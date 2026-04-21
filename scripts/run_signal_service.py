from __future__ import annotations

import argparse
import csv
import html
import json
import logging
import os
import signal
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from typing import Any, Callable

import pandas as pd
from dotenv import load_dotenv

from app.data.bingx_client import fetch_klines, interval_to_timedelta
from app.notify.telegram import SignalServiceTelegramNotifier
from app.strategy.signals import Signal, SignalAction
from app.strategy.trend_following_v2_simple import TrendFollowingV2Simple
from validation.structural_stop import compute_pivot_highs, compute_pivot_lows


DEFAULT_PARAMS = {
    "rr_ratio": 2.7,
    "adx_min": 20.0,
    "adx_strong": 35.0,
    "ema_fast": 20,
    "ema_slow": 50,
    "slope_bars": 5,
    "pullback_tolerance_atr": 1.0,
    "min_confidence": 0.0,
    "allow_short": True,
    "sig_cooldown": 5,
    "structural_stop_enabled": True,
    "structural_buffer_atr": 0.25,
    "structural_min_risk_atr": 0.8,
    "structural_pivot_left": 3,
    "structural_pivot_right": 3,
}

IDLE_POLL_SECONDS = 5

CSV_COLUMNS = [
    "timestamp_utc",
    "signal_type",
    "entry",
    "sl",
    "tp",
    "confidence",
    "htf_bias",
    "adx",
    "body_ratio",
    "macd_hist_dir",
    "risk_usd",
    "position_usd",
    "leverage",
    "qty",
    "rr_ratio",
    "sl_mode",
]

CLOSURE_COLUMNS = [
    "timestamp_utc",
    "signal_type",
    "exit_type",
    "entry_price",
    "exit_price",
    "pnl_pct",
    "pnl_usd",
    "bars_held",
    "hours_held",
]


@dataclass
class HTFSnapshot:
    bias: int
    close: float
    ema50: float
    diff_pct: float


@dataclass
class ProcessSummary:
    bars_processed: int = 0
    signals_emitted: int = 0
    closures_emitted: int = 0
    startup: bool = False
    latest_htf_bias: int = 0
    heartbeat_sent: bool = False


@dataclass
class ServiceConfig:
    symbol: str
    interval: str
    paper_equity_usd: float
    log_level: str
    fetch_limit: int = 1000
    strategy_bars: int = 300
    risk_fraction: float = 0.015
    leverage_cap: float = 3.0
    telegram_token: str = ""
    telegram_chat_id: str = ""
    csv_path: Path = Path("data/live_signals.csv")
    closures_csv_path: Path = Path("data/live_closures.csv")
    log_path: Path = Path("logs/signal_service.log")
    state_path: Path = Path("data/signal_service_state.json")
    api_error_alert_threshold: int = 3

    @property
    def interval_delta(self) -> timedelta:
        return interval_to_timedelta(self.interval)


def compute_htf_bias(df_15m: pd.DataFrame) -> int:
    return _compute_htf_snapshot(df_15m).bias


def _compute_htf_snapshot(df_15m: pd.DataFrame) -> HTFSnapshot:
    if df_15m.empty:
        return HTFSnapshot(bias=0, close=0.0, ema50=0.0, diff_pct=0.0)

    df_htf = (
        df_15m.set_index("ts")
        .resample("4h", label="right", closed="right")
        .agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
        )
        .dropna()
    )
    if df_htf.empty:
        return HTFSnapshot(bias=0, close=0.0, ema50=0.0, diff_pct=0.0)

    latest_15m_close = df_15m["ts"].iloc[-1]
    if df_htf.index[-1] > latest_15m_close:
        df_htf = df_htf.iloc[:-1]
    if df_htf.empty:
        return HTFSnapshot(bias=0, close=0.0, ema50=0.0, diff_pct=0.0)

    df_htf["ema50"] = df_htf["close"].ewm(span=50, adjust=False).mean()
    row = df_htf.iloc[-1]
    close = float(row["close"])
    ema50 = float(row["ema50"])
    if close > ema50:
        bias = 1
    elif close < ema50:
        bias = -1
    else:
        bias = 0
    diff_pct = ((close - ema50) / ema50 * 100.0) if ema50 else 0.0
    return HTFSnapshot(bias=bias, close=close, ema50=ema50, diff_pct=diff_pct)


def _bias_label(bias: int) -> str:
    return {1: "BULL", -1: "BEAR", 0: "NEUTRAL"}.get(int(bias), "NEUTRAL")


def _setup_logging(level: str, log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("signal_service")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = TimedRotatingFileHandler(
        log_path,
        when="midnight",
        backupCount=14,
        encoding="utf-8",
        utc=True,
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def _ensure_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()


def _ensure_closures_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CLOSURE_COLUMNS)
        writer.writeheader()


def _latest_confirmed_pivot_offset(df: pd.DataFrame, direction: str, left: int, right: int) -> int | None:
    if len(df) < left + right + 1:
        return None

    values = (
        df["low"].to_numpy(dtype=float)
        if direction == "LONG"
        else df["high"].to_numpy(dtype=float)
    )
    pivots = (
        compute_pivot_lows(values, left=left, right=right)
        if direction == "LONG"
        else compute_pivot_highs(values, left=left, right=right)
    )
    current_idx = len(df) - 1
    valid_indices = [idx for idx, value in enumerate(pivots) if pd.notna(value) and idx + right <= current_idx]
    if not valid_indices:
        return None
    return current_idx - valid_indices[-1]


def _compute_indicator_context(
    strategy: TrendFollowingV2Simple,
    df: pd.DataFrame,
    direction: str,
) -> dict[str, float | str | int]:
    ind_df = strategy._compute_indicators(df)
    row = ind_df.iloc[-1]
    prev_hist = float(ind_df["macd_hist"].iloc[-2]) if len(ind_df) >= 2 else 0.0
    macd_hist = float(row["macd_hist"]) if pd.notna(row["macd_hist"]) else 0.0
    body = abs(float(row["close"]) - float(row["open"]))
    candle_range = max(float(row["high"]) - float(row["low"]), 1e-10)
    body_ratio = body / candle_range
    confidence_score = strategy._compute_confidence(row, ind_df, direction)

    if direction == "LONG":
        if macd_hist > 0 and macd_hist > prev_hist:
            macd_hist_dir = "creciente (bullish ✓)"
        else:
            macd_hist_dir = "decreciente (warning)"
    else:
        if macd_hist < 0 and macd_hist < prev_hist:
            macd_hist_dir = "decreciente (bearish ✓)"
        else:
            macd_hist_dir = "creciente (warning)"

    return {
        "adx": float(row["adx"]) if pd.notna(row["adx"]) else 0.0,
        "atr": float(row["atr"]) if pd.notna(row["atr"]) else 0.0,
        "body_ratio": body_ratio,
        "macd_hist": macd_hist,
        "macd_hist_dir": macd_hist_dir,
        "confidence_score": confidence_score,
    }


def _build_sizing(entry: float, sl: float, equity: float, leverage_cap: float) -> dict[str, float]:
    sl_distance_pct = abs(entry - sl) / entry if entry else 0.0
    risk_usd = equity * 0.015
    position_usd = risk_usd / sl_distance_pct if sl_distance_pct > 0 else 0.0
    leverage = min(position_usd / equity, leverage_cap) if equity > 0 else 0.0
    qty = position_usd / entry if entry > 0 else 0.0
    return {
        "risk_usd": risk_usd,
        "sl_distance_pct": sl_distance_pct,
        "position_usd": position_usd,
        "leverage": leverage,
        "qty": qty,
    }


def build_entry_context(
    signal: Signal,
    df: pd.DataFrame,
    strategy: TrendFollowingV2Simple,
    htf: HTFSnapshot,
    equity: float,
    leverage_cap: float,
) -> tuple[dict[str, Any], dict[str, float | str]]:
    direction = "LONG" if signal.action == SignalAction.BUY else "SHORT"
    pivot_kind = "pivot low" if direction == "LONG" else "pivot high"

    entry = float(df["close"].iloc[-1])
    sl = float(signal.stop_loss) if signal.stop_loss is not None else float(signal.meta.get("sl", 0.0))
    tp = float(signal.take_profit) if signal.take_profit is not None else float(signal.meta.get("tp1", 0.0))
    rr_ratio = float(signal.meta.get("rr_ratio", DEFAULT_PARAMS["rr_ratio"]))
    sl_mode = str(signal.meta.get("sl_mode", "unknown"))

    indicators = _compute_indicator_context(strategy, df, direction)
    sizing = _build_sizing(entry, sl, equity, leverage_cap)
    pivot_offset = _latest_confirmed_pivot_offset(
        df,
        direction,
        strategy.params.get("structural_pivot_left", 3),
        strategy.params.get("structural_pivot_right", 3),
    )
    risk_pct = sizing["sl_distance_pct"] * 100.0
    tp_pct = (abs(tp - entry) / entry * 100.0) if entry else 0.0
    adx_strong = "strong ✓" if indicators["adx"] >= 35.0 else "moderate"
    body_strong = "strong candle ✓" if indicators["body_ratio"] >= 0.60 else "normal"
    htf_relation = "&gt;" if htf.close > htf.ema50 else ("&lt;" if htf.close < htf.ema50 else "=")
    context = {
        "signal_type": direction,
        "entry": entry,
        "sl": sl,
        "tp": tp,
        "rr_ratio": rr_ratio,
        "risk_pct": risk_pct,
        "tp_pct": tp_pct,
        "adx": float(indicators["adx"]),
        "adx_label": adx_strong,
        "body_ratio": float(indicators["body_ratio"]),
        "body_label": body_strong,
        "macd_hist_dir": str(indicators["macd_hist_dir"]),
        "htf_label": _bias_label(htf.bias),
        "htf_close": htf.close,
        "htf_ema50": htf.ema50,
        "htf_relation": htf_relation,
        "htf_diff_pct": htf.diff_pct,
        "confidence_pct": int(round(float(indicators["confidence_score"]) * 100)),
        "equity": equity,
        "risk_usd": sizing["risk_usd"],
        "position_usd": sizing["position_usd"],
        "leverage": sizing["leverage"],
        "leverage_cap": leverage_cap,
        "qty": sizing["qty"],
        "pivot_kind": pivot_kind,
        "pivot_offset": pivot_offset if pivot_offset is not None else "?",
    }

    csv_row = {
        "timestamp_utc": signal.ts.strftime("%Y-%m-%d %H:%M:%S"),
        "signal_type": direction,
        "entry": round(entry, 8),
        "sl": round(sl, 8),
        "tp": round(tp, 8),
        "confidence": round(float(indicators["confidence_score"]), 4),
        "htf_bias": _bias_label(htf.bias),
        "adx": round(float(indicators["adx"]), 4),
        "body_ratio": round(float(indicators["body_ratio"]), 4),
        "macd_hist_dir": str(indicators["macd_hist_dir"]),
        "risk_usd": round(sizing["risk_usd"], 8),
        "position_usd": round(sizing["position_usd"], 8),
        "leverage": round(sizing["leverage"], 8),
        "qty": round(sizing["qty"], 8),
        "rr_ratio": round(rr_ratio, 4),
        "sl_mode": sl_mode,
    }
    return context, csv_row


def _format_time_label(ts: pd.Timestamp) -> str:
    months = {
        1: "ene", 2: "feb", 3: "mar", 4: "abr", 5: "may", 6: "jun",
        7: "jul", 8: "ago", 9: "sep", 10: "oct", 11: "nov", 12: "dic",
    }
    return f"{ts.strftime('%H:%M UTC')} {ts.day:02d}-{months[ts.month]}"


def _as_utc_timestamp(value: Any) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def _format_duration(bars_held: int, interval: timedelta) -> tuple[float, str]:
    total_minutes = int(bars_held * interval.total_seconds() // 60)
    hours, minutes = divmod(total_minutes, 60)
    return total_minutes / 60.0, f"{hours}h {minutes}m"


class SignalService:
    def __init__(
        self,
        config: ServiceConfig,
        *,
        strategy: TrendFollowingV2Simple | None = None,
        fetcher: Callable[[str, str, int], pd.DataFrame] = fetch_klines,
        telegram: SignalServiceTelegramNotifier | None = None,
        logger: logging.Logger | None = None,
        dry_run: bool = False,
    ) -> None:
        self.config = config
        self.strategy = strategy or TrendFollowingV2Simple(symbol=config.symbol, params=DEFAULT_PARAMS.copy())
        self.fetcher = fetcher
        self.telegram = telegram or SignalServiceTelegramNotifier(
            config.telegram_token,
            config.telegram_chat_id,
            enabled=not dry_run,
        )
        self.logger = logger or _setup_logging(config.log_level, config.log_path)
        self.dry_run = dry_run

        self.last_processed_ts: pd.Timestamp | None = None
        self.last_signal_key: tuple[str, str] | None = None
        self.bootstrapped = False
        self.total_bars_processed = 0
        self.total_signals_emitted = 0
        self.total_closures_emitted = 0
        self.last_heartbeat_hour: str | None = None
        self.last_daily_summary_date: str | None = None
        self.latest_htf_bias: int = 0
        self.started_at_utc = datetime.now(UTC)
        self.consecutive_api_failures = 0
        self.api_error_alerted = False
        self.daily_processed_bars: dict[str, int] = {}

        _ensure_csv(config.csv_path)
        _ensure_closures_csv(config.closures_csv_path)
        self._restore_state_if_present()

    def bootstrap(self) -> ProcessSummary:
        df = self._fetch_with_observability()
        if df.empty:
            raise RuntimeError("Bootstrap failed: no bars returned by BingX")

        df = df.reset_index(drop=True)
        history = df.tail(self.config.strategy_bars).reset_index(drop=True)
        latest_htf = _compute_htf_snapshot(df)
        self.latest_htf_bias = latest_htf.bias
        summary = ProcessSummary(startup=True, latest_htf_bias=latest_htf.bias)

        for idx in range(len(history)):
            window = history.iloc[max(0, idx - self.config.strategy_bars + 1) : idx + 1].reset_index(drop=True)
            full_slice = history.iloc[: idx + 1].reset_index(drop=True)
            htf = _compute_htf_snapshot(full_slice)
            self.strategy.on_bar_all(window, htf_bias=htf.bias)
            self.last_processed_ts = window["ts"].iloc[-1]
            self.total_bars_processed += 1
            summary.bars_processed += 1

        self.bootstrapped = True
        self.logger.info(
            "Service started | %s %s | equity=%.2f USDT | HTF bias=%s",
            self.config.symbol,
            self.config.interval,
            self.config.paper_equity_usd,
            _bias_label(latest_htf.bias),
        )
        if not self.dry_run:
            sent = self.telegram.send_lifecycle_notification(
                "start",
                {
                    "symbol": self.config.symbol,
                    "interval": self.config.interval,
                    "equity": self.config.paper_equity_usd,
                    "htf_bias": _bias_label(latest_htf.bias),
                    "bootstrap_bars": summary.bars_processed,
                },
            )
            self.logger.info("Startup telegram=%s", "sent" if sent else "failed")
        return summary

    def run_once(self) -> ProcessSummary:
        if not self.bootstrapped:
            return self.bootstrap()

        try:
            df = self._fetch_with_observability()
        except Exception:
            return ProcessSummary()
        if df.empty:
            self.logger.warning("No bars returned by BingX")
            return ProcessSummary()

        df = df.reset_index(drop=True)
        summary = ProcessSummary()
        if self.last_processed_ts is None:
            return self.bootstrap()

        unseen_positions = df.index[df["ts"] > self.last_processed_ts].tolist()
        if not unseen_positions:
            return summary

        for pos in unseen_positions:
            full_slice = df.iloc[: pos + 1].reset_index(drop=True)
            window = full_slice.tail(self.config.strategy_bars).reset_index(drop=True)
            htf = _compute_htf_snapshot(full_slice)
            self.latest_htf_bias = htf.bias
            summary.latest_htf_bias = htf.bias

            signals = self.strategy.on_bar_all(window, htf_bias=htf.bias)
            latest_ts = window["ts"].iloc[-1]
            self.last_processed_ts = latest_ts
            self.total_bars_processed += 1
            summary.bars_processed += 1
            self._track_processed_bar(latest_ts)

            entry_or_close = False
            for sig in signals:
                if sig.action in (SignalAction.BUY, SignalAction.SELL):
                    entry_or_close = True
                    signal_type = "LONG" if sig.action == SignalAction.BUY else "SHORT"
                    signal_key = (latest_ts.isoformat(), signal_type)
                    if signal_key == self.last_signal_key:
                        self.logger.info(
                            "Bar closed | duplicate signal skipped | ts=%s | side=%s",
                            latest_ts.isoformat(),
                            signal_type,
                        )
                        continue
                    context, csv_row = build_entry_context(
                        sig,
                        window,
                        self.strategy,
                        htf,
                        self.config.paper_equity_usd,
                        self.config.leverage_cap,
                    )
                    sent = True if self.dry_run else self.telegram.send_entry_notification(sig, context)
                    self._append_csv(csv_row)
                    self.last_signal_key = signal_key
                    self.total_signals_emitted += 1
                    summary.signals_emitted += 1
                    self.logger.info(
                        "Bar closed | signal=%s | ts=%s | adx=%.1f | htf=%s | telegram=%s",
                        signal_type,
                        latest_ts.isoformat(),
                        context["adx"],
                        _bias_label(htf.bias),
                        "sent" if sent else "failed",
                    )
                elif sig.action == SignalAction.CLOSE:
                    entry_or_close = True
                    closure_context, closure_row = self._build_closure_context(sig, latest_ts)
                    if closure_context is None or closure_row is None:
                        self.logger.warning("Close signal without matching open trade | ts=%s", latest_ts.isoformat())
                        continue
                    sent = True if self.dry_run else self.telegram.send_exit_notification(sig, closure_context)
                    self._append_closure_csv(closure_row)
                    self.total_closures_emitted += 1
                    summary.closures_emitted += 1
                    self.logger.info(
                        "Bar closed | close=%s | side=%s | ts=%s | telegram=%s",
                        closure_context["exit_type"],
                        closure_context["signal_type"],
                        latest_ts.isoformat(),
                        "sent" if sent else "failed",
                    )

            if not entry_or_close:
                indicators = _compute_indicator_context(self.strategy, window, "LONG")
                self.logger.info(
                    "Bar closed | no signal | adx=%.1f | htf=%s | ts=%s",
                    indicators["adx"],
                    _bias_label(htf.bias),
                    latest_ts.isoformat(),
                )

        return summary

    def maybe_send_heartbeat(self, now: datetime | None = None) -> bool:
        if self.dry_run or not self.bootstrapped:
            return False

        now_utc = now or datetime.now(UTC)
        self._maybe_send_daily_summary(now_utc)
        hour_key = now_utc.strftime("%Y-%m-%d %H")
        if self.last_heartbeat_hour == hour_key:
            return False

        sent = self.telegram.send_lifecycle_notification(
            "heartbeat",
            {"message": self._heartbeat_message(now_utc)},
        )
        self.last_heartbeat_hour = hour_key
        self.logger.info(
            "Heartbeat | ts=%s | htf=%s | telegram=%s",
            now_utc.strftime("%Y-%m-%d %H:%M:%S UTC"),
            _bias_label(self.latest_htf_bias),
            "sent" if sent else "failed",
        )
        return sent

    def stop(self) -> None:
        self.logger.info("Shutdown requested | persisting service state")
        self._save_state()
        if not self.dry_run:
            self.telegram.send_lifecycle_notification(
                "stop",
                {
                    "bars_processed": self.total_bars_processed,
                    "signals": self.total_signals_emitted,
                    "closures": self.total_closures_emitted,
                    "uptime_label": self._uptime_label(),
                },
            )
        for handler in self.logger.handlers:
            handler.flush()

    def _append_csv(self, row: dict[str, float | str]) -> None:
        with self.config.csv_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
            writer.writerow(row)

    def _append_closure_csv(self, row: dict[str, float | str]) -> None:
        with self.config.closures_csv_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=CLOSURE_COLUMNS)
            writer.writerow(row)

    def _fetch_with_observability(self) -> pd.DataFrame:
        try:
            df = self.fetcher(self.config.symbol, self.config.interval, self.config.fetch_limit)
            self.consecutive_api_failures = 0
            self.api_error_alerted = False
            return df
        except Exception as exc:
            self.consecutive_api_failures += 1
            self.logger.error(
                "BingX fetch failed | consecutive=%d | error=%s",
                self.consecutive_api_failures,
                exc,
            )
            if (
                not self.dry_run
                and self.consecutive_api_failures >= self.config.api_error_alert_threshold
                and not self.api_error_alerted
            ):
                self.telegram.send_lifecycle_notification(
                    "api_error",
                    {
                        "failures": self.consecutive_api_failures,
                        "error_message": str(exc),
                    },
                )
                self.api_error_alerted = True
            raise

    def _track_processed_bar(self, ts: pd.Timestamp) -> None:
        key = ts.strftime("%Y-%m-%d")
        self.daily_processed_bars[key] = self.daily_processed_bars.get(key, 0) + 1

    def _read_csv(self, path: Path, columns: list[str]) -> pd.DataFrame:
        if not path.exists():
            return pd.DataFrame(columns=columns)
        return pd.read_csv(path)

    def _find_open_entry_context(self, signal_type: str) -> dict[str, Any] | None:
        entries = self._read_csv(self.config.csv_path, CSV_COLUMNS)
        if entries.empty:
            return None
        entries = entries[entries["signal_type"] == signal_type].copy()
        if entries.empty:
            return None
        entries["timestamp_utc"] = pd.to_datetime(entries["timestamp_utc"], utc=True)
        entries = entries.sort_values("timestamp_utc").reset_index(drop=True)

        closures = self._read_csv(self.config.closures_csv_path, CLOSURE_COLUMNS)
        if not closures.empty:
            closures = closures[closures["signal_type"] == signal_type].copy()
            closures["timestamp_utc"] = pd.to_datetime(closures["timestamp_utc"], utc=True)
            closures = closures.sort_values("timestamp_utc").reset_index(drop=True)
            closed_count = min(len(entries), len(closures))
            if closed_count:
                entries = entries.iloc[closed_count:].reset_index(drop=True)

        if entries.empty:
            return None
        return entries.iloc[-1].to_dict()

    def _build_closure_context(
        self,
        signal: Signal,
        latest_ts: pd.Timestamp,
    ) -> tuple[dict[str, Any] | None, dict[str, float | str] | None]:
        exit_type = str(signal.meta.get("exit_type", "")).lower()
        if exit_type not in {"sl", "tp"}:
            return None, None

        signal_type = self._infer_signal_type_from_open_entries()
        if signal_type is None:
            return None, None
        open_entry = self._find_open_entry_context(signal_type)
        if open_entry is None:
            return None, None

        entry_ts = _as_utc_timestamp(open_entry["timestamp_utc"])
        exit_price = float(signal.meta.get("exit_price", 0.0))
        pnl_pct = float(signal.meta.get("pnl_pct", 0.0))
        leverage = float(open_entry.get("leverage", 0.0))
        position_usd = float(open_entry.get("position_usd", 0.0))
        pnl_usd = position_usd * (pnl_pct / 100.0)
        bars_held = max(1, int(round((latest_ts - entry_ts) / pd.Timedelta(self.config.interval_delta))))
        hours_held, duration_label = _format_duration(bars_held, self.config.interval_delta)
        cooldown_bars = int(getattr(self.strategy, "_sig_cooldown", DEFAULT_PARAMS["sig_cooldown"]))
        context = {
            "signal_type": signal_type,
            "exit_type": exit_type,
            "entry_price": float(open_entry["entry"]),
            "exit_price": exit_price,
            "pnl_pct": pnl_pct,
            "pnl_usd": pnl_usd,
            "leverage": leverage,
            "bars_held": bars_held,
            "hours_held": hours_held,
            "duration_label": duration_label,
            "entry_time_label": _format_time_label(entry_ts),
            "exit_time_label": _format_time_label(latest_ts),
            "cooldown_bars": cooldown_bars,
            "cooldown_until_bar": self._cooldown_until_bar(signal_type),
        }
        row = {
            "timestamp_utc": signal.ts.strftime("%Y-%m-%d %H:%M:%S"),
            "signal_type": signal_type,
            "exit_type": exit_type,
            "entry_price": round(float(open_entry["entry"]), 8),
            "exit_price": round(exit_price, 8),
            "pnl_pct": round(pnl_pct, 8),
            "pnl_usd": round(pnl_usd, 8),
            "bars_held": bars_held,
            "hours_held": round(hours_held, 4),
        }
        return context, row

    def _infer_signal_type_from_open_entries(self) -> str | None:
        candidates: list[tuple[pd.Timestamp, str]] = []
        for signal_type in ("LONG", "SHORT"):
            entry = self._find_open_entry_context(signal_type)
            if entry is not None:
                candidates.append((_as_utc_timestamp(entry["timestamp_utc"]), signal_type))
        if not candidates:
            return None
        candidates.sort(key=lambda item: item[0])
        return candidates[-1][1]

    def _cooldown_until_bar(self, signal_type: str) -> int:
        last_bar = getattr(
            self.strategy,
            "_last_long_bar" if signal_type == "LONG" else "_last_short_bar",
            0,
        )
        cooldown = int(getattr(self.strategy, "_sig_cooldown", DEFAULT_PARAMS["sig_cooldown"]))
        return int(last_bar + cooldown)

    def _uptime_label(self) -> str:
        elapsed = datetime.now(UTC) - self.started_at_utc
        total_minutes = int(elapsed.total_seconds() // 60)
        hours, minutes = divmod(total_minutes, 60)
        return f"{hours}h {minutes}m"

    def _maybe_send_daily_summary(self, now_utc: datetime) -> None:
        previous_day = (now_utc - timedelta(days=1)).strftime("%Y-%m-%d")
        if self.last_daily_summary_date == previous_day:
            return
        summary = self._build_daily_summary(previous_day)
        if not self.dry_run:
            self.telegram.send_daily_summary(summary)
        self.last_daily_summary_date = previous_day
        self.logger.info("Daily summary sent | date=%s", previous_day)

    def _build_daily_summary(self, day_key: str) -> dict[str, Any]:
        signals = self._read_csv(self.config.csv_path, CSV_COLUMNS)
        closures = self._read_csv(self.config.closures_csv_path, CLOSURE_COLUMNS)

        if not signals.empty:
            signals["timestamp_utc"] = pd.to_datetime(signals["timestamp_utc"], utc=True)
            day_signals = signals[signals["timestamp_utc"].dt.strftime("%Y-%m-%d") == day_key]
        else:
            day_signals = pd.DataFrame(columns=CSV_COLUMNS)

        if not closures.empty:
            closures["timestamp_utc"] = pd.to_datetime(closures["timestamp_utc"], utc=True)
            day_closures = closures[closures["timestamp_utc"].dt.strftime("%Y-%m-%d") == day_key]
        else:
            day_closures = pd.DataFrame(columns=CLOSURE_COLUMNS)

        bars_processed = int(self.daily_processed_bars.get(day_key, 0))
        uptime_pct = min(100.0, bars_processed / 96 * 100) if bars_processed else 0.0
        pnl_usd = float(day_closures["pnl_usd"].sum()) if not day_closures.empty else 0.0
        pnl_pct = (pnl_usd / self.config.paper_equity_usd * 100.0) if self.config.paper_equity_usd else 0.0
        summary = {
            "date": day_key,
            "signals": int(len(day_signals)),
            "longs": int((day_signals["signal_type"] == "LONG").sum()) if not day_signals.empty else 0,
            "shorts": int((day_signals["signal_type"] == "SHORT").sum()) if not day_signals.empty else 0,
            "closures": int(len(day_closures)),
            "tp": int((day_closures["exit_type"] == "tp").sum()) if not day_closures.empty else 0,
            "sl": int((day_closures["exit_type"] == "sl").sum()) if not day_closures.empty else 0,
            "pnl_usd": pnl_usd,
            "pnl_pct": pnl_pct,
            "bars_processed": bars_processed,
            "uptime_pct": uptime_pct,
            "idle": len(day_signals) == 0 and len(day_closures) == 0,
        }
        return summary

    def _save_state(self) -> None:
        self.config.state_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "saved_at_utc": datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC"),
            "symbol": self.config.symbol,
            "interval": self.config.interval,
            "last_processed_ts": self.last_processed_ts.isoformat() if self.last_processed_ts is not None else None,
            "last_signal_key": list(self.last_signal_key) if self.last_signal_key is not None else None,
            "bootstrapped": self.bootstrapped,
            "total_bars_processed": self.total_bars_processed,
            "total_signals_emitted": self.total_signals_emitted,
            "total_closures_emitted": self.total_closures_emitted,
            "last_heartbeat_hour": self.last_heartbeat_hour,
            "last_daily_summary_date": self.last_daily_summary_date,
            "latest_htf_bias": self.latest_htf_bias,
            "daily_processed_bars": self.daily_processed_bars,
            "strategy_runtime_state": self.strategy.export_runtime_state(),
        }
        self.config.state_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        self.logger.info("State saved | path=%s", self.config.state_path)

    def _restore_state_if_present(self) -> None:
        if not self.config.state_path.exists():
            return
        try:
            payload = json.loads(self.config.state_path.read_text(encoding="utf-8"))
        except Exception as exc:
            self.logger.warning("State restore skipped | unreadable state file: %s", exc)
            return

        if payload.get("symbol") != self.config.symbol or payload.get("interval") != self.config.interval:
            self.logger.info("State restore skipped | symbol/interval mismatch")
            return

        last_processed_ts = payload.get("last_processed_ts")
        if last_processed_ts:
            self.last_processed_ts = pd.Timestamp(last_processed_ts)
        last_signal_key = payload.get("last_signal_key")
        if isinstance(last_signal_key, list) and len(last_signal_key) == 2:
            self.last_signal_key = (str(last_signal_key[0]), str(last_signal_key[1]))
        self.bootstrapped = bool(payload.get("bootstrapped", False))
        self.total_bars_processed = int(payload.get("total_bars_processed", 0))
        self.total_signals_emitted = int(payload.get("total_signals_emitted", 0))
        self.total_closures_emitted = int(payload.get("total_closures_emitted", 0))
        self.last_heartbeat_hour = payload.get("last_heartbeat_hour")
        self.last_daily_summary_date = payload.get("last_daily_summary_date")
        self.latest_htf_bias = int(payload.get("latest_htf_bias", 0))
        self.daily_processed_bars = dict(payload.get("daily_processed_bars", {}))
        self.strategy.restore_runtime_state(payload.get("strategy_runtime_state"))
        self.logger.info(
            "State restored | last_processed_ts=%s | signals=%d",
            self.last_processed_ts.isoformat() if self.last_processed_ts is not None else "none",
            self.total_signals_emitted,
        )

    def _startup_message(self, htf: HTFSnapshot) -> str:
        return (
            "🟦 <b>Signal Service started</b>\n\n"
            f"📊 Symbol: <b>{html.escape(self.config.symbol)}</b>\n"
            f"⏱️ Interval: <b>{html.escape(self.config.interval)}</b>\n"
            f"💰 Equity base: <b>{self.config.paper_equity_usd:.2f} USDT</b>\n"
            f"🧭 HTF bias: <b>{_bias_label(htf.bias)}</b> ({htf.close:.0f} vs EMA50 {htf.ema50:.0f}, {htf.diff_pct:+.2f}%)\n"
            f"🧱 Warmup bars: <b>{self.config.strategy_bars}</b>\n"
            f"📨 Heartbeat: <b>every 1 hour</b>\n\n"
            f"⏰ {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S UTC')}"
        )

    def _heartbeat_message(self, now_utc: datetime) -> str:
        return (
            "💓 <b>Signal Service heartbeat</b>\n\n"
            f"📊 Symbol: <b>{html.escape(self.config.symbol)}</b>\n"
            f"⏱️ Interval: <b>{html.escape(self.config.interval)}</b>\n"
            f"🧭 HTF bias: <b>{_bias_label(self.latest_htf_bias)}</b>\n"
            f"🪵 Bars processed: <b>{self.total_bars_processed}</b>\n"
            f"🔔 Signals emitted: <b>{self.total_signals_emitted}</b>\n"
            f"✅ Status: <b>running</b>\n\n"
            f"⏰ {now_utc.strftime('%Y-%m-%d %H:%M:%S UTC')}"
        )

    def _shutdown_message(self) -> str:
        last_bar = self.last_processed_ts.isoformat() if self.last_processed_ts is not None else "none"
        return (
            "⏹️ <b>Signal Service stopped</b>\n\n"
            f"📊 Symbol: <b>{html.escape(self.config.symbol)}</b>\n"
            f"🪵 Bars processed: <b>{self.total_bars_processed}</b>\n"
            f"🔔 Signals emitted: <b>{self.total_signals_emitted}</b>\n"
            f"🕒 Last processed bar: <b>{html.escape(last_bar)}</b>\n"
            f"💾 State file: <code>{html.escape(str(self.config.state_path))}</code>\n\n"
            f"⏰ {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S UTC')}"
        )


def _load_config() -> ServiceConfig:
    load_dotenv()
    return ServiceConfig(
        symbol=os.getenv("SYMBOL", "BTC-USDT").strip() or "BTC-USDT",
        interval=os.getenv("INTERVAL", "15m").strip() or "15m",
        paper_equity_usd=float(os.getenv("PAPER_EQUITY_USD", "100.0")),
        log_level=os.getenv("LOG_LEVEL", "INFO").strip() or "INFO",
        telegram_token=os.getenv("TELEGRAM_BOT_TOKEN", "").strip(),
        telegram_chat_id=os.getenv("TELEGRAM_CHAT_ID", "").strip(),
    )


def _sleep_idle() -> None:
    time.sleep(IDLE_POLL_SECONDS)


def _should_poll(now: datetime) -> bool:
    return now.minute % 15 == 0 and now.second >= 30


def _should_send_heartbeat(now: datetime) -> bool:
    return now.minute == 0 and now.second >= 30


def main() -> int:
    parser = argparse.ArgumentParser(description="Local TrendBot signal service prototype")
    parser.add_argument("--once", action="store_true", help="Bootstrap or process one iteration and exit")
    parser.add_argument("--dry-run", action="store_true", help="Do not send Telegram messages")
    args = parser.parse_args()

    config = _load_config()
    logger = _setup_logging(config.log_level, config.log_path)
    service = SignalService(config, logger=logger, dry_run=args.dry_run)

    stop_requested = False

    def _handle_signal(_signum: int, _frame: object | None) -> None:
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    startup = service.bootstrap()
    if args.once:
        logger.info(
            "First run successful: Service started at %s, processed %d bars, emitted %d signals",
            datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC"),
            startup.bars_processed,
            startup.signals_emitted,
        )
        service.stop()
        return 0

    try:
        while not stop_requested:
            now = datetime.now(UTC)
            if _should_send_heartbeat(now):
                service.maybe_send_heartbeat(now)
                _sleep_idle()
                continue
            if _should_poll(now):
                summary = service.run_once()
                if summary.bars_processed:
                    logger.info(
                        "Cycle complete | processed=%d | emitted=%d | htf=%s",
                        summary.bars_processed,
                        summary.signals_emitted,
                        _bias_label(summary.latest_htf_bias),
                    )
                _sleep_idle()
            else:
                _sleep_idle()
    finally:
        service.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
