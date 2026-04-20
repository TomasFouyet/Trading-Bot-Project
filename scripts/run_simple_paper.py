#!/usr/bin/env python3
"""
Simple Paper Trading Runner — executes the VALIDATED strategy.

═══════════════════════════════════════════════════════════════════════════════
This runner executes TrendFollowingV2Simple — the exact strategy that passed
statistical validation (WFA 5/5, MC RoR 0.9%, Permutation p=0.001).

NO TP1/TP2, NO trailing, NO tiers, NO reversal swap.
Single fixed SL + single fixed TP, full close only.

The strategy is the SINGLE source of truth for SL/TP/entry/exit.
This runner only: feeds bars, executes signals, sizes positions, logs, notifies.
═══════════════════════════════════════════════════════════════════════════════

Usage:
    python scripts/run_simple_paper.py                  # Paper (default)
    python scripts/run_simple_paper.py --headless       # No console, Telegram only
    python scripts/run_simple_paper.py --check          # Verify BingX
    python scripts/run_simple_paper.py --balance 200    # Custom balance
    python scripts/run_simple_paper.py --live            # REAL orders (confirmation required)
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import fcntl
import hashlib
import json
import os
import signal
import sys
import time
import uuid
from collections import deque
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ═══════════════════════════════════════════════════════════════════════════════
# VALIDATED PARAMS — do not change without re-running full WFA + PSA
# ═══════════════════════════════════════════════════════════════════════════════

VALIDATED_PARAMS = {
    # Entry (validated)
    "adx_min":               20,
    "adx_strong":            35,
    "ema_fast":              20,
    "ema_slow":              50,
    "pullback_tolerance_atr": 1.0,
    "allow_short":           True,
    "min_confidence":        0.0,   # No confidence filter — all signals
    "sig_cooldown":          5,
    "slope_bars":            5,
    # Exit (validated — rr=2.5, structural stop)
    "rr_ratio":              2.5,
    "atr_sl_mult":           2.0,
    "structural_stop_enabled": True,
    "structural_buffer_atr": 0.25,
    "structural_min_risk_atr": 0.8,
    "structural_pivot_left": 3,
    "structural_pivot_right": 3,
}

RISK_CONFIG = {
    "risk_pct_per_trade":    1.5,   # % of equity risked per trade
    "max_leverage":          3.0,   # hard cap
    "max_daily_dd_pct":      15.0,  # circuit breaker warning (reduce size)
    "circuit_breaker_pct":   25.0,  # full stop, manual review required
    "max_positions":         1,     # BTC only, 1 position at a time
}

HTF_CONFIG = {
    "enabled":    True,
    "timeframe":  "4h",
    "ema_period": 50,
}

WINDOW_SIZE   = 300
STATUS_SECS   = 21600  # 6 hours
TRADES_CSV    = Path("data/simple_paper_trades.csv")
TRADES_HEADER = [
    "timestamp", "symbol", "side", "entry_price", "exit_price", "exit_type",
    "sl", "tp", "pnl_pct", "pnl_usd", "equity_after", "leverage_used",
    "sl_dist_pct", "adx_at_entry", "atr_at_entry", "htf_bias_at_entry",
    "duration_bars", "entry_fee_type", "exit_fee_type", "sl_mode",
    "confidence",
]
# BingX Futures fees: maker (limit) 0.020%, taker (market) 0.050%
MAKER_FEE = Decimal("0.00020")
TAKER_FEE = Decimal("0.00050")

# Global headless flag — set from CLI args
_HEADLESS = False


def _log(msg: str):
    """Print to console unless headless mode."""
    if not _HEADLESS:
        print(msg)


def _log_error(msg: str):
    """Always print errors, even in headless mode."""
    print(msg)


# ═══════════════════════════════════════════════════════════════════════════════
# Safe Telegram send (never crashes)
# ═══════════════════════════════════════════════════════════════════════════════

async def _tg_send(notifier, text: str) -> bool:
    """Send Telegram message safely. Returns True on success."""
    if notifier is None:
        return False
    try:
        return await notifier.send(text)
    except Exception as e:
        _log_error(f"  [TG] Send failed: {e}")
        return False


# ═══════════════════════════════════════════════════════════════════════════════
# CSV helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _ensure_csv():
    TRADES_CSV.parent.mkdir(parents=True, exist_ok=True)
    if not TRADES_CSV.exists():
        with open(TRADES_CSV, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=TRADES_HEADER).writeheader()


def _append_csv(row: dict):
    with open(TRADES_CSV, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=TRADES_HEADER).writerow(row)


def _read_last_trades(n: int = 5) -> list[dict]:
    """Read last N trades from CSV. Returns newest first."""
    if not TRADES_CSV.exists():
        return []
    try:
        with open(TRADES_CSV, "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        return rows[-n:][::-1]
    except Exception:
        return []


# ═══════════════════════════════════════════════════════════════════════════════
# Circuit Breaker (separate from RiskManager)
# ═══════════════════════════════════════════════════════════════════════════════

class CircuitBreaker:
    """Tracks peak equity and enforces drawdown limits."""

    def __init__(self, initial_equity: float):
        self.peak_equity = initial_equity
        self.current_equity = initial_equity
        self.risk_pct = RISK_CONFIG["risk_pct_per_trade"]
        self.base_risk_pct = RISK_CONFIG["risk_pct_per_trade"]
        self.stopped = False
        self.reduced = False

    def update(self, equity: float) -> str | None:
        """Update equity and return alert message if threshold crossed."""
        self.current_equity = equity
        if equity > self.peak_equity:
            self.peak_equity = equity

        dd_pct = self.drawdown_pct

        if dd_pct >= RISK_CONFIG["circuit_breaker_pct"]:
            self.stopped = True
            return (
                f"CIRCUIT BREAKER TRIGGERED: DD={dd_pct:.1f}% from peak "
                f"(peak=${self.peak_equity:,.2f}, now=${equity:,.2f}). Bot stopped."
            )

        if dd_pct >= RISK_CONFIG["max_daily_dd_pct"]:
            if not self.reduced:
                self.reduced = True
                self.risk_pct = self.base_risk_pct / 2
                return (
                    f"CIRCUIT BREAKER WARNING: DD={dd_pct:.1f}% from peak. "
                    f"Risk reduced to {self.risk_pct:.2f}% per trade."
                )
        elif dd_pct < RISK_CONFIG["max_daily_dd_pct"] * 0.67:
            # Recovery — restore normal sizing when DD drops below 10%
            if self.reduced:
                self.reduced = False
                self.risk_pct = self.base_risk_pct

        return None

    @property
    def drawdown_pct(self) -> float:
        if self.peak_equity <= 0:
            return 0.0
        return (self.peak_equity - self.current_equity) / self.peak_equity * 100


# ═══════════════════════════════════════════════════════════════════════════════
# Position Sizing (risk-based)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_position(
    equity: float,
    entry_price: float,
    sl_price: float,
    risk_pct: float,
    max_leverage: float,
) -> tuple[float, float, float]:
    """
    Risk-based position sizing.

    Returns: (qty, position_usd, leverage)
    """
    sl_distance = abs(entry_price - sl_price)
    sl_distance_pct = sl_distance / entry_price if entry_price > 0 else 1.0

    risk_usd = equity * risk_pct / 100.0
    position_usd = risk_usd / sl_distance_pct if sl_distance_pct > 0 else 0.0
    leverage = position_usd / equity if equity > 0 else 0.0
    leverage = min(leverage, max_leverage)
    position_usd = equity * leverage
    qty = position_usd / entry_price if entry_price > 0 else 0.0

    return qty, position_usd, leverage


# ═══════════════════════════════════════════════════════════════════════════════
# Synthetic fill for paper mode
# ═══════════════════════════════════════════════════════════════════════════════

def _make_paper_fill(symbol: str, price: float, qty: Decimal, side,
                     order_type: str = "market"):
    """Synthetic fill for paper mode. order_type='limit' → maker fee, 'market' → taker fee."""
    from app.broker.base import FillResult
    px = Decimal(str(price))
    fee_rate = MAKER_FEE if order_type == "limit" else TAKER_FEE
    fee = px * qty * fee_rate
    return FillResult(
        fill_id=str(uuid.uuid4()),
        order_id=str(uuid.uuid4()),
        symbol=symbol,
        side=side,
        price=px,
        qty=qty,
        fee=fee,
        fee_currency="USDT",
        timestamp=datetime.now(timezone.utc),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# HTF Bias (lightweight, for filtering)
# ═══════════════════════════════════════════════════════════════════════════════

class SimpleHTFBias:
    """
    Lightweight 4H EMA bias. Returns +1 (bull), -1 (bear), 0 (neutral).
    Used to filter signals: reject trades against the HTF trend.
    """

    def __init__(self, symbol: str, ema_period: int = 50):
        self._symbol = symbol
        self._ema_period = ema_period
        self._bias: int = 0
        self._bias_label: str = "UNKNOWN"
        self._last_refresh: datetime | None = None

    async def warmup(self, client, store):
        """Load initial 4H bars and compute bias."""
        from app.data.ingestor import OHLCVIngestor
        try:
            ingestor = OHLCVIngestor(client, store)
            bars = await ingestor.poll_latest(self._symbol, "4h", lookback_bars=300)
            if bars:
                self._compute(bars)
        except Exception as e:
            _log(f"  [HTF] Warmup failed: {e}")

    async def maybe_refresh(self, client, store, current_ts: datetime) -> bool:
        """Refresh if 4+ hours since last refresh."""
        if self._last_refresh and (current_ts - self._last_refresh).total_seconds() < 14400:
            return False
        from app.data.ingestor import OHLCVIngestor
        try:
            ingestor = OHLCVIngestor(client, store)
            bars = await ingestor.poll_latest(self._symbol, "4h", lookback_bars=300)
            if bars:
                self._compute(bars)
                return True
        except Exception as e:
            _log(f"  [HTF] Refresh failed: {e}")
        return False

    def _compute(self, bars):
        import pandas as pd
        from app.strategy.base import BaseStrategy
        df = BaseStrategy.bars_to_df(bars)
        if len(df) < self._ema_period + 10:
            return
        close = df["close"]
        ema = close.ewm(span=self._ema_period, adjust=False).mean()
        last_close = float(close.iloc[-1])
        last_ema = float(ema.iloc[-1])
        self._last_refresh = bars[-1].ts

        if last_close > last_ema * 1.002:
            self._bias = 1
            self._bias_label = f"BULL (close={last_close:.0f} > EMA{self._ema_period}={last_ema:.0f})"
        elif last_close < last_ema * 0.998:
            self._bias = -1
            self._bias_label = f"BEAR (close={last_close:.0f} < EMA{self._ema_period}={last_ema:.0f})"
        else:
            self._bias = 0
            self._bias_label = f"NEUTRAL (close={last_close:.0f} ~ EMA{self._ema_period}={last_ema:.0f})"

    @property
    def bias(self) -> int:
        return self._bias

    @property
    def label(self) -> str:
        return self._bias_label

    def bias_emoji(self) -> str:
        if self._bias == 1:
            return "BULL \u25b2"
        if self._bias == -1:
            return "BEAR \u25bc"
        return "NEUTRAL \u25cf"

    def is_aligned(self, direction: str) -> bool:
        if self._bias == 0:
            return True
        return (direction == "LONG" and self._bias == 1) or \
               (direction == "SHORT" and self._bias == -1)


# ═══════════════════════════════════════════════════════════════════════════════
# Telegram command listener
# ═══════════════════════════════════════════════════════════════════════════════

async def _telegram_command_loop(state, symbol, shutdown_event):
    """Poll Telegram for commands and respond."""
    notifier = state.get("notifier")
    if not notifier:
        return

    offset = 0
    # Flush old messages on startup (ignore anything sent before bot started)
    try:
        _, offset = await notifier.poll_commands(offset=0)
    except Exception:
        pass

    while not shutdown_event.is_set():
        try:
            messages, offset = await notifier.poll_commands(offset)
            for msg in messages:
                text = msg["text"].strip().lower()
                # Ignore messages from other chats
                if msg.get("chat_id") != notifier.chat_id:
                    continue

                if text == "/status":
                    await _cmd_status(notifier, state, symbol)
                elif text == "/trades":
                    await _cmd_trades(notifier)
                elif text == "/equity":
                    await _cmd_equity(notifier, state)
                elif text == "/stop":
                    await _tg_send(notifier,
                        "\u26a0\ufe0f <b>Stop signal received.</b>\n"
                        "Finishing current bar and shutting down...")
                    shutdown_event.set()
                elif text.startswith("/"):
                    await _tg_send(notifier,
                        "\U0001f916 Comandos disponibles:\n"
                        "/status \u2014 estado actual\n"
                        "/trades \u2014 \u00faltimos 5 trades\n"
                        "/equity \u2014 resumen de equity\n"
                        "/stop   \u2014 detener bot")
        except Exception as e:
            _log_error(f"  [TG] Command poll error: {e}")
        try:
            await asyncio.wait_for(shutdown_event.wait(), timeout=30)
            break
        except asyncio.TimeoutError:
            pass


async def _cmd_status(notifier, state, symbol):
    cb: CircuitBreaker = state["circuit_breaker"]
    eq = state.get("equity", 0)
    w, l = state.get("wins", 0), state.get("losses", 0)
    t = w + l
    wr = f"{w/t*100:.0f}%" if t else "\u2014"
    status = _runner_status(state)
    htf: SimpleHTFBias | None = state.get("htf")
    htf_str = htf.bias_emoji() if htf else "N/A"
    ot = state.get("open_trade")
    last_bar = state.get("last_bar_ts", "")
    if last_bar and hasattr(last_bar, "strftime"):
        last_bar = last_bar.strftime("%H:%M UTC")

    pos_str = "None"
    if ot:
        upnl_pct = state.get("upnl_pct", 0)
        pos_str = f"{ot['side']} @ ${float(ot['entry_price']):,.2f} (uPnL={upnl_pct:+.2f}%)"

    await _tg_send(notifier,
        f"\U0001f4ca <b>STATUS</b>\n\n"
        f"\U0001f4b0 Equity: <code>${eq:,.2f}</code> | Peak: <code>${cb.peak_equity:,.2f}</code>\n"
        f"🫀 Runner: <code>{status}</code>\n"
        f"\U0001f4c9 DD: {cb.drawdown_pct:.1f}%\n"
        f"\U0001f4c8 Trades: {t} (W={w} L={l} WR={wr})\n"
        f"\U0001f504 Position: {pos_str}\n"
        f"\U0001f9ed HTF Bias: {htf_str}\n"
        f"\u26a1 Risk/trade: {cb.risk_pct:.1f}%\n"
        f"\U0001f551 Last bar: {last_bar or 'waiting...'}")


async def _cmd_trades(notifier):
    trades = _read_last_trades(5)
    if not trades:
        await _tg_send(notifier, "\U0001f4cb <b>No trades yet.</b>")
        return
    lines = ["\U0001f4cb <b>Last 5 Trades</b>\n"]
    total_pnl = 0.0
    for i, t in enumerate(trades, 1):
        pnl = float(t.get("pnl_pct", 0))
        pnl_usd = float(t.get("pnl_usd", 0))
        emoji = "\u2705" if pnl > 0 else "\u274c"
        side = t.get("side", "?")
        bars = t.get("duration_bars", "?")
        exit_type = t.get("exit_type", "?").upper()
        lines.append(
            f"{i}. {emoji} {side} {exit_type}  {pnl:+.2f}% (${pnl_usd:+.2f})  [{bars} bars]")
        total_pnl += pnl
    n = len(trades)
    wins = sum(1 for t in trades if float(t.get("pnl_pct", 0)) > 0)
    losses = n - wins
    lines.append(f"\nW={wins} L={losses}  Avg: {total_pnl/n:+.2f}%")
    await _tg_send(notifier, "\n".join(lines))


async def _cmd_equity(notifier, state):
    cb: CircuitBreaker = state["circuit_breaker"]
    eq = state.get("equity", 0)
    start = state.get("start_balance", eq)
    net = eq - start
    net_pct = net / start * 100 if start else 0
    await _tg_send(notifier,
        f"\U0001f4b0 <b>Equity</b>\n\n"
        f"Start:   <code>${start:,.2f}</code>\n"
        f"Current: <code>${eq:,.2f}</code>\n"
        f"Net P&L: <code>{'+' if net >= 0 else ''}${net:,.2f} ({'+' if net_pct >= 0 else ''}{net_pct:.1f}%)</code>\n"
        f"Peak:    <code>${cb.peak_equity:,.2f}</code>\n"
        f"Max DD:  {cb.drawdown_pct:.1f}%")


STATE_FILE = Path("data/bot_state.json")
HEARTBEAT_FILE = Path("data/runner_heartbeat.json")
STATE_SCHEMA_VERSION = 2
HEARTBEAT_INTERVAL_SECS = 30
MAX_API_ERROR_STREAK = 5
RUNTIME_RECONCILE_INTERVAL_SECS = 180
LOCKS_DIR = Path("data/locks")


class SafeModeRequired(RuntimeError):
    """Raised when startup or runtime divergence requires manual intervention."""

    def __init__(self, reason: str, *, details: list[str] | None = None) -> None:
        self.reason = reason
        self.details = details or []
        super().__init__(reason)


class StateValidationError(RuntimeError):
    """Raised when persisted state is malformed or incomplete."""


class InstanceLockError(RuntimeError):
    """Raised when runner exclusivity cannot be guaranteed."""


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _json_default(value):
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, datetime):
        return value.isoformat()
    raise TypeError(f"Unsupported JSON value: {type(value)!r}")


def _iso_or_none(value) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).isoformat()
    return str(value)


def _atomic_json_write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
    with open(tmp_path, "w") as f:
        json.dump(payload, f, indent=2, default=_json_default)
        f.flush()
    tmp_path.replace(path)


def _quarantine_file(path: Path, suffix: str) -> Path | None:
    if not path.exists():
        return None
    stamp = _utc_now().strftime("%Y%m%dT%H%M%SZ")
    quarantined = path.with_name(f"{path.stem}.{suffix}.{stamp}{path.suffix}")
    path.replace(quarantined)
    return quarantined


def _validate_position_payload(position: dict | None) -> dict | None:
    if position is None:
        return None
    if not isinstance(position, dict):
        raise StateValidationError("position must be an object")

    required = ("side", "entry_price", "sl", "tp", "qty")
    missing = [field for field in required if field not in position]
    if missing:
        raise StateValidationError(f"position missing fields: {', '.join(missing)}")

    side = str(position["side"]).upper()
    if side not in {"LONG", "SHORT"}:
        raise StateValidationError(f"invalid position side: {position['side']!r}")

    return {
        "side": side,
        "entry_price": float(position["entry_price"]),
        "sl": float(position["sl"]),
        "tp": float(position["tp"]),
        "qty": float(position["qty"]),
        "entry_time": position.get("entry_time", ""),
        "trade_id": position.get("trade_id", ""),
        "adx_at_entry": float(position.get("adx_at_entry", 0)),
        "sl_mode": position.get("sl_mode", "unknown"),
        "leverage": float(position.get("leverage", 0)),
        "sl_dist_pct": float(position.get("sl_dist_pct", 0)),
        "atr_at_entry": float(position.get("atr_at_entry", 0)),
        "htf_bias_at_entry": int(position.get("htf_bias_at_entry", 0)),
        "fee_in": float(position.get("fee_in", 0)),
        "entry_fee_type": position.get("entry_fee_type", "taker"),
        "confidence": float(position.get("confidence", 1.0)),
        "sl_order_id": position.get("sl_order_id"),
    }


def _validate_state_payload(raw: dict) -> dict:
    if not isinstance(raw, dict):
        raise StateValidationError("state payload must be an object")

    version = int(raw.get("version", 1))
    if version < 1:
        raise StateValidationError(f"unsupported state version: {version}")

    try:
        equity = float(raw["equity"])
        peak_equity = float(raw["peak_equity"])
        wins = int(raw["wins"])
        losses = int(raw["losses"])
    except KeyError as exc:
        raise StateValidationError(f"state missing field: {exc.args[0]}") from exc

    strategy_state = raw.get("strategy_state", {})
    if not isinstance(strategy_state, dict):
        raise StateValidationError("strategy_state must be an object")

    return {
        "version": version,
        "position": _validate_position_payload(raw.get("position")),
        "equity": equity,
        "peak_equity": peak_equity,
        "wins": wins,
        "losses": losses,
        "strategy_state": strategy_state,
        "runner_status": raw.get("runner_status", "flat"),
        "safe_mode_reason": raw.get("safe_mode_reason"),
        "last_updated": raw.get("last_updated"),
        "symbol": raw.get("symbol"),
        "timeframe": raw.get("timeframe"),
        "mode": raw.get("mode"),
    }


def _runner_status(state: dict) -> str:
    status = state.get("runner_status")
    if status in {"starting", "recovering", "safe_mode", "error", "stopped"}:
        return status
    return "in_position" if state.get("open_trade") else "flat"


def _update_runner_status(state: dict, status: str, *, note: str | None = None) -> None:
    state["runner_status"] = status
    if note:
        state["last_error"] = note


def _heartbeat_payload(state: dict, symbol: str, mode: str) -> dict:
    open_trade = state.get("open_trade")
    return {
        "updated_at": _utc_now().isoformat(),
        "status": _runner_status(state),
        "symbol": symbol,
        "mode": mode,
        "equity": round(float(state.get("equity", 0.0)), 2),
        "open_trade": {
            "side": open_trade["side"],
            "entry_price": float(open_trade["entry_price"]),
            "qty": float(open_trade["qty"]),
            "sl": float(open_trade["sl"]),
            "tp": float(open_trade["tp"]),
            "sl_order_id": open_trade.get("sl_order_id"),
        } if open_trade else None,
        "last_bar_ts": _iso_or_none(state.get("last_bar_ts")),
        "last_progress_at": _iso_or_none(state.get("last_progress_at")),
        "last_reconcile_at": _iso_or_none(state.get("last_reconcile_at")),
        "safe_mode_reason": state.get("safe_mode_reason"),
        "last_error": state.get("last_error"),
        "api_error_streak": int(state.get("api_error_streak", 0)),
    }


def _write_heartbeat(state: dict, symbol: str, mode: str) -> None:
    _atomic_json_write(HEARTBEAT_FILE, _heartbeat_payload(state, symbol, mode))


def _mark_progress(state: dict) -> None:
    state["last_progress_at"] = _utc_now()


def _record_api_success(state: dict) -> None:
    state["api_error_streak"] = 0


def _record_api_error(state: dict, *, context: str, error: Exception) -> int:
    streak = int(state.get("api_error_streak", 0)) + 1
    state["api_error_streak"] = streak
    state["last_error"] = f"{context}: {error}"
    return streak


def _expected_stop_exit_side(position_side: str) -> str:
    return "SELL" if position_side.upper() == "LONG" else "BUY"


class RunnerInstanceLock:
    def __init__(self, path: Path, metadata: dict[str, str]) -> None:
        self._path = path
        self._metadata = metadata
        self._fd: int | None = None

    def acquire(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self._path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            try:
                with open(self._path, "r") as handle:
                    existing = handle.read().strip()
            except OSError:
                existing = "unreadable lock metadata"
            os.close(fd)
            raise InstanceLockError(
                f"runner lock already held at {self._path}: {existing or 'no metadata'}"
            ) from exc

        os.ftruncate(fd, 0)
        os.write(fd, json.dumps(self._metadata, indent=2).encode("utf-8"))
        os.fsync(fd)
        self._fd = fd

    def release(self) -> None:
        if self._fd is None:
            return
        try:
            fcntl.flock(self._fd, fcntl.LOCK_UN)
        finally:
            os.close(self._fd)
            self._fd = None
            try:
                self._path.unlink()
            except FileNotFoundError:
                pass


def _build_instance_lock(symbol: str, timeframe: str, mode: str, api_key: str) -> RunnerInstanceLock:
    material = f"{api_key}|{symbol}|{timeframe}|{mode}"
    digest = hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]
    lock_path = LOCKS_DIR / f"runner_{digest}.lock"
    metadata = {
        "symbol": symbol,
        "timeframe": timeframe,
        "mode": mode,
        "pid": str(os.getpid()),
        "acquired_at": _utc_now().isoformat(),
        "lock_key": digest,
    }
    return RunnerInstanceLock(lock_path, metadata)


def _save_state(state: dict, open_trade: dict | None, equity: float,
                peak_equity: float, wins: int, losses: int, strategy_state: dict | None = None):
    """Persist bot state to disk for crash recovery."""
    data = {
        "version": STATE_SCHEMA_VERSION,
        "position": None,
        "equity": equity,
        "peak_equity": peak_equity,
        "wins": wins,
        "losses": losses,
        "strategy_state": strategy_state or {},
        "runner_status": _runner_status(state),
        "safe_mode_reason": state.get("safe_mode_reason"),
        "symbol": state.get("symbol"),
        "timeframe": state.get("timeframe"),
        "mode": state.get("mode"),
        "last_updated": _utc_now().isoformat(),
    }
    if open_trade:
        data["position"] = {
            "side": open_trade["side"],
            "entry_price": float(open_trade["entry_price"]),
            "sl": float(open_trade["sl"]),
            "tp": float(open_trade["tp"]),
            "qty": float(open_trade["qty"]),
            "entry_time": open_trade.get("entry_time", ""),
            "trade_id": open_trade.get("trade_id", ""),
            "adx_at_entry": open_trade.get("adx_at_entry", 0),
            "sl_mode": open_trade.get("sl_mode", "unknown"),
            "leverage": open_trade.get("leverage", 0),
            "sl_dist_pct": open_trade.get("sl_dist_pct", 0),
            "atr_at_entry": open_trade.get("atr_at_entry", 0),
            "htf_bias_at_entry": open_trade.get("htf_bias_at_entry", 0),
            "fee_in": float(open_trade.get("fee_in", 0)),
            "entry_fee_type": open_trade.get("entry_fee_type", "taker"),
            "confidence": open_trade.get("confidence", 1.0),
            "sl_order_id": open_trade.get("sl_order_id"),
        }
    _atomic_json_write(STATE_FILE, data)


def _load_state() -> dict | None:
    """Load saved state. Returns None if no state file."""
    if not STATE_FILE.exists():
        return None
    try:
        with open(STATE_FILE, "r") as f:
            raw = json.load(f)
        return _validate_state_payload(raw)
    except (json.JSONDecodeError, IOError, StateValidationError) as exc:
        quarantined = _quarantine_file(STATE_FILE, "corrupt")
        if quarantined:
            _log_error(f"  [STATE] Invalid state file moved to {quarantined}: {exc}")
        return None


def _clear_state():
    """Remove state file after clean close."""
    if STATE_FILE.exists():
        STATE_FILE.unlink()


def _strategy_state(strategy) -> dict:
    if hasattr(strategy, "export_runtime_state"):
        try:
            return strategy.export_runtime_state()
        except Exception:
            return {}
    return {}


def _restore_strategy(strategy, saved_state: dict | None, open_trade: dict | None) -> None:
    start_bar = None
    if saved_state and hasattr(strategy, "restore_runtime_state"):
        strategy.restore_runtime_state(saved_state)
        trade = saved_state.get("trade") or {}
        start_bar = trade.get("start_bar")
    if open_trade and hasattr(strategy, "restore_open_trade"):
        strategy.restore_open_trade(
            side=open_trade["side"],
            entry_price=float(open_trade["entry_price"]),
            stop_loss=float(open_trade["sl"]),
            take_profit=float(open_trade["tp"]),
            start_bar=start_bar,
        )


def _decimal_close(left: Decimal, right: Decimal, *, rel_tol: Decimal = Decimal("0.001")) -> bool:
    if left == right:
        return True
    base = max(abs(left), abs(right), Decimal("1"))
    return abs(left - right) <= base * rel_tol


def _relevant_stop_orders(open_orders: list[dict], side: str) -> tuple[list[dict], list[dict]]:
    position_side = side.upper()
    relevant: list[dict] = []
    unexpected: list[dict] = []
    for order in open_orders:
        order_type = str(order.get("type", order.get("origType", ""))).upper()
        order_pos_side = str(order.get("positionSide", position_side)).upper()
        if order_pos_side != position_side:
            unexpected.append(order)
            continue
        if "STOP" in order_type:
            relevant.append(order)
        else:
            unexpected.append(order)
    return relevant, unexpected


def _extract_order_qty(order: dict) -> Decimal | None:
    for key in ("origQty", "quantity", "qty"):
        value = order.get(key)
        if value is None:
            continue
        try:
            return Decimal(str(value))
        except Exception:
            return None
    return None


def _validate_stop_order_consistency(
    stop_order: dict,
    *,
    side: str,
    entry_price: Decimal,
    local_sl: Decimal,
    exchange_qty: Decimal,
) -> tuple[str, str]:
    stop_price = Decimal(str(stop_order.get("stopPrice", stop_order.get("price", 0))))
    if stop_price <= 0:
        raise SafeModeRequired(
            "invalid_exchange_stop",
            details=[f"Exchange stop order is missing stopPrice: {stop_order!r}"],
        )

    local_side = _expected_stop_exit_side(side)
    order_side = str(stop_order.get("side", local_side)).upper()
    if order_side != local_side:
        raise SafeModeRequired(
            "invalid_exchange_stop_side",
            details=[f"Expected stop side {local_side}, found {order_side}"],
        )

    order_pos_side = str(stop_order.get("positionSide", side)).upper()
    if order_pos_side != side:
        raise SafeModeRequired(
            "invalid_exchange_stop_position_side",
            details=[f"Expected stop positionSide {side}, found {order_pos_side}"],
        )

    order_qty = _extract_order_qty(stop_order)
    if order_qty is not None and not _decimal_close(order_qty, exchange_qty):
        raise SafeModeRequired(
            "exchange_stop_qty_mismatch",
            details=[f"Expected stop qty {exchange_qty}, found {order_qty}"],
        )

    if side == "LONG" and stop_price >= entry_price:
        raise SafeModeRequired(
            "invalid_exchange_stop_level",
            details=[f"LONG stop must be below entry. entry={entry_price} stop={stop_price}"],
        )
    if side == "SHORT" and stop_price <= entry_price:
        raise SafeModeRequired(
            "invalid_exchange_stop_level",
            details=[f"SHORT stop must be above entry. entry={entry_price} stop={stop_price}"],
        )

    if not _decimal_close(local_sl, stop_price):
        raise SafeModeRequired(
            "exchange_local_stop_mismatch",
            details=[f"Stop mismatch during live recovery: local={local_sl} exchange={stop_price}"],
        )

    stop_id = (
        stop_order.get("orderId")
        or stop_order.get("orderID")
        or stop_order.get("clientOrderID")
    )
    if not stop_id:
        raise SafeModeRequired(
            "invalid_exchange_stop",
            details=[f"Stop order is missing an order id: {stop_order!r}"],
        )
    return str(stop_id), str(stop_price)


async def _reconcile_live_position(
    adapter,
    client,
    symbol: str,
    open_trade: dict | None,
    *,
    phase: str = "boot",
) -> tuple[dict | None, dict]:
    """
    Reconcile local open_trade with the exchange at startup.

    Conservative rules:
      - never assume a local close if the exchange still shows an open position
      - any ambiguous divergence enters safe mode for manual intervention
    """
    try:
        positions = await adapter.get_positions(symbol)
        open_orders = await client.get_open_orders(symbol)
    except Exception as e:
        raise SafeModeRequired(
            "exchange_reconcile_failed",
            details=[f"Could not read positions/orders from exchange: {e}"],
        ) from e

    report = {
        "status": "exchange_flat",
        "messages": [
            f"Exchange positions={len(positions)} open_orders={len(open_orders)} for {symbol}"
        ],
    }

    if not positions:
        if open_orders:
            try:
                await client.cancel_all_open_orders(symbol)
                report["messages"].append(
                    f"Cancelled {len(open_orders)} stale open order(s) while exchange was flat."
                )
                report["status"] = "exchange_flat_orders_cancelled"
            except Exception as e:
                raise SafeModeRequired(
                    "exchange_flat_with_stale_orders",
                    details=[f"Exchange is flat but stale orders could not be cancelled: {e}"],
                ) from e
        if open_trade is not None:
            if phase != "boot":
                raise SafeModeRequired(
                    "runtime_exchange_flat_with_local_position",
                    details=["Runtime reconcile found no exchange position but local trade is still open."],
                )
            report["messages"].append("Local persisted position cleared because exchange is flat.")
            report["status"] = "saved_position_cleared_exchange_flat"
        return None, report

    if len(positions) != 1:
        raise SafeModeRequired(
            "multiple_exchange_positions",
            details=[f"Expected 1 exchange position, found {len(positions)}"],
        )

    pos = positions[0]
    side = pos.side.value if hasattr(pos.side, "value") else str(pos.side)
    side = side.upper()
    entry_price = Decimal(str(pos.avg_price))
    qty = Decimal(str(pos.qty))
    report["messages"].append(
        f"Exchange position detected: side={side} qty={qty} entry={entry_price}"
    )

    if open_trade is None:
        raise SafeModeRequired(
            "exchange_position_without_local_state",
            details=[
                f"Exchange has open {side} position for {symbol} but no persisted local state."
            ],
        )

    if open_trade["side"] != side:
        raise SafeModeRequired(
            "exchange_local_side_mismatch",
            details=[
                f"Side mismatch during live recovery: local={open_trade['side']} exchange={side}"
            ],
        )

    local_qty = Decimal(str(open_trade["qty"]))
    if not _decimal_close(local_qty, qty):
        raise SafeModeRequired(
            "exchange_local_qty_mismatch",
            details=[f"Qty mismatch during live recovery: local={local_qty} exchange={qty}"],
        )

    stop_orders, unexpected_orders = _relevant_stop_orders(open_orders, side)
    if unexpected_orders:
        raise SafeModeRequired(
            "unexpected_open_orders",
            details=[f"Unexpected open orders detected: {unexpected_orders!r}"],
        )

    if len(stop_orders) != 1:
        reason = "missing_exchange_stop" if not stop_orders else "multiple_exchange_stops"
        raise SafeModeRequired(
            reason,
            details=[f"Expected exactly one active stop order, found {len(stop_orders)}"],
        )

    stop_order = stop_orders[0]
    local_sl = Decimal(str(open_trade["sl"]))
    stop_id, stop_price = _validate_stop_order_consistency(
        stop_order,
        side=side,
        entry_price=entry_price,
        local_sl=local_sl,
        exchange_qty=qty,
    )

    reconciled = dict(open_trade)
    reconciled["entry_price"] = entry_price
    reconciled["qty"] = qty
    reconciled["sl_order_id"] = stop_id or reconciled.get("sl_order_id")
    report["status"] = f"{phase}_exchange_position_reconciled"
    report["messages"].append(
        f"Reconciled stop order id={reconciled['sl_order_id']} stop_price={stop_price}"
    )
    return reconciled, report


async def _maybe_runtime_reconcile(
    *,
    state: dict,
    adapter,
    client,
    symbol: str,
    open_trade: dict | None,
    reconcile_interval_secs: int,
) -> tuple[dict | None, dict | None]:
    if reconcile_interval_secs <= 0:
        return open_trade, None

    now = _utc_now()
    last_reconcile = state.get("last_reconcile_at")
    if isinstance(last_reconcile, datetime):
        elapsed = (now - last_reconcile).total_seconds()
        if elapsed < reconcile_interval_secs:
            return open_trade, None

    reconciled_trade, report = await _reconcile_live_position(
        adapter,
        client,
        symbol,
        open_trade,
        phase="runtime",
    )
    state["last_reconcile_at"] = now
    return reconciled_trade, report


async def _enter_safe_mode(state: dict, symbol: str, mode: str, notifier, reason: str, details: list[str]) -> None:
    state["safe_mode_reason"] = reason
    _update_runner_status(state, "safe_mode", note=" | ".join(details) if details else reason)
    _mark_progress(state)
    _write_heartbeat(state, symbol, mode)
    _log_error(f"  [{symbol}] SAFE MODE: {reason}")
    for detail in details:
        _log_error(f"  [{symbol}]   - {detail}")
    await _tg_send(
        notifier,
        f"🚨 <b>SAFE MODE — manual intervention required</b>\n\n"
        f"Reason: <code>{reason}</code>\n"
        + ("\n".join(details) if details else "No additional details.")
    )


async def _safe_mode_wait_loop(state: dict, symbol: str, mode: str, shutdown_event) -> None:
    while not shutdown_event.is_set():
        _mark_progress(state)
        _write_heartbeat(state, symbol, mode)
        try:
            await asyncio.wait_for(shutdown_event.wait(), timeout=HEARTBEAT_INTERVAL_SECS)
            break
        except asyncio.TimeoutError:
            pass


async def _heartbeat_loop(state: dict, symbol: str, mode: str, shutdown_event) -> None:
    while not shutdown_event.is_set():
        _write_heartbeat(state, symbol, mode)
        try:
            await asyncio.wait_for(shutdown_event.wait(), timeout=HEARTBEAT_INTERVAL_SECS)
            break
        except asyncio.TimeoutError:
            pass
    _update_runner_status(state, "stopped")
    _write_heartbeat(state, symbol, mode)


async def _persist_shutdown_state(
    *,
    notifier,
    state: dict,
    strategy,
    open_trade: dict | None,
    equity: float,
    peak_equity: float,
    is_live: bool,
) -> None:
    """Persist shutdown state without fabricating local fills."""
    if open_trade:
        shutdown_note = "live_position_left_open_for_recovery" if is_live else "paper_position_preserved_for_recovery"
        _log_error(f"  [{open_trade['symbol']}] Shutdown with open trade preserved: {shutdown_note}")
        await _tg_send(
            notifier,
            f"\u26a0\ufe0f <b>BOT STOPPED WITH OPEN POSITION</b>\n\n"
            f"Side: {open_trade['side']}\n"
            f"Entry: ${float(open_trade['entry_price']):,.2f}\n"
            f"SL: ${float(open_trade['sl']):,.2f}\n"
            f"TP: ${float(open_trade['tp']):,.2f}\n"
            f"Mode: {'LIVE' if is_live else 'PAPER'}\n"
            f"State preserved for recovery on next start."
        )

    _update_runner_status(state, "stopped")
    _save_state(
        state,
        open_trade,
        equity,
        peak_equity,
        state["wins"],
        state["losses"],
        _strategy_state(strategy),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Main trading loop
# ═══════════════════════════════════════════════════════════════════════════════

async def run_symbol(symbol, client, adapter, args, shutdown_event, state, is_live: bool):
    from app.broker.base import OrderRequest, OrderSide, OrderType
    from app.data.feed import LiveFeed
    from app.data.ingestor import OHLCVIngestor, TIMEFRAME_SECONDS
    from app.data.parquet_store import ParquetStore
    from app.strategy.base import BaseStrategy
    from app.strategy.signals import SignalAction
    from app.strategy.trend_following_v2_simple import TrendFollowingV2Simple

    s = symbol.replace("-USDT", "")
    store    = ParquetStore()
    strategy = TrendFollowingV2Simple(symbol=symbol, params=VALIDATED_PARAMS)
    notifier = state.get("notifier")
    cb: CircuitBreaker = state["circuit_breaker"]

    _log(f"  [{s}] Loading warmup ({WINDOW_SIZE} bars)...")
    _update_runner_status(state, "starting")
    _mark_progress(state)
    try:
        warmup = await OHLCVIngestor(client, store).poll_latest(
            symbol, args.timeframe, WINDOW_SIZE + 2)
    except Exception as e:
        _log_error(f"  [{s}] ERROR loading warmup: {e}")
        _update_runner_status(state, "error", note=str(e))
        return

    bw: deque = deque(warmup[:-1], maxlen=WINDOW_SIZE)
    last_processed_ts = bw[-1].ts if bw else None
    if hasattr(strategy, "force_close"):
        strategy.force_close()
    _mark_progress(state)
    _log(f"  [{s}] Ready \u2014 {len(bw)} bars loaded")

    # HTF bias
    htf = SimpleHTFBias(symbol, ema_period=HTF_CONFIG["ema_period"])
    if HTF_CONFIG["enabled"]:
        await htf.warmup(client, store)
        _log(f"  [{s}] HTF: {htf.label}")
    state["htf"] = htf

    equity = float(args.balance)
    open_trade: dict | None = None
    bar_count_in_trade = 0

    # ── State recovery (crash protection) ────────────────────
    saved = _load_state()
    _update_runner_status(state, "recovering")
    if saved:
        equity = saved.get("equity", equity)
        state["equity"] = equity
        state["wins"] = saved.get("wins", 0)
        state["losses"] = saved.get("losses", 0)
        cb.peak_equity = saved.get("peak_equity", equity)
        cb.current_equity = equity
        if saved.get("position"):
            pos = saved["position"]
            open_trade = {
                "trade_id": pos.get("trade_id", "recovered"),
                "symbol": symbol,
                "side": pos["side"],
                "entry_price": Decimal(str(pos["entry_price"])),
                "qty": Decimal(str(pos["qty"])),
                "fee_in": Decimal(str(pos.get("fee_in", 0))),
                "entry_fee_type": pos.get("entry_fee_type", "taker"),
                "sl": Decimal(str(pos["sl"])),
                "tp": Decimal(str(pos["tp"])),
                "leverage": pos.get("leverage", 1.0),
                "sl_dist_pct": pos.get("sl_dist_pct", 0),
                "sl_mode": pos.get("sl_mode", "unknown"),
                "adx_at_entry": pos.get("adx_at_entry", 0),
                "atr_at_entry": pos.get("atr_at_entry", 0),
                "htf_bias_at_entry": pos.get("htf_bias_at_entry", 0),
                "entry_time": pos.get("entry_time", ""),
                "confidence": pos.get("confidence", 1.0),
                "sl_order_id": pos.get("sl_order_id"),
            }
            if is_live:
                try:
                    open_trade, reconcile = await _reconcile_live_position(adapter, client, symbol, open_trade)
                    for message in reconcile["messages"]:
                        _log(f"  [{s}] Reconcile: {message}")
                    _log(f"  [{s}] Live reconcile: {reconcile['status']}")
                    state["last_reconcile_at"] = _utc_now()
                    _record_api_success(state)
                except SafeModeRequired as exc:
                    await _enter_safe_mode(
                        state,
                        s,
                        "live",
                        notifier,
                        exc.reason,
                        exc.details,
                    )
                    await _safe_mode_wait_loop(state, s, "live", shutdown_event)
                    return
            if open_trade is not None:
                _restore_strategy(strategy, saved.get("strategy_state"), open_trade)
                state["open_trade"] = open_trade
                _update_runner_status(state, "in_position")
                _save_state(
                    state,
                    open_trade,
                    equity,
                    cb.peak_equity,
                    state["wins"],
                    state["losses"],
                    _strategy_state(strategy),
                )
                _log(f"  [{s}] RECOVERED position: {open_trade['side']} @ ${float(open_trade['entry_price']):,.2f}"
                     f" SL=${float(open_trade['sl']):,.2f} TP=${float(open_trade['tp']):,.2f}")
                await _tg_send(notifier,
                    f"\u26a0\ufe0f <b>POSITION RECOVERED</b>\n\n"
                    f"Side: {open_trade['side']}\n"
                    f"Entry: ${float(open_trade['entry_price']):,.2f}\n"
                    f"SL: ${float(open_trade['sl']):,.2f}\n"
                    f"TP: ${float(open_trade['tp']):,.2f}\n"
                    f"Equity: ${equity:,.2f}")
            else:
                state["open_trade"] = None
                if hasattr(strategy, "force_close"):
                    strategy.force_close()
                _log(f"  [{s}] Saved position cleared during recovery")
        else:
            _restore_strategy(strategy, saved.get("strategy_state"), None)
            _log(f"  [{s}] State recovered (no open position)")
            _log(f"  [{s}] Equity=${equity:,.2f} W={state['wins']} L={state['losses']}")
    elif is_live:
        try:
            open_trade, reconcile = await _reconcile_live_position(adapter, client, symbol, None)
            for message in reconcile["messages"]:
                _log(f"  [{s}] Reconcile: {message}")
            _log(f"  [{s}] Live reconcile: {reconcile['status']}")
            state["last_reconcile_at"] = _utc_now()
            _record_api_success(state)
        except SafeModeRequired as exc:
            await _enter_safe_mode(
                state,
                s,
                "live",
                notifier,
                exc.reason,
                exc.details,
            )
            await _safe_mode_wait_loop(state, s, "live", shutdown_event)
            return

    _update_runner_status(state, "in_position" if open_trade else "flat")
    _mark_progress(state)

    tf_secs = TIMEFRAME_SECONDS.get(args.timeframe, 900)
    feed = LiveFeed(client=client, store=store, symbol=symbol,
                    timeframe=args.timeframe, max_data_delay_s=tf_secs * 3)
    feed_iter = feed.stream().__aiter__()

    try:
        while not shutdown_event.is_set():
            # Wait for next bar or shutdown
            get_bar = asyncio.ensure_future(feed_iter.__anext__())
            wait_sd = asyncio.ensure_future(shutdown_event.wait())
            done, pending = await asyncio.wait(
                [get_bar, wait_sd], return_when=asyncio.FIRST_COMPLETED)
            for t in pending:
                t.cancel()
                try:
                    await t
                except (asyncio.CancelledError, StopAsyncIteration):
                    pass
            if wait_sd in done:
                break
            try:
                bar = get_bar.result()
            except StopAsyncIteration:
                break

            if last_processed_ts is not None and bar.ts <= last_processed_ts:
                continue
            last_processed_ts = bar.ts
            state["last_bar_ts"] = bar.ts
            _mark_progress(state)

            bw.append(bar)
            if len(bw) < strategy.min_bars_required:
                continue

            # ── Circuit breaker check ────────────────────────────
            if cb.stopped:
                continue

            # ── Refresh HTF bias ─────────────────────────────────
            if HTF_CONFIG["enabled"]:
                try:
                    if await htf.maybe_refresh(client, store, bar.ts):
                        _log(f"  [{s}] HTF updated: {htf.label}")
                except Exception:
                    pass

            # ── Run strategy ─────────────────────────────────────
            df = BaseStrategy.bars_to_df(list(bw))
            try:
                signals = strategy.on_bar_all(df)
            except Exception as e:
                _log_error(f"  [{s}] Strategy error: {e}")
                continue

            # ── Refresh equity (paper mode: track manually) ──────
            if not is_live:
                pass
            else:
                try:
                    balances = await adapter.get_balance()
                    equity = float(balances[0].total) if balances else equity
                    _record_api_success(state)
                except Exception as e:
                    streak = _record_api_error(state, context="get_balance", error=e)
                    if streak >= MAX_API_ERROR_STREAK:
                        await _enter_safe_mode(
                            state,
                            s,
                            "live",
                            notifier,
                            "repeated_api_errors",
                            [state["last_error"]],
                        )
                        await _safe_mode_wait_loop(state, s, "live", shutdown_event)
                        return

                try:
                    reconciled_trade, reconcile = await _maybe_runtime_reconcile(
                        state=state,
                        adapter=adapter,
                        client=client,
                        symbol=symbol,
                        open_trade=open_trade,
                        reconcile_interval_secs=args.reconcile_secs,
                    )
                    if reconcile is not None:
                        for message in reconcile["messages"]:
                            _log(f"  [{s}] Runtime reconcile: {message}")
                        _log(f"  [{s}] Runtime reconcile: {reconcile['status']}")
                        if reconciled_trade != open_trade:
                            open_trade = reconciled_trade
                            state["open_trade"] = open_trade
                            _save_state(
                                state,
                                open_trade,
                                equity,
                                cb.peak_equity,
                                state["wins"],
                                state["losses"],
                                _strategy_state(strategy),
                            )
                        _record_api_success(state)
                except SafeModeRequired as exc:
                    await _enter_safe_mode(
                        state,
                        s,
                        "live",
                        notifier,
                        exc.reason,
                        exc.details,
                    )
                    await _safe_mode_wait_loop(state, s, "live", shutdown_event)
                    return

            state["equity"] = equity
            close = float(bar.close)

            # ── Update circuit breaker ───────────────────────────
            alert = cb.update(equity)
            if alert:
                if cb.stopped:
                    _log_error(f"\n  !!! CRITICAL: {alert}\n")
                    await _tg_send(notifier,
                        f"\U0001f6a8 <b>CIRCUIT BREAKER TRIGGERED</b>\n\n"
                        f"{alert}\n\nBot stopped. Manual review required.\n"
                        f"Send /stop to confirm shutdown.")
                    break
                else:
                    _log(f"\n  !!! WARNING: {alert}\n")
                    await _tg_send(notifier,
                        f"\u26a0\ufe0f <b>CIRCUIT BREAKER WARNING</b>\n\n"
                        f"DD: {cb.drawdown_pct:.1f}% from peak\n"
                        f"Position size reduced to {cb.risk_pct:.2f}% per trade")

            # Track bars in trade + update shared state
            if open_trade is not None:
                bar_count_in_trade += 1
                upnl = (close - float(open_trade["entry_price"])) if open_trade["side"] == "LONG" \
                    else (float(open_trade["entry_price"]) - close)
                state["upnl_pct"] = upnl / float(open_trade["entry_price"]) * 100
            else:
                state["upnl_pct"] = 0.0
            state["open_trade"] = open_trade
            _update_runner_status(state, "in_position" if open_trade else "flat")

            # ── Print bar status ─────────────────────────────────
            ts_str = bar.ts.strftime("%H:%M")
            meta   = signals[0].meta if signals else {}
            adx    = meta.get("adx", 0)
            primary = signals[0] if signals else None
            action = primary.action.value if primary else "HOLD"

            pos_str = ""
            if open_trade:
                pos_str = f"  [{open_trade['side']} {bar_count_in_trade}b uPnL={state['upnl_pct']:+.2f}%]"

            htf_str = f"HTF={'B' if htf.bias == 1 else 'S' if htf.bias == -1 else 'N'}" if HTF_CONFIG["enabled"] else ""

            if action == "HOLD":
                _log(f"  {ts_str} {s:<4} ${close:>10,.2f} ADX={adx:.1f} {htf_str}  DD={cb.drawdown_pct:.1f}%{pos_str}")
            else:
                _log(f"  {ts_str} {s:<4} ${close:>10,.2f} ADX={adx:.1f} {htf_str}  *** {action} ***{pos_str}")

            # ── Execute signals ──────────────────────────────────
            for sig in signals:
                if not sig.is_actionable():
                    continue

                # ── CLOSE ────────────────────────────────────────
                if sig.action == SignalAction.CLOSE:
                    if open_trade is None:
                        continue

                    exit_price = sig.meta.get("exit_price", close)
                    exit_type = sig.meta.get("exit_type", "close")
                    il = open_trade["side"] == "LONG"

                    if not is_live:
                        # TP → limit order (maker fee); SL/close → market (taker fee)
                        exit_order_type = "limit" if exit_type == "tp" else "market"
                        fill = _make_paper_fill(
                            symbol, exit_price, open_trade["qty"],
                            OrderSide.SELL if il else OrderSide.BUY,
                            order_type=exit_order_type)
                    else:
                        try:
                            _, fill = await adapter.place_order(OrderRequest(
                                symbol=symbol,
                                side=OrderSide.SELL if il else OrderSide.BUY,
                                order_type=OrderType.MARKET,
                                qty=open_trade["qty"],
                                strategy_id=strategy.strategy_id,
                                extra={"positionSide": "LONG" if il else "SHORT"},
                            ))
                            _record_api_success(state)
                        except Exception as e:
                            _log_error(f"  [{s}] CLOSE failed: {e}")
                            _record_api_error(state, context="close_position", error=e)
                            continue

                    if fill:
                        # Cancel exchange SL if live
                        if is_live and open_trade.get("sl_order_id"):
                            try:
                                await client.cancel_order(open_trade["sl_order_id"], symbol)
                            except Exception:
                                pass

                        # Calculate PnL
                        entry_px = open_trade["entry_price"]
                        qty_d = open_trade["qty"]
                        if il:
                            gross_pnl = (fill.price - entry_px) * qty_d
                        else:
                            gross_pnl = (entry_px - fill.price) * qty_d
                        net_pnl = gross_pnl - open_trade["fee_in"] - fill.fee
                        pnl_pct = float(net_pnl) / (float(entry_px) * float(qty_d)) * 100 if entry_px * qty_d else 0

                        # Update equity
                        equity += float(net_pnl)
                        state["equity"] = equity

                        state["wins" if net_pnl > 0 else "losses"] += 1
                        w, l = state["wins"], state["losses"]

                        emoji = "\u2705" if exit_type == "tp" else "\u274c"
                        _log(
                            f"\n  {emoji} {exit_type.upper()} {s} @ ${float(fill.price):,.2f}"
                            f"  PnL={pnl_pct:+.2f}% (${float(net_pnl):+.2f})"
                            f"  Eq=${equity:,.2f}  W={w} L={l}\n"
                        )

                        # Log CSV
                        _append_csv({
                            "timestamp": fill.timestamp.isoformat(),
                            "symbol": symbol,
                            "side": open_trade["side"],
                            "entry_price": float(entry_px),
                            "exit_price": float(fill.price),
                            "exit_type": exit_type,
                            "sl": float(open_trade["sl"]),
                            "tp": float(open_trade["tp"]),
                            "pnl_pct": round(pnl_pct, 4),
                            "pnl_usd": round(float(net_pnl), 2),
                            "equity_after": round(equity, 2),
                            "leverage_used": round(open_trade["leverage"], 2),
                            "sl_dist_pct": round(open_trade["sl_dist_pct"] * 100, 4),
                            "adx_at_entry": open_trade["adx_at_entry"],
                            "atr_at_entry": open_trade["atr_at_entry"],
                            "htf_bias_at_entry": open_trade["htf_bias_at_entry"],
                            "duration_bars": bar_count_in_trade,
                            "entry_fee_type": open_trade.get("entry_fee_type", "taker"),
                            "exit_fee_type": "maker" if exit_type == "tp" else "taker",
                            "sl_mode": open_trade.get("sl_mode", ""),
                            "confidence": open_trade.get("confidence", ""),
                        })

                        # Telegram — rich close notification
                        dur_min = bar_count_in_trade * 15
                        dur_h = dur_min // 60
                        dur_m = dur_min % 60
                        dur_str = f"{dur_h}h {dur_m}m" if dur_h > 0 else f"{dur_m}m"
                        tp_or_sl = "TP HIT" if exit_type == "tp" else "SL HIT"
                        pnl_sign = "+" if net_pnl > 0 else ""
                        wr_str = f"{w/(w+l)*100:.0f}%" if (w + l) > 0 else "\u2014"

                        await _tg_send(notifier,
                            f"{'✅' if exit_type == 'tp' else '❌'} <b>{tp_or_sl} \u2014 {symbol}</b>\n\n"
                            f"\U0001f4b5 Exit:     <code>${float(fill.price):,.2f}</code>\n"
                            f"\U0001f4cd Entry:    <code>${float(entry_px):,.2f}</code>\n"
                            f"\U0001f4b0 P&L:      <code>{pnl_sign}{pnl_pct:.2f}% ({pnl_sign}${float(net_pnl):,.2f})</code>\n"
                            f"\u23f1 Duration: {bar_count_in_trade} bars (~{dur_str})\n"
                            f"\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n"
                            f"\U0001f4bc Equity:   <code>${equity:,.2f}</code>\n"
                            f"\U0001f3d4 Peak:     <code>${cb.peak_equity:,.2f}</code>\n"
                            f"\U0001f4c9 DD/Peak:  {cb.drawdown_pct:.1f}%\n"
                            f"\U0001f4ca Session:  W={w} L={l} WR={wr_str}")

                        open_trade = None
                        bar_count_in_trade = 0
                        state["open_trade"] = None
                        state["upnl_pct"] = 0.0

                        # Persist state (position closed)
                        _save_state(state, None, equity,
                                    cb.peak_equity, state["wins"], state["losses"],
                                    _strategy_state(strategy))

                # ── BUY / SELL ───────────────────────────────────
                elif sig.action in (SignalAction.BUY, SignalAction.SELL):
                    direction = "LONG" if sig.action == SignalAction.BUY else "SHORT"

                    # If position open in opposite direction: close first
                    if open_trade is not None:
                        if open_trade["side"] == direction:
                            continue  # same direction, ignore
                        # Close existing position at market
                        il = open_trade["side"] == "LONG"
                        if not is_live:
                            close_fill = _make_paper_fill(
                                symbol, close, open_trade["qty"],
                                OrderSide.SELL if il else OrderSide.BUY)
                        else:
                            try:
                                _, close_fill = await adapter.place_order(OrderRequest(
                                    symbol=symbol,
                                    side=OrderSide.SELL if il else OrderSide.BUY,
                                    order_type=OrderType.MARKET,
                                    qty=open_trade["qty"],
                                    strategy_id=strategy.strategy_id,
                                    extra={"positionSide": "LONG" if il else "SHORT"},
                                ))
                                _record_api_success(state)
                            except Exception as e:
                                _log_error(f"  [{s}] Close-before-reverse failed: {e}")
                                _record_api_error(state, context="close_before_reverse", error=e)
                                continue

                        if close_fill:
                            entry_px = open_trade["entry_price"]
                            qty_d = open_trade["qty"]
                            if il:
                                gross_pnl = (close_fill.price - entry_px) * qty_d
                            else:
                                gross_pnl = (entry_px - close_fill.price) * qty_d
                            net_pnl = gross_pnl - open_trade["fee_in"] - close_fill.fee
                            equity += float(net_pnl)
                            state["equity"] = equity
                            state["wins" if net_pnl > 0 else "losses"] += 1
                            _log(f"  [{s}] Closed {open_trade['side']} for reversal: ${float(net_pnl):+.2f}")
                            open_trade = None
                            bar_count_in_trade = 0

                    # HTF filter: reject signals against the 4H trend
                    if HTF_CONFIG["enabled"] and not htf.is_aligned(direction):
                        _log(f"  [{s}] Signal {direction} rejected \u2014 HTF bias={htf.label}")
                        continue

                    # Position sizing — structural SL override
                    atr_val = float(sig.meta.get("atr", 0))
                    adx_val = float(sig.meta.get("adx", 0))

                    if atr_val <= 0:
                        _log(f"  [{s}] No ATR in signal — skipping")
                        continue

                    if sig.stop_loss is None or sig.take_profit is None:
                        _log(f"  [{s}] Signal missing stop_loss/take_profit — skipping")
                        continue

                    sl_price = float(sig.stop_loss)
                    tp_price = float(sig.take_profit)
                    sl_mode = sig.meta.get("sl_mode", "unknown")
                    sl_dist = abs(close - sl_price)

                    if sl_price <= 0:
                        _log(f"  [{s}] Invalid SL={sl_price} — skipping")
                        continue

                    qty, position_usd, leverage = compute_position(
                        equity=equity,
                        entry_price=close,
                        sl_price=sl_price,
                        risk_pct=cb.risk_pct,
                        max_leverage=RISK_CONFIG["max_leverage"],
                    )
                    qty_dec = Decimal(str(round(qty, 4)))
                    if qty_dec < Decimal("0.001"):
                        _log(f"  [{s}] Qty too small: {qty_dec}")
                        continue

                    sl_dist_pct = abs(close - sl_price) / close if close > 0 else 0

                    # Execute
                    il = sig.action == SignalAction.BUY
                    side = OrderSide.BUY if il else OrderSide.SELL

                    if not is_live:
                        # Entry = limit order (maker fee)
                        fill = _make_paper_fill(symbol, close, qty_dec, side,
                                                 order_type="limit")
                    else:
                        try:
                            _, fill = await adapter.place_order(OrderRequest(
                                symbol=symbol, side=side,
                                order_type=OrderType.MARKET, qty=qty_dec,
                                strategy_id=strategy.strategy_id,
                                extra={"positionSide": "LONG" if il else "SHORT"},
                            ))
                            _record_api_success(state)
                        except Exception as e:
                            _log_error(f"  [{s}] ORDER FAILED: {e}")
                            _record_api_error(state, context="place_entry_order", error=e)
                            fill = None

                    if fill:
                        open_trade = {
                            "trade_id": str(uuid.uuid4())[:8],
                            "symbol": symbol,
                            "side": direction,
                            "entry_price": fill.price,
                            "qty": fill.qty,
                            "fee_in": fill.fee,
                            "entry_fee_type": "maker" if not is_live else "taker",
                            "sl": Decimal(str(sl_price)),
                            "tp": Decimal(str(tp_price)),
                            "leverage": leverage,
                            "sl_dist_pct": sl_dist_pct,
                            "sl_mode": sl_mode,
                            "adx_at_entry": round(adx_val, 1),
                            "atr_at_entry": round(atr_val, 2),
                            "htf_bias_at_entry": htf.bias,
                            "entry_time": fill.timestamp.isoformat(),
                            "confidence": sig.meta.get("confidence_score", sig.confidence),
                            "sl_order_id": None,
                        }
                        bar_count_in_trade = 0
                        state["open_trade"] = open_trade

                        # Persist state (crash recovery)
                        _save_state(state, open_trade, equity,
                                    cb.peak_equity, state["wins"], state["losses"],
                                    _strategy_state(strategy))

                        # Place exchange SL (crash protection) in live mode
                        if is_live and sl_price > 0:
                            try:
                                sl_side = "SELL" if il else "BUY"
                                sl_resp = await client.place_stop_order(
                                    symbol=symbol,
                                    side=sl_side,
                                    stop_price=sl_price,
                                    qty=float(fill.qty),
                                    pos_side="LONG" if il else "SHORT",
                                    client_order_id=f"sl{open_trade['trade_id']}",
                                )
                                sl_order_id = (sl_resp or {}).get("orderId") or (sl_resp or {}).get("order", {}).get("orderId")
                                if not sl_order_id:
                                    raise SafeModeRequired(
                                        "missing_exchange_stop",
                                        details=["Stop order placed but no orderId was returned by BingX."],
                                    )
                                open_trade["sl_order_id"] = sl_order_id
                                _save_state(state, open_trade, equity,
                                            cb.peak_equity, state["wins"], state["losses"],
                                            _strategy_state(strategy))
                                _log(f"  [{s}] SL order @ ${sl_price:,.2f}  id={sl_order_id}")
                                _record_api_success(state)
                            except SafeModeRequired as exc:
                                await _enter_safe_mode(state, s, "live", notifier, exc.reason, exc.details)
                                await _safe_mode_wait_loop(state, s, "live", shutdown_event)
                                return
                            except Exception as e:
                                _log_error(f"  [{s}] FATAL: SL order failed: {e}")
                                _record_api_error(state, context="place_stop_order", error=e)
                                await _enter_safe_mode(
                                    state,
                                    s,
                                    "live",
                                    notifier,
                                    "missing_exchange_stop",
                                    [f"Failed to place exchange stop after entry fill: {e}"],
                                )
                                await _safe_mode_wait_loop(state, s, "live", shutdown_event)
                                return

                        sl_pct = sl_dist_pct * 100
                        tp_dist_pct = abs(tp_price - close) / close * 100 if close > 0 and tp_price > 0 else 0
                        bar_ts_str = bar.ts.strftime("%H:%M UTC")

                        _log(
                            f"\n  {'🟢' if il else '🔴'} {direction} {s} @ ${float(fill.price):,.2f}"
                            f"\n     SL=${sl_price:,.2f} (-{sl_pct:.2f}%)"
                            f"  TP=${tp_price:,.2f} (+{tp_dist_pct:.2f}%)"
                            f"\n     Risk={cb.risk_pct:.1f}% (${equity*cb.risk_pct/100:,.2f})"
                            f"  Lev={leverage:.1f}x  ATR=${atr_val:,.2f}  ADX={adx_val:.1f}  SL_mode={sl_mode}\n"
                        )

                        # Telegram — rich open notification
                        await _tg_send(notifier,
                            f"{'🟢' if il else '🔴'} <b>{direction} {symbol}</b>\n\n"
                            f"\U0001f4b5 Entry:    <code>${float(fill.price):,.2f}</code>\n"
                            f"\U0001f6e1 SL:       <code>${sl_price:,.2f} (-{sl_pct:.1f}%)</code>\n"
                            f"\U0001f3af TP:       <code>${tp_price:,.2f} (+{tp_dist_pct:.1f}%)</code>\n"
                            f"\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n"
                            f"\u26a1 Risk:     {cb.risk_pct:.1f}% (${equity*cb.risk_pct/100:,.2f})\n"
                            f"\U0001f4ca Leverage: {leverage:.1f}x\n"
                            f"\U0001f9ed HTF Bias: {htf.bias_emoji()}\n"
                            f"\U0001f4c8 ADX:      {adx_val:.1f} | ATR: ${atr_val:,.2f}\n"
                            f"\U0001f4b0 Equity:   <code>${equity:,.2f}</code>\n"
                            f"\u23f0 Bar time: {bar_ts_str}")

    except asyncio.CancelledError:
        pass
    except Exception as e:
        import traceback
        _update_runner_status(state, "error", note=str(e))
        _log_error(f"  [{s}] Error: {e}")
        traceback.print_exc()
    finally:
        feed.stop()
        await _persist_shutdown_state(
            notifier=notifier,
            state=state,
            strategy=strategy,
            open_trade=open_trade,
            equity=equity,
            peak_equity=cb.peak_equity,
            is_live=is_live,
        )

# ═══════════════════════════════════════════════════════════════════════════════
# Weekly Report
# ═══════════════════════════════════════════════════════════════════════════════

def _weekly_report_text(state) -> str:
    """Generate weekly performance report text."""
    import math
    w, l = state.get("wins", 0), state.get("losses", 0)
    t = w + l
    eq = state.get("equity", 0)
    start = state.get("start_balance", eq)
    cb: CircuitBreaker = state["circuit_breaker"]

    wr_actual = w / t if t > 0 else 0
    wr_pct = wr_actual * 100

    # Backtest reference values
    wr_bt = 0.368  # 36.8% from structural rr=2.5 validation
    expr_bt = 0.335  # +0.335R

    # CI for WR
    if t > 0:
        margin = 1.96 * math.sqrt(wr_bt * (1 - wr_bt) / t)
        ci_lo = max(0, (wr_bt - margin)) * 100
        ci_hi = min(1, (wr_bt + margin)) * 100
        in_ci = ci_lo <= wr_pct <= ci_hi
    else:
        ci_lo = ci_hi = 0
        in_ci = True

    # Actual ExpR from CSV
    trades = _read_last_trades(100)
    pnls = [float(r.get("pnl_pct", 0)) for r in trades if r.get("pnl_pct")]
    losses_pnl = [p for p in pnls if p < 0]
    avg_loss = abs(sum(losses_pnl) / len(losses_pnl)) if losses_pnl else 1
    expr_actual = (sum(pnls) / len(pnls)) / avg_loss if pnls and avg_loss > 0 else 0

    net = eq - start
    net_pct = net / start * 100 if start > 0 else 0

    return (
        f"\U0001f4cb <b>WEEKLY REPORT</b>\n\n"
        f"\U0001f4b0 Equity: <code>${eq:,.2f}</code> ({'+' if net >= 0 else ''}{net_pct:.1f}%)\n"
        f"\U0001f4c9 DD from peak: {cb.drawdown_pct:.1f}%\n"
        f"\U0001f4ca Trades: {t} (W={w} L={l})\n"
        f"\n<b>Performance vs Backtest</b>\n"
        f"WR actual: {wr_pct:.1f}% vs backtest {wr_bt*100:.1f}%\n"
        f"ExpR actual: {expr_actual:+.3f}R vs backtest {expr_bt:+.3f}R\n"
        f"95% CI for WR ({t} trades): {ci_lo:.1f}% - {ci_hi:.1f}%\n"
        f"Within CI: {'YES' if in_ci else 'NO'}\n"
        f"\nExpected WR: {wr_bt*100:.1f}% +/- {(ci_hi-ci_lo)/2:.1f}pp with {t} trades"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Status loop (every 6 hours)
# ═══════════════════════════════════════════════════════════════════════════════

async def _status_loop(state, symbol, shutdown_event):
    last_weekly = datetime.now(timezone.utc)
    await asyncio.sleep(60)  # Initial delay
    while not shutdown_event.is_set():
        try:
            now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
            eq = state.get("equity", 0)
            cb: CircuitBreaker = state["circuit_breaker"]
            w, l = state.get("wins", 0), state.get("losses", 0)
            t = w + l
            wr = f" WR={w/t*100:.0f}%" if t else ""
            status = _runner_status(state)

            htf: SimpleHTFBias | None = state.get("htf")
            htf_str = htf.bias_emoji() if htf else "N/A"
            ot = state.get("open_trade")
            pos_str = "None"
            if ot:
                upnl = state.get("upnl_pct", 0)
                pos_str = f"{ot['side']} @ ${float(ot['entry_price']):,.2f} (uPnL={upnl:+.2f}%)"

            _log(f"\n{'━'*60}")
            _log(f"  STATUS [{now}]  Eq=${eq:,.2f}  Peak=${cb.peak_equity:,.2f}  DD={cb.drawdown_pct:.1f}%  State={status}")
            _log(f"  Trades={t} (W={w} L={l}{wr})  Risk/trade={cb.risk_pct:.1f}%")
            _log(f"{'━'*60}\n")

            notifier = state.get("notifier")
            await _tg_send(notifier,
                f"\U0001f4ca <b>STATUS</b>\n\n"
                f"\U0001f4b0 Equity: <code>${eq:,.2f}</code> | Peak: <code>${cb.peak_equity:,.2f}</code>\n"
                f"🫀 Runner: <code>{status}</code>\n"
                f"\U0001f4c9 DD: {cb.drawdown_pct:.1f}%\n"
                f"\U0001f4c8 Trades: {t} (W={w} L={l}{wr})\n"
                f"\U0001f504 Position: {pos_str}\n"
                f"\U0001f9ed HTF Bias: {htf_str}\n"
                f"\u26a1 Risk/trade: {cb.risk_pct:.1f}%\n"
                f"\u23f0 {now}")

            # Weekly report (every 7 days or Monday)
            now_dt = datetime.now(timezone.utc)
            days_since = (now_dt - last_weekly).total_seconds() / 86400
            if days_since >= 7 or (now_dt.weekday() == 0 and days_since >= 1):
                report = _weekly_report_text(state)
                await _tg_send(notifier, report)
                _log(f"  [Weekly report sent]")
                last_weekly = now_dt
        except Exception:
            pass
        try:
            await asyncio.wait_for(shutdown_event.wait(), timeout=STATUS_SECS)
            break
        except asyncio.TimeoutError:
            pass


# ═══════════════════════════════════════════════════════════════════════════════
# Connection check
# ═══════════════════════════════════════════════════════════════════════════════

async def check_connection(args):
    from app.config import get_settings
    from app.core.logging import configure_logging
    s = get_settings()
    configure_logging(log_level="ERROR", log_format="console")
    from app.broker.bingx_client import BingXClient
    c = BingXClient(api_key=s.bingx_api_key, api_secret=s.bingx_api_secret,
                    base_url=s.bingx_base_url, market_type=s.bingx_market_type)
    symbol = f"{args.symbol}-USDT"
    print(f"\nChecking BingX API...")
    try:
        tk = await c.get_ticker(symbol)
        print(f"  [OK] {symbol:<12} ${float(tk['last']):>12,.4f}")
    except Exception as e:
        print(f"  [FAIL] {symbol}: {e}")
    await c.close()


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

async def main(args):
    global _HEADLESS
    _HEADLESS = args.headless

    from app.config import get_settings
    from app.core.logging import configure_logging
    settings = get_settings()
    configure_logging(log_level="WARNING", log_format="console")
    from app.broker.bingx_adapter import BingXAdapter
    from app.broker.bingx_client import BingXClient
    from app.broker.paper_adapter import PaperAdapter
    from app.notify.telegram import TelegramNotifier

    _ensure_csv()
    symbol = f"{args.symbol}-USDT"

    client = BingXClient(
        api_key=settings.bingx_api_key, api_secret=settings.bingx_api_secret,
        base_url=settings.bingx_base_url, market_type=settings.bingx_market_type,
    )

    is_live = args.live
    if is_live:
        # Safety confirmation
        print("\n" + "!" * 60)
        print("  WARNING: --live flag detected. This will use REAL USDT.")
        print("!" * 60)
        confirm = input("  Type 'YES I UNDERSTAND' to proceed: ").strip()
        if confirm != "YES I UNDERSTAND":
            print("  Aborted.")
            await client.close()
            return
        adapter = BingXAdapter(client=client)
        try:
            await client.set_leverage(symbol, int(RISK_CONFIG["max_leverage"]))
        except Exception as e:
            print(f"  Leverage warning: {e}")
        mode = "LIVE BingX"
    else:
        adapter = PaperAdapter(client=client, initial_balance=Decimal(str(args.balance)))
        mode = "PAPER simulation"

    notifier = None
    if not args.no_telegram:
        notifier = TelegramNotifier.from_env()
        if notifier:
            # Test connectivity at startup
            try:
                ok = await notifier.send("\U0001f50c Telegram test... OK")
                if ok:
                    _log("  Telegram: connected (test message sent)")
                else:
                    _log_error("  Telegram: FAILED to send test message (continuing without)")
                    notifier = None
            except Exception as e:
                _log_error(f"  Telegram: error on test send: {e} (continuing without)")
                notifier = None
        else:
            _log("  Telegram: not configured (missing TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID)")

    cb = CircuitBreaker(initial_equity=args.balance)
    headless_str = " | Modo headless \u2014 solo Telegram" if _HEADLESS else ""

    _log(f"\n{'═'*60}")
    _log(f"  SIMPLE PAPER TRADER \u2014 Validated Strategy")
    _log(f"{'═'*60}")
    _log(f"  Mode:       {mode}{headless_str}")
    _log(f"  Symbol:     {symbol}")
    _log(f"  Timeframe:  {args.timeframe}")
    _log(f"  Balance:    ${args.balance:,.2f}")
    _log(f"  Risk/trade: {RISK_CONFIG['risk_pct_per_trade']}%")
    _log(f"  Max lev:    {RISK_CONFIG['max_leverage']}x")
    _log(f"  RR ratio:   {VALIDATED_PARAMS['rr_ratio']}")
    _log(
        "  Stop:       STRUCTURAL "
        f"(pivot L/R={VALIDATED_PARAMS['structural_pivot_left']}/{VALIDATED_PARAMS['structural_pivot_right']}, "
        f"buf={VALIDATED_PARAMS['structural_buffer_atr']}ATR, "
        f"min={VALIDATED_PARAMS['structural_min_risk_atr']}ATR)"
    )
    _log(f"  ATR fallbk: {VALIDATED_PARAMS['atr_sl_mult']}x")
    _log(f"  HTF filter: {'ON (4H EMA50)' if HTF_CONFIG['enabled'] else 'OFF'}")
    _log(f"  CB warn:    {RISK_CONFIG['max_daily_dd_pct']}% DD")
    _log(f"  CB stop:    {RISK_CONFIG['circuit_breaker_pct']}% DD")
    _log(f"{'═'*60}")

    # Send rich startup notification to Telegram
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    await _tg_send(notifier,
        f"\U0001f680 <b>BOT INICIADO \u2014 {'Paper Trading' if not is_live else 'LIVE TRADING'}</b>\n\n"
        f"\U0001f4cb Estrategia: TrendFollowingV2Simple\n"
        f"\U0001f4b0 Balance: <code>${args.balance:,.2f}</code>\n"
        f"\U0001f4ca Symbol: {symbol} | {args.timeframe}\n"
        f"\U0001f3af RR: {VALIDATED_PARAMS['rr_ratio']} | Stop: STRUCTURAL (fallback {VALIDATED_PARAMS['atr_sl_mult']}\u00d7ATR)\n"
        f"\u26a1 Risk/trade: {RISK_CONFIG['risk_pct_per_trade']}%\n"
        f"\U0001f9ed HTF Filter: {'ON (4H EMA50)' if HTF_CONFIG['enabled'] else 'OFF'}\n"
        f"\U0001f6e1 Circuit Breaker: warn={RISK_CONFIG['max_daily_dd_pct']:.0f}% / stop={RISK_CONFIG['circuit_breaker_pct']:.0f}%\n"
        f"\U0001f4b8 Fees: entry limit 0.020% | SL market 0.050% | TP limit 0.020% (avg RT ~0.056%)\n"
        f"{'Modo headless \u2014 solo Telegram' + chr(10) if _HEADLESS else ''}"
        f"\nComandos disponibles:\n"
        f"/status \u2014 estado actual\n"
        f"/trades \u2014 \u00faltimos 5 trades\n"
        f"/equity \u2014 resumen de equity\n"
        f"/stop   \u2014 detener bot\n"
        f"\n\u23f0 Iniciado: {now_str}")

    _log(f"\n  Loading...\n")

    shutdown_event = asyncio.Event()
    def _shutdown():
        _log("\n  Shutting down...")
        shutdown_event.set()

    try:
        loop = asyncio.get_running_loop()
        loop.add_signal_handler(signal.SIGINT, _shutdown)
        loop.add_signal_handler(signal.SIGTERM, _shutdown)
    except NotImplementedError:
        signal.signal(signal.SIGINT, lambda s, f: _shutdown())

    state = {
        "equity": args.balance,
        "start_balance": args.balance,
        "wins": 0,
        "losses": 0,
        "notifier": notifier,
        "circuit_breaker": cb,
        "htf": None,
        "open_trade": None,
        "upnl_pct": 0.0,
        "last_bar_ts": None,
        "last_progress_at": _utc_now(),
        "last_reconcile_at": None,
        "runner_status": "starting",
        "safe_mode_reason": None,
        "last_error": None,
        "api_error_streak": 0,
        "symbol": symbol,
        "timeframe": args.timeframe,
        "mode": "live" if is_live else "paper",
    }
    _write_heartbeat(state, symbol, state["mode"])

    instance_lock = _build_instance_lock(
        symbol=symbol,
        timeframe=args.timeframe,
        mode=state["mode"],
        api_key=settings.bingx_api_key or "paper-runner",
    )
    try:
        instance_lock.acquire()
    except InstanceLockError as exc:
        _log_error(f"  Instance lock failed: {exc}")
        _update_runner_status(state, "safe_mode", note=str(exc))
        state["safe_mode_reason"] = "instance_lock_failed"
        _write_heartbeat(state, symbol, state["mode"])
        await client.close()
        return

    try:
        worker = asyncio.create_task(run_symbol(
            symbol, client, adapter, args, shutdown_event, state, is_live=is_live))
        status = asyncio.create_task(_status_loop(state, symbol, shutdown_event))
        cmd_task = asyncio.create_task(_telegram_command_loop(state, symbol, shutdown_event))
        heartbeat = asyncio.create_task(_heartbeat_loop(state, symbol, state["mode"], shutdown_event))

        results = await asyncio.gather(worker, status, cmd_task, heartbeat, return_exceptions=True)
        for r in results:
            if isinstance(r, Exception) and not isinstance(r, asyncio.CancelledError):
                _log_error(f"  Task error: {r}")

        await adapter.shutdown()

        w, l = state["wins"], state["losses"]
        t = w + l
        eq = state["equity"]
        net = eq - args.balance
        _log(f"\n{'═'*60}")
        _log(f"  SESSION COMPLETE")
        _log(f"  Trades: {t}  W={w} L={l}  {'WR=' + str(round(w/t*100)) + '%' if t else ''}")
        _log(f"  Equity: ${eq:,.2f}  (net ${net:+,.2f})")
        _log(f"{'═'*60}\n")

        if notifier:
            try:
                await notifier.bot_stopped(
                    total_trades=t, net_pnl=net, wins=w, losses=l)
            except Exception:
                pass

        _update_runner_status(state, "stopped")
        _write_heartbeat(state, symbol, state["mode"])
    finally:
        instance_lock.release()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Simple Paper Trader \u2014 Validated Strategy")
    p.add_argument("--balance",      type=float, default=100.0)
    p.add_argument("--symbol",       default="BTC")
    p.add_argument("--timeframe",    default="15m")
    p.add_argument("--live",         action="store_true", help="Send REAL orders to BingX")
    p.add_argument("--check",        action="store_true", help="Verify BingX connection")
    p.add_argument("--no-telegram",  action="store_true", dest="no_telegram")
    p.add_argument("--headless",     action="store_true", help="No console output, Telegram only")
    p.add_argument(
        "--reconcile-secs",
        type=int,
        default=RUNTIME_RECONCILE_INTERVAL_SECS,
        help="Seconds between periodic live reconciliations (0 disables runtime polling).",
    )
    args = p.parse_args()

    if args.check:
        asyncio.run(check_connection(args))
    else:
        asyncio.run(main(args))
