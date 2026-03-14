"""
RSI Divergence Strategy — multi-position edition.

Detects classical RSI divergence (price vs. momentum) on any timeframe (designed for 5m).

Bullish (LONG):
  - Price makes lower low while RSI makes higher low (bullish divergence)
  - RSI at swing low 2 must be <= rsi_oversold
  - Entry when a subsequent candle closes above EMA → limit order at EMA level

Bearish (SHORT):
  - Price makes higher high while RSI makes lower high (bearish divergence)
  - RSI at swing high 2 must be >= rsi_overbought
  - Entry when a subsequent candle closes below EMA → limit order at EMA level

Macro trend filter (TrendContext):
  - Analyzes HTF bars (default 1H) via EMA50, EMA200, and ADX-14
  - Assigns a coefficient [0.0, 1.0] per trade direction based on macro alignment
  - Trades with coefficient < min_trend_coeff are blocked

Exit (two-leg):
  - TP1 at 1:rr_ratio  → closes 70% of position, SL moves to break-even
  - TP2 at 1:tp2_ratio → closes remaining 30%
  - BE stop: if price reverses after TP1, closes at entry (no loss)

Multi-position:
  - max_concurrent_positions controls how many simultaneous trades can be open
  - Use on_bar_all() to get all signals; on_bar() returns the first for backward compat
  - Each signal carries meta["position_id"] so the engine can match entries to exits

Parameters:
    rsi_period              int,   default 9      — RSI period
    ema_period              int,   default 14     — EMA trigger period
    swing_window            int,   default 5      — bars on each side to confirm pivot
    swing_separation        int,   default 10     — min bars between two compared swings
    swing_lookback          int,   default 100    — bars back to search for swings
    trigger_window          int,   default 10     — bars to wait for EMA trigger
    rsi_oversold            float, default 30     — RSI threshold for bullish divergence
    rsi_overbought          float, default 70     — RSI threshold for bearish divergence
    allow_short             bool,  default True   — enable SHORT signals
    sl_buffer_pct           float, default 0.003  — SL placed 0.3% beyond swing extreme
    rr_ratio                float, default 1.5    — TP1 risk:reward ratio (closes 70%)
    tp2_ratio               float, default 1.75   — TP2 risk:reward ratio (closes 30%)
    min_trend_coeff         float, default 0.5    — minimum HTF coefficient
    htf_ema_fast            int,   default 50     — HTF fast EMA
    htf_ema_slow            int,   default 200    — HTF slow EMA
    htf_adx_period          int,   default 14     — HTF ADX period
    entry_window            int,   default 2      — bars to wait for limit fill
    max_concurrent_positions int,  default 1      — max simultaneous open positions
"""
from __future__ import annotations

import uuid
from decimal import Decimal
from typing import Any, Optional

import numpy as np
import pandas as pd

from app.strategy.base import BaseStrategy
from app.strategy.signals import Signal, SignalAction
from app.strategy.trend_context import TrendContext


class RSIDivergenceStrategy(BaseStrategy):
    """
    RSI divergence strategy with EMA entry trigger, multi-position support,
    and internal SL/TP management.

    Call on_bar_all(df) for full multi-position support (returns list[Signal]).
    on_bar(df) returns the first signal for backward compatibility.
    """

    def __init__(self, symbol: str, params: dict[str, Any] | None = None) -> None:
        super().__init__(symbol, params)

        # Indicator parameters
        self._rsi_period = int(self.params.get("rsi_period", 9))
        self._ema_period = int(self.params.get("ema_period", 14))

        # Swing detection
        self._swing_window = int(self.params.get("swing_window", 5))
        self._swing_separation = int(self.params.get("swing_separation", 10))
        self._swing_lookback = int(self.params.get("swing_lookback", 100))
        self._trigger_window = int(self.params.get("trigger_window", 10))

        # Signal thresholds
        self._rsi_oversold = float(self.params.get("rsi_oversold", 40.0))
        self._rsi_overbought = float(self.params.get("rsi_overbought", 60.0))
        self._allow_short = bool(self.params.get("allow_short", True))
        self._sl_buffer_pct = float(self.params.get("sl_buffer_pct", 0.003))
        self._rr_ratio = float(self.params.get("rr_ratio", 1.5))
        self._tp2_ratio = float(self.params.get("tp2_ratio", 1.75))
        self._min_trend_coeff = float(self.params.get("min_trend_coeff", 0.5))
        self._entry_window = int(self.params.get("entry_window", 15))

        # Multi-position
        self._max_concurrent = int(self.params.get("max_concurrent_positions", 5))

        # Macro trend filter (HTF)
        self._trend_ctx = TrendContext(
            ema_fast=int(self.params.get("htf_ema_fast", 50)),
            ema_slow=int(self.params.get("htf_ema_slow", 200)),
            adx_period=int(self.params.get("htf_adx_period", 14)),
        )

        # ── Scanner context (one active scanner at a time) ──────────────────
        # Detects divergences and manages ARM → ENTRY_PENDING → fill
        self._scan: dict = self._new_scan()

        # ── Open positions (list of dicts) ─────────────────────────────────
        # Each dict: {id, side, state, entry_price, sl_price, tp1_price, tp2_price}
        self._positions: list[dict] = []

    # ── HTF trend filter ──────────────────────────────────────────────────────

    def set_htf_bars(self, bars: pd.DataFrame) -> None:
        self._trend_ctx.update(bars)

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def strategy_id(self) -> str:
        return (
            f"rsi_divergence_rsi{self._rsi_period}_ema{self._ema_period}"
            f"_sw{self._swing_window}_sep{self._swing_separation}"
            f"_short{self._allow_short}"
        )

    @property
    def min_bars_required(self) -> int:
        return self._rsi_period + 1 + self._swing_lookback + self._swing_window * 2

    # ── Scan context helpers ──────────────────────────────────────────────────

    def _new_scan(self) -> dict:
        return {
            "state": "FLAT",
            "armed_direction": None,
            "armed_bars_elapsed": 0,
            "divergence_swing_price": None,
            "div_sw1_ts": None, "div_sw1_price": None, "div_sw1_rsi": None,
            "div_sw2_ts": None, "div_sw2_price": None, "div_sw2_rsi": None,
            "pending_entry_price": None,
            "pending_direction": None,
            "pending_conf_extreme": None,
            "entry_pending_bars": 0,
        }

    # ── Indicator computation ─────────────────────────────────────────────────

    def _compute_indicators(self, bars: pd.DataFrame) -> pd.DataFrame:
        df = bars.copy()
        closes = df["close"]
        delta = closes.diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        alpha = 1.0 / self._rsi_period
        avg_gain = gain.ewm(alpha=alpha, adjust=False).mean()
        avg_loss = loss.ewm(alpha=alpha, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        df["rsi"] = 100.0 - (100.0 / (1.0 + rs))
        df["ema"] = closes.ewm(span=self._ema_period, adjust=False).mean()
        return df

    # ── Swing detection ───────────────────────────────────────────────────────

    def _find_swing_lows(self, df: pd.DataFrame) -> list[int]:
        n = len(df)
        w = self._swing_window
        search_start = max(w, n - self._swing_lookback - w)
        search_end = n - w
        lows = df["low"].values
        indices = []
        for i in range(search_start, search_end):
            v = lows[i]
            if all(v < lows[i - k] for k in range(1, w + 1)) and \
               all(v < lows[i + k] for k in range(1, w + 1)):
                indices.append(i)
        return indices

    def _find_swing_highs(self, df: pd.DataFrame) -> list[int]:
        n = len(df)
        w = self._swing_window
        search_start = max(w, n - self._swing_lookback - w)
        search_end = n - w
        highs = df["high"].values
        indices = []
        for i in range(search_start, search_end):
            v = highs[i]
            if all(v > highs[i - k] for k in range(1, w + 1)) and \
               all(v > highs[i + k] for k in range(1, w + 1)):
                indices.append(i)
        return indices

    # ── Divergence detection ──────────────────────────────────────────────────

    def _detect_bullish_divergence(self, df: pd.DataFrame) -> bool:
        """Returns True and populates scan debug fields if bullish divergence detected."""
        swing_lows = self._find_swing_lows(df)
        if len(swing_lows) < 2:
            return False
        idx2, idx1 = swing_lows[-1], swing_lows[-2]
        if (idx2 - idx1) < self._swing_separation:
            return False
        price1, price2 = df["low"].iloc[idx1], df["low"].iloc[idx2]
        rsi1, rsi2 = df["rsi"].iloc[idx1], df["rsi"].iloc[idx2]
        if pd.isna(rsi1) or pd.isna(rsi2):
            return False
        if price2 < price1 and rsi2 > rsi1 and rsi1 <= self._rsi_oversold and rsi2 <= self._rsi_oversold:
            s = self._scan
            s["divergence_swing_price"] = float(price2)
            s["div_sw1_ts"] = str(df["ts"].iloc[idx1]) if "ts" in df.columns else None
            s["div_sw2_ts"] = str(df["ts"].iloc[idx2]) if "ts" in df.columns else None
            s["div_sw1_price"] = float(price1)
            s["div_sw2_price"] = float(price2)
            s["div_sw1_rsi"] = float(rsi1)
            s["div_sw2_rsi"] = float(rsi2)
            return True
        return False

    def _detect_bearish_divergence(self, df: pd.DataFrame) -> bool:
        swing_highs = self._find_swing_highs(df)
        if len(swing_highs) < 2:
            return False
        idx2, idx1 = swing_highs[-1], swing_highs[-2]
        if (idx2 - idx1) < self._swing_separation:
            return False
        price1, price2 = df["high"].iloc[idx1], df["high"].iloc[idx2]
        rsi1, rsi2 = df["rsi"].iloc[idx1], df["rsi"].iloc[idx2]
        if pd.isna(rsi1) or pd.isna(rsi2):
            return False
        if price2 > price1 and rsi2 < rsi1 and rsi1 >= self._rsi_overbought and rsi2 >= self._rsi_overbought:
            s = self._scan
            s["divergence_swing_price"] = float(price2)
            s["div_sw1_ts"] = str(df["ts"].iloc[idx1]) if "ts" in df.columns else None
            s["div_sw2_ts"] = str(df["ts"].iloc[idx2]) if "ts" in df.columns else None
            s["div_sw1_price"] = float(price1)
            s["div_sw2_price"] = float(price2)
            s["div_sw1_rsi"] = float(rsi1)
            s["div_sw2_rsi"] = float(rsi2)
            return True
        return False

    # ── EMA trigger ───────────────────────────────────────────────────────────

    def _long_trigger(self, df: pd.DataFrame) -> bool:
        ema = df["ema"].iloc[-1]
        return not pd.isna(ema) and float(df["close"].iloc[-1]) > float(ema)

    def _short_trigger(self, df: pd.DataFrame) -> bool:
        ema = df["ema"].iloc[-1]
        return not pd.isna(ema) and float(df["close"].iloc[-1]) < float(ema)

    # ── SL/TP computation ─────────────────────────────────────────────────────

    def _compute_long_sl_tps(self, entry: float, conf_low: float) -> tuple[float, float, float]:
        sl = conf_low * (1.0 - self._sl_buffer_pct)
        risk = entry - sl
        if risk <= 0:
            risk = entry * self._sl_buffer_pct
            sl = entry - risk
        return sl, entry + risk * self._rr_ratio, entry + risk * self._tp2_ratio

    def _compute_short_sl_tps(self, entry: float, conf_high: float) -> tuple[float, float, float]:
        sl = conf_high * (1.0 + self._sl_buffer_pct)
        risk = sl - entry
        if risk <= 0:
            risk = entry * self._sl_buffer_pct
            sl = entry + risk
        return sl, entry - risk * self._rr_ratio, entry - risk * self._tp2_ratio

    # ── Position management ───────────────────────────────────────────────────

    def _check_position_exit(
        self, pos: dict, ts: Any,
        cur_high: float, cur_low: float,
        base_meta: dict,
    ) -> Optional[Signal]:
        """Check if this bar hits SL or TP for the given position. Returns signal or None."""
        state = pos["state"]
        pid = pos["id"]
        pos_meta = {**base_meta, "position_id": pid, "sl": pos["sl_price"], "tp": pos["tp1_price"]}

        def mk_close(reason: str, exit_meta: dict) -> Signal:
            pos["state"] = "CLOSED"
            return Signal(
                action=SignalAction.CLOSE,
                symbol=self.symbol, ts=ts,
                strategy_id=self.strategy_id,
                reason=reason,
                meta={**pos_meta, **exit_meta},
            )

        if state == "LONG":
            # Worst-case same bar: SL takes priority over TP1
            if cur_low <= pos["sl_price"]:
                return mk_close(
                    f"long_sl_hit sl={pos['sl_price']:.4f}",
                    {"exit_type": "sl", "exit_price": pos["sl_price"]},
                )
            if cur_high >= pos["tp1_price"]:
                tp1 = pos["tp1_price"]
                pos["sl_price"] = pos["entry_price"]  # move SL to BE
                pos["tp1_price"] = None
                pos["state"] = "LONG_BE"
                return Signal(
                    action=SignalAction.PARTIAL_CLOSE,
                    symbol=self.symbol, ts=ts,
                    strategy_id=self.strategy_id,
                    reason=f"long_tp1_hit tp1={tp1:.4f} sl_be={pos['entry_price']:.4f}",
                    meta={**pos_meta, "close_pct": 0.70, "exit_type": "tp1",
                          "exit_price": tp1, "be": pos["entry_price"], "tp2": pos["tp2_price"],
                          "position_id": pid},
                )

        elif state == "LONG_BE":
            pos_meta["tp"] = pos["tp2_price"]
            if cur_low <= pos["sl_price"]:
                return mk_close(
                    f"long_be_sl_hit sl={pos['sl_price']:.4f}",
                    {"exit_type": "be_sl", "exit_price": pos["sl_price"]},
                )
            if cur_high >= pos["tp2_price"]:
                return mk_close(
                    f"long_tp2_hit tp2={pos['tp2_price']:.4f}",
                    {"exit_type": "tp2", "exit_price": pos["tp2_price"]},
                )

        elif state == "SHORT":
            if cur_high >= pos["sl_price"]:
                return mk_close(
                    f"short_sl_hit sl={pos['sl_price']:.4f}",
                    {"exit_type": "sl", "exit_price": pos["sl_price"]},
                )
            if cur_low <= pos["tp1_price"]:
                tp1 = pos["tp1_price"]
                pos["sl_price"] = pos["entry_price"]
                pos["tp1_price"] = None
                pos["state"] = "SHORT_BE"
                return Signal(
                    action=SignalAction.PARTIAL_CLOSE,
                    symbol=self.symbol, ts=ts,
                    strategy_id=self.strategy_id,
                    reason=f"short_tp1_hit tp1={tp1:.4f} sl_be={pos['entry_price']:.4f}",
                    meta={**pos_meta, "close_pct": 0.70, "exit_type": "tp1",
                          "exit_price": tp1, "be": pos["entry_price"], "tp2": pos["tp2_price"],
                          "position_id": pid},
                )

        elif state == "SHORT_BE":
            pos_meta["tp"] = pos["tp2_price"]
            if cur_high >= pos["sl_price"]:
                return mk_close(
                    f"short_be_sl_hit sl={pos['sl_price']:.4f}",
                    {"exit_type": "be_sl", "exit_price": pos["sl_price"]},
                )
            if cur_low <= pos["tp2_price"]:
                return mk_close(
                    f"short_tp2_hit tp2={pos['tp2_price']:.4f}",
                    {"exit_type": "tp2", "exit_price": pos["tp2_price"]},
                )
        return None

    # ── Scanner logic ─────────────────────────────────────────────────────────

    def _process_scanner(
        self, df: pd.DataFrame, ts: Any,
        cur_close: float, cur_high: float, cur_low: float,
        cur_ema: float, cur_rsi: float,
        base_meta: dict,
    ) -> list[Signal]:
        """Process the divergence scanner. Returns 0 or 1 signal."""
        s = self._scan
        signals: list[Signal] = []

        def hold(reason: str, extra: dict | None = None) -> Signal:
            return Signal(
                action=SignalAction.HOLD,
                symbol=self.symbol, ts=ts,
                strategy_id=self.strategy_id,
                reason=reason,
                meta={**base_meta, **(extra or {})},
            )

        # ── ENTRY_PENDING: wait for limit fill ────────────────────────────
        if s["state"] == "ENTRY_PENDING":
            s["entry_pending_bars"] += 1
            entry_price = s["pending_entry_price"]
            direction = s["pending_direction"]
            conf_extreme = s["pending_conf_extreme"]

            if direction == "LONG" and cur_low <= entry_price:
                sl, tp1, tp2 = self._compute_long_sl_tps(entry_price, conf_extreme)
                pid = str(uuid.uuid4())
                self._positions.append({
                    "id": pid, "side": "LONG", "state": "LONG",
                    "entry_price": entry_price,
                    "sl_price": sl, "tp1_price": tp1, "tp2_price": tp2,
                })
                self._scan = self._new_scan()
                signals.append(Signal(
                    action=SignalAction.BUY,
                    symbol=self.symbol, ts=ts,
                    strategy_id=self.strategy_id,
                    confidence=1.0,
                    stop_loss=Decimal(str(round(sl, 8))),
                    take_profit=Decimal(str(round(tp1, 8))),
                    reason=(
                        f"bullish_div_long entry={entry_price:.4f} "
                        f"sl={sl:.4f} tp1={tp1:.4f} tp2={tp2:.4f}"
                    ),
                    meta={
                        **base_meta, **self._trend_ctx.to_meta(),
                        "position_id": pid,
                        "limit_entry": entry_price,
                        "sl": sl, "tp1": tp1, "tp2": tp2,
                        "swing_price": s["divergence_swing_price"],
                    },
                ))
                return signals

            elif direction == "SHORT" and self._allow_short and cur_high >= entry_price:
                sl, tp1, tp2 = self._compute_short_sl_tps(entry_price, conf_extreme)
                pid = str(uuid.uuid4())
                self._positions.append({
                    "id": pid, "side": "SHORT", "state": "SHORT",
                    "entry_price": entry_price,
                    "sl_price": sl, "tp1_price": tp1, "tp2_price": tp2,
                })
                self._scan = self._new_scan()
                signals.append(Signal(
                    action=SignalAction.SELL,
                    symbol=self.symbol, ts=ts,
                    strategy_id=self.strategy_id,
                    confidence=1.0,
                    stop_loss=Decimal(str(round(sl, 8))),
                    take_profit=Decimal(str(round(tp1, 8))),
                    reason=(
                        f"bearish_div_short entry={entry_price:.4f} "
                        f"sl={sl:.4f} tp1={tp1:.4f} tp2={tp2:.4f}"
                    ),
                    meta={
                        **base_meta, **self._trend_ctx.to_meta(),
                        "position_id": pid,
                        "limit_entry": entry_price,
                        "sl": sl, "tp1": tp1, "tp2": tp2,
                        "swing_price": s["divergence_swing_price"],
                    },
                ))
                return signals

            if s["entry_pending_bars"] >= self._entry_window:
                self._scan = self._new_scan()
                return [hold("entry_pending_expired")]
            return [hold(
                f"entry_pending_{direction} limit={entry_price:.4f} "
                f"bar {s['entry_pending_bars']}/{self._entry_window}"
            )]

        # ── ARMED: wait for EMA confirmation ──────────────────────────────
        if s["state"] == "ARMED":
            s["armed_bars_elapsed"] += 1

            if s["armed_bars_elapsed"] > self._trigger_window:
                # Expired — fall through to FLAT scan this same bar
                self._scan = self._new_scan()
            else:
                direction = s["armed_direction"]

                if direction == "LONG" and self._long_trigger(df):
                    trend_coeff = self._trend_ctx.get_coefficient("LONG")
                    trend_meta = self._trend_ctx.to_meta()
                    if trend_coeff < self._min_trend_coeff:
                        self._scan = self._new_scan()
                        return [hold(
                            f"long_blocked_htf coeff={trend_coeff:.2f}<{self._min_trend_coeff}",
                            trend_meta,
                        )]
                    s["pending_entry_price"] = cur_ema
                    s["pending_direction"] = "LONG"
                    s["pending_conf_extreme"] = cur_low
                    s["entry_pending_bars"] = 0
                    s["state"] = "ENTRY_PENDING"
                    return [hold(
                        f"long_confirmed ema={cur_ema:.4f} waiting_limit",
                        self._trend_ctx.to_meta(),
                    )]

                if direction == "SHORT" and self._allow_short and self._short_trigger(df):
                    trend_coeff = self._trend_ctx.get_coefficient("SHORT")
                    trend_meta = self._trend_ctx.to_meta()
                    if trend_coeff < self._min_trend_coeff:
                        self._scan = self._new_scan()
                        return [hold(
                            f"short_blocked_htf coeff={trend_coeff:.2f}<{self._min_trend_coeff}",
                            trend_meta,
                        )]
                    s["pending_entry_price"] = cur_ema
                    s["pending_direction"] = "SHORT"
                    s["pending_conf_extreme"] = cur_high
                    s["entry_pending_bars"] = 0
                    s["state"] = "ENTRY_PENDING"
                    return [hold(
                        f"short_confirmed ema={cur_ema:.4f} waiting_limit",
                        self._trend_ctx.to_meta(),
                    )]

                return [hold(
                    f"armed_{direction} waiting {s['armed_bars_elapsed']}/{self._trigger_window}"
                )]

        # ── FLAT: scan for divergence ──────────────────────────────────────
        if self._detect_bullish_divergence(df):
            s = self._scan
            s["state"] = "ARMED"
            s["armed_direction"] = "LONG"
            s["armed_bars_elapsed"] = 0
            return [hold(
                f"bullish_div_armed sw1={s['div_sw1_price']:.4f}@{s['div_sw1_rsi']:.1f} "
                f"sw2={s['div_sw2_price']:.4f}@{s['div_sw2_rsi']:.1f}",
                {
                    "div_sw1_ts": s["div_sw1_ts"], "div_sw1_price": s["div_sw1_price"],
                    "div_sw1_rsi": s["div_sw1_rsi"], "div_sw2_ts": s["div_sw2_ts"],
                    "div_sw2_price": s["div_sw2_price"], "div_sw2_rsi": s["div_sw2_rsi"],
                },
            )]

        if self._allow_short and self._detect_bearish_divergence(df):
            s = self._scan
            s["state"] = "ARMED"
            s["armed_direction"] = "SHORT"
            s["armed_bars_elapsed"] = 0
            return [hold(
                f"bearish_div_armed sw1={s['div_sw1_price']:.4f}@{s['div_sw1_rsi']:.1f} "
                f"sw2={s['div_sw2_price']:.4f}@{s['div_sw2_rsi']:.1f}",
                {
                    "div_sw1_ts": s["div_sw1_ts"], "div_sw1_price": s["div_sw1_price"],
                    "div_sw1_rsi": s["div_sw1_rsi"], "div_sw2_ts": s["div_sw2_ts"],
                    "div_sw2_price": s["div_sw2_price"], "div_sw2_rsi": s["div_sw2_rsi"],
                },
            )]

        return []

    # ── Primary visible state for HTML viewer ─────────────────────────────────

    def _viewer_state(self) -> str:
        """Returns the state string shown in the HTML viewer bar-by-bar."""
        # Open positions take priority
        if self._positions:
            p = self._positions[0]
            return p["state"]
        # Fall back to scanner state
        return self._scan["state"]

    def _viewer_meta(self, base_meta: dict) -> dict:
        """Merge SL/TP from primary open position into meta for the viewer."""
        m = dict(base_meta)
        m["state"] = self._viewer_state()
        if self._positions:
            p = self._positions[0]
            m["sl"] = p["sl_price"]
            m["tp"] = p["tp1_price"] if p["state"] in ("LONG", "SHORT") else p.get("tp2_price")
            m["entry"] = p["entry_price"]
            m["side"] = p["side"]
        else:
            m["sl"] = None
            m["tp"] = None
        return m

    # ── Main entry points ─────────────────────────────────────────────────────

    def on_bar_all(self, bars: pd.DataFrame) -> list[Signal]:
        """
        Process one bar and return ALL signals (multi-position aware).
        May return multiple signals in one bar (e.g. one position closes + new entry).
        """
        ts = bars["ts"].iloc[-1]
        if hasattr(ts, "to_pydatetime"):
            ts = ts.to_pydatetime()

        if len(bars) < self.min_bars_required:
            return [Signal(
                action=SignalAction.HOLD,
                symbol=self.symbol, ts=ts,
                strategy_id=self.strategy_id,
                reason=f"warmup ({len(bars)}/{self.min_bars_required})",
                meta={"state": "FLAT"},
            )]

        df = self._compute_indicators(bars)
        cur_close = float(df["close"].iloc[-1])
        cur_high  = float(df["high"].iloc[-1])
        cur_low   = float(df["low"].iloc[-1])
        cur_ema   = float(df["ema"].iloc[-1]) if not pd.isna(df["ema"].iloc[-1]) else 0.0
        cur_rsi   = float(df["rsi"].iloc[-1]) if not pd.isna(df["rsi"].iloc[-1]) else 50.0

        base_meta = {"rsi": round(cur_rsi, 2), "ema": round(cur_ema, 4)}

        signals: list[Signal] = []

        # 1. Process all open positions (may emit CLOSE / PARTIAL_CLOSE)
        for pos in list(self._positions):
            sig = self._check_position_exit(pos, ts, cur_high, cur_low, base_meta)
            if sig:
                signals.append(sig)
            if pos["state"] == "CLOSED":
                self._positions.remove(pos)

        # 2. Process scanner if below max concurrent
        if len(self._positions) < self._max_concurrent:
            scan_signals = self._process_scanner(
                df, ts, cur_close, cur_high, cur_low, cur_ema, cur_rsi, base_meta
            )
            signals.extend(scan_signals)

        # 3. Attach viewer-friendly meta to all signals
        vm = self._viewer_meta(base_meta)
        for sig in signals:
            sig.meta.setdefault("state", vm["state"])
            sig.meta.setdefault("sl", vm.get("sl"))
            sig.meta.setdefault("tp", vm.get("tp"))
            sig.meta.setdefault("rsi", base_meta["rsi"])
            sig.meta.setdefault("ema", base_meta["ema"])

        if not signals:
            return [Signal(
                action=SignalAction.HOLD,
                symbol=self.symbol, ts=ts,
                strategy_id=self.strategy_id,
                reason="no_divergence",
                meta=vm,
            )]

        return signals

    def on_bar(self, bars: pd.DataFrame) -> Signal:
        """Backward-compatible: returns first signal. Use on_bar_all() for multi-position."""
        return self.on_bar_all(bars)[0]

    # ── Public utility ────────────────────────────────────────────────────────

    def compute_indicators(self, bars: pd.DataFrame) -> pd.DataFrame:
        df = self._compute_indicators(bars)
        df["swing_low"] = False
        df["swing_high"] = False
        for idx in self._find_swing_lows(df):
            df.at[idx, "swing_low"] = True
        for idx in self._find_swing_highs(df):
            df.at[idx, "swing_high"] = True
        return df
