"""
RSI Divergence Strategy with EMA14 trigger.

Detects classical RSI divergence (price vs. momentum) on any timeframe (designed for 5m).

Bullish (LONG):
  - Price makes lower low while RSI makes higher low (bullish divergence)
  - RSI at swing low 2 must be <= rsi_oversold
  - Entry when a subsequent candle closes above EMA14

Bearish (SHORT):
  - Price makes higher high while RSI makes lower high (bearish divergence)
  - RSI at swing high 2 must be >= rsi_overbought
  - Entry when a subsequent candle closes below EMA14

Exit: 100% close at 1:rr_ratio take-profit OR stop-loss.
  (The engine does not support partial closes; SL/TP are self-enforced inside on_bar.)

Parameters:
    rsi_period       int,   default 9      — RSI period (Wilder's EMA)
    ema_period       int,   default 14     — EMA trigger period
    swing_window     int,   default 5      — bars on each side to confirm a pivot
    swing_separation int,   default 10     — min bars between the two compared swings
    swing_lookback   int,   default 100    — bars back to search for swings
    trigger_window   int,   default 10     — bars to wait for EMA trigger after divergence
    rsi_oversold     float, default 30     — RSI threshold for bullish divergence
    rsi_overbought   float, default 70     — RSI threshold for bearish divergence
    allow_short      bool,  default True   — enable SHORT signals
    sl_buffer_pct    float, default 0.003  — SL placed 0.3% beyond swing extreme
    rr_ratio         float, default 1.5    — risk:reward multiplier for TP
"""
from __future__ import annotations

from decimal import Decimal
from typing import Any, Optional

import numpy as np
import pandas as pd

from app.strategy.base import BaseStrategy
from app.strategy.signals import Signal, SignalAction


class RSIDivergenceStrategy(BaseStrategy):
    """
    RSI divergence strategy with EMA14 entry trigger and internal SL/TP management.

    State machine:
        FLAT  -> (divergence detected)             -> ARMED
        ARMED -> (close crosses EMA within window) -> emit BUY/SELL -> LONG/SHORT
        ARMED -> (trigger_window expires)          -> FLAT
        LONG  -> (high >= tp or low <= sl)         -> emit CLOSE -> FLAT
        SHORT -> (low <= tp or high >= sl)         -> emit CLOSE -> FLAT
    """

    def __init__(self, symbol: str, params: dict[str, Any] | None = None) -> None:
        super().__init__(symbol, params)

        # Indicator parameters
        self._rsi_period = int(self.params.get("rsi_period", 9))
        self._ema_period = int(self.params.get("ema_period", 14))

        # Swing detection parameters
        self._swing_window = int(self.params.get("swing_window", 5))
        self._swing_separation = int(self.params.get("swing_separation", 5))
        self._swing_lookback = int(self.params.get("swing_lookback", 100))
        self._trigger_window = int(self.params.get("trigger_window", 10))

        # Signal thresholds
        self._rsi_oversold = float(self.params.get("rsi_oversold", 20.0))
        self._rsi_overbought = float(self.params.get("rsi_overbought", 80.0))
        self._allow_short = bool(self.params.get("allow_short", True))
        self._sl_buffer_pct = float(self.params.get("sl_buffer_pct", 0.003))
        self._rr_ratio = float(self.params.get("rr_ratio", 1.5))
        self._tp2_ratio = float(self.params.get("tp2_ratio", 1.75))

        # State machine:
        #   FLAT → ARMED → ENTRY_PENDING → LONG    → LONG_BE  → FLAT
        #                               → SHORT   → SHORT_BE → FLAT
        #
        #   ARMED:         divergence detected, waiting for EMA confirmation candle
        #   ENTRY_PENDING: confirmation candle closed; next bar fills at EMA level (limit sim)
        #   LONG_BE / SHORT_BE: 75% closed, remaining 25% with SL at break even
        self._state: str = "FLAT"

        # Armed state data
        self._armed_bars_elapsed: int = 0
        self._armed_direction: Optional[str] = None  # "LONG" or "SHORT"
        self._divergence_swing_price: Optional[float] = None
        self._divergence_swing_rsi: Optional[float] = None

        # ENTRY_PENDING state: limit order at EMA crossing price
        self._pending_entry_price: Optional[float] = None       # EMA level of confirmation bar
        self._pending_direction: Optional[str] = None            # "LONG" or "SHORT"
        self._pending_conf_extreme: Optional[float] = None       # low of conf bar (LONG) or high (SHORT) → SL reference
        self._entry_pending_bars: int = 0
        self._entry_window: int = int(self.params.get("entry_window", 2))  # bars to wait

        # Open position tracking
        self._entry_price: Optional[float] = None
        self._sl_price: Optional[float] = None
        self._tp1_price: Optional[float] = None   # TP1: close 70% at 1:rr_ratio
        self._tp2_price: Optional[float] = None   # TP2: close remaining 30% at 1:tp2_ratio

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
        # RSI warm-up + swing search space
        return self._rsi_period + 1 + self._swing_lookback + self._swing_window * 2

    # ── Indicator computation ─────────────────────────────────────────────────

    def _compute_indicators(self, bars: pd.DataFrame) -> pd.DataFrame:
        """Add 'rsi' and 'ema' columns to a copy of bars."""
        df = bars.copy()
        closes = df["close"]

        # RSI via Wilder's EMA (alpha = 1/period)
        delta = closes.diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        alpha = 1.0 / self._rsi_period
        avg_gain = gain.ewm(alpha=alpha, adjust=False).mean()
        avg_loss = loss.ewm(alpha=alpha, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        df["rsi"] = 100.0 - (100.0 / (1.0 + rs))

        # EMA trigger
        df["ema"] = closes.ewm(span=self._ema_period, adjust=False).mean()

        return df

    # ── Swing detection ───────────────────────────────────────────────────────

    def _find_swing_lows(self, df: pd.DataFrame) -> list[int]:
        """
        Return indices of confirmed swing lows within the lookback window.
        A pivot low at i requires: low[i] < low[i±k] for all k in 1..swing_window.
        Excludes the last swing_window bars (right side not yet confirmed).
        """
        n = len(df)
        w = self._swing_window
        search_start = max(w, n - self._swing_lookback - w)
        search_end = n - w  # exclusive

        lows = df["low"].values
        indices = []
        for i in range(search_start, search_end):
            v = lows[i]
            if all(v < lows[i - k] for k in range(1, w + 1)) and \
               all(v < lows[i + k] for k in range(1, w + 1)):
                indices.append(i)
        return indices

    def _find_swing_highs(self, df: pd.DataFrame) -> list[int]:
        """
        Return indices of confirmed swing highs within the lookback window.
        A pivot high at i requires: high[i] > high[i±k] for all k in 1..swing_window.
        """
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
        """
        Check for bullish RSI divergence on the two most recent swing lows.
        Sets self._divergence_swing_price and _divergence_swing_rsi on match.
        """
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

        if price2 < price1 and rsi2 > rsi1 and rsi2 <= self._rsi_oversold:
            self._divergence_swing_price = float(price2)
            self._divergence_swing_rsi = float(rsi2)
            return True
        return False

    def _detect_bearish_divergence(self, df: pd.DataFrame) -> bool:
        """
        Check for bearish RSI divergence on the two most recent swing highs.
        Sets self._divergence_swing_price and _divergence_swing_rsi on match.
        """
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

        if price2 > price1 and rsi2 < rsi1 and rsi2 >= self._rsi_overbought:
            self._divergence_swing_price = float(price2)
            self._divergence_swing_rsi = float(rsi2)
            return True
        return False

    # ── EMA trigger checks ────────────────────────────────────────────────────

    def _long_trigger(self, df: pd.DataFrame) -> bool:
        """Current bar closes above EMA14."""
        ema = df["ema"].iloc[-1]
        return not pd.isna(ema) and float(df["close"].iloc[-1]) > float(ema)

    def _short_trigger(self, df: pd.DataFrame) -> bool:
        """Current bar closes below EMA14."""
        ema = df["ema"].iloc[-1]
        return not pd.isna(ema) and float(df["close"].iloc[-1]) < float(ema)

    # ── SL/TP computation ─────────────────────────────────────────────────────

    def _compute_long_sl_tps(self, entry: float) -> tuple[float, float, float]:
        """LONG: SL below vela2's low (divergence swing), TP1 at 1:rr_ratio (70%), TP2 at 1:tp2_ratio (30%)."""
        sl = self._divergence_swing_price * (1.0 - self._sl_buffer_pct)
        risk = entry - sl
        if risk <= 0:
            risk = entry * self._sl_buffer_pct
            sl = entry - risk
        tp1 = entry + risk * self._rr_ratio
        tp2 = entry + risk * self._tp2_ratio
        return sl, tp1, tp2

    def _compute_short_sl_tps(self, entry: float) -> tuple[float, float, float]:
        """SHORT: SL above vela2's high (divergence swing), TP1 at 1:rr_ratio (70%), TP2 at 1:tp2_ratio (30%)."""
        sl = self._divergence_swing_price * (1.0 + self._sl_buffer_pct)
        risk = sl - entry
        if risk <= 0:
            risk = entry * self._sl_buffer_pct
            sl = entry + risk
        tp1 = entry - risk * self._rr_ratio
        tp2 = entry - risk * self._tp2_ratio
        return sl, tp1, tp2

    # ── State resets ──────────────────────────────────────────────────────────

    def _reset_to_flat(self) -> None:
        self._state = "FLAT"
        self._armed_bars_elapsed = 0
        self._armed_direction = None
        self._divergence_swing_price = None
        self._divergence_swing_rsi = None
        self._pending_entry_price = None
        self._pending_direction = None
        self._pending_conf_extreme = None
        self._entry_pending_bars = 0
        self._entry_price = None
        self._sl_price = None
        self._tp1_price = None
        self._tp2_price = None

    # ── Main signal logic ─────────────────────────────────────────────────────

    def on_bar(self, bars: pd.DataFrame) -> Signal:
        ts = bars["ts"].iloc[-1]
        if hasattr(ts, "to_pydatetime"):
            ts = ts.to_pydatetime()

        def hold(reason: str, meta: dict | None = None) -> Signal:
            return Signal(
                action=SignalAction.HOLD,
                symbol=self.symbol,
                ts=ts,
                strategy_id=self.strategy_id,
                reason=reason,
                meta=meta or {},
            )

        def close(reason: str, meta: dict | None = None) -> Signal:
            return Signal(
                action=SignalAction.CLOSE,
                symbol=self.symbol,
                ts=ts,
                strategy_id=self.strategy_id,
                reason=reason,
                meta=meta or {},
            )

        if len(bars) < self.min_bars_required:
            return hold(f"warmup ({len(bars)}/{self.min_bars_required})")

        df = self._compute_indicators(bars)
        cur_close = float(df["close"].iloc[-1])
        cur_high = float(df["high"].iloc[-1])
        cur_low = float(df["low"].iloc[-1])
        cur_ema = float(df["ema"].iloc[-1]) if not pd.isna(df["ema"].iloc[-1]) else 0.0
        cur_rsi = float(df["rsi"].iloc[-1]) if not pd.isna(df["rsi"].iloc[-1]) else 50.0

        base_meta = {
            "state": self._state,
            "rsi": round(cur_rsi, 2),
            "ema": round(cur_ema, 4),
            "sl": self._sl_price,
            "tp": self._tp1_price if self._tp1_price is not None else self._tp2_price,
        }

        # ── LONG: monitor TP1 (70%) and SL ───────────────────────────
        if self._state == "LONG":
            # Worst-case: if both SL and TP1 hit same bar → assume SL
            if cur_low <= self._sl_price:
                sl = self._sl_price
                self._reset_to_flat()
                return close(f"long_sl_hit sl={sl:.4f}", base_meta)
            if cur_high >= self._tp1_price:
                tp1 = self._tp1_price
                be = self._entry_price
                tp2 = self._tp2_price
                self._sl_price = be          # move SL to break even
                self._tp1_price = None       # TP1 consumed
                self._state = "LONG_BE"
                return Signal(
                    action=SignalAction.PARTIAL_CLOSE,
                    symbol=self.symbol, ts=ts,
                    strategy_id=self.strategy_id,
                    reason=f"long_tp1_hit tp1={tp1:.4f} sl_be={be:.4f} tp2={tp2:.4f}",
                    meta={**base_meta, "close_pct": 0.70, "be": be, "tp2": tp2},
                )
            return hold("long_open", base_meta)

        # ── LONG_BE: remaining 30% — TP2 at 1.75 R:R or BE stop ─────
        if self._state == "LONG_BE":
            if cur_low <= self._sl_price:    # BE stop hit
                sl = self._sl_price
                self._reset_to_flat()
                return close(f"long_be_sl_hit sl={sl:.4f}", base_meta)
            if cur_high >= self._tp2_price:  # TP2 hit — close remaining 30%
                tp2 = self._tp2_price
                self._reset_to_flat()
                return close(f"long_tp2_hit tp2={tp2:.4f}", base_meta)
            return hold("long_be_open", base_meta)

        # ── SHORT: monitor TP1 (70%) and SL ──────────────────────────
        if self._state == "SHORT":
            if cur_high >= self._sl_price:
                sl = self._sl_price
                self._reset_to_flat()
                return close(f"short_sl_hit sl={sl:.4f}", base_meta)
            if cur_low <= self._tp1_price:
                tp1 = self._tp1_price
                be = self._entry_price
                tp2 = self._tp2_price
                self._sl_price = be
                self._tp1_price = None       # TP1 consumed
                self._state = "SHORT_BE"
                return Signal(
                    action=SignalAction.PARTIAL_CLOSE,
                    symbol=self.symbol, ts=ts,
                    strategy_id=self.strategy_id,
                    reason=f"short_tp1_hit tp1={tp1:.4f} sl_be={be:.4f} tp2={tp2:.4f}",
                    meta={**base_meta, "close_pct": 0.70, "be": be, "tp2": tp2},
                )
            return hold("short_open", base_meta)

        # ── SHORT_BE: remaining 30% — TP2 at 1.75 R:R or BE stop ────
        if self._state == "SHORT_BE":
            if cur_high >= self._sl_price:   # BE stop hit
                sl = self._sl_price
                self._reset_to_flat()
                return close(f"short_be_sl_hit sl={sl:.4f}", base_meta)
            if cur_low <= self._tp2_price:   # TP2 hit — close remaining 30%
                tp2 = self._tp2_price
                self._reset_to_flat()
                return close(f"short_tp2_hit tp2={tp2:.4f}", base_meta)
            return hold("short_be_open", base_meta)

        # ── ENTRY_PENDING: confirmation fired; wait for limit fill at EMA ──
        if self._state == "ENTRY_PENDING":
            self._entry_pending_bars += 1
            entry_price = self._pending_entry_price

            if self._pending_direction == "LONG":
                # Fill condition: this bar's low touched (or pierced) the EMA entry level
                if cur_low <= entry_price:
                    sl, tp1, tp2 = self._compute_long_sl_tps(entry_price)
                    self._entry_price = entry_price
                    self._sl_price = sl
                    self._tp1_price = tp1
                    self._tp2_price = tp2
                    self._state = "LONG"
                    return Signal(
                        action=SignalAction.BUY,
                        symbol=self.symbol, ts=ts,
                        strategy_id=self.strategy_id,
                        confidence=1.0,
                        stop_loss=Decimal(str(round(sl, 8))),
                        take_profit=Decimal(str(round(tp1, 8))),
                        reason=(
                            f"bullish_div_long_entry_limit "
                            f"entry={entry_price:.4f} sl={sl:.4f} tp1={tp1:.4f} tp2={tp2:.4f}"
                        ),
                        meta={**base_meta, "swing_low": self._divergence_swing_price,
                              "limit_entry": entry_price, "tp2": tp2},
                    )

            elif self._pending_direction == "SHORT" and self._allow_short:
                # Fill condition: this bar's high touched (or pierced) the EMA entry level
                if cur_high >= entry_price:
                    sl, tp1, tp2 = self._compute_short_sl_tps(entry_price)
                    self._entry_price = entry_price
                    self._sl_price = sl
                    self._tp1_price = tp1
                    self._tp2_price = tp2
                    self._state = "SHORT"
                    return Signal(
                        action=SignalAction.SELL,
                        symbol=self.symbol, ts=ts,
                        strategy_id=self.strategy_id,
                        confidence=1.0,
                        stop_loss=Decimal(str(round(sl, 8))),
                        take_profit=Decimal(str(round(tp1, 8))),
                        reason=(
                            f"bearish_div_short_entry_limit "
                            f"entry={entry_price:.4f} sl={sl:.4f} tp1={tp1:.4f} tp2={tp2:.4f}"
                        ),
                        meta={**base_meta, "swing_high": self._divergence_swing_price,
                              "limit_entry": entry_price, "tp2": tp2},
                    )

            # Limit not filled within entry_window → cancel
            if self._entry_pending_bars >= self._entry_window:
                self._reset_to_flat()
                return hold("entry_pending_expired", base_meta)

            return hold(
                f"entry_pending_{self._pending_direction} "
                f"limit={entry_price:.4f} bar {self._entry_pending_bars}/{self._entry_window}",
                base_meta,
            )

        # ── ARMED: wait for EMA confirmation candle ───────────────────
        if self._state == "ARMED":
            self._armed_bars_elapsed += 1

            if self._armed_bars_elapsed > self._trigger_window:
                # Expired — fall through to FLAT scan this same bar
                self._state = "FLAT"
                self._armed_direction = None
                self._armed_bars_elapsed = 0

            else:
                if self._armed_direction == "LONG" and self._long_trigger(df):
                    # Confirmation: candle closed above EMA.
                    # Entry limit = EMA level; SL reference = this candle's LOW.
                    self._pending_entry_price = cur_ema
                    self._pending_direction = "LONG"
                    self._pending_conf_extreme = cur_low   # SL goes below this low
                    self._entry_pending_bars = 0
                    self._state = "ENTRY_PENDING"
                    return hold(
                        f"long_confirmed ema={cur_ema:.4f} conf_low={cur_low:.4f} waiting_limit_fill",
                        base_meta,
                    )

                if self._armed_direction == "SHORT" and self._allow_short and self._short_trigger(df):
                    # Confirmation: candle closed below EMA.
                    # Entry limit = EMA level; SL reference = this candle's HIGH.
                    self._pending_entry_price = cur_ema
                    self._pending_direction = "SHORT"
                    self._pending_conf_extreme = cur_high  # SL goes above this high
                    self._entry_pending_bars = 0
                    self._state = "ENTRY_PENDING"
                    return hold(
                        f"short_confirmed ema={cur_ema:.4f} conf_high={cur_high:.4f} waiting_limit_fill",
                        base_meta,
                    )

                return hold(
                    f"armed_{self._armed_direction} waiting {self._armed_bars_elapsed}/{self._trigger_window}",
                    base_meta,
                )

        # ── FLAT: scan for divergence ─────────────────────────────────
        if self._detect_bullish_divergence(df):
            self._state = "ARMED"
            self._armed_direction = "LONG"
            self._armed_bars_elapsed = 0
            return hold(
                f"bullish_div_armed swing_low={self._divergence_swing_price:.4f} rsi={self._divergence_swing_rsi:.2f}",
                base_meta,
            )

        if self._allow_short and self._detect_bearish_divergence(df):
            self._state = "ARMED"
            self._armed_direction = "SHORT"
            self._armed_bars_elapsed = 0
            return hold(
                f"bearish_div_armed swing_high={self._divergence_swing_price:.4f} rsi={self._divergence_swing_rsi:.2f}",
                base_meta,
            )

        return hold("no_divergence", base_meta)

    # ── Public utility ────────────────────────────────────────────────────────

    def compute_indicators(self, bars: pd.DataFrame) -> pd.DataFrame:
        """
        Returns bars enriched with rsi, ema, swing_low (bool), swing_high (bool).
        Useful for visualization and reporting.
        """
        df = self._compute_indicators(bars)
        df["swing_low"] = False
        df["swing_high"] = False
        for idx in self._find_swing_lows(df):
            df.at[idx, "swing_low"] = True
        for idx in self._find_swing_highs(df):
            df.at[idx, "swing_high"] = True
        return df
