"""
Hybrid RSI Divergence + Pivot Points + VWAP Strategy.

Combines three independent confirmation layers before entering a trade:
  Layer 1 — STRUCTURE:  Price near a daily Pivot Point support/resistance level.
  Layer 2 — MOMENTUM:   RSI Divergence signals exhaustion of the current move.
  Layer 3 — FLOW:       VWAP + EMA(12) trigger confirms intraday direction.

Designed for 5-minute crypto charts (BTC, ETH, SOL, etc.).

Long Setup:
  1. Price within pivot_atr_proximity × ATR(14) of a Pivot support (S1, S2, S3, PP)
  2. Bullish RSI divergence: lower low in price, higher low in RSI, RSI ≤ rsi_oversold
  3. Trigger bar: close > EMA(12)  AND  close ≥ VWAP
  4. SL  = swing_low - ATR(14) × atr_sl_mult
  5. TP  = entry + risk × rr_ratio   (capped at next Pivot resistance if closer)

Short Setup (mirror of Long):
  1. Price near Pivot resistance (R1, R2, R3, PP)
  2. Bearish RSI divergence: higher high in price, lower high in RSI, RSI ≥ rsi_overbought
  3. Trigger bar: close < EMA(12)  AND  close ≤ VWAP
  4. SL  = swing_high + ATR(14) × atr_sl_mult
  5. TP  = entry - risk × rr_ratio   (capped at next Pivot support if closer)

Worst-case: if both SL and TP hit on the same bar → assume SL (conservative).

Pre-filters applied every bar:
  • Session filter: only 08:00–21:00 UTC
  • Volume filter:  volume ≥ vol_min_ratio × rolling mean of last vol_avg_period bars

Parameters (all optional, with defaults):
    rsi_period           int,   9       — RSI period (Wilder's EMA)
    ema_period           int,   12      — EMA trigger period (12 per ZCoinTV video)
    atr_period           int,   14      — ATR lookback period
    atr_sl_mult          float, 1.5     — ATR multiplier for stop loss distance
    rr_ratio             float, 2.0     — Risk:Reward multiplier for take profit
    swing_window         int,   5       — Bars on each side to confirm a pivot
    swing_separation     int,   10      — Min bars between the two swings compared
    swing_lookback       int,   100     — Bars back to search for divergence swings
    trigger_window       int,   10      — Bars after divergence to wait for EMA+VWAP trigger
    rsi_oversold         float, 30.0    — RSI threshold for bullish divergence
    rsi_overbought       float, 70.0    — RSI threshold for bearish divergence
    pivot_atr_proximity  float, 1.5     — Max ATRs distance to a Pivot level to qualify
    vol_avg_period       int,   20      — Volume rolling average period for filter
    vol_min_ratio        float, 0.5     — Min ratio of current vol vs average
    session_start_utc    int,   8       — Session open hour (UTC, inclusive)
    session_end_utc      int,   21      — Session close hour (UTC, exclusive)
    allow_short          bool,  True    — Enable SHORT signals
    bars_per_day         int,   288     — Bars per calendar day (288 for 5m timeframe)
"""
from __future__ import annotations

from decimal import Decimal
from typing import Any, Optional

import numpy as np
import pandas as pd

from app.strategy.rsi_divergence import RSIDivergenceStrategy
from app.strategy.signals import Signal, SignalAction


class HybridRSIPivotStrategy(RSIDivergenceStrategy):
    """
    Hybrid strategy that extends RSIDivergenceStrategy with:
    - Daily Pivot Points for structural S/R zones
    - VWAP for intraday flow confirmation
    - ATR-based adaptive SL/TP
    - Session and volume pre-filters

    Inherits all swing detection and RSI divergence logic from RSIDivergenceStrategy.
    """

    def __init__(self, symbol: str, params: dict[str, Any] | None = None) -> None:
        super().__init__(symbol, params)

        # Override parent defaults for this strategy
        self._ema_period = int(self.params.get("ema_period", 12))   # 12 per ZCoinTV video
        self._rr_ratio = float(self.params.get("rr_ratio", 2.0))    # 2.0 for better edge

        # ATR-based SL
        self._atr_period = int(self.params.get("atr_period", 14))
        self._atr_sl_mult = float(self.params.get("atr_sl_mult", 1.5))

        # Pivot Point proximity filter
        self._pivot_atr_proximity = float(self.params.get("pivot_atr_proximity", 1.5))

        # Volume filter
        self._vol_avg_period = int(self.params.get("vol_avg_period", 20))
        self._vol_min_ratio = float(self.params.get("vol_min_ratio", 0.5))

        # Session filter (UTC hours)
        self._session_start = int(self.params.get("session_start_utc", 8))
        self._session_end = int(self.params.get("session_end_utc", 21))

        # Bars per day (288 for 5m, 96 for 15m, 24 for 1h, etc.)
        self._bars_per_day = int(self.params.get("bars_per_day", 288))

        # Internal: current ATR value (updated each bar, used in SL/TP and proximity)
        self._current_atr: Optional[float] = None

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def strategy_id(self) -> str:
        return (
            f"hybrid_rsi_pivot_rsi{self._rsi_period}_ema{self._ema_period}"
            f"_atr{self._atr_period}x{self._atr_sl_mult}"
            f"_rr{self._rr_ratio}_short{self._allow_short}"
        )

    @property
    def min_bars_required(self) -> int:
        # Need: 1 full previous day for Pivot Points + swing lookback + RSI warmup
        return self._bars_per_day + self._rsi_period + 1 + self._swing_lookback + self._swing_window * 2

    # ── New indicator methods ─────────────────────────────────────────────────

    def _compute_atr(self, bars: pd.DataFrame) -> pd.Series:
        """Average True Range — adaptive volatility measure."""
        high = bars["high"]
        low = bars["low"]
        prev_close = bars["close"].shift(1)

        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)

        return tr.rolling(window=self._atr_period, min_periods=self._atr_period).mean()

    def _compute_vwap(self, bars: pd.DataFrame) -> pd.Series:
        """
        Intraday VWAP that resets at UTC midnight.
        Uses typical price × volume, cumulative per calendar day.
        """
        df = bars.copy()
        ts_col = pd.to_datetime(df["ts"], utc=True)
        df["date"] = ts_col.dt.date
        df["typical_price"] = (df["high"] + df["low"] + df["close"]) / 3.0
        df["tp_x_vol"] = df["typical_price"] * df["volume"]

        # Cumulative sums per day
        df["cum_tp_vol"] = df.groupby("date")["tp_x_vol"].cumsum()
        df["cum_vol"] = df.groupby("date")["volume"].cumsum()

        vwap = df["cum_tp_vol"] / df["cum_vol"].replace(0.0, float("nan"))
        return vwap.values

    def _compute_daily_pivots(self, bars: pd.DataFrame) -> dict | None:
        """
        Calculate Standard Floor Pivot Points from the previous calendar day's OHLC.
        For 24/7 crypto: uses UTC day boundaries.

        Returns dict: {pp, r1, r2, r3, s1, s2, s3}  or None if no prior day data.
        """
        ts_col = pd.to_datetime(bars["ts"], utc=True)
        today = ts_col.iloc[-1].date()

        # Filter out current day bars to get only prior data
        prior_mask = ts_col.dt.date < today
        prior_bars = bars[prior_mask]

        if prior_bars.empty:
            return None

        # Use the most recent complete calendar day
        prior_dates = pd.to_datetime(prior_bars["ts"], utc=True).dt.date
        last_date = prior_dates.iloc[-1]
        day_bars = prior_bars[prior_dates == last_date]

        if day_bars.empty:
            return None

        H = float(day_bars["high"].max())
        L = float(day_bars["low"].min())
        C = float(day_bars["close"].iloc[-1])

        pp = (H + L + C) / 3.0
        r1 = 2.0 * pp - L
        r2 = pp + (H - L)
        r3 = H + 2.0 * (pp - L)
        s1 = 2.0 * pp - H
        s2 = pp - (H - L)
        s3 = L - 2.0 * (H - L)

        return {"pp": pp, "r1": r1, "r2": r2, "r3": r3, "s1": s1, "s2": s2, "s3": s3}

    # ── Pre-filters ───────────────────────────────────────────────────────────

    def _check_session_filter(self, ts) -> bool:
        """Only operate during active market hours (UTC)."""
        try:
            hour = pd.to_datetime(ts, utc=True).hour
        except Exception:
            return True  # Fail open if timestamp parsing fails
        return self._session_start <= hour < self._session_end

    def _check_volume_filter(self, bars: pd.DataFrame) -> bool:
        """Reject signals when current volume is too low vs. recent average."""
        if len(bars) < self._vol_avg_period:
            return True  # Not enough history, don't filter
        vol_avg = bars["volume"].rolling(window=self._vol_avg_period).mean().iloc[-1]
        if pd.isna(vol_avg) or vol_avg == 0:
            return True
        return float(bars["volume"].iloc[-1]) >= vol_avg * self._vol_min_ratio

    # ── Pivot proximity ───────────────────────────────────────────────────────

    def _is_near_pivot_support(
        self, price: float, pivots: dict | None, atr: float
    ) -> bool:
        """True if price is within pivot_atr_proximity × ATR of any support level."""
        if pivots is None or pd.isna(atr) or atr == 0:
            return False
        threshold = atr * self._pivot_atr_proximity
        for level in [pivots["s1"], pivots["s2"], pivots["s3"], pivots["pp"]]:
            if abs(price - level) <= threshold:
                return True
        return False

    def _is_near_pivot_resistance(
        self, price: float, pivots: dict | None, atr: float
    ) -> bool:
        """True if price is within pivot_atr_proximity × ATR of any resistance level."""
        if pivots is None or pd.isna(atr) or atr == 0:
            return False
        threshold = atr * self._pivot_atr_proximity
        for level in [pivots["r1"], pivots["r2"], pivots["r3"], pivots["pp"]]:
            if abs(price - level) <= threshold:
                return True
        return False

    def _next_pivot_resistance(self, price: float, pivots: dict | None) -> float | None:
        """Return the nearest Pivot resistance level above the current price."""
        if pivots is None:
            return None
        candidates = [v for v in [pivots["pp"], pivots["r1"], pivots["r2"], pivots["r3"]] if v > price]
        return min(candidates) if candidates else None

    def _next_pivot_support(self, price: float, pivots: dict | None) -> float | None:
        """Return the nearest Pivot support level below the current price."""
        if pivots is None:
            return None
        candidates = [v for v in [pivots["pp"], pivots["s1"], pivots["s2"], pivots["s3"]] if v < price]
        return max(candidates) if candidates else None

    # ── SL/TP override: ATR-based ─────────────────────────────────────────────

    def _compute_long_sl_tp(self, entry: float) -> tuple[float, float]:
        """
        Long SL/TP using ATR.
        SL = swing_low - ATR × atr_sl_mult
        TP = entry + risk × rr_ratio  (capped at next Pivot resistance if closer)
        """
        sl = self._divergence_swing_price - (self._current_atr or 0) * self._atr_sl_mult
        risk = entry - sl
        if risk <= 0:
            risk = (self._current_atr or entry * 0.005)
        tp = entry + risk * self._rr_ratio
        return sl, tp

    def _compute_short_sl_tp(self, entry: float) -> tuple[float, float]:
        """
        Short SL/TP using ATR.
        SL = swing_high + ATR × atr_sl_mult
        TP = entry - risk × rr_ratio  (capped at next Pivot support if closer)
        """
        sl = self._divergence_swing_price + (self._current_atr or 0) * self._atr_sl_mult
        risk = sl - entry
        if risk <= 0:
            risk = (self._current_atr or entry * 0.005)
        tp = entry - risk * self._rr_ratio
        return sl, tp

    # ── Main signal logic (full override) ────────────────────────────────────

    def on_bar(self, bars: pd.DataFrame) -> Signal:
        ts = bars["ts"].iloc[-1]
        if hasattr(ts, "to_pydatetime"):
            ts = ts.to_pydatetime()

        def hold(reason: str, meta: dict | None = None) -> Signal:
            return Signal(
                action=SignalAction.HOLD, symbol=self.symbol, ts=ts,
                strategy_id=self.strategy_id, reason=reason, meta=meta or {},
            )

        def close_pos(reason: str, meta: dict | None = None) -> Signal:
            return Signal(
                action=SignalAction.CLOSE, symbol=self.symbol, ts=ts,
                strategy_id=self.strategy_id, reason=reason, meta=meta or {},
            )

        if len(bars) < self.min_bars_required:
            return hold(f"warmup ({len(bars)}/{self.min_bars_required})")

        # ── Compute all indicators ────────────────────────────────────
        df = self._compute_indicators(bars)   # adds rsi, ema columns

        atr_series = self._compute_atr(bars)
        vwap_series = self._compute_vwap(bars)
        pivots = self._compute_daily_pivots(bars)

        cur_close = float(df["close"].iloc[-1])
        cur_high = float(df["high"].iloc[-1])
        cur_low = float(df["low"].iloc[-1])
        cur_ema = float(df["ema"].iloc[-1]) if not pd.isna(df["ema"].iloc[-1]) else 0.0
        cur_rsi = float(df["rsi"].iloc[-1]) if not pd.isna(df["rsi"].iloc[-1]) else 50.0
        cur_atr = float(atr_series.iloc[-1]) if not pd.isna(atr_series.iloc[-1]) else 0.0
        cur_vwap = float(vwap_series[-1]) if not pd.isna(vwap_series[-1]) else cur_close

        # Store ATR for use in SL/TP computation
        self._current_atr = cur_atr

        base_meta = {
            "state": self._state,
            "rsi": round(cur_rsi, 2),
            "ema": round(cur_ema, 4),
            "vwap": round(cur_vwap, 4),
            "atr": round(cur_atr, 4),
            "sl": self._sl_price,
            "tp": self._tp_price,
            "pivot_pp": round(pivots["pp"], 4) if pivots else None,
            "pivot_s1": round(pivots["s1"], 4) if pivots else None,
            "pivot_r1": round(pivots["r1"], 4) if pivots else None,
        }

        # ── LONG: monitor SL/TP ───────────────────────────────────────
        if self._state == "LONG":
            hit_tp = cur_high >= self._tp_price
            hit_sl = cur_low <= self._sl_price
            # Worst-case: if both hit same bar → assume SL (conservative)
            if hit_sl or (hit_tp and hit_sl):
                sl = self._sl_price
                self._reset_to_flat()
                return close_pos(f"long_sl_hit sl={sl:.4f}", base_meta)
            if hit_tp:
                tp = self._tp_price
                self._reset_to_flat()
                return close_pos(f"long_tp_hit tp={tp:.4f}", base_meta)
            return hold("long_open", base_meta)

        # ── SHORT: monitor SL/TP ──────────────────────────────────────
        if self._state == "SHORT":
            hit_tp = cur_low <= self._tp_price
            hit_sl = cur_high >= self._sl_price
            # Worst-case: if both hit same bar → assume SL
            if hit_sl or (hit_tp and hit_sl):
                sl = self._sl_price
                self._reset_to_flat()
                return close_pos(f"short_sl_hit sl={sl:.4f}", base_meta)
            if hit_tp:
                tp = self._tp_price
                self._reset_to_flat()
                return close_pos(f"short_tp_hit tp={tp:.4f}", base_meta)
            return hold("short_open", base_meta)

        # ── ARMED: wait for EMA + VWAP trigger ───────────────────────
        if self._state == "ARMED":
            self._armed_bars_elapsed += 1

            if self._armed_bars_elapsed > self._trigger_window:
                self._state = "FLAT"
                self._armed_direction = None
                self._armed_bars_elapsed = 0
                # Fall through to FLAT scan below

            else:
                if self._armed_direction == "LONG":
                    ema_trigger = cur_close > cur_ema
                    vwap_ok = cur_close >= cur_vwap
                    if ema_trigger and vwap_ok:
                        sl, tp = self._compute_long_sl_tp(cur_close)
                        # Cap TP at next Pivot resistance if it's closer
                        next_res = self._next_pivot_resistance(cur_close, pivots)
                        if next_res and next_res < tp:
                            tp = next_res
                        self._entry_price = cur_close
                        self._sl_price = sl
                        self._tp_price = tp
                        self._state = "LONG"
                        return Signal(
                            action=SignalAction.BUY,
                            symbol=self.symbol, ts=ts,
                            strategy_id=self.strategy_id,
                            confidence=1.0,
                            stop_loss=Decimal(str(round(sl, 8))),
                            take_profit=Decimal(str(round(tp, 8))),
                            reason=(
                                f"hybrid_long_entry close={cur_close:.4f} "
                                f"ema={cur_ema:.4f} vwap={cur_vwap:.4f} "
                                f"atr={cur_atr:.4f} sl={sl:.4f} tp={tp:.4f}"
                            ),
                            meta={**base_meta, "swing_low": self._divergence_swing_price},
                        )

                if self._armed_direction == "SHORT" and self._allow_short:
                    ema_trigger = cur_close < cur_ema
                    vwap_ok = cur_close <= cur_vwap
                    if ema_trigger and vwap_ok:
                        sl, tp = self._compute_short_sl_tp(cur_close)
                        # Cap TP at next Pivot support if it's closer
                        next_sup = self._next_pivot_support(cur_close, pivots)
                        if next_sup and next_sup > tp:
                            tp = next_sup
                        self._entry_price = cur_close
                        self._sl_price = sl
                        self._tp_price = tp
                        self._state = "SHORT"
                        return Signal(
                            action=SignalAction.SELL,
                            symbol=self.symbol, ts=ts,
                            strategy_id=self.strategy_id,
                            confidence=1.0,
                            stop_loss=Decimal(str(round(sl, 8))),
                            take_profit=Decimal(str(round(tp, 8))),
                            reason=(
                                f"hybrid_short_entry close={cur_close:.4f} "
                                f"ema={cur_ema:.4f} vwap={cur_vwap:.4f} "
                                f"atr={cur_atr:.4f} sl={sl:.4f} tp={tp:.4f}"
                            ),
                            meta={**base_meta, "swing_high": self._divergence_swing_price},
                        )

                return hold(
                    f"armed_{self._armed_direction} {self._armed_bars_elapsed}/{self._trigger_window} "
                    f"ema_ok={cur_close > cur_ema if self._armed_direction == 'LONG' else cur_close < cur_ema} "
                    f"vwap_ok={cur_close >= cur_vwap if self._armed_direction == 'LONG' else cur_close <= cur_vwap}",
                    base_meta,
                )

        # ── FLAT: pre-filters → scan for setups ──────────────────────

        # Pre-filter 1: session
        if not self._check_session_filter(bars["ts"].iloc[-1]):
            return hold(f"outside_session (hour={pd.to_datetime(bars['ts'].iloc[-1], utc=True).hour})", base_meta)

        # Pre-filter 2: volume
        if not self._check_volume_filter(bars):
            return hold("low_volume", base_meta)

        # LONG: divergencia bullish cerca de soporte en Pivot
        if (self._is_near_pivot_support(cur_close, pivots, cur_atr)
                and self._detect_bullish_divergence(df)):
            self._state = "ARMED"
            self._armed_direction = "LONG"
            self._armed_bars_elapsed = 0
            return hold(
                f"bullish_div_armed near_pivot_support "
                f"swing_low={self._divergence_swing_price:.4f} rsi={self._divergence_swing_rsi:.2f}",
                {**base_meta, "divergence": "bullish"},
            )

        # SHORT: divergencia bearish cerca de resistencia en Pivot
        if (self._allow_short
                and self._is_near_pivot_resistance(cur_close, pivots, cur_atr)
                and self._detect_bearish_divergence(df)):
            self._state = "ARMED"
            self._armed_direction = "SHORT"
            self._armed_bars_elapsed = 0
            return hold(
                f"bearish_div_armed near_pivot_resistance "
                f"swing_high={self._divergence_swing_price:.4f} rsi={self._divergence_swing_rsi:.2f}",
                {**base_meta, "divergence": "bearish"},
            )

        return hold("no_setup", base_meta)

    # ── Public utility ────────────────────────────────────────────────────────

    def compute_indicators(self, bars: pd.DataFrame) -> pd.DataFrame:
        """
        Returns bars enriched with rsi, ema, vwap, atr, swing_low (bool), swing_high (bool),
        and all Pivot Point levels as columns. Useful for visualization.
        """
        df = self._compute_indicators(bars)
        df["atr"] = self._compute_atr(bars)
        df["vwap"] = self._compute_vwap(bars)
        df["swing_low"] = False
        df["swing_high"] = False

        for idx in self._find_swing_lows(df):
            df.at[idx, "swing_low"] = True
        for idx in self._find_swing_highs(df):
            df.at[idx, "swing_high"] = True

        pivots = self._compute_daily_pivots(bars)
        if pivots:
            for k, v in pivots.items():
                df[f"pivot_{k}"] = v

        return df
