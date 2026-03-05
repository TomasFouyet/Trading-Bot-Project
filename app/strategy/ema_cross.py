"""
EMA Crossover Strategy with SMA200 trend filter.

Entry rules:
  LONG:  EMA20 crosses above EMA50 AND close > SMA200
  SHORT: (disabled by default — spot/no-leverage mode)
  CLOSE: EMA20 crosses below EMA50

Position sizing: fixed fraction of portfolio (configured via params).

Parameters:
  ema_fast:    int, default 20
  ema_slow:    int, default 50
  sma_trend:   int, default 200
  allow_short: bool, default False
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from app.broker.base import OHLCVBar
from app.strategy.base import BaseStrategy
from app.strategy.signals import Signal, SignalAction


class EMACrossStrategy(BaseStrategy):
    """
    Dual EMA crossover with SMA200 trend filter.

    This is the simplest viable trend-following strategy.
    Intended as a reference implementation; replace with your own.
    """

    def __init__(self, symbol: str, params: dict[str, Any] | None = None) -> None:
        super().__init__(symbol, params)
        self._ema_fast = int(self.params.get("ema_fast", 20))
        self._ema_slow = int(self.params.get("ema_slow", 50))
        self._sma_trend = int(self.params.get("sma_trend", 200))
        self._allow_short = bool(self.params.get("allow_short", False))

        # Track previous cross state to detect crossovers
        self._prev_fast_above_slow: bool | None = None

    @property
    def strategy_id(self) -> str:
        return (
            f"ema_cross_{self._ema_fast}_{self._ema_slow}_"
            f"sma{self._sma_trend}_short{self._allow_short}"
        )

    @property
    def min_bars_required(self) -> int:
        return self._sma_trend + 5  # Need full SMA200 window + buffer

    def on_bar(self, bars: pd.DataFrame) -> Signal:
        """
        Compute indicators on the given DataFrame and return a signal.
        bars must have at least min_bars_required rows.
        """
        now_ts = bars["ts"].iloc[-1]
        if isinstance(now_ts, pd.Timestamp):
            now_ts = now_ts.to_pydatetime()

        if len(bars) < self.min_bars_required:
            return Signal(
                action=SignalAction.HOLD,
                symbol=self.symbol,
                ts=now_ts,
                strategy_id=self.strategy_id,
                reason=f"not_enough_bars ({len(bars)}/{self.min_bars_required})",
            )

        closes = bars["close"]

        # ── Compute indicators ────────────────────────────────────────────
        ema_fast = closes.ewm(span=self._ema_fast, adjust=False).mean()
        ema_slow = closes.ewm(span=self._ema_slow, adjust=False).mean()
        sma_trend = closes.rolling(window=self._sma_trend).mean()

        fast_now = ema_fast.iloc[-1]
        slow_now = ema_slow.iloc[-1]
        fast_prev = ema_fast.iloc[-2]
        slow_prev = ema_slow.iloc[-2]
        sma_now = sma_trend.iloc[-1]
        close_now = closes.iloc[-1]

        if np.isnan(sma_now):
            return Signal(
                action=SignalAction.HOLD,
                symbol=self.symbol,
                ts=now_ts,
                strategy_id=self.strategy_id,
                reason="sma_not_ready",
            )

        # ── Cross detection ───────────────────────────────────────────────
        bullish_cross = (fast_prev <= slow_prev) and (fast_now > slow_now)
        bearish_cross = (fast_prev >= slow_prev) and (fast_now < slow_now)
        above_trend = close_now > sma_now

        meta = {
            "ema_fast": round(fast_now, 6),
            "ema_slow": round(slow_now, 6),
            "sma_trend": round(sma_now, 6),
            "close": round(close_now, 6),
            "above_trend": above_trend,
            "bullish_cross": bullish_cross,
            "bearish_cross": bearish_cross,
        }

        # ── Signal logic ─────────────────────────────────────────────────
        if bullish_cross and above_trend:
            return Signal(
                action=SignalAction.BUY,
                symbol=self.symbol,
                ts=now_ts,
                strategy_id=self.strategy_id,
                confidence=1.0,
                reason=f"EMA{self._ema_fast} crossed above EMA{self._ema_slow}, above SMA{self._sma_trend}",
                meta=meta,
            )

        if bearish_cross:
            return Signal(
                action=SignalAction.CLOSE,
                symbol=self.symbol,
                ts=now_ts,
                strategy_id=self.strategy_id,
                confidence=1.0,
                reason=f"EMA{self._ema_fast} crossed below EMA{self._ema_slow}",
                meta=meta,
            )

        if self._allow_short and bearish_cross and not above_trend:
            return Signal(
                action=SignalAction.SELL,
                symbol=self.symbol,
                ts=now_ts,
                strategy_id=self.strategy_id,
                confidence=0.8,
                reason=f"EMA{self._ema_fast} crossed below EMA{self._ema_slow}, below SMA{self._sma_trend}",
                meta=meta,
            )

        return Signal(
            action=SignalAction.HOLD,
            symbol=self.symbol,
            ts=now_ts,
            strategy_id=self.strategy_id,
            reason="no_cross",
            meta=meta,
        )

    def compute_indicators(self, bars: pd.DataFrame) -> pd.DataFrame:
        """Utility for external use (backtesting reports, dashboard)."""
        df = bars.copy()
        df[f"ema_{self._ema_fast}"] = df["close"].ewm(span=self._ema_fast, adjust=False).mean()
        df[f"ema_{self._ema_slow}"] = df["close"].ewm(span=self._ema_slow, adjust=False).mean()
        df[f"sma_{self._sma_trend}"] = df["close"].rolling(window=self._sma_trend).mean()
        return df
