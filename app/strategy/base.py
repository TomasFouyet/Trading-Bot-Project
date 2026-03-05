"""
Strategy ABC.

All strategies inherit from BaseStrategy and implement on_bar().
The engine feeds bars one at a time; strategies maintain their own state/indicators.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd

from app.broker.base import OHLCVBar
from app.strategy.signals import Signal


class BaseStrategy(ABC):
    """
    Abstract base for all trading strategies.

    Contract:
    - on_bar(bars) is called with a window of recent bars (pandas DataFrame).
    - Returns a Signal (possibly HOLD).
    - Must be stateless w.r.t. broker calls (those happen in the engine).
    """

    def __init__(self, symbol: str, params: dict[str, Any] | None = None) -> None:
        self.symbol = symbol
        self.params = params or {}
        self._bar_count = 0

    @property
    @abstractmethod
    def strategy_id(self) -> str:
        """Unique identifier for this strategy instance."""

    @property
    @abstractmethod
    def min_bars_required(self) -> int:
        """Minimum number of bars needed before emitting non-HOLD signals."""

    @abstractmethod
    def on_bar(self, bars: pd.DataFrame) -> Signal:
        """
        Process the latest bar window and return a Signal.

        Args:
            bars: DataFrame with columns [ts, open, high, low, close, volume].
                  Sorted ascending; last row is the most recent CLOSED bar.

        Returns:
            Signal with action = BUY | SELL | CLOSE | HOLD.
        """

    def warm_up(self, bars: list[OHLCVBar]) -> None:
        """
        Pre-load historical bars to initialize indicator state.
        Override in strategies that maintain state across bars.
        Default: no-op (DataFrame-based strategies recompute from scratch).
        """
        self._bar_count = len(bars)

    @staticmethod
    def bars_to_df(bars: list[OHLCVBar]) -> pd.DataFrame:
        """Convert list of OHLCVBar to DataFrame."""
        return pd.DataFrame(
            {
                "ts": [b.ts for b in bars],
                "open": [float(b.open) for b in bars],
                "high": [float(b.high) for b in bars],
                "low": [float(b.low) for b in bars],
                "close": [float(b.close) for b in bars],
                "volume": [float(b.volume) for b in bars],
            }
        ).sort_values("ts").reset_index(drop=True)
