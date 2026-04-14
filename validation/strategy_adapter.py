"""
Strategy adapter: wraps any BaseStrategy into a simple backtest callable.

The adapter runs bar-by-bar, collects trades, and returns a list of
trade dicts with entry/exit prices, pnl, etc.  This decouples the
validation framework from any specific strategy implementation.

Usage:
    adapter = StrategyAdapter(TrendFollowingV2, default_params={...})
    trades = adapter.run(df, params_override={"adx_min": 25})
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Type

import pandas as pd

from app.strategy.base import BaseStrategy
from app.strategy.signals import SignalAction


@dataclass
class TradeRecord:
    direction: str          # "LONG" or "SHORT"
    entry_ts: Any = None
    entry_price: float = 0.0
    exit_ts: Any = None
    exit_price: float = 0.0
    pnl_pct: float = 0.0
    exit_type: str = ""     # "sl", "tp1", "tp2", "reversal", "end_of_data"
    confidence: float = 0.0
    sl: float = 0.0
    tp1: float = 0.0
    tp2: float = 0.0
    bars_held: int = 0
    entry_bar_idx: int = -1

    def to_dict(self) -> dict:
        return {
            "direction": self.direction,
            "entry_ts": self.entry_ts,
            "entry_price": self.entry_price,
            "exit_ts": self.exit_ts,
            "exit_price": self.exit_price,
            "pnl_pct": self.pnl_pct,
            "exit_type": self.exit_type,
            "confidence": self.confidence,
            "sl": self.sl,
            "tp1": self.tp1,
            "tp2": self.tp2,
            "bars_held": self.bars_held,
            "entry_bar_idx": self.entry_bar_idx,
        }


@dataclass
class BacktestMetrics:
    """Summary metrics from a single backtest run."""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    winrate: float = 0.0
    total_pnl_pct: float = 0.0
    avg_pnl_pct: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe_ratio: float = 0.0
    annual_return_pct: float = 0.0
    trades: list[dict] = field(default_factory=list)

    def passes_minimum(self, min_annual: float = 15.0, min_sharpe: float = 1.0) -> bool:
        return self.annual_return_pct >= min_annual and self.sharpe_ratio >= min_sharpe


class StrategyAdapter:
    """
    Generic adapter that runs any BaseStrategy subclass bar-by-bar
    on a DataFrame and collects trade results.
    """

    def __init__(
        self,
        strategy_cls: Type[BaseStrategy],
        symbol: str = "BTCUSDT",
        default_params: dict[str, Any] | None = None,
        window_size: int = 250,
    ) -> None:
        self.strategy_cls = strategy_cls
        self.symbol = symbol
        self.default_params = default_params or {}
        self.window_size = window_size

    def run(
        self,
        df: pd.DataFrame,
        params_override: dict[str, Any] | None = None,
    ) -> BacktestMetrics:
        """
        Run the strategy on df and return metrics + trade list.

        Args:
            df: OHLCV DataFrame with columns [ts, open, high, low, close, volume]
            params_override: override specific params (merged with defaults)

        Returns:
            BacktestMetrics with trade details
        """
        params = {**self.default_params, **(params_override or {})}
        strategy = self.strategy_cls(self.symbol, params)

        trades: list[TradeRecord] = []
        current_trade: TradeRecord | None = None
        entry_bar_idx = 0
        equity_curve = [0.0]  # cumulative pnl %

        has_on_bar_all = hasattr(strategy, "on_bar_all")

        for i in range(strategy.min_bars_required, len(df)):
            window_start = max(0, i - self.window_size + 1)
            window = df.iloc[window_start : i + 1].copy()

            if has_on_bar_all:
                signals = strategy.on_bar_all(window)
            else:
                signals = [strategy.on_bar(window)]

            row = df.iloc[i]
            close = float(row["close"])

            for sig in signals:
                if sig.action == SignalAction.CLOSE:
                    if current_trade is not None:
                        exit_price = float(sig.meta.get("exit_price", close))
                        current_trade.exit_ts = row["ts"]
                        current_trade.exit_price = exit_price
                        current_trade.exit_type = sig.meta.get("exit_type", "close")
                        current_trade.bars_held = i - entry_bar_idx
                        current_trade.pnl_pct = self._calc_pnl(current_trade, exit_price)
                        trades.append(current_trade)
                        equity_curve.append(equity_curve[-1] + current_trade.pnl_pct)
                        current_trade = None

                elif sig.action == SignalAction.PARTIAL_CLOSE:
                    # Track partial close as reduced exposure but keep trade open
                    pass

                elif sig.action in (SignalAction.BUY, SignalAction.SELL):
                    if current_trade is not None:
                        # Force close previous (shouldn't happen normally)
                        current_trade.exit_ts = row["ts"]
                        current_trade.exit_price = close
                        current_trade.exit_type = "forced"
                        current_trade.bars_held = i - entry_bar_idx
                        current_trade.pnl_pct = self._calc_pnl(current_trade, close)
                        trades.append(current_trade)
                        equity_curve.append(equity_curve[-1] + current_trade.pnl_pct)

                    direction = "LONG" if sig.action == SignalAction.BUY else "SHORT"
                    current_trade = TradeRecord(
                        direction=direction,
                        entry_ts=row["ts"],
                        entry_price=close,
                        confidence=sig.confidence,
                        sl=float(sig.meta.get("sl", 0)),
                        tp1=float(sig.meta.get("tp1", 0)),
                        tp2=float(sig.meta.get("tp2", 0)),
                    )
                    entry_bar_idx = i

        # Close any open trade at end of data
        if current_trade is not None:
            last_close = float(df.iloc[-1]["close"])
            current_trade.exit_ts = df.iloc[-1]["ts"]
            current_trade.exit_price = last_close
            current_trade.exit_type = "end_of_data"
            current_trade.bars_held = len(df) - 1 - entry_bar_idx
            current_trade.pnl_pct = self._calc_pnl(current_trade, last_close)
            trades.append(current_trade)
            equity_curve.append(equity_curve[-1] + current_trade.pnl_pct)

        return self._compute_metrics(trades, equity_curve, df)

    @staticmethod
    def _calc_pnl(trade: TradeRecord, exit_price: float) -> float:
        if trade.direction == "LONG":
            return (exit_price - trade.entry_price) / trade.entry_price * 100
        else:
            return (trade.entry_price - exit_price) / trade.entry_price * 100

    @staticmethod
    def _compute_metrics(
        trades: list[TradeRecord],
        equity_curve: list[float],
        df: pd.DataFrame,
    ) -> BacktestMetrics:
        import numpy as np

        m = BacktestMetrics()
        m.trades = [t.to_dict() for t in trades]
        m.total_trades = len(trades)

        if not trades:
            return m

        pnls = [t.pnl_pct for t in trades]
        m.winning_trades = sum(1 for p in pnls if p > 0)
        m.losing_trades = sum(1 for p in pnls if p <= 0)
        m.winrate = m.winning_trades / m.total_trades * 100
        m.total_pnl_pct = sum(pnls)
        m.avg_pnl_pct = np.mean(pnls)

        # Max drawdown from equity curve (compounded)
        pnl_arr = np.array([t.pnl_pct for t in trades])
        mult = 1.0 + pnl_arr / 100.0
        equity = np.concatenate([[1.0], np.cumprod(mult)])
        running_max = np.maximum.accumulate(equity)
        dd = (equity - running_max) / running_max
        m.max_drawdown_pct = abs(float(np.min(dd))) * 100 if len(dd) > 0 else 0.0

        # Annualized return (approximate)
        if len(df) > 1:
            ts_start = pd.Timestamp(df["ts"].iloc[0])
            ts_end = pd.Timestamp(df["ts"].iloc[-1])
            days = max((ts_end - ts_start).total_seconds() / 86400, 1)
            m.annual_return_pct = m.total_pnl_pct * (365.0 / days)
        else:
            m.annual_return_pct = 0.0

        # Sharpe ratio (annualized, assuming ~365/avg_hold_days trades per year)
        if len(pnls) >= 2:
            pnl_arr = np.array(pnls)
            mean_r = np.mean(pnl_arr)
            std_r = np.std(pnl_arr, ddof=1)
            if std_r > 0:
                avg_bars = np.mean([t.bars_held for t in trades])
                # Estimate trades per year based on timeframe
                tf_minutes = _guess_tf_minutes(df)
                bars_per_year = 365 * 24 * 60 / tf_minutes
                trades_per_year = bars_per_year / max(avg_bars, 1)
                m.sharpe_ratio = (mean_r / std_r) * np.sqrt(trades_per_year)

        return m


def _guess_tf_minutes(df: pd.DataFrame) -> float:
    """Estimate timeframe in minutes from the first two timestamps."""
    if len(df) < 2:
        return 15.0
    delta = (pd.Timestamp(df["ts"].iloc[1]) - pd.Timestamp(df["ts"].iloc[0])).total_seconds()
    return max(delta / 60, 1.0)
