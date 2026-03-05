"""
Unit tests for EMACrossStrategy signal generation.
"""
from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

import pandas as pd
import pytest

from app.strategy.signals import SignalAction


def test_strategy_returns_hold_with_insufficient_bars(ema_strategy):
    """Strategy emits HOLD when not enough bars are available."""
    from app.broker.base import OHLCVBar
    from app.strategy.base import BaseStrategy

    bars = [
        OHLCVBar(
            symbol="BTC-USDT",
            timeframe="5m",
            ts=datetime(2024, 1, 1, tzinfo=timezone.utc),
            open=Decimal("40000"),
            high=Decimal("40100"),
            low=Decimal("39900"),
            close=Decimal("40050"),
            volume=Decimal("10"),
        )
    ]
    df = BaseStrategy.bars_to_df(bars)
    signal = ema_strategy.on_bar(df)
    assert signal.action == SignalAction.HOLD
    assert "not_enough_bars" in signal.reason


def test_strategy_produces_buy_on_bullish_cross(sample_bars):
    """
    Inject a clear bullish EMA cross by constructing price data
    where EMA20 definitively crosses above EMA50.
    Use small EMA windows so we need fewer bars.
    """
    from app.strategy.ema_cross import EMACrossStrategy
    from app.strategy.base import BaseStrategy

    strategy = EMACrossStrategy(
        symbol="BTC-USDT",
        params={"ema_fast": 5, "ema_slow": 10, "sma_trend": 20, "allow_short": False},
    )
    df = BaseStrategy.bars_to_df(sample_bars)
    # Run through all bars and collect BUY signals
    buy_signals = []
    for i in range(len(df)):
        window = df.iloc[: i + 1]
        sig = strategy.on_bar(window)
        if sig.action == SignalAction.BUY:
            buy_signals.append(sig)

    # With the synthetic trend data, we expect at least one BUY
    assert len(buy_signals) >= 1, "Expected at least one BUY signal"


def test_strategy_produces_close_on_bearish_cross(sample_bars):
    """Strategy emits CLOSE when EMA fast crosses below EMA slow."""
    from app.strategy.ema_cross import EMACrossStrategy
    from app.strategy.base import BaseStrategy

    strategy = EMACrossStrategy(
        symbol="BTC-USDT",
        params={"ema_fast": 5, "ema_slow": 10, "sma_trend": 20},
    )
    df = BaseStrategy.bars_to_df(sample_bars)
    close_signals = []
    for i in range(len(df)):
        window = df.iloc[: i + 1]
        sig = strategy.on_bar(window)
        if sig.action == SignalAction.CLOSE:
            close_signals.append(sig)

    assert len(close_signals) >= 1, "Expected at least one CLOSE signal"


def test_strategy_id_is_consistent(ema_strategy):
    """Strategy ID must be deterministic and not change."""
    id1 = ema_strategy.strategy_id
    id2 = ema_strategy.strategy_id
    assert id1 == id2
    assert "ema_cross" in id1


def test_signal_is_actionable():
    from app.strategy.signals import Signal, SignalAction

    buy = Signal(
        action=SignalAction.BUY,
        symbol="BTC-USDT",
        ts=datetime(2024, 1, 1, tzinfo=timezone.utc),
        strategy_id="test",
    )
    hold = Signal(
        action=SignalAction.HOLD,
        symbol="BTC-USDT",
        ts=datetime(2024, 1, 1, tzinfo=timezone.utc),
        strategy_id="test",
    )
    assert buy.is_actionable()
    assert not hold.is_actionable()
