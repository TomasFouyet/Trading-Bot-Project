"""
Pytest fixtures shared across all tests.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pandas as pd
import pytest

from app.broker.base import OHLCVBar


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_bars() -> list[OHLCVBar]:
    """Generate 250 synthetic bars (enough for SMA200 + buffer)."""
    bars = []
    base_price = 40000.0
    ts = datetime(2024, 1, 1, tzinfo=timezone.utc)

    # Simulate a rising trend then a fall to produce at least one EMA cross
    for i in range(250):
        if i < 80:
            close = base_price + i * 100        # rising
        elif i < 150:
            close = base_price + 8000 - (i - 80) * 50  # falling
        else:
            close = base_price + 4500 + (i - 150) * 80  # rising again

        bars.append(
            OHLCVBar(
                symbol="BTC-USDT",
                timeframe="5m",
                ts=ts + timedelta(minutes=5 * i),
                open=Decimal(str(close - 50)),
                high=Decimal(str(close + 100)),
                low=Decimal(str(close - 100)),
                close=Decimal(str(close)),
                volume=Decimal("10.5"),
            )
        )
    return bars


@pytest.fixture
def sample_df(sample_bars) -> pd.DataFrame:
    from app.strategy.base import BaseStrategy
    return BaseStrategy.bars_to_df(sample_bars)


@pytest.fixture
def ema_strategy():
    from app.strategy.ema_cross import EMACrossStrategy
    return EMACrossStrategy(
        symbol="BTC-USDT",
        params={"ema_fast": 20, "ema_slow": 50, "sma_trend": 200},
    )


@pytest.fixture
def risk_manager():
    from app.risk.manager import RiskManager
    rm = RiskManager(
        max_daily_drawdown_pct=Decimal("5.0"),
        max_position_pct=Decimal("10.0"),
        max_trade_risk_pct=Decimal("2.0"),
        max_consecutive_api_errors=5,
    )
    rm.initialize(Decimal("10000"))
    return rm
