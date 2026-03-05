"""
Unit tests for the backtest engine — deterministic with a fixed dataset.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from decimal import Decimal

import pytest

from app.broker.backtest_adapter import BacktestAdapter
from app.broker.base import OHLCVBar, OrderRequest, OrderSide, OrderStatus, OrderType


def make_bars(n: int = 10, base_price: float = 40000.0) -> list[OHLCVBar]:
    from datetime import timedelta
    bars = []
    ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
    for i in range(n):
        price = base_price + i * 100
        bars.append(
            OHLCVBar(
                symbol="BTC-USDT",
                timeframe="5m",
                ts=ts + timedelta(minutes=5 * i),
                open=Decimal(str(price)),
                high=Decimal(str(price + 50)),
                low=Decimal(str(price - 50)),
                close=Decimal(str(price + 10)),
                volume=Decimal("5.0"),
            )
        )
    return bars


@pytest.mark.asyncio
async def test_backtest_adapter_market_order_fills_on_next_bar():
    """Market order placed at bar[0] must fill at bar[1].open + slippage."""
    bars = make_bars(5)
    adapter = BacktestAdapter(bars=bars, initial_balance=Decimal("100000"))

    adapter.advance(bars[0])
    req = OrderRequest(
        symbol="BTC-USDT",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        qty=Decimal("0.01"),
    )
    order, fill = await adapter.place_order(req)
    assert order.status == OrderStatus.PENDING
    assert fill is None

    # Advance to next bar — order should fill
    fills = adapter.advance(bars[1])
    assert len(fills) == 1
    assert fills[0].order_id == order.order_id
    assert fills[0].qty == Decimal("0.01")

    # Check the order status was updated
    filled_order = await adapter.get_order(order.order_id, "BTC-USDT")
    assert filled_order.status == OrderStatus.FILLED


@pytest.mark.asyncio
async def test_backtest_adapter_balance_decreases_on_buy():
    bars = make_bars(5)
    adapter = BacktestAdapter(bars=bars, initial_balance=Decimal("100000"))
    adapter.advance(bars[0])

    req = OrderRequest(
        symbol="BTC-USDT",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        qty=Decimal("0.1"),
    )
    await adapter.place_order(req)
    adapter.advance(bars[1])  # fills the order

    balances = await adapter.get_balance()
    assert balances[0].total < Decimal("100000")


@pytest.mark.asyncio
async def test_backtest_adapter_cancel_order():
    bars = make_bars(5)
    adapter = BacktestAdapter(bars=bars, initial_balance=Decimal("100000"))
    adapter.advance(bars[0])

    req = OrderRequest(
        symbol="BTC-USDT",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        qty=Decimal("0.01"),
    )
    order, _ = await adapter.place_order(req)
    cancelled = await adapter.cancel_order(order.order_id, "BTC-USDT")
    assert cancelled

    # Advance — should NOT fill because it was cancelled
    fills = adapter.advance(bars[1])
    assert len(fills) == 0


@pytest.mark.asyncio
async def test_backtest_engine_runs_deterministically(sample_bars):
    """Full backtest run produces consistent results with same inputs."""
    from app.data.parquet_store import ParquetStore
    from app.engine.backtest import BacktestEngine
    from app.strategy.ema_cross import EMACrossStrategy

    strategy = EMACrossStrategy(
        symbol="BTC-USDT",
        params={"ema_fast": 5, "ema_slow": 10, "sma_trend": 20},
    )

    # Use a temporary in-memory store seeded with our bars
    import tempfile, os
    with tempfile.TemporaryDirectory() as tmpdir:
        store = ParquetStore(base_dir=tmpdir)
        store.write_bars("BTC-USDT", "5m", sample_bars)

        engine = BacktestEngine(
            strategy=strategy,
            store=store,
            initial_balance=Decimal("10000"),
        )

        start = sample_bars[0].ts
        end = sample_bars[-1].ts

        result1 = await engine.run("BTC-USDT", "5m", start, end)
        engine._risk.reset_kill_switch()  # reset for re-run

        result2 = await engine.run("BTC-USDT", "5m", start, end)

    # Deterministic: same result twice
    assert result1.total_trades == result2.total_trades
    assert result1.total_pnl == result2.total_pnl


@pytest.mark.asyncio
async def test_backtest_engine_metrics_computed(sample_bars):
    from app.data.parquet_store import ParquetStore
    from app.engine.backtest import BacktestEngine
    from app.strategy.ema_cross import EMACrossStrategy
    import tempfile

    strategy = EMACrossStrategy(
        symbol="BTC-USDT",
        params={"ema_fast": 5, "ema_slow": 10, "sma_trend": 20},
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        store = ParquetStore(base_dir=tmpdir)
        store.write_bars("BTC-USDT", "5m", sample_bars)

        engine = BacktestEngine(strategy=strategy, store=store)
        result = await engine.run(
            "BTC-USDT", "5m", sample_bars[0].ts, sample_bars[-1].ts
        )

    assert result.total_bars == len(sample_bars)
    assert result.max_drawdown_pct >= 0
    assert 0 <= float(result.winrate) <= 100
    assert result.equity_curve  # non-empty
    report = result.to_report()
    assert "metrics" in report
    assert "equity_curve" in report
