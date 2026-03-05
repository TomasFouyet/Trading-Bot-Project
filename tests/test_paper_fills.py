"""
Unit tests for PaperAdapter fill simulation.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.broker.base import OrderRequest, OrderSide, OrderStatus, OrderType
from app.broker.paper_adapter import PaperAdapter


def make_mock_client(last_price: float = 40000.0):
    """Create a mock BingX client that returns a fixed ticker price."""
    client = MagicMock()
    client.get_ticker = AsyncMock(
        return_value={
            "symbol": "BTC-USDT",
            "last": last_price,
            "bid": last_price - 1,
            "ask": last_price + 1,
            "ts": datetime.now(timezone.utc),
        }
    )
    client.fetch_ohlcv = AsyncMock(return_value=[])
    client.close = AsyncMock()
    return client


@pytest.mark.asyncio
async def test_paper_market_buy_fills_immediately():
    """Market buy order fills immediately with price + slippage."""
    client = make_mock_client(40000.0)
    adapter = PaperAdapter(client=client, initial_balance=Decimal("10000"))

    req = OrderRequest(
        symbol="BTC-USDT",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        qty=Decimal("0.01"),
    )
    order, fill = await adapter.place_order(req)

    assert order.status == OrderStatus.FILLED
    assert fill is not None
    assert fill.qty == Decimal("0.01")
    assert fill.price > Decimal("40000")  # price + slippage


@pytest.mark.asyncio
async def test_paper_market_sell_fills_immediately():
    client = make_mock_client(40000.0)
    adapter = PaperAdapter(client=client, initial_balance=Decimal("10000"))

    # First buy
    buy_req = OrderRequest(
        symbol="BTC-USDT",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        qty=Decimal("0.01"),
    )
    await adapter.place_order(buy_req)

    # Then sell
    sell_req = OrderRequest(
        symbol="BTC-USDT",
        side=OrderSide.SELL,
        order_type=OrderType.MARKET,
        qty=Decimal("0.01"),
    )
    order, fill = await adapter.place_order(sell_req)

    assert order.status == OrderStatus.FILLED
    assert fill is not None
    assert fill.price < Decimal("40000")  # price - slippage


@pytest.mark.asyncio
async def test_paper_balance_decreases_on_buy():
    """Balance decreases by notional + fee on buy."""
    client = make_mock_client(40000.0)
    adapter = PaperAdapter(client=client, initial_balance=Decimal("10000"))
    initial = adapter.cash_balance

    req = OrderRequest(
        symbol="BTC-USDT",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        qty=Decimal("0.1"),
    )
    await adapter.place_order(req)

    assert adapter.cash_balance < initial


@pytest.mark.asyncio
async def test_paper_limit_order_not_filled_immediately():
    """Limit orders are recorded as OPEN, not filled right away."""
    client = make_mock_client(40000.0)
    adapter = PaperAdapter(client=client, initial_balance=Decimal("10000"))

    req = OrderRequest(
        symbol="BTC-USDT",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        qty=Decimal("0.01"),
        price=Decimal("39000"),
    )
    order, fill = await adapter.place_order(req)

    assert order.status == OrderStatus.OPEN
    assert fill is None


@pytest.mark.asyncio
async def test_paper_cancel_open_order():
    client = make_mock_client(40000.0)
    adapter = PaperAdapter(client=client, initial_balance=Decimal("10000"))

    req = OrderRequest(
        symbol="BTC-USDT",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        qty=Decimal("0.01"),
        price=Decimal("39000"),
    )
    order, _ = await adapter.place_order(req)
    cancelled = await adapter.cancel_order(order.order_id, "BTC-USDT")

    assert cancelled
    retrieved = await adapter.get_order(order.order_id, "BTC-USDT")
    assert retrieved.status == OrderStatus.CANCELLED


@pytest.mark.asyncio
async def test_paper_fee_applied():
    """Fee should be positive and proportional to trade size."""
    client = make_mock_client(40000.0)
    adapter = PaperAdapter(client=client, initial_balance=Decimal("10000"))

    req = OrderRequest(
        symbol="BTC-USDT",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        qty=Decimal("0.1"),
    )
    _, fill = await adapter.place_order(req)

    assert fill.fee > 0
    # Fee should be approximately price * qty * commission_rate
    expected_fee_approx = fill.price * fill.qty * Decimal("0.00075")
    assert abs(fill.fee - expected_fee_approx) < Decimal("0.01")
