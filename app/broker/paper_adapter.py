"""
Paper trading adapter.

Simulates fills using live ticker prices + configured slippage + commissions.
Stores the same Order/Fill/Trade objects as real trading so reporting is identical.
Requires an underlying BingXClient for market data.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from decimal import Decimal
from typing import Optional

from app.broker.base import (
    Balance,
    BrokerAdapter,
    FillResult,
    OHLCVBar,
    OrderRequest,
    OrderResult,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
    TradeSide,
)
from app.broker.bingx_client import BingXClient
from app.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)
_settings = get_settings()

SLIPPAGE = _settings.slippage_rate
COMMISSION = _settings.commission_rate


class PaperAdapter(BrokerAdapter):
    """
    Paper / forward-test adapter.

    - Connects to BingX for live ticker prices only.
    - Simulates fills immediately (market orders).
    - Tracks in-memory balance and positions.
    - All orders/fills go through the same domain objects as real trading.
    """

    def __init__(
        self,
        client: BingXClient | None = None,
        initial_balance: Decimal = Decimal("10000"),
    ) -> None:
        self._client = client or BingXClient(
            api_key=_settings.bingx_api_key,
            api_secret=_settings.bingx_api_secret,
            base_url=_settings.bingx_base_url,
            market_type=_settings.bingx_market_type,
        )
        self._initial_balance = initial_balance
        self._balance = initial_balance
        self._positions: dict[str, Position] = {}
        self._orders: dict[str, OrderResult] = {}
        self._fills: list[FillResult] = []

    async def initialize(self) -> None:
        logger.info(
            "paper_adapter_initialized",
            initial_balance=str(self._initial_balance),
        )

    async def place_order(self, request: OrderRequest) -> tuple[OrderResult, FillResult | None]:
        """
        Simulate a market or limit order fill.
        For market orders: fill immediately at ticker price ± slippage.
        For limit orders: record as OPEN (filled by engine on next bar).
        """
        order_id = str(uuid.uuid4())

        if request.order_type == OrderType.MARKET:
            ticker = await self._client.get_ticker(request.symbol)
            mid_price = Decimal(str(ticker["last"]))

            # Apply slippage: buyer pays more, seller receives less
            if request.side == OrderSide.BUY:
                fill_price = mid_price * (1 + SLIPPAGE)
            else:
                fill_price = mid_price * (1 - SLIPPAGE)

            fee = fill_price * request.qty * COMMISSION

            order = OrderResult(
                order_id=order_id,
                broker_id=None,
                symbol=request.symbol,
                side=request.side,
                order_type=request.order_type,
                qty=request.qty,
                price=fill_price,
                status=OrderStatus.FILLED,
                created_at=datetime.now(timezone.utc),
            )
            fill = FillResult(
                fill_id=str(uuid.uuid4()),
                order_id=order_id,
                symbol=request.symbol,
                side=request.side,
                price=fill_price,
                qty=request.qty,
                fee=fee,
                fee_currency="USDT",
                timestamp=datetime.now(timezone.utc),
            )
            self._orders[order_id] = order
            self._fills.append(fill)
            self._apply_fill(fill)

            logger.info(
                "paper_fill",
                symbol=request.symbol,
                side=request.side.value,
                qty=str(request.qty),
                price=str(fill_price),
                fee=str(fee),
            )
            return order, fill

        else:
            # LIMIT order: record as OPEN
            order = OrderResult(
                order_id=order_id,
                broker_id=None,
                symbol=request.symbol,
                side=request.side,
                order_type=request.order_type,
                qty=request.qty,
                price=request.price,
                status=OrderStatus.OPEN,
                created_at=datetime.now(timezone.utc),
            )
            self._orders[order_id] = order
            return order, None

    def simulate_fill_at_price(
        self, order_id: str, fill_price: Decimal, ts: datetime
    ) -> FillResult | None:
        """Fill a pending limit order at the given price (called by engine on each bar)."""
        order = self._orders.get(order_id)
        if order is None or order.status != OrderStatus.OPEN:
            return None

        fee = fill_price * order.qty * COMMISSION
        fill = FillResult(
            fill_id=str(uuid.uuid4()),
            order_id=order_id,
            symbol=order.symbol,
            side=order.side,
            price=fill_price,
            qty=order.qty,
            fee=fee,
            fee_currency="USDT",
            timestamp=ts,
        )
        order.status = OrderStatus.FILLED
        order.price = fill_price
        self._fills.append(fill)
        self._apply_fill(fill)
        return fill

    def _apply_fill(self, fill: FillResult) -> None:
        """Update in-memory balance and positions."""
        notional = fill.price * fill.qty

        if fill.side == OrderSide.BUY:
            self._balance -= notional + fill.fee
            pos = self._positions.get(fill.symbol)
            if pos is None:
                self._positions[fill.symbol] = Position(
                    symbol=fill.symbol,
                    side=TradeSide.LONG,
                    qty=fill.qty,
                    avg_price=fill.price,
                    unrealized_pnl=Decimal("0"),
                )
            else:
                # Average in
                total_qty = pos.qty + fill.qty
                pos.avg_price = (pos.avg_price * pos.qty + fill.price * fill.qty) / total_qty
                pos.qty = total_qty
        else:
            # SELL — reduce or close position
            pos = self._positions.get(fill.symbol)
            if pos and pos.qty > 0:
                pnl = (fill.price - pos.avg_price) * min(fill.qty, pos.qty)
                pos.qty -= fill.qty
                if pos.qty <= 0:
                    del self._positions[fill.symbol]
                self._balance += notional - fill.fee + pnl
            else:
                self._balance += notional - fill.fee

    async def get_order(self, order_id: str, symbol: str) -> OrderResult:
        order = self._orders.get(order_id)
        if order is None:
            from app.core.exceptions import OrderNotFoundError
            raise OrderNotFoundError(f"Paper order {order_id} not found")
        return order

    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        order = self._orders.get(order_id)
        if order and order.status == OrderStatus.OPEN:
            order.status = OrderStatus.CANCELLED
            return True
        return False

    async def get_positions(self, symbol: Optional[str] = None) -> list[Position]:
        positions = list(self._positions.values())
        if symbol:
            positions = [p for p in positions if p.symbol == symbol]
        return positions

    async def get_balance(self) -> list[Balance]:
        return [
            Balance(
                currency="USDT",
                total=self._balance,
                available=self._balance,
                used=Decimal("0"),
            )
        ]

    async def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: Optional[datetime] = None,
        limit: int = 500,
    ) -> list[OHLCVBar]:
        raw = await self._client.fetch_ohlcv(symbol, timeframe, start, end, limit)
        return [
            OHLCVBar(
                symbol=symbol,
                timeframe=timeframe,
                ts=bar["ts"],
                open=Decimal(str(bar["open"])),
                high=Decimal(str(bar["high"])),
                low=Decimal(str(bar["low"])),
                close=Decimal(str(bar["close"])),
                volume=Decimal(str(bar["volume"])),
            )
            for bar in raw
        ]

    async def get_ticker(self, symbol: str) -> dict:
        return await self._client.get_ticker(symbol)

    async def shutdown(self) -> None:
        await self._client.close()

    @property
    def cash_balance(self) -> Decimal:
        return self._balance

    def get_fills(self) -> list[FillResult]:
        return list(self._fills)
