"""
BingX real-money adapter.
Wraps BingXClient and maps responses to domain types.
Sets leverage=1 on initialization unless overridden in config.
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
from app.config import MarketType, get_settings
from app.core.exceptions import BrokerError
from app.core.logging import get_logger

logger = get_logger(__name__)
_settings = get_settings()


class BingXAdapter(BrokerAdapter):
    """
    Real BingX adapter. Uses live API keys.
    Only use in REAL mode after thorough testing in paper mode.
    """

    def __init__(self, client: BingXClient | None = None) -> None:
        self._client = client or BingXClient(
            api_key=_settings.bingx_api_key,
            api_secret=_settings.bingx_api_secret,
            base_url=_settings.bingx_base_url,
            market_type=_settings.bingx_market_type,
        )

    async def initialize(self) -> None:
        leverage = _settings.leverage
        if _settings.bingx_market_type == MarketType.SWAP:
            logger.info("bingx_set_leverage", leverage=leverage, symbol=_settings.default_symbol)
            try:
                await self._client.set_leverage(_settings.default_symbol, leverage)
            except Exception as e:
                logger.warning("leverage_set_failed", error=str(e))
        logger.info("bingx_adapter_initialized", mode="real", leverage=leverage)

    async def place_order(self, request: OrderRequest) -> tuple[OrderResult, FillResult | None]:
        logger.info(
            "place_order_real",
            symbol=request.symbol,
            side=request.side,
            qty=str(request.qty),
            order_type=request.order_type,
        )
        raw = await self._client.place_order(
            symbol=request.symbol,
            side=request.side.value,
            order_type=request.order_type.value,
            qty=float(request.qty),
            price=float(request.price) if request.price else None,
            client_order_id=request.client_id,
        )
        order = raw.get("order", raw)
        result = OrderResult(
            order_id=str(order.get("orderId", request.client_id)),
            broker_id=str(order.get("orderId", "")),
            symbol=request.symbol,
            side=request.side,
            order_type=request.order_type,
            qty=request.qty,
            price=Decimal(str(order.get("price", 0))) or request.price,
            status=self._map_status(order.get("status", "NEW")),
            created_at=datetime.now(timezone.utc),
        )
        # Market orders may fill immediately; return fill if available
        fill = None
        if result.status == OrderStatus.FILLED:
            fill = FillResult(
                fill_id=str(uuid.uuid4()),
                order_id=result.order_id,
                symbol=result.symbol,
                side=result.side,
                price=Decimal(str(order.get("avgPrice", order.get("price", 0)))),
                qty=result.qty,
                fee=Decimal(str(order.get("fee", 0))),
                fee_currency="USDT",
                timestamp=datetime.now(timezone.utc),
            )
        return result, fill

    async def get_order(self, order_id: str, symbol: str) -> OrderResult:
        raw = await self._client.get_order(order_id, symbol)
        return OrderResult(
            order_id=str(raw.get("orderId", order_id)),
            broker_id=str(raw.get("orderId", "")),
            symbol=symbol,
            side=OrderSide(raw.get("side", "BUY")),
            order_type=OrderType(raw.get("type", "MARKET")),
            qty=Decimal(str(raw.get("origQty", 0))),
            price=Decimal(str(raw.get("price", 0))) if raw.get("price") else None,
            status=self._map_status(raw.get("status", "NEW")),
            created_at=datetime.now(timezone.utc),
        )

    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        try:
            await self._client.cancel_order(order_id, symbol)
            return True
        except Exception as e:
            logger.warning("cancel_order_failed", order_id=order_id, error=str(e))
            return False

    async def get_positions(self, symbol: Optional[str] = None) -> list[Position]:
        raw_list = await self._client.get_positions(symbol)
        positions = []
        for p in raw_list:
            qty = Decimal(str(p.get("positionAmt", p.get("availableAmt", 0))))
            if qty == 0:
                continue
            positions.append(
                Position(
                    symbol=p.get("symbol", symbol or ""),
                    side=TradeSide.LONG if qty > 0 else TradeSide.SHORT,
                    qty=abs(qty),
                    avg_price=Decimal(str(p.get("avgPrice", p.get("entryPrice", 0)))),
                    unrealized_pnl=Decimal(str(p.get("unrealizedProfit", p.get("unrealizedPnl", 0)))),
                    liquidation_price=Decimal(str(p.get("liquidationPrice", 0))) or None,
                )
            )
        return positions

    async def get_balance(self) -> list[Balance]:
        raw_list = await self._client.get_balance()
        balances = []
        for b in raw_list:
            asset = b.get("asset", b.get("currency", "USDT"))
            total = Decimal(str(b.get("balance", b.get("totalWalletBalance", 0))))
            available = Decimal(str(b.get("availableMargin", b.get("free", total))))
            balances.append(
                Balance(
                    currency=asset,
                    total=total,
                    available=available,
                    used=total - available,
                )
            )
        return balances

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

    @staticmethod
    def _map_status(raw: str) -> OrderStatus:
        mapping = {
            "NEW": OrderStatus.OPEN,
            "PARTIALLY_FILLED": OrderStatus.PARTIALLY_FILLED,
            "FILLED": OrderStatus.FILLED,
            "CANCELED": OrderStatus.CANCELLED,
            "CANCELLED": OrderStatus.CANCELLED,
            "REJECTED": OrderStatus.REJECTED,
            "PENDING": OrderStatus.PENDING,
        }
        return mapping.get(raw.upper(), OrderStatus.PENDING)
