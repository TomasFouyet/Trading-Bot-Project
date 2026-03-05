"""
BrokerAdapter ABC — the single interface shared by real, paper, and backtest modes.
Any strategy or engine uses only this interface; it never touches broker internals.
"""
from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# Domain types
# ─────────────────────────────────────────────────────────────────────────────

class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"


class OrderStatus(str, Enum):
    PENDING = "PENDING"
    OPEN = "OPEN"
    FILLED = "FILLED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


class TradeSide(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"


@dataclass
class OrderRequest:
    symbol: str
    side: OrderSide
    order_type: OrderType
    qty: Decimal
    price: Optional[Decimal] = None
    client_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    strategy_id: Optional[str] = None
    extra: dict = field(default_factory=dict)


@dataclass
class OrderResult:
    order_id: str
    broker_id: Optional[str]
    symbol: str
    side: OrderSide
    order_type: OrderType
    qty: Decimal
    price: Optional[Decimal]
    status: OrderStatus
    created_at: datetime
    extra: dict = field(default_factory=dict)


@dataclass
class FillResult:
    fill_id: str
    order_id: str
    symbol: str
    side: OrderSide
    price: Decimal
    qty: Decimal
    fee: Decimal
    fee_currency: str
    timestamp: datetime


@dataclass
class Position:
    symbol: str
    side: TradeSide
    qty: Decimal
    avg_price: Decimal
    unrealized_pnl: Decimal
    liquidation_price: Optional[Decimal] = None


@dataclass
class Balance:
    currency: str
    total: Decimal
    available: Decimal
    used: Decimal


@dataclass
class OHLCVBar:
    symbol: str
    timeframe: str
    ts: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal


# ─────────────────────────────────────────────────────────────────────────────
# Abstract interface
# ─────────────────────────────────────────────────────────────────────────────

class BrokerAdapter(ABC):
    """
    Unified broker interface.

    All three adapters (BingX real, paper, backtest) implement this.
    Strategy and engine code depends ONLY on this interface.
    """

    @abstractmethod
    async def place_order(self, request: OrderRequest) -> tuple[OrderResult, FillResult | None]:
        """
        Submit an order.
        Returns (OrderResult, FillResult | None).
        FillResult is populated immediately for market orders in paper/backtest.
        """

    @abstractmethod
    async def get_order(self, order_id: str, symbol: str) -> OrderResult:
        """Fetch current state of an order."""

    @abstractmethod
    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Cancel an open order. Returns True if successful."""

    @abstractmethod
    async def get_positions(self, symbol: Optional[str] = None) -> list[Position]:
        """Return open positions. Pass symbol to filter."""

    @abstractmethod
    async def get_balance(self) -> list[Balance]:
        """Return account balances."""

    @abstractmethod
    async def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: Optional[datetime] = None,
        limit: int = 500,
    ) -> list[OHLCVBar]:
        """Fetch historical OHLCV bars."""

    @abstractmethod
    async def get_ticker(self, symbol: str) -> dict:
        """Return latest ticker: {symbol, last, bid, ask, ts}."""

    # ── Lifecycle ──────────────────────────────────────────────────────────

    async def initialize(self) -> None:
        """Optional setup (e.g. set leverage, verify connection)."""

    async def shutdown(self) -> None:
        """Optional cleanup."""
