"""
Backtest adapter.

Runs purely on historical OHLCV data (no network calls).
Fills market orders on the NEXT bar's open price + slippage.
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
from app.config import get_settings
from app.core.exceptions import BacktestError, OrderNotFoundError
from app.core.logging import get_logger

logger = get_logger(__name__)
_settings = get_settings()


class BacktestAdapter(BrokerAdapter):
    """
    Backtest broker adapter.

    Usage:
        adapter = BacktestAdapter(bars=historical_bars)
        # Engine iterates bars and calls adapter.advance(bar)
        # Market orders placed on bar[i] are filled at bar[i+1].open + slippage
    """

    def __init__(
        self,
        bars: list[OHLCVBar],
        initial_balance: Decimal = Decimal("10000"),
        commission_rate: Decimal | None = None,
        slippage_rate: Decimal | None = None,
    ) -> None:
        self._bars = bars
        self._bar_index: int = 0
        self._current_bar: OHLCVBar | None = None
        self._next_bar: OHLCVBar | None = None

        self._initial_balance = initial_balance
        self._balance = initial_balance
        self._commission = commission_rate or _settings.commission_rate
        self._slippage = slippage_rate or _settings.slippage_rate

        self._positions: dict[str, Position] = {}
        self._orders: dict[str, OrderResult] = {}
        self._pending_orders: list[OrderResult] = []  # awaiting fill on next bar
        self._fills: list[FillResult] = []
        self._equity_curve: list[tuple[datetime, Decimal]] = []

    def advance(self, bar: OHLCVBar) -> list[FillResult]:
        """
        Called by the engine for each new bar.
        Fills any pending orders at this bar's open price.
        Returns list of fills generated.
        """
        self._current_bar = bar
        new_fills: list[FillResult] = []

        # Fill pending market orders at this bar's open + slippage
        still_pending = []
        for order in self._pending_orders:
            fill_price = self._apply_slippage(bar.open, order.side)
            fee = fill_price * order.qty * self._commission

            fill = FillResult(
                fill_id=str(uuid.uuid4()),
                order_id=order.order_id,
                symbol=order.symbol,
                side=order.side,
                price=fill_price,
                qty=order.qty,
                fee=fee,
                fee_currency="USDT",
                timestamp=bar.ts,
            )
            order.status = OrderStatus.FILLED
            order.price = fill_price
            self._fills.append(fill)
            self._apply_fill_to_portfolio(fill)
            new_fills.append(fill)
            logger.debug(
                "backtest_fill",
                symbol=order.symbol,
                side=order.side.value,
                qty=str(order.qty),
                price=str(fill_price),
                bar_ts=bar.ts.isoformat(),
            )

        self._pending_orders = still_pending

        # Update equity curve
        equity = self._mark_to_market()
        self._equity_curve.append((bar.ts, equity))

        return new_fills

    def _apply_slippage(self, price: Decimal, side: OrderSide) -> Decimal:
        if side == OrderSide.BUY:
            return price * (1 + self._slippage)
        return price * (1 - self._slippage)

    def _apply_fill_to_portfolio(self, fill: FillResult) -> None:
        notional = fill.price * fill.qty

        if fill.side == OrderSide.BUY:
            cost = notional + fill.fee
            if self._balance < cost:
                logger.warning("backtest_insufficient_funds", required=str(cost), available=str(self._balance))
            self._balance -= cost

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
                total_qty = pos.qty + fill.qty
                pos.avg_price = (pos.avg_price * pos.qty + fill.price * fill.qty) / total_qty
                pos.qty = total_qty

        else:  # SELL
            pos = self._positions.get(fill.symbol)
            if pos and pos.qty > 0:
                sell_qty = min(fill.qty, pos.qty)
                pnl = (fill.price - pos.avg_price) * sell_qty
                pos.qty -= sell_qty
                if pos.qty <= Decimal("1e-10"):
                    del self._positions[fill.symbol]
                self._balance += sell_qty * fill.price - fill.fee
            else:
                self._balance += notional - fill.fee

    def _mark_to_market(self) -> Decimal:
        """Total equity = cash + mark-to-market of open positions."""
        equity = self._balance
        if self._current_bar:
            for pos in self._positions.values():
                if pos.symbol == self._current_bar.symbol:
                    equity += pos.qty * self._current_bar.close
        return equity

    # ── BrokerAdapter interface ────────────────────────────────────────────

    async def place_order(self, request: OrderRequest) -> tuple[OrderResult, FillResult | None]:
        """Queue order for fill on next bar."""
        order_id = str(uuid.uuid4())
        order = OrderResult(
            order_id=order_id,
            broker_id=None,
            symbol=request.symbol,
            side=request.side,
            order_type=request.order_type,
            qty=request.qty,
            price=request.price,
            status=OrderStatus.PENDING,
            created_at=self._current_bar.ts if self._current_bar else datetime.now(timezone.utc),
        )
        self._orders[order_id] = order
        self._pending_orders.append(order)
        return order, None  # Fill happens on next bar

    async def get_order(self, order_id: str, symbol: str) -> OrderResult:
        order = self._orders.get(order_id)
        if not order:
            raise OrderNotFoundError(f"Backtest order {order_id} not found")
        return order

    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        order = self._orders.get(order_id)
        if order and order.status == OrderStatus.PENDING:
            order.status = OrderStatus.CANCELLED
            self._pending_orders = [o for o in self._pending_orders if o.order_id != order_id]
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
        """Return from in-memory bars (no network call)."""
        bars = self._bars
        if start:
            bars = [b for b in bars if b.ts >= start]
        if end:
            bars = [b for b in bars if b.ts <= end]
        return bars[:limit]

    async def get_ticker(self, symbol: str) -> dict:
        if self._current_bar is None:
            raise BacktestError("No current bar available")
        return {
            "symbol": symbol,
            "last": float(self._current_bar.close),
            "bid": float(self._current_bar.close),
            "ask": float(self._current_bar.close),
            "ts": self._current_bar.ts,
        }

    # ── Reporting ──────────────────────────────────────────────────────────

    @property
    def equity_curve(self) -> list[tuple[datetime, Decimal]]:
        return self._equity_curve

    @property
    def all_fills(self) -> list[FillResult]:
        return list(self._fills)

    def reset(self) -> None:
        self._balance = self._initial_balance
        self._positions.clear()
        self._orders.clear()
        self._pending_orders.clear()
        self._fills.clear()
        self._equity_curve.clear()
        self._bar_index = 0
        self._current_bar = None
