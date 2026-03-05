"""
Order model — represents an order submitted to the broker (real, paper, or backtest).
"""
from __future__ import annotations

from datetime import datetime
from decimal import Decimal

from sqlalchemy import DateTime, Index, Numeric, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin


class Order(Base, TimestampMixin):
    __tablename__ = "orders"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    broker_id: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    symbol: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    side: Mapped[str] = mapped_column(String(10), nullable=False)  # BUY | SELL
    order_type: Mapped[str] = mapped_column(String(20), nullable=False)  # MARKET | LIMIT
    qty: Mapped[Decimal] = mapped_column(Numeric(20, 8), nullable=False)
    price: Mapped[Decimal | None] = mapped_column(Numeric(20, 8), nullable=True)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="PENDING")
    mode: Mapped[str] = mapped_column(String(20), nullable=False, default="paper")  # paper|real|backtest
    strategy_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    trace_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    error_msg: Mapped[str | None] = mapped_column(String(500), nullable=True)

    # relationship
    fills: Mapped[list["Fill"]] = relationship("Fill", back_populates="order", lazy="selectin")

    __table_args__ = (
        Index("ix_order_symbol_status", "symbol", "status"),
        Index("ix_order_created_at", "created_at"),
    )

    def __repr__(self) -> str:
        return f"<Order {self.id} {self.symbol} {self.side} {self.qty} [{self.status}]>"
