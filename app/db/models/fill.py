"""
Fill model — represents actual execution of an order.
"""
from __future__ import annotations

from datetime import datetime
from decimal import Decimal

from sqlalchemy import DateTime, ForeignKey, Index, Numeric, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


class Fill(Base):
    __tablename__ = "fills"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    order_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("orders.id", ondelete="CASCADE"), nullable=False, index=True
    )
    symbol: Mapped[str] = mapped_column(String(20), nullable=False)
    side: Mapped[str] = mapped_column(String(10), nullable=False)
    price: Mapped[Decimal] = mapped_column(Numeric(20, 8), nullable=False)
    qty: Mapped[Decimal] = mapped_column(Numeric(20, 8), nullable=False)
    fee: Mapped[Decimal] = mapped_column(Numeric(20, 8), nullable=False, default=Decimal("0"))
    fee_currency: Mapped[str] = mapped_column(String(10), nullable=False, default="USDT")
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    mode: Mapped[str] = mapped_column(String(20), nullable=False, default="paper")

    order: Mapped["Order"] = relationship("Order", back_populates="fills")

    __table_args__ = (Index("ix_fill_ts", "ts"),)

    def __repr__(self) -> str:
        return f"<Fill {self.id} {self.symbol} {self.side} {self.qty}@{self.price}>"
