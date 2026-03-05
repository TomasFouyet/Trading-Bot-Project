"""
Trade model — a completed round-trip (entry + exit).
"""
from __future__ import annotations

from datetime import datetime
from decimal import Decimal

from sqlalchemy import DateTime, Index, Numeric, String
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, TimestampMixin


class Trade(Base, TimestampMixin):
    __tablename__ = "trades"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    symbol: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    side: Mapped[str] = mapped_column(String(10), nullable=False)  # LONG | SHORT
    mode: Mapped[str] = mapped_column(String(20), nullable=False, default="paper")
    strategy_id: Mapped[str | None] = mapped_column(String(64), nullable=True)

    entry_order_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    exit_order_id: Mapped[str | None] = mapped_column(String(64), nullable=True)

    entry_ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    exit_ts: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    entry_price: Mapped[Decimal] = mapped_column(Numeric(20, 8), nullable=False)
    exit_price: Mapped[Decimal | None] = mapped_column(Numeric(20, 8), nullable=True)
    qty: Mapped[Decimal] = mapped_column(Numeric(20, 8), nullable=False)

    pnl: Mapped[Decimal | None] = mapped_column(Numeric(20, 8), nullable=True)
    pnl_pct: Mapped[Decimal | None] = mapped_column(Numeric(10, 4), nullable=True)
    fees: Mapped[Decimal] = mapped_column(Numeric(20, 8), nullable=False, default=Decimal("0"))
    is_open: Mapped[bool] = mapped_column(default=True)

    __table_args__ = (
        Index("ix_trade_symbol_mode", "symbol", "mode"),
        Index("ix_trade_entry_ts", "entry_ts"),
    )

    def __repr__(self) -> str:
        return f"<Trade {self.id} {self.symbol} {self.side} pnl={self.pnl}>"
