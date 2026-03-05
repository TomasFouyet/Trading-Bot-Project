"""
Candle / OHLCV model.
Note: we primarily store OHLCV in Parquet for performance.
This table stores metadata + recent candles for API queries.
"""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import BigInteger, DateTime, Float, Index, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class Candle(Base):
    __tablename__ = "candles"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    timeframe: Mapped[str] = mapped_column(String(5), nullable=False)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)

    open: Mapped[float] = mapped_column(Float, nullable=False)
    high: Mapped[float] = mapped_column(Float, nullable=False)
    low: Mapped[float] = mapped_column(Float, nullable=False)
    close: Mapped[float] = mapped_column(Float, nullable=False)
    volume: Mapped[float] = mapped_column(Float, nullable=False)

    __table_args__ = (
        UniqueConstraint("symbol", "timeframe", "ts", name="uq_candle_symbol_tf_ts"),
        Index("ix_candle_symbol_tf_ts", "symbol", "timeframe", "ts"),
    )

    def __repr__(self) -> str:
        return f"<Candle {self.symbol} {self.timeframe} {self.ts} C={self.close}>"
