"""
BacktestRun — stores results of a completed backtest.
"""
from __future__ import annotations

from datetime import datetime
from decimal import Decimal

from sqlalchemy import DateTime, Index, JSON, Numeric, String
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, TimestampMixin


class BacktestRun(Base, TimestampMixin):
    __tablename__ = "backtest_runs"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    symbol: Mapped[str] = mapped_column(String(20), nullable=False)
    timeframe: Mapped[str] = mapped_column(String(5), nullable=False)
    strategy_name: Mapped[str] = mapped_column(String(64), nullable=False)
    start_date: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    end_date: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    # Strategy & cost params stored as JSON
    params: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)

    # Key metrics
    total_trades: Mapped[int] = mapped_column(default=0)
    winning_trades: Mapped[int] = mapped_column(default=0)
    losing_trades: Mapped[int] = mapped_column(default=0)
    winrate: Mapped[Decimal | None] = mapped_column(Numeric(6, 4), nullable=True)
    total_pnl: Mapped[Decimal | None] = mapped_column(Numeric(20, 8), nullable=True)
    total_pnl_pct: Mapped[Decimal | None] = mapped_column(Numeric(10, 4), nullable=True)
    max_drawdown_pct: Mapped[Decimal | None] = mapped_column(Numeric(10, 4), nullable=True)
    sharpe_ratio: Mapped[Decimal | None] = mapped_column(Numeric(8, 4), nullable=True)
    avg_trade_pnl: Mapped[Decimal | None] = mapped_column(Numeric(20, 8), nullable=True)
    exposure_pct: Mapped[Decimal | None] = mapped_column(Numeric(6, 4), nullable=True)

    # Full report JSON + parquet path
    report: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    artifacts_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="pending")

    __table_args__ = (Index("ix_backtest_symbol_strategy", "symbol", "strategy_name"),)

    def __repr__(self) -> str:
        return f"<BacktestRun {self.id} {self.symbol} {self.strategy_name} wr={self.winrate}>"
