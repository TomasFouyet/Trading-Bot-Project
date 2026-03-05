"""
BotState — singleton row representing the current state of the bot.
"""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import Boolean, DateTime, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class BotState(Base):
    __tablename__ = "bot_state"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, default=1)
    is_running: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    mode: Mapped[str] = mapped_column(String(20), nullable=False, default="paper")
    symbol: Mapped[str] = mapped_column(String(20), nullable=False, default="BTC-USDT")
    timeframe: Mapped[str] = mapped_column(String(5), nullable=False, default="5m")
    strategy_name: Mapped[str] = mapped_column(String(64), nullable=False, default="ema_cross")
    last_heartbeat: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    config_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    consecutive_api_errors: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    kill_switch_reason: Mapped[str | None] = mapped_column(String(500), nullable=True)

    def __repr__(self) -> str:
        return f"<BotState running={self.is_running} mode={self.mode} symbol={self.symbol}>"
