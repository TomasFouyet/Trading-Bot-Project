"""
Live trading engine.

Identical to ForwardTestEngine but:
- Uses BingXAdapter (real orders, real money)
- Extra safety checks before going live
- Additional logging and alerting
"""
from __future__ import annotations

from decimal import Decimal

from sqlalchemy.ext.asyncio import AsyncSession

from app.broker.bingx_adapter import BingXAdapter
from app.config import get_settings
from app.core.logging import get_logger
from app.data.feed import LiveFeed
from app.data.parquet_store import ParquetStore
from app.engine.forwardtest import ForwardTestEngine
from app.risk.manager import RiskManager
from app.strategy.base import BaseStrategy

logger = get_logger(__name__)
_settings = get_settings()


class LiveEngine(ForwardTestEngine):
    """
    Real-money trading engine.
    Reuses all logic from ForwardTestEngine with mode='real'.
    Only instantiated when bot_mode == 'real'.
    """

    def __init__(
        self,
        strategy: BaseStrategy,
        adapter: BingXAdapter,
        feed: LiveFeed,
        risk: RiskManager,
        session: AsyncSession,
    ) -> None:
        super().__init__(
            strategy=strategy,
            adapter=adapter,
            feed=feed,
            risk=risk,
            session=session,
            mode="real",
        )

    async def run(self) -> None:
        logger.warning(
            "LIVE_ENGINE_STARTING",
            symbol=_settings.default_symbol,
            strategy=self._strategy.strategy_id,
            leverage=_settings.leverage,
            message="REAL MONEY — double-check config before proceeding",
        )
        assert _settings.leverage == 1, "Leverage must be 1 for safe live trading"
        assert _settings.risk_max_daily_drawdown_pct <= Decimal("5"), (
            "Max daily drawdown should be <= 5% for live trading"
        )
        await super().run()
