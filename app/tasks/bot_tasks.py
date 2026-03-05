"""
Celery tasks for bot lifecycle management.
"""
from __future__ import annotations

import asyncio

from app.tasks.celery_app import celery_app
from app.core.logging import get_logger

logger = get_logger(__name__)

# Global reference to running engine task (for stop)
_running_engine_task = None


@celery_app.task(bind=True, name="tasks.start_bot")
def start_bot_task(
    self,
    symbol: str,
    timeframe: str,
    strategy_name: str = "ema_cross",
    mode: str = "paper",
    strategy_params: dict | None = None,
) -> dict:
    """Start the trading bot in paper or real mode."""

    async def _run() -> None:
        from app.broker.bingx_client import BingXClient
        from app.broker.paper_adapter import PaperAdapter
        from app.broker.bingx_adapter import BingXAdapter
        from app.config import get_settings, BotMode
        from app.data.feed import LiveFeed
        from app.data.parquet_store import ParquetStore
        from app.db.session import AsyncSessionLocal
        from app.db.models import BotState
        from app.engine.forwardtest import ForwardTestEngine
        from app.engine.live import LiveEngine
        from app.risk.manager import RiskManager
        from app.strategy import get_strategy
        from sqlalchemy import update

        settings = get_settings()
        strategy = get_strategy(strategy_name, symbol, strategy_params or {})
        store = ParquetStore()
        risk = RiskManager()
        client = BingXClient(
            api_key=settings.bingx_api_key,
            api_secret=settings.bingx_api_secret,
            base_url=settings.bingx_base_url,
            market_type=settings.bingx_market_type,
        )

        if mode == "real":
            adapter = BingXAdapter(client=client)
        else:
            adapter = PaperAdapter(client=client)

        feed = LiveFeed(client=client, store=store, symbol=symbol, timeframe=timeframe)

        async with AsyncSessionLocal() as session:
            # Upsert bot state
            await session.execute(
                update(BotState)
                .where(BotState.id == 1)
                .values(
                    is_running=True,
                    mode=mode,
                    symbol=symbol,
                    timeframe=timeframe,
                    strategy_name=strategy_name,
                )
            )
            await session.commit()

            EngineClass = LiveEngine if mode == "real" else ForwardTestEngine
            engine = EngineClass(
                strategy=strategy,
                adapter=adapter,
                feed=feed,
                risk=risk,
                session=session,
                mode=mode,
            )
            await engine.run()

    logger.info("bot_task_start", symbol=symbol, timeframe=timeframe, mode=mode)
    asyncio.run(_run())
    return {"status": "stopped", "symbol": symbol, "mode": mode}


@celery_app.task(name="tasks.stop_bot")
def stop_bot_task() -> dict:
    """Signal the running bot to stop gracefully."""
    # In production, use Redis pub/sub or a shared flag to signal the engine
    # For MVP: revoke the running task
    logger.info("bot_stop_requested")
    return {"status": "stop_requested"}
