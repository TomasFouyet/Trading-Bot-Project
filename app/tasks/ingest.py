"""
Celery tasks for data ingestion.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from app.tasks.celery_app import celery_app
from app.core.logging import get_logger

logger = get_logger(__name__)


@celery_app.task(bind=True, name="tasks.ingest_historical", max_retries=3)
def ingest_historical_task(
    self,
    symbol: str,
    timeframe: str,
    start_iso: str,
    end_iso: str | None = None,
) -> dict:
    """
    Fetch and store historical OHLCV data for a symbol/timeframe range.
    """
    from app.broker.bingx_client import BingXClient
    from app.config import get_settings
    from app.data.ingestor import OHLCVIngestor
    from app.data.parquet_store import ParquetStore

    settings = get_settings()

    async def _run() -> int:
        client = BingXClient(
            api_key=settings.bingx_api_key,
            api_secret=settings.bingx_api_secret,
            base_url=settings.bingx_base_url,
            market_type=settings.bingx_market_type,
        )
        store = ParquetStore()
        ingestor = OHLCVIngestor(client, store)
        try:
            start = datetime.fromisoformat(start_iso).replace(tzinfo=timezone.utc)
            end = datetime.fromisoformat(end_iso).replace(tzinfo=timezone.utc) if end_iso else None
            n = await ingestor.ingest_historical(symbol, timeframe, start, end)
            return n
        finally:
            await client.close()

    try:
        n = asyncio.run(_run())
        logger.info("ingest_task_done", symbol=symbol, timeframe=timeframe, bars=n)
        return {"status": "ok", "bars_written": n, "symbol": symbol, "timeframe": timeframe}
    except Exception as exc:
        logger.error("ingest_task_failed", error=str(exc))
        raise self.retry(exc=exc, countdown=30)


@celery_app.task(bind=True, name="tasks.poll_latest", max_retries=5)
def poll_latest_task(self, symbol: str, timeframe: str) -> dict:
    """Fetch latest bars and update Parquet (scheduled via beat)."""
    from app.broker.bingx_client import BingXClient
    from app.config import get_settings
    from app.data.ingestor import OHLCVIngestor
    from app.data.parquet_store import ParquetStore

    settings = get_settings()

    async def _run() -> int:
        client = BingXClient(
            api_key=settings.bingx_api_key,
            api_secret=settings.bingx_api_secret,
            base_url=settings.bingx_base_url,
        )
        store = ParquetStore()
        ingestor = OHLCVIngestor(client, store)
        try:
            bars = await ingestor.poll_latest(symbol, timeframe)
            return len(bars)
        finally:
            await client.close()

    try:
        n = asyncio.run(_run())
        return {"status": "ok", "bars_updated": n}
    except Exception as exc:
        raise self.retry(exc=exc, countdown=10)
