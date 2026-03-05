"""
OHLCV data ingestor.

Responsibilities:
- Fetch historical OHLCV from BingX in paginated chunks.
- Store in Parquet.
- Optionally mirror recent bars to PostgreSQL (for API queries).
- Incremental updates: only fetch bars after the last stored timestamp.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from decimal import Decimal

from app.broker.bingx_client import BingXClient
from app.broker.base import OHLCVBar
from app.config import get_settings
from app.core.logging import get_logger
from app.data.parquet_store import ParquetStore

logger = get_logger(__name__)
_settings = get_settings()

# BingX max candles per request for swap
BINGX_MAX_LIMIT = 1000

TIMEFRAME_SECONDS = {
    "1m": 60,
    "3m": 180,
    "5m": 300,
    "15m": 900,
    "30m": 1800,
    "1h": 3600,
    "2h": 7200,
    "4h": 14400,
    "6h": 21600,
    "12h": 43200,
    "1d": 86400,
    "3d": 259200,
    "1w": 604800,
}


class OHLCVIngestor:
    """
    Fetches and stores OHLCV data.

    Usage:
        ingestor = OHLCVIngestor(client, store)
        bars = await ingestor.ingest_historical(
            symbol="BTC-USDT",
            timeframe="5m",
            start=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end=datetime(2024, 6, 1, tzinfo=timezone.utc),
        )
        # Live polling:
        await ingestor.poll_latest(symbol="BTC-USDT", timeframe="5m")
    """

    def __init__(self, client: BingXClient, store: ParquetStore) -> None:
        self._client = client
        self._store = store

    async def ingest_historical(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime | None = None,
        batch_size: int = BINGX_MAX_LIMIT,
    ) -> int:
        """
        Fetch all historical bars in batches and write to Parquet.
        Supports resuming from last stored bar.
        Returns total bars written.
        """
        end = end or datetime.now(timezone.utc)
        tf_secs = TIMEFRAME_SECONDS.get(timeframe, 300)

        # Resume from last available timestamp
        _, max_ts = self._store.get_date_range(symbol, timeframe)
        if max_ts is not None:
            if hasattr(max_ts, "tzinfo") and max_ts.tzinfo is None:
                max_ts = max_ts.replace(tzinfo=timezone.utc)
            resume_from = max_ts + timedelta(seconds=tf_secs)
            logger.info(
                "ingest_resume",
                symbol=symbol,
                timeframe=timeframe,
                from_ts=resume_from.isoformat(),
            )
            start = max(start, resume_from)

        if start >= end:
            logger.info("ingest_already_up_to_date", symbol=symbol, timeframe=timeframe)
            return 0

        total_written = 0
        cursor = start
        batch_td = timedelta(seconds=tf_secs * batch_size)

        logger.info(
            "ingest_start",
            symbol=symbol,
            timeframe=timeframe,
            start=start.isoformat(),
            end=end.isoformat(),
        )

        while cursor < end:
            batch_end = min(cursor + batch_td, end)
            try:
                raw_bars = await self._client.fetch_ohlcv(
                    symbol=symbol,
                    timeframe=timeframe,
                    start=cursor,
                    end=batch_end,
                    limit=batch_size,
                )
            except Exception as e:
                logger.error("ingest_fetch_error", error=str(e), cursor=cursor.isoformat())
                break

            if not raw_bars:
                break

            bars = [
                OHLCVBar(
                    symbol=symbol,
                    timeframe=timeframe,
                    ts=b["ts"],
                    open=Decimal(str(b["open"])),
                    high=Decimal(str(b["high"])),
                    low=Decimal(str(b["low"])),
                    close=Decimal(str(b["close"])),
                    volume=Decimal(str(b["volume"])),
                )
                for b in raw_bars
            ]

            written = self._store.write_bars(symbol, timeframe, bars)
            total_written += written

            last_ts = bars[-1].ts
            cursor = last_ts + timedelta(seconds=tf_secs)

            logger.info(
                "ingest_batch",
                symbol=symbol,
                timeframe=timeframe,
                bars=len(bars),
                last_ts=last_ts.isoformat(),
                total=total_written,
            )

            # Small sleep to respect rate limits
            await asyncio.sleep(0.2)

        logger.info(
            "ingest_done",
            symbol=symbol,
            timeframe=timeframe,
            total_written=total_written,
        )
        return total_written

    async def poll_latest(
        self,
        symbol: str,
        timeframe: str,
        lookback_bars: int = 5,
    ) -> list[OHLCVBar]:
        """
        Fetch the latest N bars and update Parquet store.
        Called periodically in live/paper mode.
        """
        tf_secs = TIMEFRAME_SECONDS.get(timeframe, 300)
        start = datetime.now(timezone.utc) - timedelta(seconds=tf_secs * lookback_bars)

        try:
            raw_bars = await self._client.fetch_ohlcv(
                symbol=symbol,
                timeframe=timeframe,
                start=start,
                limit=lookback_bars,
            )
        except Exception as e:
            logger.error("poll_latest_error", error=str(e), symbol=symbol, timeframe=timeframe)
            return []

        bars = [
            OHLCVBar(
                symbol=symbol,
                timeframe=timeframe,
                ts=b["ts"],
                open=Decimal(str(b["open"])),
                high=Decimal(str(b["high"])),
                low=Decimal(str(b["low"])),
                close=Decimal(str(b["close"])),
                volume=Decimal(str(b["volume"])),
            )
            for b in raw_bars
        ]
        self._store.write_bars(symbol, timeframe, bars)
        return bars
