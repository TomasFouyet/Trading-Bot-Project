"""
DataFeed — abstraction over data sources.

Provides a unified async iterator of OHLCVBar objects.
Two implementations:
  - LiveFeed: polls BingX + updates Parquet store
  - ParquetFeed: reads from stored Parquet files (used in backtest)
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import AsyncIterator

from app.broker.base import OHLCVBar
from app.broker.bingx_client import BingXClient
from app.config import get_settings
from app.core.exceptions import DataDelayError
from app.core.logging import get_logger
from app.data.ingestor import TIMEFRAME_SECONDS
from app.data.parquet_store import ParquetStore

logger = get_logger(__name__)
_settings = get_settings()


class LiveFeed:
    """
    Polls BingX API on each closed bar interval.
    Yields the last closed bar once per timeframe period.
    """

    def __init__(
        self,
        client: BingXClient,
        store: ParquetStore,
        symbol: str,
        timeframe: str,
        max_data_delay_s: int | None = None,
    ) -> None:
        self._client = client
        self._store = store
        self._symbol = symbol
        self._timeframe = timeframe
        self._tf_secs = TIMEFRAME_SECONDS.get(timeframe, 300)
        self._max_delay = max_data_delay_s or _settings.risk_data_delay_threshold_s
        self._last_bar_ts: datetime | None = None
        self._running = False

    async def stream(self) -> AsyncIterator[OHLCVBar]:
        """
        Async generator that yields one bar per closed candle.
        Yields immediately with the last available bar, then polls.
        """
        self._running = True
        logger.info(
            "live_feed_start",
            symbol=self._symbol,
            timeframe=self._timeframe,
            poll_interval_s=self._tf_secs,
        )

        while self._running:
            now = datetime.now(timezone.utc)
            # Calculate seconds until next bar close
            seconds_since_epoch = now.timestamp()
            elapsed_in_period = seconds_since_epoch % self._tf_secs
            sleep_until_close = self._tf_secs - elapsed_in_period

            # First tick: yield current bar immediately
            if self._last_bar_ts is None:
                bar = await self._fetch_latest_closed_bar()
                if bar:
                    self._last_bar_ts = bar.ts
                    yield bar

            # Sleep until next bar close (+2s buffer for API lag)
            await asyncio.sleep(sleep_until_close + 2.0)

            bar = await self._fetch_latest_closed_bar()
            if bar is None:
                continue

            # Check data freshness
            age_s = (datetime.now(timezone.utc) - bar.ts).total_seconds()
            if age_s > self._max_delay:
                logger.warning(
                    "data_delay_warning",
                    symbol=self._symbol,
                    age_s=age_s,
                    threshold_s=self._max_delay,
                )
                raise DataDelayError(
                    f"Data delay {age_s:.0f}s exceeds threshold {self._max_delay}s"
                )

            # Only yield if it's a new bar
            if self._last_bar_ts is None or bar.ts > self._last_bar_ts:
                self._last_bar_ts = bar.ts
                self._store.write_bars(self._symbol, self._timeframe, [bar])
                yield bar

    async def _fetch_latest_closed_bar(self) -> OHLCVBar | None:
        from app.data.ingestor import OHLCVIngestor
        ingestor = OHLCVIngestor(self._client, self._store)
        bars = await ingestor.poll_latest(self._symbol, self._timeframe, lookback_bars=2)
        if not bars:
            return None
        # Return second-to-last (last closed candle, not the open current one)
        # If only one bar, it's the closed one
        return bars[-2] if len(bars) >= 2 else bars[-1]

    def stop(self) -> None:
        self._running = False


class ParquetFeed:
    """
    Reads stored bars from Parquet for backtesting.
    Yields bars in chronological order.
    """

    def __init__(
        self,
        store: ParquetStore,
        symbol: str,
        timeframe: str,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> None:
        self._store = store
        self._symbol = symbol
        self._timeframe = timeframe
        self._start = start
        self._end = end

    def load(self) -> list[OHLCVBar]:
        """Load all bars into memory (backtest mode)."""
        return self._store.read_bars(
            self._symbol, self._timeframe, self._start, self._end
        )
