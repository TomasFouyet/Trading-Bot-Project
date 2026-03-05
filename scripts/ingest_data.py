"""
Ingest historical OHLCV data from BingX into Parquet.

Usage:
    python scripts/ingest_data.py \
        --symbol BTC-USDT \
        --timeframe 5m \
        --start 2024-01-01 \
        --end 2024-06-01
"""
from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone


async def main(args: argparse.Namespace) -> None:
    from app.config import get_settings
    from app.core.logging import configure_logging
    from app.broker.bingx_client import BingXClient
    from app.data.ingestor import OHLCVIngestor
    from app.data.parquet_store import ParquetStore

    settings = get_settings()
    configure_logging(log_level="INFO", log_format="console")

    start = datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc)
    end = datetime.fromisoformat(args.end).replace(tzinfo=timezone.utc)

    print(f"Ingesting {args.symbol}/{args.timeframe} from {start.date()} to {end.date()}")

    client = BingXClient(
        api_key=settings.bingx_api_key,
        api_secret=settings.bingx_api_secret,
        base_url=settings.bingx_base_url,
        market_type=settings.bingx_market_type,
    )
    store = ParquetStore()
    ingestor = OHLCVIngestor(client, store)

    try:
        n = await ingestor.ingest_historical(args.symbol, args.timeframe, start, end)
        print(f"\nDone. Total bars written: {n}")

        min_ts, max_ts = store.get_date_range(args.symbol, args.timeframe)
        print(f"Data range: {min_ts} → {max_ts}")
    finally:
        await client.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest historical OHLCV from BingX")
    parser.add_argument("--symbol", default="BTC-USDT")
    parser.add_argument("--timeframe", default="5m")
    parser.add_argument("--start", default="2024-01-01")
    parser.add_argument("--end", default="2024-06-01")
    args = parser.parse_args()
    asyncio.run(main(args))
