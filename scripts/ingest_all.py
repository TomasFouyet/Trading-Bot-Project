"""
Batch historical data ingestion from BingX.

Downloads 1d, 4h, and 15m OHLCV data for all major crypto pairs.
Note: SP500, NASDAQ100, EURUSD are not available on BingX (crypto exchange only).

Usage:
    python scripts/ingest_all.py
    python scripts/ingest_all.py --start 2020-01-01
    python scripts/ingest_all.py --symbols BTC-USDT ETH-USDT --timeframes 1d 4h
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import datetime, timezone

SYMBOLS = [
    "BTC-USDT",
    "ETH-USDT",
    "XRP-USDT",
    "SOL-USDT",
    "BNB-USDT",
    "DOGE-USDT",
    "ADA-USDT",
    "TRX-USDT",
    "XLM-USDT",
]

TIMEFRAMES = ["1d", "4h", "15m"]

# BingX perpetual swap data availability (approximate)
DEFAULT_START = "2020-01-01"


async def ingest_one(
    symbol: str,
    timeframe: str,
    start: datetime,
    end: datetime,
) -> tuple[str, str, int | str]:
    """Ingest a single symbol/timeframe pair. Returns (symbol, tf, bars_or_error)."""
    from app.config import get_settings
    from app.broker.bingx_client import BingXClient
    from app.data.ingestor import OHLCVIngestor
    from app.data.parquet_store import ParquetStore

    settings = get_settings()
    client = BingXClient(
        api_key=settings.bingx_api_key,
        api_secret=settings.bingx_api_secret,
        base_url=settings.bingx_base_url,
        market_type=settings.bingx_market_type,
    )
    store = ParquetStore()
    ingestor = OHLCVIngestor(client, store)

    try:
        n = await ingestor.ingest_historical(symbol, timeframe, start, end)
        return symbol, timeframe, n
    except Exception as e:
        return symbol, timeframe, f"ERROR: {e}"
    finally:
        await client.close()


async def main(args: argparse.Namespace) -> None:
    from app.core.logging import configure_logging
    configure_logging(log_level="WARNING", log_format="console")

    start = datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc)
    end = datetime.fromisoformat(args.end).replace(tzinfo=timezone.utc)
    symbols = args.symbols
    timeframes = args.timeframes

    total = len(symbols) * len(timeframes)
    print(f"\nBatch ingestion: {len(symbols)} symbols × {len(timeframes)} timeframes = {total} jobs")
    print(f"Date range: {start.date()} → {end.date()}\n")

    results: list[tuple[str, str, int | str]] = []
    done = 0

    for symbol in symbols:
        for tf in timeframes:
            print(f"  [{done+1}/{total}] {symbol:12s} {tf:4s} ...", end=" ", flush=True)
            sym, timeframe, result = await ingest_one(symbol, tf, start, end)
            done += 1
            if isinstance(result, int):
                print(f"{result:6d} bars")
            else:
                print(result)
            results.append((sym, timeframe, result))
            # Small delay to avoid rate limiting between requests
            await asyncio.sleep(0.5)

    # Summary
    print("\n" + "─" * 50)
    print("SUMMARY")
    print("─" * 50)
    ok = [(s, tf, n) for s, tf, n in results if isinstance(n, int)]
    err = [(s, tf, e) for s, tf, e in results if isinstance(e, str)]
    print(f"  Success: {len(ok)}/{total}")
    if err:
        print(f"  Errors:  {len(err)}")
        for s, tf, e in err:
            print(f"    {s} {tf}: {e}")
    total_bars = sum(n for _, _, n in ok)
    print(f"  Total bars downloaded: {total_bars:,}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch ingest historical OHLCV from BingX")
    parser.add_argument(
        "--symbols", nargs="+", default=SYMBOLS,
        help="Symbols to ingest (default: all major pairs)"
    )
    parser.add_argument(
        "--timeframes", nargs="+", default=TIMEFRAMES,
        help="Timeframes to ingest (default: 1d 4h 15m)"
    )
    parser.add_argument(
        "--start", default=DEFAULT_START,
        help="Start date YYYY-MM-DD (default: 2020-01-01)"
    )
    parser.add_argument(
        "--end", default=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        help="End date YYYY-MM-DD (default: today)"
    )
    args = parser.parse_args()
    asyncio.run(main(args))
