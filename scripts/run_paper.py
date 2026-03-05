"""
Run paper trading locally without Docker/Celery.

Usage:
    python scripts/run_paper.py \
        --symbol BTC-USDT \
        --timeframe 5m \
        --balance 10000

Requirements:
    - BINGX_API_KEY and BINGX_API_SECRET in .env (for live ticker prices)
    - DATABASE_URL pointing to a running PostgreSQL (or use SQLite for local dev)
"""
from __future__ import annotations

import argparse
import asyncio
import signal
import sys


async def main(args: argparse.Namespace) -> None:
    from app.config import get_settings
    from app.core.logging import configure_logging
    settings = get_settings()
    configure_logging(log_level="INFO", log_format="console")

    from app.broker.bingx_client import BingXClient
    from app.broker.paper_adapter import PaperAdapter
    from app.data.feed import LiveFeed
    from app.data.parquet_store import ParquetStore
    from app.db.session import AsyncSessionLocal, create_all_tables
    from app.engine.forwardtest import ForwardTestEngine
    from app.risk.manager import RiskManager
    from app.strategy.ema_cross import EMACrossStrategy
    from decimal import Decimal

    # Setup
    await create_all_tables()

    strategy_params = {
        "ema_fast": args.ema_fast,
        "ema_slow": args.ema_slow,
        "sma_trend": args.sma_trend,
    }

    client = BingXClient(
        api_key=settings.bingx_api_key,
        api_secret=settings.bingx_api_secret,
        base_url=settings.bingx_base_url,
        market_type=settings.bingx_market_type,
    )
    store = ParquetStore()
    adapter = PaperAdapter(client=client, initial_balance=Decimal(str(args.balance)))
    feed = LiveFeed(client=client, store=store, symbol=args.symbol, timeframe=args.timeframe)
    strategy = EMACrossStrategy(symbol=args.symbol, params=strategy_params)
    risk = RiskManager()

    print(f"\nStarting paper trading:")
    print(f"  Symbol:    {args.symbol}")
    print(f"  Timeframe: {args.timeframe}")
    print(f"  Strategy:  EMA{args.ema_fast}/EMA{args.ema_slow} + SMA{args.sma_trend}")
    print(f"  Balance:   ${args.balance:,.2f}")
    print(f"  Mode:      PAPER (no real money)\n")
    print("Press Ctrl+C to stop.\n")

    async with AsyncSessionLocal() as session:
        engine = ForwardTestEngine(
            strategy=strategy,
            adapter=adapter,
            feed=feed,
            risk=risk,
            session=session,
            mode="paper",
        )

        # Handle Ctrl+C
        def handle_sigint(sig, frame):
            print("\n\nStopping paper trading...")
            engine.stop()
            # Print summary
            fills = adapter.get_fills()
            print(f"\nSession summary:")
            print(f"  Fills executed: {len(fills)}")
            print(f"  Final balance:  ${float(adapter.cash_balance):,.2f}")
            sys.exit(0)

        signal.signal(signal.SIGINT, handle_sigint)
        await engine.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run paper trading bot")
    parser.add_argument("--symbol", default="BTC-USDT")
    parser.add_argument("--timeframe", default="5m")
    parser.add_argument("--balance", type=float, default=10000.0)
    parser.add_argument("--ema-fast", type=int, default=20, dest="ema_fast")
    parser.add_argument("--ema-slow", type=int, default=50, dest="ema_slow")
    parser.add_argument("--sma-trend", type=int, default=200, dest="sma_trend")
    args = parser.parse_args()
    asyncio.run(main(args))
