"""
Run a backtest locally without Celery.

Usage:
    python scripts/run_backtest.py \
        --symbol BTC-USDT \
        --timeframe 5m \
        --start 2024-01-01 \
        --end 2024-06-01 \
        --balance 10000

Requires data to already be ingested in Parquet.
To ingest data first, run scripts/ingest_data.py or POST /strategy/run_backtest.
"""
from __future__ import annotations

import argparse
import asyncio
import json
from datetime import datetime, timezone
from decimal import Decimal


async def main(args: argparse.Namespace) -> None:
    # Configure logging
    from app.config import get_settings
    from app.core.logging import configure_logging
    settings = get_settings()
    configure_logging(log_level="INFO", log_format="console")

    from app.data.parquet_store import ParquetStore
    from app.engine.backtest import BacktestEngine
    from app.strategy.ema_cross import EMACrossStrategy

    start = datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc)
    end = datetime.fromisoformat(args.end).replace(tzinfo=timezone.utc)

    strategy_params = {
        "ema_fast": args.ema_fast,
        "ema_slow": args.ema_slow,
        "sma_trend": args.sma_trend,
    }

    strategy = EMACrossStrategy(symbol=args.symbol, params=strategy_params)
    store = ParquetStore()

    min_ts, max_ts = store.get_date_range(args.symbol, args.timeframe)
    if min_ts is None:
        print(f"[ERROR] No data found for {args.symbol}/{args.timeframe}")
        print("       Run data ingestion first:")
        print(f"       python scripts/ingest_data.py --symbol {args.symbol} --timeframe {args.timeframe} --start {args.start} --end {args.end}")
        return

    print(f"\nData available: {min_ts} → {max_ts}")
    print(f"Running backtest: {args.symbol} {args.timeframe} | {start.date()} → {end.date()}")
    print(f"Strategy: EMA{args.ema_fast}/EMA{args.ema_slow} + SMA{args.sma_trend}")
    print(f"Initial balance: ${args.balance:,.2f}\n")

    engine = BacktestEngine(
        strategy=strategy,
        store=store,
        initial_balance=Decimal(str(args.balance)),
        commission_rate=Decimal(str(args.commission_bps)) / 10000,
        slippage_rate=Decimal(str(args.slippage_bps)) / 10000,
    )

    result = await engine.run(args.symbol, args.timeframe, start, end, strategy_params)
    report = result.to_report()

    print("=" * 60)
    print("BACKTEST RESULTS")
    print("=" * 60)
    m = report["metrics"]
    print(f"  Total bars:       {m['total_bars']}")
    print(f"  Total trades:     {m['total_trades']}")
    print(f"  Win rate:         {m['winrate']:.1f}%")
    print(f"  Total PnL:        ${m['total_pnl']:.2f} ({m['total_pnl_pct']:.2f}%)")
    print(f"  Max drawdown:     {m['max_drawdown_pct']:.2f}%")
    print(f"  Sharpe ratio:     {m['sharpe_ratio']:.3f}")
    print(f"  Avg trade PnL:    ${m['avg_trade_pnl']:.2f}")
    print(f"  Exposure:         {m['exposure_pct']:.1f}%")
    print("=" * 60)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"\nFull report saved to: {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a trading strategy backtest")
    parser.add_argument("--symbol", default="BTC-USDT")
    parser.add_argument("--timeframe", default="5m")
    parser.add_argument("--start", default="2024-01-01")
    parser.add_argument("--end", default="2024-06-01")
    parser.add_argument("--balance", type=float, default=10000.0)
    parser.add_argument("--ema-fast", type=int, default=20, dest="ema_fast")
    parser.add_argument("--ema-slow", type=int, default=50, dest="ema_slow")
    parser.add_argument("--sma-trend", type=int, default=200, dest="sma_trend")
    parser.add_argument("--commission-bps", type=float, default=7.5, dest="commission_bps")
    parser.add_argument("--slippage-bps", type=float, default=5.0, dest="slippage_bps")
    parser.add_argument("--output", default=None, help="Save full JSON report to file")
    args = parser.parse_args()
    asyncio.run(main(args))
