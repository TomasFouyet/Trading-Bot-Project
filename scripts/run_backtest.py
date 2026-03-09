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
    from app.strategy import get_strategy

    start = datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc)
    end = datetime.fromisoformat(args.end).replace(tzinfo=timezone.utc)

    # Build strategy params depending on chosen strategy
    if args.strategy == "ema_cross":
        strategy_params = {
            "ema_fast": args.ema_fast,
            "ema_slow": args.ema_slow,
            "sma_trend": args.sma_trend,
        }
    elif args.strategy in ("rsi_divergence", "hybrid_rsi_pivot"):
        strategy_params = {
            "rsi_period": args.rsi_period,
            "ema_period": args.ema_period,
            "swing_window": args.swing_window,
            "swing_separation": args.swing_separation,
            "swing_lookback": args.swing_lookback,
            "trigger_window": args.trigger_window,
            "rsi_oversold": args.rsi_oversold,
            "rsi_overbought": args.rsi_overbought,
            "allow_short": args.allow_short,
            "sl_buffer_pct": args.sl_buffer_pct,
            "rr_ratio": args.rr_ratio,
            # Hybrid-only params
            "atr_period": args.atr_period,
            "atr_sl_mult": args.atr_sl_mult,
            "pivot_atr_proximity": args.pivot_atr_proximity,
            "vol_avg_period": args.vol_avg_period,
            "vol_min_ratio": args.vol_min_ratio,
            "session_start_utc": args.session_start_utc,
            "session_end_utc": args.session_end_utc,
            "bars_per_day": args.bars_per_day,
        }
    else:
        strategy_params = {}

    strategy = get_strategy(args.strategy, symbol=args.symbol, params=strategy_params)
    store = ParquetStore()

    min_ts, max_ts = store.get_date_range(args.symbol, args.timeframe)
    if min_ts is None:
        print(f"[ERROR] No data found for {args.symbol}/{args.timeframe}")
        print("       Run data ingestion first:")
        print(f"       python scripts/ingest_data.py --symbol {args.symbol} --timeframe {args.timeframe} --start {args.start} --end {args.end}")
        return

    print(f"\nData available: {min_ts} → {max_ts}")
    print(f"Running backtest: {args.symbol} {args.timeframe} | {start.date()} → {end.date()}")
    print(f"Strategy: {strategy.strategy_id}")
    print(f"Initial balance: ${args.balance:,.2f}\n")

    engine = BacktestEngine(
        strategy=strategy,
        store=store,
        initial_balance=Decimal(str(args.balance)),
        commission_rate=Decimal(str(args.commission_bps)) / 10000,
        slippage_rate=Decimal(str(args.slippage_bps)) / 10000,
        verbose=bool(args.output),
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
        report["initial_balance"] = args.balance
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"\nFull report saved to: {args.output}")
        print(f"Viewer: open scripts/backtest_viewer.html in your browser and load the JSON")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a trading strategy backtest")
    parser.add_argument("--symbol", default="BTC-USDT")
    parser.add_argument("--timeframe", default="5m")
    parser.add_argument("--start", default="2025-11-20")
    parser.add_argument("--end", default="2026-03-06")
    parser.add_argument("--balance", type=float, default=10000.0)
    parser.add_argument("--strategy", default="ema_cross",
                        choices=["ema_cross", "rsi_divergence", "hybrid_rsi_pivot"],
                        help="Strategy to backtest")
    parser.add_argument("--commission-bps", type=float, default=7.5, dest="commission_bps")
    parser.add_argument("--slippage-bps", type=float, default=5.0, dest="slippage_bps")
    parser.add_argument("--output", default=None, help="Save full JSON report to file")

    # EMA Cross params
    parser.add_argument("--ema-fast", type=int, default=20, dest="ema_fast")
    parser.add_argument("--ema-slow", type=int, default=50, dest="ema_slow")
    parser.add_argument("--sma-trend", type=int, default=200, dest="sma_trend")

    # RSI Divergence params
    parser.add_argument("--rsi-period", type=int, default=9, dest="rsi_period")
    parser.add_argument("--ema-period", type=int, default=14, dest="ema_period")
    parser.add_argument("--swing-window", type=int, default=5, dest="swing_window")
    parser.add_argument("--swing-separation", type=int, default=10, dest="swing_separation")
    parser.add_argument("--swing-lookback", type=int, default=100, dest="swing_lookback")
    parser.add_argument("--trigger-window", type=int, default=10, dest="trigger_window")
    parser.add_argument("--rsi-oversold", type=float, default=30.0, dest="rsi_oversold")
    parser.add_argument("--rsi-overbought", type=float, default=70.0, dest="rsi_overbought")
    parser.add_argument("--allow-short", action="store_true", default=True, dest="allow_short")
    parser.add_argument("--sl-buffer-pct", type=float, default=0.003, dest="sl_buffer_pct")
    parser.add_argument("--rr-ratio", type=float, default=1.5, dest="rr_ratio")

    # Hybrid RSI Pivot params
    parser.add_argument("--atr-period", type=int, default=14, dest="atr_period")
    parser.add_argument("--atr-sl-mult", type=float, default=1.5, dest="atr_sl_mult")
    parser.add_argument("--pivot-atr-proximity", type=float, default=1.5, dest="pivot_atr_proximity")
    parser.add_argument("--vol-avg-period", type=int, default=20, dest="vol_avg_period")
    parser.add_argument("--vol-min-ratio", type=float, default=0.5, dest="vol_min_ratio")
    parser.add_argument("--session-start-utc", type=int, default=8, dest="session_start_utc")
    parser.add_argument("--session-end-utc", type=int, default=21, dest="session_end_utc")
    parser.add_argument("--bars-per-day", type=int, default=288, dest="bars_per_day",
                        help="Bars per calendar day: 288 for 5m, 96 for 15m, 24 for 1h")

    args = parser.parse_args()
    asyncio.run(main(args))
