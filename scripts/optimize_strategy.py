"""
Strategy parameter optimizer — grid search or random search.

Runs multiple backtests in parallel and ranks combinations by a composite score.

Usage (grid search — all combinations):
    python -m scripts.optimize_strategy --strategy rsi_divergence \\
        --symbol BTC-USDT --timeframe 5m \\
        --start 2024-01-01 --end 2024-06-01 \\
        --mode grid --output optimize_results.csv

Usage (random search — N random samples):
    python -m scripts.optimize_strategy --strategy rsi_divergence \\
        --symbol BTC-USDT --timeframe 5m \\
        --start 2024-01-01 --end 2024-06-01 \\
        --mode random --samples 200 --output optimize_results.csv

Scoring:
    score = sharpe_ratio * 0.5 + winrate_normalized * 0.3 + pnl_pct_normalized * 0.2
    Combinations with fewer than --min-trades are discarded.
    Combinations with max_drawdown > --max-dd-pct are penalized.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import itertools
import json
import os
import random
import sys
import time
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

# ── Parameter search spaces ──────────────────────────────────────────────────
# Edit these lists to control the search space for each strategy.

RSI_DIVERGENCE_SPACE: dict[str, list[Any]] = {
    "rsi_period":        [7, 9, 14],
    "ema_period":        [9, 14, 21],
    "swing_window":      [3, 5, 7],
    "swing_separation":  [5, 10, 15],
    "swing_lookback":    [80, 100],
    "trigger_window":    [5, 10],
    "rsi_oversold":      [25.0, 30.0, 35.0],
    "rsi_overbought":    [65.0, 70.0, 75.0],
    "allow_short":       [True],
    "sl_buffer_pct":     [0.002, 0.003, 0.005],
    "rr_ratio":          [1.5, 1.7, 2.0, 2.5],
    "tp2_ratio":         [2.0, 2.5, 3.0],
    "min_trend_coeff":   [0.4, 0.5, 0.6],
}

HYBRID_RSI_PIVOT_SPACE: dict[str, list[Any]] = {
    "rsi_period":          [7, 9, 14],
    "ema_period":          [9, 14],
    "swing_window":        [3, 5],
    "swing_separation":    [5, 10],
    "swing_lookback":      [80, 100],
    "trigger_window":      [5, 10],
    "rsi_oversold":        [25.0, 30.0],
    "rsi_overbought":      [65.0, 70.0],
    "allow_short":         [True],
    "sl_buffer_pct":       [0.002, 0.003],
    "rr_ratio":            [1.5, 1.7, 2.0],
    "tp2_ratio":           [2.0, 2.5],
    "min_trend_coeff":     [0.4, 0.5],
    "atr_period":          [14],
    "atr_sl_mult":         [1.2, 1.5, 2.0],
    "pivot_atr_proximity": [1.0, 1.5, 2.0],
    "vol_avg_period":      [20],
    "vol_min_ratio":       [0.3, 0.5, 0.8],
    "session_start_utc":   [8],
    "session_end_utc":     [21],
    "bars_per_day":        [288],
}

PARAM_SPACES = {
    "rsi_divergence":    RSI_DIVERGENCE_SPACE,
    "hybrid_rsi_pivot":  HYBRID_RSI_PIVOT_SPACE,
}


# ── Scoring ───────────────────────────────────────────────────────────────────

def score_result(m: dict, max_dd_threshold: float) -> float:
    """
    Composite score for ranking.
      - Sharpe  (50%) — risk-adjusted returns
      - Win rate (30%) — consistency
      - PnL %   (20%) — raw profitability
    Drawdown penalty: if max_dd > threshold, multiply score by 0.5.
    Returns -inf if the combination should be excluded.
    """
    sharpe   = float(m.get("sharpe_ratio", 0))
    winrate  = float(m.get("winrate", 0)) / 100.0      # normalize to [0, 1]
    pnl_pct  = float(m.get("total_pnl_pct", 0)) / 100.0
    max_dd   = float(m.get("max_drawdown_pct", 100))

    composite = sharpe * 0.5 + winrate * 0.3 + pnl_pct * 0.2

    if max_dd > max_dd_threshold:
        composite *= 0.5

    return composite


# ── Backtest runner ───────────────────────────────────────────────────────────

async def run_single(
    strategy_name: str,
    params: dict,
    symbol: str,
    timeframe: str,
    start: datetime,
    end: datetime,
    balance: float,
    commission_bps: float,
    slippage_bps: float,
    htf_timeframe: str,
    store,
    semaphore: asyncio.Semaphore,
) -> dict | None:
    """Run one backtest and return a flat result dict, or None on error."""
    async with semaphore:
        from app.strategy import get_strategy
        from app.engine.backtest import BacktestEngine

        try:
            strategy = get_strategy(strategy_name, symbol=symbol, params=params)
            engine = BacktestEngine(
                strategy=strategy,
                store=store,
                initial_balance=Decimal(str(balance)),
                commission_rate=Decimal(str(commission_bps)) / 10000,
                slippage_rate=Decimal(str(slippage_bps)) / 10000,
                verbose=False,
            )
            result = await engine.run(
                symbol, timeframe, start, end, params,
                htf_timeframe=htf_timeframe,
            )
            m = result.to_report()["metrics"]
            return {"params": params, "metrics": m}

        except Exception as exc:
            # Skip combos that error (e.g., not enough bars)
            return None


# ── Combination generators ────────────────────────────────────────────────────

def grid_combinations(space: dict[str, list]) -> list[dict]:
    keys = list(space.keys())
    values = list(space.values())
    combos = []
    for combo in itertools.product(*values):
        combos.append(dict(zip(keys, combo)))
    return combos


def random_combinations(space: dict[str, list], n: int, seed: int = 42) -> list[dict]:
    rng = random.Random(seed)
    keys = list(space.keys())
    combos = set()
    result = []
    attempts = 0
    max_attempts = n * 20
    while len(result) < n and attempts < max_attempts:
        attempts += 1
        combo = tuple(rng.choice(space[k]) for k in keys)
        if combo not in combos:
            combos.add(combo)
            result.append(dict(zip(keys, combo)))
    return result


# ── Output helpers ────────────────────────────────────────────────────────────

def print_top(results: list[dict], top_n: int = 20) -> None:
    print(f"\n{'='*80}")
    print(f"TOP {min(top_n, len(results))} COMBINATIONS")
    print(f"{'='*80}")
    header = f"{'#':>4}  {'Score':>7}  {'Trades':>6}  {'Win%':>6}  {'PnL%':>7}  {'Sharpe':>7}  {'MaxDD%':>7}  Params"
    print(header)
    print("-" * 100)
    for i, r in enumerate(results[:top_n], 1):
        m = r["metrics"]
        p = r["params"]
        score = r["score"]
        # Compact param string — skip defaults
        param_str = "  ".join(f"{k}={v}" for k, v in p.items())
        print(
            f"{i:>4}  {score:>7.3f}  {m['total_trades']:>6}  "
            f"{m['winrate']:>5.1f}%  {m['total_pnl_pct']:>+6.2f}%  "
            f"{m['sharpe_ratio']:>7.3f}  {m['max_drawdown_pct']:>6.2f}%  "
            f"{param_str}"
        )


def save_csv(results: list[dict], path: str) -> None:
    if not results:
        return
    param_keys = list(results[0]["params"].keys())
    metric_keys = list(results[0]["metrics"].keys())
    fieldnames = ["rank", "score"] + metric_keys + param_keys

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, r in enumerate(results, 1):
            row = {"rank": i, "score": round(r["score"], 4)}
            row.update(r["metrics"])
            row.update(r["params"])
            writer.writerow(row)
    print(f"\nFull results saved to: {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

async def main(args: argparse.Namespace) -> None:
    from app.config import get_settings
    from app.core.logging import configure_logging
    from app.data.parquet_store import ParquetStore

    # Suppress verbose engine logs during optimization
    configure_logging(log_level="WARNING", log_format="console")

    store = ParquetStore()
    min_ts, max_ts = store.get_date_range(args.symbol, args.timeframe)
    if min_ts is None:
        print(f"[ERROR] No data found for {args.symbol}/{args.timeframe}")
        print(f"       Run: python -m scripts.ingest_data --symbol {args.symbol} "
              f"--timeframe {args.timeframe} --start {args.start} --end {args.end}")
        return

    start = datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc)
    end   = datetime.fromisoformat(args.end).replace(tzinfo=timezone.utc)

    space = PARAM_SPACES.get(args.strategy)
    if space is None:
        print(f"[ERROR] No param space defined for strategy '{args.strategy}'")
        print(f"       Available: {list(PARAM_SPACES.keys())}")
        return

    # Build combination list
    if args.mode == "grid":
        combos = grid_combinations(space)
        print(f"Grid search: {len(combos):,} combinations")
    else:
        combos = random_combinations(space, args.samples, seed=args.seed)
        print(f"Random search: {len(combos):,} combinations (seed={args.seed})")

    print(f"Symbol: {args.symbol}  TF: {args.timeframe}  HTF: {args.htf_timeframe}")
    print(f"Period: {args.start} -> {args.end}  Balance: ${args.balance:,.0f}")
    print(f"Min trades required: {args.min_trades}  Max DD filter: {args.max_dd_pct}%")
    print(f"Concurrency: {args.workers} workers\n")

    semaphore = asyncio.Semaphore(args.workers)
    tasks = [
        run_single(
            strategy_name=args.strategy,
            params=combo,
            symbol=args.symbol,
            timeframe=args.timeframe,
            start=start,
            end=end,
            balance=args.balance,
            commission_bps=args.commission_bps,
            slippage_bps=args.slippage_bps,
            htf_timeframe=args.htf_timeframe,
            store=store,
            semaphore=semaphore,
        )
        for combo in combos
    ]

    t0 = time.monotonic()
    done = 0
    valid_results = []

    # Run all tasks, showing progress as they complete
    for coro in asyncio.as_completed(tasks):
        result = await coro
        done += 1

        if result is not None:
            m = result["metrics"]
            if m["total_trades"] >= args.min_trades:
                result["score"] = score_result(m, args.max_dd_pct)
                valid_results.append(result)

        # Progress indicator
        elapsed = time.monotonic() - t0
        rate = done / elapsed if elapsed > 0 else 0
        eta = (len(tasks) - done) / rate if rate > 0 else 0
        print(
            f"\r  {done}/{len(tasks)}  valid={len(valid_results)}"
            f"  {rate:.1f}/s  ETA {eta:.0f}s   ",
            end="",
            flush=True,
        )

    print()  # newline after progress

    elapsed = time.monotonic() - t0
    print(f"\nCompleted {len(tasks):,} backtests in {elapsed:.1f}s  "
          f"({len(tasks)/elapsed:.1f}/s)")
    print(f"Valid combinations (>={args.min_trades} trades): {len(valid_results)}")

    if not valid_results:
        print("[WARNING] No valid combinations found. Try lowering --min-trades or expanding the date range.")
        return

    # Sort by composite score descending
    valid_results.sort(key=lambda r: r["score"], reverse=True)

    # Print top results
    print_top(valid_results, top_n=args.top_n)

    # Best result summary
    best = valid_results[0]
    print(f"\nBest combination:")
    print(f"  Score:        {best['score']:.4f}")
    print(f"  Trades:       {best['metrics']['total_trades']}")
    print(f"  Win rate:     {best['metrics']['winrate']:.1f}%")
    print(f"  Total PnL:    {best['metrics']['total_pnl_pct']:+.2f}%")
    print(f"  Sharpe ratio: {best['metrics']['sharpe_ratio']:.3f}")
    print(f"  Max drawdown: {best['metrics']['max_drawdown_pct']:.2f}%")
    print(f"  Params:")
    for k, v in best["params"].items():
        print(f"    --{k.replace('_', '-')} {v}")

    # Save CSV
    if args.output:
        save_csv(valid_results, args.output)

    # Save best params as JSON for easy reuse
    best_json_path = args.output.replace(".csv", "_best.json") if args.output else "optimize_best.json"
    with open(best_json_path, "w", encoding="utf-8") as f:
        json.dump({
            "strategy": args.strategy,
            "symbol": args.symbol,
            "timeframe": args.timeframe,
            "period": f"{args.start} -> {args.end}",
            "score": best["score"],
            "metrics": best["metrics"],
            "params": best["params"],
        }, f, indent=2)
    print(f"Best params saved to: {best_json_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimize strategy parameters via backtesting")

    # Backtest config
    parser.add_argument("--strategy", default="rsi_divergence",
                        choices=list(PARAM_SPACES.keys()),
                        help="Strategy to optimize")
    parser.add_argument("--symbol",    default="BTC-USDT")
    parser.add_argument("--timeframe", default="5m")
    parser.add_argument("--start",     default="2024-01-01")
    parser.add_argument("--end",       default="2024-06-01")
    parser.add_argument("--balance",   type=float, default=10000.0)
    parser.add_argument("--htf-timeframe", default="1h", dest="htf_timeframe")
    parser.add_argument("--commission-bps", type=float, default=7.5, dest="commission_bps")
    parser.add_argument("--slippage-bps",   type=float, default=5.0, dest="slippage_bps")

    # Search config
    parser.add_argument("--mode",    default="random", choices=["grid", "random"],
                        help="'grid' tests all combinations; 'random' samples N random ones")
    parser.add_argument("--samples", type=int, default=300,
                        help="Number of random combinations to test (--mode random)")
    parser.add_argument("--seed",    type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--workers", type=int, default=8,
                        help="Max concurrent backtests (higher = faster but more RAM)")

    # Filtering
    parser.add_argument("--min-trades", type=int, default=10, dest="min_trades",
                        help="Discard combinations with fewer than N trades")
    parser.add_argument("--max-dd-pct", type=float, default=20.0, dest="max_dd_pct",
                        help="Drawdown threshold for score penalty (%%)")

    # Output
    parser.add_argument("--output",  default="optimize_results.csv",
                        help="CSV file to save all valid results")
    parser.add_argument("--top-n",   type=int, default=20, dest="top_n",
                        help="Number of top combinations to print")

    args = parser.parse_args()
    asyncio.run(main(args))
