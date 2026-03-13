"""
Grid-search optimizer for RSI Divergence strategy — multicore edition.

Runs every parameter combination in parallel across CPU cores using
ProcessPoolExecutor, tracks live progress with ETA, and reports the best results.

Usage:
    # Full grid (all CPU cores):
    python scripts/optimize_rsi_divergence.py \
        --symbol BTC-USDT --timeframe 5m \
        --start 2025-11-20 --end 2026-03-06 \
        --balance 10000 --output optimize_results.csv

    # Quick test (small grid, 4 workers):
    python scripts/optimize_rsi_divergence.py --quick --workers 4

    # Sequential (no multiprocessing, easier to debug):
    python scripts/optimize_rsi_divergence.py --workers 1

    # Resume from previous run (skip already-computed combos):
    python scripts/optimize_rsi_divergence.py \
        --resume optimize_results/merged.csv \
        --output optimize_results.csv
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import itertools
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from decimal import Decimal


# ── Parameter grid ─────────────────────────────────────────────────────────────
# Edit these lists to control what values get tested.
# Total combinations = product of all list lengths.

PARAM_GRID = {
    "rsi_period":       [7, 9, 14],
    "ema_period":       [9, 12, 14],
    "swing_window":     [2, 3, 4, 5],
    "swing_separation": [5, 7, 10],
    "rsi_oversold":     [25.0, 30.0, 35.0],
    "rsi_overbought":   [65.0, 70.0, 75.0],
    "rr_ratio":         [1.5, 1.6, 1.75, 2.0],
    "trend_ema_period": [0, 50],
}

# Quick mode — smaller grid for a fast sanity check
PARAM_GRID_QUICK = {
    "rsi_period":       [9],
    "ema_period":       [14],
    "swing_window":     [3, 5],
    "swing_separation": [5, 10],
    "rsi_oversold":     [25.0, 30.0],
    "rsi_overbought":   [70.0, 75.0],
    "rr_ratio":         [1.5, 2.0],
    "trend_ema_period": [0, 50],
}

# Fixed parameters (not varied)
FIXED_PARAMS = {
    "swing_lookback":   100,
    "trigger_window":   10,
    "allow_short":      True,
    "sl_buffer_pct":    0.003,
    "tp2_ratio":        1.75,
    "trend_slope_bars": 5,
    "entry_window":     2,
}


# ── Worker (runs in a child process) ──────────────────────────────────────────
# Must be a top-level function to be picklable by multiprocessing.

def _worker(job: dict) -> dict | None:
    """
    Executed in a separate process for each parameter combination.
    Creates its own event loop and imports to avoid shared state issues.
    """
    return asyncio.run(_async_backtest(job))


async def _async_backtest(job: dict) -> dict | None:
    # All imports inside the worker so each process initialises cleanly
    from app.config import get_settings
    from app.core.logging import configure_logging
    from app.data.parquet_store import ParquetStore
    from app.engine.backtest import BacktestEngine
    from app.strategy import get_strategy
    from app.strategy.base import BaseStrategy

    configure_logging(log_level="ERROR", log_format="console")
    get_settings()

    symbol       = job["symbol"]
    timeframe    = job["timeframe"]
    start        = job["start"]
    end          = job["end"]
    balance      = Decimal(str(job["balance"]))
    comm_bps     = job["commission_bps"]
    slip_bps     = job["slippage_bps"]
    params       = job["params"]

    try:
        store    = ParquetStore()
        strategy = get_strategy("rsi_divergence", symbol=symbol, params=params)

        # ── Indicator cache: precompute RSI/EMA once on full series ──────────
        # The engine calls on_bar(window) where window grows then slides.
        # Normally _compute_indicators(window) recomputes EWM from scratch
        # on every call (~17k × 250 = 4.25M ops).
        # We precompute once, then slice by matching the window's last timestamp.
        all_bars     = store.read_bars(symbol, timeframe, start, end)
        full_df      = BaseStrategy.bars_to_df(all_bars)
        _precomputed = strategy._compute_indicators(full_df)
        # O(1) lookup: last timestamp of window → end position in precomputed
        _ts_to_pos   = {ts: i for i, ts in enumerate(_precomputed["ts"])}
        _orig_compute = strategy._compute_indicators  # keep for fallback

        def _cached_indicators(df: "pd.DataFrame") -> "pd.DataFrame":
            last_ts = df["ts"].iloc[-1]
            end_pos = _ts_to_pos.get(last_ts)
            if end_pos is None:
                return _orig_compute(df)   # fallback (should never happen)
            end_pos   += 1
            start_pos  = end_pos - len(df)
            # Return slice — no copy needed; strategy only reads (uses .values / .iloc)
            return _precomputed.iloc[start_pos:end_pos]

        strategy._compute_indicators = _cached_indicators
        # ─────────────────────────────────────────────────────────────────────

        engine   = BacktestEngine(
            strategy=strategy,
            store=store,
            initial_balance=balance,
            commission_rate=Decimal(str(comm_bps)) / 10000,
            slippage_rate=Decimal(str(slip_bps)) / 10000,
            verbose=False,
        )
        result = await engine.run(symbol, timeframe, start, end, params)
        m = result.to_report()["metrics"]
        return {
            **params,
            "total_trades": m["total_trades"],
            "winrate":       round(m["winrate"], 1),
            "total_pnl":     round(m["total_pnl"], 2),
            "total_pnl_pct": round(m["total_pnl_pct"], 2),
            "max_drawdown":  round(m["max_drawdown_pct"], 2),
            "sharpe":        round(m["sharpe_ratio"], 3),
            "avg_trade_pnl": round(m["avg_trade_pnl"], 2),
        }
    except Exception:
        return None  # Skip combos that fail (not enough warm-up bars, etc.)


# ── Progress helpers ───────────────────────────────────────────────────────────

# True when running in a real terminal; False in Docker / pipes
_IS_TTY = sys.stdout.isatty()

# In Docker mode, print every this many seconds (or every DOCKER_PRINT_EVERY combos)
_DOCKER_INTERVAL_SECS = 30
_DOCKER_PRINT_EVERY   = 10   # also print every N combos if interval hasn't passed


def _fmt_dur(secs: float) -> str:
    s = int(secs)
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s//60}m {s%60:02d}s"
    return f"{s//3600}h {(s%3600)//60:02d}m"


def _bar(done: int, total: int, width: int = 30) -> str:
    filled = int(width * done / total) if total > 0 else 0
    return "█" * filled + "░" * (width - filled)


def _progress_line(done: int, total: int, elapsed: float,
                   best_wr: float, best_pnl: float, workers: int,
                   best_params: dict | None = None) -> str:
    rate = done / elapsed if elapsed > 0 else 0
    eta  = (total - done) / rate if rate > 0 else 0
    pct  = done / total * 100 if total > 0 else 0
    params_str = ""
    if best_params:
        params_str = (
            f"  <- rsi={best_params.get('rsi_period')} ema={best_params.get('ema_period')}"
            f" sw={best_params.get('swing_window')} sep={best_params.get('swing_separation')}"
            f" rr={best_params.get('rr_ratio')} trend={best_params.get('trend_ema_period')}"
        )
    return (
        f"[{_bar(done, total)}] {done}/{total} ({pct:.1f}%)"
        f"  elapsed={_fmt_dur(elapsed)}  ETA={_fmt_dur(eta)}"
        f"  {rate:.1f} c/s  {workers} workers"
        f"  | best WR={best_wr:.1f}%  PnL=${best_pnl:+.2f}{params_str}"
    )


def _redraw(done: int, total: int, elapsed: float,
            best_wr: float, best_pnl: float, workers: int,
            best_params: dict | None = None,
            last_print_time: list | None = None) -> None:
    """
    TTY mode   → overwrite same line with \\r
    Docker mode → print a new line every 30s or every 10 combos
    """
    line = _progress_line(done, total, elapsed, best_wr, best_pnl, workers, best_params)

    if _IS_TTY:
        sys.stdout.write(f"\r  {line}   ")
        sys.stdout.flush()
    else:
        # Docker: only emit a line when enough time or combos have passed
        now = time.monotonic()
        t_last = last_print_time[0] if last_print_time else 0
        if (done == 0 or done == total
                or done % _DOCKER_PRINT_EVERY == 0
                or (now - t_last) >= _DOCKER_INTERVAL_SECS):
            print(f"  {line}", flush=True)
            if last_print_time is not None:
                last_print_time[0] = now


# ── Resume helpers ────────────────────────────────────────────────────────────

def _load_completed_combos(resume_path: str, keys: list[str]) -> set[tuple]:
    """
    Read an existing results CSV and return a set of param-tuples
    that have already been computed, so we can skip them.
    """
    done_set: set[tuple] = set()
    if not os.path.exists(resume_path):
        return done_set

    with open(resume_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                key = tuple(_coerce(row[k], k) for k in keys)
                done_set.add(key)
            except (KeyError, ValueError):
                continue  # skip malformed rows
    return done_set


def _coerce(value: str, key: str):
    """Convert a CSV string value back to its native Python type."""
    if key == "allow_short":
        return value.strip().lower() in ("true", "1", "yes")
    # Float params (have decimals in the grid)
    float_keys = {"rsi_oversold", "rsi_overbought", "rr_ratio", "sl_buffer_pct", "tp2_ratio"}
    if key in float_keys:
        return float(value)
    return int(float(value))  # int params (rsi_period, ema_period, etc.)


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    # Build parameter grid
    grid       = PARAM_GRID_QUICK if args.quick else PARAM_GRID
    keys       = list(grid.keys())
    all_combos = list(itertools.product(*grid.values()))
    total_all  = len(all_combos)

    # Shard: this instance handles every Nth combo starting at shard-1 (round-robin)
    shard  = max(1, min(args.shard, args.total_shards))
    combos = all_combos[shard - 1 :: args.total_shards]
    total  = len(combos)

    workers = min(args.workers, total) if total > 0 else 1

    # ── Header ────────────────────────────────────────────────────────────────
    shard_label = f"shard {shard}/{args.total_shards}  " if args.total_shards > 1 else ""
    print(f"\n{'='*65}")
    print(f"  RSI Divergence Optimizer  —  {shard_label}{workers} workers")
    print(f"{'='*65}")
    print(f"  Symbol:    {args.symbol} {args.timeframe}")
    print(f"  Period:    {args.start} → {args.end}")
    print(f"  Balance:   ${args.balance:,.2f}")
    if args.total_shards > 1:
        print(f"  Grid:      {total} combinations (shard {shard}/{args.total_shards} of {total_all} total)")
    else:
        print(f"  Grid:      {total} combinations")
    for k, v in grid.items():
        print(f"    {k:<22} {v}")
    print(f"{'='*65}\n")

    # Validate data availability before spinning up workers
    from app.config import get_settings
    from app.core.logging import configure_logging
    from app.data.parquet_store import ParquetStore
    configure_logging(log_level="ERROR", log_format="console")
    get_settings()
    store = ParquetStore()
    min_ts, max_ts = store.get_date_range(args.symbol, args.timeframe)
    if min_ts is None:
        print(f"[ERROR] No data for {args.symbol}/{args.timeframe}. Run ingest_data.py first.")
        return
    print(f"  Data range: {min_ts} → {max_ts}\n")

    start = datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc)
    end   = datetime.fromisoformat(args.end).replace(tzinfo=timezone.utc)

    # Build job list
    jobs = [
        {
            "symbol": args.symbol, "timeframe": args.timeframe,
            "start": start, "end": end,
            "balance": args.balance,
            "commission_bps": args.commission_bps,
            "slippage_bps":   args.slippage_bps,
            "params": {**FIXED_PARAMS, **dict(zip(keys, combo))},
        }
        for combo in combos
    ]

    # ── Resume: filter out already-computed combos ─────────────────────────
    if args.resume and os.path.exists(args.resume):
        done_set = _load_completed_combos(args.resume, keys)
        before   = len(jobs)
        jobs     = [
            j for j in jobs
            if tuple(j["params"][k] for k in keys) not in done_set
        ]
        skipped  = before - len(jobs)
        total    = len(jobs)
        workers  = min(args.workers, total) if total > 0 else 1
        print(f"  Resume: loaded {len(done_set)} completed combos from {args.resume}")
        print(f"  Skipping {skipped}, running {total} remaining\n")

        if total == 0:
            print("  All combinations already computed. Nothing to do!")
            return
    # ───────────────────────────────────────────────────────────────────────

    # ── Run in parallel ────────────────────────────────────────────────────────
    results:     list[dict] = []
    best_wr      = 0.0
    best_pnl     = -1e9
    best_wr_params:  dict | None = None
    best_pnl_params: dict | None = None
    done         = 0
    t0           = time.monotonic()
    last_print   = [t0]   # mutable for _redraw to update

    # ── Incremental CSV writer ────────────────────────────────────────────────
    # Results are written to disk after each completed combo so you can run
    # `python scripts/merge_optimize_results.py` at any time during execution.
    _csv_file      = open(args.output, "w", newline="") if args.output else None
    _csv_writer    = None   # initialized on first row (to know fieldnames)

    def _write_row(row: dict) -> None:
        nonlocal _csv_writer
        if _csv_file is None:
            return
        if _csv_writer is None:
            _csv_writer = csv.DictWriter(_csv_file, fieldnames=list(row.keys()))
            _csv_writer.writeheader()
        _csv_writer.writerow(row)
        _csv_file.flush()   # ensure data reaches disk immediately
    # ─────────────────────────────────────────────────────────────────────────

    print(f"  Starting {total} backtests with {workers} workers...", flush=True)
    _redraw(0, total, 0.001, best_wr, best_pnl, workers, None, last_print)

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_worker, job): job for job in jobs}

        for future in as_completed(futures):
            done   += 1
            elapsed = time.monotonic() - t0

            row = future.result()
            if row:
                results.append(row)
                _write_row(row)   # ← write immediately to disk
                if row["winrate"] > best_wr:
                    best_wr        = row["winrate"]
                    best_wr_params = {k: row[k] for k in keys}
                if row["total_pnl"] > best_pnl:
                    best_pnl        = row["total_pnl"]
                    best_pnl_params = {k: row[k] for k in keys}

            _redraw(done, total, elapsed, best_wr, best_pnl, workers,
                    best_wr_params, last_print)

    if _csv_file:
        _csv_file.close()

    elapsed_total = time.monotonic() - t0
    end_msg = (
        f"\nCompleted {total} combinations in {_fmt_dur(elapsed_total)} "
        f"({total / elapsed_total:.1f} combos/s)"
    )
    if _IS_TTY:
        print(f"\n{end_msg}")
    else:
        print(end_msg, flush=True)

    if not results:
        print("No valid results. Check data range or strategy parameters.")
        return

    # ── Rank & display ─────────────────────────────────────────────────────────
    def _score(r: dict) -> float:
        # Weighted score: WR 60% + PnL% 40%  (min_trades filter: must have >= 5 trades)
        if r["total_trades"] < 5:
            return -999.0
        return r["winrate"] * 0.6 + r["total_pnl_pct"] * 0.4

    ranked = sorted(results, key=_score, reverse=True)
    top_n  = min(20, len(ranked))

    HDR = (
        f"{'RSI':>4} {'EMA':>4} {'SW':>3} {'SEP':>4} "
        f"{'OVS':>6} {'OVB':>6} {'RR':>5} {'TREND':>6} "
        f"{'#TR':>4} {'WR%':>6} {'PnL$':>9} {'MaxDD':>7} {'Sharpe':>7}"
    )

    def _row(r: dict) -> str:
        return (
            f"{r['rsi_period']:>4} {r['ema_period']:>4} {r['swing_window']:>3} "
            f"{r['swing_separation']:>4} "
            f"{r['rsi_oversold']:>6.0f} {r['rsi_overbought']:>6.0f} "
            f"{r['rr_ratio']:>5.2f} {r['trend_ema_period']:>6} "
            f"{r['total_trades']:>4} {r['winrate']:>5.1f}% "
            f"{r['total_pnl']:>+9.2f} {r['max_drawdown']:>6.2f}% {r['sharpe']:>7.3f}"
        )

    sep = "─" * 87
    print(f"TOP {top_n} by Score (WR×0.6 + PnL%×0.4, min 5 trades)")
    print(sep)
    print(HDR)
    print(sep)
    for r in ranked[:top_n]:
        print(_row(r))
    print(sep)

    valid = [r for r in results if r["total_trades"] >= 5]
    if valid:
        best_wr_row  = max(valid, key=lambda r: (r["winrate"],  r["total_pnl"]))
        best_pnl_row = max(valid, key=lambda r: (r["total_pnl"], r["winrate"]))

        print(f"\n★  Best Win Rate:  WR={best_wr_row['winrate']:.1f}%  PnL=${best_wr_row['total_pnl']:+.2f}")
        print(f"   " + "  ".join(f"{k}={best_wr_row[k]}" for k in keys))
        print(f"\n★  Best PnL:       WR={best_pnl_row['winrate']:.1f}%  PnL=${best_pnl_row['total_pnl']:+.2f}")
        print(f"   " + "  ".join(f"{k}={best_pnl_row[k]}" for k in keys))

    if args.output:
        print(f"\nAll {len(results)} results saved to: {args.output} (written incrementally)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimize RSI Divergence parameters (multicore)")
    parser.add_argument("--symbol",    default="BTC-USDT")
    parser.add_argument("--timeframe", default="5m")
    parser.add_argument("--start",     default="2025-11-20")
    parser.add_argument("--end",       default="2026-03-06")
    parser.add_argument("--balance",   type=float, default=10000.0)
    parser.add_argument("--commission-bps", type=float, default=7.5,  dest="commission_bps")
    parser.add_argument("--slippage-bps",   type=float, default=5.0,  dest="slippage_bps")
    parser.add_argument("--output",    default="optimize_results.csv")
    parser.add_argument("--workers",      type=int, default=os.cpu_count(),
                        help=f"Parallel workers (default: all CPU cores = {os.cpu_count()})")
    parser.add_argument("--quick",        action="store_true",
                        help="Use small grid (~32 combos) for a fast test")
    # Sharding: each Docker container handles 1/total_shards of the grid
    parser.add_argument("--shard",        type=int, default=1,
                        help="Which shard to run (1-indexed, e.g. 1, 2, 3...)")
    parser.add_argument("--total-shards", type=int, default=1, dest="total_shards",
                        help="Total number of shards (Docker containers)")
    # Resume from previous run
    parser.add_argument("--resume",       default=None,
                        help="Path to existing CSV (e.g. merged.csv) — skip already-computed combos")
    args = parser.parse_args()
    main(args)