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

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view


# ── Parameter grid ─────────────────────────────────────────────────────────────
# Edit these lists to control what values get tested.
# Total combinations = product of all list lengths.

PARAM_GRID = {
    "rsi_period":       [9, 14],
    "ema_period":       [12, 14],
    "swing_window":     [2, 3, 4, 5],
    "swing_separation": [5, 7, 10],
    "rsi_oversold":     [30.0, 35.0],
    "rsi_overbought":   [65.0, 70.0],
    "rr_ratio":         [1.5, 1.6, 1.75],
    "trend_ema_period": [50],
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
    "entry_cooldown_bars": 5,
}


# ── Worker globals ─────────────────────────────────────────────────────────────
# Set by initializer once per process; shared across all jobs in that process.
# key: (rsi_period, ema_period, trend_ema_period) → (precomputed_df, ts_ms_to_pos)
_WORKER_PRECOMPUTED: "dict | None" = None
# Shared raw-bars DataFrame (ts/ohlcv, no indicators) for fast bars_to_df replacement
_WORKER_BARS_DF: "object | None" = None
# int(ms since epoch) → row-position in _WORKER_BARS_DF  (same for all indicator sets)
_WORKER_TS_MS_TO_POS: "dict | None" = None


def _init_worker_cache(precomputed: dict, bars_df: object, ts_ms_to_pos: dict) -> None:
    """
    Called once per worker process.
    Injects precomputed indicator data AND monkey-patches BaseStrategy.bars_to_df
    so the engine's per-bar window creation no longer iterates over OHLCVBar objects
    and converts Decimal fields — instead it slices a pre-built numpy-backed DataFrame.
    That alone cuts ~6 s/combo of Decimal→float overhead (250 bars × 5 fields × 17 k calls).
    """
    global _WORKER_PRECOMPUTED, _WORKER_BARS_DF, _WORKER_TS_MS_TO_POS
    _WORKER_PRECOMPUTED  = precomputed
    _WORKER_BARS_DF      = bars_df
    _WORKER_TS_MS_TO_POS = ts_ms_to_pos

    # Patch BaseStrategy.bars_to_df to use the pre-built DataFrame slice
    from app.strategy.base import BaseStrategy
    _orig_bars_to_df = BaseStrategy.__dict__["bars_to_df"].__func__

    def _fast_bars_to_df(bars_list):
        if not bars_list or _WORKER_TS_MS_TO_POS is None:
            return _orig_bars_to_df(bars_list)
        ts_ms = int(bars_list[-1].ts.timestamp() * 1_000)
        pos   = _WORKER_TS_MS_TO_POS.get(ts_ms)
        if pos is None:
            return _orig_bars_to_df(bars_list)
        n = len(bars_list)
        return _WORKER_BARS_DF.iloc[pos - n + 1 : pos + 1]

    BaseStrategy.bars_to_df = staticmethod(_fast_bars_to_df)


# ── Fast pre-filter ────────────────────────────────────────────────────────────

def _fast_divergence_count(
    rsi: np.ndarray,
    lows: np.ndarray,
    highs: np.ndarray,
    swing_window: int,
    swing_separation: int,
    rsi_oversold: float,
    rsi_overbought: float,
    allow_short: bool = True,
) -> int:
    """
    Vectorized upper-bound count of RSI divergence signals on the full series.

    Finds all swing lows/highs with numpy sliding windows, then counts consecutive
    pairs that satisfy the divergence condition.  Because this scans the full series
    (vs the rolling 250-bar window used at runtime), the result is an *upper bound*:
    if this returns < N, the backtest will also produce < N trades.
    """
    n = len(rsi)
    w = swing_window
    if n < 2 * w + 2:
        return 0

    i_range = np.arange(w, n - w)

    # ── Bullish divergence (swing lows) ─────────────────────────────────────
    left_l  = sliding_window_view(lows[:-1], w)
    right_l = sliding_window_view(lows[1:],  w)
    center_l  = lows[i_range]
    left_min  = left_l[i_range - w].min(axis=1)
    right_min = right_l[i_range].min(axis=1)
    sl_idx = i_range[(center_l < left_min) & (center_l < right_min)]

    bullish = 0
    for j in range(1, len(sl_idx)):
        i1, i2 = sl_idx[j - 1], sl_idx[j]
        if (i2 - i1) >= swing_separation:
            r1, r2 = rsi[i1], rsi[i2]
            if (lows[i2] < lows[i1] and r2 > r1
                    and r1 <= rsi_oversold and r2 <= rsi_oversold):
                bullish += 1

    if not allow_short:
        return bullish

    # ── Bearish divergence (swing highs) ────────────────────────────────────
    left_h  = sliding_window_view(highs[:-1], w)
    right_h = sliding_window_view(highs[1:],  w)
    center_h  = highs[i_range]
    left_max  = left_h[i_range - w].max(axis=1)
    right_max = right_h[i_range].max(axis=1)
    sh_idx = i_range[(center_h > left_max) & (center_h > right_max)]

    bearish = 0
    for j in range(1, len(sh_idx)):
        i1, i2 = sh_idx[j - 1], sh_idx[j]
        if (i2 - i1) >= swing_separation:
            r1, r2 = rsi[i1], rsi[i2]
            if (highs[i2] > highs[i1] and r2 < r1
                    and r1 >= rsi_overbought and r2 >= rsi_overbought):
                bearish += 1

    return bullish + bearish


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

        # ── Indicator cache ───────────────────────────────────────────────────
        # The engine calls on_bar(window) where window grows then slides.
        # We precompute RSI/EMA once and slice by the window's last timestamp.
        #
        # Fast path: use data pre-computed in the main process and shared via
        # the worker initializer (avoids re-reading parquet + re-computing ewm).
        # Slow path (fallback): compute locally (used when workers=1 or cache miss).
        ind_key = (params["rsi_period"], params["ema_period"], params["trend_ema_period"])

        if _WORKER_PRECOMPUTED is not None and ind_key in _WORKER_PRECOMPUTED:
            _precomputed, _ts_ms_to_pos_ind = _WORKER_PRECOMPUTED[ind_key]
            _orig_compute = strategy._compute_indicators

            def _cached_indicators(df: "pd.DataFrame") -> "pd.DataFrame":
                # bars_to_df already returned a precomputed-df slice (has rsi/ema cols).
                # If indicator cols are present just return the slice as-is (identity).
                if "rsi" in df.columns:
                    return df
                # Fallback: look up by int(ms) timestamp key
                ts_ms = int(df["ts"].iloc[-1].timestamp() * 1_000)
                end_pos = _ts_ms_to_pos_ind.get(ts_ms)
                if end_pos is None:
                    return _orig_compute(df)
                end_pos += 1
                return _precomputed.iloc[end_pos - len(df) : end_pos]

        else:
            # Fallback: compute indicators locally
            all_bars     = store.read_bars(symbol, timeframe, start, end)
            full_df      = BaseStrategy.bars_to_df(all_bars)
            _precomputed = strategy._compute_indicators(full_df)
            _ts_ms_to_pos_ind = {
                int(pd.Timestamp(ts).timestamp() * 1_000): i
                for i, ts in enumerate(_precomputed["ts"])
            }
            _orig_compute = strategy._compute_indicators

            def _cached_indicators(df: "pd.DataFrame") -> "pd.DataFrame":
                if "rsi" in df.columns:
                    return df
                ts_ms = int(df["ts"].iloc[-1].timestamp() * 1_000)
                end_pos = _ts_ms_to_pos_ind.get(ts_ms)
                if end_pos is None:
                    return _orig_compute(df)
                end_pos += 1
                return _precomputed.iloc[end_pos - len(df) : end_pos]

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
    from app.strategy import get_strategy as _get_strategy
    from app.strategy.base import BaseStrategy as _BaseStrategy
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

    # ── Phase 1: Pre-compute indicators for all unique (rsi, ema, trend) combos ──
    # There are only rsi×ema×trend = 3×3×2 = 18 unique indicator sets out of
    # potentially thousands of total combos.  We compute them once here and inject
    # them into every worker via the pool initializer — each worker receives the
    # dict once at startup rather than recomputing per-job.
    print("  Pre-computing indicators for unique parameter combos...", flush=True)
    _t_precomp = time.monotonic()

    _all_bars_main = store.read_bars(args.symbol, args.timeframe, start, end)
    _full_df_main  = _BaseStrategy.bars_to_df(_all_bars_main)

    _ind_keys = {
        (p["rsi_period"], p["ema_period"], p["trend_ema_period"])
        for p in (dict(zip(keys, c)) for c in combos)
    }

    _precomputed_main: dict = {}
    for (rsi_p, ema_p, trend_p) in _ind_keys:
        _p = {**FIXED_PARAMS, "rsi_period": rsi_p, "ema_period": ema_p,
              "trend_ema_period": trend_p}
        _s = _get_strategy("rsi_divergence", symbol=args.symbol, params=_p)
        _df = _s._compute_indicators(_full_df_main)
        _precomputed_main[(rsi_p, ema_p, trend_p)] = (_df, None)  # pos dict built below

    # ── Build unified int(ms)→position lookup (same bars for all indicator sets) ──
    # int(ms) keys are faster to hash/lookup than pandas Timestamps.
    # datetime.timestamp() is a native C call (~50 ns) vs Timestamp hashing (~200 ns).
    _any_df = next(iter(_precomputed_main.values()))[0]
    _ts_ms_arr = (pd.DatetimeIndex(_any_df["ts"]).asi8 // 1_000_000).tolist()
    _ts_ms_to_pos_main = dict(zip(_ts_ms_arr, range(len(_ts_ms_arr))))

    # Update all precomputed entries to share the same lookup dict
    _precomputed_main = {k: (df, _ts_ms_to_pos_main) for k, (df, _) in _precomputed_main.items()}

    # Raw bars DataFrame for workers' bars_to_df monkey-patch (no indicator columns).
    # Workers use this to slice per-bar windows without iterating over OHLCVBar objects
    # (avoids ~1250 Decimal→float conversions per bar × 17k bars = huge savings).
    _bars_raw_df = _any_df[["ts", "open", "high", "low", "close", "volume"]].copy()

    print(f"  Pre-computed {len(_precomputed_main)} indicator sets in "
          f"{_fmt_dur(time.monotonic() - _t_precomp)}\n", flush=True)

    # ── Phase 2: Fast pre-filter ───────────────────────────────────────────────
    # Count RSI divergences vectorially on the full series for each combo.
    # This is an upper bound: if the count < MIN_TRADES, the backtest will
    # definitely produce fewer trades and would be discarded anyway.
    MIN_TRADES_THRESHOLD = 2  # skip only combos with < 2 possible divergences on full series
    _lows_arr  = _full_df_main["low"].values.astype(float)
    _highs_arr = _full_df_main["high"].values.astype(float)

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

    # ── Phase 2 (continued): apply fast pre-filter ────────────────────────
    _t_filter = time.monotonic()
    _n_before = len(jobs)
    _filtered_jobs = []
    for _job in jobs:
        _p = _job["params"]
        _key = (_p["rsi_period"], _p["ema_period"], _p["trend_ema_period"])
        _rsi_arr = _precomputed_main[_key][0]["rsi"].values.astype(float)
        _count = _fast_divergence_count(
            rsi=_rsi_arr,
            lows=_lows_arr,
            highs=_highs_arr,
            swing_window=_p["swing_window"],
            swing_separation=_p["swing_separation"],
            rsi_oversold=_p["rsi_oversold"],
            rsi_overbought=_p["rsi_overbought"],
            allow_short=_p.get("allow_short", True),
        )
        if _count >= MIN_TRADES_THRESHOLD:
            _filtered_jobs.append(_job)

    _n_removed = _n_before - len(_filtered_jobs)
    print(
        f"  Fast filter: removed {_n_removed}/{_n_before} combos "
        f"(< {MIN_TRADES_THRESHOLD} potential signals) in "
        f"{_fmt_dur(time.monotonic() - _t_filter)} "
        f"→ {len(_filtered_jobs)} remaining\n",
        flush=True,
    )
    jobs    = _filtered_jobs
    total   = len(jobs)
    workers = min(args.workers, total) if total > 0 else 1

    if total == 0:
        print("  All combinations filtered out (no combo can reach min trades). Done.")
        return
    # ──────────────────────────────────────────────────────────────────────

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

    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_worker_cache,
        initargs=(_precomputed_main, _bars_raw_df, _ts_ms_to_pos_main),
    ) as pool:
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