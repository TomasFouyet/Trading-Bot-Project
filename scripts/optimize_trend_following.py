"""
═══════════════════════════════════════════════════════════════════════
  OPTIMIZER: Trend Following Strategy — 4-Layer Adaptive Parameters
═══════════════════════════════════════════════════════════════════════

Usage:
    python scripts/optimize_trend_following.py --quick          # 64 combos, ~5 min
    python scripts/optimize_trend_following.py                  # ~972 combos
    python scripts/optimize_trend_following.py --resume         # continue interrupted
    python scripts/optimize_trend_following.py --workers 8      # more parallelism

Output:
    optimize_results/trend_following/results.csv
    optimize_results/trend_following/best_params.json
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import itertools
import json
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path


# ── Parameter grids ───────────────────────────────────────────────────────

PARAM_GRID_FULL = {
    # Core entry filters
    "adx_min":                [20.0, 25.0, 30.0],
    "pullback_tolerance_atr": [1.0, 1.5, 2.0],
    # Layer 1: Confidence
    "use_confidence":         [True, False],
    "min_confidence":         [0.0, 0.40],
    "adx_strong":             [35.0],
    # Layer 2: Session
    "use_session_filter":     [True, False],
    "session_mult_eu":        [0.75],
    "session_mult_other":     [0.25, 0.50],
    # Layer 3: Streak
    "use_streak_adj":         [True, False],
    "streak_euphoria_mult":   [0.75],
    # Layer 4: Patience
    "use_patience":           [True, False],
    "soft_sl_bars":           [24, 48],
    # NOTE: tp1_close_pct is hardcoded per confidence tier in the strategy,
    # not read from params — do NOT add it here or it wastes 3x runs.
}

PARAM_GRID_QUICK = {
    "adx_min":                [25.0],
    "pullback_tolerance_atr": [1.5],
    "use_confidence":         [True, False],
    "min_confidence":         [0.0, 0.40],
    "adx_strong":             [35.0],
    "use_session_filter":     [True, False],
    "session_mult_eu":        [0.75],
    "session_mult_other":     [0.50],
    "use_streak_adj":         [True, False],
    "streak_euphoria_mult":   [0.75],
    "use_patience":           [True, False],
    "soft_sl_bars":           [48],
}

FIXED_PARAMS = {
    "ema_fast": 20,
    "ema_slow": 50,
    "slope_bars": 5,
    "min_rr": 1.5,
    "allow_short": True,
    "tight_pb_atr": 0.5,
    "us_session_start": 14,
    "us_session_end": 21,
    "eu_session_start": 8,
    "eu_session_end": 14,
    "session_mult_us": 1.0,
    "streak_euphoria_after": 2,
}

OUTPUT_DIR = Path("optimize_results/trend_following")

# Metric keys — used to separate params from results in CSV rows
METRIC_KEYS = frozenset({
    "status", "score", "total_trades", "winning_trades", "losing_trades",
    "winrate", "total_pnl", "total_pnl_pct", "max_drawdown_pct",
    "sharpe_ratio", "avg_trade_pnl", "exposure_pct",
})


# ══════════════════════════════════════════════════════════════════════════
# COMBINATION GENERATION
# ══════════════════════════════════════════════════════════════════════════

def generate_combinations(grid: dict) -> list[dict]:
    """Generate valid parameter combinations, skipping irrelevant ones."""
    keys = list(grid.keys())
    values = list(grid.values())
    combos = []
    for vals in itertools.product(*values):
        combo = dict(zip(keys, vals))

        # If confidence is OFF, only keep min_confidence=0 (the value is ignored anyway)
        if not combo["use_confidence"] and combo["min_confidence"] > 0:
            continue

        # If session is OFF, only keep one session_mult_other value (it's ignored)
        if not combo["use_session_filter"] and combo["session_mult_other"] != 0.50:
            continue

        # If patience is OFF, only keep one soft_sl_bars value (it's ignored)
        if not combo["use_patience"] and combo.get("soft_sl_bars", 48) != 48:
            continue

        combos.append(combo)
    return combos


# ══════════════════════════════════════════════════════════════════════════
# SINGLE BACKTEST RUNNER (called in subprocess)
# ══════════════════════════════════════════════════════════════════════════

def _run_one_backtest(job: dict) -> dict:
    """
    Module-level function for ProcessPoolExecutor.
    Each worker creates its own event loop.
    Returns a dict with params + metrics + status.
    """
    params = job["params"]

    try:
        # Imports inside the worker so each subprocess loads its own copy
        from app.core.logging import configure_logging
        configure_logging(log_level="ERROR", log_format="console")

        from app.data.parquet_store import ParquetStore
        from app.engine.backtest import BacktestEngine
        from app.strategy import get_strategy

        start = datetime.fromisoformat(job["start"]).replace(tzinfo=timezone.utc)
        end = datetime.fromisoformat(job["end"]).replace(tzinfo=timezone.utc)

        full_params = {**FIXED_PARAMS, **params}
        strategy = get_strategy("trend_following", symbol=job["symbol"], params=full_params)
        store = ParquetStore()

        engine = BacktestEngine(
            strategy=strategy,
            store=store,
            initial_balance=Decimal(str(job["balance"])),
            commission_rate=Decimal(str(job["commission_bps"])) / 10000,
            slippage_rate=Decimal("0.0005"),
            verbose=False,
        )

        # Run the async backtest in a fresh event loop
        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(
                engine.run(
                    job["symbol"], job["timeframe"], start, end,
                    full_params, htf_timeframe=job["htf_timeframe"],
                )
            )
        finally:
            loop.close()

        report = result.to_report()
        m = report["metrics"]

        return {
            **params,
            "total_trades":     m["total_trades"],
            "winning_trades":   m["winning_trades"],
            "losing_trades":    m["losing_trades"],
            "winrate":          m["winrate"],
            "total_pnl":        round(m["total_pnl"], 4),
            "total_pnl_pct":    round(m["total_pnl_pct"], 4),
            "max_drawdown_pct": round(m["max_drawdown_pct"], 4),
            "sharpe_ratio":     round(m["sharpe_ratio"], 4),
            "avg_trade_pnl":    round(m["avg_trade_pnl"], 4),
            "exposure_pct":     round(m["exposure_pct"], 2),
            "score":            _compute_score(m),
            "status":           "ok",
        }

    except Exception as e:
        return {
            **params,
            "total_trades": 0, "winning_trades": 0, "losing_trades": 0,
            "winrate": 0, "total_pnl": 0, "total_pnl_pct": 0,
            "max_drawdown_pct": 0, "sharpe_ratio": 0, "avg_trade_pnl": 0,
            "exposure_pct": 0, "score": -999,
            "status": f"error: {type(e).__name__}: {e}",
        }


def _compute_score(m: dict) -> float:
    """
    Composite score balancing profitability, consistency, and risk.
      PnL %           × 0.30
      Sharpe (capped)  × 0.25
      Win rate > 40%   × 0.20
      DD penalty > 2%  × 0.15
      Trade count < 30 × 0.10
    """
    pnl_pct  = m.get("total_pnl_pct", 0)
    sharpe   = m.get("sharpe_ratio", 0)
    wr       = m.get("winrate", 0)
    dd       = m.get("max_drawdown_pct", 0)
    n_trades = m.get("total_trades", 0)

    s  = pnl_pct * 0.30
    s += min(sharpe, 3.0) * 10 * 0.25
    s += max(0, (wr - 40)) * 0.5 * 0.20
    s -= max(0, dd - 2.0) * 2.0 * 0.15
    s -= max(0, 30 - n_trades) * 0.3 * 0.10
    return round(s, 4)


# ══════════════════════════════════════════════════════════════════════════
# CSV HELPERS
# ══════════════════════════════════════════════════════════════════════════

def _get_fieldnames(sample_row: dict) -> list[str]:
    param_keys = sorted(k for k in sample_row if k not in METRIC_KEYS)
    metric_list = [
        "total_trades", "winning_trades", "losing_trades", "winrate",
        "total_pnl", "total_pnl_pct", "max_drawdown_pct", "sharpe_ratio",
        "avg_trade_pnl", "exposure_pct", "score", "status",
    ]
    return param_keys + metric_list


def _combo_key(combo: dict) -> str:
    keys = sorted(k for k in combo if k not in METRIC_KEYS)
    return "|".join(f"{k}={combo[k]}" for k in keys)


def _combo_summary(combo: dict) -> str:
    parts = []
    parts.append(f"adx≥{combo.get('adx_min', '?')}")
    parts.append(f"pb={combo.get('pullback_tolerance_atr', '?')}")
    if combo.get("use_confidence"):
        parts.append(f"conf≥{combo.get('min_confidence', 0)}")
    else:
        parts.append("no_conf")
    parts.append("sess" if combo.get("use_session_filter") else "no_sess")
    parts.append("strk" if combo.get("use_streak_adj") else "no_strk")
    if combo.get("use_patience"):
        parts.append(f"pat{combo.get('soft_sl_bars', '?')}")
    return " | ".join(parts)


# ══════════════════════════════════════════════════════════════════════════
# MAIN OPTIMIZATION LOOP
# ══════════════════════════════════════════════════════════════════════════

def run_optimization(args):
    """Sequential or parallel grid search."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results_file = OUTPUT_DIR / "results.csv"

    grid = PARAM_GRID_QUICK if args.quick else PARAM_GRID_FULL
    all_combos = generate_combinations(grid)

    # Resume
    completed_keys: set[str] = set()
    if args.resume and results_file.exists():
        with open(results_file) as f:
            for row in csv.DictReader(f):
                completed_keys.add(_combo_key(row))
        print(f"  Resuming: {len(completed_keys)} already done")

    remaining = [c for c in all_combos if _combo_key(c) not in completed_keys]
    total = len(all_combos)
    done  = len(completed_keys)

    print(f"\n{'═'*60}")
    print(f"  TREND FOLLOWING OPTIMIZER")
    print(f"{'═'*60}")
    print(f"  Symbol:      {args.symbol}")
    print(f"  Timeframe:   {args.timeframe}")
    print(f"  Period:      {args.start} → {args.end}")
    print(f"  Grid:        {'QUICK' if args.quick else 'FULL'}")
    print(f"  Combinations: {total} ({len(remaining)} remaining)")
    print(f"  Workers:     {args.workers}")
    print(f"  Output:      {results_file}")
    print(f"{'═'*60}")
    sys.stdout.flush()

    if not remaining:
        print("\n  All combinations done.")
        _print_best(results_file)
        return

    # Prepare CSV
    write_header = not results_file.exists() or not args.resume
    if write_header:
        sample = {**remaining[0], "total_trades": 0, "winning_trades": 0,
                  "losing_trades": 0, "winrate": 0, "total_pnl": 0,
                  "total_pnl_pct": 0, "max_drawdown_pct": 0, "sharpe_ratio": 0,
                  "avg_trade_pnl": 0, "exposure_pct": 0, "score": 0, "status": ""}
        with open(results_file, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=_get_fieldnames(sample)).writeheader()

    # Build jobs
    jobs = [
        {
            "symbol": args.symbol, "timeframe": args.timeframe,
            "start": args.start, "end": args.end,
            "params": combo, "balance": args.balance,
            "commission_bps": args.commission_bps,
            "htf_timeframe": args.htf_timeframe,
        }
        for combo in remaining
    ]

    workers = min(args.workers, len(jobs))
    t0 = time.time()
    n_done = 0
    n_errors = 0

    print(f"\n  Starting {len(jobs)} backtests with {workers} workers...\n")
    sys.stdout.flush()

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_run_one_backtest, job): job["params"] for job in jobs}

        for future in as_completed(futures):
            n_done += 1
            combo = futures[future]

            try:
                result = future.result(timeout=300)  # 5 min max per backtest
            except Exception as exc:
                result = {
                    **combo,
                    "total_trades": 0, "winning_trades": 0, "losing_trades": 0,
                    "winrate": 0, "total_pnl": 0, "total_pnl_pct": 0,
                    "max_drawdown_pct": 0, "sharpe_ratio": 0, "avg_trade_pnl": 0,
                    "exposure_pct": 0, "score": -999,
                    "status": f"future_error: {exc}",
                }

            # Write to CSV
            with open(results_file, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=_get_fieldnames(result)).writerow(result)

            # Progress
            elapsed = time.time() - t0
            rate = n_done / elapsed if elapsed > 0 else 0
            eta = (len(remaining) - n_done) / rate / 60 if rate > 0 else 0
            i = done + n_done

            status = result.get("status", "?")
            if status == "ok":
                wr  = result.get("winrate", 0)
                pnl = result.get("total_pnl", 0)
                sc  = result.get("score", 0)
                nt  = result.get("total_trades", 0)
                print(
                    f"  [{i}/{total}] WR={wr:>5.1f}% PnL=${pnl:>8.2f} Sc={sc:>6.2f} "
                    f"T={nt:>3} | {_combo_summary(combo):<50} [{eta:.0f}m left]",
                    flush=True,
                )
            else:
                n_errors += 1
                print(f"  [{i}/{total}] ERROR: {status[:80]}", flush=True)

    # Done
    total_time = (time.time() - t0) / 60
    print(f"\n{'═'*60}")
    print(f"  COMPLETE — {n_done} runs in {total_time:.1f} min ({n_errors} errors)")
    print(f"{'═'*60}")
    _print_best(results_file)


# ══════════════════════════════════════════════════════════════════════════
# RESULTS ANALYSIS
# ══════════════════════════════════════════════════════════════════════════

def _print_best(results_file: Path):
    """Print top 10 and save best_params.json."""
    rows = []
    with open(results_file) as f:
        for row in csv.DictReader(f):
            if row.get("status") != "ok":
                continue
            for k in ["score", "winrate", "total_pnl", "total_pnl_pct",
                      "sharpe_ratio", "max_drawdown_pct", "total_trades",
                      "avg_trade_pnl", "exposure_pct"]:
                try:
                    row[k] = float(row[k])
                except (ValueError, KeyError):
                    row[k] = 0
            rows.append(row)

    if not rows:
        print("  No successful results found.")
        return

    rows.sort(key=lambda r: r["score"], reverse=True)

    print(f"\n  TOP 10 CONFIGURATIONS:")
    print(f"  {'#':<3} {'Score':>7} {'WR%':>6} {'PnL$':>9} {'PnL%':>7} {'Shrp':>6} {'DD%':>6} {'N':>4}")
    print(f"  {'─'*52}")
    for i, r in enumerate(rows[:10], 1):
        print(
            f"  {i:<3} {r['score']:>7.2f} {r['winrate']:>5.1f}% "
            f"${r['total_pnl']:>8.2f} {r['total_pnl_pct']:>6.2f}% "
            f"{r['sharpe_ratio']:>6.3f} {r['max_drawdown_pct']:>5.2f}% "
            f"{int(r['total_trades']):>4}"
        )

    # Save best
    best = rows[0]
    best_params = {}
    for k, v in best.items():
        if k in METRIC_KEYS:
            continue
        if v == "True":
            best_params[k] = True
        elif v == "False":
            best_params[k] = False
        else:
            try:
                fv = float(v)
                best_params[k] = int(fv) if fv == int(fv) else fv
            except (ValueError, TypeError):
                best_params[k] = v

    best_file = OUTPUT_DIR / "best_params.json"
    with open(best_file, "w") as f:
        json.dump({
            "params": best_params,
            "metrics": {k: best[k] for k in ["score", "winrate", "total_pnl",
                        "total_pnl_pct", "sharpe_ratio", "max_drawdown_pct", "total_trades"]},
        }, f, indent=2)

    print(f"\n  Best params → {best_file}")
    print(f"\n  BEST CONFIG:")
    for k, v in best_params.items():
        print(f"    {k}: {v}")

    # CLI command
    cli = ["python scripts/run_backtest.py --strategy trend_following"]
    flag_map = {
        "adx_min": "--adx-min", "pullback_tolerance_atr": "--pullback-tolerance-atr",
        "min_confidence": "--tf-min-confidence", "adx_strong": "--tf-adx-strong",
        "session_mult_eu": "--tf-sess-eu", "session_mult_other": "--tf-sess-other",
        "streak_euphoria_mult": "--tf-streak-mult", "soft_sl_bars": "--tf-soft-sl-bars",
        "tp1_close_pct": "--tp1-close-pct",
    }
    bool_flags = {
        "use_confidence": ("--tf-use-confidence", "--tf-no-confidence"),
        "use_session_filter": ("--tf-use-session", "--tf-no-session"),
        "use_streak_adj": ("--tf-use-streak", "--tf-no-streak"),
        "use_patience": ("--tf-use-patience", "--tf-no-patience"),
    }
    for k, (on, off) in bool_flags.items():
        if k in best_params:
            cli.append(on if best_params[k] else off)
    for k, flag in flag_map.items():
        if k in best_params:
            cli.append(f"{flag} {best_params[k]}")

    print(f"\n  Run command:")
    print(f"  {' '.join(cli)} --output best_result.json")


# ══════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimize Trend Following strategy")
    parser.add_argument("--symbol",         default="BTC-USDT")
    parser.add_argument("--timeframe",      default="15m")
    parser.add_argument("--start",          default="2025-06-01")
    parser.add_argument("--end",            default="2026-03-01")
    parser.add_argument("--balance",        type=float, default=10000.0)
    parser.add_argument("--commission-bps", type=float, default=7.5, dest="commission_bps")
    parser.add_argument("--htf-timeframe",  default="1h", dest="htf_timeframe")
    parser.add_argument("--workers",        type=int, default=4)
    parser.add_argument("--quick",          action="store_true")
    parser.add_argument("--resume",         action="store_true")
    args = parser.parse_args()

    run_optimization(args)