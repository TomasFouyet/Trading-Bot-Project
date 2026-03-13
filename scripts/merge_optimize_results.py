"""
Merge CSV results from multiple optimizer shards into a single ranked report.

Usage:
    # Default: reads optimize_results/shard_*.csv
    python scripts/merge_optimize_results.py

    # Custom folder or specific files:
    python scripts/merge_optimize_results.py --input-dir ./my_results
    python scripts/merge_optimize_results.py --files shard_1.csv shard_2.csv

    # Save merged output:
    python scripts/merge_optimize_results.py --output final_results.csv
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import sys


def _score(r: dict, balance: float) -> float:
    """Combined rank score: WR×0.6 + PnL%×0.4 (min 5 trades)."""
    try:
        trades = int(r.get("total_trades", 0))
        wr     = float(r.get("winrate", 0))
        pnl    = float(r.get("total_pnl", 0))
        pnl_pct = pnl / balance * 100 if balance > 0 else float(r.get("total_pnl_pct", 0))
    except (ValueError, ZeroDivisionError):
        return -999.0
    if trades < 5:
        return -999.0
    return wr * 0.6 + pnl_pct * 0.4


def _fmt_row(r: dict, keys: list[str]) -> str:
    def _v(k: str) -> str:
        v = r.get(k, "")
        try:
            f = float(v)
            if k in ("winrate", "max_drawdown"):
                return f"{f:6.1f}%"
            if k in ("total_pnl",):
                return f"{f:+9.2f}"
            if k in ("sharpe",):
                return f"{f:7.3f}"
            if k in ("rsi_oversold", "rsi_overbought", "rr_ratio"):
                return f"{f:6.2f}"
            return f"{int(f):>5}"
        except (ValueError, TypeError):
            return str(v)[:8]
    return "  ".join(_v(k) for k in keys)


def main(args: argparse.Namespace) -> None:
    # ── Find input files ───────────────────────────────────────────────────────
    if args.files:
        files = args.files
    else:
        pattern = os.path.join(args.input_dir, "shard_*.csv")
        files = sorted(glob.glob(pattern))

    if not files:
        print(f"[ERROR] No CSV files found. Check --input-dir ({args.input_dir}) or use --files.")
        sys.exit(1)

    print(f"\nMerging {len(files)} shard file(s):")
    for f in files:
        print(f"  {f}")

    # ── Load all rows ──────────────────────────────────────────────────────────
    all_rows: list[dict] = []
    fieldnames: list[str] = []

    for path in files:
        if not os.path.exists(path):
            print(f"  [SKIP] {path} not found")
            continue
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            if not fieldnames and reader.fieldnames:
                fieldnames = list(reader.fieldnames)
            all_rows.extend(rows)
            print(f"  {path}: {len(rows)} results")

    if not all_rows:
        print("[ERROR] No data found in any shard file.")
        sys.exit(1)

    print(f"\nTotal results: {len(all_rows)}")

    # ── Rank ───────────────────────────────────────────────────────────────────
    ranked = sorted(all_rows, key=lambda r: _score(r, args.balance), reverse=True)
    top_n  = min(20, len(ranked))

    # Columns to show in the table
    show_keys = [
        "rsi_period", "ema_period", "swing_window", "swing_separation",
        "rsi_oversold", "rsi_overbought", "rr_ratio", "trend_ema_period",
        "total_trades", "winrate", "total_pnl", "max_drawdown", "sharpe",
    ]
    show_keys = [k for k in show_keys if k in fieldnames]

    HDR = "  ".join(f"{k:>8}" for k in show_keys)
    sep = "─" * (10 * len(show_keys))

    print(f"\nTOP {top_n} by Score (WR×0.6 + PnL%×0.4, min 5 trades)")
    print(sep)
    print(HDR)
    print(sep)
    for r in ranked[:top_n]:
        print(_fmt_row(r, show_keys))
    print(sep)

    # ── Highlight best ─────────────────────────────────────────────────────────
    valid = [r for r in all_rows if int(r.get("total_trades", 0)) >= 5]
    if valid:
        param_keys = [k for k in fieldnames if k not in
                      ("total_trades", "winrate", "total_pnl", "total_pnl_pct",
                       "max_drawdown", "sharpe", "avg_trade_pnl")]

        best_wr  = max(valid, key=lambda r: (float(r.get("winrate", 0)), float(r.get("total_pnl", 0))))
        best_pnl = max(valid, key=lambda r: (float(r.get("total_pnl", 0)), float(r.get("winrate", 0))))

        print(f"\n★  Best Win Rate:  WR={float(best_wr['winrate']):.1f}%  "
              f"PnL=${float(best_wr['total_pnl']):+.2f}")
        print(f"   " + "  ".join(f"{k}={best_wr.get(k, '?')}" for k in param_keys))

        print(f"\n★  Best PnL:       WR={float(best_pnl['winrate']):.1f}%  "
              f"PnL=${float(best_pnl['total_pnl']):+.2f}")
        print(f"   " + "  ".join(f"{k}={best_pnl.get(k, '?')}" for k in param_keys))

    # ── Save merged CSV ────────────────────────────────────────────────────────
    if args.output:
        with open(args.output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(ranked)
        print(f"\nMerged results saved to: {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge optimizer shard results")
    parser.add_argument("--input-dir", default="optimize_results", dest="input_dir",
                        help="Directory containing shard_N.csv files (default: optimize_results/)")
    parser.add_argument("--files",     nargs="+",
                        help="Explicit list of CSV files to merge")
    parser.add_argument("--output",    default="optimize_results/merged.csv",
                        help="Where to save the merged ranked CSV")
    parser.add_argument("--balance",   type=float, default=10000.0,
                        help="Initial balance used in the backtest (for PnL% calculation)")
    args = parser.parse_args()
    main(args)
