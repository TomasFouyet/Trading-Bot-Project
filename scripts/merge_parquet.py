"""
Merge monthly Parquet files into one file per symbol/timeframe.

Before:  data/parquet/BTC-USDT/15m/2024-01.parquet
         data/parquet/BTC-USDT/15m/2024-02.parquet ...
After:   data/parquet/BTC-USDT/BTC-USDT_15m.parquet

Usage:
    python scripts/merge_parquet.py
    python scripts/merge_parquet.py --input-dir data/parquet --output-dir data/merged
    python scripts/merge_parquet.py --delete-originals   # remove monthly files after merge
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd


def merge_symbol_timeframe(
    monthly_dir: Path,
    output_file: Path,
) -> int:
    """Merge all parquet files in monthly_dir into output_file. Returns total rows."""
    files = sorted(monthly_dir.glob("*.parquet"))
    if not files:
        return 0

    dfs = [pd.read_parquet(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)

    # Deduplicate and sort by timestamp
    ts_col = next((c for c in df.columns if c in ("ts", "timestamp", "time", "open_time")), None)
    if ts_col:
        df = df.drop_duplicates(subset=[ts_col]).sort_values(ts_col).reset_index(drop=True)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_file, index=False)
    return len(df)


def main(args: argparse.Namespace) -> None:
    input_root = Path(args.input_dir)
    output_root = Path(args.output_dir)

    if not input_root.exists():
        print(f"Input directory not found: {input_root}")
        return

    # Find all symbol/timeframe directories (2 levels deep)
    jobs: list[tuple[str, str, Path]] = []
    for symbol_dir in sorted(input_root.iterdir()):
        if not symbol_dir.is_dir():
            continue
        for tf_dir in sorted(symbol_dir.iterdir()):
            if not tf_dir.is_dir():
                continue
            if any(tf_dir.glob("*.parquet")):
                jobs.append((symbol_dir.name, tf_dir.name, tf_dir))

    if not jobs:
        print(f"No parquet files found under {input_root}")
        return

    print(f"Merging {len(jobs)} symbol/timeframe pairs into {output_root}/\n")

    total_rows = 0
    for symbol, tf, monthly_dir in jobs:
        output_file = output_root / symbol / f"{symbol}_{tf}.parquet"
        rows = merge_symbol_timeframe(monthly_dir, output_file)
        total_rows += rows
        print(f"  {symbol:12s} {tf:4s}  →  {output_file}  ({rows:,} rows)")

        if args.delete_originals:
            shutil.rmtree(monthly_dir)

    print(f"\nDone. Total rows: {total_rows:,}")
    print(f"Output: {output_root.resolve()}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge monthly parquet files per symbol/timeframe")
    parser.add_argument("--input-dir", default="data/parquet", help="Root parquet directory")
    parser.add_argument("--output-dir", default="data/merged", help="Output directory")
    parser.add_argument(
        "--delete-originals", action="store_true",
        help="Delete monthly source files after merging"
    )
    args = parser.parse_args()
    main(args)
