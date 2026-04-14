#!/usr/bin/env python3
"""
Grid test: Fast WFA on TrendFollowingV2Simple across rr_ratio × atr_sl_mult.

Uses vectorized fast_backtest instead of bar-by-bar StrategyAdapter (~100x faster).
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import itertools
import time

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")


def run_fast_wfa(
    df: pd.DataFrame,
    rr_ratio: float,
    atr_sl_mult: float,
    wfa_grid: list[dict],
    n_windows: int = 5,
    is_ratio: float = 0.70,
) -> dict:
    """
    Run Walk-Forward Analysis using fast_backtest.

    Returns dict with per-window OOS results.
    """
    from validation.fast_backtest import compute_indicators, fast_backtest

    total = len(df)
    window_size = total // n_windows
    windows_results = []

    for w_idx in range(n_windows):
        start = w_idx * window_size
        end = min(start + window_size, total)
        if end - start < 100:
            continue

        split = start + int((end - start) * is_ratio)
        is_df = df.iloc[start:split].reset_index(drop=True)
        oos_df = df.iloc[split:end].reset_index(drop=True)

        if len(is_df) < 100 or len(oos_df) < 50:
            continue

        # Optimize on IS: try each param combo
        best_score = -np.inf
        best_params = wfa_grid[0]

        for params in wfa_grid:
            is_ind = compute_indicators(is_df, params["ema_fast"], params["ema_slow"])
            m = fast_backtest(
                is_ind,
                adx_min=params["adx_min"],
                ema_fast_p=params["ema_fast"],
                ema_slow_p=params["ema_slow"],
                rr_ratio=rr_ratio,
                atr_sl_mult=atr_sl_mult,
                precomputed=True,
            )
            score = m.sharpe_ratio if m.total_trades >= 5 else -np.inf
            if score > best_score:
                best_score = score
                best_params = params

        # Validate on OOS with best params
        oos_ind = compute_indicators(oos_df, best_params["ema_fast"], best_params["ema_slow"])
        oos_m = fast_backtest(
            oos_ind,
            adx_min=best_params["adx_min"],
            ema_fast_p=best_params["ema_fast"],
            ema_slow_p=best_params["ema_slow"],
            rr_ratio=rr_ratio,
            atr_sl_mult=atr_sl_mult,
            precomputed=True,
        )
        windows_results.append(oos_m)

    return windows_results


def main():
    from validation.data_loader import load_candles

    df = load_candles("BTC/USDT", "15m", days=730)
    print(f"Data: {len(df)} bars\n")

    # ── Grid ─────────────────────────────────────────────────────────
    rr_ratios    = [1.5, 1.7, 2.0, 2.5]
    atr_sl_mults = [1.0, 1.5, 2.0]
    combos = list(itertools.product(rr_ratios, atr_sl_mults))

    # WFA IS optimization grid (27 combos)
    adx_vals = [15, 20, 25]
    ema_fast_vals = [15, 20, 25]
    ema_slow_vals = [40, 50, 60]
    wfa_grid = [
        {"adx_min": a, "ema_fast": f, "ema_slow": s}
        for a, f, s in itertools.product(adx_vals, ema_fast_vals, ema_slow_vals)
    ]

    commission_pct = 0.08  # 0.04% per side

    results = []
    t0 = time.time()

    for idx, (rr, sl_atr) in enumerate(combos):
        t1 = time.time()
        print(f"[{idx+1}/{len(combos)}] rr={rr}  sl_atr={sl_atr} ...", end=" ", flush=True)

        windows_metrics = run_fast_wfa(
            df, rr_ratio=rr, atr_sl_mult=sl_atr,
            wfa_grid=wfa_grid, n_windows=5,
        )

        oos_annuals = [m.annual_return_pct for m in windows_metrics]
        oos_sharpes = [m.sharpe_ratio for m in windows_metrics]
        oos_positive = sum(1 for r in oos_annuals if r > 0)

        # Expectancy in R across all OOS trades
        all_oos_trades = []
        for m in windows_metrics:
            all_oos_trades.extend(m.trades)

        if all_oos_trades:
            pnls = [t["pnl_pct"] for t in all_oos_trades]
            wins = [p for p in pnls if p > 0]
            losses = [p for p in pnls if p <= 0]
            avg_win = np.mean(wins) if wins else 0
            avg_loss = abs(np.mean(losses)) if losses else 1
            winrate = len(wins) / len(pnls)
            if avg_loss > 0:
                expectancy_r = (winrate * avg_win / avg_loss) - (1 - winrate) - (commission_pct / avg_loss)
            else:
                expectancy_r = 0
            n_trades = len(pnls)
        else:
            expectancy_r = 0
            winrate = 0
            n_trades = 0

        avg_oos_annual = np.mean(oos_annuals) if oos_annuals else 0
        avg_oos_sharpe = np.mean(oos_sharpes) if oos_sharpes else 0

        elapsed = time.time() - t1
        print(f"done ({elapsed:.1f}s) | OOS: ann={avg_oos_annual:+.1f}% sh={avg_oos_sharpe:.2f} "
              f"pos={oos_positive}/5 exp={expectancy_r:+.3f}R trades={n_trades}", flush=True)

        results.append({
            "rr_ratio": rr,
            "atr_sl_mult": sl_atr,
            "oos_annual_avg": avg_oos_annual,
            "oos_sharpe_avg": avg_oos_sharpe,
            "oos_positive": oos_positive,
            "expectancy_r": expectancy_r,
            "winrate": winrate * 100,
            "n_trades": n_trades,
        })

    total_time = time.time() - t0
    print(f"\nTotal grid time: {total_time:.0f}s")

    # ── Sort and print final table ───────────────────────────────────
    results.sort(key=lambda r: (-r["oos_positive"], -r["oos_sharpe_avg"]))

    print(f"\n\n{'='*90}")
    print("GRID RESULTS — TrendFollowingV2Simple (Walk-Forward OOS)")
    print(f"{'='*90}")
    header = (
        f"{'rr':>5} | {'sl_atr':>6} | {'OOS Ann%':>9} | {'OOS Sharpe':>10} | "
        f"{'Pos/5':>5} | {'Exp R':>7} | {'WR%':>5} | {'Trades':>6}"
    )
    print(header)
    print("-" * len(header))

    for r in results:
        passes = (
            r["oos_positive"] >= 3 and
            r["oos_sharpe_avg"] > 0.5 and
            r["expectancy_r"] > 0.15
        )
        mark = " <<< PASS" if passes else ""
        print(
            f"{r['rr_ratio']:>5.1f} | {r['atr_sl_mult']:>6.1f} | "
            f"{r['oos_annual_avg']:>+9.1f} | {r['oos_sharpe_avg']:>10.2f} | "
            f"{r['oos_positive']:>5}/5 | {r['expectancy_r']:>+7.3f} | "
            f"{r['winrate']:>5.1f} | {r['n_trades']:>6}{mark}"
        )

    # Check if any passed
    passing = [r for r in results if
               r["oos_positive"] >= 3 and
               r["oos_sharpe_avg"] > 0.5 and
               r["expectancy_r"] > 0.15]

    print(f"\n{'='*90}")
    if passing:
        print(f"PASSING COMBINATIONS: {len(passing)}")
        for p in passing:
            print(f"  rr={p['rr_ratio']} sl_atr={p['atr_sl_mult']} -> "
                  f"OOS Annual={p['oos_annual_avg']:+.1f}% "
                  f"Sharpe={p['oos_sharpe_avg']:.2f} "
                  f"Exp={p['expectancy_r']:+.3f}R")
    else:
        print("NONE PASSED — no combination meets all three criteria.")
        best = results[0]
        print(f"  Best attempt: rr={best['rr_ratio']} sl_atr={best['atr_sl_mult']} -> "
              f"Pos={best['oos_positive']}/5 "
              f"Sharpe={best['oos_sharpe_avg']:.2f} "
              f"Exp={best['expectancy_r']:+.3f}R")
    print(f"{'='*90}")


if __name__ == "__main__":
    main()
