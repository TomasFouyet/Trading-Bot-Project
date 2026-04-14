#!/usr/bin/env python3
"""
R:R Plateau Test — verify rr=1.5 is a robust plateau, not an isolated spike.

Runs WFA smoke test (3 windows, 365d) across rr_ratio=[1.2..3.0]
with everything else fixed at validated baseline.
"""
from __future__ import annotations

import sys
import time
import itertools
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

import matplotlib
matplotlib.use("Agg")

from validation.fast_backtest import compute_indicators, compute_htf_bias, fast_backtest
from validation.data_loader import load_candles


# ── Config ──────────────────────────────────────────────────────────
RR_VALUES = [1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 2.0, 2.2, 2.5, 3.0]
SL_ATR = 2.0
N_WINDOWS = 3
IS_RATIO = 0.70
COMMISSION_PCT = 0.08

# Small IS optimization grid (8 combos)
IS_GRID = [
    {"adx_min": a, "ema_fast": f, "ema_slow": s}
    for a, f, s in itertools.product([20, 25], [15, 20], [45, 50])
]


def run_wfa_for_rr(df, htf_bias_full, rr_ratio):
    """Run 3-window WFA for a single rr_ratio, return per-window OOS metrics."""
    total = len(df)
    window_size = total // N_WINDOWS
    results = []

    for w_idx in range(N_WINDOWS):
        start = w_idx * window_size
        end = min(start + window_size, total)
        if end - start < 100:
            continue

        split = start + int((end - start) * IS_RATIO)
        is_df = df.iloc[start:split].reset_index(drop=True)
        oos_df = df.iloc[split:end].reset_index(drop=True)
        is_htf = htf_bias_full[start:split]
        oos_htf = htf_bias_full[split:end]

        if len(is_df) < 100 or len(oos_df) < 50:
            continue

        # Optimize on IS
        best_score = -np.inf
        best_params = IS_GRID[0]

        for params in IS_GRID:
            is_ind = compute_indicators(is_df, params["ema_fast"], params["ema_slow"])
            m = fast_backtest(
                is_ind, adx_min=params["adx_min"],
                ema_fast_p=params["ema_fast"], ema_slow_p=params["ema_slow"],
                rr_ratio=rr_ratio, atr_sl_mult=SL_ATR,
                precomputed=True, htf_bias=is_htf,
            )
            score = m.sharpe_ratio if m.total_trades >= 5 else -np.inf
            if score > best_score:
                best_score = score
                best_params = params

        # OOS with best params
        oos_ind = compute_indicators(oos_df, best_params["ema_fast"], best_params["ema_slow"])
        oos_m = fast_backtest(
            oos_ind, adx_min=best_params["adx_min"],
            ema_fast_p=best_params["ema_fast"], ema_slow_p=best_params["ema_slow"],
            rr_ratio=rr_ratio, atr_sl_mult=SL_ATR,
            precomputed=True, htf_bias=oos_htf,
        )
        results.append(oos_m)

    return results


def compute_row(rr, windows_metrics):
    """Compute summary stats for one rr_ratio."""
    if not windows_metrics:
        return None

    oos_annuals = [m.annual_return_pct for m in windows_metrics]
    oos_sharpes = [m.sharpe_ratio for m in windows_metrics]
    oos_positive = sum(1 for a in oos_annuals if a > 0)

    all_trades = []
    for m in windows_metrics:
        all_trades.extend(m.trades)

    if all_trades:
        pnls = [t["pnl_pct"] for t in all_trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        avg_win = np.mean(wins) if wins else 0
        avg_loss = abs(np.mean(losses)) if losses else 1
        winrate = len(wins) / len(pnls)
        expectancy_r = (winrate * avg_win / avg_loss) - (1 - winrate) - (COMMISSION_PCT / avg_loss) if avg_loss > 0 else 0
        n_trades = len(pnls)
    else:
        winrate = expectancy_r = 0
        n_trades = 0

    max_dd = max(m.max_drawdown_pct for m in windows_metrics)

    # Status
    passes_pos = oos_positive >= 2
    passes_sharpe = np.mean(oos_sharpes) > 1.0
    passes_exp = expectancy_r > 0.10
    marginal_sharpe = 0.5 < np.mean(oos_sharpes) <= 1.0

    if passes_pos and passes_sharpe and passes_exp:
        status = "PASS"
    elif passes_pos and marginal_sharpe:
        status = "MARGINAL"
    else:
        status = "FAIL"

    return {
        "rr": rr,
        "oos_annual": np.mean(oos_annuals),
        "oos_sharpe": np.mean(oos_sharpes),
        "oos_positive": oos_positive,
        "expectancy_r": expectancy_r,
        "n_trades": n_trades,
        "max_dd": max_dd,
        "status": status,
    }


def plot_plateau(rows, save_path):
    import matplotlib.pyplot as plt

    rr_vals = [r["rr"] for r in rows]
    sharpes = [r["oos_sharpe"] for r in rows]
    exps = [r["expectancy_r"] for r in rows]
    positives = [r["oos_positive"] for r in rows]
    statuses = [r["status"] for r in rows]

    def status_color(s):
        return {"PASS": "#2ecc71", "MARGINAL": "#f1c40f", "FAIL": "#e74c3c"}[s]

    colors = [status_color(s) for s in statuses]
    pos_colors = ["#2ecc71" if p == 3 else "#f1c40f" if p == 2 else "#e74c3c" for p in positives]

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    x = np.arange(len(rr_vals))
    labels = [str(r) for r in rr_vals]

    # Top: OOS Sharpe
    ax1.bar(x, sharpes, color=colors, edgecolor="white", width=0.7)
    ax1.axhline(1.0, color="gray", linestyle="--", linewidth=1, label="Sharpe = 1.0")
    baseline_idx = rr_vals.index(1.5)
    ax1.axvline(baseline_idx, color="navy", linestyle="--", linewidth=1.2, alpha=0.7, label="rr=1.5 (baseline)")
    ax1.set_ylabel("OOS Sharpe (avg)")
    ax1.set_title("OOS Sharpe by R:R Ratio")
    ax1.legend(fontsize=8)

    # Middle: Expectancy R
    ax2.bar(x, exps, color=colors, edgecolor="white", width=0.7)
    ax2.axhline(0.10, color="gray", linestyle="--", linewidth=1, label="Exp = +0.10R")
    ax2.axvline(baseline_idx, color="navy", linestyle="--", linewidth=1.2, alpha=0.7)
    ax2.set_ylabel("Expectancy (R)")
    ax2.set_title("Expectancy R by R:R Ratio")
    ax2.legend(fontsize=8)

    # Bottom: OOS windows positive
    ax3.bar(x, positives, color=pos_colors, edgecolor="white", width=0.7)
    ax3.axvline(baseline_idx, color="navy", linestyle="--", linewidth=1.2, alpha=0.7)
    ax3.set_ylabel("Positive windows")
    ax3.set_title("OOS Windows Positive (out of 3)")
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels)
    ax3.set_xlabel("rr_ratio")
    ax3.set_ylim(0, 3.5)
    ax3.set_yticks([0, 1, 2, 3])

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"[Plateau] Plot saved to {save_path}")
    plt.close(fig)


def main():
    print("=" * 70)
    print("R:R PLATEAU TEST — BTC/USDT 15m, HTF=ON, sl_atr=2.0")
    print(f"Testing rr_ratio: {RR_VALUES}")
    print(f"WFA: {N_WINDOWS} windows, IS grid: {len(IS_GRID)} combos, 365 days")
    print("=" * 70)

    df = load_candles("BTC/USDT", "15m", days=730)
    # Use last 365 days
    half = len(df) // 2
    df = df.iloc[half:].reset_index(drop=True)
    print(f"Data: {len(df)} bars (last 365d) from {df['ts'].iloc[0]} to {df['ts'].iloc[-1]}")

    # Pre-compute HTF bias once on the full slice
    htf_bias_full = compute_htf_bias(df)

    rows = []
    t_total = time.time()

    for idx, rr in enumerate(RR_VALUES):
        t0 = time.time()
        print(f"\n[{idx+1}/{len(RR_VALUES)}] rr={rr} ...", end=" ", flush=True)

        windows_metrics = run_wfa_for_rr(df, htf_bias_full, rr)
        row = compute_row(rr, windows_metrics)
        if row:
            rows.append(row)
            print(f"done ({time.time()-t0:.1f}s) | "
                  f"OOS: ann={row['oos_annual']:+.1f}% sh={row['oos_sharpe']:.2f} "
                  f"pos={row['oos_positive']}/3 exp={row['expectancy_r']:+.3f}R "
                  f"trades={row['n_trades']} DD={row['max_dd']:.1f}% [{row['status']}]",
                  flush=True)

    elapsed = time.time() - t_total
    print(f"\nTotal time: {elapsed:.0f}s ({elapsed/60:.1f} min)")

    # ── Results table ───────────────────────────────────────────────
    print(f"\n\n{'='*90}")
    print("R:R PLATEAU TEST RESULTS")
    print(f"{'='*90}")
    header = (f"{'rr':>5} | {'OOS Ann%':>9} | {'OOS Sharpe':>10} | "
              f"{'Pos/3':>5} | {'Exp R':>8} | {'Trades':>6} | "
              f"{'Max DD%':>7} | {'Status':>10}")
    print(header)
    print("-" * len(header))

    for r in rows:
        mark = " <-- BASELINE" if r["rr"] == 1.5 else ""
        sym = {"PASS": "V", "MARGINAL": "~", "FAIL": "X"}[r["status"]]
        print(
            f"{r['rr']:>5.1f} | {r['oos_annual']:>+9.1f} | {r['oos_sharpe']:>10.2f} | "
            f"{r['oos_positive']:>5}/3 | {r['expectancy_r']:>+8.4f} | "
            f"{r['n_trades']:>6} | {r['max_dd']:>7.1f} | "
            f"{sym} {r['status']:>8}{mark}"
        )

    # ── Plot ────────────────────────────────────────────────────────
    output_dir = ROOT / "validation" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_plateau(rows, str(output_dir / "rr_plateau_BTCUSDT.png"))

    # ── Plateau verdict ─────────────────────────────────────────────
    passing = [r for r in rows if r["status"] == "PASS"]
    passing_rr = [r["rr"] for r in passing]
    baseline_passes = 1.5 in passing_rr

    print(f"\n{'='*70}")
    print("PLATEAU VERDICT")
    print(f"{'='*70}")
    print(f"  Passing rr values: {passing_rr}")

    if len(passing_rr) >= 5:
        print(f"\n  WIDE PLATEAU confirmed. rr_ratio is robust across "
              f"{min(passing_rr)}-{max(passing_rr)}.")
        print(f"  rr=1.5 is NOT a spike. Confirmed for live trading.")
    elif len(passing_rr) >= 3 and baseline_passes:
        print(f"\n  MODERATE PLATEAU. rr=1.5 is stable but range is narrower "
              f"than ideal ({min(passing_rr)}-{max(passing_rr)}).")
        print(f"  Acceptable for live trading. Do not adjust rr without "
              f"re-running full WFA.")
    elif len(passing_rr) >= 1 and baseline_passes:
        print(f"\n  NARROW PLATEAU / POSSIBLE SPIKE. Only {len(passing_rr)} "
              f"value(s) pass.")
        print(f"  rr=1.5 works but neighbors fail. Treat as fragile.")
        print(f"  Reduce position size by 30% to account for parameter uncertainty.")
    elif not baseline_passes:
        print(f"\n  WARNING: rr=1.5 FAILS on this test. Prior full WFA used "
              f"730d, this test uses 365d.")
        print(f"  Check if it's a data period issue. Re-run with 730d before "
              f"drawing conclusions.")

    # Mathematical optimum (for information only)
    best_by_sharpe = max(rows, key=lambda r: r["oos_sharpe"])
    best_by_exp = max(rows, key=lambda r: r["expectancy_r"])

    print(f"\n  Mathematical optimum by Sharpe:     rr={best_by_sharpe['rr']} "
          f"(Sharpe={best_by_sharpe['oos_sharpe']:.2f})")
    print(f"  Mathematical optimum by Expectancy: rr={best_by_exp['rr']} "
          f"(Exp={best_by_exp['expectancy_r']:+.4f}R)")
    print(f"  NOTE: These are provided for information only. Do NOT switch")
    print(f"  to the mathematical optimum without running full WFA + PSA")
    print(f"  on the new value. That would be overfitting.")

    # ── One-line final answer ───────────────────────────────────────
    print(f"\n{'='*70}")
    if len(passing_rr) >= 5:
        print(f"rr=1.5 is CONFIRMED ROBUST — proceed as planned.")
    elif len(passing_rr) >= 3 and baseline_passes:
        print(f"rr=1.5 is CONFIRMED ROBUST (moderate plateau) — proceed as planned.")
    elif baseline_passes:
        print(f"rr=1.5 is FRAGILE — reduce position size by 30%.")
    else:
        print(f"rr=1.5 FAILED on 365d — re-test with 730d before deciding.")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
