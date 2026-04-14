#!/usr/bin/env python3
"""
Step 3 — Cross-pair validation: ETH/USDT, SOL/USDT, BNB/USDT.

Tests the validated config (rr=1.5, sl_atr=2.0, HTF=ON) on other pairs
to check whether the edge generalizes or is BTC-specific.

WFA n=3 windows, MC 2000 sims. OOS windows with <15 trades are flagged
as insufficient and excluded from the verdict.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import itertools
import numpy as np

import matplotlib
matplotlib.use("Agg")

from validation.fast_backtest import compute_indicators, compute_htf_bias, fast_backtest
from validation.monte_carlo import MonteCarloSimulation
from validation.data_loader import load_candles


# ── Config ──────────────────────────────────────────────────────────
RR = 1.5
SL_ATR = 2.0
N_WINDOWS = 3
IS_RATIO = 0.70
MC_SIMS = 2000
MIN_TRADES_PER_WINDOW = 15
COMMISSION_PCT = 0.08

PAIRS = [
    {"symbol": "ETH/USDT",  "days": 730},
    {"symbol": "SOL/USDT",  "days": 365},
    {"symbol": "BNB/USDT",  "days": 730},
]

WFA_GRID = [
    {"adx_min": a, "ema_fast": f, "ema_slow": s}
    for a, f, s in itertools.product([15, 20, 25], [15, 20, 25], [40, 50, 60])
]


def run_wfa_htf(df, n_windows=N_WINDOWS):
    """Run WFA with HTF=ON, return per-window OOS metrics."""
    htf_bias = compute_htf_bias(df)
    total = len(df)
    window_size = total // n_windows
    results = []

    for w_idx in range(n_windows):
        start = w_idx * window_size
        end = min(start + window_size, total)
        if end - start < 100:
            continue

        split = start + int((end - start) * IS_RATIO)
        is_df = df.iloc[start:split].reset_index(drop=True)
        oos_df = df.iloc[split:end].reset_index(drop=True)
        is_htf = htf_bias[start:split]
        oos_htf = htf_bias[split:end]

        if len(is_df) < 100 or len(oos_df) < 50:
            continue

        # Optimize on IS
        best_score = -np.inf
        best_params = WFA_GRID[0]

        for params in WFA_GRID:
            is_ind = compute_indicators(is_df, params["ema_fast"], params["ema_slow"])
            m = fast_backtest(
                is_ind, adx_min=params["adx_min"],
                ema_fast_p=params["ema_fast"], ema_slow_p=params["ema_slow"],
                rr_ratio=RR, atr_sl_mult=SL_ATR,
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
            rr_ratio=RR, atr_sl_mult=SL_ATR,
            precomputed=True, htf_bias=oos_htf,
        )

        sufficient = oos_m.total_trades >= MIN_TRADES_PER_WINDOW
        results.append({
            "window": w_idx,
            "metrics": oos_m,
            "params": best_params,
            "sufficient": sufficient,
        })

        flag = "" if sufficient else " ⚠ INSUFFICIENT (<15 trades)"
        print(f"    W{w_idx+1}: OOS trades={oos_m.total_trades}  "
              f"ann={oos_m.annual_return_pct:+.1f}%  "
              f"sharpe={oos_m.sharpe_ratio:.2f}  "
              f"DD={oos_m.max_drawdown_pct:.1f}%  "
              f"params={best_params}{flag}",
              flush=True)

    return results


def run_baseline_htf(df):
    """Full-data baseline backtest with HTF=ON."""
    htf_bias = compute_htf_bias(df)
    df_ind = compute_indicators(df, ema_fast_p=20, ema_slow_p=50)
    return fast_backtest(df_ind, rr_ratio=RR, atr_sl_mult=SL_ATR,
                         precomputed=True, htf_bias=htf_bias)


def summarize_pair(symbol, wfa_results, baseline):
    """Compute aggregate stats for one pair, excluding insufficient windows."""
    sufficient = [r for r in wfa_results if r["sufficient"]]
    insufficient = [r for r in wfa_results if not r["sufficient"]]

    n_total = len(wfa_results)
    n_suff = len(sufficient)

    if not sufficient:
        return {
            "symbol": symbol,
            "n_windows": n_total,
            "n_sufficient": 0,
            "n_insufficient": len(insufficient),
            "verdict": "INSUFFICIENT DATA",
            "oos_annual": 0, "oos_sharpe": 0, "oos_positive": 0,
            "winrate": 0, "expectancy_r": 0, "avg_dd": 0,
            "baseline_trades": baseline.total_trades,
            "mc_ror": None,
        }

    metrics_list = [r["metrics"] for r in sufficient]
    oos_annuals = [m.annual_return_pct for m in metrics_list]
    oos_sharpes = [m.sharpe_ratio for m in metrics_list]
    oos_positive = sum(1 for a in oos_annuals if a > 0)

    # Aggregate trades for expectancy
    all_trades = []
    for m in metrics_list:
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
        winrate = 0
        expectancy_r = 0
        n_trades = 0

    avg_annual = np.mean(oos_annuals)
    avg_sharpe = np.mean(oos_sharpes)
    avg_dd = np.mean([m.max_drawdown_pct for m in metrics_list])

    # Monte Carlo on baseline
    mc_ror = None
    if baseline.total_trades >= 10:
        mc = MonteCarloSimulation(trades=baseline.trades, n_simulations=MC_SIMS)
        mc_report = mc.run()
        mc_ror = mc_report.risk_of_ruin_pct

    # Verdict: >=2/3 sufficient windows positive, Sharpe>0.5, Exp>+0.15R
    min_positive = max(2, int(n_suff * 0.6))  # 60% of sufficient windows
    passes = (oos_positive >= min_positive and
              avg_sharpe > 0.5 and
              expectancy_r > 0.15)

    return {
        "symbol": symbol,
        "n_windows": n_total,
        "n_sufficient": n_suff,
        "n_insufficient": len(insufficient),
        "oos_annual": avg_annual,
        "oos_sharpe": avg_sharpe,
        "oos_positive": oos_positive,
        "n_trades": n_trades,
        "winrate": winrate * 100,
        "expectancy_r": expectancy_r,
        "avg_dd": avg_dd,
        "baseline_trades": baseline.total_trades,
        "mc_ror": mc_ror,
        "verdict": "PASS" if passes else "FAIL",
    }


def main():
    print("=" * 70)
    print("STEP 3 — CROSS-PAIR VALIDATION")
    print(f"Config: rr={RR}  sl_atr={SL_ATR}  HTF=ON (4H EMA50)")
    print(f"WFA: {N_WINDOWS} windows, IS/OOS={IS_RATIO:.0%}/{1-IS_RATIO:.0%}")
    print(f"Min trades/window: {MIN_TRADES_PER_WINDOW}")
    print("=" * 70)

    # Also run BTC as reference
    all_pairs = [{"symbol": "BTC/USDT", "days": 730}] + PAIRS
    pair_results = []

    for pair_cfg in all_pairs:
        symbol = pair_cfg["symbol"]
        days = pair_cfg["days"]

        print(f"\n{'─'*60}")
        print(f"  {symbol} ({days} days)")
        print(f"{'─'*60}")

        df = load_candles(symbol=symbol, timeframe="15m", days=days)
        print(f"  Data: {len(df)} bars from {df['ts'].iloc[0]} to {df['ts'].iloc[-1]}")

        print(f"\n  Baseline (full data, HTF=ON):")
        baseline = run_baseline_htf(df)
        print(f"    Trades: {baseline.total_trades}  WR: {baseline.winrate:.1f}%  "
              f"Ann: {baseline.annual_return_pct:+.1f}%  Sharpe: {baseline.sharpe_ratio:.2f}  "
              f"DD: {baseline.max_drawdown_pct:.1f}%")

        print(f"\n  Walk-Forward ({N_WINDOWS} windows, HTF=ON):")
        wfa_results = run_wfa_htf(df, n_windows=N_WINDOWS)

        summary = summarize_pair(symbol, wfa_results, baseline)
        pair_results.append(summary)

        print(f"\n  Summary for {symbol}:")
        print(f"    Sufficient windows: {summary['n_sufficient']}/{summary['n_windows']}"
              f" ({summary['n_insufficient']} excluded)")
        if summary["n_sufficient"] > 0:
            print(f"    OOS Annual avg:  {summary['oos_annual']:+.1f}%")
            print(f"    OOS Sharpe avg:  {summary['oos_sharpe']:.2f}")
            print(f"    OOS Positive:    {summary['oos_positive']}/{summary['n_sufficient']}")
            print(f"    Winrate:         {summary['winrate']:.1f}%")
            print(f"    Expectancy R:    {summary['expectancy_r']:+.4f}")
            print(f"    Avg Max DD:      {summary['avg_dd']:.1f}%")
            if summary["mc_ror"] is not None:
                print(f"    MC Risk of Ruin: {summary['mc_ror']:.1f}%")
        print(f"    VERDICT:         {summary['verdict']}")

    # ── Final comparison table ──────────────────────────────────────
    print(f"\n\n{'='*70}")
    print("CROSS-PAIR COMPARISON TABLE")
    print(f"{'='*70}")
    header = (f"{'Pair':>10} | {'Days':>4} | {'BL Trades':>9} | {'OOS Ann%':>9} | "
              f"{'Sharpe':>6} | {'Pos':>5} | {'Exp R':>7} | {'WR%':>5} | "
              f"{'DD%':>5} | {'RoR%':>5} | {'Verdict':>10}")
    print(header)
    print("-" * len(header))

    for r in pair_results:
        days = [p["days"] for p in all_pairs if p["symbol"] == r["symbol"]][0]
        ror_str = f"{r['mc_ror']:.1f}" if r["mc_ror"] is not None else "N/A"
        pos_str = f"{r['oos_positive']}/{r['n_sufficient']}" if r["n_sufficient"] > 0 else "N/A"
        print(
            f"{r['symbol']:>10} | {days:>4} | {r['baseline_trades']:>9} | "
            f"{r['oos_annual']:>+9.1f} | {r['oos_sharpe']:>6.2f} | "
            f"{pos_str:>5} | {r['expectancy_r']:>+7.4f} | "
            f"{r['winrate']:>5.1f} | {r['avg_dd']:>5.1f} | "
            f"{ror_str:>5} | {r['verdict']:>10}"
        )

    # ── Honest interpretation ───────────────────────────────────────
    non_btc = [r for r in pair_results if r["symbol"] != "BTC/USDT"]
    passing = [r for r in non_btc if r["verdict"] == "PASS"]
    failing = [r for r in non_btc if r["verdict"] == "FAIL"]
    insuff = [r for r in non_btc if r["verdict"] == "INSUFFICIENT DATA"]

    print(f"\n{'='*70}")
    print("CROSS-PAIR INTERPRETATION")
    print(f"{'='*70}")
    print(f"  Pairs tested:  {len(non_btc)}")
    print(f"  PASS:          {len(passing)}  {[r['symbol'] for r in passing]}")
    print(f"  FAIL:          {len(failing)}  {[r['symbol'] for r in failing]}")
    print(f"  INSUFFICIENT:  {len(insuff)}  {[r['symbol'] for r in insuff]}")

    if len(passing) >= 2:
        print("\n  CONCLUSION: Edge GENERALIZES across pairs. Strategy is not BTC-specific.")
    elif len(passing) == 1:
        print(f"\n  CONCLUSION: Partial generalization ({passing[0]['symbol']} passes). "
              "Edge may be pair-dependent. Paper trade BTC + passing pair only.")
    else:
        print("\n  CONCLUSION: Edge does NOT generalize. Strategy is BTC-specific.")
        print("  Recommendation: Only paper-trade BTC/USDT. The EMA+ADX+MACD+HTF approach")
        print("  may exploit BTC-specific trending behavior that other pairs lack.")

    print(f"\n{'='*70}")


if __name__ == "__main__":
    main()
