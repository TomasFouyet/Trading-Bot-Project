#!/usr/bin/env python3
"""
Compare best combo (rr=1.5, sl_atr=2.0) with and without HTF 4H filter.
Then write final summary report.
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


def run_wfa_with_htf(df, rr, sl_atr, wfa_grid, use_htf, n_windows=5, is_ratio=0.70):
    from validation.fast_backtest import compute_indicators, compute_htf_bias, fast_backtest

    # Pre-compute HTF bias on full data (if enabled)
    full_htf_bias = compute_htf_bias(df) if use_htf else None

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

        # HTF bias slices
        is_htf = full_htf_bias[start:split] if full_htf_bias is not None else None
        oos_htf = full_htf_bias[split:end] if full_htf_bias is not None else None

        # Optimize on IS
        best_score = -np.inf
        best_params = wfa_grid[0]

        for params in wfa_grid:
            is_ind = compute_indicators(is_df, params["ema_fast"], params["ema_slow"])
            m = fast_backtest(
                is_ind,
                adx_min=params["adx_min"],
                ema_fast_p=params["ema_fast"],
                ema_slow_p=params["ema_slow"],
                rr_ratio=rr, atr_sl_mult=sl_atr,
                precomputed=True,
                htf_bias=is_htf,
            )
            score = m.sharpe_ratio if m.total_trades >= 5 else -np.inf
            if score > best_score:
                best_score = score
                best_params = params

        # Validate on OOS
        oos_ind = compute_indicators(oos_df, best_params["ema_fast"], best_params["ema_slow"])
        oos_m = fast_backtest(
            oos_ind,
            adx_min=best_params["adx_min"],
            ema_fast_p=best_params["ema_fast"],
            ema_slow_p=best_params["ema_slow"],
            rr_ratio=rr, atr_sl_mult=sl_atr,
            precomputed=True,
            htf_bias=oos_htf,
        )
        windows_results.append(oos_m)

    return windows_results


def summarize(label, windows_metrics, commission_pct=0.08):
    oos_annuals = [m.annual_return_pct for m in windows_metrics]
    oos_sharpes = [m.sharpe_ratio for m in windows_metrics]
    oos_positive = sum(1 for r in oos_annuals if r > 0)

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
        expectancy_r = (winrate * avg_win / avg_loss) - (1 - winrate) - (commission_pct / avg_loss) if avg_loss > 0 else 0
        n_trades = len(pnls)
    else:
        winrate = 0
        expectancy_r = 0
        n_trades = 0

    avg_annual = np.mean(oos_annuals) if oos_annuals else 0
    avg_sharpe = np.mean(oos_sharpes) if oos_sharpes else 0
    avg_dd = np.mean([m.max_drawdown_pct for m in windows_metrics]) if windows_metrics else 0

    print(f"\n  {label}:")
    print(f"    OOS Trades:       {n_trades}")
    print(f"    OOS Annual avg:   {avg_annual:+.1f}%")
    print(f"    OOS Sharpe avg:   {avg_sharpe:.2f}")
    print(f"    OOS Positive:     {oos_positive}/5")
    print(f"    Winrate:          {winrate*100:.1f}%")
    print(f"    Expectancy R:     {expectancy_r:+.4f}")
    print(f"    Avg Max DD:       {avg_dd:.1f}%")

    passes = oos_positive >= 3 and avg_sharpe > 0.5 and expectancy_r > 0.15
    print(f"    PASSES CRITERIA:  {'YES' if passes else 'NO'}")

    return {
        "label": label,
        "n_trades": n_trades,
        "annual": avg_annual,
        "sharpe": avg_sharpe,
        "positive": oos_positive,
        "winrate": winrate * 100,
        "expectancy_r": expectancy_r,
        "avg_dd": avg_dd,
        "passes": passes,
    }


def main():
    from validation.data_loader import load_candles

    df = load_candles("BTC/USDT", "15m", days=730)
    print(f"Data: {len(df)} bars")

    # WFA grid
    wfa_grid = [
        {"adx_min": a, "ema_fast": f, "ema_slow": s}
        for a, f, s in itertools.product([15, 20, 25], [15, 20, 25], [40, 50, 60])
    ]

    rr, sl_atr = 1.5, 2.0

    print(f"\nBest combo from Part 2: rr={rr}  sl_atr={sl_atr}")
    print(f"{'='*60}")

    # ── Without HTF filter ──────────────────────────────────────
    print("\nRunning WFA WITHOUT HTF filter ...", flush=True)
    no_htf = run_wfa_with_htf(df, rr, sl_atr, wfa_grid, use_htf=False)
    r_no = summarize("Without HTF filter", no_htf)

    # ── With HTF filter ─────────────────────────────────────────
    print("\nRunning WFA WITH HTF 4H EMA50 filter ...", flush=True)
    with_htf = run_wfa_with_htf(df, rr, sl_atr, wfa_grid, use_htf=True)
    r_htf = summarize("With HTF 4H EMA50 filter", with_htf)

    # ── Comparison ──────────────────────────────────────────────
    trade_drop = (1 - r_htf["n_trades"] / max(r_no["n_trades"], 1)) * 100
    wr_delta = r_htf["winrate"] - r_no["winrate"]
    sharpe_delta = r_htf["sharpe"] - r_no["sharpe"]

    print(f"\n{'='*60}")
    print("HTF FILTER IMPACT:")
    print(f"  Trade reduction:  {trade_drop:.0f}%")
    print(f"  Winrate change:   {wr_delta:+.1f}pp")
    print(f"  Sharpe change:    {sharpe_delta:+.2f}")

    if trade_drop > 60 and sharpe_delta <= 0:
        print("  VERDICT: Filter too restrictive with no Sharpe improvement. SKIP.")
    elif r_htf["passes"]:
        print("  VERDICT: HTF filter PASSES all criteria!")
    else:
        print("  VERDICT: HTF filter does not reach +0.15R threshold.")

    # ══════════════════════════════════════════════════════════════
    # FINAL SUMMARY REPORT
    # ══════════════════════════════════════════════════════════════
    print(f"\n\n{'='*70}")
    print("FINAL VALIDATION SUMMARY REPORT")
    print(f"{'='*70}")

    print("""
1. MONTE CARLO BUG FIX
   Root cause: shuffle (without replacement) cannot change final PnL
   because both sum and product are commutative operations. Fix:
   replaced shuffle with bootstrap resampling (with replacement) for
   PnL distribution. Shuffle kept for drawdown (path-dependent).
   New MC result: P5=-40%, P50=+13%, P95=+120%, Risk of Ruin=37%.

2. ORIGINAL STRATEGY (TrendFollowingV2)
   Walk-Forward: 4/5 windows OOS negative — overfitted.
   Parameter Stability: narrow plateau, classic overfit signature.
   Monte Carlo: 37% risk of ruin — marginal edge at best.

3. SIMPLIFIED STRATEGY GRID (12 combinations)
   | rr  | sl_atr | OOS Ann% | Sharpe | Pos/5 | Exp R   |
   |-----|--------|----------|--------|-------|---------|
   | 2.0 |  1.0   |  +43.8%  |  7.45  |  5/5  | -0.049R |
   | 1.7 |  1.0   |  +24.7%  |  5.61  |  5/5  | -0.102R |
   | 1.5 |  2.0   |  +60.4%  |  5.51  |  5/5  | +0.094R | ← best exp
   | 1.5 |  1.0   |  +33.7%  |  9.88  |  4/5  | -0.076R |
   | 2.5 |  1.0   |  +20.9%  |  6.19  |  4/5  | -0.070R |
   | 1.5 |  1.5   |  +33.2%  |  5.22  |  4/5  | -0.020R |
   | 2.0 |  1.5   |  +27.4%  |  4.05  |  4/5  | -0.031R |
   | 2.5 |  1.5   |  +32.0%  |  3.69  |  4/5  | +0.002R |
   | 1.7 |  2.0   |  +27.7%  |  3.52  |  4/5  | +0.003R |
   | 1.7 |  1.5   |  +20.1%  |  3.21  |  4/5  | -0.072R |
   | 2.0 |  2.0   |  +18.5%  |  1.85  |  2/5  | -0.025R |
   | 2.5 |  2.0   |  +18.3%  | -0.51  |  2/5  | -0.023R |

   NONE PASSED all 3 criteria (>=3 pos, Sharpe>0.5, Exp>+0.15R).
   Best: rr=1.5 sl_atr=2.0 → 5/5 positive, Sharpe 5.51, but Exp=+0.094R.""")

    print(f"""
4. HTF 4H FILTER (on best combo rr=1.5, sl_atr=2.0)
   Without filter: {r_no['n_trades']} trades, WR={r_no['winrate']:.1f}%, Exp={r_no['expectancy_r']:+.4f}R
   With filter:    {r_htf['n_trades']} trades, WR={r_htf['winrate']:.1f}%, Exp={r_htf['expectancy_r']:+.4f}R
   Trade reduction: {trade_drop:.0f}%  |  Winrate delta: {wr_delta:+.1f}pp""")

    # Final recommendation
    any_pass = r_no["passes"] or r_htf["passes"]
    print(f"""
5. HONEST RECOMMENDATION""")

    if any_pass:
        print("   The HTF filter variant PASSES criteria. Worth paper-trading.")
    else:
        best_exp = max(r_no["expectancy_r"], r_htf["expectancy_r"])
        print(f"""   This EMA+ADX+MACD trend-following approach does NOT have sufficient
   edge on BTC 15m. The entry logic generates OOS-positive returns in
   most windows (a good sign), but the per-trade expectancy ({best_exp:+.3f}R)
   falls short of the +0.15R threshold after commissions.

   The simplification helped (5/5 windows positive vs 1/5 for V2), but
   the edge is too thin. Possible next hypotheses:
   a) Switch to 1H timeframe (less noise, wider SL, fewer commissions)
   b) Mean reversion instead of trend following (BTC ranges 70% of time)
   c) Volume/orderflow-based entries instead of lagging indicators
   d) Reduce to high-confidence-only entries (conf >= 0.6) to boost WR""")

    print(f"\n{'='*70}")


if __name__ == "__main__":
    main()
