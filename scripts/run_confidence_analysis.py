"""
Confidence Score vs Win Rate — Descriptive Analysis.

Strategy: TrendFollowingV2Simple + Structural Stop, rr=2.5, HTF=ON
Data: BTC/USDT 15m 730d

Steps 1-8 + Figures 1-2 + Final verdict.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from validation.data_loader import load_candles
from validation.fast_backtest import (
    fast_backtest, compute_indicators, compute_htf_bias, _fast_confidence,
)

OUTPUT_DIR = ROOT / "validation" / "output"
PLT_BG = "#0d0d0d"; PLT_FG = "#e6e6e6"

# Config
RR = 2.5
STRUCT_CFG = dict(
    stop_mode="STRUCTURAL", atr_sl_mult=2.0,
    buffer_atr=0.25, min_risk_atr=0.8,
    pivot_left=3, pivot_right=3,
)
ENTRY_PARAMS = dict(
    adx_min=20.0, ema_fast_p=20, ema_slow_p=50,
    pb_tol_atr=1.0, sig_cooldown=5, allow_short=True,
    min_confidence=0.0, adx_strong=35.0, slope_bars=5,
)


def _dark_style():
    plt.rcParams.update({
        "figure.facecolor": PLT_BG, "axes.facecolor": PLT_BG,
        "axes.edgecolor": PLT_FG, "axes.labelcolor": PLT_FG,
        "xtick.color": PLT_FG, "ytick.color": PLT_FG,
        "text.color": PLT_FG, "axes.titlecolor": PLT_FG,
        "grid.color": "#303030", "savefig.facecolor": PLT_BG,
    })


def expectancy_r(pnls):
    arr = np.array(pnls) if not isinstance(pnls, np.ndarray) else pnls
    if len(arr) == 0: return 0.0
    losses = arr[arr < 0]
    if len(losses) == 0: return float(np.mean(arr))
    r = abs(np.mean(losses))
    return float(np.mean(arr) / r) if r > 0 else 0.0


def binom_ci_95(wins, total):
    if total == 0: return 0, 0
    p = wins / total
    se = np.sqrt(p * (1 - p) / total)
    return max(0, p - 1.96 * se) * 100, min(1, p + 1.96 * se) * 100


# ═══════════════════ STEP 1: EXTRACT TRADE DATA ═══════════════════
def step1(df, htf):
    print("\n" + "=" * 70)
    print("STEP 1 — Run structural backtest and extract confidence data")
    print("=" * 70)
    m = fast_backtest(df, htf_bias=htf, rr_ratio=RR, **ENTRY_PARAMS, **STRUCT_CFG)
    trades = m.trades
    print(f"  Total trades: {len(trades)}")
    print(f"\n  {'#':>4} | {'dir':>5} | {'conf':>5} | {'outcome':>7} | {'pnl%':>7}")
    print(f"  {'-'*4}-+-{'-'*5}-+-{'-'*5}-+-{'-'*7}-+-{'-'*7}")
    for i, t in enumerate(trades[:10]):
        print(f"  {i+1:>4} | {t['direction']:>5} | {t['confidence']:.3f} | "
              f"{t['exit_type']:>7} | {t['pnl_pct']:>+6.2f}%")
    return trades


# ═══════════════════ STEP 2: DISTRIBUTION ══════════════════════════
def step2(trades):
    print("\n" + "=" * 70)
    print("STEP 2 — Confidence distribution analysis")
    print("=" * 70)
    confs = np.array([t["confidence"] for t in trades])
    print(f"\n  [2a] Overall distribution")
    print(f"  Mean:    {np.mean(confs):.3f}")
    print(f"  Median:  {np.median(confs):.3f}")
    print(f"  Std:     {np.std(confs):.3f}")
    print(f"  P25:     {np.percentile(confs, 25):.3f}")
    print(f"  P75:     {np.percentile(confs, 75):.3f}")
    print(f"  >= 0.5:  {np.mean(confs >= 0.5)*100:.1f}%")
    print(f"  >= 0.7:  {np.mean(confs >= 0.7)*100:.1f}%")

    winners = [t for t in trades if t["exit_type"] == "tp"]
    losers = [t for t in trades if t["exit_type"] == "sl"]
    wc = np.array([t["confidence"] for t in winners])
    lc = np.array([t["confidence"] for t in losers])
    print(f"\n  [2b] Distribution by outcome")
    print(f"  {'Outcome':<10} | {'N':>5} | {'Avg conf':>9} | {'Median conf':>11}")
    print(f"  {'-'*10}-+-{'-'*5}-+-{'-'*9}-+-{'-'*11}")
    print(f"  {'TP (win)':<10} | {len(wc):>5} | {np.mean(wc):>9.3f} | {np.median(wc):>11.3f}")
    print(f"  {'SL (loss)':<10} | {len(lc):>5} | {np.mean(lc):>9.3f} | {np.median(lc):>11.3f}")
    delta = np.mean(wc) - np.mean(lc)
    signal = "YES — signal exists" if abs(delta) > 0.05 else ("WEAK" if abs(delta) > 0.02 else "NO — essentially no signal")
    print(f"\n  Delta (W-L): {delta:+.4f}  → {signal}")
    return confs, wc, lc, delta


# ═══════════════════ STEP 3: WR BY BUCKET ══════════════════════════
def step3(trades, confs):
    print("\n" + "=" * 70)
    print("STEP 3 — Win Rate by Confidence Bucket")
    print("=" * 70)

    # 3a Fixed buckets
    edges = [0.0, 0.2, 0.4, 0.6, 0.8, 1.01]
    print("\n  [3a] Fixed buckets (5 equal-width)")
    print(f"  {'Bucket':<12} | {'N':>5} | {'Winners':>7} | {'WR%':>6} | {'Avg PnL':>8} | {'ExpR':>6} | Note")
    print(f"  {'-'*12}-+-{'-'*5}-+-{'-'*7}-+-{'-'*6}-+-{'-'*8}-+-{'-'*6}-+-----")
    fixed_buckets = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = [(lo <= t["confidence"] < hi) for t in trades]
        bucket_trades = [t for t, m in zip(trades, mask) if m]
        n = len(bucket_trades)
        wins = sum(1 for t in bucket_trades if t["exit_type"] == "tp")
        pnls = [t["pnl_pct"] for t in bucket_trades]
        wr = wins / n * 100 if n > 0 else 0
        avg_pnl = np.mean(pnls) if pnls else 0
        expr = expectancy_r(pnls)
        note = "INSUFFICIENT" if n < 15 else ""
        label = f"{lo:.2f}-{min(hi, 1.0):.2f}"
        print(f"  {label:<12} | {n:>5} | {wins:>7} | {wr:>5.1f}% | {avg_pnl:>+7.2f}% | {expr:>+5.2f} | {note}")
        fixed_buckets.append(dict(lo=lo, hi=hi, n=n, wins=wins, wr=wr, avg_pnl=avg_pnl, expr=expr, label=label))

    # 3b Quartile buckets
    print("\n  [3b] Quartile buckets (equal-count)")
    sorted_trades = sorted(trades, key=lambda t: t["confidence"])
    q_size = len(sorted_trades) // 4
    quartiles = []
    for qi in range(4):
        start = qi * q_size
        end = start + q_size if qi < 3 else len(sorted_trades)
        qt = sorted_trades[start:end]
        qconfs = [t["confidence"] for t in qt]
        wins = sum(1 for t in qt if t["exit_type"] == "tp")
        pnls = [t["pnl_pct"] for t in qt]
        wr = wins / len(qt) * 100
        expr = expectancy_r(pnls)
        sharpe = 0
        if len(pnls) > 1:
            mu = np.mean(pnls); sd = np.std(pnls, ddof=1)
            sharpe = mu / sd * np.sqrt(len(pnls)) if sd > 0 else 0
        quartiles.append(dict(
            qi=qi+1, lo=min(qconfs), hi=max(qconfs), n=len(qt),
            wins=wins, wr=wr, expr=expr, sharpe=sharpe, trades=qt,
        ))

    print(f"  {'Q':>3} | {'Conf range':>14} | {'N':>4} | {'WR%':>6} | {'ExpR':>6} | {'Sharpe':>7}")
    print(f"  {'-'*3}-+-{'-'*14}-+-{'-'*4}-+-{'-'*6}-+-{'-'*6}-+-{'-'*7}")
    for q in quartiles:
        print(f"  Q{q['qi']} | {q['lo']:.3f} - {q['hi']:.3f} | {q['n']:>4} | "
              f"{q['wr']:>5.1f}% | {q['expr']:>+5.2f} | {q['sharpe']:>6.2f}")

    # 3c Monotonicity
    wrs = [q["wr"] for q in quartiles]
    if all(wrs[i] < wrs[i+1] for i in range(3)):
        pattern = "MONOTONIC"
    elif wrs[-1] > wrs[0] and sum(1 for i in range(3) if wrs[i] < wrs[i+1]) >= 2:
        pattern = "PARTIAL"
    elif max(wrs) - min(wrs) < 5:
        pattern = "FLAT"
    elif wrs[-1] < wrs[0]:
        pattern = "INVERSE"
    else:
        pattern = "PARTIAL"
    print(f"\n  [3c] Monotonicity: {pattern}")
    print(f"  WR progression: Q1={wrs[0]:.1f}% → Q2={wrs[1]:.1f}% → Q3={wrs[2]:.1f}% → Q4={wrs[3]:.1f}%")
    print(f"  Max delta: {max(wrs) - min(wrs):.1f}pp")
    return fixed_buckets, quartiles, pattern


# ═══════════════════ STEP 4: STATISTICAL SIGNIFICANCE ══════════════
def step4(quartiles):
    print("\n" + "=" * 70)
    print("STEP 4 — Statistical significance (Fisher exact test)")
    print("=" * 70)

    def _fisher(qa, qb):
        w_a, l_a = qa["wins"], qa["n"] - qa["wins"]
        w_b, l_b = qb["wins"], qb["n"] - qb["wins"]
        table = [[w_a, l_a], [w_b, l_b]]
        _, p = stats.fisher_exact(table)
        return qa["wr"] - qb["wr"], p

    comparisons = [
        ("Q1 vs Q2", 0, 1), ("Q2 vs Q3", 1, 2),
        ("Q3 vs Q4", 2, 3), ("Q1 vs Q4", 0, 3),
    ]
    print(f"\n  {'Comparison':>12} | {'WR diff':>8} | {'p-value':>8} | Significant?")
    print(f"  {'-'*12}-+-{'-'*8}-+-{'-'*8}-+-------------")
    results = {}
    for label, i, j in comparisons:
        diff, p = _fisher(quartiles[i], quartiles[j])
        sig = "YES" if p < 0.05 else "NO"
        print(f"  {label:>12} | {-diff:>+7.1f}pp | {p:>8.4f} | {sig} (p<0.05)")
        results[label] = dict(diff=-diff, p=p, sig=p < 0.05)

    q14 = results["Q1 vs Q4"]
    if q14["sig"]:
        print(f"\n  Confidence score has STATISTICALLY SIGNIFICANT predictive power.")
    else:
        print(f"\n  Confidence score does NOT significantly predict WR (p={q14['p']:.4f}).")
        print(f"  Differences may be random given sample size ({sum(q['n'] for q in quartiles)} trades).")
    return results


# ═══════════════════ STEP 5: LONG VS SHORT ═════════════════════════
def step5(trades):
    print("\n" + "=" * 70)
    print("STEP 5 — Long vs Short quartile breakdown")
    print("=" * 70)

    for direction in ["LONG", "SHORT"]:
        dir_trades = sorted([t for t in trades if t["direction"] == direction],
                           key=lambda t: t["confidence"])
        if len(dir_trades) < 20:
            print(f"\n  {direction}: only {len(dir_trades)} trades — INSUFFICIENT")
            continue
        q_size = len(dir_trades) // 4
        wrs = []
        labels = []
        for qi in range(4):
            start = qi * q_size
            end = start + q_size if qi < 3 else len(dir_trades)
            qt = dir_trades[start:end]
            wins = sum(1 for t in qt if t["exit_type"] == "tp")
            wr = wins / len(qt) * 100
            wrs.append(wr)
            labels.append(f"Q{qi+1}")
        mono = all(wrs[i] < wrs[i+1] for i in range(3))
        print(f"\n  {direction} ({len(dir_trades)} trades):")
        print(f"    Q1={wrs[0]:.1f}% | Q2={wrs[1]:.1f}% | Q3={wrs[2]:.1f}% | Q4={wrs[3]:.1f}%")
        print(f"    Monotonic: {'YES' if mono else 'NO'}  delta Q4-Q1={wrs[3]-wrs[0]:+.1f}pp")


# ═══════════════════ STEP 6: FACTOR CONTRIBUTION ═══════════════════
def step6(trades, df):
    print("\n" + "=" * 70)
    print("STEP 6 — Factor contribution analysis")
    print("=" * 70)

    df_ind = compute_indicators(df, 20, 50, 5)
    atr = df_ind["atr"].values.astype(np.float64)
    adx_arr = df_ind["adx"].values.astype(np.float64)
    ema_f = df_ind["ema_fast"].values.astype(np.float64)
    ema_f_slope = df_ind["ema_fast_slope"].values.astype(np.float64)
    close = df_ind["close"].values.astype(np.float64)
    opn = df_ind["open"].values.astype(np.float64)
    high_arr = df_ind["high"].values.astype(np.float64)
    low_arr = df_ind["low"].values.astype(np.float64)
    macd_hist = df_ind["macd_hist"].values.astype(np.float64)
    volume = df_ind["volume"].values.astype(np.float64)
    vol_sma = df_ind["vol_sma"].values.astype(np.float64)

    factor_results = {}
    for fname in ["adx_strong", "pullback_tight", "macd_increasing", "body_ratio", "ema_slope", "volume_high"]:
        has_factor = []; no_factor = []
        for t in trades:
            i = t["entry_bar_idx"]
            if i < 0 or i >= len(atr): continue
            dx = adx_arr[i]; a = atr[i]; c = close[i]; ef = ema_f[i]
            h = high_arr[i]; lo = low_arr[i]; o = opn[i]
            hist = macd_hist[i]; prev_hist = macd_hist[i-1] if i > 0 else 0
            ef_slope = ema_f_slope[i]; vol = volume[i]; vsma = vol_sma[i]

            win = t["exit_type"] == "tp"
            d = t["direction"]

            if fname == "adx_strong":
                contributed = dx >= 35.0
            elif fname == "pullback_tight":
                pb_atr = abs(c - ef) / a if a > 0 else 999
                contributed = pb_atr <= 0.5
            elif fname == "macd_increasing":
                if d == "LONG":
                    contributed = hist > 0 and hist > prev_hist
                else:
                    contributed = hist < 0 and hist < prev_hist
            elif fname == "body_ratio":
                body = abs(c - o); rng = h - lo
                contributed = (body / rng >= 0.60) if rng > 0 else False
            elif fname == "ema_slope":
                if d == "LONG":
                    contributed = not np.isnan(ef_slope) and ef_slope > 0
                else:
                    contributed = not np.isnan(ef_slope) and ef_slope < 0
            elif fname == "volume_high":
                contributed = (vol / vsma >= 1.2) if vsma > 0 and not np.isnan(vsma) else False
            else:
                contributed = False

            if contributed:
                has_factor.append(win)
            else:
                no_factor.append(win)

        wr_has = np.mean(has_factor) * 100 if has_factor else 0
        wr_no = np.mean(no_factor) * 100 if no_factor else 0
        delta = wr_has - wr_no
        factor_results[fname] = dict(n_has=len(has_factor), n_no=len(no_factor),
                                     wr_has=wr_has, wr_no=wr_no, delta=delta)

    print(f"\n  {'Factor':<18} | {'Contrib':>7} | {'WR contrib':>10} | {'WR no':>10} | {'Delta':>7}")
    print(f"  {'-'*18}-+-{'-'*7}-+-{'-'*10}-+-{'-'*10}-+-{'-'*7}")
    for fname, r in sorted(factor_results.items(), key=lambda x: -abs(x[1]["delta"])):
        print(f"  {fname:<18} | {r['n_has']:>5}t  | {r['wr_has']:>9.1f}% | {r['wr_no']:>9.1f}% | {r['delta']:>+6.1f}pp")
    return factor_results


# ═══════════════════ STEP 7: SIZING SCENARIOS ══════════════════════
def step7(trades, quartiles, sig_results):
    print("\n" + "=" * 70)
    print("STEP 7 — Position sizing implications")
    print("=" * 70)

    pnls = np.array([t["pnl_pct"] for t in trades])

    # Scenario A: fixed 1.5%
    equity_a = [1.0]
    for p in pnls:
        equity_a.append(equity_a[-1] * (1 + 0.015 * p / 100 * 100))
    # Simpler: just scale pnl by risk fraction
    # Each trade's equity impact = risk% * (pnl / sl_dist)... Too complex.
    # Simpler approach: equal-risk trades, equity = prod(1 + pnl/100)
    eq_a = np.cumprod(1 + pnls / 100)
    sharpe_a = 0
    if len(pnls) > 1:
        mu = np.mean(pnls); sd = np.std(pnls, ddof=1)
        sharpe_a = mu / sd * np.sqrt(len(pnls)) if sd > 0 else 0

    # Quartile boundaries for mapping
    q_bounds = [(quartiles[i]["lo"], quartiles[i]["hi"]) for i in range(4)]
    risk_map = {0: 0.75, 1: 1.25, 2: 1.50, 3: 2.00}

    q14_diff = quartiles[3]["wr"] - quartiles[0]["wr"]
    q14_sig = sig_results.get("Q1 vs Q4", {}).get("sig", False)
    justified = q14_diff > 8.0 and q14_sig

    if not justified:
        print(f"\n  Confidence-weighted sizing NOT justified.")
        print(f"  Q1-Q4 WR diff: {q14_diff:.1f}pp (need >8pp)")
        print(f"  Q1-Q4 p-value significant: {q14_sig}")
        print(f"  → Keep fixed risk_pct = 1.5% per trade")
        return justified, None

    # Scenario B: confidence-weighted
    print(f"\n  Q1-Q4 WR diff: {q14_diff:.1f}pp > 8pp AND significant → JUSTIFIED")
    scaled_pnls = []
    for t in trades:
        conf = t["confidence"]
        # Find quartile
        qi = 0
        for j in range(4):
            if conf >= q_bounds[j][0]: qi = j
        risk = risk_map[qi]
        scaled_pnl = t["pnl_pct"] * (risk / 1.5)
        scaled_pnls.append(scaled_pnl)
    scaled_pnls = np.array(scaled_pnls)
    eq_b = np.cumprod(1 + scaled_pnls / 100)
    sharpe_b = 0
    if len(scaled_pnls) > 1:
        mu = np.mean(scaled_pnls); sd = np.std(scaled_pnls, ddof=1)
        sharpe_b = mu / sd * np.sqrt(len(scaled_pnls)) if sd > 0 else 0
    dd_a = np.min((np.cumprod(1 + pnls / 100) - np.maximum.accumulate(np.cumprod(1 + pnls / 100)))
                   / np.maximum.accumulate(np.cumprod(1 + pnls / 100))) * 100
    dd_b = np.min((np.cumprod(1 + scaled_pnls / 100) - np.maximum.accumulate(np.cumprod(1 + scaled_pnls / 100)))
                   / np.maximum.accumulate(np.cumprod(1 + scaled_pnls / 100))) * 100

    print(f"\n  {'Metric':<16} | {'Fixed 1.5%':>12} | {'Conf-weighted':>14} | {'Delta':>8}")
    print(f"  {'-'*16}-+-{'-'*12}-+-{'-'*14}-+-{'-'*8}")
    print(f"  {'Final equity':<16} | {eq_a[-1]:>12.3f} | {eq_b[-1]:>14.3f} | {eq_b[-1]-eq_a[-1]:>+7.3f}")
    print(f"  {'Sharpe':<16} | {sharpe_a:>12.2f} | {sharpe_b:>14.2f} | {sharpe_b-sharpe_a:>+7.2f}")
    print(f"  {'Max DD':<16} | {dd_a:>11.1f}% | {dd_b:>13.1f}% | {dd_b-dd_a:>+7.1f}%")
    return justified, dict(eq_a=eq_a, eq_b=eq_b, sharpe_a=sharpe_a, sharpe_b=sharpe_b)


# ═══════════════════ STEP 8: MIN_CONF SWEEP ════════════════════════
def step8(trades, df, htf):
    print("\n" + "=" * 70)
    print("STEP 8 — Optimal min_conf threshold sweep")
    print("=" * 70)

    thresholds = np.arange(0.0, 0.81, 0.1)
    base_n = len(trades)
    base_wr = sum(1 for t in trades if t["exit_type"] == "tp") / base_n * 100
    base_expr = expectancy_r([t["pnl_pct"] for t in trades])

    print(f"\n  {'min_conf':>8} | {'Trades':>6} | {'WR%':>6} | {'ExpR':>6} | {'Reduction':>10}")
    print(f"  {'-'*8}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}-+-{'-'*10}")
    sweep = []
    for mc in thresholds:
        filtered = [t for t in trades if t["confidence"] >= mc]
        n = len(filtered)
        wins = sum(1 for t in filtered if t["exit_type"] == "tp")
        wr = wins / n * 100 if n > 0 else 0
        expr = expectancy_r([t["pnl_pct"] for t in filtered])
        red = (1 - n / base_n) * 100
        print(f"  {mc:>8.1f} | {n:>6} | {wr:>5.1f}% | {expr:>+5.2f} | {red:>9.1f}%")
        sweep.append(dict(mc=mc, n=n, wr=wr, expr=expr, red=red))

    # Find sweet spot
    sweet = None
    for s in sweep[1:]:
        wr_imp = s["wr"] - base_wr
        expr_imp = s["expr"] - base_expr
        if wr_imp > 5 and s["red"] < 40 and expr_imp > 0.02:
            if sweet is None or s["expr"] > sweet["expr"]:
                sweet = s
    if sweet:
        print(f"\n  SWEET SPOT: min_conf = {sweet['mc']:.1f}")
        print(f"  WR: {base_wr:.1f}% → {sweet['wr']:.1f}% (+{sweet['wr']-base_wr:.1f}pp)")
        print(f"  ExpR: {base_expr:+.2f} → {sweet['expr']:+.2f} (+{sweet['expr']-base_expr:.2f})")
        print(f"  Trade reduction: -{sweet['red']:.1f}%")
    else:
        print(f"\n  No sweet spot found. Keep min_conf = 0.0")
    return sweep, sweet


# ═══════════════════ FIGURE 1 ══════════════════════════════════════
def figure1(trades, confs, wc, lc, fixed_buckets, quartiles, sweep, sweet, justified):
    _dark_style()
    fig = plt.figure(figsize=(18, 16))
    fig.suptitle("Confidence Score vs Win Rate — Structural rr=2.5 (BTC/USDT 15m 730d)",
                 fontsize=16, fontweight="bold", color=PLT_FG)

    overall_wr = sum(1 for t in trades if t["exit_type"] == "tp") / len(trades) * 100

    # P1: Distribution
    ax1 = fig.add_subplot(3, 2, 1)
    all_c = np.array([t["confidence"] for t in trades])
    win_c = np.array([t["confidence"] for t in trades if t["exit_type"] == "tp"])
    los_c = np.array([t["confidence"] for t in trades if t["exit_type"] == "sl"])
    bins = np.linspace(0.2, 1.0, 20)
    ax1.hist(win_c, bins=bins, alpha=0.55, color="#27ae60", edgecolor=PLT_BG, label=f"Winners ({len(win_c)})")
    ax1.hist(los_c, bins=bins, alpha=0.55, color="#e74c3c", edgecolor=PLT_BG, label=f"Losers ({len(los_c)})")
    ax1.set_xlabel("Confidence"); ax1.set_ylabel("Count")
    ax1.set_title("Confidence Distribution: Winners vs Losers")
    ax1.legend(facecolor=PLT_BG, edgecolor=PLT_FG, fontsize=9)
    ax1.grid(True, alpha=0.3)

    # P2: WR by fixed bucket
    ax2 = fig.add_subplot(3, 2, 2)
    labels = [b["label"] for b in fixed_buckets]
    wrs = [b["wr"] for b in fixed_buckets]
    ns = [b["n"] for b in fixed_buckets]
    colors = ["#27ae60" if w > overall_wr else "#e74c3c" for w in wrs]
    bars = ax2.bar(labels, wrs, color=colors, edgecolor=PLT_FG)
    for b, w, n in zip(bars, wrs, ns):
        lbl = f"{w:.0f}%\nn={n}" if n >= 15 else f"n={n}\nINSUF"
        ax2.text(b.get_x() + b.get_width()/2, w + 1, lbl,
                 ha="center", color=PLT_FG, fontsize=8)
    ax2.axhline(overall_wr, color="#f1c40f", ls="--", alpha=0.7,
                label=f"Overall WR={overall_wr:.1f}%")
    # Error bars (95% CI)
    for i, b in enumerate(fixed_buckets):
        if b["n"] >= 15:
            lo_ci, hi_ci = binom_ci_95(b["wins"], b["n"])
            ax2.plot([i, i], [lo_ci, hi_ci], color=PLT_FG, lw=2, alpha=0.7)
    ax2.set_ylabel("Win Rate %"); ax2.set_title("WR by Confidence Bucket")
    ax2.legend(facecolor=PLT_BG, edgecolor=PLT_FG, fontsize=9)
    ax2.grid(True, alpha=0.3)

    # P3: WR by quartile with CI
    ax3 = fig.add_subplot(3, 2, 3)
    q_labels = [f"Q{q['qi']}\n{q['lo']:.2f}-{q['hi']:.2f}" for q in quartiles]
    q_wrs = [q["wr"] for q in quartiles]
    cmap = plt.cm.RdYlGn
    q_colors = [cmap(0.2 + 0.6 * i / 3) for i in range(4)]
    bars3 = ax3.bar(q_labels, q_wrs, color=q_colors, edgecolor=PLT_FG)
    for i, q in enumerate(quartiles):
        lo_ci, hi_ci = binom_ci_95(q["wins"], q["n"])
        ax3.plot([i, i], [lo_ci, hi_ci], color=PLT_FG, lw=2.5, alpha=0.8)
        ax3.text(i, q["wr"] + 1.5, f"{q['wr']:.1f}%\nn={q['n']}", ha="center",
                 color=PLT_FG, fontsize=9, fontweight="bold")
    ax3.axhline(overall_wr, color="#f1c40f", ls="--", alpha=0.7)
    # Trend line
    ax3.plot(range(4), q_wrs, color="#3498db", lw=2, ls="--", marker="o", markersize=6)
    ax3.set_ylabel("Win Rate %"); ax3.set_title("WR by Confidence Quartile (with 95% CI)")
    ax3.grid(True, alpha=0.3)

    # P4: Factor heatmap (simple bar chart)
    ax4 = fig.add_subplot(3, 2, 4)
    ax4.axis("off")  # Will be drawn as text table
    # handled separately in figure2

    # P5: min_conf sweep
    ax5 = fig.add_subplot(3, 2, 5)
    mcs = [s["mc"] for s in sweep]
    wrvals = [s["wr"] for s in sweep]
    nvals = [s["n"] for s in sweep]
    ax5.plot(mcs, wrvals, color="#3498db", marker="o", lw=2, label="WR%")
    ax5.axhline(overall_wr, color="#f1c40f", ls="--", alpha=0.6, label=f"Baseline WR={overall_wr:.1f}%")
    ax5.set_xlabel("min_conf threshold"); ax5.set_ylabel("WR%", color="#3498db")
    ax5r = ax5.twinx()
    ax5r.bar(mcs, nvals, width=0.07, alpha=0.3, color="#888888")
    ax5r.set_ylabel("Trades remaining", color="#888888")
    if sweet:
        ax5.axvline(sweet["mc"], color="#27ae60", ls="--", lw=2, label=f"Sweet spot={sweet['mc']:.1f}")
    ax5.set_title("WR and Trade Count by min_conf Threshold")
    ax5.legend(facecolor=PLT_BG, edgecolor=PLT_FG, fontsize=9, loc="upper left")
    ax5.grid(True, alpha=0.3)

    # P6: Sizing equity or "not justified"
    ax6 = fig.add_subplot(3, 2, 6)
    if not justified:
        ax6.axis("off")
        ax6.text(0.5, 0.5, "Confidence-weighted sizing\nNOT JUSTIFIED\n\n"
                 f"Q1-Q4 WR delta too small\nor not statistically significant\n\n"
                 f"Keep fixed risk = 1.5% per trade",
                 transform=ax6.transAxes, ha="center", va="center",
                 fontsize=14, color="#e74c3c", fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="#1a1a1a", edgecolor="#e74c3c"))
    else:
        ax6.text(0.5, 0.5, "Confidence-weighted sizing\nJUSTIFIED\n\nSee Scenario B results",
                 transform=ax6.transAxes, ha="center", va="center",
                 fontsize=14, color="#27ae60", fontweight="bold")
    ax6.set_title("Position Sizing Verdict")

    plt.tight_layout()
    path = OUTPUT_DIR / "confidence_wr_analysis_BTCUSDT.png"
    plt.savefig(path, dpi=120); plt.close()
    print(f"\n  saved: {path}")


# ═══════════════════ FIGURE 2 ══════════════════════════════════════
def figure2(trades, df, factor_results):
    _dark_style()
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Factor Contribution Deep Dive — Structural rr=2.5 (BTC/USDT 15m 730d)",
                 fontsize=15, fontweight="bold", color=PLT_FG)

    df_ind = compute_indicators(df, 20, 50, 5)
    atr = df_ind["atr"].values.astype(np.float64)
    adx_arr = df_ind["adx"].values.astype(np.float64)
    ema_f = df_ind["ema_fast"].values.astype(np.float64)
    ema_f_slope = df_ind["ema_fast_slope"].values.astype(np.float64)
    close_arr = df_ind["close"].values.astype(np.float64)
    opn_arr = df_ind["open"].values.astype(np.float64)
    high_arr = df_ind["high"].values.astype(np.float64)
    low_arr = df_ind["low"].values.astype(np.float64)
    macd_hist = df_ind["macd_hist"].values.astype(np.float64)
    volume_arr = df_ind["volume"].values.astype(np.float64)
    vol_sma_arr = df_ind["vol_sma"].values.astype(np.float64)

    factor_defs = {
        "adx_strong": lambda t, i: adx_arr[i] >= 35.0,
        "pullback_tight": lambda t, i: (abs(close_arr[i] - ema_f[i]) / atr[i] if atr[i] > 0 else 999) <= 0.5,
        "macd_increasing": lambda t, i: (macd_hist[i] > 0 and macd_hist[i] > macd_hist[i-1]) if t["direction"] == "LONG" and i > 0 else (macd_hist[i] < 0 and macd_hist[i] < macd_hist[i-1]) if i > 0 else False,
        "body_ratio": lambda t, i: (abs(close_arr[i] - opn_arr[i]) / max(high_arr[i] - low_arr[i], 1e-10)) >= 0.60,
        "ema_slope": lambda t, i: (ema_f_slope[i] > 0 if t["direction"] == "LONG" else ema_f_slope[i] < 0) if not np.isnan(ema_f_slope[i]) else False,
        "volume_high": lambda t, i: (volume_arr[i] / vol_sma_arr[i] >= 1.2) if vol_sma_arr[i] > 0 and not np.isnan(vol_sma_arr[i]) else False,
    }

    factor_x_vals = {
        "adx_strong": lambda t, i: adx_arr[i],
        "pullback_tight": lambda t, i: abs(close_arr[i] - ema_f[i]) / atr[i] if atr[i] > 0 else 0,
        "macd_increasing": lambda t, i: abs(macd_hist[i]),
        "body_ratio": lambda t, i: abs(close_arr[i] - opn_arr[i]) / max(high_arr[i] - low_arr[i], 1e-10),
        "ema_slope": lambda t, i: abs(ema_f_slope[i]) if not np.isnan(ema_f_slope[i]) else 0,
        "volume_high": lambda t, i: volume_arr[i] / vol_sma_arr[i] if vol_sma_arr[i] > 0 and not np.isnan(vol_sma_arr[i]) else 0,
    }

    for idx, (fname, fn) in enumerate(factor_defs.items()):
        ax = axes[idx // 3, idx % 3]
        xs_win = []; xs_loss = []
        xfn = factor_x_vals[fname]
        for t in trades:
            i = t["entry_bar_idx"]
            if i < 0 or i >= len(atr): continue
            xv = xfn(t, i)
            if t["exit_type"] == "tp":
                xs_win.append(xv)
            else:
                xs_loss.append(xv)
        # Jitter pnl for visual (use x=factor_value, y=random jitter for wins/losses)
        if xs_win:
            ax.scatter(xs_win, np.random.uniform(0.5, 1.0, len(xs_win)),
                       color="#27ae60", alpha=0.4, s=12, label=f"Win ({len(xs_win)})")
        if xs_loss:
            ax.scatter(xs_loss, np.random.uniform(0.0, 0.5, len(xs_loss)),
                       color="#e74c3c", alpha=0.4, s=12, label=f"Loss ({len(xs_loss)})")
        r = factor_results[fname]
        ax.set_title(f"{fname}  Δ={r['delta']:+.1f}pp", fontsize=10)
        ax.set_xlabel(fname); ax.set_ylabel("outcome (jittered)")
        ax.legend(facecolor=PLT_BG, edgecolor=PLT_FG, fontsize=7, loc="upper right")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = OUTPUT_DIR / "confidence_factors_BTCUSDT.png"
    plt.savefig(path, dpi=120); plt.close()
    print(f"  saved: {path}")


# ═══════════════════ FINAL VERDICT ═════════════════════════════════
def verdict(trades, confs, wc, lc, delta, quartiles, pattern, sig_results,
            factor_results, justified, sweet):
    avg_conf = np.mean(confs)
    avg_wc = np.mean(wc)
    avg_lc = np.mean(lc)
    wrs = [q["wr"] for q in quartiles]
    q14 = sig_results.get("Q1 vs Q4", {})

    # Top 3 factors
    sorted_factors = sorted(factor_results.items(), key=lambda x: -abs(x[1]["delta"]))
    top3 = sorted_factors[:3]

    print("\n")
    print("  ╔══════════════════════════════════════════════════════════════╗")
    print("  ║     CONFIDENCE SCORE ANALYSIS — BTC/USDT 15m 353 trades     ║")
    print("  ╠══════════════════════════════════════════════════════════════╣")
    print("  ║ DISTRIBUTION                                                  ║")
    print(f"  ║  Mean confidence:      {avg_conf:.3f}{' '*(36-len(f'{avg_conf:.3f}'))}║")
    print(f"  ║  Winners avg conf:     {avg_wc:.3f}{' '*(36-len(f'{avg_wc:.3f}'))}║")
    print(f"  ║  Losers avg conf:      {avg_lc:.3f}{' '*(36-len(f'{avg_lc:.3f}'))}║")
    sig_txt = "signal" if abs(delta) > 0.05 else ("weak" if abs(delta) > 0.02 else "none")
    print(f"  ║  Delta (W-L):          {delta:+.4f}  ({sig_txt}){' '*(25-len(f'{delta:+.4f}  ({sig_txt})'))}║")
    print("  ╠══════════════════════════════════════════════════════════════╣")
    print("  ║ WIN RATE BY QUARTILE                                          ║")
    print(f"  ║  Q1: {wrs[0]:>4.1f}%  | Q2: {wrs[1]:>4.1f}%  | Q3: {wrs[2]:>4.1f}%  | Q4: {wrs[3]:>4.1f}%   ║")
    print(f"  ║  Pattern:    {pattern:<48}║")
    sig_label = "SIGNIFICANT" if q14.get("sig", False) else "NOT SIGNIFICANT"
    print(f"  ║  Q1 vs Q4:   p={q14.get('p', 1):.4f}  {sig_label}{' '*(28-len(sig_label)-len(f'{q14.get(chr(112),1):.4f}'))}║")
    print("  ╠══════════════════════════════════════════════════════════════╣")
    print("  ║ STRONGEST FACTORS (by WR delta)                              ║")
    for rank, (fn, fr) in enumerate(top3, 1):
        line = f"  ║  {rank}. {fn}: {fr['delta']:+.1f}pp WR delta"
        print(f"{line}{' '*(63-len(line))}║")
    print("  ╠══════════════════════════════════════════════════════════════╣")
    print("  ║ OPTIMAL min_conf THRESHOLD                                   ║")
    if sweet:
        base_wr = sum(1 for t in trades if t["exit_type"] == "tp") / len(trades) * 100
        print(f"  ║  Recommended: {sweet['mc']:.1f}{' '*(46-len(f'{sweet[chr(109)+chr(99)]:.1f}'))}║")
        print(f"  ║  Expected WR improvement: +{sweet['wr']-base_wr:.1f}pp{' '*(32-len(f'+{sweet[chr(119)+chr(114)]-base_wr:.1f}pp'))}║")
        print(f"  ║  Trade reduction: -{sweet['red']:.1f}%{' '*(38-len(f'-{sweet[chr(114)+chr(101)+chr(100)]:.1f}%'))}║")
    else:
        print("  ║  Recommended: 0.0 — no filter adds value                   ║")
    print("  ╠══════════════════════════════════════════════════════════════╣")
    print("  ║ POSITION SIZING RECOMMENDATION                               ║")
    if justified:
        print("  ║  Confidence-weighted sizing: JUSTIFIED                       ║")
        print("  ║  Q1: 0.75% | Q2: 1.25% | Q3: 1.50% | Q4: 2.00%            ║")
    else:
        print("  ║  Confidence-weighted sizing: NOT JUSTIFIED                   ║")
        reason = "WR difference not significant or too small for 353 trades"
        print(f"  ║  Reason: {reason[:51]:<51}║")
        print("  ║  Keep fixed risk_pct = 1.5% per trade                       ║")
        if sweet:
            print(f"  ║  min_conf recommendation: {sweet['mc']:.1f}{' '*(33-len(f'{sweet[chr(109)+chr(99)]:.1f}'))}║")
        else:
            print("  ║  min_conf recommendation: 0.0                               ║")
    print("  ╠══════════════════════════════════════════════════════════════╣")
    print("  ║ ACTIONS FOR run_simple_paper.py                              ║")
    if sweet:
        print(f"  ║  1. Set min_confidence = {sweet['mc']:.1f} (entry filter){' '*(25-len(f'{sweet[chr(109)+chr(99)]:.1f}'))}║")
    else:
        print("  ║  1. No change needed (keep min_confidence = 0.0)            ║")
    if justified:
        print("  ║  2. Implement Q-based risk scaling in position sizing        ║")
    else:
        print("  ║  2. No change needed (keep fixed risk = 1.5%)               ║")
    print("  ╚══════════════════════════════════════════════════════════════╝")


# ═══════════════════ MAIN ══════════════════════════════════════════
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 70)
    print("CONFIDENCE SCORE ANALYSIS — BTC/USDT 15m 730d")
    print("Structural rr=2.5, HTF=ON, min_conf=0.0")
    print("=" * 70)

    df = load_candles("BTC/USDT", "15m", days=730)
    htf = compute_htf_bias(df, htf_ema_period=50)
    print(f"  bars: {len(df)}")

    trades = step1(df, htf)
    confs, wc, lc, delta = step2(trades)
    fixed_buckets, quartiles, pattern = step3(trades, confs)
    sig_results = step4(quartiles)
    step5(trades)
    factor_results = step6(trades, df)
    justified, sizing = step7(trades, quartiles, sig_results)
    sweep, sweet = step8(trades, df, htf)

    print("\n[Figure 1]")
    figure1(trades, confs, wc, lc, fixed_buckets, quartiles, sweep, sweet, justified)
    print("[Figure 2]")
    figure2(trades, df, factor_results)

    verdict(trades, confs, wc, lc, delta, quartiles, pattern, sig_results,
            factor_results, justified, sweet)


if __name__ == "__main__":
    main()
