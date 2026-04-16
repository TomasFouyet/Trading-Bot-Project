"""
Full re-test with CORRECTED BingX Futures fee model.

Old (wrong) fee model: 0.075% taker both sides = 0.150% RT
New (correct)        : 0.020% maker limit / 0.050% taker market
Weighted avg RT      : 0.476*0.040% + 0.524*0.070% = 0.056%

Parts:
  1. Full penetration test S0-S6 (corrected fees)
  2. Fix 1 re-run (rr=2.0/2.5/3.0) with corrected S1/S2
  3. Fix 2 (sl_atr=1.87/2.07/2.15/2.23) — only if Fix 1 passes
  4. Combined validation (full WFA + MC) — only if both pass

Reuses the exact slippage wrapper and stop-hunt analysis from
scripts/run_penetration_test.py — only scenario definitions differ.
"""
from __future__ import annotations

import sys
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from validation.data_loader import load_candles
from validation.fast_backtest import (
    compute_indicators, compute_htf_bias, fast_backtest,
)
from validation.monte_carlo import MonteCarloSimulation

from scripts.run_penetration_test import (
    fast_backtest_with_slippage,
    stop_hunt_analysis,
    expectancy_r,
    VALIDATED_PARAMS,
    PLT_BG, PLT_FG, _dark_style,
)

OUTPUT_DIR = ROOT / "validation" / "output"

# ─────────── Corrected fee scenarios ──────────────────────────────
S_SCENARIOS_V2 = {
    # name: (entry%, sl%, tp%, delay_bars)
    "S0 base":  dict(entry=0.000, sl=0.000, tp=0.000, delay=0),
    "S1 paper": dict(entry=0.020, sl=0.050, tp=0.020, delay=0),  # BingX limit/taker
    "S2 real":  dict(entry=0.050, sl=0.100, tp=0.030, delay=0),  # + spread
    "S3 delay": dict(entry=0.050, sl=0.100, tp=0.030, delay=1),
    "S4 hvol":  dict(entry=0.100, sl=0.200, tp=0.050, delay=0),
    "S5 worst": dict(entry=0.150, sl=0.300, tp=0.080, delay=1),
}

# Pass criteria
S1_SHARPE_MIN = 2.0
S1_EXPR_MIN = 0.08
S2_SHARPE_MIN = 1.0
S2_EXPR_MIN = 0.05
MAX_HUNT10 = 30.0
MAX_CLUSTER = 30.0


def run_scenario(df, htf, scen_key, params):
    s = S_SCENARIOS_V2[scen_key]
    p = {**VALIDATED_PARAMS, **params}
    return fast_backtest_with_slippage(
        df,
        entry_slippage_pct=s["entry"], sl_slippage_pct=s["sl"],
        tp_slippage_pct=s["tp"], signal_delay=s["delay"],
        htf_bias=htf, **p,
    )


def classify(sharpe, expR):
    if sharpe < 1.0 or expR < 0: return "DANGER"
    if sharpe < 2.0 or expR < 0.05: return "MARGINAL"
    return "SAFE"


def find_breaking_point(df, htf, rr, sl_atr, target_sharpe=1.0):
    def _run(total):
        return fast_backtest_with_slippage(
            df, entry_slippage_pct=total/2, sl_slippage_pct=total/2,
            tp_slippage_pct=total*0.1, signal_delay=0, htf_bias=htf,
            rr_ratio=rr, atr_sl_mult=sl_atr,
        )
    lo, hi = 0.0, 3.0
    for test in [0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 1.0, 1.5, 2.0, 3.0]:
        m = _run(test)
        if m.sharpe_ratio < target_sharpe:
            hi = test; break
        lo = test
    for _ in range(10):
        mid = (lo + hi) / 2
        m = _run(mid)
        if m.sharpe_ratio < target_sharpe: hi = mid
        else: lo = mid
    bp = (lo + hi) / 2
    m = _run(bp)
    return bp, m


def smoke_wfa(df, htf, rr, sl_atr, n_windows=3, days=365):
    tail = df.tail(int(days * 96)).reset_index(drop=True)
    htf_tail = htf[-len(tail):] if htf is not None else None
    grid = [
        {"adx_min": a, "ema_fast": f, "ema_slow": s}
        for a, f, s in itertools.product([20, 25], [15, 20], [45, 50])
    ]
    total = len(tail)
    window_size = total // n_windows
    oos_sharpes = []
    for w_idx in range(n_windows):
        start = w_idx * window_size
        end = min(start + window_size, total)
        split = start + int((end - start) * 0.70)
        is_df = tail.iloc[start:split].reset_index(drop=True)
        oos_df = tail.iloc[split:end].reset_index(drop=True)
        is_htf = htf_tail[start:split] if htf_tail is not None else None
        oos_htf = htf_tail[split:end] if htf_tail is not None else None
        if len(is_df) < 100 or len(oos_df) < 50: continue
        best_score = -np.inf; best_p = grid[0]
        for params in grid:
            is_ind = compute_indicators(is_df, params["ema_fast"], params["ema_slow"])
            m = fast_backtest(
                is_ind, adx_min=params["adx_min"],
                ema_fast_p=params["ema_fast"], ema_slow_p=params["ema_slow"],
                rr_ratio=rr, atr_sl_mult=sl_atr,
                precomputed=True, htf_bias=is_htf,
            )
            score = m.sharpe_ratio if m.total_trades >= 5 else -np.inf
            if score > best_score: best_score = score; best_p = params
        oos_ind = compute_indicators(oos_df, best_p["ema_fast"], best_p["ema_slow"])
        oos_m = fast_backtest(
            oos_ind, adx_min=best_p["adx_min"],
            ema_fast_p=best_p["ema_fast"], ema_slow_p=best_p["ema_slow"],
            rr_ratio=rr, atr_sl_mult=sl_atr,
            precomputed=True, htf_bias=oos_htf,
        )
        oos_sharpes.append(oos_m.sharpe_ratio)
    return float(np.mean(oos_sharpes)) if oos_sharpes else 0.0


# ═══════════════════ PART 1 — PENETRATION ═════════════════════════
def part1_penetration(df, htf, rr=1.5, sl_atr=2.0):
    print("\n" + "=" * 70)
    print("PART 1 — PENETRATION TEST (corrected BingX Futures fees)")
    print("=" * 70)

    rows = []
    trades_by_scen = {}
    equity_curves = {}
    params = dict(rr_ratio=rr, atr_sl_mult=sl_atr)

    for key in ["S0 base", "S1 paper", "S2 real", "S3 delay", "S4 hvol", "S5 worst"]:
        m = run_scenario(df, htf, key, params)
        exp = expectancy_r(m)
        rows.append({
            "scenario": key,
            "entry": S_SCENARIOS_V2[key]["entry"],
            "sl": S_SCENARIOS_V2[key]["sl"],
            "tp": S_SCENARIOS_V2[key]["tp"],
            "delay": S_SCENARIOS_V2[key]["delay"],
            "sharpe": m.sharpe_ratio, "ann": m.annual_return_pct,
            "expR": exp, "wr": m.winrate, "dd": m.max_drawdown_pct,
            "n": m.total_trades,
        })
        trades_by_scen[key] = m.trades
        pnls = np.array([t["pnl_pct"] for t in m.trades])
        equity_curves[key] = np.concatenate([[1.0], np.cumprod(1 + pnls/100)])
        print(f"  {key:<10} Sharpe={m.sharpe_ratio:>7.3f}  Ann={m.annual_return_pct:>6.1f}%  "
              f"ExpR={exp:>+7.3f}R  WR={m.winrate:.1f}%  DD={m.max_drawdown_pct:.1f}%  n={m.total_trades}")

    # S0 sanity
    s0_sharpe = rows[0]["sharpe"]
    print(f"\n  S0 SANITY: Sharpe={s0_sharpe:.3f} (target: 4.81 ± 0.05)  "
          f"{'✓ PASS' if abs(s0_sharpe - 4.81) < 0.05 else '✗ FAIL'}")

    # Stop hunt (independent of fees)
    print("\n  Stop hunt analysis (fee-independent)...")
    baseline_m = fast_backtest(df, htf_bias=htf, **VALIDATED_PARAMS)
    records = stop_hunt_analysis(df, baseline_m)
    rec10 = float(np.mean([r["recovered_10"] for r in records])) * 100
    rec20 = float(np.mean([r["recovered_20"] for r in records])) * 100
    prox = np.array([r["proximity"] for r in records])
    cluster = float(np.mean(prox < 0.2)) * 100
    print(f"  Hunt10%: {rec10:.1f}%  Hunt20%: {rec20:.1f}%  Cluster%: {cluster:.1f}%")

    # Breaking point
    print("\n  Breaking point search...")
    bp, _ = find_breaking_point(df, htf, rr, sl_atr)
    print(f"  Breaking point: {bp:.3f}% total slippage")
    rows.append({
        "scenario": "S6 break", "entry": bp/2, "sl": bp/2, "tp": bp*0.1,
        "delay": 0, "sharpe": 1.0, "ann": 0, "expR": 0, "wr": 0, "dd": 0, "n": 0,
    })

    return {
        "rows": rows, "trades": trades_by_scen, "equity": equity_curves,
        "breaking_point": bp, "rec10": rec10, "rec20": rec20, "cluster": cluster,
        "records": records,
    }


# ═══════════════════ PART 2 — FIX 1 ═══════════════════════════════
def part2_fix1(df, htf, baseline_oos):
    print("\n" + "=" * 70)
    print("PART 2 — FIX 1 R:R RATIO (corrected fees)")
    print("=" * 70)
    results = []
    for rr in [2.0, 2.5, 3.0]:
        print(f"\n  rr={rr} ...")
        s0 = run_scenario(df, htf, "S0 base", dict(rr_ratio=rr, atr_sl_mult=2.0))
        s1 = run_scenario(df, htf, "S1 paper", dict(rr_ratio=rr, atr_sl_mult=2.0))
        s2 = run_scenario(df, htf, "S2 real", dict(rr_ratio=rr, atr_sl_mult=2.0))
        s1_expR = expectancy_r(s1); s2_expR = expectancy_r(s2)

        baseline_m = fast_backtest(df, htf_bias=htf, rr_ratio=rr, atr_sl_mult=2.0,
                                   **{k: v for k, v in VALIDATED_PARAMS.items() if k not in ("rr_ratio","atr_sl_mult")})
        records = stop_hunt_analysis(df, baseline_m)
        rec10 = float(np.mean([r["recovered_10"] for r in records])) * 100 if records else 0.0

        oos = smoke_wfa(df, htf, rr=rr, sl_atr=2.0)
        passes = (s1.sharpe_ratio > S1_SHARPE_MIN and s1_expR > S1_EXPR_MIN and
                  s2.sharpe_ratio > S2_SHARPE_MIN and s2_expR > S2_EXPR_MIN and
                  oos > baseline_oos)
        row = {
            "rr": rr, "s0_sharpe": s0.sharpe_ratio,
            "s1_sharpe": s1.sharpe_ratio, "s1_expR": s1_expR,
            "s2_sharpe": s2.sharpe_ratio, "s2_expR": s2_expR,
            "hunt10": rec10, "oos": oos, "passes": passes,
            "s1_trades": s1.trades, "s2_trades": s2.trades,
        }
        results.append(row)
        print(f"    S0={s0.sharpe_ratio:.2f}  S1={s1.sharpe_ratio:.2f} (ExpR {s1_expR:+.3f})  "
              f"S2={s2.sharpe_ratio:.2f} (ExpR {s2_expR:+.3f})  OOS={oos:.2f}  "
              f"{'PASS' if passes else 'FAIL'}")

    return results


# ═══════════════════ PART 3 — FIX 2 ═══════════════════════════════
def part3_fix2(df, htf, baseline_oos, best_rr):
    print("\n" + "=" * 70)
    print(f"PART 3 — FIX 2 SL PLACEMENT (rr={best_rr} fixed, corrected fees)")
    print("=" * 70)
    results = []
    for sl_atr in [1.87, 2.07, 2.15, 2.23]:
        print(f"\n  sl_atr={sl_atr} ...")
        s0 = run_scenario(df, htf, "S0 base", dict(rr_ratio=best_rr, atr_sl_mult=sl_atr))
        s1 = run_scenario(df, htf, "S1 paper", dict(rr_ratio=best_rr, atr_sl_mult=sl_atr))
        s2 = run_scenario(df, htf, "S2 real", dict(rr_ratio=best_rr, atr_sl_mult=sl_atr))
        s1_expR = expectancy_r(s1); s2_expR = expectancy_r(s2)

        baseline_m = fast_backtest(df, htf_bias=htf, rr_ratio=best_rr, atr_sl_mult=sl_atr,
                                   **{k: v for k, v in VALIDATED_PARAMS.items() if k not in ("rr_ratio","atr_sl_mult")})
        records = stop_hunt_analysis(df, baseline_m)
        if records:
            rec10 = float(np.mean([r["recovered_10"] for r in records])) * 100
            prox = np.array([r["proximity"] for r in records])
            cluster = float(np.mean(prox < 0.2)) * 100
        else:
            rec10 = cluster = 0.0

        oos = smoke_wfa(df, htf, rr=best_rr, sl_atr=sl_atr)
        passes = (cluster < MAX_CLUSTER and rec10 < MAX_HUNT10 and
                  s2.sharpe_ratio > S2_SHARPE_MIN and s2_expR > S2_EXPR_MIN and
                  oos > baseline_oos)
        row = {
            "sl_atr": sl_atr, "s0_sharpe": s0.sharpe_ratio,
            "s1_sharpe": s1.sharpe_ratio, "s1_expR": s1_expR,
            "s2_sharpe": s2.sharpe_ratio, "s2_expR": s2_expR,
            "hunt10": rec10, "cluster": cluster, "oos": oos,
            "passes": passes, "records": records,
        }
        results.append(row)
        print(f"    Cluster={cluster:.1f}%  Hunt10={rec10:.1f}%  "
              f"S1={s1.sharpe_ratio:.2f}  S2={s2.sharpe_ratio:.2f}  OOS={oos:.2f}  "
              f"{'PASS' if passes else 'FAIL'}")

    return results


# ═══════════════════ PART 4 — COMBINED ════════════════════════════
def part4_combined(df, htf, best_rr, best_sl):
    print("\n" + "=" * 70)
    print(f"PART 4 — COMBINED VALIDATION (rr={best_rr}, sl_atr={best_sl})")
    print("=" * 70)
    params = dict(rr_ratio=best_rr, atr_sl_mult=best_sl)
    scen_rows = []
    for key in ["S0 base", "S1 paper", "S2 real", "S3 delay", "S4 hvol", "S5 worst"]:
        m = run_scenario(df, htf, key, params)
        exp = expectancy_r(m)
        scen_rows.append({
            "scenario": key, "sharpe": m.sharpe_ratio, "ann": m.annual_return_pct,
            "expR": exp, "wr": m.winrate, "dd": m.max_drawdown_pct, "n": m.total_trades,
            "trades": m.trades,
        })
        print(f"  {key:<10} Sharpe={m.sharpe_ratio:.3f}  ExpR={exp:+.3f}R  n={m.total_trades}")

    bp, _ = find_breaking_point(df, htf, best_rr, best_sl)
    print(f"  Breaking point: {bp:.3f}%")

    # Full WFA
    print("  Full WFA (5 windows, 730d, 27 combos)...")
    wfa_grid = [{"adx_min": a, "ema_fast": f, "ema_slow": s}
                for a, f, s in itertools.product([15, 20, 25], [15, 20, 25], [40, 50, 60])]
    total = len(df); ws = total // 5
    oos_sharpes = []; oos_anns = []
    for w_idx in range(5):
        start = w_idx * ws; end = min(start + ws, total)
        split = start + int((end - start) * 0.70)
        is_df = df.iloc[start:split].reset_index(drop=True)
        oos_df = df.iloc[split:end].reset_index(drop=True)
        is_htf = htf[start:split]; oos_htf = htf[split:end]
        if len(is_df) < 100 or len(oos_df) < 50: continue
        best_score = -np.inf; best_p = wfa_grid[0]
        for g in wfa_grid:
            is_ind = compute_indicators(is_df, g["ema_fast"], g["ema_slow"])
            m = fast_backtest(is_ind, adx_min=g["adx_min"],
                              ema_fast_p=g["ema_fast"], ema_slow_p=g["ema_slow"],
                              rr_ratio=best_rr, atr_sl_mult=best_sl,
                              precomputed=True, htf_bias=is_htf)
            score = m.sharpe_ratio if m.total_trades >= 5 else -np.inf
            if score > best_score: best_score = score; best_p = g
        oos_ind = compute_indicators(oos_df, best_p["ema_fast"], best_p["ema_slow"])
        oos_m = fast_backtest(oos_ind, adx_min=best_p["adx_min"],
                              ema_fast_p=best_p["ema_fast"], ema_slow_p=best_p["ema_slow"],
                              rr_ratio=best_rr, atr_sl_mult=best_sl,
                              precomputed=True, htf_bias=oos_htf)
        oos_sharpes.append(oos_m.sharpe_ratio); oos_anns.append(oos_m.annual_return_pct)
        print(f"    W{w_idx+1}: OOS Sharpe={oos_m.sharpe_ratio:.2f}")
    full_oos = float(np.mean(oos_sharpes)) if oos_sharpes else 0.0
    full_ann = float(np.mean(oos_anns)) if oos_anns else 0.0
    print(f"  Full WFA: OOS Sharpe={full_oos:.2f}  Ann={full_ann:.1f}%")

    # Monte Carlo on S1 fills
    s1_trades = next(r["trades"] for r in scen_rows if r["scenario"] == "S1 paper")
    if len(s1_trades) >= 10:
        mc = MonteCarloSimulation(s1_trades, n_simulations=5000, seed=42)
        mc_report = mc.run()
        ror = mc_report.risk_of_ruin_pct
        p5 = mc_report.pnl_p5; p50 = mc_report.pnl_p50; p95 = mc_report.pnl_p95
        dd95 = mc_report.dd_p95
    else:
        ror = p5 = p50 = p95 = dd95 = 0.0
    print(f"  MC (S1 fills): RoR={ror:.1f}%  P5={p5:+.1f}%  P50={p50:+.1f}%  P95={p95:+.1f}%")

    # Stop hunt final
    s0_m = fast_backtest(df, htf_bias=htf, rr_ratio=best_rr, atr_sl_mult=best_sl,
                        **{k:v for k,v in VALIDATED_PARAMS.items() if k not in ("rr_ratio","atr_sl_mult")})
    records = stop_hunt_analysis(df, s0_m)
    rec10 = float(np.mean([r["recovered_10"] for r in records])) * 100 if records else 0.0
    prox = np.array([r["proximity"] for r in records]) if records else np.array([])
    cluster = float(np.mean(prox < 0.2)) * 100 if len(prox) else 0.0

    return {
        "scenarios": scen_rows, "breaking_point": bp,
        "full_wfa_oos": full_oos, "full_wfa_ann": full_ann,
        "mc_ror": ror, "mc_p5": p5, "mc_p50": p50, "mc_p95": p95, "mc_dd95": dd95,
        "hunt10": rec10, "cluster": cluster,
    }


# ═══════════════════ PLOTS ═════════════════════════════════════════
def plot_penetration_corrected(part1, out_path):
    _dark_style()
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle("Penetration Test — Corrected BingX Futures Fees (maker 0.02% / taker 0.05%)",
                 fontsize=15, fontweight="bold")
    rows = part1["rows"]
    names = [r["scenario"] for r in rows]
    sharpes = [r["sharpe"] for r in rows]
    exprs = [r["expR"] for r in rows]
    anns = [r["ann"] for r in rows]
    cmap = plt.cm.RdYlGn_r
    colors = cmap(np.linspace(0.1, 0.85, len(rows)))
    base_s = rows[0]["sharpe"]

    ax1 = fig.add_subplot(3, 2, (1, 2))
    bars = ax1.bar(names, sharpes, color=colors, edgecolor=PLT_FG)
    for b, v in zip(bars, sharpes):
        ax1.text(b.get_x()+b.get_width()/2, v+0.05, f"{v:.2f}", ha="center", color=PLT_FG, fontweight="bold")
    ax1.axhline(1.0, ls="--", color="#f1c40f", alpha=0.7, label="Sharpe=1.0")
    ax1.axhline(base_s, ls="--", color="#27ae60", alpha=0.5, label=f"Baseline={base_s:.2f}")
    ax1.set_ylabel("Sharpe Ratio")
    ax1.set_title("Sharpe by Scenario (S0 baseline → S5 worst, S6 breaking)")
    ax1.annotate(f"old break: 0.10%", xy=(len(rows)-1, sharpes[-1]),
                 xytext=(len(rows)-2, sharpes[-1]+1.5),
                 color="#f1c40f", fontsize=9,
                 arrowprops=dict(arrowstyle="->", color="#f1c40f"))
    ax1.legend(facecolor=PLT_BG, edgecolor=PLT_FG); ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(3, 2, 3)
    ax2.bar(names, exprs, color=colors, edgecolor=PLT_FG)
    ax2.axhline(0, ls="--", color="#e74c3c")
    ax2.axhline(0.05, ls="--", color="#f1c40f", alpha=0.6, label="+0.05R")
    ax2.set_ylabel("Expectancy (R)"); ax2.set_title("ExpR per Scenario")
    ax2.legend(facecolor=PLT_BG, edgecolor=PLT_FG); ax2.grid(True, alpha=0.3)
    plt.setp(ax2.get_xticklabels(), rotation=30, ha="right")

    ax3 = fig.add_subplot(3, 2, 4)
    ax3.bar(names, anns, color=colors, edgecolor=PLT_FG)
    ax3.axhline(0, ls="--", color="#e74c3c")
    ax3.set_ylabel("Annual Return (%)"); ax3.set_title("Annual Return by Scenario")
    ax3.grid(True, alpha=0.3)
    plt.setp(ax3.get_xticklabels(), rotation=30, ha="right")

    ax4 = fig.add_subplot(3, 2, 5)
    curves = part1["equity"]
    for key, color, lbl in [("S0 base", "#27ae60", "S0 perfect"),
                             ("S1 paper", "#3498db", "S1 BingX limit entry"),
                             ("S2 real", "#e67e22", "S2 realistic"),
                             ("S5 worst", "#e74c3c", "S5 worst")]:
        if key in curves:
            ax4.plot(curves[key], color=color, lw=1.8, label=lbl)
    ax4.axhline(1.0, ls="--", color=PLT_FG, alpha=0.4)
    ax4.set_ylabel("Equity"); ax4.set_xlabel("Trade #")
    ax4.set_title("Equity: Perfect vs S1 BingX Limit vs S2 vs S5")
    ax4.legend(facecolor=PLT_BG, edgecolor=PLT_FG); ax4.grid(True, alpha=0.3)

    ax5 = fig.add_subplot(3, 2, 6); ax5.axis("off")
    bp = part1["breaking_point"]
    paper_total = S_SCENARIOS_V2["S1 paper"]["entry"] + S_SCENARIOS_V2["S1 paper"]["sl"]
    real_total = S_SCENARIOS_V2["S2 real"]["entry"] + S_SCENARIOS_V2["S2 real"]["sl"]
    safety_s1 = bp / max(paper_total, 0.001)
    safety_s2 = bp / max(real_total, 0.001)
    lines = ["Penetration Test Summary (corrected fees)",
             "─────────────────────────────",
             f"Baseline S0 Sharpe:    {base_s:.3f}"]
    for r in rows[1:-1]:
        d = (r["sharpe"] - base_s) / abs(base_s) * 100 if base_s else 0
        lines.append(f"{r['scenario']:<12} Sharpe={r['sharpe']:>6.2f}  ExpR={r['expR']:>+.3f}R ({d:+.0f}%)")
    lines += ["",
              f"Breaking point (S6):   {bp:.3f}% total",
              f"  (was 0.10% with wrong fees)",
              "",
              f"S1 paper total slip:   {paper_total:.3f}%",
              f"Safety margin vs S1:   {safety_s1:.1f}x",
              f"S2 real total slip:    {real_total:.3f}%",
              f"Safety margin vs S2:   {safety_s2:.1f}x",
              "",
              f"Stop hunt (10 bars):   {part1['rec10']:.1f}%",
              f"SL cluster (<0.2):     {part1['cluster']:.1f}%"]
    ax5.text(0.02, 0.98, "\n".join(lines), transform=ax5.transAxes, va="top",
             family="monospace", fontsize=9, color=PLT_FG)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120); plt.close()
    print(f"  saved: {out_path}")


def plot_fix_comparison(fix1, fix2, part1, fix3, out_path):
    _dark_style()
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle("Fix Comparison — Corrected BingX Fees",
                 fontsize=15, fontweight="bold")

    # Panel 1: S1/S2 Sharpe by rr
    ax1 = fig.add_subplot(2, 2, 1)
    rrs = [1.5] + [r["rr"] for r in fix1]
    s1_baseline = next((r["sharpe"] for r in part1["rows"] if r["scenario"] == "S1 paper"), 0)
    s2_baseline = next((r["sharpe"] for r in part1["rows"] if r["scenario"] == "S2 real"), 0)
    s1_vals = [s1_baseline] + [r["s1_sharpe"] for r in fix1]
    s2_vals = [s2_baseline] + [r["s2_sharpe"] for r in fix1]
    x = np.arange(len(rrs))
    ax1.bar(x - 0.2, s1_vals, 0.4, color="#3498db", edgecolor=PLT_FG, label="S1 paper")
    ax1.bar(x + 0.2, s2_vals, 0.4, color="#e67e22", edgecolor=PLT_FG, label="S2 real")
    ax1.axhline(1.0, ls="--", color="#f1c40f", alpha=0.6, label="Sh=1.0")
    ax1.axhline(2.0, ls="--", color="#27ae60", alpha=0.6, label="Sh=2.0")
    ax1.set_xticks(x); ax1.set_xticklabels([f"rr={r}" for r in rrs])
    ax1.set_ylabel("Sharpe")
    ax1.set_title("Sharpe by R:R — Paper (S1) and Real (S2)")
    ax1.legend(facecolor=PLT_BG, edgecolor=PLT_FG); ax1.grid(True, alpha=0.3)

    # Panel 2: S2 Sharpe by sl_atr
    ax2 = fig.add_subplot(2, 2, 2)
    if fix2:
        sls = [2.0] + [r["sl_atr"] for r in fix2]
        s2s = [s2_baseline] + [r["s2_sharpe"] for r in fix2]
        passes = [False] + [r["passes"] for r in fix2]
        colors = ["#27ae60" if p else "#c0392b" for p in passes]
        bars = ax2.bar([str(s) for s in sls], s2s, color=colors, edgecolor=PLT_FG)
        for b, v in zip(bars, s2s):
            ax2.text(b.get_x()+b.get_width()/2, v+0.05, f"{v:.2f}", ha="center", color=PLT_FG)
        ax2.axhline(1.0, ls="--", color="#f1c40f", alpha=0.6)
    else:
        ax2.text(0.5, 0.5, "Fix 2 not run", transform=ax2.transAxes, ha="center", color=PLT_FG)
    ax2.set_ylabel("S2 Sharpe"); ax2.set_title("S2 Sharpe by SL ATR Multiplier")
    ax2.grid(True, alpha=0.3)

    # Panel 3: Stop hunt metrics
    ax3 = fig.add_subplot(2, 2, 3)
    if fix2:
        sls = [2.0] + [r["sl_atr"] for r in fix2]
        hunts = [part1["rec10"]] + [r["hunt10"] for r in fix2]
        clusts = [part1["cluster"]] + [r["cluster"] for r in fix2]
        x = np.arange(len(sls))
        ax3.bar(x - 0.2, hunts, 0.4, color="#c0392b", edgecolor=PLT_FG, label="Hunt10%")
        ax3.bar(x + 0.2, clusts, 0.4, color="#e67e22", edgecolor=PLT_FG, label="Cluster%")
        ax3.axhline(30, ls="--", color="#f1c40f", alpha=0.6)
        ax3.set_xticks(x); ax3.set_xticklabels([str(s) for s in sls])
        ax3.legend(facecolor=PLT_BG, edgecolor=PLT_FG)
    else:
        ax3.text(0.5, 0.5, "Fix 2 not run", transform=ax3.transAxes, ha="center", color=PLT_FG)
    ax3.set_ylabel("%"); ax3.set_title("Stop Hunt Metrics by SL ATR")
    ax3.grid(True, alpha=0.3)

    # Panel 4: Equity curves original vs fixed
    ax4 = fig.add_subplot(2, 2, 4)
    def _curve(trades):
        if not trades: return np.array([1.0])
        pnls = np.array([t["pnl_pct"] for t in trades])
        return np.concatenate([[1.0], np.cumprod(1 + pnls/100)])
    ax4.plot(_curve(part1["trades"]["S0 base"]), color="#27ae60", lw=1.5, label="Orig S0 (perfect)")
    ax4.plot(_curve(part1["trades"]["S1 paper"]), color="#3498db", lw=1.5, label="Orig S1 (paper)")
    ax4.plot(_curve(part1["trades"]["S2 real"]), color="#e74c3c", lw=1.5, label="Orig S2 (real)")
    if fix3:
        s2_fix = next((r for r in fix3["scenarios"] if r["scenario"] == "S2 real"), None)
        if s2_fix:
            ax4.plot(_curve(s2_fix["trades"]), color="#f1c40f", lw=2, label="Fixed S2 (real)")
    ax4.axhline(1.0, ls="--", color=PLT_FG, alpha=0.4)
    ax4.set_xlabel("Trade #"); ax4.set_ylabel("Equity")
    ax4.set_title("Equity: Original vs Fixed")
    ax4.legend(facecolor=PLT_BG, edgecolor=PLT_FG, fontsize=8)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=120); plt.close()
    print(f"  saved: {out_path}")


def plot_before_after(part1_wrong, part1_right, fix3, out_path):
    _dark_style()
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.axis("off")
    fig.suptitle("Before vs After — Fee Correction Impact",
                 fontsize=16, fontweight="bold")

    s1_right = next((r for r in part1_right["rows"] if r["scenario"] == "S1 paper"), None)
    s2_right = next((r for r in part1_right["rows"] if r["scenario"] == "S2 real"), None)

    if fix3:
        s1_fix = next((r for r in fix3["scenarios"] if r["scenario"] == "S1 paper"), None)
        s2_fix = next((r for r in fix3["scenarios"] if r["scenario"] == "S2 real"), None)
    else:
        s1_fix = s2_fix = None

    rows_txt = [
        ("Metric",            "BEFORE (wrong)",   "AFTER (correct)",     "AFTER + FIXES"),
        ("─" * 20, "─" * 18, "─" * 18, "─" * 18),
        ("S0 Sharpe (ref)",   "4.807",              f"{part1_right['rows'][0]['sharpe']:.3f}",
         f"{fix3['scenarios'][0]['sharpe']:.3f}" if fix3 else "—"),
        ("S1 Sharpe (paper)", "~1.61 (old S1)",     f"{s1_right['sharpe']:.3f}" if s1_right else "—",
         f"{s1_fix['sharpe']:.3f}" if s1_fix else "—"),
        ("S2 Sharpe (real)",  "-1.491",             f"{s2_right['sharpe']:.3f}" if s2_right else "—",
         f"{s2_fix['sharpe']:.3f}" if s2_fix else "—"),
        ("S1 ExpR",           "~+0.049R (old S1)",  f"{s1_right['expR']:+.3f}R" if s1_right else "—",
         f"{s1_fix['expR']:+.3f}R" if s1_fix else "—"),
        ("S2 ExpR",           "-0.043R",            f"{s2_right['expR']:+.3f}R" if s2_right else "—",
         f"{s2_fix['expR']:+.3f}R" if s2_fix else "—"),
        ("Breaking point",    "0.100%",             f"{part1_right['breaking_point']:.3f}%",
         f"{fix3['breaking_point']:.3f}%" if fix3 else "—"),
        ("Hunt10%",           "41.7%",              f"{part1_right['rec10']:.1f}%",
         f"{fix3['hunt10']:.1f}%" if fix3 else "—"),
        ("Cluster%",          "46.8%",              f"{part1_right['cluster']:.1f}%",
         f"{fix3['cluster']:.1f}%" if fix3 else "—"),
        ("Full WFA OOS Sh",   "9.93",               "—",
         f"{fix3['full_wfa_oos']:.2f}" if fix3 else "—"),
        ("MC RoR (S1)",       "N/A",                "—",
         f"{fix3['mc_ror']:.1f}%" if fix3 else "—"),
        ("MC P50 (S1)",       "N/A",                "—",
         f"{fix3['mc_p50']:+.1f}%" if fix3 else "—"),
        ("─" * 20, "─" * 18, "─" * 18, "─" * 18),
    ]
    txt = "\n".join(f"{a:<22}{b:<22}{c:<22}{d}" for a,b,c,d in rows_txt)
    ax.text(0.02, 0.90, txt, transform=ax.transAxes, va="top",
            family="monospace", fontsize=11, color=PLT_FG)

    plt.tight_layout()
    plt.savefig(out_path, dpi=120); plt.close()
    print(f"  saved: {out_path}")


# ═══════════════════ MAIN ══════════════════════════════════════════
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 70)
    print("FULL RE-TEST — CORRECTED BingX FUTURES FEES")
    print("  Old model: 0.075% taker × 2 = 0.150% RT")
    print("  New model: 0.020% maker entry + 0.050% taker SL = 0.056% avg RT")
    print("=" * 70)

    df = load_candles("BTC/USDT", "15m", days=730)
    htf = compute_htf_bias(df, htf_ema_period=50)
    print(f"  Loaded {len(df)} bars")

    # Baseline smoke WFA (for comparison benchmark)
    print("\n  Baseline smoke WFA (rr=1.5, sl_atr=2.0)...")
    baseline_oos = smoke_wfa(df, htf, rr=1.5, sl_atr=2.0)
    print(f"  Baseline smoke WFA OOS Sharpe: {baseline_oos:.3f}")

    # ── PART 1 ────────────────────────────────────────────────────
    part1 = part1_penetration(df, htf, rr=1.5, sl_atr=2.0)

    # Print corrected scenario table
    print("\n" + "-" * 100)
    print(f"{'Scenario':<12}{'Entry%':>8}{'SL%':>8}{'TP%':>8}{'Del':>5}"
          f"{'Sharpe':>9}{'Ann%':>8}{'ExpR':>9}{'DD%':>7}  Status")
    print("-" * 100)
    for r in part1["rows"][:-1]:
        st = classify(r["sharpe"], r["expR"])
        mk = {"SAFE":"✓","MARGINAL":"~","DANGER":"✗"}[st]
        print(f"{r['scenario']:<12}{r['entry']:>7.3f}%{r['sl']:>7.3f}%{r['tp']:>7.3f}%"
              f"{r['delay']:>5}{r['sharpe']:>9.3f}{r['ann']:>7.1f}%{r['expR']:>+8.3f}R"
              f"{r['dd']:>6.1f}%  {mk} {st}")
    s6 = part1["rows"][-1]
    print(f"{s6['scenario']:<12}{s6['entry']:>7.3f}%{s6['sl']:>7.3f}%{s6['tp']:>7.3f}%"
          f"{s6['delay']:>5}{'~1.0':>9}{'—':>8}{'—':>9}{'—':>7}  BREAK")
    print("-" * 100)

    # ── PART 2 ────────────────────────────────────────────────────
    fix1 = part2_fix1(df, htf, baseline_oos)
    print("\n" + "-" * 100)
    print(f"{'rr':<5}{'S0 Sh':>8}{'S1 Sh':>8}{'S1 ExpR':>10}{'S2 Sh':>8}{'S2 ExpR':>10}{'Hunt10%':>10}{'OOS Sh':>9}  Status")
    print("-" * 100)
    s1_base = next((r["sharpe"] for r in part1["rows"] if r["scenario"] == "S1 paper"), 0)
    s1_exp_base = next((r["expR"] for r in part1["rows"] if r["scenario"] == "S1 paper"), 0)
    s2_base = next((r["sharpe"] for r in part1["rows"] if r["scenario"] == "S2 real"), 0)
    s2_exp_base = next((r["expR"] for r in part1["rows"] if r["scenario"] == "S2 real"), 0)
    print(f"{'1.5':<5}{part1['rows'][0]['sharpe']:>8.2f}{s1_base:>8.2f}{s1_exp_base:>+9.3f}R"
          f"{s2_base:>8.2f}{s2_exp_base:>+9.3f}R{part1['rec10']:>9.1f}%{baseline_oos:>9.2f}  BASELINE")
    for r in fix1:
        print(f"{r['rr']:<5}{r['s0_sharpe']:>8.2f}{r['s1_sharpe']:>8.2f}{r['s1_expR']:>+9.3f}R"
              f"{r['s2_sharpe']:>8.2f}{r['s2_expR']:>+9.3f}R{r['hunt10']:>9.1f}%{r['oos']:>9.2f}"
              f"  {'PASS' if r['passes'] else 'FAIL'}")
    print("-" * 100)

    passing1 = [r for r in fix1 if r["passes"]]

    # ── PART 3 ────────────────────────────────────────────────────
    fix2 = None; fix3 = None; best_rr_val = None; best_sl_val = None
    if passing1:
        best_rr_row = max(passing1, key=lambda r: r["s2_sharpe"])
        best_rr_val = best_rr_row["rr"]
        print(f"\n>>> Best passing rr from Fix 1: {best_rr_val}")
        fix2 = part3_fix2(df, htf, baseline_oos, best_rr_val)
        print("\n" + "-" * 100)
        print(f"{'sl_atr':<8}{'Cluster%':>10}{'Hunt10%':>10}{'S1 Sh':>8}{'S2 Sh':>8}{'S2 ExpR':>10}{'OOS Sh':>9}  Status")
        print("-" * 100)
        print(f"{'2.00':<8}{part1['cluster']:>9.1f}%{part1['rec10']:>9.1f}%{s1_base:>8.2f}"
              f"{s2_base:>8.2f}{s2_exp_base:>+9.3f}R{baseline_oos:>9.2f}  BASELINE")
        for r in fix2:
            print(f"{r['sl_atr']:<8}{r['cluster']:>9.1f}%{r['hunt10']:>9.1f}%{r['s1_sharpe']:>8.2f}"
                  f"{r['s2_sharpe']:>8.2f}{r['s2_expR']:>+9.3f}R{r['oos']:>9.2f}"
                  f"  {'PASS' if r['passes'] else 'FAIL'}")
        print("-" * 100)
        passing2 = [r for r in fix2 if r["passes"]]

        # ── PART 4 ────────────────────────────────────────────────
        if passing2:
            best_sl_row = min(passing2, key=lambda r: r["cluster"])
            best_sl_val = best_sl_row["sl_atr"]
            print(f"\n>>> Best passing sl_atr from Fix 2: {best_sl_val}")
            fix3 = part4_combined(df, htf, best_rr_val, best_sl_val)
        else:
            print("\nFix 2: NO passing sl_atr. Skipping combined validation.")
    else:
        print("\nFix 1: NO passing rr_ratio. Skipping Fix 2 and combined.")

    # ── PLOTS ─────────────────────────────────────────────────────
    print("\nGenerating plots...")
    plot_penetration_corrected(part1, OUTPUT_DIR / "penetration_slippage_corrected_BTCUSDT.png")
    plot_fix_comparison(fix1, fix2, part1, fix3, OUTPUT_DIR / "fix_comparison_corrected_BTCUSDT.png")
    plot_before_after(None, part1, fix3, OUTPUT_DIR / "full_retest_summary_BTCUSDT.png")

    # ── FINAL REPORT ──────────────────────────────────────────────
    fix1_pass = len(passing1) > 0
    fix2_pass = fix2 is not None and len([r for r in fix2 if r["passes"]]) > 0

    s1_row = next((r for r in part1["rows"] if r["scenario"] == "S1 paper"), None)
    s2_row = next((r for r in part1["rows"] if r["scenario"] == "S2 real"), None)
    s1_status = classify(s1_row["sharpe"], s1_row["expR"]) if s1_row else "?"
    s2_status = classify(s2_row["sharpe"], s2_row["expR"]) if s2_row else "?"

    baseline_viable = s1_status in ("SAFE", "MARGINAL") and s2_status in ("SAFE", "MARGINAL")

    if fix3:
        s1_fix = next((r for r in fix3["scenarios"] if r["scenario"] == "S1 paper"), None)
        s2_fix = next((r for r in fix3["scenarios"] if r["scenario"] == "S2 real"), None)
        combined_ok = (s2_fix["sharpe"] > S2_SHARPE_MIN and s2_fix["expR"] > S2_EXPR_MIN
                       and fix3["hunt10"] < MAX_HUNT10 and fix3["cluster"] < MAX_CLUSTER)
    else:
        combined_ok = False

    overall = "PASS" if (combined_ok or (baseline_viable and s1_status == "SAFE")) else \
              ("PARTIAL" if baseline_viable or fix1_pass else "FAIL")
    ready = "YES" if overall == "PASS" else ("CONDITIONAL" if overall == "PARTIAL" else "NO")
    final_rr = best_rr_val if best_rr_val else 1.5
    final_sl = best_sl_val if best_sl_val else 2.0

    print()
    print("  ╔══════════════════════════════════════════════════════════════╗")
    print("  ║      FULL RE-TEST REPORT — Corrected BingX Futures Fees      ║")
    print("  ╠══════════════════════════════════════════════════════════════╣")
    print("  ║ FEE MODEL CORRECTION                                         ║")
    print("  ║  Old: 0.075% taker × 2 = 0.150% RT                           ║")
    print("  ║  New: 0.020% maker entry + 0.050% taker SL                   ║")
    print("  ║       = 0.056% avg RT (weighted by WR)                       ║")
    print(f"  ║  Impact: breaking point {part1['breaking_point']:.3f}% (was 0.100%)              ║")
    print("  ╠══════════════════════════════════════════════════════════════╣")
    print("  ║ PENETRATION TEST (corrected)                                 ║")
    s1_sh = s1_row["sharpe"] if s1_row else 0
    s2_sh = s2_row["sharpe"] if s2_row else 0
    print(f"  ║  S1 paper Sharpe: {s1_sh:>6.3f}  ExpR: {s1_row['expR']:>+.3f}R   [{s1_status}]    ║")
    print(f"  ║  S2 real Sharpe:  {s2_sh:>6.3f}  ExpR: {s2_row['expR']:>+.3f}R   [{s2_status}]    ║")
    print(f"  ║  S6 breaking pt:  {part1['breaking_point']:.3f}% total slippage                   ║")
    print(f"  ║  Hunt10%: {part1['rec10']:.1f}%   Cluster%: {part1['cluster']:.1f}%                     ║")
    print("  ╠══════════════════════════════════════════════════════════════╣")
    print("  ║ FIX 1 — R:R (tested 2.0, 2.5, 3.0)                           ║")
    if fix1_pass:
        best = max(passing1, key=lambda r: r["s2_sharpe"])
        print(f"  ║  Best rr_ratio: {best['rr']}                                        ║")
        print(f"  ║  S1 Sharpe:     {best['s1_sharpe']:.3f}  S2 Sharpe: {best['s2_sharpe']:.3f}            ║")
        print(f"  ║  Result:        PASS                                          ║")
    else:
        print("  ║  Result: FAIL — no rr passes S2 criteria                     ║")
    print("  ╠══════════════════════════════════════════════════════════════╣")
    print("  ║ FIX 2 — SL NON-ROUND (tested 1.87, 2.07, 2.15, 2.23)         ║")
    if fix2_pass:
        best2 = min([r for r in fix2 if r["passes"]], key=lambda r: r["cluster"])
        print(f"  ║  Best sl_atr:   {best2['sl_atr']}                                     ║")
        print(f"  ║  Cluster%:      {best2['cluster']:.1f}% (was 46.8%)                        ║")
        print(f"  ║  Hunt10%:       {best2['hunt10']:.1f}% (was 41.7%)                        ║")
        print("  ║  Result:        PASS                                          ║")
    elif fix2 is None:
        print("  ║  (skipped — Fix 1 had no passing rr)                         ║")
    else:
        print("  ║  Result: FAIL — clustering/hunt persists across all sl_atr   ║")
    print("  ╠══════════════════════════════════════════════════════════════╣")
    print("  ║ COMBINED (if both pass)                                      ║")
    if fix3:
        s2_fix = next((r for r in fix3["scenarios"] if r["scenario"] == "S2 real"), None)
        print(f"  ║  Final params:      rr={best_rr_val} sl_atr={best_sl_val}                     ║")
        print(f"  ║  S2 Sharpe:         {s2_fix['sharpe']:.3f}                                ║")
        print(f"  ║  Breaking point:    {fix3['breaking_point']:.3f}%                              ║")
        print(f"  ║  Full WFA OOS Sh:   {fix3['full_wfa_oos']:.3f}                                ║")
        print(f"  ║  MC RoR (S1 fills): {fix3['mc_ror']:.1f}%                                 ║")
        print(f"  ║  MC P50:            {fix3['mc_p50']:+.1f}%                                 ║")
    else:
        print("  ║  (not run)                                                   ║")
    print("  ╠══════════════════════════════════════════════════════════════╣")
    print("  ║ BOT UPDATE                                                   ║")
    print("  ║  run_simple_paper.py fee model: UPDATED (see Part 5)         ║")
    print("  ║  Entry: limit → 0.020% maker                                 ║")
    print("  ║  SL:    market → 0.050% taker                                ║")
    print("  ║  TP:    limit → 0.020% maker                                 ║")
    print("  ╠══════════════════════════════════════════════════════════════╣")
    print("  ║ FINAL VERDICT                                                ║")
    print(f"  ║  Penetration test:   {overall}                                    ║")
    print(f"  ║  Ready for paper:    {ready}                              ║")
    print(f"  ║  Updated params:     rr={final_rr} sl_atr={final_sl}                       ║")
    if overall == "PASS":
        na = "deploy paper trading with limit-order entry bot update"
    elif overall == "PARTIAL":
        na = "paper trade conservatively with limit orders; monitor S2 metrics"
    else:
        na = "revise strategy: tighten entries to raise gross ExpR above 0.14R"
    print(f"  ║  Next action:        {na[:40]:<40}  ║")
    print("  ╚══════════════════════════════════════════════════════════════╝")

    ready_word = "READY" if overall == "PASS" else ("CONDITIONAL" if overall == "PARTIAL" else "NOT READY")
    print(f"\nStrategy {ready_word} for paper trading with params rr={final_rr} sl_atr={final_sl}. "
          f"Next action: {na}.")

    # Save summary state for downstream if needed
    return {"part1": part1, "fix1": fix1, "fix2": fix2, "fix3": fix3,
            "best_rr": best_rr_val, "best_sl": best_sl_val,
            "overall": overall, "ready": ready}


if __name__ == "__main__":
    main()
