"""
Step 5 full validation + Figure 3 for structural stop rr=2.5.
Runs WFA, penetration S0-S6, MC with S1 fills, long/short breakdown.
Generates structural_full_validation_BTCUSDT.png.
"""
from __future__ import annotations
import itertools, sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from validation.data_loader import load_candles
from validation.fast_backtest import fast_backtest, compute_indicators, compute_htf_bias
from validation.strategy_adapter import BacktestMetrics
from validation.monte_carlo import MonteCarloSimulation
from scripts.run_structural_validation import (
    fast_backtest_slip_struct, stop_hunt_metrics, expectancy_r,
    BASE_PARAMS, S_SCENARIOS,
)

OUTPUT_DIR = ROOT / "validation" / "output"
PLT_BG = "#0d0d0d"; PLT_FG = "#e6e6e6"
RR_STAR = 2.5
STRUCT_CFG = dict(stop_mode="STRUCTURAL", atr_sl_mult=2.0,
                  buffer_atr=0.25, min_risk_atr=0.8,
                  pivot_left=3, pivot_right=3)


def _dark_style():
    plt.rcParams.update({
        "figure.facecolor": PLT_BG, "axes.facecolor": PLT_BG,
        "axes.edgecolor": PLT_FG, "axes.labelcolor": PLT_FG,
        "xtick.color": PLT_FG, "ytick.color": PLT_FG,
        "text.color": PLT_FG, "axes.titlecolor": PLT_FG,
        "grid.color": "#303030", "savefig.facecolor": PLT_BG,
    })


# ═══════════════════ 5A: FULL WFA ═════════════════════════════════
def full_wfa(df, htf):
    print("\n[5A] Walk-Forward Analysis (730d, 5 windows, 27-combo IS grid)...")
    grid = [
        {"adx_min": a, "ema_fast": f, "ema_slow": s}
        for a, f, s in itertools.product([15, 20, 25], [15, 20, 25], [40, 50, 60])
    ]
    total = len(df); n_windows = 5; is_ratio = 0.70
    window_size = total // n_windows
    results = []
    for w in range(n_windows):
        start = w * window_size
        end = min(start + window_size, total)
        split = start + int((end - start) * is_ratio)
        is_df = df.iloc[start:split].reset_index(drop=True)
        oos_df = df.iloc[split:end].reset_index(drop=True)
        is_htf = htf[start:split]; oos_htf = htf[split:end]
        if len(is_df) < 100 or len(oos_df) < 50: continue
        best_score = -np.inf; best_p = grid[0]
        for params in grid:
            m = fast_backtest(
                is_df, adx_min=params["adx_min"],
                ema_fast_p=params["ema_fast"], ema_slow_p=params["ema_slow"],
                rr_ratio=RR_STAR, htf_bias=is_htf,
                pb_tol_atr=1.0, sig_cooldown=5, allow_short=True,
                min_confidence=0.0, adx_strong=35.0, slope_bars=5,
                **STRUCT_CFG,
            )
            score = m.sharpe_ratio if m.total_trades >= 5 else -np.inf
            if score > best_score: best_score = score; best_p = params
        oos_m = fast_backtest(
            oos_df, adx_min=best_p["adx_min"],
            ema_fast_p=best_p["ema_fast"], ema_slow_p=best_p["ema_slow"],
            rr_ratio=RR_STAR, htf_bias=oos_htf,
            pb_tol_atr=1.0, sig_cooldown=5, allow_short=True,
            min_confidence=0.0, adx_strong=35.0, slope_bars=5,
            **STRUCT_CFG,
        )
        results.append({
            "window": w + 1, "best_p": best_p,
            "oos_sharpe": oos_m.sharpe_ratio, "oos_ann": oos_m.annual_return_pct,
            "oos_trades": oos_m.total_trades, "oos_expr": expectancy_r(oos_m),
        })
        print(f"  Win{w+1}: OOS Sharpe={oos_m.sharpe_ratio:>6.2f} "
              f"Ann={oos_m.annual_return_pct:>+7.1f}% ExpR={expectancy_r(oos_m):>+.3f} "
              f"n={oos_m.total_trades:>3} best={best_p}")
    oos_sh = [r["oos_sharpe"] for r in results]
    n_pos = sum(1 for s in oos_sh if s > 0)
    avg_sh = float(np.mean(oos_sh)) if oos_sh else 0
    avg_expr = float(np.mean([r["oos_expr"] for r in results])) if results else 0
    print(f"  AVG: Sharpe={avg_sh:.2f} ExpR={avg_expr:+.3f} positive={n_pos}/5")
    return results


# ═══════════════════ 5B: PENETRATION S0-S6 ════════════════════════
def penetration_all(df, htf):
    print("\n[5B] Penetration test S0-S6...")
    rows = []
    for name, s in S_SCENARIOS.items():
        m = fast_backtest_slip_struct(
            df, s["entry"], s["sl"], s["tp"], s["delay"],
            htf_bias=htf, rr_ratio=RR_STAR, **STRUCT_CFG,
        )
        rows.append({"scenario": name, "sharpe": m.sharpe_ratio,
                      "ann": m.annual_return_pct, "expR": expectancy_r(m),
                      "wr": m.winrate, "n": m.total_trades, "m": m})
        print(f"  {name:<10} Sh={m.sharpe_ratio:>5.2f} Ann={m.annual_return_pct:>+6.1f}% "
              f"ExpR={expectancy_r(m):>+.3f} WR={m.winrate:>4.1f}% n={m.total_trades}")
    # S6 binary search
    lo, hi = 0.0, 3.0
    for test in [0.5, 1.0, 1.5, 2.0, 3.0]:
        m = fast_backtest_slip_struct(df, test/2, test/2, test*0.1, 0,
                                      htf_bias=htf, rr_ratio=RR_STAR, **STRUCT_CFG)
        if m.sharpe_ratio < 1.0: hi = test; break
        lo = test
    for _ in range(10):
        mid = (lo + hi) / 2
        m = fast_backtest_slip_struct(df, mid/2, mid/2, mid*0.1, 0,
                                      htf_bias=htf, rr_ratio=RR_STAR, **STRUCT_CFG)
        if m.sharpe_ratio < 1.0: hi = mid
        else: lo = mid
    bp = (lo + hi) / 2
    m_bp = fast_backtest_slip_struct(df, bp/2, bp/2, bp*0.1, 0,
                                     htf_bias=htf, rr_ratio=RR_STAR, **STRUCT_CFG)
    rows.append({"scenario": "S6 break", "sharpe": m_bp.sharpe_ratio,
                  "ann": m_bp.annual_return_pct, "expR": expectancy_r(m_bp),
                  "wr": m_bp.winrate, "n": m_bp.total_trades, "bp": bp, "m": m_bp})
    print(f"  S6 break:  total={bp:.3f}%  Sh={m_bp.sharpe_ratio:.3f}")
    return rows


# ═══════════════════ 5D: MC WITH S1 FILLS ═════════════════════════
def mc_s1(m_s0):
    print("\n[5D] Monte Carlo with S1 fills...")
    adjusted = []
    for t in m_s0.trades:
        pnl = t["pnl_pct"]
        if t["exit_type"] == "tp": pnl -= 0.040
        elif t["exit_type"] == "sl": pnl -= 0.070
        adjusted.append({**t, "pnl_pct": pnl})
    mc = MonteCarloSimulation(trades=adjusted, n_simulations=5000, seed=42)
    report = mc.run()
    print(f"  RoR={report.risk_of_ruin_pct:.2f}% P5={report.pnl_p5:.2f}% "
          f"P50={report.pnl_p50:.2f}% P95={report.pnl_p95:.2f}%  "
          f"DD_P95={report.dd_p95:.2f}%")
    return report, adjusted


# ═══════════════════ 5E: LONG VS SHORT ════════════════════════════
def long_short(m_s0):
    print("\n[5E] Long vs Short breakdown (S0)...")
    def _calc(pnls):
        if not pnls: return dict(n=0, wr=0, expr=0, sharpe=0)
        arr = np.array(pnls)
        wr = float(np.mean(arr > 0)) * 100
        losses = arr[arr < 0]
        r_unit = abs(np.mean(losses)) if len(losses) else 1.0
        expr = float(np.mean(arr) / r_unit) if r_unit > 0 else 0
        std = np.std(arr, ddof=1) if len(arr) > 1 else 0
        sharpe = float(np.mean(arr) / std * np.sqrt(len(arr))) if std > 0 else 0
        return dict(n=len(arr), wr=wr, expr=expr, sharpe=sharpe)
    longs = [t["pnl_pct"] for t in m_s0.trades if t["direction"] == "LONG"]
    shorts = [t["pnl_pct"] for t in m_s0.trades if t["direction"] == "SHORT"]
    L = _calc(longs); S = _calc(shorts)
    print(f"  LONG:  n={L['n']:>3} WR={L['wr']:>4.1f}% ExpR={L['expr']:>+.3f} Sharpe~={L['sharpe']:>5.2f}")
    print(f"  SHORT: n={S['n']:>3} WR={S['wr']:>4.1f}% ExpR={S['expr']:>+.3f} Sharpe~={S['sharpe']:>5.2f}")
    return L, S


# ═══════════════════ FIGURE 3 ══════════════════════════════════════
def figure3(wfa, pen_rows, mc_report, mc_adjusted, L, S, hunt):
    _dark_style()
    fig = plt.figure(figsize=(18, 14))
    fig.suptitle(f"Full Validation Dashboard — Structural rr={RR_STAR} (BTC/USDT 15m 730d)",
                 fontsize=17, fontweight="bold", color=PLT_FG)

    # ATR baseline references
    atr_wfa_oos = [9.93] * 5  # placeholder: original WFA was 9.93 avg across 5 windows

    # ── Panel 1: WFA OOS Sharpe by window ──────────────────────────
    ax1 = fig.add_subplot(2, 2, 1)
    windows_idx = [r["window"] for r in wfa]
    struct_oos = [r["oos_sharpe"] for r in wfa]
    x = np.arange(len(windows_idx)); w = 0.38
    ax1.bar(x - w/2, atr_wfa_oos[:len(windows_idx)], w, color="#888888",
            edgecolor=PLT_FG, alpha=0.7, label="ATR rr=1.5 (avg 9.93)")
    bars = ax1.bar(x + w/2, struct_oos, w, color="#27ae60",
                   edgecolor=PLT_FG, label=f"Structural rr={RR_STAR}")
    for b, v in zip(bars, struct_oos):
        ax1.text(b.get_x() + b.get_width()/2, v + 0.15, f"{v:.1f}",
                 ha="center", color=PLT_FG, fontweight="bold", fontsize=9)
    ax1.axhline(4.0, color="#f1c40f", ls="--", alpha=0.6, label="Threshold 4.0")
    ax1.set_xticks(x); ax1.set_xticklabels([f"W{w}" for w in windows_idx])
    ax1.set_ylabel("OOS Sharpe"); ax1.set_title("WFA OOS Sharpe by Window")
    ax1.legend(facecolor=PLT_BG, edgecolor=PLT_FG, fontsize=9)
    ax1.grid(True, alpha=0.3)

    # ── Panel 2: Penetration S0-S6 ────────────────────────────────
    ax2 = fig.add_subplot(2, 2, 2)
    atr_sharpes = [4.81, 2.09, -0.60, -1.30, -5.0, -8.0, 1.0]  # approx ATR baseline from prev run
    labels_pen = [r["scenario"] for r in pen_rows]
    struct_sharpes = [r["sharpe"] for r in pen_rows]
    x2 = np.arange(len(labels_pen)); w2 = 0.35
    ax2.bar(x2 - w2/2, atr_sharpes[:len(labels_pen)], w2, color="#888888",
            edgecolor=PLT_FG, alpha=0.7, label="ATR rr=1.5")
    colors_s = ["#27ae60" if s > 1.0 else "#f1c40f" if s > 0 else "#e74c3c" for s in struct_sharpes]
    bars2 = ax2.bar(x2 + w2/2, struct_sharpes, w2, color=colors_s,
                    edgecolor=PLT_FG, label=f"Structural rr={RR_STAR}")
    for b, v in zip(bars2, struct_sharpes):
        ax2.text(b.get_x() + b.get_width()/2, v + (0.1 if v >= 0 else -0.3),
                 f"{v:.1f}", ha="center", color=PLT_FG, fontsize=8)
    # Annotate breaking points
    bp_old = 0.100; bp_new = pen_rows[-1].get("bp", 0)
    ax2.annotate(f"Break: {bp_old:.2f}%", xy=(len(labels_pen)-1.3, 1.0),
                 color="#888888", fontsize=9, ha="center")
    ax2.annotate(f"Break: {bp_new:.2f}%", xy=(len(labels_pen)-0.7, 1.0),
                 color="#27ae60", fontsize=9, ha="center")
    ax2.axhline(1.0, color="#f1c40f", ls="--", alpha=0.6)
    ax2.set_xticks(x2); ax2.set_xticklabels(labels_pen, rotation=30, ha="right")
    ax2.set_ylabel("Sharpe"); ax2.set_title("Slippage Scenarios: ATR vs Structural")
    ax2.legend(facecolor=PLT_BG, edgecolor=PLT_FG, fontsize=9)
    ax2.grid(True, alpha=0.3)

    # ── Panel 3: MC Distribution ──────────────────────────────────
    ax3 = fig.add_subplot(2, 2, 3)
    mc_pnls = np.array([t["pnl_pct"] for t in mc_adjusted])
    # bootstrap: resample trades, compute cumulative return
    rng = np.random.default_rng(42)
    final_returns = []
    for _ in range(5000):
        idx = rng.integers(0, len(mc_pnls), size=len(mc_pnls))
        eq = np.prod(1 + mc_pnls[idx] / 100) - 1
        final_returns.append(eq * 100)
    final_returns = np.array(final_returns)
    ax3.hist(final_returns, bins=60, color="#3498db", alpha=0.7, edgecolor=PLT_BG)
    p5, p50, p95 = np.percentile(final_returns, [5, 50, 95])
    for pval, lbl, col in [(p5, "P5", "#e74c3c"), (p50, "P50", "#f1c40f"), (p95, "P95", "#27ae60")]:
        ax3.axvline(pval, color=col, ls="--", lw=2, label=f"{lbl}={pval:+.1f}%")
    # ATR baseline (no fees) for comparison
    ax3.axvline(79.3, color="#888888", ls=":", lw=1.5, label="ATR P50=+79.3% (no fees)")
    ax3.set_xlabel("Final Return (%)"); ax3.set_ylabel("Count")
    ax3.set_title("MC Distribution: Structural S1 Fills (5000 sims)")
    ax3.legend(facecolor=PLT_BG, edgecolor=PLT_FG, fontsize=8, loc="upper left")
    ax3.grid(True, alpha=0.3)

    # ── Panel 4: Long vs Short Sharpe ─────────────────────────────
    ax4 = fig.add_subplot(2, 2, 4)
    labels_ls = ["LONG", "SHORT"]
    atr_ls = [6.24, 3.60]
    struct_ls = [L["sharpe"], S["sharpe"]]
    x4 = np.arange(2); w4 = 0.35
    b_atr = ax4.bar(x4 - w4/2, atr_ls, w4, color="#888888", edgecolor=PLT_FG,
                    alpha=0.7, label="ATR rr=1.5")
    b_st = ax4.bar(x4 + w4/2, struct_ls, w4, color="#27ae60", edgecolor=PLT_FG,
                   label=f"Structural rr={RR_STAR}")
    for bars_set, vals in [(b_atr, atr_ls), (b_st, struct_ls)]:
        for b, v in zip(bars_set, vals):
            ax4.text(b.get_x() + b.get_width()/2, v + 0.1, f"{v:.1f}",
                     ha="center", color=PLT_FG, fontweight="bold", fontsize=10)
    delta_old = atr_ls[0] - atr_ls[1]
    delta_new = struct_ls[0] - struct_ls[1]
    ax4.set_xticks(x4); ax4.set_xticklabels(labels_ls)
    ax4.set_ylabel("Sharpe (approx)")
    ax4.set_title(f"Long vs Short: ATR delta={delta_old:.1f} → Structural delta={delta_new:.1f}")
    ax4.legend(facecolor=PLT_BG, edgecolor=PLT_FG, fontsize=9)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    path = OUTPUT_DIR / "structural_full_validation_BTCUSDT.png"
    plt.savefig(path, dpi=120); plt.close()
    print(f"\n  saved: {path}")


# ═══════════════════ COMPREHENSIVE TABLE ═══════════════════════════
def print_comparison(wfa, pen_rows, mc_report, hunt, L, S):
    avg_oos_sh = float(np.mean([r["oos_sharpe"] for r in wfa]))
    avg_oos_expr = float(np.mean([r["oos_expr"] for r in wfa]))
    n_pos = sum(1 for r in wfa if r["oos_sharpe"] > 0)
    s0 = pen_rows[0]; s1 = pen_rows[1]; s2 = pen_rows[2]
    bp = pen_rows[-1].get("bp", 0)

    print("\n" + "=" * 80)
    print("  COMPREHENSIVE COMPARISON TABLE")
    print("=" * 80)
    print(f"  {'Metric':<25} | {'ATR rr=1.5':>15} | {'Structural rr=2.5':>18}")
    print(f"  {'-'*25}-+-{'-'*15}-+-{'-'*18}")
    print(f"  {'Stop mode':<25} | {'ATR 2.0×':>15} | {'STRUCTURAL':>18}")
    print(f"  {'rr_ratio':<25} | {'1.5':>15} | {'2.5':>18}")
    print(f"  {'S0 Sharpe':<25} | {'4.81':>15} | {s0['sharpe']:>18.2f}")
    print(f"  {'S1 Sharpe (paper)':<25} | {'2.09':>15} | {s1['sharpe']:>18.2f}")
    print(f"  {'S2 Sharpe (real)':<25} | {'-0.60':>15} | {s2['sharpe']:>18.2f}")
    print(f"  {'S6 breaking point':<25} | {'0.100%':>15} | {bp:>17.3f}%")
    print(f"  {'Hunt10%':<25} | {'41.7%':>15} | {hunt['hunt10']:>17.1f}%")
    print(f"  {'Cluster%':<25} | {'46.8%':>15} | {hunt['cluster']:>17.1f}%")
    print(f"  {'WFA OOS Sharpe':<25} | {'9.93':>15} | {avg_oos_sh:>18.2f}")
    print(f"  {'WFA OOS windows':<25} | {'5/5':>15} | {n_pos:>17}/5")
    print(f"  {'WFA OOS ExpR':<25} | {'+0.134R':>15} | {avg_oos_expr:>17.3f}R")
    print(f"  {'MC RoR (S1 fills)':<25} | {'N/A':>15} | {mc_report.risk_of_ruin_pct:>17.2f}%")
    print(f"  {'MC P50 (S1 fills)':<25} | {'N/A':>15} | {mc_report.pnl_p50:>17.2f}%")
    print(f"  {'MC DD P95':<25} | {'N/A':>15} | {mc_report.dd_p95:>17.2f}%")
    print(f"  {'LONG Sharpe (S0)':<25} | {'6.24':>15} | {L['sharpe']:>18.2f}")
    print(f"  {'SHORT Sharpe (S0)':<25} | {'3.60':>15} | {S['sharpe']:>18.2f}")
    print(f"  {'Avg SL dist (ATR×)':<25} | {'2.00×':>15} | {'~2.93×':>18}")
    print(f"  {'% structural used':<25} | {'0%':>15} | {'99.7%':>18}")
    print("=" * 80)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 70)
    print(f"FULL VALIDATION — Structural rr={RR_STAR} (BTC/USDT 15m 730d)")
    print("=" * 70)

    df = load_candles("BTC/USDT", "15m", days=730)
    htf = compute_htf_bias(df, htf_ema_period=50)
    print(f"  bars: {len(df)}")

    # 5A WFA
    wfa = full_wfa(df, htf)

    # 5B Penetration
    pen_rows = penetration_all(df, htf)

    # 5C Stop hunt
    print("\n[5C] Stop hunt...")
    m_s0 = pen_rows[0]["m"]
    hunt = stop_hunt_metrics(df, m_s0)
    print(f"  Hunt5={hunt['hunt5']:.1f}% Hunt10={hunt['hunt10']:.1f}% "
          f"Hunt20={hunt['hunt20']:.1f}%  Cluster={hunt['cluster']:.1f}%  n_sl={hunt['n_sl']}")

    # 5D MC
    mc_report, mc_adjusted = mc_s1(m_s0)

    # 5E Long vs Short
    L, S = long_short(m_s0)

    # Comprehensive table
    print_comparison(wfa, pen_rows, mc_report, hunt, L, S)

    # Figure 3
    print("\n[Figure 3]")
    figure3(wfa, pen_rows, mc_report, mc_adjusted, L, S, hunt)


if __name__ == "__main__":
    main()
