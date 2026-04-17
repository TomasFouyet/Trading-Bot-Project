"""Generate Figures 1 & 2 for structural stop validation (Fig 3 skipped — no CANDIDATE)."""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from validation.data_loader import load_candles
from validation.fast_backtest import fast_backtest, compute_htf_bias
from scripts.run_structural_validation import (
    fast_backtest_slip_struct, stop_hunt_metrics, sl_dist_stats,
    expectancy_r, BASE_PARAMS, S_SCENARIOS,
)

OUTPUT_DIR = ROOT / "validation" / "output"
PLT_BG = "#0d0d0d"; PLT_FG = "#e6e6e6"


def _dark_style():
    plt.rcParams.update({
        "figure.facecolor": PLT_BG, "axes.facecolor": PLT_BG,
        "axes.edgecolor": PLT_FG, "axes.labelcolor": PLT_FG,
        "xtick.color": PLT_FG, "ytick.color": PLT_FG,
        "text.color": PLT_FG, "axes.titlecolor": PLT_FG,
        "grid.color": "#303030", "savefig.facecolor": PLT_BG,
    })


def figure1(df, htf):
    """Structural vs ATR comparison. Use rr=2.5 for structural (best S2 sharpe in sweep)."""
    _dark_style()
    struct_cfg = dict(stop_mode="STRUCTURAL", atr_sl_mult=2.0,
                      buffer_atr=0.25, min_risk_atr=0.8,
                      pivot_left=3, pivot_right=3)
    # ATR baseline
    atr_s0 = fast_backtest_slip_struct(df, 0, 0, 0, 0, htf_bias=htf, stop_mode="ATR",
                                        rr_ratio=1.5, atr_sl_mult=2.0)
    s1 = S_SCENARIOS["S1 paper"]
    atr_s1 = fast_backtest_slip_struct(df, s1["entry"], s1["sl"], s1["tp"], s1["delay"],
                                        htf_bias=htf, stop_mode="ATR",
                                        rr_ratio=1.5, atr_sl_mult=2.0)
    # Structural rr=2.5 (best S2 Sharpe)
    st_s0 = fast_backtest_slip_struct(df, 0, 0, 0, 0, htf_bias=htf,
                                       rr_ratio=2.5, **struct_cfg)
    st_s1 = fast_backtest_slip_struct(df, s1["entry"], s1["sl"], s1["tp"], s1["delay"],
                                       htf_bias=htf, rr_ratio=2.5, **struct_cfg)

    def _eq(m):
        pnls = np.array([t["pnl_pct"] for t in m.trades])
        return np.concatenate([[1.0], np.cumprod(1 + pnls / 100)])

    atr_hunt = stop_hunt_metrics(df, atr_s0)
    st_hunt = stop_hunt_metrics(df, st_s0)
    atr_dist = sl_dist_stats(atr_s0, df)
    st_dist = sl_dist_stats(st_s0, df)

    fig = plt.figure(figsize=(18, 13))
    fig.suptitle("Structural vs ATR Stop — BTC/USDT 15m 730d",
                 fontsize=17, fontweight="bold", color=PLT_FG)

    # P1: Equity curves
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(_eq(atr_s0), color="#e74c3c", lw=1.8, ls="--", label="ATR rr=1.5 (S0)", alpha=0.9)
    ax1.plot(_eq(atr_s1), color="#888888", lw=1.5, ls="--", label="ATR rr=1.5 (S1)", alpha=0.8)
    ax1.plot(_eq(st_s0), color="#27ae60", lw=1.8, label="Structural rr=2.5 (S0)")
    ax1.plot(_eq(st_s1), color="#3498db", lw=1.8, label="Structural rr=2.5 (S1)")
    ax1.axhline(1.0, color=PLT_FG, ls="--", alpha=0.3)
    ax1.set_title("Equity Curves: ATR vs Structural")
    ax1.set_xlabel("Trade #"); ax1.set_ylabel("Equity multiplier")
    ax1.legend(facecolor=PLT_BG, edgecolor=PLT_FG, fontsize=9)
    ax1.grid(True, alpha=0.3)

    # P2: SL distance dist
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.hist(atr_dist["dists"], bins=30, color="#e74c3c", alpha=0.55,
             edgecolor=PLT_FG, label=f"ATR mean={atr_dist['mean']:.2f}×")
    ax2.hist(st_dist["dists"], bins=30, color="#3498db", alpha=0.55,
             edgecolor=PLT_FG, label=f"Structural mean={st_dist['mean']:.2f}×")
    ax2.axvline(2.0, color="#f1c40f", ls="--", alpha=0.6, label="ATR target=2.0")
    ax2.set_xlabel("SL distance (× ATR)"); ax2.set_ylabel("Count")
    ax2.set_title("SL Distance Distribution")
    ax2.legend(facecolor=PLT_BG, edgecolor=PLT_FG, fontsize=9)
    ax2.grid(True, alpha=0.3)

    # P3: Stop hunt rate
    ax3 = fig.add_subplot(2, 2, 3)
    labels = ["5 bars", "10 bars", "20 bars"]
    atr_h = [atr_hunt["hunt5"], atr_hunt["hunt10"], atr_hunt["hunt20"]]
    st_h = [st_hunt["hunt5"], st_hunt["hunt10"], st_hunt["hunt20"]]
    x = np.arange(len(labels))
    w = 0.38
    b1 = ax3.bar(x - w/2, atr_h, w, color="#c0392b", edgecolor=PLT_FG, label="ATR rr=1.5")
    b2 = ax3.bar(x + w/2, st_h, w, color="#27ae60", edgecolor=PLT_FG, label="Structural rr=2.5")
    for bars, vals in [(b1, atr_h), (b2, st_h)]:
        for b, v in zip(bars, vals):
            ax3.text(b.get_x() + b.get_width()/2, v + 0.4, f"{v:.1f}%",
                     ha="center", color=PLT_FG, fontweight="bold", fontsize=9)
    ax3.axhline(30, color="#f1c40f", ls="--", alpha=0.6, label="30% threshold")
    ax3.set_xticks(x); ax3.set_xticklabels(labels)
    ax3.set_ylabel("% SL trades recovered to entry")
    ax3.set_title("Stop Hunt Rate: ATR vs Structural")
    ax3.legend(facecolor=PLT_BG, edgecolor=PLT_FG, fontsize=9)
    ax3.grid(True, alpha=0.3)

    # P4: SL penetration side-by-side
    ax4 = fig.add_subplot(2, 2, 4)
    atr_prox = np.clip(np.array(atr_hunt["prox"]), 0, 2)
    st_prox = np.clip(np.array(st_hunt["prox"]), 0, 2)
    ax4.hist(atr_prox, bins=25, alpha=0.55, color="#e74c3c",
             edgecolor=PLT_FG, label=f"ATR cluster={atr_hunt['cluster']:.1f}%")
    ax4.hist(st_prox, bins=25, alpha=0.55, color="#3498db",
             edgecolor=PLT_FG, label=f"Structural cluster={st_hunt['cluster']:.1f}%")
    ax4.axvline(0.2, color="#f1c40f", ls="--", alpha=0.7, label="0.2 cluster threshold")
    ax4.set_xlabel("SL penetration (× SL distance)"); ax4.set_ylabel("Count")
    ax4.set_title("SL Penetration Depth Distribution")
    ax4.legend(facecolor=PLT_BG, edgecolor=PLT_FG, fontsize=9)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    path = OUTPUT_DIR / "structural_vs_atr_BTCUSDT.png"
    plt.savefig(path, dpi=120); plt.close()
    print(f"  saved: {path}")


def figure2(df, htf):
    """R:R sweep with structural stop."""
    _dark_style()
    struct_cfg = dict(stop_mode="STRUCTURAL", atr_sl_mult=2.0,
                      buffer_atr=0.25, min_risk_atr=0.8,
                      pivot_left=3, pivot_right=3)
    rrs = [1.5, 2.0, 2.5, 2.7, 3.0, 3.5]
    s1 = S_SCENARIOS["S1 paper"]; s2 = S_SCENARIOS["S2 real"]
    rows = []
    eq_s1 = {}
    for rr in rrs:
        m0 = fast_backtest_slip_struct(df, 0, 0, 0, 0, htf_bias=htf, rr_ratio=rr, **struct_cfg)
        m1 = fast_backtest_slip_struct(df, s1["entry"], s1["sl"], s1["tp"], 0,
                                        htf_bias=htf, rr_ratio=rr, **struct_cfg)
        m2 = fast_backtest_slip_struct(df, s2["entry"], s2["sl"], s2["tp"], 0,
                                        htf_bias=htf, rr_ratio=rr, **struct_cfg)
        hunt = stop_hunt_metrics(df, m0)
        pnls1 = np.array([t["pnl_pct"] for t in m1.trades])
        eq_s1[rr] = np.concatenate([[1.0], np.cumprod(1 + pnls1 / 100)])
        rows.append({"rr": rr, "s0": m0.sharpe_ratio, "s1": m1.sharpe_ratio,
                      "s2": m2.sharpe_ratio, "s2_expr": expectancy_r(m2),
                      "hunt10": hunt["hunt10"], "cluster": hunt["cluster"]})

    fig = plt.figure(figsize=(18, 14))
    fig.suptitle("R:R Sweep — Structural Stop (BTC/USDT 15m 730d)",
                 fontsize=17, fontweight="bold", color=PLT_FG)

    # P1 top wide: S0/S1/S2 Sharpe grouped bars
    ax1 = fig.add_subplot(3, 2, (1, 2))
    x = np.arange(len(rrs)); w = 0.25
    ax1.bar(x - w, [r["s0"] for r in rows], w, color="#27ae60", edgecolor=PLT_FG, label="S0 perfect")
    ax1.bar(x, [r["s1"] for r in rows], w, color="#3498db", edgecolor=PLT_FG, label="S1 paper")
    ax1.bar(x + w, [r["s2"] for r in rows], w, color="#e67e22", edgecolor=PLT_FG, label="S2 real")
    ax1.axhline(1.0, color="#f1c40f", ls="--", alpha=0.6, label="Sharpe=1.0")
    ax1.axhline(2.0, color="#27ae60", ls="--", alpha=0.5, label="Sharpe=2.0")
    ax1.set_xticks(x); ax1.set_xticklabels([str(r) for r in rrs])
    ax1.set_xlabel("R:R ratio"); ax1.set_ylabel("Sharpe")
    ax1.set_title("Sharpe by R:R — Structural Stop")
    ax1.legend(facecolor=PLT_BG, edgecolor=PLT_FG, fontsize=9)
    ax1.grid(True, alpha=0.3)

    # P2 middle-left: S2 ExpR
    ax2 = fig.add_subplot(3, 2, 3)
    expr = [r["s2_expr"] for r in rows]
    colors = ["#27ae60" if e > 0.05 else ("#f1c40f" if e > 0 else "#e74c3c") for e in expr]
    bars = ax2.bar([str(r) for r in rrs], expr, color=colors, edgecolor=PLT_FG)
    for b, v in zip(bars, expr):
        ax2.text(b.get_x() + b.get_width()/2, v + (0.005 if v >= 0 else -0.015),
                 f"{v:+.3f}", ha="center", color=PLT_FG, fontweight="bold", fontsize=9)
    ax2.axhline(0.05, color="#27ae60", ls="--", alpha=0.7, label="+0.05R threshold")
    ax2.axhline(0, color=PLT_FG, ls="-", alpha=0.3)
    ax2.set_xlabel("R:R ratio"); ax2.set_ylabel("S2 Expectancy (R)")
    ax2.set_title("S2 Expectancy R by R:R")
    ax2.legend(facecolor=PLT_BG, edgecolor=PLT_FG, fontsize=9)
    ax2.grid(True, alpha=0.3)

    # P3 middle-right: Hunt10 + Cluster lines
    ax3 = fig.add_subplot(3, 2, 4)
    ax3.plot([str(r) for r in rrs], [r["hunt10"] for r in rows],
             color="#e74c3c", marker="o", lw=2, label="Hunt10%")
    ax3.plot([str(r) for r in rrs], [r["cluster"] for r in rows],
             color="#e67e22", marker="s", lw=2, label="Cluster%")
    ax3.axhline(30, color="#f1c40f", ls="--", alpha=0.7, label="30% threshold")
    ax3.axhline(41.7, color="#e74c3c", ls=":", alpha=0.5, label="ATR Hunt10=41.7%")
    ax3.axhline(46.8, color="#e67e22", ls=":", alpha=0.5, label="ATR Cluster=46.8%")
    ax3.set_xlabel("R:R ratio"); ax3.set_ylabel("%")
    ax3.set_title("Stop Hunt (10b) + Cluster% by R:R")
    ax3.legend(facecolor=PLT_BG, edgecolor=PLT_FG, fontsize=8)
    ax3.grid(True, alpha=0.3)

    # P4 bottom: equity curves S1 by rr
    ax4 = fig.add_subplot(3, 1, 3)
    cmap = plt.cm.viridis
    for idx, rr in enumerate(rrs):
        ax4.plot(eq_s1[rr], color=cmap(idx / max(len(rrs) - 1, 1)),
                 lw=1.6, label=f"rr={rr}")
    ax4.axhline(1.0, color=PLT_FG, ls="--", alpha=0.3)
    ax4.set_xlabel("Trade #"); ax4.set_ylabel("Equity multiplier")
    ax4.set_title("S1 Equity Curves by R:R (realistic BingX fees)")
    ax4.legend(facecolor=PLT_BG, edgecolor=PLT_FG, fontsize=9)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    path = OUTPUT_DIR / "structural_rr_sweep_BTCUSDT.png"
    plt.savefig(path, dpi=120); plt.close()
    print(f"  saved: {path}")


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("[loading data]")
    df = load_candles("BTC/USDT", "15m", days=730)
    htf = compute_htf_bias(df, htf_ema_period=50)
    print(f"  bars: {len(df)}")
    print("\n[Figure 1] structural_vs_atr_BTCUSDT.png")
    figure1(df, htf)
    print("\n[Figure 2] structural_rr_sweep_BTCUSDT.png")
    figure2(df, htf)
    print("\n[Figure 3] skipped — NO CANDIDATE (per Critical Rule #6)")
