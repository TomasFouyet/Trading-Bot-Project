#!/usr/bin/env python3
"""
Regime Analysis — classify market conditions and measure strategy
performance per regime. Tests hypotheses about when the strategy
works and when it doesn't.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")

from validation.fast_backtest import compute_indicators, compute_htf_bias, fast_backtest
from validation.data_loader import load_candles

# ── Regime constants ────────────────────────────────────────────────
REGIME_TRENDING_STRONG = 0
REGIME_TRENDING_NORMAL = 1
REGIME_RANGING = 2
REGIME_HIGH_VOLATILITY = 3

REGIME_NAMES = {
    REGIME_TRENDING_STRONG: "TRENDING_STRONG",
    REGIME_TRENDING_NORMAL: "TRENDING_NORMAL",
    REGIME_RANGING:         "RANGING",
    REGIME_HIGH_VOLATILITY: "HIGH_VOLATILITY",
}

COMMISSION_PCT = 0.08


# ═══════════════════════════════════════════════════════════════════
# STEP 1 — Regime Classification
# ═══════════════════════════════════════════════════════════════════

def classify_regimes(df: pd.DataFrame) -> np.ndarray:
    """
    Classify each bar into a regime based on ADX and ATR percentile.

    Returns array of regime indices, same length as df.
    """
    close = df["close"].values.astype(np.float64)
    high = df["high"].values.astype(np.float64)
    low = df["low"].values.astype(np.float64)

    # ATR(14)
    tr = np.maximum(
        high - low,
        np.maximum(
            np.abs(high - np.roll(close, 1)),
            np.abs(low - np.roll(close, 1))
        )
    )
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).ewm(com=13, adjust=False).mean().values

    # ADX(14)
    up = np.diff(high, prepend=high[0])
    down = -np.diff(low, prepend=low[0])
    dm_p = np.where((up > down) & (up > 0), up, 0.0)
    dm_m = np.where((down > up) & (down > 0), down, 0.0)
    atr_safe = np.where(atr > 0, atr, 1e-10)
    di_p = 100 * pd.Series(dm_p).ewm(com=13, adjust=False).mean().values / atr_safe
    di_m = 100 * pd.Series(dm_m).ewm(com=13, adjust=False).mean().values / atr_safe
    di_sum = di_p + di_m
    di_sum_safe = np.where(di_sum > 0, di_sum, 1e-10)
    dx = 100 * np.abs(di_p - di_m) / di_sum_safe
    adx = pd.Series(dx).ewm(com=13, adjust=False).mean().values

    # ATR as % of price
    atr_pct = atr / np.where(close > 0, close, 1e-10) * 100

    # Rolling ATR percentile (100-bar window)
    atr_pct_series = pd.Series(atr_pct)
    atr_percentile = atr_pct_series.rolling(100, min_periods=20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
    ).values

    # Classify with priority: HIGH_VOL > TRENDING_STRONG > TRENDING_NORMAL > RANGING
    n = len(df)
    regimes = np.full(n, REGIME_RANGING, dtype=np.int8)

    for i in range(n):
        if np.isnan(atr_percentile[i]) or np.isnan(adx[i]):
            continue
        if atr_percentile[i] >= 80:
            regimes[i] = REGIME_HIGH_VOLATILITY
        elif adx[i] >= 30:
            regimes[i] = REGIME_TRENDING_STRONG
        elif adx[i] >= 20:
            regimes[i] = REGIME_TRENDING_NORMAL
        else:
            regimes[i] = REGIME_RANGING

    return regimes, adx, atr_pct, atr_percentile


def print_regime_distribution(regimes, adx, atr_pct):
    """Print regime distribution table."""
    print(f"\n{'='*70}")
    print("STEP 1 — REGIME DISTRIBUTION")
    print(f"{'='*70}")
    header = f"{'Regime':>20} | {'Bars':>7} | {'% time':>7} | {'Avg ADX':>8} | {'Avg ATR%':>8}"
    print(header)
    print("-" * len(header))

    n = len(regimes)
    for rid in [REGIME_TRENDING_STRONG, REGIME_TRENDING_NORMAL,
                REGIME_RANGING, REGIME_HIGH_VOLATILITY]:
        mask = regimes == rid
        count = int(mask.sum())
        pct = count / n * 100
        avg_adx = float(np.mean(adx[mask])) if count > 0 else 0
        avg_atr = float(np.mean(atr_pct[mask])) if count > 0 else 0
        print(f"{REGIME_NAMES[rid]:>20} | {count:>7} | {pct:>6.1f}% | {avg_adx:>8.1f} | {avg_atr:>7.2f}%")


# ═══════════════════════════════════════════════════════════════════
# STEP 2 — Performance by Regime
# ═══════════════════════════════════════════════════════════════════

def compute_regime_performance(trades, regimes, n_bars, tf_minutes=15.0):
    """Compute per-regime performance metrics from trade list."""
    results = {}

    for rid in [REGIME_TRENDING_STRONG, REGIME_TRENDING_NORMAL,
                REGIME_RANGING, REGIME_HIGH_VOLATILITY]:
        regime_trades = [t for t in trades if t.entry_bar_idx >= 0
                         and t.entry_bar_idx < len(regimes)
                         and regimes[t.entry_bar_idx] == rid]

        pnls = np.array([t.pnl_pct for t in regime_trades])
        n_trades = len(pnls)

        if n_trades == 0:
            results[rid] = {
                "n_trades": 0, "pct_total": 0, "winrate": 0,
                "avg_pnl": 0, "expectancy_r": 0, "sharpe": 0, "max_dd": 0,
            }
            continue

        wins = pnls[pnls > 0]
        losses = pnls[pnls <= 0]
        winrate = len(wins) / n_trades * 100
        avg_pnl = float(np.mean(pnls))
        avg_win = float(np.mean(wins)) if len(wins) > 0 else 0
        avg_loss = abs(float(np.mean(losses))) if len(losses) > 0 else 1
        expectancy_r = (
            (len(wins)/n_trades * avg_win / avg_loss) -
            (len(losses)/n_trades) -
            (COMMISSION_PCT / avg_loss)
        ) if avg_loss > 0 else 0

        # Sharpe
        sharpe = 0.0
        if n_trades >= 2:
            mean_r = np.mean(pnls)
            std_r = np.std(pnls, ddof=1)
            if std_r > 0:
                avg_bars = np.mean([t.bars_held for t in regime_trades])
                bars_per_year = 365 * 24 * 60 / tf_minutes
                trades_per_year = bars_per_year / max(avg_bars, 1)
                sharpe = float((mean_r / std_r) * np.sqrt(trades_per_year))

        # Max DD (compounded)
        mult = 1.0 + pnls / 100.0
        equity = np.concatenate([[1.0], np.cumprod(mult)])
        running_max = np.maximum.accumulate(equity)
        dd = (equity - running_max) / running_max
        max_dd = abs(float(np.min(dd))) * 100

        results[rid] = {
            "n_trades": n_trades,
            "pct_total": n_trades / len(trades) * 100,
            "winrate": winrate,
            "avg_pnl": avg_pnl,
            "expectancy_r": expectancy_r,
            "sharpe": sharpe,
            "max_dd": max_dd,
        }

    return results


def print_performance_table(perf, total_metrics):
    """Print performance by regime table."""
    print(f"\n{'='*90}")
    print("STEP 2 — PERFORMANCE BY REGIME")
    print(f"{'='*90}")
    header = (f"{'Regime':>20} | {'Trades':>6} | {'% tot':>5} | {'WR%':>5} | "
              f"{'Avg PnL':>8} | {'Exp R':>8} | {'Sharpe':>7} | {'Max DD%':>8}")
    print(header)
    print("-" * len(header))

    for rid in [REGIME_TRENDING_STRONG, REGIME_TRENDING_NORMAL,
                REGIME_RANGING, REGIME_HIGH_VOLATILITY]:
        r = perf[rid]
        print(f"{REGIME_NAMES[rid]:>20} | {r['n_trades']:>6} | {r['pct_total']:>4.1f}% | "
              f"{r['winrate']:>4.1f}% | {r['avg_pnl']:>+7.3f}% | "
              f"{r['expectancy_r']:>+8.4f} | {r['sharpe']:>7.2f} | {r['max_dd']:>7.1f}%")

    print("-" * len(header))
    print(f"{'TOTAL':>20} | {total_metrics.total_trades:>6} | {'100':>4}.0% | "
          f"{total_metrics.winrate:>4.1f}% | {total_metrics.avg_pnl_pct:>+7.3f}% | "
          f"{'':>8} | {total_metrics.sharpe_ratio:>7.2f} | {total_metrics.max_drawdown_pct:>7.1f}%")


def test_hypotheses(perf):
    """Test H1, H2, H3 and return results dict."""
    results = {}

    # H1: RANGING loses money
    ranging = perf[REGIME_RANGING]
    if ranging["n_trades"] < 5:
        results["H1"] = ("INSUFFICIENT DATA", f"Only {ranging['n_trades']} trades in RANGING")
    elif ranging["sharpe"] < 0 and ranging["expectancy_r"] < 0:
        results["H1"] = ("CONFIRMED", f"RANGING Sharpe={ranging['sharpe']:.2f}, Exp={ranging['expectancy_r']:+.4f}R")
    elif ranging["sharpe"] < 1.0:
        results["H1"] = ("PARTIAL", f"RANGING Sharpe={ranging['sharpe']:.2f} (positive but weak)")
    else:
        results["H1"] = ("REJECTED", f"RANGING Sharpe={ranging['sharpe']:.2f} >= 1.0")

    # H2: TRENDING_STRONG outperforms TRENDING_NORMAL
    strong = perf[REGIME_TRENDING_STRONG]
    normal = perf[REGIME_TRENDING_NORMAL]
    if strong["n_trades"] < 5 or normal["n_trades"] < 5:
        results["H2"] = ("INSUFFICIENT DATA", f"Strong={strong['n_trades']}, Normal={normal['n_trades']} trades")
    elif strong["sharpe"] > normal["sharpe"]:
        results["H2"] = ("CONFIRMED", f"Strong Sharpe={strong['sharpe']:.2f} > Normal Sharpe={normal['sharpe']:.2f}")
    else:
        results["H2"] = ("REJECTED", f"Strong Sharpe={strong['sharpe']:.2f} <= Normal Sharpe={normal['sharpe']:.2f}")

    # H3: HIGH_VOLATILITY degrades performance
    highvol = perf[REGIME_HIGH_VOLATILITY]
    if highvol["n_trades"] < 5:
        results["H3"] = ("INSUFFICIENT DATA", f"Only {highvol['n_trades']} trades in HIGH_VOL")
    elif highvol["sharpe"] < normal["sharpe"]:
        results["H3"] = ("CONFIRMED", f"HighVol Sharpe={highvol['sharpe']:.2f} < Normal Sharpe={normal['sharpe']:.2f}")
    else:
        results["H3"] = ("REJECTED", f"HighVol Sharpe={highvol['sharpe']:.2f} >= Normal Sharpe={normal['sharpe']:.2f}")

    print(f"\n{'='*70}")
    print("HYPOTHESIS TESTING")
    print(f"{'='*70}")
    for h, (verdict, detail) in results.items():
        print(f"  {h}: {verdict}")
        print(f"       {detail}")

    return results


# ═══════════════════════════════════════════════════════════════════
# STEP 3 — Dynamic R:R / Regime Filter
# ═══════════════════════════════════════════════════════════════════

def run_comparison(df, htf_bias, regimes, baseline_metrics, hyp_results):
    """Run regime-filtered backtests and compare to baseline."""
    comparisons = {}
    h1_confirmed = hyp_results["H1"][0] in ("CONFIRMED", "PARTIAL")
    h2_confirmed = hyp_results["H2"][0] == "CONFIRMED"

    if not h1_confirmed and not h2_confirmed:
        print(f"\n{'='*70}")
        print("STEP 3 — SKIPPED (neither H1 nor H2 confirmed)")
        print(f"{'='*70}")
        return comparisons

    print(f"\n{'='*70}")
    print("STEP 3 — REGIME FILTER COMPARISON")
    print(f"{'='*70}")

    df_ind = compute_indicators(df, ema_fast_p=20, ema_slow_p=50)

    # Test A: Skip RANGING
    if h1_confirmed:
        m_no_range = fast_backtest(
            df_ind, rr_ratio=1.5, atr_sl_mult=2.0,
            precomputed=True, htf_bias=htf_bias,
            regime_labels=regimes, regime_skip={REGIME_RANGING},
        )
        comparisons["no_ranging"] = m_no_range

    # Test B: Dynamic R:R (only if H2 confirmed)
    if h2_confirmed:
        rr_map = {
            REGIME_TRENDING_STRONG: 2.0,
            REGIME_TRENDING_NORMAL: 1.5,
            REGIME_HIGH_VOLATILITY: 1.2,
        }
        skip = {REGIME_RANGING} if h1_confirmed else set()
        m_dynamic = fast_backtest(
            df_ind, rr_ratio=1.5, atr_sl_mult=2.0,
            precomputed=True, htf_bias=htf_bias,
            regime_labels=regimes, regime_skip=skip, regime_rr_map=rr_map,
        )
        comparisons["dynamic_rr"] = m_dynamic

    # Print comparison table
    bl = baseline_metrics
    header = f"{'Metric':>20} | {'Baseline':>12} | "
    labels = []
    if "no_ranging" in comparisons:
        header += f"{'No-Ranging':>12} | {'Delta':>8} | "
        labels.append("no_ranging")
    if "dynamic_rr" in comparisons:
        header += f"{'Dynamic RR':>12} | {'Delta':>8}"
        labels.append("dynamic_rr")
    print(header)
    print("-" * len(header))

    def row(name, bl_val, comp_dict, fmt=".1f"):
        line = f"{name:>20} | {bl_val:>12{fmt}} | "
        for label in labels:
            m = comp_dict.get(label)
            if m is not None:
                val = getattr(m, {
                    "Total trades": "total_trades",
                    "Win rate %": "winrate",
                    "Annual return %": "annual_return_pct",
                    "Sharpe": "sharpe_ratio",
                    "Max DD %": "max_drawdown_pct",
                }.get(name, ""), 0)
                delta = val - bl_val
                line += f"{val:>12{fmt}} | {delta:>+8{fmt}} | "
            else:
                line += f"{'N/A':>12} | {'':>8} | "
        print(line)

    metrics_map = [
        ("Total trades", bl.total_trades, "total_trades", "d"),
        ("Win rate %", bl.winrate, "winrate", ".1f"),
        ("Annual return %", bl.annual_return_pct, "annual_return_pct", ".1f"),
        ("Sharpe", bl.sharpe_ratio, "sharpe_ratio", ".2f"),
        ("Max DD %", bl.max_drawdown_pct, "max_drawdown_pct", ".1f"),
    ]

    for name, bl_val, attr, fmt in metrics_map:
        line = f"{name:>20} | {bl_val:>12{fmt}} | "
        for label in labels:
            m = comparisons[label]
            val = getattr(m, attr)
            delta = val - bl_val
            line += f"{val:>12{fmt}} | {delta:>+8{fmt}} | "
        print(line)

    # Check if improvement > 15%
    for label in labels:
        m = comparisons[label]
        sharpe_improvement = (m.sharpe_ratio - bl.sharpe_ratio) / max(abs(bl.sharpe_ratio), 0.01) * 100
        print(f"\n  {label}: Sharpe improvement = {sharpe_improvement:+.1f}%", end="")
        if sharpe_improvement > 15:
            print(" --> SIGNIFICANT (>15%)")
        else:
            print(" --> MARGINAL (<= 15%)")

    return comparisons


# ═══════════════════════════════════════════════════════════════════
# STEP 4 — Regime Timeline Plot
# ═══════════════════════════════════════════════════════════════════

def plot_regime_timeline(df, regimes, adx, atr_percentile, trades, save_path):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    ts = pd.DatetimeIndex(df["ts"])
    close = df["close"].values
    n = len(df)

    # Downsample for plotting (every 96 bars = 1 day)
    step = 96
    idx = np.arange(0, n, step)
    ts_plot = ts[idx]
    close_plot = close[idx]
    adx_plot = adx[idx]
    atr_pct_plot = atr_percentile[idx]
    regimes_plot = regimes[idx]

    regime_colors = {
        REGIME_TRENDING_STRONG: "#2ecc71",
        REGIME_TRENDING_NORMAL: "#3498db",
        REGIME_RANGING:         "#bdc3c7",
        REGIME_HIGH_VOLATILITY: "#e74c3c",
    }

    fig, axes = plt.subplots(4, 1, figsize=(16, 14), sharex=True)

    # Plot 1: Price with regime background
    ax1 = axes[0]
    ax1.plot(ts_plot, close_plot, color="black", linewidth=0.8, label="BTC Close")
    # Color background by regime
    for i in range(len(idx) - 1):
        ax1.axvspan(ts_plot[i], ts_plot[i+1],
                     alpha=0.3, color=regime_colors[regimes_plot[i]], linewidth=0)
    ax1.set_ylabel("Price (USDT)")
    ax1.set_title("BTC/USDT 15m — Regime Classification")
    legend_patches = [Patch(facecolor=regime_colors[rid], alpha=0.4, label=REGIME_NAMES[rid])
                      for rid in [REGIME_TRENDING_STRONG, REGIME_TRENDING_NORMAL,
                                  REGIME_RANGING, REGIME_HIGH_VOLATILITY]]
    ax1.legend(handles=legend_patches, fontsize=8, loc="upper left")

    # Plot 2: ADX
    ax2 = axes[1]
    ax2.plot(ts_plot, adx_plot, color="purple", linewidth=0.8)
    ax2.axhline(20, color="orange", linestyle="--", linewidth=0.8, label="ADX=20")
    ax2.axhline(30, color="red", linestyle="--", linewidth=0.8, label="ADX=30")
    ax2.set_ylabel("ADX(14)")
    ax2.legend(fontsize=8)
    ax2.set_title("ADX Over Time")

    # Plot 3: ATR percentile
    ax3 = axes[2]
    ax3.plot(ts_plot, atr_pct_plot, color="brown", linewidth=0.8)
    ax3.axhline(80, color="red", linestyle="--", linewidth=0.8, label="P80 threshold")
    ax3.set_ylabel("ATR Percentile")
    ax3.legend(fontsize=8)
    ax3.set_title("ATR Percentile (100-bar rolling)")

    # Plot 4: Cumulative PnL by regime
    ax4 = axes[3]
    # Build per-regime equity curves
    for rid in [REGIME_TRENDING_STRONG, REGIME_TRENDING_NORMAL,
                REGIME_RANGING, REGIME_HIGH_VOLATILITY]:
        regime_trades_sorted = sorted(
            [t for t in trades if t.entry_bar_idx >= 0
             and t.entry_bar_idx < len(regimes)
             and regimes[t.entry_bar_idx] == rid],
            key=lambda t: t.entry_bar_idx
        )
        if not regime_trades_sorted:
            continue
        cum_pnl = np.cumsum([t.pnl_pct for t in regime_trades_sorted])
        trade_ts = [ts[min(t.entry_bar_idx, n-1)] for t in regime_trades_sorted]
        ax4.plot(trade_ts, cum_pnl, color=regime_colors[rid],
                 linewidth=1.2, label=f"{REGIME_NAMES[rid]} ({len(regime_trades_sorted)} trades)")

    ax4.set_ylabel("Cumulative PnL %")
    ax4.set_xlabel("Date")
    ax4.legend(fontsize=8)
    ax4.set_title("Cumulative PnL by Entry Regime")
    ax4.axhline(0, color="gray", linestyle="-", linewidth=0.5)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"[Regime] Plot saved to {save_path}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════
# STEP 5 — Trading Rules
# ═══════════════════════════════════════════════════════════════════

def print_trading_rules(perf, hyp_results, comparisons, baseline_sharpe):
    """Print actionable trading rules box."""

    # Determine actions per regime
    h1_confirmed = hyp_results["H1"][0] in ("CONFIRMED", "PARTIAL")
    h2_confirmed = hyp_results["H2"][0] == "CONFIRMED"
    h3_confirmed = hyp_results["H3"][0] == "CONFIRMED"

    # Check if any filter improves Sharpe > 15%
    best_label = None
    best_improvement = 0
    for label, m in comparisons.items():
        imp = (m.sharpe_ratio - baseline_sharpe) / max(abs(baseline_sharpe), 0.01) * 100
        if imp > best_improvement:
            best_improvement = imp
            best_label = label

    adopt = best_improvement > 15

    # Build rules
    rules = {}
    if adopt and best_label == "dynamic_rr":
        rules[REGIME_TRENDING_STRONG] = ("TRADE", "2.0", "Strong trend supports wider TP target")
        rules[REGIME_TRENDING_NORMAL] = ("TRADE", "1.5", "Validated baseline, reliable edge")
        rules[REGIME_RANGING] = ("SKIP", "N/A", f"Negative/weak edge (Sharpe={perf[REGIME_RANGING]['sharpe']:.2f})")
        rules[REGIME_HIGH_VOLATILITY] = ("TRADE", "1.2", "Tighter TP to capture gains before reversal")
    elif adopt and best_label == "no_ranging":
        rules[REGIME_TRENDING_STRONG] = ("TRADE", "1.5", "Strong trend, baseline rr works")
        rules[REGIME_TRENDING_NORMAL] = ("TRADE", "1.5", "Validated baseline")
        rules[REGIME_RANGING] = ("SKIP", "N/A", f"Negative/weak edge (Sharpe={perf[REGIME_RANGING]['sharpe']:.2f})")
        rules[REGIME_HIGH_VOLATILITY] = ("TRADE", "1.5", "Baseline rr, monitor closely")
    else:
        # Keep baseline simple
        rules[REGIME_TRENDING_STRONG] = ("TRADE", "1.5", "Baseline — complexity not justified")
        rules[REGIME_TRENDING_NORMAL] = ("TRADE", "1.5", "Baseline — validated")
        rules[REGIME_RANGING] = (
            "SKIP" if h1_confirmed else "TRADE",
            "N/A" if h1_confirmed else "1.5",
            f"Sharpe={perf[REGIME_RANGING]['sharpe']:.2f}" + (" — skip recommended" if h1_confirmed else "")
        )
        rules[REGIME_HIGH_VOLATILITY] = ("TRADE", "1.5", "Baseline")

    adopt_label = "YES" if adopt else ("PARTIAL" if h1_confirmed else "NO")
    complexity = "LOW" if adopt_label == "PARTIAL" else ("MEDIUM" if adopt else "LOW")
    sharpe_str = f"+{best_improvement:.0f}%" if best_improvement > 15 else '"marginal"'

    print(f"\n")
    print(f"  {'='*54}")
    print(f"  |{'REGIME-BASED TRADING RULES - BTC/USDT':^53}|")
    print(f"  {'='*54}")
    for rid in [REGIME_TRENDING_STRONG, REGIME_TRENDING_NORMAL,
                REGIME_RANGING, REGIME_HIGH_VOLATILITY]:
        action, rr, rationale = rules[rid]
        thresholds = {
            REGIME_TRENDING_STRONG: "ADX>=30, ATR<P80",
            REGIME_TRENDING_NORMAL: "ADX 20-30, ATR<P80",
            REGIME_RANGING: "ADX<20",
            REGIME_HIGH_VOLATILITY: "ATR>=P80",
        }[rid]
        print(f"  |{'':-<53}|")
        print(f"  | {REGIME_NAMES[rid]} ({thresholds}){' '*(53-len(REGIME_NAMES[rid])-len(thresholds)-4)}|")
        print(f"  |   Action:    {action:<39}|")
        print(f"  |   rr_ratio:  {rr:<39}|")
        print(f"  |   Rationale: {rationale:<39}|")

    print(f"  |{'':-<53}|")
    print(f"  | {'FINAL VERDICT':<52}|")
    print(f"  |   Adopt regime filter: {adopt_label:<29}|")
    print(f"  |   Expected Sharpe improvement: {sharpe_str:<21}|")
    print(f"  |   Complexity added: {complexity:<32}|")
    print(f"  {'='*54}")

    return adopt_label, best_improvement


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    t_start = time.time()

    print("=" * 70)
    print("REGIME ANALYSIS — BTC/USDT 15m, HTF=ON, rr=1.5, sl_atr=2.0")
    print("=" * 70)

    df = load_candles("BTC/USDT", "15m", days=730)
    print(f"Data: {len(df)} bars\n")

    # ── Step 1: Regime classification ────────────────────────────
    regimes, adx, atr_pct, atr_percentile = classify_regimes(df)
    print_regime_distribution(regimes, adx, atr_pct)

    # ── Baseline backtest ────────────────────────────────────────
    htf_bias = compute_htf_bias(df)
    df_ind = compute_indicators(df, ema_fast_p=20, ema_slow_p=50)
    baseline = fast_backtest(
        df_ind, rr_ratio=1.5, atr_sl_mult=2.0,
        precomputed=True, htf_bias=htf_bias,
    )
    print(f"\nBaseline: {baseline.total_trades} trades, Sharpe={baseline.sharpe_ratio:.2f}, "
          f"WR={baseline.winrate:.1f}%, Ann={baseline.annual_return_pct:+.1f}%")

    # Get TradeRecord objects (not dicts) for regime tagging
    # Re-run to get TradeRecord list — fast_backtest returns BacktestMetrics
    # which has .trades as dicts. Need to re-extract TradeRecords.
    # Hack: re-run and capture from _build_metrics input
    from validation.strategy_adapter import TradeRecord as TR

    # Reconstruct TradeRecords from the dict list
    trade_records = []
    for td in baseline.trades:
        tr = TR(
            direction=td["direction"],
            entry_price=td["entry_price"],
            exit_price=td["exit_price"],
            pnl_pct=td["pnl_pct"],
            exit_type=td["exit_type"],
            bars_held=td["bars_held"],
            entry_bar_idx=td.get("entry_bar_idx", -1),
        )
        trade_records.append(tr)

    # ── Step 2: Performance by regime ────────────────────────────
    perf = compute_regime_performance(trade_records, regimes, len(df))
    print_performance_table(perf, baseline)
    hyp_results = test_hypotheses(perf)

    # ── Step 3: Dynamic R:R / Regime filter ──────────────────────
    comparisons = run_comparison(df, htf_bias, regimes, baseline, hyp_results)

    # ── Step 4: Plot ─────────────────────────────────────────────
    output_dir = ROOT / "validation" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_regime_timeline(df, regimes, adx, atr_percentile, trade_records,
                         str(output_dir / "regime_analysis_BTCUSDT.png"))

    # ── Step 5: Trading rules ────────────────────────────────────
    adopt_label, improvement = print_trading_rules(
        perf, hyp_results, comparisons, baseline.sharpe_ratio
    )

    total_time = time.time() - t_start
    print(f"\nTotal time: {total_time:.0f}s ({total_time/60:.1f} min)")

    # ── Final line ───────────────────────────────────────────────
    print(f"\n{'='*70}")
    if adopt_label == "YES":
        print(f"Regime filter ADOPTED — Sharpe improvement of +{improvement:.0f}% justifies the added complexity.")
    elif adopt_label == "PARTIAL":
        print(f"Regime filter PARTIALLY ADOPTED — skip RANGING regime only, keep rr=1.5 fixed everywhere else.")
    else:
        print(f"Regime filter NOT ADOPTED — improvement is marginal, keep baseline simple.")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
