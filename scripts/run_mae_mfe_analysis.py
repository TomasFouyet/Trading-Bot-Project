#!/usr/bin/env python3
"""
MAE/MFE/Duration Analysis — Descriptive statistics on trade excursions.

Re-runs the validated backtest (rr=1.5, atr_sl=2.0, HTF=ON, 730d)
with per-bar MAE/MFE tracking, then analyses:
  1. Duration — do late trades have worse outcomes?
  2. MAE — does adverse excursion predict failure?
  3. MFE — are significant gains being wasted (reversed to SL)?
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from validation.data_loader import load_candles
from validation.fast_backtest import compute_indicators, compute_htf_bias

OUTPUT_DIR = ROOT / "validation" / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BASELINE_WR = 47.6  # from prior full backtest


# ═══════════════════════════════════════════════════════════════════════════════
# Backtest with MAE/MFE tracking
# ═══════════════════════════════════════════════════════════════════════════════

def backtest_with_mae_mfe(
    df: pd.DataFrame,
    htf_bias: np.ndarray,
    rr_ratio: float = 1.5,
    atr_sl_mult: float = 2.0,
    adx_min: float = 20.0,
    sig_cooldown: int = 5,
    pb_tol_atr: float = 1.0,
    allow_short: bool = True,
    min_confidence: float = 0.0,
    adx_strong: float = 35.0,
) -> list[dict]:
    """
    Identical entry/exit logic to fast_backtest, but tracks
    per-bar MAE/MFE during each trade.
    """
    from validation.fast_backtest import _fast_confidence

    close = df["close"].values.astype(np.float64)
    high = df["high"].values.astype(np.float64)
    low = df["low"].values.astype(np.float64)
    opn = df["open"].values.astype(np.float64)
    atr = df["atr"].values.astype(np.float64)
    adx = df["adx"].values.astype(np.float64)
    ema_f = df["ema_fast"].values.astype(np.float64)
    ema_s = df["ema_slow"].values.astype(np.float64)
    ema_s_slope = df["ema_slow_slope"].values.astype(np.float64)
    ema_f_slope = df["ema_fast_slope"].values.astype(np.float64)
    macd_v = df["macd"].values.astype(np.float64)
    macd_sig = df["macd_signal"].values.astype(np.float64)
    macd_hist = df["macd_hist"].values.astype(np.float64)
    volume = df["volume"].values.astype(np.float64)
    vol_sma = df["vol_sma"].values.astype(np.float64)
    n = len(df)
    min_bars = 70  # max(60, ema_slow_p + 20)

    trades = []
    in_trade = False
    trade_dir = 0
    entry_price = 0.0
    sl_price = 0.0
    tp_price = 0.0
    sl_dist = 0.0
    entry_bar = 0
    mae = 0.0  # most adverse excursion (negative for longs going down)
    mfe = 0.0  # most favorable excursion (positive for longs going up)
    bar_at_mfe = 0

    prev_long_sig = False
    prev_short_sig = False
    last_long_bar = -999
    last_short_bar = -999

    for i in range(min_bars, n):
        c = close[i]
        h = high[i]
        lo = low[i]
        a = atr[i]
        dx = adx[i]

        if np.isnan(dx) or np.isnan(ema_s[i]) or a <= 0:
            continue

        # ── Check exits first ─────────────────────────────────────
        if in_trade:
            # Update MAE/MFE
            if trade_dir == 1:  # LONG
                excursion_low = lo - entry_price
                excursion_high = h - entry_price
                if excursion_low < mae:
                    mae = excursion_low
                if excursion_high > mfe:
                    mfe = excursion_high
                    bar_at_mfe = i - entry_bar
            else:  # SHORT
                excursion_low = entry_price - h  # adverse = price goes up
                excursion_high = entry_price - lo  # favorable = price goes down
                if excursion_low < mae:
                    mae = excursion_low
                if excursion_high > mfe:
                    mfe = excursion_high
                    bar_at_mfe = i - entry_bar

            sl_hit = (trade_dir == 1 and lo <= sl_price) or \
                     (trade_dir == -1 and h >= sl_price)
            tp_hit = (trade_dir == 1 and h >= tp_price) or \
                     (trade_dir == -1 and lo <= tp_price)

            if sl_hit:
                pnl = _pnl(trade_dir, entry_price, sl_price)
                trades.append(_make_trade(
                    trade_dir, entry_price, sl_price, pnl, "sl",
                    i - entry_bar, entry_bar, sl_dist, mae, mfe, bar_at_mfe))
                in_trade = False
                continue
            elif tp_hit:
                pnl = _pnl(trade_dir, entry_price, tp_price)
                trades.append(_make_trade(
                    trade_dir, entry_price, tp_price, pnl, "tp",
                    i - entry_bar, entry_bar, sl_dist, mae, mfe, bar_at_mfe))
                in_trade = False
                continue

        # ── Entry conditions ──────────────────────────────────────
        pb_zone = abs(c - ema_f[i]) < a * pb_tol_atr
        sl_rising = ema_s_slope[i] > 0 if not np.isnan(ema_s_slope[i]) else False
        sl_falling = ema_s_slope[i] < 0 if not np.isnan(ema_s_slope[i]) else False
        p_above = c > ema_s[i]
        p_below = c < ema_s[i]
        m_bull = macd_v[i] > macd_sig[i]
        m_bear = macd_v[i] < macd_sig[i]
        c_bull = c > opn[i]
        c_bear = c < opn[i]
        adx_ok = dx >= adx_min

        long_base = adx_ok and sl_rising and p_above and m_bull and pb_zone and c_bull
        short_base = adx_ok and sl_falling and p_below and m_bear and pb_zone and c_bear and allow_short

        conf_l = _fast_confidence(dx, adx_strong, a, c, ema_f[i],
                                  macd_hist[i], macd_hist[i-1] if i > 0 else 0,
                                  "LONG", opn[i], h, lo,
                                  ema_f_slope[i], volume[i], vol_sma[i]) if long_base else 0.0
        conf_s = _fast_confidence(dx, adx_strong, a, c, ema_f[i],
                                  macd_hist[i], macd_hist[i-1] if i > 0 else 0,
                                  "SHORT", opn[i], h, lo,
                                  ema_f_slope[i], volume[i], vol_sma[i]) if short_base else 0.0

        long_signal = long_base and conf_l >= min_confidence
        short_signal = short_base and conf_s >= min_confidence

        long_trigger_raw = long_signal and not prev_long_sig
        short_trigger_raw = short_signal and not prev_short_sig
        prev_long_sig = long_signal
        prev_short_sig = short_signal

        long_trigger = long_trigger_raw and (i - last_long_bar) >= sig_cooldown
        short_trigger = short_trigger_raw and (i - last_short_bar) >= sig_cooldown
        if long_trigger:
            last_long_bar = i
        if short_trigger:
            last_short_bar = i

        # HTF filter
        if htf_bias is not None:
            bias = htf_bias[i]
            if long_trigger and bias == -1:
                long_trigger = False
            if short_trigger and bias == 1:
                short_trigger = False

        # Open trade
        if not in_trade:
            if long_trigger:
                sl_dist = a * atr_sl_mult
                entry_price = c
                sl_price = c - sl_dist
                tp_price = c + sl_dist * rr_ratio
                trade_dir = 1
                entry_bar = i
                in_trade = True
                mae = 0.0
                mfe = 0.0
                bar_at_mfe = 0
            elif short_trigger:
                sl_dist = a * atr_sl_mult
                entry_price = c
                sl_price = c + sl_dist
                tp_price = c - sl_dist * rr_ratio
                trade_dir = -1
                entry_bar = i
                in_trade = True
                mae = 0.0
                mfe = 0.0
                bar_at_mfe = 0

    # Close open trade at end (exclude from analysis)
    if in_trade:
        pnl = _pnl(trade_dir, entry_price, close[-1])
        trades.append(_make_trade(
            trade_dir, entry_price, float(close[-1]), pnl, "end_of_data",
            n - 1 - entry_bar, entry_bar, sl_dist, mae, mfe, bar_at_mfe))

    return trades


def _pnl(direction: int, entry: float, exit_p: float) -> float:
    if direction == 1:
        return (exit_p - entry) / entry * 100
    else:
        return (entry - exit_p) / entry * 100


def _make_trade(trade_dir, entry_price, exit_price, pnl, exit_type,
                bars_held, entry_bar, sl_dist, mae, mfe, bar_at_mfe) -> dict:
    mae_r = mae / sl_dist if sl_dist > 0 else 0.0
    mfe_r = mfe / sl_dist if sl_dist > 0 else 0.0
    return {
        "direction": "LONG" if trade_dir == 1 else "SHORT",
        "entry_price": entry_price,
        "exit_price": exit_price,
        "pnl_pct": pnl,
        "exit_type": exit_type,
        "bars_held": bars_held,
        "entry_bar_idx": entry_bar,
        "sl_dist": sl_dist,
        "mae": mae,
        "mfe": mfe,
        "mae_r": mae_r,
        "mfe_r": mfe_r,
        "bar_at_mfe": bar_at_mfe,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Analysis functions
# ═══════════════════════════════════════════════════════════════════════════════

def duration_analysis(trades: list[dict]):
    """Step 2: Duration analysis."""
    winners = [t for t in trades if t["exit_type"] == "tp"]
    losers = [t for t in trades if t["exit_type"] == "sl"]

    print("\n" + "=" * 70)
    print("STEP 2a — DURATION STATISTICS")
    print("=" * 70)

    def stats(tlist, label):
        bars = np.array([t["bars_held"] for t in tlist])
        return {
            "label": label, "count": len(bars),
            "avg": np.mean(bars), "median": np.median(bars),
            "p25": np.percentile(bars, 25), "p75": np.percentile(bars, 75),
            "p95": np.percentile(bars, 95),
            "min": np.min(bars), "max": np.max(bars),
        }

    sw = stats(winners, "Winners (TP)")
    sl = stats(losers, "Losers (SL)")

    print(f"\n  {'Metric':<20} {'Winners (TP)':>14} {'Losers (SL)':>14}")
    print(f"  {'-'*48}")
    for key in ["count", "avg", "median", "p25", "p75", "p95", "min", "max"]:
        fmt = ".0f" if key == "count" else ".1f"
        print(f"  {key.capitalize():<20} {sw[key]:>14{fmt}} {sl[key]:>14{fmt}}")

    # Step 2b — Conditional probability by duration (survival analysis)
    print(f"\n{'=' * 70}")
    print("STEP 2b — CONDITIONAL PROBABILITY BY DURATION")
    print("=" * 70)

    buckets = [(1, 5), (6, 10), (11, 15), (16, 20), (21, 25),
               (26, 30), (31, 40), (41, 50), (51, 999)]

    print(f"\n  {'Bars open':<12} {'Active':>8} {'Won(TP)':>8} {'Lost(SL)':>8} {'P(TP)':>8} {'P(SL)':>8}")
    print(f"  {'-'*52}")

    structural_break = None
    bucket_results = []

    for lo_b, hi_b in buckets:
        # Trades still active at start of this bucket = resolved at or after lo_b
        active = [t for t in trades if t["bars_held"] >= lo_b and t["exit_type"] in ("tp", "sl")]
        # Resolved within this bucket
        resolved = [t for t in trades if lo_b <= t["bars_held"] <= hi_b and t["exit_type"] in ("tp", "sl")]
        won = sum(1 for t in resolved if t["exit_type"] == "tp")
        lost = sum(1 for t in resolved if t["exit_type"] == "sl")
        total_resolved = won + lost
        p_tp = won / total_resolved * 100 if total_resolved > 0 else 0
        p_sl = lost / total_resolved * 100 if total_resolved > 0 else 0
        label = f"{lo_b}-{hi_b}" if hi_b < 999 else f"{lo_b}+"

        sufficient = total_resolved >= 15
        marker = ""
        if not sufficient and total_resolved > 0:
            marker = " *"
        if sufficient and p_sl > 60 and structural_break is None:
            structural_break = (lo_b, p_sl, total_resolved)
            marker = " <-- BREAK"

        bucket_results.append({
            "lo": lo_b, "hi": hi_b, "active": len(active),
            "won": won, "lost": lost, "total": total_resolved,
            "p_tp": p_tp, "p_sl": p_sl, "sufficient": sufficient,
        })

        print(f"  {label:<12} {len(active):>8} {won:>8} {lost:>8} {p_tp:>7.1f}% {p_sl:>7.1f}%{marker}")

    print(f"\n  (* = fewer than 15 trades, insufficient for conclusions)")

    if structural_break:
        print(f"\n  STRUCTURAL BREAK at bar {structural_break[0]}: P(SL)={structural_break[1]:.1f}% with {structural_break[2]} trades")
    else:
        print(f"\n  NO STRUCTURAL BREAK found — duration timeout not justified")

    return sw, sl, bucket_results, structural_break


def mae_analysis(trades: list[dict]):
    """Step 3: MAE analysis."""
    valid = [t for t in trades if t["exit_type"] in ("tp", "sl")]
    winners = [t for t in valid if t["exit_type"] == "tp"]
    losers = [t for t in valid if t["exit_type"] == "sl"]

    print(f"\n{'=' * 70}")
    print("STEP 3a — MAE DISTRIBUTION")
    print("=" * 70)

    mae_buckets = [
        (0.0, -0.2, "0 to -0.2R"),
        (-0.2, -0.4, "-0.2 to -0.4R"),
        (-0.4, -0.6, "-0.4 to -0.6R"),
        (-0.6, -0.8, "-0.6 to -0.8R"),
        (-0.8, -1.0, "-0.8 to -1.0R"),
        (-1.0, -99, "> -1.0R"),
    ]

    print(f"\n  {'MAE bucket (R)':<18} {'Winners':>8} {'Losers':>8} {'P(TP|MAE)':>10}")
    print(f"  {'-'*44}")

    soft_sl = None
    mae_bucket_results = []

    for hi_r, lo_r, label in mae_buckets:
        w = sum(1 for t in winners if lo_r < t["mae_r"] <= hi_r)
        l = sum(1 for t in losers if lo_r < t["mae_r"] <= hi_r)
        total = w + l
        p_tp = w / total * 100 if total > 0 else 0
        sufficient = total >= 15
        marker = ""
        if not sufficient and total > 0:
            marker = " *"
        if sufficient and p_tp < 25 and soft_sl is None:
            soft_sl = (lo_r, hi_r, p_tp, total)
            marker = " <-- SOFT SL"

        mae_bucket_results.append({
            "label": label, "hi_r": hi_r, "lo_r": lo_r,
            "winners": w, "losers": l, "total": total,
            "p_tp": p_tp, "sufficient": sufficient,
        })

        print(f"  {label:<18} {w:>8} {l:>8} {p_tp:>9.1f}%{marker}")

    if soft_sl:
        print(f"\n  SOFT SL candidate: if MAE reaches {soft_sl[0]:.1f}R, P(TP)={soft_sl[2]:.1f}% ({soft_sl[3]} trades)")
        print(f"  Current hard SL is at -1.0R by definition")
    else:
        print(f"\n  NO SOFT SL justified — MAE does not predict outcome below 25%")

    # Check if MAE separates winners from losers
    w_mae = np.array([t["mae_r"] for t in winners])
    l_mae = np.array([t["mae_r"] for t in losers])
    print(f"\n  Winner avg MAE: {np.mean(w_mae):.3f}R  median: {np.median(w_mae):.3f}R")
    print(f"  Loser  avg MAE: {np.mean(l_mae):.3f}R  median: {np.median(l_mae):.3f}R")
    separation = abs(np.mean(w_mae) - np.mean(l_mae))
    mae_predicts = separation > 0.2
    print(f"  Separation: {separation:.3f}R → {'YES (clear)' if mae_predicts else 'NO (mixed)'}")

    return mae_bucket_results, soft_sl, mae_predicts


def mfe_analysis(trades: list[dict]):
    """Step 4: MFE analysis."""
    valid = [t for t in trades if t["exit_type"] in ("tp", "sl")]
    winners = [t for t in valid if t["exit_type"] == "tp"]
    losers = [t for t in valid if t["exit_type"] == "sl"]

    print(f"\n{'=' * 70}")
    print("STEP 4a — MFE vs OUTCOME")
    print("=" * 70)

    mfe_buckets = [
        (0.0, 0.5, "0 to 0.5R"),
        (0.5, 0.8, "0.5 to 0.8R"),
        (0.8, 1.0, "0.8 to 1.0R"),
        (1.0, 1.2, "1.0 to 1.2R"),
        (1.2, 99, "> 1.2R"),
    ]

    print(f"\n  {'MFE bucket (R)':<16} {'Count':>6} {'TP':>6} {'SL':>6} {'P(wasted)':>10}")
    print(f"  {'-'*44}")

    mfe_bucket_results = []
    wasted_high_mfe = 0
    total_high_mfe = 0

    for lo_r, hi_r, label in mfe_buckets:
        bucket = [t for t in valid if lo_r <= t["mfe_r"] < hi_r]
        tp_count = sum(1 for t in bucket if t["exit_type"] == "tp")
        sl_count = sum(1 for t in bucket if t["exit_type"] == "sl")
        total = len(bucket)
        p_wasted = sl_count / total * 100 if total > 0 else 0
        sufficient = total >= 15

        if lo_r >= 0.8:
            wasted_high_mfe += sl_count
            total_high_mfe += total

        marker = ""
        if not sufficient and total > 0:
            marker = " *"

        mfe_bucket_results.append({
            "label": label, "lo_r": lo_r, "hi_r": hi_r,
            "count": total, "tp": tp_count, "sl": sl_count,
            "p_wasted": p_wasted, "sufficient": sufficient,
        })

        print(f"  {label:<16} {total:>6} {tp_count:>6} {sl_count:>6} {p_wasted:>9.1f}%{marker}")

    p_wasted_high = wasted_high_mfe / total_high_mfe * 100 if total_high_mfe > 0 else 0
    trailing_justified = p_wasted_high > 20 and total_high_mfe >= 15

    print(f"\n  Trades reaching MFE >= 0.8R: {total_high_mfe}")
    print(f"  Of those, ended SL (wasted): {wasted_high_mfe} ({p_wasted_high:.1f}%)")

    if trailing_justified:
        print(f"  TRAILING STOP candidate: {p_wasted_high:.1f}% of trades reached 0.8R+ but reversed to SL")
    else:
        if total_high_mfe < 15:
            print(f"  Insufficient data (< 15 trades) in MFE >= 0.8R bucket")
        else:
            print(f"  NO trailing stop justified — few trades waste significant MFE")

    # Step 4b — Time-to-MFE
    print(f"\n{'=' * 70}")
    print("STEP 4b — TIME-TO-MFE FOR WINNERS")
    print("=" * 70)

    w_btm = np.array([t["bar_at_mfe"] for t in winners])
    w_btr = np.array([t["bars_held"] for t in winners])

    if len(w_btm) > 0:
        print(f"\n  bars_to_mfe:        P25={np.percentile(w_btm, 25):.0f}  P50={np.percentile(w_btm, 50):.0f}  P75={np.percentile(w_btm, 75):.0f}  P95={np.percentile(w_btm, 95):.0f}")
        print(f"  bars_to_resolution: P25={np.percentile(w_btr, 25):.0f}  P50={np.percentile(w_btr, 50):.0f}  P75={np.percentile(w_btr, 75):.0f}  P95={np.percentile(w_btr, 95):.0f}")

        if np.percentile(w_btm, 75) < np.percentile(w_btr, 75) * 0.8:
            print(f"  Winners typically reach max gain well before closing — price oscillates near TP")
        else:
            print(f"  Winners reach MFE near resolution — price moves cleanly to TP")

    return mfe_bucket_results, trailing_justified, p_wasted_high


# ═══════════════════════════════════════════════════════════════════════════════
# Plots
# ═══════════════════════════════════════════════════════════════════════════════

def plot_duration(trades, bucket_results, structural_break):
    """Figure 1 — Duration Analysis."""
    valid = [t for t in trades if t["exit_type"] in ("tp", "sl")]
    winners = [t for t in valid if t["exit_type"] == "tp"]
    losers = [t for t in valid if t["exit_type"] == "sl"]

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))

    # Panel 1: Histogram
    w_bars = [t["bars_held"] for t in winners]
    l_bars = [t["bars_held"] for t in losers]
    max_bar = max(max(w_bars), max(l_bars))
    bins = np.arange(0, min(max_bar + 2, 80), 2)
    ax1.hist(w_bars, bins=bins, alpha=0.6, color="#2ecc71", label=f"Winners ({len(winners)})", edgecolor="white")
    ax1.hist(l_bars, bins=bins, alpha=0.6, color="#e74c3c", label=f"Losers ({len(losers)})", edgecolor="white")
    ax1.set_xlabel("Bars held")
    ax1.set_ylabel("Count")
    ax1.set_title("Duration Distribution — Winners vs Losers")
    ax1.legend()
    if structural_break:
        ax1.axvline(structural_break[0], color="navy", linestyle="--", alpha=0.7, label=f"Break @ bar {structural_break[0]}")
        ax1.legend()

    # Panel 2: P(TP) by duration bucket
    labels = []
    p_tps = []
    colors = []
    for b in bucket_results:
        if b["total"] == 0:
            continue
        lbl = f"{b['lo']}-{b['hi']}" if b["hi"] < 999 else f"{b['lo']}+"
        labels.append(lbl)
        p_tps.append(b["p_tp"])
        colors.append("#2ecc71" if b["p_tp"] > BASELINE_WR else "#e74c3c")

    x = np.arange(len(labels))
    ax2.bar(x, p_tps, color=colors, edgecolor="white", width=0.7)
    ax2.axhline(BASELINE_WR, color="gray", linestyle="--", linewidth=1, label=f"Baseline WR={BASELINE_WR}%")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=45)
    ax2.set_ylabel("P(TP) %")
    ax2.set_title("Win Rate by Duration Bucket")
    ax2.legend(fontsize=8)

    # Panel 3: Survival curves
    max_duration = 80
    w_bars_arr = np.array(w_bars)
    l_bars_arr = np.array(l_bars)
    bars_range = np.arange(1, max_duration + 1)
    w_survival = np.array([np.sum(w_bars_arr >= b) / len(w_bars_arr) for b in bars_range])
    l_survival = np.array([np.sum(l_bars_arr >= b) / len(l_bars_arr) for b in bars_range])

    ax3.plot(bars_range, w_survival * 100, color="#2ecc71", linewidth=2, label="Winners")
    ax3.plot(bars_range, l_survival * 100, color="#e74c3c", linewidth=2, label="Losers")
    ax3.set_xlabel("Bars elapsed")
    ax3.set_ylabel("% still active")
    ax3.set_title("Survival Curves")
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    if structural_break:
        ax3.axvline(structural_break[0], color="navy", linestyle="--", alpha=0.7)

    plt.tight_layout()
    path = str(OUTPUT_DIR / "mae_mfe_duration_BTCUSDT.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"\n  [Plot 1] Saved to {path}")
    plt.close(fig)


def plot_mae_mfe(trades, mae_bucket_results, mfe_bucket_results):
    """Figure 2 — MAE/MFE Analysis."""
    valid = [t for t in trades if t["exit_type"] in ("tp", "sl")]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))

    # Panel 1: Scatter MAE vs MFE
    for t in valid:
        color = "#2ecc71" if t["exit_type"] == "tp" else "#e74c3c"
        alpha = 0.4
        ax1.scatter(t["mae_r"], t["mfe_r"], c=color, alpha=alpha, s=12, edgecolors="none")

    ax1.axvline(-0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax1.axhline(0.8, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax1.set_xlabel("MAE (R)")
    ax1.set_ylabel("MFE (R)")
    ax1.set_title("MAE vs MFE Scatter")
    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#2ecc71', markersize=8, label='TP'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#e74c3c', markersize=8, label='SL'),
    ]
    ax1.legend(handles=legend_elements, fontsize=8)
    ax1.set_xlim(-1.3, 0.1)

    # Panel 2: P(TP) by MAE bucket
    labels = [b["label"] for b in mae_bucket_results if b["total"] > 0]
    p_tps = [b["p_tp"] for b in mae_bucket_results if b["total"] > 0]
    colors = ["#2ecc71" if p > BASELINE_WR else "#e74c3c" for p in p_tps]
    x = np.arange(len(labels))
    ax2.bar(x, p_tps, color=colors, edgecolor="white", width=0.7)
    ax2.axhline(BASELINE_WR, color="gray", linestyle="--", linewidth=1, label=f"Baseline WR={BASELINE_WR}%")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=45, fontsize=7)
    ax2.set_ylabel("P(TP) %")
    ax2.set_title("P(TP) by MAE Bucket")
    ax2.legend(fontsize=7)

    # Panel 3: P(wasted MFE) by MFE bucket
    labels = [b["label"] for b in mfe_bucket_results if b["count"] > 0]
    p_wasted = [b["p_wasted"] for b in mfe_bucket_results if b["count"] > 0]
    colors2 = ["#e74c3c" if p > 20 else "#f1c40f" if p > 10 else "#2ecc71" for p in p_wasted]
    x = np.arange(len(labels))
    ax3.bar(x, p_wasted, color=colors2, edgecolor="white", width=0.7)
    ax3.axhline(20, color="gray", linestyle="--", linewidth=1, label="20% threshold")
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels, rotation=45, fontsize=7)
    ax3.set_ylabel("P(wasted MFE) %")
    ax3.set_title("P(wasted MFE) by MFE Bucket")
    ax3.legend(fontsize=7)

    plt.tight_layout()
    path = str(OUTPUT_DIR / "mae_mfe_scatter_BTCUSDT.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  [Plot 2] Saved to {path}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# Timeout WFA (Step 5, only if structural break found)
# ═══════════════════════════════════════════════════════════════════════════════

def run_timeout_wfa(df_full, htf_bias_full, break_bar):
    """Run WFA smoke test for timeout variants."""
    import itertools
    from validation.fast_backtest import compute_indicators as ci, compute_htf_bias as chtf

    print(f"\n{'=' * 70}")
    print(f"STEP 5 — TIMEOUT WFA (break_bar={break_bar})")
    print("=" * 70)

    # Use last 365 days
    half = len(df_full) // 2
    df = df_full.iloc[half:].reset_index(drop=True)
    htf = compute_htf_bias(df)

    N_WINDOWS = 3
    IS_RATIO = 0.70
    IS_GRID = [
        {"adx_min": a, "ema_fast": f, "ema_slow": s}
        for a, f, s in itertools.product([20, 25], [15, 20], [45, 50])
    ]

    def run_variant(variant_name, timeout_bar, only_losers):
        total = len(df)
        window_size = total // N_WINDOWS
        all_metrics = []

        for w_idx in range(N_WINDOWS):
            start = w_idx * window_size
            end = min(start + window_size, total)
            split = start + int((end - start) * IS_RATIO)
            is_df = df.iloc[start:split].reset_index(drop=True)
            oos_df = df.iloc[split:end].reset_index(drop=True)
            is_htf = htf[start:split]
            oos_htf = htf[split:end]

            if len(is_df) < 100 or len(oos_df) < 50:
                continue

            best_score = -np.inf
            best_params = IS_GRID[0]

            for params in IS_GRID:
                is_ind = ci(is_df, params["ema_fast"], params["ema_slow"])
                trades = backtest_with_mae_mfe(is_ind, is_htf, rr_ratio=1.5, atr_sl_mult=2.0,
                                               adx_min=params["adx_min"])
                # Apply timeout
                trades_adj = _apply_timeout(trades, timeout_bar, only_losers)
                m = _trades_to_metrics(trades_adj, is_df)
                score = m["sharpe"] if m["n"] >= 5 else -np.inf
                if score > best_score:
                    best_score = score
                    best_params = params

            oos_ind = ci(oos_df, best_params["ema_fast"], best_params["ema_slow"])
            oos_trades = backtest_with_mae_mfe(oos_ind, oos_htf, rr_ratio=1.5, atr_sl_mult=2.0,
                                                adx_min=best_params["adx_min"])
            oos_adj = _apply_timeout(oos_trades, timeout_bar, only_losers)
            m = _trades_to_metrics(oos_adj, oos_df)
            all_metrics.append(m)

        if not all_metrics:
            return None

        avg = {k: np.mean([m[k] for m in all_metrics]) for k in all_metrics[0]}
        avg["n"] = sum(m["n"] for m in all_metrics)
        return avg

    baseline = run_variant("Baseline", timeout_bar=9999, only_losers=False)
    variant_a = run_variant("Variant A", timeout_bar=break_bar, only_losers=False)
    variant_b = run_variant("Variant B", timeout_bar=break_bar, only_losers=True)

    print(f"\n  {'Metric':<18} {'Baseline':>12} {'Variant A':>12} {'Variant B':>12}")
    print(f"  {'-'*54}")
    for key, label, fmt in [
        ("sharpe", "OOS Sharpe", ".2f"), ("annual", "OOS Annual %", "+.1f"),
        ("wr", "OOS WR %", ".1f"), ("exp_r", "Expectancy R", "+.4f"),
        ("avg_bars", "Avg bars held", ".1f"), ("max_dd", "OOS DD %", ".1f"),
    ]:
        b_val = baseline[key] if baseline else 0
        a_val = variant_a[key] if variant_a else 0
        bv_val = variant_b[key] if variant_b else 0
        b_s = format(b_val, fmt).rjust(12)
        a_s = format(a_val, fmt).rjust(12)
        bv_s = format(bv_val, fmt).rjust(12)
        print(f"  {label:<18} {b_s} {a_s} {bv_s}")

    return baseline, variant_a, variant_b


def _apply_timeout(trades, timeout_bar, only_losers):
    """Simulate timeout: close trades at timeout bar."""
    result = []
    for t in trades:
        if t["exit_type"] == "end_of_data":
            continue
        if t["bars_held"] > timeout_bar:
            # Would timeout apply?
            if only_losers:
                # Only exit if losing at timeout point — approximate with mae
                # If MAE < 0 at that point, likely losing
                # We don't have exact bar-by-bar PnL, so use a proxy:
                # the trade lasted longer than timeout, just keep the original outcome
                # This is a rough approximation
                result.append(t)
            else:
                # Close at "neutral" — approximate PnL = 0 (worst case for the rule)
                # More accurate: partial PnL, but we don't have bar-by-bar close
                modified = dict(t)
                modified["exit_type"] = "timeout"
                modified["pnl_pct"] = 0.0  # conservative estimate
                modified["bars_held"] = timeout_bar
                result.append(modified)
        else:
            result.append(t)
    return result


def _trades_to_metrics(trades, df):
    """Compute summary metrics from trade list."""
    valid = [t for t in trades if t["exit_type"] != "end_of_data"]
    n = len(valid)
    if n == 0:
        return {"n": 0, "sharpe": 0, "annual": 0, "wr": 0, "exp_r": 0, "avg_bars": 0, "max_dd": 0}

    pnls = np.array([t["pnl_pct"] for t in valid])
    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]
    wr = len(wins) / n * 100
    avg_bars = np.mean([t["bars_held"] for t in valid])

    avg_win = np.mean(wins) if len(wins) > 0 else 0
    avg_loss = abs(np.mean(losses)) if len(losses) > 0 else 1
    exp_r = (wr / 100 * avg_win / avg_loss) - (1 - wr / 100) if avg_loss > 0 else 0

    mult = 1.0 + pnls / 100.0
    equity = np.concatenate([[1.0], np.cumprod(mult)])
    running_max = np.maximum.accumulate(equity)
    dd = (equity - running_max) / running_max
    max_dd = abs(float(np.min(dd))) * 100

    ts_start = pd.Timestamp(df["ts"].iloc[0])
    ts_end = pd.Timestamp(df["ts"].iloc[-1])
    days = max((ts_end - ts_start).total_seconds() / 86400, 1)
    total_return = equity[-1]
    annual = (total_return ** (365.0 / days) - 1) * 100

    std_r = np.std(pnls, ddof=1) if n >= 2 else 1
    if std_r > 0:
        delta = (pd.Timestamp(df["ts"].iloc[1]) - pd.Timestamp(df["ts"].iloc[0])).total_seconds()
        tf_min = max(delta / 60, 1)
        bars_per_year = 365 * 24 * 60 / tf_min
        trades_per_year = bars_per_year / max(avg_bars, 1)
        sharpe = float((np.mean(pnls) / std_r) * np.sqrt(trades_per_year))
    else:
        sharpe = 0

    return {"n": n, "sharpe": sharpe, "annual": annual, "wr": wr,
            "exp_r": exp_r, "avg_bars": avg_bars, "max_dd": max_dd}


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("MAE/MFE/DURATION ANALYSIS — BTC/USDT 15m, 730d")
    print("rr=1.5, atr_sl=2.0, HTF=ON")
    print("=" * 70)

    # Load data
    df = load_candles("BTC/USDT", "15m", days=730)
    print(f"Data: {len(df)} bars from {df['ts'].iloc[0]} to {df['ts'].iloc[-1]}")

    # Compute indicators + HTF bias
    ind = compute_indicators(df, ema_fast_p=20, ema_slow_p=50)
    htf_bias = compute_htf_bias(df)

    # Run backtest with MAE/MFE tracking
    print("Running backtest with MAE/MFE tracking...")
    trades = backtest_with_mae_mfe(ind, htf_bias, rr_ratio=1.5, atr_sl_mult=2.0)

    # Filter out end_of_data
    valid = [t for t in trades if t["exit_type"] in ("tp", "sl")]
    print(f"Total trades: {len(valid)} (tp={sum(1 for t in valid if t['exit_type']=='tp')}, sl={sum(1 for t in valid if t['exit_type']=='sl')})")
    print(f"Excluded: {len(trades) - len(valid)} end_of_data trades")

    # Step 2: Duration
    sw, sl, bucket_results, structural_break = duration_analysis(valid)

    # Step 3: MAE
    mae_bucket_results, soft_sl, mae_predicts = mae_analysis(valid)

    # Step 4: MFE
    mfe_bucket_results, trailing_justified, p_wasted_high = mfe_analysis(valid)

    # Step 5: Timeout WFA (only if structural break)
    timeout_results = None
    if structural_break:
        timeout_results = run_timeout_wfa(ind, htf_bias, structural_break[0])
    else:
        print(f"\n{'=' * 70}")
        print("STEP 5 — No timeout rule tested — no structural break found.")
        print("=" * 70)

    # Step 6: Plots
    print(f"\n{'=' * 70}")
    print("STEP 6 — GENERATING PLOTS")
    print("=" * 70)
    plot_duration(valid, bucket_results, structural_break)
    plot_mae_mfe(valid, mae_bucket_results, mfe_bucket_results)

    # Step 7: Final verdict
    print_verdict(sw, sl, structural_break, soft_sl, mae_predicts,
                  trailing_justified, p_wasted_high, timeout_results)


def print_verdict(sw, sl, structural_break, soft_sl, mae_predicts,
                  trailing_justified, p_wasted_high, timeout_results):
    sb_str = f"bar {structural_break[0]} (P(SL)={structural_break[1]:.1f}%)" if structural_break else "NOT FOUND"
    timeout_just = "JUSTIFIED" if structural_break else "NOT JUSTIFIED"
    soft_sl_str = f"-{abs(soft_sl[0]):.1f}R (P(TP)={soft_sl[2]:.1f}%)" if soft_sl else "NOT FOUND"
    mae_str = "YES (clear separation)" if mae_predicts else "NO (mixed)"
    wasted_str = f"{p_wasted_high:.1f}% of trades" if p_wasted_high > 0 else "NOT SIGNIFICANT"
    trail_str = "JUSTIFIED" if trailing_justified else "NOT JUSTIFIED"

    adopt_timeout = False
    if timeout_results and structural_break:
        baseline, va, vb = timeout_results
        if baseline and va and va["sharpe"] > baseline["sharpe"] * 1.15:
            adopt_timeout = True
        if baseline and vb and vb["sharpe"] > baseline["sharpe"] * 1.15:
            adopt_timeout = True

    adopt_soft_sl = soft_sl is not None
    adopt_trail = trailing_justified

    # Determine if baseline is optimal
    baseline_optimal = not adopt_timeout and not adopt_soft_sl and not adopt_trail

    print(f"""

{'='*66}
  MAE/MFE/DURATION ANALYSIS — BTC/USDT 15m
{'='*66}
  DURATION ANALYSIS
   Winners avg bars:    {sw['avg']:.1f}  | Losers avg bars: {sl['avg']:.1f}
   Structural break:    {sb_str}
   Timeout rule:        {timeout_just}
{'─'*66}
  MAE ANALYSIS
   Soft SL threshold:   {soft_sl_str}
   MAE predicts outcome: {mae_str}
{'─'*66}
  MFE ANALYSIS
   Wasted MFE (>0.8R):  {wasted_str}
   Trailing stop:       {trail_str}
{'─'*66}""")

    if timeout_results and structural_break:
        baseline, va, vb = timeout_results
        va_s = va["sharpe"] if va else 0
        vb_s = vb["sharpe"] if vb else 0
        b_s = baseline["sharpe"] if baseline else 0
        print(f"""  TIMEOUT BACKTEST
   Variant A OOS Sharpe: {va_s:.2f} vs baseline {b_s:.2f}
   Variant B OOS Sharpe: {vb_s:.2f} vs baseline {b_s:.2f}
   Adopt timeout:        {'YES' if adopt_timeout else 'NO'}
{'─'*66}""")
    else:
        print(f"""  TIMEOUT BACKTEST
   Not tested — no structural break found.
{'─'*66}""")

    print(f"""  FINAL RECOMMENDATION
   Keep baseline (fixed SL/TP):  {'YES' if baseline_optimal else 'NO'}
   Add timeout rule:             {'YES (bar ' + str(structural_break[0]) + ')' if adopt_timeout else 'NO'}
   Add soft SL:                  {'YES (' + soft_sl_str + ')' if adopt_soft_sl else 'NO'}
   Add trailing stop:            {'YES' if adopt_trail else 'NO'}""")

    if baseline_optimal:
        print(f"""
   Rationale: Fixed SL/TP at rr=1.5, sl_atr=2.0 is the correct exit
   mechanism. No structural breaks found in duration, MAE, or MFE
   distributions that justify additional exit rules.""")
    else:
        reasons = []
        if adopt_timeout:
            reasons.append(f"timeout at bar {structural_break[0]}")
        if adopt_soft_sl:
            reasons.append(f"soft SL at {soft_sl_str}")
        if adopt_trail:
            reasons.append(f"trailing stop (wasted MFE={p_wasted_high:.0f}%)")
        print(f"""
   Rationale: Data shows structural justification for: {', '.join(reasons)}.
   Verify with full WFA before deploying.""")

    print(f"{'='*66}")


if __name__ == "__main__":
    main()
