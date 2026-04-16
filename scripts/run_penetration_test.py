"""
Penetration test for TrendFollowingV2Simple — BTC/USDT 15m 730d.

Attack 1: Stop Hunt Analysis — classify SL exits, measure proximity
          clustering and post-SL recovery.
Attack 2: Slippage & Latency Stress Test — run 7 execution scenarios
          + binary search for breaking point.

Baseline: rr_ratio=1.5, atr_sl_mult=2.0, HTF=ON, 730d.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from validation.data_loader import load_candles
from validation.fast_backtest import (
    compute_indicators,
    compute_htf_bias,
    fast_backtest,
    _pnl,
    _fast_confidence,
    _build_metrics,
)
from validation.strategy_adapter import TradeRecord, BacktestMetrics


# ─────────────────── config ────────────────────────────────────────
VALIDATED_PARAMS = dict(
    adx_min=20.0,
    ema_fast_p=20,
    ema_slow_p=50,
    rr_ratio=1.5,
    atr_sl_mult=2.0,
    pb_tol_atr=1.0,
    sig_cooldown=5,
    allow_short=True,
    min_confidence=0.0,
    adx_strong=35.0,
    slope_bars=5,
)
OUTPUT_DIR = ROOT / "validation" / "output"
PLT_BG = "#0d0d0d"
PLT_FG = "#e6e6e6"


def _dark_style():
    plt.rcParams.update({
        "figure.facecolor": PLT_BG,
        "axes.facecolor": PLT_BG,
        "axes.edgecolor": PLT_FG,
        "axes.labelcolor": PLT_FG,
        "xtick.color": PLT_FG,
        "ytick.color": PLT_FG,
        "text.color": PLT_FG,
        "axes.titlecolor": PLT_FG,
        "grid.color": "#303030",
        "savefig.facecolor": PLT_BG,
    })


# ═══════════════════ SLIPPAGE BACKTEST ═════════════════════════════
def fast_backtest_with_slippage(
    df: pd.DataFrame,
    entry_slippage_pct: float = 0.0,
    sl_slippage_pct: float = 0.0,
    tp_slippage_pct: float = 0.0,
    signal_delay: int = 0,
    htf_bias: np.ndarray | None = None,
    **params,
) -> BacktestMetrics:
    """Fork of fast_backtest that applies slippage to fills + optional 1-bar delay.

    Slippage direction (adverse to trader):
      LONG entry:  fill = close * (1 + slip)
      LONG SL:     fill = sl    * (1 - slip)   (gap-through, worse than SL)
      LONG TP:     fill = tp    * (1 - slip)   (filled slightly below limit)
      SHORT entry: fill = close * (1 - slip)
      SHORT SL:    fill = sl    * (1 + slip)
      SHORT TP:    fill = tp    * (1 + slip)

    signal_delay = 1: entry at next-bar open; SL/TP recomputed from new entry.
    """
    p = {**VALIDATED_PARAMS, **params}
    df = compute_indicators(df, p["ema_fast_p"], p["ema_slow_p"], p["slope_bars"])
    min_bars = max(60, p["ema_slow_p"] + 20)

    close = df["close"].values.astype(np.float64)
    high  = df["high"].values.astype(np.float64)
    low   = df["low"].values.astype(np.float64)
    opn   = df["open"].values.astype(np.float64)
    atr   = df["atr"].values.astype(np.float64)
    adx   = df["adx"].values.astype(np.float64)
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

    es = entry_slippage_pct / 100.0
    ss = sl_slippage_pct / 100.0
    ts = tp_slippage_pct / 100.0

    trades: list[TradeRecord] = []
    in_trade = False
    trade_dir = 0
    entry_price = 0.0
    sl_price = 0.0
    tp_price = 0.0
    entry_bar = 0

    prev_long_sig = False
    prev_short_sig = False
    last_long_bar = -999
    last_short_bar = -999

    pending_long = 0   # bars remaining until delayed entry (0 = none)
    pending_short = 0

    for i in range(min_bars, n):
        c = close[i]
        h = high[i]
        lo = low[i]
        a = atr[i]
        dx = adx[i]

        if np.isnan(dx) or np.isnan(ema_s[i]) or a <= 0:
            continue

        # ── Delayed entry fill at bar i open ──────────────────────
        if not in_trade and signal_delay > 0:
            if pending_long == 1:
                raw_entry = opn[i]
                entry_price = raw_entry * (1 + es)
                sl_dist = a * p["atr_sl_mult"]
                sl_price = entry_price - sl_dist
                tp_price = entry_price + sl_dist * p["rr_ratio"]
                trade_dir = 1
                entry_bar = i
                in_trade = True
                pending_long = 0
            elif pending_short == 1:
                raw_entry = opn[i]
                entry_price = raw_entry * (1 - es)
                sl_dist = a * p["atr_sl_mult"]
                sl_price = entry_price + sl_dist
                tp_price = entry_price - sl_dist * p["rr_ratio"]
                trade_dir = -1
                entry_bar = i
                in_trade = True
                pending_short = 0
            else:
                if pending_long > 1: pending_long -= 1
                if pending_short > 1: pending_short -= 1

        # ── Exits ────────────────────────────────────────────────
        if in_trade:
            sl_hit = (trade_dir == 1 and lo <= sl_price) or \
                     (trade_dir == -1 and h >= sl_price)
            tp_hit = (trade_dir == 1 and h >= tp_price) or \
                     (trade_dir == -1 and lo <= tp_price)

            if tp_hit:
                if trade_dir == 1:
                    fill = tp_price * (1 - ts)
                else:
                    fill = tp_price * (1 + ts)
                trades.append(TradeRecord(
                    direction="LONG" if trade_dir == 1 else "SHORT",
                    entry_price=entry_price, exit_price=fill,
                    pnl_pct=_pnl(trade_dir, entry_price, fill),
                    exit_type="tp", bars_held=i - entry_bar,
                    entry_bar_idx=entry_bar,
                    sl=sl_price, tp1=tp_price,
                ))
                in_trade = False
                continue
            elif sl_hit:
                if trade_dir == 1:
                    fill = sl_price * (1 - ss)
                else:
                    fill = sl_price * (1 + ss)
                trades.append(TradeRecord(
                    direction="LONG" if trade_dir == 1 else "SHORT",
                    entry_price=entry_price, exit_price=fill,
                    pnl_pct=_pnl(trade_dir, entry_price, fill),
                    exit_type="sl", bars_held=i - entry_bar,
                    entry_bar_idx=entry_bar,
                    sl=sl_price, tp1=tp_price,
                ))
                in_trade = False
                continue

        # ── Signals ──────────────────────────────────────────────
        pb_zone   = abs(c - ema_f[i]) < a * p["pb_tol_atr"]
        sl_rising = ema_s_slope[i] > 0 if not np.isnan(ema_s_slope[i]) else False
        sl_falling= ema_s_slope[i] < 0 if not np.isnan(ema_s_slope[i]) else False
        p_above   = c > ema_s[i]
        p_below   = c < ema_s[i]
        m_bull    = macd_v[i] > macd_sig[i]
        m_bear    = macd_v[i] < macd_sig[i]
        c_bull    = c > opn[i]
        c_bear    = c < opn[i]
        adx_ok    = dx >= p["adx_min"]

        long_base  = adx_ok and sl_rising and p_above and m_bull and pb_zone and c_bull
        short_base = adx_ok and sl_falling and p_below and m_bear and pb_zone and c_bear and p["allow_short"]

        conf_l = _fast_confidence(dx, p["adx_strong"], a, c, ema_f[i],
                                  macd_hist[i], macd_hist[i-1] if i > 0 else 0,
                                  "LONG", opn[i], h, lo,
                                  ema_f_slope[i], volume[i], vol_sma[i]) if long_base else 0.0
        conf_s = _fast_confidence(dx, p["adx_strong"], a, c, ema_f[i],
                                  macd_hist[i], macd_hist[i-1] if i > 0 else 0,
                                  "SHORT", opn[i], h, lo,
                                  ema_f_slope[i], volume[i], vol_sma[i]) if short_base else 0.0

        long_signal  = long_base  and conf_l >= p["min_confidence"]
        short_signal = short_base and conf_s >= p["min_confidence"]

        long_trig_raw  = long_signal  and not prev_long_sig
        short_trig_raw = short_signal and not prev_short_sig
        prev_long_sig  = long_signal
        prev_short_sig = short_signal

        long_trig  = long_trig_raw  and (i - last_long_bar)  >= p["sig_cooldown"]
        short_trig = short_trig_raw and (i - last_short_bar) >= p["sig_cooldown"]
        if long_trig:  last_long_bar = i
        if short_trig: last_short_bar = i

        if htf_bias is not None:
            bias = htf_bias[i]
            if long_trig  and bias == -1: long_trig  = False
            if short_trig and bias == 1:  short_trig = False

        # ── Open trade ───────────────────────────────────────────
        if not in_trade:
            if signal_delay == 0:
                if long_trig:
                    entry_price = c * (1 + es)
                    sl_dist = a * p["atr_sl_mult"]
                    sl_price = entry_price - sl_dist
                    tp_price = entry_price + sl_dist * p["rr_ratio"]
                    trade_dir = 1
                    entry_bar = i
                    in_trade = True
                elif short_trig:
                    entry_price = c * (1 - es)
                    sl_dist = a * p["atr_sl_mult"]
                    sl_price = entry_price + sl_dist
                    tp_price = entry_price - sl_dist * p["rr_ratio"]
                    trade_dir = -1
                    entry_bar = i
                    in_trade = True
            else:  # delayed
                if long_trig:  pending_long = signal_delay
                elif short_trig: pending_short = signal_delay

    if in_trade:
        trades.append(TradeRecord(
            direction="LONG" if trade_dir == 1 else "SHORT",
            entry_price=entry_price, exit_price=float(close[-1]),
            pnl_pct=_pnl(trade_dir, entry_price, float(close[-1])),
            exit_type="end_of_data", bars_held=n - 1 - entry_bar,
            entry_bar_idx=entry_bar,
            sl=sl_price, tp1=tp_price,
        ))

    return _build_metrics(trades, df)


# ═══════════════════ ATTACK 1: STOP HUNT ══════════════════════════
def stop_hunt_analysis(df: pd.DataFrame, metrics: BacktestMetrics):
    """Classify every SL exit and compute proximity + recovery stats."""
    df_ind = compute_indicators(df, VALIDATED_PARAMS["ema_fast_p"],
                                VALIDATED_PARAMS["ema_slow_p"],
                                VALIDATED_PARAMS["slope_bars"])
    high = df_ind["high"].values.astype(np.float64)
    low  = df_ind["low"].values.astype(np.float64)
    close = df_ind["close"].values.astype(np.float64)
    atr = df_ind["atr"].values.astype(np.float64)
    n = len(df_ind)

    sl_trades = [t for t in metrics.trades if t["exit_type"] == "sl"]

    records = []
    for t in sl_trades:
        direction = t["direction"]
        entry_idx = t["entry_bar_idx"]
        bars_held = t["bars_held"]
        entry = t["entry_price"]
        sl = t["sl"]
        sl_dist_abs = abs(entry - sl)

        sl_bar = entry_idx + bars_held
        if sl_bar >= n or sl_dist_abs <= 0:
            continue

        # Proximity: how deep past SL the bar went, normalized by sl_dist
        if direction == "LONG":
            penetration = max(sl - low[sl_bar], 0.0)
        else:
            penetration = max(high[sl_bar] - sl, 0.0)
        proximity = penetration / sl_dist_abs

        # Post-SL recovery: did price return to entry within N bars?
        end_20 = min(sl_bar + 20, n - 1)
        future_high = high[sl_bar + 1: end_20 + 1]
        future_low  = low [sl_bar + 1: end_20 + 1]

        tol = 0.001  # 0.1% of entry
        if direction == "LONG":
            # recovered if high returns to within 0.1% of entry (>= entry*(1-tol))
            threshold = entry * (1 - tol)
            recovered_5  = bool(np.any(future_high[:5]  >= threshold))
            recovered_10 = bool(np.any(future_high[:10] >= threshold))
            recovered_20 = bool(np.any(future_high[:20] >= threshold))
            max_rec = (np.max(future_high[:20]) - entry) / entry * 100 if len(future_high) else 0.0
            # normalized price path (relative to entry, in %)
            path_end = min(sl_bar + 20, n - 1)
            path = (close[sl_bar: path_end + 1] - entry) / entry * 100
        else:
            threshold = entry * (1 + tol)
            recovered_5  = bool(np.any(future_low[:5]  <= threshold))
            recovered_10 = bool(np.any(future_low[:10] <= threshold))
            recovered_20 = bool(np.any(future_low[:20] <= threshold))
            max_rec = (entry - np.min(future_low[:20])) / entry * 100 if len(future_low) else 0.0
            path_end = min(sl_bar + 20, n - 1)
            path = (entry - close[sl_bar: path_end + 1]) / entry * 100

        records.append({
            "direction": direction,
            "proximity": proximity,
            "recovered_5": recovered_5,
            "recovered_10": recovered_10,
            "recovered_20": recovered_20,
            "max_recovery_pct": max_rec,
            "path": path,
        })

    return records


def plot_stop_hunt(records, metrics, out_path):
    _dark_style()
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle("Penetration Test — Stop Hunt Analysis (BTC/USDT 15m)",
                 fontsize=16, fontweight="bold")

    n_sl = len(records)
    if n_sl == 0:
        plt.savefig(out_path); plt.close()
        return {}

    prox = np.array([r["proximity"] for r in records])
    rec5  = np.mean([r["recovered_5"]  for r in records]) * 100
    rec10 = np.mean([r["recovered_10"] for r in records]) * 100
    rec20 = np.mean([r["recovered_20"] for r in records]) * 100

    # Panel 1: Stop hunt rate
    ax1 = fig.add_subplot(2, 2, 1)
    bars = ax1.bar(["5 bars", "10 bars", "20 bars"], [rec5, rec10, rec20],
                   color="#c0392b", edgecolor=PLT_FG)
    for b, v in zip(bars, [rec5, rec10, rec20]):
        ax1.text(b.get_x() + b.get_width()/2, v + 0.5, f"{v:.1f}%",
                 ha="center", color=PLT_FG, fontweight="bold")
    ax1.axhline(10, ls="--", color="#f1c40f", alpha=0.6, label="10% threshold")
    ax1.axhline(20, ls="--", color="#e67e22", alpha=0.6, label="20% threshold")
    ax1.set_ylabel("% of SL trades recovered to entry")
    ax1.set_title("Stop Hunt Rate by Recovery Window")
    ax1.legend(facecolor=PLT_BG, edgecolor=PLT_FG)
    ax1.grid(True, alpha=0.3)

    # Panel 2: Proximity histogram (clamp to [0,2] for display)
    ax2 = fig.add_subplot(2, 2, 2)
    clipped = np.clip(prox, 0, 2)
    counts, edges, patches = ax2.hist(clipped, bins=20, edgecolor=PLT_FG)
    for i, patch in enumerate(patches):
        center = (edges[i] + edges[i+1]) / 2
        if center < 0.2: patch.set_facecolor("#c0392b")
        elif center < 0.5: patch.set_facecolor("#e67e22")
        else: patch.set_facecolor("#27ae60")
    ax2.axvline(0.2, ls="--", color="#f1c40f", label="0.2 clustering thresh")
    pct_below_02 = float(np.mean(prox < 0.2)) * 100
    ax2.set_xlabel("SL Penetration (× SL distance)")
    ax2.set_ylabel("Count")
    ax2.set_title(f"SL Penetration Depth — {pct_below_02:.1f}% below 0.2")
    ax2.text(0.98, 0.95, "Clustering if >30% below 0.2",
             transform=ax2.transAxes, ha="right", va="top",
             color="#f1c40f", fontsize=9)
    ax2.legend(facecolor=PLT_BG, edgecolor=PLT_FG)
    ax2.grid(True, alpha=0.3)

    # Panel 3: Post-SL price paths
    ax3 = fig.add_subplot(2, 2, 3)
    max_len = max(len(r["path"]) for r in records)
    matrix = np.full((len(records), max_len), np.nan)
    for i, r in enumerate(records):
        matrix[i, :len(r["path"])] = r["path"]
    hunt_mask = np.array([r["recovered_10"] for r in records])
    for i in range(len(records)):
        color = "#e67e22" if hunt_mask[i] else "#555555"
        alpha = 0.35 if hunt_mask[i] else 0.15
        ax3.plot(matrix[i], color=color, alpha=alpha, lw=0.7)
    median = np.nanmedian(matrix, axis=0)
    ax3.plot(median, color="#ecf0f1", lw=2.5, label="Median")
    ax3.axhline(0, ls="--", color=PLT_FG, alpha=0.5, label="Entry level")
    ax3.set_xlabel("Bars after SL exit")
    ax3.set_ylabel("Price vs entry (%)")
    ax3.set_title("Post-SL Price Path (normalized, entry=0%)")
    ax3.legend(facecolor=PLT_BG, edgecolor=PLT_FG)
    ax3.grid(True, alpha=0.3)

    # Panel 4: Summary text
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis("off")
    prox_low  = float(np.mean(prox < 0.2)) * 100
    prox_mid  = float(np.mean((prox >= 0.2) & (prox < 0.5))) * 100
    prox_high = float(np.mean(prox >= 0.5)) * 100
    if rec10 > 20:
        verdict = "HIGH risk"; sl_verdict = "QUESTIONABLE"
    elif rec10 > 10:
        verdict = "MODERATE risk"; sl_verdict = "ROBUST"
    else:
        verdict = "LOW risk"; sl_verdict = "ROBUST"
    cluster = "DETECTED" if prox_low > 30 else "NOT DETECTED"
    txt = (
        "Stop Hunt Analysis Summary\n"
        "─────────────────────────────\n"
        f"Total SL exits:        {n_sl}\n"
        f"Stop hunts (5 bars):   {int(rec5*n_sl/100)}  ({rec5:.1f}%)\n"
        f"Stop hunts (10 bars):  {int(rec10*n_sl/100)}  ({rec10:.1f}%)\n"
        f"Stop hunts (20 bars):  {int(rec20*n_sl/100)}  ({rec20:.1f}%)\n\n"
        "SL Penetration:\n"
        f"  <0.2 (suspicious):   {prox_low:.1f}%\n"
        f"  0.2-0.5 (moderate):  {prox_mid:.1f}%\n"
        f"  >0.5 (genuine):      {prox_high:.1f}%\n\n"
        f"Clustering:            {cluster}\n"
        f"Verdict:               {verdict}\n"
        f"SL at 2.0×ATR:         {sl_verdict}"
    )
    ax4.text(0.02, 0.98, txt, transform=ax4.transAxes, va="top",
             family="monospace", fontsize=10, color=PLT_FG)

    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"  saved: {out_path}")

    return {
        "n_sl": n_sl, "rec5": rec5, "rec10": rec10, "rec20": rec20,
        "prox_low": prox_low, "prox_mid": prox_mid, "prox_high": prox_high,
        "clustering": cluster, "verdict": verdict, "sl_verdict": sl_verdict,
    }


# ═══════════════════ ATTACK 2: SLIPPAGE ════════════════════════════
def expectancy_r(metrics: BacktestMetrics) -> float:
    """Expectancy in R units (assuming baseline SL = 2×ATR with rr=1.5).
    Approximation: each trade's R = pnl_pct / (avg_loss_pct if lost else use mean)."""
    if not metrics.trades:
        return 0.0
    pnls = np.array([t["pnl_pct"] for t in metrics.trades])
    losses = pnls[pnls < 0]
    if len(losses) == 0:
        return float(np.mean(pnls) / abs(pnls.min()) if len(pnls) else 0.0)
    r_unit = abs(np.mean(losses))  # average losing trade defines 1R
    if r_unit <= 0:
        return 0.0
    return float(np.mean(pnls) / r_unit)


def run_scenarios(df: pd.DataFrame, htf: np.ndarray):
    scenarios = [
        ("S0 base",  0.00, 0.00, 0.00, 0),
        ("S1 paper", 0.04, 0.04, 0.02, 0),
        ("S2 real",  0.08, 0.10, 0.02, 0),
        ("S3 delay", 0.08, 0.10, 0.02, 1),
        ("S4 hvol",  0.20, 0.30, 0.05, 0),
        ("S5 worst", 0.30, 0.50, 0.10, 1),
    ]
    rows = []
    equity_curves = {}
    for name, es, ss, ts, delay in scenarios:
        m = fast_backtest_with_slippage(
            df, entry_slippage_pct=es, sl_slippage_pct=ss,
            tp_slippage_pct=ts, signal_delay=delay,
            htf_bias=htf, **VALIDATED_PARAMS,
        )
        rows.append({
            "scenario": name, "es": es, "ss": ss, "ts": ts, "delay": delay,
            "sharpe": m.sharpe_ratio, "ann": m.annual_return_pct,
            "expR": expectancy_r(m), "wr": m.winrate,
            "dd": m.max_drawdown_pct, "n": m.total_trades,
        })
        pnls = np.array([t["pnl_pct"] for t in m.trades])
        equity_curves[name] = np.concatenate([[1.0], np.cumprod(1 + pnls / 100)])
        print(f"  {name}: Sharpe={m.sharpe_ratio:.3f} Ann={m.annual_return_pct:.1f}% "
              f"ExpR={rows[-1]['expR']:.3f} WR={m.winrate:.1f}% DD={m.max_drawdown_pct:.1f}% n={m.total_trades}")

    # S6 breaking point — binary search on total slippage
    # Apply half to entry, half to SL; tp slip scales 0.1×
    print("  Searching breaking point (S6)...")
    lo, hi = 0.0, 3.0
    target_sharpe = 1.0
    best_total = None
    # First bracket: find a `hi` where Sharpe < target_sharpe
    for test in [0.5, 1.0, 1.5, 2.0, 3.0]:
        m = fast_backtest_with_slippage(
            df, entry_slippage_pct=test/2, sl_slippage_pct=test/2,
            tp_slippage_pct=test*0.1, signal_delay=0,
            htf_bias=htf, **VALIDATED_PARAMS,
        )
        if m.sharpe_ratio < target_sharpe:
            hi = test
            break
        lo = test
    # Binary search
    for _ in range(10):
        mid = (lo + hi) / 2
        m = fast_backtest_with_slippage(
            df, entry_slippage_pct=mid/2, sl_slippage_pct=mid/2,
            tp_slippage_pct=mid*0.1, signal_delay=0,
            htf_bias=htf, **VALIDATED_PARAMS,
        )
        if m.sharpe_ratio < target_sharpe:
            hi = mid
        else:
            lo = mid
    best_total = (lo + hi) / 2
    m_break = fast_backtest_with_slippage(
        df, entry_slippage_pct=best_total/2, sl_slippage_pct=best_total/2,
        tp_slippage_pct=best_total*0.1, signal_delay=0,
        htf_bias=htf, **VALIDATED_PARAMS,
    )
    rows.append({
        "scenario": "S6 break",
        "es": best_total/2, "ss": best_total/2, "ts": best_total*0.1, "delay": 0,
        "sharpe": m_break.sharpe_ratio, "ann": m_break.annual_return_pct,
        "expR": expectancy_r(m_break), "wr": m_break.winrate,
        "dd": m_break.max_drawdown_pct, "n": m_break.total_trades,
        "total_slip": best_total,
    })
    print(f"  S6 break: total={best_total:.3f}% Sharpe={m_break.sharpe_ratio:.3f}")

    return rows, equity_curves


def classify_scenario(sharpe, expR):
    if sharpe < 1.0 or expR < 0:
        return "DANGER"
    if sharpe < 2.0 or expR < 0.05:
        return "MARGINAL"
    return "SAFE"


def plot_slippage(rows, equity_curves, out_path):
    _dark_style()
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle("Penetration Test — Slippage & Latency Stress (BTC/USDT 15m)",
                 fontsize=16, fontweight="bold")

    names = [r["scenario"] for r in rows]
    sharpes = [r["sharpe"] for r in rows]
    exprs = [r["expR"] for r in rows]
    anns = [r["ann"] for r in rows]
    baseline_sharpe = rows[0]["sharpe"]

    # Colors: green → yellow → red by index
    cmap = plt.cm.RdYlGn_r
    colors = cmap(np.linspace(0.1, 0.85, len(rows)))

    # Panel 1: Sharpe (top wide)
    ax1 = fig.add_subplot(3, 2, (1, 2))
    bars = ax1.bar(names, sharpes, color=colors, edgecolor=PLT_FG)
    for b, v in zip(bars, sharpes):
        ax1.text(b.get_x() + b.get_width()/2, v + 0.05, f"{v:.2f}",
                 ha="center", color=PLT_FG, fontweight="bold")
    ax1.axhline(1.0, ls="--", color="#f1c40f", alpha=0.7, label="Sharpe=1.0 (min)")
    ax1.axhline(baseline_sharpe, ls="--", color="#27ae60", alpha=0.5,
                label=f"Baseline={baseline_sharpe:.2f}")
    ax1.set_ylabel("Sharpe Ratio")
    ax1.set_title("Sharpe Degradation Under Slippage Scenarios")
    ax1.legend(facecolor=PLT_BG, edgecolor=PLT_FG)
    ax1.grid(True, alpha=0.3)

    # Panel 2: ExpR
    ax2 = fig.add_subplot(3, 2, 3)
    ax2.bar(names, exprs, color=colors, edgecolor=PLT_FG)
    ax2.axhline(0, ls="--", color="#e74c3c", label="Breakeven")
    ax2.axhline(rows[0]["expR"], ls="--", color="#27ae60", alpha=0.5,
                label=f"Baseline={rows[0]['expR']:.3f}R")
    ax2.set_ylabel("Expectancy (R)")
    ax2.set_title("Expectancy R — Slippage Impact")
    ax2.legend(facecolor=PLT_BG, edgecolor=PLT_FG, fontsize=8)
    ax2.grid(True, alpha=0.3)
    plt.setp(ax2.get_xticklabels(), rotation=30, ha="right")

    # Panel 3: Annual %
    ax3 = fig.add_subplot(3, 2, 4)
    ax3.bar(names, anns, color=colors, edgecolor=PLT_FG)
    ax3.axhline(0, ls="--", color="#e74c3c")
    ax3.set_ylabel("Annual Return (%)")
    ax3.set_title("Annual Return by Execution Quality")
    ax3.grid(True, alpha=0.3)
    plt.setp(ax3.get_xticklabels(), rotation=30, ha="right")

    # Panel 4: Equity curves S0 / S2 / S5
    ax4 = fig.add_subplot(3, 2, 5)
    for scen, color, label in [("S0 base", "#27ae60", "S0 perfect"),
                                ("S2 real", "#3498db", "S2 realistic"),
                                ("S5 worst", "#e74c3c", "S5 worst")]:
        if scen in equity_curves:
            ax4.plot(equity_curves[scen], color=color, lw=1.8, label=label)
    ax4.axhline(1.0, ls="--", color=PLT_FG, alpha=0.4)
    ax4.set_ylabel("Equity (multiplier)")
    ax4.set_xlabel("Trade #")
    ax4.set_title("Equity Curve: Perfect vs Realistic vs Worst")
    ax4.legend(facecolor=PLT_BG, edgecolor=PLT_FG)
    ax4.grid(True, alpha=0.3)

    # Panel 5: summary
    ax5 = fig.add_subplot(3, 2, 6)
    ax5.axis("off")
    s0 = rows[0]; s6 = rows[-1]
    def degr(v, base): return (v - base) / abs(base) * 100 if base else 0
    lines = ["Slippage Stress Test Summary",
             "─────────────────────────────",
             f"Baseline Sharpe (S0):   {s0['sharpe']:.3f}"]
    for r in rows[1:-1]:
        d = degr(r["sharpe"], s0["sharpe"])
        lines.append(f"{r['scenario']:10s}           {r['sharpe']:.3f}  ({d:+.1f}%)")
    lines += ["",
              "Breaking point (S6):",
              f"  Total slippage: {s6.get('total_slip', 0):.3f}%",
              f"  Entry: {s6['es']:.3f}%  SL: {s6['ss']:.3f}%",
              f"  Sharpe at break: {s6['sharpe']:.3f}",
              "",
              f"Paper sim (S1 total): {rows[1]['es']+rows[1]['ss']:.2f}%",
              f"Safety margin: {s6.get('total_slip', 0) / max(rows[1]['es']+rows[1]['ss'], 0.01):.1f}x"]
    ax5.text(0.02, 0.98, "\n".join(lines), transform=ax5.transAxes, va="top",
             family="monospace", fontsize=9, color=PLT_FG)

    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"  saved: {out_path}")


# ═══════════════════ MAIN ══════════════════════════════════════════
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 66)
    print("PENETRATION TEST — BTC/USDT 15m 730d")
    print("=" * 66)

    print("\n[1/4] Loading data...")
    df = load_candles("BTC/USDT", "15m", days=730)
    htf = compute_htf_bias(df, htf_ema_period=50)
    print(f"  bars: {len(df)}  htf bias computed")

    print("\n[2/4] Baseline backtest (HTF=ON, validated params)...")
    baseline = fast_backtest(df, htf_bias=htf, **VALIDATED_PARAMS)
    print(f"  trades={baseline.total_trades} Sharpe={baseline.sharpe_ratio:.3f} "
          f"Ann={baseline.annual_return_pct:.1f}% WR={baseline.winrate:.1f}% "
          f"DD={baseline.max_drawdown_pct:.1f}%")

    # Sanity: S0 via slippage=0 must match baseline
    s0_check = fast_backtest_with_slippage(
        df, entry_slippage_pct=0, sl_slippage_pct=0, tp_slippage_pct=0,
        signal_delay=0, htf_bias=htf, **VALIDATED_PARAMS,
    )
    delta = abs(s0_check.sharpe_ratio - baseline.sharpe_ratio)
    print(f"  S0 sanity: slippage-wrapper Sharpe={s0_check.sharpe_ratio:.3f} "
          f"(delta={delta:.4f})")
    if delta > 0.05:
        print(f"  !! WARNING: S0 deviates from baseline by {delta:.3f} — bug in wrapper")

    print("\n[3/4] ATTACK 1 — Stop hunt analysis...")
    records = stop_hunt_analysis(df, baseline)
    sh = plot_stop_hunt(records, baseline, OUTPUT_DIR / "penetration_stop_hunt_BTCUSDT.png")

    print("\n[4/4] ATTACK 2 — Slippage & latency stress...")
    rows, curves = run_scenarios(df, htf)
    plot_slippage(rows, curves, OUTPUT_DIR / "penetration_slippage_BTCUSDT.png")

    # ── Scenario table ────────────────────────────────────────────
    print("\n" + "=" * 90)
    print("SLIPPAGE SCENARIO TABLE")
    print("=" * 90)
    print(f"{'Scenario':<11}{'Entry':>8}{'SL':>8}{'TP':>8}{'Del':>5}"
          f"{'Sharpe':>9}{'Ann%':>8}{'ExpR':>8}{'WR%':>7}{'DD%':>7}{'vs S0':>9}  Status")
    base_s = rows[0]["sharpe"]
    for r in rows:
        vs = (r["sharpe"] - base_s) / abs(base_s) * 100 if base_s else 0
        status = classify_scenario(r["sharpe"], r["expR"])
        mark = {"SAFE": "✓", "MARGINAL": "~", "DANGER": "✗"}[status]
        print(f"{r['scenario']:<11}{r['es']:>7.2f}%{r['ss']:>7.2f}%{r['ts']:>7.2f}%"
              f"{r['delay']:>5}{r['sharpe']:>9.3f}{r['ann']:>7.1f}%{r['expR']:>8.3f}"
              f"{r['wr']:>6.1f}%{r['dd']:>6.1f}%{vs:>8.1f}%  {mark} {status}")

    # ── Final report ──────────────────────────────────────────────
    s1 = rows[1]; s2 = rows[2]; s5 = rows[5]; s6 = rows[6]
    s1_st = classify_scenario(s1["sharpe"], s1["expR"])
    s2_st = classify_scenario(s2["sharpe"], s2["expR"])
    s5_st = classify_scenario(s5["sharpe"], s5["expR"])
    total_slip_break = s6.get("total_slip", 0)
    paper_total = s1["es"] + s1["ss"]
    safety = total_slip_break / max(paper_total, 0.01)

    # Stop hunt recommendation
    if sh.get("rec10", 0) > 20:
        rec1 = "Test 2.15×ATR or randomize SL offset"
    elif sh.get("rec10", 0) > 10:
        rec1 = "Monitor in paper trading"
    else:
        rec1 = "Keep 2.0×ATR"

    # Slippage recommendation
    if s2_st == "DANGER" or s1_st == "DANGER":
        rec2 = "STOP — not viable at realistic slippage"
    elif s2_st == "MARGINAL":
        rec2 = "Increase slippage buffer before live"
    else:
        rec2 = "Safe to paper trade"

    # Overall verdict
    critical = []
    if sh.get("rec10", 0) > 20:
        critical.append("High stop hunt rate")
    if s2_st == "DANGER":
        critical.append("Strategy fails at realistic slippage")
    if safety < 2:
        critical.append(f"Safety margin < 2x ({safety:.1f}x)")

    if not critical:
        verdict = "YES"; action = "NONE"
    elif len(critical) == 1 and "Monitor" in rec1:
        verdict = "PARTIAL"; action = "; ".join(critical)
    else:
        verdict = "PARTIAL" if len(critical) <= 1 else "NO"
        action = "; ".join(critical)

    print("\n")
    print("  ╔══════════════════════════════════════════════════════════════╗")
    print("  ║         PENETRATION TEST REPORT — BTC/USDT 15m              ║")
    print("  ╠══════════════════════════════════════════════════════════════╣")
    print("  ║ ATTACK 1 — STOP HUNT                                         ║")
    print(f"  ║  SL trades analyzed:    {sh.get('n_sl',0):<37} ║")
    print(f"  ║  Stop hunt rate (10b):  {sh.get('rec10',0):.1f}%  [{sh.get('verdict','?')}]{' '*(33-len(sh.get('verdict','?'))-len(f'{sh.get(chr(114)+chr(101)+chr(99)+chr(49)+chr(48),0):.1f}'))}║")
    print(f"  ║  SL proximity cluster:  {sh.get('prox_low',0):.1f}%  [{sh.get('clustering','?')}]{' '*(35-len(sh.get('clustering','?'))-len(f'{sh.get(chr(112)+chr(114)+chr(111)+chr(120)+chr(95)+chr(108)+chr(111)+chr(119),0):.1f}'))}║")
    print(f"  ║  SL at 2.0×ATR:         [{sh.get('sl_verdict','?')}]{' '*(36-len(sh.get('sl_verdict','?')))}║")
    print("  ║                                                              ║")
    print(f"  ║  Recommendation: {rec1:<44}║")
    print("  ╠══════════════════════════════════════════════════════════════╣")
    print("  ║ ATTACK 2 — SLIPPAGE                                          ║")
    print(f"  ║  Paper sim (S1):        Sharpe={s1['sharpe']:.2f}  [{s1_st}]{' '*(28-len(s1_st))}║")
    print(f"  ║  Real exchange (S2):    Sharpe={s2['sharpe']:.2f}  [{s2_st}]{' '*(28-len(s2_st))}║")
    print(f"  ║  Worst case (S5):       Sharpe={s5['sharpe']:.2f}  [{s5_st}]{' '*(28-len(s5_st))}║")
    print(f"  ║  Breaking point:        {total_slip_break:.3f}% total slippage{' '*(21-len(f'{total_slip_break:.3f}'))}║")
    print(f"  ║  Safety margin:         {safety:.1f}x above paper sim{' '*(21-len(f'{safety:.1f}'))}║")
    print("  ║                                                              ║")
    print(f"  ║  Recommendation: {rec2:<44}║")
    print("  ╠══════════════════════════════════════════════════════════════╣")
    print("  ║ OVERALL VERDICT                                              ║")
    print(f"  ║  Strategy passes penetration test: [{verdict}]{' '*(22-len(verdict))}║")
    print("  ║                                                              ║")
    print(f"  ║  Critical vulnerabilities: {action[:33]:<33} ║")
    print("  ╚══════════════════════════════════════════════════════════════╝")

    final = "PASSES" if verdict == "YES" else ("PARTIALLY PASSES" if verdict == "PARTIAL" else "FAILS")
    action_txt = "Safe to proceed" if not critical else f"Fix: {action}"
    print(f"\nStrategy {final} penetration test. {action_txt}.")


if __name__ == "__main__":
    main()
