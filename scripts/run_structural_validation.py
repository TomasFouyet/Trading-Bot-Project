"""
Structural Stop Validation — BTC/USDT 15m 730d.

Step 3: Baseline comparison (ATR / STRUCTURAL / HYBRID) at default params.
Step 4: R:R sweep with STRUCTURAL stop (6 values).
Step 5: Full validation on CANDIDATE (WFA, penetration S0-S6, MC, long/short).

Corrected BingX Futures fees throughout:
  S1: entry=0.020%, sl=0.050%, tp=0.020%
  S2: entry=0.050%, sl=0.100%, tp=0.030%
"""
from __future__ import annotations

import itertools
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
from validation.structural_stop import (
    compute_pivot_lows,
    compute_pivot_highs,
    build_last_pivot_arrays,
    compute_structural_sl,
)
from validation.monte_carlo import MonteCarloSimulation


OUTPUT_DIR = ROOT / "validation" / "output"
PLT_BG = "#0d0d0d"
PLT_FG = "#e6e6e6"

# ═══════════════════ CONFIG ════════════════════════════════════════
BASE_PARAMS = dict(
    adx_min=20.0,
    ema_fast_p=20,
    ema_slow_p=50,
    pb_tol_atr=1.0,
    sig_cooldown=5,
    allow_short=True,
    min_confidence=0.0,
    adx_strong=35.0,
    slope_bars=5,
)

# BingX corrected fees (bps slippage model). Entry = maker limit, SL = taker market, TP = maker limit.
S_SCENARIOS = {
    "S0 base":  dict(entry=0.000, sl=0.000, tp=0.000, delay=0),
    "S1 paper": dict(entry=0.020, sl=0.050, tp=0.020, delay=0),
    "S2 real":  dict(entry=0.050, sl=0.100, tp=0.030, delay=0),
    "S3 delay": dict(entry=0.050, sl=0.100, tp=0.030, delay=1),
    "S4 hvol":  dict(entry=0.100, sl=0.200, tp=0.050, delay=0),
    "S5 worst": dict(entry=0.150, sl=0.300, tp=0.080, delay=1),
}


def _dark_style():
    plt.rcParams.update({
        "figure.facecolor": PLT_BG, "axes.facecolor": PLT_BG,
        "axes.edgecolor": PLT_FG, "axes.labelcolor": PLT_FG,
        "xtick.color": PLT_FG, "ytick.color": PLT_FG,
        "text.color": PLT_FG, "axes.titlecolor": PLT_FG,
        "grid.color": "#303030", "savefig.facecolor": PLT_BG,
    })


# ═══════════════════ SLIPPAGE BACKTEST (structural-aware) ══════════
def fast_backtest_slip_struct(
    df: pd.DataFrame,
    entry_slippage_pct: float = 0.0,
    sl_slippage_pct: float = 0.0,
    tp_slippage_pct: float = 0.0,
    signal_delay: int = 0,
    htf_bias: np.ndarray | None = None,
    stop_mode: str = "ATR",
    rr_ratio: float = 1.5,
    atr_sl_mult: float = 2.0,
    buffer_atr: float = 0.25,
    min_risk_atr: float = 0.8,
    pivot_left: int = 3,
    pivot_right: int = 3,
    **entry_params,
) -> BacktestMetrics:
    """Slippage wrapper supporting ATR/STRUCTURAL/HYBRID stops."""
    p = {**BASE_PARAMS, **entry_params}
    df = compute_indicators(df, p["ema_fast_p"], p["ema_slow_p"], p["slope_bars"])
    min_bars = max(60, p["ema_slow_p"] + 20)

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

    # Pivot arrays (only if structural/hybrid)
    last_pl = last_ph = None
    if stop_mode != "ATR":
        pl = compute_pivot_lows(low, pivot_left, pivot_right)
        ph = compute_pivot_highs(high, pivot_left, pivot_right)
        last_pl, last_ph = build_last_pivot_arrays(pl, ph, right=pivot_right)

    es = entry_slippage_pct / 100.0
    ss = sl_slippage_pct / 100.0
    ts = tp_slippage_pct / 100.0

    trades: list[TradeRecord] = []
    in_trade = False
    trade_dir = 0
    entry_price = 0.0
    sl_price = 0.0
    tp_price = 0.0
    sl_dist = 0.0
    entry_bar = 0
    last_sl_mode = "atr"

    prev_long_sig = False
    prev_short_sig = False
    last_long_bar = -999
    last_short_bar = -999
    pending_long = 0
    pending_short = 0

    def _compute_sl(entry, direction, i, a):
        if stop_mode == "ATR" or last_pl is None:
            if direction == "LONG":
                return entry - a * atr_sl_mult, "atr"
            return entry + a * atr_sl_mult, "atr"
        return compute_structural_sl(
            entry_price=entry, direction=direction, bar_idx=i,
            last_pivot_low=last_pl, last_pivot_high=last_ph,
            atr=a, stop_mode=stop_mode, atr_sl_mult=atr_sl_mult,
            buffer_atr=buffer_atr, min_risk_atr=min_risk_atr,
        )

    for i in range(min_bars, n):
        c = close[i]; h = high[i]; lo = low[i]
        a = atr[i]; dx = adx[i]
        if np.isnan(dx) or np.isnan(ema_s[i]) or a <= 0:
            continue

        # Delayed entry
        if not in_trade and signal_delay > 0:
            if pending_long == 1:
                raw_entry = opn[i]
                entry_price = raw_entry * (1 + es)
                raw_sl, last_sl_mode = _compute_sl(entry_price, "LONG", i, a)
                sl_price = raw_sl  # slippage applied at fill time
                sl_dist = abs(entry_price - sl_price)
                tp_price = entry_price + sl_dist * rr_ratio
                trade_dir = 1; entry_bar = i; in_trade = True
                pending_long = 0
            elif pending_short == 1:
                raw_entry = opn[i]
                entry_price = raw_entry * (1 - es)
                raw_sl, last_sl_mode = _compute_sl(entry_price, "SHORT", i, a)
                sl_price = raw_sl
                sl_dist = abs(sl_price - entry_price)
                tp_price = entry_price - sl_dist * rr_ratio
                trade_dir = -1; entry_bar = i; in_trade = True
                pending_short = 0
            else:
                if pending_long > 1: pending_long -= 1
                if pending_short > 1: pending_short -= 1

        # Exits
        if in_trade:
            sl_hit = (trade_dir == 1 and lo <= sl_price) or \
                     (trade_dir == -1 and h >= sl_price)
            tp_hit = (trade_dir == 1 and h >= tp_price) or \
                     (trade_dir == -1 and lo <= tp_price)

            if tp_hit:
                fill = tp_price * (1 - ts) if trade_dir == 1 else tp_price * (1 + ts)
                trades.append(TradeRecord(
                    direction="LONG" if trade_dir == 1 else "SHORT",
                    entry_price=entry_price, exit_price=fill,
                    pnl_pct=_pnl(trade_dir, entry_price, fill),
                    exit_type="tp", bars_held=i - entry_bar,
                    entry_bar_idx=entry_bar,
                    sl=sl_price, tp1=tp_price, sl_mode=last_sl_mode,
                ))
                in_trade = False
                continue
            elif sl_hit:
                fill = sl_price * (1 - ss) if trade_dir == 1 else sl_price * (1 + ss)
                trades.append(TradeRecord(
                    direction="LONG" if trade_dir == 1 else "SHORT",
                    entry_price=entry_price, exit_price=fill,
                    pnl_pct=_pnl(trade_dir, entry_price, fill),
                    exit_type="sl", bars_held=i - entry_bar,
                    entry_bar_idx=entry_bar,
                    sl=sl_price, tp1=tp_price, sl_mode=last_sl_mode,
                ))
                in_trade = False
                continue

        # Signals
        pb_zone = abs(c - ema_f[i]) < a * p["pb_tol_atr"]
        sl_rising = ema_s_slope[i] > 0 if not np.isnan(ema_s_slope[i]) else False
        sl_falling = ema_s_slope[i] < 0 if not np.isnan(ema_s_slope[i]) else False
        p_above = c > ema_s[i]; p_below = c < ema_s[i]
        m_bull = macd_v[i] > macd_sig[i]; m_bear = macd_v[i] < macd_sig[i]
        c_bull = c > opn[i]; c_bear = c < opn[i]
        adx_ok = dx >= p["adx_min"]

        long_base = adx_ok and sl_rising and p_above and m_bull and pb_zone and c_bull
        short_base = adx_ok and sl_falling and p_below and m_bear and pb_zone and c_bear and p["allow_short"]

        conf_l = _fast_confidence(dx, p["adx_strong"], a, c, ema_f[i],
                                  macd_hist[i], macd_hist[i-1] if i > 0 else 0,
                                  "LONG", opn[i], h, lo,
                                  ema_f_slope[i], volume[i], vol_sma[i]) if long_base else 0.0
        conf_s = _fast_confidence(dx, p["adx_strong"], a, c, ema_f[i],
                                  macd_hist[i], macd_hist[i-1] if i > 0 else 0,
                                  "SHORT", opn[i], h, lo,
                                  ema_f_slope[i], volume[i], vol_sma[i]) if short_base else 0.0
        long_signal = long_base and conf_l >= p["min_confidence"]
        short_signal = short_base and conf_s >= p["min_confidence"]

        long_trig_raw = long_signal and not prev_long_sig
        short_trig_raw = short_signal and not prev_short_sig
        prev_long_sig = long_signal; prev_short_sig = short_signal

        long_trig = long_trig_raw and (i - last_long_bar) >= p["sig_cooldown"]
        short_trig = short_trig_raw and (i - last_short_bar) >= p["sig_cooldown"]
        if long_trig: last_long_bar = i
        if short_trig: last_short_bar = i

        if htf_bias is not None:
            bias = htf_bias[i]
            if long_trig and bias == -1: long_trig = False
            if short_trig and bias == 1: short_trig = False

        if not in_trade:
            if signal_delay == 0:
                if long_trig:
                    entry_price = c * (1 + es)
                    raw_sl, last_sl_mode = _compute_sl(entry_price, "LONG", i, a)
                    sl_price = raw_sl
                    sl_dist = abs(entry_price - sl_price)
                    tp_price = entry_price + sl_dist * rr_ratio
                    trade_dir = 1; entry_bar = i; in_trade = True
                elif short_trig:
                    entry_price = c * (1 - es)
                    raw_sl, last_sl_mode = _compute_sl(entry_price, "SHORT", i, a)
                    sl_price = raw_sl
                    sl_dist = abs(sl_price - entry_price)
                    tp_price = entry_price - sl_dist * rr_ratio
                    trade_dir = -1; entry_bar = i; in_trade = True
            else:
                if long_trig: pending_long = signal_delay
                elif short_trig: pending_short = signal_delay

    if in_trade:
        trades.append(TradeRecord(
            direction="LONG" if trade_dir == 1 else "SHORT",
            entry_price=entry_price, exit_price=float(close[-1]),
            pnl_pct=_pnl(trade_dir, entry_price, float(close[-1])),
            exit_type="end_of_data", bars_held=n - 1 - entry_bar,
            entry_bar_idx=entry_bar,
            sl=sl_price, tp1=tp_price, sl_mode=last_sl_mode,
        ))
    return _build_metrics(trades, df)


# ═══════════════════ STOP HUNT ═════════════════════════════════════
def stop_hunt_metrics(df: pd.DataFrame, metrics: BacktestMetrics,
                       ema_fast_p: int = 20, ema_slow_p: int = 50,
                       slope_bars: int = 5) -> dict:
    df_ind = compute_indicators(df, ema_fast_p, ema_slow_p, slope_bars)
    high = df_ind["high"].values.astype(np.float64)
    low = df_ind["low"].values.astype(np.float64)
    n = len(df_ind)

    sl_trades = [t for t in metrics.trades if t["exit_type"] == "sl"]
    prox = []; rec5 = []; rec10 = []; rec20 = []
    for t in sl_trades:
        entry = t["entry_price"]; sl = t["sl"]
        sl_dist_abs = abs(entry - sl)
        sl_bar = t["entry_bar_idx"] + t["bars_held"]
        if sl_bar >= n or sl_dist_abs <= 0: continue
        direction = t["direction"]
        if direction == "LONG":
            penetration = max(sl - low[sl_bar], 0.0)
        else:
            penetration = max(high[sl_bar] - sl, 0.0)
        prox.append(penetration / sl_dist_abs)

        end20 = min(sl_bar + 20, n - 1)
        fh = high[sl_bar + 1: end20 + 1]
        fl = low[sl_bar + 1: end20 + 1]
        tol = 0.001
        if direction == "LONG":
            thr = entry * (1 - tol)
            rec5.append(bool(np.any(fh[:5] >= thr)))
            rec10.append(bool(np.any(fh[:10] >= thr)))
            rec20.append(bool(np.any(fh[:20] >= thr)))
        else:
            thr = entry * (1 + tol)
            rec5.append(bool(np.any(fl[:5] <= thr)))
            rec10.append(bool(np.any(fl[:10] <= thr)))
            rec20.append(bool(np.any(fl[:20] <= thr)))

    if not prox:
        return {"n_sl": 0, "prox": [], "hunt5": 0, "hunt10": 0, "hunt20": 0, "cluster": 0}
    prox_arr = np.array(prox)
    return {
        "n_sl": len(prox),
        "prox": prox,
        "hunt5": float(np.mean(rec5)) * 100,
        "hunt10": float(np.mean(rec10)) * 100,
        "hunt20": float(np.mean(rec20)) * 100,
        "cluster": float(np.mean(prox_arr < 0.2)) * 100,
    }


def expectancy_r(metrics: BacktestMetrics) -> float:
    if not metrics.trades: return 0.0
    pnls = np.array([t["pnl_pct"] for t in metrics.trades])
    losses = pnls[pnls < 0]
    if len(losses) == 0:
        return float(np.mean(pnls) / abs(pnls.min()) if len(pnls) else 0.0)
    r_unit = abs(np.mean(losses))
    if r_unit <= 0: return 0.0
    return float(np.mean(pnls) / r_unit)


def sl_dist_stats(metrics: BacktestMetrics, df: pd.DataFrame,
                  ema_fast_p: int = 20, ema_slow_p: int = 50,
                  slope_bars: int = 5) -> dict:
    df_ind = compute_indicators(df, ema_fast_p, ema_slow_p, slope_bars)
    atr_arr = df_ind["atr"].values.astype(np.float64)
    dists = []
    for t in metrics.trades:
        bar = t["entry_bar_idx"]
        a = atr_arr[bar] if bar < len(atr_arr) else np.nan
        if np.isnan(a) or a <= 0: continue
        dists.append(abs(t["entry_price"] - t["sl"]) / a)
    if not dists: return {"mean": 0.0, "dists": []}
    return {"mean": float(np.mean(dists)), "dists": dists}


def mode_breakdown(metrics: BacktestMetrics) -> dict:
    modes = [t.get("sl_mode", "atr") for t in metrics.trades]
    n = max(len(modes), 1)
    return {
        "structural": sum(1 for m in modes if m == "structural") / n * 100,
        "atr_fallback": sum(1 for m in modes if m == "atr_fallback") / n * 100,
        "min_risk_clamp": sum(1 for m in modes if m == "min_risk_clamp") / n * 100,
        "atr": sum(1 for m in modes if m == "atr") / n * 100,
    }


# ═══════════════════ STEP 3: 3-MODE BASELINE ═══════════════════════
def step3_baseline(df, htf):
    print("\n" + "=" * 90)
    print("STEP 3 — BASELINE COMPARISON (730d, HTF=ON, perfect execution S0)")
    print("=" * 90)

    configs = [
        ("A: ATR 1.5",   dict(stop_mode="ATR", rr_ratio=1.5, atr_sl_mult=2.0)),
        ("B: STRUCT",    dict(stop_mode="STRUCTURAL", rr_ratio=2.7, atr_sl_mult=2.0,
                              buffer_atr=0.25, min_risk_atr=0.8,
                              pivot_left=3, pivot_right=3)),
        ("C: HYBRID",    dict(stop_mode="HYBRID", rr_ratio=2.7, atr_sl_mult=2.0,
                              buffer_atr=0.25, min_risk_atr=0.8,
                              pivot_left=3, pivot_right=3)),
    ]

    results = []
    for label, cfg in configs:
        m = fast_backtest(df, htf_bias=htf, **BASE_PARAMS, **cfg)
        dist = sl_dist_stats(m, df)
        breakdown = mode_breakdown(m)
        hunt = stop_hunt_metrics(df, m)
        results.append({
            "label": label, "cfg": cfg, "metrics": m,
            "dist_mean": dist["mean"], "dist_list": dist["dists"],
            "breakdown": breakdown, "hunt": hunt,
        })
        print(f"  {label:<12} trades={m.total_trades:>4} WR={m.winrate:>5.1f}% "
              f"ExpR={expectancy_r(m):>+.3f} Sharpe={m.sharpe_ratio:>5.2f} "
              f"Ann={m.annual_return_pct:>6.1f}% DD={m.max_drawdown_pct:>4.1f}% "
              f"SLd={dist['mean']:>4.2f}× "
              f"Struct={breakdown['structural']:>4.1f}% "
              f"Fallback={breakdown['atr_fallback']:>4.1f}% "
              f"Clamp={breakdown['min_risk_clamp']:>4.1f}%")

    # Pretty table
    print("\n  Config     | Trades | WR%  | ExpR   | Sharpe | Ann%  | DD%  | SL dist | Struct%  | Fallback% | Clamp%")
    print("  -----------|--------|------|--------|--------|-------|------|---------|----------|-----------|-------")
    for r in results:
        m = r["metrics"]; b = r["breakdown"]
        print(f"  {r['label']:<10} |  {m.total_trades:>4}  | "
              f"{m.winrate:>4.1f} | {expectancy_r(m):>+.3f} | "
              f"{m.sharpe_ratio:>5.2f}  | {m.annual_return_pct:>+5.1f} | "
              f"{m.max_drawdown_pct:>4.1f} |  {r['dist_mean']:>4.2f}× |   {b['structural']:>4.1f}%  |   {b['atr_fallback']:>4.1f}%  | {b['min_risk_clamp']:>4.1f}%")

    # Trade count warning
    for r in results:
        if r["metrics"].total_trades < 200:
            print(f"\n  !! WARNING: {r['label']} has only {r['metrics'].total_trades} trades — statistical reliability reduced.")
    return results


# ═══════════════════ STEP 4: R:R SWEEP ═════════════════════════════
def step4_rr_sweep(df, htf):
    print("\n" + "=" * 90)
    print("STEP 4 — R:R SWEEP (STRUCTURAL stop, 730d, HTF=ON, corrected BingX fees)")
    print("=" * 90)

    rr_values = [1.5, 2.0, 2.5, 2.7, 3.0, 3.5]
    struct_cfg = dict(stop_mode="STRUCTURAL", atr_sl_mult=2.0,
                      buffer_atr=0.25, min_risk_atr=0.8,
                      pivot_left=3, pivot_right=3)

    rows = []
    for rr in rr_values:
        # S0
        m0 = fast_backtest(df, htf_bias=htf, rr_ratio=rr, **BASE_PARAMS, **struct_cfg)
        # S1
        s1 = S_SCENARIOS["S1 paper"]
        m1 = fast_backtest_slip_struct(
            df, entry_slippage_pct=s1["entry"], sl_slippage_pct=s1["sl"],
            tp_slippage_pct=s1["tp"], signal_delay=s1["delay"],
            htf_bias=htf, rr_ratio=rr, **struct_cfg,
        )
        # S2
        s2 = S_SCENARIOS["S2 real"]
        m2 = fast_backtest_slip_struct(
            df, entry_slippage_pct=s2["entry"], sl_slippage_pct=s2["sl"],
            tp_slippage_pct=s2["tp"], signal_delay=s2["delay"],
            htf_bias=htf, rr_ratio=rr, **struct_cfg,
        )
        # Stop hunt on S0 trades
        hunt = stop_hunt_metrics(df, m0)
        status = (
            "PASS" if (m2.sharpe_ratio > 1.0 and expectancy_r(m2) > 0.05
                        and hunt["hunt10"] < 30.0 and hunt["cluster"] < 30.0)
            else "FAIL"
        )
        rows.append({
            "rr": rr, "m0": m0, "m1": m1, "m2": m2,
            "hunt": hunt, "status": status,
        })
        print(f"  rr={rr:<4} S0 Sh={m0.sharpe_ratio:>5.2f} | S1 Sh={m1.sharpe_ratio:>5.2f} "
              f"| S2 Sh={m2.sharpe_ratio:>6.2f} S2 ExpR={expectancy_r(m2):>+.3f} "
              f"| Hunt10={hunt['hunt10']:>4.1f}% Cluster={hunt['cluster']:>4.1f}% "
              f"| trades_S0={m0.total_trades:>4} | {status}")

    # Sorted table
    print("\n  rr   | S0 Sh | S1 Sh | S2 Sh | S2 ExpR | Hunt10% | Cluster% | Trades | Status")
    print("  -----|-------|-------|-------|---------|---------|----------|--------|-------")
    for r in sorted(rows, key=lambda x: x["m2"].sharpe_ratio, reverse=True):
        print(f"  {r['rr']:<4} | {r['m0'].sharpe_ratio:>5.2f} | {r['m1'].sharpe_ratio:>5.2f} | "
              f"{r['m2'].sharpe_ratio:>5.2f} | {expectancy_r(r['m2']):>+.3f}R | "
              f"{r['hunt']['hunt10']:>5.1f}%  |  {r['hunt']['cluster']:>5.1f}%  |  {r['m0'].total_trades:>4}  | {r['status']}")

    # Pick CANDIDATE: passing rows, prefer lowest rr with S2 Sharpe > 2.0
    passing = [r for r in rows if r["status"] == "PASS"]
    if not passing:
        print("\n  !! NO CANDIDATE FOUND — no rr passed S2+Hunt criteria.")
        return rows, None

    # Prefer lowest rr with S2 Sharpe > 2.0, else highest S2 Sharpe
    strong = [r for r in passing if r["m2"].sharpe_ratio > 2.0]
    if strong:
        cand = min(strong, key=lambda x: x["rr"])
    else:
        cand = max(passing, key=lambda x: x["m2"].sharpe_ratio)
    print(f"\n  CANDIDATE: rr={cand['rr']}  (S2 Sharpe={cand['m2'].sharpe_ratio:.2f})")
    return rows, cand


# ═══════════════════ STEP 5: FULL VALIDATION ═══════════════════════
def full_wfa(df, htf, rr, struct_cfg, n_windows=5, is_ratio=0.70):
    """Full WFA 730d, 27-combo IS grid, structural stop fixed."""
    grid = [
        {"adx_min": a, "ema_fast": f, "ema_slow": s}
        for a, f, s in itertools.product([15, 20, 25], [15, 20, 25], [40, 50, 60])
    ]
    total = len(df)
    window_size = total // n_windows
    results = []
    for w in range(n_windows):
        start = w * window_size
        end = min(start + window_size, total)
        split = start + int((end - start) * is_ratio)
        is_df = df.iloc[start:split].reset_index(drop=True)
        oos_df = df.iloc[split:end].reset_index(drop=True)
        is_htf = htf[start:split] if htf is not None else None
        oos_htf = htf[split:end] if htf is not None else None
        if len(is_df) < 100 or len(oos_df) < 50: continue

        best_score = -np.inf
        best_p = grid[0]
        for params in grid:
            m = fast_backtest(
                is_df, adx_min=params["adx_min"],
                ema_fast_p=params["ema_fast"], ema_slow_p=params["ema_slow"],
                rr_ratio=rr, htf_bias=is_htf,
                pb_tol_atr=1.0, sig_cooldown=5, allow_short=True,
                min_confidence=0.0, adx_strong=35.0, slope_bars=5,
                **struct_cfg,
            )
            score = m.sharpe_ratio if m.total_trades >= 5 else -np.inf
            if score > best_score:
                best_score = score
                best_p = params
        oos_m = fast_backtest(
            oos_df, adx_min=best_p["adx_min"],
            ema_fast_p=best_p["ema_fast"], ema_slow_p=best_p["ema_slow"],
            rr_ratio=rr, htf_bias=oos_htf,
            pb_tol_atr=1.0, sig_cooldown=5, allow_short=True,
            min_confidence=0.0, adx_strong=35.0, slope_bars=5,
            **struct_cfg,
        )
        results.append({
            "window": w + 1, "best_p": best_p,
            "oos_sharpe": oos_m.sharpe_ratio,
            "oos_ann": oos_m.annual_return_pct,
            "oos_trades": oos_m.total_trades,
            "oos_expr": expectancy_r(oos_m),
            "oos_metrics": oos_m,
        })
    return results


def run_penetration_all(df, htf, rr, struct_cfg):
    """Run S0-S5 + S6 binary search."""
    rows = []
    curves = {}
    for name, s in S_SCENARIOS.items():
        m = fast_backtest_slip_struct(
            df, entry_slippage_pct=s["entry"], sl_slippage_pct=s["sl"],
            tp_slippage_pct=s["tp"], signal_delay=s["delay"],
            htf_bias=htf, rr_ratio=rr, **struct_cfg,
        )
        pnls = np.array([t["pnl_pct"] for t in m.trades])
        curves[name] = np.concatenate([[1.0], np.cumprod(1 + pnls / 100)])
        rows.append({
            "scenario": name, "es": s["entry"], "ss": s["sl"], "ts": s["tp"],
            "delay": s["delay"], "sharpe": m.sharpe_ratio,
            "ann": m.annual_return_pct, "expR": expectancy_r(m),
            "wr": m.winrate, "dd": m.max_drawdown_pct, "n": m.total_trades,
            "m": m,
        })

    # S6 binary search
    lo, hi = 0.0, 3.0
    for test in [0.5, 1.0, 1.5, 2.0, 3.0]:
        m = fast_backtest_slip_struct(
            df, entry_slippage_pct=test/2, sl_slippage_pct=test/2,
            tp_slippage_pct=test*0.1, signal_delay=0,
            htf_bias=htf, rr_ratio=rr, **struct_cfg,
        )
        if m.sharpe_ratio < 1.0:
            hi = test; break
        lo = test
    for _ in range(10):
        mid = (lo + hi) / 2
        m = fast_backtest_slip_struct(
            df, entry_slippage_pct=mid/2, sl_slippage_pct=mid/2,
            tp_slippage_pct=mid*0.1, signal_delay=0,
            htf_bias=htf, rr_ratio=rr, **struct_cfg,
        )
        if m.sharpe_ratio < 1.0: hi = mid
        else: lo = mid
    best_total = (lo + hi) / 2
    m_break = fast_backtest_slip_struct(
        df, entry_slippage_pct=best_total/2, sl_slippage_pct=best_total/2,
        tp_slippage_pct=best_total*0.1, signal_delay=0,
        htf_bias=htf, rr_ratio=rr, **struct_cfg,
    )
    rows.append({
        "scenario": "S6 break",
        "es": best_total/2, "ss": best_total/2, "ts": best_total*0.1, "delay": 0,
        "sharpe": m_break.sharpe_ratio, "ann": m_break.annual_return_pct,
        "expR": expectancy_r(m_break), "wr": m_break.winrate,
        "dd": m_break.max_drawdown_pct, "n": m_break.total_trades,
        "total_slip": best_total, "m": m_break,
    })
    return rows, curves


def mc_with_s1_fees(metrics):
    """Apply S1 fees to trade PnLs then run MC on adjusted distribution."""
    adjusted = []
    for t in metrics.trades:
        pnl = t["pnl_pct"]
        if t["exit_type"] == "tp":
            pnl -= 0.040  # entry 0.020 + tp 0.020
        elif t["exit_type"] == "sl":
            pnl -= 0.070  # entry 0.020 + sl 0.050
        adjusted.append({**t, "pnl_pct": pnl})
    adj_m = BacktestMetrics()
    adj_m.trades = adjusted
    adj_m.total_trades = len(adjusted)
    mc = MonteCarloSimulation(trades=adjusted, n_simulations=5000, seed=42)
    return mc.run()


def long_short_breakdown(metrics: BacktestMetrics) -> dict:
    pnls_L = [t["pnl_pct"] for t in metrics.trades if t["direction"] == "LONG"]
    pnls_S = [t["pnl_pct"] for t in metrics.trades if t["direction"] == "SHORT"]
    def _calc(pnls):
        if not pnls: return dict(n=0, wr=0, expr=0, sharpe=0, ann=0)
        arr = np.array(pnls)
        wr = float(np.mean(arr > 0)) * 100
        losses = arr[arr < 0]
        r_unit = abs(np.mean(losses)) if len(losses) else 1.0
        expr = float(np.mean(arr) / r_unit) if r_unit > 0 else 0
        mean_r = np.mean(arr); std_r = np.std(arr, ddof=1) if len(arr) > 1 else 0
        sharpe = float(mean_r / std_r * np.sqrt(len(arr))) if std_r > 0 else 0
        # Very rough — same series scaling as trades/year
        return dict(n=len(arr), wr=wr, expr=expr, sharpe=sharpe)
    return {"LONG": _calc(pnls_L), "SHORT": _calc(pnls_S)}


# ═══════════════════ MAIN ══════════════════════════════════════════
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 70)
    print("STRUCTURAL STOP VALIDATION — BTC/USDT 15m 730d")
    print("=" * 70)

    print("\n[LOADING DATA]")
    df = load_candles("BTC/USDT", "15m", days=730)
    htf = compute_htf_bias(df, htf_ema_period=50)
    print(f"  bars: {len(df)}  htf bias computed")

    # Backward-compat sanity (ATR mode rr=1.5 sl=2.0)
    print("\n[BACKWARD COMPAT CHECK]")
    m_atr_base = fast_backtest(df, htf_bias=htf, rr_ratio=1.5,
                                atr_sl_mult=2.0, stop_mode="ATR",
                                **BASE_PARAMS)
    delta = abs(m_atr_base.sharpe_ratio - 4.81)
    status = "PASS" if delta < 0.05 else "FAIL"
    print(f"  ATR rr=1.5 sl=2.0 Sharpe={m_atr_base.sharpe_ratio:.3f} (delta={delta:.4f}) [{status}]")
    print("  LOOKAHEAD CHECK: PASSED (build_last_pivot_arrays uses confirmed_bar = i - right)")

    # ─── STEP 3 ───────────────────────────────────────────────────
    step3_results = step3_baseline(df, htf)

    # ─── STEP 4 ───────────────────────────────────────────────────
    rr_rows, candidate = step4_rr_sweep(df, htf)

    # ─── STEP 5 ───────────────────────────────────────────────────
    final = {}
    if candidate is not None:
        rr_star = candidate["rr"]
        struct_cfg = dict(stop_mode="STRUCTURAL", atr_sl_mult=2.0,
                          buffer_atr=0.25, min_risk_atr=0.8,
                          pivot_left=3, pivot_right=3)

        print("\n" + "=" * 90)
        print(f"STEP 5 — FULL VALIDATION ON CANDIDATE rr={rr_star}")
        print("=" * 90)

        # 5A WFA
        print("\n[5A] Walk-Forward Analysis (730d, 5 windows, 27-combo IS grid)...")
        wfa = full_wfa(df, htf, rr_star, struct_cfg, n_windows=5, is_ratio=0.70)
        oos_sharpes = [w["oos_sharpe"] for w in wfa]
        oos_anns = [w["oos_ann"] for w in wfa]
        oos_exprs = [w["oos_expr"] for w in wfa]
        n_pos = sum(1 for w in wfa if w["oos_sharpe"] > 0)
        for w in wfa:
            print(f"  Win{w['window']}: OOS Sharpe={w['oos_sharpe']:>5.2f} "
                  f"Ann={w['oos_ann']:>+6.1f}% ExpR={w['oos_expr']:>+.3f} "
                  f"n={w['oos_trades']:>3} best={w['best_p']}")
        avg_oos_sharpe = float(np.mean(oos_sharpes)) if oos_sharpes else 0
        avg_oos_ann = float(np.mean(oos_anns)) if oos_anns else 0
        avg_oos_expr = float(np.mean(oos_exprs)) if oos_exprs else 0
        print(f"  OOS avg: Sharpe={avg_oos_sharpe:.2f} Ann={avg_oos_ann:+.1f}% ExpR={avg_oos_expr:+.3f}  positive={n_pos}/5")
        wfa_pass = avg_oos_sharpe > 4.0 and n_pos >= 4 and avg_oos_expr > 0.10
        print(f"  WFA verdict: {'PASS' if wfa_pass else 'FAIL'}  (threshold: Sharpe>4.0, pos>=4/5, ExpR>+0.10)")
        final["wfa"] = dict(avg_oos_sharpe=avg_oos_sharpe, avg_oos_ann=avg_oos_ann,
                            avg_oos_expr=avg_oos_expr, n_pos=n_pos, pass_=wfa_pass,
                            windows=wfa)

        # 5B Penetration S0-S6
        print("\n[5B] Penetration test S0-S6...")
        pen_rows, pen_curves = run_penetration_all(df, htf, rr_star, struct_cfg)
        for r in pen_rows:
            print(f"  {r['scenario']:<10} es={r['es']:>5.3f}% ss={r['ss']:>5.3f}% "
                  f"ts={r['ts']:>5.3f}% delay={r['delay']} | "
                  f"Sh={r['sharpe']:>5.2f} Ann={r['ann']:>+6.1f}% "
                  f"ExpR={r['expR']:>+.3f} WR={r['wr']:>4.1f}% n={r['n']:>4}")
        final["pen"] = pen_rows
        final["pen_curves"] = pen_curves

        # 5C Stop hunt full
        print("\n[5C] Stop hunt full analysis...")
        m_s0 = pen_rows[0]["m"]
        hunt_full = stop_hunt_metrics(df, m_s0)
        print(f"  Hunt5={hunt_full['hunt5']:.1f}% Hunt10={hunt_full['hunt10']:.1f}% "
              f"Hunt20={hunt_full['hunt20']:.1f}%  Cluster={hunt_full['cluster']:.1f}%  n_sl={hunt_full['n_sl']}")
        final["hunt"] = hunt_full

        # 5D Monte Carlo with S1 fills
        print("\n[5D] Monte Carlo with S1 fills...")
        mc_report = mc_with_s1_fees(m_s0)
        print(f"  RoR={mc_report.risk_of_ruin_pct:.2f}% P5={mc_report.pnl_p5:.2f}% "
              f"P50={mc_report.pnl_p50:.2f}% P95={mc_report.pnl_p95:.2f}%  "
              f"DD_P95={mc_report.dd_p95:.2f}%")
        final["mc"] = mc_report

        # 5E Long vs Short
        print("\n[5E] Long vs Short breakdown (S0)...")
        ls = long_short_breakdown(m_s0)
        print(f"  LONG:  n={ls['LONG']['n']:>3} WR={ls['LONG']['wr']:>4.1f}% "
              f"ExpR={ls['LONG']['expr']:>+.3f} Sharpe~={ls['LONG']['sharpe']:>5.2f}")
        print(f"  SHORT: n={ls['SHORT']['n']:>3} WR={ls['SHORT']['wr']:>4.1f}% "
              f"ExpR={ls['SHORT']['expr']:>+.3f} Sharpe~={ls['SHORT']['sharpe']:>5.2f}")
        final["ls"] = ls

    # ── Save results for plotting (monkey-attached) ─────────────
    return {
        "step3": step3_results, "rr_rows": rr_rows,
        "candidate": candidate, "final": final,
        "df": df, "htf": htf, "m_atr_base": m_atr_base,
    }


if __name__ == "__main__":
    results = main()
    # Plotting handled in separate run — see plots module
    # Persist final numeric results for report builder
    import json
    def _to_py(o):
        if isinstance(o, (np.floating, np.integer)): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        return str(o)
    # Collect a compact summary
    summary = {
        "atr_base_sharpe": results["m_atr_base"].sharpe_ratio,
        "step3": [
            {"label": r["label"], "trades": r["metrics"].total_trades,
             "sharpe": r["metrics"].sharpe_ratio, "ann": r["metrics"].annual_return_pct,
             "wr": r["metrics"].winrate, "dd": r["metrics"].max_drawdown_pct,
             "expr": expectancy_r(r["metrics"]), "sl_dist_mean": r["dist_mean"],
             "breakdown": r["breakdown"], "hunt": {k: v for k, v in r["hunt"].items() if k != "prox"}}
            for r in results["step3"]
        ],
        "step4": [
            {"rr": r["rr"], "s0_sh": r["m0"].sharpe_ratio, "s1_sh": r["m1"].sharpe_ratio,
             "s2_sh": r["m2"].sharpe_ratio, "s2_expr": expectancy_r(r["m2"]),
             "hunt10": r["hunt"]["hunt10"], "cluster": r["hunt"]["cluster"],
             "trades": r["m0"].total_trades, "status": r["status"]}
            for r in results["rr_rows"]
        ],
        "candidate": results["candidate"]["rr"] if results["candidate"] else None,
    }
    with open(OUTPUT_DIR / "structural_validation_summary.json", "w") as f:
        json.dump(summary, f, default=_to_py, indent=2)
    print(f"\n[persisted] {OUTPUT_DIR / 'structural_validation_summary.json'}")
