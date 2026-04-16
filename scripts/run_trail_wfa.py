#!/usr/bin/env python3
"""
Trailing Stop WFA Analysis — Steps 2-7
Tests hypothesis: trailing stop improves OOS Sharpe by >15%.
"""
import sys, os, itertools
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from validation.data_loader import load_candles
from validation.fast_backtest import (
    compute_indicators, compute_htf_bias, fast_backtest, _pnl
)

# ══════════════════════════════════════════════════════════════════
# STEP 2 — Single trade verification
# ══════════════════════════════════════════════════════════════════

def step2_verify_single_trade():
    """Find a wasted-MFE trade and verify trail improves it."""
    print("=" * 70)
    print("STEP 2 — SINGLE TRADE VERIFICATION")
    print("=" * 70)

    df = load_candles("BTC/USDT", "15m", days=730)
    htf_bias = compute_htf_bias(df, htf_ema_period=50)

    # Run baseline
    result_base = fast_backtest(
        df, adx_min=20, ema_fast_p=20, ema_slow_p=50,
        rr_ratio=1.5, atr_sl_mult=2.0,
        htf_bias=htf_bias, use_trail=False,
    )

    # Precompute indicators for bar-level access
    df_ind = compute_indicators(df, ema_fast_p=20, ema_slow_p=50, slope_bars=5)
    high = df_ind["high"].values
    low = df_ind["low"].values
    close = df_ind["close"].values
    atr_arr = df_ind["atr"].values

    # Find wasted-MFE trades (SL with MFE >= 0.8R)
    wasted = []
    for t in result_base.trades:
        if t["exit_type"] != "sl":
            continue
        idx = t["entry_bar_idx"]
        ep = t["entry_price"]
        direction = t["direction"]
        bars = t["bars_held"]
        sl_d = atr_arr[idx] * 2.0
        if sl_d <= 0:
            continue

        peak = ep
        for b in range(idx + 1, min(idx + bars + 1, len(high))):
            if direction == "LONG":
                peak = max(peak, high[b])
            else:
                peak = min(peak, low[b])

        mfe_r = ((peak - ep) / sl_d) if direction == "LONG" else ((ep - peak) / sl_d)
        if mfe_r >= 0.8:
            wasted.append((t, mfe_r, sl_d))

    print(f"Found {len(wasted)} wasted-MFE trades in baseline\n")

    # Pick one
    trade, mfe_val, sl_d = wasted[0]
    idx = trade["entry_bar_idx"]
    ep = trade["entry_price"]
    direction = trade["direction"]
    bars = trade["bars_held"]
    td = 1 if direction == "LONG" else -1
    sl_price = ep - sl_d if td == 1 else ep + sl_d
    tp_price = (ep + sl_d * 1.5) if td == 1 else (ep - sl_d * 1.5)

    print(f"Trade: {direction}, Entry bar={idx}, Entry={ep:.2f}")
    print(f"SL={sl_price:.2f}, TP={tp_price:.2f}, SL dist={sl_d:.2f}")
    print(f"Baseline: {trade['exit_type']}, PnL={trade['pnl_pct']:.4f}%, MFE={mfe_val:.3f}R\n")

    # Simulate bar-by-bar with trail
    peak_fav = ep
    trail_active = False
    trail_stop_v = 0.0
    act_r, tr_r = 0.8, 0.5
    exit_type = None
    exit_price = None

    print(f"{'Bar':>4} | {'High':>10} | {'Low':>10} | {'PeakFav':>10} | {'MFE_R':>6} | {'Trail?':>6} | {'TrailStop':>10} | Exit?")
    print("-" * 82)

    for b in range(idx + 1, idx + bars + 2):
        if b >= len(high):
            break
        h, lo_b = high[b], low[b]

        if td == 1:
            peak_fav = max(peak_fav, h)
            mfe_r = (peak_fav - ep) / sl_d
        else:
            peak_fav = min(peak_fav, lo_b)
            mfe_r = (ep - peak_fav) / sl_d

        if not trail_active and mfe_r >= act_r:
            trail_active = True
            trail_stop_v = (peak_fav - tr_r * sl_d) if td == 1 else (peak_fav + tr_r * sl_d)

        if trail_active:
            if td == 1:
                trail_stop_v = max(trail_stop_v, peak_fav - tr_r * sl_d)
                trail_stop_v = max(trail_stop_v, sl_price)
            else:
                trail_stop_v = min(trail_stop_v, peak_fav + tr_r * sl_d)
                trail_stop_v = min(trail_stop_v, sl_price)

        tp_hit = (td == 1 and h >= tp_price) or (td == -1 and lo_b <= tp_price)
        trail_hit = trail_active and ((td == 1 and lo_b <= trail_stop_v) or (td == -1 and h >= trail_stop_v))
        sl_hit = (td == 1 and lo_b <= sl_price) or (td == -1 and h >= sl_price)

        exit_str = ""
        if tp_hit:
            exit_str = "TP"
            if not exit_type: exit_type, exit_price = "tp", tp_price
        elif trail_hit:
            exit_str = "TRAIL"
            if not exit_type: exit_type, exit_price = "trail", trail_stop_v
        elif sl_hit:
            exit_str = "SL"
            if not exit_type: exit_type, exit_price = "sl", sl_price

        ts_str = f"{trail_stop_v:.2f}" if trail_active else "N/A"
        print(f"{b-idx:>4} | {h:>10.2f} | {lo_b:>10.2f} | {peak_fav:>10.2f} | {mfe_r:>6.3f} | {'Yes' if trail_active else 'No':>6} | {ts_str:>10} | {exit_str}")
        if exit_str:
            break

    print()
    if exit_type == "trail":
        trail_pnl = _pnl(td, ep, exit_price)
        base_pnl = trade["pnl_pct"]
        print(f"WITHOUT trail: exits at SL ({sl_price:.2f}), PnL = {base_pnl:+.4f}%")
        print(f"WITH trail:    exits at trail ({exit_price:.2f}), PnL = {trail_pnl:+.4f}%")
        print(f"Improvement: {trail_pnl - base_pnl:+.4f}%")
        print(f"\n>>> PASS — trail exit is better than SL exit")
        return True
    else:
        print(f"Exit type: {exit_type} — trying another trade...")
        # Try more trades
        for trade2, mfe2, sl_d2 in wasted[1:10]:
            idx2 = trade2["entry_bar_idx"]
            ep2 = trade2["entry_price"]
            dir2 = trade2["direction"]
            bars2 = trade2["bars_held"]
            td2 = 1 if dir2 == "LONG" else -1
            sl2 = ep2 - sl_d2 if td2 == 1 else ep2 + sl_d2
            tp2 = (ep2 + sl_d2 * 1.5) if td2 == 1 else (ep2 - sl_d2 * 1.5)

            pf2 = ep2
            ta2 = False
            ts2 = 0.0
            et2 = None
            ep2_exit = None

            for b in range(idx2 + 1, idx2 + bars2 + 2):
                if b >= len(high): break
                h2, lo2 = high[b], low[b]
                if td2 == 1: pf2 = max(pf2, h2); mr2 = (pf2 - ep2) / sl_d2
                else: pf2 = min(pf2, lo2); mr2 = (ep2 - pf2) / sl_d2
                if not ta2 and mr2 >= 0.8:
                    ta2 = True
                    ts2 = (pf2 - 0.5*sl_d2) if td2==1 else (pf2 + 0.5*sl_d2)
                if ta2:
                    if td2==1: ts2 = max(ts2, pf2-0.5*sl_d2); ts2 = max(ts2, sl2)
                    else: ts2 = min(ts2, pf2+0.5*sl_d2); ts2 = min(ts2, sl2)
                tp_h = (td2==1 and h2>=tp2) or (td2==-1 and lo2<=tp2)
                tr_h = ta2 and ((td2==1 and lo2<=ts2) or (td2==-1 and h2>=ts2))
                sl_h = (td2==1 and lo2<=sl2) or (td2==-1 and h2>=sl2)
                if tp_h: et2="tp"; ep2_exit=tp2; break
                elif tr_h: et2="trail"; ep2_exit=ts2; break
                elif sl_h: et2="sl"; ep2_exit=sl2; break

            if et2 == "trail":
                trail_pnl2 = _pnl(td2, trade2["entry_price"], ep2_exit)
                base_pnl2 = trade2["pnl_pct"]
                print(f"\nFound trail exit on trade at bar {idx2} ({dir2}):")
                print(f"WITHOUT trail: SL at {sl2:.2f}, PnL = {base_pnl2:+.4f}%")
                print(f"WITH trail:    trail at {ep2_exit:.2f}, PnL = {trail_pnl2:+.4f}%")
                print(f"Improvement: {trail_pnl2 - base_pnl2:+.4f}%")
                print(f"\n>>> PASS — trail exit is better than SL exit")
                return True

        print("\n>>> FAIL — could not find a trade where trail activates before SL")
        return False


# ══════════════════════════════════════════════════════════════════
# STEP 3+4 — WFA Smoke Test
# ══════════════════════════════════════════════════════════════════

def run_mini_wfa(df, htf_bias, n_windows, is_grid, trail_params, label=""):
    """
    Run a mini WFA using fast_backtest directly.
    Returns dict with avg OOS metrics.
    """
    total = len(df)
    window_size = total // n_windows

    oos_sharpes = []
    oos_annuals = []
    oos_wrs = []
    oos_dds = []
    oos_exp_rs = []
    oos_avg_bars_list = []
    oos_trail_pcts = []
    oos_positive = 0

    for w in range(n_windows):
        start = w * window_size
        end = min(start + window_size, total)
        split = start + int((end - start) * 0.70)

        is_df = df.iloc[start:split].reset_index(drop=True)
        oos_df = df.iloc[split:end].reset_index(drop=True)
        is_htf = htf_bias[start:split]
        oos_htf = htf_bias[split:end]

        # Optimize on IS
        best_score = -np.inf
        best_params = {}
        for combo in is_grid:
            m = fast_backtest(
                is_df,
                adx_min=combo["adx_min"],
                ema_fast_p=combo["ema_fast_p"],
                ema_slow_p=combo["ema_slow_p"],
                rr_ratio=1.5, atr_sl_mult=2.0,
                htf_bias=is_htf,
                **trail_params,
            )
            if m.total_trades >= 5 and m.sharpe_ratio > best_score:
                best_score = m.sharpe_ratio
                best_params = combo

        if not best_params:
            best_params = is_grid[0]

        # Test on OOS
        oos_m = fast_backtest(
            oos_df,
            adx_min=best_params["adx_min"],
            ema_fast_p=best_params["ema_fast_p"],
            ema_slow_p=best_params["ema_slow_p"],
            rr_ratio=1.5, atr_sl_mult=2.0,
            htf_bias=oos_htf,
            **trail_params,
        )

        oos_sharpes.append(oos_m.sharpe_ratio)
        oos_annuals.append(oos_m.annual_return_pct)
        oos_wrs.append(oos_m.winrate)
        oos_dds.append(oos_m.max_drawdown_pct)
        if oos_m.sharpe_ratio > 0:
            oos_positive += 1

        # Expectancy in R
        if oos_m.trades:
            pnls = [t["pnl_pct"] for t in oos_m.trades]
            # Approximate R: avg pnl / avg |loss|
            losses = [abs(p) for p in pnls if p < 0]
            avg_loss_pct = np.mean(losses) if losses else 1.0
            exp_r = np.mean(pnls) / avg_loss_pct if avg_loss_pct > 0 else 0
            oos_exp_rs.append(exp_r)
            avg_bars = np.mean([t["bars_held"] for t in oos_m.trades])
            oos_avg_bars_list.append(avg_bars)
            trail_count = sum(1 for t in oos_m.trades if t["exit_type"] == "trail")
            trail_pct = trail_count / len(oos_m.trades) * 100
            oos_trail_pcts.append(trail_pct)
        else:
            oos_exp_rs.append(0)
            oos_avg_bars_list.append(0)
            oos_trail_pcts.append(0)

    return {
        "sharpe": np.mean(oos_sharpes) if oos_sharpes else 0,
        "annual": np.mean(oos_annuals) if oos_annuals else 0,
        "wr": np.mean(oos_wrs) if oos_wrs else 0,
        "dd": np.mean(oos_dds) if oos_dds else 0,
        "exp_r": np.mean(oos_exp_rs) if oos_exp_rs else 0,
        "avg_bars": np.mean(oos_avg_bars_list) if oos_avg_bars_list else 0,
        "trail_pct": np.mean(oos_trail_pcts) if oos_trail_pcts else 0,
        "pos_windows": oos_positive,
        "n_windows": n_windows,
    }


def step3_smoke_test():
    """Run 6 trail combinations + baseline through 3-window WFA on 365d."""
    print("\n" + "=" * 70)
    print("STEP 3 — WFA SMOKE TEST (365d, 3 windows)")
    print("=" * 70)

    df_full = load_candles("BTC/USDT", "15m", days=730)
    # Use last 365 days
    bars_365 = int(365 * 24 * 60 / 15)
    df = df_full.iloc[-bars_365:].reset_index(drop=True)
    htf_bias = compute_htf_bias(df, htf_ema_period=50)

    print(f"Data: {len(df)} bars")

    # IS grid: 8 combos
    is_grid = []
    for adx in [20, 25]:
        for ef in [15, 20]:
            for es in [45, 50]:
                is_grid.append({"adx_min": adx, "ema_fast_p": ef, "ema_slow_p": es})

    # Trail combinations
    trail_combos = [
        (0.6, 0.3), (0.6, 0.5),
        (0.8, 0.3), (0.8, 0.5),
        (1.0, 0.3), (1.0, 0.5),
    ]

    results = []

    # Baseline first
    print("\nRunning baseline (no trail)...")
    base_r = run_mini_wfa(df, htf_bias, 3, is_grid,
                          {"use_trail": False})
    base_r["act_r"] = None
    base_r["trail_r_param"] = None
    results.append(("BASE", base_r))

    # Trail combos
    for act_r, tr_r in trail_combos:
        label = f"act={act_r} tr={tr_r}"
        print(f"Running {label}...")
        r = run_mini_wfa(df, htf_bias, 3, is_grid,
                         {"use_trail": True, "activation_r": act_r, "trail_r": tr_r})
        r["act_r"] = act_r
        r["trail_r_param"] = tr_r
        results.append((label, r))

    return results, base_r


def step4_print_results(results, base_r):
    """Print results table."""
    print("\n" + "=" * 70)
    print("STEP 4 — RESULTS TABLE (sorted by OOS Sharpe)")
    print("=" * 70)

    base_sharpe = base_r["sharpe"]

    # Sort by Sharpe (skip baseline for sorting, add it at end)
    trail_results = [(l, r) for l, r in results if l != "BASE"]
    trail_results.sort(key=lambda x: x[1]["sharpe"], reverse=True)

    print(f"\n  {'act_r':>5} | {'trail_r':>7} | {'OOS Sharpe':>10} | {'OOS Ann%':>8} | {'Pos/3':>5} | {'Exp R':>7} | {'Trail%':>6} | {'DD%':>5} | {'vs Base':>8} | Status")
    print(f"  {'-'*5}-+-{'-'*7}-+-{'-'*10}-+-{'-'*8}-+-{'-'*5}-+-{'-'*7}-+-{'-'*6}-+-{'-'*5}-+-{'-'*8}-+-{'-'*8}")

    passing = []
    for label, r in trail_results:
        act = r["act_r"]
        tr = r["trail_r_param"]
        sharpe = r["sharpe"]
        ann = r["annual"]
        pos = r["pos_windows"]
        exp = r["exp_r"]
        tpct = r["trail_pct"]
        dd = r["dd"]

        if base_sharpe > 0:
            vs_base = (sharpe - base_sharpe) / base_sharpe * 100
        else:
            vs_base = 100.0 if sharpe > 0 else 0.0

        # Status
        if sharpe > base_sharpe * 1.15 and pos >= 2 and exp > base_r["exp_r"]:
            status = "PASS"
            passing.append((label, r))
        elif sharpe > base_sharpe:
            status = "~ MARGINAL"
        else:
            status = "FAIL"

        print(f"  {act:>5.1f} | {tr:>7.1f} | {sharpe:>10.2f} | {ann:>+7.1f}% | {pos:>3}/3 | {exp:>+6.4f} | {tpct:>5.1f}% | {dd:>4.1f}% | {vs_base:>+7.1f}% | {status}")

    # Baseline row
    print(f"  {'─'*95}")
    r = base_r
    print(f"  {'BASE':>5} | {'N/A':>7} | {r['sharpe']:>10.2f} | {r['annual']:>+7.1f}% | {r['pos_windows']:>3}/3 | {r['exp_r']:>+6.4f} | {0:>5.1f}% | {r['dd']:>4.1f}% | {'BASELINE':>8} |")

    return passing


# ══════════════════════════════════════════════════════════════════
# STEP 5 — Full WFA (if any pass)
# ══════════════════════════════════════════════════════════════════

def step5_full_wfa(best_act, best_tr):
    """Run full 5-window WFA on 730d with 27 IS combos."""
    print("\n" + "=" * 70)
    print(f"STEP 5 — FULL WFA (730d, 5 windows) — act={best_act} trail={best_tr}")
    print("=" * 70)

    df = load_candles("BTC/USDT", "15m", days=730)
    htf_bias = compute_htf_bias(df, htf_ema_period=50)

    # 27 IS combos
    is_grid = []
    for adx in [15, 20, 25]:
        for ef in [15, 20, 25]:
            for es in [40, 50, 60]:
                is_grid.append({"adx_min": adx, "ema_fast_p": ef, "ema_slow_p": es})

    print(f"Data: {len(df)} bars, IS grid: {len(is_grid)} combos")

    # Baseline
    print("\nRunning baseline (no trail)...")
    base_full = run_mini_wfa(df, htf_bias, 5, is_grid, {"use_trail": False})

    # Trail
    print(f"Running trail (act={best_act}, tr={best_tr})...")
    trail_full = run_mini_wfa(df, htf_bias, 5, is_grid,
                              {"use_trail": True, "activation_r": best_act, "trail_r": best_tr})

    print(f"\n  {'Metric':<22} | {'Baseline':>12} | {'Trail':>12}")
    print(f"  {'-'*22}-+-{'-'*12}-+-{'-'*12}")
    print(f"  {'OOS Sharpe avg':<22} | {base_full['sharpe']:>12.2f} | {trail_full['sharpe']:>12.2f}")
    print(f"  {'OOS Annual % avg':<22} | {base_full['annual']:>+11.1f}% | {trail_full['annual']:>+11.1f}%")
    print(f"  {'OOS Positive wins':<22} | {base_full['pos_windows']:>10}/5  | {trail_full['pos_windows']:>10}/5 ")
    print(f"  {'Expectancy R':<22} | {base_full['exp_r']:>+11.4f}R | {trail_full['exp_r']:>+11.4f}R")
    print(f"  {'Avg bars held':<22} | {base_full['avg_bars']:>12.1f} | {trail_full['avg_bars']:>12.1f}")
    print(f"  {'OOS Max DD avg':<22} | {base_full['dd']:>11.1f}% | {trail_full['dd']:>11.1f}%")
    print(f"  {'% trail exits':<22} | {0:>11.1f}% | {trail_full['trail_pct']:>11.1f}%")

    # Adoption criteria
    adopt = (
        trail_full["sharpe"] > base_full["sharpe"] * 1.15
        and trail_full["pos_windows"] >= 5
        and trail_full["exp_r"] > base_full["exp_r"]
    )

    print(f"\n  Adoption criteria:")
    print(f"    OOS Sharpe > baseline*1.15 ({base_full['sharpe']*1.15:.2f}): {trail_full['sharpe']:.2f} → {'PASS' if trail_full['sharpe'] > base_full['sharpe']*1.15 else 'FAIL'}")
    print(f"    5/5 OOS positive: {trail_full['pos_windows']}/5 → {'PASS' if trail_full['pos_windows']>=5 else 'FAIL'}")
    print(f"    Exp R > baseline ({base_full['exp_r']:+.4f}): {trail_full['exp_r']:+.4f} → {'PASS' if trail_full['exp_r']>base_full['exp_r'] else 'FAIL'}")

    return adopt, base_full, trail_full


# ══════════════════════════════════════════════════════════════════
# STEP 6 — Monte Carlo
# ══════════════════════════════════════════════════════════════════

def step6_monte_carlo(best_act, best_tr):
    """Run MC comparison."""
    print("\n" + "=" * 70)
    print("STEP 6 — MONTE CARLO (5000 simulations)")
    print("=" * 70)

    df = load_candles("BTC/USDT", "15m", days=730)
    htf_bias = compute_htf_bias(df, htf_ema_period=50)

    # Baseline trades
    base_m = fast_backtest(
        df, adx_min=20, ema_fast_p=20, ema_slow_p=50,
        rr_ratio=1.5, atr_sl_mult=2.0,
        htf_bias=htf_bias, use_trail=False,
    )
    base_pnls = np.array([t["pnl_pct"] for t in base_m.trades if t["exit_type"] != "end_of_data"])

    # Trail trades
    trail_m = fast_backtest(
        df, adx_min=20, ema_fast_p=20, ema_slow_p=50,
        rr_ratio=1.5, atr_sl_mult=2.0,
        htf_bias=htf_bias, use_trail=True,
        activation_r=best_act, trail_r=best_tr,
    )
    trail_pnls = np.array([t["pnl_pct"] for t in trail_m.trades if t["exit_type"] != "end_of_data"])

    n_sims = 5000
    n_trades = min(len(base_pnls), len(trail_pnls))

    def mc_sim(pnls, n_sims, n_trades):
        rng = np.random.default_rng(42)
        finals = np.zeros(n_sims)
        max_dds = np.zeros(n_sims)
        for s in range(n_sims):
            sample = rng.choice(pnls, size=n_trades, replace=True)
            mult = 1.0 + sample / 100.0
            equity = np.cumprod(mult)
            finals[s] = (equity[-1] - 1) * 100
            rm = np.maximum.accumulate(equity)
            dd = (equity - rm) / rm
            max_dds[s] = abs(np.min(dd)) * 100
        return finals, max_dds

    base_finals, base_dds = mc_sim(base_pnls, n_sims, n_trades)
    trail_finals, trail_dds = mc_sim(trail_pnls, n_sims, n_trades)

    base_ror = np.mean(base_finals < -50) * 100
    trail_ror = np.mean(trail_finals < -50) * 100

    print(f"\n  {'Metric':<18} | {'Baseline':>12} | {'With Trail':>12}")
    print(f"  {'-'*18}-+-{'-'*12}-+-{'-'*12}")
    print(f"  {'Risk of Ruin':<18} | {base_ror:>11.1f}% | {trail_ror:>11.1f}%")
    print(f"  {'MC P5':<18} | {np.percentile(base_finals, 5):>+11.1f}% | {np.percentile(trail_finals, 5):>+11.1f}%")
    print(f"  {'MC P50':<18} | {np.percentile(base_finals, 50):>+11.1f}% | {np.percentile(trail_finals, 50):>+11.1f}%")
    print(f"  {'MC P95':<18} | {np.percentile(base_finals, 95):>+11.1f}% | {np.percentile(trail_finals, 95):>+11.1f}%")
    print(f"  {'Max DD P95':<18} | {np.percentile(base_dds, 95):>11.1f}% | {np.percentile(trail_dds, 95):>11.1f}%")

    mc_worse = trail_ror > base_ror * 1.5 or np.percentile(trail_finals, 5) < np.percentile(base_finals, 5) * 0.8
    if mc_worse:
        print("\n  WARNING: MC results are WORSE with trail — suspicious")
    else:
        print("\n  MC results are consistent with trail improvement")

    return {
        "base_ror": base_ror, "trail_ror": trail_ror,
        "base_p50": np.percentile(base_finals, 50),
        "trail_p50": np.percentile(trail_finals, 50),
        "base_dd95": np.percentile(base_dds, 95),
        "trail_dd95": np.percentile(trail_dds, 95),
    }


# ══════════════════════════════════════════════════════════════════
# STEP 7 — Final verdict
# ══════════════════════════════════════════════════════════════════

def step7_verdict(smoke_results, smoke_base, passing, adopted, best_act=None, best_tr=None,
                  full_base=None, full_trail=None, mc=None):
    print("\n")
    print("  " + "=" * 62)
    print("       TRAILING STOP WFA RESULTS — BTC/USDT 15m")
    print("  " + "=" * 62)
    print("  SMOKE TEST (365d, 3 windows)")
    print(f"   Combinations tested:  6")
    print(f"   Combinations passing: {len(passing)}")
    if passing:
        best_l, best_r = passing[0]
        print(f"   Best combination:     act={best_r['act_r']} trail={best_r['trail_r_param']}")
        print(f"   Best OOS Sharpe:      {best_r['sharpe']:.2f} vs baseline {smoke_base['sharpe']:.2f}")
    else:
        print(f"   Best combination:     N/A")
        print(f"   Best OOS Sharpe:      N/A vs baseline {smoke_base['sharpe']:.2f}")
    print("  " + "-" * 62)

    if full_base and full_trail:
        print("  FULL WFA (730d, 5 windows)")
        print(f"   OOS Sharpe:           {full_trail['sharpe']:.2f} vs baseline {full_base['sharpe']:.2f}")
        print(f"   OOS windows positive: {full_trail['pos_windows']}/5")
        print(f"   Expectancy R:         {full_trail['exp_r']:+.4f} vs {full_base['exp_r']:+.4f}")
        print(f"   % trail exits:        {full_trail['trail_pct']:.1f}%")
        print("  " + "-" * 62)

    if mc:
        print("  MONTE CARLO")
        print(f"   Risk of Ruin:         {mc['trail_ror']:.1f}% vs {mc['base_ror']:.1f}%")
        print(f"   MC P50:               {mc['trail_p50']:+.1f}% vs {mc['base_p50']:+.1f}%")
        print("  " + "-" * 62)

    print("  FINAL DECISION")
    if adopted:
        print(f"   Trailing stop ADOPTED")
        print(f"   activation_r = {best_act}")
        print(f"   trail_r      = {best_tr}")
        print(f"   Update VALIDATED_PARAMS in run_simple_paper.py")
        print(f"   Update Pine Script with trail logic")
    else:
        print(f"   Trailing stop REJECTED")
        print(f"   \"Baseline fixed SL/TP is optimal. 22.5% wasted MFE")
        print(f"    exists but trailing stop does not improve OOS")
        print(f"    performance. Bot ready for paper trading as-is.\"")
    print("  " + "=" * 62)


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════

def main():
    # Step 2
    ok = step2_verify_single_trade()
    if not ok:
        print("\nSTOPPED: Step 2 verification failed.")
        return

    # Step 3
    smoke_results, smoke_base = step3_smoke_test()

    # Step 4
    passing = step4_print_results(smoke_results, smoke_base)

    adopted = False
    best_act = None
    best_tr = None
    full_base = None
    full_trail = None
    mc = None

    if passing:
        # Best passing combo
        best_label, best_r = passing[0]
        best_act = best_r["act_r"]
        best_tr = best_r["trail_r_param"]

        # Step 5
        adopted, full_base, full_trail = step5_full_wfa(best_act, best_tr)

        # Step 6 (only if adopted)
        if adopted:
            mc = step6_monte_carlo(best_act, best_tr)
    else:
        print("\n  No combinations passed smoke test. REJECTED.")

    # Step 7
    step7_verdict(smoke_results, smoke_base, passing, adopted,
                  best_act, best_tr, full_base, full_trail, mc)


if __name__ == "__main__":
    main()
