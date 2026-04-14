#!/usr/bin/env python3
"""
Run statistical validation suite on ANY registered strategy.

For trend_v2_simple, uses the fast vectorized backtester + HTF filter.
For other strategies, falls back to the generic bar-by-bar StrategyAdapter.

Usage:
    # Full suite on the new simplified strategy (fast, ~2 min):
    python scripts/run_validation.py --strategy trend_v2_simple --save-plots

    # Full suite on original V2 (slow, ~30 min):
    python scripts/run_validation.py --strategy trend_v2 --save-plots

    # List available strategies:
    python scripts/run_validation.py --list

    # Individual tests:
    python scripts/run_validation.py --strategy trend_v2_simple --test wfa
    python scripts/run_validation.py --strategy trend_v2_simple --test mc
    python scripts/run_validation.py --strategy trend_v2_simple --test psa
"""
from __future__ import annotations

import argparse
import itertools
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")


def parse_args():
    p = argparse.ArgumentParser(description="Statistical Validation Suite")
    p.add_argument("--strategy", "-s", default="trend_v2_simple",
                   help="Strategy to validate (use --list to see options)")
    p.add_argument("--list", action="store_true",
                   help="List all available strategies and exit")
    p.add_argument("--test", choices=["wfa", "mc", "psa", "all"], default="all",
                   help="Which test to run (default: all)")
    p.add_argument("--symbol", default="BTC/USDT")
    p.add_argument("--timeframe", default="15m")
    p.add_argument("--days", type=int, default=730, help="Days of historical data")
    p.add_argument("--wfa-windows", type=int, default=5, help="WFA rolling windows")
    p.add_argument("--mc-sims", type=int, default=5000, help="Monte Carlo simulations")
    p.add_argument("--htf", action="store_true", default=True,
                   help="Enable HTF 4H trend filter (default: on)")
    p.add_argument("--no-htf", action="store_true", help="Disable HTF filter")
    p.add_argument("--save-plots", action="store_true", help="Save plots to validation/output/")
    p.add_argument("--no-cache", action="store_true", help="Force re-download data")
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════
# FAST PATH — for trend_v2_simple using vectorized backtester
# ═══════════════════════════════════════════════════════════════════════

def run_fast_baseline(df, rr, sl_atr, use_htf):
    """Run a single fast backtest with default entry params."""
    import numpy as np
    from validation.fast_backtest import compute_indicators, compute_htf_bias, fast_backtest

    htf_bias = compute_htf_bias(df) if use_htf else None
    df_ind = compute_indicators(df, ema_fast_p=20, ema_slow_p=50)
    return fast_backtest(df_ind, rr_ratio=rr, atr_sl_mult=sl_atr,
                         precomputed=True, htf_bias=htf_bias)


def run_fast_wfa(df, rr, sl_atr, use_htf, n_windows=5):
    """Run fast WFA and return a WalkForwardReport-compatible structure."""
    import numpy as np
    from validation.fast_backtest import compute_indicators, compute_htf_bias, fast_backtest
    from validation.walk_forward import WalkForwardReport, WindowResult
    from validation.strategy_adapter import BacktestMetrics

    htf_full = compute_htf_bias(df) if use_htf else None

    wfa_grid = [
        {"adx_min": a, "ema_fast": f, "ema_slow": s}
        for a, f, s in itertools.product([15, 20, 25], [15, 20, 25], [40, 50, 60])
    ]

    total = len(df)
    window_size = total // n_windows
    report = WalkForwardReport()

    for w_idx in range(n_windows):
        start = w_idx * window_size
        end = min(start + window_size, total)
        split = start + int((end - start) * 0.70)

        is_df = df.iloc[start:split].reset_index(drop=True)
        oos_df = df.iloc[split:end].reset_index(drop=True)
        is_htf = htf_full[start:split] if htf_full is not None else None
        oos_htf = htf_full[split:end] if htf_full is not None else None

        if len(is_df) < 100 or len(oos_df) < 50:
            continue

        # Optimize on IS
        best_score = -np.inf
        best_params = wfa_grid[0]
        best_is_m = BacktestMetrics()

        for params in wfa_grid:
            is_ind = compute_indicators(is_df, params["ema_fast"], params["ema_slow"])
            m = fast_backtest(
                is_ind, adx_min=params["adx_min"],
                ema_fast_p=params["ema_fast"], ema_slow_p=params["ema_slow"],
                rr_ratio=rr, atr_sl_mult=sl_atr,
                precomputed=True, htf_bias=is_htf,
            )
            score = m.sharpe_ratio if m.total_trades >= 5 else -np.inf
            if score > best_score:
                best_score = score
                best_params = params
                best_is_m = m

        # OOS with best params
        oos_ind = compute_indicators(oos_df, best_params["ema_fast"], best_params["ema_slow"])
        oos_m = fast_backtest(
            oos_ind, adx_min=best_params["adx_min"],
            ema_fast_p=best_params["ema_fast"], ema_slow_p=best_params["ema_slow"],
            rr_ratio=rr, atr_sl_mult=sl_atr,
            precomputed=True, htf_bias=oos_htf,
        )

        result = WindowResult(
            window_idx=w_idx,
            is_start=is_df["ts"].iloc[0], is_end=is_df["ts"].iloc[-1],
            oos_start=oos_df["ts"].iloc[0], oos_end=oos_df["ts"].iloc[-1],
            best_params=best_params,
            is_metrics=best_is_m,
            oos_metrics=oos_m,
        )
        report.windows.append(result)

        print(f"  Window {w_idx+1}: IS ann={best_is_m.annual_return_pct:+.1f}% "
              f"→ OOS ann={oos_m.annual_return_pct:+.1f}%  "
              f"sharpe={oos_m.sharpe_ratio:.2f}  trades={oos_m.total_trades}",
              flush=True)

    return report


# ═══════════════════════════════════════════════════════════════════════
# SLOW PATH — generic, for any BaseStrategy
# ═══════════════════════════════════════════════════════════════════════

def run_slow_path(args, strategy_cls, df, output_dir):
    from validation.strategy_adapter import StrategyAdapter
    from validation.walk_forward import WalkForwardAnalysis
    from validation.monte_carlo import MonteCarloSimulation
    from validation.param_stability import ParamStabilityAnalysis

    adapter = StrategyAdapter(
        strategy_cls=strategy_cls,
        symbol=args.symbol.replace("/", ""),
        default_params={},
    )

    print("\n--- Baseline Backtest (default params) ---")
    baseline = adapter.run(df)
    print(f"  Trades:  {baseline.total_trades}")
    print(f"  Winrate: {baseline.winrate:.1f}%")
    print(f"  Annual:  {baseline.annual_return_pct:.1f}%")
    print(f"  Sharpe:  {baseline.sharpe_ratio:.2f}")
    print(f"  Max DD:  {baseline.max_drawdown_pct:.1f}%")

    if args.test in ("wfa", "all"):
        param_grid = {
            "adx_min": [15, 20, 25, 30],
            "ema_fast": [15, 20, 25],
            "ema_slow": [40, 50, 60],
        }
        wfa = WalkForwardAnalysis(adapter=adapter, param_grid=param_grid)
        wfa_report = wfa.run(df, n_windows=args.wfa_windows)
        if args.save_plots:
            wfa_report.plot(save_path=str(output_dir / "wfa_report.png"))

    if args.test in ("mc", "all") and baseline.total_trades >= 5:
        mc = MonteCarloSimulation(trades=baseline.trades, n_simulations=args.mc_sims)
        mc_report = mc.run()
        mc_report.print_summary()
        if args.save_plots:
            mc_report.plot(save_path=str(output_dir / "monte_carlo.png"))

    if args.test in ("psa", "all"):
        stability_grid = {
            "adx_min": [15, 18, 20, 22, 25, 28, 30],
            "ema_fast": [12, 15, 18, 20, 22, 25, 28],
            "ema_slow": [40, 45, 50, 55, 60],
        }
        psa = ParamStabilityAnalysis(adapter=adapter, param_grid=stability_grid)
        psa_report = psa.run(df)
        if args.save_plots:
            psa_report.plot_heatmaps(save_path=str(output_dir / "param_stability.png"))


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    use_htf = args.htf and not args.no_htf

    from validation.registry import get_strategy_class, list_strategies

    if args.list:
        print("\nAvailable strategies:")
        for name in list_strategies():
            try:
                cls = get_strategy_class(name)
                doc = (cls.__doc__ or "").strip().split("\n")[0]
            except Exception:
                doc = "(import error)"
            print(f"  {name:20s} → {doc}")
        return

    strategy_cls = get_strategy_class(args.strategy)
    from validation.data_loader import load_candles

    print(f"\n{'='*70}")
    print(f"STATISTICAL VALIDATION SUITE")
    print(f"{'='*70}")
    print(f"  Strategy:  {args.strategy} → {strategy_cls.__name__}")
    print(f"  Symbol:    {args.symbol}")
    print(f"  Timeframe: {args.timeframe}")
    print(f"  Days:      {args.days}")
    print(f"  Test:      {args.test}")
    print(f"  HTF:       {'ON (4H EMA50)' if use_htf else 'OFF'}")
    print()

    df = load_candles(
        symbol=args.symbol, timeframe=args.timeframe,
        days=args.days, cache=not args.no_cache,
    )
    print(f"  Data: {len(df)} bars from {df['ts'].iloc[0]} to {df['ts'].iloc[-1]}")

    output_dir = ROOT / "validation" / "output"
    if args.save_plots:
        output_dir.mkdir(parents=True, exist_ok=True)

    # ── Route: fast path for trend_v2_simple, slow for others ────
    is_fast = args.strategy == "trend_v2_simple"

    if not is_fast:
        print(f"\n[Using slow bar-by-bar adapter for {args.strategy}]")
        run_slow_path(args, strategy_cls, df, output_dir)

        print(f"\n{'='*70}")
        print("VALIDATION COMPLETE")
        print(f"{'='*70}")
        return

    # ── FAST PATH ────────────────────────────────────────────────
    rr = 1.5
    sl_atr = 2.0
    print(f"\n[Fast backtester] rr={rr}  sl_atr={sl_atr}  htf={'ON' if use_htf else 'OFF'}")

    # Baseline
    t0 = time.time()
    print("\n--- Baseline Backtest ---")
    baseline = run_fast_baseline(df, rr, sl_atr, use_htf)
    print(f"  Trades:  {baseline.total_trades}")
    print(f"  Winrate: {baseline.winrate:.1f}%")
    print(f"  Annual:  {baseline.annual_return_pct:.1f}%")
    print(f"  Sharpe:  {baseline.sharpe_ratio:.2f}")
    print(f"  Max DD:  {baseline.max_drawdown_pct:.1f}%")
    print(f"  ({time.time()-t0:.1f}s)")

    # WFA
    if args.test in ("wfa", "all"):
        t0 = time.time()
        print(f"\n--- Walk-Forward Analysis ({args.wfa_windows} windows) ---")
        wfa_report = run_fast_wfa(df, rr, sl_atr, use_htf, n_windows=args.wfa_windows)
        wfa_report.print_summary()
        print(f"  ({time.time()-t0:.1f}s)")

        if args.save_plots:
            wfa_report.plot(save_path=str(output_dir / "wfa_report.png"))

    # Monte Carlo
    if args.test in ("mc", "all"):
        t0 = time.time()
        print(f"\n--- Monte Carlo ({args.mc_sims} simulations) ---")
        if baseline.total_trades >= 5:
            from validation.monte_carlo import MonteCarloSimulation
            mc = MonteCarloSimulation(trades=baseline.trades, n_simulations=args.mc_sims)
            mc_report = mc.run()
            mc_report.print_summary()
            print(f"  ({time.time()-t0:.1f}s)")

            if args.save_plots:
                mc_report.plot(save_path=str(output_dir / "monte_carlo.png"))
        else:
            print("  Skipped — not enough trades.")

    # Parameter Stability (FAST)
    if args.test in ("psa", "all"):
        t0 = time.time()
        print(f"\n--- Parameter Stability (fast) ---")
        from validation.fast_param_stability import fast_param_stability

        stability_grid = {
            "adx_min": [15, 18, 20, 22, 25, 28, 30],
            "ema_fast": [12, 15, 18, 20, 22, 25, 28],
            "ema_slow": [40, 45, 50, 55, 60],
            "rr_ratio": [1.3, 1.5, 1.7],
            "atr_sl_mult": [1.5, 2.0, 2.5],
        }

        psa_report = fast_param_stability(
            df, param_grid=stability_grid,
            use_htf=use_htf,
        )
        print(f"  ({time.time()-t0:.1f}s)")

        if args.save_plots:
            psa_report.plot_heatmaps(save_path=str(output_dir / "param_stability.png"))

    # Done
    print(f"\n{'='*70}")
    print("VALIDATION COMPLETE")
    if args.save_plots:
        print(f"Plots saved to: {output_dir}/")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
