"""
Fast Parameter Stability Analysis using the vectorized backtester.

Pre-computes indicators per (ema_fast, ema_slow) pair, then sweeps
the remaining params (adx_min, rr_ratio, atr_sl_mult) without
recomputing indicators. ~50-100x faster than the generic PSA.
"""
from __future__ import annotations

import itertools
from typing import Any

import numpy as np
import pandas as pd

from validation.fast_backtest import compute_indicators, compute_htf_bias, fast_backtest
from validation.param_stability import ParamStabilityReport


def fast_param_stability(
    df: pd.DataFrame,
    param_grid: dict[str, list],
    use_htf: bool = False,
    min_annual: float = 15.0,
    min_sharpe: float = 1.0,
    verbose: bool = True,
) -> ParamStabilityReport:
    """
    Run parameter stability analysis using the fast backtester.

    param_grid keys can include: adx_min, ema_fast, ema_slow,
    rr_ratio, atr_sl_mult, and any other fast_backtest param.
    """
    # Separate indicator params (require recomputation) from trade params
    ema_fast_vals = param_grid.get("ema_fast", [20])
    ema_slow_vals = param_grid.get("ema_slow", [50])
    other_keys = [k for k in param_grid if k not in ("ema_fast", "ema_slow")]
    other_vals = [param_grid[k] for k in other_keys]
    other_combos = list(itertools.product(*other_vals)) if other_vals else [()]

    # Pre-compute HTF bias once
    htf_bias = compute_htf_bias(df) if use_htf else None

    # Pre-compute indicators for each (ema_fast, ema_slow) pair
    indicator_cache: dict[tuple[int, int], pd.DataFrame] = {}
    for ef, es in itertools.product(ema_fast_vals, ema_slow_vals):
        if ef >= es:
            continue  # skip invalid: fast EMA must be < slow EMA
        indicator_cache[(ef, es)] = compute_indicators(df, ef, es)

    total = len(indicator_cache) * len(other_combos)
    if verbose:
        print(f"[Fast PSA] {len(indicator_cache)} indicator sets × "
              f"{len(other_combos)} param combos = {total} total")

    rows = []
    best_sharpe = -np.inf
    best_params: dict[str, Any] = {}
    best_annual = 0.0
    count = 0

    for (ef, es), df_ind in indicator_cache.items():
        for combo in other_combos:
            params = dict(zip(other_keys, combo))
            params["ema_fast"] = ef
            params["ema_slow"] = es

            metrics = fast_backtest(
                df_ind,
                adx_min=params.get("adx_min", 20.0),
                ema_fast_p=ef,
                ema_slow_p=es,
                rr_ratio=params.get("rr_ratio", 1.5),
                atr_sl_mult=params.get("atr_sl_mult", 2.0),
                precomputed=True,
                htf_bias=htf_bias,
            )

            row = {
                **params,
                "total_trades": metrics.total_trades,
                "winrate": metrics.winrate,
                "total_pnl_pct": metrics.total_pnl_pct,
                "annual_return_pct": metrics.annual_return_pct,
                "sharpe_ratio": metrics.sharpe_ratio,
                "max_drawdown_pct": metrics.max_drawdown_pct,
            }
            rows.append(row)

            if metrics.sharpe_ratio > best_sharpe:
                best_sharpe = metrics.sharpe_ratio
                best_params = params
                best_annual = metrics.annual_return_pct

            count += 1
            if verbose and count % max(1, total // 20) == 0:
                print(f"[Fast PSA] {count}/{total} ...", end="\r", flush=True)

    if verbose:
        print(f"[Fast PSA] {total}/{total} done.     ")

    results_df = pd.DataFrame(rows)
    profitable = int((results_df["total_pnl_pct"] > 0).sum()) if not results_df.empty else 0
    robust = int(
        ((results_df["annual_return_pct"] >= min_annual) &
         (results_df["sharpe_ratio"] >= min_sharpe)).sum()
    ) if not results_df.empty else 0

    report = ParamStabilityReport(
        param_grid=param_grid,
        results=results_df,
        total_combos=len(rows),
        profitable_combos=profitable,
        robust_combos=robust,
        best_params=best_params,
        best_sharpe=float(best_sharpe),
        best_annual=best_annual,
    )

    report.print_summary()
    return report
