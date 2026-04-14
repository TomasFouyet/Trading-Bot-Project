"""
Parameter Stability Analysis — detect overfitting via parameter sensitivity.

Tests a grid of parameters near the "optimal" values.  If only a narrow
spike of combinations works, the strategy is likely overfit.  If a wide
plateau of parameters yields similar results, the strategy is robust.

Generates heatmaps (2D slices) of Sharpe / annual return across param pairs.

Usage:
    from validation.param_stability import ParamStabilityAnalysis
    from validation.strategy_adapter import StrategyAdapter
    from app.strategy.trend_following_v2 import TrendFollowingV2

    adapter = StrategyAdapter(TrendFollowingV2)
    psa = ParamStabilityAnalysis(adapter, param_grid={
        "adx_min": [15, 18, 20, 22, 25, 28, 30],
        "ema_fast": [10, 15, 20, 25, 30],
        "ema_slow": [40, 45, 50, 55, 60],
    })
    report = psa.run(df)
    report.print_summary()
    report.plot_heatmaps()
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from validation.strategy_adapter import BacktestMetrics, StrategyAdapter


@dataclass
class ParamStabilityReport:
    """Results of parameter stability analysis."""
    param_grid: dict[str, list] = field(default_factory=dict)
    results: pd.DataFrame = field(default_factory=pd.DataFrame)
    total_combos: int = 0
    profitable_combos: int = 0
    robust_combos: int = 0  # meets annual >= 15%, sharpe >= 1.0
    best_params: dict[str, Any] = field(default_factory=dict)
    best_sharpe: float = 0.0
    best_annual: float = 0.0

    @property
    def robustness_ratio(self) -> float:
        """Fraction of combos that are robust (meet minimum criteria)."""
        if self.total_combos == 0:
            return 0.0
        return self.robust_combos / self.total_combos

    @property
    def profitability_ratio(self) -> float:
        """Fraction of combos that are profitable."""
        if self.total_combos == 0:
            return 0.0
        return self.profitable_combos / self.total_combos

    @property
    def is_stable(self) -> bool:
        """
        Strategy is stable if >= 20% of parameter combos are robust.
        This means the strategy works across a wide range, not just one spike.
        """
        return self.robustness_ratio >= 0.20

    def print_summary(self) -> None:
        print("\n" + "=" * 70)
        print("PARAMETER STABILITY REPORT")
        print("=" * 70)
        print(f"  Total combinations tested: {self.total_combos}")
        print(f"  Profitable (PnL > 0):      {self.profitable_combos} ({self.profitability_ratio:.0%})")
        print(f"  Robust (ann>=15%, S>=1):    {self.robust_combos} ({self.robustness_ratio:.0%})")
        print()
        print(f"  Best params: {self.best_params}")
        print(f"  Best Sharpe: {self.best_sharpe:.2f}")
        print(f"  Best Annual: {self.best_annual:.1f}%")
        print()

        if not self.results.empty:
            print(f"  Sharpe distribution across grid:")
            sharpes = self.results["sharpe_ratio"]
            print(f"    Mean:   {sharpes.mean():.2f}")
            print(f"    Std:    {sharpes.std():.2f}")
            print(f"    Min:    {sharpes.min():.2f}")
            print(f"    Max:    {sharpes.max():.2f}")
            print(f"    >0:     {(sharpes > 0).sum()}")
            print(f"    >1:     {(sharpes > 1).sum()}")

        verdict = "STABLE" if self.is_stable else "UNSTABLE (possible overfit)"
        print(f"\n  VERDICT: {verdict}")
        print("=" * 70)

    def plot_heatmaps(
        self,
        metric: str = "sharpe_ratio",
        save_path: str | None = None,
    ) -> None:
        """
        Plot 2D heatmaps for each pair of parameters.

        For grids with >2 params, creates one heatmap per pair,
        averaging over all other dimensions.
        """
        import matplotlib.pyplot as plt

        params = list(self.param_grid.keys())
        if len(params) < 2:
            print("[PSA] Need at least 2 parameters for heatmaps.")
            return

        pairs = list(itertools.combinations(params, 2))
        n_pairs = len(pairs)
        cols = min(3, n_pairs)
        rows = (n_pairs + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
        fig.suptitle(f"Parameter Stability — {metric}", fontsize=14, fontweight="bold")

        if n_pairs == 1:
            axes = np.array([axes])
        axes = np.atleast_2d(axes)

        for idx, (p1, p2) in enumerate(pairs):
            r, c = divmod(idx, cols)
            ax = axes[r][c] if rows > 1 else axes[0][c]

            # Pivot: average metric across all other params
            pivot = self.results.groupby([p1, p2])[metric].mean().reset_index()
            pivot_table = pivot.pivot(index=p1, columns=p2, values=metric)

            im = ax.imshow(
                pivot_table.values,
                aspect="auto",
                cmap="RdYlGn",
                interpolation="nearest",
            )
            ax.set_xticks(range(len(pivot_table.columns)))
            ax.set_xticklabels([f"{v:.4g}" for v in pivot_table.columns], rotation=45, fontsize=8)
            ax.set_yticks(range(len(pivot_table.index)))
            ax.set_yticklabels([f"{v:.4g}" for v in pivot_table.index], fontsize=8)
            ax.set_xlabel(p2)
            ax.set_ylabel(p1)
            ax.set_title(f"{p1} vs {p2}")
            fig.colorbar(im, ax=ax, shrink=0.8)

        # Hide unused axes
        for idx in range(n_pairs, rows * cols):
            r, c = divmod(idx, cols)
            ax = axes[r][c] if rows > 1 else axes[0][c]
            ax.axis("off")

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"[PSA] Heatmap saved to {save_path}")
        plt.show()


class ParamStabilityAnalysis:
    """
    Grid-search all parameter combos and measure performance landscape.
    """

    def __init__(
        self,
        adapter: StrategyAdapter,
        param_grid: dict[str, list],
        min_annual: float = 15.0,
        min_sharpe: float = 1.0,
    ) -> None:
        self.adapter = adapter
        self.param_grid = param_grid
        self.min_annual = min_annual
        self.min_sharpe = min_sharpe

        keys = list(param_grid.keys())
        vals = list(param_grid.values())
        self._combos = [dict(zip(keys, combo)) for combo in itertools.product(*vals)]
        print(f"[PSA] Parameter grid: {len(self._combos)} combinations")

    def run(self, df: pd.DataFrame, verbose: bool = True) -> ParamStabilityReport:
        """Run all parameter combos and collect metrics."""
        rows = []
        best_sharpe = -np.inf
        best_params = {}
        best_annual = 0.0

        total = len(self._combos)
        for idx, params in enumerate(self._combos):
            if verbose and (idx + 1) % max(1, total // 20) == 0:
                print(f"[PSA] Testing {idx + 1}/{total} ...", end="\r")

            metrics = self.adapter.run(df, params_override=params)

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

        if verbose:
            print(f"[PSA] Testing {total}/{total} done.     ")

        results_df = pd.DataFrame(rows)

        profitable = int((results_df["total_pnl_pct"] > 0).sum())
        robust = int(
            ((results_df["annual_return_pct"] >= self.min_annual) &
             (results_df["sharpe_ratio"] >= self.min_sharpe)).sum()
        )

        report = ParamStabilityReport(
            param_grid=self.param_grid,
            results=results_df,
            total_combos=total,
            profitable_combos=profitable,
            robust_combos=robust,
            best_params=best_params,
            best_sharpe=float(best_sharpe),
            best_annual=best_annual,
        )

        report.print_summary()
        return report
