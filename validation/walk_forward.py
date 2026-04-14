"""
Walk-Forward Analysis (WFA) for trading strategy validation.

How it works:
  1. Split historical data into N rolling windows
  2. Each window: 70% in-sample (IS) + 30% out-of-sample (OOS)
  3. Optimize parameters on IS via grid search
  4. Test best params on OOS (unseen data)
  5. Slide window forward, repeat
  6. Report: IS vs OOS degradation, combined OOS equity curve

A strategy is robust if OOS performance doesn't collapse vs IS.
Target: 15%+ annual return, Sharpe > 1.0 on combined OOS.

Usage:
    from validation.walk_forward import WalkForwardAnalysis
    from validation.strategy_adapter import StrategyAdapter
    from app.strategy.trend_following_v2 import TrendFollowingV2

    adapter = StrategyAdapter(TrendFollowingV2)
    wfa = WalkForwardAnalysis(adapter, param_grid={
        "adx_min": [15, 20, 25, 30],
        "ema_fast": [10, 20, 30],
        "ema_slow": [40, 50, 60],
    })
    report = wfa.run(df, n_windows=5)
    report.print_summary()
    report.plot()
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from validation.strategy_adapter import BacktestMetrics, StrategyAdapter


@dataclass
class WindowResult:
    """Results for a single walk-forward window."""
    window_idx: int
    is_start: Any
    is_end: Any
    oos_start: Any
    oos_end: Any
    best_params: dict[str, Any]
    is_metrics: BacktestMetrics
    oos_metrics: BacktestMetrics

    @property
    def pnl_degradation(self) -> float:
        """% degradation from IS to OOS annual return."""
        if self.is_metrics.annual_return_pct == 0:
            return 0.0
        return (
            (self.is_metrics.annual_return_pct - self.oos_metrics.annual_return_pct)
            / abs(self.is_metrics.annual_return_pct)
            * 100
        )

    @property
    def sharpe_degradation(self) -> float:
        if self.is_metrics.sharpe_ratio == 0:
            return 0.0
        return (
            (self.is_metrics.sharpe_ratio - self.oos_metrics.sharpe_ratio)
            / abs(self.is_metrics.sharpe_ratio)
            * 100
        )


@dataclass
class WalkForwardReport:
    """Aggregated WFA results across all windows."""
    windows: list[WindowResult] = field(default_factory=list)
    combined_oos_metrics: BacktestMetrics | None = None

    @property
    def avg_oos_annual_return(self) -> float:
        if not self.windows:
            return 0.0
        return np.mean([w.oos_metrics.annual_return_pct for w in self.windows])

    @property
    def avg_oos_sharpe(self) -> float:
        if not self.windows:
            return 0.0
        return np.mean([w.oos_metrics.sharpe_ratio for w in self.windows])

    @property
    def avg_pnl_degradation(self) -> float:
        if not self.windows:
            return 0.0
        return np.mean([w.pnl_degradation for w in self.windows])

    @property
    def avg_sharpe_degradation(self) -> float:
        if not self.windows:
            return 0.0
        return np.mean([w.sharpe_degradation for w in self.windows])

    @property
    def is_robust(self) -> bool:
        """Passes if avg OOS: annual >= 15%, sharpe >= 1.0, degradation < 50%."""
        return (
            self.avg_oos_annual_return >= 15.0
            and self.avg_oos_sharpe >= 1.0
            and self.avg_pnl_degradation < 50.0
        )

    def print_summary(self) -> None:
        print("\n" + "=" * 70)
        print("WALK-FORWARD ANALYSIS REPORT")
        print("=" * 70)

        for w in self.windows:
            print(f"\n--- Window {w.window_idx + 1} ---")
            print(f"  IS:  {w.is_start} → {w.is_end}")
            print(f"  OOS: {w.oos_start} → {w.oos_end}")
            print(f"  Best params: {w.best_params}")
            print(f"  IS  → trades={w.is_metrics.total_trades:>3}  "
                  f"annual={w.is_metrics.annual_return_pct:>7.1f}%  "
                  f"sharpe={w.is_metrics.sharpe_ratio:>5.2f}  "
                  f"dd={w.is_metrics.max_drawdown_pct:>5.1f}%  "
                  f"wr={w.is_metrics.winrate:>4.1f}%")
            print(f"  OOS → trades={w.oos_metrics.total_trades:>3}  "
                  f"annual={w.oos_metrics.annual_return_pct:>7.1f}%  "
                  f"sharpe={w.oos_metrics.sharpe_ratio:>5.2f}  "
                  f"dd={w.oos_metrics.max_drawdown_pct:>5.1f}%  "
                  f"wr={w.oos_metrics.winrate:>4.1f}%")
            print(f"  Degradation: PnL={w.pnl_degradation:>+.1f}%  "
                  f"Sharpe={w.sharpe_degradation:>+.1f}%")

        print(f"\n{'=' * 70}")
        print(f"AGGREGATE OOS:")
        print(f"  Avg annual return: {self.avg_oos_annual_return:.1f}%")
        print(f"  Avg Sharpe:        {self.avg_oos_sharpe:.2f}")
        print(f"  Avg PnL degrad:    {self.avg_pnl_degradation:+.1f}%")
        print(f"  Avg Sharpe degrad: {self.avg_sharpe_degradation:+.1f}%")
        verdict = "PASS ✓" if self.is_robust else "FAIL ✗"
        print(f"\n  VERDICT: {verdict}")
        print("=" * 70)

    def plot(self, save_path: str | None = None) -> None:
        """Plot IS vs OOS metrics per window + combined OOS equity."""
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("Walk-Forward Analysis", fontsize=14, fontweight="bold")

        n = len(self.windows)
        x = list(range(1, n + 1))

        # Annual return IS vs OOS
        ax = axes[0, 0]
        is_ret = [w.is_metrics.annual_return_pct for w in self.windows]
        oos_ret = [w.oos_metrics.annual_return_pct for w in self.windows]
        ax.bar([i - 0.15 for i in x], is_ret, 0.3, label="In-Sample", alpha=0.8)
        ax.bar([i + 0.15 for i in x], oos_ret, 0.3, label="Out-of-Sample", alpha=0.8)
        ax.axhline(y=15, color="green", linestyle="--", alpha=0.5, label="Target 15%")
        ax.set_title("Annual Return (%)")
        ax.set_xlabel("Window")
        ax.legend()

        # Sharpe IS vs OOS
        ax = axes[0, 1]
        is_sh = [w.is_metrics.sharpe_ratio for w in self.windows]
        oos_sh = [w.oos_metrics.sharpe_ratio for w in self.windows]
        ax.bar([i - 0.15 for i in x], is_sh, 0.3, label="In-Sample", alpha=0.8)
        ax.bar([i + 0.15 for i in x], oos_sh, 0.3, label="Out-of-Sample", alpha=0.8)
        ax.axhline(y=1.0, color="green", linestyle="--", alpha=0.5, label="Target 1.0")
        ax.set_title("Sharpe Ratio")
        ax.set_xlabel("Window")
        ax.legend()

        # Max drawdown per window
        ax = axes[1, 0]
        oos_dd = [w.oos_metrics.max_drawdown_pct for w in self.windows]
        ax.bar(x, oos_dd, color="red", alpha=0.6)
        ax.set_title("OOS Max Drawdown (%)")
        ax.set_xlabel("Window")

        # Combined OOS equity curve
        ax = axes[1, 1]
        all_oos_pnls = []
        for w in self.windows:
            for t in w.oos_metrics.trades:
                all_oos_pnls.append(t["pnl_pct"])
        if all_oos_pnls:
            cum_pnl = np.cumsum(all_oos_pnls)
            ax.plot(cum_pnl, color="blue", linewidth=1.5)
            ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        ax.set_title("Combined OOS Equity Curve (cum %)")
        ax.set_xlabel("Trade #")

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"[WFA] Plot saved to {save_path}")
        plt.show()


class WalkForwardAnalysis:
    """
    Walk-Forward optimizer + validator.

    Splits data into rolling windows, optimizes on IS, validates on OOS.
    """

    def __init__(
        self,
        adapter: StrategyAdapter,
        param_grid: dict[str, list],
        is_ratio: float = 0.70,
        optimization_target: str = "sharpe",  # "sharpe", "annual_return", "pnl"
    ) -> None:
        self.adapter = adapter
        self.param_grid = param_grid
        self.is_ratio = is_ratio
        self.optimization_target = optimization_target

        # Expand grid
        keys = list(param_grid.keys())
        vals = list(param_grid.values())
        self._param_combos = [dict(zip(keys, combo)) for combo in itertools.product(*vals)]
        print(f"[WFA] Parameter grid: {len(self._param_combos)} combinations")

    def run(
        self,
        df: pd.DataFrame,
        n_windows: int = 5,
        overlap: float = 0.0,
        verbose: bool = True,
    ) -> WalkForwardReport:
        """
        Run walk-forward analysis.

        Args:
            df: Full OHLCV dataset
            n_windows: Number of rolling windows
            overlap: Fraction of overlap between consecutive windows (0.0 = no overlap)
            verbose: Print progress

        Returns:
            WalkForwardReport with per-window and aggregate results
        """
        windows = self._split_windows(df, n_windows, overlap)
        report = WalkForwardReport()

        for idx, (is_df, oos_df) in enumerate(windows):
            if verbose:
                print(f"\n[WFA] Window {idx + 1}/{n_windows}")
                print(f"  IS:  {is_df['ts'].iloc[0]} → {is_df['ts'].iloc[-1]} ({len(is_df)} bars)")
                print(f"  OOS: {oos_df['ts'].iloc[0]} → {oos_df['ts'].iloc[-1]} ({len(oos_df)} bars)")

            # Optimize on IS
            best_params, is_metrics = self._optimize_is(is_df, verbose)

            if verbose:
                print(f"  Best params: {best_params}")
                print(f"  IS  → annual={is_metrics.annual_return_pct:.1f}%  "
                      f"sharpe={is_metrics.sharpe_ratio:.2f}")

            # Validate on OOS with best params
            oos_metrics = self.adapter.run(oos_df, params_override=best_params)

            if verbose:
                print(f"  OOS → annual={oos_metrics.annual_return_pct:.1f}%  "
                      f"sharpe={oos_metrics.sharpe_ratio:.2f}")

            result = WindowResult(
                window_idx=idx,
                is_start=is_df["ts"].iloc[0],
                is_end=is_df["ts"].iloc[-1],
                oos_start=oos_df["ts"].iloc[0],
                oos_end=oos_df["ts"].iloc[-1],
                best_params=best_params,
                is_metrics=is_metrics,
                oos_metrics=oos_metrics,
            )
            report.windows.append(result)

        report.print_summary()
        return report

    def _split_windows(
        self,
        df: pd.DataFrame,
        n_windows: int,
        overlap: float,
    ) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
        """Split df into n rolling (IS, OOS) pairs."""
        total = len(df)
        # Each window covers (total / (n_windows * (1 - overlap) + overlap)) bars
        # Simplified: equal-sized windows that advance by step_size
        window_size = total // n_windows if overlap == 0 else int(total / (n_windows * (1 - overlap) + overlap))
        step_size = int(window_size * (1 - overlap)) if overlap > 0 else window_size

        windows = []
        for i in range(n_windows):
            start = i * step_size
            end = min(start + window_size, total)
            if end - start < 100:  # skip tiny windows
                continue

            split = start + int((end - start) * self.is_ratio)
            is_df = df.iloc[start:split].reset_index(drop=True)
            oos_df = df.iloc[split:end].reset_index(drop=True)

            if len(is_df) < 50 or len(oos_df) < 20:
                continue

            windows.append((is_df, oos_df))

        return windows

    def _optimize_is(
        self,
        is_df: pd.DataFrame,
        verbose: bool,
    ) -> tuple[dict[str, Any], BacktestMetrics]:
        """Grid search on IS data, return best params + IS metrics."""
        best_score = -np.inf
        best_params: dict[str, Any] = {}
        best_metrics = BacktestMetrics()

        total = len(self._param_combos)
        for idx, params in enumerate(self._param_combos):
            if verbose and (idx + 1) % max(1, total // 10) == 0:
                print(f"    Optimizing: {idx + 1}/{total} ...", end="\r")

            metrics = self.adapter.run(is_df, params_override=params)
            score = self._score(metrics)

            if score > best_score:
                best_score = score
                best_params = params
                best_metrics = metrics

        if verbose:
            print(f"    Optimizing: {total}/{total} done.     ")

        return best_params, best_metrics

    def _score(self, m: BacktestMetrics) -> float:
        """Score a backtest result for optimization ranking."""
        if m.total_trades < 5:
            return -np.inf

        if self.optimization_target == "sharpe":
            return m.sharpe_ratio
        elif self.optimization_target == "annual_return":
            return m.annual_return_pct
        elif self.optimization_target == "pnl":
            return m.total_pnl_pct
        else:
            # Default: weighted combination
            return m.sharpe_ratio * 0.6 + m.annual_return_pct * 0.004
