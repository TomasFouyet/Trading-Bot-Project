"""
Permutation Test — Block Bootstrap on daily returns.

Shuffles daily log returns to build synthetic price series,
runs the strategy on each, and builds a null distribution of Sharpe ratios.
If the real Sharpe sits in the top 5% → statistically significant edge.

This is stronger than Monte Carlo (which only reshuffles trade sequence):
permutation tests whether the *market structure itself* is necessary
for the edge to exist.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from validation.fast_backtest import compute_indicators, compute_htf_bias, fast_backtest


# ─── Synthetic series construction ───────────────────────────────────

def _compute_daily_returns(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute daily log returns and the day index for each 15m bar.

    Returns:
        daily_log_returns: array of shape (n_days,)
        bar_day_idx: array of shape (n_bars,) mapping each bar to its day
    """
    ts = pd.DatetimeIndex(df["ts"])
    dates = ts.normalize()  # floor to date
    unique_dates = dates.unique().sort_values()
    date_to_idx = {d: i for i, d in enumerate(unique_dates)}
    bar_day_idx = np.array([date_to_idx[d] for d in dates])

    close = df["close"].values.astype(np.float64)
    n_days = len(unique_dates)
    daily_log_returns = np.zeros(n_days)

    for day_i in range(n_days):
        mask = bar_day_idx == day_i
        day_closes = close[mask]
        if len(day_closes) >= 2:
            daily_log_returns[day_i] = np.log(day_closes[-1] / day_closes[0])
        # else: 0.0 (single bar day)

    return daily_log_returns, bar_day_idx


def _build_synthetic_df(
    df: pd.DataFrame,
    shuffled_daily_returns: np.ndarray,
    bar_day_idx: np.ndarray,
    original_daily_returns: np.ndarray,
) -> pd.DataFrame:
    """
    Reconstruct a synthetic OHLCV DataFrame from shuffled daily returns.

    The ratio between synthetic and real cumulative returns is applied
    proportionally to all price columns. Volume is unchanged.
    """
    n_days = len(original_daily_returns)
    n_bars = len(df)

    # Cumulative log return per day (original and synthetic)
    orig_cum = np.cumsum(original_daily_returns)
    synth_cum = np.cumsum(shuffled_daily_returns)

    # Per-bar cumulative offset: for each bar, the cumulative return
    # UP TO the start of its day (exclusive of intraday movement)
    # Plus its fractional intraday position
    close = df["close"].values.astype(np.float64)

    # Build per-bar price ratio
    # For each bar: ratio = exp(synth_cumulative - orig_cumulative) at that bar's day
    # But we need intraday structure preserved within each day.
    # Approach: scale entire day's prices by the ratio at day boundaries.

    # Day boundary cumulative returns (before day 0 = 0)
    orig_day_start_cum = np.zeros(n_days)
    synth_day_start_cum = np.zeros(n_days)
    orig_day_start_cum[1:] = orig_cum[:-1]
    synth_day_start_cum[1:] = synth_cum[:-1]

    # Per-bar scaling factor: shift from original day position to synthetic
    day_idx = bar_day_idx
    bar_orig_offset = orig_day_start_cum[day_idx]
    bar_synth_offset = synth_day_start_cum[day_idx]
    ratio = np.exp(bar_synth_offset - bar_orig_offset)

    # Apply ratio to OHLC
    syn_df = df.copy()
    syn_df["open"] = df["open"].values * ratio
    syn_df["high"] = df["high"].values * ratio
    syn_df["low"] = df["low"].values * ratio
    syn_df["close"] = df["close"].values * ratio
    # Volume unchanged

    return syn_df


# ─── Report ──────────────────────────────────────────────────────────

@dataclass
class PermutationReport:
    real_sharpe: float = 0.0
    null_sharpes: np.ndarray = field(default_factory=lambda: np.array([]))
    p_value: float = 1.0
    is_significant: bool = False
    null_mean: float = 0.0
    null_std: float = 0.0
    null_p95: float = 0.0
    z_score: float = 0.0
    n_permutations: int = 0
    real_trades: int = 0

    def print_summary(self) -> None:
        print(f"\n{'='*60}")
        print("PERMUTATION TEST RESULTS")
        print(f"{'='*60}")
        print(f"  Real Sharpe:       {self.real_sharpe:.2f}  ({self.real_trades} trades)")
        print(f"  Null mean:         {self.null_mean:.2f} +/- {self.null_std:.2f}")
        print(f"  Null P95:          {self.null_p95:.2f}")
        print(f"  Z-score:           {self.z_score:.2f}")
        print(f"  P-value:           {self.p_value:.4f}")
        sig = "YES (p < 0.05)" if self.is_significant else "NO (p >= 0.05)"
        print(f"  Significant:       {sig}")
        print(f"  Permutations:      {self.n_permutations}")

        # Null distribution percentiles
        if len(self.null_sharpes) > 0:
            ps = np.percentile(self.null_sharpes, [5, 25, 50, 75, 95])
            print(f"  Null distribution: P5={ps[0]:.2f}  P25={ps[1]:.2f}  "
                  f"P50={ps[2]:.2f}  P75={ps[3]:.2f}  P95={ps[4]:.2f}")

        print(f"{'='*60}")

    def plot(self, save_path: str | None = None) -> None:
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5),
                                        gridspec_kw={"width_ratios": [2, 1]})

        # Left: histogram
        valid = self.null_sharpes[np.isfinite(self.null_sharpes)]
        n_bins = min(60, max(15, len(np.unique(valid)) // 3))
        ax1.hist(valid, bins=n_bins, alpha=0.7, color="steelblue",
                 edgecolor="white", label="Null distribution")
        ax1.axvline(self.real_sharpe, color="red", linewidth=2,
                    label=f"Real Sharpe = {self.real_sharpe:.2f}")
        ax1.axvline(self.null_p95, color="green", linewidth=1.5,
                    linestyle="--", label=f"Null P95 = {self.null_p95:.2f}")
        ax1.set_xlabel("Sharpe Ratio")
        ax1.set_ylabel("Count")
        ax1.set_title("Permutation Test — Null Distribution")
        ax1.legend(fontsize=9)

        sig_text = "SIGNIFICANT" if self.is_significant else "NOT SIGNIFICANT"
        sig_color = "green" if self.is_significant else "red"
        ax1.annotate(f"p-value = {self.p_value:.4f}\n{sig_text}",
                     xy=(0.97, 0.95), xycoords="axes fraction",
                     ha="right", va="top", fontsize=11, fontweight="bold",
                     color=sig_color,
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                               edgecolor=sig_color, alpha=0.9))

        # Right: text summary
        ax2.axis("off")
        lines = [
            f"Real Sharpe:     {self.real_sharpe:.2f}",
            f"Real Trades:     {self.real_trades}",
            "",
            f"Null mean:       {self.null_mean:.2f} +/- {self.null_std:.2f}",
            f"Null P95:        {self.null_p95:.2f}",
            "",
            f"Z-score:         {self.z_score:.2f}",
            f"P-value:         {self.p_value:.4f}",
            "",
            f"Result:          {sig_text}",
            f"Permutations:    {self.n_permutations}",
        ]
        text = "\n".join(lines)
        ax2.text(0.1, 0.95, text, transform=ax2.transAxes,
                 fontsize=11, verticalalignment="top",
                 fontfamily="monospace",
                 bbox=dict(boxstyle="round", facecolor="lightyellow",
                           edgecolor="gray", alpha=0.9))
        ax2.set_title("Summary")

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"[Permutation] Plot saved to {save_path}")
        plt.close(fig)


# ─── Main class ──────────────────────────────────────────────────────

class PermutationTest:
    """
    Block Bootstrap Permutation Test.

    Shuffles daily log returns to destroy cross-day momentum while
    preserving intraday structure. Runs strategy on each synthetic
    series and builds a null Sharpe distribution.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        n_permutations: int = 1000,
        seed: int = 42,
        rr_ratio: float = 1.5,
        atr_sl_mult: float = 2.0,
        use_htf: bool = True,
        verbose: bool = True,
    ):
        self.df = df
        self.n_permutations = n_permutations
        self.seed = seed
        self.rr_ratio = rr_ratio
        self.atr_sl_mult = atr_sl_mult
        self.use_htf = use_htf
        self.verbose = verbose

    def run(self) -> PermutationReport:
        import time

        df = self.df

        # Pre-compute daily returns and day mapping
        daily_returns, bar_day_idx = _compute_daily_returns(df)
        n_days = len(daily_returns)

        if self.verbose:
            print(f"[Permutation] {n_days} daily returns, "
                  f"{len(df)} bars, {self.n_permutations} permutations")

        # ── Real backtest ────────────────────────────────────────
        t0 = time.time()
        htf_bias = compute_htf_bias(df) if self.use_htf else None
        df_ind = compute_indicators(df, ema_fast_p=20, ema_slow_p=50)
        real_m = fast_backtest(
            df_ind, rr_ratio=self.rr_ratio, atr_sl_mult=self.atr_sl_mult,
            precomputed=True, htf_bias=htf_bias,
        )
        real_sharpe = real_m.sharpe_ratio
        real_trades = real_m.total_trades

        if self.verbose:
            print(f"[Permutation] Real: Sharpe={real_sharpe:.2f}  "
                  f"Trades={real_trades}  ({time.time()-t0:.1f}s)")

        if real_sharpe < 0:
            print("[Permutation] WARNING: Real Sharpe is negative. "
                  "Permutation test is meaningless. Stopping.")
            return PermutationReport(
                real_sharpe=real_sharpe,
                real_trades=real_trades,
                n_permutations=0,
            )

        # ── Null distribution ────────────────────────────────────
        null_sharpes = np.zeros(self.n_permutations)
        t_start = time.time()

        for perm_i in range(self.n_permutations):
            rng = np.random.default_rng(self.seed + perm_i)

            # Shuffle daily returns
            shuffled = daily_returns.copy()
            rng.shuffle(shuffled)

            # Build synthetic DataFrame
            syn_df = _build_synthetic_df(df, shuffled, bar_day_idx, daily_returns)

            # Recompute HTF bias on synthetic data (synchronized)
            syn_htf = compute_htf_bias(syn_df) if self.use_htf else None

            # Run backtest on synthetic data
            syn_ind = compute_indicators(syn_df, ema_fast_p=20, ema_slow_p=50)
            syn_m = fast_backtest(
                syn_ind, rr_ratio=self.rr_ratio, atr_sl_mult=self.atr_sl_mult,
                precomputed=True, htf_bias=syn_htf,
            )
            null_sharpes[perm_i] = syn_m.sharpe_ratio

            if self.verbose and (perm_i + 1) % 100 == 0:
                elapsed = time.time() - t_start
                rate = (perm_i + 1) / elapsed
                eta = (self.n_permutations - perm_i - 1) / rate
                print(f"[Permutation] {perm_i+1}/{self.n_permutations}  "
                      f"({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)  "
                      f"null_mean={np.mean(null_sharpes[:perm_i+1]):.2f}",
                      flush=True)

        total_time = time.time() - t_start

        # ── Compute statistics ───────────────────────────────────
        valid_nulls = null_sharpes[np.isfinite(null_sharpes)]
        null_mean = float(np.mean(valid_nulls))
        null_std = float(np.std(valid_nulls, ddof=1)) if len(valid_nulls) > 1 else 1.0
        null_p95 = float(np.percentile(valid_nulls, 95))
        p_value = float(np.mean(valid_nulls >= real_sharpe))
        z_score = (real_sharpe - null_mean) / null_std if null_std > 0 else 0.0

        if self.verbose:
            print(f"[Permutation] Done in {total_time:.0f}s "
                  f"({total_time/self.n_permutations:.1f}s per permutation)")

        report = PermutationReport(
            real_sharpe=real_sharpe,
            null_sharpes=null_sharpes,
            p_value=p_value,
            is_significant=p_value < 0.05,
            null_mean=null_mean,
            null_std=null_std,
            null_p95=null_p95,
            z_score=z_score,
            n_permutations=self.n_permutations,
            real_trades=real_trades,
        )

        return report
