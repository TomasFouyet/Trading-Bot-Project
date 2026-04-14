"""
Monte Carlo simulation for trade-sequence robustness testing.

Two complementary simulation methods:

1. **Bootstrap resampling** (with replacement): Answers "what range of
   outcomes could this strategy produce?" by sampling N trades from the
   observed pool with replacement. Each simulation gets a different MIX
   of trades, so the final PnL varies. This produces real distributions
   of final PnL and tests whether the observed result is statistically
   significant.

2. **Shuffle** (without replacement): Answers "how bad can the path get
   even if we keep the same trades?" by reordering the exact same set.
   Final PnL is always identical (commutative), but drawdown and
   consecutive losses vary. This measures path-dependency risk.

Usage:
    from validation.monte_carlo import MonteCarloSimulation
    mc = MonteCarloSimulation(trades, n_simulations=5000)
    report = mc.run()
    report.print_summary()
    report.plot()
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class MonteCarloReport:
    """Results of Monte Carlo simulation."""
    n_simulations: int = 0
    n_trades: int = 0
    method: str = ""

    # Original sequence metrics
    original_final_pnl: float = 0.0
    original_max_dd: float = 0.0

    # Distribution of final PnL (from bootstrap)
    pnl_p5: float = 0.0
    pnl_p25: float = 0.0
    pnl_p50: float = 0.0
    pnl_p75: float = 0.0
    pnl_p95: float = 0.0
    pnl_mean: float = 0.0

    # Distribution of max drawdown (from shuffle)
    dd_p5: float = 0.0
    dd_p25: float = 0.0
    dd_p50: float = 0.0
    dd_p75: float = 0.0
    dd_p95: float = 0.0
    dd_mean: float = 0.0

    # Distribution of max consecutive losses (from shuffle)
    consec_loss_p50: float = 0.0
    consec_loss_p95: float = 0.0

    # Raw distributions for plotting
    final_pnls: np.ndarray = field(default_factory=lambda: np.array([]))
    max_dds: np.ndarray = field(default_factory=lambda: np.array([]))

    @property
    def risk_of_ruin_pct(self) -> float:
        """% of bootstrap simulations that ended negative."""
        if len(self.final_pnls) == 0:
            return 0.0
        return float(np.mean(self.final_pnls < 0) * 100)

    def print_summary(self) -> None:
        print("\n" + "=" * 70)
        print("MONTE CARLO SIMULATION REPORT")
        print("=" * 70)
        print(f"  Simulations:  {self.n_simulations:,}")
        print(f"  Trades:       {self.n_trades}")
        print(f"  Method:       bootstrap (PnL) + shuffle (DD)")
        print()
        print(f"  Original sequence:")
        print(f"    Final PnL:    {self.original_final_pnl:>+8.2f}%")
        print(f"    Max Drawdown: {self.original_max_dd:>8.2f}%")
        print()
        print(f"  Final PnL Distribution (bootstrap with replacement):")
        print(f"    P5  = {self.pnl_p5:>+8.2f}%   (worst case)")
        print(f"    P25 = {self.pnl_p25:>+8.2f}%")
        print(f"    P50 = {self.pnl_p50:>+8.2f}%   (median)")
        print(f"    P75 = {self.pnl_p75:>+8.2f}%")
        print(f"    P95 = {self.pnl_p95:>+8.2f}%   (best case)")
        print()
        print(f"  Max Drawdown Distribution (shuffle, path risk):")
        print(f"    P5  = {self.dd_p5:>8.2f}%   (best case)")
        print(f"    P50 = {self.dd_p50:>8.2f}%   (median)")
        print(f"    P95 = {self.dd_p95:>8.2f}%   (worst case)")
        print()
        print(f"  Max Consecutive Losses:")
        print(f"    P50 = {self.consec_loss_p50:.0f}   P95 = {self.consec_loss_p95:.0f}")
        print()
        print(f"  Risk of Ruin (negative final PnL): {self.risk_of_ruin_pct:.1f}%")
        print("=" * 70)

    def plot(self, save_path: str | None = None) -> None:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("Monte Carlo Simulation", fontsize=14, fontweight="bold")

        # Final PnL histogram (bootstrap)
        ax = axes[0, 0]
        pnl_bins = min(80, max(10, len(np.unique(self.final_pnls)) // 2))
        ax.hist(self.final_pnls, bins=pnl_bins, alpha=0.7, color="steelblue", edgecolor="none")
        ax.axvline(self.original_final_pnl, color="red", linewidth=2, label="Original")
        ax.axvline(self.pnl_p5, color="orange", linestyle="--", label=f"P5={self.pnl_p5:.1f}%")
        ax.axvline(self.pnl_p50, color="green", linestyle="--", label=f"P50={self.pnl_p50:.1f}%")
        ax.axvline(self.pnl_p95, color="blue", linestyle="--", label=f"P95={self.pnl_p95:.1f}%")
        ax.set_title("Final PnL Distribution — Bootstrap (%)")
        ax.legend(fontsize=8)

        # Max drawdown histogram (shuffle)
        ax = axes[0, 1]
        dd_bins = min(80, max(10, len(np.unique(self.max_dds)) // 2))
        ax.hist(self.max_dds, bins=dd_bins, alpha=0.7, color="salmon", edgecolor="none")
        ax.axvline(self.original_max_dd, color="red", linewidth=2, label="Original")
        ax.axvline(self.dd_p50, color="green", linestyle="--", label=f"P50={self.dd_p50:.1f}%")
        ax.axvline(self.dd_p95, color="orange", linestyle="--", label=f"P95={self.dd_p95:.1f}%")
        ax.set_title("Max Drawdown Distribution — Shuffle (%)")
        ax.legend(fontsize=8)

        # Sample bootstrap equity curves
        ax = axes[1, 0]
        if hasattr(self, "_pnl_array"):
            rng = np.random.default_rng(42)
            n = len(self._pnl_array)
            for _ in range(100):
                sample = rng.choice(self._pnl_array, size=n, replace=True)
                eq = (np.cumprod(1.0 + sample / 100.0) - 1.0) * 100
                ax.plot(eq, alpha=0.05, color="steelblue", linewidth=0.5)
            orig_eq = (np.cumprod(1.0 + self._pnl_array / 100.0) - 1.0) * 100
            ax.plot(orig_eq, color="red", linewidth=2, label="Original")
        ax.set_title("Sample Bootstrap Equity Curves")
        ax.set_xlabel("Trade #")
        ax.legend()

        # Risk summary text
        ax = axes[1, 1]
        ax.axis("off")
        text = (
            f"Monte Carlo Summary\n"
            f"{'─' * 35}\n"
            f"Simulations:    {self.n_simulations:,}\n"
            f"Trades:         {self.n_trades}\n\n"
            f"Expected DD:    {self.dd_p50:.1f}% (P50)\n"
            f"Worst DD:       {self.dd_p95:.1f}% (P95)\n\n"
            f"Expected PnL:   {self.pnl_p50:+.1f}% (P50)\n"
            f"Worst PnL:      {self.pnl_p5:+.1f}% (P5)\n\n"
            f"Risk of Ruin:   {self.risk_of_ruin_pct:.1f}%\n"
            f"Max Consec Loss: {self.consec_loss_p95:.0f} (P95)"
        )
        ax.text(0.1, 0.5, text, transform=ax.transAxes, fontsize=12,
                verticalalignment="center", fontfamily="monospace",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"[MC] Plot saved to {save_path}")
        plt.show()


class MonteCarloSimulation:
    """
    Combined bootstrap + shuffle Monte Carlo for trading strategies.

    - Bootstrap (with replacement): varies which trades occur → PnL distribution
    - Shuffle (without replacement): varies trade order → drawdown distribution
    """

    def __init__(
        self,
        trades: list[dict],
        n_simulations: int = 5000,
        seed: int = 42,
    ) -> None:
        self.trades = trades
        self.n_simulations = n_simulations
        self.rng = np.random.default_rng(seed)

        self._pnls = np.array([t["pnl_pct"] for t in trades], dtype=np.float64)

    def run(self) -> MonteCarloReport:
        """Run combined bootstrap + shuffle simulation."""
        n = len(self._pnls)
        if n < 5:
            print("[MC] Warning: fewer than 5 trades — results unreliable.")
            return MonteCarloReport(n_simulations=0, n_trades=n)

        print(f"[MC] Running {self.n_simulations:,} simulations on {n} trades ...")
        print(f"  Trade stats: mean={self._pnls.mean():.4f}%  "
              f"std={self._pnls.std():.4f}%  "
              f"min={self._pnls.min():.4f}%  max={self._pnls.max():.4f}%")

        # ═══════════════════════════════════════════════════════════════
        # BOOTSTRAP (with replacement) → Final PnL distribution
        # Each sim draws n trades from the pool, allowing repeats.
        # This varies WHICH trades occur, so final PnL changes.
        # ═══════════════════════════════════════════════════════════════
        bootstrap_indices = self.rng.choice(n, size=(self.n_simulations, n), replace=True)
        bootstrap_samples = self._pnls[bootstrap_indices]  # (n_sims, n_trades)

        bootstrap_mult = 1.0 + bootstrap_samples / 100.0
        bootstrap_equity = np.cumprod(bootstrap_mult, axis=1)
        final_pnls = (bootstrap_equity[:, -1] - 1.0) * 100

        # ═══════════════════════════════════════════════════════════════
        # SHUFFLE (without replacement) → Drawdown distribution
        # Same trades, different order. Final PnL is invariant (by
        # commutativity), but the equity PATH changes → DD changes.
        # ═══════════════════════════════════════════════════════════════
        all_shuffled = np.tile(self._pnls, (self.n_simulations, 1))
        for i in range(self.n_simulations):
            self.rng.shuffle(all_shuffled[i])

        shuffle_mult = 1.0 + all_shuffled / 100.0
        shuffle_equity = np.cumprod(shuffle_mult, axis=1)

        running_max = np.maximum.accumulate(shuffle_equity, axis=1)
        drawdowns = (shuffle_equity - running_max) / running_max * 100
        max_dds = np.abs(np.min(drawdowns, axis=1))

        # Max consecutive losses per shuffle
        consec_losses = np.array([
            self._max_consecutive_losses(all_shuffled[i])
            for i in range(self.n_simulations)
        ])

        # ═══════════════════════════════════════════════════════════════
        # ORIGINAL sequence metrics
        # ═══════════════════════════════════════════════════════════════
        orig_mult = 1.0 + self._pnls / 100.0
        orig_equity = np.cumprod(orig_mult)
        orig_final = float((orig_equity[-1] - 1.0) * 100)
        orig_rm = np.maximum.accumulate(orig_equity)
        orig_dd = float(np.abs(np.min((orig_equity - orig_rm) / orig_rm * 100)))

        print(f"  Bootstrap PnL: P5={np.percentile(final_pnls,5):+.1f}%  "
              f"P50={np.percentile(final_pnls,50):+.1f}%  "
              f"P95={np.percentile(final_pnls,95):+.1f}%")
        print(f"  Shuffle DD:    P50={np.percentile(max_dds,50):.1f}%  "
              f"P95={np.percentile(max_dds,95):.1f}%")

        report = MonteCarloReport(
            n_simulations=self.n_simulations,
            n_trades=n,
            method="bootstrap+shuffle",
            original_final_pnl=orig_final,
            original_max_dd=orig_dd,
            pnl_p5=float(np.percentile(final_pnls, 5)),
            pnl_p25=float(np.percentile(final_pnls, 25)),
            pnl_p50=float(np.percentile(final_pnls, 50)),
            pnl_p75=float(np.percentile(final_pnls, 75)),
            pnl_p95=float(np.percentile(final_pnls, 95)),
            pnl_mean=float(np.mean(final_pnls)),
            dd_p5=float(np.percentile(max_dds, 5)),
            dd_p25=float(np.percentile(max_dds, 25)),
            dd_p50=float(np.percentile(max_dds, 50)),
            dd_p75=float(np.percentile(max_dds, 75)),
            dd_p95=float(np.percentile(max_dds, 95)),
            dd_mean=float(np.mean(max_dds)),
            consec_loss_p50=float(np.percentile(consec_losses, 50)),
            consec_loss_p95=float(np.percentile(consec_losses, 95)),
            final_pnls=final_pnls,
            max_dds=max_dds,
        )
        report._pnl_array = self._pnls

        print(f"[MC] Done.")
        return report

    @staticmethod
    def _max_consecutive_losses(pnls: np.ndarray) -> int:
        """Count maximum consecutive losing trades."""
        max_streak = 0
        current = 0
        for p in pnls:
            if p <= 0:
                current += 1
                max_streak = max(max_streak, current)
            else:
                current = 0
        return max_streak
