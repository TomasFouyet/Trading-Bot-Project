#!/usr/bin/env python3
"""
Run Permutation Test on BTC/USDT 15m with HTF=ON, rr=1.5, sl_atr=2.0.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")

from validation.data_loader import load_candles
from validation.permutation_test import PermutationTest


def main():
    print("=" * 70)
    print("PERMUTATION TEST — BTC/USDT 15m, HTF=ON")
    print("Config: rr=1.5, sl_atr=2.0, n_permutations=1000")
    print("=" * 70)

    df = load_candles("BTC/USDT", "15m", days=730)
    print(f"Data: {len(df)} bars\n")

    output_dir = ROOT / "validation" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()

    pt = PermutationTest(
        df=df,
        n_permutations=1000,
        seed=42,
        rr_ratio=1.5,
        atr_sl_mult=2.0,
        use_htf=True,
    )
    report = pt.run()
    report.print_summary()
    report.plot(save_path=str(output_dir / "permutation_test_BTCUSDT.png"))

    total = time.time() - t0
    print(f"\nTotal time: {total:.0f}s ({total/60:.1f} min)")

    # ── Interpretation ───────────────────────────────────────────
    print(f"\n{'='*70}")
    print("INTERPRETATION")
    print(f"{'='*70}")

    if report.real_sharpe < 0:
        print("  Real Sharpe is NEGATIVE. Test is meaningless.")
    elif report.p_value < 0.01:
        print(f"  STRONG EVIDENCE of real edge. The strategy's Sharpe of "
              f"{report.real_sharpe:.2f}")
        print(f"  exceeds 99% of random permutations. "
              f"Probability this is noise: <1%.")
        print(f"  Z-score of {report.z_score:.1f} means the real edge is "
              f"{report.z_score:.1f} standard deviations")
        print(f"  above what random price structure produces.")
    elif report.p_value < 0.05:
        print(f"  EVIDENCE of real edge. The strategy's Sharpe of "
              f"{report.real_sharpe:.2f}")
        print(f"  exceeds 95% of random permutations. "
              f"Standard statistical threshold met.")
    elif report.p_value < 0.10:
        print(f"  WEAK EVIDENCE. The strategy exceeds 90% of permutations but")
        print(f"  does not meet the standard p<0.05 threshold. "
              f"Treat with caution.")
    else:
        print(f"  NO STATISTICAL EVIDENCE of edge. The Sharpe could appear by")
        print(f"  chance on random data. Do NOT proceed to paper trading.")
        print(f"  Revisit strategy hypothesis from scratch.")

    print(f"\n{'='*70}")


if __name__ == "__main__":
    main()
