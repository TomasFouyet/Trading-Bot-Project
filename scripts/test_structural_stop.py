"""Unit tests for validation/structural_stop.py."""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from validation.structural_stop import (
    compute_pivot_lows,
    compute_pivot_highs,
    build_last_pivot_arrays,
    compute_structural_sl,
)


def check(name, cond):
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {name}")
    return cond


def test_basic_pivots():
    # Strict pivot at index 3 (value 95 < 100,99,98 on left; < 97,99,101 on right)
    low = np.array([100.0, 99, 98, 95, 97, 99, 101, 102, 103, 104, 105, 104, 103])
    high = np.array([105.0, 104, 103, 100, 102, 104, 106, 107, 108, 109, 110, 109, 108])

    pl = compute_pivot_lows(low, left=3, right=3)
    ph = compute_pivot_highs(high, left=3, right=3)

    pl_idx = np.where(~np.isnan(pl))[0].tolist()
    ph_idx = np.where(~np.isnan(ph))[0].tolist()
    print(f"    pivot lows detected at: {pl_idx} (value={[float(pl[i]) for i in pl_idx]})")
    print(f"    pivot highs detected at: {ph_idx} (value={[float(ph[i]) for i in ph_idx]})")

    ok1 = check("pivot low at idx 3 is 95.0", 3 in pl_idx and pl[3] == 95.0)
    # Pivot high at idx 10 (110): left=109,108,107 all <110 ok; right=109,108 but only 2 bars after... n=13, right=3 needs i<=9
    # So high[10]=110 fails because right needs i+3 < n=13 → i<10. Actually i<n-right=13-3=10 → i<10. So idx 10 is excluded.
    # Let's check which ones valid: range(left, n-right)=range(3,10). So only indices 3..9.
    # idx 6: high=106, left: 103,104,100 - 100<106 so 106>100 yes. 106>104 yes. 106>103 yes. OK.
    # right: 107,108,109 - 106<107 fails. So idx 6 not pivot.
    # idx 3: high=100, left: 103,104,105 - 100<103 so fails (not pivot high since 100 not max).
    # No pivot highs in range!
    ok2 = check("pivot highs list in valid range [3,9]", all(3 <= i <= 9 for i in ph_idx))
    return ok1 and ok2


def test_lookahead_arrays():
    low = np.array([100.0, 99, 98, 95, 97, 99, 101, 102, 103, 104, 105, 104, 103])
    high = np.array([105.0, 104, 103, 100, 102, 104, 106, 107, 108, 109, 110, 109, 108])
    pl = compute_pivot_lows(low, 3, 3)
    ph = compute_pivot_highs(high, 3, 3)
    last_low, last_high = build_last_pivot_arrays(pl, ph, right=3)

    # Pivot at bar 3 confirmed at bar 3+3=6. So last_low[5]=nan, last_low[6]=95.0.
    print(f"    last_low[3..9]: {last_low[3:10].tolist()}")
    ok1 = check("last_low[5] is nan (before confirmation)", np.isnan(last_low[5]))
    ok2 = check("last_low[6] = 95.0 (confirmed at bar 6)", last_low[6] == 95.0)
    ok3 = check("last_low[9] = 95.0 (still most recent)", last_low[9] == 95.0)
    return ok1 and ok2 and ok3


def test_structural_sl_long():
    low = np.array([100.0, 99, 98, 95, 97, 99, 101, 102, 103, 104, 105, 104, 103])
    high = np.array([105.0, 104, 103, 100, 102, 104, 106, 107, 108, 109, 110, 109, 108])
    pl = compute_pivot_lows(low, 3, 3)
    ph = compute_pivot_highs(high, 3, 3)
    last_low, last_high = build_last_pivot_arrays(pl, ph, right=3)

    # At bar 9, last pivot low = 95.0, atr=2.0, buffer=0.25 → struct=95-0.5=94.5
    # entry=104, min_risk_atr=0.8 → min_stop = 104 - 0.8*2 = 102.4
    # final = min(94.5, 102.4) = 94.5 → structural used
    sl, mode = compute_structural_sl(
        entry_price=104.0, direction="LONG", bar_idx=9,
        last_pivot_low=last_low, last_pivot_high=last_high,
        atr=2.0, stop_mode="STRUCTURAL", buffer_atr=0.25, min_risk_atr=0.8,
    )
    print(f"    STRUCTURAL LONG: sl={sl:.3f} mode={mode} (expect 94.5, structural)")
    ok1 = check("STRUCTURAL sl = 94.5", abs(sl - 94.5) < 1e-9)
    ok2 = check("mode = structural", mode == "structural")

    # ATR mode: sl = 104 - 2.0*2.0 = 100.0
    sl2, m2 = compute_structural_sl(
        entry_price=104.0, direction="LONG", bar_idx=9,
        last_pivot_low=last_low, last_pivot_high=last_high,
        atr=2.0, stop_mode="ATR", atr_sl_mult=2.0,
    )
    print(f"    ATR LONG: sl={sl2:.3f} mode={m2} (expect 100.0, atr)")
    ok3 = check("ATR sl = 100.0", abs(sl2 - 100.0) < 1e-9)
    ok4 = check("mode = atr", m2 == "atr")

    # HYBRID mode: min(atr_stop=100, struct_stop=94.5) = 94.5
    sl3, m3 = compute_structural_sl(
        entry_price=104.0, direction="LONG", bar_idx=9,
        last_pivot_low=last_low, last_pivot_high=last_high,
        atr=2.0, stop_mode="HYBRID", atr_sl_mult=2.0, buffer_atr=0.25,
    )
    print(f"    HYBRID LONG: sl={sl3:.3f} mode={m3} (expect 94.5)")
    ok5 = check("HYBRID sl = 94.5 (tighter of two)", abs(sl3 - 94.5) < 1e-9)

    # min_risk clamp: if pivot is TOO CLOSE (pivot=103.9 → struct=103.4), min_stop=102.4, final=min(103.4,102.4)=102.4
    # Create synthetic last_low with close pivot
    fake_last = np.full(20, 103.9)
    fake_high = np.full(20, np.nan)
    sl4, m4 = compute_structural_sl(
        entry_price=104.0, direction="LONG", bar_idx=10,
        last_pivot_low=fake_last, last_pivot_high=fake_high,
        atr=2.0, stop_mode="STRUCTURAL", buffer_atr=0.25, min_risk_atr=0.8,
    )
    # struct = 103.9 - 0.5 = 103.4. min_stop = 104 - 1.6 = 102.4. final = min(103.4, 102.4) = 102.4
    print(f"    MIN_RISK_CLAMP LONG: sl={sl4:.3f} mode={m4} (expect 102.4, min_risk_clamp)")
    ok6 = check("clamped sl = 102.4", abs(sl4 - 102.4) < 1e-9)
    ok7 = check("mode = min_risk_clamp", m4 == "min_risk_clamp")

    # atr_fallback: no pivot
    nan_last = np.full(20, np.nan)
    sl5, m5 = compute_structural_sl(
        entry_price=104.0, direction="LONG", bar_idx=10,
        last_pivot_low=nan_last, last_pivot_high=nan_last,
        atr=2.0, stop_mode="STRUCTURAL", atr_sl_mult=2.0, buffer_atr=0.25, min_risk_atr=0.8,
    )
    # struct falls back to atr_stop=100.0. min_stop=102.4. final=min(100,102.4)=100.
    # pivot not available → mode="atr_fallback"
    print(f"    ATR_FALLBACK LONG: sl={sl5:.3f} mode={m5} (expect 100.0, atr_fallback)")
    ok8 = check("fallback sl = 100.0", abs(sl5 - 100.0) < 1e-9)
    ok9 = check("mode = atr_fallback", m5 == "atr_fallback")

    return all([ok1, ok2, ok3, ok4, ok5, ok6, ok7, ok8, ok9])


def test_structural_sl_short():
    high = np.array([100.0, 101, 102, 105, 103, 101, 99, 98, 97, 96, 95, 96, 97])
    low = np.array([95.0, 96, 97, 100, 98, 96, 94, 93, 92, 91, 90, 91, 92])
    pl = compute_pivot_lows(low, 3, 3)
    ph = compute_pivot_highs(high, 3, 3)
    last_low, last_high = build_last_pivot_arrays(pl, ph, right=3)

    # At bar 9, last pivot high = 105.0, atr=2.0, buffer=0.25 → struct=105+0.5=105.5
    # entry=96, min_stop=96+0.8*2=97.6
    # final = max(105.5, 97.6) = 105.5 → structural
    sl, mode = compute_structural_sl(
        entry_price=96.0, direction="SHORT", bar_idx=9,
        last_pivot_low=last_low, last_pivot_high=last_high,
        atr=2.0, stop_mode="STRUCTURAL", buffer_atr=0.25, min_risk_atr=0.8,
    )
    print(f"    STRUCTURAL SHORT: sl={sl:.3f} mode={mode} (expect 105.5, structural)")
    ok1 = check("SHORT struct sl = 105.5", abs(sl - 105.5) < 1e-9)
    ok2 = check("SHORT mode = structural", mode == "structural")
    return ok1 and ok2


if __name__ == "__main__":
    print("=" * 60)
    print("UNIT TESTS — validation/structural_stop.py")
    print("=" * 60)
    print("\n[1/4] Pivot detection")
    r1 = test_basic_pivots()
    print("\n[2/4] Lookahead enforcement")
    r2 = test_lookahead_arrays()
    print("\n[3/4] Structural SL LONG (all modes)")
    r3 = test_structural_sl_long()
    print("\n[4/4] Structural SL SHORT")
    r4 = test_structural_sl_short()

    print("\n" + "=" * 60)
    if all([r1, r2, r3, r4]):
        print("ALL TESTS PASSED")
    else:
        print("!! FAILURES DETECTED")
        sys.exit(1)
    print("=" * 60)
