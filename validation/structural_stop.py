"""
Structural stop module — pivot-based SL with ATR fallback and min-risk clamp.
Mirrors Pine Script "TrendV2Simple + HTF + Structural Stops" exactly.

No lookahead: pivots at bar i confirmed only at bar i+right.
build_last_pivot_arrays enforces this via (i - right) window.
"""
from __future__ import annotations

import numpy as np


def compute_pivot_lows(low: np.ndarray, left: int = 3, right: int = 3) -> np.ndarray:
    """Confirmed pivot lows. pivot_lows[i] = low[i] if strict minimum over window."""
    n = len(low)
    result = np.full(n, np.nan)
    for i in range(left, n - right):
        v = low[i]
        is_pivot = True
        for j in range(1, left + 1):
            if v >= low[i - j]:
                is_pivot = False
                break
        if is_pivot:
            for j in range(1, right + 1):
                if v >= low[i + j]:
                    is_pivot = False
                    break
        if is_pivot:
            result[i] = v
    return result


def compute_pivot_highs(high: np.ndarray, left: int = 3, right: int = 3) -> np.ndarray:
    """Confirmed pivot highs. Symmetric to compute_pivot_lows."""
    n = len(high)
    result = np.full(n, np.nan)
    for i in range(left, n - right):
        v = high[i]
        is_pivot = True
        for j in range(1, left + 1):
            if v <= high[i - j]:
                is_pivot = False
                break
        if is_pivot:
            for j in range(1, right + 1):
                if v <= high[i + j]:
                    is_pivot = False
                    break
        if is_pivot:
            result[i] = v
    return result


def build_last_pivot_arrays(
    pivot_lows: np.ndarray,
    pivot_highs: np.ndarray,
    right: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """
    last_pivot_low[i]  = most recent pivot_low confirmed at bar i-right or earlier.
    last_pivot_high[i] = same for highs.

    Enforces no-lookahead: at runtime bar i, pivot at bar k is only visible if
    k + right <= i (i.e. k <= i - right).
    """
    n = len(pivot_lows)
    last_low = np.full(n, np.nan)
    last_high = np.full(n, np.nan)
    running_low = np.nan
    running_high = np.nan
    for i in range(n):
        confirmed_bar = i - right
        if confirmed_bar >= 0:
            if not np.isnan(pivot_lows[confirmed_bar]):
                running_low = pivot_lows[confirmed_bar]
            if not np.isnan(pivot_highs[confirmed_bar]):
                running_high = pivot_highs[confirmed_bar]
        last_low[i] = running_low
        last_high[i] = running_high
    return last_low, last_high


def compute_structural_sl(
    entry_price: float,
    direction: str,
    bar_idx: int,
    last_pivot_low: np.ndarray,
    last_pivot_high: np.ndarray,
    atr: float,
    stop_mode: str = "STRUCTURAL",
    atr_sl_mult: float = 2.0,
    buffer_atr: float = 0.25,
    min_risk_atr: float = 0.8,
) -> tuple[float, str]:
    """Return (sl_price, mode_label). mode_label ∈ {atr, structural, atr_fallback, min_risk_clamp}."""
    if direction == "LONG":
        atr_stop = entry_price - atr * atr_sl_mult
        pivot = last_pivot_low[bar_idx] if bar_idx < len(last_pivot_low) else np.nan
        pivot_available = not np.isnan(pivot)
        struct_stop = (pivot - atr * buffer_atr) if pivot_available else atr_stop
        hybrid_stop = min(atr_stop, struct_stop)

        if stop_mode == "ATR":
            raw_stop = atr_stop
        elif stop_mode == "STRUCTURAL":
            raw_stop = struct_stop
        else:
            raw_stop = hybrid_stop

        min_stop = entry_price - atr * min_risk_atr
        final_stop = min(raw_stop, min_stop)

        if stop_mode == "ATR":
            mode = "atr"
        elif not pivot_available:
            mode = "atr_fallback"
        elif final_stop == min_stop and raw_stop > min_stop:
            mode = "min_risk_clamp"
        else:
            mode = "structural"
        return final_stop, mode

    else:  # SHORT
        atr_stop = entry_price + atr * atr_sl_mult
        pivot = last_pivot_high[bar_idx] if bar_idx < len(last_pivot_high) else np.nan
        pivot_available = not np.isnan(pivot)
        struct_stop = (pivot + atr * buffer_atr) if pivot_available else atr_stop
        hybrid_stop = max(atr_stop, struct_stop)

        if stop_mode == "ATR":
            raw_stop = atr_stop
        elif stop_mode == "STRUCTURAL":
            raw_stop = struct_stop
        else:
            raw_stop = hybrid_stop

        min_stop = entry_price + atr * min_risk_atr
        final_stop = max(raw_stop, min_stop)

        if stop_mode == "ATR":
            mode = "atr"
        elif not pivot_available:
            mode = "atr_fallback"
        elif final_stop == min_stop and raw_stop < min_stop:
            mode = "min_risk_clamp"
        else:
            mode = "structural"
        return final_stop, mode
