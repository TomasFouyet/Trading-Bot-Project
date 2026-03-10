"""
Multi-timeframe macro trend filter.

Analyzes higher timeframe (HTF) bars using:
  - EMA 50  → medium-term trend direction
  - EMA 200 → long-term trend direction
  - ADX 14  → trend strength

Returns a trade coefficient [0.0, 1.0] based on how well the proposed
micro-timeframe trade direction aligns with the macro trend.

Coefficient matrix (direction × strength):

  Macro direction | ADX zone | LONG coeff | SHORT coeff
  ────────────────|──────────|────────────|────────────
  UP              | STRONG   |    1.00    |    0.00
  UP              | MODERATE |    0.85    |    0.20
  UP              | WEAK     |    0.70    |    0.45
  SIDEWAYS        | any      |    0.65    |    0.65
  DOWN            | WEAK     |    0.45    |    0.70
  DOWN            | MODERATE |    0.20    |    0.85
  DOWN            | STRONG   |    0.00    |    1.00

With default min_trend_coeff = 0.5 in the strategy:
  - Trades aligned with macro trend:        always allowed
  - Counter-trend in sideways market:       allowed (0.65 ≥ 0.5)
  - Counter-trend with weak macro trend:    blocked (0.45 < 0.5)
  - Counter-trend with strong macro trend:  blocked
"""
from __future__ import annotations

import numpy as np
import pandas as pd


class TrendContext:
    """
    Analyzes HTF bars and exposes a per-direction trade coefficient.

    Usage:
        ctx = TrendContext()
        ctx.update(htf_bars_df)           # call whenever new HTF bars arrive
        coeff = ctx.get_coefficient("LONG")  # or "SHORT"
        if coeff >= min_coeff:
            # proceed with entry
    """

    # (macro_direction, adx_zone) → (long_coeff, short_coeff)
    _COEFF_TABLE: dict[tuple[str, str], tuple[float, float]] = {
        ("UP",       "STRONG"):   (1.00, 0.00),
        ("UP",       "MODERATE"): (0.85, 0.20),
        ("UP",       "WEAK"):     (0.70, 0.45),
        ("SIDEWAYS", "STRONG"):   (0.65, 0.65),
        ("SIDEWAYS", "MODERATE"): (0.65, 0.65),
        ("SIDEWAYS", "WEAK"):     (0.65, 0.65),
        ("DOWN",     "WEAK"):     (0.45, 0.70),
        ("DOWN",     "MODERATE"): (0.20, 0.85),
        ("DOWN",     "STRONG"):   (0.00, 1.00),
    }

    def __init__(
        self,
        ema_fast: int = 50,
        ema_slow: int = 200,
        adx_period: int = 14,
    ) -> None:
        self._ema_fast = ema_fast
        self._ema_slow = ema_slow
        self._adx_period = adx_period

        # Latest computed state (public for meta/logging)
        self.direction: str = "SIDEWAYS"
        self.adx_zone: str = "WEAK"
        self.adx: float = 0.0
        self.ema_fast_val: float = 0.0
        self.ema_slow_val: float = 0.0
        self._initialized: bool = False

    # ── Public API ────────────────────────────────────────────────────────────

    def update(self, bars: pd.DataFrame) -> None:
        """
        Recompute trend direction and strength from HTF bars.
        bars must have columns: close, high, low (numeric).
        Requires at least ema_slow + adx_period bars to produce a result.
        """
        min_bars = self._ema_slow + self._adx_period * 2 + 5
        if len(bars) < min_bars:
            self._initialized = False
            return

        closes = bars["close"].astype(float)
        highs = bars["high"].astype(float)
        lows = bars["low"].astype(float)

        # ── EMA fast & slow ───────────────────────────────────────────────
        ema_fast_s = closes.ewm(span=self._ema_fast, adjust=False).mean()
        ema_slow_s = closes.ewm(span=self._ema_slow, adjust=False).mean()

        last_close = float(closes.iloc[-1])
        last_ema_fast = float(ema_fast_s.iloc[-1])
        last_ema_slow = float(ema_slow_s.iloc[-1])

        self.ema_fast_val = last_ema_fast
        self.ema_slow_val = last_ema_slow

        # Trend direction: price AND ema_fast must both be on the same side of ema_slow
        if last_close > last_ema_slow and last_ema_fast > last_ema_slow:
            self.direction = "UP"
        elif last_close < last_ema_slow and last_ema_fast < last_ema_slow:
            self.direction = "DOWN"
        else:
            self.direction = "SIDEWAYS"

        # ── ADX (Wilder) ──────────────────────────────────────────────────
        adx_val = self._compute_adx(highs, lows, closes)
        self.adx = adx_val

        if adx_val >= 25.0:
            self.adx_zone = "STRONG"
        elif adx_val >= 15.0:
            self.adx_zone = "MODERATE"
        else:
            self.adx_zone = "WEAK"

        self._initialized = True

    def get_coefficient(self, direction: str) -> float:
        """
        Return the trade coefficient for a given micro direction ("LONG" or "SHORT").
        Returns 1.0 (no filter) when there is not enough HTF data yet.
        """
        if not self._initialized:
            return 1.0

        key = (self.direction, self.adx_zone)
        long_c, short_c = self._COEFF_TABLE.get(key, (0.65, 0.65))
        return long_c if direction == "LONG" else short_c

    def to_meta(self) -> dict:
        """Return current trend state as a metadata dict for signal logging."""
        return {
            "htf_direction": self.direction,
            "htf_adx":       round(self.adx, 2),
            "htf_adx_zone":  self.adx_zone,
            "htf_ema_fast":  round(self.ema_fast_val, 4),
            "htf_ema_slow":  round(self.ema_slow_val, 4),
        }

    # ── ADX calculation ───────────────────────────────────────────────────────

    def _compute_adx(
        self,
        highs: pd.Series,
        lows: pd.Series,
        closes: pd.Series,
    ) -> float:
        """Wilder's ADX using self._adx_period."""
        period = self._adx_period
        h = highs.values.astype(float)
        lo = lows.values.astype(float)
        c = closes.values.astype(float)
        n = len(c)

        if n < period * 2 + 2:
            return 0.0

        tr = np.zeros(n)
        plus_dm = np.zeros(n)
        minus_dm = np.zeros(n)

        for i in range(1, n):
            hl = h[i] - lo[i]
            hc = abs(h[i] - c[i - 1])
            lc = abs(lo[i] - c[i - 1])
            tr[i] = max(hl, hc, lc)

            up_move = h[i] - h[i - 1]
            down_move = lo[i - 1] - lo[i]
            plus_dm[i] = up_move if (up_move > down_move and up_move > 0) else 0.0
            minus_dm[i] = down_move if (down_move > up_move and down_move > 0) else 0.0

        # Wilder's initial sums (skip index 0 which is 0)
        s_tr = tr[1:].astype(float)
        s_pdm = plus_dm[1:].astype(float)
        s_mdm = minus_dm[1:].astype(float)
        m = len(s_tr)

        if m < period:
            return 0.0

        atr_arr = np.zeros(m)
        pdm_arr = np.zeros(m)
        mdm_arr = np.zeros(m)

        # Seed: first Wilder sum = simple sum of first `period` bars
        atr_arr[period - 1] = s_tr[:period].sum()
        pdm_arr[period - 1] = s_pdm[:period].sum()
        mdm_arr[period - 1] = s_mdm[:period].sum()

        for i in range(period, m):
            atr_arr[i] = atr_arr[i - 1] - atr_arr[i - 1] / period + s_tr[i]
            pdm_arr[i] = pdm_arr[i - 1] - pdm_arr[i - 1] / period + s_pdm[i]
            mdm_arr[i] = mdm_arr[i - 1] - mdm_arr[i - 1] / period + s_mdm[i]

        with np.errstate(divide="ignore", invalid="ignore"):
            plus_di  = np.where(atr_arr > 0, 100.0 * pdm_arr / atr_arr, 0.0)
            minus_di = np.where(atr_arr > 0, 100.0 * mdm_arr / atr_arr, 0.0)
            di_sum = plus_di + minus_di
            dx = np.where(di_sum > 0, 100.0 * np.abs(plus_di - minus_di) / di_sum, 0.0)

        # ADX = Wilder's EMA of DX, seeded at index (2*period - 2)
        adx_arr = np.zeros(m)
        seed_idx = 2 * period - 2
        if seed_idx >= m:
            return 0.0

        # Seed ADX as simple mean of first DX window
        adx_arr[seed_idx] = dx[period - 1 : seed_idx + 1].mean()

        for i in range(seed_idx + 1, m):
            adx_arr[i] = (adx_arr[i - 1] * (period - 1) + dx[i]) / period

        return float(adx_arr[-1])
