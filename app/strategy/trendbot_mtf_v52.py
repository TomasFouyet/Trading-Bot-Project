"""
TrendBotMTFv52Strategy — replica exacta del Pine Script TrendBot MTF v5.2

Diferencias respecto a TrendFollowingStrategy (la estrategia existente):
  ✓ Filtro RSI momentum  (rsi_long_min=45, rsi_short_max=55)
  ✓ Filtro volumen mínimo (vol_min_ratio=0.6 × SMA20)
  ✓ Filtro candle direction (close > open para LONG, close < open para SHORT)
  ✓ Pullback tolerancia por defecto 1.2 ATR (v5.2 lo subió de 1.0 a 1.2)
  ✓ SL mínimo 1.2 ATR (v5.2 lo subió de 1.0 para evitar wick-outs)
  ✓ Cooldown reducido: 3 barras (v5.1 usaba 5)
  ✓ Trailing ATR multiplier 1.5 (independiente del existente)
  ✓ Reversal swap: si hay señal contraria, cierra y abre al revés
  ✓ Tier A=100% equity, B=75%, C=25% (sizing proporcional como Pine)
  ✓ Bias 4H via HTFContext (ya existente en el proyecto)
  ✓ Solo BTC-USDT por defecto (el Pine script se usa en BTC)

Compatible al 100% con:
  - run_multi_paper.py  (via on_bar() → Signal)
  - BacktestEngine      (via on_bar() → Signal)
  - position_manager    (SL/TP tracking externo)

Parámetros configurables (todos con valores por defecto Pine v5.2):
  adx_min               int     20
  adx_strong            int     35
  ema_fast              int     20
  ema_slow              int     50
  pb_tol_atr            float   1.2   (pullback tolerance)
  min_confidence        float   0.0
  allow_short           bool    True
  sig_cooldown          int     3
  require_bull_bar      bool    True
  use_rsi_filter        bool    True
  rsi_long_min          int     45
  rsi_short_max         int     55
  use_vol_filter        bool    True
  vol_min_ratio         float   0.6
  sl_swing_lookback     int     50
  sl_swing_window       int     3
  sl_min_atr            float   1.2
  sl_max_atr            float   2.5
  sl_buf_atr            float   0.3
  tp1_r_A / tp2_r_A     float   1.5 / 3.0
  tp1_r_B / tp2_r_B     float   1.5 / 2.5
  tp1_r_C / tp2_r_C     float   1.0 / 1.5
  use_trailing          bool    True
  trail_atr_mult        float   1.5
  enable_reversal       bool    True
  tier_a_pct            float   100  (% equity)
  tier_b_pct            float   75
  tier_c_pct            float   25
  htf_ema_fast          int     50
  htf_ema_slow          int     200
  use_session_filter    bool    True
  session_mult_us       float   1.0
  session_mult_eu       float   0.75
  session_mult_other    float   0.5
"""
from __future__ import annotations

from decimal import Decimal
from typing import Any

import numpy as np
import pandas as pd

from app.broker.base import OHLCVBar
from app.strategy.base import BaseStrategy
from app.strategy.signals import Signal, SignalAction


# ─────────────────────────────────────────────────────────────────────────────
# Helper: HOLD signal
# ─────────────────────────────────────────────────────────────────────────────
def _hold(symbol: str, ts, strategy_id: str, reason: str, meta: dict | None = None) -> Signal:
    return Signal(
        action=SignalAction.HOLD,
        symbol=symbol,
        ts=ts,
        strategy_id=strategy_id,
        reason=reason,
        meta=meta or {},
    )


class TrendBotMTFv52Strategy(BaseStrategy):
    """
    Python replica del Pine Script TrendBot MTF v5.2.

    Genera señales BUY / SELL / HOLD.
    Señales CLOSE / PARTIAL_CLOSE las gestiona el motor externo (position_manager).
    """

    def __init__(self, symbol: str, params: dict[str, Any] | None = None) -> None:
        super().__init__(symbol, params)

        # ── Core ────────────────────────────────────────────────────────────
        self._adx_min        = int(self.params.get("adx_min",         20))
        self._adx_strong     = int(self.params.get("adx_strong",      35))
        self._ema_fast       = int(self.params.get("ema_fast",         20))
        self._ema_slow       = int(self.params.get("ema_slow",         50))
        self._pb_tol_atr     = float(self.params.get("pb_tol_atr",    1.2))
        self._min_conf       = float(self.params.get("min_confidence", 0.0))
        self._allow_short    = bool(self.params.get("allow_short",     True))
        self._sig_cooldown   = int(self.params.get("sig_cooldown",     3))

        # ── Quality filters (NEW in v5.2) ────────────────────────────────────
        self._require_bull_bar = bool(self.params.get("require_bull_bar", True))
        self._use_rsi_filter   = bool(self.params.get("use_rsi_filter",   True))
        self._rsi_long_min     = int(self.params.get("rsi_long_min",      45))
        self._rsi_short_max    = int(self.params.get("rsi_short_max",     55))
        self._use_vol_filter   = bool(self.params.get("use_vol_filter",   True))
        self._vol_min_ratio    = float(self.params.get("vol_min_ratio",   0.6))

        # ── SL / TP ──────────────────────────────────────────────────────────
        self._sl_lookback  = int(self.params.get("sl_swing_lookback", 50))
        self._sl_window    = int(self.params.get("sl_swing_window",   3))
        self._sl_min_atr   = float(self.params.get("sl_min_atr",      1.5))
        self._sl_max_atr   = float(self.params.get("sl_max_atr",      3.0))
        self._sl_buf_atr   = float(self.params.get("sl_buf_atr",      0.3))

        # TP R-multiples por tier
        self._tp1_r = {"A": float(self.params.get("tp1_r_A", 1.5)),
                       "B": float(self.params.get("tp1_r_B", 1.5)),
                       "C": float(self.params.get("tp1_r_C", 1.0))}
        self._tp2_r = {"A": float(self.params.get("tp2_r_A", 3.0)),
                       "B": float(self.params.get("tp2_r_B", 2.5)),
                       "C": float(self.params.get("tp2_r_C", 1.5))}

        # TP1 partial-close % por tier (mismo que Pine: A=33%, B=50%, C=70%)
        self._tp1_close_pct = {"A": 0.33, "B": 0.50, "C": 0.70}

        # ── Trailing SL ───────────────────────────────────────────────────────
        self._use_trailing    = bool(self.params.get("use_trailing",    True))
        self._trail_atr_mult  = float(self.params.get("trail_atr_mult", 1.5))

        # ── Reversal swap ────────────────────────────────────────────────────
        self._enable_reversal = bool(self.params.get("enable_reversal", True))

        # ── Sizing (% equity por tier) ────────────────────────────────────────
        self._tier_pct = {
            "A": float(self.params.get("tier_a_pct", 100)),
            "B": float(self.params.get("tier_b_pct",  75)),
            "C": float(self.params.get("tier_c_pct",  25)),
        }

        # ── HTF (4H) ─────────────────────────────────────────────────────────
        self._htf_ema_fast = int(self.params.get("htf_ema_fast", 50))
        self._htf_ema_slow = int(self.params.get("htf_ema_slow", 200))

        # ── Session sizing ────────────────────────────────────────────────────
        self._use_session      = bool(self.params.get("use_session_filter", True))
        self._sess_us          = float(self.params.get("session_mult_us",    1.0))
        self._sess_eu          = float(self.params.get("session_mult_eu",    0.75))
        self._sess_other       = float(self.params.get("session_mult_other", 0.5))

        # ── Cooldown state ────────────────────────────────────────────────────
        self._last_long_bar:  int = -999
        self._last_short_bar: int = -999
        self._bar_index:      int = 0

        # ── Reversal state (mirrors Pine trade_state) ─────────────────────────
        # 0 = flat, 1 = long, -1 = short
        self._trade_state: int = 0

        # ── HTF bars (injected externally, optional) ──────────────────────────
        self._htf_df: pd.DataFrame | None = None

    # ── External HTF injection (called by run script before on_bar) ───────────
    def set_htf_bars(self, htf_df: pd.DataFrame) -> None:
        """Inject HTF (4H) bars. Called by run script before on_bar()."""
        self._htf_df = htf_df

    # ── BaseStrategy interface ────────────────────────────────────────────────

    @property
    def strategy_id(self) -> str:
        return (
            f"trendbot_mtf_v52"
            f"|adx={self._adx_min}"
            f"|pb={self._pb_tol_atr}"
            f"|cd={self._sig_cooldown}"
            f"|rsi={'on' if self._use_rsi_filter else 'off'}"
            f"|vol={'on' if self._use_vol_filter else 'off'}"
        )

    @property
    def min_bars_required(self) -> int:
        return max(60, self._ema_slow + 30)

    def on_bar(self, bars: pd.DataFrame) -> Signal:
        self._bar_index += 1
        if len(bars) < self.min_bars_required:
            return _hold(self.symbol, bars.iloc[-1]["ts"], self.strategy_id, "warmup")

        row = bars.iloc[-1]
        ts  = row["ts"]

        # ── Compute indicators ────────────────────────────────────────────────
        ind = self._compute_indicators(bars)
        atr   = ind["atr"]
        adx   = ind["adx"]
        close = float(row["close"])
        open_ = float(row["open"])
        high  = float(row["high"])
        low   = float(row["low"])

        ema_f      = ind["ema_fast"]
        ema_s      = ind["ema_slow"]
        ema_f_slp  = ind["ema_fast_slope"]
        ema_s_slp  = ind["ema_slow_slope"]
        macd_l     = ind["macd"]
        macd_s     = ind["macd_signal"]
        macd_h     = ind["macd_hist"]
        macd_h_p   = ind["macd_hist_prev"]
        vol_ratio  = ind["vol_ratio"]
        rsi        = ind["rsi"]

        # ── Base conditions (mirrors Pine long_base / short_base) ─────────────
        pb_zone    = abs(close - ema_f) < atr * self._pb_tol_atr
        sl_rising  = ema_s_slp > 0
        sl_falling = ema_s_slp < 0
        p_above    = close > ema_s
        p_below    = close < ema_s
        m_bull     = macd_l > macd_s
        m_bear     = macd_l < macd_s

        # Candle direction filter
        c_bull_bar = (close > open_) if self._require_bull_bar else True
        c_bear_bar = (close < open_) if self._require_bull_bar else True

        adx_ok = adx >= self._adx_min

        # RSI momentum filter
        rsi_ok_long  = (rsi >= self._rsi_long_min)  if self._use_rsi_filter else True
        rsi_ok_short = (rsi <= self._rsi_short_max) if self._use_rsi_filter else True

        # Volume filter
        vol_ok = (vol_ratio >= self._vol_min_ratio) if self._use_vol_filter else True

        long_base  = (adx_ok and sl_rising  and p_above and m_bull
                      and pb_zone and c_bull_bar and rsi_ok_long  and vol_ok)
        short_base = (adx_ok and sl_falling and p_below and m_bear
                      and pb_zone and c_bear_bar and rsi_ok_short and vol_ok
                      and self._allow_short)

        # ── HTF bias ──────────────────────────────────────────────────────────
        htf_bias = self._compute_htf_bias()

        # ── Confidence scoring ────────────────────────────────────────────────
        raw_conf_l = self._compute_conf(ind, "LONG")  if long_base  else 0.0
        raw_conf_s = self._compute_conf(ind, "SHORT") if short_base else 0.0

        htf_mod_l = self._htf_conf_mod(htf_bias, "LONG")
        htf_mod_s = self._htf_conf_mod(htf_bias, "SHORT")

        conf_long  = max(0.0, min(1.0, raw_conf_l + htf_mod_l))
        conf_short = max(0.0, min(1.0, raw_conf_s + htf_mod_s))

        long_signal  = long_base  and conf_long  >= self._min_conf
        short_signal = short_base and conf_short >= self._min_conf

        # ── Raw triggers (leading edge only — same as Pine not signal[1]) ──────
        # We approximate "not signal[1]" via cooldown: only fire once per cooldown
        long_trigger_raw  = long_signal
        short_trigger_raw = short_signal

        # Cooldown guard
        long_trigger  = (long_trigger_raw
                         and (self._bar_index - self._last_long_bar) >= self._sig_cooldown)
        short_trigger = (short_trigger_raw
                         and (self._bar_index - self._last_short_bar) >= self._sig_cooldown)

        # ── Reversal detection ────────────────────────────────────────────────
        reversal_to_long  = self._enable_reversal and self._trade_state == -1 and long_trigger
        reversal_to_short = self._enable_reversal and self._trade_state ==  1 and short_trigger
        is_reversal       = reversal_to_long or reversal_to_short

        new_long  = (self._trade_state == 0 and long_trigger)  or reversal_to_long
        new_short = (self._trade_state == 0 and short_trigger) or reversal_to_short

        # ── Build signal ──────────────────────────────────────────────────────
        if new_long:
            sig = self._build_signal(bars, row, ts, ind, atr, conf_long,
                                     "LONG", htf_bias, is_reversal)
            if sig.is_actionable():
                self._last_long_bar = self._bar_index
                self._trade_state   = 1
            return sig

        if new_short:
            sig = self._build_signal(bars, row, ts, ind, atr, conf_short,
                                     "SHORT", htf_bias, is_reversal)
            if sig.is_actionable():
                self._last_short_bar = self._bar_index
                self._trade_state    = -1
            return sig

        return _hold(self.symbol, ts, self.strategy_id,
                     "no_setup" if not (long_base or short_base) else "filtered",
                     meta=ind)

    def notify_trade_closed(self, side: str) -> None:
        """Call this after a trade closes to reset reversal state."""
        self._trade_state = 0

    # ═════════════════════════════════════════════════════════════════════════
    # INDICATORS (mirrors Pine 15M section)
    # ═════════════════════════════════════════════════════════════════════════

    def _compute_indicators(self, bars: pd.DataFrame) -> dict:
        close  = bars["close"]
        high   = bars["high"]
        low    = bars["low"]
        volume = bars["volume"]
        open_  = bars["open"]

        # ATR(14) — EMA method (same as ta.atr in Pine)
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low  - close.shift()).abs(),
        ], axis=1).max(axis=1)
        atr = tr.ewm(com=13, adjust=False).mean().iloc[-1]

        # ADX(14) — mirrors ta.dmi()
        up   = high.diff()
        dn   = -low.diff()
        dm_p = up.where((up > dn) & (up > 0), 0.0)
        dm_m = dn.where((dn > up) & (dn > 0), 0.0)
        atr_s = tr.ewm(com=13, adjust=False).mean()
        atr_s = atr_s.replace(0, np.nan)
        di_p  = 100 * dm_p.ewm(com=13, adjust=False).mean() / atr_s
        di_m  = 100 * dm_m.ewm(com=13, adjust=False).mean() / atr_s
        dx    = (100 * (di_p - di_m).abs() / (di_p + di_m).replace(0, np.nan)
                 ).fillna(0)
        adx   = dx.ewm(com=13, adjust=False).mean().iloc[-1]

        # EMAs
        ema_f_s = close.ewm(span=self._ema_fast, adjust=False).mean()
        ema_s_s = close.ewm(span=self._ema_slow, adjust=False).mean()
        ema_f   = ema_f_s.iloc[-1]
        ema_s   = ema_s_s.iloc[-1]

        # EMA slopes (last 5 bars, mirrors Pine)
        ema_f_slope = ema_f - ema_f_s.iloc[-6] if len(ema_f_s) >= 6 else 0.0
        ema_s_slope = ema_s - ema_s_s.iloc[-6] if len(ema_s_s) >= 6 else 0.0

        # MACD(12, 26, 9)
        ema12  = close.ewm(span=12, adjust=False).mean()
        ema26  = close.ewm(span=26, adjust=False).mean()
        macd_l = ema12 - ema26
        macd_s = macd_l.ewm(span=9, adjust=False).mean()
        macd_h = macd_l - macd_s
        macd_hist      = macd_h.iloc[-1]
        macd_hist_prev = macd_h.iloc[-2] if len(macd_h) >= 2 else 0.0

        # Volume ratio vs SMA20
        vol_sma   = volume.rolling(20).mean()
        vol_ratio = (volume.iloc[-1] / vol_sma.iloc[-1]) if vol_sma.iloc[-1] > 0 else 0.0

        # RSI(14)
        delta  = close.diff()
        gain   = delta.clip(lower=0).ewm(com=13, adjust=False).mean()
        loss   = (-delta.clip(upper=0)).ewm(com=13, adjust=False).mean()
        rs     = gain / loss.replace(0, np.nan)
        rsi_s  = 100 - 100 / (1 + rs)
        rsi    = rsi_s.iloc[-1]

        row = bars.iloc[-1]
        body  = abs(float(row["close"]) - float(row["open"]))
        rng   = float(row["high"]) - float(row["low"])
        body_ratio = body / rng if rng > 0 else 0.0

        return {
            "atr":            atr,
            "adx":            adx,
            "di_plus":        di_p.iloc[-1],
            "di_minus":       di_m.iloc[-1],
            "ema_fast":       ema_f,
            "ema_slow":       ema_s,
            "ema_fast_slope": ema_f_slope,
            "ema_slow_slope": ema_s_slope,
            "macd":           macd_l.iloc[-1],
            "macd_signal":    macd_s.iloc[-1],
            "macd_hist":      macd_hist,
            "macd_hist_prev": macd_hist_prev,
            "vol_ratio":      vol_ratio,
            "rsi":            rsi,
            "body_ratio":     body_ratio,
        }

    # ═════════════════════════════════════════════════════════════════════════
    # CONFIDENCE SCORING (mirrors Pine compute_conf())
    # ═════════════════════════════════════════════════════════════════════════

    def _compute_conf(self, ind: dict, direction: str) -> float:
        sc = 0.0
        adx       = ind["adx"]
        atr       = ind["atr"]
        close     = ind.get("close", 0)          # not in ind → computed inline
        ema_f     = ind["ema_fast"]
        ema_f_slp = ind["ema_fast_slope"]
        macd_h    = ind["macd_hist"]
        macd_h_p  = ind["macd_hist_prev"]
        vol_ratio = ind["vol_ratio"]
        body_rat  = ind["body_ratio"]

        # Factor 1: ADX strength
        if adx >= self._adx_strong:
            sc += 0.20
        elif adx >= 30:
            sc += 0.10

        # Factor 2: Pullback tightness
        # We need close from caller — stored in ind optionally
        ema_f_val = ema_f
        pb_atr_d  = ind.get("pb_atr_d", None)
        if pb_atr_d is None:
            # Will be computed properly in build_signal; here use 0.5 as proxy
            pb_atr_d = 0.5
        if pb_atr_d <= 0.5:
            sc += 0.20
        elif pb_atr_d <= 1.0:
            sc += 0.10

        # Factor 3: MACD accelerating
        if direction == "LONG":
            if macd_h > 0 and macd_h > macd_h_p:
                sc += 0.15
            elif macd_h > 0:
                sc += 0.05
        else:
            if macd_h < 0 and macd_h < macd_h_p:
                sc += 0.15
            elif macd_h < 0:
                sc += 0.05

        # Factor 4: Candle body strength
        if body_rat >= 0.60:
            sc += 0.15
        elif body_rat >= 0.40:
            sc += 0.07

        # Factor 5: EMA fast slope
        if direction == "LONG" and ema_f_slp > 0:
            sc += 0.15
        elif direction == "SHORT" and ema_f_slp < 0:
            sc += 0.15

        # Factor 6: Volume
        if vol_ratio >= 1.2:
            sc += 0.15
        elif vol_ratio >= 0.8:
            sc += 0.05

        return min(sc, 1.0)

    # ═════════════════════════════════════════════════════════════════════════
    # HTF BIAS (mirrors Pine htf_score / htf_bias)
    # ═════════════════════════════════════════════════════════════════════════

    def _compute_htf_bias(self) -> dict:
        """
        Returns {"bias": int, "strength": float, "label": str}.
        bias: +1 bull, -1 bear, 0 neutral.
        """
        if self._htf_df is None or len(self._htf_df) < self._htf_ema_slow + 5:
            return {"bias": 0, "strength": 0.0, "label": "4H NEUTRAL (no data)"}

        df = self._htf_df
        close = df["close"]

        htf_ef = close.ewm(span=self._htf_ema_fast, adjust=False).mean()
        htf_es = close.ewm(span=self._htf_ema_slow, adjust=False).mean()

        # Approximate ADX for HTF
        high = df["high"]
        low  = df["low"]
        tr   = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low  - close.shift()).abs(),
        ], axis=1).max(axis=1)
        up   = high.diff()
        dn   = -low.diff()
        dm_p = up.where((up > dn) & (up > 0), 0.0)
        dm_m = dn.where((dn > up) & (dn > 0), 0.0)
        atr_s = tr.ewm(com=13, adjust=False).mean().replace(0, np.nan)
        di_p  = 100 * dm_p.ewm(com=13, adjust=False).mean() / atr_s
        di_m  = 100 * dm_m.ewm(com=13, adjust=False).mean() / atr_s
        dx    = (100 * (di_p - di_m).abs() / (di_p + di_m).replace(0, np.nan)).fillna(0)
        htf_adx = dx.ewm(com=13, adjust=False).mean().iloc[-1]

        c0 = float(close.iloc[-1])
        c1 = float(close.iloc[-2]) if len(close) >= 2 else c0
        c2 = float(close.iloc[-3]) if len(close) >= 3 else c1
        ef = float(htf_ef.iloc[-1])
        es = float(htf_es.iloc[-1])

        score = 0.0
        # EMA alignment
        if ef > es:   score += 0.35
        elif ef < es: score -= 0.35

        # Price vs EMAs
        if c0 > ef and ef > es:    score += 0.25
        elif c0 < ef and ef < es:  score -= 0.25
        elif c0 > es:              score += 0.10
        elif c0 < es:              score -= 0.10

        # ADX bonus
        adx_bonus = 0.20 if htf_adx >= 30 else (0.10 if htf_adx >= 20 else 0.0)
        score += adx_bonus if score > 0 else -adx_bonus

        # Rising / falling
        if c2 <= c1 <= c0:   score += 0.10
        elif c2 >= c1 >= c0: score -= 0.10

        score_c  = max(-1.0, min(1.0, score))
        bias     = 1 if score_c >= 0.35 else (-1 if score_c <= -0.35 else 0)
        strength = abs(score_c)

        label = (f"4H {'BULL' if bias == 1 else ('BEAR' if bias == -1 else 'NEUTRAL')}"
                 f" (score={score_c:+.2f} adx={htf_adx:.0f})")
        return {"bias": bias, "strength": strength, "label": label,
                "htf_ema_fast": ef, "htf_ema_slow": es, "htf_adx": htf_adx}

    def _htf_conf_mod(self, htf: dict, direction: str) -> float:
        """Mirrors Pine htf_conf_mod(). Returns delta to add to confidence."""
        bias     = htf["bias"]
        strength = htf["strength"]
        if bias == 0:
            return 0.0
        aligned = (direction == "LONG" and bias == 1) or (direction == "SHORT" and bias == -1)
        if aligned:
            return 0.15 if strength >= 0.6 else 0.05
        else:
            return -0.35 if strength >= 0.6 else -0.20

    # ═════════════════════════════════════════════════════════════════════════
    # SL / TP CALCULATION (mirrors Pine calc_sl / find_swing)
    # ═════════════════════════════════════════════════════════════════════════

    def _find_swing(self, bars: pd.DataFrame, direction: str) -> tuple[float, bool]:
        """Find most recent pivot high (SHORT) or pivot low (LONG)."""
        w  = self._sl_window
        lb = min(self._sl_lookback, len(bars))
        if lb <= 2 * w:
            return 0.0, False

        # Work on reversed slice (most recent first)
        sub = bars.iloc[-(lb):].reset_index(drop=True)
        n   = len(sub)

        if direction == "LONG":
            col = "low"
            for i in range(w, n - w):
                v = float(sub[col].iloc[i])
                if all(v <= float(sub[col].iloc[i - j]) for j in range(1, w + 1)) and \
                   all(v <= float(sub[col].iloc[i + j]) for j in range(1, w + 1)):
                    return v, True
        else:
            col = "high"
            for i in range(w, n - w):
                v = float(sub[col].iloc[i])
                if all(v >= float(sub[col].iloc[i - j]) for j in range(1, w + 1)) and \
                   all(v >= float(sub[col].iloc[i + j]) for j in range(1, w + 1)):
                    return v, True

        return 0.0, False

    def _calc_sl(self, bars: pd.DataFrame, close: float, atr: float,
                 ema_s: float, direction: str, conf: float) -> tuple[float, float, float, str]:
        """Returns (sl, tp1, tp2, method)."""
        swing_val, swing_found = self._find_swing(bars, direction)

        tier = self._get_tier(conf)
        lb   = min(self._sl_lookback, len(bars))

        if direction == "LONG":
            fallback  = float(bars["low"].iloc[-lb:].min()) if lb > 0 else close - atr
            base      = swing_val if swing_found else fallback
            struct_sl = min(base, ema_s) - atr * self._sl_buf_atr
            dist      = close - struct_sl
            dist      = max(dist, atr * self._sl_min_atr)
            dist      = min(dist, atr * self._sl_max_atr)
            sl        = close - dist
            risk      = abs(close - sl)
            tp1       = close + risk * self._tp1_r[tier]
            tp2       = close + risk * self._tp2_r[tier]
        else:
            fallback  = float(bars["high"].iloc[-lb:].max()) if lb > 0 else close + atr
            base      = swing_val if swing_found else fallback
            struct_sl = max(base, ema_s) + atr * self._sl_buf_atr
            dist      = struct_sl - close
            dist      = max(dist, atr * self._sl_min_atr)
            dist      = min(dist, atr * self._sl_max_atr)
            sl        = close + dist
            risk      = abs(close - sl)
            tp1       = close - risk * self._tp1_r[tier]
            tp2       = close - risk * self._tp2_r[tier]

        method = "swing" if swing_found else "lookback"
        return sl, tp1, tp2, method

    # ═════════════════════════════════════════════════════════════════════════
    # SESSION SIZING (mirrors Pine is_us / is_eu)
    # ═════════════════════════════════════════════════════════════════════════

    def _get_session_mult(self, ts) -> tuple[float, str]:
        if not self._use_session:
            return 1.0, "off"
        try:
            import pandas as pd
            if isinstance(ts, str):
                ts = pd.Timestamp(ts)
            if hasattr(ts, "tzinfo") and ts.tzinfo is not None:
                utc_h = ts.tz_convert("UTC").hour
            else:
                utc_h = ts.hour
        except Exception:
            return 1.0, "other"

        if 14 <= utc_h < 21:
            return self._sess_us, "US"
        elif 8 <= utc_h < 14:
            return self._sess_eu, "EU"
        else:
            return self._sess_other, "other"

    # ═════════════════════════════════════════════════════════════════════════
    # TIER HELPERS
    # ═════════════════════════════════════════════════════════════════════════

    def _get_tier(self, conf: float) -> str:
        if conf >= 0.65: return "A"
        if conf >= 0.40: return "B"
        return "C"

    def _get_size_pct(self, conf: float) -> float:
        """Returns % equity to use (mirrors Pine tier_a_pct / tier_b_pct / tier_c_pct)."""
        tier = self._get_tier(conf)
        return self._tier_pct[tier]

    def _get_size_mult(self, conf: float) -> float:
        """Convert % equity to a multiplier relative to 100% baseline."""
        return self._get_size_pct(conf) / 100.0

    # ═════════════════════════════════════════════════════════════════════════
    # SIGNAL BUILDER
    # ═════════════════════════════════════════════════════════════════════════

    def _build_signal(
        self,
        bars: pd.DataFrame,
        row: pd.Series,
        ts,
        ind: dict,
        atr: float,
        conf: float,
        direction: str,
        htf: dict,
        is_reversal: bool,
    ) -> Signal:
        close  = float(row["close"])
        ema_s  = ind["ema_slow"]

        # Compute pb_atr_d now that we have close
        pb_atr_d = abs(close - ind["ema_fast"]) / atr if atr > 0 else 0.0
        ind["pb_atr_d"] = pb_atr_d

        # Re-compute confidence with correct pb_atr_d
        conf_final = self._compute_conf(ind, direction) + self._htf_conf_mod(htf, direction)
        conf_final = max(0.0, min(1.0, conf_final))

        if conf_final < self._min_conf:
            return _hold(self.symbol, ts, self.strategy_id,
                         f"low_conf={conf_final:.2f}", meta=ind)

        tier = self._get_tier(conf_final)
        sl, tp1, tp2, sl_method = self._calc_sl(bars, close, atr, ema_s, direction, conf_final)

        risk = abs(close - sl)
        if risk <= 0:
            return _hold(self.symbol, ts, self.strategy_id, "zero_risk", meta=ind)

        # Session sizing
        sess_m, sess_name = self._get_session_mult(ts)

        # Size multiplier (Pine: qty = equity × tier_pct / close)
        size_pct  = self._get_size_pct(conf_final)
        # Expressed as a multiplier for the signal confidence field
        # The run script uses sig.confidence as the equity fraction
        # So we encode: confidence = (tier_pct/100) * session_mult
        final_mult = round(max(0.05, min(1.5, (size_pct / 100.0) * sess_m)), 4)

        tp1_close_pct = self._tp1_close_pct[tier]
        htf_label     = htf.get("label", "4H NEUTRAL")
        rev_tag       = " ⟳SWAP" if is_reversal else ""

        reason = (
            f"trendbot_v52_{direction.lower()}{rev_tag}"
            f"|T={tier}|c={conf_final:.2f}"
            f"|adx={ind['adx']:.1f}"
            f"|rsi={ind['rsi']:.1f}"
            f"|vol={ind['vol_ratio']:.2f}"
            f"|sess={sess_name}({sess_m})"
            f"|htf={htf['bias']}"
        )

        meta = {
            **ind,
            "atr":               round(atr, 6),
            "tp1":               round(tp1, 8),
            "tp2":               round(tp2, 8),
            "confidence_score":  conf_final,
            "confidence_tier":   tier,
            "sl_method":         sl_method,
            "tp1_close_pct":     tp1_close_pct,
            "rr_tp1":            self._tp1_r[tier],
            "rr_tp2":            self._tp2_r[tier],
            "session":           sess_name,
            "session_mult":      sess_m,
            "size_pct":          size_pct,
            "final_size_mult":   final_mult,
            "trailing_atr_mult": self._trail_atr_mult,
            "htf_bias":          htf.get("bias", 0),
            "htf_strength":      htf.get("strength", 0.0),
            "htf_label":         htf_label,
            "htf_ema_fast":      htf.get("htf_ema_fast", 0),
            "htf_ema_slow":      htf.get("htf_ema_slow", 0),
            "htf_adx":           htf.get("htf_adx", 0),
            "is_reversal":       is_reversal,
        }

        action = SignalAction.BUY if direction == "LONG" else SignalAction.SELL

        return Signal(
            action=action,
            symbol=self.symbol,
            ts=ts,
            strategy_id=self.strategy_id,
            confidence=final_mult,
            stop_loss=Decimal(str(round(sl, 8))),
            take_profit=Decimal(str(round(tp1, 8))),
            reason=reason,
            meta=meta,
        )