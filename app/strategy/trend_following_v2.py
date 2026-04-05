"""
TrendFollowing V2 — Pine Script v5.1 Aligned.

═══════════════════════════════════════════════════════════════════════════════
This strategy mirrors the TrendBot MTF v5.1 Pine Script EXACTLY.

Key differences vs V1 (trend_following.py):
  1. STATEFUL TRADE MACHINE: tracks trade_state (FLAT/LONG/SHORT) internally
  2. EDGE DETECTION: fires only on first bar of setup (signal AND NOT signal[1])
  3. INTEGRATED SL/TP: checks SL/TP/TP1→BE/TP2 every bar using bar high/low
  4. REVERSAL SWAP: closes opposite trade + opens new in same bar
  5. EMITS CLOSE/PARTIAL_CLOSE SIGNALS: engine doesn't manage SL/TP

The strategy emits:
  - BUY/SELL for new entries (with SL/TP in meta)
  - PARTIAL_CLOSE when TP1 hits (with close_pct and new BE stop)
  - CLOSE when SL or TP2 hits (or reversal swap close)
  - HOLD otherwise

The caller (run_multi_paper.py or BacktestEngine) just executes the signals.
═══════════════════════════════════════════════════════════════════════════════

Parameters: same as V1 (trend_following.py) plus:
  - enable_reversal  bool, default True  — allow reversal swap
"""
from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

import pandas as pd

from app.strategy.base import BaseStrategy
from app.strategy.signals import Signal, SignalAction


# ─────────────────────────────────────────────────────────────────────────────
# Trade state (mirrors Pine's var declarations)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TradeState:
    """Internal trade state — mirrors Pine Script trade_state variables."""
    state: int = 0              # 0=flat, 1=long, -1=short
    entry: float = 0.0
    sl: float = 0.0
    tp1: float = 0.0
    tp2: float = 0.0
    tp1_hit: bool = False
    start_bar: int = 0
    tier: str = "X"
    sl_method: str = ""
    confidence: float = 0.0
    signal_reason: str = ""

    def is_flat(self) -> bool:
        return self.state == 0

    def is_long(self) -> bool:
        return self.state == 1

    def is_short(self) -> bool:
        return self.state == -1

    def reset(self):
        self.state = 0
        self.entry = 0.0
        self.sl = 0.0
        self.tp1 = 0.0
        self.tp2 = 0.0
        self.tp1_hit = False
        self.start_bar = 0
        self.tier = "X"
        self.sl_method = ""
        self.confidence = 0.0
        self.signal_reason = ""


def _hold(symbol: str, ts, strategy_id: str, reason: str,
          meta: dict | None = None) -> Signal:
    return Signal(
        action=SignalAction.HOLD, symbol=symbol, ts=ts,
        strategy_id=strategy_id, reason=reason, meta=meta or {},
    )


class TrendFollowingV2(BaseStrategy):
    """
    Trend Following V2 — stateful, Pine Script v5.1 aligned.

    Integrates trade state machine, SL/TP management, edge detection,
    cooldown, and reversal swap — all in one class.
    """

    def __init__(self, symbol: str, params: dict[str, Any] | None = None) -> None:
        super().__init__(symbol, params)

        # ── Core parameters (same as Pine) ────────────────────────────
        self._adx_min        = float(self.params.get("adx_min",                20.0))
        self._adx_strong     = float(self.params.get("adx_strong",             35.0))
        self._ema_fast_p     = int(self.params.get("ema_fast",                   20))
        self._ema_slow_p     = int(self.params.get("ema_slow",                   50))
        self._slope_bars     = int(self.params.get("slope_bars",                  5))
        self._pb_tol_atr     = float(self.params.get("pullback_tolerance_atr",  1.0))
        self._min_confidence = float(self.params.get("min_confidence",          0.0))
        self._allow_short    = bool(self.params.get("allow_short",              True))

        # ── Signal cooldown (Pine: sig_cooldown) ─────────────────────
        self._sig_cooldown   = int(self.params.get("sig_cooldown",              5))

        # ── Reversal swap (Pine: enable_reversal) ────────────────────
        self._enable_reversal = bool(self.params.get("enable_reversal",         True))

        # ── SL structure (Pine: SL CALCULATION section) ──────────────
        self._sl_lookback    = int(self.params.get("sl_swing_lookback",         50))
        self._sl_window      = int(self.params.get("sl_swing_window",           3))
        self._sl_min_atr     = float(self.params.get("sl_min_atr",              1.0))
        self._sl_max_atr     = float(self.params.get("sl_max_atr",              2.5))
        self._sl_buf_atr     = float(self.params.get("sl_buffer_atr",           0.3))

        # ── TP per tier (Pine: TP1/TP2 R inputs) ─────────────────────
        self._tp1_r_A = float(self.params.get("tp1_r_A", 1.5))
        self._tp2_r_A = float(self.params.get("tp2_r_A", 3.0))
        self._tp1_r_B = float(self.params.get("tp1_r_B", 1.5))
        self._tp2_r_B = float(self.params.get("tp2_r_B", 2.5))
        self._tp1_r_C = float(self.params.get("tp1_r_C", 1.0))
        self._tp2_r_C = float(self.params.get("tp2_r_C", 1.5))

        # ── Layer 2: Session sizing ──────────────────────────────────
        self._use_session    = bool(self.params.get("use_session_filter",       True))
        self._us_start       = int(self.params.get("us_session_start",          14))
        self._us_end         = int(self.params.get("us_session_end",            21))
        self._eu_start       = int(self.params.get("eu_session_start",           8))
        self._eu_end         = int(self.params.get("eu_session_end",            14))
        self._sess_mult_us   = float(self.params.get("session_mult_us",         1.0))
        self._sess_mult_eu   = float(self.params.get("session_mult_eu",         0.75))
        self._sess_mult_oth  = float(self.params.get("session_mult_other",      0.50))

        # ── Layer 3: Streak adjuster ─────────────────────────────────
        self._use_streak     = bool(self.params.get("use_streak_adj",           True))
        self._euph_after     = int(self.params.get("streak_euphoria_after",      2))
        self._euph_mult      = float(self.params.get("streak_euphoria_mult",    0.75))
        self._consecutive_wins: int  = 0
        self._consecutive_losses: int = 0

        # ── Layer 4: Patience timer ──────────────────────────────────
        self._use_patience   = bool(self.params.get("use_patience",             True))
        self._soft_sl_bars   = int(self.params.get("soft_sl_bars",              48))

        # ── Internal state (mirrors Pine var declarations) ────────────
        self._trade = TradeState()
        self._bar_index: int = 0
        self._last_long_bar:  int = -999
        self._last_short_bar: int = -999

        # Previous bar signal state for edge detection
        # (mirrors Pine: long_signal[1], short_signal[1])
        self._prev_long_signal:  bool = False
        self._prev_short_signal: bool = False

    # ── Properties ────────────────────────────────────────────────────

    @property
    def strategy_id(self) -> str:
        return (
            f"trend_v2|adx={self._adx_min}"
            f"|ema={self._ema_fast_p}/{self._ema_slow_p}"
            f"|rev={'on' if self._enable_reversal else 'off'}"
        )

    @property
    def min_bars_required(self) -> int:
        return max(60, self._ema_slow_p + 20)

    @property
    def engine_manages_sl_tp(self) -> bool:
        # V2 manages SL/TP internally — disable engine auto-close
        return False

    def notify_trade_result(self, won: bool) -> None:
        if won:
            self._consecutive_wins += 1
            self._consecutive_losses = 0
        else:
            self._consecutive_losses += 1
            self._consecutive_wins = 0

    # ══════════════════════════════════════════════════════════════════
    # INDICATORS (same as Pine)
    # ══════════════════════════════════════════════════════════════════

    def _compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        close, high, low = df["close"], df["high"], df["low"]

        # ATR(14)
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low  - close.shift()).abs(),
        ], axis=1).max(axis=1)
        df["atr"] = tr.ewm(com=13, adjust=False).mean()

        # ADX(14) — same as Pine ta.dmi(14, 14)
        atr14 = df["atr"]
        up   = high.diff()
        down = -low.diff()
        dm_p = up.where((up > down) & (up > 0), 0.0)
        dm_m = down.where((down > up) & (down > 0), 0.0)
        di_p = 100 * dm_p.ewm(com=13, adjust=False).mean() / atr14.replace(0, 1e-10)
        di_m = 100 * dm_m.ewm(com=13, adjust=False).mean() / atr14.replace(0, 1e-10)
        dx   = 100 * (di_p - di_m).abs() / (di_p + di_m).replace(0, 1e-10)
        df["adx"] = dx.ewm(com=13, adjust=False).mean()

        # EMAs
        df["ema_fast"]       = close.ewm(span=self._ema_fast_p, adjust=False).mean()
        df["ema_slow"]       = close.ewm(span=self._ema_slow_p, adjust=False).mean()
        df["ema_slow_slope"] = df["ema_slow"].diff(self._slope_bars)
        df["ema_fast_slope"] = df["ema_fast"].diff(self._slope_bars)

        # MACD (12, 26, 9)
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd  = ema12 - ema26
        df["macd"]        = macd
        df["macd_signal"] = macd.ewm(span=9, adjust=False).mean()
        df["macd_hist"]   = macd - df["macd_signal"]

        # Volume SMA
        df["vol_sma"] = df["volume"].rolling(20).mean()

        return df

    # ══════════════════════════════════════════════════════════════════
    # CONFIDENCE SCORING (mirrors Pine compute_conf exactly)
    # ══════════════════════════════════════════════════════════════════

    def _compute_confidence(self, row: pd.Series, df: pd.DataFrame,
                            direction: str) -> float:
        sc = 0.0

        # Factor 1: ADX strength
        adx = float(row["adx"]) if not pd.isna(row["adx"]) else 0
        if adx >= self._adx_strong:
            sc += 0.20
        elif adx >= 30:
            sc += 0.10

        # Factor 2: Pullback tightness
        atr = float(row["atr"]) if not pd.isna(row["atr"]) else 1e-10
        ema_f = float(row["ema_fast"])
        pb_atr = abs(float(row["close"]) - ema_f) / atr if atr > 0 else 999
        if pb_atr <= 0.5:
            sc += 0.20
        elif pb_atr <= 1.0:
            sc += 0.10

        # Factor 3: MACD histogram
        hist = float(row["macd_hist"]) if not pd.isna(row["macd_hist"]) else 0
        prev_hist = float(df["macd_hist"].iloc[-2]) if len(df) >= 2 and not pd.isna(df["macd_hist"].iloc[-2]) else 0
        if direction == "LONG" and hist > 0 and hist > prev_hist:
            sc += 0.15
        elif direction == "SHORT" and hist < 0 and hist < prev_hist:
            sc += 0.15
        elif (direction == "LONG" and hist > 0) or (direction == "SHORT" and hist < 0):
            sc += 0.05

        # Factor 4: Candle body strength
        body = abs(float(row["close"]) - float(row["open"]))
        rng  = float(row["high"]) - float(row["low"])
        ratio = body / rng if rng > 0 else 0
        if ratio >= 0.60:
            sc += 0.15
        elif ratio >= 0.40:
            sc += 0.07

        # Factor 5: EMA fast slope
        efs = float(row["ema_fast_slope"]) if not pd.isna(row["ema_fast_slope"]) else 0
        if direction == "LONG" and efs > 0:
            sc += 0.15
        elif direction == "SHORT" and efs < 0:
            sc += 0.15

        # Factor 6: Volume
        vol     = float(row["volume"]) if not pd.isna(row["volume"]) else 0
        vol_sma = float(row["vol_sma"]) if not pd.isna(row["vol_sma"]) else 1
        vol_r   = vol / vol_sma if vol_sma > 0 else 0
        if vol_r >= 1.2:
            sc += 0.15
        elif vol_r >= 0.8:
            sc += 0.05

        return round(min(sc, 1.0), 3)

    # ══════════════════════════════════════════════════════════════════
    # TIER CLASSIFICATION (mirrors Pine get_tier / tp1_r_X / etc.)
    # ══════════════════════════════════════════════════════════════════

    def _get_tier(self, conf: float) -> str:
        if conf >= 0.65: return "A"
        if conf >= 0.40: return "B"
        return "C"

    def _get_size_mult(self, conf: float) -> float:
        if conf >= 0.65: return 2.0
        if conf >= 0.40: return 1.5
        return 0.5

    def _get_tp1_close_pct(self, conf: float) -> float:
        if conf >= 0.65: return 0.33
        if conf >= 0.40: return 0.50
        return 0.70

    def _get_tp_ratios(self, conf: float) -> tuple[float, float]:
        if conf >= 0.65: return self._tp1_r_A, self._tp2_r_A
        if conf >= 0.40: return self._tp1_r_B, self._tp2_r_B
        return self._tp1_r_C, self._tp2_r_C

    # ══════════════════════════════════════════════════════════════════
    # SL CALCULATION (mirrors Pine calc_sl exactly)
    # ══════════════════════════════════════════════════════════════════

    def _find_swing(self, df: pd.DataFrame, direction: str) -> tuple[float | None, bool]:
        """Find confirmed swing pivot. Returns (level, found)."""
        w  = self._sl_window
        lb = min(self._sl_lookback, len(df) - 1)
        if lb <= 2 * w:
            return None, False

        if direction == "LONG":
            vals = df["low"].values
            for i in range(w, lb - w):
                is_pivot = True
                for j in range(1, w + 1):
                    if i - j < 0:
                        is_pivot = False
                        break
                    if vals[-(i+1)] > vals[-(i-j+1)] or vals[-(i+1)] > vals[-(i+j+1)]:
                        is_pivot = False
                        break
                if is_pivot:
                    return float(vals[-(i+1)]), True
        else:
            vals = df["high"].values
            for i in range(w, lb - w):
                is_pivot = True
                for j in range(1, w + 1):
                    if i - j < 0:
                        is_pivot = False
                        break
                    if vals[-(i+1)] < vals[-(i-j+1)] or vals[-(i+1)] < vals[-(i+j+1)]:
                        is_pivot = False
                        break
                if is_pivot:
                    return float(vals[-(i+1)]), True

        return None, False

    def _calc_sl(self, df: pd.DataFrame, close: float, atr: float,
                 ema_slow: float, direction: str,
                 conf: float) -> tuple[float, float, float, str]:
        """
        Compute SL, TP1, TP2 — mirrors Pine calc_sl() exactly.
        Returns (sl, tp1, tp2, method).
        """
        lb = min(self._sl_lookback, len(df) - 1)
        swing, swing_found = self._find_swing(df, direction)
        method = "swing" if swing_found else "lookback"

        tp1_r, tp2_r = self._get_tp_ratios(conf)

        if direction == "LONG":
            fallback = float(df["low"].iloc[-lb:].min()) if lb > 0 else close
            base = swing if swing_found else fallback
            struct_sl = min(base, ema_slow) - atr * self._sl_buf_atr
            dist = close - struct_sl
            dist = max(dist, atr * self._sl_min_atr)
            dist = min(dist, atr * self._sl_max_atr)
            sl = close - dist
            risk = abs(close - sl)
            tp1 = close + risk * tp1_r
            tp2 = close + risk * tp2_r
        else:
            fallback = float(df["high"].iloc[-lb:].max()) if lb > 0 else close
            base = swing if swing_found else fallback
            struct_sl = max(base, ema_slow) + atr * self._sl_buf_atr
            dist = struct_sl - close
            dist = max(dist, atr * self._sl_min_atr)
            dist = min(dist, atr * self._sl_max_atr)
            sl = close + dist
            risk = abs(close - sl)
            tp1 = close - risk * tp1_r
            tp2 = close - risk * tp2_r

        return sl, tp1, tp2, method

    # ══════════════════════════════════════════════════════════════════
    # SESSION / STREAK (same as V1)
    # ══════════════════════════════════════════════════════════════════

    def _get_session(self, ts) -> tuple[float, str]:
        if not self._use_session:
            return 1.0, "off"
        try:
            if hasattr(ts, 'utcoffset') and ts.utcoffset() is not None:
                utc_h = (ts.hour + ts.utcoffset().total_seconds() / 3600) % 24
            elif hasattr(ts, 'hour'):
                utc_h = ts.hour
            else:
                return 1.0, "?"
        except Exception:
            return 1.0, "?"
        if self._us_start <= utc_h < self._us_end:
            return self._sess_mult_us, "US"
        if self._eu_start <= utc_h < self._eu_end:
            return self._sess_mult_eu, "EU"
        return self._sess_mult_oth, "off_hours"

    def _get_streak_mult(self) -> tuple[float, str]:
        if not self._use_streak:
            return 1.0, "off"
        if self._consecutive_wins >= self._euph_after:
            return self._euph_mult, f"anti_euph({self._consecutive_wins}W)"
        return 1.0, "normal"

    # ══════════════════════════════════════════════════════════════════
    # HTF BIAS MODIFIER (mirrors Pine htf_conf_mod)
    # ══════════════════════════════════════════════════════════════════

    def _htf_conf_mod(self, direction: str, htf_bias) -> float:
        """Apply HTF confidence modifier — mirrors Pine htf_conf_mod()."""
        if htf_bias is None:
            return 0.0
        bias_val = htf_bias.bias if hasattr(htf_bias, 'bias') else 0
        strength = htf_bias.strength if hasattr(htf_bias, 'strength') else 0

        if bias_val == 0:
            return 0.0

        aligned = (direction == "LONG" and bias_val == 1) or \
                  (direction == "SHORT" and bias_val == -1)
        if aligned:
            return 0.15 if strength >= 0.6 else 0.05
        else:
            return -0.35 if strength >= 0.6 else -0.20

    # ══════════════════════════════════════════════════════════════════
    # MAIN: on_bar_all() — the Pine Script state machine
    # ══════════════════════════════════════════════════════════════════

    def on_bar_all(self, df: pd.DataFrame, htf_bias=None) -> list[Signal]:
        """
        Process one bar through the Pine Script v5.1 state machine.

        Returns a list of signals (may contain multiple: e.g. CLOSE + BUY
        for a reversal swap, or PARTIAL_CLOSE for TP1).

        This method mirrors the Pine Script flow:
          1. Compute indicators
          2. Compute setup conditions + confidence
          3. Edge detection + cooldown → trigger
          4. Check SL/TP exits for open trades
          5. Process reversal swap
          6. Open new trades
        """
        self._bar_index += 1
        ts = df["ts"].iloc[-1]
        if hasattr(ts, "to_pydatetime"):
            ts = ts.to_pydatetime()

        if len(df) < self.min_bars_required:
            return [_hold(self.symbol, ts, self.strategy_id, "warmup")]

        df = self._compute_indicators(df)
        row = df.iloc[-1]

        if pd.isna(row["adx"]) or pd.isna(row["ema_slow"]):
            return [_hold(self.symbol, ts, self.strategy_id, "indicators_not_ready")]

        close = float(row["close"])
        high  = float(row["high"])
        low   = float(row["low"])
        atr   = float(row["atr"]) if not pd.isna(row["atr"]) else 0.0
        adx   = float(row["adx"])
        ema_f = float(row["ema_fast"])
        ema_s = float(row["ema_slow"])

        # ── Build indicator meta (for logging/display) ────────────────
        ind: dict = {
            "adx": round(adx, 2),
            "atr": round(atr, 6),
            "ema_fast": round(ema_f, 4),
            "ema_slow": round(ema_s, 4),
        }

        # ══════════════════════════════════════════════════════════════
        # STEP 1: Setup conditions (mirrors Pine SETUP CONDITIONS)
        # ══════════════════════════════════════════════════════════════
        pb_zone    = abs(close - ema_f) < atr * self._pb_tol_atr if atr > 0 else False
        sl_rising  = float(row["ema_slow_slope"]) > 0 if not pd.isna(row["ema_slow_slope"]) else False
        sl_falling = float(row["ema_slow_slope"]) < 0 if not pd.isna(row["ema_slow_slope"]) else False
        p_above    = close > ema_s
        p_below    = close < ema_s
        m_bull     = float(row["macd"]) > float(row["macd_signal"])
        m_bear     = float(row["macd"]) < float(row["macd_signal"])
        c_bull     = close > float(row["open"])
        c_bear     = close < float(row["open"])
        adx_ok     = adx >= self._adx_min

        long_base  = adx_ok and sl_rising  and p_above and m_bull and pb_zone and c_bull
        short_base = adx_ok and sl_falling and p_below and m_bear and pb_zone and c_bear and self._allow_short

        # ── Confidence + HTF modifier ─────────────────────────────────
        raw_conf_l = self._compute_confidence(row, df, "LONG") if long_base else 0.0
        raw_conf_s = self._compute_confidence(row, df, "SHORT") if short_base else 0.0

        htf_mod_l = self._htf_conf_mod("LONG", htf_bias)
        htf_mod_s = self._htf_conf_mod("SHORT", htf_bias)

        conf_long  = max(0.0, min(1.0, raw_conf_l + htf_mod_l))
        conf_short = max(0.0, min(1.0, raw_conf_s + htf_mod_s))

        long_signal  = long_base  and conf_long  >= self._min_confidence
        short_signal = short_base and conf_short >= self._min_confidence

        # ── Edge detection (Pine: signal AND NOT signal[1]) ───────────
        long_trigger_raw  = long_signal  and not self._prev_long_signal
        short_trigger_raw = short_signal and not self._prev_short_signal

        # Update previous signal state for next bar
        self._prev_long_signal  = long_signal
        self._prev_short_signal = short_signal

        # ── Cooldown (Pine: bar_index - last_long_bar >= sig_cooldown) ─
        long_trigger  = long_trigger_raw  and (self._bar_index - self._last_long_bar)  >= self._sig_cooldown
        short_trigger = short_trigger_raw and (self._bar_index - self._last_short_bar) >= self._sig_cooldown

        if long_trigger:
            self._last_long_bar = self._bar_index
        if short_trigger:
            self._last_short_bar = self._bar_index

        # ── Compute SL/TP for potential entries ───────────────────────
        sl_l, tp1_l, tp2_l, sl_method_l = self._calc_sl(df, close, atr, ema_s, "LONG", conf_long) if long_trigger else (0, 0, 0, "")
        sl_s, tp1_s, tp2_s, sl_method_s = self._calc_sl(df, close, atr, ema_s, "SHORT", conf_short) if short_trigger else (0, 0, 0, "")

        # ══════════════════════════════════════════════════════════════
        # STEP 2: Reversal detection (Pine: REVERSAL SWAP section)
        # ══════════════════════════════════════════════════════════════
        reversal_to_long  = (self._enable_reversal and
                             self._trade.is_short() and long_trigger)
        reversal_to_short = (self._enable_reversal and
                             self._trade.is_long() and short_trigger and
                             not self._trade.tp1_hit)
        is_reversal = reversal_to_long or reversal_to_short

        # ── Determine new entry direction ─────────────────────────────
        normal_long_entry  = self._trade.is_flat() and long_trigger
        normal_short_entry = self._trade.is_flat() and short_trigger
        new_long  = normal_long_entry  or reversal_to_long
        new_short = normal_short_entry or reversal_to_short

        signals: list[Signal] = []

        # ══════════════════════════════════════════════════════════════
        # STEP 3: Exit detection (Pine: SL/TP check section)
        # Only if NOT a reversal (reversal handles its own close)
        # ══════════════════════════════════════════════════════════════
        if not self._trade.is_flat() and not is_reversal:
            exit_signal = self._check_exits(ts, high, low, close, ind)
            if exit_signal:
                signals.append(exit_signal)
                # If trade was fully closed (SL or TP2), state is now flat
                # TP1 partial close doesn't reset state

        # ══════════════════════════════════════════════════════════════
        # STEP 4: Process TP1 hit → move SL to BE
        # (Already handled inside _check_exits)
        # ══════════════════════════════════════════════════════════════

        # ══════════════════════════════════════════════════════════════
        # STEP 5: Reversal swap close (Pine: REVERSAL SWAP section)
        # ══════════════════════════════════════════════════════════════
        if is_reversal:
            old_dir = "L" if self._trade.is_long() else "S"
            new_dir = "L" if reversal_to_long else "S"
            pnl_pct = self._calc_unrealized_pnl_pct(close)

            signals.append(Signal(
                action=SignalAction.CLOSE,
                symbol=self.symbol, ts=ts,
                strategy_id=self.strategy_id,
                reason=f"reversal_swap|{old_dir}→{new_dir}|pnl={pnl_pct:+.2f}%",
                meta={
                    **ind,
                    "exit_type": "reversal_swap",
                    "exit_price": close,
                    "old_direction": "LONG" if self._trade.is_long() else "SHORT",
                    "new_direction": "LONG" if reversal_to_long else "SHORT",
                },
            ))
            # Reset trade state — new trade opens below
            self._trade.reset()

        # ══════════════════════════════════════════════════════════════
        # STEP 6: Open new trade (Pine: new_long/new_short section)
        # ══════════════════════════════════════════════════════════════
        if new_long and self._trade.is_flat():
            entry_signal = self._open_trade(
                df, row, ts, ind, atr, "LONG",
                conf_long, sl_l, tp1_l, tp2_l, sl_method_l,
                htf_bias, htf_mod_l,
            )
            if entry_signal:
                signals.append(entry_signal)

        elif new_short and self._trade.is_flat():
            entry_signal = self._open_trade(
                df, row, ts, ind, atr, "SHORT",
                conf_short, sl_s, tp1_s, tp2_s, sl_method_s,
                htf_bias, htf_mod_s,
            )
            if entry_signal:
                signals.append(entry_signal)

        # ── Add position state to all signals' meta ───────────────────
        pos_meta = self._position_meta()
        for sig in signals:
            sig.meta.update(pos_meta)

        if not signals:
            reason = self._build_hold_reason(
                long_signal, short_signal, long_trigger_raw,
                short_trigger_raw, adx, ind,
            )
            return [_hold(self.symbol, ts, self.strategy_id, reason,
                          meta={**ind, **pos_meta})]

        return signals

    def on_bar(self, df: pd.DataFrame, htf_bias=None) -> Signal:
        """Backward-compatible single-signal interface."""
        sigs = self.on_bar_all(df, htf_bias=htf_bias)
        return sigs[0] if sigs else _hold(
            self.symbol, df["ts"].iloc[-1], self.strategy_id, "no_signal")

    # ══════════════════════════════════════════════════════════════════
    # EXIT DETECTION (mirrors Pine's SL/TP1/TP2 checks)
    # ══════════════════════════════════════════════════════════════════

    def _check_exits(self, ts, bar_high: float, bar_low: float,
                     bar_close: float, ind: dict) -> Signal | None:
        """
        Check SL, TP1, TP2 against current bar — mirrors Pine exactly.

        Priority: SL > TP (if both hit same bar, SL wins).
        TP1 → partial close + move SL to entry (BE).
        TP2 → full close.
        SL → full close.
        """
        t = self._trade
        is_long = t.is_long()

        # ── SL hit ────────────────────────────────────────────────────
        sl_hit_long  = is_long      and bar_low  <= t.sl
        sl_hit_short = t.is_short() and bar_high >= t.sl

        if sl_hit_long or sl_hit_short:
            exit_price = t.sl
            pnl_pct = self._pnl_pct(exit_price)
            is_win = pnl_pct > 0
            exit_type = "sl"
            if t.tp1_hit:
                exit_type = "trailing_sl"  # was trailing after TP1

            self._trade.reset()
            return Signal(
                action=SignalAction.CLOSE,
                symbol=self.symbol, ts=ts,
                strategy_id=self.strategy_id,
                reason=f"{exit_type}|pnl={pnl_pct:+.2f}%",
                meta={
                    **ind,
                    "exit_type": exit_type,
                    "exit_price": exit_price,
                    "pnl_pct": pnl_pct,
                },
            )

        # ── TP1 hit (partial close) ───────────────────────────────────
        tp1_hit_long  = is_long      and not t.tp1_hit and bar_high >= t.tp1
        tp1_hit_short = t.is_short() and not t.tp1_hit and bar_low  <= t.tp1

        if tp1_hit_long or tp1_hit_short:
            t.tp1_hit = True
            old_sl = t.sl
            t.sl = t.entry  # Move SL to breakeven (BE)
            close_pct = self._get_tp1_close_pct(t.confidence)

            return Signal(
                action=SignalAction.PARTIAL_CLOSE,
                symbol=self.symbol, ts=ts,
                strategy_id=self.strategy_id,
                reason=f"tp1_hit|close={close_pct*100:.0f}%|sl→BE={t.entry:.2f}",
                meta={
                    **ind,
                    "exit_type": "tp1",
                    "exit_price": t.tp1,
                    "close_pct": close_pct,
                    "new_sl": t.entry,  # BE stop
                    "tp2": t.tp2,
                },
            )

        # ── TP2 hit (full close) ──────────────────────────────────────
        tp2_hit_long  = is_long      and t.tp1_hit and bar_high >= t.tp2
        tp2_hit_short = t.is_short() and t.tp1_hit and bar_low  <= t.tp2

        if tp2_hit_long or tp2_hit_short:
            pnl_pct = self._pnl_pct(t.tp2)
            self._trade.reset()
            return Signal(
                action=SignalAction.CLOSE,
                symbol=self.symbol, ts=ts,
                strategy_id=self.strategy_id,
                reason=f"tp2_hit|pnl={pnl_pct:+.2f}%",
                meta={
                    **ind,
                    "exit_type": "tp2",
                    "exit_price": t.tp2,
                    "pnl_pct": pnl_pct,
                },
            )

        return None

    # ══════════════════════════════════════════════════════════════════
    # OPEN TRADE (mirrors Pine: new_long/new_short block)
    # ══════════════════════════════════════════════════════════════════

    def _open_trade(self, df, row, ts, ind, atr, direction,
                    conf, sl, tp1, tp2, sl_method,
                    htf_bias, htf_mod) -> Signal | None:
        """Set internal trade state and return entry signal."""
        close = float(row["close"])
        risk = abs(close - sl)
        if risk <= 0:
            return None

        tier = self._get_tier(conf)
        size_mult = self._get_size_mult(conf)
        sess_m, sess_name = self._get_session(ts)
        strk_m, strk_reason = self._get_streak_mult()
        final_mult = round(max(0.10, min(size_mult * sess_m * strk_m, 2.0)), 3)

        # Set trade state
        self._trade.state = 1 if direction == "LONG" else -1
        self._trade.entry = close
        self._trade.sl = sl
        self._trade.tp1 = tp1
        self._trade.tp2 = tp2
        self._trade.tp1_hit = False
        self._trade.start_bar = self._bar_index
        self._trade.tier = tier
        self._trade.sl_method = sl_method
        self._trade.confidence = conf

        # Build HTF tags (mirrors Pine label text)
        htf_tag = ""
        htf_meta = {}
        if htf_bias is not None:
            bias_val = htf_bias.bias if hasattr(htf_bias, 'bias') else 0
            if direction == "LONG":
                htf_tag = "↑4H" if bias_val == 1 else ("↓4H!" if bias_val == -1 else "~4H")
            else:
                htf_tag = "↓4H" if bias_val == -1 else ("↑4H!" if bias_val == 1 else "~4H")
            htf_meta = {
                "htf_bias": bias_val,
                "htf_strength": htf_bias.strength if hasattr(htf_bias, 'strength') else 0,
                "htf_mod": round(htf_mod, 3),
                "htf_label": htf_tag,
            }

        # Patience timer
        soft_sl = self._soft_sl_bars if self._use_patience else 0

        reason = (
            f"trend_{direction.lower()}"
            f"|T={tier}|c={conf:.2f}"
            f"|s={sess_name}({sess_m})"
            f"|k={strk_reason}"
            f"|adx={float(row['adx']):.1f}"
        )
        if htf_tag:
            reason += f"|{htf_tag}"

        self._trade.signal_reason = reason

        action = SignalAction.BUY if direction == "LONG" else SignalAction.SELL

        return Signal(
            action=action,
            symbol=self.symbol, ts=ts,
            strategy_id=self.strategy_id,
            confidence=final_mult,
            stop_loss=Decimal(str(round(sl, 8))),
            take_profit=Decimal(str(round(tp1, 8))),
            reason=reason,
            meta={
                **ind,
                **htf_meta,
                "tp1": round(tp1, 8),
                "tp2": round(tp2, 8),
                "sl": round(sl, 8),
                "sl_method": sl_method,
                "tp1_close_pct": self._get_tp1_close_pct(conf),
                "soft_sl_bars": soft_sl,
                "confidence_score": conf,
                "confidence_tier": tier,
                "session": sess_name,
                "session_mult": sess_m,
                "streak_reason": strk_reason,
                "streak_mult": strk_m,
                "final_size_mult": final_mult,
                "rr_tp1": self._get_tp_ratios(conf)[0],
                "rr_tp2": self._get_tp_ratios(conf)[1],
            },
        )

    # ══════════════════════════════════════════════════════════════════
    # HELPERS
    # ══════════════════════════════════════════════════════════════════

    def _pnl_pct(self, exit_price: float) -> float:
        t = self._trade
        if t.is_long():
            return (exit_price - t.entry) / t.entry * 100
        else:
            return (t.entry - exit_price) / t.entry * 100

    def _calc_unrealized_pnl_pct(self, current_price: float) -> float:
        return self._pnl_pct(current_price)

    def _position_meta(self) -> dict:
        """Current position state for meta/display."""
        t = self._trade
        if t.is_flat():
            return {"state": "FLAT"}
        state_str = "LONG" if t.is_long() else "SHORT"
        if t.tp1_hit:
            state_str += "_BE"
        return {
            "state": state_str,
            "entry": t.entry,
            "sl": t.sl,
            "tp": t.tp1 if not t.tp1_hit else t.tp2,
            "tp2": t.tp2,
        }

    def _build_hold_reason(self, long_signal, short_signal,
                           long_trigger_raw, short_trigger_raw,
                           adx, ind) -> str:
        """Build descriptive hold reason (for display/logging)."""
        if not self._trade.is_flat():
            t = self._trade
            bars = self._bar_index - t.start_bar
            phase = "TRAIL" if t.tp1_hit else ("LONG" if t.is_long() else "SHORT")
            return f"in_trade|{phase}|{bars}bars|sl={t.sl:.2f}"

        if adx < self._adx_min:
            return f"adx={adx:.1f}<{self._adx_min}"

        if long_signal and not long_trigger_raw:
            return "long_signal_active_no_edge"
        if short_signal and not short_trigger_raw:
            return "short_signal_active_no_edge"

        if long_trigger_raw:
            bars_left = self._sig_cooldown - (self._bar_index - self._last_long_bar)
            return f"cooldown_long({bars_left}bars)"
        if short_trigger_raw:
            bars_left = self._sig_cooldown - (self._bar_index - self._last_short_bar)
            return f"cooldown_short({bars_left}bars)"

        return "no_setup"

    # ══════════════════════════════════════════════════════════════════
    # PUBLIC: Get current trade state (for run_multi_paper.py)
    # ══════════════════════════════════════════════════════════════════

    @property
    def trade_state(self) -> TradeState:
        """Access internal trade state (read-only reference)."""
        return self._trade

    def force_close(self) -> None:
        """Force reset trade state (e.g. on shutdown)."""
        self._trade.reset()

    def update_sl_after_tp1(self, new_sl: float) -> None:
        """Update SL after TP1 partial close (called by executor)."""
        self._trade.sl = new_sl
