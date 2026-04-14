"""
TrendFollowing V2 Simple — Simplified exit management.

═══════════════════════════════════════════════════════════════════════════════
IDENTICAL to TrendFollowingV2:
  - Entry logic: EMA20/50 + ADX + MACD + pullback zone + edge detection
  - Confidence scoring
  - Signal cooldown

REMOVED (vs V2):
  - TP1 partial close / TP2 second target
  - Move SL to breakeven after TP1
  - Trailing stop
  - Tier-based sizing (A/B/C)
  - Session multiplier / streak adjuster
  - Reversal swap

REPLACED WITH:
  - Single fixed SL: atr_sl_mult × ATR below/above entry
  - Single fixed TP: rr_ratio × SL distance from entry
  - Full close at TP or SL hit
  - Fixed position size (confidence=1.0 always)

Tunable params for grid search:
  - rr_ratio:    reward/risk ratio (default 2.0)
  - atr_sl_mult: ATR multiplier for SL distance (default 1.5)
═══════════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any

import pandas as pd

from app.strategy.base import BaseStrategy
from app.strategy.signals import Signal, SignalAction


@dataclass
class _Trade:
    """Minimal trade state."""
    state: int = 0        # 0=flat, 1=long, -1=short
    entry: float = 0.0
    sl: float = 0.0
    tp: float = 0.0
    start_bar: int = 0

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
        self.tp = 0.0
        self.start_bar = 0


def _hold(symbol: str, ts, strategy_id: str, reason: str,
          meta: dict | None = None) -> Signal:
    return Signal(
        action=SignalAction.HOLD, symbol=symbol, ts=ts,
        strategy_id=strategy_id, reason=reason, meta=meta or {},
    )


class TrendFollowingV2Simple(BaseStrategy):
    """
    Simplified TrendFollowing — same entries as V2, fixed SL/TP exits.
    """

    def __init__(self, symbol: str, params: dict[str, Any] | None = None) -> None:
        super().__init__(symbol, params)

        # ── Entry parameters (IDENTICAL to V2) ────────────────────────
        self._adx_min        = float(self.params.get("adx_min",                20.0))
        self._adx_strong     = float(self.params.get("adx_strong",             35.0))
        self._ema_fast_p     = int(self.params.get("ema_fast",                   20))
        self._ema_slow_p     = int(self.params.get("ema_slow",                   50))
        self._slope_bars     = int(self.params.get("slope_bars",                  5))
        self._pb_tol_atr     = float(self.params.get("pullback_tolerance_atr",  1.0))
        self._min_confidence = float(self.params.get("min_confidence",          0.0))
        self._allow_short    = bool(self.params.get("allow_short",              True))
        self._sig_cooldown   = int(self.params.get("sig_cooldown",              5))

        # ── Simple exit parameters (NEW) ──────────────────────────────
        self._rr_ratio       = float(self.params.get("rr_ratio",               2.0))
        self._atr_sl_mult    = float(self.params.get("atr_sl_mult",            1.5))

        # ── Internal state ────────────────────────────────────────────
        self._trade = _Trade()
        self._bar_index: int = 0
        self._last_long_bar:  int = -999
        self._last_short_bar: int = -999
        self._prev_long_signal:  bool = False
        self._prev_short_signal: bool = False

    @property
    def strategy_id(self) -> str:
        return (
            f"trend_v2_simple|adx={self._adx_min}"
            f"|ema={self._ema_fast_p}/{self._ema_slow_p}"
            f"|rr={self._rr_ratio}|sl_atr={self._atr_sl_mult}"
        )

    @property
    def min_bars_required(self) -> int:
        return max(60, self._ema_slow_p + 20)

    @property
    def engine_manages_sl_tp(self) -> bool:
        return False

    # ══════════════════════════════════════════════════════════════════
    # INDICATORS — copied verbatim from V2
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

        # ADX(14)
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
    # CONFIDENCE — copied verbatim from V2
    # ══════════════════════════════════════════════════════════════════

    def _compute_confidence(self, row: pd.Series, df: pd.DataFrame,
                            direction: str) -> float:
        sc = 0.0

        adx = float(row["adx"]) if not pd.isna(row["adx"]) else 0
        if adx >= self._adx_strong:
            sc += 0.20
        elif adx >= 30:
            sc += 0.10

        atr = float(row["atr"]) if not pd.isna(row["atr"]) else 1e-10
        ema_f = float(row["ema_fast"])
        pb_atr = abs(float(row["close"]) - ema_f) / atr if atr > 0 else 999
        if pb_atr <= 0.5:
            sc += 0.20
        elif pb_atr <= 1.0:
            sc += 0.10

        hist = float(row["macd_hist"]) if not pd.isna(row["macd_hist"]) else 0
        prev_hist = float(df["macd_hist"].iloc[-2]) if len(df) >= 2 and not pd.isna(df["macd_hist"].iloc[-2]) else 0
        if direction == "LONG" and hist > 0 and hist > prev_hist:
            sc += 0.15
        elif direction == "SHORT" and hist < 0 and hist < prev_hist:
            sc += 0.15
        elif (direction == "LONG" and hist > 0) or (direction == "SHORT" and hist < 0):
            sc += 0.05

        body = abs(float(row["close"]) - float(row["open"]))
        rng  = float(row["high"]) - float(row["low"])
        ratio = body / rng if rng > 0 else 0
        if ratio >= 0.60:
            sc += 0.15
        elif ratio >= 0.40:
            sc += 0.07

        efs = float(row["ema_fast_slope"]) if not pd.isna(row["ema_fast_slope"]) else 0
        if direction == "LONG" and efs > 0:
            sc += 0.15
        elif direction == "SHORT" and efs < 0:
            sc += 0.15

        vol     = float(row["volume"]) if not pd.isna(row["volume"]) else 0
        vol_sma = float(row["vol_sma"]) if not pd.isna(row["vol_sma"]) else 1
        vol_r   = vol / vol_sma if vol_sma > 0 else 0
        if vol_r >= 1.2:
            sc += 0.15
        elif vol_r >= 0.8:
            sc += 0.05

        return round(min(sc, 1.0), 3)

    # ══════════════════════════════════════════════════════════════════
    # MAIN: on_bar_all() — simplified state machine
    # ══════════════════════════════════════════════════════════════════

    def on_bar_all(self, df: pd.DataFrame, htf_bias=None) -> list[Signal]:
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

        ind: dict = {
            "adx": round(adx, 2),
            "atr": round(atr, 6),
            "ema_fast": round(ema_f, 4),
            "ema_slow": round(ema_s, 4),
        }

        # ── STEP 1: Setup conditions (IDENTICAL to V2) ───────────────
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

        # ── Confidence (IDENTICAL to V2, but no tier/sizing) ──────────
        conf_long  = self._compute_confidence(row, df, "LONG") if long_base else 0.0
        conf_short = self._compute_confidence(row, df, "SHORT") if short_base else 0.0

        long_signal  = long_base  and conf_long  >= self._min_confidence
        short_signal = short_base and conf_short >= self._min_confidence

        # ── Edge detection (IDENTICAL to V2) ──────────────────────────
        long_trigger_raw  = long_signal  and not self._prev_long_signal
        short_trigger_raw = short_signal and not self._prev_short_signal
        self._prev_long_signal  = long_signal
        self._prev_short_signal = short_signal

        # ── Cooldown (IDENTICAL to V2) ────────────────────────────────
        long_trigger  = long_trigger_raw  and (self._bar_index - self._last_long_bar)  >= self._sig_cooldown
        short_trigger = short_trigger_raw and (self._bar_index - self._last_short_bar) >= self._sig_cooldown
        if long_trigger:
            self._last_long_bar = self._bar_index
        if short_trigger:
            self._last_short_bar = self._bar_index

        signals: list[Signal] = []

        # ── STEP 2: Check exits for open trade (SIMPLIFIED) ──────────
        if not self._trade.is_flat():
            exit_sig = self._check_exit(ts, high, low, close, ind)
            if exit_sig:
                signals.append(exit_sig)

        # ── STEP 3: Open new trade if flat (NO reversal swap) ────────
        if self._trade.is_flat() and (long_trigger or short_trigger):
            if long_trigger:
                sl_dist = atr * self._atr_sl_mult
                sl = close - sl_dist
                tp = close + sl_dist * self._rr_ratio
                signals.append(self._open_trade(ts, close, sl, tp, "LONG", ind))
            elif short_trigger:
                sl_dist = atr * self._atr_sl_mult
                sl = close + sl_dist
                tp = close - sl_dist * self._rr_ratio
                signals.append(self._open_trade(ts, close, sl, tp, "SHORT", ind))

        if not signals:
            return [_hold(self.symbol, ts, self.strategy_id, "no_setup", meta=ind)]

        return signals

    def on_bar(self, df: pd.DataFrame, htf_bias=None) -> Signal:
        sigs = self.on_bar_all(df, htf_bias=htf_bias)
        return sigs[0] if sigs else _hold(
            self.symbol, df["ts"].iloc[-1], self.strategy_id, "no_signal")

    # ══════════════════════════════════════════════════════════════════
    # EXIT: single SL or single TP, full close
    # ══════════════════════════════════════════════════════════════════

    def _check_exit(self, ts, bar_high: float, bar_low: float,
                    bar_close: float, ind: dict) -> Signal | None:
        t = self._trade

        # SL hit (priority over TP — same bar, SL wins)
        sl_hit = (t.is_long() and bar_low <= t.sl) or \
                 (t.is_short() and bar_high >= t.sl)

        if sl_hit:
            exit_price = t.sl
            pnl = self._pnl_pct(exit_price)
            self._trade.reset()
            return Signal(
                action=SignalAction.CLOSE,
                symbol=self.symbol, ts=ts,
                strategy_id=self.strategy_id,
                reason=f"sl_hit|pnl={pnl:+.2f}%",
                meta={**ind, "exit_type": "sl", "exit_price": exit_price, "pnl_pct": pnl},
            )

        # TP hit
        tp_hit = (t.is_long() and bar_high >= t.tp) or \
                 (t.is_short() and bar_low <= t.tp)

        if tp_hit:
            exit_price = t.tp
            pnl = self._pnl_pct(exit_price)
            self._trade.reset()
            return Signal(
                action=SignalAction.CLOSE,
                symbol=self.symbol, ts=ts,
                strategy_id=self.strategy_id,
                reason=f"tp_hit|pnl={pnl:+.2f}%",
                meta={**ind, "exit_type": "tp", "exit_price": exit_price, "pnl_pct": pnl},
            )

        return None

    # ══════════════════════════════════════════════════════════════════
    # OPEN TRADE — fixed size, no tiers
    # ══════════════════════════════════════════════════════════════════

    def _open_trade(self, ts, close: float, sl: float, tp: float,
                    direction: str, ind: dict) -> Signal:
        self._trade.state = 1 if direction == "LONG" else -1
        self._trade.entry = close
        self._trade.sl = sl
        self._trade.tp = tp
        self._trade.start_bar = self._bar_index

        action = SignalAction.BUY if direction == "LONG" else SignalAction.SELL
        risk = abs(close - sl)

        return Signal(
            action=action,
            symbol=self.symbol, ts=ts,
            strategy_id=self.strategy_id,
            confidence=1.0,  # fixed size
            stop_loss=Decimal(str(round(sl, 8))),
            take_profit=Decimal(str(round(tp, 8))),
            reason=f"trend_{direction.lower()}|rr={self._rr_ratio}|sl_atr={self._atr_sl_mult}",
            meta={
                **ind,
                "sl": round(sl, 8),
                "tp1": round(tp, 8),
                "tp2": round(tp, 8),
                "exit_type": "",
                "rr_ratio": self._rr_ratio,
                "atr_sl_mult": self._atr_sl_mult,
                "risk": round(risk, 8),
            },
        )

    def _pnl_pct(self, exit_price: float) -> float:
        t = self._trade
        if t.is_long():
            return (exit_price - t.entry) / t.entry * 100
        else:
            return (t.entry - exit_price) / t.entry * 100
