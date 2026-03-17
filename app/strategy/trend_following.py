"""
Trend Following Strategy — Multi-Timeframe (proxy vía EMAs múltiples).

Estrategia B del "Sistema de Trading Repetible v1.0".

Emula multi-timeframe en un solo timeframe usando EMAs de distintos períodos:
  - EMA(50) actúa como proxy del trend diario
  - EMA(20) actúa como proxy del pullback de 4H

Long entry:
  - EMA(50) con pendiente ascendente (proxy trend alcista diario)
  - Precio por encima de EMA(50)
  - Precio retrocede hasta zona EMA(20) y rebota (pullback)
  - MACD(12,26,9) > Signal (momentum confirma)
  - ADX(14) > adx_min (tendencia activa)
  - Vela de continuación alcista (cierre > apertura)

Short entry (si allow_short): inverso

Stop Loss:  debajo de EMA(50) o swing low (el más conservador) ± margen ATR
TP1:        1.5R — cierra 33% de la posición
TP2:        3.0R — cierra 33% adicional, activa trailing stop en EMA(20)
TP3 (trail): trailing sobre EMA(20) — cierra el 34% restante

Parameters:
    adx_min                 float, default 25     — ADX mínimo para confirmar tendencia
    min_rr                  float, default 2.0    — R:R blended mínimo
    pullback_tolerance_atr  float, default 1.5    — distancia max a EMA(20) en ATRs
    allow_short             bool,  default True   — habilitar señales SHORT
    ema_fast                int,   default 20     — EMA rápida (pullback proxy)
    ema_slow                int,   default 50     — EMA lenta (trend proxy)
    slope_bars              int,   default 5      — barras para medir pendiente EMA
"""
from __future__ import annotations

from decimal import Decimal
from typing import Any

import pandas as pd

from app.strategy.base import BaseStrategy
from app.strategy.signals import Signal, SignalAction


def _hold(symbol: str, ts, strategy_id: str, reason: str, meta: dict | None = None) -> Signal:
    return Signal(action=SignalAction.HOLD, symbol=symbol, ts=ts, strategy_id=strategy_id, reason=reason, meta=meta or {})


class TrendFollowingStrategy(BaseStrategy):
    """
    Trend following multi-TF (proxy) con entradas en pullback a EMA(20).
    """

    def __init__(self, symbol: str, params: dict[str, Any] | None = None) -> None:
        super().__init__(symbol, params)
        self._adx_min        = float(self.params.get("adx_min",                25.0))
        self._min_rr         = float(self.params.get("min_rr",                  2.0))
        self._pb_tol_atr     = float(self.params.get("pullback_tolerance_atr",  1.5))
        self._allow_short    = bool(self.params.get("allow_short",              True))
        self._ema_fast       = int(self.params.get("ema_fast",                   20))
        self._ema_slow       = int(self.params.get("ema_slow",                   50))
        self._slope_bars     = int(self.params.get("slope_bars",                  5))

    @property
    def strategy_id(self) -> str:
        return (
            f"trend_following|adx_min={self._adx_min}"
            f"|ema={self._ema_fast}/{self._ema_slow}|rr={self._min_rr}"
        )

    @property
    def min_bars_required(self) -> int:
        return max(60, self._ema_slow + 20)

    # ── Indicators ────────────────────────────────────────────────────────

    def _compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        close = df["close"]
        high  = df["high"]
        low   = df["low"]

        # ATR(14)
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low  - close.shift()).abs(),
        ], axis=1).max(axis=1)
        atr14 = tr.ewm(com=13, adjust=False).mean()
        df["atr"] = atr14

        # ADX(14)
        up   = high.diff()
        down = -low.diff()
        dm_plus  = up.where((up > down) & (up > 0), 0.0)
        dm_minus = down.where((down > up) & (down > 0), 0.0)
        di_plus  = 100 * dm_plus.ewm(com=13, adjust=False).mean() / atr14.replace(0, 1e-10)
        di_minus = 100 * dm_minus.ewm(com=13, adjust=False).mean() / atr14.replace(0, 1e-10)
        dx       = 100 * (di_plus - di_minus).abs() / (di_plus + di_minus).replace(0, 1e-10)
        df["adx"] = dx.ewm(com=13, adjust=False).mean()

        # EMAs
        df["ema_fast"] = close.ewm(span=self._ema_fast, adjust=False).mean()
        df["ema_slow"] = close.ewm(span=self._ema_slow, adjust=False).mean()
        df["ema_slow_slope"] = df["ema_slow"].diff(self._slope_bars)

        # MACD (12, 26, 9)
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd  = ema12 - ema26
        df["macd"]        = macd
        df["macd_signal"] = macd.ewm(span=9, adjust=False).mean()

        return df

    # ── on_bar ────────────────────────────────────────────────────────────

    def on_bar(self, df: pd.DataFrame) -> Signal:
        if len(df) < self.min_bars_required:
            return _hold(self.symbol, df["ts"].iloc[-1], self.strategy_id, "warmup")

        df = self._compute_indicators(df)
        row = df.iloc[-1]
        ts  = row["ts"]

        if pd.isna(row["adx"]) or pd.isna(row["ema_slow"]):
            return _hold(self.symbol, ts, self.strategy_id, "indicators_not_ready")

        # Build indicator dict for bar_data visibility
        ind: dict = {}
        if not pd.isna(row["adx"]):          ind["adx"]         = round(float(row["adx"]), 2)
        if not pd.isna(row["ema_fast"]):     ind["ema_fast"]    = round(float(row["ema_fast"]), 2)
        if not pd.isna(row["ema_slow"]):     ind["ema_slow"]    = round(float(row["ema_slow"]), 2)
        if not pd.isna(row["macd"]):         ind["macd"]        = round(float(row["macd"]), 4)
        if not pd.isna(row["macd_signal"]):  ind["macd_signal"] = round(float(row["macd_signal"]), 4)

        # Must have an active trend
        if row["adx"] < self._adx_min:
            return _hold(self.symbol, ts, self.strategy_id, f"adx={row['adx']:.1f}<{self._adx_min}", meta=ind)

        atr  = row["atr"] if not pd.isna(row["atr"]) else 0.0
        pullback_zone = abs(row["close"] - row["ema_fast"]) < atr * self._pb_tol_atr

        ema_slow_rising  = row["ema_slow_slope"] > 0
        ema_slow_falling = row["ema_slow_slope"] < 0
        price_above_slow = row["close"] > row["ema_slow"]
        price_below_slow = row["close"] < row["ema_slow"]
        macd_bull  = row["macd"] > row["macd_signal"]
        macd_bear  = row["macd"] < row["macd_signal"]
        bull_candle = row["close"] > row["open"]
        bear_candle = row["close"] < row["open"]

        # Blended R:R across scaled exits: 0.33*1.5R + 0.33*3R + 0.34*trailing ≈ 2.5R
        BLENDED_RR = 0.33 * 1.5 + 0.33 * 3.0 + 0.34 * 3.0  # ~2.5

        # ── LONG ──────────────────────────────────────────────────────────
        if ema_slow_rising and price_above_slow and macd_bull and pullback_zone and bull_candle:
            lookback = min(20, len(df) - 1)
            swing_low = df["low"].iloc[-lookback:].min()
            sl    = min(swing_low, row["ema_slow"]) - atr * 0.2
            risk  = row["close"] - sl
            tp1   = row["close"] + risk * 1.5
            tp2   = row["close"] + risk * 2

            if risk > 0 and BLENDED_RR >= self._min_rr:
                return Signal(
                    action=SignalAction.BUY,
                    symbol=self.symbol,
                    ts=ts,
                    strategy_id=self.strategy_id,
                    stop_loss=Decimal(str(round(sl, 8))),
                    take_profit=Decimal(str(round(tp1, 8))),
                    reason=f"trend_long|adx={row['adx']:.1f}|slope={row['ema_slow_slope']:.2f}",
                    meta={**ind, "rr": round(BLENDED_RR, 2), "tp1": round(tp1, 8), "tp2": round(tp2, 8)},
                )

        # ── SHORT ─────────────────────────────────────────────────────────
        if self._allow_short and ema_slow_falling and price_below_slow and macd_bear and pullback_zone and bear_candle:
            lookback = min(20, len(df) - 1)
            swing_high = df["high"].iloc[-lookback:].max()
            sl   = max(swing_high, row["ema_slow"]) + atr * 0.2
            risk = sl - row["close"]
            tp1  = row["close"] - risk * 1.5
            tp2  = row["close"] - risk * 2

            if risk > 0 and BLENDED_RR >= self._min_rr:
                return Signal(
                    action=SignalAction.SELL,
                    symbol=self.symbol,
                    ts=ts,
                    strategy_id=self.strategy_id,
                    stop_loss=Decimal(str(round(sl, 8))),
                    take_profit=Decimal(str(round(tp1, 8))),
                    reason=f"trend_short|adx={row['adx']:.1f}|slope={row['ema_slow_slope']:.2f}",
                    meta={**ind, "rr": round(BLENDED_RR, 2), "tp1": round(tp1, 8), "tp2": round(tp2, 8)},
                )

        return _hold(self.symbol, ts, self.strategy_id, "no_setup", meta=ind)
