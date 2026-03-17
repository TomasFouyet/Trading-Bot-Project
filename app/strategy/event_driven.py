"""
Event-Driven / Breakout Strategy — proxy via volume spike + range breakout.

Estrategia C del "Sistema de Trading Repetible v1.0".

En backtest no podemos modelar noticias reales, así que usamos un proxy:
detección de "eventos" via volumen anómalo + breakout de rango.
Esto captura la dinámica del documento: dislocaciones de precio causadas por
catálisis que se manifiestan como spikes de volumen + breakout confirmado.

Entry:
  - Volumen > vol_multiplier × SMA_vol(lookback) (evento detectado)
  - Cierre fuera del rango de las últimas `lookback` velas (breakout)
  - ADX subiendo respecto a la barra anterior (momentum activándose)
  - R:R calculado >= min_rr

Tamaño:  50% del normal (confidence=0.5), pre-confirmación
Exit:    TP a 2.5R, SL en el límite opuesto del rango
Timeout: max_hold_bars barras sin confirmación → cierre por tiempo

Parameters:
    vol_multiplier  float, default 2.0   — múltiplo de volumen para detectar el evento
    lookback        int,   default 20    — barras para definir el rango y vol promedio
    min_rr          float, default 2.0   — R:R mínimo para tomar el trade
    max_hold_bars   int,   default 20    — barras máximas antes de cierre por tiempo
    allow_short     bool,  default True  — habilitar señales SHORT (breakdown)
    tp_rr           float, default 2.5   — target en múltiplos de R
"""
from __future__ import annotations

from decimal import Decimal
from typing import Any

import pandas as pd

from app.strategy.base import BaseStrategy
from app.strategy.signals import Signal, SignalAction


def _hold(symbol: str, ts, strategy_id: str, reason: str, meta: dict | None = None) -> Signal:
    return Signal(action=SignalAction.HOLD, symbol=symbol, ts=ts, strategy_id=strategy_id, reason=reason, meta=meta or {})


class EventDrivenStrategy(BaseStrategy):
    """
    Event-driven breakout: detecta spikes de volumen + breakout de rango.
    """

    def __init__(self, symbol: str, params: dict[str, Any] | None = None) -> None:
        super().__init__(symbol, params)
        self._vol_mult    = float(self.params.get("vol_multiplier", 2.0))
        self._lookback    = int(self.params.get("lookback",         20))
        self._min_rr      = float(self.params.get("min_rr",          2.0))
        self._max_hold    = int(self.params.get("max_hold_bars",     20))
        self._allow_short = bool(self.params.get("allow_short",      True))
        self._tp_rr       = float(self.params.get("tp_rr",           2.5))

    @property
    def strategy_id(self) -> str:
        return (
            f"event_driven|vol_x={self._vol_mult}"
            f"|lookback={self._lookback}|rr={self._min_rr}"
        )

    @property
    def min_bars_required(self) -> int:
        return self._lookback + 30

    # ── Indicators ────────────────────────────────────────────────────────

    def _compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        high  = df["high"]
        low   = df["low"]
        close = df["close"]

        # ATR(14)
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low  - close.shift()).abs(),
        ], axis=1).max(axis=1)
        df["atr"] = tr.ewm(com=13, adjust=False).mean()

        # ADX(14)
        up   = high.diff()
        down = -low.diff()
        dm_plus  = up.where((up > down) & (up > 0), 0.0)
        dm_minus = down.where((down > up) & (down > 0), 0.0)
        atr14    = tr.ewm(com=13, adjust=False).mean()
        di_plus  = 100 * dm_plus.ewm(com=13, adjust=False).mean() / atr14.replace(0, 1e-10)
        di_minus = 100 * dm_minus.ewm(com=13, adjust=False).mean() / atr14.replace(0, 1e-10)
        dx       = 100 * (di_plus - di_minus).abs() / (di_plus + di_minus).replace(0, 1e-10)
        df["adx"] = dx.ewm(com=13, adjust=False).mean()

        # Volume SMA
        df["vol_sma"] = df["volume"].rolling(self._lookback).mean()

        return df

    # ── on_bar ────────────────────────────────────────────────────────────

    def on_bar(self, df: pd.DataFrame) -> Signal:
        if len(df) < self.min_bars_required:
            return _hold(self.symbol, df["ts"].iloc[-1], self.strategy_id, "warmup")

        df = self._compute_indicators(df)
        row  = df.iloc[-1]
        prev = df.iloc[-2]
        ts   = row["ts"]

        if pd.isna(row["adx"]) or pd.isna(row["vol_sma"]):
            return _hold(self.symbol, ts, self.strategy_id, "indicators_not_ready")

        # Volume spike check
        vol_avg = row["vol_sma"]
        if vol_avg <= 0:
            return _hold(self.symbol, ts, self.strategy_id, "no_volume")

        vol_ratio = row["volume"] / vol_avg

        # Build indicator dict for bar_data visibility
        ind: dict = {}
        if not pd.isna(row["adx"]):   ind["adx"]       = round(float(row["adx"]), 2)
        ind["vol_ratio"] = round(float(vol_ratio), 3)

        if vol_ratio < self._vol_mult:
            return _hold(self.symbol, ts, self.strategy_id, f"vol_ratio={vol_ratio:.1f}<{self._vol_mult}", meta=ind)

        # Range of previous `lookback` bars (excluding current)
        window = df.iloc[-(self._lookback + 1):-1]
        range_high = window["high"].max()
        range_low  = window["low"].min()

        # ADX rising → momentum activating
        adx_rising = (
            not pd.isna(row["adx"]) and
            not pd.isna(prev["adx"]) and
            row["adx"] > prev["adx"]
        )

        # ── LONG breakout ──────────────────────────────────────────────────
        if row["close"] > range_high and adx_rising:
            sl   = range_low
            risk = row["close"] - sl
            tp   = row["close"] + risk * self._tp_rr
            rr   = self._tp_rr  # by construction

            if rr >= self._min_rr and risk > 0:
                return Signal(
                    action=SignalAction.BUY,
                    symbol=self.symbol,
                    ts=ts,
                    strategy_id=self.strategy_id,
                    confidence=0.5,
                    stop_loss=Decimal(str(round(sl, 8))),
                    take_profit=Decimal(str(round(tp, 8))),
                    reason=f"event_breakout_long|vol_x={vol_ratio:.1f}|adx={row['adx']:.1f}",
                    meta={**ind, "rr": round(rr, 2), "max_hold": self._max_hold,
                          "range_high": round(range_high, 8), "range_low": round(range_low, 8)},
                )

        # ── SHORT breakdown ────────────────────────────────────────────────
        if self._allow_short and row["close"] < range_low and adx_rising:
            sl   = range_high
            risk = sl - row["close"]
            tp   = row["close"] - risk * self._tp_rr
            rr   = self._tp_rr

            if rr >= self._min_rr and risk > 0:
                return Signal(
                    action=SignalAction.SELL,
                    symbol=self.symbol,
                    ts=ts,
                    strategy_id=self.strategy_id,
                    confidence=0.5,
                    stop_loss=Decimal(str(round(sl, 8))),
                    take_profit=Decimal(str(round(tp, 8))),
                    reason=f"event_breakout_short|vol_x={vol_ratio:.1f}|adx={row['adx']:.1f}",
                    meta={**ind, "rr": round(rr, 2), "max_hold": self._max_hold,
                          "range_high": round(range_high, 8), "range_low": round(range_low, 8)},
                )

        return _hold(self.symbol, ts, self.strategy_id, "no_event", meta=ind)
