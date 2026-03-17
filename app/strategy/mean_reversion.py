"""
Mean Reversion Strategy — RSI + Bollinger Bands.

Estrategia A del "Sistema de Trading Repetible v1.0".

Pre-condiciones:
  - ADX(14) < 25 (no hay tendencia fuerte, mercado en rango)

Long entry:
  - RSI(14) <= rsi_oversold (sobreventa)
  - Precio toca Bollinger Band inferior
  - Vela gatillo: Hammer o Bullish Engulfing
  - R:R calculado >= min_rr

Short entry (si allow_short):
  - RSI(14) >= rsi_overbought (sobrecompra)
  - Precio toca Bollinger Band superior
  - Vela gatillo: Shooting Star o Bearish Engulfing

Stop Loss:  mínimo/máximo de los últimos 20 barras ± 1 ATR
Take Profit: SMA(20) — primer target de regreso a la media

Parameters:
    rsi_oversold       float,  default 30     — RSI threshold para LONG
    rsi_overbought     float,  default 70     — RSI threshold para SHORT
    adx_max            float,  default 25     — ADX máximo (encima = tendencia, no operar)
    min_rr             float,  default 1.5    — R:R mínimo para tomar el trade
    allow_short        bool,   default True   — habilitar señales SHORT
    swing_lookback     int,    default 20     — barras para buscar swing extremo
"""
from __future__ import annotations

from decimal import Decimal
from typing import Any

import pandas as pd

from app.strategy.base import BaseStrategy
from app.strategy.signals import Signal, SignalAction


def _hold(symbol: str, ts, strategy_id: str, reason: str, meta: dict | None = None) -> Signal:
    return Signal(action=SignalAction.HOLD, symbol=symbol, ts=ts, strategy_id=strategy_id, reason=reason, meta=meta or {})


class MeanReversionStrategy(BaseStrategy):
    """
    Mean Reversion: compra en sobreventa con confirmación de vela y Bollinger.
    """

    def __init__(self, symbol: str, params: dict[str, Any] | None = None) -> None:
        super().__init__(symbol, params)
        self._rsi_oversold   = float(self.params.get("rsi_oversold",  30.0))
        self._rsi_overbought = float(self.params.get("rsi_overbought", 70.0))
        self._adx_max        = float(self.params.get("adx_max",        25.0))
        self._min_rr         = float(self.params.get("min_rr",          1.5))
        self._allow_short    = bool(self.params.get("allow_short",      True))
        self._swing_lookback = int(self.params.get("swing_lookback",     20))

    @property
    def strategy_id(self) -> str:
        return (
            f"mean_reversion|rsi_os={self._rsi_oversold}"
            f"|adx_max={self._adx_max}|rr={self._min_rr}"
        )

    @property
    def min_bars_required(self) -> int:
        return 60  # BB(20) + ADX(14) + warm-up buffer

    # ── Indicators ────────────────────────────────────────────────────────

    def _compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        close = df["close"]
        high  = df["high"]
        low   = df["low"]

        # RSI(14)
        delta = close.diff()
        gain  = delta.clip(lower=0)
        loss  = (-delta).clip(lower=0)
        avg_g = gain.ewm(com=13, adjust=False).mean()
        avg_l = loss.ewm(com=13, adjust=False).mean()
        rs = avg_g / avg_l.replace(0, 1e-10)
        df["rsi"] = 100 - 100 / (1 + rs)

        # ATR(14)
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs(),
        ], axis=1).max(axis=1)
        df["atr"] = tr.ewm(com=13, adjust=False).mean()

        # ADX(14)
        up   = high.diff()
        down = -low.diff()
        dm_plus  = up.where((up > down) & (up > 0), 0.0)
        dm_minus = down.where((down > up) & (down > 0), 0.0)
        atr14 = tr.ewm(com=13, adjust=False).mean()
        di_plus  = 100 * dm_plus.ewm(com=13, adjust=False).mean() / atr14.replace(0, 1e-10)
        di_minus = 100 * dm_minus.ewm(com=13, adjust=False).mean() / atr14.replace(0, 1e-10)
        dx       = (100 * (di_plus - di_minus).abs() / (di_plus + di_minus).replace(0, 1e-10))
        df["adx"] = dx.ewm(com=13, adjust=False).mean()

        # Bollinger Bands (20, 2σ)
        sma20 = close.rolling(20).mean()
        std20 = close.rolling(20).std()
        df["bb_upper"] = sma20 + 2 * std20
        df["bb_lower"] = sma20 - 2 * std20
        df["sma20"]    = sma20

        return df

    # ── Candle patterns ───────────────────────────────────────────────────

    @staticmethod
    def _patterns(row: pd.Series, prev: pd.Series) -> dict[str, bool]:
        o, h, l, c   = row["open"],  row["high"],  row["low"],  row["close"]
        po, ph, pl, pc = prev["open"], prev["high"], prev["low"], prev["close"]
        body = abs(c - o)

        lower_wick = min(o, c) - l
        upper_wick = h - max(o, c)

        hammer    = body > 0 and lower_wick > 2 * body and upper_wick < body * 0.5
        star      = body > 0 and upper_wick > 2 * body and lower_wick < body * 0.5
        bull_eng  = pc < po and c > o and c > po and o < pc
        bear_eng  = pc > po and c < o and c < po and o > pc
        return {
            "hammer":            hammer,
            "shooting_star":     star,
            "bullish_engulfing": bull_eng,
            "bearish_engulfing": bear_eng,
        }

    # ── on_bar ────────────────────────────────────────────────────────────

    def on_bar(self, df: pd.DataFrame) -> Signal:
        if len(df) < self.min_bars_required:
            return _hold(self.symbol, df["ts"].iloc[-1], self.strategy_id, "warmup")

        df = self._compute_indicators(df)
        row  = df.iloc[-1]
        prev = df.iloc[-2]
        ts   = row["ts"]

        if pd.isna(row["adx"]) or pd.isna(row["rsi"]):
            return _hold(self.symbol, ts, self.strategy_id, "indicators_not_ready")

        # Build indicator dict for bar_data visibility
        ind: dict = {}
        if not pd.isna(row["rsi"]):      ind["rsi"]      = round(float(row["rsi"]), 2)
        if not pd.isna(row["adx"]):      ind["adx"]      = round(float(row["adx"]), 2)
        if not pd.isna(row["bb_upper"]): ind["bb_upper"] = round(float(row["bb_upper"]), 2)
        if not pd.isna(row["bb_lower"]): ind["bb_lower"] = round(float(row["bb_lower"]), 2)
        if not pd.isna(row["sma20"]):    ind["ema"]      = round(float(row["sma20"]), 2)

        # Pre-condition: no trending market
        if row["adx"] >= self._adx_max:
            return _hold(self.symbol, ts, self.strategy_id, f"adx={row['adx']:.1f}>={self._adx_max}", meta=ind)

        pat = self._patterns(row, prev)
        atr = row["atr"] if not pd.isna(row["atr"]) else 0.0

        # ── LONG ──────────────────────────────────────────────────────────
        if row["rsi"] <= self._rsi_oversold and row["low"] <= row["bb_lower"]:
            if pat["hammer"] or pat["bullish_engulfing"]:
                lookback = min(self._swing_lookback, len(df) - 1)
                swing_low = df["low"].iloc[-lookback:].min()
                sl = swing_low - atr
                tp = row["sma20"]

                risk   = row["close"] - sl
                reward = tp - row["close"]
                rr = reward / risk if risk > 0 else 0.0

                if rr >= self._min_rr and risk > 0:
                    pattern = "hammer" if pat["hammer"] else "bull_engulfing"
                    return Signal(
                        action=SignalAction.BUY,
                        symbol=self.symbol,
                        ts=ts,
                        strategy_id=self.strategy_id,
                        stop_loss=Decimal(str(round(sl, 8))),
                        take_profit=Decimal(str(round(tp, 8))),
                        reason=f"mean_rev_long|rsi={row['rsi']:.1f}|rr={rr:.2f}|{pattern}",
                        meta={**ind, "rr": round(rr, 3), "pattern": pattern},
                    )

        # ── SHORT ─────────────────────────────────────────────────────────
        if self._allow_short and row["rsi"] >= self._rsi_overbought and row["high"] >= row["bb_upper"]:
            if pat["shooting_star"] or pat["bearish_engulfing"]:
                lookback = min(self._swing_lookback, len(df) - 1)
                swing_high = df["high"].iloc[-lookback:].max()
                sl = swing_high + atr
                tp = row["sma20"]

                risk   = sl - row["close"]
                reward = row["close"] - tp
                rr = reward / risk if risk > 0 else 0.0

                if rr >= self._min_rr and risk > 0:
                    pattern = "shooting_star" if pat["shooting_star"] else "bear_engulfing"
                    return Signal(
                        action=SignalAction.SELL,
                        symbol=self.symbol,
                        ts=ts,
                        strategy_id=self.strategy_id,
                        stop_loss=Decimal(str(round(sl, 8))),
                        take_profit=Decimal(str(round(tp, 8))),
                        reason=f"mean_rev_short|rsi={row['rsi']:.1f}|rr={rr:.2f}|{pattern}",
                        meta={**ind, "rr": round(rr, 3), "pattern": pattern},
                    )

        return _hold(self.symbol, ts, self.strategy_id, "no_setup", meta=ind)
