"""
Trend Following Strategy — Multi-Timeframe (proxy vía EMAs múltiples).
With 4-Layer Adaptive Intelligence System.

Estrategia B del "Sistema de Trading Repetible v1.0".

═══════════════════════════════════════════════════════════════════════
  4 CAPAS ADAPTATIVAS
═══════════════════════════════════════════════════════════════════════

CAPA 1 — CONFIDENCE SCORING (calidad de entrada)
  Evalúa 6 factores de confluencia (0.0–1.0) que determinan:
    - Tamaño de posición (Signal.confidence)
    - Distancias TP1/TP2
    - % de posición cerrado en TP1
  Factores:
    +0.20  ADX fuerte (>35)
    +0.20  Pullback ajustado (<0.5 ATR de EMA rápida)
    +0.15  MACD histograma acelerando
    +0.15  Vela fuerte (body >60% del rango)
    +0.15  EMA rápida alineada con dirección del trade
    +0.15  Volumen por encima del promedio (>1.2x SMA20)
  Tiers:
    A (>=0.65): 100% size, TP1=1.5R (33% close), TP2=3R
    B (0.40-0.65): 75% size, TP1=1.5R (50% close), TP2=2.5R
    C (<0.40): filtrado por defecto (min_confidence=0.40)

CAPA 2 — SESSION-AWARE SIZING (sesión de mercado)
  Multiplica el tamaño según la sesión UTC de la entrada:
    US (14-21 UTC):    1.0x  (91% del profit histórico)
    EU (08-14 UTC):    0.75x
    Asia/Night:        0.50x

CAPA 3 — STREAK ADJUSTER (anti-euforia / anti-tilt)
  After 2+ wins consecutivos: 0.75x (WR cae a 36% post-win)
  After losses: 1.0x (WR sube a 51% post-loss)

CAPA 4 — PATIENCE TIMER (SL suave)
  Flag en meta["soft_sl_bars"] para que el engine use cierre-por-cierre
  en las primeras N barras (evita wicks <12h que tienen 34% WR).

═══════════════════════════════════════════════════════════════════════

Parameters:
    # Core
    adx_min                 float, default 25
    min_rr                  float, default 2.0
    pullback_tolerance_atr  float, default 1.5
    allow_short             bool,  default True
    ema_fast                int,   default 20
    ema_slow                int,   default 50
    slope_bars              int,   default 5
    # Layer 1: Confidence
    use_confidence          bool,  default True
    adx_strong              float, default 35
    tight_pb_atr            float, default 0.5
    min_confidence          float, default 0.40
    # Layer 2: Session
    use_session_filter      bool,  default True
    us_session_start        int,   default 14
    us_session_end          int,   default 21
    eu_session_start        int,   default 8
    eu_session_end          int,   default 14
    session_mult_us         float, default 1.0
    session_mult_eu         float, default 0.75
    session_mult_other      float, default 0.50
    # Layer 3: Streak
    use_streak_adj          bool,  default True
    streak_euphoria_after   int,   default 2
    streak_euphoria_mult    float, default 0.75
    # Layer 4: Patience
    use_patience            bool,  default True
    soft_sl_bars            int,   default 48      (15min bars = 12h)
"""
from __future__ import annotations

from decimal import Decimal
from typing import Any

import pandas as pd

from app.strategy.base import BaseStrategy
from app.strategy.signals import Signal, SignalAction


def _hold(symbol: str, ts, strategy_id: str, reason: str, meta: dict | None = None) -> Signal:
    return Signal(
        action=SignalAction.HOLD, symbol=symbol, ts=ts,
        strategy_id=strategy_id, reason=reason, meta=meta or {},
    )


class TrendFollowingStrategy(BaseStrategy):
    """
    Trend following multi-TF (proxy) con pullback a EMA(20).
    4 capas adaptativas: confidence, session, streak, patience.
    """

    def __init__(self, symbol: str, params: dict[str, Any] | None = None) -> None:
        super().__init__(symbol, params)

        # ── Core parameters ───────────────────────────────────────────
        self._adx_min        = float(self.params.get("adx_min",                25.0))
        self._min_rr         = float(self.params.get("min_rr",                  2.0))
        self._pb_tol_atr     = float(self.params.get("pullback_tolerance_atr",  1.5))
        self._allow_short    = bool(self.params.get("allow_short",              True))
        self._ema_fast       = int(self.params.get("ema_fast",                   20))
        self._ema_slow       = int(self.params.get("ema_slow",                   50))
        self._slope_bars     = int(self.params.get("slope_bars",                  5))

        # ── Layer 1: Confidence scoring ───────────────────────────────
        self._use_confidence = bool(self.params.get("use_confidence",           True))
        self._adx_strong     = float(self.params.get("adx_strong",             35.0))
        self._tight_pb_atr   = float(self.params.get("tight_pb_atr",            0.5))
        self._min_confidence = float(self.params.get("min_confidence",           0.40))

        # ── Layer 2: Session-aware sizing ─────────────────────────────
        self._use_session    = bool(self.params.get("use_session_filter",       True))
        self._us_start       = int(self.params.get("us_session_start",          14))
        self._us_end         = int(self.params.get("us_session_end",            21))
        self._eu_start       = int(self.params.get("eu_session_start",           8))
        self._eu_end         = int(self.params.get("eu_session_end",            14))
        self._sess_mult_us   = float(self.params.get("session_mult_us",         1.0))
        self._sess_mult_eu   = float(self.params.get("session_mult_eu",         0.75))
        self._sess_mult_oth  = float(self.params.get("session_mult_other",      0.50))

        # ── Layer 3: Streak adjuster ──────────────────────────────────
        self._use_streak     = bool(self.params.get("use_streak_adj",           True))
        self._euph_after     = int(self.params.get("streak_euphoria_after",      2))
        self._euph_mult      = float(self.params.get("streak_euphoria_mult",    0.75))
        self._consecutive_wins: int  = 0
        self._consecutive_losses: int = 0

        # ── Layer 4: Patience timer ───────────────────────────────────
        self._use_patience   = bool(self.params.get("use_patience",             True))
        self._soft_sl_bars   = int(self.params.get("soft_sl_bars",              48))

    # ── Public: called by engine after position closes ────────────────

    def notify_trade_result(self, won: bool) -> None:
        """
        Update streak counters. Call from BacktestEngine after each
        position close to enable Layer 3.
        """
        if won:
            self._consecutive_wins += 1
            self._consecutive_losses = 0
        else:
            self._consecutive_losses += 1
            self._consecutive_wins = 0

    # ── Properties ────────────────────────────────────────────────────

    @property
    def strategy_id(self) -> str:
        layers = []
        if self._use_confidence: layers.append("conf")
        if self._use_session:    layers.append("sess")
        if self._use_streak:     layers.append("strk")
        if self._use_patience:   layers.append("pat")
        tag = "+".join(layers) if layers else "none"
        return (
            f"trend_following|adx={self._adx_min}"
            f"|ema={self._ema_fast}/{self._ema_slow}|L={tag}"
        )

    @property
    def min_bars_required(self) -> int:
        return max(60, self._ema_slow + 20)

    # ══════════════════════════════════════════════════════════════════
    # INDICATORS
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
        df["ema_fast"]       = close.ewm(span=self._ema_fast, adjust=False).mean()
        df["ema_slow"]       = close.ewm(span=self._ema_slow, adjust=False).mean()
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
    # LAYER 1: CONFIDENCE SCORING
    # ══════════════════════════════════════════════════════════════════

    def _compute_confidence(self, row: pd.Series, df: pd.DataFrame,
                            direction: str) -> tuple[float, list[str]]:
        score = 0.0
        reasons: list[str] = []

        # Factor 1: ADX strength (+0.20)
        adx = float(row["adx"]) if not pd.isna(row["adx"]) else 0
        if adx >= self._adx_strong:
            score += 0.20; reasons.append(f"adx_strong({adx:.0f})")
        elif adx >= 30:
            score += 0.10; reasons.append(f"adx_mod({adx:.0f})")

        # Factor 2: Pullback tightness (+0.20)
        atr = float(row["atr"]) if not pd.isna(row["atr"]) else 1e-10
        ema_f = float(row["ema_fast"]) if not pd.isna(row["ema_fast"]) else float(row["close"])
        pb_atr = abs(float(row["close"]) - ema_f) / atr if atr > 0 else 999
        if pb_atr <= self._tight_pb_atr:
            score += 0.20; reasons.append(f"tight_pb({pb_atr:.2f}ATR)")
        elif pb_atr <= 1.0:
            score += 0.10; reasons.append(f"ok_pb({pb_atr:.2f}ATR)")

        # Factor 3: MACD histogram accelerating (+0.15)
        hist = float(row["macd_hist"]) if not pd.isna(row["macd_hist"]) else 0
        prev = float(df["macd_hist"].iloc[-2]) if len(df) >= 2 and not pd.isna(df["macd_hist"].iloc[-2]) else 0
        if direction == "LONG" and hist > 0 and hist > prev:
            score += 0.15; reasons.append("macd_accel_bull")
        elif direction == "SHORT" and hist < 0 and hist < prev:
            score += 0.15; reasons.append("macd_accel_bear")
        elif (direction == "LONG" and hist > 0) or (direction == "SHORT" and hist < 0):
            score += 0.05; reasons.append("macd_aligned")

        # Factor 4: Candle body strength (+0.15)
        body = abs(float(row["close"]) - float(row["open"]))
        rng  = float(row["high"]) - float(row["low"])
        ratio = body / rng if rng > 0 else 0
        if ratio >= 0.60:
            score += 0.15; reasons.append(f"strong_candle({ratio:.0%})")
        elif ratio >= 0.40:
            score += 0.07; reasons.append(f"decent_candle({ratio:.0%})")

        # Factor 5: EMA fast slope alignment (+0.15)
        efs = float(row["ema_fast_slope"]) if not pd.isna(row["ema_fast_slope"]) else 0
        if direction == "LONG" and efs > 0:
            score += 0.15; reasons.append("ema_fast_rising")
        elif direction == "SHORT" and efs < 0:
            score += 0.15; reasons.append("ema_fast_falling")

        # Factor 6: Volume confirmation (+0.15)
        vol     = float(row["volume"]) if not pd.isna(row["volume"]) else 0
        vol_sma = float(row["vol_sma"]) if not pd.isna(row["vol_sma"]) else 1
        vol_r   = vol / vol_sma if vol_sma > 0 else 0
        if vol_r >= 1.2:
            score += 0.15; reasons.append(f"vol_high({vol_r:.1f}x)")
        elif vol_r >= 0.8:
            score += 0.05; reasons.append(f"vol_ok({vol_r:.1f}x)")

        return round(min(score, 1.0), 3), reasons

    def _get_tier_params(self, confidence: float) -> dict:
        if confidence >= 0.65:
            return {"tier": "A", "size_mult": 2.0,  "tp1_r": 1.5, "tp2_r": 3.0, "tp1_close_pct": 0.33}
        elif confidence >= 0.40:
            return {"tier": "B", "size_mult": 1.5, "tp1_r": 1.5, "tp2_r": 2.5, "tp1_close_pct": 0.50}
        else:
            return {"tier": "C", "size_mult": 0.50, "tp1_r": 1.0, "tp2_r": 1.5, "tp1_close_pct": 0.70}

    # ══════════════════════════════════════════════════════════════════
    # LAYER 2: SESSION-AWARE SIZING
    # ══════════════════════════════════════════════════════════════════

    def _get_session_mult(self, ts) -> tuple[float, str]:
        if not self._use_session:
            return 1.0, "off"
        if hasattr(ts, 'utcoffset') and ts.utcoffset() is not None:
            utc_h = (ts.hour + ts.utcoffset().total_seconds() / 3600) % 24
        elif hasattr(ts, 'hour'):
            utc_h = ts.hour
        else:
            return 1.0, "?"
        if self._us_start <= utc_h < self._us_end:
            return self._sess_mult_us, "US"
        elif self._eu_start <= utc_h < self._eu_end:
            return self._sess_mult_eu, "EU"
        return self._sess_mult_oth, "off_hours"

    # ══════════════════════════════════════════════════════════════════
    # LAYER 3: STREAK ADJUSTER
    # ══════════════════════════════════════════════════════════════════

    def _get_streak_mult(self) -> tuple[float, str]:
        if not self._use_streak:
            return 1.0, "off"
        if self._consecutive_wins >= self._euph_after:
            return self._euph_mult, f"anti_euph({self._consecutive_wins}W)"
        return 1.0, "normal"

    # ══════════════════════════════════════════════════════════════════
    # MAIN SIGNAL LOGIC
    # ══════════════════════════════════════════════════════════════════

    def on_bar(self, df: pd.DataFrame) -> Signal:
        if len(df) < self.min_bars_required:
            return _hold(self.symbol, df["ts"].iloc[-1], self.strategy_id, "warmup")

        df = self._compute_indicators(df)
        row = df.iloc[-1]
        ts  = row["ts"]

        if pd.isna(row["adx"]) or pd.isna(row["ema_slow"]):
            return _hold(self.symbol, ts, self.strategy_id, "indicators_not_ready")

        ind: dict = {}
        for key in ["adx", "ema_fast", "ema_slow", "macd", "macd_signal", "macd_hist"]:
            val = row.get(key)
            if val is not None and not pd.isna(val):
                ind[key] = round(float(val), 4 if "macd" in key else 2)

        if row["adx"] < self._adx_min:
            return _hold(self.symbol, ts, self.strategy_id,
                         f"adx={row['adx']:.1f}<{self._adx_min}", meta=ind)

        atr = float(row["atr"]) if not pd.isna(row["atr"]) else 0.0
        pb_zone = abs(float(row["close"]) - float(row["ema_fast"])) < atr * self._pb_tol_atr

        sl_rising  = row["ema_slow_slope"] > 0
        sl_falling = row["ema_slow_slope"] < 0
        p_above    = float(row["close"]) > float(row["ema_slow"])
        p_below    = float(row["close"]) < float(row["ema_slow"])
        m_bull     = float(row["macd"]) > float(row["macd_signal"])
        m_bear     = float(row["macd"]) < float(row["macd_signal"])
        c_bull     = float(row["close"]) > float(row["open"])
        c_bear     = float(row["close"]) < float(row["open"])

        if sl_rising and p_above and m_bull and pb_zone and c_bull:
            return self._build_signal(df, row, ts, ind, atr, "LONG")

        if self._allow_short and sl_falling and p_below and m_bear and pb_zone and c_bear:
            return self._build_signal(df, row, ts, ind, atr, "SHORT")

        return _hold(self.symbol, ts, self.strategy_id, "no_setup", meta=ind)

    # ── Signal builder ────────────────────────────────────────────────

    def _build_signal(self, df: pd.DataFrame, row: pd.Series,
                      ts, ind: dict, atr: float, direction: str) -> Signal:
        close = float(row["close"])

        # Layer 1
        if self._use_confidence:
            conf, conf_reasons = self._compute_confidence(row, df, direction)
            if conf < self._min_confidence:
                return _hold(self.symbol, ts, self.strategy_id,
                             f"low_conf={conf:.2f}", meta=ind)
            tier = self._get_tier_params(conf)
        else:
            conf, conf_reasons = 1.0, ["off"]
            tier = {"tier": "X", "size_mult": 1.0, "tp1_r": 1.5,
                    "tp2_r": 3.0, "tp1_close_pct": 0.33}

        # Layer 2
        sess_m, sess_name = self._get_session_mult(ts)

        # Layer 3
        strk_m, strk_reason = self._get_streak_mult()

        # Combined sizing
        final_mult = round(max(0.10, min(tier["size_mult"] * sess_m * strk_m, 2.0)), 3)

        # SL / TP
        lb = min(20, len(df) - 1)
        if direction == "LONG":
            swing = float(df["low"].iloc[-lb:].min())
            sl   = min(swing, float(row["ema_slow"])) - atr * 0.2
            risk = close - sl
            tp1  = close + risk * tier["tp1_r"]
            tp2  = close + risk * tier["tp2_r"]
            action = SignalAction.BUY
        else:
            swing = float(df["high"].iloc[-lb:].max())
            sl   = max(swing, float(row["ema_slow"])) + atr * 0.2
            risk = sl - close
            tp1  = close - risk * tier["tp1_r"]
            tp2  = close - risk * tier["tp2_r"]
            action = SignalAction.SELL

        if risk <= 0:
            return _hold(self.symbol, ts, self.strategy_id, "zero_risk", meta=ind)

        # Layer 4
        soft_sl = self._soft_sl_bars if self._use_patience else 0

        reason = (
            f"trend_{direction.lower()}"
            f"|T={tier['tier']}|c={conf:.2f}"
            f"|s={sess_name}({sess_m})"
            f"|k={strk_reason}"
            f"|adx={float(row['adx']):.1f}"
        )

        meta = {
            **ind,
            "tp1": round(tp1, 8), "tp2": round(tp2, 8),
            "rr_tp1": tier["tp1_r"], "rr_tp2": tier["tp2_r"],
            "tp1_close_pct": tier["tp1_close_pct"],
            "confidence_score": conf, "confidence_tier": tier["tier"],
            "confidence_reasons": conf_reasons,
            "session": sess_name, "session_mult": sess_m,
            "streak_reason": strk_reason, "streak_mult": strk_m,
            "consecutive_wins": self._consecutive_wins,
            "consecutive_losses": self._consecutive_losses,
            "soft_sl_bars": soft_sl,
            "final_size_mult": final_mult,
            "size_breakdown": f"tier={tier['size_mult']}*sess={sess_m}*strk={strk_m}",
        }

        return Signal(
            action=action, symbol=self.symbol, ts=ts,
            strategy_id=self.strategy_id,
            confidence=final_mult,
            stop_loss=Decimal(str(round(sl, 8))),
            take_profit=Decimal(str(round(tp1, 8))),
            reason=reason, meta=meta,
        )