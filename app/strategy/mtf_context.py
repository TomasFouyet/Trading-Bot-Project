"""
Multi-Timeframe Context — HTF Bias + LTF Entry Precision.

Provides two things to TrendFollowingStrategy:

  1. HTFContext  — reads 4H bars and returns a directional bias (+1 long, -1 short, 0 neutral).
                   The strategy uses this as a hard filter: a 15M signal in the OPPOSITE
                   direction of the 4H bias is rejected or downgraded.

  2. LTFEntry   — given an approved 15M setup direction, scans the last N 5M bars for
                   a tighter entry price (micro-pullback to 5M EMA9 or consolidation low/high).
                   Returns an adjusted entry suggestion and updated SL.

Design principles:
  - These are pure data-analysis helpers; no broker calls, no state mutations.
  - The 15M strategy remains the source of truth for SL/TP/size.
  - HTF filter: if 4H trend is AGAINST signal, confidence is reduced by a flat penalty
    (not a hard block — allows the caller to decide whether to trade or skip).
  - LTF entry: best-effort. If no micro-pullback is found, returns the original entry.
  - All computations work on a plain pandas DataFrame of OHLCVBars (same format as
    BaseStrategy.bars_to_df).

Usage in run_multi_paper.py (per symbol, per bar):
    # At startup — fetch HTF warmup bars
    htf_ctx = HTFContext(symbol, htf_tf="4h")
    await htf_ctx.warmup(client, store)

    # Each 15M bar — update HTF (only changes every 4h candle)
    await htf_ctx.maybe_refresh(client, store, bar_ts)
    bias, bias_str = htf_ctx.get_bias()

    # After 15M signal is approved — find LTF entry
    ltf = LTFEntry(symbol, ltf_tf="5m")
    ltf_bars = await ltf.fetch(client, store)
    entry_adj = ltf.find_entry(ltf_bars, direction=sig_direction, sl_price=sl, atr=atr)
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Optional

import pandas as pd

from app.broker.base import OHLCVBar
from app.broker.bingx_client import BingXClient
from app.core.logging import get_logger
from app.data.ingestor import OHLCVIngestor, TIMEFRAME_SECONDS
from app.data.parquet_store import ParquetStore
from app.strategy.base import BaseStrategy

logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Data classes
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class HTFBias:
    """
    Directional bias from the higher timeframe.
    bias:       +1 = bullish, -1 = bearish, 0 = neutral/unclear
    strength:   0.0–1.0  (used as confidence modifier)
    reason:     human-readable string for logs/Telegram
    ema_fast:   4H EMA50 value
    ema_slow:   4H EMA200 value
    adx:        4H ADX(14)
    """
    bias: int            # +1 / -1 / 0
    strength: float      # 0.0 – 1.0
    reason: str
    ema_fast: float = 0.0
    ema_slow: float = 0.0
    adx: float = 0.0
    last_close: float = 0.0

    @property
    def label(self) -> str:
        if self.bias == 1:
            return f"4H BULL ({self.reason})"
        if self.bias == -1:
            return f"4H BEAR ({self.reason})"
        return f"4H NEUTRAL ({self.reason})"

    def is_aligned(self, direction: str) -> bool:
        """True if signal direction matches or is at least not opposed."""
        if self.bias == 0:
            return True   # Neutral → allow trade at reduced size
        return (direction == "LONG" and self.bias == 1) or \
               (direction == "SHORT" and self.bias == -1)

    def confidence_modifier(self, direction: str) -> float:
        """
        Confidence multiplier to apply to the 15M signal.
          Aligned + strong  →  +0.15 bonus
          Aligned + weak    →  +0.05 bonus
          Neutral           →  0.00
          Opposed + weak    →  -0.20 penalty
          Opposed + strong  →  -0.35 penalty
        """
        if self.bias == 0:
            return 0.0
        aligned = self.is_aligned(direction)
        if aligned:
            return 0.15 if self.strength >= 0.6 else 0.05
        else:
            return -0.35 if self.strength >= 0.6 else -0.20


@dataclass
class LTFEntryResult:
    """
    Result of LTF entry search.
    found:         True if a tighter entry was identified
    entry_price:   suggested entry (may equal original if not found)
    sl_price:      adjusted SL (tighter because entry is better)
    improvement_pct: how much better the entry is vs. original (0.0 if not found)
    reason:        description of what was found
    """
    found: bool
    entry_price: float
    sl_price: float
    improvement_pct: float
    reason: str


# ═══════════════════════════════════════════════════════════════════════════════
# HTF Context  (Higher Timeframe — 4H by default)
# ═══════════════════════════════════════════════════════════════════════════════

class HTFContext:
    """
    Fetches and caches 4H bars per symbol.
    Refreshes automatically when a new 4H candle closes.

    Bias logic (in order of priority):
      1. EMA50 vs EMA200 alignment (golden/death cross context)
      2. Price position relative to both EMAs
      3. ADX(14) for trend strength
      4. Recent candle structure (last 3 closed 4H candles)
    """

    HTF_BARS   = 300   # warmup bars (300 × 4H = 50 days)
    HTF_RECENT = 3     # candles used for structure check

    def __init__(self, symbol: str, htf_tf: str = "4h") -> None:
        self._symbol  = symbol
        self._htf_tf  = htf_tf
        self._tf_secs = TIMEFRAME_SECONDS.get(htf_tf, 14400)
        self._bars:  list[OHLCVBar] = []
        self._bias:  HTFBias = HTFBias(bias=0, strength=0.0, reason="not_loaded")
        self._last_bar_ts: datetime | None = None

    # ── Data loading ──────────────────────────────────────────────────────────

    async def warmup(self, client: BingXClient, store: ParquetStore) -> None:
        """Fetch initial HTF bar history. Call once at startup."""
        try:
            ingestor = OHLCVIngestor(client, store)
            bars = await ingestor.poll_latest(self._symbol, self._htf_tf,
                                              lookback_bars=self.HTF_BARS)
            if bars:
                self._bars = bars
                self._last_bar_ts = bars[-1].ts
                self._bias = self._compute_bias()
                logger.info("htf_warmup_done", symbol=self._symbol, tf=self._htf_tf,
                            bars=len(bars), bias=self._bias.label)
            else:
                logger.warning("htf_warmup_empty", symbol=self._symbol)
        except Exception as e:
            logger.warning("htf_warmup_failed", symbol=self._symbol, error=str(e))

    async def maybe_refresh(self, client: BingXClient, store: ParquetStore,
                             current_ts: datetime) -> bool:
        """
        Refresh HTF bars if a new 4H candle has closed since last fetch.
        Returns True if refreshed.
        """
        if self._last_bar_ts is None:
            await self.warmup(client, store)
            return True

        # Check if at least one new 4H bar should have closed
        elapsed = (current_ts - self._last_bar_ts).total_seconds()
        if elapsed < self._tf_secs:
            return False

        try:
            ingestor = OHLCVIngestor(client, store)
            new_bars = await ingestor.poll_latest(self._symbol, self._htf_tf, lookback_bars=5)
            if new_bars and new_bars[-1].ts > self._last_bar_ts:
                # Merge — keep full history up to HTF_BARS
                existing_ts = {b.ts for b in self._bars}
                fresh = [b for b in new_bars if b.ts not in existing_ts]
                if fresh:
                    self._bars = (self._bars + fresh)[-self.HTF_BARS:]
                    self._last_bar_ts = self._bars[-1].ts
                    self._bias = self._compute_bias()
                    logger.info("htf_refreshed", symbol=self._symbol,
                                new_bars=len(fresh), bias=self._bias.label)
                    return True
        except Exception as e:
            logger.warning("htf_refresh_failed", symbol=self._symbol, error=str(e))
        return False

    # ── Bias computation ──────────────────────────────────────────────────────

    def get_bias(self) -> HTFBias:
        return self._bias

    def _compute_bias(self) -> HTFBias:
        if len(self._bars) < 50:
            return HTFBias(bias=0, strength=0.0, reason="insufficient_bars")

        df = BaseStrategy.bars_to_df(self._bars)
        df = self._add_indicators(df)
        row = df.iloc[-1]

        if pd.isna(row.get("ema_fast")) or pd.isna(row.get("ema_slow")):
            return HTFBias(bias=0, strength=0.0, reason="indicators_not_ready")

        close    = float(row["close"])
        ema_fast = float(row["ema_fast"])   # EMA50
        ema_slow = float(row["ema_slow"])   # EMA200
        adx      = float(row["adx"]) if not pd.isna(row.get("adx", float("nan"))) else 0.0
        atr      = float(row["atr"]) if not pd.isna(row.get("atr", float("nan"))) else 0.0

        score    = 0.0
        reasons: list[str] = []

        # ── Factor 1: EMA50 vs EMA200 alignment ──────────────────────────────
        ema_bull = ema_fast > ema_slow
        ema_bear = ema_fast < ema_slow
        ema_sep  = abs(ema_fast - ema_slow) / ema_slow if ema_slow > 0 else 0

        if ema_bull:
            score += 0.35; reasons.append(f"EMA50>200({ema_sep:.2%})")
        elif ema_bear:
            score -= 0.35; reasons.append(f"EMA50<200({ema_sep:.2%})")

        # ── Factor 2: Price vs both EMAs ─────────────────────────────────────
        if close > ema_fast > ema_slow:
            score += 0.25; reasons.append("P>EMA50>200")
        elif close < ema_fast < ema_slow:
            score -= 0.25; reasons.append("P<EMA50<200")
        elif close > ema_slow:
            score += 0.10; reasons.append("P>EMA200")
        elif close < ema_slow:
            score -= 0.10; reasons.append("P<EMA200")

        # ── Factor 3: ADX strength ────────────────────────────────────────────
        # ADX confirms the trend is real, not ranging
        if adx >= 30:
            adx_bonus = 0.20; reasons.append(f"ADX={adx:.0f}★")
        elif adx >= 20:
            adx_bonus = 0.10; reasons.append(f"ADX={adx:.0f}✓")
        else:
            adx_bonus = 0.0; reasons.append(f"ADX={adx:.0f}✗")
        # ADX amplifies direction but doesn't set it
        if score > 0:
            score += adx_bonus
        elif score < 0:
            score -= adx_bonus

        # ── Factor 4: Recent candle structure (last HTF_RECENT bars) ─────────
        recent = df.tail(self.HTF_RECENT + 1).iloc[:-1]  # exclude current forming bar
        if len(recent) >= 2:
            closes = recent["close"].values
            bull_structure = all(closes[i] <= closes[i+1] for i in range(len(closes)-1))
            bear_structure = all(closes[i] >= closes[i+1] for i in range(len(closes)-1))
            if bull_structure:
                score += 0.10; reasons.append("rising_structure")
            elif bear_structure:
                score -= 0.10; reasons.append("falling_structure")

        # ── Determine bias ─────────────────────────────────────────────────────
        # Normalise score to [-1, 1]
        raw_score = max(-1.0, min(1.0, score))
        strength  = abs(raw_score)

        if raw_score >= 0.35:
            bias = 1
        elif raw_score <= -0.35:
            bias = -1
        else:
            bias = 0

        return HTFBias(
            bias     = bias,
            strength = round(strength, 3),
            reason   = " | ".join(reasons) if reasons else "no_signal",
            ema_fast = round(ema_fast, 4),
            ema_slow = round(ema_slow, 4),
            adx      = round(adx, 1),
            last_close = round(close, 4),
        )

    @staticmethod
    def _add_indicators(df: pd.DataFrame) -> pd.DataFrame:
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

        # EMA50 (fast) and EMA200 (slow) on 4H
        df["ema_fast"] = close.ewm(span=50,  adjust=False).mean()
        df["ema_slow"] = close.ewm(span=200, adjust=False).mean()

        return df


# ═══════════════════════════════════════════════════════════════════════════════
# LTF Entry  (Lower Timeframe — 5M by default)
# ═══════════════════════════════════════════════════════════════════════════════

class LTFEntry:
    """
    Scans recent 5M bars to find a tighter entry after a 15M setup is detected.

    Entry patterns searched (in priority order):
      LONG:
        1. Micro-pullback to 5M EMA9 (price came back to EMA9 after impulse)
        2. Inside bar / tight consolidation after impulse (low of last 3 bars > EMA9)
        3. First green 5M candle after a sequence of red candles (momentum reset)

      SHORT:
        1. Micro-pullback to 5M EMA9 (price came up to EMA9 after impulse)
        2. Inside bar / tight consolidation after impulse (high of last 3 bars < EMA9)
        3. First red 5M candle after a sequence of green candles (momentum reset)

    If a pattern is found, entry_price is tighter (closer to current price than the
    15M close) and the SL is recalculated as:
        LONG:  entry - (original_sl_distance * 0.85)   # tighter by up to 15%
        SHORT: entry + (original_sl_distance * 0.85)

    The 15% tightening is conservative — avoids over-optimising on noise.
    """

    LTF_BARS = 50    # how many 5M bars to fetch

    def __init__(self, symbol: str, ltf_tf: str = "5m") -> None:
        self._symbol = symbol
        self._ltf_tf = ltf_tf

    async def fetch(self, client: BingXClient, store: ParquetStore) -> list[OHLCVBar]:
        """Fetch the latest LTF bars."""
        try:
            ingestor = OHLCVIngestor(client, store)
            return await ingestor.poll_latest(self._symbol, self._ltf_tf,
                                              lookback_bars=self.LTF_BARS)
        except Exception as e:
            logger.warning("ltf_fetch_failed", symbol=self._symbol, error=str(e))
            return []

    def find_entry(
        self,
        bars: list[OHLCVBar],
        direction: str,        # "LONG" or "SHORT"
        sl_price: float,       # SL calculated by 15M strategy
        entry_price: float,    # current 15M close (baseline entry)
        atr_15m: float,        # ATR from 15M bar (used for quality gate)
    ) -> LTFEntryResult:
        """
        Analyse last LTF_BARS bars and return an entry recommendation.
        Always returns a result — if no pattern found, returns the original entry.
        """
        if len(bars) < 10:
            return LTFEntryResult(
                found=False, entry_price=entry_price, sl_price=sl_price,
                improvement_pct=0.0, reason="insufficient_ltf_bars",
            )

        df = BaseStrategy.bars_to_df(bars)
        df = self._add_ltf_indicators(df)

        # Only look at last 12 bars (1 hour of 5M candles)
        window = df.tail(12).reset_index(drop=True)
        if len(window) < 5:
            return LTFEntryResult(
                found=False, entry_price=entry_price, sl_price=sl_price,
                improvement_pct=0.0, reason="insufficient_window",
            )

        if direction == "LONG":
            return self._find_long_entry(window, sl_price, entry_price, atr_15m)
        else:
            return self._find_short_entry(window, sl_price, entry_price, atr_15m)

    # ── LONG entry patterns ───────────────────────────────────────────────────

    def _find_long_entry(self, w: pd.DataFrame, sl: float, orig_entry: float,
                          atr: float) -> LTFEntryResult:
        last    = w.iloc[-1]
        ema9    = float(last["ema9"]) if not pd.isna(last.get("ema9", float("nan"))) else 0.0
        close   = float(last["close"])
        low     = float(last["low"])

        # Quality gate: entry must be ABOVE the 15M SL
        def _make_result(candidate: float, reason: str) -> LTFEntryResult:
            if candidate <= sl:
                return LTFEntryResult(False, orig_entry, sl, 0.0, "entry_at_or_below_sl")
            sl_dist_orig = orig_entry - sl
            sl_dist_new  = candidate - sl
            # Don't allow SL to tighten more than 15% (avoids getting stopped by noise)
            new_sl = candidate - sl_dist_new * 0.85
            new_sl = max(new_sl, sl)   # never worse than original
            impr   = (orig_entry - candidate) / orig_entry * 100 if orig_entry > 0 else 0.0
            return LTFEntryResult(
                found=True,
                entry_price=round(candidate, 8),
                sl_price=round(new_sl, 8),
                improvement_pct=round(impr, 3),
                reason=reason,
            )

        # Pattern 1: Price touching EMA9 from above (micro-pullback)
        if ema9 > 0 and close >= ema9 * 0.9995 and close <= ema9 * 1.003:
            return _make_result(close, f"ltf_pb_ema9({ema9:.4f})")

        # Pattern 2: Inside bar / tight consolidation — last 3 bars all above EMA9
        last3 = w.tail(3)
        if ema9 > 0 and (last3["low"] > ema9).all():
            # Use the low of the last 3 bars as the reference (consolidation bottom)
            consol_low = float(last3["low"].min())
            if consol_low > sl:
                # Entry: current close (wait for breakout is implied in 15M signal)
                # Benefit: the SL can be placed tighter at the consolidation low
                new_sl    = consol_low - (atr * 0.1)
                new_sl    = max(new_sl, sl)
                impr      = (orig_entry - close) / orig_entry * 100
                return LTFEntryResult(
                    found=True,
                    entry_price=round(close, 8),
                    sl_price=round(new_sl, 8),
                    improvement_pct=round(impr, 3),
                    reason=f"ltf_consolidation(low={consol_low:.4f})",
                )

        # Pattern 3: First green bar after 2+ red bars (momentum reset)
        is_green = float(last["close"]) > float(last["open"])
        prev_red = all(
            float(w.iloc[i]["close"]) < float(w.iloc[i]["open"])
            for i in range(len(w) - 3, len(w) - 1)
        )
        if is_green and prev_red and close > sl:
            return _make_result(close, "ltf_momentum_reset_green")

        return LTFEntryResult(
            found=False, entry_price=orig_entry, sl_price=sl,
            improvement_pct=0.0, reason="ltf_no_pattern",
        )

    # ── SHORT entry patterns ──────────────────────────────────────────────────

    def _find_short_entry(self, w: pd.DataFrame, sl: float, orig_entry: float,
                           atr: float) -> LTFEntryResult:
        last  = w.iloc[-1]
        ema9  = float(last["ema9"]) if not pd.isna(last.get("ema9", float("nan"))) else 0.0
        close = float(last["close"])
        high  = float(last["high"])

        def _make_result(candidate: float, reason: str) -> LTFEntryResult:
            if candidate >= sl:
                return LTFEntryResult(False, orig_entry, sl, 0.0, "entry_at_or_above_sl")
            sl_dist_new = sl - candidate
            new_sl = candidate + sl_dist_new * 0.85
            new_sl = min(new_sl, sl)
            impr   = (candidate - orig_entry) / orig_entry * 100 if orig_entry > 0 else 0.0
            return LTFEntryResult(
                found=True,
                entry_price=round(candidate, 8),
                sl_price=round(new_sl, 8),
                improvement_pct=round(impr, 3),
                reason=reason,
            )

        # Pattern 1: Price touching EMA9 from below (micro-pullback up)
        if ema9 > 0 and close <= ema9 * 1.0005 and close >= ema9 * 0.997:
            return _make_result(close, f"ltf_pb_ema9({ema9:.4f})")

        # Pattern 2: Tight consolidation — last 3 bars all below EMA9
        last3 = w.tail(3)
        if ema9 > 0 and (last3["high"] < ema9).all():
            consol_high = float(last3["high"].max())
            if consol_high < sl:
                new_sl  = consol_high + (atr * 0.1)
                new_sl  = min(new_sl, sl)
                impr    = (close - orig_entry) / orig_entry * 100
                return LTFEntryResult(
                    found=True,
                    entry_price=round(close, 8),
                    sl_price=round(new_sl, 8),
                    improvement_pct=round(impr, 3),
                    reason=f"ltf_consolidation(high={consol_high:.4f})",
                )

        # Pattern 3: First red bar after 2+ green bars (momentum reset)
        is_red   = float(last["close"]) < float(last["open"])
        prev_grn = all(
            float(w.iloc[i]["close"]) > float(w.iloc[i]["open"])
            for i in range(len(w) - 3, len(w) - 1)
        )
        if is_red and prev_grn and close < sl:
            return _make_result(close, "ltf_momentum_reset_red")

        return LTFEntryResult(
            found=False, entry_price=orig_entry, sl_price=sl,
            improvement_pct=0.0, reason="ltf_no_pattern",
        )

    # ── Indicators ────────────────────────────────────────────────────────────

    @staticmethod
    def _add_ltf_indicators(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["ema9"]  = df["close"].ewm(span=9,  adjust=False).mean()
        df["ema21"] = df["close"].ewm(span=21, adjust=False).mean()
        return df