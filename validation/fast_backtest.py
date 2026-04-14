"""
Fast vectorized backtester for TrendFollowingV2Simple.

Pre-computes all indicators ONCE, then simulates entry/exit logic
in a single pass. ~100x faster than bar-by-bar StrategyAdapter.

This is specific to the V2Simple entry logic (EMA+ADX+MACD+pullback)
with fixed SL/TP exits. Not a general-purpose adapter.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from validation.strategy_adapter import BacktestMetrics, TradeRecord


def compute_indicators(
    df: pd.DataFrame,
    ema_fast_p: int = 20,
    ema_slow_p: int = 50,
    slope_bars: int = 5,
) -> pd.DataFrame:
    """Pre-compute all indicators on the full DataFrame."""
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
    df["ema_fast"]       = close.ewm(span=ema_fast_p, adjust=False).mean()
    df["ema_slow"]       = close.ewm(span=ema_slow_p, adjust=False).mean()
    df["ema_slow_slope"] = df["ema_slow"].diff(slope_bars)
    df["ema_fast_slope"] = df["ema_fast"].diff(slope_bars)

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


def compute_htf_bias(
    df_15m: pd.DataFrame,
    htf_ema_period: int = 50,
) -> np.ndarray:
    """
    Compute 4H trend bias aligned to 15m bars.

    Returns array of same length as df_15m:
      +1 = bullish (4H close > 4H EMA50)
      -1 = bearish (4H close < 4H EMA50)
       0 = neutral / not enough data
    """
    # Resample 15m → 4H
    df_htf = df_15m.set_index("ts").resample("4h").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }).dropna()

    df_htf["ema_htf"] = df_htf["close"].ewm(span=htf_ema_period, adjust=False).mean()
    df_htf["bias"] = 0
    df_htf.loc[df_htf["close"] > df_htf["ema_htf"], "bias"] = 1
    df_htf.loc[df_htf["close"] < df_htf["ema_htf"], "bias"] = -1

    # Forward-fill 4H bias to 15m bars
    bias_htf = df_htf[["bias"]].rename(columns={"bias": "htf_bias"})
    df_aligned = df_15m[["ts"]].copy().set_index("ts")
    df_aligned = df_aligned.join(bias_htf, how="left")
    df_aligned["htf_bias"] = df_aligned["htf_bias"].ffill().fillna(0)

    return df_aligned["htf_bias"].values.astype(np.int8)


def fast_backtest(
    df: pd.DataFrame,
    adx_min: float = 20.0,
    ema_fast_p: int = 20,
    ema_slow_p: int = 50,
    rr_ratio: float = 2.0,
    atr_sl_mult: float = 1.5,
    pb_tol_atr: float = 1.0,
    sig_cooldown: int = 5,
    allow_short: bool = True,
    min_confidence: float = 0.0,
    adx_strong: float = 35.0,
    slope_bars: int = 5,
    precomputed: bool = False,
    htf_bias: np.ndarray | None = None,
    regime_labels: np.ndarray | None = None,
    regime_skip: set | None = None,
    regime_rr_map: dict | None = None,
) -> BacktestMetrics:
    """
    Run a fast single-pass backtest.

    If precomputed=True, assumes df already has indicator columns.

    Optional regime parameters:
        regime_labels: per-bar regime index (0-3), same length as df
        regime_skip: set of regime indices to skip (no entry)
        regime_rr_map: dict mapping regime index → rr_ratio override
    Otherwise computes them first.
    """
    if not precomputed:
        df = compute_indicators(df, ema_fast_p, ema_slow_p, slope_bars)

    min_bars = max(60, ema_slow_p + 20)

    # Extract arrays for speed
    close  = df["close"].values.astype(np.float64)
    high   = df["high"].values.astype(np.float64)
    low    = df["low"].values.astype(np.float64)
    opn    = df["open"].values.astype(np.float64)
    atr    = df["atr"].values.astype(np.float64)
    adx    = df["adx"].values.astype(np.float64)
    ema_f  = df["ema_fast"].values.astype(np.float64)
    ema_s  = df["ema_slow"].values.astype(np.float64)
    ema_s_slope = df["ema_slow_slope"].values.astype(np.float64)
    ema_f_slope = df["ema_fast_slope"].values.astype(np.float64)
    macd_v = df["macd"].values.astype(np.float64)
    macd_sig = df["macd_signal"].values.astype(np.float64)
    macd_hist = df["macd_hist"].values.astype(np.float64)
    volume = df["volume"].values.astype(np.float64)
    vol_sma = df["vol_sma"].values.astype(np.float64)
    n = len(df)

    # State
    trades: list[TradeRecord] = []
    in_trade = False
    trade_dir = 0   # 1=long, -1=short
    entry_price = 0.0
    sl_price = 0.0
    tp_price = 0.0
    entry_bar = 0

    prev_long_sig = False
    prev_short_sig = False
    last_long_bar = -999
    last_short_bar = -999

    for i in range(min_bars, n):
        c = close[i]
        h = high[i]
        lo = low[i]
        a = atr[i]
        dx = adx[i]

        if np.isnan(dx) or np.isnan(ema_s[i]) or a <= 0:
            continue

        # ── Check exits first ─────────────────────────────────────
        if in_trade:
            sl_hit = (trade_dir == 1 and lo <= sl_price) or \
                     (trade_dir == -1 and h >= sl_price)
            tp_hit = (trade_dir == 1 and h >= tp_price) or \
                     (trade_dir == -1 and lo <= tp_price)

            if sl_hit:
                pnl = _pnl(trade_dir, entry_price, sl_price)
                trades.append(TradeRecord(
                    direction="LONG" if trade_dir == 1 else "SHORT",
                    entry_price=entry_price,
                    exit_price=sl_price,
                    pnl_pct=pnl,
                    exit_type="sl",
                    bars_held=i - entry_bar,
                    entry_bar_idx=entry_bar,
                ))
                in_trade = False
                continue
            elif tp_hit:
                pnl = _pnl(trade_dir, entry_price, tp_price)
                trades.append(TradeRecord(
                    direction="LONG" if trade_dir == 1 else "SHORT",
                    entry_price=entry_price,
                    exit_price=tp_price,
                    pnl_pct=pnl,
                    exit_type="tp",
                    bars_held=i - entry_bar,
                    entry_bar_idx=entry_bar,
                ))
                in_trade = False
                continue

        # ── Entry conditions (IDENTICAL to V2) ────────────────────
        pb_zone    = abs(c - ema_f[i]) < a * pb_tol_atr
        sl_rising  = ema_s_slope[i] > 0 if not np.isnan(ema_s_slope[i]) else False
        sl_falling = ema_s_slope[i] < 0 if not np.isnan(ema_s_slope[i]) else False
        p_above    = c > ema_s[i]
        p_below    = c < ema_s[i]
        m_bull     = macd_v[i] > macd_sig[i]
        m_bear     = macd_v[i] < macd_sig[i]
        c_bull     = c > opn[i]
        c_bear     = c < opn[i]
        adx_ok     = dx >= adx_min

        long_base  = adx_ok and sl_rising and p_above and m_bull and pb_zone and c_bull
        short_base = adx_ok and sl_falling and p_below and m_bear and pb_zone and c_bear and allow_short

        # Confidence (simplified — just check min_confidence gate)
        conf_l = _fast_confidence(dx, adx_strong, a, c, ema_f[i],
                                  macd_hist[i], macd_hist[i-1] if i > 0 else 0,
                                  "LONG", opn[i], h, lo,
                                  ema_f_slope[i], volume[i], vol_sma[i]) if long_base else 0.0
        conf_s = _fast_confidence(dx, adx_strong, a, c, ema_f[i],
                                  macd_hist[i], macd_hist[i-1] if i > 0 else 0,
                                  "SHORT", opn[i], h, lo,
                                  ema_f_slope[i], volume[i], vol_sma[i]) if short_base else 0.0

        long_signal  = long_base  and conf_l >= min_confidence
        short_signal = short_base and conf_s >= min_confidence

        # Edge detection
        long_trigger_raw  = long_signal  and not prev_long_sig
        short_trigger_raw = short_signal and not prev_short_sig
        prev_long_sig  = long_signal
        prev_short_sig = short_signal

        # Cooldown
        long_trigger  = long_trigger_raw  and (i - last_long_bar)  >= sig_cooldown
        short_trigger = short_trigger_raw and (i - last_short_bar) >= sig_cooldown
        if long_trigger:
            last_long_bar = i
        if short_trigger:
            last_short_bar = i

        # ── HTF filter (if provided) ─────────────────────────────
        if htf_bias is not None:
            bias = htf_bias[i]
            if long_trigger and bias == -1:
                long_trigger = False   # skip LONG when HTF bearish
            if short_trigger and bias == 1:
                short_trigger = False  # skip SHORT when HTF bullish

        # ── Regime filter (if provided) ──────────────────────────
        if regime_labels is not None and (long_trigger or short_trigger):
            bar_regime = regime_labels[i]
            if regime_skip and bar_regime in regime_skip:
                long_trigger = False
                short_trigger = False

        # ── Open trade if flat ────────────────────────────────────
        if not in_trade:
            # Determine rr for this trade (regime override or default)
            effective_rr = rr_ratio
            if regime_rr_map is not None and regime_labels is not None:
                bar_regime = regime_labels[i]
                if bar_regime in regime_rr_map:
                    effective_rr = regime_rr_map[bar_regime]

            if long_trigger:
                sl_dist = a * atr_sl_mult
                entry_price = c
                sl_price = c - sl_dist
                tp_price = c + sl_dist * effective_rr
                trade_dir = 1
                entry_bar = i
                in_trade = True
            elif short_trigger:
                sl_dist = a * atr_sl_mult
                entry_price = c
                sl_price = c + sl_dist
                tp_price = c - sl_dist * effective_rr
                trade_dir = -1
                entry_bar = i
                in_trade = True

    # Close open trade at end
    if in_trade:
        pnl = _pnl(trade_dir, entry_price, close[-1])
        trades.append(TradeRecord(
            direction="LONG" if trade_dir == 1 else "SHORT",
            entry_price=entry_price,
            exit_price=float(close[-1]),
            pnl_pct=pnl,
            exit_type="end_of_data",
            bars_held=n - 1 - entry_bar,
            entry_bar_idx=entry_bar,
        ))

    return _build_metrics(trades, df)


def _pnl(direction: int, entry: float, exit_p: float) -> float:
    if direction == 1:
        return (exit_p - entry) / entry * 100
    else:
        return (entry - exit_p) / entry * 100


def _fast_confidence(
    adx: float, adx_strong: float, atr: float, close: float, ema_f: float,
    hist: float, prev_hist: float, direction: str,
    opn: float, high: float, low: float,
    ema_f_slope: float, volume: float, vol_sma: float,
) -> float:
    """Fast confidence scoring — identical logic to V2."""
    sc = 0.0

    if adx >= adx_strong:
        sc += 0.20
    elif adx >= 30:
        sc += 0.10

    pb_atr = abs(close - ema_f) / atr if atr > 0 else 999
    if pb_atr <= 0.5:
        sc += 0.20
    elif pb_atr <= 1.0:
        sc += 0.10

    if not np.isnan(hist) and not np.isnan(prev_hist):
        if direction == "LONG" and hist > 0 and hist > prev_hist:
            sc += 0.15
        elif direction == "SHORT" and hist < 0 and hist < prev_hist:
            sc += 0.15
        elif (direction == "LONG" and hist > 0) or (direction == "SHORT" and hist < 0):
            sc += 0.05

    body = abs(close - opn)
    rng = high - low
    ratio = body / rng if rng > 0 else 0
    if ratio >= 0.60:
        sc += 0.15
    elif ratio >= 0.40:
        sc += 0.07

    if not np.isnan(ema_f_slope):
        if direction == "LONG" and ema_f_slope > 0:
            sc += 0.15
        elif direction == "SHORT" and ema_f_slope < 0:
            sc += 0.15

    vol_r = volume / vol_sma if vol_sma > 0 and not np.isnan(vol_sma) else 0
    if vol_r >= 1.2:
        sc += 0.15
    elif vol_r >= 0.8:
        sc += 0.05

    return min(sc, 1.0)


def _build_metrics(trades: list[TradeRecord], df: pd.DataFrame) -> BacktestMetrics:
    """Build BacktestMetrics from trade list."""
    m = BacktestMetrics()
    m.trades = [t.to_dict() for t in trades]
    m.total_trades = len(trades)

    if not trades:
        return m

    pnls = np.array([t.pnl_pct for t in trades])
    m.winning_trades = int(np.sum(pnls > 0))
    m.losing_trades = int(np.sum(pnls <= 0))
    m.winrate = m.winning_trades / m.total_trades * 100
    m.total_pnl_pct = float(np.sum(pnls))
    m.avg_pnl_pct = float(np.mean(pnls))

    # Equity curve & drawdown (compounded)
    mult = 1.0 + pnls / 100.0
    equity = np.concatenate([[1.0], np.cumprod(mult)])
    running_max = np.maximum.accumulate(equity)
    drawdowns = (equity - running_max) / running_max  # percentage DD
    m.max_drawdown_pct = abs(float(np.min(drawdowns))) * 100

    # Annualized return (compounded)
    ts_start = pd.Timestamp(df["ts"].iloc[0])
    ts_end = pd.Timestamp(df["ts"].iloc[-1])
    days = max((ts_end - ts_start).total_seconds() / 86400, 1)
    total_return = equity[-1] / equity[0]  # uses compounded equity from DD calc
    m.annual_return_pct = (total_return ** (365.0 / days) - 1) * 100

    # Sharpe
    if len(pnls) >= 2:
        mean_r = np.mean(pnls)
        std_r = np.std(pnls, ddof=1)
        if std_r > 0:
            avg_bars = np.mean([t.bars_held for t in trades])
            delta = (pd.Timestamp(df["ts"].iloc[1]) - pd.Timestamp(df["ts"].iloc[0])).total_seconds()
            tf_min = max(delta / 60, 1)
            bars_per_year = 365 * 24 * 60 / tf_min
            trades_per_year = bars_per_year / max(avg_bars, 1)
            m.sharpe_ratio = float((mean_r / std_r) * np.sqrt(trades_per_year))

    return m
