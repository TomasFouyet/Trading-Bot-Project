from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from validation.structural_stop import build_last_pivot_arrays, compute_pivot_highs, compute_pivot_lows, compute_structural_sl

from .common import (
    PINE_SIGNALS_CSV,
    VALIDATED_PARAMS,
    build_verified_htf,
    compute_pine_htf_bias,
    ema,
    ensure_float,
    load_reference_dataset,
    pine_adx,
    pine_atr,
)


@dataclass(slots=True)
class PineAuditResult:
    signals: pd.DataFrame
    per_bar: pd.DataFrame


def _indicator_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["atr"] = pine_atr(out, 14)
    _, _, adx = pine_adx(out, 14)
    out["adx"] = adx
    out["ema_fast"] = ema(out["close"], VALIDATED_PARAMS["ema_fast"])
    out["ema_slow"] = ema(out["close"], VALIDATED_PARAMS["ema_slow"])
    out["ema_fast_slope"] = out["ema_fast"] - out["ema_fast"].shift(VALIDATED_PARAMS["slope_bars"])
    out["ema_slow_slope"] = out["ema_slow"] - out["ema_slow"].shift(VALIDATED_PARAMS["slope_bars"])
    ema12 = ema(out["close"], 12)
    ema26 = ema(out["close"], 26)
    out["macd"] = ema12 - ema26
    out["macd_signal"] = ema(out["macd"], 9)
    out["macd_hist"] = out["macd"] - out["macd_signal"]
    out["vol_sma"] = out["volume"].rolling(20).mean()
    return out


def _htf_frame(df_15m: pd.DataFrame) -> pd.DataFrame:
    df_htf = build_verified_htf(df_15m)
    df_htf["htf_4h_ema50"] = ema(df_htf["close"], 50)
    df_htf["htf_bias"] = [
        compute_pine_htf_bias(close_value, ema50_value)
        for close_value, ema50_value in zip(df_htf["close"], df_htf["htf_4h_ema50"], strict=False)
    ]
    return df_htf.rename(columns={"close": "htf_4h_close"})[
        ["timestamp_utc", "htf_4h_close", "htf_4h_ema50", "htf_bias"]
    ]


def _compute_confidence(df: pd.DataFrame, idx: int, direction: str) -> float:
    row = df.iloc[idx]
    prev_hist = float(df.iloc[idx - 1]["macd_hist"]) if idx > 0 and not pd.isna(df.iloc[idx - 1]["macd_hist"]) else 0.0
    score = 0.0
    adx = float(row["adx"]) if not pd.isna(row["adx"]) else 0.0
    atr = float(row["atr"]) if not pd.isna(row["atr"]) else 0.0
    ema_fast = float(row["ema_fast"]) if not pd.isna(row["ema_fast"]) else 0.0
    close = float(row["close"])

    if adx >= VALIDATED_PARAMS["adx_strong"]:
        score += 0.20
    elif adx >= 30:
        score += 0.10

    pb_atr = abs(close - ema_fast) / atr if atr > 0 else 999.0
    if pb_atr <= 0.5:
        score += 0.20
    elif pb_atr <= 1.0:
        score += 0.10

    hist = float(row["macd_hist"]) if not pd.isna(row["macd_hist"]) else 0.0
    if direction == "LONG" and hist > 0 and hist > prev_hist:
        score += 0.15
    elif direction == "SHORT" and hist < 0 and hist < prev_hist:
        score += 0.15
    elif (direction == "LONG" and hist > 0) or (direction == "SHORT" and hist < 0):
        score += 0.05

    rng = float(row["high"]) - float(row["low"])
    body_ratio = abs(float(row["close"]) - float(row["open"])) / rng if rng > 0 else 0.0
    if body_ratio >= 0.60:
        score += 0.15
    elif body_ratio >= 0.40:
        score += 0.07

    ema_fast_slope = float(row["ema_fast_slope"]) if not pd.isna(row["ema_fast_slope"]) else 0.0
    if direction == "LONG" and ema_fast_slope > 0:
        score += 0.15
    elif direction == "SHORT" and ema_fast_slope < 0:
        score += 0.15

    vol_sma = float(row["vol_sma"]) if not pd.isna(row["vol_sma"]) else 0.0
    vol_ratio = float(row["volume"]) / vol_sma if vol_sma > 0 else 0.0
    if vol_ratio >= 1.2:
        score += 0.15
    elif vol_ratio >= 0.8:
        score += 0.05

    return round(min(score, 1.0), 3)


def _evaluate_conditions(row: pd.Series, signal_type: str, allow_short: bool) -> dict[str, bool]:
    close = float(row["close"])
    atr = float(row["atr"]) if not pd.isna(row["atr"]) else 0.0
    ema_fast = float(row["ema_fast"]) if not pd.isna(row["ema_fast"]) else 0.0
    ema_slow = float(row["ema_slow"]) if not pd.isna(row["ema_slow"]) else 0.0
    adx = float(row["adx"]) if not pd.isna(row["adx"]) else 0.0
    slope = float(row["ema_slow_slope"]) if not pd.isna(row["ema_slow_slope"]) else 0.0
    macd = float(row["macd"]) if not pd.isna(row["macd"]) else 0.0
    macd_signal = float(row["macd_signal"]) if not pd.isna(row["macd_signal"]) else 0.0
    is_long = signal_type == "LONG"
    return {
        "adx_ok": adx >= VALIDATED_PARAMS["adx_min"],
        "slope_ok": slope > 0 if is_long else slope < 0,
        "price_ok": close > ema_slow if is_long else close < ema_slow,
        "macd_ok": macd > macd_signal if is_long else macd < macd_signal,
        "pullback_ok": abs(close - ema_fast) < atr * VALIDATED_PARAMS["pullback_tolerance_atr"] if atr > 0 else False,
        "candle_ok": close > float(row["open"]) if is_long else close < float(row["open"]),
        "allow_short_ok": True if is_long else allow_short,
    }


def run_pine_replica(dataset_path=None) -> PineAuditResult:
    df = load_reference_dataset(dataset_path)
    ind = _indicator_frame(df)
    htf = _htf_frame(df)

    pivot_lows = compute_pivot_lows(
        ind["low"].astype(float).to_numpy(),
        left=VALIDATED_PARAMS["structural_pivot_left"],
        right=VALIDATED_PARAMS["structural_pivot_right"],
    )
    pivot_highs = compute_pivot_highs(
        ind["high"].astype(float).to_numpy(),
        left=VALIDATED_PARAMS["structural_pivot_left"],
        right=VALIDATED_PARAMS["structural_pivot_right"],
    )
    last_pivot_low, last_pivot_high = build_last_pivot_arrays(
        pivot_lows, pivot_highs, right=VALIDATED_PARAMS["structural_pivot_right"]
    )

    signal_rows: list[dict[str, Any]] = []
    per_bar_rows: list[dict[str, Any]] = []

    last_long_bar = -999
    last_short_bar = -999
    prev_long_signal = False
    prev_short_signal = False
    in_trade = False
    trade_state = 0
    trade_sl = None
    trade_tp = None

    for idx in range(len(ind)):
        row = ind.iloc[idx]
        in_trade_before = in_trade

        if in_trade:
            bar_high = float(row["high"])
            bar_low = float(row["low"])
            sl_hit = (trade_state == 1 and bar_low <= trade_sl) or (trade_state == -1 and bar_high >= trade_sl)
            tp_hit = (trade_state == 1 and bar_high >= trade_tp) or (trade_state == -1 and bar_low <= trade_tp)
            if sl_hit or tp_hit:
                in_trade = False
                trade_state = 0
                trade_sl = None
                trade_tp = None

        htf_visible = htf[htf["timestamp_utc"] <= row["timestamp_utc"]]
        htf_row = htf_visible.iloc[-1] if not htf_visible.empty else pd.Series(dtype=float)
        htf_bias = htf_row.get("htf_bias", "NEUTRAL")

        conditions_long = _evaluate_conditions(row, "LONG", VALIDATED_PARAMS["allow_short"])
        conditions_short = _evaluate_conditions(row, "SHORT", VALIDATED_PARAMS["allow_short"])
        indicators_ready = idx + 1 >= max(60, VALIDATED_PARAMS["ema_slow"] + 20)
        conf_long = _compute_confidence(ind, idx, "LONG") if indicators_ready and all(conditions_long.values()) else 0.0
        conf_short = _compute_confidence(ind, idx, "SHORT") if indicators_ready and all(conditions_short.values()) else 0.0
        long_signal = indicators_ready and all(conditions_long.values()) and conf_long >= VALIDATED_PARAMS["min_confidence"]
        short_signal = indicators_ready and all(conditions_short.values()) and conf_short >= VALIDATED_PARAMS["min_confidence"]

        long_trigger_raw = long_signal and not prev_long_signal
        short_trigger_raw = short_signal and not prev_short_signal
        prev_long_signal = long_signal
        prev_short_signal = short_signal

        long_cooldown_ok = (idx - last_long_bar) >= VALIDATED_PARAMS["sig_cooldown"]
        short_cooldown_ok = (idx - last_short_bar) >= VALIDATED_PARAMS["sig_cooldown"]
        long_trigger = long_trigger_raw and long_cooldown_ok
        short_trigger = short_trigger_raw and short_cooldown_ok
        if long_trigger:
            last_long_bar = idx
        if short_trigger:
            last_short_bar = idx

        if long_trigger and htf_bias == "BEAR":
            long_trigger = False
        if short_trigger and htf_bias == "BULL":
            short_trigger = False

        common_row = {
            "bar_index": idx,
            "timestamp_utc": row["timestamp_utc"],
            "close": float(row["close"]),
            "ema20": ensure_float(row["ema_fast"]),
            "ema50": ensure_float(row["ema_slow"]),
            "ema20_slope": ensure_float(row["ema_fast_slope"]),
            "ema50_slope": ensure_float(row["ema_slow_slope"]),
            "atr14": ensure_float(row["atr"]),
            "adx14": ensure_float(row["adx"]),
            "macd_line": ensure_float(row["macd"]),
            "macd_signal": ensure_float(row["macd_signal"]),
            "macd_hist": ensure_float(row["macd_hist"]),
            "pullback_atr_distance": (
                abs(float(row["close"]) - float(row["ema_fast"])) / float(row["atr"])
                if not pd.isna(row["atr"]) and float(row["atr"]) > 0
                else None
            ),
            "body_ratio": (
                abs(float(row["close"]) - float(row["open"])) / (float(row["high"]) - float(row["low"]))
                if float(row["high"]) > float(row["low"])
                else 0.0
            ),
            "volume_ratio": (
                float(row["volume"]) / float(row["vol_sma"])
                if not pd.isna(row["vol_sma"]) and float(row["vol_sma"]) > 0
                else None
            ),
            "htf_4h_close": ensure_float(htf_row.get("htf_4h_close")),
            "htf_4h_ema50": ensure_float(htf_row.get("htf_4h_ema50")),
            "htf_bias": htf_bias,
            "pivot_low_used": ensure_float(last_pivot_low[idx]),
            "pivot_high_used": ensure_float(last_pivot_high[idx]),
            "signal_type": None,
            "sl_calculado": None,
            "tp_calculado": None,
            "confidence_score": None,
            "in_trade_before": in_trade_before,
            "long_conditions": conditions_long,
            "short_conditions": conditions_short,
            "long_signal_state": long_signal,
            "short_signal_state": short_signal,
            "long_raw_trigger": long_trigger_raw,
            "short_raw_trigger": short_trigger_raw,
            "long_cooldown_ok": long_cooldown_ok,
            "short_cooldown_ok": short_cooldown_ok,
            "long_trigger_pre_htf": long_trigger_raw and long_cooldown_ok,
            "short_trigger_pre_htf": short_trigger_raw and short_cooldown_ok,
            "long_htf_aligned": htf_bias in ("BULL", "NEUTRAL"),
            "short_htf_aligned": htf_bias in ("BEAR", "NEUTRAL"),
        }

        if not in_trade and (long_trigger or short_trigger):
            signal_type = "LONG" if long_trigger else "SHORT"
            conf = conf_long if long_trigger else conf_short
            atr = float(row["atr"]) if not pd.isna(row["atr"]) else 0.0
            close = float(row["close"])
            stop_loss, _sl_mode = compute_structural_sl(
                entry_price=close,
                direction=signal_type,
                bar_idx=idx,
                last_pivot_low=last_pivot_low,
                last_pivot_high=last_pivot_high,
                atr=atr,
                stop_mode="STRUCTURAL",
                atr_sl_mult=VALIDATED_PARAMS["atr_sl_mult"],
                buffer_atr=VALIDATED_PARAMS["structural_buffer_atr"],
                min_risk_atr=VALIDATED_PARAMS["structural_min_risk_atr"],
            )
            risk = abs(close - stop_loss)
            take_profit = close + risk * VALIDATED_PARAMS["rr_ratio"] if signal_type == "LONG" else close - risk * VALIDATED_PARAMS["rr_ratio"]
            row_out = dict(common_row)
            row_out.update(
                {
                    "signal_type": signal_type,
                    "sl_calculado": stop_loss,
                    "tp_calculado": take_profit,
                    "confidence_score": conf,
                    "sl_mode": _sl_mode,
                }
            )
            signal_rows.append(row_out)
            common_row["signal_type"] = signal_type
            common_row["sl_calculado"] = stop_loss
            common_row["tp_calculado"] = take_profit
            common_row["confidence_score"] = conf
            in_trade = True
            trade_state = 1 if signal_type == "LONG" else -1
            trade_sl = stop_loss
            trade_tp = take_profit

        per_bar_rows.append(common_row)

    signals_df = pd.DataFrame(signal_rows)
    if not signals_df.empty:
        signals_df.to_csv(PINE_SIGNALS_CSV, index=False, date_format="%Y-%m-%dT%H:%M:%SZ")
    else:
        pd.DataFrame(columns=[]).to_csv(PINE_SIGNALS_CSV, index=False)
    return PineAuditResult(signals=signals_df, per_bar=pd.DataFrame(per_bar_rows))
