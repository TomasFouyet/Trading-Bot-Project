from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any

import pandas as pd

from app.strategy.signals import SignalAction
from app.strategy.trend_following_v2_simple import TrendFollowingV2Simple
from scripts.run_simple_paper import VALIDATED_PARAMS
from validation.structural_stop import build_last_pivot_arrays, compute_pivot_highs, compute_pivot_lows

from .common import (
    PYTHON_SIGNALS_CSV,
    build_verified_htf,
    compute_python_htf_bias,
    ema,
    ensure_float,
    load_reference_dataset,
)


@dataclass(slots=True)
class PythonAuditResult:
    signals: pd.DataFrame
    per_bar: pd.DataFrame


def _compute_htf_frame(df_15m: pd.DataFrame) -> pd.DataFrame:
    df_htf = build_verified_htf(df_15m)
    df_htf["htf_4h_ema50"] = ema(df_htf["close"], 50)
    df_htf["htf_bias"] = [
        compute_python_htf_bias(close_value, ema50_value)
        for close_value, ema50_value in zip(df_htf["close"], df_htf["htf_4h_ema50"], strict=False)
    ]
    return df_htf.rename(columns={"close": "htf_4h_close"})[
        ["timestamp_utc", "htf_4h_close", "htf_4h_ema50", "htf_bias"]
    ]


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


def run_python_audit(dataset_path=None) -> PythonAuditResult:
    df = load_reference_dataset(dataset_path)
    df_htf = _compute_htf_frame(df)
    strategy = TrendFollowingV2Simple("BTC-USDT", params=dict(VALIDATED_PARAMS))
    executed_trade_open = False

    per_bar_rows: list[dict[str, Any]] = []
    signal_rows: list[dict[str, Any]] = []

    for idx in range(len(df)):
        window = df.iloc[: idx + 1].copy().rename(columns={"timestamp_utc": "ts"})
        trade_state_before = strategy._trade.state
        executed_trade_open_before = executed_trade_open
        prev_long_signal_before = strategy._prev_long_signal
        prev_short_signal_before = strategy._prev_short_signal
        last_long_bar_before = strategy._last_long_bar
        last_short_bar_before = strategy._last_short_bar
        runtime_bar_index = strategy._bar_index + 1
        signals = strategy.on_bar_all(window)
        indicators_df = strategy._compute_indicators(window)
        row = indicators_df.iloc[-1]

        htf_visible = df_htf[df_htf["timestamp_utc"] <= row["ts"]]
        htf_row = htf_visible.iloc[-1] if not htf_visible.empty else pd.Series(dtype=float)
        htf_bias = htf_row.get("htf_bias", "NEUTRAL")

        pivot_lows = compute_pivot_lows(
            indicators_df["low"].astype(float).to_numpy(),
            left=VALIDATED_PARAMS["structural_pivot_left"],
            right=VALIDATED_PARAMS["structural_pivot_right"],
        )
        pivot_highs = compute_pivot_highs(
            indicators_df["high"].astype(float).to_numpy(),
            left=VALIDATED_PARAMS["structural_pivot_left"],
            right=VALIDATED_PARAMS["structural_pivot_right"],
        )
        last_pivot_low, last_pivot_high = build_last_pivot_arrays(
            pivot_lows, pivot_highs, right=VALIDATED_PARAMS["structural_pivot_right"]
        )

        common_row = {
            "bar_index": idx,
            "timestamp_utc": row["ts"],
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
            "pivot_low_used": ensure_float(last_pivot_low[-1]),
            "pivot_high_used": ensure_float(last_pivot_high[-1]),
            "signal_type": None,
            "sl_calculado": None,
            "tp_calculado": None,
            "confidence_score": None,
            "in_trade_before": trade_state_before != 0,
            "ghost_trade_active": (trade_state_before != 0) and (not executed_trade_open_before),
        }

        emitted_entry = None
        rejected_by_htf = None
        close_emitted = any(sig.action == SignalAction.CLOSE for sig in signals)
        if close_emitted and executed_trade_open:
            executed_trade_open = False
        for sig in signals:
            if sig.action not in (SignalAction.BUY, SignalAction.SELL):
                continue
            signal_type = "LONG" if sig.action == SignalAction.BUY else "SHORT"
            aligned = (
                htf_bias == "NEUTRAL"
                or (signal_type == "LONG" and htf_bias == "BULL")
                or (signal_type == "SHORT" and htf_bias == "BEAR")
            )
            if not aligned:
                rejected_by_htf = signal_type
                continue
            emitted_entry = (sig, signal_type)
            break

        if emitted_entry is not None:
            sig, signal_type = emitted_entry
            executed_trade_open = True
            signal_row = dict(common_row)
            signal_row.update(
                {
                    "signal_type": signal_type,
                    "sl_calculado": float(sig.stop_loss) if isinstance(sig.stop_loss, Decimal) else ensure_float(sig.stop_loss),
                    "tp_calculado": float(sig.take_profit) if isinstance(sig.take_profit, Decimal) else ensure_float(sig.take_profit),
                    "confidence_score": strategy._compute_confidence(row, indicators_df, signal_type),
                    "sl_mode": sig.meta.get("sl_mode"),
                    "reason": sig.reason,
                }
            )
            signal_rows.append(signal_row)
            common_row["signal_type"] = signal_type
            common_row["sl_calculado"] = signal_row["sl_calculado"]
            common_row["tp_calculado"] = signal_row["tp_calculado"]
            common_row["confidence_score"] = signal_row["confidence_score"]

        for side in ("LONG", "SHORT"):
            conditions = _evaluate_conditions(row, side, VALIDATED_PARAMS["allow_short"])
            common_row[f"{side.lower()}_conditions"] = conditions
            conf = strategy._compute_confidence(row, indicators_df, side) if all(conditions.values()) else 0.0
            signal_state = all(conditions.values()) and conf >= VALIDATED_PARAMS["min_confidence"]
            prev_state = prev_long_signal_before if side == "LONG" else prev_short_signal_before
            last_bar = last_long_bar_before if side == "LONG" else last_short_bar_before
            raw_trigger = signal_state and not prev_state
            cooldown_ok = (runtime_bar_index - last_bar) >= VALIDATED_PARAMS["sig_cooldown"]
            common_row[f"{side.lower()}_signal_state"] = signal_state
            common_row[f"{side.lower()}_raw_trigger"] = raw_trigger
            common_row[f"{side.lower()}_cooldown_ok"] = cooldown_ok
            common_row[f"{side.lower()}_trigger_pre_htf"] = raw_trigger and cooldown_ok
            common_row[f"{side.lower()}_htf_aligned"] = (
                htf_bias == "NEUTRAL"
                or (side == "LONG" and htf_bias == "BULL")
                or (side == "SHORT" and htf_bias == "BEAR")
            )
        common_row["rejected_by_htf_signal_type"] = rejected_by_htf
        per_bar_rows.append(common_row)

    signals_df = pd.DataFrame(signal_rows)
    if not signals_df.empty:
        signals_df.to_csv(PYTHON_SIGNALS_CSV, index=False, date_format="%Y-%m-%dT%H:%M:%SZ")
    else:
        pd.DataFrame(columns=[]).to_csv(PYTHON_SIGNALS_CSV, index=False)
    return PythonAuditResult(signals=signals_df, per_bar=pd.DataFrame(per_bar_rows))
