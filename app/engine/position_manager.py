"""
Unified position manager for SL / TP / trailing / breakeven logic.

Single source of truth used by:
  - BacktestEngine (backtest.py)
  - run_multi_paper.py (live & paper trading)

This ensures that backtest results match live/paper behaviour exactly.

Design:
  - Pure logic, no I/O: takes bar data + trade dict, returns actions.
  - The caller (backtest or live script) handles order execution.
  - Trade dicts follow the same schema everywhere.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any


# ── Commission & breakeven constants ─────────────────────────────────────────
COMMISSION_RATE = Decimal("0.00075")   # per-leg (entry + exit each pay this)
BE_BUFFER       = Decimal("0.0002")    # extra 2 bps above fees


@dataclass
class ExitAction:
    """Returned by check_exit() when a position should be (partially) closed."""
    exit_type: str          # "sl", "be_sl", "trailing_sl", "tp1", "tp2"
    trigger_price: float    # price at which the exit triggers
    close_qty: Decimal      # quantity to close
    close_pct: float        # fraction of original qty this represents
    tp1_updates: dict | None = None  # if tp1 hit: updates to apply to trade dict


def create_trade(
    *,
    trade_id: str,
    symbol: str,
    side: str,                  # "LONG" or "SHORT"
    entry_price: Decimal,
    qty: Decimal,
    fee_in: Decimal,
    sl_price: float | None,
    tp1_price: float | None,
    tp2_price: float | None,
    tp1_close_pct: float = 0.33,
    soft_sl_bars: int = 0,
    trailing_atr: float = 1.0,
    current_atr: float = 0.0,
    signal_reason: str = "",
    extra: dict | None = None,
) -> dict:
    """Create a standardised trade dict. Used by both backtest and live."""
    trade: dict[str, Any] = {
        "trade_id":           trade_id,
        "symbol":             symbol,
        "side":               side,
        "entry_price":        entry_price,
        "qty":                qty,
        "original_qty":       qty,
        "fee_in":             fee_in,
        "total_partial_pnl":  Decimal("0"),
        "total_partial_fees": Decimal("0"),
        "signal_reason":      signal_reason,
        # SL / TP levels
        "sl_price":           sl_price,
        "tp1_price":          tp1_price,
        "tp2_price":          tp2_price,
        # State
        "tp1_hit":            False,
        "tp1_close_pct":      tp1_close_pct,
        "soft_sl_bars":       soft_sl_bars,
        "bars_in_trade":      0,
        "trailing_atr":       trailing_atr,
        "current_atr":        current_atr,
        "best_price":         float(entry_price),
    }
    if extra:
        trade.update(extra)
    return trade


def advance_bar(trade: dict, bar_high: float, bar_low: float) -> None:
    """
    Call once per bar BEFORE check_exit().
    Updates bars_in_trade and trailing stop.
    """
    trade["bars_in_trade"] = trade.get("bars_in_trade", 0) + 1
    _update_trailing(trade, bar_high, bar_low)


def update_atr(trade: dict, atr: float) -> None:
    """Update the current ATR from the latest bar's indicators."""
    if atr > 0:
        trade["current_atr"] = atr


def _update_trailing(trade: dict, bar_high: float, bar_low: float) -> None:
    """Advance trailing stop after TP1 has been hit."""
    if not trade.get("tp1_hit"):
        return

    trail_atr = trade.get("trailing_atr", 1.0)
    cur_atr   = trade.get("current_atr", 0.0)
    if cur_atr <= 0:
        return

    il = trade["side"] == "LONG"
    sl = float(trade["sl_price"]) if trade.get("sl_price") is not None else None

    if il:
        best = max(trade.get("best_price", float(trade["entry_price"])), bar_high)
        trade["best_price"] = best
        new_sl = best - cur_atr * trail_atr
        if sl is None or new_sl > sl:
            trade["sl_price"] = new_sl
    else:
        best = min(trade.get("best_price", float(trade["entry_price"])), bar_low)
        trade["best_price"] = best
        new_sl = best + cur_atr * trail_atr
        if sl is None or new_sl < sl:
            trade["sl_price"] = new_sl


def check_exit(
    trade: dict,
    bar_high: float,
    bar_low: float,
    bar_close: float,
) -> ExitAction | None:
    """
    Check if the current bar triggers SL, TP1, or TP2.

    Priority: SL > TP (conservative — if both trigger on same bar, SL wins).

    Patience timer: during the first `soft_sl_bars` bars, SL triggers on bar
    CLOSE instead of bar wick (avoids wick-out in first N bars).

    Returns ExitAction or None.
    """
    il = trade["side"] == "LONG"

    sl  = float(trade["sl_price"])  if trade.get("sl_price")  is not None else None
    t1  = float(trade["tp1_price"]) if trade.get("tp1_price") is not None else None
    t2  = float(trade["tp2_price"]) if trade.get("tp2_price") is not None else None
    t1h = trade.get("tp1_hit", False)

    soft = trade.get("soft_sl_bars", 0)
    bit  = trade["bars_in_trade"]
    in_patience = soft > 0 and bit <= soft

    # ── Trigger detection ────────────────────────────────────────────────
    if il:
        price_for_sl = bar_close if in_patience else bar_low
        sl_hit = sl is not None and price_for_sl <= sl
        t1_hit = t1 is not None and not t1h and bar_high >= t1 and not sl_hit
        t2_hit = t2 is not None and t1h     and bar_high >= t2 and not sl_hit
    else:
        price_for_sl = bar_close if in_patience else bar_high
        sl_hit = sl is not None and price_for_sl >= sl
        t1_hit = t1 is not None and not t1h and bar_low <= t1 and not sl_hit
        t2_hit = t2 is not None and t1h     and bar_low <= t2 and not sl_hit

    # ── Build exit action ────────────────────────────────────────────────
    if sl_hit:
        exit_type = "trailing_sl" if t1h else ("be_sl" if _is_breakeven(trade) else "sl")
        # Fill at SL price, but can't fill beyond the bar range
        if il:
            trigger_price = max(sl, bar_low)
        else:
            trigger_price = min(sl, bar_high)
        return ExitAction(
            exit_type=exit_type,
            trigger_price=trigger_price,
            close_qty=trade["qty"],
            close_pct=1.0,
        )

    if t1_hit:
        cp = float(trade.get("tp1_close_pct", 0.33))
        cq = min(
            (trade["qty"] * Decimal(str(cp))).quantize(Decimal("0.001")),
            trade["qty"],
        )
        if cq <= Decimal("0"):
            return None

        # Compute TP1 post-hit updates (BE stop, best_price init, patience off)
        entry = trade["entry_price"]
        fee_buffer = entry * (COMMISSION_RATE * 2 + BE_BUFFER)
        if il:
            new_sl = float(entry + fee_buffer)
        else:
            new_sl = float(entry - fee_buffer)

        return ExitAction(
            exit_type="tp1",
            trigger_price=t1,
            close_qty=cq,
            close_pct=cp,
            tp1_updates={
                "tp1_hit": True,
                "tp1_price": None,
                "sl_price": new_sl,
                "best_price": t1,       # start trailing from TP1 fill
                "soft_sl_bars": 0,       # disable patience — trailing uses wicks
            },
        )

    if t2_hit:
        return ExitAction(
            exit_type="tp2",
            trigger_price=t2,
            close_qty=trade["qty"],
            close_pct=1.0,
        )

    return None


def apply_tp1_updates(trade: dict, updates: dict) -> None:
    """Apply the tp1_updates dict returned by check_exit() to the trade."""
    trade.update(updates)


def compute_leg_pnl(
    trade: dict,
    exit_price: Decimal,
    closed_qty: Decimal,
    fee_out: Decimal,
) -> Decimal:
    """Compute net PnL for a single exit leg."""
    il = trade["side"] == "LONG"
    entry = trade["entry_price"]
    price_diff = (exit_price - entry) if il else (entry - exit_price)
    gross = price_diff * closed_qty
    pct_of_orig = closed_qty / trade["original_qty"] if trade["original_qty"] > 0 else Decimal("0")
    fee_in_part = trade["fee_in"] * pct_of_orig
    return gross - fee_in_part - fee_out


def _is_breakeven(trade: dict) -> bool:
    """True if the current SL is at or above the breakeven threshold."""
    entry = float(trade["entry_price"])
    sl    = float(trade.get("sl_price", 0))
    il    = trade["side"] == "LONG"
    be_threshold = entry * float(COMMISSION_RATE * 2 + BE_BUFFER)
    if il:
        return sl >= entry - be_threshold
    else:
        return sl <= entry + be_threshold


def check_reversal(trade: dict, new_direction: str) -> bool:
    """
    Check if a new signal should trigger a reversal (close + re-open).

    Returns True if:
      - There IS an open trade
      - The new signal is in the OPPOSITE direction
      - The trade hasn't already hit TP1 (trailing — let it ride)

    When True, the caller should:
      1. Close the current position at market
      2. Immediately open in `new_direction` using the new signal's SL/TP
    """
    if trade is None:
        return False

    current_side = trade["side"]
    opposite = (
        (current_side == "LONG"  and new_direction == "SHORT") or
        (current_side == "SHORT" and new_direction == "LONG")
    )

    if not opposite:
        return False

    # Don't reverse if TP1 was already hit — we're trailing and the original
    # direction might still be profitable.
    if trade.get("tp1_hit", False):
        return False

    return True


def compute_reversal_savings(trade: dict, current_price: float) -> dict:
    """
    Compute how much a reversal saves vs waiting for SL.

    Returns dict with:
      - sl_price: the current SL level
      - unrealised_pnl: current P&L at market price
      - sl_pnl: P&L if SL hits
      - savings: unrealised - sl_pnl (positive = reversal is better)
    """
    entry = float(trade["entry_price"])
    sl    = float(trade.get("sl_price", entry))
    il    = trade["side"] == "LONG"

    if il:
        unrealised = current_price - entry
        sl_pnl     = sl - entry
    else:
        unrealised = entry - current_price
        sl_pnl     = entry - sl

    return {
        "sl_price":       sl,
        "unrealised_pnl": unrealised,
        "sl_pnl":         sl_pnl,
        "savings":        unrealised - sl_pnl,
    }