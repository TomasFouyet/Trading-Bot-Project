#!/usr/bin/env python3
"""
Simple Paper Trading Runner — executes the VALIDATED strategy.

═══════════════════════════════════════════════════════════════════════════════
This runner executes TrendFollowingV2Simple — the exact strategy that passed
statistical validation (WFA 5/5, MC RoR 0.9%, Permutation p=0.001).

NO TP1/TP2, NO trailing, NO tiers, NO reversal swap.
Single fixed SL + single fixed TP, full close only.

The strategy is the SINGLE source of truth for SL/TP/entry/exit.
This runner only: feeds bars, executes signals, sizes positions, logs, notifies.
═══════════════════════════════════════════════════════════════════════════════

Usage:
    python scripts/run_simple_paper.py                  # Paper (default)
    python scripts/run_simple_paper.py --headless       # No console, Telegram only
    python scripts/run_simple_paper.py --check          # Verify BingX
    python scripts/run_simple_paper.py --balance 200    # Custom balance
    python scripts/run_simple_paper.py --live            # REAL orders (confirmation required)
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import signal
import sys
import time
import uuid
from collections import deque
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np


# ═══════════════════════════════════════════════════════════════════════════════
# VALIDATED PARAMS — do not change without re-running full WFA + PSA
# ═══════════════════════════════════════════════════════════════════════════════

VALIDATED_PARAMS = {
    # Entry (validated)
    "adx_min":               20,
    "adx_strong":            35,
    "ema_fast":              20,
    "ema_slow":              50,
    "pullback_tolerance_atr": 1.0,
    "allow_short":           True,
    "min_confidence":        0.0,   # No confidence filter — all signals
    "sig_cooldown":          5,
    "slope_bars":            5,
    # Exit (validated — rr=2.5, structural stop)
    "rr_ratio":              2.5,
    "atr_sl_mult":           2.0,
}

STRUCTURAL_STOP_CFG = {
    "enabled":        True,
    "buffer_atr":     0.25,
    "min_risk_atr":   0.8,
    "pivot_left":     3,
    "pivot_right":    3,
}

RISK_CONFIG = {
    "risk_pct_per_trade":    1.5,   # % of equity risked per trade
    "max_leverage":          3.0,   # hard cap
    "max_daily_dd_pct":      15.0,  # circuit breaker warning (reduce size)
    "circuit_breaker_pct":   25.0,  # full stop, manual review required
    "max_positions":         1,     # BTC only, 1 position at a time
}

HTF_CONFIG = {
    "enabled":    True,
    "timeframe":  "4h",
    "ema_period": 50,
}

WINDOW_SIZE   = 300
STATUS_SECS   = 21600  # 6 hours
TRADES_CSV    = Path("data/simple_paper_trades.csv")
TRADES_HEADER = [
    "timestamp", "symbol", "side", "entry_price", "exit_price", "exit_type",
    "sl", "tp", "pnl_pct", "pnl_usd", "equity_after", "leverage_used",
    "sl_dist_pct", "adx_at_entry", "atr_at_entry", "htf_bias_at_entry",
    "duration_bars", "entry_fee_type", "exit_fee_type", "sl_mode",
    "confidence",
]
# BingX Futures fees: maker (limit) 0.020%, taker (market) 0.050%
MAKER_FEE = Decimal("0.00020")
TAKER_FEE = Decimal("0.00050")

# Global headless flag — set from CLI args
_HEADLESS = False


def _log(msg: str):
    """Print to console unless headless mode."""
    if not _HEADLESS:
        print(msg)


def _log_error(msg: str):
    """Always print errors, even in headless mode."""
    print(msg)


# ═══════════════════════════════════════════════════════════════════════════════
# Safe Telegram send (never crashes)
# ═══════════════════════════════════════════════════════════════════════════════

async def _tg_send(notifier, text: str) -> bool:
    """Send Telegram message safely. Returns True on success."""
    if notifier is None:
        return False
    try:
        return await notifier.send(text)
    except Exception as e:
        _log_error(f"  [TG] Send failed: {e}")
        return False


# ═══════════════════════════════════════════════════════════════════════════════
# CSV helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _ensure_csv():
    TRADES_CSV.parent.mkdir(parents=True, exist_ok=True)
    if not TRADES_CSV.exists():
        with open(TRADES_CSV, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=TRADES_HEADER).writeheader()


def _append_csv(row: dict):
    with open(TRADES_CSV, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=TRADES_HEADER).writerow(row)


def _read_last_trades(n: int = 5) -> list[dict]:
    """Read last N trades from CSV. Returns newest first."""
    if not TRADES_CSV.exists():
        return []
    try:
        with open(TRADES_CSV, "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        return rows[-n:][::-1]
    except Exception:
        return []


# ═══════════════════════════════════════════════════════════════════════════════
# Circuit Breaker (separate from RiskManager)
# ═══════════════════════════════════════════════════════════════════════════════

class CircuitBreaker:
    """Tracks peak equity and enforces drawdown limits."""

    def __init__(self, initial_equity: float):
        self.peak_equity = initial_equity
        self.current_equity = initial_equity
        self.risk_pct = RISK_CONFIG["risk_pct_per_trade"]
        self.base_risk_pct = RISK_CONFIG["risk_pct_per_trade"]
        self.stopped = False
        self.reduced = False

    def update(self, equity: float) -> str | None:
        """Update equity and return alert message if threshold crossed."""
        self.current_equity = equity
        if equity > self.peak_equity:
            self.peak_equity = equity

        dd_pct = self.drawdown_pct

        if dd_pct >= RISK_CONFIG["circuit_breaker_pct"]:
            self.stopped = True
            return (
                f"CIRCUIT BREAKER TRIGGERED: DD={dd_pct:.1f}% from peak "
                f"(peak=${self.peak_equity:,.2f}, now=${equity:,.2f}). Bot stopped."
            )

        if dd_pct >= RISK_CONFIG["max_daily_dd_pct"]:
            if not self.reduced:
                self.reduced = True
                self.risk_pct = self.base_risk_pct / 2
                return (
                    f"CIRCUIT BREAKER WARNING: DD={dd_pct:.1f}% from peak. "
                    f"Risk reduced to {self.risk_pct:.2f}% per trade."
                )
        elif dd_pct < RISK_CONFIG["max_daily_dd_pct"] * 0.67:
            # Recovery — restore normal sizing when DD drops below 10%
            if self.reduced:
                self.reduced = False
                self.risk_pct = self.base_risk_pct

        return None

    @property
    def drawdown_pct(self) -> float:
        if self.peak_equity <= 0:
            return 0.0
        return (self.peak_equity - self.current_equity) / self.peak_equity * 100


# ═══════════════════════════════════════════════════════════════════════════════
# Position Sizing (risk-based)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_position(
    equity: float,
    entry_price: float,
    sl_price: float,
    risk_pct: float,
    max_leverage: float,
) -> tuple[float, float, float]:
    """
    Risk-based position sizing.

    Returns: (qty, position_usd, leverage)
    """
    sl_distance = abs(entry_price - sl_price)
    sl_distance_pct = sl_distance / entry_price if entry_price > 0 else 1.0

    risk_usd = equity * risk_pct / 100.0
    position_usd = risk_usd / sl_distance_pct if sl_distance_pct > 0 else 0.0
    leverage = position_usd / equity if equity > 0 else 0.0
    leverage = min(leverage, max_leverage)
    position_usd = equity * leverage
    qty = position_usd / entry_price if entry_price > 0 else 0.0

    return qty, position_usd, leverage


# ═══════════════════════════════════════════════════════════════════════════════
# Synthetic fill for paper mode
# ═══════════════════════════════════════════════════════════════════════════════

def _make_paper_fill(symbol: str, price: float, qty: Decimal, side,
                     order_type: str = "market"):
    """Synthetic fill for paper mode. order_type='limit' → maker fee, 'market' → taker fee."""
    from app.broker.base import FillResult
    px = Decimal(str(price))
    fee_rate = MAKER_FEE if order_type == "limit" else TAKER_FEE
    fee = px * qty * fee_rate
    return FillResult(
        fill_id=str(uuid.uuid4()),
        order_id=str(uuid.uuid4()),
        symbol=symbol,
        side=side,
        price=px,
        qty=qty,
        fee=fee,
        fee_currency="USDT",
        timestamp=datetime.now(timezone.utc),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# HTF Bias (lightweight, for filtering)
# ═══════════════════════════════════════════════════════════════════════════════

class SimpleHTFBias:
    """
    Lightweight 4H EMA bias. Returns +1 (bull), -1 (bear), 0 (neutral).
    Used to filter signals: reject trades against the HTF trend.
    """

    def __init__(self, symbol: str, ema_period: int = 50):
        self._symbol = symbol
        self._ema_period = ema_period
        self._bias: int = 0
        self._bias_label: str = "UNKNOWN"
        self._last_refresh: datetime | None = None

    async def warmup(self, client, store):
        """Load initial 4H bars and compute bias."""
        from app.data.ingestor import OHLCVIngestor
        try:
            ingestor = OHLCVIngestor(client, store)
            bars = await ingestor.poll_latest(self._symbol, "4h", lookback_bars=300)
            if bars:
                self._compute(bars)
        except Exception as e:
            _log(f"  [HTF] Warmup failed: {e}")

    async def maybe_refresh(self, client, store, current_ts: datetime) -> bool:
        """Refresh if 4+ hours since last refresh."""
        if self._last_refresh and (current_ts - self._last_refresh).total_seconds() < 14400:
            return False
        from app.data.ingestor import OHLCVIngestor
        try:
            ingestor = OHLCVIngestor(client, store)
            bars = await ingestor.poll_latest(self._symbol, "4h", lookback_bars=300)
            if bars:
                self._compute(bars)
                return True
        except Exception as e:
            _log(f"  [HTF] Refresh failed: {e}")
        return False

    def _compute(self, bars):
        import pandas as pd
        from app.strategy.base import BaseStrategy
        df = BaseStrategy.bars_to_df(bars)
        if len(df) < self._ema_period + 10:
            return
        close = df["close"]
        ema = close.ewm(span=self._ema_period, adjust=False).mean()
        last_close = float(close.iloc[-1])
        last_ema = float(ema.iloc[-1])
        self._last_refresh = bars[-1].ts

        if last_close > last_ema * 1.002:
            self._bias = 1
            self._bias_label = f"BULL (close={last_close:.0f} > EMA{self._ema_period}={last_ema:.0f})"
        elif last_close < last_ema * 0.998:
            self._bias = -1
            self._bias_label = f"BEAR (close={last_close:.0f} < EMA{self._ema_period}={last_ema:.0f})"
        else:
            self._bias = 0
            self._bias_label = f"NEUTRAL (close={last_close:.0f} ~ EMA{self._ema_period}={last_ema:.0f})"

    @property
    def bias(self) -> int:
        return self._bias

    @property
    def label(self) -> str:
        return self._bias_label

    def bias_emoji(self) -> str:
        if self._bias == 1:
            return "BULL \u25b2"
        if self._bias == -1:
            return "BEAR \u25bc"
        return "NEUTRAL \u25cf"

    def is_aligned(self, direction: str) -> bool:
        if self._bias == 0:
            return True
        return (direction == "LONG" and self._bias == 1) or \
               (direction == "SHORT" and self._bias == -1)


# ═══════════════════════════════════════════════════════════════════════════════
# Telegram command listener
# ═══════════════════════════════════════════════════════════════════════════════

async def _telegram_command_loop(state, symbol, shutdown_event):
    """Poll Telegram for commands and respond."""
    notifier = state.get("notifier")
    if not notifier:
        return

    offset = 0
    # Flush old messages on startup (ignore anything sent before bot started)
    try:
        _, offset = await notifier.poll_commands(offset=0)
    except Exception:
        pass

    while not shutdown_event.is_set():
        try:
            messages, offset = await notifier.poll_commands(offset)
            for msg in messages:
                text = msg["text"].strip().lower()
                # Ignore messages from other chats
                if msg.get("chat_id") != notifier.chat_id:
                    continue

                if text == "/status":
                    await _cmd_status(notifier, state, symbol)
                elif text == "/trades":
                    await _cmd_trades(notifier)
                elif text == "/equity":
                    await _cmd_equity(notifier, state)
                elif text == "/stop":
                    await _tg_send(notifier,
                        "\u26a0\ufe0f <b>Stop signal received.</b>\n"
                        "Finishing current bar and shutting down...")
                    shutdown_event.set()
                elif text.startswith("/"):
                    await _tg_send(notifier,
                        "\U0001f916 Comandos disponibles:\n"
                        "/status \u2014 estado actual\n"
                        "/trades \u2014 \u00faltimos 5 trades\n"
                        "/equity \u2014 resumen de equity\n"
                        "/stop   \u2014 detener bot")
        except Exception as e:
            _log_error(f"  [TG] Command poll error: {e}")
        try:
            await asyncio.wait_for(shutdown_event.wait(), timeout=30)
            break
        except asyncio.TimeoutError:
            pass


async def _cmd_status(notifier, state, symbol):
    cb: CircuitBreaker = state["circuit_breaker"]
    eq = state.get("equity", 0)
    w, l = state.get("wins", 0), state.get("losses", 0)
    t = w + l
    wr = f"{w/t*100:.0f}%" if t else "\u2014"
    htf: SimpleHTFBias | None = state.get("htf")
    htf_str = htf.bias_emoji() if htf else "N/A"
    ot = state.get("open_trade")
    last_bar = state.get("last_bar_ts", "")
    if last_bar and hasattr(last_bar, "strftime"):
        last_bar = last_bar.strftime("%H:%M UTC")

    pos_str = "None"
    if ot:
        upnl_pct = state.get("upnl_pct", 0)
        pos_str = f"{ot['side']} @ ${float(ot['entry_price']):,.2f} (uPnL={upnl_pct:+.2f}%)"

    await _tg_send(notifier,
        f"\U0001f4ca <b>STATUS</b>\n\n"
        f"\U0001f4b0 Equity: <code>${eq:,.2f}</code> | Peak: <code>${cb.peak_equity:,.2f}</code>\n"
        f"\U0001f4c9 DD: {cb.drawdown_pct:.1f}%\n"
        f"\U0001f4c8 Trades: {t} (W={w} L={l} WR={wr})\n"
        f"\U0001f504 Position: {pos_str}\n"
        f"\U0001f9ed HTF Bias: {htf_str}\n"
        f"\u26a1 Risk/trade: {cb.risk_pct:.1f}%\n"
        f"\U0001f551 Last bar: {last_bar or 'waiting...'}")


async def _cmd_trades(notifier):
    trades = _read_last_trades(5)
    if not trades:
        await _tg_send(notifier, "\U0001f4cb <b>No trades yet.</b>")
        return
    lines = ["\U0001f4cb <b>Last 5 Trades</b>\n"]
    total_pnl = 0.0
    for i, t in enumerate(trades, 1):
        pnl = float(t.get("pnl_pct", 0))
        pnl_usd = float(t.get("pnl_usd", 0))
        emoji = "\u2705" if pnl > 0 else "\u274c"
        side = t.get("side", "?")
        bars = t.get("duration_bars", "?")
        exit_type = t.get("exit_type", "?").upper()
        lines.append(
            f"{i}. {emoji} {side} {exit_type}  {pnl:+.2f}% (${pnl_usd:+.2f})  [{bars} bars]")
        total_pnl += pnl
    n = len(trades)
    wins = sum(1 for t in trades if float(t.get("pnl_pct", 0)) > 0)
    losses = n - wins
    lines.append(f"\nW={wins} L={losses}  Avg: {total_pnl/n:+.2f}%")
    await _tg_send(notifier, "\n".join(lines))


async def _cmd_equity(notifier, state):
    cb: CircuitBreaker = state["circuit_breaker"]
    eq = state.get("equity", 0)
    start = state.get("start_balance", eq)
    net = eq - start
    net_pct = net / start * 100 if start else 0
    await _tg_send(notifier,
        f"\U0001f4b0 <b>Equity</b>\n\n"
        f"Start:   <code>${start:,.2f}</code>\n"
        f"Current: <code>${eq:,.2f}</code>\n"
        f"Net P&L: <code>{'+' if net >= 0 else ''}${net:,.2f} ({'+' if net_pct >= 0 else ''}{net_pct:.1f}%)</code>\n"
        f"Peak:    <code>${cb.peak_equity:,.2f}</code>\n"
        f"Max DD:  {cb.drawdown_pct:.1f}%")


# ═══════════════════════════════════════════════════════════════════════════════
# Structural Stop Override (applies validated pivot-based SL)
# ═══════════════════════════════════════════════════════════════════════════════

def _compute_structural_sl_from_bars(bars_deque, entry_price: float,
                                      direction: str, atr_val: float) -> tuple[float, str]:
    """Compute structural SL from the bar buffer. Returns (sl_price, mode_label)."""
    from validation.structural_stop import (
        compute_pivot_lows, compute_pivot_highs,
        build_last_pivot_arrays, compute_structural_sl,
    )
    cfg = STRUCTURAL_STOP_CFG
    if not cfg["enabled"]:
        sl_dist = atr_val * VALIDATED_PARAMS["atr_sl_mult"]
        if direction == "LONG":
            return entry_price - sl_dist, "atr"
        else:
            return entry_price + sl_dist, "atr"

    lows = np.array([float(b.low) for b in bars_deque])
    highs = np.array([float(b.high) for b in bars_deque])

    pl = compute_pivot_lows(lows, cfg["pivot_left"], cfg["pivot_right"])
    ph = compute_pivot_highs(highs, cfg["pivot_left"], cfg["pivot_right"])
    last_pl, last_ph = build_last_pivot_arrays(pl, ph, right=cfg["pivot_right"])

    bar_idx = len(bars_deque) - 1
    return compute_structural_sl(
        entry_price=entry_price,
        direction=direction,
        bar_idx=bar_idx,
        last_pivot_low=last_pl,
        last_pivot_high=last_ph,
        atr=atr_val,
        stop_mode="STRUCTURAL",
        atr_sl_mult=VALIDATED_PARAMS["atr_sl_mult"],
        buffer_atr=cfg["buffer_atr"],
        min_risk_atr=cfg["min_risk_atr"],
    )


# ═══════════════════════════════════════════════════════════════════════════════
# State Persistence (crash recovery)
# ═══════════════════════════════════════════════════════════════════════════════

import json

STATE_FILE = Path("data/bot_state.json")


def _save_state(state: dict, open_trade: dict | None, equity: float,
                peak_equity: float, wins: int, losses: int):
    """Persist bot state to disk for crash recovery."""
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "position": None,
        "equity": equity,
        "peak_equity": peak_equity,
        "wins": wins,
        "losses": losses,
        "last_updated": datetime.now(timezone.utc).isoformat(),
    }
    if open_trade:
        data["position"] = {
            "side": open_trade["side"],
            "entry_price": float(open_trade["entry_price"]),
            "sl": float(open_trade["sl"]),
            "tp": float(open_trade["tp"]),
            "qty": float(open_trade["qty"]),
            "entry_time": open_trade.get("entry_time", ""),
            "trade_id": open_trade.get("trade_id", ""),
            "adx_at_entry": open_trade.get("adx_at_entry", 0),
            "sl_mode": open_trade.get("sl_mode", "unknown"),
            "leverage": open_trade.get("leverage", 0),
            "sl_dist_pct": open_trade.get("sl_dist_pct", 0),
            "atr_at_entry": open_trade.get("atr_at_entry", 0),
            "htf_bias_at_entry": open_trade.get("htf_bias_at_entry", 0),
            "fee_in": float(open_trade.get("fee_in", 0)),
            "entry_fee_type": open_trade.get("entry_fee_type", "taker"),
        }
    with open(STATE_FILE, "w") as f:
        json.dump(data, f, indent=2)


def _load_state() -> dict | None:
    """Load saved state. Returns None if no state file."""
    if not STATE_FILE.exists():
        return None
    try:
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return None


def _clear_state():
    """Remove state file after clean close."""
    if STATE_FILE.exists():
        STATE_FILE.unlink()


# ═══════════════════════════════════════════════════════════════════════════════
# Main trading loop
# ═══════════════════════════════════════════════════════════════════════════════

async def run_symbol(symbol, client, adapter, args, shutdown_event, state, is_live: bool):
    from app.broker.base import OrderRequest, OrderSide, OrderType
    from app.data.feed import LiveFeed
    from app.data.ingestor import OHLCVIngestor, TIMEFRAME_SECONDS
    from app.data.parquet_store import ParquetStore
    from app.strategy.base import BaseStrategy
    from app.strategy.signals import SignalAction
    from app.strategy.trend_following_v2_simple import TrendFollowingV2Simple

    s = symbol.replace("-USDT", "")
    store    = ParquetStore()
    strategy = TrendFollowingV2Simple(symbol=symbol, params=VALIDATED_PARAMS)
    notifier = state.get("notifier")
    cb: CircuitBreaker = state["circuit_breaker"]

    _log(f"  [{s}] Loading warmup ({WINDOW_SIZE} bars)...")
    try:
        warmup = await OHLCVIngestor(client, store).poll_latest(
            symbol, args.timeframe, WINDOW_SIZE + 2)
    except Exception as e:
        _log_error(f"  [{s}] ERROR loading warmup: {e}")
        return

    bw: deque = deque(warmup[:-1], maxlen=WINDOW_SIZE)
    last_processed_ts = bw[-1].ts if bw else None
    if hasattr(strategy, 'force_close'):
        strategy.force_close()
    _log(f"  [{s}] Ready \u2014 {len(bw)} bars loaded")

    # HTF bias
    htf = SimpleHTFBias(symbol, ema_period=HTF_CONFIG["ema_period"])
    if HTF_CONFIG["enabled"]:
        await htf.warmup(client, store)
        _log(f"  [{s}] HTF: {htf.label}")
    state["htf"] = htf

    equity = float(args.balance)
    open_trade: dict | None = None
    bar_count_in_trade = 0

    # ── State recovery (crash protection) ────────────────────
    saved = _load_state()
    if saved:
        equity = saved.get("equity", equity)
        state["equity"] = equity
        state["wins"] = saved.get("wins", 0)
        state["losses"] = saved.get("losses", 0)
        cb.peak_equity = saved.get("peak_equity", equity)
        cb.current_equity = equity
        if saved.get("position"):
            pos = saved["position"]
            open_trade = {
                "trade_id": pos.get("trade_id", "recovered"),
                "symbol": symbol,
                "side": pos["side"],
                "entry_price": Decimal(str(pos["entry_price"])),
                "qty": Decimal(str(pos["qty"])),
                "fee_in": Decimal(str(pos.get("fee_in", 0))),
                "entry_fee_type": pos.get("entry_fee_type", "taker"),
                "sl": Decimal(str(pos["sl"])),
                "tp": Decimal(str(pos["tp"])),
                "leverage": pos.get("leverage", 1.0),
                "sl_dist_pct": pos.get("sl_dist_pct", 0),
                "sl_mode": pos.get("sl_mode", "unknown"),
                "adx_at_entry": pos.get("adx_at_entry", 0),
                "atr_at_entry": pos.get("atr_at_entry", 0),
                "htf_bias_at_entry": pos.get("htf_bias_at_entry", 0),
                "entry_time": pos.get("entry_time", ""),
                "sl_order_id": None,
            }
            state["open_trade"] = open_trade
            _log(f"  [{s}] RECOVERED position: {pos['side']} @ ${pos['entry_price']:,.2f}"
                 f" SL=${pos['sl']:,.2f} TP=${pos['tp']:,.2f}")
            await _tg_send(notifier,
                f"\u26a0\ufe0f <b>POSITION RECOVERED</b>\n\n"
                f"Side: {pos['side']}\n"
                f"Entry: ${pos['entry_price']:,.2f}\n"
                f"SL: ${pos['sl']:,.2f}\n"
                f"TP: ${pos['tp']:,.2f}\n"
                f"Equity: ${equity:,.2f}")
        else:
            _log(f"  [{s}] State recovered (no open position)")
            _log(f"  [{s}] Equity=${equity:,.2f} W={state['wins']} L={state['losses']}")

    tf_secs = TIMEFRAME_SECONDS.get(args.timeframe, 900)
    feed = LiveFeed(client=client, store=store, symbol=symbol,
                    timeframe=args.timeframe, max_data_delay_s=tf_secs * 3)
    feed_iter = feed.stream().__aiter__()

    try:
        while not shutdown_event.is_set():
            # Wait for next bar or shutdown
            get_bar = asyncio.ensure_future(feed_iter.__anext__())
            wait_sd = asyncio.ensure_future(shutdown_event.wait())
            done, pending = await asyncio.wait(
                [get_bar, wait_sd], return_when=asyncio.FIRST_COMPLETED)
            for t in pending:
                t.cancel()
                try:
                    await t
                except (asyncio.CancelledError, StopAsyncIteration):
                    pass
            if wait_sd in done:
                break
            try:
                bar = get_bar.result()
            except StopAsyncIteration:
                break

            if last_processed_ts is not None and bar.ts <= last_processed_ts:
                continue
            last_processed_ts = bar.ts
            state["last_bar_ts"] = bar.ts

            bw.append(bar)
            if len(bw) < strategy.min_bars_required:
                continue

            # ── Circuit breaker check ────────────────────────────
            if cb.stopped:
                continue

            # ── Refresh HTF bias ─────────────────────────────────
            if HTF_CONFIG["enabled"]:
                try:
                    if await htf.maybe_refresh(client, store, bar.ts):
                        _log(f"  [{s}] HTF updated: {htf.label}")
                except Exception:
                    pass

            # ── Run strategy ─────────────────────────────────────
            df = BaseStrategy.bars_to_df(list(bw))
            try:
                signals = strategy.on_bar_all(df)
            except Exception as e:
                _log_error(f"  [{s}] Strategy error: {e}")
                continue

            # ── Refresh equity (paper mode: track manually) ──────
            if not is_live:
                pass
            else:
                try:
                    balances = await adapter.get_balance()
                    equity = float(balances[0].total) if balances else equity
                except Exception:
                    pass

            state["equity"] = equity
            close = float(bar.close)

            # ── Update circuit breaker ───────────────────────────
            alert = cb.update(equity)
            if alert:
                if cb.stopped:
                    _log_error(f"\n  !!! CRITICAL: {alert}\n")
                    await _tg_send(notifier,
                        f"\U0001f6a8 <b>CIRCUIT BREAKER TRIGGERED</b>\n\n"
                        f"{alert}\n\nBot stopped. Manual review required.\n"
                        f"Send /stop to confirm shutdown.")
                    break
                else:
                    _log(f"\n  !!! WARNING: {alert}\n")
                    await _tg_send(notifier,
                        f"\u26a0\ufe0f <b>CIRCUIT BREAKER WARNING</b>\n\n"
                        f"DD: {cb.drawdown_pct:.1f}% from peak\n"
                        f"Position size reduced to {cb.risk_pct:.2f}% per trade")

            # Track bars in trade + update shared state
            if open_trade is not None:
                bar_count_in_trade += 1
                upnl = (close - float(open_trade["entry_price"])) if open_trade["side"] == "LONG" \
                    else (float(open_trade["entry_price"]) - close)
                state["upnl_pct"] = upnl / float(open_trade["entry_price"]) * 100
            else:
                state["upnl_pct"] = 0.0
            state["open_trade"] = open_trade

            # ── Print bar status ─────────────────────────────────
            ts_str = bar.ts.strftime("%H:%M")
            meta   = signals[0].meta if signals else {}
            adx    = meta.get("adx", 0)
            primary = signals[0] if signals else None
            action = primary.action.value if primary else "HOLD"

            pos_str = ""
            if open_trade:
                pos_str = f"  [{open_trade['side']} {bar_count_in_trade}b uPnL={state['upnl_pct']:+.2f}%]"

            htf_str = f"HTF={'B' if htf.bias == 1 else 'S' if htf.bias == -1 else 'N'}" if HTF_CONFIG["enabled"] else ""

            if action == "HOLD":
                _log(f"  {ts_str} {s:<4} ${close:>10,.2f} ADX={adx:.1f} {htf_str}  DD={cb.drawdown_pct:.1f}%{pos_str}")
            else:
                _log(f"  {ts_str} {s:<4} ${close:>10,.2f} ADX={adx:.1f} {htf_str}  *** {action} ***{pos_str}")

            # ── Execute signals ──────────────────────────────────
            for sig in signals:
                if not sig.is_actionable():
                    continue

                # ── CLOSE ────────────────────────────────────────
                if sig.action == SignalAction.CLOSE:
                    if open_trade is None:
                        continue

                    exit_price = sig.meta.get("exit_price", close)
                    exit_type = sig.meta.get("exit_type", "close")
                    il = open_trade["side"] == "LONG"

                    if not is_live:
                        # TP → limit order (maker fee); SL/close → market (taker fee)
                        exit_order_type = "limit" if exit_type == "tp" else "market"
                        fill = _make_paper_fill(
                            symbol, exit_price, open_trade["qty"],
                            OrderSide.SELL if il else OrderSide.BUY,
                            order_type=exit_order_type)
                    else:
                        try:
                            _, fill = await adapter.place_order(OrderRequest(
                                symbol=symbol,
                                side=OrderSide.SELL if il else OrderSide.BUY,
                                order_type=OrderType.MARKET,
                                qty=open_trade["qty"],
                                strategy_id=strategy.strategy_id,
                                extra={"positionSide": "LONG" if il else "SHORT"},
                            ))
                        except Exception as e:
                            _log_error(f"  [{s}] CLOSE failed: {e}")
                            continue

                    if fill:
                        # Cancel exchange SL if live
                        if is_live and open_trade.get("sl_order_id"):
                            try:
                                await client.cancel_order(open_trade["sl_order_id"], symbol)
                            except Exception:
                                pass

                        # Calculate PnL
                        entry_px = open_trade["entry_price"]
                        qty_d = open_trade["qty"]
                        if il:
                            gross_pnl = (fill.price - entry_px) * qty_d
                        else:
                            gross_pnl = (entry_px - fill.price) * qty_d
                        net_pnl = gross_pnl - open_trade["fee_in"] - fill.fee
                        pnl_pct = float(net_pnl) / (float(entry_px) * float(qty_d)) * 100 if entry_px * qty_d else 0

                        # Update equity
                        equity += float(net_pnl)
                        state["equity"] = equity

                        state["wins" if net_pnl > 0 else "losses"] += 1
                        w, l = state["wins"], state["losses"]

                        emoji = "\u2705" if exit_type == "tp" else "\u274c"
                        _log(
                            f"\n  {emoji} {exit_type.upper()} {s} @ ${float(fill.price):,.2f}"
                            f"  PnL={pnl_pct:+.2f}% (${float(net_pnl):+.2f})"
                            f"  Eq=${equity:,.2f}  W={w} L={l}\n"
                        )

                        # Log CSV
                        _append_csv({
                            "timestamp": fill.timestamp.isoformat(),
                            "symbol": symbol,
                            "side": open_trade["side"],
                            "entry_price": float(entry_px),
                            "exit_price": float(fill.price),
                            "exit_type": exit_type,
                            "sl": float(open_trade["sl"]),
                            "tp": float(open_trade["tp"]),
                            "pnl_pct": round(pnl_pct, 4),
                            "pnl_usd": round(float(net_pnl), 2),
                            "equity_after": round(equity, 2),
                            "leverage_used": round(open_trade["leverage"], 2),
                            "sl_dist_pct": round(open_trade["sl_dist_pct"] * 100, 4),
                            "adx_at_entry": open_trade["adx_at_entry"],
                            "atr_at_entry": open_trade["atr_at_entry"],
                            "htf_bias_at_entry": open_trade["htf_bias_at_entry"],
                            "duration_bars": bar_count_in_trade,
                            "entry_fee_type": open_trade.get("entry_fee_type", "taker"),
                            "exit_fee_type": "maker" if exit_type == "tp" else "taker",
                            "sl_mode": open_trade.get("sl_mode", ""),
                            "confidence": open_trade.get("confidence", ""),
                        })

                        # Telegram — rich close notification
                        dur_min = bar_count_in_trade * 15
                        dur_h = dur_min // 60
                        dur_m = dur_min % 60
                        dur_str = f"{dur_h}h {dur_m}m" if dur_h > 0 else f"{dur_m}m"
                        tp_or_sl = "TP HIT" if exit_type == "tp" else "SL HIT"
                        pnl_sign = "+" if net_pnl > 0 else ""
                        wr_str = f"{w/(w+l)*100:.0f}%" if (w + l) > 0 else "\u2014"

                        await _tg_send(notifier,
                            f"{'✅' if exit_type == 'tp' else '❌'} <b>{tp_or_sl} \u2014 {symbol}</b>\n\n"
                            f"\U0001f4b5 Exit:     <code>${float(fill.price):,.2f}</code>\n"
                            f"\U0001f4cd Entry:    <code>${float(entry_px):,.2f}</code>\n"
                            f"\U0001f4b0 P&L:      <code>{pnl_sign}{pnl_pct:.2f}% ({pnl_sign}${float(net_pnl):,.2f})</code>\n"
                            f"\u23f1 Duration: {bar_count_in_trade} bars (~{dur_str})\n"
                            f"\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n"
                            f"\U0001f4bc Equity:   <code>${equity:,.2f}</code>\n"
                            f"\U0001f3d4 Peak:     <code>${cb.peak_equity:,.2f}</code>\n"
                            f"\U0001f4c9 DD/Peak:  {cb.drawdown_pct:.1f}%\n"
                            f"\U0001f4ca Session:  W={w} L={l} WR={wr_str}")

                        open_trade = None
                        bar_count_in_trade = 0
                        state["open_trade"] = None
                        state["upnl_pct"] = 0.0

                        # Persist state (position closed)
                        _save_state(state, None, equity,
                                    cb.peak_equity, state["wins"], state["losses"])

                # ── BUY / SELL ───────────────────────────────────
                elif sig.action in (SignalAction.BUY, SignalAction.SELL):
                    direction = "LONG" if sig.action == SignalAction.BUY else "SHORT"

                    # If position open in opposite direction: close first
                    if open_trade is not None:
                        if open_trade["side"] == direction:
                            continue  # same direction, ignore
                        # Close existing position at market
                        il = open_trade["side"] == "LONG"
                        if not is_live:
                            close_fill = _make_paper_fill(
                                symbol, close, open_trade["qty"],
                                OrderSide.SELL if il else OrderSide.BUY)
                        else:
                            try:
                                _, close_fill = await adapter.place_order(OrderRequest(
                                    symbol=symbol,
                                    side=OrderSide.SELL if il else OrderSide.BUY,
                                    order_type=OrderType.MARKET,
                                    qty=open_trade["qty"],
                                    strategy_id=strategy.strategy_id,
                                    extra={"positionSide": "LONG" if il else "SHORT"},
                                ))
                            except Exception as e:
                                _log_error(f"  [{s}] Close-before-reverse failed: {e}")
                                continue

                        if close_fill:
                            entry_px = open_trade["entry_price"]
                            qty_d = open_trade["qty"]
                            if il:
                                gross_pnl = (close_fill.price - entry_px) * qty_d
                            else:
                                gross_pnl = (entry_px - close_fill.price) * qty_d
                            net_pnl = gross_pnl - open_trade["fee_in"] - close_fill.fee
                            equity += float(net_pnl)
                            state["equity"] = equity
                            state["wins" if net_pnl > 0 else "losses"] += 1
                            _log(f"  [{s}] Closed {open_trade['side']} for reversal: ${float(net_pnl):+.2f}")
                            open_trade = None
                            bar_count_in_trade = 0

                    # HTF filter: reject signals against the 4H trend
                    if HTF_CONFIG["enabled"] and not htf.is_aligned(direction):
                        _log(f"  [{s}] Signal {direction} rejected \u2014 HTF bias={htf.label}")
                        continue

                    # Position sizing — structural SL override
                    atr_val = float(sig.meta.get("atr", 0))
                    adx_val = float(sig.meta.get("adx", 0))

                    if atr_val <= 0:
                        _log(f"  [{s}] No ATR in signal — skipping")
                        continue

                    sl_price, sl_mode = _compute_structural_sl_from_bars(
                        bw, close, direction, atr_val)
                    sl_dist = abs(close - sl_price)
                    tp_price = close + sl_dist * VALIDATED_PARAMS["rr_ratio"] if direction == "LONG" \
                        else close - sl_dist * VALIDATED_PARAMS["rr_ratio"]

                    if sl_price <= 0:
                        _log(f"  [{s}] Invalid SL={sl_price} — skipping")
                        continue

                    qty, position_usd, leverage = compute_position(
                        equity=equity,
                        entry_price=close,
                        sl_price=sl_price,
                        risk_pct=cb.risk_pct,
                        max_leverage=RISK_CONFIG["max_leverage"],
                    )
                    qty_dec = Decimal(str(round(qty, 4)))
                    if qty_dec < Decimal("0.001"):
                        _log(f"  [{s}] Qty too small: {qty_dec}")
                        continue

                    sl_dist_pct = abs(close - sl_price) / close if close > 0 else 0

                    # Execute
                    il = sig.action == SignalAction.BUY
                    side = OrderSide.BUY if il else OrderSide.SELL

                    if not is_live:
                        # Entry = limit order (maker fee)
                        fill = _make_paper_fill(symbol, close, qty_dec, side,
                                                 order_type="limit")
                    else:
                        try:
                            _, fill = await adapter.place_order(OrderRequest(
                                symbol=symbol, side=side,
                                order_type=OrderType.MARKET, qty=qty_dec,
                                strategy_id=strategy.strategy_id,
                                extra={"positionSide": "LONG" if il else "SHORT"},
                            ))
                        except Exception as e:
                            _log_error(f"  [{s}] ORDER FAILED: {e}")
                            fill = None

                    if fill:
                        open_trade = {
                            "trade_id": str(uuid.uuid4())[:8],
                            "symbol": symbol,
                            "side": direction,
                            "entry_price": fill.price,
                            "qty": fill.qty,
                            "fee_in": fill.fee,
                            "entry_fee_type": "maker" if not is_live else "taker",
                            "sl": Decimal(str(sl_price)),
                            "tp": Decimal(str(tp_price)),
                            "leverage": leverage,
                            "sl_dist_pct": sl_dist_pct,
                            "sl_mode": sl_mode,
                            "adx_at_entry": round(adx_val, 1),
                            "atr_at_entry": round(atr_val, 2),
                            "htf_bias_at_entry": htf.bias,
                            "entry_time": fill.timestamp.isoformat(),
                            "sl_order_id": None,
                        }
                        bar_count_in_trade = 0
                        state["open_trade"] = open_trade

                        # Persist state (crash recovery)
                        _save_state(state, open_trade, equity,
                                    cb.peak_equity, state["wins"], state["losses"])

                        # Place exchange SL (crash protection) in live mode
                        if is_live and sl_price > 0:
                            try:
                                sl_side = "SELL" if il else "BUY"
                                sl_resp = await client.place_stop_order(
                                    symbol=symbol,
                                    side=sl_side,
                                    stop_price=sl_price,
                                    qty=float(fill.qty),
                                    pos_side="LONG" if il else "SHORT",
                                    client_order_id=f"sl{open_trade['trade_id']}",
                                )
                                sl_order_id = (sl_resp or {}).get("orderId") or (sl_resp or {}).get("order", {}).get("orderId")
                                open_trade["sl_order_id"] = sl_order_id
                                _log(f"  [{s}] SL order @ ${sl_price:,.2f}  id={sl_order_id}")
                            except Exception as e:
                                _log_error(f"  [{s}] WARNING: SL order failed: {e}")

                        sl_pct = sl_dist_pct * 100
                        tp_dist_pct = abs(tp_price - close) / close * 100 if close > 0 and tp_price > 0 else 0
                        bar_ts_str = bar.ts.strftime("%H:%M UTC")

                        _log(
                            f"\n  {'🟢' if il else '🔴'} {direction} {s} @ ${float(fill.price):,.2f}"
                            f"\n     SL=${sl_price:,.2f} (-{sl_pct:.2f}%)"
                            f"  TP=${tp_price:,.2f} (+{tp_dist_pct:.2f}%)"
                            f"\n     Risk={cb.risk_pct:.1f}% (${equity*cb.risk_pct/100:,.2f})"
                            f"  Lev={leverage:.1f}x  ATR=${atr_val:,.2f}  ADX={adx_val:.1f}  SL_mode={sl_mode}\n"
                        )

                        # Telegram — rich open notification
                        await _tg_send(notifier,
                            f"{'🟢' if il else '🔴'} <b>{direction} {symbol}</b>\n\n"
                            f"\U0001f4b5 Entry:    <code>${float(fill.price):,.2f}</code>\n"
                            f"\U0001f6e1 SL:       <code>${sl_price:,.2f} (-{sl_pct:.1f}%)</code>\n"
                            f"\U0001f3af TP:       <code>${tp_price:,.2f} (+{tp_dist_pct:.1f}%)</code>\n"
                            f"\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n"
                            f"\u26a1 Risk:     {cb.risk_pct:.1f}% (${equity*cb.risk_pct/100:,.2f})\n"
                            f"\U0001f4ca Leverage: {leverage:.1f}x\n"
                            f"\U0001f9ed HTF Bias: {htf.bias_emoji()}\n"
                            f"\U0001f4c8 ADX:      {adx_val:.1f} | ATR: ${atr_val:,.2f}\n"
                            f"\U0001f4b0 Equity:   <code>${equity:,.2f}</code>\n"
                            f"\u23f0 Bar time: {bar_ts_str}")

    except asyncio.CancelledError:
        pass
    except Exception as e:
        import traceback
        _log_error(f"  [{s}] Error: {e}")
        traceback.print_exc()
    finally:
        feed.stop()
        # Close open trade on shutdown
        if open_trade and open_trade["qty"] > Decimal("0.001"):
            _log(f"  [{s}] Closing on shutdown...")
            try:
                il = open_trade["side"] == "LONG"
                bc = float(bw[-1].close) if bw else float(open_trade["entry_price"])
                fill = _make_paper_fill(
                    symbol, bc, open_trade["qty"],
                    OrderSide.SELL if il else OrderSide.BUY)
                entry_px = open_trade["entry_price"]
                if il:
                    gross_pnl = (fill.price - entry_px) * fill.qty
                else:
                    gross_pnl = (entry_px - fill.price) * fill.qty
                net_pnl = gross_pnl - open_trade["fee_in"] - fill.fee
                _log(f"  [{s}] Closed @ ${bc:,.2f}  pnl=${float(net_pnl):+.2f}")
                equity += float(net_pnl)
                state["equity"] = equity
            except Exception as e:
                _log_error(f"  [{s}] Shutdown close failed: {e}")

        # Persist final state on clean shutdown
        _save_state(state, None, equity, cb.peak_equity, state["wins"], state["losses"])

# ═══════════════════════════════════════════════════════════════════════════════
# Weekly Report
# ═══════════════════════════════════════════════════════════════════════════════

def _weekly_report_text(state) -> str:
    """Generate weekly performance report text."""
    import math
    w, l = state.get("wins", 0), state.get("losses", 0)
    t = w + l
    eq = state.get("equity", 0)
    start = state.get("start_balance", eq)
    cb: CircuitBreaker = state["circuit_breaker"]

    wr_actual = w / t if t > 0 else 0
    wr_pct = wr_actual * 100

    # Backtest reference values
    wr_bt = 0.368  # 36.8% from structural rr=2.5 validation
    expr_bt = 0.335  # +0.335R

    # CI for WR
    if t > 0:
        margin = 1.96 * math.sqrt(wr_bt * (1 - wr_bt) / t)
        ci_lo = max(0, (wr_bt - margin)) * 100
        ci_hi = min(1, (wr_bt + margin)) * 100
        in_ci = ci_lo <= wr_pct <= ci_hi
    else:
        ci_lo = ci_hi = 0
        in_ci = True

    # Actual ExpR from CSV
    trades = _read_last_trades(100)
    pnls = [float(r.get("pnl_pct", 0)) for r in trades if r.get("pnl_pct")]
    losses_pnl = [p for p in pnls if p < 0]
    avg_loss = abs(sum(losses_pnl) / len(losses_pnl)) if losses_pnl else 1
    expr_actual = (sum(pnls) / len(pnls)) / avg_loss if pnls and avg_loss > 0 else 0

    net = eq - start
    net_pct = net / start * 100 if start > 0 else 0

    return (
        f"\U0001f4cb <b>WEEKLY REPORT</b>\n\n"
        f"\U0001f4b0 Equity: <code>${eq:,.2f}</code> ({'+' if net >= 0 else ''}{net_pct:.1f}%)\n"
        f"\U0001f4c9 DD from peak: {cb.drawdown_pct:.1f}%\n"
        f"\U0001f4ca Trades: {t} (W={w} L={l})\n"
        f"\n<b>Performance vs Backtest</b>\n"
        f"WR actual: {wr_pct:.1f}% vs backtest {wr_bt*100:.1f}%\n"
        f"ExpR actual: {expr_actual:+.3f}R vs backtest {expr_bt:+.3f}R\n"
        f"95% CI for WR ({t} trades): {ci_lo:.1f}% - {ci_hi:.1f}%\n"
        f"Within CI: {'YES' if in_ci else 'NO'}\n"
        f"\nExpected WR: {wr_bt*100:.1f}% +/- {(ci_hi-ci_lo)/2:.1f}pp with {t} trades"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Status loop (every 6 hours)
# ═══════════════════════════════════════════════════════════════════════════════

async def _status_loop(state, symbol, shutdown_event):
    last_weekly = datetime.now(timezone.utc)
    await asyncio.sleep(60)  # Initial delay
    while not shutdown_event.is_set():
        try:
            now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
            eq = state.get("equity", 0)
            cb: CircuitBreaker = state["circuit_breaker"]
            w, l = state.get("wins", 0), state.get("losses", 0)
            t = w + l
            wr = f" WR={w/t*100:.0f}%" if t else ""

            htf: SimpleHTFBias | None = state.get("htf")
            htf_str = htf.bias_emoji() if htf else "N/A"
            ot = state.get("open_trade")
            pos_str = "None"
            if ot:
                upnl = state.get("upnl_pct", 0)
                pos_str = f"{ot['side']} @ ${float(ot['entry_price']):,.2f} (uPnL={upnl:+.2f}%)"

            _log(f"\n{'━'*60}")
            _log(f"  STATUS [{now}]  Eq=${eq:,.2f}  Peak=${cb.peak_equity:,.2f}  DD={cb.drawdown_pct:.1f}%")
            _log(f"  Trades={t} (W={w} L={l}{wr})  Risk/trade={cb.risk_pct:.1f}%")
            _log(f"{'━'*60}\n")

            notifier = state.get("notifier")
            await _tg_send(notifier,
                f"\U0001f4ca <b>STATUS</b>\n\n"
                f"\U0001f4b0 Equity: <code>${eq:,.2f}</code> | Peak: <code>${cb.peak_equity:,.2f}</code>\n"
                f"\U0001f4c9 DD: {cb.drawdown_pct:.1f}%\n"
                f"\U0001f4c8 Trades: {t} (W={w} L={l}{wr})\n"
                f"\U0001f504 Position: {pos_str}\n"
                f"\U0001f9ed HTF Bias: {htf_str}\n"
                f"\u26a1 Risk/trade: {cb.risk_pct:.1f}%\n"
                f"\u23f0 {now}")

            # Weekly report (every 7 days or Monday)
            now_dt = datetime.now(timezone.utc)
            days_since = (now_dt - last_weekly).total_seconds() / 86400
            if days_since >= 7 or (now_dt.weekday() == 0 and days_since >= 1):
                report = _weekly_report_text(state)
                await _tg_send(notifier, report)
                _log(f"  [Weekly report sent]")
                last_weekly = now_dt
        except Exception:
            pass
        try:
            await asyncio.wait_for(shutdown_event.wait(), timeout=STATUS_SECS)
            break
        except asyncio.TimeoutError:
            pass


# ═══════════════════════════════════════════════════════════════════════════════
# Connection check
# ═══════════════════════════════════════════════════════════════════════════════

async def check_connection(args):
    from app.config import get_settings
    from app.core.logging import configure_logging
    s = get_settings()
    configure_logging(log_level="ERROR", log_format="console")
    from app.broker.bingx_client import BingXClient
    c = BingXClient(api_key=s.bingx_api_key, api_secret=s.bingx_api_secret,
                    base_url=s.bingx_base_url, market_type=s.bingx_market_type)
    symbol = f"{args.symbol}-USDT"
    print(f"\nChecking BingX API...")
    try:
        tk = await c.get_ticker(symbol)
        print(f"  [OK] {symbol:<12} ${float(tk['last']):>12,.4f}")
    except Exception as e:
        print(f"  [FAIL] {symbol}: {e}")
    await c.close()


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

async def main(args):
    global _HEADLESS
    _HEADLESS = args.headless

    from app.config import get_settings
    from app.core.logging import configure_logging
    settings = get_settings()
    configure_logging(log_level="WARNING", log_format="console")
    from app.broker.bingx_adapter import BingXAdapter
    from app.broker.bingx_client import BingXClient
    from app.broker.paper_adapter import PaperAdapter
    from app.notify.telegram import TelegramNotifier

    _ensure_csv()
    symbol = f"{args.symbol}-USDT"

    client = BingXClient(
        api_key=settings.bingx_api_key, api_secret=settings.bingx_api_secret,
        base_url=settings.bingx_base_url, market_type=settings.bingx_market_type,
    )

    is_live = args.live
    if is_live:
        # Safety confirmation
        print("\n" + "!" * 60)
        print("  WARNING: --live flag detected. This will use REAL USDT.")
        print("!" * 60)
        confirm = input("  Type 'YES I UNDERSTAND' to proceed: ").strip()
        if confirm != "YES I UNDERSTAND":
            print("  Aborted.")
            await client.close()
            return
        adapter = BingXAdapter(client=client)
        try:
            await client.set_leverage(symbol, int(RISK_CONFIG["max_leverage"]))
        except Exception as e:
            print(f"  Leverage warning: {e}")
        mode = "LIVE BingX"
    else:
        adapter = PaperAdapter(client=client, initial_balance=Decimal(str(args.balance)))
        mode = "PAPER simulation"

    notifier = None
    if not args.no_telegram:
        notifier = TelegramNotifier.from_env()
        if notifier:
            # Test connectivity at startup
            try:
                ok = await notifier.send("\U0001f50c Telegram test... OK")
                if ok:
                    _log("  Telegram: connected (test message sent)")
                else:
                    _log_error("  Telegram: FAILED to send test message (continuing without)")
                    notifier = None
            except Exception as e:
                _log_error(f"  Telegram: error on test send: {e} (continuing without)")
                notifier = None
        else:
            _log("  Telegram: not configured (missing TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID)")

    cb = CircuitBreaker(initial_equity=args.balance)
    headless_str = " | Modo headless \u2014 solo Telegram" if _HEADLESS else ""

    _log(f"\n{'═'*60}")
    _log(f"  SIMPLE PAPER TRADER \u2014 Validated Strategy")
    _log(f"{'═'*60}")
    _log(f"  Mode:       {mode}{headless_str}")
    _log(f"  Symbol:     {symbol}")
    _log(f"  Timeframe:  {args.timeframe}")
    _log(f"  Balance:    ${args.balance:,.2f}")
    _log(f"  Risk/trade: {RISK_CONFIG['risk_pct_per_trade']}%")
    _log(f"  Max lev:    {RISK_CONFIG['max_leverage']}x")
    _log(f"  RR ratio:   {VALIDATED_PARAMS['rr_ratio']}")
    _log(f"  Stop:       STRUCTURAL (pivot L/R={STRUCTURAL_STOP_CFG['pivot_left']}/{STRUCTURAL_STOP_CFG['pivot_right']}, buf={STRUCTURAL_STOP_CFG['buffer_atr']}ATR, min={STRUCTURAL_STOP_CFG['min_risk_atr']}ATR)")
    _log(f"  ATR fallbk: {VALIDATED_PARAMS['atr_sl_mult']}x")
    _log(f"  HTF filter: {'ON (4H EMA50)' if HTF_CONFIG['enabled'] else 'OFF'}")
    _log(f"  CB warn:    {RISK_CONFIG['max_daily_dd_pct']}% DD")
    _log(f"  CB stop:    {RISK_CONFIG['circuit_breaker_pct']}% DD")
    _log(f"{'═'*60}")

    # Send rich startup notification to Telegram
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    await _tg_send(notifier,
        f"\U0001f680 <b>BOT INICIADO \u2014 {'Paper Trading' if not is_live else 'LIVE TRADING'}</b>\n\n"
        f"\U0001f4cb Estrategia: TrendFollowingV2Simple\n"
        f"\U0001f4b0 Balance: <code>${args.balance:,.2f}</code>\n"
        f"\U0001f4ca Symbol: {symbol} | {args.timeframe}\n"
        f"\U0001f3af RR: {VALIDATED_PARAMS['rr_ratio']} | Stop: STRUCTURAL (fallback {VALIDATED_PARAMS['atr_sl_mult']}\u00d7ATR)\n"
        f"\u26a1 Risk/trade: {RISK_CONFIG['risk_pct_per_trade']}%\n"
        f"\U0001f9ed HTF Filter: {'ON (4H EMA50)' if HTF_CONFIG['enabled'] else 'OFF'}\n"
        f"\U0001f6e1 Circuit Breaker: warn={RISK_CONFIG['max_daily_dd_pct']:.0f}% / stop={RISK_CONFIG['circuit_breaker_pct']:.0f}%\n"
        f"\U0001f4b8 Fees: entry limit 0.020% | SL market 0.050% | TP limit 0.020% (avg RT ~0.056%)\n"
        f"{'Modo headless \u2014 solo Telegram' + chr(10) if _HEADLESS else ''}"
        f"\nComandos disponibles:\n"
        f"/status \u2014 estado actual\n"
        f"/trades \u2014 \u00faltimos 5 trades\n"
        f"/equity \u2014 resumen de equity\n"
        f"/stop   \u2014 detener bot\n"
        f"\n\u23f0 Iniciado: {now_str}")

    _log(f"\n  Loading...\n")

    shutdown_event = asyncio.Event()
    def _shutdown():
        _log("\n  Shutting down...")
        shutdown_event.set()

    try:
        loop = asyncio.get_running_loop()
        loop.add_signal_handler(signal.SIGINT, _shutdown)
        loop.add_signal_handler(signal.SIGTERM, _shutdown)
    except NotImplementedError:
        signal.signal(signal.SIGINT, lambda s, f: _shutdown())

    state = {
        "equity": args.balance,
        "start_balance": args.balance,
        "wins": 0,
        "losses": 0,
        "notifier": notifier,
        "circuit_breaker": cb,
        "htf": None,
        "open_trade": None,
        "upnl_pct": 0.0,
        "last_bar_ts": None,
    }

    worker = asyncio.create_task(run_symbol(
        symbol, client, adapter, args, shutdown_event, state, is_live=is_live))
    status = asyncio.create_task(_status_loop(state, symbol, shutdown_event))
    cmd_task = asyncio.create_task(_telegram_command_loop(state, symbol, shutdown_event))

    results = await asyncio.gather(worker, status, cmd_task, return_exceptions=True)
    for r in results:
        if isinstance(r, Exception) and not isinstance(r, asyncio.CancelledError):
            _log_error(f"  Task error: {r}")

    await adapter.shutdown()

    w, l = state["wins"], state["losses"]
    t = w + l
    eq = state["equity"]
    net = eq - args.balance
    _log(f"\n{'═'*60}")
    _log(f"  SESSION COMPLETE")
    _log(f"  Trades: {t}  W={w} L={l}  {'WR=' + str(round(w/t*100)) + '%' if t else ''}")
    _log(f"  Equity: ${eq:,.2f}  (net ${net:+,.2f})")
    _log(f"{'═'*60}\n")

    if notifier:
        try:
            await notifier.bot_stopped(
                total_trades=t, net_pnl=net, wins=w, losses=l)
        except Exception:
            pass


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Simple Paper Trader \u2014 Validated Strategy")
    p.add_argument("--balance",      type=float, default=100.0)
    p.add_argument("--symbol",       default="BTC")
    p.add_argument("--timeframe",    default="15m")
    p.add_argument("--live",         action="store_true", help="Send REAL orders to BingX")
    p.add_argument("--check",        action="store_true", help="Verify BingX connection")
    p.add_argument("--no-telegram",  action="store_true", dest="no_telegram")
    p.add_argument("--headless",     action="store_true", help="No console output, Telegram only")
    args = p.parse_args()

    if args.check:
        asyncio.run(check_connection(args))
    else:
        asyncio.run(main(args))
