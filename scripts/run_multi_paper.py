"""
Multi-symbol paper trading — TrendFollowing on BingX.

Monitors BTC, ETH, SOL, XRP, BNB simultaneously for entries.
Supports both LONG and SHORT positions.

Modes:
  DEFAULT:      PaperAdapter — simulates locally with real BingX prices.
                No orders sent. Safe. No money touched.
  --live-bingx: BingXAdapter — sends REAL orders to BingX with REAL USDT.

Usage:
    python scripts/run_multi_paper.py                        # Safe simulation
    python scripts/run_multi_paper.py --check                # Test API
    python scripts/run_multi_paper.py --symbols BTC ETH SOL  # Fewer symbols
    python scripts/run_multi_paper.py --live-bingx            # REAL orders!

═══════════════════════════════════════════════════════════════════════
FIXES vs previous version
═══════════════════════════════════════════════════════════════════════

FIX 1 — SL/TP fill price in paper mode
  OLD: adapter.place_order() fetches live ticker → wrong price if bar rebounded
  NEW: _make_paper_fill() creates a synthetic fill at the EXACT trigger price
       (sl_price, tp1_price, tp2_price). Paper mode never calls the broker for
       SL/TP closes. Live mode still calls place_order() as before.

FIX 2 — ATR updated every bar
  OLD: current_atr stored once at entry and never updated → stale trailing SL
  NEW: strategy meta["atr"] is read from the latest bar's indicators on every
       tick and stored in open_trade["current_atr"] each bar.

FIX 3 — best_price initialised at fill price not bar extremes
  OLD: after TP1, best_price = bar.high (LONG) or bar.low (SHORT), which could
       be BETTER than TP1 fill → trailing SL immediately too tight
  NEW: best_price = float(fill.price) at TP1 hit, then updated each bar.

FIX 4 — BE_SL uses exact commission rate
  OLD: hardcoded 0.002 (20 bps) as fee buffer
  NEW: reads COMMISSION constant (7.5 bps × 2 legs = 15 bps) from config,
       keeping a small extra buffer so BE is never negative.

FIX 5 — be_sl exit_type when patience is active
  OLD: a trailing SL closing before TP1 was logged as "trailing_sl"
  NEW: exits before TP1 are always "sl" or "be_sl"; "trailing_sl" only
       applies after TP1.

FIX 6 — SL/TP check priority: SL wins over TP on same bar
  OLD: TP1 had priority over SL even if both triggered on the same bar
  NEW: if SL and TP1 both trigger on the same bar, SL wins (conservative).
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import signal
import time
import uuid
from collections import deque
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

from app.engine.position_manager import (
    COMMISSION_RATE,
    ExitAction,
    advance_bar,
    apply_tp1_updates,
    check_exit,
    compute_leg_pnl,
    create_trade,
    update_atr,
)

HTF_TIMEFRAME = "4h"   # Higher timeframe for bias
LTF_TIMEFRAME = "5m"   # Lower timeframe for entry precision

BEST_PARAMS: dict = {
    # Core — aligned with Pine Script defaults
    "adx_min": 20, "adx_strong": 35,
    "pullback_tolerance_atr": 1.0,
    "allow_short": True,
    # Confidence — min_confidence=0 matches Pine (no tier-C filter)
    "min_confidence": 0, "use_confidence": True,
    # Session sizing
    "session_mult_us": 1.0, "session_mult_eu": 0.75, "session_mult_other": 0.5,
    "use_session_filter": True,
    # Streak
    "streak_euphoria_mult": 0.75, "use_streak_adj": True,
    # Patience (SL suave)
    "soft_sl_bars": 48, "use_patience": True,
    # Cooldown (mirrors Pine Script sig_cooldown=5 barras)
    "sig_cooldown": 5,
}

DEFAULT_BASES = ["BTC", "ETH", "SOL", "XRP", "BNB"]
WINDOW_SIZE   = 300
STATUS_SECS   = 300
TRADES_CSV    = Path("data/paper_trades.csv")
TRADES_HEADER = [
    "trade_id", "symbol", "side", "strategy", "tier", "leverage",
    "entry_time", "entry_price", "exit_time", "exit_price", "exit_type",
    "qty", "pnl", "pnl_pct", "fees", "signal_reason",
]


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


# ═══════════════════════════════════════════════════════════════════════════════
# Synthetic fill for paper-mode SL/TP closes
# ═══════════════════════════════════════════════════════════════════════════════

def _make_paper_fill(ot: dict, trigger_price: float, qty: Decimal, side):
    """Create a synthetic FillResult at the exact trigger price (paper mode only)."""
    from app.broker.base import FillResult
    price = Decimal(str(trigger_price))
    fee   = price * qty * COMMISSION_RATE
    return FillResult(
        fill_id=str(uuid.uuid4()),
        order_id=str(uuid.uuid4()),
        symbol=ot["symbol"],
        side=side,
        price=price,
        qty=qty,
        fee=fee,
        fee_currency="USDT",
        timestamp=datetime.now(timezone.utc),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# PnL recording
# ═══════════════════════════════════════════════════════════════════════════════

def _record_close(ot: dict, fill, exit_type: str) -> Decimal:
    net = compute_leg_pnl(ot, fill.price, fill.qty, fill.fee)
    pct = fill.qty / ot["original_qty"]
    fi  = ot["fee_in"] * pct
    tp  = ot["total_partial_pnl"] + net
    tf  = ot["total_partial_fees"] + fi + fill.fee
    n   = ot["entry_price"] * ot["original_qty"]
    _append_csv({
        "trade_id": ot["trade_id"], "symbol": ot["symbol"], "side": ot["side"],
        "strategy": ot["strategy"], "tier": ot["tier"], "leverage": ot["leverage"],
        "entry_time": ot["entry_time"], "entry_price": float(ot["entry_price"]),
        "exit_time": fill.timestamp.isoformat(), "exit_price": float(fill.price),
        "exit_type": exit_type, "qty": float(ot["original_qty"]),
        "pnl": float(tp.quantize(Decimal("0.01"))),
        "pnl_pct": round(float(tp / n * 100) if n else 0, 4),
        "fees": float(tf.quantize(Decimal("0.01"))),
        "signal_reason": ot["signal_reason"],
    })
    return tp


# ═══════════════════════════════════════════════════════════════════════════════
# Core SL / TP check — called every bar for open positions
# ═══════════════════════════════════════════════════════════════════════════════

async def _check_sl_tp(ot: dict, bar, adapter, sid: str, is_live: bool, notifier=None):
    """
    Evaluate SL/TP triggers against the current bar using shared PositionManager.

    Returns (exit_type, fill, close_pct) or (None, None, None) if no trigger.
    """
    from app.broker.base import OrderRequest, OrderSide, OrderType

    bh, bl, bc = float(bar.high), float(bar.low), float(bar.close)
    il = ot["side"] == "LONG"

    # Delegate to shared position manager (trailing + trigger detection)
    advance_bar(ot, bh, bl)
    action = check_exit(ot, bh, bl, bc)

    if action is None:
        return None, None, None

    et = action.exit_type
    cq = action.close_qty
    cp = action.close_pct
    trigger_price = action.trigger_price

    # Apply TP1 updates immediately (BE stop, best_price, patience off)
    if action.tp1_updates:
        apply_tp1_updates(ot, action.tp1_updates)

    # ── Execute close ─────────────────────────────────────────────────────────
    close_side     = OrderSide.SELL if il else OrderSide.BUY
    close_pos_side = "LONG"         if il else "SHORT"

    if not is_live:
        fill = _make_paper_fill(ot, trigger_price, cq, close_side)
    else:
        try:
            _, fill = await adapter.place_order(
                OrderRequest(
                    symbol=ot["symbol"], side=close_side,
                    order_type=OrderType.MARKET, qty=cq, strategy_id=sid,
                    extra={"positionSide": close_pos_side},
                )
            )
        except Exception as e:
            err_str = str(e)
            sl = float(ot.get("sl_price", 0))
            print(f"  ⚠️  [{ot['symbol']}] {et.upper()} CLOSE ORDER FAILED: {err_str}")
            if notifier:
                asyncio.create_task(notifier.send(
                    f"⚠️ <b>CLOSE ORDER FAILED — {ot['symbol']}</b>\n"
                    f"🚫 Exit: <b>{et.upper()}</b>\n"
                    f"🚫 Error: <code>{err_str[:200]}</code>\n"
                    f"📌 Will retry next bar\n"
                    f"  {ot['side']}  entry=<code>${float(ot['entry_price']):,.2f}</code>\n"
                    f"  SL=<code>${sl:,.2f}</code>  qty={float(cq):.4f}\n"
                ))
            return et + "_order_failed", None, cp

    return et, fill, cp


# ═══════════════════════════════════════════════════════════════════════════════
# Bar status line helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _analyze_bar(bar, sig, meta: dict, params: dict) -> tuple[str, bool]:
    adx_min  = float(params.get("adx_min", 20))
    adx      = meta.get("adx", 0)
    ema_fast = meta.get("ema_fast", 0)
    ema_slow = meta.get("ema_slow", 0)
    macd     = meta.get("macd", 0)
    macd_sig = meta.get("macd_signal", 0)
    close    = float(bar.close)
    reason   = sig.reason or ""

    if sig.is_actionable():
        tier = meta.get("confidence_tier", "?")
        conf = meta.get("confidence_score", 0)
        return f"SIGNAL {sig.action.value}  Tier={tier}  conf={conf:.2f}", False

    if reason in ("warmup", "indicators_not_ready"):
        return "initializing indicators...", False

    if reason.startswith("adx="):
        need = adx_min - adx
        pct  = adx / adx_min * 100
        bar_count = "█" * int(pct // 10) + "·" * (10 - int(pct // 10))
        return f"ADX {adx:.1f}/{adx_min:.0f} [{bar_count}] needs +{need:.1f}", False

    if reason == "no_setup":
        p_above = close > ema_slow if ema_slow else None
        p_below = close < ema_slow if ema_slow else None
        m_bull  = macd > macd_sig
        m_bear  = macd < macd_sig

        if p_below:
            if not m_bear:
                return f"MACD bullish in downtrend ({macd:.2f} > {macd_sig:.2f})", False
            if ema_fast:
                dist_pct = abs(close - ema_fast) / close * 100
                if close < ema_fast:
                    return f"WATCHLIST SHORT — price {dist_pct:.1f}% below EMA20, waiting pullback up", True
                else:
                    return f"WATCHLIST SHORT — price at EMA20, checking bearish candle", True
            return "waiting for short pullback", True

        if p_above:
            if not m_bull:
                return f"MACD bearish ({macd:.2f} < {macd_sig:.2f})", False
            if ema_fast:
                dist_pct = (close - ema_fast) / close * 100
                if dist_pct > 0:
                    return f"WATCHLIST LONG — price {dist_pct:.1f}% above EMA20, waiting pullback", True
                else:
                    return f"WATCHLIST LONG — price at EMA20, checking candle", True
            return "waiting for pullback", True

        return "price near EMA50 — no clear direction", False

    if reason.startswith("low_conf"):
        try: conf = float(reason.split("=")[1])
        except: conf = 0.0
        return f"confluence weak (score={conf:.2f})", False

    if reason == "zero_risk":
        return "SL too close (0 risk)", False

    # Cooldown: setup valid but waiting for bar gap to expire
    if reason.startswith("cooldown_long("):
        try: bars_left = reason.split("(")[1].rstrip("bars)")
        except: bars_left = "?"
        return f"★ COOLDOWN LONG — setup ready, {bars_left} bars remaining", True

    if reason.startswith("cooldown_short("):
        try: bars_left = reason.split("(")[1].rstrip("bars)")
        except: bars_left = "?"
        return f"★ COOLDOWN SHORT — setup ready, {bars_left} bars remaining", True

    return reason, False


def _print_bar_line(sym, bar, sig, meta, ot, equity, params):
    ts    = bar.ts.strftime("%H:%M")
    close = float(bar.close)
    adx   = meta.get("adx", 0)
    ema_slow = meta.get("ema_slow", 0)
    trend = "↑" if (ema_slow and close > ema_slow) else ("↓" if ema_slow else "~")
    adx_min = float(params.get("adx_min", 20))
    adx_badge = f"ADX={adx:.0f}★" if adx >= 35 else (f"ADX={adx:.0f}✓" if adx >= adx_min else f"ADX={adx:.0f}✗")
    msg, wl = _analyze_bar(bar, sig, meta, params)
    prefix = "★ WATCH" if wl else "  hold "
    pos = ""
    if ot:
        e = float(ot["entry_price"])
        u = (close - e) * float(ot["qty"]) if ot["side"] == "LONG" else (e - close) * float(ot["qty"])
        sl = ot.get("sl_price")
        sl_str = f" SL=${float(sl):,.2f}" if sl is not None else ""
        phase = "TRAIL" if ot.get("tp1_hit") else ot["side"]
        pos = f"  [{phase} {ot['bars_in_trade']}bars ${u:+.0f}{sl_str}]"
    s = sym.replace("-USDT", "")
    print(f"  {ts} {s:<4} ${close:>10,.2f} {trend} {adx_badge:<9}  {prefix}: {msg}{pos}")


# ═══════════════════════════════════════════════════════════════════════════════
# Status loop (periodic market scan summary)
# ═══════════════════════════════════════════════════════════════════════════════

async def _status_loop(client, state, symbols, shutdown_event):
    await asyncio.sleep(STATUS_SECS)
    while not shutdown_event.is_set():
        try:
            now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
            eq = state.get("equity", 0)
            w, l = state.get("wins", 0), state.get("losses", 0)
            t = w + l; pos = state.get("positions", {}); ss = state.get("sym_states", {})
            wr = f" WR={w/t*100:.0f}%" if t else ""
            print(f"\n{'━'*72}")
            print(f"  MARKET SCAN  [{now}]  Eq=${eq:,.2f}  Trades={t} (W={w} L={l}{wr})")
            print(f"  {'Symbol':<12} {'Price':>10}  {'Trend':5}  {'ADX':8}  {'Status'}")
            print(f"  {'─'*68}")
            for sym in symbols:
                si = ss.get(sym, {})
                if not si: print(f"  {sym:<12} loading..."); continue
                c = si.get("close", 0); a = si.get("adx", 0); tr = si.get("trend", "~")
                am = si.get("adx_min", 20)
                ab = f"{a:.1f}★" if a >= 35 else (f"{a:.1f}✓" if a >= am else f"{a:.1f}✗")
                msg = si.get("msg", "—"); wl = si.get("watchlist", False)
                if sym in pos:
                    ot = pos[sym]
                    try:
                        tk = await client.get_ticker(sym); p = float(tk["last"])
                        e = float(ot["entry_price"]); il = ot["side"] == "LONG"
                        u = (p - e) * float(ot["qty"]) if il else (e - p) * float(ot["qty"])
                        phase = "TRAIL" if ot.get("tp1_hit") else ot["side"]
                        sl_now = ot.get("sl_price")
                        sl_str = f" SL=${float(sl_now):,.2f}" if sl_now else ""
                        status = f"IN TRADE {phase} T={ot.get('tier','?')} {ot.get('leverage',1)}x uPnL=${u:+.0f}{sl_str}"
                    except: status = f"IN TRADE {ot['side']}"
                elif wl: status = f"★ {msg}"
                else: status = msg[:45]
                print(f"  {sym:<12} ${c:>10,.2f}  {tr:5}  {ab:<7}  {status}")
            print(f"  {'─'*68}")
            wls = [s for s in symbols if ss.get(s, {}).get("watchlist")]
            if pos:
                open_list = ', '.join(f"{s}({pos[s]['side']})" for s in pos)
                print(f"  Open: {open_list}")
            elif wls: print(f"  ★ Close to entry: {', '.join(s.replace('-USDT','') for s in wls)}")
            else: print(f"  Scanning {len(symbols)} symbols — no setups yet")
            print(f"{'━'*72}\n")
        except Exception as e: print(f"  [status: {e}]")

        try:
            await asyncio.wait_for(shutdown_event.wait(), timeout=STATUS_SECS)
            break
        except asyncio.TimeoutError:
            pass


# ═══════════════════════════════════════════════════════════════════════════════
# Daily Telegram summary (fires at midnight UTC)
# ═══════════════════════════════════════════════════════════════════════════════

async def _daily_summary_loop(state, args, shutdown_event):
    """
    Fires a Telegram daily_summary once per UTC day at midnight.
    Tracks session-start equity per day so the daily PnL is correct
    even across multi-day sessions.
    """
    from app.notify.telegram import TelegramNotifier
    daily_start_equity = state.get("equity", args.balance)
    daily_wins:   int = 0
    daily_losses: int = 0
    top_pnl:   float = 0.0
    worst_pnl: float = 0.0

    last_day = datetime.now(timezone.utc).date()
    state["_daily_start_equity"] = daily_start_equity
    state["_daily_wins"]   = daily_wins
    state["_daily_losses"] = daily_losses
    state["_daily_top"]    = top_pnl
    state["_daily_worst"]  = worst_pnl

    # Wait until just after next midnight UTC
    while not shutdown_event.is_set():
        now     = datetime.now(timezone.utc)
        today   = now.date()

        # Check if a new day started since last loop
        if today > last_day:
            notifier = state.get("notifier")
            eq       = state.get("equity", args.balance)
            dse      = state.get("_daily_start_equity", args.balance)
            dw       = state.get("_daily_wins",   0)
            dl       = state.get("_daily_losses", 0)
            dt       = state.get("_daily_top",    0.0)
            dwo      = state.get("_daily_worst",  0.0)

            if notifier:
                try:
                    await notifier.daily_summary(
                        equity=eq,
                        starting_equity=dse,
                        total_trades=dw + dl,
                        wins=dw,
                        losses=dl,
                        top_pnl=dt,
                        worst_pnl=dwo,
                    )
                except Exception as _e:
                    print(f"  [daily_summary] Telegram error: {_e}")

            # Reset counters for the new day
            state["_daily_start_equity"] = eq
            state["_daily_wins"]   = 0
            state["_daily_losses"] = 0
            state["_daily_top"]    = 0.0
            state["_daily_worst"]  = 0.0
            last_day = today

        # Sleep until 30s after next midnight UTC
        tomorrow = datetime(today.year, today.month, today.day,
                            tzinfo=timezone.utc) + __import__("datetime").timedelta(days=1)
        secs_until_midnight = (tomorrow - now).total_seconds() + 30
        secs_to_wait = min(secs_until_midnight, 300)  # check at most every 5 min
        try:
            await asyncio.wait_for(shutdown_event.wait(), timeout=secs_to_wait)
            break
        except asyncio.TimeoutError:
            pass


# ═══════════════════════════════════════════════════════════════════════════════
# Per-symbol worker
# ═══════════════════════════════════════════════════════════════════════════════

async def run_symbol(symbol, client, adapter, args, settings, tier_lev, params,
                     shutdown_event, state, is_live: bool):
    from app.broker.base import OrderRequest, OrderSide, OrderType
    from app.data.feed import LiveFeed
    from app.data.ingestor import OHLCVIngestor, TIMEFRAME_SECONDS
    from app.data.parquet_store import ParquetStore
    from app.risk.manager import RiskManager
    from app.strategy.base import BaseStrategy
    from app.strategy.signals import Signal, SignalAction
    from app.strategy.trend_following import TrendFollowingStrategy
    from app.strategy.mtf_context import HTFContext, LTFEntry
    from app.notify.telegram import TelegramNotifier

    s = symbol.replace("-USDT", "")
    store    = ParquetStore()
    strategy = TrendFollowingStrategy(symbol=symbol, params=params)
    risk     = RiskManager()
    notifier: TelegramNotifier | None = state.get("notifier")

    print(f"  [{s}] Loading warmup...")
    try:
        warmup = await OHLCVIngestor(client, store).poll_latest(symbol, args.timeframe, WINDOW_SIZE + 2)
    except Exception as e:
        print(f"  [{s}] ERROR loading warmup: {e}")
        return
    bw: deque = deque(warmup[:-1], maxlen=WINDOW_SIZE)

    if len(bw) >= strategy.min_bars_required:
        sig0 = strategy.on_bar(BaseStrategy.bars_to_df(list(bw)))
        m0   = sig0.meta or {}
        msg0, _ = _analyze_bar(bw[-1], sig0, m0, params)
        print(f"  [{s}] Ready  ${float(bw[-1].close):,.2f}  ADX={m0.get('adx',0):.1f}  → {msg0}")
    else:
        print(f"  [{s}] {len(bw)} bars (need {strategy.min_bars_required})")

    # ── MTF: initialise Higher and Lower TF helpers ───────────────────────────
    htf_ctx = HTFContext(symbol, htf_tf=HTF_TIMEFRAME)
    ltf_ent = LTFEntry(symbol, ltf_tf=LTF_TIMEFRAME)
    await htf_ctx.warmup(client, store)
    htf_bias = htf_ctx.get_bias()
    print(f"  [{s}] HTF bias: {htf_bias.label}  strength={htf_bias.strength:.2f}"
          f"  4H EMA50={htf_bias.ema_fast:,.2f}  ADX={htf_bias.adx:.1f}")

    balances = await adapter.get_balance()
    equity   = balances[0].total if balances else Decimal(str(args.balance))
    risk.initialize(equity)

    open_trade = None
    tf_secs    = TIMEFRAME_SECONDS.get(args.timeframe, 900)
    feed       = LiveFeed(client=client, store=store, symbol=symbol,
                          timeframe=args.timeframe, max_data_delay_s=tf_secs * 3)
    feed_iter  = feed.stream().__aiter__()

    try:
        while not shutdown_event.is_set():
            get_bar = asyncio.ensure_future(feed_iter.__anext__())
            wait_sd = asyncio.ensure_future(shutdown_event.wait())
            done, pending = await asyncio.wait([get_bar, wait_sd], return_when=asyncio.FIRST_COMPLETED)
            for t in pending:
                t.cancel()
                try: await t
                except (asyncio.CancelledError, StopAsyncIteration): pass
            if wait_sd in done: break
            try: bar = get_bar.result()
            except StopAsyncIteration: break

            bw.append(bar)
            if len(bw) < strategy.min_bars_required:
                continue

            # ── 1. SL / TP check (MUST run first — never skip) ────────────────
            # This runs BEFORE strategy so that even if strategy/HTF/LTF
            # crashes, open positions are always monitored and closed.
            # advance_bar() is called inside _check_sl_tp() via PositionManager.
            if open_trade:
                et, fill, cp = await _check_sl_tp(
                    open_trade, bar, adapter, strategy.strategy_id,
                    is_live=is_live, notifier=notifier,
                )

                # Live-mode retry on failed order (paper mode never fails)
                if et and et.endswith("_order_failed"):
                    print(f"  [{s}] Retrying close next bar ({et})")
                    continue  # skip strategy — just wait for next bar to retry

                elif et == "tp1" and fill:
                    # ── Partial close at TP1 ──────────────────────────────────
                    # TP1 updates (BE stop, best_price, patience) already applied
                    # by _check_sl_tp via apply_tp1_updates()
                    np_ = compute_leg_pnl(open_trade, fill.price, fill.qty, fill.fee)
                    fp = fill.qty / open_trade["original_qty"]
                    fi = open_trade["fee_in"] * fp
                    open_trade["total_partial_pnl"]  += np_
                    open_trade["total_partial_fees"] += fi + fill.fee
                    open_trade["qty"] -= fill.qty
                    trail_sl = float(open_trade["sl_price"])

                    print(
                        f"\n  [{s}] TP1 {cp*100:.0f}% @ ${float(fill.price):,.2f}"
                        f"  pnl=${float(np_):+.2f}"
                        f"  → trailing SL (floor=${trail_sl:,.2f})\n"
                    )
                    if notifier:
                        asyncio.create_task(notifier.tp1_hit(
                            symbol=symbol, exit_price=float(fill.price), close_pct=cp,
                            partial_pnl=float(np_), remaining_qty=float(open_trade["qty"]),
                            entry_price=float(open_trade["entry_price"]),
                        ))
                    state["positions"][symbol] = open_trade

                    # Edge case: TP1 consumed entire qty (e.g. cp=1.0).
                    # total_partial_pnl already holds np_ from line above,
                    # so we must NOT call _record_close (it would recompute and
                    # double-count the same leg). Write CSV directly instead.
                    if open_trade["qty"] < Decimal("0.001"):
                        tp = open_trade["total_partial_pnl"]
                        n  = open_trade["entry_price"] * open_trade["original_qty"]
                        _append_csv({
                            "trade_id": open_trade["trade_id"],
                            "symbol":   open_trade["symbol"],
                            "side":     open_trade["side"],
                            "strategy": open_trade["strategy"],
                            "tier":     open_trade["tier"],
                            "leverage": open_trade["leverage"],
                            "entry_time":  open_trade["entry_time"],
                            "entry_price": float(open_trade["entry_price"]),
                            "exit_time":   fill.timestamp.isoformat(),
                            "exit_price":  float(fill.price),
                            "exit_type":   et,
                            "qty":     float(open_trade["original_qty"]),
                            "pnl":     float(tp.quantize(Decimal("0.01"))),
                            "pnl_pct": round(float(tp / n * 100) if n else 0, 4),
                            "fees":    float(open_trade["total_partial_fees"].quantize(Decimal("0.01"))),
                            "signal_reason": open_trade["signal_reason"],
                        })
                        risk.record_fill(tp)
                        state["wins" if tp > 0 else "losses"] += 1
                        # Update daily counters for Telegram daily_summary
                        _pnl_f = float(tp)
                        if tp > 0: state["_daily_wins"]   = state.get("_daily_wins",   0) + 1
                        else:      state["_daily_losses"] = state.get("_daily_losses", 0) + 1
                        state["_daily_top"]   = max(state.get("_daily_top",   0.0), _pnl_f)
                        state["_daily_worst"] = min(state.get("_daily_worst", 0.0), _pnl_f)
                        # Notify Telegram — full close at TP1
                        w, l = state["wins"], state["losses"]; tt = w + l
                        if notifier:
                            asyncio.create_task(notifier.trade_closed(
                                symbol=symbol, side=open_trade["side"], exit_type="tp1",
                                exit_price=float(fill.price),
                                entry_price=float(open_trade["entry_price"]),
                                total_pnl=float(tp), qty=float(open_trade["original_qty"]),
                                total_trades=tt, wins=w, losses=l,
                            ))
                        if hasattr(strategy, "notify_trade_result"):
                            strategy.notify_trade_result(won=tp > 0)
                        if hasattr(strategy, "_bar_index"):
                            if open_trade["side"] == "LONG":
                                strategy._last_long_bar = -999
                            else:
                                strategy._last_short_bar = -999
                        # Update equity immediately after close
                        try:
                            balances = await adapter.get_balance()
                            equity   = balances[0].total if balances else equity
                            state["_balance_cache"] = (equity, time.monotonic())
                            state["equity"] = float(equity)
                        except Exception:
                            pass
                        open_trade = None
                        state["positions"].pop(symbol, None)

                elif et in ("tp2", "sl", "be_sl", "trailing_sl") and fill:
                    # ── Full close ────────────────────────────────────────────
                    tp = _record_close(open_trade, fill, et)
                    risk.record_fill(tp)
                    state["wins" if tp > 0 else "losses"] += 1
                    # Update daily counters for Telegram daily_summary
                    _pnl_f = float(tp)
                    if tp > 0: state["_daily_wins"]   = state.get("_daily_wins",   0) + 1
                    else:      state["_daily_losses"] = state.get("_daily_losses", 0) + 1
                    state["_daily_top"]   = max(state.get("_daily_top",   0.0), _pnl_f)
                    state["_daily_worst"] = min(state.get("_daily_worst", 0.0), _pnl_f)
                    w, l = state["wins"], state["losses"]; tt = w + l
                    label = "WIN" if tp > 0 else "LOSS"
                    print(
                        f"\n  [{s}] {et.upper():<12} @ ${float(fill.price):,.2f}"
                        f"  pnl=${float(tp):+.2f} [{label}]"
                        f"  total={tt} WR={w/tt*100:.0f}%\n"
                    )
                    if notifier:
                        asyncio.create_task(notifier.trade_closed(
                            symbol=symbol, side=open_trade["side"], exit_type=et,
                            exit_price=float(fill.price),
                            entry_price=float(open_trade["entry_price"]),
                            total_pnl=float(tp), qty=float(open_trade["original_qty"]),
                            total_trades=tt, wins=w, losses=l,
                        ))
                    if hasattr(strategy, "notify_trade_result"):
                        strategy.notify_trade_result(won=tp > 0)
                    # Reset cooldown for the direction just closed so the strategy
                    # can fire again immediately on the next valid bar — avoids an
                    # artificial wait after SL/TP closes, matching Pine behaviour.
                    if hasattr(strategy, "_bar_index"):
                        closed_side = open_trade["side"]
                        if closed_side == "LONG":
                            strategy._last_long_bar = -999
                        else:
                            strategy._last_short_bar = -999
                    # Refresh equity immediately so the next bar uses the correct
                    # post-close balance for position sizing (avoids 60s cache lag).
                    try:
                        balances = await adapter.get_balance()
                        equity   = balances[0].total if balances else equity
                        async with state["_balance_lock"]:
                            state["_balance_cache"] = (equity, time.monotonic())
                        state["equity"] = float(equity)
                    except Exception:
                        pass
                    open_trade = None
                    state["positions"].pop(symbol, None)

            # ── 2. Strategy + HTF (wrapped — crash here must NOT kill SL monitoring)
            df = BaseStrategy.bars_to_df(list(bw))
            try:
                refreshed = await htf_ctx.maybe_refresh(client, store, bar.ts)
                if refreshed:
                    htf_bias = htf_ctx.get_bias()
                    print(f"  [{s}] HTF updated: {htf_bias.label}  str={htf_bias.strength:.2f}")
            except Exception:
                pass  # non-fatal — keep using cached bias

            try:
                sig  = strategy.on_bar(df, htf_bias=htf_bias)
                meta = sig.meta or {}
            except Exception as _strat_e:
                print(f"  [{s}] Strategy error (non-fatal): {_strat_e}")
                sig  = Signal(action=SignalAction.HOLD, symbol=symbol,
                              ts=bar.ts, strategy_id=strategy.strategy_id,
                              reason="strategy_error")
                meta = {}

            # Update ATR for trailing SL on NEXT bar's check
            if open_trade:
                update_atr(open_trade, float(meta.get("atr", 0)))

            # ── 3. Balance refresh (shared cache) ─────────────────────────────
            _bal_cache = state["_balance_cache"]
            if _bal_cache is None or (time.monotonic() - _bal_cache[1]) > 60:
                async with state["_balance_lock"]:
                    _bal_cache = state["_balance_cache"]
                    if _bal_cache is None or (time.monotonic() - _bal_cache[1]) > 60:
                        try:
                            balances = await adapter.get_balance()
                            equity   = balances[0].total if balances else Decimal(str(args.balance))
                            state["_balance_cache"] = (equity, time.monotonic())
                        except Exception as _be:
                            equity = _bal_cache[0] if _bal_cache else Decimal(str(args.balance))
                            if state["_balance_cache"] is None:
                                state["_balance_cache"] = (equity, time.monotonic())
                    else:
                        equity = _bal_cache[0]
            else:
                equity = _bal_cache[0]

            risk.update_equity(equity)
            state["equity"] = float(equity)

            # ── 4. Status bar ─────────────────────────────────────────────────
            tier    = meta.get("confidence_tier", "X")
            lev     = tier_lev.get(tier, 1)
            msg, wl = _analyze_bar(bar, sig, meta, params)
            ema_slow = meta.get("ema_slow", 0)
            state.setdefault("sym_states", {})[symbol] = {
                "close": float(bar.close), "adx": meta.get("adx", 0),
                "trend": ("↑" if float(bar.close) > ema_slow else "↓") if ema_slow else "~",
                "adx_min": float(params.get("adx_min", 20)), "msg": msg, "watchlist": wl,
            }
            _print_bar_line(symbol, bar, sig, meta, open_trade, float(equity), params)

            # ── 5. Signal routing ─────────────────────────────────────────────
            #
            # REVERSAL SWAP: split into 5a (close if opposite) + 5b (open).
            # When a signal fires in the opposite direction of an open trade,
            # step 5a closes the position and step 5b immediately re-opens
            # in the new direction — all on the same bar. This avoids eating
            # a full SL loss when the trend has clearly reversed.
            #
            if not sig.is_actionable():
                continue
            ok, reason = risk.validate_signal(sig, equity)
            if not ok:
                print(f"  [{s}] Rejected: {reason}")
                continue
            if len(state["positions"]) >= args.max_positions and symbol not in state["positions"]:
                print(f"  [{s}] Max {args.max_positions} positions reached")
                continue

            tk    = await adapter.get_ticker(symbol)
            price = Decimal(str(tk["last"]))

            # ── 5a. Reversal close: if in trade and signal is opposite, close first ──
            if open_trade:
                is_long  = open_trade["side"] == "LONG"
                should_close = (
                    sig.action == SignalAction.CLOSE
                    or (is_long      and sig.action == SignalAction.SELL)
                    or (not is_long  and sig.action == SignalAction.BUY)
                )
                if should_close:
                    close_side     = OrderSide.SELL if is_long else OrderSide.BUY
                    close_pos_side = "LONG"          if is_long else "SHORT"
                    if not is_live:
                        fill = _make_paper_fill(open_trade, float(bar.close),
                                                open_trade["qty"], close_side)
                    else:
                        try:
                            _, fill = await adapter.place_order(OrderRequest(
                                symbol=symbol, side=close_side,
                                order_type=OrderType.MARKET,
                                qty=open_trade["qty"],
                                strategy_id=strategy.strategy_id,
                                extra={"positionSide": close_pos_side},
                            ))
                        except Exception as ce:
                            print(f"  [{s}] ⚠️  Reversal close failed: {ce}")
                            fill = None

                    if fill:
                        is_reversal = sig.action in (SignalAction.BUY, SignalAction.SELL)
                        exit_type = "reversal_swap" if is_reversal else "signal_close"
                        tp = _record_close(open_trade, fill, exit_type)
                        risk.record_fill(tp)
                        state["wins" if tp > 0 else "losses"] += 1
                        # Update daily counters for Telegram daily_summary
                        _pnl_f = float(tp)
                        if tp > 0: state["_daily_wins"]   = state.get("_daily_wins",   0) + 1
                        else:      state["_daily_losses"] = state.get("_daily_losses", 0) + 1
                        state["_daily_top"]   = max(state.get("_daily_top",   0.0), _pnl_f)
                        state["_daily_worst"] = min(state.get("_daily_worst", 0.0), _pnl_f)
                        label = "WIN" if tp > 0 else "LOSS"
                        if hasattr(strategy, "notify_trade_result"):
                            strategy.notify_trade_result(won=tp > 0)
                        w, l = state["wins"], state["losses"]
                        new_dir = "LONG" if sig.action == SignalAction.BUY else "SHORT"
                        swap_tag = f" → SWAP to {new_dir}" if is_reversal else ""
                        print(
                            f"\n  ┌{'─'*60}┐"
                            f"\n  │  TRADE CLOSED [{s}] [{label}]  ({exit_type})"
                            f"\n  │  {open_trade['side']} @ ${float(fill.price):,.2f}"
                            f"\n  │  pnl=${float(tp):+.2f}{swap_tag}"
                            f"\n  └{'─'*60}┘\n"
                        )
                        if notifier:
                            asyncio.create_task(notifier.trade_closed(
                                symbol=symbol, side=open_trade["side"],
                                exit_type=exit_type, exit_price=float(fill.price),
                                entry_price=float(open_trade["entry_price"]),
                                total_pnl=float(tp), qty=float(open_trade["original_qty"]),
                                total_trades=w + l, wins=w, losses=l,
                            ))
                        open_trade = None
                        state["positions"].pop(symbol, None)
                        # Reset cooldown so step 5b fires immediately on this bar —
                        # mirrors Pine Script reversal swap (cierra + reabre en la
                        # misma barra sin esperar el cooldown de la nueva dirección).
                        if is_reversal and hasattr(strategy, "_bar_index"):
                            if sig.action == SignalAction.BUY:
                                strategy._last_long_bar = -999
                            else:
                                strategy._last_short_bar = -999
                        # Bug 2 fix: refresh equity immediately after reversal close
                        # so step 5b uses the correct post-close balance for sizing,
                        # not the stale 60s cache.
                        try:
                            balances = await adapter.get_balance()
                            equity   = balances[0].total if balances else equity
                            async with state["_balance_lock"]:
                                state["_balance_cache"] = (equity, time.monotonic())
                            state["equity"] = float(equity)
                            price = Decimal(str((await adapter.get_ticker(symbol))["last"]))
                        except Exception:
                            pass
                        # open_trade is now None → step 5b will immediately re-open
                    else:
                        continue  # close failed — retry next bar
                else:
                    # Same-direction signal while in trade — ignore
                    continue

            # ── 5b. Open new position (also fires after reversal swap) ────────
            if not open_trade and sig.action in (SignalAction.BUY, SignalAction.SELL):
                    il  = sig.action == SignalAction.BUY
                    bq  = risk.compute_order_qty(sig, equity, price)
                    max_notional = equity * Decimal(str(args.alloc_pct)) / Decimal("100")
                    mq  = (max_notional / price).quantize(Decimal("0.001"))
                    qty = min(bq * lev, mq).quantize(Decimal("0.001"))
                    if qty < Decimal("0.001"):
                        continue
                    side      = OrderSide.BUY  if il else OrderSide.SELL
                    pos_side  = "LONG"          if il else "SHORT"
                    sd        = "LONG"          if il else "SHORT"
                    sf        = float(sig.stop_loss)   if sig.stop_loss   else 0
                    tp1       = float(sig.take_profit) if sig.take_profit else 0
                    tp2       = float(meta.get("tp2", 0))

                    # ── LTF entry refinement ──────────────────────────────────
                    # After the 15M signal is approved, scan the 5M chart to see
                    # if we can get a tighter entry (micro-pullback / consolidation).
                    # If found: use the better entry price and a tighter SL.
                    # The order is still placed at market — the benefit is that we
                    # enter on a slightly better candle rather than chasing the close.
                    ltf_result = None
                    atr_15m = float(meta.get("atr", 0))
                    if atr_15m > 0:
                        try:
                            ltf_bars = await ltf_ent.fetch(client, store)
                            ltf_result = ltf_ent.find_entry(
                                ltf_bars,
                                direction=sd,
                                sl_price=sf,
                                entry_price=float(price),
                                atr_15m=atr_15m,
                            )
                            if ltf_result.found:
                                # Override SL with the tighter one from LTF
                                sf = ltf_result.sl_price
                                sig._stop_loss = Decimal(str(round(sf, 8)))
                                # Recalculate TP1/TP2 using same R multiples
                                rr1 = meta.get("rr_tp1", 1.5)
                                rr2 = meta.get("rr_tp2", 2.5)
                                risk_new = abs(float(price) - sf)
                                if sd == "LONG":
                                    tp1 = float(price) + risk_new * rr1
                                    tp2 = float(price) + risk_new * rr2
                                else:
                                    tp1 = float(price) - risk_new * rr1
                                    tp2 = float(price) - risk_new * rr2
                                print(
                                    f"  [{s}] LTF entry: {ltf_result.reason}"
                                    f"  SL improved by {ltf_result.improvement_pct:+.3f}%"
                                    f"  new_SL=${sf:,.4f}"
                                )
                        except Exception as _ltf_e:
                            pass  # non-fatal — use original SL/TP

                    try:
                        _, fill = await adapter.place_order(OrderRequest(
                            symbol=symbol, side=side, order_type=OrderType.MARKET,
                            qty=qty, strategy_id=strategy.strategy_id,
                            extra={"positionSide": pos_side},
                        ))
                    except Exception as order_err:
                        err_msg = str(order_err)
                        print(f"\n  [{s}] ⚠️  ORDER FAILED: {err_msg}")
                        if notifier:
                            asyncio.create_task(notifier.send(
                                f"⚠️ <b>ORDER FAILED — {symbol}</b>\n"
                                f"🚫 Error: <code>{err_msg[:200]}</code>\n"
                                f"  {sd} Tier {tier} {lev}x\n"
                                f"  Entry ≈ <code>${float(price):,.2f}</code>  qty={float(qty):.4f}\n"
                                f"  SL=<code>${sf:,.2f}</code>  TP1=<code>${tp1:,.2f}</code>\n"
                                f"  {sig.reason[:80]}\n"
                            ))
                        fill = None

                    if fill:
                        rr = abs(float(fill.price) - sf) / float(fill.price) * 100 if sf else 0
                        open_trade = create_trade(
                            trade_id=str(uuid.uuid4())[:8],
                            symbol=symbol,
                            side=sd,
                            entry_price=fill.price,
                            qty=fill.qty,
                            fee_in=fill.fee,
                            sl_price=round(sf, 8) if sf else None,
                            tp1_price=round(tp1, 8) if tp1 else None,
                            tp2_price=round(tp2, 8) if tp2 else None,
                            tp1_close_pct=meta.get("tp1_close_pct", 0.33),
                            soft_sl_bars=meta.get("soft_sl_bars", 0),
                            trailing_atr=meta.get("trailing_atr", 1.0),
                            current_atr=float(meta.get("atr", 0)) if meta.get("atr") else 0,
                            signal_reason=sig.reason,
                            extra={
                                "strategy":   strategy.strategy_id,
                                "tier":       tier,
                                "leverage":   lev,
                                "entry_time": fill.timestamp.isoformat(),
                                "htf_bias":   htf_bias.label if htf_bias else "off",
                                "ltf_reason": ltf_result.reason if ltf_result else "off",
                            },
                        )
                        state["positions"][symbol] = open_trade
                        notional   = float(qty) * float(fill.price)
                        alloc_used = notional / float(equity) * 100
                        try:
                            _htf_str = f"{htf_bias.strength:.2f}" if htf_bias else "0"
                            _ltf_str = ltf_result.reason if ltf_result else "off"
                            print(
                                f"\n  ┌{'─'*60}┐"
                                f"\n  │  TRADE OPENED [{s}] Tier {tier} {lev}x"
                                f"\n  │  {sd} @ ${float(fill.price):,.2f}  qty={float(qty):.4f}"
                                f"\n  │  Notional: ${notional:,.2f} ({alloc_used:.1f}% of equity)"
                                f"\n  │  SL=${sf:,.2f} ({rr:.1f}%)  TP1=${tp1:,.2f}  TP2=${tp2:,.2f}"
                                f"\n  │  soft_sl_bars={meta.get('soft_sl_bars',0)} ({meta.get('soft_sl_bars',0)*15//60}h patience)"
                                f"\n  │  HTF: {htf_bias.label if htf_bias else 'off'}  str={_htf_str}"
                                f"\n  │  LTF: {_ltf_str}"
                                f"\n  │  {sig.reason}"
                                f"\n  └{'─'*60}┘\n"
                            )
                        except Exception as _pe:
                            print(f"\n  [{s}] TRADE OPENED {sd} @ ${float(fill.price):,.2f} (print error: {_pe})\n")
                        if notifier:
                            asyncio.create_task(notifier.trade_opened(
                                symbol=symbol, side=sd, tier=tier, leverage=lev,
                                entry_price=float(fill.price), qty=float(qty),
                                stop_loss=sf or None, tp1=tp1 or None, tp2=tp2 or None,
                                session=meta.get("session", ""),
                                reason=sig.reason, equity=float(equity),
                            ))

            # (old else block removed — reversal logic now in step 5a above)

    except asyncio.CancelledError:
        pass
    except Exception as e:
        import traceback
        print(f"  [{s}] Error: {e}")
        traceback.print_exc()
    finally:
        feed.stop()
        if open_trade and open_trade["qty"] > Decimal("0.001"):
            print(f"  [{s}] Closing {open_trade['side']} on shutdown...")
            try:
                il         = open_trade["side"] == "LONG"
                close_side = OrderSide.SELL if il else OrderSide.BUY
                bc_price   = float(bw[-1].close) if bw else float(open_trade["entry_price"])
                if not is_live:
                    fill = _make_paper_fill(open_trade, bc_price,
                                            open_trade["qty"], close_side)
                else:
                    _, fill = await adapter.place_order(OrderRequest(
                        symbol=symbol, side=close_side,
                        order_type=OrderType.MARKET,
                        qty=open_trade["qty"],
                        strategy_id=strategy.strategy_id,
                        extra={"positionSide": "LONG" if il else "SHORT"},
                    ))
                if fill:
                    tp = _record_close(open_trade, fill, "shutdown")
                    print(f"  [{s}] Closed @ ${float(fill.price):,.2f}  pnl=${float(tp):+.2f}")
            except Exception as e:
                print(f"  [{s}] WARNING: shutdown close failed: {e}")


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
    symbols = [f"{b}-USDT" for b in args.symbols]
    print(f"\nChecking BingX API for {len(symbols)} symbols...")
    for sym in symbols:
        try:
            tk = await c.get_ticker(sym)
            print(f"  [OK] {sym:<12} ${float(tk['last']):>12,.4f}")
        except Exception as e:
            print(f"  [FAIL] {sym}: {e}")
    try:
        raw = await c.get_balance()
        u = next((a for a in raw if "USDT" in str(a.get("asset", ""))), None)
        print(f"\n  Balance: {u.get('balance','?') if u else '?'} USDT")
    except Exception as e:
        print(f"\n  Balance: {e}")
    await c.close()
    print()


# ═══════════════════════════════════════════════════════════════════════════════
# Main entry point
# ═══════════════════════════════════════════════════════════════════════════════

async def main(args):
    from app.config import get_settings
    from app.core.logging import configure_logging
    settings = get_settings()
    configure_logging(log_level="WARNING", log_format="console")
    from app.broker.bingx_adapter import BingXAdapter
    from app.broker.bingx_client import BingXClient
    from app.broker.paper_adapter import PaperAdapter
    from app.notify.telegram import TelegramNotifier

    _ensure_csv()
    symbols  = [f"{b}-USDT" for b in args.symbols]
    tier_lev = {"A": args.lev_a, "B": args.lev_b, "C": args.lev_c, "X": 1}
    params   = dict(BEST_PARAMS)
    if args.params_file:
        with open(args.params_file) as f:
            d = json.load(f)
        params.update(d.get("params", d))

    client = BingXClient(
        api_key=settings.bingx_api_key, api_secret=settings.bingx_api_secret,
        base_url=settings.bingx_base_url, market_type=settings.bingx_market_type,
    )

    is_live = args.live_bingx
    if is_live:
        adapter = BingXAdapter(client=client)
        mx = max(args.lev_a, args.lev_b, args.lev_c)
        for sym in symbols:
            try:
                await client.set_leverage(sym, mx)
                print(f"  [{sym}] leverage set to {mx}x")
            except Exception as e:
                print(f"  [{sym}] leverage warning: {e}")
        mode = "LIVE BingX — REAL USDT!"
    else:
        adapter = PaperAdapter(client=client, initial_balance=Decimal(str(args.balance)))
        mode    = "PAPER simulation (real prices, virtual balance)"

    short_status = "enabled" if params.get("allow_short", True) else "disabled"
    print(f"\n{'═'*72}")
    print(f"  MULTI-SYMBOL PAPER TRADING — TrendFollowing (LONG + SHORT)")
    print(f"{'═'*72}")
    print(f"  Mode:           {mode}")
    print(f"  Symbols:        {', '.join(symbols)}")
    print(f"  Timeframe:      {args.timeframe}")
    print(f"  Balance:        ${args.balance:,.2f}")
    print(f"  Max positions:  {args.max_positions}")
    print(f"  Alloc/trade:    {args.alloc_pct}% of equity")
    print(f"  Leverage:       A={args.lev_a}x B={args.lev_b}x C={args.lev_c}x")
    print(f"  Directions:     LONG + SHORT ({short_status})")
    print(f"  ADX:            min={params['adx_min']} strong={params['adx_strong']}")
    print(f"  Patience:       {params['soft_sl_bars']} bars ({params['soft_sl_bars']*15//60}h)")
    print(f"  Trade log:      {TRADES_CSV}")
    if not is_live:
        print(f"\n  Safe mode: no real orders. Use --live-bingx for real trading.")
    print(f"{'═'*72}")

    notifier = TelegramNotifier.from_env()
    if notifier:
        tg_ok = await notifier.bot_started(symbols=symbols, mode=mode, balance=args.balance)
        if tg_ok:
            print(f"  Telegram: ✓ notifications enabled (test message sent)")
        else:
            print(f"  Telegram: ✗ FAILED to send — check TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env")
            notifier = None  # disable to avoid silent failures
    else:
        print(f"  Telegram: not configured")
    print(f"\n  Ctrl+C to stop. Loading warmup...\n")

    shutdown_event = asyncio.Event()
    loop           = asyncio.get_running_loop()
    worker_tasks: list[asyncio.Task] = []

    def _shutdown():
        print("\n  Shutting down all symbols...")
        shutdown_event.set()
        for t in worker_tasks:
            if not t.done():
                t.cancel()

    try:
        loop.add_signal_handler(signal.SIGINT,  _shutdown)
        loop.add_signal_handler(signal.SIGTERM, _shutdown)
    except NotImplementedError:
        signal.signal(signal.SIGINT, lambda s, f: _shutdown())

    state = {
        "equity":    args.balance,
        "wins":      0,
        "losses":    0,
        "positions": {},
        "sym_states": {},
        "notifier":  notifier,
        "_balance_cache":     None,
        "_balance_lock":      asyncio.Lock(),
        # Daily summary counters (updated by every trade close, read by _daily_summary_loop)
        "_daily_start_equity": args.balance,
        "_daily_wins":    0,
        "_daily_losses":  0,
        "_daily_top":     0.0,
        "_daily_worst":   0.0,
    }

    for sym in symbols:
        t = asyncio.create_task(run_symbol(
            sym, client, adapter, args, settings, tier_lev, params,
            shutdown_event, state, is_live=is_live,
        ))
        worker_tasks.append(t)

    status_task = asyncio.create_task(
        _status_loop(client, state, symbols, shutdown_event)
    )
    worker_tasks.append(status_task)

    daily_task = asyncio.create_task(
        _daily_summary_loop(state, args, shutdown_event)
    )
    worker_tasks.append(daily_task)

    results = await asyncio.gather(*worker_tasks, return_exceptions=True)
    for i, r in enumerate(results):
        if isinstance(r, Exception) and not isinstance(r, asyncio.CancelledError):
            print(f"  Task {i} error: {r}")

    await adapter.shutdown()

    w, l    = state["wins"], state["losses"]
    t       = w + l
    final_eq = state.get("equity", args.balance)
    net_pnl  = final_eq - args.balance
    if notifier:
        await notifier.bot_stopped(total_trades=t, net_pnl=net_pnl, wins=w, losses=l)
    print(f"\n{'═'*72}\n  SESSION SUMMARY\n{'═'*72}")
    print(f"  Symbols: {', '.join(symbols)}")
    if t:
        print(f"  Trades:  {t}  (W={w} L={l} WR={w/t*100:.0f}%)")
    else:
        print(f"  Trades:  0")
    print(f"  PnL:     ${net_pnl:+,.2f}")
    print(f"  Log:     {TRADES_CSV}\n{'═'*72}\n")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Multi-symbol paper trading (LONG+SHORT)")
    p.add_argument("--symbols",       nargs="+", default=DEFAULT_BASES)
    p.add_argument("--timeframe",     default="15m")
    p.add_argument("--balance",       type=float, default=10000.0)
    p.add_argument("--max-positions", type=int,   default=3,    dest="max_positions")
    p.add_argument("--alloc-pct",     type=float, default=10.0, dest="alloc_pct",
                   help="Max %% of equity per trade BEFORE leverage (default: 10%%)")
    p.add_argument("--lev-a",  type=int, default=5, dest="lev_a")
    p.add_argument("--lev-b",  type=int, default=3, dest="lev_b")
    p.add_argument("--lev-c",  type=int, default=1, dest="lev_c")
    p.add_argument("--params-file", default=None, dest="params_file")
    p.add_argument("--check",      action="store_true")
    p.add_argument("--live-bingx", action="store_true", dest="live_bingx",
                   help="Send REAL orders to BingX (real USDT!)")
    args = p.parse_args()
    if args.check:
        asyncio.run(check_connection(args))
    else:
        asyncio.run(main(args))