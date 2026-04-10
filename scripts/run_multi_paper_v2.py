"""
Multi-symbol paper trading — V2 (Pine Script aligned).

═══════════════════════════════════════════════════════════════════════════════
KEY CHANGE vs run_multi_paper.py:
  The strategy (TrendFollowingV2) now manages the ENTIRE trade state machine
  internally — SL/TP/TP1→BE/TP2/reversal swap — just like Pine Script v5.1.

  This runner ONLY:
    1. Feeds bars to the strategy
    2. Executes the signals it returns (BUY/SELL/CLOSE/PARTIAL_CLOSE)
    3. Sends Telegram notifications
    4. Logs trades to CSV

  NO SL/TP logic lives here anymore. The strategy IS the source of truth.
═══════════════════════════════════════════════════════════════════════════════

Usage:
    python scripts/run_multi_paper_v2.py                        # Safe simulation
    python scripts/run_multi_paper_v2.py --check                # Test API
    python scripts/run_multi_paper_v2.py --symbols BTC ETH SOL  # Fewer symbols
    python scripts/run_multi_paper_v2.py --live-bingx            # REAL orders!
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


HTF_TIMEFRAME = "4h"
DEFAULT_BASES = ["BTC", "ETH", "SOL", "XRP", "BNB"]
WINDOW_SIZE   = 300
STATUS_SECS   = 300
TRADES_CSV    = Path("data/paper_trades_v2.csv")
TRADES_HEADER = [
    "trade_id", "symbol", "side", "strategy", "tier", "leverage",
    "entry_time", "entry_price", "exit_time", "exit_price", "exit_type",
    "qty", "pnl", "pnl_pct", "fees", "signal_reason",
]

BEST_PARAMS: dict = {
    "adx_min": 20, "adx_strong": 35,
    "pullback_tolerance_atr": 1.0,
    "allow_short": True,
    "min_confidence": 0,
    "sig_cooldown": 5,
    "enable_reversal": True,
    # SL structure
    "sl_swing_lookback": 50, "sl_swing_window": 3,
    "sl_min_atr": 1.0, "sl_max_atr": 2.5, "sl_buffer_atr": 0.3,
    # TP per tier
    "tp1_r_A": 1.5, "tp2_r_A": 3.0,
    "tp1_r_B": 1.5, "tp2_r_B": 2.5,
    "tp1_r_C": 1.0, "tp2_r_C": 1.5,
    # Session
    "session_mult_us": 1.0, "session_mult_eu": 0.75, "session_mult_other": 0.5,
    "use_session_filter": True,
    # Streak
    "streak_euphoria_mult": 0.75, "use_streak_adj": True,
    # Patience
    "soft_sl_bars": 48, "use_patience": True,
}

COMMISSION_RATE = Decimal("0.00075")


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
# Synthetic fill for paper-mode
# ═══════════════════════════════════════════════════════════════════════════════

def _make_paper_fill(symbol: str, price: float, qty: Decimal, side):
    from app.broker.base import FillResult
    px = Decimal(str(price))
    fee = px * qty * COMMISSION_RATE
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
# Per-symbol worker (SIMPLIFIED — strategy handles all trade logic)
# ═══════════════════════════════════════════════════════════════════════════════

async def run_symbol(symbol, client, adapter, args, settings, tier_lev, params,
                     shutdown_event, state, is_live: bool):
    from app.broker.base import OrderRequest, OrderSide, OrderType
    from app.data.feed import LiveFeed
    from app.data.ingestor import OHLCVIngestor, TIMEFRAME_SECONDS
    from app.data.parquet_store import ParquetStore
    from app.risk.manager import RiskManager
    from app.strategy.base import BaseStrategy
    from app.strategy.signals import SignalAction
    from app.strategy.trend_following_v2 import TrendFollowingV2
    from app.strategy.mtf_context import HTFContext
    from app.notify.telegram import TelegramNotifier

    s = symbol.replace("-USDT", "")
    store    = ParquetStore()
    strategy = TrendFollowingV2(symbol=symbol, params=params)
    risk     = RiskManager()
    notifier: TelegramNotifier | None = state.get("notifier")

    print(f"  [{s}] Loading warmup...")
    try:
        warmup = await OHLCVIngestor(client, store).poll_latest(
            symbol, args.timeframe, WINDOW_SIZE + 2)
    except Exception as e:
        print(f"  [{s}] ERROR loading warmup: {e}")
        return
    bw: deque = deque(warmup[:-1], maxlen=WINDOW_SIZE)
    last_processed_ts = bw[-1].ts if bw else None
    # Force flat state — runner has no open position at startup
    strategy.force_close()
    print(f"  [{s}] Ready — {len(bw)} bars loaded")

    # HTF bias
    htf_ctx = HTFContext(symbol, htf_tf=HTF_TIMEFRAME)
    await htf_ctx.warmup(client, store)
    htf_bias = htf_ctx.get_bias()
    print(f"  [{s}] HTF: {htf_bias.label}  str={htf_bias.strength:.2f}")

    balances = await adapter.get_balance()
    equity   = balances[0].total if balances else Decimal(str(args.balance))
    risk.initialize(equity)

    # ── Open trade tracking (for execution only — strategy owns the logic) ──
    open_trade: dict | None = None  # {entry_price, qty, side, ...}

    tf_secs = TIMEFRAME_SECONDS.get(args.timeframe, 900)
    feed    = LiveFeed(client=client, store=store, symbol=symbol,
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
                try: await t
                except (asyncio.CancelledError, StopAsyncIteration): pass
            if wait_sd in done:
                break
            try:
                bar = get_bar.result()
            except StopAsyncIteration:
                break

            # Skip bars already seen in warmup
            if last_processed_ts is not None and bar.ts <= last_processed_ts:
                continue
            last_processed_ts = bar.ts

            bw.append(bar)
            if len(bw) < strategy.min_bars_required:
                continue

            # ── Refresh HTF bias ──────────────────────────────────────
            try:
                if await htf_ctx.maybe_refresh(client, store, bar.ts):
                    htf_bias = htf_ctx.get_bias()
                    print(f"  [{s}] HTF updated: {htf_bias.label}")
            except Exception:
                pass

            # ── Run strategy (THE source of truth) ────────────────────
            df = BaseStrategy.bars_to_df(list(bw))
            try:
                signals = strategy.on_bar_all(df, htf_bias=htf_bias)
            except Exception as e:
                print(f"  [{s}] Strategy error: {e}")
                continue

            # ── Refresh equity ────────────────────────────────────────
            _bal_cache = state.get("_balance_cache")
            if _bal_cache is None or (time.monotonic() - _bal_cache[1]) > 60:
                try:
                    balances = await adapter.get_balance()
                    equity = balances[0].total if balances else equity
                    state["_balance_cache"] = (equity, time.monotonic())
                except Exception:
                    pass
            else:
                equity = _bal_cache[0]
            state["equity"] = float(equity)

            # ── Print bar status ──────────────────────────────────────
            ts_str = bar.ts.strftime("%H:%M")
            close  = float(bar.close)
            meta   = signals[0].meta if signals else {}
            adx    = meta.get("adx", 0)
            trade_st = strategy.trade_state
            pos_str = ""
            if not trade_st.is_flat():
                phase = "TRAIL" if trade_st.tp1_hit else ("LONG" if trade_st.is_long() else "SHORT")
                bars_in = strategy._bar_index - trade_st.start_bar
                upnl = (close - trade_st.entry) if trade_st.is_long() else (trade_st.entry - close)
                pos_str = f"  [{phase} {bars_in}b SL={trade_st.sl:.2f} uPnL={upnl:+.2f}]"

            primary = signals[0] if signals else None
            reason = primary.reason if primary else "?"
            action = primary.action.value if primary else "HOLD"
            if action == "HOLD":
                print(f"  {ts_str} {s:<4} ${close:>10,.2f} ADX={adx:.1f}  {reason[:50]}{pos_str}")
            else:
                print(f"  {ts_str} {s:<4} ${close:>10,.2f} ADX={adx:.1f}  *** {action} *** {reason[:40]}{pos_str}")

            # ── Execute signals ───────────────────────────────────────
            for sig in signals:
                if not sig.is_actionable():
                    continue

                # ── CLOSE (SL/TP2/reversal swap close) ────────────────
                if sig.action == SignalAction.CLOSE:
                    if open_trade is None:
                        continue

                    exit_price = sig.meta.get("exit_price", close)
                    exit_type  = sig.meta.get("exit_type", "close")
                    il = open_trade["side"] == "LONG"

                    if not is_live:
                        fill = _make_paper_fill(
                            symbol, exit_price, open_trade["qty"],
                            OrderSide.SELL if il else OrderSide.BUY)
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
                            print(f"  [{s}] CLOSE failed: {e}")
                            continue

                    if fill:
                        pnl = _calc_pnl(open_trade, fill)
                        _log_trade(open_trade, fill, exit_type, pnl, strategy)
                        state["wins" if pnl > 0 else "losses"] += 1
                        w, l = state["wins"], state["losses"]
                        print(f"\n  [{s}] {exit_type.upper()} @ ${float(fill.price):,.2f}"
                              f"  pnl=${float(pnl):+.2f}  W={w} L={l}\n")

                        if notifier:
                            asyncio.create_task(notifier.trade_closed(
                                symbol=symbol, side=open_trade["side"],
                                exit_type=exit_type, exit_price=float(fill.price),
                                entry_price=float(open_trade["entry_price"]),
                                total_pnl=float(pnl),
                                qty=float(open_trade["original_qty"]),
                                total_trades=w + l, wins=w, losses=l,
                            ))
                        if hasattr(strategy, "notify_trade_result"):
                            strategy.notify_trade_result(won=pnl > 0)

                        open_trade = None
                        state["positions"].pop(symbol, None)

                        # Refresh equity after close
                        try:
                            balances = await adapter.get_balance()
                            equity = balances[0].total if balances else equity
                            state["_balance_cache"] = (equity, time.monotonic())
                            state["equity"] = float(equity)
                        except Exception:
                            pass

                # ── PARTIAL_CLOSE (TP1 hit) ───────────────────────────
                elif sig.action == SignalAction.PARTIAL_CLOSE:
                    if open_trade is None:
                        continue

                    exit_price = sig.meta.get("exit_price", close)
                    close_pct  = sig.meta.get("close_pct", 0.33)
                    il = open_trade["side"] == "LONG"
                    close_qty = (open_trade["qty"] * Decimal(str(close_pct))).quantize(Decimal("0.001"))
                    close_qty = min(close_qty, open_trade["qty"])

                    if not is_live:
                        fill = _make_paper_fill(
                            symbol, exit_price, close_qty,
                            OrderSide.SELL if il else OrderSide.BUY)
                    else:
                        try:
                            _, fill = await adapter.place_order(OrderRequest(
                                symbol=symbol,
                                side=OrderSide.SELL if il else OrderSide.BUY,
                                order_type=OrderType.MARKET,
                                qty=close_qty,
                                strategy_id=strategy.strategy_id,
                                extra={"positionSide": "LONG" if il else "SHORT"},
                            ))
                        except Exception as e:
                            print(f"  [{s}] TP1 CLOSE failed: {e}")
                            continue

                    if fill:
                        partial_pnl = _calc_partial_pnl(open_trade, fill)
                        open_trade["qty"] -= fill.qty
                        open_trade["total_partial_pnl"] += partial_pnl
                        open_trade["total_partial_fees"] += fill.fee

                        print(f"\n  [{s}] TP1 {close_pct*100:.0f}% @ ${float(fill.price):,.2f}"
                              f"  pnl=${float(partial_pnl):+.2f}  → trailing SL\n")

                        if notifier:
                            asyncio.create_task(notifier.tp1_hit(
                                symbol=symbol, exit_price=float(fill.price),
                                close_pct=close_pct, partial_pnl=float(partial_pnl),
                                remaining_qty=float(open_trade["qty"]),
                                entry_price=float(open_trade["entry_price"]),
                            ))

                        # If qty exhausted (all closed via partials)
                        if open_trade["qty"] < Decimal("0.001"):
                            total_pnl = open_trade["total_partial_pnl"]
                            state["wins" if total_pnl > 0 else "losses"] += 1
                            open_trade = None
                            state["positions"].pop(symbol, None)

                # ── BUY / SELL (new entry) ────────────────────────────
                elif sig.action in (SignalAction.BUY, SignalAction.SELL):
                    if open_trade is not None:
                        continue  # shouldn't happen — strategy manages state

                    if len(state["positions"]) >= args.max_positions and symbol not in state["positions"]:
                        print(f"  [{s}] Max {args.max_positions} positions reached")
                        continue

                    ok, reject_reason = risk.validate_signal(sig, equity)
                    if not ok:
                        print(f"  [{s}] Rejected: {reject_reason}")
                        continue

                    tk = await adapter.get_ticker(symbol)
                    price = Decimal(str(tk["last"]))

                    # Position sizing
                    il   = sig.action == SignalAction.BUY
                    tier = sig.meta.get("confidence_tier", "X")
                    lev  = tier_lev.get(tier, 1)
                    bq   = risk.compute_order_qty(sig, equity, price)
                    max_notional = equity * Decimal(str(args.alloc_pct)) / Decimal("100")
                    mq   = (max_notional / price).quantize(Decimal("0.001"))
                    qty  = min(bq * lev, mq).quantize(Decimal("0.001"))
                    if qty < Decimal("0.001"):
                        continue

                    side     = OrderSide.BUY if il else OrderSide.SELL
                    pos_side = "LONG" if il else "SHORT"

                    try:
                        _, fill = await adapter.place_order(OrderRequest(
                            symbol=symbol, side=side,
                            order_type=OrderType.MARKET, qty=qty,
                            strategy_id=strategy.strategy_id,
                            extra={"positionSide": pos_side},
                        ))
                    except Exception as e:
                        print(f"  [{s}] ORDER FAILED: {e}")
                        fill = None

                    if fill:
                        sl  = sig.meta.get("sl", 0)
                        tp1 = sig.meta.get("tp1", 0)
                        tp2 = sig.meta.get("tp2", 0)

                        open_trade = {
                            "trade_id": str(uuid.uuid4())[:8],
                            "symbol": symbol,
                            "side": "LONG" if il else "SHORT",
                            "entry_price": fill.price,
                            "qty": fill.qty,
                            "original_qty": fill.qty,
                            "fee_in": fill.fee,
                            "total_partial_pnl": Decimal("0"),
                            "total_partial_fees": Decimal("0"),
                            "strategy": strategy.strategy_id,
                            "tier": tier,
                            "leverage": lev,
                            "entry_time": fill.timestamp.isoformat(),
                            "signal_reason": sig.reason,
                        }
                        state["positions"][symbol] = open_trade

                        notional = float(qty) * float(fill.price)
                        print(
                            f"\n  ┌{'─'*55}┐"
                            f"\n  │  TRADE OPENED [{s}] Tier {tier} {lev}x"
                            f"\n  │  {'LONG' if il else 'SHORT'} @ ${float(fill.price):,.2f}"
                            f"  qty={float(qty):.4f}"
                            f"\n  │  SL=${sl:.2f}  TP1=${tp1:.2f}  TP2=${tp2:.2f}"
                            f"\n  │  {sig.reason[:60]}"
                            f"\n  └{'─'*55}┘\n"
                        )

                        if notifier:
                            asyncio.create_task(notifier.trade_opened(
                                symbol=symbol,
                                side="LONG" if il else "SHORT",
                                tier=tier, leverage=lev,
                                entry_price=float(fill.price),
                                qty=float(qty),
                                stop_loss=sl or None,
                                tp1=tp1 or None, tp2=tp2 or None,
                                session=sig.meta.get("session", ""),
                                reason=sig.reason,
                                equity=float(equity),
                            ))

    except asyncio.CancelledError:
        pass
    except Exception as e:
        import traceback
        print(f"  [{s}] Error: {e}")
        traceback.print_exc()
    finally:
        feed.stop()
        # Close open trade on shutdown
        if open_trade and open_trade["qty"] > Decimal("0.001"):
            print(f"  [{s}] Closing on shutdown...")
            try:
                il = open_trade["side"] == "LONG"
                bc = float(bw[-1].close) if bw else float(open_trade["entry_price"])
                fill = _make_paper_fill(
                    symbol, bc, open_trade["qty"],
                    OrderSide.SELL if il else OrderSide.BUY)
                pnl = _calc_pnl(open_trade, fill)
                _log_trade(open_trade, fill, "shutdown", pnl, strategy)
                print(f"  [{s}] Closed @ ${bc:,.2f}  pnl=${float(pnl):+.2f}")
            except Exception as e:
                print(f"  [{s}] Shutdown close failed: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# PnL helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _calc_pnl(ot: dict, fill) -> Decimal:
    il = ot["side"] == "LONG"
    entry = ot["entry_price"]
    diff = (fill.price - entry) if il else (entry - fill.price)
    gross = diff * fill.qty
    fee_pct = fill.qty / ot["original_qty"]
    fee_in = ot["fee_in"] * fee_pct
    return gross - fee_in - fill.fee + ot.get("total_partial_pnl", Decimal("0"))


def _calc_partial_pnl(ot: dict, fill) -> Decimal:
    il = ot["side"] == "LONG"
    entry = ot["entry_price"]
    diff = (fill.price - entry) if il else (entry - fill.price)
    gross = diff * fill.qty
    fee_pct = fill.qty / ot["original_qty"]
    fee_in = ot["fee_in"] * fee_pct
    return gross - fee_in - fill.fee


def _log_trade(ot: dict, fill, exit_type: str, pnl: Decimal, strategy):
    n = ot["entry_price"] * ot["original_qty"]
    # fee_in is the full entry fee; prorate it for the remaining qty at close
    # (the rest was already accounted in partial closes)
    fee_pct = fill.qty / ot["original_qty"] if ot["original_qty"] else Decimal("1")
    proportional_fee_in = ot["fee_in"] * fee_pct
    total_fees = proportional_fee_in + ot.get("total_partial_fees", Decimal("0")) + fill.fee
    _append_csv({
        "trade_id": ot["trade_id"], "symbol": ot["symbol"], "side": ot["side"],
        "strategy": ot.get("strategy", ""), "tier": ot.get("tier", ""),
        "leverage": ot.get("leverage", 1),
        "entry_time": ot.get("entry_time", ""),
        "entry_price": float(ot["entry_price"]),
        "exit_time": fill.timestamp.isoformat(),
        "exit_price": float(fill.price),
        "exit_type": exit_type,
        "qty": float(ot["original_qty"]),
        "pnl": float(pnl.quantize(Decimal("0.01"))),
        "pnl_pct": round(float(pnl / n * 100) if n else 0, 4),
        "fees": float(total_fees.quantize(Decimal("0.01"))),
        "signal_reason": ot.get("signal_reason", ""),
    })


# ═══════════════════════════════════════════════════════════════════════════════
# Status loop
# ═══════════════════════════════════════════════════════════════════════════════

async def _status_loop(state, symbols, shutdown_event):
    await asyncio.sleep(STATUS_SECS)
    while not shutdown_event.is_set():
        try:
            now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
            eq = state.get("equity", 0)
            w, l = state.get("wins", 0), state.get("losses", 0)
            t = w + l
            wr = f" WR={w/t*100:.0f}%" if t else ""
            print(f"\n{'━'*60}")
            print(f"  STATUS [{now}]  Eq=${eq:,.2f}  Trades={t} (W={w} L={l}{wr})")
            pos = state.get("positions", {})
            if pos:
                for sym, ot in pos.items():
                    print(f"    {sym} {ot['side']} @ ${float(ot['entry_price']):,.2f}")
            print(f"{'━'*60}\n")
        except Exception:
            pass
        try:
            await asyncio.wait_for(shutdown_event.wait(), timeout=STATUS_SECS)
            break
        except asyncio.TimeoutError:
            pass


# ═══════════════════════════════════════════════════════════════════════════════
# Telegram command loop
# ═══════════════════════════════════════════════════════════════════════════════

async def _telegram_command_loop(state, shutdown_event):
    """Poll Telegram for commands and respond. Supported: /equity"""
    notifier = state.get("notifier")
    if not notifier:
        return
    offset = 0
    while not shutdown_event.is_set():
        try:
            messages, offset = await notifier.poll_commands(offset)
            for msg in messages:
                text = msg["text"].strip().lower()
                if text == "/equity":
                    eq  = state.get("equity", 0)
                    w   = state.get("wins", 0)
                    l   = state.get("losses", 0)
                    t   = w + l
                    wr  = f"{w/t*100:.0f}%" if t else "—"
                    pos = state.get("positions", {})
                    pos_lines = ""
                    for sym, ot in pos.items():
                        pos_lines += f"\n  📌 {sym} {ot['side']} @ ${float(ot['entry_price']):,.2f}"
                    await notifier.send(
                        f"💰 <b>EQUITY</b>\n"
                        f"\n"
                        f"💵 Balance:  <code>${eq:,.2f}</code>\n"
                        f"📊 Trades:   {t}  (W={w}  L={l}  WR={wr})\n"
                        f"{'📌 Positions:' + pos_lines if pos_lines else '📭 No open positions'}\n"
                        f"\n⏰ {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
                    )
        except Exception:
            pass
        try:
            await asyncio.wait_for(shutdown_event.wait(), timeout=5)
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
    symbols = [f"{b}-USDT" for b in args.symbols]
    print(f"\nChecking BingX API...")
    for sym in symbols:
        try:
            tk = await c.get_ticker(sym)
            print(f"  [OK] {sym:<12} ${float(tk['last']):>12,.4f}")
        except Exception as e:
            print(f"  [FAIL] {sym}: {e}")
    await c.close()


# ═══════════════════════════════════════════════════════════════════════════════
# Main
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
            except Exception as e:
                print(f"  [{sym}] leverage warning: {e}")
        mode = "LIVE BingX"
    else:
        adapter = PaperAdapter(client=client, initial_balance=Decimal(str(args.balance)))
        mode = "PAPER simulation"

    print(f"\n{'═'*60}")
    print(f"  TREND BOT V2 — Pine Script v5.1 Aligned")
    print(f"{'═'*60}")
    print(f"  Mode:      {mode}")
    print(f"  Symbols:   {', '.join(symbols)}")
    print(f"  Timeframe: {args.timeframe}")
    print(f"  Balance:   ${args.balance:,.2f}")
    print(f"  Reversal:  {'ON' if params.get('enable_reversal', True) else 'OFF'}")
    print(f"  Leverage:  A={args.lev_a}x B={args.lev_b}x C={args.lev_c}x")
    print(f"{'═'*60}")

    notifier = TelegramNotifier.from_env()
    if notifier:
        ok = await notifier.bot_started(symbols=symbols, mode=mode, balance=args.balance)
        print(f"  Telegram: {'✓' if ok else '✗'}")
    else:
        print(f"  Telegram: not configured")
    print(f"\n  Loading...\n")

    shutdown_event = asyncio.Event()
    worker_tasks: list[asyncio.Task] = []

    def _shutdown():
        print("\n  Shutting down...")
        shutdown_event.set()

    try:
        loop = asyncio.get_running_loop()
        loop.add_signal_handler(signal.SIGINT, _shutdown)
        loop.add_signal_handler(signal.SIGTERM, _shutdown)
    except NotImplementedError:
        signal.signal(signal.SIGINT, lambda s, f: _shutdown())

    state = {
        "equity": args.balance, "wins": 0, "losses": 0,
        "positions": {}, "notifier": notifier,
        "_balance_cache": None,
    }

    for sym in symbols:
        t = asyncio.create_task(run_symbol(
            sym, client, adapter, args, settings, tier_lev, params,
            shutdown_event, state, is_live=is_live,
        ))
        worker_tasks.append(t)

    status_task = asyncio.create_task(
        _status_loop(state, symbols, shutdown_event))
    worker_tasks.append(status_task)

    cmd_task = asyncio.create_task(
        _telegram_command_loop(state, shutdown_event))
    worker_tasks.append(cmd_task)

    results = await asyncio.gather(*worker_tasks, return_exceptions=True)
    for i, r in enumerate(results):
        if isinstance(r, Exception) and not isinstance(r, asyncio.CancelledError):
            print(f"  Task {i} error: {r}")

    await adapter.shutdown()

    w, l = state["wins"], state["losses"]
    t = w + l
    print(f"\n{'═'*60}")
    print(f"  SESSION: {t} trades  W={w} L={l}"
          f"  {'WR=' + str(round(w/t*100)) + '%' if t else ''}")
    print(f"{'═'*60}\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="TrendBot V2 — Pine Script aligned")
    p.add_argument("--symbols",       nargs="+", default=DEFAULT_BASES)
    p.add_argument("--timeframe",     default="15m")
    p.add_argument("--balance",       type=float, default=10000.0)
    p.add_argument("--max-positions", type=int,   default=3,    dest="max_positions")
    p.add_argument("--alloc-pct",     type=float, default=10.0, dest="alloc_pct")
    p.add_argument("--lev-a",  type=int, default=5, dest="lev_a")
    p.add_argument("--lev-b",  type=int, default=3, dest="lev_b")
    p.add_argument("--lev-c",  type=int, default=1, dest="lev_c")
    p.add_argument("--params-file", default=None, dest="params_file")
    p.add_argument("--check",      action="store_true")
    p.add_argument("--live-bingx", action="store_true", dest="live_bingx")
    args = p.parse_args()
    if args.check:
        asyncio.run(check_connection(args))
    else:
        asyncio.run(main(args))
