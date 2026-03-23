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
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import signal
import uuid
from collections import deque
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

BEST_PARAMS: dict = {
    "adx_min": 20, "adx_strong": 35, "min_confidence": 0,
    "pullback_tolerance_atr": 1, "session_mult_eu": 0.75,
    "session_mult_other": 0.5, "soft_sl_bars": 48,
    "streak_euphoria_mult": 0.75, "use_confidence": True,
    "use_patience": True, "use_session_filter": True, "use_streak_adj": True,
    "allow_short": True,
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


def _ensure_csv():
    TRADES_CSV.parent.mkdir(parents=True, exist_ok=True)
    if not TRADES_CSV.exists():
        with open(TRADES_CSV, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=TRADES_HEADER).writeheader()

def _append_csv(row: dict):
    with open(TRADES_CSV, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=TRADES_HEADER).writerow(row)

def _record_close(ot: dict, fill, exit_type: str) -> Decimal:
    il  = ot["side"] == "LONG"
    gp  = ((fill.price - ot["entry_price"]) if il else (ot["entry_price"] - fill.price)) * fill.qty
    pct = fill.qty / ot["original_qty"]
    fi  = ot["fee_in"] * pct
    net = gp - fi - fill.fee
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


async def _check_sl_tp(ot, bar, adapter, sid):
    from app.broker.base import OrderRequest, OrderSide, OrderType
    bh, bl, bc = float(bar.high), float(bar.low), float(bar.close)
    il  = ot["side"] == "LONG"
    sl  = float(ot["sl_price"])  if ot.get("sl_price")  else None
    t1  = float(ot["tp1_price"]) if ot.get("tp1_price") else None
    t2  = float(ot["tp2_price"]) if ot.get("tp2_price") else None
    t1h = ot.get("tp1_hit", False)
    soft, bit = ot.get("soft_sl_bars", 0), ot["bars_in_trade"]
    pat = soft > 0 and bit <= soft

    if il:
        st = bc if pat else bl
        sh = sl is not None and st <= sl
        t1n = t1 is not None and not t1h and bh >= t1
        t2n = t2 is not None and t1h and bh >= t2
    else:
        st = bc if pat else bh
        sh = sl is not None and st >= sl
        t1n = t1 is not None and not t1h and bl <= t1
        t2n = t2 is not None and t1h and bl <= t2

    et, cq, cp = None, Decimal("0"), 1.0
    if t1n and not sh:
        et  = "tp1"
        cp  = float(ot.get("tp1_close_pct", 0.33))
        cq  = min((ot["qty"] * Decimal(str(cp))).quantize(Decimal("0.001")), ot["qty"])
    elif t2n and not sh:
        et, cq = "tp2", ot["qty"]
    elif sh:
        et, cq = ("be_sl" if t1h else "sl"), ot["qty"]

    if not et or cq <= 0:
        return None, None, None
    side = OrderSide.SELL if il else OrderSide.BUY
    _, fill = await adapter.place_order(
        OrderRequest(symbol=ot["symbol"], side=side, order_type=OrderType.MARKET,
                     qty=cq, strategy_id=sid))
    return et, fill, cp


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

        # Bearish trend — potential short territory
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

        # Bullish trend — potential long territory
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
        ph = "BE" if ot.get("tp1_hit") else ot["side"]
        pos = f"  [{ph} {ot['bars_in_trade']}bars ${u:+.0f}]"
    s = sym.replace("-USDT", "")
    print(f"  {ts} {s:<4} ${close:>10,.2f} {trend} {adx_badge:<9}  {prefix}: {msg}{pos}")


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
                        ph = "BE" if ot.get("tp1_hit") else ot["side"]
                        status = f"IN TRADE {ph} T={ot.get('tier','?')} {ot.get('leverage',1)}x uPnL=${u:+.0f}"
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

async def run_symbol(symbol, client, adapter, args, settings, tier_lev, params,
                     shutdown_event, state):
    from app.broker.base import OrderRequest, OrderSide, OrderType
    from app.data.feed import LiveFeed
    from app.data.ingestor import OHLCVIngestor, TIMEFRAME_SECONDS
    from app.data.parquet_store import ParquetStore
    from app.risk.manager import RiskManager
    from app.strategy.base import BaseStrategy
    from app.strategy.signals import SignalAction
    from app.strategy.trend_following import TrendFollowingStrategy

    from app.notify.telegram import TelegramNotifier
    s = symbol.replace("-USDT", "")
    store = ParquetStore(); strategy = TrendFollowingStrategy(symbol=symbol, params=params)
    risk = RiskManager()
    notifier: TelegramNotifier | None = state.get("notifier")

    print(f"  [{s}] Loading warmup...")
    try:
        warmup = await OHLCVIngestor(client, store).poll_latest(symbol, args.timeframe, WINDOW_SIZE + 2)
    except Exception as e: print(f"  [{s}] ERROR: {e}"); return
    bw: deque = deque(warmup[:-1], maxlen=WINDOW_SIZE)

    if len(bw) >= strategy.min_bars_required:
        sig0 = strategy.on_bar(BaseStrategy.bars_to_df(list(bw)))
        m0 = sig0.meta or {}; msg0, _ = _analyze_bar(bw[-1], sig0, m0, params)
        print(f"  [{s}] Ready  ${float(bw[-1].close):,.2f}  ADX={m0.get('adx',0):.1f}  → {msg0}")
    else:
        print(f"  [{s}] {len(bw)} bars (need {strategy.min_bars_required})")

    balances = await adapter.get_balance()
    equity = balances[0].total if balances else Decimal(str(args.balance))
    risk.initialize(equity)

    open_trade = None
    tf_secs = TIMEFRAME_SECONDS.get(args.timeframe, 900)
    feed = LiveFeed(client=client, store=store, symbol=symbol,
                    timeframe=args.timeframe, max_data_delay_s=tf_secs * 3)
    feed_iter = feed.stream().__aiter__()

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
            if len(bw) < strategy.min_bars_required: continue

            # ── 1. SL / TP check ─────────────────────────────────────
            if open_trade:
                open_trade["bars_in_trade"] += 1
                et, fill, cp = await _check_sl_tp(open_trade, bar, adapter, strategy.strategy_id)
                if et == "tp1" and fill:
                    il = open_trade["side"] == "LONG"
                    gp = ((fill.price - open_trade["entry_price"]) if il else (open_trade["entry_price"] - fill.price)) * fill.qty
                    fp = fill.qty / open_trade["original_qty"]; fi = open_trade["fee_in"] * fp
                    np_ = gp - fi - fill.fee
                    open_trade["total_partial_pnl"] += np_; open_trade["total_partial_fees"] += fi + fill.fee
                    open_trade["qty"] -= fill.qty; open_trade["tp1_hit"] = True
                    open_trade["tp1_price"] = None
                    # Move SL to breakeven+ (entry ± 0.2% to cover fees)
                    entry = open_trade["entry_price"]
                    fee_buffer = entry * Decimal("0.002")  # 20 bps covers round-trip fees
                    if open_trade["side"] == "LONG":
                        open_trade["sl_price"] = entry + fee_buffer
                    else:
                        open_trade["sl_price"] = entry - fee_buffer
                    open_trade["soft_sl_bars"] = 0  # disable patience — BE SL must use wicks
                    be_price = float(open_trade["sl_price"])
                    print(f"\n  [{s}] TP1 {cp*100:.0f}% @ ${float(fill.price):,.2f}  pnl=${float(np_):+.2f}  SL→BE+ ${be_price:,.4f}\n")
                    if notifier:
                        asyncio.create_task(notifier.tp1_hit(
                            symbol=symbol, exit_price=float(fill.price), close_pct=cp,
                            partial_pnl=float(np_), remaining_qty=float(open_trade["qty"]),
                            entry_price=float(open_trade["entry_price"]),
                        ))
                    state["positions"][symbol] = open_trade
                    if open_trade["qty"] < Decimal("0.001"):
                        tp = _record_close(open_trade, fill, et); risk.record_fill(tp)
                        state["wins" if tp > 0 else "losses"] += 1
                        open_trade = None; state["positions"].pop(symbol, None)
                elif et in ("tp2", "sl", "be_sl") and fill:
                    tp = _record_close(open_trade, fill, et); risk.record_fill(tp)
                    state["wins" if tp > 0 else "losses"] += 1
                    w, l = state["wins"], state["losses"]; tt = w + l
                    print(f"\n  [{s}] {et.upper():<5} @ ${float(fill.price):,.2f}  pnl=${float(tp):+.2f} [{'WIN' if tp>0 else 'LOSS'}]  total={tt} WR={w/tt*100:.0f}%\n")
                    if notifier:
                        asyncio.create_task(notifier.trade_closed(
                            symbol=symbol, side=open_trade["side"], exit_type=et,
                            exit_price=float(fill.price), entry_price=float(open_trade["entry_price"]),
                            total_pnl=float(tp), qty=float(open_trade["original_qty"]),
                            total_trades=tt, wins=w, losses=l,
                        ))
                    if hasattr(strategy, "notify_trade_result"): strategy.notify_trade_result(won=tp > 0)
                    open_trade = None; state["positions"].pop(symbol, None)

            # ── 2. Strategy signal ────────────────────────────────────
            df = BaseStrategy.bars_to_df(list(bw)); sig = strategy.on_bar(df); meta = sig.meta or {}
            balances = await adapter.get_balance()
            equity = balances[0].total if balances else Decimal(str(args.balance))
            risk.update_equity(equity); state["equity"] = float(equity)
            tier = meta.get("confidence_tier", "X"); lev = tier_lev.get(tier, 1)
            msg, wl = _analyze_bar(bar, sig, meta, params)
            ema_slow = meta.get("ema_slow", 0)
            state.setdefault("sym_states", {})[symbol] = {
                "close": float(bar.close), "adx": meta.get("adx", 0),
                "trend": ("↑" if float(bar.close) > ema_slow else "↓") if ema_slow else "~",
                "adx_min": float(params.get("adx_min", 20)), "msg": msg, "watchlist": wl,
            }
            _print_bar_line(symbol, bar, sig, meta, open_trade, float(equity), params)

            if not sig.is_actionable(): continue
            ok, reason = risk.validate_signal(sig, equity)
            if not ok: print(f"  [{s}] Rejected: {reason}"); continue
            if len(state["positions"]) >= args.max_positions and symbol not in state["positions"]:
                print(f"  [{s}] Max {args.max_positions} positions reached"); continue

            tk = await adapter.get_ticker(symbol); price = Decimal(str(tk["last"]))

            # ── 3. Direction-aware signal routing ─────────────────────
            #
            #   No position + BUY  → open LONG
            #   No position + SELL → open SHORT
            #   LONG  + SELL/CLOSE → close LONG
            #   SHORT + BUY/CLOSE  → close SHORT
            #   LONG  + BUY        → ignore (already long)
            #   SHORT + SELL       → ignore (already short)

            if not open_trade:
                # ── OPEN NEW POSITION ─────────────────────────────────
                if sig.action in (SignalAction.BUY, SignalAction.SELL):
                    il = sig.action == SignalAction.BUY
                    bq = risk.compute_order_qty(sig, equity, price)
                    max_notional = equity * Decimal(str(args.alloc_pct)) / Decimal("100")
                    mq = (max_notional / price).quantize(Decimal("0.001"))
                    qty = min(bq * lev, mq).quantize(Decimal("0.001"))
                    if qty < Decimal("0.001"): continue
                    side = OrderSide.BUY if il else OrderSide.SELL
                    _, fill = await adapter.place_order(OrderRequest(
                        symbol=symbol, side=side, order_type=OrderType.MARKET,
                        qty=qty, strategy_id=strategy.strategy_id))
                    if fill:
                        sd = "LONG" if il else "SHORT"
                        sf = float(sig.stop_loss) if sig.stop_loss else 0
                        tp1 = float(sig.take_profit) if sig.take_profit else 0
                        tp2 = float(meta.get("tp2", 0))
                        rr = abs(float(fill.price) - sf) / float(fill.price) * 100 if sf else 0
                        open_trade = {
                            "trade_id": str(uuid.uuid4())[:8], "symbol": symbol,
                            "side": sd, "strategy": strategy.strategy_id,
                            "tier": tier, "leverage": lev,
                            "entry_time": fill.timestamp.isoformat(),
                            "entry_price": fill.price,
                            "qty": fill.qty, "original_qty": fill.qty, "fee_in": fill.fee,
                            "total_partial_pnl": Decimal("0"),
                            "total_partial_fees": Decimal("0"),
                            "signal_reason": sig.reason,
                            "sl_price": sig.stop_loss, "tp1_price": sig.take_profit,
                            "tp2_price": Decimal(str(meta["tp2"])) if meta.get("tp2") else None,
                            "tp1_hit": False,
                            "tp1_close_pct": meta.get("tp1_close_pct", 0.33),
                            "soft_sl_bars": meta.get("soft_sl_bars", 0),
                            "bars_in_trade": 0,
                        }
                        state["positions"][symbol] = open_trade
                        notional = float(qty) * float(fill.price)
                        alloc_used = notional / float(equity) * 100
                        print(
                            f"\n  ┌{'─'*55}┐"
                            f"\n  │  TRADE OPENED [{s}] Tier {tier} {lev}x"
                            f"\n  │  {sd} @ ${float(fill.price):,.2f}  qty={float(qty):.4f}"
                            f"\n  │  Notional: ${notional:,.2f} ({alloc_used:.1f}% of equity)"
                            f"\n  │  SL=${sf:,.2f}({rr:.1f}%)  TP1=${tp1:,.2f}  TP2=${tp2:,.2f}"
                            f"\n  │  {sig.reason}"
                            f"\n  └{'─'*55}┘\n"
                        )
                        if notifier:
                            asyncio.create_task(notifier.trade_opened(
                                symbol=symbol, side=sd, tier=tier, leverage=lev,
                                entry_price=float(fill.price), qty=float(qty),
                                stop_loss=sf or None, tp1=tp1 or None, tp2=tp2 or None,
                                session=meta.get("session", ""),
                                reason=sig.reason, equity=float(equity),
                            ))

            else:
                # ── CLOSE EXISTING POSITION ───────────────────────────
                is_long  = open_trade["side"] == "LONG"

                should_close = (
                    sig.action == SignalAction.CLOSE
                    or (is_long  and sig.action == SignalAction.SELL)
                    or (not is_long and sig.action == SignalAction.BUY)
                )

                if should_close:
                    close_side = OrderSide.SELL if is_long else OrderSide.BUY
                    _, fill = await adapter.place_order(OrderRequest(
                        symbol=symbol, side=close_side,
                        order_type=OrderType.MARKET,
                        qty=open_trade["qty"],
                        strategy_id=strategy.strategy_id))
                    if fill:
                        if sig.action == SignalAction.CLOSE:
                            exit_type = "signal_close"
                        else:
                            exit_type = "reversal_close"

                        tp = _record_close(open_trade, fill, exit_type)
                        risk.record_fill(tp)
                        state["wins" if tp > 0 else "losses"] += 1
                        if hasattr(strategy, "notify_trade_result"):
                            strategy.notify_trade_result(won=tp > 0)
                        label = "WIN" if tp > 0 else "LOSS"
                        print(
                            f"\n  ┌{'─'*55}┐"
                            f"\n  │  TRADE CLOSED [{s}] [{label}]  ({exit_type})"
                            f"\n  │  {open_trade['side']} @ ${float(fill.price):,.2f}"
                            f"  pnl=${float(tp):+.2f}"
                            f"\n  └{'─'*55}┘\n"
                        )
                        if notifier:
                            ww, ll = state["wins"], state["losses"]
                            asyncio.create_task(notifier.trade_closed(
                                symbol=symbol, side=open_trade["side"],
                                exit_type=exit_type,
                                exit_price=float(fill.price),
                                entry_price=float(open_trade["entry_price"]),
                                total_pnl=float(tp),
                                qty=float(open_trade["original_qty"]),
                                total_trades=ww + ll, wins=ww, losses=ll,
                            ))
                        open_trade = None
                        state["positions"].pop(symbol, None)

    except asyncio.CancelledError:
        pass
    except Exception as e:
        import traceback; print(f"  [{s}] Error: {e}"); traceback.print_exc()
    finally:
        feed.stop()
        if open_trade and open_trade["qty"] > Decimal("0.001"):
            print(f"  [{s}] Closing {open_trade['side']} on shutdown...")
            try:
                il = open_trade["side"] == "LONG"
                close_side = OrderSide.SELL if il else OrderSide.BUY
                _, fill = await adapter.place_order(OrderRequest(
                    symbol=symbol, side=close_side,
                    order_type=OrderType.MARKET,
                    qty=open_trade["qty"],
                    strategy_id=strategy.strategy_id))
                if fill:
                    tp = _record_close(open_trade, fill, "shutdown")
                    print(f"  [{s}] Closed @ ${float(fill.price):,.2f}  pnl=${float(tp):+.2f}")
            except Exception as e: print(f"  [{s}] WARNING close failed: {e}")


async def check_connection(args):
    from app.config import get_settings; from app.core.logging import configure_logging
    s = get_settings(); configure_logging(log_level="ERROR", log_format="console")
    from app.broker.bingx_client import BingXClient
    c = BingXClient(api_key=s.bingx_api_key, api_secret=s.bingx_api_secret,
                    base_url=s.bingx_base_url, market_type=s.bingx_market_type)
    symbols = [f"{b}-USDT" for b in args.symbols]
    print(f"\nChecking BingX API for {len(symbols)} symbols...")
    for sym in symbols:
        try: tk = await c.get_ticker(sym); print(f"  [OK] {sym:<12} ${float(tk['last']):>12,.4f}")
        except Exception as e: print(f"  [FAIL] {sym}: {e}")
    try:
        raw = await c.get_balance()
        u = next((a for a in raw if "USDT" in str(a.get("asset", ""))), None)
        print(f"\n  Balance: {u.get('balance','?') if u else '?'} USDT")
    except Exception as e: print(f"\n  Balance: {e}")
    await c.close(); print()


async def main(args):
    from app.config import get_settings; from app.core.logging import configure_logging
    settings = get_settings(); configure_logging(log_level="WARNING", log_format="console")
    from app.broker.bingx_adapter import BingXAdapter
    from app.broker.bingx_client import BingXClient
    from app.broker.paper_adapter import PaperAdapter
    from app.notify.telegram import TelegramNotifier

    _ensure_csv()
    symbols = [f"{b}-USDT" for b in args.symbols]
    tier_lev = {"A": args.lev_a, "B": args.lev_b, "C": args.lev_c, "X": 1}
    params = dict(BEST_PARAMS)
    if args.params_file:
        with open(args.params_file) as f: d = json.load(f)
        params.update(d.get("params", d))

    client = BingXClient(api_key=settings.bingx_api_key, api_secret=settings.bingx_api_secret,
                         base_url=settings.bingx_base_url, market_type=settings.bingx_market_type)

    if args.live_bingx:
        adapter = BingXAdapter(client=client)
        mx = max(args.lev_a, args.lev_b, args.lev_c)
        for sym in symbols:
            try: await client.set_leverage(sym, mx)
            except Exception as e: print(f"  [{sym}] leverage: {e}")
        mode = "LIVE BingX — REAL USDT!"
    else:
        adapter = PaperAdapter(client=client, initial_balance=Decimal(str(args.balance)))
        mode = "LOCAL simulation (real prices, virtual balance)"

    short_status = "enabled" if params.get("allow_short", True) else "disabled"
    print(f"\n{'═'*72}")
    print(f"  MULTI-SYMBOL PAPER TRADING — TrendFollowing (LONG + SHORT)")
    print(f"{'═'*72}")
    print(f"  Mode:           {mode}")
    print(f"  Symbols:        {', '.join(symbols)}")
    print(f"  Timeframe:      {args.timeframe}")
    print(f"  Balance:        ${args.balance:,.2f}")
    print(f"  Max positions:  {args.max_positions}")
    print(f"  Alloc/trade:    {args.alloc_pct}% of equity (${args.balance * args.alloc_pct / 100:,.0f} notional cap)")
    print(f"  Leverage:       A={args.lev_a}x B={args.lev_b}x C={args.lev_c}x")
    print(f"  Directions:     LONG + SHORT ({short_status})")
    print(f"  ADX:            min={params['adx_min']} strong={params['adx_strong']}")
    print(f"  Patience:       {params['soft_sl_bars']} bars ({params['soft_sl_bars']*15//60}h)")
    print(f"  Trade log:      {TRADES_CSV}")
    if not args.live_bingx:
        print(f"\n  Safe mode: no real orders. Use --live-bingx for real trading.")
    print(f"{'═'*72}")

    notifier = TelegramNotifier.from_env()
    if notifier:
        print(f"  Telegram: notifications enabled")
        await notifier.bot_started(symbols=symbols, mode=mode, balance=args.balance)
    else:
        print(f"  Telegram: not configured (add TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID to .env)")
    print(f"\n  Ctrl+C to stop. Loading warmup...\n")

    shutdown_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    worker_tasks: list[asyncio.Task] = []

    def _shutdown():
        print("\n  Shutting down all symbols...")
        shutdown_event.set()
        for t in worker_tasks:
            if not t.done():
                t.cancel()

    try:
        loop.add_signal_handler(signal.SIGINT, _shutdown)
        loop.add_signal_handler(signal.SIGTERM, _shutdown)
    except NotImplementedError:
        signal.signal(signal.SIGINT, lambda s, f: _shutdown())

    state = {"equity": args.balance, "wins": 0, "losses": 0, "positions": {},
             "sym_states": {}, "notifier": notifier}

    for sym in symbols:
        t = asyncio.create_task(run_symbol(
            sym, client, adapter, args, settings, tier_lev, params,
            shutdown_event, state))
        worker_tasks.append(t)

    status_task = asyncio.create_task(_status_loop(client, state, symbols, shutdown_event))
    worker_tasks.append(status_task)

    results = await asyncio.gather(*worker_tasks, return_exceptions=True)
    for i, r in enumerate(results):
        if isinstance(r, Exception) and not isinstance(r, asyncio.CancelledError):
            print(f"  Task {i} error: {r}")

    await adapter.shutdown()

    w, l = state["wins"], state["losses"]; t = w + l
    final_eq = state.get("equity", args.balance)
    net_pnl  = final_eq - args.balance
    if notifier:
        await notifier.bot_stopped(total_trades=t, net_pnl=net_pnl, wins=w, losses=l)
    print(f"\n{'═'*72}\n  SESSION SUMMARY\n{'═'*72}")
    print(f"  Symbols: {', '.join(symbols)}")
    if t: print(f"  Trades:  {t}  (W={w} L={l} WR={w/t*100:.0f}%)")
    else: print(f"  Trades:  0")
    print(f"  Log:     {TRADES_CSV}\n{'═'*72}\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Multi-symbol paper trading (LONG+SHORT)")
    p.add_argument("--symbols", nargs="+", default=DEFAULT_BASES)
    p.add_argument("--timeframe", default="15m")
    p.add_argument("--balance", type=float, default=10000.0)
    p.add_argument("--max-positions", type=int, default=3, dest="max_positions")
    p.add_argument("--alloc-pct", type=float, default=10.0, dest="alloc_pct",
                   help="Max %% of equity per trade BEFORE leverage (default: 10%%)")
    p.add_argument("--lev-a", type=int, default=5, dest="lev_a")
    p.add_argument("--lev-b", type=int, default=3, dest="lev_b")
    p.add_argument("--lev-c", type=int, default=1, dest="lev_c")
    p.add_argument("--params-file", default=None, dest="params_file")
    p.add_argument("--check", action="store_true")
    p.add_argument("--live-bingx", action="store_true", dest="live_bingx",
                   help="Send REAL orders to BingX (real USDT!)")
    args = p.parse_args()
    if args.check: asyncio.run(check_connection(args))
    else: asyncio.run(main(args))