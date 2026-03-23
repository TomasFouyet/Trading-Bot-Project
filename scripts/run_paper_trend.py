"""
Paper trading for TrendFollowingStrategy using BingX virtual USDT.

Features:
  - Leverage per confidence tier (A > B > C)
  - Bar-by-bar SL / TP1 / TP2 monitoring (matches BacktestEngine logic)
  - Per-bar indicator dashboard (ADX, EMAs, MACD, pullback, session, streak)
  - 5-minute live status updates (price, equity, open position)
  - BingX API connection check (--check flag)
  - Trade history saved to data/paper_trades.csv
  - --live-bingx: send real orders to BingX (visible on exchange)

Usage:
    python scripts/run_paper_trend.py --check                # Test API
    python scripts/run_paper_trend.py                        # Local simulation
    python scripts/run_paper_trend.py --live-bingx           # Real BingX orders
    python scripts/run_paper_trend.py --quiet                # Less output
    python scripts/run_paper_trend.py --live-bingx --lev-a 5 --lev-b 3 --lev-c 1
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
}

WINDOW_SIZE       = 300
STATUS_INTERVAL_S = 300
TRADES_CSV        = Path("data/paper_trades.csv")
TRADES_HEADER     = [
    "trade_id", "symbol", "side", "strategy", "tier", "leverage",
    "entry_time", "entry_price", "exit_time", "exit_price", "exit_type",
    "qty", "pnl", "pnl_pct", "fees", "signal_reason",
]


# ── Trade history helpers ─────────────────────────────────────────────────────

def _ensure_trades_file():
    TRADES_CSV.parent.mkdir(parents=True, exist_ok=True)
    if not TRADES_CSV.exists():
        with open(TRADES_CSV, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=TRADES_HEADER).writeheader()

def _append_trade(row: dict):
    with open(TRADES_CSV, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=TRADES_HEADER).writerow(row)

def _record_trade_close(ot: dict, fill, exit_type: str) -> Decimal:
    is_long = ot["side"] == "LONG"
    gross = ((fill.price - ot["entry_price"]) if is_long else (ot["entry_price"] - fill.price)) * fill.qty
    pct = fill.qty / ot["original_qty"]
    fee_in = ot["fee_in"] * pct
    net = gross - fee_in - fill.fee
    total_pnl  = ot["total_partial_pnl"] + net
    total_fees = ot["total_partial_fees"] + fee_in + fill.fee
    notional = ot["entry_price"] * ot["original_qty"]
    _append_trade({
        "trade_id": ot["trade_id"], "symbol": ot["symbol"], "side": ot["side"],
        "strategy": ot["strategy"], "tier": ot["tier"], "leverage": ot["leverage"],
        "entry_time": ot["entry_time"], "entry_price": float(ot["entry_price"]),
        "exit_time": fill.timestamp.isoformat(), "exit_price": float(fill.price),
        "exit_type": exit_type, "qty": float(ot["original_qty"]),
        "pnl": float(total_pnl.quantize(Decimal("0.01"))),
        "pnl_pct": round(float(total_pnl / notional * 100) if notional else 0, 4),
        "fees": float(total_fees.quantize(Decimal("0.01"))),
        "signal_reason": ot["signal_reason"],
    })
    return total_pnl


# ── SL / TP monitor ──────────────────────────────────────────────────────────

async def _check_sl_tp(ot: dict, bar, adapter, strategy_id: str):
    from app.broker.base import OrderRequest, OrderSide, OrderType
    bh, bl, bc = float(bar.high), float(bar.low), float(bar.close)
    is_long = ot["side"] == "LONG"
    sl  = float(ot["sl_price"]) if ot.get("sl_price") else None
    tp1 = float(ot["tp1_price"]) if ot.get("tp1_price") else None
    tp2 = float(ot["tp2_price"]) if ot.get("tp2_price") else None
    tp1_hit = ot.get("tp1_hit", False)
    soft = ot.get("soft_sl_bars", 0)
    patience = soft > 0 and ot["bars_in_trade"] <= soft

    if is_long:
        sl_trig = bc if patience else bl
        sl_hit = sl is not None and sl_trig <= sl
        tp1_now = tp1 is not None and not tp1_hit and bh >= tp1
        tp2_now = tp2 is not None and tp1_hit and bh >= tp2
    else:
        sl_trig = bc if patience else bh
        sl_hit = sl is not None and sl_trig >= sl
        tp1_now = tp1 is not None and not tp1_hit and bl <= tp1
        tp2_now = tp2 is not None and tp1_hit and bl <= tp2

    exit_type, close_qty, close_pct = None, Decimal("0"), 1.0
    if tp1_now and not sl_hit:
        exit_type = "tp1"
        close_pct = float(ot.get("tp1_close_pct", 0.33))
        close_qty = min((ot["qty"] * Decimal(str(close_pct))).quantize(Decimal("0.001")), ot["qty"])
    elif tp2_now and not sl_hit:
        exit_type, close_qty = "tp2", ot["qty"]
    elif sl_hit:
        exit_type = "be_sl" if tp1_hit else "sl"
        close_qty = ot["qty"]

    if not exit_type or close_qty <= 0:
        return None, None, None

    side = OrderSide.SELL if is_long else OrderSide.BUY
    req = OrderRequest(symbol=ot["symbol"], side=side, order_type=OrderType.MARKET,
                       qty=close_qty, strategy_id=strategy_id)
    order, fill = await adapter.place_order(req)
    return exit_type, fill, close_pct


# ── Per-bar dashboard ─────────────────────────────────────────────────────────

def _print_bar_dashboard(bar, sig, meta, ot, equity, sw, sl):
    ts = bar.ts.strftime("%Y-%m-%d %H:%M")
    c, h, l = float(bar.close), float(bar.high), float(bar.low)
    adx = meta.get("adx", 0)
    ef, es = meta.get("ema_fast", 0), meta.get("ema_slow", 0)
    mh = meta.get("macd_hist", 0)
    above = c > es if es else None
    sess = meta.get("session", "?")
    ai = "▲" if adx >= 25 else "▽"
    mi = "+" if mh > 0 else "−"
    ti = "↑" if above else "↓" if above is not None else "?"

    print(f"  [{ts}]  C={c:>10,.2f}  H={h:,.2f}  L={l:,.2f}  "
          f"│ ADX={adx:>5.1f}{ai} │ EMAf={ef:>10,.2f}  EMAs={es:>10,.2f}  "
          f"│ MACD_H={mh:>+8.2f}{mi} │ {ti} {sess}")

    action = sig.action.value
    if ot:
        entry = float(ot["entry_price"])
        il = ot["side"] == "LONG"
        upnl = (c - entry) * float(ot["qty"]) if il else (entry - c) * float(ot["qty"])
        sf = float(ot["sl_price"]) if ot.get("sl_price") else 0
        atp = float(ot["tp2_price"] or 0) if ot.get("tp1_hit") else float(ot["tp1_price"] or 0)
        sd = abs(entry - sf) / entry * 100 if sf and entry else 0
        td = abs(atp - entry) / entry * 100 if atp and entry else 0
        phase = "BE" if ot.get("tp1_hit") else ot["side"]
        bars = ot.get("bars_in_trade", 0)
        tpl = "TP2" if ot.get("tp1_hit") else "TP1"
        print(f"           Sig={action:<14} │ POS={phase}  bars={bars:>3}  uPnL=${upnl:>+9.2f}  "
              f"│ SL={sf:>10,.2f}({sd:.1f}%)  {tpl}={atp:>10,.2f}({td:.1f}%)  │ Eq=${equity:>10,.2f}")
    else:
        reason = sig.reason[:50] if sig.reason else ""
        print(f"           Sig={action:<14} │ {reason:<50} │ Eq=${equity:>10,.2f}")

def _print_bar_compact(bar, sig, ot, equity):
    ts = bar.ts.strftime("%m-%d %H:%M")
    c = float(bar.close)
    pos = ""
    if ot:
        e = float(ot["entry_price"])
        il = ot["side"] == "LONG"
        u = (c - e) * float(ot["qty"]) if il else (e - c) * float(ot["qty"])
        p = "BE" if ot.get("tp1_hit") else ot["side"][:1]
        pos = f"  [{p} uPnL=${u:+.2f}]"
    print(f"  {ts} C={c:>10,.2f} {sig.action.value:<6}{pos}  Eq=${float(equity):,.2f}")


# ── Status loop ───────────────────────────────────────────────────────────────

async def _status_loop(client, symbol, state):
    await asyncio.sleep(STATUS_INTERVAL_S)
    while state.get("running", True):
        try:
            tk = await client.get_ticker(symbol)
            p = float(tk["last"])
            now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
            ot = state.get("open_trade")
            eq = state.get("equity", 0)
            w, l = state.get("total_wins", 0), state.get("total_losses", 0)
            t = w + l
            print(f"\n{'━'*70}")
            print(f"  STATUS  [{now}]  {symbol}=${p:,.2f}  Eq=${eq:,.2f}")
            print(f"     Trades: {t} (W={w} L={l} WR={w/t*100:.0f}%)" if t else "     Trades: 0")
            if ot:
                e = float(ot["entry_price"])
                il = ot["side"] == "LONG"
                u = (p - e) * float(ot["qty"]) if il else (e - p) * float(ot["qty"])
                ph = "BE" if ot.get("tp1_hit") else ot["side"]
                print(f"     OPEN {ph}  T={ot.get('tier','?')} {ot.get('leverage',1)}x  "
                      f"entry={e:,.2f}  bars={ot.get('bars_in_trade',0)}  uPnL=${u:+,.2f}")
            else:
                print(f"     No position — waiting")
            print(f"{'━'*70}\n")
        except Exception as e:
            print(f"  [status error: {e}]")
        await asyncio.sleep(STATUS_INTERVAL_S)


# ── API check ─────────────────────────────────────────────────────────────────

async def check_connection(args):
    from app.config import get_settings
    from app.core.logging import configure_logging
    s = get_settings(); configure_logging(log_level="ERROR", log_format="console")
    from app.broker.bingx_client import BingXClient
    c = BingXClient(api_key=s.bingx_api_key, api_secret=s.bingx_api_secret,
                    base_url=s.bingx_base_url, market_type=s.bingx_market_type)
    print(f"\nChecking BingX API...")
    errs = []
    try:
        tk = await c.get_ticker(args.symbol); print(f"  [OK] Ticker: ${float(tk['last']):,.2f}")
    except Exception as e: print(f"  [FAIL] Ticker: {e}"); errs.append("ticker")
    try:
        from app.data.ingestor import OHLCVIngestor; from app.data.parquet_store import ParquetStore
        bars = await OHLCVIngestor(c, ParquetStore()).poll_latest(args.symbol, args.timeframe, 3)
        print(f"  [OK] OHLCV: {len(bars)} bars")
    except Exception as e: print(f"  [FAIL] OHLCV: {e}"); errs.append("ohlcv")
    try:
        raw = await c.get_balance()
        u = next((a for a in raw if "USDT" in str(a.get("asset",""))), None)
        print(f"  [OK] Balance: {u.get('balance','?') if u else 'reachable'}")
    except Exception as e: print(f"  [FAIL] Balance: {e}"); errs.append("balance")
    await c.close()
    print(f"\n  {'All OK' if not errs else f'{len(errs)} error(s): {errs}'}\n")


# ── Main ──────────────────────────────────────────────────────────────────────

async def main(args):
    from app.config import get_settings
    from app.core.logging import configure_logging
    settings = get_settings(); configure_logging(log_level="WARNING", log_format="console")

    from app.broker.base import OrderRequest, OrderSide, OrderType
    from app.broker.bingx_adapter import BingXAdapter
    from app.broker.bingx_client import BingXClient
    from app.broker.paper_adapter import PaperAdapter
    from app.data.feed import LiveFeed
    from app.data.ingestor import OHLCVIngestor, TIMEFRAME_SECONDS
    from app.data.parquet_store import ParquetStore
    from app.risk.manager import RiskManager
    from app.strategy.base import BaseStrategy
    from app.strategy.signals import SignalAction
    from app.strategy.trend_following import TrendFollowingStrategy

    _ensure_trades_file()
    tier_lev = {"A": args.lev_a, "B": args.lev_b, "C": args.lev_c, "X": 1}
    params = dict(BEST_PARAMS)
    if args.params_file:
        with open(args.params_file) as f: d = json.load(f)
        params.update(d.get("params", d))

    client = BingXClient(api_key=settings.bingx_api_key, api_secret=settings.bingx_api_secret,
                         base_url=settings.bingx_base_url, market_type=settings.bingx_market_type)
    store = ParquetStore()
    strategy = TrendFollowingStrategy(symbol=args.symbol, params=params)
    risk = RiskManager()

    use_bingx = args.live_bingx
    if use_bingx:
        adapter = BingXAdapter(client=client)
        mx = max(args.lev_a, args.lev_b, args.lev_c)
        try: await client.set_leverage(args.symbol, mx); print(f"  BingX leverage={mx}x")
        except Exception as e: print(f"  Leverage warning: {e}")
        mode_label = "LIVE BingX — orders on exchange"
    else:
        adapter = PaperAdapter(client=client, initial_balance=Decimal(str(args.balance)))
        mode_label = "LOCAL simulation"

    print(f"\n{'═'*70}")
    print(f"  PAPER TRADING — TrendFollowing")
    print(f"{'═'*70}")
    print(f"  Mode:           {mode_label}")
    print(f"  Symbol:         {args.symbol}  TF: {args.timeframe}")
    print(f"  Balance:        ${args.balance:,.2f}")
    print(f"  Risk/trade:     {settings.risk_max_trade_risk_pct}%")
    print(f"  Leverage:       A={args.lev_a}x B={args.lev_b}x C={args.lev_c}x")
    print(f"  ADX:            {params.get('adx_min')}/{params.get('adx_strong')}")
    print(f"  Patience:       {params.get('soft_sl_bars')} bars")
    print(f"{'═'*70}\n  Press Ctrl+C to stop.\n")

    # Warmup
    print(f"  Loading warmup bars...")
    warmup = await OHLCVIngestor(client, store).poll_latest(args.symbol, args.timeframe, WINDOW_SIZE + 2)
    bw: deque = deque(warmup[:-1], maxlen=WINDOW_SIZE)
    print(f"  Loaded {len(bw)} bars (need {strategy.min_bars_required})")

    if len(bw) >= strategy.min_bars_required:
        st = strategy.on_bar(BaseStrategy.bars_to_df(list(bw)))
        m = st.meta or {}
        lb = bw[-1]
        print(f"\n  Market now: ${float(lb.close):,.2f}  ADX={m.get('adx',0):.1f}  "
              f"EMAf={m.get('ema_fast',0):,.2f}  EMAs={m.get('ema_slow',0):,.2f}")
        print(f"  Signal: {st.action.value} — {st.reason[:80]}")
    print(f"\n  Live loop started.\n")

    balances = await adapter.get_balance()
    equity = balances[0].total if balances else Decimal(str(args.balance))
    risk.initialize(equity)

    state = {"running": True, "open_trade": None, "equity": float(equity),
             "total_wins": 0, "total_losses": 0}
    open_trade = None

    # ── Shutdown event ────────────────────────────────────────────────────
    shutdown_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    def _shutdown():
        print("\n  Shutting down...")
        shutdown_event.set()
        state["running"] = False
    try:
        loop.add_signal_handler(signal.SIGINT, _shutdown)
        loop.add_signal_handler(signal.SIGTERM, _shutdown)
    except NotImplementedError:
        signal.signal(signal.SIGINT, lambda s, f: _shutdown())

    status_task = asyncio.create_task(_status_loop(client, args.symbol, state))
    tf_secs = TIMEFRAME_SECONDS.get(args.timeframe, 900)
    feed = LiveFeed(client=client, store=store, symbol=args.symbol,
                    timeframe=args.timeframe, max_data_delay_s=tf_secs * 3)

    try:
        feed_iter = feed.stream().__aiter__()
        while not shutdown_event.is_set():
            # Race: next bar vs shutdown — so Ctrl+C exits immediately
            get_bar = asyncio.ensure_future(feed_iter.__anext__())
            wait_sd = asyncio.ensure_future(shutdown_event.wait())
            done, pending = await asyncio.wait([get_bar, wait_sd], return_when=asyncio.FIRST_COMPLETED)
            for t in pending:
                t.cancel()
                try: await t
                except (asyncio.CancelledError, StopAsyncIteration): pass
            if wait_sd in done:
                break
            try: bar = get_bar.result()
            except StopAsyncIteration: break

            bw.append(bar)
            if len(bw) < strategy.min_bars_required:
                print(f"  Warming up ({len(bw)}/{strategy.min_bars_required})..."); continue

            # 1. SL/TP
            if open_trade:
                open_trade["bars_in_trade"] += 1
                et, fill, cp = await _check_sl_tp(open_trade, bar, adapter, strategy.strategy_id)

                if et == "tp1" and fill:
                    il = open_trade["side"] == "LONG"
                    gp = ((fill.price - open_trade["entry_price"]) if il else (open_trade["entry_price"] - fill.price)) * fill.qty
                    fp = fill.qty / open_trade["original_qty"]
                    fi = open_trade["fee_in"] * fp
                    np_ = gp - fi - fill.fee
                    open_trade["total_partial_pnl"] += np_
                    open_trade["total_partial_fees"] += fi + fill.fee
                    open_trade["qty"] -= fill.qty
                    open_trade["tp1_hit"] = True
                    open_trade["tp1_price"] = None
                    open_trade["sl_price"] = open_trade["entry_price"]
                    print(f"\n  TP1 {cp*100:.0f}%  @ {float(fill.price):,.2f}  pnl={float(np_):+.2f}  "
                          f"remaining={float(open_trade['qty']):.4f}  SL->BE\n")
                    state["open_trade"] = open_trade
                    if open_trade["qty"] < Decimal("0.001"):
                        tp = _record_trade_close(open_trade, fill, et); risk.record_fill(tp)
                        if tp > 0: state["total_wins"] += 1
                        else: state["total_losses"] += 1
                        open_trade = None; state["open_trade"] = None

                elif et in ("tp2", "sl", "be_sl") and fill:
                    tp = _record_trade_close(open_trade, fill, et); risk.record_fill(tp)
                    w = "WIN" if tp > 0 else "LOSS"
                    if tp > 0: state["total_wins"] += 1
                    else: state["total_losses"] += 1
                    print(f"\n  {et.upper():<5}  @ {float(fill.price):,.2f}  pnl={float(tp):+.2f} [{w}]\n")
                    if hasattr(strategy, "notify_trade_result"): strategy.notify_trade_result(won=tp > 0)
                    open_trade = None; state["open_trade"] = None

            # 2. Strategy
            df = BaseStrategy.bars_to_df(list(bw))
            sig = strategy.on_bar(df)
            meta = sig.meta or {}
            balances = await adapter.get_balance()
            equity = balances[0].total if balances else Decimal(str(args.balance))
            risk.update_equity(equity)
            state["equity"] = float(equity)
            tier = meta.get("confidence_tier", "X")
            lev = tier_lev.get(tier, 1)

            if args.quiet:
                _print_bar_compact(bar, sig, open_trade, float(equity))
            else:
                _print_bar_dashboard(bar, sig, meta, open_trade, float(equity),
                                     strategy._consecutive_wins, strategy._consecutive_losses)

            if not sig.is_actionable(): continue
            ok, reason = risk.validate_signal(sig, equity)
            if not ok: print(f"           >> Rejected: {reason}"); continue

            tk = await adapter.get_ticker(args.symbol)
            price = Decimal(str(tk["last"]))

            # BUY / SELL
            if sig.action in (SignalAction.BUY, SignalAction.SELL) and not open_trade:
                il = sig.action == SignalAction.BUY
                bq = risk.compute_order_qty(sig, equity, price)
                mq = (equity / price).quantize(Decimal("0.001"))
                qty = min(bq * lev, mq).quantize(Decimal("0.001"))
                side = OrderSide.BUY if il else OrderSide.SELL
                order, fill = await adapter.place_order(
                    OrderRequest(symbol=args.symbol, side=side, order_type=OrderType.MARKET,
                                 qty=qty, strategy_id=strategy.strategy_id))
                if fill:
                    open_trade = {
                        "trade_id": str(uuid.uuid4())[:8], "symbol": args.symbol,
                        "side": "LONG" if il else "SHORT", "strategy": strategy.strategy_id,
                        "tier": tier, "leverage": lev,
                        "entry_time": fill.timestamp.isoformat(), "entry_price": fill.price,
                        "qty": fill.qty, "original_qty": fill.qty, "fee_in": fill.fee,
                        "total_partial_pnl": Decimal("0"), "total_partial_fees": Decimal("0"),
                        "signal_reason": sig.reason,
                        "sl_price": sig.stop_loss, "tp1_price": sig.take_profit,
                        "tp2_price": Decimal(str(meta["tp2"])) if meta.get("tp2") else None,
                        "tp1_hit": False, "tp1_close_pct": meta.get("tp1_close_pct", 0.33),
                        "soft_sl_bars": meta.get("soft_sl_bars", 0), "bars_in_trade": 0,
                    }
                    state["open_trade"] = open_trade
                    sd = "LONG" if il else "SHORT"
                    sf = float(sig.stop_loss) if sig.stop_loss else 0
                    rd = abs(float(fill.price) - sf) / float(fill.price) * 100 if sf else 0
                    print(f"\n  OPENED {sd}  Tier={tier} {lev}x  qty={float(qty):.4f}  @ {float(fill.price):,.2f}")
                    print(f"     SL={sf:,.2f}({rd:.1f}%)  TP1={float(sig.take_profit or 0):,.2f}  "
                          f"TP2={float(meta.get('tp2',0)):,.2f}")
                    print(f"     {sig.reason}\n")

            elif sig.action in (SignalAction.SELL, SignalAction.CLOSE) and open_trade:
                il = open_trade["side"] == "LONG"
                side = OrderSide.SELL if il else OrderSide.BUY
                order, fill = await adapter.place_order(
                    OrderRequest(symbol=args.symbol, side=side, order_type=OrderType.MARKET,
                                 qty=open_trade["qty"], strategy_id=strategy.strategy_id))
                if fill:
                    tp = _record_trade_close(open_trade, fill, "signal_close"); risk.record_fill(tp)
                    w = "WIN" if tp > 0 else "LOSS"
                    if tp > 0: state["total_wins"] += 1
                    else: state["total_losses"] += 1
                    if hasattr(strategy, "notify_trade_result"): strategy.notify_trade_result(won=tp > 0)
                    print(f"\n  CLOSE  @ {float(fill.price):,.2f}  pnl={float(tp):+.2f} [{w}]\n")
                    open_trade = None; state["open_trade"] = None

    except asyncio.CancelledError: pass
    except Exception as e:
        import traceback; print(f"\n  Error: {e}"); traceback.print_exc()
    finally:
        feed.stop(); state["running"] = False

        # Close position on BingX before exit
        if use_bingx and open_trade and open_trade["qty"] > Decimal("0.001"):
            print(f"  Closing position on BingX...")
            try:
                il = open_trade["side"] == "LONG"
                side = OrderSide.SELL if il else OrderSide.BUY
                _, fill = await adapter.place_order(
                    OrderRequest(symbol=args.symbol, side=side, order_type=OrderType.MARKET,
                                 qty=open_trade["qty"], strategy_id=strategy.strategy_id))
                if fill:
                    tp = _record_trade_close(open_trade, fill, "shutdown")
                    print(f"  Closed @ {float(fill.price):,.2f}  pnl={float(tp):+.2f}")
            except Exception as e:
                print(f"  WARNING: could not close: {e}\n  CHECK BINGX MANUALLY!")

        status_task.cancel()
        try: await status_task
        except asyncio.CancelledError: pass
        await adapter.shutdown()

    # Summary
    if use_bingx:
        try: bl = await adapter.get_balance(); fb = float(bl[0].total) if bl else args.balance
        except: fb = args.balance
    else:
        fb = float(adapter.cash_balance)
    np_ = fb - args.balance
    w, l = state["total_wins"], state["total_losses"]; t = w + l
    print(f"\n{'═'*70}\n  SESSION SUMMARY\n{'═'*70}")
    print(f"  Trades: {t}  (W={w} L={l} WR={w/t*100:.0f}%)" if t else "  Trades: 0")
    print(f"  Balance: ${args.balance:,.2f} -> ${fb:,.2f}  PnL=${np_:+,.2f} ({np_/args.balance*100:+.2f}%)")
    print(f"  Log: {TRADES_CSV}\n{'═'*70}\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Paper trading — TrendFollowing")
    p.add_argument("--symbol", default="BTC-USDT")
    p.add_argument("--timeframe", default="15m")
    p.add_argument("--balance", type=float, default=10000.0)
    p.add_argument("--lev-a", type=int, default=5, dest="lev_a")
    p.add_argument("--lev-b", type=int, default=3, dest="lev_b")
    p.add_argument("--lev-c", type=int, default=1, dest="lev_c")
    p.add_argument("--params-file", default=None, dest="params_file")
    p.add_argument("--check", action="store_true")
    p.add_argument("--quiet", action="store_true", help="One-line per bar")
    p.add_argument("--live-bingx", action="store_true", dest="live_bingx",
                   help="Send real orders to BingX (visible on exchange)")
    args = p.parse_args()
    if args.check: asyncio.run(check_connection(args))
    else: asyncio.run(main(args))