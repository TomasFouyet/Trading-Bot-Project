"""
run_trendbot_btc.py — TrendBot MTF v5.2 en BTC-USDT

Replica exacta del Pine Script TrendBot MTF v5.2, conectado a BingX.
Soporta paper trading (simulación) y live trading (órdenes reales).

Modos:
  DEFAULT:       PaperAdapter — simula con precios reales de BingX.
                 No se envían órdenes. Dinero virtual. 100% seguro.
  --live-bingx:  BingXAdapter — órdenes REALES en BingX con USDT real.

Uso:
    python scripts/run_trendbot_btc.py                    # Paper seguro
    python scripts/run_trendbot_btc.py --check            # Verifica API
    python scripts/run_trendbot_btc.py --live-bingx       # ¡REAL!
    python scripts/run_trendbot_btc.py --balance 5000     # Capital virtual
    python scripts/run_trendbot_btc.py --leverage 3       # Apalancamiento

Sizing proporcional (replica Pine v5.2):
    qty = equity × (tier_pct / 100) × session_mult / close
    Con $10K y Tier A (100%) → $10,000 en BTC
    Con $50K y Tier A (100%) → $50,000 en BTC
    El % de ganancia/pérdida es IDÉNTICO en ambos casos.

Parámetros de la estrategia:
    --adx-min       ADX mínimo (default: 20)
    --pb-tol        Pullback tolerancia ATR (default: 1.2)
    --sl-min-atr    SL mínimo en ATR (default: 1.2)
    --cooldown      Cooldown entre señales en barras (default: 3)
    --no-rsi        Desactiva filtro RSI
    --no-vol        Desactiva filtro de volumen
    --no-short      Solo LONGs
    --no-reversal   Desactiva reversal swap
    --trail-mult    Trailing ATR multiplier (default: 1.5)
    --min-conf      Confianza mínima (default: 0.0)
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
    advance_bar,
    apply_tp1_updates,
    check_exit,
    compute_leg_pnl,
    create_trade,
    update_atr,
)

SYMBOL        = "BTC-USDT"
TIMEFRAME     = "15m"
HTF_TIMEFRAME = "4h"
WINDOW_SIZE   = 350   # barras de warmup (>= ema_slow=50 + htf_ema_slow=200 + buffer)

TRADES_CSV    = Path("data/trendbot_btc_trades.csv")
TRADES_HEADER = [
    "trade_id", "symbol", "side", "strategy", "tier", "leverage",
    "entry_time", "entry_price", "exit_time", "exit_price", "exit_type",
    "qty", "pnl", "pnl_pct", "fees", "signal_reason",
]

# ── Parámetros v5.2 por defecto ───────────────────────────────────────────────
DEFAULT_PARAMS: dict = {
    # Core
    "adx_min": 20, "adx_strong": 35,
    "ema_fast": 20, "ema_slow": 50,
    "pb_tol_atr": 1.2,
    "min_confidence": 0.0,
    "allow_short": True,
    "sig_cooldown": 3,
    # Filtros v5.2
    "require_bull_bar": True,
    "use_rsi_filter": True, "rsi_long_min": 45, "rsi_short_max": 55,
    "use_vol_filter": True, "vol_min_ratio": 0.6,
    # SL / TP
    "sl_swing_lookback": 50, "sl_swing_window": 3,
    "sl_min_atr": 1.2, "sl_max_atr": 2.5, "sl_buf_atr": 0.3,
    "tp1_r_A": 1.5, "tp2_r_A": 3.0,
    "tp1_r_B": 1.5, "tp2_r_B": 2.5,
    "tp1_r_C": 1.0, "tp2_r_C": 1.5,
    # Trailing
    "use_trailing": True, "trail_atr_mult": 1.5,
    # Reversal
    "enable_reversal": True,
    # Sizing por tier (% equity — Pine v5.2)
    "tier_a_pct": 100, "tier_b_pct": 75, "tier_c_pct": 25,
    # HTF
    "htf_ema_fast": 50, "htf_ema_slow": 200,
    # Session
    "use_session_filter": True,
    "session_mult_us": 1.0, "session_mult_eu": 0.75, "session_mult_other": 0.5,
}


# ═══════════════════════════════════════════════════════════════════════════════
# CSV helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _ensure_csv() -> None:
    TRADES_CSV.parent.mkdir(parents=True, exist_ok=True)
    if not TRADES_CSV.exists():
        with open(TRADES_CSV, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=TRADES_HEADER).writeheader()

def _append_csv(row: dict) -> None:
    with open(TRADES_CSV, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=TRADES_HEADER).writerow(row)


# ═══════════════════════════════════════════════════════════════════════════════
# Synthetic fill (paper mode SL/TP)
# ═══════════════════════════════════════════════════════════════════════════════

def _make_paper_fill(ot: dict, trigger_price: float, qty: Decimal, side):
    from app.broker.base import FillResult, OrderSide
    import uuid
    from datetime import datetime, timezone
    fee = qty * Decimal(str(trigger_price)) * COMMISSION_RATE
    return FillResult(
        fill_id=str(uuid.uuid4()),
        order_id=str(uuid.uuid4()),
        symbol=ot["symbol"],
        side=side,
        price=Decimal(str(trigger_price)),
        qty=qty,
        fee=fee,
        fee_currency="USDT",
        timestamp=datetime.now(timezone.utc),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# PnL helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _record_close(ot: dict, fill, exit_type: str) -> Decimal:
    """Log trade to CSV, return final PnL (including partial TP1 legs)."""
    exit_price = fill.price
    qty        = fill.qty
    side       = ot["side"]
    entry_p    = ot["entry_price"]

    # PnL for remaining leg
    final_leg_pnl = compute_leg_pnl(ot, exit_price, qty, fill.fee)
    total_pnl     = ot.get("total_partial_pnl", Decimal("0")) + final_leg_pnl
    total_fees    = (ot.get("total_partial_fees", Decimal("0"))
                     + ot["fee_in"] * (qty / ot["original_qty"])
                     + fill.fee)

    # PnL%  relative to initial value (entry × original_qty)
    entry_value = entry_p * ot["original_qty"]
    pnl_pct     = float(total_pnl / entry_value * 100) if entry_value > 0 else 0.0

    tier = ot.get("tier", "?")
    lev  = ot.get("leverage", 1)
    row  = {
        "trade_id":    ot["trade_id"],
        "symbol":      ot["symbol"],
        "side":        side,
        "strategy":    "trendbot_mtf_v52",
        "tier":        tier,
        "leverage":    lev,
        "entry_time":  ot.get("entry_time", ""),
        "entry_price": float(entry_p),
        "exit_time":   datetime.now(timezone.utc).isoformat(),
        "exit_price":  float(exit_price),
        "exit_type":   exit_type,
        "qty":         float(ot["original_qty"]),
        "pnl":         round(float(total_pnl), 4),
        "pnl_pct":     round(pnl_pct, 4),
        "fees":        round(float(total_fees), 4),
        "signal_reason": ot.get("signal_reason", ""),
    }
    _append_csv(row)
    return total_pnl


# ═══════════════════════════════════════════════════════════════════════════════
# SL/TP check per bar
# ═══════════════════════════════════════════════════════════════════════════════

async def _check_sl_tp(ot: dict, bar, adapter, sid: str, is_live: bool, notifier=None):
    """
    Returns (exit_type, fill, close_pct) or (None, None, None).
    Mirrors run_multi_paper._check_sl_tp().
    """
    from app.broker.base import OrderRequest, OrderSide, OrderType

    bh, bl, bc = float(bar.high), float(bar.low), float(bar.close)
    il         = ot["side"] == "LONG"

    advance_bar(ot, bh, bl)
    action = check_exit(ot, bh, bl, bc)

    if action is None:
        return None, None, None

    et  = action.exit_type
    cq  = action.close_qty
    cp  = action.close_pct
    tpx = action.trigger_price

    if action.tp1_updates:
        apply_tp1_updates(ot, action.tp1_updates)

    close_side     = OrderSide.SELL if il else OrderSide.BUY
    close_pos_side = "LONG"         if il else "SHORT"

    if not is_live:
        fill = _make_paper_fill(ot, tpx, cq, close_side)
    else:
        try:
            _, fill = await adapter.place_order(OrderRequest(
                symbol=ot["symbol"], side=close_side,
                order_type=OrderType.MARKET, qty=cq, strategy_id=sid,
                extra={"positionSide": close_pos_side},
            ))
        except Exception as e:
            print(f"  ⚠️  [BTC] {et.upper()} CLOSE ORDER FAILED: {e}")
            if notifier:
                asyncio.create_task(notifier.send(
                    f"⚠️ <b>CLOSE ORDER FAILED — BTC-USDT</b>\n"
                    f"🚫 Exit: <b>{et.upper()}</b>\n"
                    f"🚫 Error: <code>{str(e)[:200]}</code>\n"
                    f"📌 Retry next bar\n"
                ))
            return et + "_order_failed", None, cp

    return et, fill, cp


# ═══════════════════════════════════════════════════════════════════════════════
# HTF refresh helper
# ═══════════════════════════════════════════════════════════════════════════════

async def _refresh_htf(client, store, symbol: str, htf_df_holder: list) -> None:
    """Fetch/update 4H bars for HTF bias. Stores in htf_df_holder[0]."""
    from app.strategy.mtf_context import HTFContext
    # Use existing HTFContext if available for caching
    try:
        from app.data.ingestor import OHLCVIngestor, TIMEFRAME_SECONDS
        ingestor = OHLCVIngestor(client=client, store=store)
        bars = await ingestor.poll_latest(symbol, "4h", lookback_bars=250)
        if bars:
            df = __import__("app.strategy.base", fromlist=["BaseStrategy"]).BaseStrategy.bars_to_df(bars)
            htf_df_holder[0] = df
    except Exception as e:
        print(f"  [HTF] refresh failed: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# Status printer
# ═══════════════════════════════════════════════════════════════════════════════

def _print_status(bar, sig, ot: dict | None, wins: int, losses: int, equity: float) -> None:
    from app.strategy.signals import SignalAction
    ts    = datetime.now(timezone.utc).strftime("%H:%M:%S")
    close = float(bar.close)
    meta  = sig.meta or {}
    adx   = meta.get("adx", 0)
    rsi   = meta.get("rsi", 0)
    vol   = meta.get("vol_ratio", 0)
    act   = sig.action.value if sig else "HOLD"

    pos_str = ""
    if ot:
        sl  = float(ot.get("sl_price", 0) or 0)
        tp1 = float(ot.get("tp1_price", 0) or 0)
        tp1h = ot.get("tp1_hit", False)
        bars_in = ot.get("bars_in_trade", 0)
        pnl_unrealized = ((close - float(ot["entry_price"])) * float(ot["qty"])
                          * (1 if ot["side"] == "LONG" else -1))
        pos_str = (f"  [{ot['side']} T{ot.get('tier','?')} "
                   f"bar#{bars_in} pnl=${pnl_unrealized:+.2f} "
                   f"sl={sl:,.0f}{'(trail)' if tp1h else ''}]")

    wr = wins / (wins + losses) * 100 if (wins + losses) > 0 else 0
    eq_str = f"${equity:,.2f}" if equity else ""
    print(
        f"  {ts} BTC ${close:>10,.2f}"
        f"  ADX={adx:.1f} RSI={rsi:.1f} VOL={vol:.2f}"
        f"  {act if act != 'HOLD' else '─':>6}"
        f"  W={wins} L={losses} WR={wr:.0f}%"
        f"  {eq_str}{pos_str}"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Main trading loop (single symbol: BTC-USDT)
# ═══════════════════════════════════════════════════════════════════════════════

async def _run_btc(args):
    from app.config import get_settings
    from app.core.logging import configure_logging
    settings = get_settings()
    configure_logging(log_level="WARNING", log_format="console")

    from app.broker.bingx_adapter import BingXAdapter
    from app.broker.bingx_client import BingXClient
    from app.broker.base import OrderRequest, OrderSide, OrderType
    from app.broker.paper_adapter import PaperAdapter
    from app.core.exceptions import DataDelayError
    from app.data.feed import LiveFeed
    from app.data.ingestor import OHLCVIngestor
    from app.data.parquet_store import ParquetStore
    from app.notify.telegram import TelegramNotifier
    from app.strategy.signals import SignalAction
    from app.strategy.trendbot_mtf_v52 import TrendBotMTFv52Strategy

    _ensure_csv()

    # ── Build params ──────────────────────────────────────────────────────────
    params = dict(DEFAULT_PARAMS)
    params.update({
        "adx_min":          args.adx_min,
        "pb_tol_atr":       args.pb_tol,
        "sl_min_atr":       args.sl_min_atr,
        "sig_cooldown":     args.cooldown,
        "min_confidence":   args.min_conf,
        "use_rsi_filter":   not args.no_rsi,
        "use_vol_filter":   not args.no_vol,
        "allow_short":      not args.no_short,
        "enable_reversal":  not args.no_reversal,
        "trail_atr_mult":   args.trail_mult,
    })
    if args.params_file:
        with open(args.params_file) as f:
            d = json.load(f)
        params.update(d.get("params", d))

    # ── BingX client ──────────────────────────────────────────────────────────
    client = BingXClient(
        api_key=settings.bingx_api_key,
        api_secret=settings.bingx_api_secret,
        base_url=settings.bingx_base_url,
        market_type=settings.bingx_market_type,
    )

    # ── Adapter ───────────────────────────────────────────────────────────────
    is_live = args.live_bingx
    if is_live:
        adapter = BingXAdapter(client=client)
        try:
            await client.set_leverage(SYMBOL, args.leverage)
            print(f"  [BTC] Leverage set to {args.leverage}x")
        except Exception as e:
            print(f"  [BTC] Leverage warning: {e}")
        mode = f"LIVE BingX — REAL USDT!  leverage={args.leverage}x"
    else:
        adapter = PaperAdapter(client=client, initial_balance=Decimal(str(args.balance)))
        mode    = f"PAPER simulation (real prices, virtual ${args.balance:,.2f})"

    await adapter.initialize()

    # ── Telegram notifier (optional — only if .env has token+chat_id) ─────────
    notifier = TelegramNotifier.from_env()
    if notifier:
        print(f"  Telegram: enabled")
    else:
        print(f"  Telegram: disabled (set TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID in .env)")

    # ── Strategy ──────────────────────────────────────────────────────────────
    strategy = TrendBotMTFv52Strategy(symbol=SYMBOL, params=params)
    store    = ParquetStore()

    # ── Print banner ──────────────────────────────────────────────────────────
    short_s = "enabled" if params["allow_short"]   else "disabled"
    rsi_s   = "enabled" if params["use_rsi_filter"] else "disabled"
    vol_s   = "enabled" if params["use_vol_filter"] else "disabled"
    rev_s   = "enabled" if params["enable_reversal"] else "disabled"
    print(f"\n{'═'*72}")
    print(f"  TRENDBOT MTF v5.2 — BTC-USDT @ {TIMEFRAME}")
    print(f"{'═'*72}")
    print(f"  Mode:       {mode}")
    print(f"  ADX min:    {params['adx_min']}    Pullback: {params['pb_tol_atr']} ATR")
    print(f"  SL min:     {params['sl_min_atr']} ATR    Cooldown: {params['sig_cooldown']} bars")
    print(f"  RSI filter: {rsi_s} (long≥{params['rsi_long_min']} short≤{params['rsi_short_max']})")
    print(f"  Vol filter: {vol_s} (min {params['vol_min_ratio']}× SMA20)")
    print(f"  Trailing:   {params['trail_atr_mult']} ATR    Reversal swap: {rev_s}")
    print(f"  Shorts:     {short_s}")
    print(f"  Sizing:     Tier A={params['tier_a_pct']}%  B={params['tier_b_pct']}%  C={params['tier_c_pct']}% of equity")
    print(f"  Trade log:  {TRADES_CSV}")
    if not is_live:
        print(f"\n  Safe mode: no real orders. Use --live-bingx for real trading.")
    print(f"{'═'*72}\n")
    print("  Timestamp        Price         ADX   RSI   VOL    Signal    Wins/Losses")
    print("  " + "─" * 68)

    # ── Warm up 15M data ──────────────────────────────────────────────────────
    ingestor = OHLCVIngestor(client=client, store=store)
    print(f"\n  Fetching {WINDOW_SIZE} bars of {TIMEFRAME} data...")
    bars_warmup = await ingestor.poll_latest(SYMBOL, TIMEFRAME, lookback_bars=WINDOW_SIZE)
    print(f"  Got {len(bars_warmup)} bars. Fetching 4H HTF data...")

    # ── Warm up 4H data ───────────────────────────────────────────────────────
    htf_df_holder = [None]
    htf_bars = await ingestor.poll_latest(SYMBOL, HTF_TIMEFRAME, lookback_bars=250)
    if htf_bars:
        from app.strategy.base import BaseStrategy as _BS
        htf_df_holder[0] = _BS.bars_to_df(htf_bars)
        print(f"  Got {len(htf_bars)} 4H bars for HTF bias.\n")
    else:
        print("  Warning: no 4H data, running without HTF bias.\n")

    # Inject HTF into strategy
    if htf_df_holder[0] is not None:
        strategy.set_htf_bars(htf_df_holder[0])

    # ── State ─────────────────────────────────────────────────────────────────
    open_trade: dict | None = None
    equity     = Decimal(str(args.balance))
    wins       = 0
    losses     = 0
    bw: deque  = deque(bars_warmup, maxlen=WINDOW_SIZE)
    last_processed_ts = bw[-1].ts if bw else None
    # Force flat state — runner has no open position at startup
    strategy.notify_trade_closed("LONG")  # resets _trade_state to 0

    # ── Detect existing open position on BingX at startup ─────────────────────
    if is_live:
        try:
            positions = await client.get_positions(SYMBOL)
            for pos in positions:
                qty_raw = Decimal(str(pos.get("positionAmt", pos.get("qty", 0))))
                if abs(qty_raw) >= Decimal("0.0001"):
                    side = "LONG" if qty_raw > 0 else "SHORT"
                    entry_px = Decimal(str(pos.get("avgPrice", pos.get("entry_price", 0))))
                    open_trade = create_trade(
                        trade_id      = str(uuid.uuid4()),
                        symbol        = SYMBOL,
                        side          = side,
                        entry_price   = entry_px,
                        qty           = abs(qty_raw),
                        fee_in        = Decimal("0"),
                        sl_price      = None,
                        tp1_price     = None,
                        tp2_price     = None,
                        tp1_close_pct = 0.33,
                        soft_sl_bars  = 0,
                        trailing_atr  = 1.5,
                        current_atr   = 0.0,
                        signal_reason = "recovered_on_startup",
                        extra         = {"tier": "?", "leverage": args.leverage,
                                         "entry_time": datetime.now(timezone.utc).isoformat()},
                    )
                    print(f"\n  [BTC] ⚠️  Posición abierta detectada al iniciar:")
                    print(f"  [BTC]    {side} {abs(qty_raw):.4f} BTC @ ${float(entry_px):,.2f}")
                    print(f"  [BTC]    SL/TP no disponibles — se gestionará con trailing ATR\n")
                    break
        except Exception as e:
            print(f"  [BTC] No se pudo consultar posiciones abiertas: {e}")

    # Shutdown flag
    _stop = asyncio.Event()
    _main_task: asyncio.Task | None = None

    # ── Live feed ─────────────────────────────────────────────────────────────
    feed = LiveFeed(client=client, store=store, symbol=SYMBOL, timeframe=TIMEFRAME)
    _main_task = asyncio.current_task()

    def _handle_signal(*_):
        print("\n\n  Ctrl+C — deteniendo bot (posición queda abierta en BingX)...")
        _stop.set()
        feed.stop()
        if _main_task and not _main_task.done():
            _main_task.cancel()

    signal.signal(signal.SIGINT,  _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    last_htf_refresh = time.time()
    HTF_REFRESH_SECS = 3600  # refresh 4H every hour

    print("  Waiting for live 15M bars...\n")
    try:
        while not _stop.is_set():
          try:
            async for bar in feed.stream():
                if _stop.is_set():
                    break

                # Skip bars already seen in warmup
                if last_processed_ts is not None and bar.ts <= last_processed_ts:
                    continue
                last_processed_ts = bar.ts

                bw.append(bar)
                if len(bw) < strategy.min_bars_required:
                    continue

                # Refresh HTF periodically
                now = time.time()
                if now - last_htf_refresh > HTF_REFRESH_SECS:
                    await _refresh_htf(client, store, SYMBOL, htf_df_holder)
                    if htf_df_holder[0] is not None:
                        strategy.set_htf_bars(htf_df_holder[0])
                    last_htf_refresh = now

                from app.strategy.base import BaseStrategy as _BS
                df = _BS.bars_to_df(list(bw))

                # ── 1. SL/TP check (MUST run first) ───────────────────────────
                if open_trade:
                    et, fill, cp = await _check_sl_tp(
                        open_trade, bar, adapter, strategy.strategy_id,
                        is_live=is_live,
                    )

                    if et and et.endswith("_order_failed"):
                        print(f"  [BTC] Retrying close next bar ({et})")
                        continue

                    elif et == "tp1" and fill:
                        np_ = compute_leg_pnl(open_trade, fill.price, fill.qty, fill.fee)
                        fp  = fill.qty / open_trade["original_qty"]
                        fi  = open_trade["fee_in"] * fp
                        open_trade["total_partial_pnl"]  += np_
                        open_trade["total_partial_fees"] += fi + fill.fee
                        open_trade["qty"] -= fill.qty
                        trail_sl = float(open_trade["sl_price"])
                        print(
                            f"\n  [BTC] TP1 {cp*100:.0f}% @ ${float(fill.price):,.2f}"
                            f"  pnl=${float(np_):+.2f}"
                            f"  → trailing SL ${trail_sl:,.2f}\n"
                        )
                        if notifier:
                            asyncio.create_task(notifier.tp1_hit(
                                symbol=SYMBOL,
                                exit_price=float(fill.price),
                                close_pct=cp,
                                partial_pnl=float(np_),
                                remaining_qty=float(open_trade["qty"]),
                                entry_price=float(open_trade["entry_price"]),
                            ))

                    elif et in ("sl", "be_sl", "trailing_sl", "tp2") and fill:
                        _ot_snap = dict(open_trade)
                        pnl = _record_close(open_trade, fill, et)
                        won = pnl > 0
                        wins   += 1 if won else 0
                        losses += 0 if won else 1
                        strategy.notify_trade_closed(open_trade["side"])

                        emoji = "✅" if won else "❌"
                        wr = wins / (wins + losses) * 100
                        print(
                            f"\n  [BTC] {emoji} {et.upper()} @ ${float(fill.price):,.2f}"
                            f"  pnl=${float(pnl):+.2f}"
                            f"  W={wins} L={losses} WR={wr:.0f}%\n"
                        )
                        if notifier:
                            asyncio.create_task(notifier.trade_closed(
                                symbol=SYMBOL,
                                side=_ot_snap["side"],
                                exit_type=et,
                                exit_price=float(fill.price),
                                entry_price=float(_ot_snap["entry_price"]),
                                total_pnl=float(pnl),
                                qty=float(_ot_snap["original_qty"]),
                                total_trades=wins + losses,
                                wins=wins,
                                losses=losses,
                            ))
                        open_trade = None

                # ── 2. Strategy signal ─────────────────────────────────────────
                sig = strategy.on_bar(df)

                if open_trade and sig.meta.get("atr"):
                    update_atr(open_trade, sig.meta["atr"])

                # ── 3. Print status ────────────────────────────────────────────
                try:
                    eq_bal = await adapter.get_balance()
                    for b in eq_bal:
                        if hasattr(b, "currency") and "USDT" in b.currency:
                            equity = b.total
                            break
                except Exception:
                    pass

                _print_status(bar, sig, open_trade, wins, losses, float(equity))

                # ── 4. Execute signal ──────────────────────────────────────────
                if not sig.is_actionable():
                    continue
                if open_trade is not None:
                    if sig.meta.get("is_reversal"):
                        from app.broker.base import OrderRequest, OrderSide, OrderType
                        il         = open_trade["side"] == "LONG"
                        close_side = OrderSide.SELL if il else OrderSide.BUY
                        close_px   = float(bar.close)
                        if not is_live:
                            fill = _make_paper_fill(open_trade, close_px,
                                                    open_trade["qty"], close_side)
                        else:
                            _, fill = await adapter.place_order(OrderRequest(
                                symbol=SYMBOL, side=close_side,
                                order_type=OrderType.MARKET,
                                qty=open_trade["qty"],
                                strategy_id=strategy.strategy_id,
                                extra={"positionSide": "LONG" if il else "SHORT"},
                            ))
                        if fill:
                            pnl = _record_close(open_trade, fill, "reversal_swap")
                            wins   += 1 if pnl > 0 else 0
                            losses += 0 if pnl > 0 else 1
                            print(f"\n  [BTC] ⟳ REVERSAL SWAP @ ${float(fill.price):,.2f}"
                                  f"  pnl=${float(pnl):+.2f}\n")
                        open_trade = None
                    else:
                        continue

                # ── 5. Open new trade ──────────────────────────────────────────
                meta      = sig.meta or {}
                conf      = meta.get("confidence_score", 0.5)
                tier      = meta.get("confidence_tier", "C")
                atr       = meta.get("atr", 0.0)
                sl_price  = float(sig.stop_loss)   if sig.stop_loss   else None
                tp1_px    = float(sig.take_profit) if sig.take_profit else None
                tp2_px    = meta.get("tp2")
                tp1_cp    = meta.get("tp1_close_pct", 0.33)
                trail_m   = meta.get("trailing_atr_mult", 1.5)
                sess_m    = meta.get("session_mult", 1.0)
                direction = "LONG" if sig.action.value == "BUY" else "SHORT"

                try:
                    bal = await adapter.get_balance()
                    for b in bal:
                        if hasattr(b, "currency") and "USDT" in b.currency:
                            equity = b.total
                            break
                except Exception:
                    pass

                close_px   = float(bar.close)
                # Sizing fijo: 1% del capital × leverage (máx 5x)
                lev_capped = min(args.leverage, 5)
                alloc_usdt = float(equity) * 0.01 * lev_capped
                # BTC-USDT en BingX tiene quantityPrecision=4 (mínimo 0.0001)
                qty        = Decimal(str(alloc_usdt / close_px)).quantize(Decimal("0.0001"))

                if qty <= Decimal("0.0001"):
                    print(f"  [BTC] Skipping signal — qty too small ({qty})")
                    continue

                from app.broker.base import OrderRequest, OrderSide, OrderType
                entry_side = OrderSide.BUY  if direction == "LONG" else OrderSide.SELL
                pos_side   = "LONG"         if direction == "LONG" else "SHORT"

                if not is_live:
                    fee = qty * Decimal(str(close_px)) * COMMISSION_RATE
                    from app.broker.base import FillResult
                    fill = FillResult(
                        fill_id=str(uuid.uuid4()),
                        order_id=str(uuid.uuid4()),
                        symbol=SYMBOL,
                        side=entry_side,
                        price=Decimal(str(close_px)),
                        qty=qty,
                        fee=fee,
                        fee_currency="USDT",
                        timestamp=datetime.now(timezone.utc),
                    )
                else:
                    _, fill = await adapter.place_order(OrderRequest(
                        symbol=SYMBOL, side=entry_side,
                        order_type=OrderType.MARKET, qty=qty,
                        strategy_id=strategy.strategy_id,
                        extra={"positionSide": pos_side},
                    ))

                if not fill:
                    print(f"  [BTC] Entry order failed — no fill returned")
                    continue

                open_trade = create_trade(
                    trade_id      = str(uuid.uuid4()),
                    symbol        = SYMBOL,
                    side          = direction,
                    entry_price   = fill.price,
                    qty           = fill.qty,
                    fee_in        = fill.fee,
                    sl_price      = sl_price,
                    tp1_price     = tp1_px,
                    tp2_price     = float(tp2_px) if tp2_px else None,
                    tp1_close_pct = tp1_cp,
                    soft_sl_bars  = 0,
                    trailing_atr  = trail_m,
                    current_atr   = atr,
                    signal_reason = sig.reason,
                    extra         = {
                        "tier":       tier,
                        "leverage":   args.leverage,
                        "entry_time": datetime.now(timezone.utc).isoformat(),
                    },
                )

                htf_tag = {1: "↑4H", -1: "↓4H⚠", 0: "~4H"}.get(meta.get("htf_bias", 0), "~4H")
                print(
                    f"\n  ┌{'─'*60}┐"
                    f"\n  │  {'▲ LONG' if direction == 'LONG' else '▼ SHORT'}"
                    f" Tier {tier}  conf={conf:.2f}  {htf_tag}"
                    f"\n  │  entry=${float(fill.price):,.2f}"
                    f"  SL=${sl_price:,.2f}"
                    f"  TP1=${tp1_px:,.2f}"
                    f"  TP2=${float(tp2_px) if tp2_px else 0:,.2f}"
                    f"\n  │  qty={float(fill.qty):.6f} BTC"
                    f"  alloc=${alloc_usdt:,.2f} USDT"
                    f"  sess={meta.get('session','?')}({sess_m})"
                    f"\n  │  RSI={meta.get('rsi',0):.1f}"
                    f"  VOL={meta.get('vol_ratio',0):.2f}"
                    f"  ADX={meta.get('adx',0):.1f}"
                    f"\n  └{'─'*60}┘\n"
                )

                if notifier:
                    asyncio.create_task(notifier.trade_opened(
                        symbol=SYMBOL,
                        side=direction,
                        tier=tier,
                        leverage=args.leverage,
                        entry_price=float(fill.price),
                        qty=float(fill.qty),
                        stop_loss=sl_price,
                        tp1=tp1_px,
                        tp2=float(tp2_px) if tp2_px else None,
                        session=meta.get("session", ""),
                        reason=sig.reason or "",
                        equity=float(equity),
                    ))

          except DataDelayError as e:
              print(f"  [BTC] ⚠️  Data delay ({e}) — reconnecting in 30s...")
              if notifier:
                  asyncio.create_task(notifier.send(
                      f"⚠️ <b>Data delay — BTC-USDT</b>\n"
                      f"🕐 {e}\nReconectando en 30s..."
                  ))
              await asyncio.sleep(30)
              feed = LiveFeed(client=client, store=store, symbol=SYMBOL, timeframe=TIMEFRAME)
    except asyncio.CancelledError:
        print("  [BTC] Loop cancelado por Ctrl+C")
    except Exception as e:
        import traceback
        print(f"  [BTC] Error: {e}")
        traceback.print_exc()
    finally:
        feed.stop()

        # ── Shutdown: NO cerrar posición — queda abierta en BingX ───────────────
        if open_trade and open_trade["qty"] > Decimal("0.0001"):
            print(f"\n  [BTC] Posición {open_trade['side']} {float(open_trade['qty']):.4f} BTC"
                  f" @ ${float(open_trade['entry_price']):,.2f} — quedó abierta en BingX.")
            print(f"  [BTC] Al reiniciar el bot la detectará automáticamente.")
        else:
            print(f"\n  [BTC] Sin posición abierta al cerrar.")

        print(f"\n  Trades guardados en {TRADES_CSV}")
        print(f"  Final: W={wins} L={losses}")
        await client.close()


# ═══════════════════════════════════════════════════════════════════════════════
# Connection check
# ═══════════════════════════════════════════════════════════════════════════════

async def _check_connection(args):
    from app.config import get_settings
    from app.core.logging import configure_logging
    s = get_settings()
    configure_logging(log_level="ERROR", log_format="console")
    from app.broker.bingx_client import BingXClient
    c = BingXClient(api_key=s.bingx_api_key, api_secret=s.bingx_api_secret,
                    base_url=s.bingx_base_url, market_type=s.bingx_market_type)
    print(f"\nChecking BingX API connection...")
    try:
        tk = await c.get_ticker(SYMBOL)
        print(f"  [OK] {SYMBOL:<12} ${float(tk['last']):>12,.4f}")
    except Exception as e:
        print(f"  [FAIL] {SYMBOL}: {e}")
    try:
        raw = await c.get_balance()
        u = next((a for a in raw if "USDT" in str(a.get("asset", ""))), None)
        print(f"  Balance: {u.get('balance','?') if u else '?'} USDT")
    except Exception as e:
        print(f"  Balance: {e}")
    await c.close()
    print()


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def _parse_args():
    p = argparse.ArgumentParser(
        description="TrendBot MTF v5.2 — BTC-USDT via BingX",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--check",       action="store_true",  help="Test API connection and exit")
    p.add_argument("--live-bingx",  action="store_true",  help="REAL orders on BingX (USDT real!)")
    p.add_argument("--balance",     type=float, default=10000, help="Paper balance in USDT (default: 10000)")
    p.add_argument("--leverage",    type=int,   default=5,     help="Leverage (default: 5, max aplicado: 5x)")
    p.add_argument("--params-file", type=str,   default=None,  help="JSON file with extra params to override")

    # Strategy params
    p.add_argument("--adx-min",      type=int,   default=20,   help="ADX mínimo (default: 20)")
    p.add_argument("--pb-tol",       type=float, default=1.2,  help="Pullback tolerance ATR (default: 1.2)")
    p.add_argument("--sl-min-atr",   type=float, default=1.2,  help="SL mínimo en ATR (default: 1.2)")
    p.add_argument("--cooldown",     type=int,   default=3,    help="Cooldown entre señales en barras (default: 3)")
    p.add_argument("--min-conf",     type=float, default=0.0,  help="Confianza mínima (default: 0.0)")
    p.add_argument("--trail-mult",   type=float, default=1.5,  help="Trailing ATR multiplier (default: 1.5)")
    p.add_argument("--no-rsi",       action="store_true",      help="Desactiva filtro RSI")
    p.add_argument("--no-vol",       action="store_true",      help="Desactiva filtro de volumen")
    p.add_argument("--no-short",     action="store_true",      help="Solo LONGs")
    p.add_argument("--no-reversal",  action="store_true",      help="Desactiva reversal swap")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if args.live_bingx:
        print("\n" + "█" * 72)
        print("  ⚠️   MODO LIVE — SE ENVIARÁN ÓRDENES REALES A BINGX")
        print("  ⚠️   Solo continúa si has probado en paper mode primero")
        print("█" * 72)
        confirm = input("\n  Escribe CONFIRMAR para continuar (o Enter para cancelar): ")
        if confirm.strip() != "CONFIRMAR":
            print("  Cancelado.")
            exit(0)

    if args.check:
        asyncio.run(_check_connection(args))
    else:
        asyncio.run(_run_btc(args))