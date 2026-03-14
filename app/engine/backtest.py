"""
Backtest engine — multi-position edition.

Flow:
  1. Load bars from Parquet
  2. Warm up strategy (min_bars_required)
  3. Iterate bar by bar:
     a. adapter.advance(bar) → fills pending orders
     b. strategy.on_bar_all(window) → list[Signal]
     c. risk.validate_signal()
     d. if approved: adapter.place_order() → queued for next bar fill
     e. record fills, update equity, check risk
  4. Close any open positions at end
  5. Compute metrics

Multi-position support:
  - open_trades: dict[position_id, trade_dict]
  - Each BUY/SELL signal creates a new entry keyed by position_id
  - CLOSE/PARTIAL_CLOSE signals reference position_id from meta
  - Falls back to closing the oldest trade if no position_id in meta

SHORT support:
  - SignalAction.SELL (when no open trade) opens a SHORT position
  - PnL = (entry - exit) * qty  for SHORT  (reversed from LONG)

SL/TP stored in trade records:
  - sl_price, tp1_price, tp2_price extracted from entry signal meta
  - exit_type stored on close: "sl", "tp1", "tp2", "be_sl", "forced"
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

import numpy as np
import pandas as pd

from app.broker.backtest_adapter import BacktestAdapter
from app.broker.base import OHLCVBar, OrderRequest, OrderSide, OrderType, TradeSide
from app.config import get_settings
from app.core.exceptions import KillSwitchTriggered
from app.core.logging import get_logger
from app.data.parquet_store import ParquetStore
from app.risk.manager import RiskManager
from app.strategy.base import BaseStrategy
from app.strategy.signals import Signal, SignalAction

logger = get_logger(__name__)
_settings = get_settings()

WINDOW_SIZE = 250


@dataclass
class BacktestResult:
    run_id: str
    symbol: str
    timeframe: str
    strategy_id: str
    start_date: datetime
    end_date: datetime
    params: dict

    total_bars: int = 0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: Decimal = Decimal("0")
    total_pnl_pct: Decimal = Decimal("0")
    max_drawdown_pct: Decimal = Decimal("0")
    sharpe_ratio: Decimal = Decimal("0")
    avg_trade_pnl: Decimal = Decimal("0")
    winrate: Decimal = Decimal("0")
    exposure_pct: Decimal = Decimal("0")

    trades: list[dict] = field(default_factory=list)
    equity_curve: list[dict] = field(default_factory=list)
    signals: list[dict] = field(default_factory=list)
    bar_data: list[dict] = field(default_factory=list)

    def to_report(self) -> dict:
        report = {
            "run_id": self.run_id,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "strategy_id": self.strategy_id,
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "params": self.params,
            "metrics": {
                "total_bars": self.total_bars,
                "total_trades": self.total_trades,
                "winning_trades": self.winning_trades,
                "losing_trades": self.losing_trades,
                "winrate": float(self.winrate),
                "total_pnl": float(self.total_pnl),
                "total_pnl_pct": float(self.total_pnl_pct),
                "max_drawdown_pct": float(self.max_drawdown_pct),
                "sharpe_ratio": float(self.sharpe_ratio),
                "avg_trade_pnl": float(self.avg_trade_pnl),
                "exposure_pct": float(self.exposure_pct),
            },
            "trades": self.trades,
            "equity_curve": [
                {"ts": e["ts"].isoformat(), "equity": float(e["equity"])}
                for e in self.equity_curve
            ],
        }
        if self.bar_data:
            report["bar_data"] = self.bar_data
        return report


class BacktestEngine:
    """
    Deterministic single-symbol backtest engine with multi-position support.

    Handles BUY (open LONG), SELL (open SHORT), CLOSE, PARTIAL_CLOSE signals.
    """

    def __init__(
        self,
        strategy: BaseStrategy,
        store: ParquetStore,
        initial_balance: Decimal = Decimal("10000"),
        commission_rate: Decimal | None = None,
        slippage_rate: Decimal | None = None,
        verbose: bool = False,
    ) -> None:
        self._strategy = strategy
        self._store = store
        self._initial_balance = initial_balance
        self._commission = commission_rate or _settings.commission_rate
        self._slippage = slippage_rate or _settings.slippage_rate
        self._risk = RiskManager()
        self._verbose = verbose

    async def run(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
        params: dict | None = None,
        htf_timeframe: str = "1h",
    ) -> BacktestResult:
        run_id = str(uuid.uuid4())
        logger.info(
            "backtest_start",
            run_id=run_id, symbol=symbol, timeframe=timeframe,
            htf_timeframe=htf_timeframe,
            start=start.isoformat(), end=end.isoformat(),
        )

        bars = self._store.read_bars(symbol, timeframe, start, end)
        if len(bars) < self._strategy.min_bars_required + 10:
            raise ValueError(
                f"Not enough bars: {len(bars)} < {self._strategy.min_bars_required + 10}"
            )

        # Load HTF bars for trend filter
        _strategy_has_htf = hasattr(self._strategy, "set_htf_bars")
        _htf_df: pd.DataFrame | None = None
        if _strategy_has_htf:
            from datetime import timedelta
            _htf_warmup_days = 250
            htf_bars_raw = self._store.read_bars(
                symbol, htf_timeframe,
                start - timedelta(days=_htf_warmup_days), end,
            )
            if htf_bars_raw:
                _htf_df = pd.DataFrame({
                    "ts":     pd.to_datetime([b.ts for b in htf_bars_raw], utc=True),
                    "open":   [float(b.open)   for b in htf_bars_raw],
                    "high":   [float(b.high)   for b in htf_bars_raw],
                    "low":    [float(b.low)    for b in htf_bars_raw],
                    "close":  [float(b.close)  for b in htf_bars_raw],
                    "volume": [float(b.volume) for b in htf_bars_raw],
                }).sort_values("ts").reset_index(drop=True)
                logger.info("htf_bars_loaded", symbol=symbol, htf_timeframe=htf_timeframe, bars=len(_htf_df))
            else:
                logger.warning("htf_bars_not_found", symbol=symbol, htf_timeframe=htf_timeframe)

        adapter = BacktestAdapter(
            bars=bars,
            initial_balance=self._initial_balance,
            commission_rate=self._commission,
            slippage_rate=self._slippage,
        )
        self._risk.initialize(self._initial_balance, as_of_date=bars[0].ts.date())

        result = BacktestResult(
            run_id=run_id, symbol=symbol, timeframe=timeframe,
            strategy_id=self._strategy.strategy_id,
            start_date=start, end_date=end,
            params=params or self._strategy.params,
        )

        # ── Multi-position tracking ────────────────────────────────────────
        # open_trades: position_id → trade dict
        open_trades: dict[str, dict] = {}
        # pending_entries: order_id → {position_id, signal}
        pending_entries: dict[str, dict] = {}
        # pending_closes: order_id → {position_id, close_pct}  (close_pct=1.0 = full)
        pending_closes: dict[str, dict] = {}

        bars_in_position = 0
        total_bars = len(bars)

        # ── Main loop ──────────────────────────────────────────────────────
        for i, bar in enumerate(bars):
            # 1. Advance adapter (fills pending orders at this bar's open)
            fills = adapter.advance(bar)

            for fill in fills:
                # ── Entry fill ─────────────────────────────────────────────
                if fill.order_id in pending_entries:
                    pe = pending_entries.pop(fill.order_id)
                    pid = pe["position_id"]
                    sig = pe["signal"]
                    side_str = "LONG" if sig.action == SignalAction.BUY else "SHORT"
                    open_trades[pid] = {
                        "trade_id": str(uuid.uuid4()),
                        "position_id": pid,
                        "symbol": symbol,
                        "side": side_str,
                        "entry_ts": fill.timestamp,
                        "entry_price": fill.price,
                        "qty": fill.qty,
                        "original_qty": fill.qty,
                        "fee_in": fill.fee,
                        "total_partial_pnl": Decimal("0"),
                        "total_partial_fees": Decimal("0"),
                        "mode": "backtest",
                        # SL / TP from entry signal
                        "sl_price": float(sig.stop_loss) if sig.stop_loss else None,
                        "tp1_price": float(sig.take_profit) if sig.take_profit else None,
                        "tp2_price": sig.meta.get("tp2"),
                        "entry_rsi": sig.meta.get("rsi"),
                        "entry_ema": sig.meta.get("ema"),
                    }
                    logger.debug("backtest_entry", side=side_str, price=str(fill.price),
                                 ts=fill.timestamp.isoformat())

                # ── Partial close fill ─────────────────────────────────────
                elif fill.order_id in pending_closes:
                    pc = pending_closes.pop(fill.order_id)
                    pid = pc["position_id"]
                    close_pct = pc["close_pct"]
                    exit_type = pc.get("exit_type", "tp1")

                    if pid in open_trades:
                        trade = open_trades[pid]
                        # Use the strategy-provided limit price (TP/SL level) when available,
                        # so PnL reflects the actual trigger level rather than next-bar open.
                        intended_price = pc.get("limit_price")
                        if intended_price is not None:
                            exit_price = Decimal(str(intended_price))
                            fee_out = exit_price * fill.qty * self._commission
                        else:
                            exit_price = fill.price
                            fee_out = fill.fee
                        entry_price = trade["entry_price"]
                        closed_qty = fill.qty
                        is_long = trade["side"] == "LONG"

                        # PnL direction: LONG = exit-entry; SHORT = entry-exit
                        price_diff = (exit_price - entry_price) if is_long else (entry_price - exit_price)
                        gross_pnl = price_diff * closed_qty

                        pct_of_orig = closed_qty / trade["original_qty"]
                        fee_in_part = trade["fee_in"] * pct_of_orig
                        net_pnl = gross_pnl - fee_in_part - fee_out

                        trade["total_partial_pnl"] += net_pnl
                        trade["total_partial_fees"] += fee_in_part + fee_out
                        trade["qty"] -= closed_qty

                        if trade["qty"] < Decimal("0.0001"):
                            self._record_trade(result, trade, exit_price, fill.timestamp,
                                               fee_out, exit_type=exit_type, is_final_partial=True)
                            del open_trades[pid]

            # Count bars in position
            bars_in_position += len(open_trades)

            # 2. Build strategy window
            window_start = max(0, i + 1 - WINDOW_SIZE)
            window = bars[window_start: i + 1]
            if len(window) < 2:
                continue

            df = BaseStrategy.bars_to_df(window)

            # 2b. Feed HTF bars (no lookahead)
            if _strategy_has_htf and _htf_df is not None:
                cur_ts = (pd.Timestamp(bar.ts).tz_convert("UTC")
                          if bar.ts.tzinfo else pd.Timestamp(bar.ts, tz="UTC"))
                htf_window = _htf_df[_htf_df["ts"] < cur_ts].tail(300)
                if len(htf_window) > 0:
                    self._strategy.set_htf_bars(htf_window)

            # 3. Get signals (multi-position aware)
            if hasattr(self._strategy, "on_bar_all"):
                signals: list[Signal] = self._strategy.on_bar_all(df)
            else:
                signals = [self._strategy.on_bar(df)]

            # Record actionable signals for reporting
            for sig in signals:
                if sig.is_actionable():
                    result.signals.append({
                        "ts": bar.ts.isoformat(),
                        "action": sig.action.value,
                        "reason": sig.reason,
                    })

            if self._verbose:
                # Merge all signal metas into one bar_data entry
                merged_meta = {}
                for sig in signals:
                    merged_meta.update(sig.meta)
                primary_sig = signals[0] if signals else None
                result.bar_data.append({
                    "ts": bar.ts.isoformat(),
                    "o": float(bar.open), "h": float(bar.high),
                    "l": float(bar.low), "c": float(bar.close),
                    "v": float(bar.volume),
                    "signal": primary_sig.action.value if primary_sig else "HOLD",
                    "reason": primary_sig.reason if primary_sig else "",
                    "meta": merged_meta,
                })

            # 4. Risk + equity
            balances = await adapter.get_balance()
            cash = balances[0].total if balances else self._initial_balance
            equity = cash
            for pid, trade in open_trades.items():
                pos_qty = trade["qty"]
                is_long = trade["side"] == "LONG"
                if is_long:
                    equity += pos_qty * bar.close
                else:
                    equity -= pos_qty * bar.close

            kill_switch_active = False
            try:
                self._risk.update_equity(equity, as_of_date=bar.ts.date())
            except KillSwitchTriggered:
                kill_switch_active = True

            # 5. Execute signals
            for sig in signals:
                # Skip HOLD
                if sig.action == SignalAction.HOLD:
                    continue

                if kill_switch_active and sig.action not in (SignalAction.CLOSE, SignalAction.PARTIAL_CLOSE):
                    continue

                if not kill_switch_active:
                    try:
                        approved, reason = self._risk.validate_signal(sig, equity)
                    except Exception as e:
                        logger.warning("backtest_risk_error", error=str(e))
                        continue
                    if not approved:
                        continue

                pid = sig.meta.get("position_id", "")

                # ── Open LONG ──────────────────────────────────────────────
                if sig.action == SignalAction.BUY:
                    # Don't double-open same position
                    if pid and pid in open_trades:
                        continue
                    qty = self._risk.compute_order_qty(sig, equity, bar.close)
                    req = OrderRequest(
                        symbol=symbol, side=OrderSide.BUY,
                        order_type=OrderType.MARKET, qty=qty,
                        strategy_id=self._strategy.strategy_id,
                    )
                    order, _ = await adapter.place_order(req)
                    pending_entries[order.order_id] = {
                        "position_id": pid or str(uuid.uuid4()),
                        "signal": sig,
                    }

                # ── Open SHORT ─────────────────────────────────────────────
                elif sig.action == SignalAction.SELL and not (pid and pid in open_trades):
                    qty = self._risk.compute_order_qty(sig, equity, bar.close)
                    req = OrderRequest(
                        symbol=symbol, side=OrderSide.SELL,
                        order_type=OrderType.MARKET, qty=qty,
                        strategy_id=self._strategy.strategy_id,
                    )
                    order, _ = await adapter.place_order(req)
                    pending_entries[order.order_id] = {
                        "position_id": pid or str(uuid.uuid4()),
                        "signal": sig,
                    }

                # ── Partial close ──────────────────────────────────────────
                elif sig.action == SignalAction.PARTIAL_CLOSE:
                    # Find the trade to partially close
                    target_trade = None
                    if pid and pid in open_trades:
                        target_trade = open_trades[pid]
                    elif open_trades:
                        target_trade = next(iter(open_trades.values()))

                    if target_trade:
                        close_pct = sig.close_pct
                        close_qty = (target_trade["qty"] * Decimal(str(close_pct))).quantize(Decimal("0.001"))
                        close_qty = min(close_qty, target_trade["qty"])
                        if close_qty > Decimal("0"):
                            is_long = target_trade["side"] == "LONG"
                            req = OrderRequest(
                                symbol=symbol,
                                side=OrderSide.SELL if is_long else OrderSide.BUY,
                                order_type=OrderType.MARKET, qty=close_qty,
                                strategy_id=self._strategy.strategy_id,
                            )
                            order, _ = await adapter.place_order(req)
                            pending_closes[order.order_id] = {
                                "position_id": target_trade["position_id"],
                                "close_pct": close_pct,
                                "exit_type": sig.meta.get("exit_type", "tp1"),
                                "limit_price": sig.meta.get("exit_price"),
                            }

                # ── Full close ─────────────────────────────────────────────
                elif sig.action == SignalAction.CLOSE:
                    target_trade = None
                    if pid and pid in open_trades:
                        target_trade = open_trades[pid]
                    elif open_trades:
                        target_trade = next(iter(open_trades.values()))

                    if target_trade:
                        qty = target_trade["qty"]
                        is_long = target_trade["side"] == "LONG"
                        req = OrderRequest(
                            symbol=symbol,
                            side=OrderSide.SELL if is_long else OrderSide.BUY,
                            order_type=OrderType.MARKET, qty=qty,
                            strategy_id=self._strategy.strategy_id,
                        )
                        order, _ = await adapter.place_order(req)
                        pending_closes[order.order_id] = {
                            "position_id": target_trade["position_id"],
                            "close_pct": 1.0,
                            "exit_type": sig.meta.get("exit_type", "close"),
                            "limit_price": sig.meta.get("exit_price"),
                        }

            # Handle fills queued by the close orders placed above
            # (these are placed at this bar; filled at next bar — adapter handles it)

            # 6. Process any immediate fills from the close orders just placed
            # Note: close orders placed above are queued and filled on the NEXT bar.
            # The adapter.advance() at the start of the next iteration will handle them.
            # But we need to handle any closes that ARE pending now to close correctly.
            # Actually we handle this via a second fill scan on the same bar for immediate closes:
            # (The BacktestAdapter queues orders and fills them on advance(), so no action needed here.)

            result.equity_curve.append({"ts": bar.ts, "equity": equity})

        # ── Process any remaining fills from the last bar ─────────────────
        # (advance() is called at the START of each iteration, so there may be
        # pending orders from the last bar that never got advanced)
        if pending_entries or pending_closes:
            if bars:
                last_bar = bars[-1]
                leftover_fills = adapter.advance(last_bar)
                for fill in leftover_fills:
                    if fill.order_id in pending_entries:
                        pe = pending_entries.pop(fill.order_id)
                        pid = pe["position_id"]
                        sig = pe["signal"]
                        side_str = "LONG" if sig.action == SignalAction.BUY else "SHORT"
                        open_trades[pid] = {
                            "trade_id": str(uuid.uuid4()),
                            "position_id": pid,
                            "symbol": symbol,
                            "side": side_str,
                            "entry_ts": fill.timestamp,
                            "entry_price": fill.price,
                            "qty": fill.qty,
                            "original_qty": fill.qty,
                            "fee_in": fill.fee,
                            "total_partial_pnl": Decimal("0"),
                            "total_partial_fees": Decimal("0"),
                            "mode": "backtest",
                            "sl_price": float(sig.stop_loss) if sig.stop_loss else None,
                            "tp1_price": float(sig.take_profit) if sig.take_profit else None,
                            "tp2_price": sig.meta.get("tp2"),
                            "entry_rsi": sig.meta.get("rsi"),
                            "entry_ema": sig.meta.get("ema"),
                        }
                    elif fill.order_id in pending_closes:
                        pc = pending_closes.pop(fill.order_id)
                        pid = pc["position_id"]
                        if pid in open_trades:
                            trade = open_trades[pid]
                            intended_price = pc.get("limit_price")
                            if intended_price is not None:
                                exit_price = Decimal(str(intended_price))
                                fee_out = exit_price * fill.qty * self._commission
                            else:
                                exit_price = fill.price
                                fee_out = fill.fee
                            entry_price = trade["entry_price"]
                            closed_qty = fill.qty
                            is_long = trade["side"] == "LONG"
                            price_diff = (exit_price - entry_price) if is_long else (entry_price - exit_price)
                            gross_pnl = price_diff * closed_qty
                            pct_of_orig = closed_qty / trade["original_qty"]
                            fee_in_part = trade["fee_in"] * pct_of_orig
                            net_pnl = gross_pnl - fee_in_part - fee_out
                            trade["total_partial_pnl"] += net_pnl
                            trade["total_partial_fees"] += fee_in_part + fee_out
                            trade["qty"] -= closed_qty
                            if trade["qty"] < Decimal("0.0001"):
                                self._record_trade(result, trade, exit_price, fill.timestamp,
                                                   fee_out, exit_type=pc.get("exit_type", "close"),
                                                   is_final_partial=True)
                                del open_trades[pid]

        # ── Force-close remaining open positions at end ───────────────────
        if open_trades and bars:
            last_bar = bars[-1]
            for pid, trade in list(open_trades.items()):
                exit_price = last_bar.close
                entry_price = trade["entry_price"]
                qty = trade["qty"]
                is_long = trade["side"] == "LONG"
                price_diff = (exit_price - entry_price) if is_long else (entry_price - exit_price)
                gross_pnl = price_diff * qty
                remaining_pct = qty / trade["original_qty"]
                fee_in_portion = trade["fee_in"] * remaining_pct
                fee_out = exit_price * qty * self._commission
                net_pnl_this_leg = gross_pnl - fee_in_portion - fee_out
                total_net_pnl = trade["total_partial_pnl"] + net_pnl_this_leg
                total_fees = trade["total_partial_fees"] + fee_in_portion + fee_out

                trade_record = {
                    "trade_id":    trade["trade_id"],
                    "position_id": pid,
                    "symbol":      trade["symbol"],
                    "side":        trade["side"],
                    "entry_ts":    trade["entry_ts"],
                    "entry_price": trade["entry_price"],
                    "qty":         trade["original_qty"],
                    "fee_in":      trade["fee_in"],
                    "mode":        trade["mode"],
                    "exit_ts":     last_bar.ts,
                    "exit_price":  exit_price,
                    "pnl":         total_net_pnl,
                    "pnl_pct":     total_net_pnl / (entry_price * trade["original_qty"]) * 100,
                    "fees":        total_fees,
                    "had_partial_close": trade["total_partial_pnl"] != Decimal("0"),
                    "forced_close": True,
                    "exit_type":   "forced",
                    "sl_price":    trade.get("sl_price"),
                    "tp1_price":   trade.get("tp1_price"),
                    "tp2_price":   trade.get("tp2_price"),
                    "entry_rsi":   trade.get("entry_rsi"),
                    "entry_ema":   trade.get("entry_ema"),
                }
                result.trades.append(
                    {k: str(v) if isinstance(v, Decimal) else v
                     for k, v in trade_record.items()}
                )
                result.total_trades += 1
                if total_net_pnl > 0:
                    result.winning_trades += 1
                else:
                    result.losing_trades += 1
                result.total_pnl += total_net_pnl
            open_trades.clear()

        # ── Summary metrics ────────────────────────────────────────────────
        result.total_bars = total_bars
        if result.total_trades > 0:
            result.winrate = Decimal(str(
                round(result.winning_trades / result.total_trades * 100, 1)
            ))
            result.avg_trade_pnl = result.total_pnl / result.total_trades
        result.total_pnl_pct = Decimal(str(
            round(float(result.total_pnl / self._initial_balance * 100), 2)
        ))
        result.exposure_pct = Decimal(str(
            round(bars_in_position / max(total_bars, 1) * 100, 2)
        ))

        if result.equity_curve:
            equities = [float(e["equity"]) for e in result.equity_curve]
            peak = equities[0]
            max_dd = 0.0
            for eq in equities:
                peak = max(peak, eq)
                dd = (peak - eq) / peak * 100 if peak > 0 else 0
                max_dd = max(max_dd, dd)
            result.max_drawdown_pct = Decimal(str(round(max_dd, 2)))

        if result.trades:
            pnls = [float(t["pnl"]) for t in result.trades]
            if len(pnls) >= 2 and np.std(pnls) > 0:
                sharpe = np.mean(pnls) / np.std(pnls) * np.sqrt(len(pnls))
                result.sharpe_ratio = Decimal(str(round(sharpe, 3)))

        logger.info(
            "backtest_complete",
            run_id=run_id, total_trades=result.total_trades,
            total_pnl=str(result.total_pnl), winrate=str(result.winrate),
            max_dd=str(result.max_drawdown_pct),
        )
        return result

    def _record_trade(
        self,
        result: "BacktestResult",
        trade: dict,
        final_exit_price: Decimal,
        exit_ts: datetime,
        final_fee_out: Decimal,
        exit_type: str = "close",
        is_final_partial: bool = False,
    ) -> None:
        """Record a completed trade (accumulated partials or full close)."""
        entry_price = trade["entry_price"]
        total_net_pnl = trade["total_partial_pnl"]
        total_fees = trade["total_partial_fees"]

        trade_record = {
            "trade_id":    trade["trade_id"],
            "position_id": trade.get("position_id", ""),
            "symbol":      trade["symbol"],
            "side":        trade["side"],
            "entry_ts":    trade["entry_ts"],
            "entry_price": entry_price,
            "qty":         trade["original_qty"],
            "fee_in":      trade["fee_in"],
            "mode":        trade["mode"],
            "exit_ts":     exit_ts,
            "exit_price":  final_exit_price,
            "pnl":         total_net_pnl,
            "pnl_pct":     total_net_pnl / (entry_price * trade["original_qty"]) * 100,
            "fees":        total_fees,
            "had_partial_close": True,
            "exit_type":   exit_type,
            "sl_price":    trade.get("sl_price"),
            "tp1_price":   trade.get("tp1_price"),
            "tp2_price":   trade.get("tp2_price"),
            "entry_rsi":   trade.get("entry_rsi"),
            "entry_ema":   trade.get("entry_ema"),
        }
        result.trades.append(
            {k: str(v) if isinstance(v, Decimal) else v
             for k, v in trade_record.items()}
        )
        result.total_trades += 1
        if total_net_pnl > 0:
            result.winning_trades += 1
        else:
            result.losing_trades += 1
        result.total_pnl += total_net_pnl
        self._risk.record_fill(total_net_pnl)

    # Keep legacy _record_completed_trade for any external callers
    def _record_completed_trade(self, result, open_trade, final_exit_price, exit_ts, final_fee_out, is_final_partial=False):
        self._record_trade(result, open_trade, final_exit_price, exit_ts, final_fee_out, is_final_partial=is_final_partial)
