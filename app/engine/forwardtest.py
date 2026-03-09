"""
Forward test (paper trading) engine.

Flow (per closed bar):
  1. Receive bar from LiveFeed
  2. Maintain rolling window of recent bars
  3. strategy.on_bar(window) → Signal
  4. risk.validate_signal()
  5. If approved: adapter.place_order() → immediate fill at ticker + slippage
  6. Persist Order + Fill + Trade to PostgreSQL
  7. Update Prometheus metrics
  8. Heartbeat to bot_state table

Supports PARTIAL_CLOSE:
  - Closes a fraction (signal.meta["close_pct"]) of the current position
  - Records a partial fill but keeps the trade open with reduced qty

Can be switched to real mode by substituting PaperAdapter → BingXAdapter.
"""
from __future__ import annotations

import asyncio
import uuid
from collections import deque
from datetime import datetime, timezone
from decimal import Decimal
from typing import Deque

from sqlalchemy.ext.asyncio import AsyncSession

from app.broker.base import (
    BrokerAdapter,
    FillResult,
    OHLCVBar,
    OrderRequest,
    OrderResult,
    OrderSide,
    OrderStatus,
    OrderType,
)
from app.config import get_settings
from app.core.exceptions import KillSwitchTriggered
from app.core.logging import get_logger
from app.data.feed import LiveFeed
from app.db.models import BotState, Fill, Order, Trade
from app.risk.manager import RiskManager
from app.strategy.base import BaseStrategy
from app.strategy.signals import Signal, SignalAction

logger = get_logger(__name__)
_settings = get_settings()

WINDOW_SIZE = 250  # Rolling bar window for strategy


class ForwardTestEngine:
    """
    Paper (or real) trading engine driven by live market data.
    Swap PaperAdapter for BingXAdapter to go live.
    """

    def __init__(
        self,
        strategy: BaseStrategy,
        adapter: BrokerAdapter,
        feed: LiveFeed,
        risk: RiskManager,
        session: AsyncSession,
        mode: str = "paper",
    ) -> None:
        self._strategy = strategy
        self._adapter = adapter
        self._feed = feed
        self._risk = risk
        self._session = session
        self._mode = mode
        self._running = False
        self._bar_window: Deque[OHLCVBar] = deque(maxlen=WINDOW_SIZE)
        self._open_trade: dict | None = None
        self._trace_id = str(uuid.uuid4())[:8]

    async def run(self) -> None:
        """
        Main event loop. Runs until stopped or kill switch triggers.
        """
        self._running = True
        symbol = self._strategy.symbol
        logger.info(
            "engine_start",
            mode=self._mode,
            strategy=self._strategy.strategy_id,
            symbol=symbol,
            trace_id=self._trace_id,
        )

        # Initialize risk manager with current balance
        balances = await self._adapter.get_balance()
        equity = balances[0].total if balances else Decimal("10000")
        self._risk.initialize(equity)

        try:
            async for bar in self._feed.stream():
                if not self._running:
                    break
                self._bar_window.append(bar)
                try:
                    await self._process_bar(bar)
                except KillSwitchTriggered as e:
                    logger.error("kill_switch", reason=str(e), trace_id=self._trace_id)
                    await self._update_bot_state(is_running=False)
                    break
                except Exception as e:
                    logger.error("bar_processing_error", error=str(e), trace_id=self._trace_id)
                    self._risk.on_api_error(str(e))
        finally:
            self._running = False
            await self._update_bot_state(is_running=False)
            logger.info("engine_stopped", trace_id=self._trace_id)

    async def stop(self) -> None:
        self._running = False

    async def _process_bar(self, bar: OHLCVBar) -> None:
        """Process a single closed bar."""
        if len(self._bar_window) < 2:
            return

        from app.strategy.base import BaseStrategy
        bars_list = list(self._bar_window)
        df = BaseStrategy.bars_to_df(bars_list)
        signal = self._strategy.on_bar(df)

        # Get current equity
        balances = await self._adapter.get_balance()
        equity = balances[0].total if balances else Decimal("10000")

        try:
            self._risk.update_equity(equity)
        except KillSwitchTriggered:
            raise

        logger.debug(
            "bar_processed",
            ts=bar.ts.isoformat(),
            close=str(bar.close),
            signal=signal.action.value,
            reason=signal.reason,
        )

        # Emit metrics
        try:
            from app.metrics.prometheus import TRADING_SIGNALS
            TRADING_SIGNALS.labels(
                symbol=bar.symbol, action=signal.action.value
            ).inc()
        except Exception:
            pass

        if not signal.is_actionable():
            await self._heartbeat()
            return

        approved, reason = self._risk.validate_signal(signal, equity)
        if not approved:
            logger.info("signal_rejected", reason=reason, signal=str(signal))
            await self._heartbeat()
            return

        await self._execute_signal(signal, bar, equity)
        await self._heartbeat()

    async def _execute_signal(
        self, signal: Signal, bar: OHLCVBar, equity: Decimal
    ) -> None:
        """Translate signal into broker order and persist results."""
        ticker = await self._adapter.get_ticker(bar.symbol)
        price = Decimal(str(ticker["last"]))

        if signal.action == SignalAction.BUY and self._open_trade is None:
            qty = self._risk.compute_order_qty(signal, equity, price)
            req = OrderRequest(
                symbol=bar.symbol,
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                qty=qty,
                strategy_id=self._strategy.strategy_id,
            )
            order, fill = await self._adapter.place_order(req)
            if fill:
                await self._persist_order_fill(order, fill)
                self._open_trade = {
                    "trade_id": str(uuid.uuid4()),
                    "entry_order_id": order.order_id,
                    "symbol": bar.symbol,
                    "side": "LONG",
                    "entry_ts": fill.timestamp,
                    "entry_price": fill.price,
                    "qty": fill.qty,
                    "original_qty": fill.qty,
                    "fee_in": fill.fee,
                    "total_partial_pnl": Decimal("0"),
                    "total_partial_fees": Decimal("0"),
                }
                logger.info(
                    "trade_opened",
                    symbol=bar.symbol,
                    qty=str(qty),
                    price=str(fill.price),
                    trade_id=self._open_trade["trade_id"],
                )

        elif signal.action == SignalAction.PARTIAL_CLOSE and self._open_trade:
            # ── PARTIAL CLOSE ──────────────────────────────────────
            close_pct = signal.close_pct
            close_qty = (self._open_trade["qty"] * Decimal(str(close_pct))).quantize(Decimal("0.001"))
            close_qty = min(close_qty, self._open_trade["qty"])

            if close_qty > Decimal("0"):
                req = OrderRequest(
                    symbol=bar.symbol,
                    side=OrderSide.SELL,
                    order_type=OrderType.MARKET,
                    qty=close_qty,
                    strategy_id=self._strategy.strategy_id,
                )
                order, fill = await self._adapter.place_order(req)
                if fill:
                    await self._persist_order_fill(order, fill)

                    # Calculate PnL for the closed portion
                    entry_price = self._open_trade["entry_price"]
                    gross_pnl = (fill.price - entry_price) * close_qty
                    pct_of_original = close_qty / self._open_trade["original_qty"]
                    fee_in_portion = self._open_trade["fee_in"] * pct_of_original
                    net_pnl = gross_pnl - fee_in_portion - fill.fee

                    # Accumulate partial PnL
                    self._open_trade["total_partial_pnl"] += net_pnl
                    self._open_trade["total_partial_fees"] += fee_in_portion + fill.fee
                    self._open_trade["qty"] -= close_qty

                    logger.info(
                        "partial_close",
                        symbol=bar.symbol,
                        close_pct=close_pct,
                        closed_qty=str(close_qty),
                        remaining_qty=str(self._open_trade["qty"]),
                        partial_pnl=str(net_pnl),
                        trade_id=self._open_trade["trade_id"],
                    )

                    # If qty exhausted (shouldn't happen normally, but safety check)
                    if self._open_trade["qty"] < Decimal("0.0001"):
                        await self._close_trade_final(fill)

        elif signal.action in (SignalAction.SELL, SignalAction.CLOSE) and self._open_trade:
            # ── FULL CLOSE of remaining position ───────────────────
            qty = self._open_trade["qty"]
            req = OrderRequest(
                symbol=bar.symbol,
                side=OrderSide.SELL,
                order_type=OrderType.MARKET,
                qty=qty,
                strategy_id=self._strategy.strategy_id,
            )
            order, fill = await self._adapter.place_order(req)
            if fill:
                await self._persist_order_fill(order, fill)
                await self._close_trade_final(fill)

    async def _close_trade_final(self, fill: FillResult) -> None:
        """
        Close the trade, computing total PnL including any partial closes.
        """
        if not self._open_trade:
            return

        entry_price = self._open_trade["entry_price"]
        closed_qty = fill.qty
        gross_pnl = (fill.price - entry_price) * closed_qty

        remaining_pct = closed_qty / self._open_trade["original_qty"]
        fee_in_portion = self._open_trade["fee_in"] * remaining_pct
        net_pnl_this_leg = gross_pnl - fee_in_portion - fill.fee

        total_net_pnl = self._open_trade["total_partial_pnl"] + net_pnl_this_leg
        total_fees = self._open_trade["total_partial_fees"] + fee_in_portion + fill.fee

        # Persist to DB
        trade = Trade(
            id=self._open_trade["trade_id"],
            symbol=self._open_trade["symbol"],
            side=self._open_trade["side"],
            entry_ts=self._open_trade["entry_ts"],
            entry_price=self._open_trade["entry_price"],
            entry_order_id=self._open_trade.get("entry_order_id"),
            exit_ts=fill.timestamp,
            exit_price=fill.price,
            exit_order_id=fill.order_id,
            qty=self._open_trade["original_qty"],
            fees=total_fees,
            pnl=total_net_pnl,
            pnl_pct=total_net_pnl / (entry_price * self._open_trade["original_qty"]) * 100,
            mode=self._mode,
            strategy_id=self._strategy.strategy_id,
        )
        self._session.add(trade)
        await self._session.commit()

        # Update metrics
        try:
            from app.metrics.prometheus import (
                TRADES_TOTAL, TRADE_PNL, PORTFOLIO_EQUITY, DAILY_PNL,
            )
            TRADES_TOTAL.labels(symbol=trade.symbol, mode=self._mode).inc()
            TRADE_PNL.labels(symbol=trade.symbol, mode=self._mode).observe(float(total_net_pnl))
        except Exception:
            pass

        self._risk.record_fill(total_net_pnl)

        logger.info(
            "trade_closed",
            trade_id=trade.id,
            symbol=trade.symbol,
            entry=str(trade.entry_price),
            exit=str(trade.exit_price),
            pnl=str(total_net_pnl),
        )
        self._open_trade = None

    async def _persist_order_fill(self, order: OrderResult, fill: FillResult) -> None:
        """Save Order and Fill to DB."""
        db_order = Order(
            order_id=order.order_id,
            broker_id=order.broker_id,
            symbol=order.symbol,
            side=order.side.value,
            order_type=order.order_type.value,
            qty=order.qty,
            price=order.price,
            status=order.status.value,
            strategy_id=order.extra.get("strategy_id"),
            mode=self._mode,
            created_at=order.created_at,
        )
        db_fill = Fill(
            fill_id=fill.fill_id,
            order_id=fill.order_id,
            symbol=fill.symbol,
            side=fill.side.value,
            price=fill.price,
            qty=fill.qty,
            fee=fill.fee,
            fee_currency=fill.fee_currency,
            timestamp=fill.timestamp,
        )
        self._session.add(db_order)
        self._session.add(db_fill)
        await self._session.commit()

    async def _heartbeat(self) -> None:
        """Update bot heartbeat in DB."""
        try:
            from sqlalchemy import update
            await self._session.execute(
                update(BotState)
                .where(BotState.id == 1)
                .values(last_heartbeat=datetime.now(timezone.utc))
            )
            await self._session.commit()
        except Exception:
            pass

    async def _update_bot_state(self, is_running: bool) -> None:
        """Update bot running state in DB."""
        try:
            from sqlalchemy import update
            await self._session.execute(
                update(BotState)
                .where(BotState.id == 1)
                .values(is_running=is_running)
            )
            await self._session.commit()
        except Exception:
            pass
