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
        await self._adapter.initialize()

        # Get initial balance for risk manager
        balances = await self._adapter.get_balance()
        initial_equity = balances[0].total if balances else Decimal("10000")
        self._risk.initialize(initial_equity)

        logger.info(
            "forward_engine_start",
            mode=self._mode,
            symbol=_settings.default_symbol,
            strategy=self._strategy.strategy_id,
            initial_equity=str(initial_equity),
            trace_id=self._trace_id,
        )

        self._running = True
        try:
            async for bar in self._feed.stream():
                if not self._running:
                    break
                await self._process_bar(bar)
        except KillSwitchTriggered as e:
            logger.error("kill_switch_stop", reason=str(e), trace_id=self._trace_id)
            await self._update_bot_state(is_running=False, kill_reason=str(e))
        except asyncio.CancelledError:
            logger.info("engine_cancelled", trace_id=self._trace_id)
        except Exception as e:
            logger.error("engine_error", error=str(e), trace_id=self._trace_id)
            raise
        finally:
            self._running = False
            await self._adapter.shutdown()
            logger.info("forward_engine_stopped", trace_id=self._trace_id)

    async def _process_bar(self, bar: OHLCVBar) -> None:
        self._bar_window.append(bar)
        self._risk.on_api_success()

        # Update equity
        balances = await self._adapter.get_balance()
        equity = balances[0].total if balances else Decimal("0")
        self._risk.update_equity(equity)

        # Need enough bars for strategy
        if len(self._bar_window) < self._strategy.min_bars_required:
            logger.debug(
                "warming_up",
                bars=len(self._bar_window),
                needed=self._strategy.min_bars_required,
            )
            await self._heartbeat()
            return

        df = BaseStrategy.bars_to_df(list(self._bar_window))
        signal = self._strategy.on_bar(df)

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
                    "fee_in": fill.fee,
                }
                logger.info(
                    "trade_opened",
                    symbol=bar.symbol,
                    qty=str(qty),
                    price=str(fill.price),
                    trade_id=self._open_trade["trade_id"],
                )

        elif signal.action in (SignalAction.SELL, SignalAction.CLOSE) and self._open_trade:
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
                await self._close_trade(fill)

    async def _persist_order_fill(self, order: OrderResult, fill: FillResult) -> None:
        """Save Order and Fill to DB."""
        db_order = Order(
            id=order.order_id,
            broker_id=order.broker_id,
            symbol=order.symbol,
            side=order.side.value,
            order_type=order.order_type.value,
            qty=order.qty,
            price=order.price,
            status=order.status.value,
            mode=self._mode,
            strategy_id=order.extra.get("strategy_id") if order.extra else None,
            trace_id=self._trace_id,
        )
        db_fill = Fill(
            id=fill.fill_id,
            order_id=fill.order_id,
            symbol=fill.symbol,
            side=fill.side.value,
            price=fill.price,
            qty=fill.qty,
            fee=fill.fee,
            fee_currency=fill.fee_currency,
            ts=fill.timestamp,
            mode=self._mode,
        )
        self._session.add(db_order)
        self._session.add(db_fill)
        await self._session.flush()

    async def _close_trade(self, exit_fill: FillResult) -> None:
        if self._open_trade is None:
            return

        entry = self._open_trade
        gross_pnl = (exit_fill.price - entry["entry_price"]) * entry["qty"]
        fees = entry["fee_in"] + exit_fill.fee
        net_pnl = gross_pnl - fees
        pnl_pct = net_pnl / (entry["entry_price"] * entry["qty"]) * 100

        trade = Trade(
            id=entry["trade_id"],
            symbol=entry["symbol"],
            side=entry["side"],
            mode=self._mode,
            strategy_id=self._strategy.strategy_id,
            entry_order_id=entry["entry_order_id"],
            exit_order_id=exit_fill.order_id,
            entry_ts=entry["entry_ts"],
            exit_ts=exit_fill.timestamp,
            entry_price=entry["entry_price"],
            exit_price=exit_fill.price,
            qty=entry["qty"],
            pnl=net_pnl,
            pnl_pct=pnl_pct,
            fees=fees,
            is_open=False,
        )
        self._session.add(trade)
        await self._session.flush()

        self._risk.record_fill(net_pnl)
        self._open_trade = None

        logger.info(
            "trade_closed",
            symbol=entry["symbol"],
            entry_price=str(entry["entry_price"]),
            exit_price=str(exit_fill.price),
            pnl=str(net_pnl),
            pnl_pct=str(pnl_pct.quantize(Decimal("0.01"))),
        )

        # Prometheus
        try:
            from app.metrics.prometheus import TRADE_PNL, TRADES_TOTAL
            TRADES_TOTAL.labels(symbol=entry["symbol"], mode=self._mode).inc()
            TRADE_PNL.labels(symbol=entry["symbol"], mode=self._mode).observe(float(net_pnl))
        except Exception:
            pass

    async def _heartbeat(self) -> None:
        """Update bot_state.last_heartbeat."""
        try:
            from sqlalchemy import update
            from app.db.models import BotState
            await self._session.execute(
                update(BotState)
                .where(BotState.id == 1)
                .values(last_heartbeat=datetime.now(timezone.utc))
            )
            await self._session.flush()
        except Exception:
            pass

    async def _update_bot_state(self, is_running: bool, kill_reason: str = "") -> None:
        try:
            from sqlalchemy import update
            await self._session.execute(
                update(BotState)
                .where(BotState.id == 1)
                .values(
                    is_running=is_running,
                    kill_switch_reason=kill_reason if kill_reason else None,
                )
            )
            await self._session.flush()
        except Exception:
            pass

    def stop(self) -> None:
        self._running = False
        self._feed.stop()
