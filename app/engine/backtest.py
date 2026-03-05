"""
Backtest engine.

Flow:
  1. Load bars from Parquet
  2. Warm up strategy (min_bars_required)
  3. Iterate bar by bar:
     a. adapter.advance(bar) → fills pending orders
     b. strategy.on_bar(window) → Signal
     c. risk.validate_signal()
     d. if approved: adapter.place_order() → queued for next bar fill
     e. record fills, update equity, check risk
  4. Close any open positions at end
  5. Compute metrics, save BacktestRun to DB

Market orders are filled on the NEXT bar's open (realistic simulation).
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
from app.core.logging import get_logger
from app.data.parquet_store import ParquetStore
from app.risk.manager import RiskManager
from app.strategy.base import BaseStrategy
from app.strategy.signals import Signal, SignalAction

logger = get_logger(__name__)
_settings = get_settings()

WINDOW_SIZE = 250  # Number of bars to pass to strategy at each step


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

    def to_report(self) -> dict:
        return {
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


class BacktestEngine:
    """
    Deterministic single-symbol backtest engine.
    """

    def __init__(
        self,
        strategy: BaseStrategy,
        store: ParquetStore,
        initial_balance: Decimal = Decimal("10000"),
        commission_rate: Decimal | None = None,
        slippage_rate: Decimal | None = None,
    ) -> None:
        self._strategy = strategy
        self._store = store
        self._initial_balance = initial_balance
        self._commission = commission_rate or _settings.commission_rate
        self._slippage = slippage_rate or _settings.slippage_rate
        self._risk = RiskManager()

    async def run(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
        params: dict | None = None,
    ) -> BacktestResult:
        run_id = str(uuid.uuid4())
        logger.info(
            "backtest_start",
            run_id=run_id,
            symbol=symbol,
            timeframe=timeframe,
            start=start.isoformat(),
            end=end.isoformat(),
        )

        # Load bars
        bars = self._store.read_bars(symbol, timeframe, start, end)
        if len(bars) < self._strategy.min_bars_required + 10:
            raise ValueError(
                f"Not enough bars: {len(bars)} < {self._strategy.min_bars_required + 10}"
            )

        adapter = BacktestAdapter(
            bars=bars,
            initial_balance=self._initial_balance,
            commission_rate=self._commission,
            slippage_rate=self._slippage,
        )
        self._risk.initialize(self._initial_balance)

        result = BacktestResult(
            run_id=run_id,
            symbol=symbol,
            timeframe=timeframe,
            strategy_id=self._strategy.strategy_id,
            start_date=start,
            end_date=end,
            params=params or self._strategy.params,
        )

        # ── Open trade tracker ─────────────────────────────────────────
        open_trade: dict | None = None
        pending_order_id: str | None = None
        close_order_id: str | None = None
        bars_in_position = 0
        total_bars = len(bars)

        # ── Main loop ─────────────────────────────────────────────────
        for i, bar in enumerate(bars):
            # 1. Advance adapter (fills pending orders at this bar's open)
            fills = adapter.advance(bar)

            for fill in fills:
                if pending_order_id and fill.order_id == pending_order_id:
                    # Entry fill
                    open_trade = {
                        "trade_id": str(uuid.uuid4()),
                        "symbol": symbol,
                        "side": TradeSide.LONG.value,
                        "entry_ts": fill.timestamp,
                        "entry_price": fill.price,
                        "qty": fill.qty,
                        "fee_in": fill.fee,
                        "mode": "backtest",
                    }
                    pending_order_id = None
                    logger.debug("backtest_entry", price=str(fill.price), ts=fill.timestamp.isoformat())

                elif close_order_id and fill.order_id == close_order_id:
                    # Exit fill
                    if open_trade:
                        exit_price = fill.price
                        entry_price = open_trade["entry_price"]
                        qty = open_trade["qty"]
                        gross_pnl = (exit_price - entry_price) * qty
                        fees = open_trade["fee_in"] + fill.fee
                        net_pnl = gross_pnl - fees
                        pnl_pct = net_pnl / (entry_price * qty) * 100

                        trade_record = {
                            **open_trade,
                            "exit_ts": fill.timestamp,
                            "exit_price": exit_price,
                            "pnl": net_pnl,
                            "pnl_pct": pnl_pct,
                            "fees": fees,
                        }
                        result.trades.append(
                            {k: str(v) if isinstance(v, Decimal) else v
                             for k, v in trade_record.items()}
                        )
                        result.total_trades += 1
                        if net_pnl > 0:
                            result.winning_trades += 1
                        else:
                            result.losing_trades += 1
                        result.total_pnl += net_pnl
                        self._risk.record_fill(net_pnl)
                        open_trade = None
                    close_order_id = None

            if open_trade:
                bars_in_position += 1

            # 2. Build window for strategy
            window_start = max(0, i + 1 - WINDOW_SIZE)
            window = bars[window_start : i + 1]
            if len(window) < 2:
                continue

            df = BaseStrategy.bars_to_df(window)

            # 3. Get signal
            signal = self._strategy.on_bar(df)
            if signal.is_actionable():
                result.signals.append(
                    {
                        "ts": bar.ts.isoformat(),
                        "action": signal.action.value,
                        "reason": signal.reason,
                    }
                )

            # 4. Risk validation
            balances = await adapter.get_balance()
            equity = balances[0].total if balances else self._initial_balance
            self._risk.update_equity(equity)

            try:
                approved, reason = self._risk.validate_signal(signal, equity)
            except Exception as e:
                logger.warning("backtest_risk_error", error=str(e))
                break

            if not approved:
                continue

            # 5. Execute signal
            if signal.action == SignalAction.BUY and open_trade is None and pending_order_id is None:
                qty = self._risk.compute_order_qty(signal, equity, bar.close)
                req = OrderRequest(
                    symbol=symbol,
                    side=OrderSide.BUY,
                    order_type=OrderType.MARKET,
                    qty=qty,
                    strategy_id=self._strategy.strategy_id,
                )
                order, _ = await adapter.place_order(req)
                pending_order_id = order.order_id

            elif signal.action in (SignalAction.SELL, SignalAction.CLOSE) and open_trade is not None:
                qty = open_trade["qty"]
                req = OrderRequest(
                    symbol=symbol,
                    side=OrderSide.SELL,
                    order_type=OrderType.MARKET,
                    qty=qty,
                    strategy_id=self._strategy.strategy_id,
                )
                order, _ = await adapter.place_order(req)
                close_order_id = order.order_id

            # 6. Record equity
            result.equity_curve.append({"ts": bar.ts, "equity": equity})

        # ── Close open position at end ─────────────────────────────────
        if open_trade is not None:
            last_bar = bars[-1]
            close_price = last_bar.close * (1 - self._slippage)
            fee = close_price * open_trade["qty"] * self._commission
            net_pnl = (close_price - open_trade["entry_price"]) * open_trade["qty"] - fee - open_trade["fee_in"]
            result.trades.append(
                {
                    **{k: str(v) if isinstance(v, Decimal) else v for k, v in open_trade.items()},
                    "exit_ts": last_bar.ts.isoformat(),
                    "exit_price": str(close_price),
                    "pnl": str(net_pnl),
                    "pnl_pct": str(net_pnl / (open_trade["entry_price"] * open_trade["qty"]) * 100),
                    "fees": str(fee),
                    "forced_close": True,
                }
            )
            result.total_trades += 1
            if net_pnl > 0:
                result.winning_trades += 1
            else:
                result.losing_trades += 1
            result.total_pnl += net_pnl

        # ── Compute metrics ────────────────────────────────────────────
        result.total_bars = total_bars
        result.total_pnl_pct = (result.total_pnl / self._initial_balance * 100).quantize(
            Decimal("0.01")
        )
        if result.total_trades > 0:
            result.winrate = Decimal(result.winning_trades) / Decimal(result.total_trades) * 100
            result.avg_trade_pnl = result.total_pnl / Decimal(result.total_trades)

        if total_bars > 0:
            result.exposure_pct = Decimal(bars_in_position) / Decimal(total_bars) * 100

        result.max_drawdown_pct = self._compute_max_drawdown(result.equity_curve)
        result.sharpe_ratio = self._compute_sharpe(result.equity_curve)

        logger.info(
            "backtest_done",
            run_id=run_id,
            trades=result.total_trades,
            pnl_pct=str(result.total_pnl_pct),
            winrate=str(result.winrate),
            max_dd=str(result.max_drawdown_pct),
            sharpe=str(result.sharpe_ratio),
        )
        return result

    @staticmethod
    def _compute_max_drawdown(equity_curve: list[dict]) -> Decimal:
        if not equity_curve:
            return Decimal("0")
        equities = [float(e["equity"]) for e in equity_curve]
        peak = equities[0]
        max_dd = 0.0
        for eq in equities:
            if eq > peak:
                peak = eq
            dd = (peak - eq) / peak * 100 if peak > 0 else 0
            max_dd = max(max_dd, dd)
        return Decimal(str(round(max_dd, 2)))

    @staticmethod
    def _compute_sharpe(equity_curve: list[dict], periods_per_year: int = 252) -> Decimal:
        """Simplified Sharpe ratio (annualized, assuming 0% risk-free rate)."""
        if len(equity_curve) < 2:
            return Decimal("0")
        equities = [float(e["equity"]) for e in equity_curve]
        returns = np.diff(equities) / np.array(equities[:-1])
        if returns.std() == 0:
            return Decimal("0")
        sharpe = (returns.mean() / returns.std()) * np.sqrt(periods_per_year)
        return Decimal(str(round(sharpe, 3)))
