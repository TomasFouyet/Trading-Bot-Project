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

PARTIAL_CLOSE support:
  - When signal.action == PARTIAL_CLOSE, close signal.close_pct of position
  - Remaining qty stays in open_trade with updated fees
  - Final CLOSE closes whatever remains
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
    bar_data: list[dict] = field(default_factory=list)  # verbose mode only

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
    Deterministic single-symbol backtest engine.

    Supports BUY, SELL, CLOSE, and PARTIAL_CLOSE signals.
    PARTIAL_CLOSE closes a fraction of the position (specified in signal.meta["close_pct"])
    and records a partial trade. The remaining position continues with the original entry price.
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
            run_id=run_id,
            symbol=symbol,
            timeframe=timeframe,
            htf_timeframe=htf_timeframe,
            start=start.isoformat(),
            end=end.isoformat(),
        )

        # Load micro-timeframe bars
        bars = self._store.read_bars(symbol, timeframe, start, end)
        if len(bars) < self._strategy.min_bars_required + 10:
            raise ValueError(
                f"Not enough bars: {len(bars)} < {self._strategy.min_bars_required + 10}"
            )

        # Load higher-timeframe bars for trend filter (if strategy supports it)
        _strategy_has_htf = hasattr(self._strategy, "set_htf_bars")
        _htf_df: pd.DataFrame | None = None
        if _strategy_has_htf:
            # Load extra bars before `start` for indicator warm-up (EMA200 needs ~200 HTF bars)
            from datetime import timedelta
            _htf_warmup_days = 250  # enough for EMA200 on any HTF
            htf_bars_raw = self._store.read_bars(
                symbol,
                htf_timeframe,
                start - timedelta(days=_htf_warmup_days),
                end,
            )
            if htf_bars_raw:
                _htf_df = pd.DataFrame(
                    {
                        "ts":     pd.to_datetime([b.ts for b in htf_bars_raw], utc=True),
                        "open":   [float(b.open)   for b in htf_bars_raw],
                        "high":   [float(b.high)   for b in htf_bars_raw],
                        "low":    [float(b.low)    for b in htf_bars_raw],
                        "close":  [float(b.close)  for b in htf_bars_raw],
                        "volume": [float(b.volume) for b in htf_bars_raw],
                    }
                ).sort_values("ts").reset_index(drop=True)
                logger.info(
                    "htf_bars_loaded",
                    symbol=symbol,
                    htf_timeframe=htf_timeframe,
                    bars=len(_htf_df),
                )
            else:
                logger.warning(
                    "htf_bars_not_found",
                    symbol=symbol,
                    htf_timeframe=htf_timeframe,
                    message="Trend filter disabled — no HTF data available",
                )

        adapter = BacktestAdapter(
            bars=bars,
            initial_balance=self._initial_balance,
            commission_rate=self._commission,
            slippage_rate=self._slippage,
        )
        self._risk.initialize(self._initial_balance, as_of_date=bars[0].ts.date())

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
        partial_close_order_id: str | None = None   # NEW: track partial close orders
        partial_close_pct: float = 0.0               # NEW: % being closed
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
                        "original_qty": fill.qty,   # NEW: track original size
                        "fee_in": fill.fee,
                        "total_partial_pnl": Decimal("0"),  # NEW: accumulate partial PnL
                        "total_partial_fees": Decimal("0"),  # NEW: accumulate partial fees
                        "mode": "backtest",
                    }
                    pending_order_id = None
                    logger.debug("backtest_entry", price=str(fill.price), ts=fill.timestamp.isoformat())

                elif partial_close_order_id and fill.order_id == partial_close_order_id:
                    # ── PARTIAL CLOSE fill ─────────────────────────────
                    if open_trade:
                        exit_price = fill.price
                        entry_price = open_trade["entry_price"]
                        closed_qty = fill.qty
                        gross_pnl = (exit_price - entry_price) * closed_qty
                        fee_out = fill.fee
                        # Proportional entry fee for the closed portion
                        pct_closed_of_original = closed_qty / open_trade["original_qty"]
                        fee_in_portion = open_trade["fee_in"] * pct_closed_of_original
                        net_pnl = gross_pnl - fee_in_portion - fee_out

                        # Accumulate partial PnL
                        open_trade["total_partial_pnl"] += net_pnl
                        open_trade["total_partial_fees"] += fee_in_portion + fee_out

                        # Reduce remaining qty
                        open_trade["qty"] -= closed_qty
                        if open_trade["qty"] < Decimal("0.0001"):
                            # Fully closed via partials (edge case: close_pct summed to 100%)
                            self._record_completed_trade(
                                result, open_trade, exit_price, fill.timestamp,
                                fee_out, is_final_partial=True,
                            )
                            open_trade = None
                        else:
                            logger.debug(
                                "backtest_partial_close",
                                closed_qty=str(closed_qty),
                                remaining_qty=str(open_trade["qty"]),
                                partial_pnl=str(net_pnl),
                                ts=fill.timestamp.isoformat(),
                            )

                    partial_close_order_id = None
                    partial_close_pct = 0.0

                elif close_order_id and fill.order_id == close_order_id:
                    # ── FULL CLOSE fill ────────────────────────────────
                    if open_trade:
                        exit_price = fill.price
                        entry_price = open_trade["entry_price"]
                        qty = fill.qty
                        gross_pnl = (exit_price - entry_price) * qty

                        # Fee for the remaining portion
                        remaining_pct = qty / open_trade["original_qty"]
                        fee_in_portion = open_trade["fee_in"] * remaining_pct
                        fee_out = fill.fee
                        net_pnl_this_leg = gross_pnl - fee_in_portion - fee_out

                        # Total PnL = partial legs + final leg
                        total_net_pnl = open_trade["total_partial_pnl"] + net_pnl_this_leg
                        total_fees = open_trade["total_partial_fees"] + fee_in_portion + fee_out

                        trade_record = {
                            "trade_id": open_trade["trade_id"],
                            "symbol": open_trade["symbol"],
                            "side": open_trade["side"],
                            "entry_ts": open_trade["entry_ts"],
                            "entry_price": open_trade["entry_price"],
                            "qty": open_trade["original_qty"],  # Report original full qty
                            "fee_in": open_trade["fee_in"],
                            "mode": open_trade["mode"],
                            "exit_ts": fill.timestamp,
                            "exit_price": exit_price,
                            "pnl": total_net_pnl,
                            "pnl_pct": total_net_pnl / (entry_price * open_trade["original_qty"]) * 100,
                            "fees": total_fees,
                            "had_partial_close": open_trade["total_partial_pnl"] != Decimal("0"),
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

            # 2b. Feed HTF bars to trend filter (no lookahead: only bars closed before current bar)
            if _strategy_has_htf and _htf_df is not None:
                cur_ts = pd.Timestamp(bar.ts).tz_convert("UTC") if bar.ts.tzinfo else pd.Timestamp(bar.ts, tz="UTC")
                htf_window = _htf_df[_htf_df["ts"] < cur_ts].tail(300)
                if len(htf_window) > 0:
                    self._strategy.set_htf_bars(htf_window)

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

            if self._verbose:
                result.bar_data.append({
                    "ts": bar.ts.isoformat(),
                    "o": float(bar.open), "h": float(bar.high),
                    "l": float(bar.low), "c": float(bar.close),
                    "v": float(bar.volume),
                    "signal": signal.action.value,
                    "reason": signal.reason,
                    "meta": signal.meta,
                })

            # 4. Risk validation — use mark-to-market equity
            # cash already reflects position cost deducted (LONG) or proceeds received (SHORT)
            # MTM = cash + qty*close (LONG) or cash - qty*close (SHORT)
            balances = await adapter.get_balance()
            cash = balances[0].total if balances else self._initial_balance
            equity = cash
            if open_trade:
                pos_qty = open_trade["qty"]
                is_long = open_trade["side"] == TradeSide.LONG.value
                if is_long:
                    equity += pos_qty * bar.close
                else:  # SHORT: subtract current liability
                    equity -= pos_qty * bar.close
            kill_switch_active = False
            try:
                self._risk.update_equity(equity, as_of_date=bar.ts.date())
            except KillSwitchTriggered:
                kill_switch_active = True

            if kill_switch_active:
                # Daily drawdown limit — allow CLOSE/PARTIAL_CLOSE to protect existing positions,
                # but skip new entries.
                if signal.action not in (SignalAction.CLOSE, SignalAction.PARTIAL_CLOSE):
                    result.equity_curve.append({"ts": bar.ts, "equity": equity})
                    continue
                # Fall through so the close signal gets executed below

            if not kill_switch_active:
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

            elif signal.action == SignalAction.PARTIAL_CLOSE and open_trade is not None and partial_close_order_id is None:
                # ── NEW: PARTIAL CLOSE ─────────────────────────────────
                close_pct = signal.close_pct
                close_qty = (open_trade["qty"] * Decimal(str(close_pct))).quantize(Decimal("0.001"))
                close_qty = min(close_qty, open_trade["qty"])  # Safety: never close more than we have

                if close_qty > Decimal("0"):
                    req = OrderRequest(
                        symbol=symbol,
                        side=OrderSide.SELL,
                        order_type=OrderType.MARKET,
                        qty=close_qty,
                        strategy_id=self._strategy.strategy_id,
                    )
                    order, _ = await adapter.place_order(req)
                    partial_close_order_id = order.order_id
                    partial_close_pct = close_pct
                    logger.debug(
                        "backtest_partial_close_ordered",
                        close_pct=close_pct,
                        close_qty=str(close_qty),
                        remaining_qty=str(open_trade["qty"] - close_qty),
                    )

            elif signal.action in (SignalAction.SELL, SignalAction.CLOSE) and open_trade is not None:
                # Full close of remaining position
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

            # 6. Equity curve (reuse the MTM equity already computed above)
            result.equity_curve.append({"ts": bar.ts, "equity": equity})

        # ── Close any open position at end ────────────────────────────
        if open_trade:
            last_bar = bars[-1]
            exit_price = last_bar.close
            entry_price = open_trade["entry_price"]
            qty = open_trade["qty"]
            gross_pnl = (exit_price - entry_price) * qty

            remaining_pct = qty / open_trade["original_qty"]
            fee_in_portion = open_trade["fee_in"] * remaining_pct
            fee_out = exit_price * qty * self._commission
            net_pnl_this_leg = gross_pnl - fee_in_portion - fee_out

            total_net_pnl = open_trade["total_partial_pnl"] + net_pnl_this_leg
            total_fees = open_trade["total_partial_fees"] + fee_in_portion + fee_out

            trade_record = {
                "trade_id": open_trade["trade_id"],
                "symbol": open_trade["symbol"],
                "side": open_trade["side"],
                "entry_ts": open_trade["entry_ts"],
                "entry_price": open_trade["entry_price"],
                "qty": open_trade["original_qty"],
                "fee_in": open_trade["fee_in"],
                "mode": open_trade["mode"],
                "exit_ts": last_bar.ts,
                "exit_price": exit_price,
                "pnl": total_net_pnl,
                "pnl_pct": total_net_pnl / (entry_price * open_trade["original_qty"]) * 100,
                "fees": total_fees,
                "had_partial_close": open_trade["total_partial_pnl"] != Decimal("0"),
                "forced_close": True,
            }
            result.trades.append(
                {k: str(v) if isinstance(v, Decimal) else v for k, v in trade_record.items()}
            )
            result.total_trades += 1
            if total_net_pnl > 0:
                result.winning_trades += 1
            else:
                result.losing_trades += 1
            result.total_pnl += total_net_pnl
            open_trade = None

        # ── Compute summary metrics ───────────────────────────────────
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

        # Max drawdown
        if result.equity_curve:
            equities = [float(e["equity"]) for e in result.equity_curve]
            peak = equities[0]
            max_dd = 0.0
            for eq in equities:
                peak = max(peak, eq)
                dd = (peak - eq) / peak * 100 if peak > 0 else 0
                max_dd = max(max_dd, dd)
            result.max_drawdown_pct = Decimal(str(round(max_dd, 2)))

        # Sharpe ratio (annualized, 5m bars)
        if result.trades:
            pnls = [float(t["pnl"]) if isinstance(t["pnl"], str) else float(t["pnl"])
                    for t in result.trades]
            if len(pnls) >= 2 and np.std(pnls) > 0:
                sharpe = np.mean(pnls) / np.std(pnls) * np.sqrt(len(pnls))
                result.sharpe_ratio = Decimal(str(round(sharpe, 3)))

        logger.info(
            "backtest_complete",
            run_id=run_id,
            total_trades=result.total_trades,
            total_pnl=str(result.total_pnl),
            winrate=str(result.winrate),
            max_dd=str(result.max_drawdown_pct),
        )
        return result

    def _record_completed_trade(
        self,
        result: BacktestResult,
        open_trade: dict,
        final_exit_price: Decimal,
        exit_ts: datetime,
        final_fee_out: Decimal,
        is_final_partial: bool = False,
    ) -> None:
        """Record a trade that was fully closed via accumulated partial closes."""
        entry_price = open_trade["entry_price"]
        total_net_pnl = open_trade["total_partial_pnl"]
        total_fees = open_trade["total_partial_fees"]

        trade_record = {
            "trade_id": open_trade["trade_id"],
            "symbol": open_trade["symbol"],
            "side": open_trade["side"],
            "entry_ts": open_trade["entry_ts"],
            "entry_price": entry_price,
            "qty": open_trade["original_qty"],
            "fee_in": open_trade["fee_in"],
            "mode": open_trade["mode"],
            "exit_ts": exit_ts,
            "exit_price": final_exit_price,
            "pnl": total_net_pnl,
            "pnl_pct": total_net_pnl / (entry_price * open_trade["original_qty"]) * 100,
            "fees": total_fees,
            "had_partial_close": True,
        }
        result.trades.append(
            {k: str(v) if isinstance(v, Decimal) else v for k, v in trade_record.items()}
        )
        result.total_trades += 1
        if total_net_pnl > 0:
            result.winning_trades += 1
        else:
            result.losing_trades += 1
        result.total_pnl += total_net_pnl
        self._risk.record_fill(total_net_pnl)
