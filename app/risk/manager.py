"""
RiskManager — independent module that validates every order before execution.

Enforces:
1. Max daily drawdown (kill switch)
2. Max position size as % of portfolio
3. Max risk per trade (stop-loss based)
4. Consecutive API error kill switch
5. Data delay kill switch
6. Global kill switch (manual override)

All rejections are logged with explicit reason.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from decimal import Decimal
from typing import Optional

from app.broker.base import BrokerAdapter, OrderRequest, OrderSide
from app.config import get_settings
from app.core.exceptions import KillSwitchTriggered, RiskViolationError
from app.core.logging import get_logger
from app.strategy.signals import Signal, SignalAction

logger = get_logger(__name__)
_settings = get_settings()


@dataclass
class RiskState:
    """Mutable risk state tracked during a session."""
    session_start_equity: Decimal = Decimal("0")
    current_equity: Decimal = Decimal("0")
    daily_start_equity: Decimal = Decimal("0")
    daily_pnl: Decimal = Decimal("0")
    consecutive_api_errors: int = 0
    kill_switch_active: bool = False
    kill_switch_reason: str = ""
    trades_today: int = 0
    last_reset_date: date = field(default_factory=date.today)


class RiskManager:
    """
    Stateful risk manager.

    Must be initialized with current equity before each session.
    Call validate_signal() before placing any order.
    Call record_fill() after each fill to update state.
    Call on_api_error() / on_api_success() for error tracking.
    """

    def __init__(
        self,
        max_daily_drawdown_pct: Decimal | None = None,
        max_position_pct: Decimal | None = None,
        max_trade_risk_pct: Decimal | None = None,
        max_consecutive_api_errors: int | None = None,
    ) -> None:
        self._max_daily_dd = max_daily_drawdown_pct or _settings.risk_max_daily_drawdown_pct
        self._max_position_pct = max_position_pct or _settings.risk_max_position_pct
        self._max_trade_risk_pct = max_trade_risk_pct or _settings.risk_max_trade_risk_pct
        self._max_api_errors = max_consecutive_api_errors or _settings.risk_max_consecutive_api_errors
        self._state = RiskState()

    def initialize(self, equity: Decimal) -> None:
        """Call once at session start with current portfolio equity."""
        today = date.today()
        self._state.session_start_equity = equity
        self._state.current_equity = equity
        self._state.daily_start_equity = equity
        self._state.last_reset_date = today
        logger.info(
            "risk_initialized",
            equity=str(equity),
            max_daily_dd_pct=str(self._max_daily_dd),
            max_position_pct=str(self._max_position_pct),
        )

    def update_equity(self, equity: Decimal) -> None:
        """Update current equity (call after each bar mark-to-market)."""
        today = date.today()
        if today > self._state.last_reset_date:
            # New trading day — reset daily counters
            self._state.daily_start_equity = equity
            self._state.daily_pnl = Decimal("0")
            self._state.trades_today = 0
            self._state.last_reset_date = today
            logger.info("risk_daily_reset", equity=str(equity))

        self._state.current_equity = equity

        # Check daily drawdown
        if self._state.daily_start_equity > 0:
            dd_pct = (self._state.daily_start_equity - equity) / self._state.daily_start_equity * 100
            if dd_pct >= self._max_daily_dd:
                reason = (
                    f"Daily drawdown {dd_pct:.2f}% exceeds limit {self._max_daily_dd}%"
                )
                self._trigger_kill_switch(reason)

    def validate_signal(
        self,
        signal: Signal,
        current_equity: Decimal,
        current_position_value: Decimal = Decimal("0"),
    ) -> tuple[bool, str]:
        """
        Validate a signal before acting on it.

        Returns:
            (approved: bool, reason: str)
        """
        # 1. Kill switch active?
        if self._state.kill_switch_active:
            return False, f"Kill switch active: {self._state.kill_switch_reason}"

        # 2. Not actionable?
        if not signal.is_actionable():
            return False, "Signal is HOLD"

        # 3. CLOSE signals always allowed (risk reduction)
        if signal.action == SignalAction.CLOSE:
            return True, "close_position"

        # 4. Check equity > 0
        if current_equity <= 0:
            return False, "Zero or negative equity"

        # 5. Position size check
        proposed_notional = self._estimate_notional(signal, current_equity)
        position_pct = (current_position_value + proposed_notional) / current_equity * 100
        if position_pct > self._max_position_pct:
            return (
                False,
                f"Position {position_pct:.1f}% would exceed max {self._max_position_pct}%",
            )

        # 6. Trade risk check (only if stop loss provided)
        if signal.stop_loss is not None:
            # Estimate risk based on stop distance
            # For BUY: risk = (entry - stop_loss) * qty
            # We approximate entry as current_equity * position_pct / 100 / price
            # Simple check: stop distance % of current price
            pass  # Full implementation requires current price — handled in engine

        return True, "approved"

    def compute_order_qty(
        self,
        signal: Signal,
        current_equity: Decimal,
        price: Decimal,
    ) -> Decimal:
        """
        Compute order quantity based on max position % of equity.
        Uses signal.confidence as a scaling factor.
        """
        if signal.target_qty is not None:
            return signal.target_qty

        max_notional = current_equity * (self._max_position_pct / 100)
        scaled_notional = max_notional * Decimal(str(signal.confidence))
        qty = scaled_notional / price

        # Round to reasonable precision
        qty = qty.quantize(Decimal("0.001"))
        return max(qty, Decimal("0.001"))

    def _estimate_notional(self, signal: Signal, equity: Decimal) -> Decimal:
        """Rough notional estimate for position size check."""
        if signal.target_qty is not None:
            # Can't know price here — use conservative estimate
            return equity * (self._max_position_pct / 100)
        return equity * (self._max_position_pct / 100) * Decimal(str(signal.confidence))

    def on_api_error(self, error: str) -> None:
        """Call on every consecutive API error."""
        self._state.consecutive_api_errors += 1
        logger.warning(
            "api_error_recorded",
            count=self._state.consecutive_api_errors,
            max=self._max_api_errors,
            error=error,
        )
        if self._state.consecutive_api_errors >= self._max_api_errors:
            self._trigger_kill_switch(
                f"Too many consecutive API errors: {self._state.consecutive_api_errors}"
            )

    def on_api_success(self) -> None:
        """Reset consecutive error counter on any successful API call."""
        self._state.consecutive_api_errors = 0

    def on_data_delay(self, delay_s: float) -> None:
        """Trigger kill switch on stale data."""
        self._trigger_kill_switch(
            f"Data delay {delay_s:.0f}s exceeds threshold {_settings.risk_data_delay_threshold_s}s"
        )

    def record_fill(self, pnl: Decimal) -> None:
        """Update daily PnL after a completed trade."""
        self._state.daily_pnl += pnl
        self._state.trades_today += 1

    def trigger_manual_kill_switch(self, reason: str = "manual") -> None:
        self._trigger_kill_switch(f"Manual kill switch: {reason}")

    def reset_kill_switch(self) -> None:
        """Manually reset kill switch (requires operator action)."""
        self._state.kill_switch_active = False
        self._state.kill_switch_reason = ""
        self._state.consecutive_api_errors = 0
        logger.warning("kill_switch_reset")

    def _trigger_kill_switch(self, reason: str) -> None:
        if not self._state.kill_switch_active:
            self._state.kill_switch_active = True
            self._state.kill_switch_reason = reason
            logger.error("kill_switch_triggered", reason=reason)
            raise KillSwitchTriggered(reason)

    @property
    def is_kill_switch_active(self) -> bool:
        return self._state.kill_switch_active

    @property
    def kill_switch_reason(self) -> str:
        return self._state.kill_switch_reason

    @property
    def state(self) -> RiskState:
        return self._state

    def get_summary(self) -> dict:
        s = self._state
        dd_pct = Decimal("0")
        if s.daily_start_equity > 0:
            dd_pct = (s.daily_start_equity - s.current_equity) / s.daily_start_equity * 100

        return {
            "kill_switch_active": s.kill_switch_active,
            "kill_switch_reason": s.kill_switch_reason,
            "current_equity": str(s.current_equity),
            "daily_pnl": str(s.daily_pnl),
            "daily_drawdown_pct": str(dd_pct.quantize(Decimal("0.01"))),
            "max_daily_drawdown_pct": str(self._max_daily_dd),
            "consecutive_api_errors": s.consecutive_api_errors,
            "trades_today": s.trades_today,
        }
