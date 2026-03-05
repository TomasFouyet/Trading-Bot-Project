"""
Unit tests for RiskManager.
"""
from __future__ import annotations

from decimal import Decimal
from datetime import datetime, timezone

import pytest

from app.core.exceptions import KillSwitchTriggered
from app.risk.manager import RiskManager
from app.strategy.signals import Signal, SignalAction


def make_signal(action: SignalAction) -> Signal:
    return Signal(
        action=action,
        symbol="BTC-USDT",
        ts=datetime(2024, 1, 1, tzinfo=timezone.utc),
        strategy_id="test",
    )


def test_risk_manager_approves_buy(risk_manager):
    signal = make_signal(SignalAction.BUY)
    approved, reason = risk_manager.validate_signal(signal, Decimal("10000"))
    assert approved
    assert reason == "approved"


def test_risk_manager_approves_close(risk_manager):
    """CLOSE signals should always be approved (risk reduction)."""
    signal = make_signal(SignalAction.CLOSE)
    approved, reason = risk_manager.validate_signal(signal, Decimal("10000"))
    assert approved
    assert reason == "close_position"


def test_risk_manager_rejects_hold(risk_manager):
    signal = make_signal(SignalAction.HOLD)
    approved, reason = risk_manager.validate_signal(signal, Decimal("10000"))
    assert not approved


def test_risk_manager_rejects_on_zero_equity(risk_manager):
    signal = make_signal(SignalAction.BUY)
    approved, reason = risk_manager.validate_signal(signal, Decimal("0"))
    assert not approved
    assert "equity" in reason.lower()


def test_kill_switch_on_max_api_errors(risk_manager):
    """Trigger kill switch after N consecutive API errors."""
    with pytest.raises(KillSwitchTriggered):
        for i in range(10):
            risk_manager.on_api_error("connection refused")


def test_api_success_resets_error_count(risk_manager):
    """Successful API call resets consecutive error counter."""
    risk_manager.on_api_error("timeout")
    risk_manager.on_api_error("timeout")
    risk_manager.on_api_success()
    assert risk_manager.state.consecutive_api_errors == 0


def test_kill_switch_blocks_signals(risk_manager):
    """Once kill switch is active, all signals are rejected."""
    risk_manager._state.kill_switch_active = True
    risk_manager._state.kill_switch_reason = "test"
    signal = make_signal(SignalAction.BUY)
    approved, reason = risk_manager.validate_signal(signal, Decimal("10000"))
    assert not approved
    assert "kill switch" in reason.lower()


def test_manual_kill_switch(risk_manager):
    with pytest.raises(KillSwitchTriggered):
        risk_manager.trigger_manual_kill_switch("operator halt")
    assert risk_manager.is_kill_switch_active
    assert "operator halt" in risk_manager.kill_switch_reason


def test_reset_kill_switch(risk_manager):
    risk_manager._state.kill_switch_active = True
    risk_manager.reset_kill_switch()
    assert not risk_manager.is_kill_switch_active


def test_compute_order_qty(risk_manager):
    signal = make_signal(SignalAction.BUY)
    qty = risk_manager.compute_order_qty(signal, Decimal("10000"), Decimal("40000"))
    # With 10% max position, equity=10000, price=40000:
    # max_notional = 10000 * 10/100 = 1000
    # qty = 1000 / 40000 = 0.025
    assert qty > 0
    assert qty <= Decimal("0.025") + Decimal("0.001")


def test_daily_drawdown_kill_switch():
    """Kill switch triggers when daily drawdown exceeds limit."""
    rm = RiskManager(max_daily_drawdown_pct=Decimal("2.0"))
    rm.initialize(Decimal("10000"))
    with pytest.raises(KillSwitchTriggered):
        rm.update_equity(Decimal("9700"))  # 3% drawdown > 2% limit
