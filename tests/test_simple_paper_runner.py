from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from app.broker.bingx_client import BingXClient
from app.broker.base import Position, TradeSide
from app.core.exceptions import BrokerError
from app.strategy.trend_following_v2_simple import TrendFollowingV2Simple
from scripts import run_simple_paper as runner


def _validated_strategy() -> TrendFollowingV2Simple:
    return TrendFollowingV2Simple(symbol="BTC-USDT", params=runner.VALIDATED_PARAMS)


def test_structural_stop_rr_comes_from_strategy(sample_df):
    strategy = _validated_strategy()
    window = sample_df.tail(120).reset_index(drop=True)
    ind_df = strategy._compute_indicators(window)
    close = float(ind_df.iloc[-1]["close"])
    atr = float(ind_df.iloc[-1]["atr"])

    stop_loss, take_profit, sl_mode = strategy._compute_entry_levels(
        ind_df, close, atr, "LONG"
    )

    risk = close - stop_loss
    reward = take_profit - close

    assert stop_loss < close
    assert take_profit > close
    assert sl_mode in {"structural", "atr_fallback", "min_risk_clamp", "atr"}
    assert reward / risk == pytest.approx(runner.VALIDATED_PARAMS["rr_ratio"], rel=1e-9)


def test_strategy_runtime_state_restores_open_trade():
    strategy = _validated_strategy()
    strategy.restore_open_trade(
        side="LONG",
        entry_price=42000.0,
        stop_loss=41000.0,
        take_profit=44500.0,
        start_bar=17,
    )

    snapshot = strategy.export_runtime_state()
    restored = _validated_strategy()
    restored.restore_runtime_state(snapshot)

    assert restored.export_runtime_state()["trade"] == snapshot["trade"]


class _FlatAdapter:
    async def get_positions(self, symbol: str):
        return []


class _LongAdapter:
    async def get_positions(self, symbol: str):
        return [
            Position(
                symbol=symbol,
                side=TradeSide.LONG,
                qty=Decimal("0.010"),
                avg_price=Decimal("43123.5"),
                unrealized_pnl=Decimal("12.5"),
            )
        ]


class _OrdersClient:
    def __init__(self, orders=None, cancel_result=None):
        self.orders = orders or []
        self.cancelled_symbols = []
        self.cancel_result = cancel_result or {}

    async def get_open_orders(self, symbol: str):
        return list(self.orders)

    async def cancel_all_open_orders(self, symbol: str):
        self.cancelled_symbols.append(symbol)
        return self.cancel_result


@pytest.mark.asyncio
async def test_live_reconcile_clears_stale_local_state():
    open_trade = {
        "symbol": "BTC-USDT",
        "side": "LONG",
        "entry_price": Decimal("43000"),
        "qty": Decimal("0.010"),
        "sl": Decimal("42000"),
        "tp": Decimal("45500"),
    }
    client = _OrdersClient()

    reconciled, report = await runner._reconcile_live_position(
        _FlatAdapter(), client, "BTC-USDT", open_trade
    )

    assert reconciled is None
    assert report["status"] == "saved_position_cleared_exchange_flat"


@pytest.mark.asyncio
async def test_live_reconcile_requires_persisted_state_for_exchange_position():
    client = _OrdersClient(
        orders=[
            {
                "orderId": "sl123",
                "type": "STOP_MARKET",
                "positionSide": "LONG",
                "stopPrice": "42000",
            }
        ]
    )
    with pytest.raises(runner.SafeModeRequired, match="exchange_position_without_local_state"):
        await runner._reconcile_live_position(_LongAdapter(), client, "BTC-USDT", None)


@pytest.mark.asyncio
async def test_live_reconcile_links_exchange_stop_to_local_state():
    open_trade = {
        "symbol": "BTC-USDT",
        "side": "LONG",
        "entry_price": Decimal("43000"),
        "qty": Decimal("0.010"),
        "sl": Decimal("42000"),
        "tp": Decimal("45500"),
        "sl_order_id": None,
    }
    client = _OrdersClient(
        orders=[
            {
                "orderId": "sl123",
                "type": "STOP_MARKET",
                "positionSide": "LONG",
                "stopPrice": "42000",
            }
        ]
    )

    reconciled, report = await runner._reconcile_live_position(
        _LongAdapter(), client, "BTC-USDT", open_trade
    )

    assert report["status"] == "boot_exchange_position_reconciled"
    assert reconciled["sl_order_id"] == "sl123"
    assert reconciled["qty"] == Decimal("0.010")


@pytest.mark.asyncio
async def test_live_reconcile_enters_safe_mode_when_exchange_stop_missing():
    open_trade = {
        "symbol": "BTC-USDT",
        "side": "LONG",
        "entry_price": Decimal("43000"),
        "qty": Decimal("0.010"),
        "sl": Decimal("42000"),
        "tp": Decimal("45500"),
        "sl_order_id": None,
    }
    client = _OrdersClient()

    with pytest.raises(runner.SafeModeRequired, match="missing_exchange_stop"):
        await runner._reconcile_live_position(_LongAdapter(), client, "BTC-USDT", open_trade)


@pytest.mark.asyncio
async def test_live_reconcile_cancels_stale_orders_when_exchange_flat():
    client = _OrdersClient(
        orders=[
            {
                "orderId": "old-stop",
                "type": "STOP_MARKET",
                "positionSide": "LONG",
                "stopPrice": "42000",
            }
        ]
    )

    reconciled, report = await runner._reconcile_live_position(
        _FlatAdapter(), client, "BTC-USDT", None
    )

    assert reconciled is None
    assert report["status"] == "exchange_flat_orders_cancelled"
    assert client.cancelled_symbols == ["BTC-USDT"]


@pytest.mark.asyncio
async def test_runtime_reconcile_detects_missing_exchange_position():
    open_trade = {
        "symbol": "BTC-USDT",
        "side": "LONG",
        "entry_price": Decimal("43000"),
        "qty": Decimal("0.010"),
        "sl": Decimal("42000"),
        "tp": Decimal("45500"),
    }
    with pytest.raises(runner.SafeModeRequired, match="runtime_exchange_flat_with_local_position"):
        await runner._reconcile_live_position(
            _FlatAdapter(), _OrdersClient(), "BTC-USDT", open_trade, phase="runtime"
        )


@pytest.mark.asyncio
async def test_runtime_reconcile_updates_stop_link():
    open_trade = {
        "symbol": "BTC-USDT",
        "side": "LONG",
        "entry_price": Decimal("43000"),
        "qty": Decimal("0.010"),
        "sl": Decimal("42000"),
        "tp": Decimal("45500"),
        "sl_order_id": None,
    }
    state = {
        "last_reconcile_at": datetime.now(timezone.utc) - timedelta(seconds=600),
    }
    client = _OrdersClient(
        orders=[
            {
                "orderId": "sl123",
                "type": "STOP_MARKET",
                "side": "SELL",
                "positionSide": "LONG",
                "stopPrice": "42000",
                "quantity": "0.010",
            }
        ]
    )

    reconciled, report = await runner._maybe_runtime_reconcile(
        state=state,
        adapter=_LongAdapter(),
        client=client,
        symbol="BTC-USDT",
        open_trade=open_trade,
        reconcile_interval_secs=120,
    )

    assert report["status"] == "runtime_exchange_position_reconciled"
    assert reconciled["sl_order_id"] == "sl123"
    assert state["last_reconcile_at"] is not None


@pytest.mark.asyncio
async def test_runtime_reconcile_enters_safe_mode_when_stop_missing():
    open_trade = {
        "symbol": "BTC-USDT",
        "side": "LONG",
        "entry_price": Decimal("43000"),
        "qty": Decimal("0.010"),
        "sl": Decimal("42000"),
        "tp": Decimal("45500"),
        "sl_order_id": None,
    }
    state = {
        "last_reconcile_at": datetime.now(timezone.utc) - timedelta(seconds=600),
    }

    with pytest.raises(runner.SafeModeRequired, match="missing_exchange_stop"):
        await runner._maybe_runtime_reconcile(
            state=state,
            adapter=_LongAdapter(),
            client=_OrdersClient(),
            symbol="BTC-USDT",
            open_trade=open_trade,
            reconcile_interval_secs=120,
        )


@pytest.mark.asyncio
async def test_live_reconcile_rejects_inconsistent_stop_side():
    open_trade = {
        "symbol": "BTC-USDT",
        "side": "LONG",
        "entry_price": Decimal("43000"),
        "qty": Decimal("0.010"),
        "sl": Decimal("42000"),
        "tp": Decimal("45500"),
        "sl_order_id": None,
    }
    client = _OrdersClient(
        orders=[
            {
                "orderId": "sl123",
                "type": "STOP_MARKET",
                "side": "BUY",
                "positionSide": "LONG",
                "stopPrice": "42000",
                "quantity": "0.010",
            }
        ]
    )

    with pytest.raises(runner.SafeModeRequired, match="invalid_exchange_stop_side"):
        await runner._reconcile_live_position(_LongAdapter(), client, "BTC-USDT", open_trade)


@pytest.mark.asyncio
async def test_shutdown_persists_open_trade_without_fake_close(tmp_path, monkeypatch):
    state_file = tmp_path / "bot_state.json"
    heartbeat_file = tmp_path / "runner_heartbeat.json"
    monkeypatch.setattr(runner, "STATE_FILE", state_file)
    monkeypatch.setattr(runner, "HEARTBEAT_FILE", heartbeat_file)

    strategy = _validated_strategy()
    strategy.restore_open_trade(
        side="LONG",
        entry_price=42000.0,
        stop_loss=41000.0,
        take_profit=44500.0,
        start_bar=9,
    )

    state = {
        "wins": 2,
        "losses": 1,
    }
    open_trade = {
        "trade_id": "abc123",
        "symbol": "BTC-USDT",
        "side": "LONG",
        "entry_price": Decimal("42000"),
        "qty": Decimal("0.010"),
        "fee_in": Decimal("0.5"),
        "entry_fee_type": "taker",
        "sl": Decimal("41000"),
        "tp": Decimal("44500"),
        "leverage": 1.5,
        "sl_dist_pct": 0.0238,
        "sl_mode": "structural",
        "adx_at_entry": 29.4,
        "atr_at_entry": 180.0,
        "htf_bias_at_entry": 1,
        "entry_time": datetime(2024, 1, 1, tzinfo=timezone.utc).isoformat(),
        "confidence": 0.65,
        "sl_order_id": "sl123",
    }

    await runner._persist_shutdown_state(
        notifier=None,
        state=state,
        strategy=strategy,
        open_trade=open_trade,
        equity=1000.0,
        peak_equity=1100.0,
        is_live=True,
    )

    saved = json.loads(state_file.read_text())
    assert saved["equity"] == 1000.0
    assert saved["position"]["side"] == "LONG"
    assert saved["position"]["sl"] == 41000.0
    assert saved["position"]["tp"] == 44500.0
    assert saved["position"]["sl_order_id"] == "sl123"
    assert saved["strategy_state"]["trade"]["state"] == 1


def test_state_load_quarantines_corrupt_file(tmp_path, monkeypatch):
    state_file = tmp_path / "bot_state.json"
    monkeypatch.setattr(runner, "STATE_FILE", state_file)
    state_file.write_text("{not-json")

    loaded = runner._load_state()

    assert loaded is None
    quarantined = list(tmp_path.glob("bot_state.corrupt.*.json"))
    assert quarantined
    assert not state_file.exists()


def test_state_load_quarantines_partial_state(tmp_path, monkeypatch):
    state_file = tmp_path / "bot_state.json"
    monkeypatch.setattr(runner, "STATE_FILE", state_file)
    state_file.write_text(json.dumps({"equity": 1000.0, "wins": 1}))

    loaded = runner._load_state()

    assert loaded is None
    quarantined = list(tmp_path.glob("bot_state.corrupt.*.json"))
    assert quarantined


def test_state_write_and_read_round_trip(tmp_path, monkeypatch):
    state_file = tmp_path / "bot_state.json"
    monkeypatch.setattr(runner, "STATE_FILE", state_file)
    state = {
        "symbol": "BTC-USDT",
        "timeframe": "15m",
        "mode": "paper",
        "runner_status": "in_position",
        "safe_mode_reason": None,
    }
    open_trade = {
        "trade_id": "abc123",
        "symbol": "BTC-USDT",
        "side": "LONG",
        "entry_price": Decimal("42000"),
        "qty": Decimal("0.010"),
        "fee_in": Decimal("0.5"),
        "entry_fee_type": "taker",
        "sl": Decimal("41000"),
        "tp": Decimal("44500"),
        "leverage": 1.5,
        "sl_dist_pct": 0.0238,
        "sl_mode": "structural",
        "adx_at_entry": 29.4,
        "atr_at_entry": 180.0,
        "htf_bias_at_entry": 1,
        "entry_time": datetime(2024, 1, 1, tzinfo=timezone.utc).isoformat(),
        "confidence": 0.65,
        "sl_order_id": "sl123",
    }

    runner._save_state(
        state,
        open_trade,
        equity=1000.0,
        peak_equity=1100.0,
        wins=2,
        losses=1,
        strategy_state={"trade": {"state": 1}},
    )

    loaded = runner._load_state()

    assert loaded["version"] == runner.STATE_SCHEMA_VERSION
    assert loaded["position"]["side"] == "LONG"
    assert loaded["position"]["sl_order_id"] == "sl123"
    assert loaded["strategy_state"]["trade"]["state"] == 1


def test_heartbeat_file_contains_runner_status(tmp_path, monkeypatch):
    heartbeat_file = tmp_path / "runner_heartbeat.json"
    monkeypatch.setattr(runner, "HEARTBEAT_FILE", heartbeat_file)
    state = {
        "equity": 1000.0,
        "open_trade": {
            "side": "LONG",
            "entry_price": Decimal("42000"),
            "qty": Decimal("0.010"),
            "sl": Decimal("41000"),
            "tp": Decimal("44500"),
            "sl_order_id": "sl123",
        },
        "last_bar_ts": datetime(2024, 1, 1, tzinfo=timezone.utc),
        "last_progress_at": datetime(2024, 1, 1, tzinfo=timezone.utc),
        "runner_status": "in_position",
        "safe_mode_reason": None,
        "last_error": None,
        "api_error_streak": 0,
    }

    runner._write_heartbeat(state, "BTC-USDT", "paper")

    payload = json.loads(heartbeat_file.read_text())
    assert payload["status"] == "in_position"
    assert payload["symbol"] == "BTC-USDT"
    assert payload["mode"] == "paper"
    assert payload["open_trade"]["sl_order_id"] == "sl123"


def test_runner_instance_lock_enforces_single_owner(tmp_path):
    lock_path = tmp_path / "runner.lock"
    first = runner.RunnerInstanceLock(lock_path, {"pid": "1"})
    second = runner.RunnerInstanceLock(lock_path, {"pid": "2"})

    first.acquire()
    try:
        with pytest.raises(runner.InstanceLockError, match="runner lock already held"):
            second.acquire()
    finally:
        first.release()

    second.acquire()
    second.release()


@pytest.mark.asyncio
async def test_bingx_client_request_with_retry_uses_async_sleep(monkeypatch):
    sleeps = []
    client = BingXClient(api_key="key", api_secret="secret")

    async def fake_sleep(delay: float):
        sleeps.append(delay)

    attempts = {"count": 0}

    async def fake_request(method, path, *, params=None, json=None, signed=False):
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise BrokerError("temporary broker failure")
        return {"ok": True}

    monkeypatch.setattr("app.broker.bingx_client.asyncio.sleep", fake_sleep)
    monkeypatch.setattr(client, "_request", fake_request)

    try:
        result = await client._request_with_retry("GET", "/health")
    finally:
        await client.close()

    assert result == {"ok": True}
    assert attempts["count"] == 3
    assert len(sleeps) == 2
