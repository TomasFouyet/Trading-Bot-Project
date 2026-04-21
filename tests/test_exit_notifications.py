from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pandas as pd

from app.notify.telegram import SignalServiceTelegramNotifier
from app.strategy.signals import Signal, SignalAction
from app.strategy.trend_following_v2_simple import TrendFollowingV2Simple
from scripts.run_signal_service import DEFAULT_PARAMS, ServiceConfig, SignalService


def _make_df(count: int, start: datetime | None = None, step_minutes: int = 15, drift: float = 20.0) -> pd.DataFrame:
    start = start or datetime(2026, 4, 21, tzinfo=UTC)
    rows = []
    price = 76000.0
    for idx in range(count):
        close = price + drift * idx
        rows.append(
            {
                "ts": pd.Timestamp(start + timedelta(minutes=step_minutes * idx)),
                "open": close - 10.0,
                "high": close + 15.0,
                "low": close - 20.0,
                "close": close,
                "volume": 1000.0 + idx,
            }
        )
    return pd.DataFrame(rows)


@dataclass
class _FakeNotifier:
    entry_calls: list[dict]
    exit_calls: list[dict]
    lifecycle_calls: list[tuple[str, dict]]
    summary_calls: list[dict]

    def send_entry_notification(self, signal, context) -> bool:
        self.entry_calls.append({"signal": signal, "context": context})
        return True

    def send_exit_notification(self, signal, context) -> bool:
        self.exit_calls.append({"signal": signal, "context": context})
        return True

    def send_lifecycle_notification(self, event_type: str, payload: dict) -> bool:
        self.lifecycle_calls.append((event_type, payload))
        return True

    def send_daily_summary(self, summary: dict) -> bool:
        self.summary_calls.append(summary)
        return True


def _build_service(tmp_path, fetcher, notifier):
    config = ServiceConfig(
        symbol="BTC-USDT",
        interval="15m",
        paper_equity_usd=100.0,
        log_level="INFO",
        csv_path=tmp_path / "live_signals.csv",
        closures_csv_path=tmp_path / "live_closures.csv",
        log_path=tmp_path / "signal_service.log",
        state_path=tmp_path / "signal_service_state.json",
    )
    strategy = TrendFollowingV2Simple(symbol="BTC-USDT", params=DEFAULT_PARAMS.copy())
    return SignalService(config, strategy=strategy, fetcher=fetcher, telegram=notifier, dry_run=False)


def test_close_detection_writes_closure_csv_and_notifies(tmp_path, monkeypatch):
    base_df = _make_df(300)
    entry_df = _make_df(301)
    exit_df = _make_df(311)
    outputs = [base_df, entry_df, exit_df]

    def fetcher(symbol: str, interval: str, limit: int) -> pd.DataFrame:
        return outputs.pop(0).copy()

    notifier = _FakeNotifier([], [], [], [])
    service = _build_service(tmp_path, fetcher, notifier)
    strategy = service.strategy
    entry_ts = entry_df["ts"].iloc[-1]
    exit_ts = exit_df["ts"].iloc[-1]

    def fake_on_bar_all(df: pd.DataFrame, htf_bias: int | None = None):
        ts = df["ts"].iloc[-1]
        if ts == entry_ts:
            entry = float(df["close"].iloc[-1])
            return [
                Signal(
                    action=SignalAction.BUY,
                    symbol="BTC-USDT",
                    ts=entry_ts.to_pydatetime(),
                    strategy_id="trend_v2_simple",
                    confidence=1.0,
                    stop_loss=Decimal(str(entry - 664.06)),
                    take_profit=Decimal(str(entry + 1792.97)),
                    meta={"rr_ratio": 2.7, "sl_mode": "STRUCTURAL", "sl": entry - 664.06, "tp1": entry + 1792.97},
                )
            ]
        if ts == exit_ts:
            strategy.force_close()
            return [
                Signal(
                    action=SignalAction.CLOSE,
                    symbol="BTC-USDT",
                    ts=exit_ts.to_pydatetime(),
                    strategy_id="trend_v2_simple",
                    meta={"exit_type": "sl", "exit_price": 75605.74, "pnl_pct": -0.87},
                )
            ]
        return []

    monkeypatch.setattr(strategy, "on_bar_all", fake_on_bar_all)

    service.bootstrap()
    service.run_once()
    summary = service.run_once()

    assert summary.closures_emitted == 1
    assert notifier.exit_calls
    exit_ctx = notifier.exit_calls[0]["context"]
    assert exit_ctx["exit_type"] == "sl"
    assert exit_ctx["entry_price"] == float(notifier.entry_calls[0]["context"]["entry"])

    closures = pd.read_csv(service.config.closures_csv_path)
    assert len(closures) == 1
    assert closures.iloc[0]["exit_type"] == "sl"
    assert closures.iloc[0]["entry_price"] == round(float(notifier.entry_calls[0]["context"]["entry"]), 8)


def test_exit_notifier_formats_tp_and_sl_messages():
    notifier = SignalServiceTelegramNotifier("token", "chat", enabled=True)
    messages: list[str] = []
    notifier.send_html = lambda text: messages.append(text) or True  # type: ignore[method-assign]
    signal = Signal(
        action=SignalAction.CLOSE,
        symbol="BTC-USDT",
        ts=datetime(2026, 4, 21, 12, 0, tzinfo=UTC),
        strategy_id="trend_v2_simple",
        meta={"exit_type": "tp", "exit_price": 78062.77, "pnl_pct": 2.35},
    )
    tp_context = {
        "signal_type": "LONG",
        "exit_type": "tp",
        "entry_price": 76269.80,
        "exit_price": 78062.77,
        "pnl_pct": 2.35,
        "pnl_usd": 3.62,
        "leverage": 1.72,
        "duration_label": "14h 30m",
        "bars_held": 58,
        "entry_time_label": "09:45 UTC 21-abr",
        "exit_time_label": "00:15 UTC 22-abr",
        "cooldown_bars": 5,
    }
    notifier.send_exit_notification(signal, tp_context)

    sl_signal = Signal(
        action=SignalAction.CLOSE,
        symbol="BTC-USDT",
        ts=datetime(2026, 4, 21, 12, 0, tzinfo=UTC),
        strategy_id="trend_v2_simple",
        meta={"exit_type": "sl", "exit_price": 75605.74, "pnl_pct": -0.87},
    )
    sl_context = dict(tp_context)
    sl_context.update({"exit_type": "sl", "pnl_pct": -0.87, "pnl_usd": -1.50, "exit_price": 75605.74})
    notifier.send_exit_notification(sl_signal, sl_context)

    assert "🎯" in messages[0] and "CERRADO en TP" in messages[0]
    assert "❌" in messages[1] and "CERRADO en SL" in messages[1]


def test_restart_reconstructs_open_trade_from_csv(tmp_path, monkeypatch):
    base_df = _make_df(300)
    entry_df = _make_df(301)
    exit_df = _make_df(305)
    notifier = _FakeNotifier([], [], [], [])
    first_outputs = [base_df, entry_df]

    def first_fetcher(symbol: str, interval: str, limit: int) -> pd.DataFrame:
        return first_outputs.pop(0).copy()

    service = _build_service(tmp_path, first_fetcher, notifier)
    strategy = service.strategy
    entry_ts = entry_df["ts"].iloc[-1]

    def first_on_bar_all(df: pd.DataFrame, htf_bias: int | None = None):
        if df["ts"].iloc[-1] == entry_ts:
            entry = float(df["close"].iloc[-1])
            return [
                Signal(
                    action=SignalAction.BUY,
                    symbol="BTC-USDT",
                    ts=entry_ts.to_pydatetime(),
                    strategy_id="trend_v2_simple",
                    confidence=1.0,
                    stop_loss=Decimal(str(entry - 400.0)),
                    take_profit=Decimal(str(entry + 1080.0)),
                    meta={"rr_ratio": 2.7, "sl_mode": "STRUCTURAL", "sl": entry - 400.0, "tp1": entry + 1080.0},
                )
            ]
        return []

    monkeypatch.setattr(strategy, "on_bar_all", first_on_bar_all)
    service.bootstrap()
    service.run_once()
    service.stop()

    restart_notifier = _FakeNotifier([], [], [], [])
    second_outputs = [exit_df]

    def second_fetcher(symbol: str, interval: str, limit: int) -> pd.DataFrame:
        return second_outputs.pop(0).copy()

    restarted = _build_service(tmp_path, second_fetcher, restart_notifier)
    exit_ts = exit_df["ts"].iloc[-1]

    def second_on_bar_all(df: pd.DataFrame, htf_bias: int | None = None):
        if df["ts"].iloc[-1] == exit_ts:
            return [
                Signal(
                    action=SignalAction.CLOSE,
                    symbol="BTC-USDT",
                    ts=exit_ts.to_pydatetime(),
                    strategy_id="trend_v2_simple",
                    meta={"exit_type": "sl", "exit_price": 75605.74, "pnl_pct": -0.87},
                )
            ]
        return []

    monkeypatch.setattr(restarted.strategy, "on_bar_all", second_on_bar_all)
    summary = restarted.run_once()

    assert summary.closures_emitted == 1
    assert restart_notifier.exit_calls
    assert restart_notifier.exit_calls[0]["context"]["entry_price"] > 0


def test_lifecycle_notifications_fire_on_start_stop_and_api_error(tmp_path):
    base_df = _make_df(300)
    notifier = _FakeNotifier([], [], [], [])
    failures = {"count": 0}

    def fetcher(symbol: str, interval: str, limit: int) -> pd.DataFrame:
        if failures["count"] == 0:
            failures["count"] += 1
            return base_df.copy()
        raise RuntimeError("network down")

    service = _build_service(tmp_path, fetcher, notifier)
    service.config.api_error_alert_threshold = 3
    service.bootstrap()
    for _ in range(3):
        service.run_once()
    service.stop()

    lifecycle_events = [event for event, _payload in notifier.lifecycle_calls]
    assert "start" in lifecycle_events
    assert "api_error" in lifecycle_events
    assert "stop" in lifecycle_events
