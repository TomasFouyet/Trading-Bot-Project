from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pandas as pd

from app.strategy.signals import Signal, SignalAction
from app.strategy.trend_following_v2_simple import TrendFollowingV2Simple
from scripts.run_signal_service import CSV_COLUMNS, DEFAULT_PARAMS, ServiceConfig, SignalService, compute_htf_bias


def _make_df(count: int, start: datetime | None = None, step_minutes: int = 15, drift: float = 20.0) -> pd.DataFrame:
    start = start or datetime(2026, 4, 1, tzinfo=UTC)
    rows = []
    price = 65000.0
    for idx in range(count):
        close = price + drift * idx
        rows.append(
            {
                "ts": pd.Timestamp(start + timedelta(minutes=step_minutes * idx)),
                "open": close - 8.0,
                "high": close + 14.0,
                "low": close - 12.0,
                "close": close,
                "volume": 1000.0 + idx,
            }
        )
    return pd.DataFrame(rows)


class _Response:
    def __init__(self, payload: dict, status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError("http error")

    def json(self) -> dict:
        return self._payload


class _FakeClient:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def __enter__(self) -> "_FakeClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def get(self, url: str, params: dict) -> _Response:
        return _Response(self.payload)


@dataclass
class _FakeTelegram:
    messages: list[str]

    def send(self, text: str) -> bool:
        self.messages.append(text)
        return True


def test_fetch_klines_returns_sorted_dataframe(monkeypatch):
    from app.data import bingx_client

    payload = {
        "code": 0,
        "data": [
            [1711929600000, "65000", "65100", "64950", "65080", "1200"],
            [1711928700000, "64900", "65020", "64880", "64980", "1100"],
        ],
    }
    monkeypatch.setattr(bingx_client.httpx, "Client", lambda timeout=10.0: _FakeClient(payload))
    monkeypatch.setattr(
        bingx_client,
        "_drop_in_progress_candle",
        lambda df, interval, now=None: df,
    )

    df = bingx_client.fetch_klines("BTC-USDT", "15m", limit=2)

    assert list(df.columns) == ["ts", "open", "high", "low", "close", "volume"]
    assert df["ts"].is_monotonic_increasing
    assert df.iloc[0]["close"] == 64980.0
    assert str(df.iloc[0]["ts"].tz) == "UTC"


def test_compute_htf_bias_returns_expected_states():
    bullish = _make_df(16 * 60, drift=10.0)
    bearish = _make_df(16 * 60, drift=-10.0)
    neutral = _make_df(16 * 60, drift=0.0)

    assert compute_htf_bias(bullish) == 1
    assert compute_htf_bias(bearish) == -1
    assert compute_htf_bias(neutral) == 0


def test_signal_service_end_to_end_writes_csv_and_formats_message(tmp_path, monkeypatch):
    base_df = _make_df(300)
    next_bar = _make_df(301).iloc[[-1]]
    live_df = pd.concat([base_df, next_bar], ignore_index=True)

    outputs = [base_df, live_df]

    def fake_fetcher(symbol: str, interval: str, limit: int) -> pd.DataFrame:
        return outputs.pop(0).copy()

    config = ServiceConfig(
        symbol="BTC-USDT",
        interval="15m",
        paper_equity_usd=100.0,
        log_level="INFO",
        csv_path=tmp_path / "live_signals.csv",
        log_path=tmp_path / "signal_service.log",
    )
    strategy = TrendFollowingV2Simple(symbol="BTC-USDT", params=DEFAULT_PARAMS.copy())
    telegram = _FakeTelegram(messages=[])
    service = SignalService(
        config,
        strategy=strategy,
        fetcher=fake_fetcher,
        telegram=telegram,
        dry_run=False,
    )

    target_ts = live_df["ts"].iloc[-1]

    def fake_on_bar_all(df: pd.DataFrame, htf_bias: int | None = None):
        if df["ts"].iloc[-1] == target_ts:
            entry = float(df["close"].iloc[-1])
            return [
                Signal(
                    action=SignalAction.BUY,
                    symbol="BTC-USDT",
                    ts=target_ts.to_pydatetime(),
                    strategy_id="trend_v2_simple",
                    confidence=1.0,
                    stop_loss=Decimal(str(entry - 250.0)),
                    take_profit=Decimal(str(entry + 675.0)),
                    reason="trend_long|rr=2.7",
                    meta={"rr_ratio": 2.7, "sl_mode": "STRUCTURAL", "sl": entry - 250.0, "tp1": entry + 675.0},
                )
            ]
        return []

    monkeypatch.setattr(strategy, "on_bar_all", fake_on_bar_all)

    startup = service.bootstrap()
    summary = service.run_once()

    assert startup.startup is True
    assert "Signal Service started" in telegram.messages[0]
    assert summary.signals_emitted == 1
    assert len(telegram.messages) == 2
    assert "LONG BTC-USDT 15m" in telegram.messages[1]
    assert "Sizing sugerido" in telegram.messages[1]

    csv_lines = config.csv_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(csv_lines) == 2
    assert csv_lines[0].split(",") == CSV_COLUMNS


def test_signal_service_is_idempotent_for_same_bar(tmp_path, monkeypatch):
    base_df = _make_df(300)
    next_bar = _make_df(301).iloc[[-1]]
    live_df = pd.concat([base_df, next_bar], ignore_index=True)

    outputs = [base_df, live_df, live_df]

    def fake_fetcher(symbol: str, interval: str, limit: int) -> pd.DataFrame:
        return outputs.pop(0).copy()

    config = ServiceConfig(
        symbol="BTC-USDT",
        interval="15m",
        paper_equity_usd=100.0,
        log_level="INFO",
        csv_path=tmp_path / "live_signals.csv",
        log_path=tmp_path / "signal_service.log",
    )
    strategy = TrendFollowingV2Simple(symbol="BTC-USDT", params=DEFAULT_PARAMS.copy())
    telegram = _FakeTelegram(messages=[])
    service = SignalService(
        config,
        strategy=strategy,
        fetcher=fake_fetcher,
        telegram=telegram,
        dry_run=False,
    )

    target_ts = live_df["ts"].iloc[-1]

    def fake_on_bar_all(df: pd.DataFrame, htf_bias: int | None = None):
        if df["ts"].iloc[-1] == target_ts:
            entry = float(df["close"].iloc[-1])
            return [
                Signal(
                    action=SignalAction.BUY,
                    symbol="BTC-USDT",
                    ts=target_ts.to_pydatetime(),
                    strategy_id="trend_v2_simple",
                    confidence=1.0,
                    stop_loss=Decimal(str(entry - 220.0)),
                    take_profit=Decimal(str(entry + 594.0)),
                    reason="trend_long|rr=2.7",
                    meta={"rr_ratio": 2.7, "sl_mode": "STRUCTURAL", "sl": entry - 220.0, "tp1": entry + 594.0},
                )
            ]
        return []

    monkeypatch.setattr(strategy, "on_bar_all", fake_on_bar_all)

    service.bootstrap()
    first = service.run_once()
    second = service.run_once()

    assert first.signals_emitted == 1
    assert second.signals_emitted == 0
    assert len(telegram.messages) == 2


def test_signal_service_sends_hourly_heartbeat_once_per_hour(tmp_path):
    base_df = _make_df(300)

    def fake_fetcher(symbol: str, interval: str, limit: int) -> pd.DataFrame:
        return base_df.copy()

    config = ServiceConfig(
        symbol="BTC-USDT",
        interval="15m",
        paper_equity_usd=100.0,
        log_level="INFO",
        csv_path=tmp_path / "live_signals.csv",
        log_path=tmp_path / "signal_service.log",
    )
    telegram = _FakeTelegram(messages=[])
    service = SignalService(config, fetcher=fake_fetcher, telegram=telegram, dry_run=False)

    service.bootstrap()
    first = service.maybe_send_heartbeat(datetime(2026, 4, 20, 12, 5, tzinfo=UTC))
    second = service.maybe_send_heartbeat(datetime(2026, 4, 20, 12, 35, tzinfo=UTC))
    third = service.maybe_send_heartbeat(datetime(2026, 4, 20, 13, 1, tzinfo=UTC))

    assert first is True
    assert second is False
    assert third is True
    assert len(telegram.messages) == 3
    assert "Signal Service heartbeat" in telegram.messages[1]
