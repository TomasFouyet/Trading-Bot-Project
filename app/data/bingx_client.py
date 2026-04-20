from __future__ import annotations

import logging
import time
from datetime import UTC, datetime, timedelta

import httpx
import pandas as pd

from app.broker.bingx_client import TIMEFRAME_MAP, normalize_symbol

logger = logging.getLogger(__name__)

BASE_URL = "https://open-api.bingx.com"
KLINES_PATH = "/openApi/swap/v2/quote/klines"
MAX_RETRIES = 3

_INTERVAL_TO_DELTA = {
    "1m": timedelta(minutes=1),
    "3m": timedelta(minutes=3),
    "5m": timedelta(minutes=5),
    "15m": timedelta(minutes=15),
    "30m": timedelta(minutes=30),
    "1h": timedelta(hours=1),
    "2h": timedelta(hours=2),
    "4h": timedelta(hours=4),
    "6h": timedelta(hours=6),
    "12h": timedelta(hours=12),
    "1d": timedelta(days=1),
    "3d": timedelta(days=3),
    "1w": timedelta(weeks=1),
}


def interval_to_timedelta(interval: str) -> timedelta:
    try:
        return _INTERVAL_TO_DELTA[interval]
    except KeyError as exc:
        raise ValueError(f"Unsupported interval: {interval}") from exc


def _parse_rows(payload: object, interval: str) -> pd.DataFrame:
    delta = interval_to_timedelta(interval)
    raw_list = payload if isinstance(payload, list) else []
    rows: list[dict[str, object]] = []

    for item in raw_list:
        if isinstance(item, list):
            open_ms, o, h, l, c, v = item[0], item[1], item[2], item[3], item[4], item[5]
        elif isinstance(item, dict):
            open_ms = item.get("time", item.get("openTime", item.get("t")))
            o = item.get("open", item.get("o"))
            h = item.get("high", item.get("h"))
            l = item.get("low", item.get("l"))
            c = item.get("close", item.get("c"))
            v = item.get("volume", item.get("v"))
        else:
            continue

        if open_ms is None:
            continue

        open_ts = pd.Timestamp(int(open_ms), unit="ms", tz="UTC")
        close_ts = open_ts + delta
        rows.append(
            {
                "ts": close_ts,
                "open": float(o),
                "high": float(h),
                "low": float(l),
                "close": float(c),
                "volume": float(v),
            }
        )

    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
    if df.empty:
        return df
    return df.sort_values("ts").drop_duplicates(subset=["ts"]).reset_index(drop=True)


def _drop_in_progress_candle(df: pd.DataFrame, interval: str, now: datetime | None = None) -> pd.DataFrame:
    if df.empty:
        return df

    now_utc = now or datetime.now(UTC)
    last_close = df["ts"].iloc[-1].to_pydatetime()
    if last_close > now_utc:
        logger.info(
            "dropping_in_progress_candle",
            extra={"interval": interval, "last_close_utc": last_close.isoformat()},
        )
        return df.iloc[:-1].reset_index(drop=True)
    return df


def _warn_on_temporal_gaps(df: pd.DataFrame, interval: str) -> None:
    if len(df) < 2:
        return

    expected = pd.Timedelta(interval_to_timedelta(interval))
    diffs = df["ts"].diff().dropna()
    gap_rows = diffs[diffs > expected]
    for idx, diff in gap_rows.items():
        missing = int(round(diff / expected)) - 1
        if missing > 1:
            logger.warning(
                "bingx_gap_detected",
                extra={
                    "interval": interval,
                    "missing_candles": missing,
                    "prev_ts_utc": df.loc[idx - 1, "ts"].isoformat(),
                    "curr_ts_utc": df.loc[idx, "ts"].isoformat(),
                },
            )


def fetch_klines(symbol: str, interval: str, limit: int = 300) -> pd.DataFrame:
    """
    Return OHLCV candles as a UTC DataFrame sorted oldest→newest.

    The `ts` column is the candle close timestamp in UTC. Any still-open
    last candle is discarded to avoid lookahead bias.
    """
    params = {
        "symbol": normalize_symbol(symbol),
        "interval": TIMEFRAME_MAP.get(interval, interval),
        "limit": int(limit),
    }
    url = f"{BASE_URL}{KLINES_PATH}"
    last_error: Exception | None = None

    with httpx.Client(timeout=10.0) as client:
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = client.get(url, params=params)
                if response.status_code >= 500:
                    raise RuntimeError(f"BingX server error {response.status_code}")
                response.raise_for_status()
                payload = response.json()
                code = payload.get("code", 0)
                if code != 0:
                    msg = payload.get("msg", "Unknown BingX error")
                    raise RuntimeError(f"BingX API error {code}: {msg}")

                data = payload.get("data", payload)
                if isinstance(data, dict):
                    data = data.get("data", [])

                df = _parse_rows(data, interval)
                df = _drop_in_progress_candle(df, interval)
                _warn_on_temporal_gaps(df, interval)
                return df
            except (httpx.HTTPError, ValueError, RuntimeError) as exc:
                last_error = exc
                if attempt >= MAX_RETRIES:
                    break
                sleep_s = 0.5 * (2 ** (attempt - 1))
                logger.warning(
                    "bingx_fetch_retry",
                    extra={
                        "symbol": params["symbol"],
                        "interval": params["interval"],
                        "attempt": attempt,
                        "error": str(exc),
                        "sleep_s": sleep_s,
                    },
                )
                time.sleep(sleep_s)

    raise RuntimeError(
        f"Failed to fetch BingX klines for {params['symbol']} {params['interval']} after {MAX_RETRIES} attempts: {last_error}"
    )


__all__ = ["fetch_klines", "interval_to_timedelta"]
