"""
Download OHLCV candles from Binance via ccxt and cache to local parquet.

Usage:
    from validation.data_loader import load_candles
    df = load_candles("BTC/USDT", "15m", days=730)
"""
from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

CACHE_DIR = Path(__file__).parent / "cache"


def _fetch_ccxt(symbol: str, timeframe: str, since_ms: int, limit: int = 1000):
    """Fetch one page of candles from Binance via ccxt."""
    import ccxt

    exchange = ccxt.binance({"enableRateLimit": True})
    return exchange.fetch_ohlcv(symbol, timeframe, since=since_ms, limit=limit)


def download_candles(
    symbol: str = "BTC/USDT",
    timeframe: str = "15m",
    days: int = 730,
    cache: bool = True,
) -> pd.DataFrame:
    """
    Download historical OHLCV and return a clean DataFrame.

    Columns: ts (datetime UTC), open, high, low, close, volume.
    Cached to validation/cache/<symbol>_<tf>.parquet for fast reload.
    """
    safe_sym = symbol.replace("/", "")
    cache_path = CACHE_DIR / f"{safe_sym}_{timeframe}_{days}d.parquet"

    if cache and cache_path.exists():
        df = pd.read_parquet(cache_path)
        print(f"[data_loader] Loaded {len(df)} bars from cache: {cache_path.name}")
        return df

    print(f"[data_loader] Downloading {symbol} {timeframe} last {days} days ...")

    tf_ms = _timeframe_to_ms(timeframe)
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    since_ms = now_ms - days * 86_400_000

    all_candles = []
    cursor = since_ms

    while cursor < now_ms:
        batch = _fetch_ccxt(symbol, timeframe, since_ms=cursor, limit=1000)
        if not batch:
            break
        all_candles.extend(batch)
        cursor = batch[-1][0] + tf_ms
        print(f"  ... {len(all_candles)} candles", end="\r")
        time.sleep(0.05)

    print(f"\n[data_loader] Downloaded {len(all_candles)} candles total.")

    df = pd.DataFrame(all_candles, columns=["ts_ms", "open", "high", "low", "close", "volume"])
    df["ts"] = pd.to_datetime(df["ts_ms"], unit="ms", utc=True)
    df = df[["ts", "open", "high", "low", "close", "volume"]].copy()
    df = df.drop_duplicates(subset="ts").sort_values("ts").reset_index(drop=True)

    # Cast to float
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)

    if cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_path, index=False)
        print(f"[data_loader] Cached to {cache_path.name}")

    return df


def _timeframe_to_ms(tf: str) -> int:
    """Convert timeframe string to milliseconds."""
    unit = tf[-1]
    val = int(tf[:-1])
    multipliers = {"m": 60_000, "h": 3_600_000, "d": 86_400_000, "w": 604_800_000}
    return val * multipliers[unit]


# Convenience alias
load_candles = download_candles
