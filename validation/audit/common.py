from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
import json
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import urlopen

import numpy as np
import pandas as pd

from scripts.run_simple_paper import HTF_CONFIG, VALIDATED_PARAMS


AUDIT_DIR = Path(__file__).resolve().parent
PLOTS_DIR = AUDIT_DIR / "plots"
DATASET_CSV = AUDIT_DIR / "bingx_btcusdt_15m_audit.csv"
PYTHON_SIGNALS_CSV = AUDIT_DIR / "python_signals.csv"
PINE_SIGNALS_CSV = AUDIT_DIR / "pine_signals.csv"
DIFF_SUMMARY_JSON = AUDIT_DIR / "diff_summary.json"
AUDIT_REPORT_MD = AUDIT_DIR / "AUDIT_REPORT.md"

BINGX_BASE_URL = "https://open-api.bingx.com/openApi/swap/v2/quote/klines"
SYMBOL = "BTC-USDT"
TIMEFRAME = "15m"
BAR_SECONDS = 15 * 60
HTF_SECONDS = 4 * 60 * 60
DEFAULT_BARS = 2000


@dataclass(slots=True)
class AuditConfig:
    symbol: str = SYMBOL
    timeframe: str = TIMEFRAME
    bars: int = DEFAULT_BARS
    htf_ema_period: int = HTF_CONFIG["ema_period"]
    validated_params: dict[str, Any] = None

    def __post_init__(self) -> None:
        if self.validated_params is None:
            self.validated_params = dict(VALIDATED_PARAMS)


def ensure_dirs() -> None:
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def floor_time(dt: datetime, seconds: int) -> datetime:
    ts = int(dt.timestamp())
    return datetime.fromtimestamp(ts - (ts % seconds), tz=UTC)


def fetch_bingx_klines(
    *,
    symbol: str,
    interval: str,
    limit: int,
    end_time_ms: int | None = None,
) -> list[dict[str, Any]]:
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    if end_time_ms is not None:
        params["endTime"] = end_time_ms
    url = f"{BINGX_BASE_URL}?{urlencode(params)}"
    with urlopen(url, timeout=20) as response:
        payload = json.loads(response.read().decode("utf-8"))
    code = payload.get("code")
    if code != 0:
        raise RuntimeError(f"BingX error {code}: {payload.get('msg', 'unknown')}")
    return payload.get("data", [])


def download_reference_dataset(config: AuditConfig) -> pd.DataFrame:
    ensure_dirs()

    now_utc = datetime.now(UTC)
    current_bar_open = floor_time(now_utc, BAR_SECONDS)
    last_closed_bar_open = current_bar_open - timedelta(seconds=BAR_SECONDS)
    end_time_ms = int((last_closed_bar_open + timedelta(seconds=BAR_SECONDS) - timedelta(milliseconds=1)).timestamp() * 1000)

    rows: list[dict[str, Any]] = []
    seen_times: set[int] = set()

    while len(rows) < config.bars:
        batch_limit = min(1440, config.bars - len(rows))
        batch = fetch_bingx_klines(
            symbol=config.symbol,
            interval=config.timeframe,
            limit=batch_limit,
            end_time_ms=end_time_ms,
        )
        if not batch:
            raise RuntimeError("BingX returned no data while building the reference dataset.")

        batch_sorted = sorted(batch, key=lambda item: int(item["time"]))
        new_rows = []
        for item in batch_sorted:
            ts_ms = int(item["time"])
            if ts_ms in seen_times:
                continue
            seen_times.add(ts_ms)
            close_ts = datetime.fromtimestamp(ts_ms / 1000, tz=UTC) + timedelta(seconds=BAR_SECONDS)
            new_rows.append(
                {
                    "timestamp_utc": close_ts,
                    "open": float(item["open"]),
                    "high": float(item["high"]),
                    "low": float(item["low"]),
                    "close": float(item["close"]),
                    "volume": float(item["volume"]),
                }
            )

        if not new_rows:
            raise RuntimeError("BingX pagination stalled: the API repeated only duplicate candles.")

        rows = new_rows + rows
        earliest_open_ms = int(batch_sorted[0]["time"])
        end_time_ms = earliest_open_ms - (BAR_SECONDS * 1000)

    df = pd.DataFrame(rows).sort_values("timestamp_utc").drop_duplicates("timestamp_utc")
    df = df.tail(config.bars).reset_index(drop=True)
    if len(df) != config.bars:
        raise RuntimeError(f"Expected {config.bars} bars, got {len(df)}.")

    expected_delta = pd.Timedelta(minutes=15)
    deltas = df["timestamp_utc"].diff().dropna()
    if not deltas.eq(expected_delta).all():
        raise RuntimeError("15m dataset contains timestamp gaps or duplicates after pagination.")

    df.to_csv(DATASET_CSV, index=False, date_format="%Y-%m-%dT%H:%M:%SZ")
    return df


def load_reference_dataset(path: Path | None = None) -> pd.DataFrame:
    path = path or DATASET_CSV
    df = pd.read_csv(path, parse_dates=["timestamp_utc"])
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    return df


def build_verified_htf(df_15m: pd.DataFrame) -> pd.DataFrame:
    df_htf = (
        df_15m.set_index("timestamp_utc")
        .resample("4h", label="right", closed="right")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
        .dropna()
        .reset_index()
    )
    hours = sorted(df_htf["timestamp_utc"].dt.hour.unique().tolist())
    if hours != [0, 4, 8, 12, 16, 20]:
        raise RuntimeError(
            "4H resample misaligned. Expected close timestamps at 00:00/04:00/08:00/12:00/16:00/20:00 UTC, "
            f"got hours={hours}."
        )
    return df_htf


def align_last_closed_htf(df_15m: pd.DataFrame, df_htf: pd.DataFrame) -> pd.DataFrame:
    left = df_15m.sort_values("timestamp_utc").copy()
    right = df_htf.sort_values("timestamp_utc").copy()
    aligned = pd.merge_asof(
        left,
        right,
        on="timestamp_utc",
        direction="backward",
        suffixes=("", "_4h"),
    )
    return aligned


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def pine_rma(series: pd.Series, length: int) -> pd.Series:
    values = series.astype(float).to_numpy(dtype=float)
    result = np.full(len(values), np.nan, dtype=float)
    if len(values) < length:
        return pd.Series(result, index=series.index, dtype=float)

    seed = float(np.nanmean(values[:length]))
    result[length - 1] = seed
    alpha = 1.0 / length
    for i in range(length, len(values)):
        prev = result[i - 1]
        result[i] = alpha * values[i] + (1.0 - alpha) * prev
    return pd.Series(result, index=series.index, dtype=float)


def pine_atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    tr = pd.concat(
        [
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return pine_rma(tr, length)


def pine_adx(df: pd.DataFrame, length: int = 14) -> tuple[pd.Series, pd.Series, pd.Series]:
    high = df["high"].astype(float)
    low = df["low"].astype(float)

    up = high.diff()
    down = -low.diff()
    plus_dm = up.where((up > down) & (up > 0), 0.0).fillna(0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0).fillna(0.0)
    atr = pine_atr(df, length).replace(0, np.nan)
    plus_di = 100.0 * pine_rma(plus_dm, length) / atr
    minus_di = 100.0 * pine_rma(minus_dm, length) / atr
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = pine_rma(dx.fillna(0.0), length)
    return plus_di.fillna(0.0), minus_di.fillna(0.0), adx.fillna(0.0)


def compute_python_htf_bias(close_value: float, ema50_value: float) -> str:
    if pd.isna(close_value) or pd.isna(ema50_value):
        return "NEUTRAL"
    if close_value > ema50_value * 1.002:
        return "BULL"
    if close_value < ema50_value * 0.998:
        return "BEAR"
    return "NEUTRAL"


def compute_pine_htf_bias(close_value: float, ema50_value: float) -> str:
    if pd.isna(close_value) or pd.isna(ema50_value):
        return "NEUTRAL"
    if close_value > ema50_value:
        return "BULL"
    if close_value < ema50_value:
        return "BEAR"
    return "NEUTRAL"


def signal_type_from_action(action: str) -> str:
    return "LONG" if action == "BUY" else "SHORT"


def relative_diff(a: float, b: float) -> float | None:
    if pd.isna(a) or pd.isna(b):
        return None
    if b == 0:
        return 0.0 if a == 0 else None
    return abs(a - b) / abs(b) * 100.0


def ensure_float(value: Any) -> float | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    try:
        return float(value)
    except Exception:
        return None
