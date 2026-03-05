"""
Parquet-based OHLCV storage.

Layout:
  {parquet_dir}/{symbol}/{timeframe}/YYYY-MM.parquet

Uses PyArrow for writes and DuckDB for fast analytical queries.
"""
from __future__ import annotations

import os
from datetime import datetime
from decimal import Decimal
from pathlib import Path

import duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from app.broker.base import OHLCVBar
from app.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)
_settings = get_settings()

SCHEMA = pa.schema(
    [
        pa.field("ts", pa.timestamp("us", tz="UTC")),
        pa.field("open", pa.float64()),
        pa.field("high", pa.float64()),
        pa.field("low", pa.float64()),
        pa.field("close", pa.float64()),
        pa.field("volume", pa.float64()),
    ]
)


class ParquetStore:
    """
    Append-only Parquet store for OHLCV data.
    Each (symbol, timeframe) is stored in monthly partitions.
    """

    def __init__(self, base_dir: str | None = None) -> None:
        self._base = Path(base_dir or _settings.parquet_dir)
        self._base.mkdir(parents=True, exist_ok=True)
        self._duck = duckdb.connect(":memory:")

    def _partition_path(self, symbol: str, timeframe: str, year: int, month: int) -> Path:
        safe_symbol = symbol.replace("/", "-").replace(":", "-")
        d = self._base / safe_symbol / timeframe
        d.mkdir(parents=True, exist_ok=True)
        return d / f"{year:04d}-{month:02d}.parquet"

    def write_bars(self, symbol: str, timeframe: str, bars: list[OHLCVBar]) -> int:
        """
        Write bars to parquet, partitioned by month.
        Returns number of bars written.
        """
        if not bars:
            return 0

        # Group bars by (year, month)
        groups: dict[tuple[int, int], list[OHLCVBar]] = {}
        for bar in bars:
            key = (bar.ts.year, bar.ts.month)
            groups.setdefault(key, []).append(bar)

        total_written = 0
        for (year, month), group_bars in groups.items():
            path = self._partition_path(symbol, timeframe, year, month)

            new_df = pd.DataFrame(
                {
                    "ts": pd.to_datetime([b.ts for b in group_bars], utc=True),
                    "open": [float(b.open) for b in group_bars],
                    "high": [float(b.high) for b in group_bars],
                    "low": [float(b.low) for b in group_bars],
                    "close": [float(b.close) for b in group_bars],
                    "volume": [float(b.volume) for b in group_bars],
                }
            )
            new_df = new_df.sort_values("ts").drop_duplicates("ts")

            if path.exists():
                existing_df = pd.read_parquet(path)
                combined = pd.concat([existing_df, new_df]).drop_duplicates("ts").sort_values("ts")
                combined.to_parquet(path, index=False, schema=SCHEMA)
                written = len(new_df)
            else:
                new_df.to_parquet(path, index=False, schema=SCHEMA)
                written = len(new_df)

            total_written += written
            logger.debug(
                "parquet_write",
                symbol=symbol,
                timeframe=timeframe,
                partition=f"{year}-{month:02d}",
                bars=written,
            )

        return total_written

    def read_bars(
        self,
        symbol: str,
        timeframe: str,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> list[OHLCVBar]:
        """
        Read OHLCV bars using DuckDB for fast filtering.
        """
        safe_symbol = symbol.replace("/", "-").replace(":", "-")
        pattern = str(self._base / safe_symbol / timeframe / "*.parquet")

        files = list((self._base / safe_symbol / timeframe).glob("*.parquet"))
        if not files:
            logger.warning("parquet_no_files", symbol=symbol, timeframe=timeframe)
            return []

        where_clauses = []
        if start:
            where_clauses.append(f"ts >= '{start.isoformat()}'")
        if end:
            where_clauses.append(f"ts <= '{end.isoformat()}'")

        where_sql = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""

        query = f"""
            SELECT ts, open, high, low, close, volume
            FROM read_parquet('{pattern}')
            {where_sql}
            ORDER BY ts
        """
        try:
            df = self._duck.execute(query).df()
        except Exception as e:
            logger.error("parquet_read_error", error=str(e), symbol=symbol)
            return []

        return [
            OHLCVBar(
                symbol=symbol,
                timeframe=timeframe,
                ts=row.ts.to_pydatetime() if hasattr(row.ts, "to_pydatetime") else row.ts,
                open=Decimal(str(row.open)),
                high=Decimal(str(row.high)),
                low=Decimal(str(row.low)),
                close=Decimal(str(row.close)),
                volume=Decimal(str(row.volume)),
            )
            for row in df.itertuples()
        ]

    def query_df(self, sql: str) -> pd.DataFrame:
        """Run arbitrary DuckDB SQL against all parquet files."""
        return self._duck.execute(sql).df()

    def get_date_range(self, symbol: str, timeframe: str) -> tuple[datetime | None, datetime | None]:
        """Return (min_ts, max_ts) for available data."""
        safe_symbol = symbol.replace("/", "-").replace(":", "-")
        files = list((self._base / safe_symbol / timeframe).glob("*.parquet"))
        if not files:
            return None, None

        pattern = str(self._base / safe_symbol / timeframe / "*.parquet")
        try:
            result = self._duck.execute(
                f"SELECT MIN(ts) as min_ts, MAX(ts) as max_ts FROM read_parquet('{pattern}')"
            ).fetchone()
            if result:
                return result[0], result[1]
        except Exception:
            pass
        return None, None

    def list_symbols(self) -> list[dict]:
        """Return list of available (symbol, timeframe) pairs with row counts."""
        result = []
        for symbol_dir in self._base.iterdir():
            if not symbol_dir.is_dir():
                continue
            for tf_dir in symbol_dir.iterdir():
                if not tf_dir.is_dir():
                    continue
                files = list(tf_dir.glob("*.parquet"))
                result.append(
                    {
                        "symbol": symbol_dir.name,
                        "timeframe": tf_dir.name,
                        "files": len(files),
                    }
                )
        return result
