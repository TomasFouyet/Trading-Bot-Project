"""
GET /ohlcv — fetch OHLCV candles from Parquet store.
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from app.data.parquet_store import ParquetStore

router = APIRouter()


@router.get("/ohlcv")
async def get_ohlcv(
    symbol: str = Query(..., description="Symbol, e.g. BTC-USDT"),
    tf: str = Query("5m", description="Timeframe, e.g. 1m, 5m, 1h"),
    start: Optional[str] = Query(None, description="ISO datetime, e.g. 2024-01-01T00:00:00"),
    end: Optional[str] = Query(None, description="ISO datetime"),
    limit: int = Query(500, ge=1, le=5000),
) -> dict:
    store = ParquetStore()

    start_dt = datetime.fromisoformat(start) if start else None
    end_dt = datetime.fromisoformat(end) if end else None

    bars = store.read_bars(symbol, tf, start_dt, end_dt)

    if not bars:
        raise HTTPException(status_code=404, detail=f"No data found for {symbol}/{tf}")

    # Apply limit from end
    bars = bars[-limit:]

    return {
        "symbol": symbol,
        "timeframe": tf,
        "count": len(bars),
        "bars": [
            {
                "ts": b.ts.isoformat(),
                "open": float(b.open),
                "high": float(b.high),
                "low": float(b.low),
                "close": float(b.close),
                "volume": float(b.volume),
            }
            for b in bars
        ],
    }
