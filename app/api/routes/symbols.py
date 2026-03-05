"""
GET /symbols — list available symbols with stored data.
"""
from __future__ import annotations

from fastapi import APIRouter

from app.data.parquet_store import ParquetStore

router = APIRouter()


@router.get("/symbols")
async def list_symbols() -> dict:
    store = ParquetStore()
    return {"symbols": store.list_symbols()}
