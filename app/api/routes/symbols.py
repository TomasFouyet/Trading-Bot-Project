"""
GET /symbols — list available symbols with stored data.
GET /symbols/contracts — list all perpetual swap contracts from BingX.
"""
from __future__ import annotations

from fastapi import APIRouter

from app.broker.bingx_client import BingXClient
from app.config import get_settings
from app.data.parquet_store import ParquetStore

router = APIRouter()


@router.get("/symbols")
async def list_symbols() -> dict:
    store = ParquetStore()
    return {"symbols": store.list_symbols()}


@router.get("/symbols/contracts")
async def list_contracts() -> dict:
    """Return all perpetual swap contracts available on BingX."""
    settings = get_settings()
    async with BingXClient(
        api_key=settings.bingx_api_key,
        api_secret=settings.bingx_api_secret,
        base_url=settings.bingx_base_url,
        market_type=settings.bingx_market_type,
    ) as client:
        contracts = await client.get_contracts()
    return {"contracts": contracts, "count": len(contracts)}
