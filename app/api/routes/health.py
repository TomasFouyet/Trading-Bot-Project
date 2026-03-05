"""
GET /health — liveness and readiness check.
"""
from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_session

router = APIRouter()


@router.get("/health")
async def health_check(session: AsyncSession = Depends(get_session)) -> dict:
    """
    Returns OK if API, database, and Redis are reachable.
    """
    checks: dict = {"api": "ok", "database": "unknown", "redis": "unknown"}

    try:
        await session.execute(text("SELECT 1"))
        checks["database"] = "ok"
    except Exception as e:
        checks["database"] = f"error: {e}"

    try:
        import redis.asyncio as aioredis
        from app.config import get_settings
        r = aioredis.from_url(get_settings().redis_url)
        await r.ping()
        await r.aclose()
        checks["redis"] = "ok"
    except Exception as e:
        checks["redis"] = f"error: {e}"

    status = "ok" if all(v == "ok" for v in checks.values()) else "degraded"

    return {
        "status": status,
        "ts": datetime.now(timezone.utc).isoformat(),
        "checks": checks,
    }
