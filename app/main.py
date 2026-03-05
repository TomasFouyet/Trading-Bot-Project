"""
FastAPI application entry point.

Exposes:
  GET  /health
  GET  /symbols
  GET  /ohlcv
  POST /strategy/run_backtest
  GET  /strategy/backtest/{run_id}
  POST /bot/start
  POST /bot/stop
  GET  /bot/status
  GET  /trades
  GET  /trades/{trade_id}
  GET  /metrics/summary
  GET  /metrics/prometheus
"""
from __future__ import annotations

import structlog
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.core.logging import configure_logging, get_logger

# Configure logging immediately
_settings = get_settings()
configure_logging(log_level=_settings.log_level, log_format=_settings.log_format.value)
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ──────────────────────────────────────────────────────────
    logger.info("app_startup", config=_settings.safe_repr())

    # Create DB tables (dev mode; use Alembic in production)
    from app.db.session import create_all_tables
    await create_all_tables()

    # Ensure bot_state singleton exists
    from app.db.session import AsyncSessionLocal
    from app.db.models import BotState
    async with AsyncSessionLocal() as session:
        existing = await session.get(BotState, 1)
        if not existing:
            session.add(BotState(id=1))
            await session.commit()
            logger.info("bot_state_initialized")

    logger.info("app_ready", host="0.0.0.0", port=8000)
    yield

    # ── Shutdown ─────────────────────────────────────────────────────────
    logger.info("app_shutdown")


app = FastAPI(
    title="Trading Bot API",
    description="BingX trading bot with backtest/paper/live modes",
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Exception handlers ─────────────────────────────────────────────────────

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.error("unhandled_exception", path=str(request.url), error=str(exc))
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "error": str(exc)},
    )


# ── Routers ────────────────────────────────────────────────────────────────

from app.api.routes.health import router as health_router
from app.api.routes.symbols import router as symbols_router
from app.api.routes.ohlcv import router as ohlcv_router
from app.api.routes.strategy import router as strategy_router
from app.api.routes.bot import router as bot_router
from app.api.routes.trades import router as trades_router
from app.api.routes.metrics_routes import router as metrics_router

app.include_router(health_router, tags=["Health"])
app.include_router(symbols_router, tags=["Data"])
app.include_router(ohlcv_router, tags=["Data"])
app.include_router(strategy_router, tags=["Strategy"])
app.include_router(bot_router, tags=["Bot"])
app.include_router(trades_router, tags=["Trades"])
app.include_router(metrics_router, tags=["Metrics"])
