"""
POST /strategy/run_backtest — submit a backtest job to Celery.
GET  /strategy/backtest/{run_id} — check backtest status/result.
"""
from __future__ import annotations

import uuid
from datetime import datetime

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select

from app.tasks.backtest_tasks import run_backtest_task

router = APIRouter(prefix="/strategy")


class BacktestRequest(BaseModel):
    symbol: str = "BTC-USDT"
    timeframe: str = "5m"
    start: str = Field(..., description="ISO datetime, e.g. 2024-01-01T00:00:00")
    end: str = Field(..., description="ISO datetime, e.g. 2024-06-01T00:00:00")
    strategy_name: str = "ema_cross"
    strategy_params: dict = Field(default_factory=dict)
    initial_balance: float = 10000.0
    commission_bps: float | None = None
    slippage_bps: float | None = None


@router.post("/run_backtest")
async def run_backtest(req: BacktestRequest) -> dict:
    """
    Submit a backtest job. Returns run_id immediately.
    Poll GET /strategy/backtest/{run_id} for results.
    """
    run_id = str(uuid.uuid4())
    task = run_backtest_task.apply_async(
        kwargs={
            "symbol": req.symbol,
            "timeframe": req.timeframe,
            "start_iso": req.start,
            "end_iso": req.end,
            "strategy_name": req.strategy_name,
            "strategy_params": req.strategy_params,
            "initial_balance": req.initial_balance,
            "commission_bps": req.commission_bps,
            "slippage_bps": req.slippage_bps,
            "run_id": run_id,
        },
        task_id=run_id,
    )
    return {"run_id": run_id, "task_id": task.id, "status": "submitted"}


@router.get("/backtest/{run_id}")
async def get_backtest_result(run_id: str) -> dict:
    """
    Fetch backtest run results from DB.
    """
    from app.db.session import AsyncSessionLocal
    from app.db.models import BacktestRun

    async with AsyncSessionLocal() as session:
        result = await session.get(BacktestRun, run_id)
        if not result:
            raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

        return {
            "run_id": result.id,
            "symbol": result.symbol,
            "timeframe": result.timeframe,
            "strategy_name": result.strategy_name,
            "status": result.status,
            "metrics": {
                "total_trades": result.total_trades,
                "winning_trades": result.winning_trades,
                "winrate": float(result.winrate) if result.winrate else None,
                "total_pnl": float(result.total_pnl) if result.total_pnl else None,
                "total_pnl_pct": float(result.total_pnl_pct) if result.total_pnl_pct else None,
                "max_drawdown_pct": float(result.max_drawdown_pct) if result.max_drawdown_pct else None,
                "sharpe_ratio": float(result.sharpe_ratio) if result.sharpe_ratio else None,
            },
            "report": result.report,
        }
