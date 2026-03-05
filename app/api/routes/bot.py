"""
POST /bot/start  — start bot in paper or real mode
POST /bot/stop   — stop the running bot
GET  /bot/status — current bot state
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_session
from app.db.models import BotState

router = APIRouter(prefix="/bot")


class BotStartRequest(BaseModel):
    symbol: str = "BTC-USDT"
    timeframe: str = "5m"
    strategy_name: str = "ema_cross"
    mode: str = "paper"  # paper | real
    strategy_params: dict = {}


@router.post("/start")
async def start_bot(
    req: BotStartRequest,
    session: AsyncSession = Depends(get_session),
) -> dict:
    # Check not already running
    bot_state = await session.get(BotState, 1)
    if bot_state and bot_state.is_running:
        raise HTTPException(status_code=409, detail="Bot is already running")

    if req.mode == "real":
        from app.config import get_settings
        s = get_settings()
        if not s.bingx_api_key or not s.bingx_api_secret:
            raise HTTPException(
                status_code=400,
                detail="BINGX_API_KEY and BINGX_API_SECRET required for real mode",
            )

    from app.tasks.bot_tasks import start_bot_task
    task = start_bot_task.apply_async(
        kwargs={
            "symbol": req.symbol,
            "timeframe": req.timeframe,
            "strategy_name": req.strategy_name,
            "mode": req.mode,
            "strategy_params": req.strategy_params,
        }
    )
    return {"status": "starting", "task_id": task.id, "mode": req.mode}


@router.post("/stop")
async def stop_bot(session: AsyncSession = Depends(get_session)) -> dict:
    await session.execute(
        update(BotState).where(BotState.id == 1).values(is_running=False)
    )
    # Also revoke Celery task if we had stored the task_id
    return {"status": "stop_requested"}


@router.get("/status")
async def get_bot_status(session: AsyncSession = Depends(get_session)) -> dict:
    bot_state = await session.get(BotState, 1)
    if not bot_state:
        return {
            "is_running": False,
            "mode": None,
            "symbol": None,
            "timeframe": None,
            "strategy_name": None,
            "last_heartbeat": None,
            "kill_switch_active": False,
            "kill_switch_reason": None,
        }
    return {
        "is_running": bot_state.is_running,
        "mode": bot_state.mode,
        "symbol": bot_state.symbol,
        "timeframe": bot_state.timeframe,
        "strategy_name": bot_state.strategy_name,
        "last_heartbeat": bot_state.last_heartbeat.isoformat() if bot_state.last_heartbeat else None,
        "kill_switch_active": bool(bot_state.kill_switch_reason),
        "kill_switch_reason": bot_state.kill_switch_reason,
    }
