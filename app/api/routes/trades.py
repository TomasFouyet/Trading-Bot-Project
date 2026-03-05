"""
GET /trades — list trades with filters.
GET /trades/{trade_id} — single trade.
"""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy import select, desc
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_session
from app.db.models import Trade

router = APIRouter(prefix="/trades")


@router.get("")
async def list_trades(
    symbol: Optional[str] = Query(None),
    mode: Optional[str] = Query(None, description="paper | real | backtest"),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    session: AsyncSession = Depends(get_session),
) -> dict:
    q = select(Trade).order_by(desc(Trade.entry_ts)).limit(limit).offset(offset)
    if symbol:
        q = q.where(Trade.symbol == symbol)
    if mode:
        q = q.where(Trade.mode == mode)

    result = await session.execute(q)
    trades = result.scalars().all()

    return {
        "total": len(trades),
        "trades": [
            {
                "id": t.id,
                "symbol": t.symbol,
                "side": t.side,
                "mode": t.mode,
                "strategy_id": t.strategy_id,
                "entry_ts": t.entry_ts.isoformat(),
                "exit_ts": t.exit_ts.isoformat() if t.exit_ts else None,
                "entry_price": float(t.entry_price),
                "exit_price": float(t.exit_price) if t.exit_price else None,
                "qty": float(t.qty),
                "pnl": float(t.pnl) if t.pnl else None,
                "pnl_pct": float(t.pnl_pct) if t.pnl_pct else None,
                "fees": float(t.fees),
                "is_open": t.is_open,
            }
            for t in trades
        ],
    }


@router.get("/{trade_id}")
async def get_trade(
    trade_id: str,
    session: AsyncSession = Depends(get_session),
) -> dict:
    from fastapi import HTTPException
    trade = await session.get(Trade, trade_id)
    if not trade:
        raise HTTPException(status_code=404, detail=f"Trade {trade_id} not found")
    return {
        "id": trade.id,
        "symbol": trade.symbol,
        "side": trade.side,
        "mode": trade.mode,
        "entry_ts": trade.entry_ts.isoformat(),
        "exit_ts": trade.exit_ts.isoformat() if trade.exit_ts else None,
        "entry_price": float(trade.entry_price),
        "exit_price": float(trade.exit_price) if trade.exit_price else None,
        "qty": float(trade.qty),
        "pnl": float(trade.pnl) if trade.pnl else None,
        "pnl_pct": float(trade.pnl_pct) if trade.pnl_pct else None,
        "fees": float(trade.fees),
        "is_open": trade.is_open,
    }
