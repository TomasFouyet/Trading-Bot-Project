"""
GET /metrics/summary — PnL, drawdown, win rate, exposure summary.
GET /metrics/prometheus — raw Prometheus metrics (for scraping).
"""
from __future__ import annotations

from decimal import Decimal
from typing import Optional

from fastapi import APIRouter, Depends, Query, Response
from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_session
from app.db.models import Trade

router = APIRouter(prefix="/metrics")


@router.get("/summary")
async def get_metrics_summary(
    symbol: Optional[str] = Query(None),
    mode: Optional[str] = Query("paper"),
    session: AsyncSession = Depends(get_session),
) -> dict:
    """
    Compute PnL, drawdown, win rate, and other metrics from the trades table.
    """
    q = select(Trade).where(Trade.is_open == False)  # noqa: E712
    if symbol:
        q = q.where(Trade.symbol == symbol)
    if mode:
        q = q.where(Trade.mode == mode)

    result = await session.execute(q)
    trades = result.scalars().all()

    if not trades:
        return {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "winrate": 0.0,
            "total_pnl": 0.0,
            "avg_trade_pnl": 0.0,
            "max_drawdown_pct": 0.0,
            "total_fees": 0.0,
            "message": "No completed trades",
        }

    pnls = [float(t.pnl) for t in trades if t.pnl is not None]
    winning = [p for p in pnls if p > 0]
    losing = [p for p in pnls if p <= 0]
    total_fees = sum(float(t.fees) for t in trades)

    # Equity curve (cumulative PnL)
    cumulative = 0.0
    peak = 0.0
    max_dd = 0.0
    for p in pnls:
        cumulative += p
        if cumulative > peak:
            peak = cumulative
        dd = (peak - cumulative) / peak * 100 if peak > 0 else 0
        max_dd = max(max_dd, dd)

    winrate = len(winning) / len(pnls) * 100 if pnls else 0

    return {
        "total_trades": len(trades),
        "winning_trades": len(winning),
        "losing_trades": len(losing),
        "winrate": round(winrate, 2),
        "total_pnl": round(sum(pnls), 4),
        "avg_trade_pnl": round(sum(pnls) / len(pnls), 4) if pnls else 0,
        "best_trade": round(max(pnls), 4) if pnls else 0,
        "worst_trade": round(min(pnls), 4) if pnls else 0,
        "max_drawdown_pct": round(max_dd, 2),
        "total_fees": round(total_fees, 4),
        "symbol": symbol,
        "mode": mode,
    }


@router.get("/prometheus")
async def prometheus_metrics() -> Response:
    """Expose Prometheus metrics in text format."""
    from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
