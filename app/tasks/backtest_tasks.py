"""
Celery tasks for running backtests asynchronously.
"""
from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone
from decimal import Decimal

from app.tasks.celery_app import celery_app
from app.core.logging import get_logger

logger = get_logger(__name__)


@celery_app.task(bind=True, name="tasks.run_backtest", max_retries=2)
def run_backtest_task(
    self,
    symbol: str,
    timeframe: str,
    start_iso: str,
    end_iso: str,
    strategy_name: str = "ema_cross",
    strategy_params: dict | None = None,
    initial_balance: float = 10000.0,
    commission_bps: float | None = None,
    slippage_bps: float | None = None,
    run_id: str | None = None,
) -> dict:
    """
    Run a full backtest and persist results to DB.
    Returns the BacktestRun dict.
    """
    run_id = run_id or str(uuid.uuid4())

    async def _run() -> dict:
        from app.config import get_settings
        from app.data.parquet_store import ParquetStore
        from app.db.session import AsyncSessionLocal
        from app.db.models import BacktestRun
        from app.engine.backtest import BacktestEngine
        from app.strategy import get_strategy

        settings = get_settings()

        start = datetime.fromisoformat(start_iso).replace(tzinfo=timezone.utc)
        end = datetime.fromisoformat(end_iso).replace(tzinfo=timezone.utc)

        strategy = get_strategy(strategy_name, symbol, strategy_params or {})
        store = ParquetStore()

        commission = Decimal(str(commission_bps)) / 10000 if commission_bps else None
        slippage = Decimal(str(slippage_bps)) / 10000 if slippage_bps else None

        engine = BacktestEngine(
            strategy=strategy,
            store=store,
            initial_balance=Decimal(str(initial_balance)),
            commission_rate=commission,
            slippage_rate=slippage,
        )

        # Mark as running
        async with AsyncSessionLocal() as session:
            run_record = BacktestRun(
                id=run_id,
                symbol=symbol,
                timeframe=timeframe,
                strategy_name=strategy_name,
                start_date=start,
                end_date=end,
                params={
                    "strategy_params": strategy_params or {},
                    "initial_balance": initial_balance,
                    "commission_bps": commission_bps,
                    "slippage_bps": slippage_bps,
                },
                status="running",
            )
            session.add(run_record)
            await session.commit()

        try:
            result = await engine.run(
                symbol=symbol,
                timeframe=timeframe,
                start=start,
                end=end,
                params=strategy_params,
            )
        except Exception as exc:
            async with AsyncSessionLocal() as session:
                from sqlalchemy import update
                await session.execute(
                    update(BacktestRun)
                    .where(BacktestRun.id == run_id)
                    .values(status="failed", report={"error": str(exc)})
                )
                await session.commit()
            raise

        # Persist results
        report = result.to_report()
        async with AsyncSessionLocal() as session:
            from sqlalchemy import update
            await session.execute(
                update(BacktestRun)
                .where(BacktestRun.id == run_id)
                .values(
                    status="completed",
                    total_trades=result.total_trades,
                    winning_trades=result.winning_trades,
                    losing_trades=result.losing_trades,
                    winrate=result.winrate,
                    total_pnl=result.total_pnl,
                    total_pnl_pct=result.total_pnl_pct,
                    max_drawdown_pct=result.max_drawdown_pct,
                    sharpe_ratio=result.sharpe_ratio,
                    avg_trade_pnl=result.avg_trade_pnl,
                    exposure_pct=result.exposure_pct,
                    report=report,
                )
            )
            await session.commit()

        logger.info(
            "backtest_task_done",
            run_id=run_id,
            trades=result.total_trades,
            pnl_pct=str(result.total_pnl_pct),
        )
        return {"run_id": run_id, "status": "completed", "metrics": report["metrics"]}

    try:
        return asyncio.run(_run())
    except Exception as exc:
        logger.error("backtest_task_failed", run_id=run_id, error=str(exc))
        raise self.retry(exc=exc, countdown=5)
