"""
Celery application factory.
"""
from __future__ import annotations

from celery import Celery

from app.config import get_settings

_settings = get_settings()

celery_app = Celery(
    "trading_bot",
    broker=_settings.celery_broker_url,
    backend=_settings.celery_result_backend,
    include=[
        "app.tasks.ingest",
        "app.tasks.backtest_tasks",
        "app.tasks.bot_tasks",
    ],
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    task_routes={
        "app.tasks.ingest.*": {"queue": "data"},
        "app.tasks.backtest_tasks.*": {"queue": "compute"},
        "app.tasks.bot_tasks.*": {"queue": "trading"},
    },
)
