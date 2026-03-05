"""
Structured logging setup with structlog.
All log records are JSON in production, colorful console in dev.
"""
from __future__ import annotations

import logging
import sys
import uuid

import structlog
from structlog.types import EventDict, WrappedLogger


def add_trace_id(logger: WrappedLogger, method: str, event_dict: EventDict) -> EventDict:
    """Inject a trace_id if not already present."""
    if "trace_id" not in event_dict:
        event_dict["trace_id"] = str(uuid.uuid4())[:8]
    return event_dict


def drop_color_message_key(logger: WrappedLogger, method: str, event_dict: EventDict) -> EventDict:
    """Remove uvicorn's color_message to reduce noise."""
    event_dict.pop("color_message", None)
    return event_dict


def configure_logging(log_level: str = "INFO", log_format: str = "json") -> None:
    log_level_int = getattr(logging, log_level.upper(), logging.INFO)

    shared_processors: list = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        add_trace_id,
        drop_color_message_key,
    ]

    if log_format == "json":
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    structlog.configure(
        processors=shared_processors
        + [
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers = [handler]
    root_logger.setLevel(log_level_int)

    # Silence noisy libs
    for lib in ("httpx", "httpcore", "asyncio", "celery.utils.functional"):
        logging.getLogger(lib).setLevel(logging.WARNING)


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)
