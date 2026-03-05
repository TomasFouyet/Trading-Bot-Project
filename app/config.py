"""
Central configuration via Pydantic Settings.
All secrets come from environment variables or .env file.
NEVER log API keys.
"""
from __future__ import annotations

from decimal import Decimal
from enum import Enum
from functools import lru_cache

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class BotMode(str, Enum):
    PAPER = "paper"
    REAL = "real"
    BACKTEST = "backtest"


class MarketType(str, Enum):
    SWAP = "swap"   # perpetual futures
    SPOT = "spot"


class LogFormat(str, Enum):
    JSON = "json"
    CONSOLE = "console"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── BingX ──────────────────────────────────────────────
    bingx_api_key: str = Field(default="", repr=False)
    bingx_api_secret: str = Field(default="", repr=False)
    bingx_base_url: str = "https://open-api.bingx.com"
    bingx_market_type: MarketType = MarketType.SWAP

    # ── Database ────────────────────────────────────────────
    database_url: str = "postgresql+asyncpg://trading:trading@localhost:5432/trading"
    database_pool_size: int = 10
    database_max_overflow: int = 20

    # ── Redis / Celery ──────────────────────────────────────
    redis_url: str = "redis://localhost:6379/0"
    celery_broker_url: str = "redis://localhost:6379/0"
    celery_result_backend: str = "redis://localhost:6379/1"

    # ── Storage ─────────────────────────────────────────────
    parquet_dir: str = "./data/parquet"

    # ── Bot defaults ────────────────────────────────────────
    bot_mode: BotMode = BotMode.PAPER
    default_symbol: str = "BTC-USDT"
    default_timeframe: str = "5m"
    leverage: int = Field(default=1, ge=1, le=125)  # 1 = no leverage

    # ── Risk manager ────────────────────────────────────────
    risk_max_daily_drawdown_pct: Decimal = Decimal("2.0")
    risk_max_position_pct: Decimal = Decimal("5.0")
    risk_max_trade_risk_pct: Decimal = Decimal("1.0")
    risk_max_consecutive_api_errors: int = 5
    risk_data_delay_threshold_s: int = 60

    # ── Costs ───────────────────────────────────────────────
    commission_bps: Decimal = Decimal("7.5")  # 0.075%
    slippage_bps: Decimal = Decimal("5.0")    # 0.05%

    # ── Observability ───────────────────────────────────────
    log_level: str = "INFO"
    log_format: LogFormat = LogFormat.JSON
    prometheus_port: int = 8001
    environment: str = "development"

    @model_validator(mode="after")
    def _warn_missing_keys(self) -> "Settings":
        if self.bot_mode == BotMode.REAL:
            assert self.bingx_api_key, "BINGX_API_KEY required for real trading"
            assert self.bingx_api_secret, "BINGX_API_SECRET required for real trading"
        return self

    @property
    def commission_rate(self) -> Decimal:
        return self.commission_bps / Decimal("10000")

    @property
    def slippage_rate(self) -> Decimal:
        return self.slippage_bps / Decimal("10000")

    def safe_repr(self) -> dict:
        """Return config dict safe to log (no secrets)."""
        d = self.model_dump()
        d["bingx_api_key"] = "***" if d["bingx_api_key"] else ""
        d["bingx_api_secret"] = "***" if d["bingx_api_secret"] else ""
        return d


@lru_cache
def get_settings() -> Settings:
    return Settings()
