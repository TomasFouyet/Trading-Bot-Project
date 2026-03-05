"""
Domain exceptions for the trading bot.
"""
from __future__ import annotations


class TradingBotError(Exception):
    """Base exception."""


class BrokerError(TradingBotError):
    """Error communicating with the broker."""


class RateLimitError(BrokerError):
    """API rate limit exceeded."""


class AuthenticationError(BrokerError):
    """Invalid API credentials."""


class InsufficientFundsError(BrokerError):
    """Not enough balance to place order."""


class OrderNotFoundError(BrokerError):
    """Order ID does not exist."""


class RiskViolationError(TradingBotError):
    """Risk manager rejected the order."""


class KillSwitchTriggered(TradingBotError):
    """Kill switch activated. Bot stopped."""


class DataFeedError(TradingBotError):
    """Problem with market data feed."""


class DataDelayError(DataFeedError):
    """Data is too stale."""


class BacktestError(TradingBotError):
    """Error running backtest."""


class ConfigurationError(TradingBotError):
    """Invalid configuration."""
