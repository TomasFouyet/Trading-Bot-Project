"""
Strategy registry — maps short names to strategy classes.

This is the single source of truth for which strategies the validation
framework can run. Add new strategies here.

Usage:
    from validation.registry import get_strategy_class, list_strategies

    cls = get_strategy_class("trend_v2")
    print(list_strategies())
"""
from __future__ import annotations

from typing import Type

from app.strategy.base import BaseStrategy

# Lazy imports to avoid loading all strategies at module level
STRATEGY_REGISTRY: dict[str, tuple[str, str]] = {
    # name → (module_path, class_name)
    "trend_v2":     ("app.strategy.trend_following_v2",  "TrendFollowingV2"),
    "trend_v2_simple": ("app.strategy.trend_following_v2_simple", "TrendFollowingV2Simple"),
    "trend_v1":     ("app.strategy.trend_following",     "TrendFollowingStrategy"),
    "trendbot_mtf": ("app.strategy.trendbot_mtf_v52",   "TrendBotMTFv52Strategy"),
    "ema_cross":    ("app.strategy.ema_cross",           "EMACrossStrategy"),
    "rsi_div":      ("app.strategy.rsi_divergence",      "RSIDivergenceStrategy"),
    "mean_rev":     ("app.strategy.mean_reversion",      "MeanReversionStrategy"),
    "event":        ("app.strategy.event_driven",        "EventDrivenStrategy"),
}


def get_strategy_class(name: str) -> Type[BaseStrategy]:
    """Resolve strategy name to class. Raises KeyError if not found."""
    if name not in STRATEGY_REGISTRY:
        available = ", ".join(sorted(STRATEGY_REGISTRY.keys()))
        raise KeyError(f"Unknown strategy '{name}'. Available: {available}")

    module_path, class_name = STRATEGY_REGISTRY[name]
    import importlib
    mod = importlib.import_module(module_path)
    cls = getattr(mod, class_name)
    return cls


def list_strategies() -> list[str]:
    """Return sorted list of available strategy names."""
    return sorted(STRATEGY_REGISTRY.keys())
