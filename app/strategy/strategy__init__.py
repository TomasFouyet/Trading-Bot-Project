from app.strategy.ema_cross import EMACrossStrategy
from app.strategy.rsi_divergence import RSIDivergenceStrategy
from app.strategy.hybrid_rsi_pivot import HybridRSIPivotStrategy
from app.strategy.mean_reversion import MeanReversionStrategy
from app.strategy.trend_following import TrendFollowingStrategy
from app.strategy.trend_following_v2 import TrendFollowingV2
from app.strategy.event_driven import EventDrivenStrategy
from app.strategy.trendbot_mtf_v52 import TrendBotMTFv52Strategy
from app.strategy.base import BaseStrategy
from app.strategy.signals import Signal, SignalAction


STRATEGY_REGISTRY: dict[str, type[BaseStrategy]] = {
    "ema_cross":           EMACrossStrategy,
    "rsi_divergence":      RSIDivergenceStrategy,
    "hybrid_rsi_pivot":    HybridRSIPivotStrategy,
    "mean_reversion":      MeanReversionStrategy,
    "trend_following":     TrendFollowingStrategy,
    "trend_following_v2":  TrendFollowingV2,
    "event_driven":        EventDrivenStrategy,
    "trendbot_mtf_v52":    TrendBotMTFv52Strategy,
}


def get_strategy(name: str, symbol: str, params: dict | None = None) -> BaseStrategy:
    cls = STRATEGY_REGISTRY.get(name)
    if cls is None:
        raise ValueError(f"Unknown strategy: {name!r}. Available: {list(STRATEGY_REGISTRY)}")
    return cls(symbol=symbol, params=params or {})


__all__ = [
    "BaseStrategy", "Signal", "SignalAction",
    "EMACrossStrategy", "RSIDivergenceStrategy", "HybridRSIPivotStrategy",
    "MeanReversionStrategy", "TrendFollowingStrategy", "TrendFollowingV2",
    "EventDrivenStrategy",
    "get_strategy","TrendBotMTFv52Strategy",
]
