from app.strategy.ema_cross import EMACrossStrategy
from app.strategy.rsi_divergence import RSIDivergenceStrategy
from app.strategy.base import BaseStrategy
from app.strategy.signals import Signal, SignalAction

STRATEGY_REGISTRY: dict[str, type[BaseStrategy]] = {
    "ema_cross": EMACrossStrategy,
    "rsi_divergence": RSIDivergenceStrategy,
}


def get_strategy(name: str, symbol: str, params: dict | None = None) -> BaseStrategy:
    cls = STRATEGY_REGISTRY.get(name)
    if cls is None:
        raise ValueError(f"Unknown strategy: {name!r}. Available: {list(STRATEGY_REGISTRY)}")
    return cls(symbol=symbol, params=params or {})


__all__ = [
    "BaseStrategy", "Signal", "SignalAction",
    "EMACrossStrategy", "RSIDivergenceStrategy",
    "get_strategy",
]
