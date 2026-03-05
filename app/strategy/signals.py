"""
Signal types emitted by strategies.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Optional


class SignalAction(str, Enum):
    BUY = "BUY"       # Open or add to long
    SELL = "SELL"     # Open or add to short
    CLOSE = "CLOSE"   # Close current position
    HOLD = "HOLD"     # No action


@dataclass
class Signal:
    """
    A trading signal produced by a strategy.

    The engine + risk manager decide whether to act on it.
    """
    action: SignalAction
    symbol: str
    ts: datetime
    strategy_id: str
    confidence: float = 1.0          # 0.0 - 1.0, used for position sizing
    target_qty: Optional[Decimal] = None   # Explicit qty override
    stop_loss: Optional[Decimal] = None    # Absolute price for SL
    take_profit: Optional[Decimal] = None  # Absolute price for TP
    reason: str = ""                 # Human-readable reason for audit log
    meta: dict = field(default_factory=dict)  # Strategy-specific metadata

    def is_actionable(self) -> bool:
        return self.action not in (SignalAction.HOLD,)

    def __repr__(self) -> str:
        return (
            f"<Signal {self.action.value} {self.symbol} "
            f"@ {self.ts.isoformat()} conf={self.confidence:.2f} reason={self.reason!r}>"
        )
