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
    BUY = "BUY"             # Open or add to long
    SELL = "SELL"           # Open or add to short
    CLOSE = "CLOSE"         # Close current position (100%)
    PARTIAL_CLOSE = "PARTIAL_CLOSE"  # Close a fraction of the position
    HOLD = "HOLD"           # No action


@dataclass
class Signal:
    """
    A trading signal produced by a strategy.

    The engine + risk manager decide whether to act on it.

    For PARTIAL_CLOSE signals:
      - meta["close_pct"] indicates the fraction to close (e.g. 0.70 = 70%)
      - Remaining position stays open with updated SL/TP
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

    @property
    def close_pct(self) -> float:
        """Fraction of position to close (for PARTIAL_CLOSE). Default 1.0."""
        if self.action == SignalAction.PARTIAL_CLOSE:
            return float(self.meta.get("close_pct", 0.70))
        return 1.0

    def __repr__(self) -> str:
        extra = ""
        if self.action == SignalAction.PARTIAL_CLOSE:
            extra = f" close_pct={self.close_pct:.0%}"
        return (
            f"<Signal {self.action.value} {self.symbol} "
            f"@ {self.ts.isoformat()} conf={self.confidence:.2f}{extra} reason={self.reason!r}>"
        )
