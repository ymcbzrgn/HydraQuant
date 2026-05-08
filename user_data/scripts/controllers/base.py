"""Phase 30 C.1 — Base controller interface."""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class ControllerContext:
    """Common context object passed through the controller chain."""

    pair: str
    side: str = "long"  # 'long' | 'short'
    portfolio_value: float = 0.0
    open_positions: list = field(default_factory=list)
    market_regime: str = "unknown"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ControllerDecision:
    proceed: bool
    stake_amount: float = 0.0
    reason: str = ""
    blocked_by: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseController(ABC):
    """Abstract controller — subclasses implement decide()."""

    name: str = "base"

    @abstractmethod
    def decide(self, ctx: ControllerContext) -> ControllerDecision:
        ...

    def __repr__(self) -> str:
        return f"<Controller {self.name}>"
