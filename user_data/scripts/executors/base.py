"""Phase 30 C.1 — Base executor."""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class OrderRequest:
    pair: str
    side: str
    stake_amount: float
    rate: float
    order_type: str = "limit"
    leverage: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OrderResult:
    success: bool
    order_id: Optional[str] = None
    filled_amount: float = 0.0
    avg_price: float = 0.0
    error: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseExecutor(ABC):
    name: str = "base"

    @abstractmethod
    def submit(self, req: OrderRequest) -> OrderResult:
        ...
