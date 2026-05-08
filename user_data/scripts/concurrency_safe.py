"""Phase 30 A.3 — Per-tool ConcurrencySafe etiketi for AGENT_REGISTRY.

Marks each agent / tool with a `concurrency_safe` flag:
- True  -> can run in agent_pool Round 1 paralel batch
- False -> must run serially after paralel batch (mutates shared state)

Provides agent_partition() that splits a list of (agent_name, fn) into
(parallel_safe, serial_only) lists for agent_pool.run_debate().
"""
from __future__ import annotations

import logging
from typing import Callable, Dict, List, Tuple

logger = logging.getLogger(__name__)

# Default registry: well-known mutating tools must be serial.
_DEFAULT_SAFE: Dict[str, bool] = {
    # Pure-read / pure-compute
    "TrendAgent": True,
    "MomentumAgent": True,
    "CrowdAgent": True,
    "EvidenceAgent": True,
    "MacroAgent": True,
    "RiskAgent": True,
    "RegimeAgent": True,
    "FundingAgent": True,
    "OnchainAgent": True,
    "SentimentAgent": True,
    # Stateful / writes shared state
    "MagmaWriter": False,
    "MagmaPromoter": False,
    "HippocampusWriter": False,
    "BayesianKellyUpdater": False,
    "AutonomyPromoter": False,
}

_REGISTRY: Dict[str, bool] = dict(_DEFAULT_SAFE)


def mark_safe(name: str, is_safe: bool) -> None:
    _REGISTRY[name] = bool(is_safe)


def is_concurrency_safe(name: str) -> bool:
    return bool(_REGISTRY.get(name, False))


def partition(items: List[Tuple[str, Callable]]) -> Tuple[List, List]:
    parallel_safe: List[Tuple[str, Callable]] = []
    serial_only: List[Tuple[str, Callable]] = []
    for name, fn in items:
        if is_concurrency_safe(name):
            parallel_safe.append((name, fn))
        else:
            serial_only.append((name, fn))
    return parallel_safe, serial_only
