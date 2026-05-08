"""Phase 30 D.3 — Strategy marketplace skeleton.

Operator-driven runtime add/remove of agent personas + strategy modules.
- register_strategy(name, module_path, weight)
- approve_strategy(name) — requires backtest verify gate (D.X) + shadow promotion
- list_active() / list_pending()
"""
from __future__ import annotations

import importlib
import json
import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


@dataclass
class StrategyEntry:
    name: str
    module_path: str
    weight: float = 0.0
    status: str = "pending"  # pending / approved / disabled
    metadata: Dict[str, Any] = field(default_factory=dict)


_REGISTRY: Dict[str, StrategyEntry] = {}
_LOCK = threading.RLock()


def register(name: str, module_path: str, weight: float = 0.0,
             metadata: Dict[str, Any] = None) -> StrategyEntry:
    with _LOCK:
        entry = StrategyEntry(name=name, module_path=module_path,
                              weight=float(weight),
                              metadata=metadata or {})
        _REGISTRY[name] = entry
    return entry


def approve(name: str, dry_run: bool = True) -> bool:
    with _LOCK:
        entry = _REGISTRY.get(name)
        if not entry:
            return False
        try:
            importlib.import_module(entry.module_path)
        except Exception as e:
            logger.error(f"[Marketplace] cannot import {entry.module_path}: {e}")
            return False
        if not dry_run:
            entry.status = "approved"
        return True


def disable(name: str) -> bool:
    with _LOCK:
        if name in _REGISTRY:
            _REGISTRY[name].status = "disabled"
            return True
    return False


def list_active() -> List[StrategyEntry]:
    with _LOCK:
        return [e for e in _REGISTRY.values() if e.status == "approved"]


def list_pending() -> List[StrategyEntry]:
    with _LOCK:
        return [e for e in _REGISTRY.values() if e.status == "pending"]


def export_state() -> str:
    with _LOCK:
        return json.dumps(
            [{
                "name": e.name, "module": e.module_path,
                "weight": e.weight, "status": e.status,
                "metadata": e.metadata,
            } for e in _REGISTRY.values()],
            default=str, indent=2,
        )
