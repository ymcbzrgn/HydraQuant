"""Phase 30 B.12 — Memory flush before compaction.

Forces persistence of in-memory caches (rag_graph state, agent_pool history,
hippocampus pending writes) before context_compressor runs to ensure no
data is lost during compression replay.
"""
from __future__ import annotations

import logging
from typing import Callable, List

logger = logging.getLogger(__name__)

_FLUSH_HOOKS: List[Callable[[], None]] = []


def register_flush(cb: Callable[[], None]) -> None:
    _FLUSH_HOOKS.append(cb)


def flush_all() -> int:
    n_ok = 0
    for cb in _FLUSH_HOOKS:
        try:
            cb()
            n_ok += 1
        except Exception as e:
            logger.error(f"[MemoryFlush] hook failed: {e}")
    logger.info(f"[MemoryFlush] flushed {n_ok}/{len(_FLUSH_HOOKS)} hooks")
    return n_ok


def reset_for_tests() -> None:
    _FLUSH_HOOKS.clear()
