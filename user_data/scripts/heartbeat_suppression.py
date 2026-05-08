"""Phase 30 A.5 — Cron heartbeat suppression.

Suppresses repetitive identical messages within an interval window so Telegram
operator-channel doesn't drown in noise (e.g. rag_health_unreachable each 5s).

Two-tier:
- exact_dedup: same (kind, message) within `window_sec` -> drop
- pattern_compress: count and emit summary every `summary_window_sec`
"""
from __future__ import annotations

import threading
import time
from collections import defaultdict
from typing import Dict, Optional, Tuple

DEFAULT_WINDOW_SEC = 300
DEFAULT_SUMMARY_WINDOW_SEC = 3600

_LAST: Dict[Tuple[str, str], float] = {}
_COUNTERS: Dict[Tuple[str, str], int] = defaultdict(int)
_BUCKET_START: Dict[Tuple[str, str], float] = {}
_LOCK = threading.RLock()


def should_emit(
    kind: str,
    message: str,
    window_sec: int = DEFAULT_WINDOW_SEC,
) -> bool:
    """Return True if this (kind, message) should be sent now."""
    key = (kind, message[:120])
    now = time.time()
    with _LOCK:
        last = _LAST.get(key, 0.0)
        if (now - last) < window_sec:
            _COUNTERS[key] += 1
            return False
        _LAST[key] = now
        _COUNTERS[key] = 0
    return True


def get_compressed_summary(
    kind: str,
    summary_window_sec: int = DEFAULT_SUMMARY_WINDOW_SEC,
) -> Optional[str]:
    """Return aggregate "kind X happened N times in last M minutes" or None."""
    now = time.time()
    parts = []
    with _LOCK:
        for (k, msg), n in list(_COUNTERS.items()):
            if k != kind or n < 2:
                continue
            start = _BUCKET_START.get((k, msg), now - summary_window_sec)
            if (now - start) >= summary_window_sec:
                parts.append(f"{msg[:40]} x{n+1}")
                _COUNTERS[(k, msg)] = 0
                _BUCKET_START[(k, msg)] = now
    if not parts:
        return None
    return f"[{kind} compressed] " + " | ".join(parts[:5])


def reset_for_tests() -> None:
    with _LOCK:
        _LAST.clear()
        _COUNTERS.clear()
        _BUCKET_START.clear()
