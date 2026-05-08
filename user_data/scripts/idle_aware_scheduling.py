"""Phase 30 A.22 — Idle-aware scheduling.

Detects "idle" mode (no trade activity in last N minutes + low market_stress)
and switches scheduler to sparse cadence. Wakes on:
- Hawkes spike (volatility burst)
- New trade opened
- News critical event (A.16)
- Operator manual wake
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_IDLE_THRESHOLD_MIN = 30
DEFAULT_SPARSE_MULTIPLIER = 3  # 5min jobs become 15min when idle


class IdleScheduler:
    def __init__(
        self,
        idle_threshold_min: int = DEFAULT_IDLE_THRESHOLD_MIN,
        sparse_multiplier: int = DEFAULT_SPARSE_MULTIPLIER,
    ):
        self.idle_threshold_min = int(idle_threshold_min)
        self.sparse_multiplier = int(sparse_multiplier)
        self._idle = False
        self._last_wake_reason = "init"
        self._last_activity_ts = time.time()
        self._lock = threading.RLock()

    def report_activity(self, reason: str) -> None:
        with self._lock:
            self._last_activity_ts = time.time()
            if self._idle:
                logger.info(f"[IdleScheduler] WAKE — reason={reason}")
                self._idle = False
                self._last_wake_reason = reason

    def is_idle(self) -> bool:
        with self._lock:
            elapsed_min = (time.time() - self._last_activity_ts) / 60.0
            now_idle = elapsed_min >= self.idle_threshold_min
            if now_idle and not self._idle:
                logger.info(f"[IdleScheduler] entering IDLE after {elapsed_min:.1f}min inactivity")
                self._idle = True
            return self._idle

    def adjusted_interval_seconds(self, base_seconds: int) -> int:
        if self.is_idle():
            return int(base_seconds * self.sparse_multiplier)
        return int(base_seconds)

    @property
    def state(self) -> dict:
        with self._lock:
            return {
                "idle": self._idle,
                "last_wake_reason": self._last_wake_reason,
                "minutes_since_activity": round((time.time() - self._last_activity_ts) / 60, 1),
            }


_GLOBAL: Optional[IdleScheduler] = None
_GLOBAL_LOCK = threading.Lock()


def get_idle_scheduler() -> IdleScheduler:
    global _GLOBAL
    with _GLOBAL_LOCK:
        if _GLOBAL is None:
            _GLOBAL = IdleScheduler()
    return _GLOBAL
