"""Phase 30 B.5 — Adaptive concurrency LLM router.

Per-provider concurrency window auto-tunes:
- Latency p95 rising -> reduce concurrency (back-pressure).
- Latency p95 stable + queue idle -> raise concurrency.

State persisted in provider_capabilities.current_concurrency.
"""
from __future__ import annotations

import logging
import threading
import time
from collections import deque
from typing import Deque, Dict

logger = logging.getLogger(__name__)

DEFAULT_MIN_CONCURRENCY = 1
DEFAULT_MAX_CONCURRENCY = 8
DEFAULT_TARGET_P95_MS = 12_000
SAMPLE_WINDOW = 50


class AdaptiveConcurrency:
    def __init__(
        self,
        min_concurrency: int = DEFAULT_MIN_CONCURRENCY,
        max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
        target_p95_ms: int = DEFAULT_TARGET_P95_MS,
    ):
        self.min_c = int(min_concurrency)
        self.max_c = int(max_concurrency)
        self.target_p95 = int(target_p95_ms)
        self._latencies: Dict[str, Deque[int]] = {}
        self._concurrency: Dict[str, int] = {}
        self._lock = threading.RLock()

    def record(self, provider: str, latency_ms: int, success: bool = True) -> None:
        with self._lock:
            dq = self._latencies.setdefault(provider, deque(maxlen=SAMPLE_WINDOW))
            dq.append(int(latency_ms))
            self._maybe_adjust(provider)

    def _maybe_adjust(self, provider: str) -> None:
        dq = self._latencies.get(provider)
        if not dq or len(dq) < 10:
            return
        sorted_lat = sorted(dq)
        p95 = sorted_lat[int(0.95 * (len(sorted_lat) - 1))]
        current = self._concurrency.get(provider, 2)
        if p95 > self.target_p95 * 1.5:
            new = max(self.min_c, current - 1)
        elif p95 < self.target_p95 * 0.6:
            new = min(self.max_c, current + 1)
        else:
            new = current
        if new != current:
            self._concurrency[provider] = new
            logger.info(
                f"[AdaptiveConcurrency] {provider} {current} -> {new} (p95={p95}ms target={self.target_p95}ms)"
            )
            self._persist(provider, new)

    def _persist(self, provider: str, current: int) -> None:
        try:
            from db import AI_DB_PATH, get_db_connection

            with get_db_connection(AI_DB_PATH) as conn:
                conn.execute(
                    """INSERT INTO provider_capabilities (model, current_concurrency, target_concurrency, updated_at)
                       VALUES (?, ?, ?, datetime('now'))
                       ON CONFLICT(model) DO UPDATE SET
                           current_concurrency=excluded.current_concurrency,
                           updated_at=datetime('now')""",
                    (provider, current, current),
                )
                conn.commit()
        except Exception:
            pass

    def get_concurrency(self, provider: str) -> int:
        with self._lock:
            return self._concurrency.get(provider, 2)

    def reset_for_tests(self) -> None:
        with self._lock:
            self._latencies.clear()
            self._concurrency.clear()


_GLOBAL = AdaptiveConcurrency()
