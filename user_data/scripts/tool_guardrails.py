"""Phase 30 B.8 — Tool-loop guardrails MADAM (3-pattern detection).

Patterns:
- exact_failure: same tool + same input -> same error 3 times.
- same_tool_failure: same tool failed N times in a row.
- no_progress: tool returned same output 3+ times.

Triggers: log + DB persist + force-stop run when threshold breached.
"""
from __future__ import annotations

import hashlib
import logging
import threading
import time
from collections import defaultdict, deque
from typing import Any, Deque, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

DEFAULT_THRESHOLD = 3
DEFAULT_WINDOW_SEC = 600


class ToolGuardrails:
    def __init__(self, threshold: int = DEFAULT_THRESHOLD, window_sec: int = DEFAULT_WINDOW_SEC):
        self.threshold = int(threshold)
        self.window = int(window_sec)
        self._failure_history: Dict[str, Deque[Tuple[float, str, str]]] = defaultdict(
            lambda: deque(maxlen=20)
        )
        self._success_history: Dict[str, Deque[Tuple[float, str]]] = defaultdict(
            lambda: deque(maxlen=20)
        )
        self._lock = threading.RLock()

    @staticmethod
    def _hash_input(input_repr: str) -> str:
        return hashlib.sha1(input_repr.encode("utf-8", errors="ignore")).hexdigest()[:16]

    def record_failure(self, tool_name: str, input_repr: str, error_text: str) -> Optional[str]:
        now = time.time()
        cutoff = now - self.window
        in_hash = self._hash_input(input_repr)
        err_hash = hashlib.sha1((error_text or "")[:200].encode()).hexdigest()[:16]
        with self._lock:
            dq = self._failure_history[tool_name]
            dq.append((now, in_hash, err_hash))
            while dq and dq[0][0] < cutoff:
                dq.popleft()
            same_tool_count = len(dq)
            same_input_same_err = sum(
                1 for t, ih, eh in dq if ih == in_hash and eh == err_hash
            )
        if same_input_same_err >= self.threshold:
            return self._fire("exact_failure", tool_name, {
                "input_hash": in_hash, "err_hash": err_hash, "count": same_input_same_err,
            })
        if same_tool_count >= self.threshold * 2:
            return self._fire("same_tool_failure", tool_name, {"count": same_tool_count})
        return None

    def record_success(self, tool_name: str, output_repr: str) -> Optional[str]:
        out_hash = self._hash_input(output_repr)
        now = time.time()
        cutoff = now - self.window
        with self._lock:
            dq = self._success_history[tool_name]
            dq.append((now, out_hash))
            while dq and dq[0][0] < cutoff:
                dq.popleft()
            same = sum(1 for t, h in dq if h == out_hash)
        if same >= self.threshold:
            return self._fire("no_progress", tool_name, {"output_hash": out_hash, "count": same})
        return None

    def _fire(self, pattern: str, tool: str, ctx: Dict[str, Any]) -> str:
        msg = f"[ToolGuardrails] FIRE pattern={pattern} tool={tool} ctx={ctx}"
        logger.warning(msg)
        try:
            from db import AI_DB_PATH, get_db_connection
            import json

            with get_db_connection(AI_DB_PATH) as conn:
                conn.execute(
                    """INSERT INTO tool_loop_events (pattern, context_json) VALUES (?, ?)""",
                    (f"{pattern}:{tool}", json.dumps(ctx, default=str)),
                )
                conn.commit()
        except Exception:
            pass
        try:
            from severity_router import emit
            emit(kind=f"tool.{pattern}", severity="warn", message=msg, payload={"tool": tool, **ctx})
        except Exception:
            pass
        return pattern

    def reset_for_tests(self) -> None:
        with self._lock:
            self._failure_history.clear()
            self._success_history.clear()


_GLOBAL = ToolGuardrails()
