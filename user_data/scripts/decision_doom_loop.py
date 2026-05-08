"""Phase 30 A.6 — Doom loop detector.

Detects when the same decision_hash repeats N times in a sliding window
(e.g. agent stuck producing identical NEUTRAL output, or trade re-attempts
after rejection). Emits CRITICAL after threshold; persists to doom_loop_events.

decision_hash = sha256(pair + signal + entry_bucket + reasoning_summary[:200]).
"""
from __future__ import annotations

import hashlib
import logging
import threading
import time
from collections import deque
from typing import Deque, Dict, Optional

logger = logging.getLogger(__name__)

DEFAULT_THRESHOLD = 5
DEFAULT_WINDOW_SEC = 1800

_HISTORY: Dict[str, Deque[tuple]] = {}
_LOCK = threading.RLock()


def hash_decision(pair: str, signal: str, entry_price: Optional[float], reasoning: str = "") -> str:
    bucket = f"{entry_price:.4f}" if entry_price else "0"
    h = hashlib.sha256()
    h.update(pair.encode())
    h.update(b"|")
    h.update((signal or "").encode())
    h.update(b"|")
    h.update(bucket.encode())
    h.update(b"|")
    h.update((reasoning or "")[:200].encode())
    return h.hexdigest()[:32]


def record(
    pair: str,
    decision_hash: str,
    threshold: int = DEFAULT_THRESHOLD,
    window_sec: int = DEFAULT_WINDOW_SEC,
) -> Optional[Dict]:
    """Record a decision; return event dict if threshold breached."""
    now = time.time()
    cutoff = now - window_sec
    with _LOCK:
        dq = _HISTORY.setdefault(pair, deque(maxlen=200))
        dq.append((decision_hash, now))
        while dq and dq[0][1] < cutoff:
            dq.popleft()
        same = sum(1 for h, _ in dq if h == decision_hash)
        if same >= threshold:
            ev = {
                "pair": pair,
                "decision_hash": decision_hash,
                "consecutive_count": same,
                "window_start": dq[0][1],
                "window_end": now,
                "severity": "critical" if same >= threshold * 2 else "warn",
            }
            try:
                from db import AI_DB_PATH, get_db_connection

                with get_db_connection(AI_DB_PATH) as conn:
                    conn.execute(
                        """INSERT INTO doom_loop_events
                           (pair, decision_hash, consecutive_count,
                            window_start, window_end, severity)
                           VALUES (?, ?, ?, datetime(?, 'unixepoch'), datetime(?, 'unixepoch'), ?)""",
                        (pair, decision_hash, same, ev["window_start"], ev["window_end"], ev["severity"]),
                    )
                    conn.commit()
            except Exception:
                pass
            try:
                from severity_router import emit

                emit(
                    kind="decision.doom_loop",
                    severity=ev["severity"],
                    message=f"{pair} same decision x{same} in {window_sec}s",
                    payload=ev,
                )
            except Exception:
                pass
            return ev
    return None


def reset_for_tests() -> None:
    with _LOCK:
        _HISTORY.clear()
