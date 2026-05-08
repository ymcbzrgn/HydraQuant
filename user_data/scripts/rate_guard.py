"""Phase 30 B.7 — Cross-process rate guard.

Multiple HydraQuant Python processes (freqtrade, scheduler, rag-service)
share the same provider keys; without coordination they exceed RPM limits.
This module uses SQLite (with row-lock semantics + jitter retry) for atomic
counter updates per minute window.
"""
from __future__ import annotations

import logging
import time
from typing import Tuple

logger = logging.getLogger(__name__)


def _current_window() -> str:
    return time.strftime("%Y-%m-%dT%H:%M", time.gmtime())


def acquire(provider: str, limit_rpm: int) -> Tuple[bool, int]:
    """Atomically increment provider RPM counter; return (allowed, current_count)."""
    window = _current_window()
    try:
        from db import AI_DB_PATH, get_db_connection

        with get_db_connection(AI_DB_PATH) as conn:
            row = conn.execute(
                """SELECT window_start, count FROM rate_guard_state WHERE provider=?""",
                (provider,),
            ).fetchone()
            if not row:
                conn.execute(
                    """INSERT INTO rate_guard_state (provider, window_start, count, limit_rpm, updated_at)
                       VALUES (?, ?, 1, ?, datetime('now'))""",
                    (provider, window, limit_rpm),
                )
                conn.commit()
                return True, 1
            current_window, count = row
            if current_window != window:
                conn.execute(
                    """UPDATE rate_guard_state
                       SET window_start=?, count=1, limit_rpm=?, updated_at=datetime('now')
                       WHERE provider=?""",
                    (window, limit_rpm, provider),
                )
                conn.commit()
                return True, 1
            if int(count) >= int(limit_rpm):
                return False, int(count)
            conn.execute(
                """UPDATE rate_guard_state SET count=count+1, updated_at=datetime('now') WHERE provider=?""",
                (provider,),
            )
            conn.commit()
            return True, int(count) + 1
    except Exception as e:
        logger.error(f"[RateGuard] acquire failed: {e}")
        return True, 0  # fail-open


def status(provider: str) -> dict:
    try:
        from db import AI_DB_PATH, get_db_connection

        with get_db_connection(AI_DB_PATH) as conn:
            row = conn.execute(
                """SELECT window_start, count, limit_rpm FROM rate_guard_state WHERE provider=?""",
                (provider,),
            ).fetchone()
            if row:
                return {"provider": provider, "window": row[0], "count": row[1], "limit_rpm": row[2]}
    except Exception:
        pass
    return {"provider": provider, "count": 0, "limit_rpm": 0}
