"""Phase 30 B.9 — Error classification taxonomy.

Maps Python exceptions to {retryable, permanent, unknown} buckets used by
LLM router, RAG client, exchange API. Persists hits to error_taxonomy_log
for offline analysis (which providers fail in which way).
"""
from __future__ import annotations

import logging
from typing import Tuple

logger = logging.getLogger(__name__)

RETRYABLE = {
    "TimeoutError", "ConnectionError", "ConnectionResetError",
    "ConnectionRefusedError", "ReadTimeout", "ConnectTimeout",
    "RateLimitError", "ServiceUnavailable", "InternalServerError",
    "APIConnectionError", "APIError", "ChunkedEncodingError",
    "ProtocolError", "RemoteDisconnected", "IncompleteRead",
    "OperationalError",  # SQLite locked etc.
    "BusyLoadingError",  # Redis equivalents
}
PERMANENT = {
    "KeyError", "TypeError", "ValueError", "AttributeError",
    "AuthenticationError", "PermissionError", "NotFoundError",
    "InvalidRequestError", "ValidationError",
    "FileNotFoundError",
}


def classify(exc: BaseException) -> Tuple[str, bool, bool]:
    name = type(exc).__name__
    if name in RETRYABLE:
        label = "retryable"
        retryable, permanent = True, False
    elif name in PERMANENT:
        label = "permanent"
        retryable, permanent = False, True
    else:
        msg = str(exc).lower()
        if any(k in msg for k in ("timeout", "timed out", "connection", "rate limit", "503", "429", "502")):
            label = "retryable"
            retryable, permanent = True, False
        elif any(k in msg for k in ("invalid", "unauthorized", "forbidden", "not found", "401", "403", "404")):
            label = "permanent"
            retryable, permanent = False, True
        else:
            label = "unknown"
            retryable, permanent = False, False
    _persist(name, label, retryable, permanent)
    return label, retryable, permanent


def _persist(error_class: str, label: str, retryable: bool, permanent: bool) -> None:
    try:
        from db import AI_DB_PATH, get_db_connection

        with get_db_connection(AI_DB_PATH) as conn:
            cur = conn.execute(
                "SELECT id, count FROM error_taxonomy_log WHERE error_class=? AND taxonomy_label=?",
                (error_class, label),
            )
            row = cur.fetchone()
            if row:
                conn.execute(
                    "UPDATE error_taxonomy_log SET count=count+1, last_seen=datetime('now') WHERE id=?",
                    (row[0],),
                )
            else:
                conn.execute(
                    """INSERT INTO error_taxonomy_log
                       (error_class, taxonomy_label, retryable, permanent, count)
                       VALUES (?, ?, ?, ?, 1)""",
                    (error_class, label, int(retryable), int(permanent)),
                )
            conn.commit()
    except Exception:
        pass
