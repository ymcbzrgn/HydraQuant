"""Phase 30 A.17 — Hash-based LLM response cache.

Wraps LLM router invoke() with a deterministic SHA-256 cache_key over
(model, prompt, temperature, max_tokens). Cache hits short-circuit network call.

Cache lives in `llm_response_cache` SQLite table. TTL via PARAM:
- Stable (default 6h): regular trade signals.
- Long (24h): pure deterministic prompts (e.g., classification).
"""
from __future__ import annotations

import hashlib
import logging
import time
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

DEFAULT_TTL_SECONDS = 6 * 3600


def make_cache_key(model: str, prompt: str, temperature: float = 0.0, max_tokens: int = 0) -> str:
    h = hashlib.sha256()
    h.update(model.encode("utf-8"))
    h.update(b"\0")
    h.update(prompt.encode("utf-8", errors="ignore"))
    h.update(b"\0")
    h.update(f"t={temperature}".encode("utf-8"))
    h.update(b"\0")
    h.update(f"mt={max_tokens}".encode("utf-8"))
    return h.hexdigest()


def get_cached(cache_key: str, ttl_seconds: int = DEFAULT_TTL_SECONDS) -> Optional[Tuple[str, str]]:
    try:
        from db import AI_DB_PATH, get_db_connection

        with get_db_connection(AI_DB_PATH) as conn:
            cur = conn.execute(
                """SELECT response_text, model, created_at FROM llm_response_cache
                   WHERE cache_key = ?""",
                (cache_key,),
            )
            row = cur.fetchone()
            if not row:
                return None
            response_text, model, created_at = row
            try:
                from datetime import datetime as _dt
                created = _dt.fromisoformat(str(created_at).replace("Z", "+00:00"))
                age = (time.time() - created.timestamp())
                if age > ttl_seconds:
                    return None
            except Exception:
                pass
            conn.execute(
                """UPDATE llm_response_cache
                   SET hit_count = hit_count + 1,
                       last_hit_at = datetime('now')
                   WHERE cache_key = ?""",
                (cache_key,),
            )
            conn.commit()
            return response_text, model
    except Exception as e:
        logger.error(f"[LLMCache] get failed: {e}")
        return None


def put_cached(
    cache_key: str,
    model: str,
    response_text: str,
    tokens_in: int = 0,
    tokens_out: int = 0,
) -> None:
    try:
        from db import AI_DB_PATH, get_db_connection

        with get_db_connection(AI_DB_PATH) as conn:
            conn.execute(
                """INSERT OR REPLACE INTO llm_response_cache
                   (cache_key, model, response_text, tokens_in, tokens_out, hit_count, created_at, last_hit_at)
                   VALUES (?, ?, ?, ?, ?, 0, datetime('now'), datetime('now'))""",
                (cache_key, model, response_text, tokens_in, tokens_out),
            )
            conn.commit()
    except Exception as e:
        logger.error(f"[LLMCache] put failed: {e}")


def cleanup_stale(ttl_seconds: int = 7 * 24 * 3600) -> int:
    try:
        from db import AI_DB_PATH, get_db_connection

        with get_db_connection(AI_DB_PATH) as conn:
            cur = conn.execute(
                """DELETE FROM llm_response_cache
                   WHERE last_hit_at < datetime('now', ?)""",
                (f"-{ttl_seconds} seconds",),
            )
            conn.commit()
            return cur.rowcount
    except Exception as e:
        logger.error(f"[LLMCache] cleanup failed: {e}")
        return 0
