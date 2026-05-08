"""Phase 30 A.10 — SHA-256 prompt integrity check.

Records expected hash of each agent's system prompt at "register" time;
verifies before each invocation. Tampering -> CRITICAL alert + run abort.

Prevents silent prompt drift (someone editing prompts at deploy without review).
"""
from __future__ import annotations

import hashlib
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def hash_prompt(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def register(agent_name: str, prompt: str) -> str:
    h = hash_prompt(prompt)
    try:
        from db import AI_DB_PATH, get_db_connection

        with get_db_connection(AI_DB_PATH) as conn:
            conn.execute(
                """INSERT OR REPLACE INTO prompt_hashes
                   (agent_name, prompt_sha256, recorded_at, last_verified_at)
                   VALUES (?, ?, datetime('now'), datetime('now'))""",
                (agent_name, h),
            )
            conn.commit()
    except Exception as e:
        logger.error(f"[PromptIntegrity] register failed: {e}")
    return h


def expected_hash(agent_name: str) -> Optional[str]:
    try:
        from db import AI_DB_PATH, get_db_connection

        with get_db_connection(AI_DB_PATH) as conn:
            cur = conn.execute(
                "SELECT prompt_sha256 FROM prompt_hashes WHERE agent_name = ?",
                (agent_name,),
            )
            row = cur.fetchone()
            return row[0] if row else None
    except Exception:
        return None


def verify(agent_name: str, prompt: str, raise_on_fail: bool = False) -> bool:
    actual = hash_prompt(prompt)
    expected = expected_hash(agent_name)
    if expected is None:
        register(agent_name, prompt)
        return True
    if actual == expected:
        try:
            from db import AI_DB_PATH, get_db_connection

            with get_db_connection(AI_DB_PATH) as conn:
                conn.execute(
                    "UPDATE prompt_hashes SET last_verified_at=datetime('now') WHERE agent_name=?",
                    (agent_name,),
                )
                conn.commit()
        except Exception:
            pass
        return True
    try:
        from db import AI_DB_PATH, get_db_connection

        with get_db_connection(AI_DB_PATH) as conn:
            conn.execute(
                """INSERT INTO prompt_integrity_violations
                   (agent_name, expected_hash, actual_hash) VALUES (?, ?, ?)""",
                (agent_name, expected, actual),
            )
            conn.commit()
    except Exception:
        pass
    msg = f"[PromptIntegrity] VIOLATION agent={agent_name} expected={expected[:16]} actual={actual[:16]}"
    logger.critical(msg)
    try:
        from severity_router import emit

        emit(kind="prompt.integrity_violation", severity="critical", message=msg,
             payload={"agent": agent_name, "expected": expected, "actual": actual})
    except Exception:
        pass
    if raise_on_fail:
        raise RuntimeError(msg)
    return False
