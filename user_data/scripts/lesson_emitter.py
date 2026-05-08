"""Phase 30 A.28 + A.9 — Single canonical lesson emitter.

All callers (rag_graph, agent_pool, schedulers) should funnel ai_lessons writes
through emit_lesson() to:
- Apply think_scrubber on lesson_text (no <think> leakage).
- INSERT OR IGNORE for (decision_id, pair) UNIQUE dedup.
- Log skipped duplicates without raising.
"""
from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def emit_lesson(
    decision_id: int,
    pair: str,
    signal: str,
    pnl: Optional[float],
    lesson_text: str,
) -> bool:
    """Returns True if inserted, False if duplicate / failed."""
    try:
        from think_scrubber import scrub
        lesson_text = scrub(lesson_text or "")
    except Exception:
        pass
    try:
        from db import AI_DB_PATH, get_db_connection

        with get_db_connection(AI_DB_PATH) as conn:
            cur = conn.execute(
                """INSERT OR IGNORE INTO ai_lessons
                   (decision_id, pair, signal, outcome_pnl, lesson_text)
                   VALUES (?, ?, ?, ?, ?)""",
                (decision_id, pair, signal, pnl, lesson_text),
            )
            inserted = cur.rowcount > 0
            conn.commit()
        if not inserted:
            logger.debug(f"[LessonDedup] skip duplicate decision_id={decision_id} pair={pair}")
        return inserted
    except Exception as e:
        logger.error(f"[LessonEmitter] failed: {e}")
        return False
