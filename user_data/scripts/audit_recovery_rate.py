"""Phase 30 A.8 — Unsuccessful decisions + recovery rate audit.

Tracks agent_pool round failures + recovery (later round succeeded).
Weekly report: per-agent failure-class distribution + recovery_rate.
"""
from __future__ import annotations

import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


def record_failure(
    pair: str,
    agent_name: str,
    round_idx: int,
    failure_class: str,
    failure_text: str,
) -> int:
    try:
        from db import AI_DB_PATH, get_db_connection

        with get_db_connection(AI_DB_PATH) as conn:
            cur = conn.execute(
                """INSERT INTO agent_pool_unsuccessful_decisions
                   (pair, agent_name, round_idx, failure_class, failure_text)
                   VALUES (?, ?, ?, ?, ?)""",
                (pair, agent_name, round_idx, failure_class, failure_text[:1000]),
            )
            conn.commit()
            return int(cur.lastrowid)
    except Exception as e:
        logger.error(f"[A.8] record_failure: {e}")
        return -1


def mark_recovered(failure_id: int) -> None:
    try:
        from db import AI_DB_PATH, get_db_connection

        with get_db_connection(AI_DB_PATH) as conn:
            conn.execute(
                "UPDATE agent_pool_unsuccessful_decisions SET recovered=1 WHERE id=?",
                (failure_id,),
            )
            conn.commit()
    except Exception:
        pass


def weekly_summary() -> Dict[str, Any]:
    from db import AI_DB_PATH, get_db_connection

    with get_db_connection(AI_DB_PATH) as conn:
        rows = conn.execute(
            """SELECT agent_name, failure_class,
                      COUNT(*) AS n,
                      SUM(recovered) AS recovered_n
               FROM agent_pool_unsuccessful_decisions
               WHERE ts >= datetime('now', '-7 days')
               GROUP BY agent_name, failure_class
               ORDER BY n DESC"""
        ).fetchall()
    out: Dict[str, Any] = {"per_agent_class": []}
    total_n = total_rec = 0
    for agent, klass, n, rec in rows:
        out["per_agent_class"].append({
            "agent": agent, "failure_class": klass, "n": n,
            "recovered": rec, "recovery_rate": round(rec / n, 4) if n else 0,
        })
        total_n += n
        total_rec += rec or 0
    out["overall_recovery_rate"] = round(total_rec / total_n, 4) if total_n else 0.0
    out["total_failures_7d"] = total_n
    return out
