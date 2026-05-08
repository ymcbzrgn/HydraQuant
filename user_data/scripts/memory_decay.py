"""Phase 30 B.11 (1/2) — Temporal-decay memory weights.

Every recall computes weight = exp(-lambda * age_days). Older memories fade
unless reinforced by recent retrieval. Lambda PARAM-driven per-memory-class.
"""
from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Optional

DEFAULT_LAMBDA = 0.05  # half-life ~14 days


def age_days(ts_iso: str) -> float:
    try:
        dt = datetime.fromisoformat(str(ts_iso).replace("Z", "+00:00"))
        return max(0.0, (datetime.now(timezone.utc) - dt).total_seconds() / 86400.0)
    except Exception:
        return 999.0


def weight(ts_iso: str, lam: float = DEFAULT_LAMBDA) -> float:
    return math.exp(-lam * age_days(ts_iso))


def reinforce(memory_id: int, table: str = "agent_memory", boost: float = 0.1) -> Optional[float]:
    try:
        from db import AI_DB_PATH, get_db_connection

        with get_db_connection(AI_DB_PATH) as conn:
            cur = conn.execute(f"PRAGMA table_info({table})")
            cols = {r[1] for r in cur.fetchall()}
            if "decay_weight" not in cols:
                try:
                    conn.execute(f"ALTER TABLE {table} ADD COLUMN decay_weight REAL DEFAULT 1.0")
                except Exception:
                    pass
            conn.execute(
                f"UPDATE {table} SET decay_weight = MIN(1.0, COALESCE(decay_weight, 1.0) + ?) WHERE id=?",
                (boost, memory_id),
            )
            conn.commit()
        return None
    except Exception:
        return None
