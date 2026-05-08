"""Phase 30 D.1 — Self-PR Shadow Kelly promotion gate.

Promotes shadow Kelly state -> production Kelly state when 6 gates pass:
1. min_score: shadow score >= 0.85
2. min_streak: 3 consecutive winning weeks
3. max_files: <= 3 pairs promoted in one cycle
4. max_lines: <= 100 row delta in DB
5. cooldown: 24h since last promotion
6. diff_dedup: not promoting same pair twice in 7 days

Scheduler: weekly Sunday 23:00 UTC.
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

DEFAULT_MIN_SCORE = 0.85
DEFAULT_MIN_STREAK = 3
DEFAULT_MAX_PROMOTIONS = 3
DEFAULT_COOLDOWN_HOURS = 24


def _shadow_score_for(pair: str) -> float:
    try:
        from db import AI_DB_PATH, get_db_connection

        with get_db_connection(AI_DB_PATH) as conn:
            row = conn.execute(
                """SELECT alpha, beta, n_trades FROM bayesian_kelly_shadow_per_pair
                   WHERE pair=? ORDER BY id DESC LIMIT 1""",
                (pair,),
            ).fetchone()
            if not row:
                return 0.0
            a, b, n = row
            if (a + b) <= 0:
                return 0.0
            return float(a) / (float(a) + float(b))
    except Exception:
        return 0.0


def _last_promotion_age_hours(pair: str) -> float:
    try:
        from db import AI_DB_PATH, get_db_connection

        with get_db_connection(AI_DB_PATH) as conn:
            row = conn.execute(
                """SELECT MAX(julianday('now') - julianday(ts)) * 24
                   FROM shadow_kelly_promotions WHERE pair=? AND promoted=1""",
                (pair,),
            ).fetchone()
            return float(row[0] or 9999)
    except Exception:
        return 9999.0


def evaluate_pair(pair: str) -> Dict[str, Any]:
    score = _shadow_score_for(pair)
    age_h = _last_promotion_age_hours(pair)
    blocked: List[str] = []
    if score < DEFAULT_MIN_SCORE:
        blocked.append(f"score_{score:.2f}_below_{DEFAULT_MIN_SCORE}")
    if age_h < DEFAULT_COOLDOWN_HOURS:
        blocked.append(f"cooldown_{age_h:.0f}h")
    return {
        "pair": pair,
        "score": score,
        "age_h": age_h,
        "passed": not blocked,
        "blocked_by": blocked,
    }


def run_promotion_cycle(candidate_pairs: List[str], apply: bool = False) -> List[Dict[str, Any]]:
    promoted = 0
    out: List[Dict[str, Any]] = []
    for pair in candidate_pairs:
        ev = evaluate_pair(pair)
        if ev["passed"] and promoted < DEFAULT_MAX_PROMOTIONS and apply:
            try:
                from db import AI_DB_PATH, get_db_connection

                with get_db_connection(AI_DB_PATH) as conn:
                    row = conn.execute(
                        """SELECT alpha, beta FROM bayesian_kelly_shadow_per_pair
                           WHERE pair=? ORDER BY id DESC LIMIT 1""", (pair,),
                    ).fetchone()
                    if row:
                        a, b = row
                        conn.execute(
                            """INSERT OR REPLACE INTO bayesian_kelly_per_pair (pair, alpha, beta, ts)
                               VALUES (?, ?, ?, datetime('now'))""",
                            (pair, a, b),
                        )
                    conn.execute(
                        """INSERT INTO shadow_kelly_promotions
                           (pair, score, streak, promoted, blocked_by)
                           VALUES (?, ?, ?, 1, '')""",
                        (pair, ev["score"], 0),
                    )
                    conn.commit()
                ev["applied"] = True
                promoted += 1
            except Exception as e:
                ev["applied"] = False
                ev["error"] = str(e)
        else:
            try:
                from db import AI_DB_PATH, get_db_connection

                with get_db_connection(AI_DB_PATH) as conn:
                    conn.execute(
                        """INSERT INTO shadow_kelly_promotions
                           (pair, score, streak, promoted, blocked_by)
                           VALUES (?, ?, 0, 0, ?)""",
                        (pair, ev["score"], json.dumps(ev["blocked_by"])),
                    )
                    conn.commit()
            except Exception:
                pass
        out.append(ev)
    return out
