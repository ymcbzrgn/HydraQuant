"""Phase 30 A.28 — ai_lessons retro-dedup migration script.

Removes duplicate (decision_id, pair) rows keeping the most recent.
Also ensures UNIQUE index is in place. Idempotent.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "user_data" / "scripts"))

from db import AI_DB_PATH, get_db_connection  # noqa: E402


def main(apply: bool = False) -> int:
    with get_db_connection(AI_DB_PATH) as conn:
        before = conn.execute("SELECT COUNT(*) FROM ai_lessons").fetchone()[0]
        dups = conn.execute(
            """SELECT decision_id, pair, COUNT(*) c FROM ai_lessons
               GROUP BY decision_id, pair HAVING c > 1"""
        ).fetchall()
        n_dup = sum(r[2] - 1 for r in dups)
        print(f"[migrate] before={before} duplicates={n_dup} affected_groups={len(dups)}")
        if not apply:
            print("[dry-run] use --apply to actually delete")
            return 0
        conn.execute(
            """DELETE FROM ai_lessons
               WHERE id NOT IN (
                   SELECT MAX(id) FROM ai_lessons GROUP BY decision_id, pair
               )"""
        )
        conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_ai_lessons_uniq ON ai_lessons(decision_id, pair)"
        )
        conn.commit()
        after = conn.execute("SELECT COUNT(*) FROM ai_lessons").fetchone()[0]
        print(f"[migrate] after={after} removed={before - after}")
    return 0


if __name__ == "__main__":
    sys.exit(main(apply="--apply" in sys.argv))
