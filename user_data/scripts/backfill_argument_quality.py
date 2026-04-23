"""One-shot backfill for argument_quality after Mega Sprint 2026-04-23 (B.4).

Resets every row to the Beta(1,1) prior then re-plays `agent_memory` rows
with a resolved `final_outcome_pnl` through the canonical agent-direction
rule (BULLISH+pnl>0, BEARISH+pnl<0, NEUTRAL+|pnl|<0.5). Previous rows were
graded against the aggregated trade signal, which marked contrarian agents
wrong every time they were actually right.
"""
from __future__ import annotations

import logging
import sys

from ai_config import AI_DB_PATH
from agent_pool import AgentPool
from db import get_db_connection

logging.basicConfig(
    level=logging.INFO,
    format="[ArgQualityBackfill] %(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> int:
    pool = AgentPool(db_path=AI_DB_PATH)
    conn = get_db_connection()
    try:
        conn.execute(
            "UPDATE argument_quality "
            "SET times_used = 0, times_correct = 0, "
            "    avg_pnl_when_used = 0, quality_score = 0.5"
        )
        conn.commit()

        rows = conn.execute(
            """
            SELECT agent_type, regime, signal, key_argument, final_outcome_pnl
            FROM agent_memory
            WHERE final_outcome_pnl IS NOT NULL
            """
        ).fetchall()
    finally:
        conn.close()

    processed = 0
    skipped = 0
    for row in rows:
        arg = row["key_argument"] or ""
        pattern = pool._extract_argument_pattern(arg)
        if not pattern:
            skipped += 1
            continue
        agent_signal = row["signal"]
        try:
            pnl = float(row["final_outcome_pnl"])
        except (TypeError, ValueError):
            skipped += 1
            continue

        if agent_signal == "BULLISH":
            correct = pnl > 0
        elif agent_signal == "BEARISH":
            correct = pnl < 0
        elif agent_signal == "NEUTRAL":
            try:
                from neural_organism import _p
                band = float(_p("agent.neutral_correct_band_pct", 0.5))
            except Exception:
                band = 0.5
            correct = abs(pnl) < band
        else:
            skipped += 1
            continue

        try:
            pool._update_argument_quality(
                agent_type=row["agent_type"],
                pattern=pattern,
                regime=row["regime"] or "_global",
                was_correct=correct,
                outcome_pnl=pnl,
            )
            processed += 1
        except Exception as exc:
            skipped += 1
            logger.debug("update failed for %s: %s", row["agent_type"], exc)

    logger.info("processed=%d skipped=%d total_rows=%d", processed, skipped, len(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
