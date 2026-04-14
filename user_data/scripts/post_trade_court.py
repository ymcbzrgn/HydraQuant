"""
post_trade_court.py — Phase 26 Sprint 2, Task 12C

Post-Trade Court — Trade autopsy with blame assignment.

After each trade closes, the "court" investigates:
  - Which modules contributed to the decision?
  - Was the signal quality good or bad?
  - Was the execution quality good or bad?
  - Was the sizing appropriate?
  - What was the root cause of loss (if any)?

Blame assignment feeds back to:
  - ablation_league (module contribution tracking)
  - self_model (competence map updates)
  - EWC (regime-specific learning)

Integration:
  - Reads from: ai_decisions, evidence_audit_log, organism_audit
  - Writes verdicts to: organism_audit (event_type='trade_verdict')
  - Consumed by: ablation_league (12A), self_model (9A)
"""

import os
import sys
import json
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
logger = logging.getLogger("post_trade_court")

from ai_config import AI_DB_PATH
from db import get_db_connection, get_connection, execute_with_retry, init_db


class PostTradeCourt:
    """Trade autopsy and blame assignment."""

    def __init__(self):
        init_db()

    def investigate_trade(self, trade_id: int) -> Dict:
        """Full investigation of a completed trade."""
        conn = get_db_connection(AI_DB_PATH)
        try:
            trade = conn.execute("""
                SELECT * FROM ai_decisions WHERE id = ?
            """, (trade_id,)).fetchone()

            if not trade:
                return {"error": f"trade {trade_id} not found"}

            evidence = conn.execute("""
                SELECT * FROM evidence_audit_log
                WHERE pair = ? AND ABS(JULIANDAY(timestamp) - JULIANDAY(?)) < 0.01
                LIMIT 1
            """, (trade["pair"], trade["timestamp"])).fetchone()

        finally:
            conn.close()

        trade_dict = dict(trade)
        evidence_dict = dict(evidence) if evidence else {}

        verdict = {
            "trade_id": trade_id,
            "pair": trade_dict.get("pair"),
            "signal": trade_dict.get("signal_type"),
            "confidence": trade_dict.get("confidence"),
            "pnl": trade_dict.get("outcome_pnl"),
            "duration": trade_dict.get("outcome_duration"),
            "regime": trade_dict.get("regime"),
            "timestamp": trade_dict.get("timestamp"),
        }

        # Determine outcome
        pnl = trade_dict.get("outcome_pnl", 0) or 0
        verdict["outcome"] = "WIN" if pnl > 0 else "LOSS" if pnl < 0 else "BREAKEVEN"

        # Blame analysis
        blame = self._assign_blame(trade_dict, evidence_dict)
        verdict["blame"] = blame

        # Lessons learned
        verdict["lessons"] = self._extract_lessons(trade_dict, evidence_dict, blame)

        # Persist verdict
        self._persist_verdict(verdict)

        return verdict

    def _assign_blame(self, trade: Dict, evidence: Dict) -> Dict:
        """Assign blame to specific components."""
        blame = {
            "signal_quality": "good",
            "sizing_quality": "appropriate",
            "timing_quality": "neutral",
            "execution_quality": "unknown",
            "root_cause": "none",
            "blamed_modules": [],
        }

        pnl = trade.get("outcome_pnl", 0) or 0
        conf = trade.get("confidence", 0.5) or 0.5

        if pnl >= 0:
            # Winning trade — identify what worked
            if conf > 0.7:
                blame["signal_quality"] = "excellent"
            return blame

        # Losing trade analysis
        # 1. Signal quality
        if conf > 0.7 and pnl < -1:
            blame["signal_quality"] = "overconfident"
            blame["blamed_modules"].append("confidence_calibrator")
            blame["root_cause"] = "high confidence + significant loss = calibration failure"

        elif conf < 0.4:
            blame["signal_quality"] = "low_confidence_traded"
            blame["blamed_modules"].append("entry_filter")
            blame["root_cause"] = "should not have traded at low confidence"

        # 2. Sizing
        if pnl < -3:
            blame["sizing_quality"] = "too_large"
            blame["blamed_modules"].append("position_sizer")

        # 3. Timing
        duration = trade.get("outcome_duration", 0) or 0
        if duration < 300:  # Less than 5 minutes
            blame["timing_quality"] = "premature_exit"
        elif duration > 86400:  # More than 24 hours
            blame["timing_quality"] = "held_too_long"

        # 4. Sub-score analysis
        if evidence:
            sub_scores_json = evidence.get("sub_scores_json", "{}")
            try:
                sub = json.loads(sub_scores_json) if sub_scores_json else {}
                # Find the most misleading sub-score
                for score_name, score_val in sub.items():
                    if isinstance(score_val, (int, float)) and score_val > 0.7 and pnl < -1:
                        blame["blamed_modules"].append(f"sub_{score_name}")
            except Exception:
                pass

        if not blame["root_cause"] or blame["root_cause"] == "none":
            blame["root_cause"] = "market_noise"

        return blame

    def _extract_lessons(self, trade: Dict, evidence: Dict, blame: Dict) -> List[str]:
        """Extract actionable lessons from the trade."""
        lessons = []

        if blame["signal_quality"] == "overconfident":
            lessons.append("Reduce confidence boost for this regime")

        if blame["sizing_quality"] == "too_large":
            lessons.append("Implement stricter position limits for high-loss scenarios")

        if blame["timing_quality"] == "held_too_long":
            lessons.append("Consider tighter time-based exit rules")

        if "market_noise" in blame["root_cause"]:
            lessons.append("Accept — market noise loss within risk budget")

        return lessons

    def _persist_verdict(self, verdict: Dict):
        """Write verdict to organism_audit."""
        try:
            execute_with_retry(
                """INSERT INTO organism_audit
                   (timestamp, event_type, details_json)
                   VALUES (datetime('now'), 'trade_verdict', ?)""",
                (json.dumps(verdict, default=str),),
                max_retries=3,
            )
        except Exception as e:
            logger.debug(f"[Court] Persist failed: {e}")

    def investigate_recent(self, n_trades: int = 10) -> List[Dict]:
        """Investigate N most recent completed trades."""
        conn = get_db_connection(AI_DB_PATH)
        try:
            rows = conn.execute("""
                SELECT id FROM ai_decisions
                WHERE outcome_pnl IS NOT NULL
                ORDER BY timestamp DESC LIMIT ?
            """, (n_trades,)).fetchall()
        finally:
            conn.close()

        verdicts = []
        for row in rows:
            verdict = self.investigate_trade(row["id"])
            if "error" not in verdict:
                verdicts.append(verdict)

        losses = [v for v in verdicts if v["outcome"] == "LOSS"]
        wins = [v for v in verdicts if v["outcome"] == "WIN"]

        logger.info(f"[Court] Investigated {len(verdicts)} trades: "
                    f"{len(wins)} wins, {len(losses)} losses")

        return verdicts


# Singleton
_court_instance = None

def get_court() -> PostTradeCourt:
    global _court_instance
    if _court_instance is None:
        _court_instance = PostTradeCourt()
    return _court_instance
