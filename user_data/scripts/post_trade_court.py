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

        # Phase 27 Task 15: RLAIF reward — grade the verdict with the
        # 3-judge WCO rubric and persist to rlaif_rewards.
        #
        # Post-audit fix: Phase 26 used `asyncio.ensure_future` when a loop
        # was already running. On APScheduler's thread pool there is no
        # running loop, so the coroutine was silently DROPPED — no rlaif_rewards
        # row was ever written from the court path. We now always run the
        # coroutine to completion on this thread via asyncio.run. If a loop
        # somehow IS already running (rare — only nested async contexts),
        # fall through to nest_asyncio when available, else log and skip.
        # RLAIF is OPTIONAL — scoring failure must NOT break the verdict.
        try:
            import asyncio as _asyncio
            from rlaif_reward import get_rlaif
            rlaif = get_rlaif()
            context = {
                "verified_regime": trade_dict.get("regime"),
                "n_recent_losses": self._count_recent_losses(),
            }
            coro = rlaif.score_trade(verdict, context)
            try:
                _asyncio.run(coro)
            except RuntimeError:
                # "asyncio.run() cannot be called from a running event loop"
                try:
                    import nest_asyncio  # type: ignore
                    nest_asyncio.apply()
                    _asyncio.run(rlaif.score_trade(verdict, context))
                except ImportError:
                    logger.debug(
                        "[Court:RLAIF] running loop detected and nest_asyncio "
                        "unavailable — RLAIF scoring skipped (env reward intact)"
                    )
        except Exception as e:
            logger.warning(f"[Court:RLAIF] scoring failed: {e}")

        return verdict

    def _count_recent_losses(self) -> int:
        """Phase 27 Task 15 helper: context input for RLAIF rubric."""
        try:
            conn = get_db_connection(AI_DB_PATH)
            row = conn.execute("""
                SELECT COUNT(*) AS n FROM ai_decisions
                WHERE outcome_pnl < 0
                  AND timestamp > datetime('now', '-2 days')
            """).fetchone()
            conn.close()
            return int(row["n"] or 0)
        except Exception:
            return 0

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

        # Mega Sprint 2026-04-23 (B.5): replace the generic "market_noise"
        # bucket with a concrete root cause when evidence supports one. The
        # observability improvement lets the nightly autopsy job cluster
        # losses by actual failure mode instead of collapsing them all
        # under one label.
        if blame["root_cause"] == "market_noise":
            try:
                atr_ratio = float(evidence.get("atr_ratio_at_entry", 1.0) or 1.0)
            except (TypeError, ValueError):
                atr_ratio = 1.0
            try:
                funding = float(evidence.get("funding_rate", 0.0) or 0.0)
            except (TypeError, ValueError):
                funding = 0.0
            try:
                vol_z = float(evidence.get("volume_z", 0.0) or 0.0)
            except (TypeError, ValueError):
                vol_z = 0.0
            duration_s = trade.get("outcome_duration", 0) or 0

            if atr_ratio > 1.8:
                blame["root_cause"] = "volatility_burst"
            elif abs(funding) > 0.0005:
                blame["root_cause"] = "funding_flip"
            elif vol_z < -1.5:
                blame["root_cause"] = "thin_liquidity"
            elif duration_s and duration_s < 900 and pnl < -0.5:
                blame["root_cause"] = "stop_hunt"

        # Opposite-consensus detection: the aggregator chose LONG but the
        # debate clearly leaned bearish → blame the aggregator, not the
        # agents that dissented correctly.
        try:
            consensus = json.loads(trade.get("agent_votes_json", "{}") or "{}")
        except Exception:
            consensus = {}
        if consensus:
            bears = sum(1 for v in consensus.values()
                        if isinstance(v, (int, float)) and v < 0)
            bulls = len(consensus) - bears
            signal_label = (trade.get("signal_type") or "").upper()
            if signal_label in ("LONG", "BULLISH") and bears > bulls * 1.5:
                blame["root_cause"] = "ignored_bear_consensus"
                blame["blamed_modules"].append("agent_pool_aggregator")
            elif signal_label in ("SHORT", "BEARISH") and bulls > bears * 1.5:
                blame["root_cause"] = "ignored_bull_consensus"
                blame["blamed_modules"].append("agent_pool_aggregator")

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
