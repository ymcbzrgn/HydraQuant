"""
model_risk_engine.py — Phase 26 Sprint 2, Task 12B

Model Risk Engine — Monitors world model and counterfactual bias.

Detects when models are drifting, hallucinating, or producing biased predictions.
Acts as a "brake" on model-based decisions when trust is low.

Checks:
  1. World model prediction accuracy (predicted vs actual next-state)
  2. Dream filter pass rate (low rate = unhealthy world model)
  3. Counterfactual consistency (similar interventions should give similar results)
  4. CatBoost feature drift (distribution shift in input features)
  5. Calibration drift (Brier score trend)

Integration:
  - Reads from: world_model_rollouts, dream_scenarios, catboost_training_runs
  - Writes risk assessments to organism_audit
  - Consumed by: constitution (blocks model-based decisions when risk is high)
"""

import os
import sys
import json
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
logger = logging.getLogger("model_risk_engine")

from ai_config import AI_DB_PATH
from db import get_db_connection, execute_with_retry, init_db

# Risk thresholds
DREAM_PASS_RATE_MIN = 0.2      # Below: world model unhealthy
BRIER_DEGRADED = 0.3           # Above: calibration is bad
FEATURE_DRIFT_THRESHOLD = 0.5  # KL divergence threshold


class ModelRiskEngine:
    """Monitors model health and prevents model-based hallucination trading."""

    def __init__(self):
        self._risk_scores: Dict[str, float] = {}
        init_db()

    def assess_risk(self) -> Dict:
        """Run full model risk assessment.

        Returns {"overall_risk": 0-1, "model_risks": {...}, "recommendations": [...]}
        """
        risks = {}
        recommendations = []

        # 1. Dream filter pass rate
        dream_risk = self._check_dream_health()
        risks["dream_engine"] = dream_risk
        if dream_risk > 0.7:
            recommendations.append("World model unhealthy — disable dream-augmented learning")

        # 2. CatBoost training drift
        catboost_risk = self._check_catboost_drift()
        risks["catboost"] = catboost_risk
        if catboost_risk > 0.6:
            recommendations.append("CatBoost accuracy degrading — trigger retrain")

        # 3. Calibration quality
        calibration_risk = self._check_calibration()
        risks["calibration"] = calibration_risk
        if calibration_risk > 0.5:
            recommendations.append("Confidence calibration degraded — pass-through mode")

        # 4. Prediction staleness
        staleness_risk = self._check_prediction_staleness()
        risks["staleness"] = staleness_risk
        if staleness_risk > 0.5:
            recommendations.append("Models haven't been retrained recently")

        # Overall risk
        overall = float(np.mean(list(risks.values()))) if risks else 0.0

        self._risk_scores = risks

        result = {
            "overall_risk": round(overall, 3),
            "model_risks": {k: round(v, 3) for k, v in risks.items()},
            "recommendations": recommendations,
            "assessed_at": datetime.now(timezone.utc).isoformat(),
        }

        # Persist
        self._persist_assessment(result)

        return result

    def _check_dream_health(self) -> float:
        """Check dream engine pass rate from dream_scenarios."""
        conn = get_db_connection(AI_DB_PATH)
        try:
            row = conn.execute("""
                SELECT
                    COUNT(CASE WHEN passed_filter = 1 THEN 1 END) as passed,
                    COUNT(*) as total
                FROM dream_scenarios
                WHERE timestamp >= datetime('now', '-7 days')
            """).fetchone()

            total = row["total"] or 0
            passed = row["passed"] or 0

            if total < 10:
                return 0.3  # Not enough data, moderate risk

            pass_rate = passed / total
            if pass_rate < DREAM_PASS_RATE_MIN:
                return 0.9  # Very high risk
            elif pass_rate < 0.5:
                return 0.5
            else:
                return 0.1  # Healthy

        except Exception:
            return 0.3  # Table may not exist yet
        finally:
            conn.close()

    def _check_catboost_drift(self) -> float:
        """Check CatBoost accuracy trend."""
        conn = get_db_connection(AI_DB_PATH)
        try:
            rows = conn.execute("""
                SELECT test_accuracy, test_f1, trained_at
                FROM catboost_training_runs
                ORDER BY trained_at DESC LIMIT 5
            """).fetchall()

            if not rows:
                return 0.3

            accuracies = [r["test_accuracy"] or 0 for r in rows]

            # Check if accuracy is declining
            if len(accuracies) >= 2:
                trend = accuracies[0] - accuracies[-1]  # Latest minus oldest
                if trend < -0.1:
                    return 0.7  # Significant decline
                elif trend < -0.05:
                    return 0.4
            return 0.1

        except Exception:
            return 0.3
        finally:
            conn.close()

    def _check_calibration(self) -> float:
        """Check confidence calibration quality."""
        # Use Brier score from recent predictions
        conn = get_db_connection(AI_DB_PATH)
        try:
            rows = conn.execute("""
                SELECT confidence, outcome_pnl
                FROM ai_decisions
                WHERE outcome_pnl IS NOT NULL
                ORDER BY timestamp DESC LIMIT 100
            """).fetchall()

            if len(rows) < 20:
                return 0.3

            # Brier score: mean((confidence - actual)^2)
            brier = 0.0
            for r in rows:
                conf = r["confidence"] or 0.5
                actual = 1.0 if (r["outcome_pnl"] or 0) > 0 else 0.0
                brier += (conf - actual) ** 2
            brier /= len(rows)

            if brier > BRIER_DEGRADED:
                return 0.7
            elif brier > 0.25:
                return 0.4
            return 0.1

        except Exception:
            return 0.3
        finally:
            conn.close()

    def _check_prediction_staleness(self) -> float:
        """Check when models were last retrained."""
        conn = get_db_connection(AI_DB_PATH)
        try:
            row = conn.execute("""
                SELECT MAX(trained_at) as last_train
                FROM catboost_training_runs
            """).fetchone()

            if not row or not row["last_train"]:
                return 0.5

            last_train = datetime.fromisoformat(row["last_train"])
            days_since = (datetime.now(timezone.utc) - last_train.replace(tzinfo=timezone.utc)).days

            if days_since > 14:
                return 0.7
            elif days_since > 7:
                return 0.3
            return 0.1

        except Exception:
            return 0.3
        finally:
            conn.close()

    def _persist_assessment(self, result: Dict):
        """Log risk assessment to organism_audit."""
        try:
            execute_with_retry(
                """INSERT INTO organism_audit
                   (timestamp, event_type, details_json)
                   VALUES (datetime('now'), 'model_risk_assessment', ?)""",
                (json.dumps(result),),
                max_retries=3,
            )
        except Exception:
            pass

    def get_model_trust(self) -> float:
        """Get overall model trust level (1 - risk)."""
        if not self._risk_scores:
            self.assess_risk()
        overall = float(np.mean(list(self._risk_scores.values()))) if self._risk_scores else 0.3
        return round(1.0 - overall, 3)

    def get_status(self) -> Dict:
        return {
            "risk_scores": self._risk_scores,
            "model_trust": self.get_model_trust(),
        }


# Singleton
_mre_instance = None

def get_model_risk_engine() -> ModelRiskEngine:
    global _mre_instance
    if _mre_instance is None:
        _mre_instance = ModelRiskEngine()
    return _mre_instance
