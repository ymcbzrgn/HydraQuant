"""
Phase 6.3: Confidence Calibration
Measures and corrects AI confidence estimates using historical trade outcomes.

Key question: When the AI says "0.80 confidence", is it really right 80% of the time?
If not, we calibrate using Platt scaling (logistic regression on confidence → actual outcome).
"""

import os
import sys
import math
import sqlite3
import logging
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Optional

sys.path.append(os.path.dirname(__file__))

logger = logging.getLogger(__name__)

from ai_config import AI_DB_PATH as DB_PATH

# Phase 24: Neural Organism — adaptive parameters
try:
    from neural_organism import _p
except ImportError:
    def _p(param_id, fallback=0.5, regime="_global"):
        return fallback


class ConfidenceCalibrator:
    """
    Measures AI confidence quality and provides calibrated estimates.
    """

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._platt_a = 0.0  # Platt scaling parameter A
        self._platt_b = 0.0  # Platt scaling parameter B
        self._calibrated = False
        self._brier_disabled = False  # Phase 21: prevent fit-disable-fit cycle

    def _get_conn(self):
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _get_history(self, min_trades: int = 20) -> List[Tuple[float, float]]:
        """
        Fetch historical (confidence, actual_outcome) pairs from ai_decisions with REAL trade results.
        Returns list of (confidence, 1.0_if_profitable_else_0.0).
        Only uses trades that have resolved outcomes (outcome_pnl IS NOT NULL).
        """
        try:
            with self._get_conn() as conn:
                rows = conn.execute("""
                    SELECT d.confidence,
                           CASE WHEN d.outcome_pnl > 0 THEN 1.0 ELSE 0.0 END as outcome
                    FROM ai_decisions d
                    WHERE d.confidence IS NOT NULL AND d.confidence > 0
                      AND d.outcome_pnl IS NOT NULL
                    ORDER BY d.timestamp DESC
                    LIMIT 500
                """).fetchall()

                if len(rows) < min_trades:
                    return []
                return [(float(r['confidence']), float(r['outcome'])) for r in rows]
        except Exception as e:
            logger.warning(f"[Calibrator] History fetch failed: {e}")
            return []

    def brier_score(self, min_trades: int = 20) -> float:
        """
        Brier Score: mean((confidence - actual_outcome)^2)
        Lower is better. Perfect = 0.0, random = 0.25.
        """
        history = self._get_history(min_trades=min_trades)
        if not history:
            return -1.0  # No data

        total = sum((conf - outcome) ** 2 for conf, outcome in history)
        return total / len(history)

    def calibration_curve(self, n_bins: int = 10, min_trades: int = 20) -> Dict[str, List]:
        """
        Compute calibration curve: for each confidence bin, what's the actual hit rate?

        Returns:
            {
                "bins": [(0.0, 0.1), (0.1, 0.2), ...],
                "predicted": [0.05, 0.15, ...],  # mean predicted confidence per bin
                "actual": [0.10, 0.12, ...],      # actual hit rate per bin
                "counts": [5, 12, ...],           # number of samples per bin
            }
        """
        history = self._get_history(min_trades=min_trades)
        if not history:
            return {"bins": [], "predicted": [], "actual": [], "counts": []}

        bin_size = 1.0 / n_bins
        bins = []
        predicted = []
        actual = []
        counts = []

        for i in range(n_bins):
            lo = i * bin_size
            hi = (i + 1) * bin_size
            bin_data = [(c, o) for c, o in history if lo <= c < hi]

            bins.append((round(lo, 2), round(hi, 2)))
            if bin_data:
                mean_conf = sum(c for c, _ in bin_data) / len(bin_data)
                mean_outcome = sum(o for _, o in bin_data) / len(bin_data)
                predicted.append(round(mean_conf, 3))
                actual.append(round(mean_outcome, 3))
                counts.append(len(bin_data))
            else:
                predicted.append(0.0)
                actual.append(0.0)
                counts.append(0)

        return {"bins": bins, "predicted": predicted, "actual": actual, "counts": counts}

    def fit_platt_scaling(self):
        """
        Fit Platt scaling parameters using logistic regression.
        P(y=1|f) = 1 / (1 + exp(A*f + B))
        """
        history = self._get_history(min_trades=20)
        if not history:
            logger.warning("[Calibrator] Not enough data for Platt scaling.")
            return

        # Simple gradient descent for A, B
        a, b = 0.0, 0.0
        lr = 0.01

        for _ in range(1000):
            grad_a, grad_b = 0.0, 0.0
            for conf, outcome in history:
                z = a * conf + b
                sigmoid = 1.0 / (1.0 + math.exp(-z)) if z > -500 else 0.0
                error = sigmoid - outcome
                grad_a += error * conf
                grad_b += error

            grad_a /= len(history)
            grad_b /= len(history)
            a -= lr * grad_a
            b -= lr * grad_b

        self._platt_a = a
        self._platt_b = b
        self._calibrated = True
        logger.info(f"[Calibrator] Platt scaling fitted: A={a:.4f}, B={b:.4f}")

    def adjust_confidence(self, raw_confidence: float) -> float:
        """
        Calibrate raw AI confidence using Platt scaling.
        If not calibrated yet or Brier >= 0.25, returns raw confidence unchanged.
        """
        # Phase 21: If Brier check already disabled calibration, pass through without re-fitting
        if self._brier_disabled:
            return raw_confidence

        if not self._calibrated:
            self.fit_platt_scaling()

        if not self._calibrated:
            return raw_confidence  # Still no data → pass through

        # Brier safety guard: if calibration is WORSE than random, disable until next explicit re-fit
        brier_thr = _p("calibrator.brier_threshold", 0.25)
        brier = self.brier_score(min_trades=int(_p("calibrator.min_trades", 20)))
        if brier >= brier_thr:
            self._calibrated = False
            self._brier_disabled = True  # Prevent fit-disable-fit cycle
            logger.warning(f"[Calibrator] Brier {brier:.4f} >= {brier_thr} (worse than random), disabling until re-fit")
            return raw_confidence

        z = self._platt_a * raw_confidence + self._platt_b
        try:
            calibrated = 1.0 / (1.0 + math.exp(-z))
        except OverflowError:
            calibrated = 0.0 if z < 0 else 1.0

        return max(0.01, min(calibrated, 0.99))

    def report(self) -> str:
        """Generate calibration quality report."""
        brier = self.brier_score()
        curve = self.calibration_curve()

        if brier < 0:
            return "No calibration data available yet."

        lines = [
            "📐 Confidence Calibration Report",
            "━" * 35,
            f"Brier Score: {brier:.4f} {'✅' if brier < 0.15 else '⚠️' if brier < 0.25 else '❌'}",
            f"  (Perfect=0.0, Random=0.25)",
            "",
            "Bin       | Predicted | Actual | N",
            "----------|-----------|--------|---",
        ]

        for i, (lo_hi) in enumerate(curve["bins"]):
            pred = curve["predicted"][i]
            act = curve["actual"][i]
            n = curve["counts"][i]
            diff_marker = "✅" if abs(pred - act) < 0.1 else "⚠️"
            lines.append(f"{lo_hi[0]:.1f}-{lo_hi[1]:.1f}   | {pred:.3f}     | {act:.3f}  | {n} {diff_marker}")

        return "\n".join(lines)
