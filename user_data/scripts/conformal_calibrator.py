"""
Phase 26: Conformal Calibrator — CQR + ACI

Replaces Platt scaling (which DEGRADES CatBoost calibration per 2025 finding).

CQR (Conformalized Quantile Regression):
  - Distribution-free prediction intervals with MATHEMATICAL %95 coverage guarantee
  - Adapts to regime changes via ACI (Adaptive Conformal Inference)
  - Raw quantile → conformal → guaranteed coverage

ACI (Adaptive Conformal Inference):
  - Online adjustment of alpha → intervals automatically widen in volatile periods
  - Intervals shrink when model is accurate → more aggressive sizing

Novel Contribution #6 (Dual-Axis Calibration):
  This module provides the INTERVAL axis. CatBoost provides the PROBABILITY axis.
  sizing = catboost_probability × (1 / cqr_interval_width)

Library: MAPIE (pip install mapie)
"""

import logging
import os
import json
import time
import numpy as np
import pandas as pd
from typing import Dict, Optional, List, Tuple
from collections import deque

logger = logging.getLogger(__name__)

try:
    from neural_organism import _p
except ImportError:
    def _p(param_id, fallback=0.5, regime="_global"):
        return fallback


class ConformalCalibrator:
    """CQR + ACI conformal prediction for trading.

    Provides calibrated prediction intervals with mathematical coverage guarantee.
    Replaces the old Platt-based confidence_calibrator.py.

    Usage:
        cal = ConformalCalibrator()
        cal.fit(historical_predictions, historical_outcomes)
        interval = cal.predict_interval(new_prediction)
        # interval = {"lower": -0.03, "upper": 0.08, "width": 0.11, "coverage_target": 0.95}
    """

    def __init__(self, target_coverage: float = 0.95, aci_gamma: float = 0.01):
        """
        Args:
            target_coverage: Desired coverage probability (default 0.95)
            aci_gamma: ACI step size for online alpha adjustment (default 0.01)
        """
        self.target_coverage = target_coverage
        self.aci_gamma = aci_gamma

        # ACI state
        self._alpha = 1.0 - target_coverage  # current miscoverage rate
        self._alpha_history = deque(maxlen=500)

        # Nonconformity scores from calibration set
        self._calibration_scores: Optional[np.ndarray] = None
        self._fitted = False

        # Online tracking
        self._coverage_tracker = deque(maxlen=200)  # 1=covered, 0=not covered

        # Persistence
        self._state_path = os.path.join(
            os.path.dirname(__file__), "..", "models", "conformal_state.json"
        )
        self._load_state()

    def fit(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray,
        lower_quantiles: Optional[np.ndarray] = None,
        upper_quantiles: Optional[np.ndarray] = None,
    ) -> None:
        """Fit conformal calibrator from historical prediction/outcome pairs.

        For CQR: provide quantile predictions (lower, upper) from Chronos-Bolt.
        For standard conformal: provide point predictions only.

        Args:
            predictions: Point predictions (e.g., predicted PnL %)
            actuals: Actual outcomes (e.g., realized PnL %)
            lower_quantiles: Optional lower quantile predictions (P10)
            upper_quantiles: Optional upper quantile predictions (P90)
        """
        if len(predictions) < 10:
            logger.warning("[Conformal] Not enough calibration data (<10)")
            return

        if lower_quantiles is not None and upper_quantiles is not None:
            # CQR: nonconformity score = max(lower - actual, actual - upper)
            scores = np.maximum(lower_quantiles - actuals, actuals - upper_quantiles)
        else:
            # Standard conformal: nonconformity score = |prediction - actual|
            scores = np.abs(predictions - actuals)

        self._calibration_scores = np.sort(scores)
        self._fitted = True
        self._save_state()

        logger.info(
            f"[Conformal] Fit: {len(scores)} calibration scores, "
            f"median={np.median(scores):.4f}, p95={np.percentile(scores, 95):.4f}"
        )

    def predict_interval(
        self,
        prediction: float,
        lower_quantile: Optional[float] = None,
        upper_quantile: Optional[float] = None,
    ) -> Dict[str, float]:
        """Compute conformal prediction interval.

        Args:
            prediction: Point prediction (e.g., predicted PnL %)
            lower_quantile: Chronos P10 prediction (optional, for CQR)
            upper_quantile: Chronos P90 prediction (optional, for CQR)

        Returns:
            {
                "lower": float,     # lower bound of interval
                "upper": float,     # upper bound of interval
                "width": float,     # interval width (narrow = certain)
                "coverage": float,  # target coverage (0.95)
                "alpha": float,     # current ACI alpha
                "empirical_coverage": float,  # actual coverage from recent predictions
            }
        """
        if not self._fitted or self._calibration_scores is None:
            # Not fitted — return wide default interval
            width = _p("conformal.default_width", 0.10)
            return {
                "lower": prediction - width / 2,
                "upper": prediction + width / 2,
                "width": width,
                "coverage": self.target_coverage,
                "alpha": self._alpha,
                "empirical_coverage": 0.5,
            }

        # Compute conformal quantile with ACI-adjusted alpha
        n = len(self._calibration_scores)
        adjusted_alpha = np.clip(self._alpha, 0.01, 0.50)  # safety bounds
        quantile_index = int(np.ceil((1 - adjusted_alpha) * (n + 1))) - 1
        quantile_index = np.clip(quantile_index, 0, n - 1)
        q_hat = self._calibration_scores[quantile_index]

        if lower_quantile is not None and upper_quantile is not None:
            # CQR: interval = [lower_quantile - q_hat, upper_quantile + q_hat]
            lower = lower_quantile - q_hat
            upper = upper_quantile + q_hat
        else:
            # Standard: interval = [prediction - q_hat, prediction + q_hat]
            lower = prediction - q_hat
            upper = prediction + q_hat

        width = upper - lower

        # Empirical coverage from recent tracking
        empirical_coverage = (
            sum(self._coverage_tracker) / len(self._coverage_tracker)
            if self._coverage_tracker else self.target_coverage
        )

        return {
            "lower": round(float(lower), 6),
            "upper": round(float(upper), 6),
            "width": round(float(max(width, 1e-6)), 6),
            "coverage": self.target_coverage,
            "alpha": round(float(self._alpha), 4),
            "empirical_coverage": round(float(empirical_coverage), 4),
        }

    def update(self, prediction: float, actual: float,
               lower: Optional[float] = None, upper: Optional[float] = None) -> None:
        """Online ACI update after observing actual outcome.

        This is the KEY to adaptive conformal: alpha adjusts based on
        whether the interval actually covered the true outcome.

        Args:
            prediction: What we predicted
            actual: What actually happened
            lower: The interval lower bound we used
            upper: The interval upper bound we used
        """
        if lower is not None and upper is not None:
            covered = int(lower <= actual <= upper)
        else:
            # Recompute interval for this prediction
            interval = self.predict_interval(prediction)
            covered = int(interval["lower"] <= actual <= interval["upper"])

        self._coverage_tracker.append(covered)

        # ACI update (Gibbs & Candes 2021):
        # α_t+1 = α_t + γ * (α_target - (1 - covered_t))
        # α_target = 1 - target_coverage (fixed, e.g., 0.05 for 95% coverage)
        # If we missed (covered=0): alpha increases → wider intervals next time
        # If we covered (covered=1): alpha decreases → tighter intervals next time
        alpha_target = 1.0 - self.target_coverage  # fixed at 0.05
        self._alpha = float(np.clip(
            self._alpha + self.aci_gamma * (alpha_target - (1 - covered)),
            0.01, 0.50
        ))
        self._alpha_history.append(self._alpha)

        # Periodically re-save state
        if len(self._coverage_tracker) % 50 == 0:
            self._save_state()

    def _save_state(self) -> None:
        try:
            os.makedirs(os.path.dirname(self._state_path), exist_ok=True)
            state = {
                "alpha": self._alpha,
                "calibration_scores": self._calibration_scores.tolist() if self._calibration_scores is not None else [],
                "coverage_tracker": list(self._coverage_tracker),
                "target_coverage": self.target_coverage,
            }
            with open(self._state_path, "w") as f:
                json.dump(state, f)
        except Exception as e:
            logger.debug(f"[Conformal] Save state failed: {e}")

    def _load_state(self) -> None:
        try:
            if os.path.exists(self._state_path):
                with open(self._state_path) as f:
                    state = json.load(f)
                scores = state.get("calibration_scores", [])
                if scores:
                    self._calibration_scores = np.array(scores)
                    self._fitted = True
                self._alpha = state.get("alpha", 1.0 - self.target_coverage)
                tracker = state.get("coverage_tracker", [])
                self._coverage_tracker = deque(tracker, maxlen=200)
                if self._fitted:
                    logger.info(f"[Conformal] Loaded state: {len(scores)} scores, alpha={self._alpha:.4f}")
        except Exception as e:
            logger.debug(f"[Conformal] No saved state: {e}")
