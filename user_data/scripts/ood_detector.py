"""
Phase 26: Out-of-Distribution (OOD) Detector — Mahalanobis Distance

"Bu piyasayı daha önce hiç görmedim" → defansif mod (%50-75 sizing azaltma).

Per-regime Gaussian fit. When new data point's Mahalanobis distance exceeds
chi-squared threshold → OOD flag → organism enters defensive mode.

NeurIPS 2024: "Mahalanobis performs best for OOD detection"
"""

import logging
import os
import json
import numpy as np
import pandas as pd
from typing import Dict, Optional, List

try:
    from scipy import stats as scipy_stats
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False
    scipy_stats = None
    logging.getLogger(__name__).warning("[OOD] scipy not installed — OOD detection disabled")

logger = logging.getLogger(__name__)

try:
    from neural_organism import _p
except ImportError:
    def _p(param_id, fallback=0.5, regime="_global"):
        return fallback


class MarketOODDetector:
    """Mahalanobis distance-based OOD detection per market regime.

    Fit phase: learn mean + precision matrix per regime from historical features.
    Detect phase: compute Mahalanobis distance for new feature vector.
                  If distance > chi-squared threshold → OOD.

    Usage:
        detector = MarketOODDetector()
        detector.fit(feature_matrix, regime_labels)  # offline, from backtests
        result = detector.detect(new_features)
        if result["is_ood"]:
            sizing *= result["defensive_multiplier"]  # reduce position
    """

    REGIMES = ["trending_bull", "trending_bear", "ranging", "crash", "recovery", "transition"]

    def __init__(self, confidence_level: float = 0.95):
        """
        Args:
            confidence_level: Chi-squared threshold for OOD detection (default 0.95 = 5% false positive)
        """
        self.confidence_level = confidence_level
        self._fitted = False
        self._regime_stats: Dict[str, Dict] = {}  # regime → {mean, precision, n_features}
        self._global_stats: Optional[Dict] = None   # fallback when regime unknown
        self._model_path = os.path.join(
            os.path.dirname(__file__), "..", "models", "ood_detector_state.json"
        )
        self._load_state()

    def fit(self, features: pd.DataFrame, regimes: pd.Series) -> None:
        """Fit per-regime Gaussian distributions from historical data.

        Args:
            features: DataFrame of feature vectors (rows=samples, cols=features)
            regimes: Series of regime labels per sample
        """
        if len(features) < 30:
            logger.warning("[OOD] Not enough data to fit (<30 samples)")
            return

        feature_cols = features.columns.tolist()
        n_features = len(feature_cols)

        # Global stats (fallback)
        self._global_stats = self._fit_gaussian(features.values, n_features)

        # Per-regime stats
        for regime in self.REGIMES:
            mask = regimes == regime
            regime_data = features[mask].values
            if len(regime_data) >= max(n_features + 1, 15):  # need n+1 samples minimum for covariance
                self._regime_stats[regime] = self._fit_gaussian(regime_data, n_features)
                logger.info(f"[OOD] Fitted regime '{regime}': {len(regime_data)} samples, {n_features} features")
            else:
                logger.debug(f"[OOD] Skipping regime '{regime}': only {len(regime_data)} samples")

        self._fitted = True
        self._save_state(feature_cols)
        logger.info(f"[OOD] Fit complete: {len(self._regime_stats)} regimes, {n_features} features")

    def detect(
        self,
        features: Dict[str, float],
        regime: str = "_global",
    ) -> Dict[str, any]:
        """Detect if new feature vector is OOD.

        Args:
            features: Dict of feature_name → value
            regime: Current market regime (for regime-specific detection)

        Returns:
            {
                "is_ood": bool,
                "distance": float (Mahalanobis distance),
                "threshold": float (chi-squared critical value),
                "p_value": float (probability of seeing this distance under fitted distribution),
                "defensive_multiplier": float (1.0 if normal, 0.25-0.75 if OOD),
                "closest_regime": str (which regime this is closest to),
            }
        """
        result = {
            "is_ood": False,
            "distance": 0.0,
            "threshold": 0.0,
            "p_value": 1.0,
            "defensive_multiplier": 1.0,
            "closest_regime": regime,
        }

        if not self._fitted or not _SCIPY_AVAILABLE:
            return result

        # Get stats for current regime (or global fallback)
        regime_stat = self._regime_stats.get(regime, self._global_stats)
        if regime_stat is None:
            regime_stat = self._global_stats
        if regime_stat is None:
            return result

        # Build feature vector in correct order
        feature_names = regime_stat.get("feature_names", [])
        x = np.array([features.get(name, 0.0) for name in feature_names])

        # Compute Mahalanobis distance
        mean = np.array(regime_stat["mean"])
        precision = np.array(regime_stat["precision"])
        n_features = len(mean)

        if n_features == 0:
            return result

        try:
            diff = x - mean
            distance = float(np.sqrt(np.clip(diff @ precision @ diff, 0, None)))
        except Exception as e:
            logger.debug(f"[OOD] Mahalanobis computation failed: {e}")
            return result

        # Chi-squared test
        threshold = float(np.sqrt(scipy_stats.chi2.ppf(self.confidence_level, df=n_features)))
        p_value = float(1.0 - scipy_stats.chi2.cdf(distance ** 2, df=n_features))

        is_ood = distance > threshold

        # Defensive multiplier — graduated response
        if is_ood:
            # How far beyond threshold? Clip to reasonable range
            severity = min(distance / threshold, 3.0)  # cap at 3x threshold
            base_defense = _p("ood.defensive_base", 0.5)
            # severity 1.0 → multiplier = base_defense, severity 3.0 → multiplier = base_defense * 0.5
            defensive_multiplier = base_defense * (1.0 / severity)
            defensive_multiplier = max(defensive_multiplier, 0.15)  # never reduce below 15%
        else:
            defensive_multiplier = 1.0

        # Find closest regime (useful when regime detection is uncertain)
        closest_regime = regime
        min_dist = distance
        for r, stats in self._regime_stats.items():
            try:
                r_mean = np.array(stats["mean"])
                r_prec = np.array(stats["precision"])
                r_diff = x - r_mean
                r_dist = float(np.sqrt(np.clip(r_diff @ r_prec @ r_diff, 0, None)))
                if r_dist < min_dist:
                    min_dist = r_dist
                    closest_regime = r
            except Exception:
                continue

        result["is_ood"] = is_ood
        result["distance"] = round(distance, 4)
        result["threshold"] = round(threshold, 4)
        result["p_value"] = round(p_value, 6)
        result["defensive_multiplier"] = round(defensive_multiplier, 4)
        result["closest_regime"] = closest_regime

        if is_ood:
            logger.warning(
                f"[OOD] DETECTED! distance={distance:.2f} > threshold={threshold:.2f} "
                f"(p={p_value:.4f}) → defensive_mult={defensive_multiplier:.2f}, "
                f"closest_regime={closest_regime}"
            )

        return result

    def _fit_gaussian(self, data: np.ndarray, n_features: int) -> Dict:
        """Fit multivariate Gaussian and compute precision matrix."""
        mean = data.mean(axis=0)
        cov = np.cov(data, rowvar=False)

        # Regularize covariance (prevent singular matrix)
        regularization = 1e-6 * np.eye(n_features)
        cov_reg = cov + regularization

        try:
            precision = np.linalg.inv(cov_reg)
        except np.linalg.LinAlgError:
            # Fallback: diagonal precision
            precision = np.diag(1.0 / (np.diag(cov) + 1e-6))

        return {
            "mean": mean.tolist(),
            "precision": precision.tolist(),
            "n_features": n_features,
            "n_samples": len(data),
        }

    def _save_state(self, feature_names: List[str]) -> None:
        """Persist fitted state to JSON for reuse across restarts."""
        try:
            os.makedirs(os.path.dirname(self._model_path), exist_ok=True)
            state = {
                "feature_names": feature_names,
                "global": self._global_stats,
                "regimes": {r: s for r, s in self._regime_stats.items()},
            }
            # Add feature_names to each stat dict
            for key in ["global"] + list(state["regimes"].keys()):
                if state.get(key) or state.get("regimes", {}).get(key):
                    target = state[key] if key == "global" else state["regimes"][key]
                    if target:
                        target["feature_names"] = feature_names

            with open(self._model_path, "w") as f:
                json.dump(state, f)
            logger.info(f"[OOD] State saved to {self._model_path}")
        except Exception as e:
            logger.warning(f"[OOD] Failed to save state: {e}")

    def _load_state(self) -> None:
        """Load previously fitted state."""
        try:
            if os.path.exists(self._model_path):
                with open(self._model_path) as f:
                    state = json.load(f)
                self._global_stats = state.get("global")
                self._regime_stats = state.get("regimes", {})
                if self._global_stats or self._regime_stats:
                    self._fitted = True
                    n_regimes = len(self._regime_stats)
                    logger.info(f"[OOD] Loaded state: {n_regimes} regimes from {self._model_path}")
        except Exception as e:
            logger.debug(f"[OOD] No saved state found: {e}")
