"""
Phase 27 Fix 1 (D4 Ajani): Out-of-Distribution (OOD) Detector — Wasserstein W1

"Bu piyasayı daha önce hiç görmedim" → defansif mod (sizing azaltma).

Per-regime empirical reference distributions. When new data point's per-feature
Wasserstein-1 distance exceeds the 95th-percentile training score → OOD flag.

WHY WASSERSTEIN (replaces Phase 26 Mahalanobis):
  Mahalanobis assumed Gaussian. Crypto returns have kurtosis 9-18 (not Gaussian).
  In d=20 features, Mahalanobis distance concentrates → every point is "OOD"
  (distance=500, defensive_mult=0.17 permanently). Wasserstein-1 compares the
  empirical CDFs and degrades gracefully under fat tails.

Usage contract (UNCHANGED from Phase 26 to preserve callers):
  detector = MarketOODDetector()
  detector.fit(features_df, regime_series)      # scheduler._ood_refit, Sun 04:15 UTC
  result = detector.detect(feature_dict, regime)
  if result["is_ood"]:
      sizing *= result["defensive_multiplier"]

Return dict keys (callers in triple_perception.py:190, dual_axis_calibrator.py:109):
  is_ood, distance, threshold, p_value, defensive_multiplier, closest_regime

State format v2 persisted at user_data/models/ood_detector_state.json.
Version mismatch triggers auto-refit on next fit() call.
"""

import logging
import math
import os
import json
from typing import Dict, Optional, List

import numpy as np
import pandas as pd

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


STATE_FORMAT_VERSION = 2  # v1 = Gaussian (Phase 26), v2 = Wasserstein (Phase 27 Fix 1)


class MarketOODDetector:
    """Wasserstein-1 based OOD detection per market regime.

    Fit:    store per-feature sorted reference arrays + empirical W1 threshold
            from training-set self-distances (95th percentile).
    Detect: compute mean per-feature W1(reference, {sample}) → compare to threshold.
            Sigmoid defensive_multiplier smoothly scales sizing — no binary cliff.

    Phase 27 changes from Phase 26:
      - _fit_gaussian → _fit_reference (sorted arrays instead of mean/precision)
      - Mahalanobis distance → per-feature W1 mean
      - Chi-squared threshold → 95th-percentile empirical threshold
      - Binary defensive_multiplier → sigmoid on distance/threshold ratio
    """

    # Audit fix (2026-04-19): regime labels MUST match RegimeClassifier exactly.
    # Pre-fix list had 'transition' (wrong, RegimeClassifier emits
    # 'transitional') + 'crash'/'recovery' (never produced by the classifier),
    # so per-regime fit found 0 matching rows and refit reported "0 regimes
    # fit". Now mirrors regime_classifier.RegimeClassifier.ALL_REGIMES.
    REGIMES = ["trending_bull", "trending_bear", "ranging",
               "high_volatility", "transitional"]

    def __init__(self, confidence_level: float = 0.95):
        """
        Args:
            confidence_level: Percentile for OOD threshold (default 0.95 = 5% false positive).
        """
        self.confidence_level = confidence_level
        self._fitted = False
        self._regime_stats: Dict[str, Dict] = {}  # regime → {ref_distributions, w1_threshold, train_scores}
        self._global_stats: Optional[Dict] = None
        self._model_path = os.path.join(
            os.path.dirname(__file__), "..", "models", "ood_detector_state.json"
        )
        self._load_state()

    # ─── FIT ────────────────────────────────────────────────────────────────

    def fit(self, features: pd.DataFrame, regimes: pd.Series) -> None:
        """Fit per-regime empirical reference distributions.

        Args:
            features: DataFrame of feature vectors (rows=samples, cols=features).
            regimes:  Series of regime labels per sample.
        """
        if len(features) < 30:
            logger.warning("[OOD] Not enough data to fit (<30 samples)")
            return

        feature_cols = features.columns.tolist()
        n_features = len(feature_cols)

        # Global stats (fallback for unseen regimes)
        self._global_stats = self._fit_reference(features.values, feature_cols)

        # Per-regime stats
        self._regime_stats = {}
        for regime in self.REGIMES:
            mask = regimes == regime
            regime_data = features[mask].values
            if len(regime_data) >= max(n_features + 1, 15):
                self._regime_stats[regime] = self._fit_reference(regime_data, feature_cols)
                logger.info(
                    f"[OOD] Fitted regime '{regime}': {len(regime_data)} samples, "
                    f"{n_features} features, w1_threshold={self._regime_stats[regime]['w1_threshold']:.4f}"
                )
            else:
                logger.debug(f"[OOD] Skipping regime '{regime}': only {len(regime_data)} samples")

        self._fitted = True
        self._save_state(feature_cols)
        logger.info(
            f"[OOD] Fit complete: {len(self._regime_stats)} regimes, {n_features} features, "
            f"format=v{STATE_FORMAT_VERSION} (Wasserstein)"
        )

    def _fit_reference(self, data: np.ndarray, feature_names: List[str]) -> Dict:
        """Build per-feature sorted reference arrays + empirical W1 threshold.

        For each feature column: sort the column, store as reference CDF support.
        For each training sample: compute its per-feature W1 against the (leave-one-out
        would be ideal but for speed we use full) reference → mean → training score.
        Threshold = 95th percentile of training scores.
        """
        ref_distributions = []
        for col_idx in range(data.shape[1]):
            col_sorted = np.sort(data[:, col_idx]).tolist()
            ref_distributions.append(col_sorted)

        # Per-sample training W1 scores (for threshold calibration)
        train_scores = []
        for i in range(len(data)):
            score = self._compute_w1_from_refs(data[i], ref_distributions)
            train_scores.append(float(score))

        if not _SCIPY_AVAILABLE or not train_scores:
            w1_threshold = 1.0  # degraded mode — scipy missing or no data
        else:
            w1_threshold = float(np.percentile(train_scores, self.confidence_level * 100))

        return {
            "ref_distributions": ref_distributions,
            "w1_threshold": w1_threshold,
            "train_scores": sorted(train_scores),  # sorted for fast percentile rank lookup
            "n_features": len(feature_names),
            "n_samples": len(data),
            "feature_names": feature_names,
        }

    @staticmethod
    def _compute_w1_from_refs(sample: np.ndarray, ref_distributions: List[list]) -> float:
        """Mean per-feature Wasserstein-1 distance between reference arrays and a single sample point.

        A single scalar vs a reference array: W1 = |ref_mean - sample| is exact
        (the optimal transport of a Dirac mass to the reference CDF).
        We use scipy's wasserstein_distance for robustness / future multi-point support.
        """
        if not _SCIPY_AVAILABLE:
            return 0.0
        distances = []
        for col_idx, ref in enumerate(ref_distributions):
            if col_idx >= len(sample):
                continue
            if not ref:
                continue
            val = float(sample[col_idx])
            d = scipy_stats.wasserstein_distance(ref, [val])
            distances.append(float(d))
        return float(np.mean(distances)) if distances else 0.0

    # ─── DETECT ─────────────────────────────────────────────────────────────

    def detect(
        self,
        features: Dict[str, float],
        regime: str = "_global",
    ) -> Dict[str, any]:
        """Detect if new feature vector is OOD.

        Args:
            features: Dict of feature_name → value.
            regime:   Current market regime (for regime-specific detection).

        Returns (contract preserved from Phase 26 — callers depend on these keys):
            is_ood:                bool
            distance:              float (mean per-feature W1)
            threshold:             float (95th-percentile training W1)
            p_value:               float (1 - empirical CDF of distance in train scores)
            defensive_multiplier:  float (sigmoid on ratio, floor 0.10, ceiling 1.0)
            closest_regime:        str  (regime with minimum distance to this sample)
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

        regime_stat = self._regime_stats.get(regime, self._global_stats)
        if regime_stat is None:
            regime_stat = self._global_stats
        if regime_stat is None:
            return result

        feature_names = regime_stat.get("feature_names", [])
        if not feature_names:
            return result
        x = np.array([features.get(name, 0.0) for name in feature_names])

        ref_distributions = regime_stat.get("ref_distributions", [])
        threshold = float(regime_stat.get("w1_threshold", 1.0))
        train_scores = regime_stat.get("train_scores", [])

        distance = self._compute_w1_from_refs(x, ref_distributions)

        # Empirical p-value: fraction of training scores >= observed distance
        if train_scores:
            rank = float(np.searchsorted(train_scores, distance))
            p_value = max(0.0, 1.0 - rank / max(len(train_scores), 1))
        else:
            p_value = 1.0

        is_ood = distance > threshold

        # Sigmoid-based defensive multiplier (smooth, no binary cliff)
        # ratio = distance / threshold:
        #   0.5 → mult ≈ 0.82 (well inside distribution)
        #   1.0 → mult ≈ 0.50 (at threshold)
        #   2.0 → mult ≈ 0.17 (2x threshold, deep OOD)
        #   3.0 → mult ≈ 0.10 (floor)
        if threshold > 1e-10:
            ratio = distance / threshold
            # Sprint 2026-05-04: floor raised 0.10 → 0.50. Old setting let the
            # OOD detector cut sizing by 90% on extreme distance, which combined
            # with cortisol/streak multipliers and Platt-collapsed confidence
            # routinely pushed final stake under min_notional → silent SHADOW.
            # 0.50 floor keeps a usable channel even at high distance.
            defensive_multiplier = 1.0 / (1.0 + math.exp(3.0 * (ratio - 1.0)))
            floor = float(_p("ood.defensive_floor", 0.50))
            defensive_multiplier = max(floor, min(1.0, defensive_multiplier))
        else:
            defensive_multiplier = 1.0

        # Closest regime (diagnostic — does not change sizing)
        closest_regime = regime
        min_dist = distance
        for r, stats in self._regime_stats.items():
            try:
                r_refs = stats.get("ref_distributions", [])
                r_feature_names = stats.get("feature_names", [])
                if not r_refs or not r_feature_names:
                    continue
                r_x = np.array([features.get(name, 0.0) for name in r_feature_names])
                r_dist = self._compute_w1_from_refs(r_x, r_refs)
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
                f"[OOD] DETECTED! w1={distance:.4f} > thr={threshold:.4f} "
                f"(p={p_value:.4f}) → defensive_mult={defensive_multiplier:.2f}, "
                f"closest_regime={closest_regime}"
            )

        return result

    # ─── STATE PERSISTENCE ──────────────────────────────────────────────────

    def _save_state(self, feature_names: List[str]) -> None:
        """Persist fitted state (format v2) to JSON for reuse across restarts."""
        try:
            os.makedirs(os.path.dirname(self._model_path), exist_ok=True)
            state = {
                "format_version": STATE_FORMAT_VERSION,
                "feature_names": feature_names,
                "confidence_level": self.confidence_level,
                "global": self._global_stats,
                "regimes": self._regime_stats,
            }
            with open(self._model_path, "w") as f:
                json.dump(state, f)
            logger.info(f"[OOD] State saved (v{STATE_FORMAT_VERSION}) to {self._model_path}")
        except Exception as e:
            logger.warning(f"[OOD] Failed to save state: {e}")

    def _load_state(self) -> None:
        """Load previously fitted state. Version mismatch → refit required.

        v1 (Phase 26 Gaussian): has "mean" and "precision" keys in regime stats.
        v2 (Phase 27 Wasserstein): has "ref_distributions" and "w1_threshold".

        If a v1 file is found, we log a warning and mark as NOT fitted so the
        next scheduler refit cycle regenerates it in v2 format.
        """
        try:
            if not os.path.exists(self._model_path):
                return

            with open(self._model_path) as f:
                state = json.load(f)

            file_version = int(state.get("format_version", 1))

            if file_version != STATE_FORMAT_VERSION:
                logger.warning(
                    f"[OOD] State format v{file_version} found, expected v{STATE_FORMAT_VERSION}. "
                    f"Auto-refit required (scheduler._ood_refit, Sun 04:15 UTC, "
                    f"or call detector.fit(...) with ≥30 trades). Running in fallback mode "
                    f"(defensive_multiplier=1.0) until refit completes."
                )
                # Do NOT load incompatible Gaussian stats — stay unfitted so detect()
                # short-circuits with defensive_multiplier=1.0 (safe default, no false OOD).
                self._fitted = False
                self._global_stats = None
                self._regime_stats = {}
                return

            self._global_stats = state.get("global")
            self._regime_stats = state.get("regimes", {})
            if self._global_stats or self._regime_stats:
                self._fitted = True
                n_regimes = len(self._regime_stats)
                logger.info(
                    f"[OOD] Loaded state v{file_version}: {n_regimes} regimes from {self._model_path}"
                )
        except Exception as e:
            logger.debug(f"[OOD] No saved state loaded: {e}")
            self._fitted = False
            self._global_stats = None
            self._regime_stats = {}
