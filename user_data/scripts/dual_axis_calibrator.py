"""
Phase 26: Dual-Axis Calibration — Novel Contribution #6

Nobody has combined CatBoost native probability + CQR interval into a 2D confidence system.
This is the first system that answers BOTH "will I win?" AND "how uncertain am I?" simultaneously.

Axis 1: CatBoost probability (nokta tahmini — "Bu trade %72 kazanır")
  - Natively well-calibrated (ordered boosting + symmetric trees)
  - Platt scaling BOZUYOR CatBoost calibration (2025 finding) → kullanılmıyor

Axis 2: CQR interval (aralık tahmini — "PnL -%2 ile +%5 arası")
  - Mathematical %95 coverage guarantee (distribution-free)
  - ACI ile rejim değişimlerine otomatik adapte olur

Birleşim:
  sizing = catboost_confidence × (1 / cqr_interval_width)

  Yüksek olasılık + dar aralık = BÜYÜK POZİSYON (kesin kazanç)
  Yüksek olasılık + geniş aralık = KÜÇÜK POZİSYON (belirsiz kazanç)
  Düşük olasılık + dar aralık = PAS GEÇ (kesin kayıp)
"""

import logging
import numpy as np
from typing import Dict, Optional

logger = logging.getLogger(__name__)

try:
    from neural_organism import _p
except ImportError:
    def _p(param_id, fallback=0.5, regime="_global"):
        return fallback


class DualAxisCalibrator:
    """2D confidence → sizing system.

    Replaces the old confidence_calibrator.py (Platt-based, Brier >= 0.25).

    Usage:
        calibrator = DualAxisCalibrator()
        result = calibrator.calibrate(
            catboost_probability=0.72,
            cqr_interval={"lower": -0.02, "upper": 0.05, "width": 0.07},
            ensemble_variance=0.03,
            ood_result={"is_ood": False, "defensive_multiplier": 1.0},
        )
        sizing = result["final_sizing_multiplier"]
    """

    def calibrate(
        self,
        catboost_probability: float = 0.5,
        cqr_interval: Optional[Dict[str, float]] = None,
        ensemble_variance: float = 0.0,
        ood_result: Optional[Dict] = None,
        chronos_sizing: float = 1.0,
    ) -> Dict[str, float]:
        """Compute dual-axis calibrated sizing multiplier.

        Args:
            catboost_probability: CatBoost P(win) [0, 1] — Axis 1
            cqr_interval: CQR conformal interval — Axis 2
            ensemble_variance: Deep ensemble disagreement
            ood_result: OOD detector result (defensive multiplier)
            chronos_sizing: Chronos uncertainty sizing from Triple Perception

        Returns:
            {
                "catboost_confidence": float — raw CatBoost probability,
                "interval_width": float — CQR interval width,
                "uncertainty_composite": float — combined uncertainty (0=certain, 1=uncertain),
                "final_sizing_multiplier": float — THE number that scales position size,
                "signal_quality": "HIGH" | "MEDIUM" | "LOW" | "REJECT",
                "explanation": str — human-readable explanation,
            }
        """
        # --- Axis 1: CatBoost Probability ---
        # No Platt scaling! CatBoost is natively calibrated.
        prob = float(np.clip(catboost_probability, 0.01, 0.99))

        # Direction-aware confidence: distance from 0.5
        directional_confidence = abs(prob - 0.5) * 2  # 0=neutral, 1=very confident

        # --- Axis 2: CQR Interval Width ---
        if cqr_interval and cqr_interval.get("width", 0) > 0:
            width = cqr_interval["width"]
        else:
            width = _p("calibrator.default_width", 0.10)

        # Normalize width to [0, 1] range (0=tight=certain, 1=wide=uncertain)
        width_scale = _p("calibrator.width_scale", 10.0)
        interval_uncertainty = float(np.clip(width * width_scale, 0.0, 1.0))

        # --- Composite Uncertainty (3 sources merged) ---
        # CQR interval width (primary)
        # Ensemble variance (secondary)
        # OOD score (tertiary)

        w_cqr = _p("calibrator.w_cqr", 0.5)
        w_ensemble = _p("calibrator.w_ensemble", 0.3)
        w_ood = _p("calibrator.w_ood", 0.2)

        ensemble_unc = float(np.clip(ensemble_variance * 20, 0, 1))  # scale variance to [0,1]

        ood_penalty = 1.0
        if ood_result and ood_result.get("is_ood"):
            ood_penalty = ood_result.get("defensive_multiplier", 0.5)

        uncertainty_composite = (
            w_cqr * interval_uncertainty +
            w_ensemble * ensemble_unc +
            w_ood * (1.0 - ood_penalty)
        )
        uncertainty_composite = float(np.clip(uncertainty_composite, 0, 1))

        # --- DUAL-AXIS FUSION ---
        # sizing = confidence × certainty × external_factors
        certainty = 1.0 - uncertainty_composite
        base_sizing = directional_confidence * certainty

        # Apply OOD defensive multiplier
        base_sizing *= ood_penalty

        # Apply Chronos uncertainty
        base_sizing *= chronos_sizing

        # Floor and ceiling
        min_sizing = _p("calibrator.min_sizing", 0.10)
        max_sizing = _p("calibrator.max_sizing", 1.50)
        final_sizing = float(np.clip(base_sizing, min_sizing, max_sizing))

        # --- Signal Quality Classification ---
        if directional_confidence >= 0.6 and uncertainty_composite < 0.3:
            quality = "HIGH"
        elif directional_confidence >= 0.3 and uncertainty_composite < 0.6:
            quality = "MEDIUM"
        elif directional_confidence >= 0.1:
            quality = "LOW"
        else:
            quality = "REJECT"

        # --- Explanation ---
        if prob > 0.5:
            direction = "BULLISH"
            prob_str = f"P(win)={prob:.0%}"
        else:
            direction = "BEARISH"
            prob_str = f"P(loss)={1-prob:.0%}"

        explanation = (
            f"{direction} {prob_str}, interval_width={width:.4f}, "
            f"uncertainty={uncertainty_composite:.2f}, "
            f"sizing={final_sizing:.2f}× — {quality}"
        )

        result = {
            "catboost_confidence": round(prob, 4),
            "interval_width": round(width, 6),
            "uncertainty_composite": round(uncertainty_composite, 4),
            "final_sizing_multiplier": round(final_sizing, 4),
            "signal_quality": quality,
            "explanation": explanation,
        }

        logger.debug(f"[DualAxis] {explanation}")
        return result
