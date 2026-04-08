"""
Phase 26: Triple Perception — TTM + Chronos-Bolt + CatBoost Fusion

Three different architectures, three different perspectives, ONE fused signal.
  TTM:          Directional signal (up/down/flat + magnitude) — MLP-Mixer, multivariate
  Chronos-Bolt: Uncertainty distribution (P10-P90 quantiles) — Transformer, quantile
  CatBoost:     Final decision (calibrated probability + SHAP) — Gradient boosting, tabular king

Fusion logic:
  direction = TTM (best at directional detection)
  uncertainty = Chronos-Bolt (best at uncertainty quantification)
  final_decision = CatBoost (best at combining all features into a prediction)
  sizing_multiplier = 1 / (1 + 5 * interval_width) — Chronos uncertainty scales position

Novel Contribution #7: Triple Perception Ensemble (TTM + Chronos-Bolt + CatBoost)
No existing system combines MLP-Mixer + quantile Transformer + gradient boosting.
"""

import logging
import time
import numpy as np
import pandas as pd
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# Phase 24: Neural Organism adaptive parameters
try:
    from neural_organism import _p
except ImportError:
    def _p(param_id, fallback=0.5, regime="_global"):
        return fallback


class TriplePerception:
    """Fuses TTM embedding + Chronos uncertainty + CatBoost decision.

    This runs as Tier-2 (<5s) in the signal pipeline, BEFORE LLM/RAG analysis.
    Evidence Engine gets the benefit of ML perception as additional input.
    """

    def __init__(self):
        self._catboost_model = None
        self._catboost_available = False
        self._initialized = False

    def _get_or_create(self, attr_name: str, factory):
        """Lazy singleton helper — create instance on first use, reuse after."""
        if not hasattr(self, attr_name) or getattr(self, attr_name) is None:
            try:
                obj = factory() if callable(factory) else factory
                setattr(self, attr_name, obj)
            except Exception as e:
                logger.debug(f"[TriplePerception] {attr_name} init failed: {e}")
                setattr(self, attr_name, None)
        return getattr(self, attr_name)

    def _ensure_init(self):
        """Lazy initialization — models loaded on first call."""
        if self._initialized:
            return
        self._initialized = True

        # CatBoost standalone model for signal prediction
        try:
            self._load_catboost()
        except Exception as e:
            logger.warning(f"[TriplePerception] CatBoost load failed: {e}")

    def _load_catboost(self):
        """Load pre-trained CatBoost signal prediction model from SQLite/file."""
        import os
        model_path = os.path.join(
            os.path.dirname(__file__), "..", "models", "catboost_signal_v1.cbm"
        )
        if os.path.exists(model_path):
            try:
                from catboost import CatBoostClassifier
                self._catboost_model = CatBoostClassifier()
                self._catboost_model.load_model(model_path)
                self._catboost_available = True
                logger.info(f"[TriplePerception] CatBoost loaded from {model_path}")
            except Exception as e:
                logger.warning(f"[TriplePerception] CatBoost load error: {e}")
        else:
            logger.info("[TriplePerception] No pre-trained CatBoost model yet. Will use TTM+Chronos only until first training.")

    def perceive(
        self,
        df_1h: pd.DataFrame,
        df_4h: Optional[pd.DataFrame] = None,
        df_1d: Optional[pd.DataFrame] = None,
        ee_subscores: Optional[Dict[str, float]] = None,
        chart_features: Optional[Dict[str, float]] = None,
    ) -> Dict[str, any]:
        """Run Triple Perception pipeline.

        Args:
            df_1h: Primary 1h OHLCV DataFrame
            df_4h: Optional 4h DataFrame
            df_1d: Optional 1d DataFrame
            ee_subscores: Evidence Engine sub-scores (Q1-Q6)
            chart_features: Pre-computed chart structure features (from chart_features.py)

        Returns:
            {
                "signal": "BULLISH" | "BEARISH" | "NEUTRAL",
                "confidence": float (0.0-1.0, from CatBoost or fusion),
                "sizing_multiplier": float (Chronos uncertainty → position scaling),
                "ttm_direction": float (-1 to 1),
                "ttm_magnitude": float (0 to 1),
                "ttm_embedding": np.ndarray (64-dim),
                "chronos_p10": float, "chronos_p50": float, "chronos_p90": float,
                "chronos_interval_width": float,
                "catboost_probability": float (native calibrated, Novel #6),
                "disagreement": float (0-1, how much models disagree → uncertainty),
                "shap_top_features": list (top 5 feature importances),
                "latency_ms": float,
                "components_available": {"ttm": bool, "chronos": bool, "catboost": bool}
            }
        """
        self._ensure_init()
        start = time.time()

        result = {
            "signal": "NEUTRAL",
            "confidence": 0.0,
            "sizing_multiplier": 1.0,
            "ttm_direction": 0.0,
            "ttm_magnitude": 0.0,
            "ttm_embedding": np.zeros(64),
            "chronos_p10": 0.0, "chronos_p50": 0.0, "chronos_p90": 0.0,
            "chronos_interval_width": 0.1,
            "catboost_probability": 0.5,
            "disagreement": 0.0,
            "shap_top_features": [],
            "latency_ms": 0.0,
            "components_available": {"ttm": False, "chronos": False, "catboost": False},
        }

        # --- 1. TTM: Directional Signal + Embedding ---
        try:
            from ttm_perception import compute_ttm_embedding
            ttm = compute_ttm_embedding(df_1h)
            result["ttm_direction"] = ttm["direction"]
            result["ttm_magnitude"] = ttm["magnitude"]
            result["ttm_embedding"] = ttm["embedding"]
            result["components_available"]["ttm"] = ttm["available"]
        except Exception as e:
            logger.warning(f"[TriplePerception] TTM failed: {e}")

        # --- 2. Chronos-Bolt: Uncertainty Quantiles ---
        try:
            from chronos_perception import compute_chronos_quantiles
            chronos = compute_chronos_quantiles(df_1h)
            result["chronos_p10"] = chronos["p10"]
            result["chronos_p50"] = chronos["p50"]
            result["chronos_p90"] = chronos["p90"]
            result["chronos_interval_width"] = chronos["interval_width"]
            result["components_available"]["chronos"] = chronos["available"]

            # Sizing multiplier from uncertainty (narrow interval = larger position)
            uncertainty_scale = _p("perception.uncertainty_scale", 5.0)
            result["sizing_multiplier"] = 1.0 / (1.0 + uncertainty_scale * chronos["interval_width"])
        except Exception as e:
            logger.warning(f"[TriplePerception] Chronos failed: {e}")

        # --- 3. CatBoost: Final Decision (if model available) ---
        if self._catboost_available and self._catboost_model is not None:
            try:
                catboost_result = self._catboost_predict(
                    df_1h, result["ttm_embedding"], ee_subscores, chart_features
                )
                result["catboost_probability"] = catboost_result["probability"]
                result["shap_top_features"] = catboost_result.get("shap_top", [])
                result["components_available"]["catboost"] = True
            except Exception as e:
                logger.warning(f"[TriplePerception] CatBoost predict failed: {e}")

        # --- 4. OOD Detection: "Bu piyasayı daha önce gördüm mü?" ---
        ood_result = {"is_ood": False, "defensive_multiplier": 1.0}
        try:
            from ood_detector import MarketOODDetector
            detector = self._get_or_create("_ood_detector", MarketOODDetector)
            if detector is not None:
                feature_dict = dict(chart_features) if chart_features else {}
                if result["components_available"]["ttm"]:
                    for i, v in enumerate(result["ttm_embedding"][:16]):
                        feature_dict[f"ttm_{i}"] = float(v)
                ood_result = detector.detect(feature_dict)
            if ood_result["is_ood"]:
                result["sizing_multiplier"] *= ood_result["defensive_multiplier"]
                logger.warning(f"[TriplePerception:OOD] dist={ood_result['distance']:.2f} → defensive_mult={ood_result['defensive_multiplier']:.2f}")
        except Exception as e:
            logger.debug(f"[TriplePerception:OOD] {e}")

        # --- 5. Deep Ensemble: Model disagreement → uncertainty ---
        ensemble_variance = 0.0
        try:
            from deep_ensemble import DeepEnsemble
            ensemble = self._get_or_create("_ensemble", lambda: DeepEnsemble(input_dim=50))
            if ensemble is not None and ensemble._fitted:
                feature_vec = np.array([chart_features.get(k, 0.0) for k in sorted(chart_features.keys())[:50]]) if chart_features else np.zeros(50)
                ens_result = ensemble.predict_with_uncertainty(feature_vec)
                ensemble_variance = ens_result["variance"]
                result["sizing_multiplier"] *= ens_result["sizing_multiplier"]
        except Exception as e:
            logger.debug(f"[TriplePerception:Ensemble] {e}")

        # --- 6. Conformal Calibrator: CQR interval with ACI ---
        cqr_interval = None
        try:
            from conformal_calibrator import ConformalCalibrator
            conformal = self._get_or_create("_conformal", ConformalCalibrator)
            if conformal is None:
                raise ValueError("Conformal calibrator unavailable")
            cqr_interval = conformal.predict_interval(
                prediction=result["chronos_p50"],
                lower_quantile=result["chronos_p10"],
                upper_quantile=result["chronos_p90"],
            )
        except Exception as e:
            logger.debug(f"[TriplePerception:Conformal] {e}")

        # --- 7. Dual-Axis Calibration (Novel #6): Final sizing ---
        try:
            from dual_axis_calibrator import DualAxisCalibrator
            dual_cal = self._get_or_create("_dual_cal", DualAxisCalibrator)
            if dual_cal is None:
                raise ValueError("DualAxis calibrator unavailable")
            dual_result = dual_cal.calibrate(
                catboost_probability=result["catboost_probability"],
                cqr_interval=cqr_interval,
                ensemble_variance=ensemble_variance,
                ood_result=ood_result,
                chronos_sizing=result["sizing_multiplier"],
            )
            result["sizing_multiplier"] = dual_result["final_sizing_multiplier"]
            result["signal_quality"] = dual_result["signal_quality"]
            result["dual_axis_explanation"] = dual_result["explanation"]
        except Exception as e:
            logger.debug(f"[TriplePerception:DualAxis] {e}")

        # --- 8. FUSION: Combine three perspectives ---
        result.update(self._fuse(result))

        # --- 9. PHEROMONE: Deposit perception results for other modules ---
        try:
            from pheromone_field import get_pheromone_field, PheromoneField
            field = get_pheromone_field()
            field.deposit("triple_perception", PheromoneField.SIGNAL_PREDICTION, {
                "signal": result["signal"],
                "confidence": result["confidence"],
                "sizing_multiplier": result["sizing_multiplier"],
                "disagreement": result["disagreement"],
            }, half_life=120)  # 2 min — valid until next candle
            field.deposit("triple_perception", PheromoneField.SIGNAL_UNCERTAINTY, {
                "interval_width": result["chronos_interval_width"],
                "ensemble_disagreement": result["disagreement"],
            }, half_life=120)
        except Exception:
            pass  # Pheromone is advisory — never block trade

        result["latency_ms"] = round((time.time() - start) * 1000, 1)
        logger.debug(
            f"[TriplePerception] {result['signal']} conf={result['confidence']:.2f} "
            f"sizing={result['sizing_multiplier']:.2f} in {result['latency_ms']:.0f}ms "
            f"(TTM={'Y' if result['components_available']['ttm'] else 'N'} "
            f"Chr={'Y' if result['components_available']['chronos'] else 'N'} "
            f"CB={'Y' if result['components_available']['catboost'] else 'N'})"
        )

        return result

    def _catboost_predict(
        self,
        df: pd.DataFrame,
        ttm_embedding: np.ndarray,
        ee_subscores: Optional[Dict[str, float]],
        chart_features: Optional[Dict[str, float]],
    ) -> Dict:
        """Run CatBoost prediction with all available features."""
        # Build feature vector
        features = {}

        # Raw indicators from last candle
        last = df.iloc[-1]
        for col in ["rsi", "adx", "atr", "macd", "macdhist"]:
            if col in df.columns:
                val = last[col]
                features[col] = float(val) if not pd.isna(val) else 0.0

        # TTM embedding as features
        for i, v in enumerate(ttm_embedding[:64]):
            features[f"ttm_{i}"] = float(v)

        # Evidence Engine sub-scores
        if ee_subscores:
            for k, v in ee_subscores.items():
                features[f"ee_{k}"] = float(v)

        # Chart structure features
        if chart_features:
            features.update(chart_features)

        # Create DataFrame for CatBoost
        feature_df = pd.DataFrame([features])

        # Align columns with model's expected features
        expected_cols = self._catboost_model.feature_names_
        for col in expected_cols:
            if col not in feature_df.columns:
                feature_df[col] = 0.0
        feature_df = feature_df[expected_cols]

        # Predict probability
        proba = self._catboost_model.predict_proba(feature_df)[0]

        # Assume binary classification: [P(bearish), P(bullish)]
        if len(proba) >= 2:
            probability = float(proba[1])  # P(bullish)
        else:
            probability = float(proba[0])

        # SHAP (if available)
        shap_top = []
        try:
            shap_values = self._catboost_model.get_feature_importance(
                data=feature_df, type="ShapValues"
            )
            if shap_values is not None:
                importances = abs(shap_values[0][:-1])  # exclude bias term
                top_idx = np.argsort(importances)[-5:][::-1]
                shap_top = [
                    {"feature": expected_cols[i], "importance": round(float(importances[i]), 4)}
                    for i in top_idx
                ]
        except Exception:
            pass

        return {"probability": probability, "shap_top": shap_top}

    def _fuse(self, result: Dict) -> Dict:
        """Fuse TTM + Chronos + CatBoost into final signal."""
        ttm_dir = result["ttm_direction"]
        chronos_p50 = result["chronos_p50"]
        catboost_prob = result["catboost_probability"]
        has_catboost = result["components_available"]["catboost"]

        # Disagreement: how much do the models disagree?
        signals = []
        if abs(ttm_dir) > 0.01:
            signals.append(1.0 if ttm_dir > 0 else -1.0)
        if abs(chronos_p50) > 0.001:
            signals.append(1.0 if chronos_p50 > 0 else -1.0)
        if has_catboost:
            signals.append(1.0 if catboost_prob > 0.5 else -1.0)

        if len(signals) >= 2:
            agreement = sum(signals) / len(signals)
            disagreement = 1.0 - abs(agreement)
        else:
            disagreement = 0.5

        # Final signal determination
        if has_catboost:
            # CatBoost is primary decision maker
            bull_threshold = _p("perception.bull_threshold", 0.55)
            bear_threshold = _p("perception.bear_threshold", 0.45)

            if catboost_prob > bull_threshold:
                signal = "BULLISH"
                confidence = catboost_prob
            elif catboost_prob < bear_threshold:
                signal = "BEARISH"
                confidence = 1.0 - catboost_prob
            else:
                signal = "NEUTRAL"
                confidence = 0.5 - abs(catboost_prob - 0.5)
        else:
            # TTM + Chronos only (no CatBoost yet)
            ttm_threshold = _p("perception.ttm_threshold", 0.1)

            if ttm_dir > ttm_threshold and chronos_p50 > 0:
                signal = "BULLISH"
                confidence = min(abs(ttm_dir) * result["ttm_magnitude"], 0.85)
            elif ttm_dir < -ttm_threshold and chronos_p50 < 0:
                signal = "BEARISH"
                confidence = min(abs(ttm_dir) * result["ttm_magnitude"], 0.85)
            else:
                signal = "NEUTRAL"
                confidence = 0.3

        # Disagreement penalty: high disagreement → lower confidence
        disagreement_penalty = _p("perception.disagreement_penalty", 0.3)
        confidence *= (1.0 - disagreement * disagreement_penalty)

        return {
            "signal": signal,
            "confidence": round(float(np.clip(confidence, 0.0, 1.0)), 4),
            "disagreement": round(float(disagreement), 4),
        }


# Module-level singleton
_triple_perception = None


def get_triple_perception() -> TriplePerception:
    """Get or create singleton TriplePerception instance."""
    global _triple_perception
    if _triple_perception is None:
        _triple_perception = TriplePerception()
    return _triple_perception
