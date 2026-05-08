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
import os
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
        # Phase 27 Task 13 — attributes exist immediately so callers that check
        # availability before the first perceive() don't AttributeError.
        self._kronos_model = None
        self._kronos_predictor = None
        self._kronos_available = False
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

        # Phase 27 Task 13 (C5 Ajani): Kronos-mini — optional 4th perception
        # stream (OHLCV-native financial foundation model). Disabled by default;
        # opt-in via HQ_ENABLE_KRONOS=1 so CPU inference cost is a conscious
        # choice. Failure to load is NOT fatal — the pipeline still runs as
        # triple perception. (Attributes were pre-seeded in __init__.)
        self._try_load_kronos()

    def _try_load_kronos(self) -> None:
        """Phase 27 Task 13 (default-on): load NeoQuasar/Kronos-mini via the
        vendored loader (kronos_vendor) — upstream HF AutoModel can't dispatch
        the custom Kronos model_type, so we use the upstream classes via
        snapshot_download + dynamic import. Falls back to a small Transformer
        encoder if the snapshot mirror strips the Python source.
        """
        if os.environ.get("HQ_ENABLE_KRONOS", "1") == "0":
            logger.debug("[TriplePerception:Kronos] explicitly disabled (HQ_ENABLE_KRONOS=0)")
            return
        try:
            import time as _time
            from kronos_vendor import Kronos, KronosTokenizer, KronosPredictor
        except Exception as e:
            logger.info(f"[TriplePerception:Kronos] vendor import failed: {e}; skipping")
            return
        try:
            tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-2k")
            model = Kronos.from_pretrained("NeoQuasar/Kronos-mini")
            self._kronos_predictor = KronosPredictor(
                model, tokenizer, device="cpu", max_context=2048,
            )
        except Exception as e:
            logger.warning(
                f"[TriplePerception:Kronos] vendor load failed ({str(e)[:140]}); "
                "staying with triple perception"
            )
            return
        # CPU benchmark — single dummy predict to enforce the 200ms budget.
        try:
            import pandas as _pd
            dummy_df = _pd.DataFrame({
                "open": [1.0] * 64, "high": [1.01] * 64,
                "low": [0.99] * 64, "close": [1.0] * 64,
                "volume": [1000.0] * 64,
            })
            t0 = _time.perf_counter()
            self._kronos_predictor.predict(dummy_df, pred_len=4)
            elapsed_ms = (_time.perf_counter() - t0) * 1000.0
            if elapsed_ms > 200.0:
                logger.warning(
                    f"[TriplePerception:Kronos] benchmark {elapsed_ms:.0f}ms > 200ms; disabling"
                )
                self._kronos_predictor = None
                return
            self._kronos_model = model
            self._kronos_available = True
            upstream = (model.upstream_loaded if hasattr(model, "upstream_loaded") else False)
            logger.info(
                f"[TriplePerception:Kronos] loaded via vendor (upstream={upstream}, "
                f"benchmark {elapsed_ms:.0f}ms)"
            )
        except Exception as e:
            logger.warning(
                f"[TriplePerception:Kronos] benchmark failed ({e}); staying with triple perception"
            )
            self._kronos_predictor = None

    def _kronos_predict(self, df) -> Optional[float]:
        """Run Kronos via the vendored predictor over the last 64 candles.

        Returns a scalar direction score in [-1, +1]: (predicted_close /
        last_close − 1) of the 4-step horizon mean, clamped. None when the
        predictor isn't loaded or inference fails.
        """
        predictor = getattr(self, "_kronos_predictor", None)
        if not self._kronos_available or predictor is None:
            return None
        try:
            tail = df.tail(64)
            if len(tail) < 10:
                return None
            forecast = predictor.predict(tail, pred_len=4)
            if forecast is None or len(forecast) == 0:
                return None
            try:
                last_close = float(tail["close"].iloc[-1])
            except Exception:
                return None
            if last_close <= 0:
                return None
            mean_pred = float(forecast.mean()) if hasattr(forecast, "mean") else float(forecast[-1])
            direction = (mean_pred / last_close) - 1.0
            # Scale into [-1, +1] — anything beyond ±5% caps to the boundary.
            return max(-1.0, min(1.0, direction * 20.0))
        except Exception as e:
            logger.debug(f"[TriplePerception:Kronos] predict failed: {e}")
            return None

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
                # Data Acceleration audit fix 4: record the checkpoint file
                # mtime + path so _reload_catboost_if_updated can detect new
                # weekly retrain artefacts without restarting the bot.
                self._catboost_path = model_path
                try:
                    self._catboost_mtime = os.path.getmtime(model_path)
                except Exception:
                    self._catboost_mtime = 0.0
                logger.info(f"[TriplePerception] CatBoost loaded from {model_path}")
            except Exception as e:
                logger.warning(f"[TriplePerception] CatBoost load error: {e}")
        else:
            logger.info("[TriplePerception] No pre-trained CatBoost model yet. Will use TTM+Chronos only until first training.")

    def _reload_catboost_if_updated(self) -> None:
        """Data Acceleration audit fix 4: hot-reload CatBoost when the weekly
        retrain writes a newer `.cbm` file. Called at the top of perceive()
        so a fresh model takes effect without a bot restart.
        """
        import os
        path = getattr(self, "_catboost_path", None)
        if not path or not os.path.exists(path):
            return
        try:
            current_mtime = os.path.getmtime(path)
        except Exception:
            return
        if current_mtime <= getattr(self, "_catboost_mtime", 0.0):
            return
        try:
            from catboost import CatBoostClassifier
            new_model = CatBoostClassifier()
            new_model.load_model(path)
            self._catboost_model = new_model
            self._catboost_mtime = current_mtime
            self._catboost_available = True
            logger.info(
                f"[TriplePerception:HotReload] CatBoost re-loaded "
                f"(mtime={current_mtime:.0f}) from {path}"
            )
        except Exception as e:
            logger.warning(f"[TriplePerception:HotReload] failed: {e}")

    def perceive(
        self,
        df_1h: pd.DataFrame,
        df_4h: Optional[pd.DataFrame] = None,
        df_1d: Optional[pd.DataFrame] = None,
        ee_subscores: Optional[Dict[str, float]] = None,
        chart_features: Optional[Dict[str, float]] = None,
        pair: Optional[str] = None,
        timeframe: str = "1h",
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
        # Data Acceleration audit fix 4: check for a fresh CatBoost checkpoint
        # every perceive() call. Cheap (one stat call); skips re-load if
        # mtime unchanged. Weekly retrain artefacts are picked up without
        # restarting the bot.
        self._reload_catboost_if_updated()
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
            "components_available": {"ttm": False, "chronos": False,
                                     "catboost": False, "kronos": False},
            "kronos_direction": 0.0,
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
                # Phase 27 Fix 4 (G3): publish natural-language SHAP so MADAM can
                # see which neural-model features actually drove the prediction.
                result["shap_narrative"] = catboost_result.get("shap_narrative", "")
                if result["shap_narrative"]:
                    try:
                        from pheromone_field import get_pheromone_field, PheromoneField
                        get_pheromone_field().deposit(
                            "triple_perception", PheromoneField.SIGNAL_SHAP,
                            result["shap_narrative"], half_life=300.0,
                        )
                    except Exception:
                        pass
                result["components_available"]["catboost"] = True
            except Exception as e:
                logger.warning(f"[TriplePerception] CatBoost predict failed: {e}")

        # --- 3b. Kronos-mini (optional, Task 13) ---
        # Disabled unless HQ_ENABLE_KRONOS=1 set at process startup. When active,
        # the scalar direction score is added to result so _fuse / MADAM can
        # use it. We log the raw score even when we don't yet weight it into
        # the final signal — avoids the "defined but never called" dead code
        # audit finding.
        if self._kronos_available:
            try:
                kronos_dir = self._kronos_predict(df_1h)
                if kronos_dir is not None:
                    result["kronos_direction"] = float(kronos_dir)
                    result["components_available"]["kronos"] = True
                    logger.info(f"[TriplePerception:Kronos] direction={kronos_dir:+.3f}")
            except Exception as e:
                logger.debug(f"[TriplePerception:Kronos] predict error: {e}")

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
            # Audit fix (2026-04-19): DualAxis was returning 0.10 floor on every
            # uncalibrated pair, multiplying through to $0.02 stakes → 100%
            # of signals shadow-routed → ZERO real trades for 2 days. CAAT in
            # custom_stake_amount already applies its own 8-part sizing
            # discipline; we only need DualAxis to add information when it
            # has high-confidence intervals, NOT to pessimistically halve
            # everything during cold-start. Hard floor 0.30.
            result["sizing_multiplier"] = max(0.30, float(dual_result["final_sizing_multiplier"]))
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

        if pair and result["components_available"].get("ttm") and df_1h is not None and len(df_1h):
            self._persist_world_model_state(
                pair=pair, timeframe=timeframe, df_1h=df_1h,
                ttm_embedding=result["ttm_embedding"],
                chart_features=chart_features,
            )

        # ═══ PHASE 30 D.7 — Quintuple Perception (5th + 6th source overlay) ═══
        # Adds long-horizon (TimesFM-style) and visual chart pattern (YOLO)
        # alongside existing 9-stage triple_perception output. Result fields:
        # - quintuple_direction, quintuple_confidence, quintuple_n_sources,
        #   quintuple_disagreement_penalty, quintuple_breakdown
        # When foundation/long-horizon/YOLO weights are absent the modules
        # gracefully fall back; this is observation-mode and never overrides
        # the primary fused signal — sizing_multiplier remains canonical.
        try:
            from quintuple_perception import collect_votes, fuse
            _closes_1m = []
            try:
                if df_1h is not None and len(df_1h):
                    _closes_1m = list(df_1h["close"].tail(120).astype(float).values)
            except Exception:
                _closes_1m = []
            _closes_1h = list(_closes_1m)
            _votes = collect_votes(
                pair=pair or "",
                ohlcv_close=_closes_1m,
                ohlcv_1h_close=_closes_1h,
                png_path=None,
            )
            if _votes:
                _fused = fuse(_votes)
                result["quintuple_direction"] = _fused.direction
                result["quintuple_confidence"] = _fused.confidence
                result["quintuple_n_sources"] = _fused.n_sources
                result["quintuple_disagreement_penalty"] = _fused.disagreement_penalty
                result["quintuple_breakdown"] = _fused.breakdown
                # Telemetry — record observation-mode fused signal
                try:
                    from telemetry import record as _phase30_tm
                    _phase30_tm(
                        kind="perception.quintuple",
                        severity="debug",
                        source_module="triple_perception",
                        payload={
                            "pair": pair, "n_sources": _fused.n_sources,
                            "direction": round(_fused.direction, 3),
                            "confidence": round(_fused.confidence, 3),
                            "disagreement": round(_fused.disagreement_penalty, 3),
                        },
                    )
                except Exception:
                    pass
        except Exception:
            pass

        return result

    def _persist_world_model_state(
        self,
        pair: str,
        timeframe: str,
        df_1h: pd.DataFrame,
        ttm_embedding: np.ndarray,
        chart_features: Optional[Dict[str, float]],
    ) -> None:
        """Write the current perception snapshot to `world_model_states`.

        This table is the input to world_model.train_from_buffer() AND the
        source for TTM/Chronos fine-tuning pipelines. Before the Signal
        Quality sprint (2026-04-20) no module actually wrote to it — only
        readers existed — so WorldModel perpetually reported
        `insufficient data: 0 < 64` and dream_scenarios stayed empty.

        One row per (pair, timeframe, timestamp) — UNIQUE constraint makes
        the write idempotent when perceive() fires multiple times within a
        single candle. Schema columns missing from the live perception
        (`fng`, `hormones_json`) are left NULL; downstream readers already
        tolerate NULLs there.
        """
        try:
            import json as _json
            from datetime import datetime as _dt, timezone as _tz
            from db import get_db_connection

            last_ts = df_1h.index[-1] if hasattr(df_1h, "index") else _dt.now(tz=_tz.utc)
            if hasattr(last_ts, "to_pydatetime"):
                last_ts = last_ts.to_pydatetime()
            if getattr(last_ts, "tzinfo", None) is None:
                last_ts = last_ts.replace(tzinfo=_tz.utc) if hasattr(last_ts, "replace") else last_ts
            ts_iso = last_ts.isoformat() if hasattr(last_ts, "isoformat") else str(last_ts)

            embedding_bytes = np.asarray(ttm_embedding, dtype=np.float32).tobytes()
            chart_json = _json.dumps(chart_features) if chart_features else None

            regime = None
            try:
                from rag_graph import _resolve_pair_regime
                regime = _resolve_pair_regime(pair)
            except Exception:
                pass

            conn = get_db_connection()
            try:
                conn.execute(
                    """INSERT OR REPLACE INTO world_model_states
                       (pair, timeframe, timestamp, ttm_embedding,
                        chart_features_json, regime, fng, hormones_json)
                       VALUES (?, ?, ?, ?, ?, ?, NULL, NULL)""",
                    (pair, timeframe, ts_iso, embedding_bytes, chart_json, regime),
                )
                conn.commit()
            finally:
                conn.close()
        except Exception as e:
            logger.debug(f"[TriplePerception:Persist] {pair} {timeframe} failed: {e}")

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

        # Evidence Engine sub-scores (both ee_ and sub_ prefixed for trainer compatibility)
        if ee_subscores:
            for k, v in ee_subscores.items():
                features[f"ee_{k}"] = float(v)
                features[f"sub_{k}"] = float(v)  # Phase 28: trainer uses sub_ prefix

        # Phase 28: Add trainer-compatible features from result context
        # These match catboost_trainer.py feature_names
        features["confidence"] = features.get("ee_confidence", 0.5)
        features["trust_score"] = features.get("ee_trust", 0.5)
        features["max_cap"] = features.get("ee_max_cap", 0.35)
        regime_map = {"trending_bull": 4, "ranging": 2, "transitional": 3,
                      "trending_bear": 1, "high_volatility": 5}
        features["regime_code"] = regime_map.get(
            features.get("regime", "transitional"), 3)

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

        # MultiClass: [P(bearish), P(neutral), P(bullish)]
        if len(proba) >= 3:
            probability = float(proba[2])  # P(bullish)
        elif len(proba) >= 2:
            probability = float(proba[1])  # Binary fallback: P(bullish)
        else:
            probability = float(proba[0])

        # SHAP (if available) — Phase 26 returned raw importances; Phase 27 Fix 4
        # (G3) also turns them into natural-language for MADAM / rag_graph.
        shap_top = []
        shap_signed: list = []  # list of (feature, signed_shap)
        try:
            shap_values = self._catboost_model.get_feature_importance(
                data=feature_df, type="ShapValues"
            )
            if shap_values is not None:
                raw = shap_values[0][:-1]  # exclude bias term
                importances = abs(raw)
                top_idx = np.argsort(importances)[-5:][::-1]
                shap_top = [
                    {"feature": expected_cols[i], "importance": round(float(importances[i]), 4)}
                    for i in top_idx
                ]
                shap_signed = [
                    (expected_cols[i], float(raw[i])) for i in top_idx
                ]
        except Exception:
            pass

        # Phase 27 Fix 4: build natural-language SHAP narrative for MADAM.
        direction = "BULLISH" if probability >= 0.55 else "BEARISH" if probability <= 0.45 else "NEUTRAL"
        shap_narrative = self._format_shap_narrative(shap_signed, direction, probability)

        return {
            "probability": probability,
            "shap_top": shap_top,
            "shap_narrative": shap_narrative,
        }

    def _format_shap_narrative(self, shap_signed, prediction: str,
                                probability: float) -> str:
        """Phase 27 Fix 4 (G3): CatBoost SHAP → MADAM-readable explanation."""
        if not shap_signed:
            return ""
        direction_word = {"BULLISH": "LONG", "BEARISH": "SHORT"}.get(prediction, "NEUTRAL")
        lines = [
            f"CatBoost predicts {direction_word} ({probability:.0%} probability).",
            "Top contributing features:",
        ]
        for feat, shap_val in shap_signed:
            direction = "supporting" if shap_val > 0 else "opposing"
            lines.append(f"  • {feat}: {shap_val:+.3f} ({direction})")
        supporting = sum(1 for _, v in shap_signed if v > 0)
        opposing = len(shap_signed) - supporting
        if opposing == 0:
            lines.append("All top features AGREE on direction — high conviction.")
        elif opposing >= 3:
            lines.append(
                f"WARNING: {opposing}/{len(shap_signed)} top features OPPOSE the prediction — "
                "mixed signal, reduce sizing."
            )
        return "\n".join(lines)

    def _fuse(self, result: Dict) -> Dict:
        """Fuse TTM + Chronos + CatBoost (+ optional Kronos) into final signal."""
        ttm_dir = result["ttm_direction"]
        chronos_p50 = result["chronos_p50"]
        catboost_prob = result["catboost_probability"]
        has_catboost = result["components_available"]["catboost"]
        # Phase 27 Task 13: Kronos adds a fourth direction vote when active.
        kronos_dir = result.get("kronos_direction", 0.0)
        has_kronos = result["components_available"].get("kronos", False)

        # Disagreement: how much do the models disagree?
        signals = []
        if abs(ttm_dir) > 0.01:
            signals.append(1.0 if ttm_dir > 0 else -1.0)
        if abs(chronos_p50) > 0.001:
            signals.append(1.0 if chronos_p50 > 0 else -1.0)
        if has_catboost:
            signals.append(1.0 if catboost_prob > 0.5 else -1.0)
        if has_kronos and abs(kronos_dir) > 0.01:
            signals.append(1.0 if kronos_dir > 0 else -1.0)

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
