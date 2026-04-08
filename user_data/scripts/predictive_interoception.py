"""
Phase 26: Predictive Interoception — Predict Own Failures BEFORE They Happen

Not just monitoring (reactive) — PREDICTING (proactive).
"10 dakika sonra latency spike olacak" → ŞİMDİDEN önlem al.

Hierarchical prediction:
  Low level:  "RAM kullanımım 30 dk sonra ne olacak?"
  Mid level:  "Bu modülün performansı düşecek mi?"
  High level: "Genel sağlığım bu hafta nereye gidiyor?"

Source: arXiv 2511.13668, 2025 — Interoceptive Predictive Coding
"""

import logging
import time
import os
import json
import numpy as np
from typing import Dict, List, Optional
from collections import deque
from datetime import datetime

logger = logging.getLogger(__name__)

try:
    from neural_organism import _p
except ImportError:
    def _p(param_id, fallback=0.5, regime="_global"):
        return fallback


class PredictiveInteroception:
    """Predicts organism's own health metrics BEFORE degradation happens.

    Maintains rolling windows of metrics, fits simple predictors,
    and triggers proactive responses when predictions indicate trouble.

    Metrics tracked:
      - signal_latency_ms: RAG pipeline latency
      - win_rate_7d: rolling 7-day win rate
      - ram_usage_pct: server RAM utilization
      - api_error_rate: LLM/embedding API error rate
      - organism_health: composite interoception score
      - prediction_error: how wrong our last predictions were
    """

    METRICS = [
        "signal_latency_ms",
        "win_rate_7d",
        "api_error_rate",
        "organism_health",
        "prediction_error_avg",
    ]

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self._history: Dict[str, deque] = {
            m: deque(maxlen=window_size) for m in self.METRICS
        }
        self._predictions: Dict[str, float] = {}  # last prediction per metric
        self._alerts: List[Dict] = []
        self._state_path = os.path.join(
            os.path.dirname(__file__), "..", "models", "interoception_state.json"
        )
        self._load_state()

    def record(self, metric: str, value: float) -> None:
        """Record a new metric observation."""
        if metric in self._history:
            self._history[metric].append({
                "value": value,
                "time": time.time(),
            })

    def predict_and_act(self) -> Dict[str, any]:
        """Predict future metric values and take proactive action.

        Returns:
            {
                "predictions": {metric: predicted_value},
                "alerts": [{metric, predicted, threshold, action}],
                "health_trend": "improving" | "stable" | "degrading",
                "proactive_actions": [str]
            }
        """
        predictions = {}
        alerts = []
        actions = []

        for metric in self.METRICS:
            history = self._history[metric]
            if len(history) < 10:
                continue

            values = [h["value"] for h in history]
            predicted = self._predict_next(values)
            predictions[metric] = round(predicted, 4)

            # Store for error tracking
            self._predictions[metric] = predicted

            # Check thresholds and generate proactive responses
            alert = self._check_threshold(metric, predicted, values)
            if alert:
                alerts.append(alert)
                actions.extend(alert.get("actions", []))

        # Health trend (high-level prediction)
        health_trend = self._assess_health_trend()

        # Track prediction errors (meta-interoception)
        self._track_prediction_errors()

        result = {
            "predictions": predictions,
            "alerts": alerts,
            "health_trend": health_trend,
            "proactive_actions": actions,
        }

        if alerts:
            logger.warning(
                f"[Interoception] {len(alerts)} proactive alerts: "
                + ", ".join(a["metric"] for a in alerts)
            )

        # Deposit to pheromone field
        try:
            from pheromone_field import get_pheromone_field, PheromoneField
            field = get_pheromone_field()
            field.deposit("interoception", PheromoneField.SIGNAL_HEALTH, {
                "trend": health_trend,
                "n_alerts": len(alerts),
                "predictions": predictions,
            }, half_life=300)  # 5 min
        except Exception:
            pass

        # Persist periodically
        if len(self._history["organism_health"]) % 20 == 0:
            self._save_state()

        return result

    def _predict_next(self, values: List[float]) -> float:
        """Simple linear extrapolation for next value.
        Uses exponentially weighted moving average + trend.
        """
        if len(values) < 3:
            return values[-1] if values else 0.0

        recent = np.array(values[-20:])

        # EWMA
        alpha = 0.3
        ewma = recent[0]
        for v in recent[1:]:
            ewma = alpha * v + (1 - alpha) * ewma

        # Trend (slope of last 10 points)
        if len(recent) >= 5:
            x = np.arange(len(recent[-10:]))
            y = recent[-10:]
            try:
                slope = np.polyfit(x, y, 1)[0]
            except (np.linalg.LinAlgError, ValueError):
                slope = 0.0
        else:
            slope = 0.0

        # Predicted next = EWMA + trend
        return float(ewma + slope)

    def _check_threshold(self, metric: str, predicted: float, history: List[float]) -> Optional[Dict]:
        """Check if predicted value crosses a danger threshold."""
        recent_avg = np.mean(history[-10:]) if len(history) >= 10 else np.mean(history)

        thresholds = {
            "signal_latency_ms": {
                "danger": _p("interoception.latency_danger", 30000),  # 30s
                "action": "Pre-warm caches, reduce RAG depth",
            },
            "win_rate_7d": {
                "danger_below": _p("interoception.winrate_danger", 0.35),
                "action": "Enter defensive mode, reduce sizing",
            },
            "api_error_rate": {
                "danger": _p("interoception.error_rate_danger", 0.3),  # 30% error rate
                "action": "Switch to fallback APIs, increase circuit breaker cooldowns",
            },
            "organism_health": {
                "danger_below": _p("interoception.health_danger", 0.3),
                "action": "Reduce trade frequency, run diagnostic",
            },
            "prediction_error_avg": {
                "danger": _p("interoception.pred_error_danger", 0.5),
                "action": "Models degrading, trigger retraining",
            },
        }

        config = thresholds.get(metric, {})
        if not config:
            return None

        if "danger" in config and predicted > config["danger"]:
            return {
                "metric": metric,
                "predicted": predicted,
                "threshold": config["danger"],
                "direction": "above",
                "severity": min(predicted / config["danger"], 3.0),
                "actions": [config["action"]],
            }
        elif "danger_below" in config and predicted < config["danger_below"]:
            return {
                "metric": metric,
                "predicted": predicted,
                "threshold": config["danger_below"],
                "direction": "below",
                "severity": min(config["danger_below"] / max(predicted, 0.01), 3.0),
                "actions": [config["action"]],
            }

        return None

    def _assess_health_trend(self) -> str:
        """High-level: is the organism getting better or worse?"""
        health_history = [h["value"] for h in self._history["organism_health"]]
        if len(health_history) < 20:
            return "stable"

        recent = np.mean(health_history[-10:])
        older = np.mean(health_history[-20:-10])

        if recent > older * 1.05:
            return "improving"
        elif recent < older * 0.95:
            return "degrading"
        return "stable"

    def _track_prediction_errors(self) -> None:
        """Meta-interoception: how accurate are our predictions?"""
        errors = []
        for metric in self.METRICS:
            if metric not in self._predictions:
                continue
            history = self._history[metric]
            if len(history) < 2:
                continue

            predicted = self._predictions[metric]
            actual = history[-1]["value"]
            if actual != 0:
                error = abs(predicted - actual) / abs(actual)
            else:
                error = abs(predicted - actual)
            errors.append(error)

        if errors:
            avg_error = float(np.mean(errors))
            self.record("prediction_error_avg", avg_error)

    def _save_state(self) -> None:
        try:
            os.makedirs(os.path.dirname(self._state_path), exist_ok=True)
            state = {
                metric: [{"value": h["value"], "time": h["time"]}
                         for h in list(self._history[metric])[-50:]]
                for metric in self.METRICS
            }
            with open(self._state_path, "w") as f:
                json.dump(state, f)
        except Exception as e:
            logger.debug(f"[Interoception] Save failed: {e}")

    def _load_state(self) -> None:
        try:
            if os.path.exists(self._state_path):
                with open(self._state_path) as f:
                    state = json.load(f)
                for metric in self.METRICS:
                    if metric in state:
                        for entry in state[metric]:
                            self._history[metric].append(entry)
                logger.info(f"[Interoception] Loaded state for {len(state)} metrics")
        except Exception as e:
            logger.debug(f"[Interoception] No saved state: {e}")


# Singleton
_interoception: Optional[PredictiveInteroception] = None


def get_interoception() -> PredictiveInteroception:
    global _interoception
    if _interoception is None:
        _interoception = PredictiveInteroception()
    return _interoception
