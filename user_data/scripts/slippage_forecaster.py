"""
slippage_forecaster.py — Phase 26 Sprint 2, Task 11E

Execution Quality Prediction — forecasts expected slippage before trade entry.

Lightweight model (5 features → bps estimate):
  1. Spread (current bid-ask spread)
  2. Depth (orderbook depth at best levels)
  3. Volatility (recent price volatility)
  4. Urgency (how quickly we need to fill)
  5. Order size percentile (relative to avg volume)

If expected_slippage > threshold:
  - Reduce notional size
  - Switch to passive limit orders
  - Or skip the trade entirely

Integration:
  - Reads from: lob_encoder (spread, depth), chart_features (volatility)
  - Consumed by: custom_entry_price(), custom_exit_price() in strategy
  - Writes post-trade actual slippage for model improvement
"""

import os
import sys
import logging
from typing import Dict, Optional
from collections import deque

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
logger = logging.getLogger("slippage_forecaster")

from ai_config import AI_DB_PATH
from db import get_db_connection, execute_with_retry, init_db

# Slippage thresholds (basis points)
SLIPPAGE_ACCEPTABLE_BPS = 5.0    # Up to 5 bps — normal
SLIPPAGE_CAUTION_BPS = 15.0      # 5-15 bps — reduce size
SLIPPAGE_SKIP_BPS = 30.0         # >30 bps — skip trade

# Model weights (simple linear model, updated with SGD)
DEFAULT_WEIGHTS = np.array([2.0, -0.5, 3.0, 1.5, 2.5], dtype=np.float32)
DEFAULT_BIAS = 1.0


class SlippageForecaster:
    """Lightweight slippage prediction model."""

    def __init__(self):
        self._weights = DEFAULT_WEIGHTS.copy()
        self._bias = DEFAULT_BIAS
        self._history: deque = deque(maxlen=500)
        self._lr = 0.01
        init_db()

    def predict(self, spread_bps: float, depth_log: float,
                volatility: float, urgency: float,
                order_size_percentile: float) -> Dict:
        """Predict expected slippage in basis points.

        Args:
            spread_bps: Current bid-ask spread in bps
            depth_log: log(best_bid_size + best_ask_size)
            volatility: Recent ATR / price (fraction)
            urgency: 0.0 (can wait) to 1.0 (must fill immediately)
            order_size_percentile: order_size / avg_volume (0-1+)

        Returns:
            {"expected_slippage_bps": float, "recommendation": str,
             "sizing_adjustment": float}
        """
        features = np.array([
            spread_bps,
            depth_log,
            volatility * 10000,  # Convert to bps
            urgency,
            order_size_percentile,
        ], dtype=np.float32)

        # Linear prediction
        slippage_bps = float(np.dot(self._weights, features) + self._bias)
        slippage_bps = max(slippage_bps, 0.0)  # Slippage can't be negative

        # Recommendation
        if slippage_bps <= SLIPPAGE_ACCEPTABLE_BPS:
            recommendation = "market_order"
            sizing_adj = 1.0
        elif slippage_bps <= SLIPPAGE_CAUTION_BPS:
            recommendation = "limit_order"
            sizing_adj = 0.8
        elif slippage_bps <= SLIPPAGE_SKIP_BPS:
            recommendation = "passive_limit"
            sizing_adj = 0.5
        else:
            recommendation = "skip"
            sizing_adj = 0.0

        return {
            "expected_slippage_bps": round(slippage_bps, 2),
            "recommendation": recommendation,
            "sizing_adjustment": sizing_adj,
            "features": {
                "spread_bps": round(spread_bps, 2),
                "depth_log": round(depth_log, 2),
                "volatility_bps": round(volatility * 10000, 2),
                "urgency": round(urgency, 2),
                "order_size_pct": round(order_size_percentile, 3),
            },
        }

    def predict_from_lob(self, lob_result: Dict, volatility: float = 0.001,
                         order_size: float = 100, avg_volume: float = 10000) -> Dict:
        """Predict slippage from LOB encoder result."""
        spread_bps = lob_result.get("spread_pct", 0.01) * 100  # Convert % to bps
        depth = lob_result.get("depth_ratio_5", 1.0)
        depth_log = float(np.log1p(depth))
        urgency = 0.5  # Default moderate urgency
        size_pct = order_size / max(avg_volume, 1)

        return self.predict(spread_bps, depth_log, volatility, urgency, size_pct)

    def update_model(self, predicted_bps: float, actual_bps: float,
                     features: np.ndarray):
        """Online SGD update with actual slippage feedback.

        Called after trade execution with actual realized slippage.
        """
        error = predicted_bps - actual_bps
        gradient = error * features

        self._weights -= self._lr * gradient
        self._bias -= self._lr * error

        # Clip weights to reasonable range
        self._weights = np.clip(self._weights, -10, 10)
        self._bias = np.clip(self._bias, -5, 10)

        self._history.append({
            "predicted": round(predicted_bps, 2),
            "actual": round(actual_bps, 2),
            "error": round(error, 2),
        })

    def record_actual_slippage(self, pair: str, expected_bps: float,
                               actual_bps: float, order_type: str = "market"):
        """Record actual slippage for model improvement and analytics."""
        try:
            execute_with_retry(
                """INSERT INTO organ_performance_history
                   (organ, regime, metric, value, sample_size)
                   VALUES ('slippage_forecaster', ?, 'slippage_error_bps', ?, 1)""",
                (order_type, actual_bps - expected_bps),
                max_retries=3,
            )
        except Exception:
            pass

    def get_accuracy(self) -> Dict:
        """Get model accuracy from recent predictions."""
        if not self._history:
            return {"n_predictions": 0, "mae_bps": 0, "direction_accuracy": 0}

        errors = [abs(h["error"]) for h in self._history]
        return {
            "n_predictions": len(self._history),
            "mae_bps": round(float(np.mean(errors)), 2),
            "max_error_bps": round(float(max(errors)), 2),
        }


# Singleton
_sf_instance = None

def get_slippage_forecaster() -> SlippageForecaster:
    global _sf_instance
    if _sf_instance is None:
        _sf_instance = SlippageForecaster()
    return _sf_instance
