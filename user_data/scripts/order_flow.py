"""
order_flow.py — Phase 26 Sprint 2, Task 11C

Order Flow Intelligence — CVD + Liquidation Radar.

CVD (Cumulative Volume Delta): tracks aggressive buyer vs seller accumulation.
  CVD rising + price flat = hidden accumulation (bullish divergence)
  CVD falling + price flat = hidden distribution (bearish divergence)

VPIN-lite: Volume-synchronized informed flow estimate.
  High VPIN = toxic flow (informed traders), widen spread or reduce size.

Liquidation Radar: estimates squeeze probability from crowding metrics.
  squeeze_prob > 0.6 → clamp sizing
  squeeze_prob > 0.8 + same direction signal → VETO trade

Integration:
  - Reads from: derivatives_data (funding, OI, L/S ratio)
  - Writes to: pheromone_field ("order_flow_state")
  - Consumed by: market_maker_mode (11D), constitution checks, evidence_engine
"""

import os
import sys
import logging
from collections import deque
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
logger = logging.getLogger("order_flow")

from ai_config import AI_DB_PATH
from db import get_db_connection, init_db

# VPIN thresholds
VPIN_SAFE = 0.3
VPIN_CAUTION = 0.6
VPIN_DANGER = 0.7
VPIN_TOXIC = 0.8

# Squeeze thresholds
SQUEEZE_CLAMP_THRESHOLD = 0.6
SQUEEZE_VETO_THRESHOLD = 0.8


class OrderFlowAnalyzer:
    """CVD + VPIN + Liquidation radar."""

    def __init__(self):
        self._cvd_history: Dict[str, deque] = {}
        self._volume_buckets: Dict[str, deque] = {}
        init_db()

    def analyze(self, pair: str, trades: List[Dict] = None) -> Dict:
        """Analyze order flow for a pair.

        Args:
            trades: Recent trades [{"price": float, "amount": float, "side": "buy"|"sell"}, ...]
                    If None, reads from derivatives_data.

        Returns: {"cvd_slope": float, "flow_toxicity": float, "squeeze_prob_long": float, ...}
        """
        result = {
            "cvd": 0.0,
            "cvd_slope": 0.0,
            "flow_toxicity": 0.0,
            "vpin": 0.0,
            "aggression_state": "neutral",
            "squeeze_probability_long": 0.0,
            "squeeze_probability_short": 0.0,
            "liq_cluster_distance": 1.0,
            "large_lot_detected": False,
        }

        # CVD from trades
        if trades:
            result.update(self._compute_cvd(pair, trades))
            result.update(self._compute_vpin(pair, trades))

        # Liquidation radar from derivatives data
        liq_data = self._compute_liquidation_radar(pair)
        result.update(liq_data)

        # Aggression state
        if result["cvd_slope"] > 0.5:
            result["aggression_state"] = "aggressive_buying"
        elif result["cvd_slope"] < -0.5:
            result["aggression_state"] = "aggressive_selling"
        elif abs(result["cvd_slope"]) < 0.1:
            result["aggression_state"] = "balanced"

        return result

    def _compute_cvd(self, pair: str, trades: List[Dict]) -> Dict:
        """Compute Cumulative Volume Delta."""
        if pair not in self._cvd_history:
            self._cvd_history[pair] = deque(maxlen=500)

        buy_vol = sum(t["amount"] for t in trades if t.get("side") == "buy")
        sell_vol = sum(t["amount"] for t in trades if t.get("side") == "sell")
        delta = buy_vol - sell_vol

        self._cvd_history[pair].append(delta)
        cvd_values = list(self._cvd_history[pair])

        # CVD cumulative
        cvd = sum(cvd_values)

        # CVD slope (trend)
        if len(cvd_values) >= 10:
            recent = np.array(cvd_values[-10:])
            x = np.arange(len(recent))
            if recent.std() > 0:
                slope = float(np.polyfit(x, recent, 1)[0])
            else:
                slope = 0.0
        else:
            slope = 0.0

        # Large lot detection
        total_vol = buy_vol + sell_vol
        large_lot = False
        if trades:
            max_trade = max(t["amount"] for t in trades)
            avg_trade = total_vol / len(trades) if trades else 1
            large_lot = max_trade > avg_trade * 5

        return {
            "cvd": round(float(cvd), 4),
            "cvd_slope": round(float(slope), 4),
            "large_lot_detected": large_lot,
        }

    def _compute_vpin(self, pair: str, trades: List[Dict]) -> Dict:
        """Compute VPIN-lite (Volume-synchronized Probability of Informed trading)."""
        if not trades:
            return {"vpin": 0.0, "flow_toxicity": 0.0}

        buy_vol = sum(t["amount"] for t in trades if t.get("side") == "buy")
        sell_vol = sum(t["amount"] for t in trades if t.get("side") == "sell")
        total_vol = buy_vol + sell_vol

        if total_vol == 0:
            return {"vpin": 0.0, "flow_toxicity": 0.0}

        # VPIN = |buy_vol - sell_vol| / total_vol
        vpin = abs(buy_vol - sell_vol) / total_vol

        # Flow toxicity mapping
        if vpin > VPIN_TOXIC:
            toxicity = 1.0
        elif vpin > VPIN_DANGER:
            toxicity = 0.8
        elif vpin > VPIN_CAUTION:
            toxicity = 0.5
        elif vpin > VPIN_SAFE:
            toxicity = 0.3
        else:
            toxicity = 0.1

        return {
            "vpin": round(float(vpin), 4),
            "flow_toxicity": round(float(toxicity), 2),
        }

    def _compute_liquidation_radar(self, pair: str) -> Dict:
        """Estimate squeeze probability from derivatives data."""
        conn = get_db_connection(AI_DB_PATH)
        try:
            row = conn.execute("""
                SELECT funding_rate, long_short_ratio, open_interest_usd
                FROM derivatives_data
                WHERE pair = ?
                ORDER BY timestamp DESC LIMIT 1
            """, (pair,)).fetchone()

            if not row:
                return {
                    "squeeze_probability_long": 0.0,
                    "squeeze_probability_short": 0.0,
                    "liq_cluster_distance": 1.0,
                }

            funding = row["funding_rate"] or 0
            ls_ratio = row["long_short_ratio"] or 1.0
            oi = row["open_interest_usd"] or 0

            # Squeeze probability estimation
            # High funding + crowded longs → long squeeze risk
            squeeze_long = 0.0
            if funding > 0.01 and ls_ratio > 1.5:
                squeeze_long = min(0.3 + (funding - 0.01) * 50 + (ls_ratio - 1.5) * 0.2, 1.0)

            # Negative funding + crowded shorts → short squeeze risk
            squeeze_short = 0.0
            if funding < -0.01 and ls_ratio < 0.7:
                squeeze_short = min(0.3 + abs(funding + 0.01) * 50 + (0.7 - ls_ratio) * 0.3, 1.0)

            # Liquidation cluster distance (proxy: inverse of OI concentration)
            liq_distance = 1.0 / (1.0 + abs(funding) * 1000)

            return {
                "squeeze_probability_long": round(float(squeeze_long), 3),
                "squeeze_probability_short": round(float(squeeze_short), 3),
                "liq_cluster_distance": round(float(liq_distance), 3),
            }

        finally:
            conn.close()

    def should_veto_trade(self, pair: str, signal: str) -> tuple:
        """Check if order flow conditions warrant a trade veto.

        Returns (should_veto: bool, reason: str)
        """
        result = self.analyze(pair)

        # Squeeze veto: high squeeze prob + signal in same direction as crowded side
        if signal == "BEARISH" and result["squeeze_probability_short"] > SQUEEZE_VETO_THRESHOLD:
            return True, f"short squeeze risk {result['squeeze_probability_short']:.0%}"
        if signal == "BULLISH" and result["squeeze_probability_long"] > SQUEEZE_VETO_THRESHOLD:
            return True, f"long squeeze risk {result['squeeze_probability_long']:.0%}"

        # Toxic flow veto
        if result["flow_toxicity"] > 0.9:
            return True, f"toxic flow (VPIN={result['vpin']:.2f})"

        return False, ""

    def get_sizing_adjustment(self, pair: str) -> float:
        """Get sizing multiplier based on order flow.

        Returns 0.3-1.0 multiplier.
        """
        result = self.analyze(pair)

        mult = 1.0
        if result["flow_toxicity"] > 0.75:
            mult *= 0.6
        if result["squeeze_probability_long"] > SQUEEZE_CLAMP_THRESHOLD:
            mult *= 0.7
        if result["squeeze_probability_short"] > SQUEEZE_CLAMP_THRESHOLD:
            mult *= 0.7

        return max(mult, 0.3)

    def publish_to_pheromone(self, result: Dict, pair: str):
        """Publish order flow state to pheromone field."""
        try:
            from pheromone_field import get_pheromone_field
            pfield = get_pheromone_field()
            pfield.deposit("order_flow", "order_flow_state", {
                "pair": pair,
                "cvd_slope": result["cvd_slope"],
                "flow_toxicity": result["flow_toxicity"],
                "squeeze_long": result["squeeze_probability_long"],
                "squeeze_short": result["squeeze_probability_short"],
                "aggression": result["aggression_state"],
            })
        except Exception:
            pass


# Singleton
_of_instance = None

def get_order_flow() -> OrderFlowAnalyzer:
    global _of_instance
    if _of_instance is None:
        _of_instance = OrderFlowAnalyzer()
    return _of_instance
