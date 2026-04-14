"""
active_learner.py — Phase 26 Sprint 2, Task 9B

Active Learning — The organism actively SEEKS information about its weaknesses.

Instead of only learning from trades that happen to occur, the organism:
  1. Identifies knowledge gaps from SelfModel (9A) competence map
  2. Finds high-uncertainty regions from conformal calibrator
  3. Suggests exploration trades (minimum sizing, maximum information gain)
  4. Information-theoretic exploration/exploitation balance

Integration:
  - Reads from self_model.py (9A) — competence map, weak areas
  - Reads from conformal_calibrator.py — uncertainty per pair/regime
  - Reads from rl_environment.py — available pairs from config
  - Writes suggestions to pheromone_field for strategy consumption
  - Used by AIFreqtradeSizer for trade selection

Reference: Information-theoretic active learning for trading (novel)
"""

import os
import sys
import json
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
logger = logging.getLogger("active_learner")

from ai_config import AI_DB_PATH
from db import get_db_connection, get_connection, init_db

# Exploration parameters
MIN_EXPLORATION_SIZE = 0.25    # Minimum sizing for exploration trades (25% of normal)
MAX_EXPLORATION_TRADES = 3     # Max concurrent exploration trades
UNCERTAINTY_THRESHOLD = 0.7    # High uncertainty = exploration candidate
COMPETENCE_GAP_THRESHOLD = 0.4  # Below this = knowledge gap


class ActiveLearner:
    """Actively seeks information about the organism's weaknesses."""

    def __init__(self, db_path: str = AI_DB_PATH):
        self.db_path = db_path
        self._self_model = None
        self._available_pairs = None
        init_db()

    def _get_self_model(self):
        """Lazy-load self model."""
        if self._self_model is None:
            try:
                from self_model import get_self_model
                self._self_model = get_self_model()
                self._self_model.load_from_db()
            except Exception as e:
                logger.warning(f"[ActiveLearner] SelfModel not available: {e}")
        return self._self_model

    def _get_available_pairs(self) -> List[str]:
        """Get pairs that have recent trade data."""
        if self._available_pairs is not None:
            return self._available_pairs

        conn = get_db_connection(self.db_path)
        try:
            rows = conn.execute("""
                SELECT DISTINCT pair FROM ai_decisions
                WHERE timestamp >= datetime('now', '-30 days')
                ORDER BY pair
            """).fetchall()
            self._available_pairs = [r["pair"] for r in rows]
            return self._available_pairs
        finally:
            conn.close()

    def _get_regimes(self) -> List[str]:
        """Get known regimes."""
        return ["trending_bull", "trending_bear", "ranging",
                "high_volatility", "transitional"]

    # ===================================================================
    # Knowledge Gap Identification
    # ===================================================================

    def identify_knowledge_gaps(self) -> List[Dict]:
        """Identify where the organism has knowledge gaps.

        Returns sorted list of gaps (highest uncertainty first).
        """
        gaps = []
        sm = self._get_self_model()
        pairs = self._get_available_pairs()
        regimes = self._get_regimes()

        for pair in pairs:
            for regime in regimes:
                gap_info = self._assess_gap(pair, regime, sm)
                if gap_info:
                    gaps.append(gap_info)

        # Sort by information value (highest first)
        gaps.sort(key=lambda x: x.get("info_value", 0), reverse=True)

        logger.info(f"[ActiveLearner] Found {len(gaps)} knowledge gaps "
                    f"across {len(pairs)} pairs × {len(regimes)} regimes")

        return gaps

    def _assess_gap(self, pair: str, regime: str,
                    sm=None) -> Optional[Dict]:
        """Assess knowledge gap for a specific pair+regime."""
        gap = {
            "pair": pair,
            "regime": regime,
            "gap_types": [],
            "uncertainty": 0.5,
            "competence": 0.5,
            "info_value": 0.0,
        }

        # 1. Check competence map
        if sm:
            competence = sm.get_competence(pair, regime)
            gap["competence"] = competence

            if competence < COMPETENCE_GAP_THRESHOLD:
                gap["gap_types"].append("low_competence")
                gap["info_value"] += (1.0 - competence) * 0.4

            # Check if no experience at all
            if (pair, regime) not in sm.competence_map:
                gap["gap_types"].append("no_experience")
                gap["info_value"] += 0.5

        # 2. Check uncertainty from recent predictions
        conn = get_db_connection(self.db_path)
        try:
            recent = conn.execute("""
                SELECT COUNT(*) as cnt,
                       AVG(confidence) as avg_conf,
                       COUNT(CASE WHEN outcome_pnl IS NOT NULL THEN 1 END) as completed
                FROM ai_decisions
                WHERE pair = ? AND regime = ?
                  AND timestamp >= datetime('now', '-14 days')
            """, (pair, regime)).fetchone()

            trade_count = recent["cnt"] or 0
            avg_conf = recent["avg_conf"] or 0.5
            completed = recent["completed"] or 0

            # Few trades = high uncertainty
            if trade_count < 5:
                gap["uncertainty"] = 0.8
                gap["gap_types"].append("few_trades")
                gap["info_value"] += 0.3

            # Low average confidence = model is uncertain
            if avg_conf < 0.45:
                gap["uncertainty"] = max(gap["uncertainty"], 1.0 - avg_conf)
                gap["gap_types"].append("low_confidence_area")
                gap["info_value"] += 0.2

        finally:
            conn.close()

        # Only return if there's a meaningful gap
        if gap["gap_types"]:
            return gap

        return None

    # ===================================================================
    # Exploration Suggestions
    # ===================================================================

    def suggest_exploration_trades(self, max_trades: int = MAX_EXPLORATION_TRADES) -> List[Dict]:
        """Suggest trades for information gathering.

        These are MINIMUM-SIZE trades designed to LEARN, not profit.
        """
        gaps = self.identify_knowledge_gaps()

        suggestions = []
        for gap in gaps[:max_trades]:
            suggestion = {
                "pair": gap["pair"],
                "regime": gap["regime"],
                "sizing_multiplier": MIN_EXPLORATION_SIZE,
                "reason": f"active_learning: {', '.join(gap['gap_types'])}",
                "info_value": gap["info_value"],
                "competence": gap["competence"],
                "uncertainty": gap["uncertainty"],
                "is_exploration": True,
            }
            suggestions.append(suggestion)

        if suggestions:
            logger.info(f"[ActiveLearner] Suggesting {len(suggestions)} exploration trades: "
                       f"{[s['pair'] for s in suggestions]}")

        return suggestions

    def get_exploration_bias(self, pair: str, regime: str) -> float:
        """Get exploration bias for a specific pair+regime.

        Returns 0.0-1.0:
          0.0 = pure exploitation (well-known area, maximize profit)
          1.0 = pure exploration (unknown area, minimize size, gather info)
        """
        sm = self._get_self_model()
        if not sm:
            return 0.0  # Default to exploitation

        competence = sm.get_competence(pair, regime)

        # Higher exploration when competence is low
        if competence < 0.3:
            return 0.8  # Strong exploration
        elif competence < 0.5:
            return 0.4  # Moderate exploration
        elif competence > 0.7:
            return 0.05  # Almost pure exploitation
        else:
            return 0.15  # Mild exploration

    # ===================================================================
    # Deposit to Pheromone for Strategy Consumption
    # ===================================================================

    def publish_suggestions(self):
        """Publish exploration suggestions to pheromone field."""
        try:
            from pheromone_field import get_pheromone_field

            suggestions = self.suggest_exploration_trades()
            if not suggestions:
                return

            pfield = get_pheromone_field()
            pfield.deposit("active_learner", "exploration_suggestions", {
                "suggestions": suggestions,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "count": len(suggestions),
            })

            logger.info(f"[ActiveLearner] Published {len(suggestions)} exploration suggestions "
                       f"to pheromone field")

        except Exception as e:
            logger.warning(f"[ActiveLearner] Failed to publish: {e}")

    # ===================================================================
    # Status
    # ===================================================================

    def get_status(self) -> Dict:
        gaps = self.identify_knowledge_gaps()
        suggestions = self.suggest_exploration_trades()
        return {
            "knowledge_gaps": len(gaps),
            "top_gaps": gaps[:5],
            "suggestions": suggestions,
            "available_pairs": len(self._get_available_pairs()),
        }

    def print_status(self):
        status = self.get_status()
        print(f"\n{'='*55}")
        print(f"  ACTIVE LEARNER STATUS")
        print(f"{'='*55}")
        print(f"  Knowledge gaps: {status['knowledge_gaps']}")
        print(f"  Available pairs: {status['available_pairs']}")
        if status["top_gaps"]:
            print(f"  Top gaps:")
            for g in status["top_gaps"]:
                print(f"    {g['pair']} + {g['regime']}: "
                      f"{', '.join(g['gap_types'])} (info_value={g['info_value']:.2f})")
        if status["suggestions"]:
            print(f"  Exploration suggestions:")
            for s in status["suggestions"]:
                print(f"    {s['pair']}: {s['reason']} (size={s['sizing_multiplier']}x)")
        print(f"{'='*55}\n")


# Singleton
_active_learner_instance = None

def get_active_learner() -> ActiveLearner:
    global _active_learner_instance
    if _active_learner_instance is None:
        _active_learner_instance = ActiveLearner()
    return _active_learner_instance
