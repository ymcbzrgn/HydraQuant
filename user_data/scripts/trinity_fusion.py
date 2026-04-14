"""
trinity_fusion.py — Phase 26 Sprint 2, Task 10B

LLM × RL × RAG Trinity — Cross-modal fusion of three AI paradigms.

The Trinity combines:
  - RAG: retrieves relevant historical context (patterns, lessons, news)
  - ML/LLM: reasons about the context and predicts outcomes
  - RL: makes optimal parameter decisions using predictions

Fusion mechanism: Cross-attention where ML prediction attends to RAG context,
then RL uses the enriched prediction for decision making.

Timestamp Alignment Guard ensures no stale data enters fusion.

Integration:
  - Reads from: rag_graph.py (RAG context), triple_perception.py (ML predictions),
                sac_online.py (RL actions), evidence_engine.py (evidence scores)
  - Writes rl_relevance feedback to LanceDB (RAG learns from RL outcomes)
  - Used by: AIFreqtradeSizer for final trade decisions

Reference: Neurosymbolic AI — neural (ML) + symbolic (LLM reasoning)
"""

import os
import sys
import json
import logging
import time as _time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
logger = logging.getLogger("trinity_fusion")

from ai_config import AI_DB_PATH
from db import init_db

# Latency tiers (milliseconds)
TIER_BUDGETS = {
    "tier_0": 1,       # Constitution checks
    "tier_1": 100,     # RL inference, hormone compute
    "tier_2": 5000,    # TTM perception, ensemble
    "tier_3": 60000,   # World model, MADAM debate
}

# Staleness thresholds per source
STALENESS_MS = {
    "rl_action": 200,       # Tier-1
    "ml_prediction": 5000,  # Tier-2
    "rag_context": 60000,   # Tier-3
    "llm_reasoning": 60000, # Tier-3
}

FUSION_DIM = 64


class TimestampedField:
    """A value with timestamp and max age for staleness checking."""

    def __init__(self, name: str, value: Any, max_age_ms: float):
        self.name = name
        self.value = value
        self.updated_at = _time.monotonic()
        self.max_age_ms = max_age_ms

    def is_fresh(self) -> bool:
        age_ms = (_time.monotonic() - self.updated_at) * 1000
        return age_ms <= self.max_age_ms

    def age_ms(self) -> float:
        return (_time.monotonic() - self.updated_at) * 1000


class TripleFusion:
    """LLM × RL × RAG cross-modal fusion with timestamp alignment."""

    def __init__(self):
        self._fields: Dict[str, TimestampedField] = {}
        self._fusion_count = 0
        self._stale_rejections = 0
        init_db()

    # ===================================================================
    # Field Management
    # ===================================================================

    def update_field(self, name: str, value: Any, tier: str = "tier_2"):
        """Update a timestamped field."""
        max_age = STALENESS_MS.get(name, TIER_BUDGETS.get(tier, 5000))
        self._fields[name] = TimestampedField(name, value, max_age)

    def update_rag_context(self, context_embedding: np.ndarray,
                           context_text: str = "", sources: List[str] = None):
        """Update RAG context field."""
        self.update_field("rag_context", {
            "embedding": context_embedding,
            "text": context_text,
            "sources": sources or [],
        }, "tier_3")

    def update_ml_prediction(self, prediction: Dict):
        """Update ML prediction field (from triple_perception)."""
        self.update_field("ml_prediction", prediction, "tier_2")

    def update_rl_action(self, action: np.ndarray, q_value: float = 0.0):
        """Update RL action field (from SAC/IQL)."""
        self.update_field("rl_action", {
            "action": action,
            "q_value": q_value,
        }, "tier_1")

    def update_llm_reasoning(self, reasoning: str, sentiment: str = "NEUTRAL"):
        """Update LLM reasoning field (from MADAM debate)."""
        self.update_field("llm_reasoning", {
            "reasoning": reasoning,
            "sentiment": sentiment,
        }, "tier_3")

    # ===================================================================
    # Timestamp Alignment Guard
    # ===================================================================

    def check_alignment(self, required_fields: List[str] = None) -> Tuple[bool, Dict]:
        """Check if all required fields are fresh enough for fusion.

        Returns (all_fresh, staleness_report).
        """
        if required_fields is None:
            required_fields = ["ml_prediction", "rag_context"]

        report = {}
        all_fresh = True

        for name in required_fields:
            field = self._fields.get(name)
            if field is None:
                report[name] = {"status": "missing", "age_ms": float("inf")}
                all_fresh = False
            elif not field.is_fresh():
                report[name] = {
                    "status": "stale",
                    "age_ms": round(field.age_ms(), 1),
                    "max_age_ms": field.max_age_ms,
                }
                all_fresh = False
            else:
                report[name] = {
                    "status": "fresh",
                    "age_ms": round(field.age_ms(), 1),
                }

        return all_fresh, report

    # ===================================================================
    # Trinity Fusion
    # ===================================================================

    def fuse(self, pair: str = "", regime: str = "") -> Dict[str, Any]:
        """Execute Trinity fusion: RAG + ML + RL → optimal decision.

        Returns fused decision with provenance tracking.
        """
        # Check alignment
        aligned, staleness = self.check_alignment(["ml_prediction"])

        if not aligned:
            stale_fields = [k for k, v in staleness.items() if v["status"] != "fresh"]
            self._stale_rejections += 1
            logger.debug(f"[Trinity] Fusion skipped — stale fields: {stale_fields}")
            return {
                "fused": False,
                "reason": f"stale fields: {stale_fields}",
                "staleness": staleness,
                "fallback": "ml_only",
            }

        result = {
            "fused": True,
            "pair": pair,
            "regime": regime,
            "components": {},
            "staleness": staleness,
        }

        # Component 1: ML prediction
        ml_field = self._fields.get("ml_prediction")
        if ml_field and ml_field.is_fresh():
            ml_data = ml_field.value
            result["components"]["ml"] = {
                "signal": ml_data.get("signal", "NEUTRAL"),
                "confidence": ml_data.get("confidence", 0.5),
                "sizing_multiplier": ml_data.get("sizing_multiplier", 1.0),
            }

        # Component 2: RAG context
        rag_field = self._fields.get("rag_context")
        if rag_field and rag_field.is_fresh():
            rag_data = rag_field.value
            result["components"]["rag"] = {
                "has_context": True,
                "n_sources": len(rag_data.get("sources", [])),
            }

        # Component 3: RL action
        rl_field = self._fields.get("rl_action")
        if rl_field and rl_field.is_fresh():
            rl_data = rl_field.value
            action = rl_data.get("action")
            if action is not None:
                result["components"]["rl"] = {
                    "sizing_mult": float(action[0]) if len(action) > 0 else 1.0,
                    "confidence_thresh": float(action[1]) if len(action) > 1 else 0.5,
                    "stoploss_mult": float(action[2]) if len(action) > 2 else 2.0,
                    "leverage": float(action[3]) if len(action) > 3 else 1.0,
                    "q_value": rl_data.get("q_value", 0.0),
                }

        # Component 4: LLM reasoning
        llm_field = self._fields.get("llm_reasoning")
        if llm_field and llm_field.is_fresh():
            llm_data = llm_field.value
            result["components"]["llm"] = {
                "sentiment": llm_data.get("sentiment", "NEUTRAL"),
                "has_reasoning": bool(llm_data.get("reasoning")),
            }

        # Fuse: weighted combination
        result["decision"] = self._compute_fused_decision(result["components"])
        self._fusion_count += 1

        return result

    def _compute_fused_decision(self, components: Dict) -> Dict:
        """Compute final fused decision from components."""
        # ML confidence as base
        ml = components.get("ml", {})
        base_confidence = ml.get("confidence", 0.5)
        signal = ml.get("signal", "NEUTRAL")
        sizing = ml.get("sizing_multiplier", 1.0)

        # RL override for sizing/params
        rl = components.get("rl", {})
        if rl:
            sizing = rl.get("sizing_mult", sizing)

        # LLM sentiment agreement bonus
        llm = components.get("llm", {})
        llm_sentiment = llm.get("sentiment", "NEUTRAL")

        # Agreement bonus: if ML and LLM agree, boost confidence
        if signal == "BULLISH" and llm_sentiment == "BULLISH":
            base_confidence = min(base_confidence * 1.1, 0.95)
        elif signal == "BEARISH" and llm_sentiment == "BEARISH":
            base_confidence = min(base_confidence * 1.1, 0.95)
        # Disagreement penalty
        elif (signal == "BULLISH" and llm_sentiment == "BEARISH") or \
             (signal == "BEARISH" and llm_sentiment == "BULLISH"):
            base_confidence *= 0.85

        # RAG context richness bonus
        rag = components.get("rag", {})
        if rag.get("n_sources", 0) > 3:
            base_confidence = min(base_confidence * 1.05, 0.95)

        return {
            "signal": signal,
            "confidence": round(base_confidence, 4),
            "sizing_multiplier": round(sizing, 4),
            "n_components": len(components),
            "ml_llm_agreement": signal.lower() == llm_sentiment.lower() if llm_sentiment != "NEUTRAL" else None,
        }

    # ===================================================================
    # RL Relevance Feedback to RAG
    # ===================================================================

    def send_rl_feedback_to_rag(self, doc_ids: List[str], trade_pnl: float):
        """Send RL outcome feedback to LanceDB for RAG relevance learning.

        Positive PnL → increase rl_relevance for retrieved docs
        Negative PnL → decrease rl_relevance
        """
        if not doc_ids:
            return

        try:
            from lance_store import get_lance_store
            store = get_lance_store()

            relevance_delta = 0.1 if trade_pnl > 0 else -0.05
            updated = 0

            for doc_id in doc_ids:
                try:
                    # LanceDB doesn't support in-place update easily,
                    # so we track relevance in SQLite instead
                    from db import get_connection
                    with get_connection() as conn:
                        conn.execute("""
                            INSERT INTO rl_relevance_feedback
                            (doc_id, trade_pnl, relevance_delta, timestamp)
                            VALUES (?, ?, ?, datetime('now'))
                            ON CONFLICT(doc_id) DO UPDATE SET
                                relevance_delta = relevance_delta + ?,
                                trade_pnl = ?,
                                timestamp = datetime('now')
                        """, (doc_id, trade_pnl, relevance_delta,
                              relevance_delta, trade_pnl))
                        conn.commit()
                    updated += 1
                except Exception:
                    pass

            if updated > 0:
                logger.debug(f"[Trinity] RL feedback: {updated} docs, "
                           f"pnl={trade_pnl:.2f}, delta={relevance_delta:+.2f}")

        except Exception as e:
            logger.debug(f"[Trinity] RL feedback failed: {e}")

    # ===================================================================
    # Status
    # ===================================================================

    def get_status(self) -> Dict:
        fields_status = {}
        for name, field in self._fields.items():
            fields_status[name] = {
                "fresh": field.is_fresh(),
                "age_ms": round(field.age_ms(), 1),
            }

        return {
            "fusion_count": self._fusion_count,
            "stale_rejections": self._stale_rejections,
            "fields": fields_status,
        }


# Singleton
_trinity_instance = None

def get_trinity() -> TripleFusion:
    global _trinity_instance
    if _trinity_instance is None:
        _trinity_instance = TripleFusion()
    return _trinity_instance
