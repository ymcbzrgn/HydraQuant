"""
decision_contract.py — Phase 26 Sprint 2, Task 12D

Decision Contract — JSON provenance audit trail for every trade decision.

Every trade decision gets a "contract" documenting:
  - WHO decided (which modules were active)
  - WHAT they decided (signal, sizing, params)
  - WHY they decided (evidence, reasoning, confidence breakdown)
  - WHEN (timestamps for each component)
  - Constitutional compliance check

Extends organism_audit with rich provenance data.
Enables post-hoc analysis: "What went wrong in trade #42?"

Integration:
  - Called by: HydraSizer at confirm_trade_entry
  - Reads from: all active modules (pheromone field, trinity, lifecycle)
  - Writes to: organism_audit (event_type='decision_contract')
"""

import os
import sys
import json
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
logger = logging.getLogger("decision_contract")

from ai_config import AI_DB_PATH
from db import execute_with_retry, init_db


class DecisionContract:
    """Creates and manages decision provenance contracts."""

    def __init__(self):
        init_db()

    def create_contract(self,
                        pair: str,
                        signal: str,
                        confidence: float,
                        sizing_multiplier: float,
                        regime: str,
                        modules_active: List[str],
                        evidence: Dict = None,
                        perception: Dict = None,
                        rl_action: Dict = None,
                        lifecycle_state: Dict = None,
                        constitution_check: Dict = None,
                        trinity_result: Dict = None) -> Dict:
        """Create a decision contract for a trade entry.

        Returns the contract dict and persists it to organism_audit.
        """
        contract = {
            "version": "2.0",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "pair": pair,
            "signal": signal,
            "confidence": round(confidence, 4),
            "sizing_multiplier": round(sizing_multiplier, 4),
            "regime": regime,

            # WHO decided
            "modules_active": modules_active,
            "n_modules": len(modules_active),

            # WHAT was decided
            "decision": {
                "signal": signal,
                "confidence": round(confidence, 4),
                "sizing": round(sizing_multiplier, 4),
                "rl_action": rl_action,
            },

            # WHY — evidence breakdown
            "evidence": {
                "sub_scores": evidence.get("sub_scores", {}) if evidence else {},
                "contradictions": evidence.get("contradictions", 0) if evidence else 0,
                "n_sources": evidence.get("n_sources", 0) if evidence else 0,
            },

            # Perception state
            "perception": {
                "catboost_prob": perception.get("catboost_probability", 0) if perception else 0,
                "ttm_direction": perception.get("ttm_direction", 0) if perception else 0,
                "chronos_p50": perception.get("chronos_p50", 0) if perception else 0,
                "ood_active": perception.get("ood_active", False) if perception else False,
                "disagreement": perception.get("disagreement", 0) if perception else 0,
            },

            # Lifecycle state
            "lifecycle": {
                "danger_level": lifecycle_state.get("danger_level", 0) if lifecycle_state else 0,
                "danger_response": lifecycle_state.get("danger_response", "NORMAL") if lifecycle_state else "NORMAL",
                "circadian_mode": lifecycle_state.get("circadian_mode", "normal") if lifecycle_state else "normal",
                "hormesis": lifecycle_state.get("hormesis", "comfortable") if lifecycle_state else "comfortable",
            },

            # Constitutional compliance
            "constitution": {
                "allowed": constitution_check.get("allowed", True) if constitution_check else True,
                "violations": constitution_check.get("violations", []) if constitution_check else [],
                "clamped": constitution_check.get("clamped_params", {}) if constitution_check else {},
            },

            # Trinity fusion
            "trinity": {
                "fused": trinity_result.get("fused", False) if trinity_result else False,
                "ml_llm_agreement": trinity_result.get("decision", {}).get("ml_llm_agreement") if trinity_result else None,
            },
        }

        # Persist
        self._persist_contract(contract)

        return contract

    def _persist_contract(self, contract: Dict):
        """Write contract to organism_audit."""
        try:
            execute_with_retry(
                """INSERT INTO organism_audit
                   (timestamp, event_type, details_json)
                   VALUES (?, 'decision_contract', ?)""",
                (contract["created_at"], json.dumps(contract, default=str)),
                max_retries=3,
            )
        except Exception as e:
            logger.debug(f"[Contract] Persist failed: {e}")

    def get_contract(self, trade_timestamp: str) -> Optional[Dict]:
        """Retrieve a decision contract by trade timestamp."""
        from db import get_db_connection
        conn = get_db_connection(AI_DB_PATH)
        try:
            row = conn.execute("""
                SELECT details_json FROM organism_audit
                WHERE event_type = 'decision_contract'
                AND ABS(JULIANDAY(timestamp) - JULIANDAY(?)) < 0.001
                ORDER BY timestamp DESC LIMIT 1
            """, (trade_timestamp,)).fetchone()

            if row and row["details_json"]:
                return json.loads(row["details_json"])
            return None
        finally:
            conn.close()

    def get_recent_contracts(self, n: int = 10) -> List[Dict]:
        """Get N most recent decision contracts."""
        from db import get_db_connection
        conn = get_db_connection(AI_DB_PATH)
        try:
            rows = conn.execute("""
                SELECT details_json FROM organism_audit
                WHERE event_type = 'decision_contract'
                ORDER BY timestamp DESC LIMIT ?
            """, (n,)).fetchall()

            contracts = []
            for row in rows:
                try:
                    contracts.append(json.loads(row["details_json"]))
                except Exception:
                    pass
            return contracts
        finally:
            conn.close()


# Singleton
_contract_instance = None

def get_decision_contract() -> DecisionContract:
    global _contract_instance
    if _contract_instance is None:
        _contract_instance = DecisionContract()
    return _contract_instance
