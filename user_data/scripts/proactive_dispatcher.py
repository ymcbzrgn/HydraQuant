"""
proactive_dispatcher.py — PredictiveInteroception efferent output.

The interoceptive loop produced `proactive_actions: List[str]` but no
module ever read that key. `_check_threshold` fires every 15 minutes
with the same prescription strings ("Enter defensive mode, reduce
sizing") and nothing in the body changes state. This dispatcher maps
each prescription to a concrete effector:

    str payload                                         → effector
    "Enter defensive mode, reduce sizing"               → pheromone defensive_mode + sizing cap
    "Pre-warm caches, reduce RAG depth"                 → pheromone rag_depth ↓
    "Switch to fallback APIs, increase circuit breaker" → llm_router cooldown bump
    "Reduce trade frequency, run diagnostic"            → pheromone trade_frequency ↓
    "Models degrading, trigger retraining"              → retrain_queue flag

The dispatcher is side-effecting but read-only for data it doesn't
own — it only deposits pheromones and writes a small `proactive_queue`
table that consumers (scheduler retrain, HydraSizer defensive cap) read
asynchronously. An action that cannot be dispatched logs and continues.
Cooldown (default 1800s) prevents the same alert firing a fresh
action every tick — the organism speaks loudly but acts deliberately.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from typing import List, Dict, Optional, Any

sys.path.append(os.path.dirname(__file__))

logger = logging.getLogger(__name__)


# ─── Canonical action identifiers ────────────────────────────────────────────
ACTION_DEFENSIVE_MODE = "defensive_mode"
ACTION_CACHE_PREWARM = "cache_prewarm"
ACTION_LLM_FALLBACK = "llm_fallback"
ACTION_REDUCE_FREQUENCY = "reduce_frequency"
ACTION_TRIGGER_RETRAIN = "trigger_retrain"

# Map the free-form strings that `PredictiveInteroception` emits to a
# stable action id. Matching is substring-based (case-insensitive) so
# small wording changes in the source don't silently disable dispatch.
_ACTION_MAP = [
    ("defensive mode", ACTION_DEFENSIVE_MODE),
    ("reduce sizing", ACTION_DEFENSIVE_MODE),
    ("pre-warm cache", ACTION_CACHE_PREWARM),
    ("prewarm cache", ACTION_CACHE_PREWARM),
    ("reduce rag depth", ACTION_CACHE_PREWARM),
    ("fallback api", ACTION_LLM_FALLBACK),
    ("circuit breaker cooldown", ACTION_LLM_FALLBACK),
    ("trade frequency", ACTION_REDUCE_FREQUENCY),
    ("run diagnostic", ACTION_REDUCE_FREQUENCY),
    ("retrain", ACTION_TRIGGER_RETRAIN),
    ("models degrading", ACTION_TRIGGER_RETRAIN),
]


def _classify(action_text: str) -> Optional[str]:
    lo = (action_text or "").lower()
    for needle, action_id in _ACTION_MAP:
        if needle in lo:
            return action_id
    return None


class ProactiveDispatcher:
    """Translates predict_and_act's string prescriptions into concrete
    effector calls and records them in the `proactive_queue` table so
    scheduler/HydraSizer consumers can pick them up asynchronously.
    """

    DEFAULT_COOLDOWN_SEC = 1800.0  # 30 min — matches scheduler tick cadence

    def __init__(self, db_path: Optional[str] = None):
        # Lazy imports guard against module boot ordering at the hot-path.
        try:
            from ai_config import AI_DB_PATH as _DBP
        except Exception:
            _DBP = None
        self.db_path = db_path or _DBP
        self._last_dispatch: Dict[str, float] = {}
        self._ensure_table()

    def _ensure_table(self) -> None:
        try:
            from db import get_db_connection
            with get_db_connection(self.db_path) as conn:
                conn.execute(
                    """CREATE TABLE IF NOT EXISTS proactive_queue (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        action_id TEXT NOT NULL,
                        source_metric TEXT,
                        severity REAL,
                        payload TEXT,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                        handled_at TEXT
                    )"""
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_proactive_pending "
                    "ON proactive_queue(handled_at, created_at)"
                )
                conn.commit()
        except Exception as e:
            logger.debug(f"[ProactiveDispatcher] ensure_table failed: {e}")

    # ─── Dispatch surface ───────────────────────────────────────────────────

    def dispatch(self, result: Dict[str, Any]) -> Dict[str, int]:
        """Consume a `PredictiveInteroception.predict_and_act` result and
        fire the concrete effectors. Returns counters for observability.
        """
        counters = {"actions_seen": 0, "dispatched": 0, "cooldown_skipped": 0,
                    "unclassified": 0, "effector_errors": 0}

        if not isinstance(result, dict):
            return counters

        alerts: List[Dict[str, Any]] = result.get("alerts") or []
        actions: List[str] = list(result.get("proactive_actions") or [])
        if not actions:
            for alert in alerts:
                for raw in (alert.get("actions") or []):
                    actions.append(raw)

        now = time.monotonic()
        for raw in actions:
            counters["actions_seen"] += 1
            action_id = _classify(raw)
            if action_id is None:
                counters["unclassified"] += 1
                logger.debug(f"[ProactiveDispatcher] unclassified action: {raw!r}")
                continue
            last = self._last_dispatch.get(action_id, 0.0)
            if now - last < self.DEFAULT_COOLDOWN_SEC:
                counters["cooldown_skipped"] += 1
                continue

            ok = self._fire(action_id, raw, alerts)
            if ok:
                self._last_dispatch[action_id] = now
                counters["dispatched"] += 1
            else:
                counters["effector_errors"] += 1

        if counters["dispatched"]:
            logger.info(
                f"[ProactiveDispatcher] {counters['dispatched']} action(s) dispatched "
                f"(seen={counters['actions_seen']}, cooldown={counters['cooldown_skipped']}, "
                f"unclassified={counters['unclassified']})"
            )
        return counters

    # ─── Concrete effectors ─────────────────────────────────────────────────

    def _fire(self, action_id: str, raw_text: str,
              alerts: List[Dict[str, Any]]) -> bool:
        severity = 1.0
        source_metric = ""
        for a in alerts:
            if raw_text in (a.get("actions") or []):
                severity = float(a.get("severity", 1.0))
                source_metric = str(a.get("metric", ""))
                break

        try:
            if action_id == ACTION_DEFENSIVE_MODE:
                self._effector_defensive_mode(severity)
            elif action_id == ACTION_CACHE_PREWARM:
                self._effector_cache_prewarm(severity)
            elif action_id == ACTION_LLM_FALLBACK:
                self._effector_llm_fallback(severity)
            elif action_id == ACTION_REDUCE_FREQUENCY:
                self._effector_reduce_frequency(severity)
            elif action_id == ACTION_TRIGGER_RETRAIN:
                self._effector_trigger_retrain(severity)
            else:
                return False
            self._enqueue(action_id, source_metric, severity, raw_text)
            return True
        except Exception as e:
            logger.warning(f"[ProactiveDispatcher] effector {action_id} failed: {e}")
            return False

    def _effector_defensive_mode(self, severity: float) -> None:
        """Deposit a defensive_mode pheromone HydraSizer's sizing pipe can
        read to clamp its stake multiplier. Half-life equals dispatcher
        cooldown so the effect auto-decays if the condition clears."""
        try:
            from pheromone_field import get_pheromone_field
            pfield = get_pheromone_field()
            pfield.deposit(
                "proactive_dispatcher", "defensive_mode",
                {"sizing_cap": max(0.25, 1.0 - 0.25 * min(severity, 2.0)),
                 "severity": severity},
                half_life=self.DEFAULT_COOLDOWN_SEC,
            )
        except Exception as e:
            raise RuntimeError(f"pheromone unavailable: {e}")

    def _effector_cache_prewarm(self, severity: float) -> None:
        try:
            from pheromone_field import get_pheromone_field
            pfield = get_pheromone_field()
            # RAG depth = how many chunks agent_pool retrieves. Lowering it
            # cuts LLM input size → latency → rescues the budget.
            pfield.deposit(
                "proactive_dispatcher", "rag_depth_hint",
                {"depth_cap": max(3, int(8 - 2 * min(severity, 2.0))),
                 "severity": severity},
                half_life=self.DEFAULT_COOLDOWN_SEC,
            )
        except Exception as e:
            raise RuntimeError(f"pheromone unavailable: {e}")

    def _effector_llm_fallback(self, severity: float) -> None:
        """Nudge llm_router's fleet-wide cooldown ceiling upward. Reads
        inside llm_router.LLMSlot.record_failure pick up the deposit via
        pheromone_field.read on the 'llm_router_cooldown_hint' signal."""
        try:
            from pheromone_field import get_pheromone_field
            pfield = get_pheromone_field()
            pfield.deposit(
                "proactive_dispatcher", "llm_router_cooldown_hint",
                {"multiplier": 1.0 + 0.5 * min(severity, 2.0),
                 "severity": severity},
                half_life=self.DEFAULT_COOLDOWN_SEC,
            )
        except Exception as e:
            raise RuntimeError(f"pheromone unavailable: {e}")

    def _effector_reduce_frequency(self, severity: float) -> None:
        """HydraSizer reads 'trade_frequency_hint' to widen the inter-
        signal gap when organism health is degrading. Deposit a hint;
        HydraSizer interprets the specifics."""
        try:
            from pheromone_field import get_pheromone_field
            pfield = get_pheromone_field()
            pfield.deposit(
                "proactive_dispatcher", "trade_frequency_hint",
                {"min_gap_seconds": int(300 + 300 * min(severity, 2.0)),
                 "severity": severity},
                half_life=self.DEFAULT_COOLDOWN_SEC,
            )
        except Exception as e:
            raise RuntimeError(f"pheromone unavailable: {e}")

    def _effector_trigger_retrain(self, severity: float) -> None:
        """Flag the scheduler's retrain queue. CatBoost weekly cron picks
        this up on its next tick and advances the run time."""
        try:
            from pheromone_field import get_pheromone_field
            pfield = get_pheromone_field()
            pfield.deposit(
                "proactive_dispatcher", "retrain_request",
                {"reason": "prediction_error_high", "severity": severity},
                half_life=6 * 3600.0,   # retrain requests persist longer
            )
        except Exception as e:
            raise RuntimeError(f"pheromone unavailable: {e}")

    # ─── Persistence ────────────────────────────────────────────────────────

    def _enqueue(self, action_id: str, source_metric: str,
                 severity: float, raw_text: str) -> None:
        try:
            import json
            from db import get_db_connection
            with get_db_connection(self.db_path) as conn:
                conn.execute(
                    "INSERT INTO proactive_queue (action_id, source_metric, severity, payload) "
                    "VALUES (?, ?, ?, ?)",
                    (action_id, source_metric, float(severity),
                     json.dumps({"raw": raw_text})),
                )
                conn.commit()
        except Exception as e:
            logger.debug(f"[ProactiveDispatcher] enqueue failed: {e}")


# ─── Singleton ──────────────────────────────────────────────────────────────

_dispatcher: Optional[ProactiveDispatcher] = None


def get_proactive_dispatcher() -> ProactiveDispatcher:
    global _dispatcher
    if _dispatcher is None:
        _dispatcher = ProactiveDispatcher()
    return _dispatcher
