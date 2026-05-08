"""Phase 30 C.10 — Contradiction matrix + time decay.

For every pair of agents in agent_pool, tracks the rate at which they disagree
on signals. Decays older observations exponentially. A high contradiction rate
signals either:
- Genuine regime conflict (legitimate)
- One agent is broken / drifted (audit)

Used by aggregator to discount over-correlated agents (when they always agree)
and to up-weight robust independent voters.
"""
from __future__ import annotations

import logging
import math
import threading
from collections import defaultdict
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

DECAY_HALF_LIFE_HOURS = 72


class ContradictionMatrix:
    def __init__(self):
        self._counts: Dict[Tuple[str, str], Dict[str, float]] = defaultdict(
            lambda: {"agree": 0.0, "disagree": 0.0, "last_ts": 0.0}
        )
        self._lock = threading.RLock()

    def record(self, agent_a: str, agent_b: str, agree: bool, pair: str = "",
               ts: float = 0.0) -> None:
        import time
        ts = ts or time.time()
        if agent_a > agent_b:
            agent_a, agent_b = agent_b, agent_a
        key = (agent_a, agent_b)
        with self._lock:
            entry = self._counts[key]
            self._decay(entry, ts)
            if agree:
                entry["agree"] += 1.0
            else:
                entry["disagree"] += 1.0
            entry["last_ts"] = ts
        try:
            from db import AI_DB_PATH, get_db_connection

            with get_db_connection(AI_DB_PATH) as conn:
                conn.execute(
                    """INSERT INTO contradiction_events
                       (pair, agent_a, agent_b, disagreement, time_decay_weight)
                       VALUES (?, ?, ?, ?, ?)""",
                    (pair, agent_a, agent_b, 0 if agree else 1, 1.0),
                )
                conn.commit()
        except Exception:
            pass

    def _decay(self, entry: Dict[str, float], now_ts: float) -> None:
        last = entry["last_ts"]
        if last <= 0:
            return
        elapsed_h = (now_ts - last) / 3600.0
        if elapsed_h <= 0:
            return
        factor = math.exp(-math.log(2) * elapsed_h / DECAY_HALF_LIFE_HOURS)
        entry["agree"] *= factor
        entry["disagree"] *= factor

    def disagreement_rate(self, agent_a: str, agent_b: str) -> float:
        if agent_a > agent_b:
            agent_a, agent_b = agent_b, agent_a
        with self._lock:
            entry = self._counts.get((agent_a, agent_b))
            if not entry:
                return 0.0
            total = entry["agree"] + entry["disagree"]
            if total <= 0:
                return 0.0
            return entry["disagree"] / total

    def top_disagreements(self, top_n: int = 10) -> List[Dict]:
        with self._lock:
            items = []
            for (a, b), e in self._counts.items():
                rate = e["disagree"] / (e["agree"] + e["disagree"]) if (e["agree"] + e["disagree"]) else 0.0
                items.append({"agent_a": a, "agent_b": b, "rate": rate,
                              "n_total": e["agree"] + e["disagree"]})
        items.sort(key=lambda x: x["rate"], reverse=True)
        return items[:top_n]


_GLOBAL = ContradictionMatrix()
