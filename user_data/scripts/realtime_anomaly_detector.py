"""Real-time price anomaly detector — Phase 30 A.27.

Defends against testnet SHORT bias and real flash-crashes:
- 1m bar |dp/p| >= threshold -> halt new entries for cooldown_seconds.
- Halt state per-pair, TTL-based (no DB read on hot path after first record).

LINK trade #2187 (9.658 -> 19.315 +%100, liquidated -1016 USDT) would have been
blocked: detector would have flagged the +%100 bar and halted entries 300s.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

DEFAULT_THRESHOLD_PCT = 0.05
DEFAULT_COOLDOWN_SECONDS = 300


class RealtimeAnomalyDetector:
    """Tracks per-pair anomaly state with TTL-based halt."""

    def __init__(
        self,
        threshold_pct: float = DEFAULT_THRESHOLD_PCT,
        cooldown_seconds: int = DEFAULT_COOLDOWN_SECONDS,
        record_to_db: bool = True,
    ):
        self.threshold_pct = float(threshold_pct)
        self.cooldown_seconds = int(cooldown_seconds)
        self.record_to_db = bool(record_to_db)
        self._halt_until: Dict[str, float] = {}
        self._last_close: Dict[str, float] = {}
        self._lock = threading.RLock()

    def check_bar(
        self,
        pair: str,
        close: float,
        volume: float = 0.0,
    ) -> Optional[str]:
        """Return reason string if anomaly, else None.

        Updates internal last_close + halt state.
        """
        if close is None or close <= 0:
            return None
        now = time.time()
        with self._lock:
            prev = self._last_close.get(pair)
            self._last_close[pair] = float(close)
            if prev is None or prev <= 0:
                return None
            delta_pct = abs(close - prev) / prev
            if delta_pct >= self.threshold_pct:
                self._halt_until[pair] = now + self.cooldown_seconds
                self._record(pair, "single_bar_jump", delta_pct, close, prev)
                return f"single_bar_jump_{delta_pct:.2%}"
        return None

    def is_halted(self, pair: str) -> Tuple[bool, str]:
        with self._lock:
            until = self._halt_until.get(pair, 0.0)
            now = time.time()
            if now < until:
                return True, f"halted_until_{until:.0f}_remaining_{int(until - now)}s"
        return False, ""

    def force_halt(self, pair: str, seconds: Optional[int] = None) -> None:
        seconds = self.cooldown_seconds if seconds is None else int(seconds)
        with self._lock:
            self._halt_until[pair] = time.time() + seconds
        logger.warning(f"[RealtimeAnomaly] FORCE halt {pair} {seconds}s")

    def clear_halt(self, pair: str) -> None:
        with self._lock:
            self._halt_until.pop(pair, None)

    def _record(self, pair: str, kind: str, magnitude: float, close: float, prev: float) -> None:
        logger.warning(
            f"[RealtimeAnomaly] {pair} {kind} mag={magnitude:.2%} close={close} prev={prev}"
        )
        if not self.record_to_db:
            return
        try:
            from db import get_db_connection, AI_DB_PATH

            with get_db_connection(AI_DB_PATH) as conn:
                conn.execute(
                    """INSERT INTO price_anomaly_events
                       (pair, kind, magnitude, close, prev_close)
                       VALUES (?, ?, ?, ?, ?)""",
                    (pair, kind, float(magnitude), float(close), float(prev)),
                )
                conn.commit()
        except Exception as e:
            logger.error(f"[RealtimeAnomaly] DB write failed: {e}")


_GLOBAL_DETECTOR: Optional[RealtimeAnomalyDetector] = None
_GLOBAL_LOCK = threading.Lock()


def get_detector(
    threshold_pct: Optional[float] = None,
    cooldown_seconds: Optional[int] = None,
) -> RealtimeAnomalyDetector:
    """Singleton accessor; PARAM-driven thresholds via neural_organism.PARAM_REGISTRY."""
    global _GLOBAL_DETECTOR
    with _GLOBAL_LOCK:
        if _GLOBAL_DETECTOR is None:
            tp = DEFAULT_THRESHOLD_PCT if threshold_pct is None else threshold_pct
            cs = DEFAULT_COOLDOWN_SECONDS if cooldown_seconds is None else cooldown_seconds
            try:
                from neural_organism import PARAM_REGISTRY  # type: ignore

                tp = float(PARAM_REGISTRY.get("anomaly.threshold_pct", {}).get("default", tp))
                cs = int(PARAM_REGISTRY.get("anomaly.cooldown_seconds", {}).get("default", cs))
            except Exception:
                pass
            _GLOBAL_DETECTOR = RealtimeAnomalyDetector(
                threshold_pct=tp, cooldown_seconds=cs
            )
    return _GLOBAL_DETECTOR


def reset_detector_for_tests() -> None:
    global _GLOBAL_DETECTOR
    with _GLOBAL_LOCK:
        _GLOBAL_DETECTOR = None
