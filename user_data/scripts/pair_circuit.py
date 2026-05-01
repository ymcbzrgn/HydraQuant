"""
pair_circuit.py — Per-pair exchange-call circuit breaker.

LLMSlot (llm_router.py:227-370) protected the AI pipeline from a dead
provider by counting consecutive failures, opening the circuit at 5, and
backing off 15s → 30s → 60s → 120s → 300s (capped). That zarif pattern
never made it to the exchange side, so ICP's 18,768 empty orderbooks
turned into 4,693 `Unable to exit trade` warnings — same call, same
failure, no memory. This class ports the LLMSlot behaviour to pair-
level exchange interactions.

Behaviour:
  * `record_success(pair)` zeros the failure counter and closes circuits.
  * `record_failure(pair, reason)` increments the counter; at ≥5 it sets
    `blacklisted_until = now + min(300, 15 * 2 ** (n_fail - 5))`.
  * `is_dormant(pair)` returns True while `blacklisted_until` is in the
    future — callers (custom_exit, populate_entry_trend, custom_entry_price)
    short-circuit their exchange work when this fires.
  * `revive_probe(pair)` runs a lightweight orderbook fetch; dolu book
    resets the circuit, empty one extends the blacklist without spamming
    retries.

State lives in-process (per strategy PID). For multi-process alignment
the pheromone field carries a `pair_dormant::<pair>` deposit so other
listeners (agent_pool, sizing hooks in different services) can see the
state without a DB round-trip.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

sys.path.append(os.path.dirname(__file__))

logger = logging.getLogger(__name__)


@dataclass
class _PairSlot:
    """Mirrors llm_router.LLMSlot's circuit-breaker state — slimmed down
    to what pair-level calls need (no latency histogram, no bandit)."""
    consecutive_failures: int = 0
    blacklisted_until: float = 0.0
    backoff_level: int = 0
    last_event_at: float = 0.0
    total_failures: int = 0
    total_successes: int = 0
    last_failure_reason: str = ""
    # D1 (2026-04-25): rolling fill-rate window. Each entry is
    # (ts, filled_bool, age_seconds). Trimmed to the last hour on every
    # update so the rate naturally tracks the present.
    order_attempts: list = field(default_factory=list)


# Thresholds mirror LLMSlot's values so the two subsystems stay easy to
# reason about together. 5 consecutive failures opens the circuit; each
# additional failure doubles the cooldown up to the 300 second cap.
_OPEN_THRESHOLD = 5
_BASE_COOLDOWN = 15.0
_COOLDOWN_CAP = 300.0


class PairCircuitBreaker:
    """Process-local registry of pair circuits.

    Task 28.1: `_slots` is now lock-guarded. APScheduler (revive tick) and
    the main freqtrade bot loop (populate_entry_trend / custom_exit /
    custom_entry_price) mutate this dict from different threads; the
    previous lock-free implementation had a classic read-check-insert
    race where two concurrent first-touches on the same pair could each
    allocate their own _PairSlot, losing one's state.
    """

    def __init__(self):
        import threading
        self._slots: Dict[str, _PairSlot] = {}
        self._lock = threading.Lock()

    def _slot(self, pair: str) -> _PairSlot:
        with self._lock:
            slot = self._slots.get(pair)
            if slot is None:
                slot = _PairSlot()
                self._slots[pair] = slot
            return slot

    def is_dormant(self, pair: str) -> bool:
        with self._lock:
            slot = self._slots.get(pair)
        if slot is None:
            return False
        return time.time() < slot.blacklisted_until

    def status(self, pair: str) -> Dict[str, float]:
        with self._lock:
            slot = self._slots.get(pair)
        if slot is None:
            return {"dormant": False, "consecutive_failures": 0,
                    "seconds_until_revival": 0.0}
        remaining = max(0.0, slot.blacklisted_until - time.time())
        return {
            "dormant": remaining > 0,
            "consecutive_failures": slot.consecutive_failures,
            "total_failures": slot.total_failures,
            "total_successes": slot.total_successes,
            "seconds_until_revival": remaining,
            "last_reason": slot.last_failure_reason,
        }

    def record_success(self, pair: str) -> None:
        slot = self._slot(pair)
        slot.consecutive_failures = 0
        slot.backoff_level = 0
        slot.blacklisted_until = 0.0
        slot.last_event_at = time.time()
        slot.total_successes += 1
        self._publish_state(pair, slot)

    def record_failure(self, pair: str, reason: str = "") -> bool:
        """Returns True if the failure pushed the pair into dormant state."""
        slot = self._slot(pair)
        slot.consecutive_failures += 1
        slot.total_failures += 1
        slot.last_event_at = time.time()
        slot.last_failure_reason = (reason or "unspecified")[:120]

        if slot.consecutive_failures >= _OPEN_THRESHOLD:
            slot.backoff_level = slot.consecutive_failures - _OPEN_THRESHOLD
            cooldown = min(
                _COOLDOWN_CAP,
                _BASE_COOLDOWN * (2 ** slot.backoff_level),
            )
            slot.blacklisted_until = time.time() + cooldown
            logger.warning(
                f"[PairCircuit] {pair} DORMANT reason={slot.last_failure_reason} "
                f"consec={slot.consecutive_failures} cooldown={cooldown:.0f}s"
            )
            self._publish_state(pair, slot)
            return True
        # Even sub-threshold failures get published so other listeners see
        # a rising consecutive_failures count.
        self._publish_state(pair, slot)
        return False

    def record_order_attempt(self, pair: str, filled: bool,
                             age_seconds: float = 0.0) -> bool:
        """Record an order outcome for fill-rate tracking.

        D1 (2026-04-25): production observed the same 4 pairs (ICP, WIF,
        UNI, 0G) re-attempting limit-short trades every cycle for 3+
        hours without filling. The chronic non-fill pattern is invisible
        to the consecutive_failures circuit because each attempt isn't
        itself an "exception" — the order is just sitting there. With
        fill-rate tracking, a pair whose 1h rolling rate drops below 20%
        with at least 5 attempts gets flipped into a 30-min soft dormant
        and the entry path skips it.

        AUDIT-13 (2026-04-25): publication snapshot is taken INSIDE the
        lock so the pheromone deposit reflects exactly the state we just
        wrote — not a state another thread mutated between unlock and
        publish.

        Returns True if this attempt pushed the pair into dormant.
        """
        now = time.time()
        publish_snapshot: Optional[Dict[str, Any]] = None
        with self._lock:
            slot = self._slots.get(pair)
            if slot is None:
                slot = _PairSlot()
                self._slots[pair] = slot
            slot.order_attempts.append((now, bool(filled), float(age_seconds)))
            # Trim to last hour
            slot.order_attempts = [
                a for a in slot.order_attempts if now - a[0] < 3600.0
            ]
            n = len(slot.order_attempts)
            filled_count = sum(1 for a in slot.order_attempts if a[1])
            fill_rate = filled_count / n if n > 0 else 1.0
            went_dormant = False
            if n >= 5 and fill_rate < 0.20 and not (slot.blacklisted_until > now):
                slot.blacklisted_until = now + 1800.0
                slot.last_failure_reason = f"low_fill_rate:{fill_rate:.2f}"
                went_dormant = True
                logger.warning(
                    f"[PairCircuit] {pair} SOFT-DORMANT (fill_rate={fill_rate:.2%} "
                    f"n={n} window=1h cooldown=30min)"
                )
                # Capture exact state for publication while we still hold the lock.
                publish_snapshot = {
                    "consecutive_failures": slot.consecutive_failures,
                    "blacklisted_until": slot.blacklisted_until,
                    "last_failure_reason": slot.last_failure_reason,
                    "order_attempts_snapshot": list(slot.order_attempts),
                }
        if went_dormant and publish_snapshot is not None:
            self._publish_snapshot(pair, publish_snapshot)
        return went_dormant

    def _publish_snapshot(self, pair: str, snap: Dict[str, Any]) -> None:
        """Publish a previously-captured slot snapshot. Used by paths
        that need to release the lock before doing pheromone IO. The
        deposited payload reflects the snapshot, not the live slot.
        """
        try:
            from pheromone_field import get_pheromone_field
            pfield = get_pheromone_field()
            now_t = time.time()
            attempts = snap.get("order_attempts_snapshot", [])
            recent = [a for a in attempts if now_t - a[0] < 3600.0]
            fill_rate = (
                sum(1 for a in recent if a[1]) / len(recent)
                if recent else None
            )
            blacklisted_until = float(snap.get("blacklisted_until", 0.0) or 0.0)
            pfield.deposit(
                "pair_circuit", f"dormant::{pair}",
                {
                    "pair": pair,
                    "consecutive_failures": snap.get("consecutive_failures", 0),
                    "blacklisted_until": blacklisted_until,
                    "reason": snap.get("last_failure_reason", ""),
                    "dormant": now_t < blacklisted_until,
                    "fill_rate_1h": fill_rate,
                    "n_attempts_1h": len(recent),
                },
                half_life=max(60.0, blacklisted_until - now_t)
                if blacklisted_until > now_t else 30.0,
            )
        except Exception:
            pass

    def get_fill_rate(self, pair: str) -> Optional[float]:
        """Return current 1h rolling fill rate for a pair (None if no data)."""
        now = time.time()
        with self._lock:
            slot = self._slots.get(pair)
            if slot is None or not slot.order_attempts:
                return None
            recent = [a for a in slot.order_attempts if now - a[0] < 3600.0]
            if not recent:
                return None
            filled_count = sum(1 for a in recent if a[1])
            return filled_count / len(recent)

    # ─── Sprint 2026-05-01: spread/age/volume telemetry ──────────────────
    # Rolling window of observed bid/ask spreads per pair. Fed by
    # HydraSizer.bot_loop_start which reads dp.orderbook(pair, 1) for
    # every active pair. Callers (scheduler._adaptive_pairlist_tune)
    # query the aggregate distribution to compute exchange-wide medians
    # so SpreadFilter / VolumePairList / AgeFilter values can be derived
    # rather than hardcoded.

    def record_pair_observation(self, pair: str, spread: Optional[float] = None,
                                quote_volume: Optional[float] = None) -> None:
        """Append a (timestamp, spread, quote_volume) sample to the slot.

        Trimmed to last hour. None values are skipped per-field so a single
        ticker that has spread but not volume doesn't poison the window.
        """
        if spread is None and quote_volume is None:
            return
        now = time.time()
        with self._lock:
            slot = self._slots.get(pair)
            if slot is None:
                slot = _PairSlot()
                self._slots[pair] = slot
            # Use a separate observations list so it doesn't pollute
            # order_attempts. We piggyback by stashing it on the slot dict.
            obs = getattr(slot, "_observations", None)
            if obs is None:
                obs = []
                setattr(slot, "_observations", obs)
            obs.append((now, spread, quote_volume))
            # Trim to last hour
            cutoff = now - 3600.0
            slot._observations = [o for o in obs if o[0] >= cutoff]

    def _all_recent_observations(self) -> Dict[str, list]:
        """Snapshot of every pair's last-hour observations. Used by aggregate
        statistics — never mutates state."""
        now = time.time()
        cutoff = now - 3600.0
        snap: Dict[str, list] = {}
        with self._lock:
            for pair, slot in list(self._slots.items()):
                obs = getattr(slot, "_observations", None) or []
                recent = [o for o in obs if o[0] >= cutoff]
                if recent:
                    snap[pair] = recent
        return snap

    def get_exchange_spread_distribution(self) -> Dict[str, float]:
        """Per-pair median 1h-rolling spread. Empty dict if no samples yet.

        Uses observations recorded via record_pair_observation(). The
        median per pair is the noise-robust per-pair "true" spread; the
        aggregate distribution is what adaptive filters consume.
        """
        result: Dict[str, float] = {}
        for pair, recent in self._all_recent_observations().items():
            spreads = [o[1] for o in recent if o[1] is not None]
            if not spreads:
                continue
            spreads.sort()
            mid = len(spreads) // 2
            if len(spreads) % 2 == 0:
                med = 0.5 * (spreads[mid - 1] + spreads[mid])
            else:
                med = spreads[mid]
            result[pair] = float(med)
        return result

    def adaptive_spread_cap(self, fallback: float = 0.05) -> float:
        """Adaptive max-spread-ratio for SpreadFilter consumers.

        Computed as exchange-wide median × hormonal panic factor. Calm:
        median × 1.5 (admit pairs up to 1.5× median spread). Panic:
        median × 1.05 (only the cleanest books).

        `fallback` is returned only when there are NO spread observations
        yet (cold start within first hour). The fallback should itself
        come from a config or envelope — we keep a permissive 5% so the
        bot can build observation history without locking everyone out.
        """
        dist = self.get_exchange_spread_distribution()
        if not dist:
            return float(fallback)
        spreads = sorted(dist.values())
        mid = len(spreads) // 2
        if len(spreads) % 2 == 0:
            median_spread = 0.5 * (spreads[mid - 1] + spreads[mid])
        else:
            median_spread = spreads[mid]
        # Hormonal modulation: cortisol 1.0 (calm) → factor 1.5,
        # cortisol 0.5 (panic) → factor 1.05.
        try:
            from neural_organism import get_organism
            cort = float(get_organism().hormones.cortisol)
        except Exception:
            cort = 1.0
        factor = 1.05 + 0.45 * max(0.0, min(1.0, (cort - 0.5) / 0.5))
        cap = median_spread * factor
        # Hard floor 0.005 (0.5%) — never admit a pair with sub-half-percent
        # noise floor; hard ceiling 0.20 (20%) — anything wider is broken book.
        return max(0.005, min(0.20, cap))

    def adaptive_age_floor(self, fallback: int = 7) -> int:
        """Adaptive AgeFilter min_days_listed, hormone-driven.

        Calm cortisol (1.0) → floor 7 days (permissive — accept newer pairs).
        Panic (0.5) → floor 30 days (demand more history under stress).
        """
        try:
            from neural_organism import get_organism
            cort = float(get_organism().hormones.cortisol)
        except Exception:
            cort = 1.0
        # cort 1.0 → 7, cort 0.5 → 30
        days = int(round(7 + (1.0 - cort) * 46))
        return max(1, min(60, days))

    def adaptive_volume_floor(self, exchange_quote_volumes: Optional[list] = None,
                              fallback: float = 0.0) -> float:
        """Adaptive VolumePairList min_value (5th-percentile of exchange volumes).

        If no list is provided, derive from observations. Returns 0.0 when
        there's no data — the safest permissive floor for cold start.
        """
        vols = list(exchange_quote_volumes) if exchange_quote_volumes else []
        if not vols:
            for recent in self._all_recent_observations().values():
                vols.extend([o[2] for o in recent if o[2] is not None])
        if not vols:
            return float(fallback)
        vols = sorted(v for v in vols if v and v > 0)
        if not vols:
            return float(fallback)
        idx = max(0, int(len(vols) * 0.05) - 1)
        return float(vols[idx])

    def revive_probe(self, pair: str, orderbook: Optional[Dict]) -> bool:
        """Run a manual probe from outside (scheduler revival job). Returns
        True if the pair woke up as a result of this probe."""
        if not self.is_dormant(pair):
            return False
        if orderbook and orderbook.get("bids") and orderbook.get("asks"):
            logger.info(f"[PairCircuit] {pair} REVIVED — orderbook healthy")
            self.record_success(pair)
            return True
        return False

    # ─── Pheromone publication (read-only side-channel) ─────────────────────

    def _publish_state(self, pair: str, slot: _PairSlot) -> None:
        try:
            from pheromone_field import get_pheromone_field
            pfield = get_pheromone_field()
            # AUDIT-6 (2026-04-25): include rolling fill_rate so dashboards +
            # downstream consumers (HydraSizer DCA gates, telemetry) can read
            # the chronic-non-filler signal directly. Previously get_fill_rate
            # was implemented but had no consumer.
            now_t = time.time()
            recent = [a for a in slot.order_attempts if now_t - a[0] < 3600.0]
            fill_rate = (
                sum(1 for a in recent if a[1]) / len(recent)
                if recent else None
            )
            pfield.deposit(
                "pair_circuit", f"dormant::{pair}",
                {
                    "pair": pair,
                    "consecutive_failures": slot.consecutive_failures,
                    "blacklisted_until": slot.blacklisted_until,
                    "reason": slot.last_failure_reason,
                    "dormant": now_t < slot.blacklisted_until,
                    "fill_rate_1h": fill_rate,
                    "n_attempts_1h": len(recent),
                },
                half_life=max(60.0, slot.blacklisted_until - now_t)
                if slot.blacklisted_until > now_t else 30.0,
            )
        except Exception:
            # Pheromone outage must not break pair-circuit state updates.
            pass


# ─── Singleton ──────────────────────────────────────────────────────────────

_breaker: Optional[PairCircuitBreaker] = None


def get_pair_circuit() -> PairCircuitBreaker:
    global _breaker
    if _breaker is None:
        _breaker = PairCircuitBreaker()
    return _breaker
