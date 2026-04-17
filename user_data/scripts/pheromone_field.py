"""
Phase 26: Stigmergic Pheromone Field — Lock-Free Module Coordination
Phase 27 Fix 3 (F2 Ajani): Ring-buffer trails replace single-deposit overwrite.

Replaces Global Workspace RLock with bio-inspired pheromone communication.
Modules don't send messages — they DEPOSIT pheromones. Others READ them.
Pheromones naturally DECAY over time → stale data problem solved inherently.

Novel Contribution #9: Stigmergic Cognitive Architecture
No existing financial system uses pheromone-based module coordination.

Key properties:
  - Lock-free: no RLock, no race conditions, no deadlocks
  - Natural temporal decay: old signals fade, new signals dominate
  - Emergent coordination: no central controller, modules self-organize
  - Timestamp alignment FREE: decay IS the alignment mechanism
  - Trails (Phase 27): each key keeps the last K deposits in a deque, enabling
    integrals / gradients / acceleration instead of only the latest sample.

Phase 27 references:
  - Dorigo & Stutzle (2004) ACO: τ(t+1)=(1-ρ)·τ(t)+Σ Δτ  — leaky accumulation
  - Usher & McClelland (2001) LIF neuron: dx/dt = -k·x + I(t)
  - Stutzle & Hoos (2000) MMAS bounds [τ_min, τ_max] prevent saturation.

Source: S-MADRL, Nature Communications Engineering 2024
"""

import logging
import time
from collections import deque
from typing import Any, Dict, Optional, List
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class Pheromone:
    """Single pheromone deposit in the field."""
    value: Any                    # payload (float, dict, list, etc.)
    source: str                   # which module deposited this
    deposited_at: float           # time.monotonic() timestamp
    half_life: float              # seconds until signal strength halves
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PheromoneTrail:
    """Ring buffer of deposits for a single compound_key (Phase 27 Fix 3).

    `deposits` is a deque(maxlen=K) so that old pheromones fall off naturally.
    `accumulated_value` is a LIF (leaky-integrate-and-fire) membrane potential
    used for fast integral reads — it decays each deposit by the half-life
    elapsed since the previous deposit, then adds the new extracted float.
    MMAS bounds [TAU_MIN, TAU_MAX] prevent saturation (Stutzle & Hoos 2000).
    """
    deposits: deque
    accumulated_value: float = 0.0

    @property
    def latest(self) -> Optional[Pheromone]:
        return self.deposits[-1] if self.deposits else None


class PheromoneField:
    """Shared pheromone environment for all cognitive modules.

    Thread-safe via dict operations (CPython GIL makes dict read/write atomic).
    Trails replace the Phase 26 single-deposit overwrite; this unlocks HQ-1
    (Confidence Integral) and HQ-3 (Pheromone Gradient) from PHASE27_ALPHA.md.

    Usage:
        field = get_pheromone_field()

        # Module deposits a signal (appends to trail)
        field.deposit("catboost", "prediction",
                      {"signal": "BULLISH", "confidence": 0.78}, half_life=60)

        # Read latest (decayed)            — unchanged Phase 26 behavior
        pred = field.read("prediction")
        all_signals = field.read_all()

        # Phase 27 — trail-aware reads:
        integ = field.read_integral("prediction", window_seconds=14400)
        grad  = field.read_gradient("prediction")
        acc   = field.read_acceleration("prediction")
    """

    # MMAS saturation bounds — accumulated_value is clamped into [τ_min, τ_max]
    # so no single signal can dominate and no signal is fully wiped.
    TAU_MIN = 0.01
    TAU_MAX = 10.0

    def __init__(self, trail_length: int = 32):
        self._field: Dict[str, PheromoneTrail] = {}
        self._trail_length = int(max(1, trail_length))
        self._deposit_count = 0
        self._read_count = 0
        self._history: List[Dict] = []  # last N deposits for debugging
        self._history_max = 100

    # ─── Deposit ────────────────────────────────────────────────────────────

    def deposit(
        self,
        source: str,
        signal_type: str,
        value: Any,
        half_life: float = 30.0,
        metadata: Optional[Dict] = None,
    ) -> None:
        """Append a pheromone deposit to the trail for (source, signal_type).

        Args:
            source: Module name (e.g., "catboost", "hormones", "agent_pool")
            signal_type: Signal identifier (e.g., "prediction", "cortisol")
            value: Payload — can be float, dict, list, etc.
            half_life: Seconds until signal strength decays to 50% (default 30s)
            metadata: Optional extra data (e.g., pair, regime)

        Key format: "{source}::{signal_type}" — allows multiple sources for
        the same signal type. Read by signal_type returns the FRESHEST deposit
        across all matching trails.
        """
        pheromone = Pheromone(
            value=value,
            source=source,
            deposited_at=time.monotonic(),
            half_life=max(half_life, 0.1),
            metadata=metadata or {},
        )
        compound_key = f"{source}::{signal_type}"

        trail = self._field.get(compound_key)
        if trail is None:
            trail = PheromoneTrail(
                deposits=deque(maxlen=self._trail_length),
                accumulated_value=0.0,
            )
            self._field[compound_key] = trail

        # LIF leaky accumulation — decay first, then add new signal.
        added = self._extract_float(value)
        if trail.deposits:
            last = trail.deposits[-1]
            dt = pheromone.deposited_at - last.deposited_at
            decay_factor = 0.5 ** (dt / max(last.half_life, 0.1))
            trail.accumulated_value = trail.accumulated_value * decay_factor + added
        else:
            trail.accumulated_value = added

        # MMAS bounds — saturation protection.
        trail.accumulated_value = max(
            self.TAU_MIN, min(self.TAU_MAX, trail.accumulated_value)
        )

        trail.deposits.append(pheromone)
        self._deposit_count += 1

        # Debug history
        if len(self._history) >= self._history_max:
            self._history.pop(0)
        self._history.append({
            "source": source,
            "signal": signal_type,
            "time": pheromone.deposited_at,
            "half_life": half_life,
        })

    # ─── Latest-sample reads (Phase 26 contract preserved) ─────────────────

    def read(self, signal_type: str, source: Optional[str] = None,
             raw: bool = False) -> Any:
        """Read the FRESHEST pheromone in the matching trail(s), decayed.

        Args:
            signal_type: Signal to read (matches all sources unless specified)
            source: Optional specific source module to read from
            raw: If True, return Pheromone object without decay.

        Returns:
            Decayed value from the freshest matching deposit, or None if fully
            decayed / absent.
        """
        self._read_count += 1
        pheromone = self._latest_pheromone(signal_type, source)
        if pheromone is None:
            return None
        if raw:
            return pheromone

        age = time.monotonic() - pheromone.deposited_at
        decay_factor = 0.5 ** (age / pheromone.half_life)
        if decay_factor < 0.01:
            return None

        if isinstance(pheromone.value, (int, float)):
            return pheromone.value * decay_factor
        if isinstance(pheromone.value, dict):
            result = pheromone.value.copy()
            result["_decay"] = round(decay_factor, 4)
            result["_age_s"] = round(age, 1)
            result["_source"] = pheromone.source
            return result
        return pheromone.value

    def read_float(self, signal_type: str, default: float = 0.0) -> float:
        """Convenience: read a numeric pheromone with decay applied."""
        val = self.read(signal_type)
        if val is None:
            return default
        if isinstance(val, (int, float)):
            return float(val)
        if isinstance(val, dict):
            for key in ("value", "confidence", "score", "level"):
                if key in val:
                    return float(val[key]) * val.get("_decay", 1.0)
        return default

    def read_all(self) -> Dict[str, Any]:
        """Read all active (non-fully-decayed) pheromones.

        Outer-dict keys are `"{source}::{signal_type}"` (Phase 26 format — two
        sources publishing the same signal_type stay separate). Each value dict
        includes both `signal_type` (bare name) and `key` (alias) so consumers
        can look up signals without parsing the compound key themselves.
        """
        result = {}
        now = time.monotonic()
        for compound_key, trail in list(self._field.items()):
            pheromone = trail.latest
            if pheromone is None:
                continue
            age = now - pheromone.deposited_at
            decay = 0.5 ** (age / pheromone.half_life)
            if decay >= 0.01:
                bare_signal = (compound_key.split("::", 1)[-1]
                               if "::" in compound_key else compound_key)
                result[compound_key] = {
                    "value": pheromone.value,
                    "source": pheromone.source,
                    "signal_type": bare_signal,
                    # Backward-compat alias — lets code that used to key by
                    # signal_type still locate the record (value["key"]).
                    "key": bare_signal,
                    "compound_key": compound_key,
                    "decay": round(decay, 4),
                    "age_seconds": round(age, 1),
                    "trail_depth": len(trail.deposits),
                    "accumulated_value": round(trail.accumulated_value, 6),
                }
        return result

    def read_signals(self) -> Dict[str, Any]:
        """Flat dict keyed by bare `signal_type` (not compound). When multiple
        sources publish the same signal_type, the freshest deposit wins (tie
        broken by raw monotonic timestamp). Useful for consumers that don't
        care about the originating module.
        """
        grouped: Dict[str, Dict[str, Any]] = {}
        latest_ts: Dict[str, float] = {}
        for compound_key, trail in self._field.items():
            pheromone = trail.latest
            if pheromone is None:
                continue
            bare = (compound_key.split("::", 1)[-1]
                    if "::" in compound_key else compound_key)
            if bare not in latest_ts or pheromone.deposited_at > latest_ts[bare]:
                entry = self.read_all().get(compound_key)
                if entry is None:
                    continue
                grouped[bare] = entry
                latest_ts[bare] = pheromone.deposited_at
        return grouped

    def read_by_source(self, source: str) -> Dict[str, Any]:
        """Read latest pheromones deposited by a specific module."""
        result = {}
        for compound_key, trail in self._field.items():
            pheromone = trail.latest
            if pheromone is None or pheromone.source != source:
                continue
            signal_type = (compound_key.split("::", 1)[-1]
                           if "::" in compound_key else compound_key)
            val = self.read(signal_type, source=source)
            if val is not None:
                result[signal_type] = val
        return result

    def get_freshness(self, signal_type: str, source: Optional[str] = None) -> float:
        """Get freshness of a signal (1.0 = just deposited, 0.0 = fully decayed)."""
        pheromone = self._latest_pheromone(signal_type, source)
        if pheromone is None:
            return 0.0
        age = time.monotonic() - pheromone.deposited_at
        return max(0.0, 0.5 ** (age / pheromone.half_life))

    # ─── Phase 27 trail reads — integral / gradient / acceleration ─────────

    def read_integral(self, signal_type: str,
                      window_seconds: float = 3600.0,
                      source: Optional[str] = None) -> float:
        """HQ-1: Leaky integral of a signal over a time window.

        Two trails with the same "bullish" headline but different integrals
        mean different things: a 4h-old-but-just-repeated bullish signal
        outweighs a 5-min-old flip.

        Computes a decay-weighted average of the scalar-extracted values in
        each matching trail's deposits whose age is within window_seconds.
        """
        now = time.monotonic()
        trails = self._matching_trails(signal_type, source)
        if not trails:
            return 0.0

        total = 0.0
        weight_sum = 0.0
        for trail in trails:
            for pheromone in trail.deposits:
                age = now - pheromone.deposited_at
                if age > window_seconds:
                    continue
                decay = 0.5 ** (age / max(pheromone.half_life, 0.1))
                value = self._extract_float(pheromone.value)
                total += value * decay
                weight_sum += decay

        if weight_sum <= 0:
            return 0.0
        return total / weight_sum

    def read_gradient(self, signal_type: str,
                      source: Optional[str] = None) -> float:
        """HQ-3: First derivative dS/dt between the last two deposits.

        > 0 rising conviction, < 0 fading conviction. Always 0 with <2 samples.
        """
        for trail in self._matching_trails(signal_type, source):
            if len(trail.deposits) < 2:
                return 0.0
            latest = trail.deposits[-1]
            previous = trail.deposits[-2]
            dt = latest.deposited_at - previous.deposited_at
            if dt < 1e-3:
                return 0.0
            v_latest = self._extract_float(latest.value)
            v_previous = self._extract_float(previous.value)
            return (v_latest - v_previous) / dt
        return 0.0

    def read_acceleration(self, signal_type: str,
                          source: Optional[str] = None) -> float:
        """HQ-3: Second derivative d²S/dt² — is conviction accelerating?
        Returns 0 when fewer than 3 deposits exist in any matching trail."""
        for trail in self._matching_trails(signal_type, source):
            if len(trail.deposits) < 3:
                return 0.0
            p1, p2, p3 = trail.deposits[-3], trail.deposits[-2], trail.deposits[-1]
            dt1 = p2.deposited_at - p1.deposited_at
            dt2 = p3.deposited_at - p2.deposited_at
            if dt1 < 1e-3 or dt2 < 1e-3:
                return 0.0
            v1 = self._extract_float(p1.value)
            v2 = self._extract_float(p2.value)
            v3 = self._extract_float(p3.value)
            g1 = (v2 - v1) / dt1
            g2 = (v3 - v2) / dt2
            return (g2 - g1) / ((dt1 + dt2) / 2.0)
        return 0.0

    def read_accumulated(self, signal_type: str,
                         source: Optional[str] = None) -> float:
        """Return the LIF accumulated_value of the matching trail, MMAS-clamped."""
        trails = self._matching_trails(signal_type, source)
        if not trails:
            return 0.0
        # Pick the trail whose LATEST pheromone is freshest.
        best_trail = max(trails,
                         key=lambda t: (t.latest.deposited_at if t.latest else 0.0))
        return best_trail.accumulated_value

    # ─── Maintenance ────────────────────────────────────────────────────────

    def cleanup(self) -> int:
        """Remove trails whose latest deposit is fully decayed.
        Called periodically by scheduler."""
        now = time.monotonic()
        dead_keys = []
        for compound_key, trail in self._field.items():
            pheromone = trail.latest
            if pheromone is None:
                dead_keys.append(compound_key)
                continue
            age = now - pheromone.deposited_at
            decay = 0.5 ** (age / max(pheromone.half_life, 0.1))
            if decay < 0.001:  # <0.1% strength → dead
                dead_keys.append(compound_key)

        for key in dead_keys:
            del self._field[key]

        if dead_keys:
            logger.debug(f"[Pheromone] Cleaned {len(dead_keys)} dead trails")
        return len(dead_keys)

    def get_field_health(self) -> Dict[str, Any]:
        """Get field status for monitoring/visualization."""
        now = time.monotonic()
        active_count = 0
        sources = set()
        total_freshness = 0.0
        total_deposits_in_trails = 0

        for trail in self._field.values():
            pheromone = trail.latest
            if pheromone is None:
                continue
            age = now - pheromone.deposited_at
            decay = 0.5 ** (age / max(pheromone.half_life, 0.1))
            if decay >= 0.01:
                active_count += 1
                sources.add(pheromone.source)
                total_freshness += decay
                total_deposits_in_trails += len(trail.deposits)

        avg_freshness = total_freshness / active_count if active_count > 0 else 0.0

        return {
            "active_signals": active_count,
            "total_deposits": self._deposit_count,
            "total_reads": self._read_count,
            "active_sources": sorted(sources),
            "avg_freshness": round(avg_freshness, 4),
            "field_size": len(self._field),
            "trail_length": self._trail_length,
            "deposits_in_trails": total_deposits_in_trails,
        }

    # ─── Internals ──────────────────────────────────────────────────────────

    def _latest_pheromone(self, signal_type: str,
                          source: Optional[str]) -> Optional[Pheromone]:
        """Return the freshest pheromone across matching trails."""
        if source:
            trail = self._field.get(f"{source}::{signal_type}")
            return trail.latest if trail else None

        freshest: Optional[Pheromone] = None
        best_freshness = 0.0
        now = time.monotonic()
        for key, trail in self._field.items():
            if not key.endswith(f"::{signal_type}"):
                continue
            pheromone = trail.latest
            if pheromone is None:
                continue
            age = now - pheromone.deposited_at
            freshness = 0.5 ** (age / max(pheromone.half_life, 0.1))
            if freshness > best_freshness:
                best_freshness = freshness
                freshest = pheromone
        return freshest

    def _matching_trails(self, signal_type: str,
                         source: Optional[str]) -> List[PheromoneTrail]:
        """Return all trails matching the given signal_type (and optional source)."""
        if source:
            trail = self._field.get(f"{source}::{signal_type}")
            return [trail] if trail else []
        return [t for key, t in self._field.items()
                if key.endswith(f"::{signal_type}")]

    @staticmethod
    def _extract_float(value: Any) -> float:
        """Best-effort scalar extraction for integral / LIF math.

        Numbers → themselves. Dicts → look up common fields
        (value, confidence, score, level, strength). Sequences → length. Else 0.
        """
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, dict):
            for key in ("value", "confidence", "score", "level", "strength"):
                if key in value:
                    try:
                        return float(value[key])
                    except (TypeError, ValueError):
                        pass
            return 0.0
        if isinstance(value, (list, tuple)):
            return float(len(value))
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    # --- Pre-defined signal types (type safety) ---

    SIGNAL_PREDICTION = "prediction"             # Triple Perception output
    SIGNAL_CORTISOL = "cortisol"                 # Hormone level
    SIGNAL_DOPAMINE = "dopamine"                 # Hormone level
    SIGNAL_SEROTONIN = "serotonin"               # Hormone level
    SIGNAL_ADRENALINE = "adrenaline"             # Hormone level
    SIGNAL_FEAR = "fear_level"                   # Amygdala fear
    SIGNAL_OOD = "ood_score"                     # Out-of-distribution
    SIGNAL_UNCERTAINTY = "uncertainty"            # Ensemble variance
    SIGNAL_CONFIDENCE = "confidence_interval"    # CQR interval
    SIGNAL_CROWD = "crowd_direction"             # Mirror neurons
    SIGNAL_REGIME = "market_regime"              # Regime classifier
    SIGNAL_HEALTH = "organism_health"            # Interoception composite
    SIGNAL_CAUSAL = "causal_insight"             # Causal engine discovery
    SIGNAL_WORLD = "world_prediction"            # World model simulation
    # Phase 27 Fix 2E / Fix 4 additions
    SIGNAL_AGENT_CONSENSUS = "agent_consensus"   # AgentPool debate majority signal
    SIGNAL_AGENT_DISSENT = "agent_dissent"       # AgentPool strong two-sided split
    SIGNAL_SHAP = "shap_narrative"               # CatBoost SHAP natural-language summary


# =============================================================================
# Singleton
# =============================================================================

_pheromone_field: Optional[PheromoneField] = None


def get_pheromone_field() -> PheromoneField:
    """Get or create the global pheromone field singleton."""
    global _pheromone_field
    if _pheromone_field is None:
        _pheromone_field = PheromoneField()
        logger.info(
            "[Pheromone] Global pheromone field initialized "
            f"(trail_length={_pheromone_field._trail_length}, "
            f"τ∈[{PheromoneField.TAU_MIN}, {PheromoneField.TAU_MAX}])"
        )
    return _pheromone_field
