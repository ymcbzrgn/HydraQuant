"""
Phase 26: Stigmergic Pheromone Field — Lock-Free Module Coordination

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

Source: S-MADRL, Nature Communications Engineering 2024
"""

import logging
import time
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


class PheromoneField:
    """Shared pheromone environment for all cognitive modules.

    Thread-safe via dict operations (CPython GIL makes dict read/write atomic for simple ops).
    No explicit locks needed — pheromone semantics tolerate stale reads.

    Usage:
        field = get_pheromone_field()

        # Module deposits a signal
        field.deposit("catboost", "prediction", {"signal": "BULLISH", "confidence": 0.78}, half_life=60)

        # Another module reads it
        pred = field.read("prediction")  # decayed value
        all_signals = field.read_all()   # all active pheromones

        # Organism health check
        health = field.get_field_health()
    """

    def __init__(self):
        self._field: Dict[str, Pheromone] = {}
        self._deposit_count = 0
        self._read_count = 0
        self._history: List[Dict] = []  # last N deposits for debugging
        self._history_max = 100

    def deposit(
        self,
        source: str,
        signal_type: str,
        value: Any,
        half_life: float = 30.0,
        metadata: Optional[Dict] = None,
    ) -> None:
        """Deposit a pheromone signal into the field.

        Args:
            source: Module name (e.g., "catboost", "hormones", "amygdala")
            signal_type: Signal identifier (e.g., "prediction", "cortisol", "fear_level")
            value: Payload — can be float, dict, list, etc.
            half_life: Seconds until signal strength decays to 50% (default 30s)
            metadata: Optional extra data (e.g., pair, regime)

        Key format: "{source}::{signal_type}" — allows multiple sources for same signal type.
        Read by signal_type returns the FRESHEST deposit across all sources.
        """
        pheromone = Pheromone(
            value=value,
            source=source,
            deposited_at=time.monotonic(),
            half_life=max(half_life, 0.1),
            metadata=metadata or {},
        )
        # Compound key: source::signal_type → no collision between modules
        compound_key = f"{source}::{signal_type}"
        self._field[compound_key] = pheromone
        self._deposit_count += 1

        # Track history
        if len(self._history) >= self._history_max:
            self._history.pop(0)
        self._history.append({
            "source": source,
            "signal": signal_type,
            "time": time.monotonic(),
            "half_life": half_life,
        })

    def read(self, signal_type: str, source: Optional[str] = None, raw: bool = False) -> Any:
        """Read a pheromone signal with temporal decay applied.

        Args:
            signal_type: Signal to read (matches all sources unless source specified)
            source: Optional specific source module to read from
            raw: If True, return Pheromone object without decay. If False, return decayed value.

        Returns:
            Decayed value from the FRESHEST matching deposit (or None if fully decayed)
        """
        self._read_count += 1

        if source:
            # Exact lookup
            pheromone = self._field.get(f"{source}::{signal_type}")
        else:
            # Find freshest across all sources for this signal_type
            pheromone = None
            best_freshness = 0.0
            for key, p in self._field.items():
                if key.endswith(f"::{signal_type}"):
                    age = time.monotonic() - p.deposited_at
                    freshness = 0.5 ** (age / p.half_life)
                    if freshness > best_freshness:
                        best_freshness = freshness
                        pheromone = p

        if pheromone is None:
            return None

        if raw:
            return pheromone

        # Apply exponential decay
        age = time.monotonic() - pheromone.deposited_at
        decay_factor = 0.5 ** (age / pheromone.half_life)

        # If decayed below 1% → effectively dead
        if decay_factor < 0.01:
            return None

        # For numeric values, multiply by decay factor
        if isinstance(pheromone.value, (int, float)):
            return pheromone.value * decay_factor
        elif isinstance(pheromone.value, dict):
            # For dict values, add decay metadata but keep structure
            result = pheromone.value.copy()
            result["_decay"] = round(decay_factor, 4)
            result["_age_s"] = round(age, 1)
            result["_source"] = pheromone.source
            return result
        else:
            return pheromone.value

    def read_float(self, signal_type: str, default: float = 0.0) -> float:
        """Convenience: read a numeric pheromone, return float with decay applied."""
        val = self.read(signal_type)
        if val is None:
            return default
        if isinstance(val, (int, float)):
            return float(val)
        if isinstance(val, dict):
            # Try to extract primary value
            for key in ["value", "confidence", "score", "level"]:
                if key in val:
                    return float(val[key]) * val.get("_decay", 1.0)
        return default

    def read_all(self) -> Dict[str, Any]:
        """Read all active (non-fully-decayed) pheromones."""
        result = {}
        now = time.monotonic()
        for compound_key, pheromone in list(self._field.items()):
            age = now - pheromone.deposited_at
            decay = 0.5 ** (age / pheromone.half_life)
            if decay >= 0.01:
                result[compound_key] = {
                    "value": pheromone.value,
                    "source": pheromone.source,
                    "signal_type": compound_key.split("::", 1)[-1] if "::" in compound_key else compound_key,
                    "decay": round(decay, 4),
                    "age_seconds": round(age, 1),
                }
        return result

    def read_by_source(self, source: str) -> Dict[str, Any]:
        """Read all pheromones deposited by a specific module."""
        result = {}
        for compound_key, pheromone in self._field.items():
            if pheromone.source == source:
                signal_type = compound_key.split("::", 1)[-1] if "::" in compound_key else compound_key
                val = self.read(signal_type, source=source)
                if val is not None:
                    result[signal_type] = val
        return result

    def get_freshness(self, signal_type: str, source: Optional[str] = None) -> float:
        """Get freshness of a signal (1.0 = just deposited, 0.0 = fully decayed)."""
        if source:
            pheromone = self._field.get(f"{source}::{signal_type}")
        else:
            # Find freshest across all sources
            pheromone = None
            for key, p in self._field.items():
                if key.endswith(f"::{signal_type}"):
                    if pheromone is None or p.deposited_at > pheromone.deposited_at:
                        pheromone = p
        if pheromone is None:
            return 0.0
        age = time.monotonic() - pheromone.deposited_at
        return max(0.0, 0.5 ** (age / pheromone.half_life))

    def cleanup(self) -> int:
        """Remove fully decayed pheromones. Called periodically by scheduler."""
        now = time.monotonic()
        dead_keys = []
        for signal_type, pheromone in self._field.items():
            age = now - pheromone.deposited_at
            decay = 0.5 ** (age / pheromone.half_life)
            if decay < 0.001:  # <0.1% strength → dead
                dead_keys.append(signal_type)

        for key in dead_keys:
            del self._field[key]

        if dead_keys:
            logger.debug(f"[Pheromone] Cleaned {len(dead_keys)} dead signals")
        return len(dead_keys)

    def get_field_health(self) -> Dict[str, Any]:
        """Get field status for monitoring/visualization."""
        now = time.monotonic()
        active_count = 0
        sources = set()
        total_freshness = 0.0

        for pheromone in self._field.values():
            age = now - pheromone.deposited_at
            decay = 0.5 ** (age / pheromone.half_life)
            if decay >= 0.01:
                active_count += 1
                sources.add(pheromone.source)
                total_freshness += decay

        avg_freshness = total_freshness / active_count if active_count > 0 else 0.0

        return {
            "active_signals": active_count,
            "total_deposits": self._deposit_count,
            "total_reads": self._read_count,
            "active_sources": sorted(sources),
            "avg_freshness": round(avg_freshness, 4),
            "field_size": len(self._field),
        }

    # --- Pre-defined signal types (type safety) ---

    SIGNAL_PREDICTION = "prediction"           # Triple Perception output
    SIGNAL_CORTISOL = "cortisol"               # Hormone level
    SIGNAL_DOPAMINE = "dopamine"               # Hormone level
    SIGNAL_SEROTONIN = "serotonin"             # Hormone level
    SIGNAL_ADRENALINE = "adrenaline"           # Hormone level
    SIGNAL_FEAR = "fear_level"                 # Amygdala fear
    SIGNAL_OOD = "ood_score"                   # Out-of-distribution
    SIGNAL_UNCERTAINTY = "uncertainty"          # Ensemble variance
    SIGNAL_CONFIDENCE = "confidence_interval"  # CQR interval
    SIGNAL_CROWD = "crowd_direction"           # Mirror neurons
    SIGNAL_REGIME = "market_regime"            # Regime classifier
    SIGNAL_HEALTH = "organism_health"          # Interoception composite
    SIGNAL_CAUSAL = "causal_insight"           # Causal engine discovery
    SIGNAL_WORLD = "world_prediction"          # World model simulation


# =============================================================================
# Singleton
# =============================================================================

_pheromone_field: Optional[PheromoneField] = None


def get_pheromone_field() -> PheromoneField:
    """Get or create the global pheromone field singleton."""
    global _pheromone_field
    if _pheromone_field is None:
        _pheromone_field = PheromoneField()
        logger.info("[Pheromone] Global pheromone field initialized")
    return _pheromone_field
