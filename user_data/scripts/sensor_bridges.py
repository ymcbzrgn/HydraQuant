"""
sensor_bridges.py — Exteroceptive → Interoceptive translation layer.

The organism's hormonal system (neural_organism.Hormones) only reacted to
endogenous inputs (Fear & Greed index, own drawdown, consecutive losses).
Four exogenous pain signals that were observed in production — empty
orderbooks, websocket disconnects, data starvation, shadow-Kelly winrate
collapse — never reached cortisol. The audit called this "organism
cannot feel organ ischemia". This module is the afferent nerve: each
sensor deposits a decayed pheromone, and `aggregate_sensor_stress()`
aggregates them into a [0..1] scalar the organism's compute() pipe
now consumes as an extra stress input.

Design choices (organism philosophy):
  * Nothing blocks. A sensor that cannot reach the pheromone field logs
    and continues — the organism is already degraded if the field is down.
  * Half-lives are calibrated to the event's expected recovery horizon.
    Orderbook flicker (60s) vs shadow-Kelly posterior collapse (1h).
  * The aggregation is idempotent and read-only — callers can aggregate
    as often as they want without mutating state.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Optional, Dict, Any, Tuple

sys.path.append(os.path.dirname(__file__))

logger = logging.getLogger(__name__)

try:
    from pheromone_field import get_pheromone_field
    _PFIELD_AVAILABLE = True
except Exception:
    _PFIELD_AVAILABLE = False


# ─── Source/signal canonical names ──────────────────────────────────────────
# Consumers of aggregate_sensor_stress and the organism compute pathway
# must use these exact keys. Centralising them here lets the registry ride
# on import time instead of tangling magic strings across HydraSizer.

SOURCE_LIQUIDITY = "liquidity_sensor"
SOURCE_EXCHANGE = "exchange_sensor"
SOURCE_DATA = "data_freshness_sensor"
SOURCE_SHADOW = "shadow_kelly_sensor"
SOURCE_MEMORY = "memory_pressure_sensor"

SIG_EMPTY_ORDERBOOK = "empty_orderbook"
SIG_WS_DISCONNECT = "ws_disconnect"
SIG_DATA_STARVATION = "data_starvation"
SIG_SHADOW_DIVERGENCE = "shadow_divergence"
SIG_MEMORY_STRESS = "memory_stress"


def _safe_deposit(source: str, sig: str, value: Any, half_life: float,
                  metadata: Optional[Dict] = None) -> bool:
    if not _PFIELD_AVAILABLE:
        return False
    try:
        pfield = get_pheromone_field()
        pfield.deposit(source, sig, value, half_life=half_life, metadata=metadata or {})
        return True
    except Exception as e:
        logger.debug(f"[SensorBridge] deposit {source}::{sig} failed: {e}")
        return False


def _safe_read_float(source: str, sig: str) -> Optional[float]:
    """Read the LIF accumulated magnitude for (source, signal_type).

    Task 28.6: previously reached into pheromone_field's private `_field`
    dict, which would silently start returning None if the backing
    storage got refactored. `read_accumulated` is the canonical public
    API — returns 0.0 (not None) when the trail is missing, which makes
    the aggregator's `raw <= 0` short-circuit trivially correct.
    """
    if not _PFIELD_AVAILABLE:
        return None
    try:
        pfield = get_pheromone_field()
        return float(pfield.read_accumulated(sig, source=source))
    except Exception:
        return None


# ─── Afferent (sensor → field) ──────────────────────────────────────────────

def record_empty_orderbook(pair: str, side: str = "both",
                           call_site: str = "") -> bool:
    """Orderbook returned empty bids or asks.

    Observed 18,768x for ICP in a single 17h window — the loudest single
    signal in the log stream and previously invisible to the organism.
    half_life=60s — if liquidity returns within a minute the pheromone
    decays on its own; if it keeps firing the trail saturates and the
    aggregator reports high stress.
    """
    payload = {
        "pair": pair,
        "side": side,
        "call_site": call_site,
        "intensity": 1.0,
    }
    return _safe_deposit(SOURCE_LIQUIDITY, SIG_EMPTY_ORDERBOOK, payload,
                         half_life=60.0, metadata={"pair": pair})


def record_ws_disconnect(exchange: str, pair: str = "",
                         timeframe: str = "", code: Optional[int] = None) -> bool:
    """Websocket closed by the remote. Self-recovers on reconnect; the
    pheromone tracks recency so repeated disconnects aggregate into a
    sustained stress signal.
    """
    payload = {
        "exchange": exchange,
        "pair": pair,
        "timeframe": timeframe,
        "code": code,
        "intensity": 1.0,
    }
    return _safe_deposit(SOURCE_EXCHANGE, SIG_WS_DISCONNECT, payload,
                         half_life=120.0, metadata={"exchange": exchange})


def record_data_starvation(pair: str, source: str = "tech_data",
                           reason: str = "db_fallback_only") -> bool:
    """Live data path failed and the module fell back to a stale DB row
    (observed 64x per 17h as 'No real-time tech_data'). Stress accumulates
    if this keeps firing across ticks.
    """
    payload = {
        "pair": pair,
        "source": source,
        "reason": reason,
        "intensity": 1.0,
    }
    return _safe_deposit(SOURCE_DATA, SIG_DATA_STARVATION, payload,
                         half_life=180.0, metadata={"pair": pair})


def record_shadow_divergence(pair: str, real_winrate: Optional[float],
                             shadow_winrate: float, n_shadow: int) -> bool:
    """Shadow-Kelly posterior winrate collapsed (<25% observed for 40+
    pairs in production). Half-life is long (3600s) because the signal is
    statistical — it should survive through many ticks.
    intensity is proportional to how bad the shadow winrate is, capped
    at 1.0 so a catastrophic pair does not dwarf everything else via
    the MMAS bounds.
    """
    # Intensity scales with distance from healthy (winrate ≥ 0.5) and
    # shrinks on small samples.
    gap = max(0.0, 0.50 - float(shadow_winrate))
    sample_weight = min(1.0, float(n_shadow) / 10.0)
    intensity = min(1.0, gap * 2.0 * sample_weight)
    payload = {
        "pair": pair,
        "real_winrate": real_winrate,
        "shadow_winrate": shadow_winrate,
        "n_shadow": n_shadow,
        "intensity": intensity,
    }
    return _safe_deposit(SOURCE_SHADOW, SIG_SHADOW_DIVERGENCE, payload,
                         half_life=3600.0, metadata={"pair": pair})


# ─── Efferent (field → organism) ────────────────────────────────────────────

# Per-sensor contribution cap. Empty orderbooks can get VERY loud in a
# single minute so we cap each sensor's raw contribution independently
# before summing.
_SENSOR_CAPS: Dict[Tuple[str, str], float] = {
    (SOURCE_LIQUIDITY, SIG_EMPTY_ORDERBOOK): 0.45,
    (SOURCE_EXCHANGE, SIG_WS_DISCONNECT): 0.25,
    (SOURCE_DATA, SIG_DATA_STARVATION): 0.20,
    (SOURCE_SHADOW, SIG_SHADOW_DIVERGENCE): 0.35,
}
_FIELD_TAU_MAX = 10.0  # pheromone_field.PheromoneField.TAU_MAX

# Memory pressure intentionally LIVES OUTSIDE _SENSOR_CAPS. Hormones.compute()
# weights memory more heavily (cap 0.40) than the exogenous sensor_stress
# aggregate (cap 0.30); blending memory into _SENSOR_CAPS as well would
# double-count it (audit 2026-04-25 caught this — a saturated memory trail
# was contributing 0.49 instead of the design 0.40). aggregate_memory_stress
# reads the trail directly via _MEMORY_KEY.
_MEMORY_KEY: Tuple[str, str] = (SOURCE_MEMORY, SIG_MEMORY_STRESS)


def aggregate_sensor_stress() -> float:
    """Return a [0..1] exogenous stress scalar built from the decayed
    trails of the four EXOGENOUS sensor channels (orderbook, ws, data,
    shadow). Memory pressure is its own dedicated channel — see
    aggregate_memory_stress(). Safe to call from any hot path —
    never raises, returns 0.0 when the field is unavailable or cold.

    Semantics: 0.0 = no exogenous pain, 1.0 = all sensors saturated.
    Feeds into neural_organism.Hormones.compute() as `sensor_stress`.
    """
    total = 0.0
    for (src, sig), cap in _SENSOR_CAPS.items():
        raw = _safe_read_float(src, sig)
        if raw is None or raw <= 0:
            continue
        # Normalise against the field's saturation bound so the sensor
        # reaches its cap when the trail itself reaches TAU_MAX.
        norm = min(1.0, raw / _FIELD_TAU_MAX)
        total += cap * norm
    return min(1.0, max(0.0, total))


def aggregate_memory_stress() -> float:
    """Return a [0..1] memory-only stress scalar.

    The most recent memory_sensor deposit carries `intensity ∈ [0,1]` in
    its payload; we read the LIF accumulated magnitude (already MMAS-clamped
    by pheromone_field) and normalise the same way aggregate_sensor_stress
    does — so a saturated memory trail returns ~1.0 and a quiet one 0.0.

    This is intentionally a separate channel from aggregate_sensor_stress:
    Hormones.compute() weights memory pressure more heavily (cap 0.4 vs
    sensor_stress cap 0.3) because RAM/swap saturation is an existential
    organism state, not a transient external irritation.
    """
    raw = _safe_read_float(SOURCE_MEMORY, SIG_MEMORY_STRESS)
    if raw is None or raw <= 0:
        return 0.0
    return min(1.0, max(0.0, raw / _FIELD_TAU_MAX))


def active_sensor_count() -> int:
    """Number of sensors currently carrying a non-zero pheromone trail.

    Counts all 5 channels: 4 exogenous (orderbook, ws, data, shadow)
    plus the memory homeostasis channel. NOTE: a sensor "firing" here
    means its PAIN signal is active (orderbook empty, ws dropped, etc.),
    not "the channel is healthy". Callers that map this to info_q should
    interpret high counts as DEGRADED state, not as more data feeds.
    """
    count = 0
    for (src, sig) in _SENSOR_CAPS.keys():
        raw = _safe_read_float(src, sig)
        if raw is not None and raw > 0:
            count += 1
    # Memory channel is not in _SENSOR_CAPS but is a 5th afferent.
    raw_mem = _safe_read_float(SOURCE_MEMORY, SIG_MEMORY_STRESS)
    if raw_mem is not None and raw_mem > 0:
        count += 1
    return count


def snapshot() -> Dict[str, Any]:
    """Introspection helper for the /api/ai endpoints + daily postmortem."""
    out: Dict[str, Any] = {
        "sensor_stress": aggregate_sensor_stress(),
        "memory_stress": aggregate_memory_stress(),
    }
    for (src, sig), cap in _SENSOR_CAPS.items():
        raw = _safe_read_float(src, sig)
        out[f"{src}.{sig}"] = {
            "accumulated": raw,
            "cap": cap,
            "contribution": cap * min(1.0, (raw or 0.0) / _FIELD_TAU_MAX),
        }
    # Memory channel — separate cap (0.40 in Hormones.compute), not in
    # _SENSOR_CAPS to avoid double-counting.
    raw_mem = _safe_read_float(SOURCE_MEMORY, SIG_MEMORY_STRESS)
    out[f"{SOURCE_MEMORY}.{SIG_MEMORY_STRESS}"] = {
        "accumulated": raw_mem,
        "cap": 0.40,
        "contribution": 0.40 * min(1.0, (raw_mem or 0.0) / _FIELD_TAU_MAX),
    }
    return out


# ─── Convenience: orderbook probe helper ─────────────────────────────────────

def probe_orderbook(pair: str, orderbook: Optional[Dict],
                    call_site: str = "") -> bool:
    """Inspect an orderbook dict; if either side is empty, deposit a stress
    pheromone AND drive the per-pair circuit breaker.

    Returns True when the book is bad (caller should skip exchange-bound
    work). Zero-dep wrapper so HydraSizer can drop it in everywhere
    orderbooks are read.

    Side-effect on pair_circuit (module-level import kept lazy to avoid
    circulars): healthy books reset the pair's consecutive_failures,
    bad books increment and at ≥5 they flip the pair to DORMANT for a
    backoff window.
    """
    try:
        from pair_circuit import get_pair_circuit
        _circuit = get_pair_circuit()
    except Exception:
        _circuit = None

    def _on_bad(side: str) -> None:
        record_empty_orderbook(pair, side=side, call_site=call_site)
        if _circuit is not None:
            try:
                _circuit.record_failure(pair, reason=f"empty_orderbook:{side}")
            except Exception:
                pass

    if not orderbook:
        _on_bad("both")
        return True
    bids = orderbook.get("bids") or []
    asks = orderbook.get("asks") or []
    if not bids and not asks:
        _on_bad("both")
        return True
    if not bids:
        _on_bad("bid")
        return True
    if not asks:
        _on_bad("ask")
        return True
    # Healthy book — clear the circuit so a pair that just came back to
    # life is allowed to transact again.
    if _circuit is not None:
        try:
            _circuit.record_success(pair)
        except Exception:
            pass
    return False
