"""
memory_sensor.py — Internal homeostasis afferent: RAM, swap and glibc heap
fragmentation collapsed into a single [0..1] memory_stress scalar.

The organism observed swap=2.0Gi/2.0Gi (100% saturated) for 11.5h post-deploy
while cortisol stayed at 1.0 (calm). Cortisol is INVERTED in this codebase
(1.0 = calm, lower = stressed) so a memory-saturated organism MUST drive
cortisol DOWN to trigger defensive sizing automatically — restart is no
longer the only escape valve.

Architecture (organism philosophy, not workaround):
  * record_memory_pressure(): sample → score → pheromone deposit
  * aggregate_memory_stress():  efferent reader for Hormones.compute()
  * tick(): scheduler convenience wrapper
  * Half-life 300s — memory pressure builds and recovers over minutes,
    matching gc/malloc_trim cycle cadence.
  * Score has 5 components weighted to make swap saturation the loudest
    voice (it was the ground-truth signal in production).

Failure semantics: nothing raises. psutil missing → /proc fallback. Both
missing → score 0.0 and we accept that this organ is dark.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from typing import Any, Dict, Optional, Tuple

sys.path.append(os.path.dirname(__file__))

logger = logging.getLogger(__name__)

try:
    import psutil  # type: ignore
    _PSUTIL_AVAILABLE = True
except Exception:
    _PSUTIL_AVAILABLE = False


# ─── Component thresholds ────────────────────────────────────────────────────
# Piecewise-linear mapping: value <= floor → 0.0, >= ceil → 1.0.
# Tuned against the 11.5h prod baseline (sys=85%, swap=100%, self_swap=50%).
_SYS_FLOOR, _SYS_CEIL = 0.70, 0.97
_RSS_FLOOR, _RSS_CEIL = 0.20, 0.60
_SWAP_FLOOR, _SWAP_CEIL = 0.10, 0.80
_SELF_SWAP_FLOOR, _SELF_SWAP_CEIL = 0.05, 0.50
_FRAG_FLOOR, _FRAG_CEIL = 1.5, 4.0  # vsz / rss surrogate

# Component weights (sum=1.0). Swap is the loudest voice (prod ground truth).
_W_SYS = 0.30
_W_RSS = 0.20
_W_SWAP = 0.30
_W_SELF_SWAP = 0.15
_W_FRAG = 0.05


def _read_self_smaps_swap_kb() -> int:
    """Per-process swap usage from smaps_rollup (Linux). 0 on failure."""
    try:
        with open("/proc/self/smaps_rollup", "r") as fh:
            for ln in fh:
                if ln.startswith("Swap:"):
                    return int(ln.split()[1])
    except Exception:
        return 0
    return 0


def _read_proc_meminfo() -> Dict[str, int]:
    """System-wide memory snapshot from /proc/meminfo (Linux). {} on failure."""
    out: Dict[str, int] = {}
    try:
        with open("/proc/meminfo", "r") as fh:
            for ln in fh:
                key, _, rest = ln.partition(":")
                rest = rest.strip().split()
                if rest:
                    try:
                        out[key] = int(rest[0])
                    except ValueError:
                        pass
    except Exception:
        return {}
    return out


def collect_pressure() -> Dict[str, float]:
    """Sample raw memory + swap + fragmentation signals.

    Returns scalars in [0..1] (or fragmentation ratio for frag_score).
    Keys:
      sys_used_pct   — system used RAM / total
      rss_pct        — own process RSS / total RAM
      swap_pct       — system swap used / total swap
      self_swap_pct  — own process swap pages / total swap
      frag_score     — vsz / rss (1.0 ideal, ≥4.0 severely fragmented)
    """
    snap: Dict[str, float] = {
        "rss_pct": 0.0,
        "sys_used_pct": 0.0,
        "swap_pct": 0.0,
        "self_swap_pct": 0.0,
        "frag_score": 1.0,
    }

    if _PSUTIL_AVAILABLE:
        try:
            vm = psutil.virtual_memory()
            sm = psutil.swap_memory()
            proc = psutil.Process()
            mi = proc.memory_info()
            rss = int(mi.rss)
            vsz = int(mi.vms)
            if vm.total > 0:
                snap["sys_used_pct"] = float(vm.percent) / 100.0
                snap["rss_pct"] = rss / vm.total
            if sm.total > 0:
                snap["swap_pct"] = float(sm.used) / sm.total
                self_swap_kb = _read_self_smaps_swap_kb()
                if self_swap_kb > 0:
                    snap["self_swap_pct"] = (self_swap_kb * 1024) / sm.total
            if rss > 0:
                snap["frag_score"] = vsz / rss
            return snap
        except Exception as e:
            logger.debug(f"[MemorySensor] psutil sample failed: {e}")

    # /proc fallback — Linux only, but that is our prod surface.
    mi = _read_proc_meminfo()
    total_kb = mi.get("MemTotal", 0)
    avail_kb = mi.get("MemAvailable", mi.get("MemFree", 0))
    swap_t_kb = mi.get("SwapTotal", 0)
    swap_f_kb = mi.get("SwapFree", 0)
    if total_kb > 0:
        snap["sys_used_pct"] = (total_kb - avail_kb) / total_kb
    if swap_t_kb > 0:
        snap["swap_pct"] = (swap_t_kb - swap_f_kb) / swap_t_kb
        self_swap_kb = _read_self_smaps_swap_kb()
        if self_swap_kb > 0:
            snap["self_swap_pct"] = self_swap_kb / swap_t_kb
    return snap


def _piecewise(value: float, floor: float, ceil: float) -> float:
    if value <= floor:
        return 0.0
    if value >= ceil:
        return 1.0
    return (value - floor) / max(1e-9, (ceil - floor))


def compute_pressure_score(
    snap: Optional[Dict[str, float]] = None,
) -> Tuple[float, Dict[str, float]]:
    """Combine raw signals into a [0..1] pressure score + per-component breakdown.

    Returns (score, components). Score 0.0 = no pressure, 1.0 = saturated.
    """
    if snap is None:
        snap = collect_pressure()

    sys_pct = _piecewise(snap.get("sys_used_pct", 0.0), _SYS_FLOOR, _SYS_CEIL)
    rss_pct = _piecewise(snap.get("rss_pct", 0.0), _RSS_FLOOR, _RSS_CEIL)
    swap_pct = _piecewise(snap.get("swap_pct", 0.0), _SWAP_FLOOR, _SWAP_CEIL)
    self_swap = _piecewise(snap.get("self_swap_pct", 0.0), _SELF_SWAP_FLOOR, _SELF_SWAP_CEIL)
    frag_pct = _piecewise(snap.get("frag_score", 1.0), _FRAG_FLOOR, _FRAG_CEIL)

    score = (
        _W_SYS * sys_pct
        + _W_RSS * rss_pct
        + _W_SWAP * swap_pct
        + _W_SELF_SWAP * self_swap
        + _W_FRAG * frag_pct
    )
    components = {
        "sys": round(sys_pct, 4),
        "rss": round(rss_pct, 4),
        "swap": round(swap_pct, 4),
        "self_swap": round(self_swap, 4),
        "frag": round(frag_pct, 4),
    }
    return min(1.0, max(0.0, score)), components


def record_memory_pressure(
    snap: Optional[Dict[str, float]] = None,
) -> Tuple[float, Dict[str, Any]]:
    """Sample, score, deposit a memory-stress pheromone.

    Returns (score, payload). half_life=300s mirrors the time scale of
    memory pressure recovery via gc + malloc_trim.
    """
    if snap is None:
        snap = collect_pressure()
    score, components = compute_pressure_score(snap)
    payload: Dict[str, Any] = {
        "intensity": score,
        "components": components,
        "raw": {k: round(float(v), 4) for k, v in snap.items()},
        "ts": time.time(),
    }
    try:
        from sensor_bridges import (
            SOURCE_MEMORY,
            SIG_MEMORY_STRESS,
            _safe_deposit,
        )
        _safe_deposit(
            SOURCE_MEMORY,
            SIG_MEMORY_STRESS,
            payload,
            half_life=300.0,
            metadata={"score": round(score, 4)},
        )
    except Exception as e:
        logger.debug(f"[MemorySensor] deposit failed: {e}")
    return score, payload


def tick() -> Dict[str, Any]:
    """Scheduler convenience: sample → deposit → return brief snapshot.

    Sprint 2026-05-02: also feeds PredictiveMemoryGuardian so the bot
    can FORECAST OOM rather than just react to it.
    """
    snap = collect_pressure()
    score, payload = record_memory_pressure(snap)
    # Feed forecaster — non-fatal if module missing (cold-start safety)
    try:
        from predictive_memory_guardian import record_sample, publish_forecast
        record_sample(score)
        publish_forecast()
    except Exception as e:
        logger.debug(f"[MemorySensor] guardian feed failed: {e}")
    return {
        "score": round(score, 4),
        "components": payload["components"],
        "raw": payload["raw"],
    }
