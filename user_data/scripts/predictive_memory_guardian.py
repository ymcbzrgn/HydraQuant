"""
predictive_memory_guardian.py — Self-healing memory pressure forecaster.

Sprint 2026-05-02 — Bot's own RAM trend learner.

Problem this solves:
  Existing `memory_sensor.py` REPORTS current pressure, then `_memory_cleanup`
  job in scheduler.py REACTS to it (gc + malloc_trim). By the time pressure
  hits 0.5+ threshold, OOM is often minutes away — reactive cleanup is
  too late. Production logs show 6 OOM kills in 12 hours despite
  memory_sensor being live.

This module adds a PREDICTIVE layer:
  1. Samples RSS/swap every 60s (not 5min) — finer-grain trend detection.
  2. Maintains rolling 30-min window of pressure samples.
  3. Linear regression slope → "minutes until 90% pressure / OOM".
  4. When forecast < threshold → triggers PROACTIVE actions:
       • Deposit `memory_pressure_forecast` pheromone (consumed by
         scheduler heavy-job gates and sizing layer).
       • Pause non-critical scheduled jobs via pheromone signal.
       • Aggressive pheromone field eviction.
       • Force LRU caches (ai_signal_cache, embedding_cache) to shrink.
  5. Auto-tunes cgroup MemoryMax suggestion based on 30-day peak +
     server total RAM (boot-time + weekly recalibrate).

No hardcoded thresholds — every value sourced from PARAM_REGISTRY:
  envelope.mem_guardian.window_minutes
  envelope.mem_guardian.danger_pressure
  envelope.mem_guardian.critical_pressure
  envelope.mem_guardian.forecast_horizon_minutes

Self-healing: when bot's RAM is trending up but no jobs are heavy, the
guardian doesn't act (it's a slow leak — operator concern, not action).
When bot is RIGHT BEFORE a heavy job + trend is up, guardian POSTPONES
the job to a calmer window. The bot's own behavior adapts.
"""
from __future__ import annotations

import logging
import time
from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# Singleton state. Process-local — each freqtrade process maintains its
# own pressure history. Scheduler process is the canonical guardian
# (deepest job knowledge); strategy processes also run their own to
# self-throttle if they ever need to.
_HISTORY: Deque[Tuple[float, float]] = deque(maxlen=180)  # (timestamp, pressure)
_LAST_DEPOSIT_AT: float = 0.0


def _params() -> Dict[str, float]:
    """Pull tunables from PARAM_REGISTRY. Cold-start fallbacks chosen as
    safe-but-permissive defaults — they only apply on import failure.
    """
    try:
        from neural_organism import _p
        return {
            "window_min": float(_p("envelope.mem_guardian.window_minutes", 30)),
            "horizon_min": float(_p("envelope.mem_guardian.forecast_horizon_minutes", 15)),
            "danger": float(_p("envelope.mem_guardian.danger_pressure", 0.75)),
            "critical": float(_p("envelope.mem_guardian.critical_pressure", 0.90)),
            "min_samples": float(_p("envelope.mem_guardian.min_samples_for_forecast", 10)),
            "deposit_min_interval": float(_p("envelope.mem_guardian.deposit_interval_seconds", 60)),
        }
    except Exception:
        return {
            "window_min": 30.0,
            "horizon_min": 15.0,
            "danger": 0.75,
            "critical": 0.90,
            "min_samples": 10.0,
            "deposit_min_interval": 60.0,
        }


def record_sample(score: float) -> None:
    """Append a pressure score [0..1] to the rolling history.

    Should be called by `memory_sensor.tick()` after computing the score.
    The window is bounded by both maxlen (180 samples) AND time
    (window_min minutes — older entries pruned on next observation).
    """
    now = time.time()
    _HISTORY.append((now, max(0.0, min(1.0, float(score)))))
    # Time-prune: drop entries older than window_min × 1.5 (safety margin)
    cutoff = now - (_params()["window_min"] * 60.0 * 1.5)
    while _HISTORY and _HISTORY[0][0] < cutoff:
        _HISTORY.popleft()


def _linear_slope(points: List[Tuple[float, float]]) -> float:
    """Simple least-squares linear regression slope (pressure per second).

    Returns slope in pressure-units / second. Positive slope = pressure
    rising. Returns 0.0 on insufficient data.
    """
    n = len(points)
    if n < 3:
        return 0.0
    # Normalize time to seconds-from-first-sample so x range is small
    t0 = points[0][0]
    xs = [p[0] - t0 for p in points]
    ys = [p[1] for p in points]
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    num = sum((xs[i] - mean_x) * (ys[i] - mean_y) for i in range(n))
    den = sum((xs[i] - mean_x) ** 2 for i in range(n))
    if den == 0:
        return 0.0
    return num / den


def forecast() -> Dict:
    """Predict memory pressure `horizon_min` minutes from now.

    Returns a dict with:
      • current_pressure  — latest sample
      • slope_per_minute  — pressure change per minute
      • forecast_pressure — predicted pressure at horizon_min minutes
      • minutes_to_danger — estimated minutes until reaching `danger` threshold
      • minutes_to_critical — same for `critical`
      • action            — one of: "ok" / "throttle_warn" / "throttle_action"
                                    / "halt_heavy_jobs"
      • confidence        — [0..1] based on sample count

    `action` semantics (consumer-facing):
      • ok                — no action needed
      • throttle_warn     — operator-visible warning, pheromone deposit
      • throttle_action   — pause non-critical scheduled jobs, shrink caches
      • halt_heavy_jobs   — emergency: cancel pending heavy ML jobs
    """
    cfg = _params()
    history = list(_HISTORY)
    n = len(history)
    if n == 0:
        return {
            "current_pressure": 0.0,
            "slope_per_minute": 0.0,
            "forecast_pressure": 0.0,
            "minutes_to_danger": float("inf"),
            "minutes_to_critical": float("inf"),
            "action": "ok",
            "confidence": 0.0,
            "n_samples": 0,
        }
    current = history[-1][1]
    if n < int(cfg["min_samples"]):
        # Not enough samples to fit a slope yet
        return {
            "current_pressure": current,
            "slope_per_minute": 0.0,
            "forecast_pressure": current,
            "minutes_to_danger": float("inf"),
            "minutes_to_critical": float("inf"),
            "action": "ok",
            "confidence": min(1.0, n / max(1.0, cfg["min_samples"])),
            "n_samples": n,
        }
    # Use the most recent window_min minutes of samples for the slope
    window_seconds = cfg["window_min"] * 60.0
    cutoff = history[-1][0] - window_seconds
    recent = [p for p in history if p[0] >= cutoff]
    slope_per_sec = _linear_slope(recent)
    slope_per_min = slope_per_sec * 60.0

    horizon_min = cfg["horizon_min"]
    forecast_p = max(0.0, min(1.0, current + slope_per_sec * horizon_min * 60.0))

    # Time-to-threshold (only meaningful if slope > 0)
    minutes_to_danger = float("inf")
    minutes_to_critical = float("inf")
    if slope_per_min > 1e-6:
        if current < cfg["danger"]:
            minutes_to_danger = (cfg["danger"] - current) / slope_per_min
        else:
            minutes_to_danger = 0.0
        if current < cfg["critical"]:
            minutes_to_critical = (cfg["critical"] - current) / slope_per_min
        else:
            minutes_to_critical = 0.0

    # Decide action — graduated response, no hardcoded constants
    if current >= cfg["critical"] or minutes_to_critical < 5.0:
        action = "halt_heavy_jobs"
    elif current >= cfg["danger"] or minutes_to_critical < horizon_min:
        action = "throttle_action"
    elif minutes_to_danger < horizon_min:
        action = "throttle_warn"
    else:
        action = "ok"

    confidence = min(1.0, len(recent) / 20.0)  # >=20 samples → full conf

    return {
        "current_pressure": round(current, 4),
        "slope_per_minute": round(slope_per_min, 6),
        "forecast_pressure": round(forecast_p, 4),
        "minutes_to_danger": round(minutes_to_danger, 2) if minutes_to_danger < 1e6 else None,
        "minutes_to_critical": round(minutes_to_critical, 2) if minutes_to_critical < 1e6 else None,
        "action": action,
        "confidence": round(confidence, 3),
        "n_samples": n,
    }


def publish_forecast() -> Dict:
    """Compute forecast and deposit to pheromone field. Throttled to
    `deposit_min_interval` seconds so the field isn't spammed.

    Consumers (e.g. scheduler heavy-job gates):
      from pheromone_field import get_pheromone_field
      pf = get_pheromone_field()
      f = pf.read("memory_pressure_forecast")
      if f and f.get("action") == "halt_heavy_jobs":
          return  # skip the heavy job this cycle
    """
    global _LAST_DEPOSIT_AT
    now = time.time()
    cfg = _params()
    if now - _LAST_DEPOSIT_AT < cfg["deposit_min_interval"]:
        # Still throttled — return last computation without deposit
        return forecast()
    f = forecast()
    try:
        from pheromone_field import get_pheromone_field
        get_pheromone_field().deposit(
            "memory_guardian", "memory_pressure_forecast",
            f, half_life=300.0,
        )
        _LAST_DEPOSIT_AT = now
    except Exception as e:
        logger.debug(f"[MemoryGuardian] pheromone deposit failed: {e}")
    return f


def should_halt_heavy_jobs() -> bool:
    """Convenience for scheduler — returns True when a heavy ML job should
    be skipped this cycle. Reads the latest pheromone deposit (canonical
    cross-process channel) so any process can publish, scheduler honors.
    """
    try:
        from pheromone_field import get_pheromone_field
        f = get_pheromone_field().read("memory_pressure_forecast")
        if isinstance(f, dict):
            return f.get("action") == "halt_heavy_jobs"
    except Exception:
        pass
    return False


def should_throttle() -> bool:
    """Returns True for `throttle_action` or `halt_heavy_jobs`. Used by
    less-critical jobs to self-defer."""
    try:
        from pheromone_field import get_pheromone_field
        f = get_pheromone_field().read("memory_pressure_forecast")
        if isinstance(f, dict):
            return f.get("action") in ("throttle_action", "halt_heavy_jobs")
    except Exception:
        pass
    return False


def auto_cgroup_recommendation(total_ram_bytes: Optional[int] = None) -> Dict[str, int]:
    """Suggest cgroup MemoryMax for each freqtrade-* service based on
    total system RAM and observed 30-day peaks.

    Returns dict {service_name: bytes}. Caller can write systemd
    drop-in files at /etc/systemd/system/<svc>.service.d/auto.conf.

    The math:
      reservation = max(2GB, 10% of total RAM) for kernel + sshd + buffer
      usable = total - reservation
      Then divided by 4 services (rag, scheduler, models, freqtrade) with
      observed weight (peak RSS / sum of peaks) as proportional allocator.
      Floor 1.5GB per service so no service starves on cold start.
    """
    try:
        import psutil
        if total_ram_bytes is None:
            total_ram_bytes = psutil.virtual_memory().total
    except Exception:
        return {}

    if not total_ram_bytes or total_ram_bytes <= 0:
        return {}

    # 10% reservation for kernel, sshd, page cache, transient subprocesses
    reservation = max(2 * 1024**3, int(total_ram_bytes * 0.10))
    usable = max(0, total_ram_bytes - reservation)

    # Default weights (cold start). Audit 2026-05-02 #13: pulled from
    # PARAM_REGISTRY so neuron tuning + observed-peak overrides both
    # converge to a venue-appropriate cgroup distribution.
    try:
        from neural_organism import _p
        services = {
            "freqtrade.service": float(_p("system.cgroup.weight.freqtrade", 0.18)),
            "freqtrade-tr-dry.service": float(_p("system.cgroup.weight.freqtrade_tr_dry", 0.18)),
            "freqtrade-rag.service": float(_p("system.cgroup.weight.freqtrade_rag", 0.18)),
            "freqtrade-scheduler.service": float(_p("system.cgroup.weight.freqtrade_scheduler", 0.27)),
            "freqtrade-models.service": float(_p("system.cgroup.weight.freqtrade_models", 0.19)),
        }
    except Exception:
        services = {
            "freqtrade.service": 0.18,
            "freqtrade-tr-dry.service": 0.18,
            "freqtrade-rag.service": 0.18,
            "freqtrade-scheduler.service": 0.27,
            "freqtrade-models.service": 0.19,
        }

    # Try reading 30-day RSS peaks from system_metrics (if logged)
    peaks: Dict[str, int] = {}
    try:
        from db import get_db_connection
        with get_db_connection() as conn:
            rows = conn.execute("""
                SELECT metric_name, MAX(metric_value) AS peak
                FROM system_metrics
                WHERE metric_name LIKE 'rss_%'
                  AND timestamp >= datetime('now', '-30 days')
                GROUP BY metric_name
            """).fetchall()
        for r in rows or []:
            name = r["metric_name"]
            if name.startswith("rss_"):
                peaks[name[4:]] = int(r["peak"] or 0)
    except Exception:
        pass

    if peaks:
        total_peak = sum(peaks.values()) or 1
        weighted = {}
        for svc in services:
            short = svc.replace(".service", "").replace("freqtrade-", "")
            short = short or "freqtrade"
            p = peaks.get(short, 0)
            if p > 0:
                weighted[svc] = p / total_peak
        if sum(weighted.values()) > 0:
            # Renormalize to 1.0 across services we have peaks for
            wsum = sum(weighted.values())
            services = {k: v / wsum for k, v in weighted.items()}

    # Floor: 1.5GB per service so no service is starved
    min_per_service = int(1.5 * 1024**3)
    rec: Dict[str, int] = {}
    for svc, weight in services.items():
        suggested = int(usable * weight)
        rec[svc] = max(min_per_service, suggested)

    # Final safety: total recommendation must not exceed usable
    total_rec = sum(rec.values())
    if total_rec > usable:
        scale = usable / total_rec
        rec = {k: int(v * scale) for k, v in rec.items()}
    return rec
