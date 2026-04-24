"""
autonomous_lifecycle.py — Phase 26 Sprint 2, Task 9C

13-Layer Autonomous Organism Lifecycle — Zero human intervention.

After deployment, the organism:
  - Learns from every trade (Active Inference)
  - Grows new connections when needed (Morphogenesis)
  - Maintains criticality balance (HAG)
  - Coordinates via pheromones (Stigmergy — already exists, extended)
  - Predicts failures before they happen (Predictive Coding)
  - Detects threats (Danger Theory Immunity)
  - Gets stronger from stress (Hormesis)
  - Allocates compute resources (Dual-Brain Market)
  - Maintains identity across changes (Autopoiesis)
  - Safely deploys changes (Canary Deployment)
  - Follows circadian rhythms (Sirkadyen)
  - Enables collective intelligence (Kolektif Zeka)
  - Orchestrates all layers (Meta-Controller)

Integration:
  - Reads from: neural_organism, pheromone_field, self_model, causal_engine
  - Writes to: pheromone_field, organism_audit, hormone_history
  - Called by scheduler (hourly lifecycle tick)

Reference: Free Energy Principle (Friston), Active Inference (MDPI 2026)
"""

import os
import sys
import json
import logging
import math
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
logger = logging.getLogger("autonomous_lifecycle")

from ai_config import AI_DB_PATH
from db import get_db_connection, get_connection, execute_with_retry, init_db

# ===========================================================================
# Layer Constants
# ===========================================================================

# Active Inference
EFE_PRAGMATIC_WEIGHT = 0.6   # Exploitation weight
EFE_EPISTEMIC_WEIGHT = 0.4   # Exploration weight

# Morphogenesis
COACTIVATION_THRESHOLD = 0.7  # Correlation threshold for new synapse
MIN_COACTIVATION_SAMPLES = 10

# Criticality (HAG)
CRITICALITY_TARGET = 0.5      # Edge of chaos — not too ordered, not too chaotic
CRITICALITY_TOLERANCE = 0.15

# Hormesis
HORMESIS_MILD_STRESS = 0.3    # Mild stress → adaptation
HORMESIS_TOXIC_STRESS = 0.8   # Too much stress → damage

# Circadian
AGGRESSIVE_HOURS = [8, 9, 10, 11, 14, 15, 16, 17]  # UTC: US+EU market overlap
CONSERVATIVE_HOURS = [0, 1, 2, 3, 4, 5]              # Low volume hours
SLEEP_HOURS = [22, 23]                                 # Maintenance window


class ActiveInferenceCore:
    """Layer 1: Expected Free Energy (EFE) decision framework.

    Replaces pure reward maximization with surprise minimization.
    EFE = pragmatic_value + epistemic_value
    """

    def compute_efe(self, action_candidates: List[Dict],
                    current_state: Dict) -> List[Tuple[Dict, float]]:
        """Rank actions by Expected Free Energy.

        Each candidate: {"action": np.array, "expected_pnl": float, "uncertainty": float}
        Returns: [(action, efe_score)] sorted best-first.
        """
        scored = []
        for candidate in action_candidates:
            # Pragmatic value: expected profit
            pragmatic = candidate.get("expected_pnl", 0) * EFE_PRAGMATIC_WEIGHT

            # Epistemic value: information gain (high uncertainty = high value)
            uncertainty = candidate.get("uncertainty", 0.5)
            epistemic = uncertainty * EFE_EPISTEMIC_WEIGHT

            efe = pragmatic + epistemic
            scored.append((candidate, efe))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored


class MorphogenesisEngine:
    """Layer 2: Developmental Morphogenesis — Architecture grows itself.

    When two modules are consistently co-activated and produce good outcomes,
    create a new synapse between them. When a synapse is never used, prune it.
    """

    def __init__(self):
        self._coactivation_log: Dict[Tuple[str, str], List[float]] = {}

    def log_coactivation(self, module_a: str, module_b: str, outcome_pnl: float):
        """Record that two modules were co-active during a trade."""
        key = tuple(sorted([module_a, module_b]))
        if key not in self._coactivation_log:
            self._coactivation_log[key] = []
        self._coactivation_log[key].append(outcome_pnl)

        # Limit history
        if len(self._coactivation_log[key]) > 100:
            self._coactivation_log[key] = self._coactivation_log[key][-100:]

    def suggest_new_synapses(self) -> List[Dict]:
        """Suggest new synapses based on co-activation patterns."""
        suggestions = []
        for (mod_a, mod_b), outcomes in self._coactivation_log.items():
            if len(outcomes) < MIN_COACTIVATION_SAMPLES:
                continue

            win_rate = sum(1 for o in outcomes if o > 0) / len(outcomes)
            avg_pnl = np.mean(outcomes)

            if win_rate > COACTIVATION_THRESHOLD and avg_pnl > 0:
                suggestions.append({
                    "source": mod_a,
                    "target": mod_b,
                    "strength": round(win_rate, 3),
                    "avg_pnl": round(float(avg_pnl), 4),
                    "n_samples": len(outcomes),
                    "reason": "high co-activation win rate",
                })

        return suggestions

    def suggest_pruning(self) -> List[Dict]:
        """Suggest synapses to prune (consistently negative co-activation)."""
        prune = []
        for (mod_a, mod_b), outcomes in self._coactivation_log.items():
            if len(outcomes) < MIN_COACTIVATION_SAMPLES:
                continue

            win_rate = sum(1 for o in outcomes if o > 0) / len(outcomes)
            if win_rate < 0.25:
                prune.append({
                    "source": mod_a, "target": mod_b,
                    "win_rate": round(win_rate, 3),
                    "reason": "consistently negative co-activation",
                })

        return prune


class CriticalityMonitor:
    """Layer 3: Self-Organized Criticality — Balance order and chaos.

    The organism must operate at the "edge of chaos":
    - Too ordered → can't adapt to new patterns
    - Too chaotic → unstable, oscillating parameters
    """

    def compute_criticality(self, param_changes: List[float]) -> float:
        """Compute criticality measure from recent parameter changes.

        Returns 0.0 (frozen) to 1.0 (chaotic).
        Optimal: ~0.5 (edge of chaos).
        """
        if not param_changes or len(param_changes) < 5:
            return 0.5  # Default: balanced

        changes = np.array(param_changes)
        variance = float(np.var(changes))
        max_change = float(np.max(np.abs(changes)))

        # Normalize: low variance = ordered, high variance = chaotic
        # Use log scale for better resolution
        log_var = np.log1p(variance * 100)  # Scale up small variances
        criticality = float(np.clip(log_var / 5.0, 0.0, 1.0))

        return round(criticality, 3)

    def get_adjustment(self, criticality: float) -> Dict[str, float]:
        """Suggest adjustments to restore criticality balance."""
        if abs(criticality - CRITICALITY_TARGET) <= CRITICALITY_TOLERANCE:
            return {"action": "none", "learning_rate_mult": 1.0}

        if criticality < CRITICALITY_TARGET - CRITICALITY_TOLERANCE:
            # Too ordered → increase exploration
            return {
                "action": "increase_exploration",
                "learning_rate_mult": 1.3,
                "noise_injection": 0.05,
            }
        else:
            # Too chaotic → stabilize
            return {
                "action": "stabilize",
                "learning_rate_mult": 0.7,
                "noise_injection": 0.0,
            }


class DangerTheoryImmunity:
    """Layer 6: Danger Theory — Smart threat detection.

    Unlike simple anomaly detection, danger theory distinguishes:
    - Danger signals (genuine threats): flash crash, liquidity crisis
    - Non-danger anomalies (opportunities): unusual volume, regime shift
    """

    def __init__(self):
        self._threat_history: List[Dict] = []
        self._false_alarm_count = 0
        self._true_alarm_count = 0

    def assess_danger(self, market_state: Dict) -> Dict:
        """Assess current danger level.

        Returns {"danger_level": 0-1, "threat_type": str, "response": str}
        """
        danger_signals = []
        danger_level = 0.0

        # Signal 1: Extreme drawdown
        drawdown = market_state.get("drawdown", 0)
        if drawdown > 0.15:
            danger_signals.append("high_drawdown")
            danger_level += 0.3

        # Signal 2: Cortisol spike (stress indicator)
        cortisol = market_state.get("cortisol", 0.5)
        if cortisol > 0.8:
            danger_signals.append("stress_spike")
            danger_level += 0.2

        # Signal 3: Multiple consecutive losses
        consecutive_losses = market_state.get("consecutive_losses", 0)
        if consecutive_losses >= 4:
            danger_signals.append("loss_streak")
            danger_level += 0.25

        # Signal 4: OOD detection firing
        ood_active = market_state.get("ood_distance", 0) > 100
        if ood_active:
            danger_signals.append("ood_market")
            danger_level += 0.15

        # Signal 5: Extreme FNG
        fng = market_state.get("fng_value", 50)
        if fng < 10 or fng > 90:
            danger_signals.append("extreme_sentiment")
            danger_level += 0.1

        danger_level = min(danger_level, 1.0)

        # Determine response
        if danger_level > 0.7:
            response = "DEFENSIVE"  # Reduce all positions, tighten stoploss
        elif danger_level > 0.4:
            response = "CAUTIOUS"   # Reduce sizing, skip low-confidence trades
        else:
            response = "NORMAL"

        return {
            "danger_level": round(danger_level, 3),
            "signals": danger_signals,
            "response": response,
            "n_signals": len(danger_signals),
        }

    def record_outcome(self, was_danger: bool, actual_loss: bool):
        """Record whether danger assessment was correct."""
        if was_danger and actual_loss:
            self._true_alarm_count += 1
        elif was_danger and not actual_loss:
            self._false_alarm_count += 1

    def get_precision(self) -> float:
        """Danger detection precision."""
        total = self._true_alarm_count + self._false_alarm_count
        if total == 0:
            return 0.5
        return self._true_alarm_count / total


class HormesisEngine:
    """Layer 7: Antifragile Hormesis — Get stronger from controlled stress."""

    def compute_adaptation(self, stress_level: float,
                           recent_performance: List[float]) -> Dict:
        """Compute adaptation response to stress.

        Mild stress → grow stronger (hormesis)
        Toxic stress → protect (damage control)
        """
        if stress_level < HORMESIS_MILD_STRESS:
            return {
                "response": "comfortable",
                "adaptation_rate": 1.0,
                "message": "normal operating conditions",
            }
        elif stress_level < HORMESIS_TOXIC_STRESS:
            # Hormesis zone — beneficial stress
            adaptation = 1.0 + (stress_level - HORMESIS_MILD_STRESS) * 0.5
            return {
                "response": "hormesis",
                "adaptation_rate": round(adaptation, 3),
                "message": "beneficial stress — increasing learning rate",
            }
        else:
            # Toxic stress — protect
            return {
                "response": "toxic",
                "adaptation_rate": 0.5,
                "message": "excessive stress — reducing exposure",
            }


class CircadianRhythm:
    """Layer 11: Circadian Rhythm — Time-based behavior modulation.

    Phase 27 Fix 10 (I4): the per-hour multiplier is now sourced from
    `cerebellum_timing.get_timing_multiplier(hour)` so it is LEARNED from
    live 24-slot performance instead of the old hardcoded AGGRESSIVE/
    CONSERVATIVE/SLEEP buckets. When cerebellum is unavailable (cold start,
    not enough trades) we fall back to the legacy table below.

    Phase 27 Task 19 (I4): Borbely Two-Process Model overlay for the
    sleep-wake cycle.

      Process S (homeostatic): compute fatigue accumulates while awake.
        S(t) = S_max · (1 − exp(−t_wake / τ_rise))
      Process C (circadian):   market opportunity rhythm (learned C-factor
        from cerebellum hourly multiplier, default sinusoid).
      Mode trigger:
        S(t) − C(t) > +δ_upper  → advance SLEEP (savings mode)
        S(t) − C(t) < −δ_lower  → advance WAKE  (opportunity mode)
      EMERGENCY_WAKE: 3σ+ volatility shock instantly forces FULL_WAKE.
    """

    # Three operational modes — scheduler uses these to gate non-critical jobs.
    MODE_FULL_WAKE = "full_wake"      # 52 jobs active
    MODE_LIGHT_SLEEP = "light_sleep"  # ~18 jobs (heavy learning paused)
    MODE_DEEP_SLEEP = "deep_sleep"    # ~6 jobs (only safety + heartbeat)

    # Borbely constants — S rises toward S_MAX with τ_rise, decays with τ_fall.
    S_MAX = 1.0
    TAU_RISE = 16.0 * 3600.0   # 16h awake → 63% saturation
    TAU_FALL = 6.0 * 3600.0    # 6h sleep → 63% recovery
    DELTA_UPPER = 0.15
    DELTA_LOWER = 0.10
    VOL_EMERGENCY_SIGMA = 3.0

    def __init__(self):
        # Process S state (homeostatic pressure)
        self._process_s: float = 0.0
        self._last_tick_ts: Optional[float] = None
        self._current_mode: str = self.MODE_FULL_WAKE
        self._emergency_wake_until: float = 0.0

    def _process_s_update(self, now_ts: float, mode_is_wake: bool) -> float:
        """Evolve Process S: rises while awake, falls while asleep."""
        import math as _math
        if self._last_tick_ts is None:
            self._last_tick_ts = now_ts
            return self._process_s
        dt = max(0.0, now_ts - self._last_tick_ts)
        if mode_is_wake:
            self._process_s = self.S_MAX - (self.S_MAX - self._process_s) * _math.exp(-dt / self.TAU_RISE)
        else:
            self._process_s = self._process_s * _math.exp(-dt / self.TAU_FALL)
        self._last_tick_ts = now_ts
        return self._process_s

    def _process_c(self, hour_utc: int) -> float:
        """Circadian opportunity — learned from cerebellum when possible, else
        sinusoidal day/night approximation centred on UTC 14:00 (US open)."""
        import math as _math
        learned = self._cerebellum_multiplier(hour_utc)
        if learned is not None:
            # Map learned multiplier ∈ [0.3, 1.5] into [0, 1]
            return max(0.0, min(1.0, (learned - 0.3) / 1.2))
        # Fallback: sinusoidal peak at 14:00 UTC, trough at 02:00 UTC
        phase = 2.0 * _math.pi * ((hour_utc - 14.0) / 24.0)
        return 0.5 + 0.5 * _math.cos(phase)

    def evaluate_mode(self,
                      hour_utc: int,
                      recent_volatility_sigma: float = 0.0) -> str:
        """Phase 27 Task 19: pick FULL / LIGHT / DEEP sleep mode.

        Called by the scheduler tick (hourly is enough — Process S drifts
        slowly). `recent_volatility_sigma` = zscore of recent 1h returns against
        their 7d std; >3σ triggers EMERGENCY_WAKE regardless of Borbely state.
        """
        import time as _time
        now = _time.monotonic()

        # Emergency override — short-lived but hard-overrides cycle.
        if recent_volatility_sigma >= self.VOL_EMERGENCY_SIGMA:
            self._emergency_wake_until = now + 3600.0  # 1h forced wake
            logger.warning(
                f"[Circadian] EMERGENCY_WAKE — volatility {recent_volatility_sigma:.1f}σ "
                "forces FULL_WAKE for 1h"
            )
        if now < self._emergency_wake_until:
            self._current_mode = self.MODE_FULL_WAKE
            self._process_s_update(now, mode_is_wake=True)
            return self._current_mode

        mode_is_wake = self._current_mode == self.MODE_FULL_WAKE
        s = self._process_s_update(now, mode_is_wake=mode_is_wake)
        c = self._process_c(hour_utc)

        # Decision rule (Borbely): the delta S − C gates the transition.
        delta = s - c
        if delta > self.DELTA_UPPER:
            # Homeostatic pressure exceeds opportunity → go down.
            if self._current_mode == self.MODE_FULL_WAKE:
                self._current_mode = self.MODE_LIGHT_SLEEP
            elif self._current_mode == self.MODE_LIGHT_SLEEP and s > 0.8:
                self._current_mode = self.MODE_DEEP_SLEEP
        elif delta < -self.DELTA_LOWER:
            # Opportunity beats fatigue → come up.
            if self._current_mode == self.MODE_DEEP_SLEEP:
                self._current_mode = self.MODE_LIGHT_SLEEP
            elif self._current_mode == self.MODE_LIGHT_SLEEP and s < 0.3:
                self._current_mode = self.MODE_FULL_WAKE
        return self._current_mode

    def get_mode(self) -> str:
        return self._current_mode

    def state_snapshot(self) -> Dict:
        """Diagnostic snapshot for logs / dashboards."""
        import time as _time
        return {
            "mode": self._current_mode,
            "process_s": round(self._process_s, 3),
            "emergency_wake_remaining_s": max(0.0, self._emergency_wake_until - _time.monotonic()),
        }

    def _cerebellum_multiplier(self, hour_utc: int) -> Optional[float]:
        """Read learned timing multiplier from the cerebellum, or None."""
        try:
            from cerebellum_timing import get_cerebellum
        except Exception:
            return None
        try:
            cereb = get_cerebellum()
            # Try the exact-hour signature first; fall back to the arg-less one
            # some earlier Sprint 2 revisions exposed.
            try:
                mult = cereb.get_timing_multiplier(hour_utc)
            except TypeError:
                mult = cereb.get_timing_multiplier()
            if mult is None:
                return None
            return float(mult)
        except Exception:
            return None

    def get_time_modulation(self, hour_utc: int) -> Dict:
        """Get behavior modulation based on time of day."""
        # Phase 27 Fix 10: prefer learned timing, fall back to hardcoded
        learned = self._cerebellum_multiplier(hour_utc)
        if learned is not None:
            if learned >= 1.10:
                mode = "aggressive"
                description = f"learned peak (cerebellum mult={learned:.2f})"
                skip_low = False
                maintenance = False
            elif learned <= 0.50:
                mode = "maintenance"
                description = f"learned low-value window (cerebellum mult={learned:.2f})"
                skip_low = True
                maintenance = True
            elif learned <= 0.85:
                mode = "conservative"
                description = f"learned low volume (cerebellum mult={learned:.2f})"
                skip_low = True
                maintenance = False
            else:
                mode = "normal"
                description = f"learned neutral (cerebellum mult={learned:.2f})"
                skip_low = False
                maintenance = False
            return {
                "mode": mode,
                "sizing_mult": max(0.2, min(1.5, learned)),
                "skip_low_confidence": skip_low,
                "run_maintenance": maintenance,
                "description": description,
                "source": "cerebellum",
            }

        # Legacy fallback — cerebellum unavailable / insufficient history.
        if hour_utc in SLEEP_HOURS:
            return {
                "mode": "maintenance",
                "sizing_mult": 0.3,
                "skip_low_confidence": True,
                "run_maintenance": True,
                "description": "maintenance window — minimal trading",
                "source": "legacy_hardcoded",
            }
        elif hour_utc in AGGRESSIVE_HOURS:
            return {
                "mode": "aggressive",
                "sizing_mult": 1.2,
                "skip_low_confidence": False,
                "run_maintenance": False,
                "description": "peak hours — US+EU market overlap",
                "source": "legacy_hardcoded",
            }
        elif hour_utc in CONSERVATIVE_HOURS:
            return {
                "mode": "conservative",
                "sizing_mult": 0.7,
                "skip_low_confidence": True,
                "run_maintenance": False,
                "description": "low volume hours — reduced exposure",
                "source": "legacy_hardcoded",
            }
        else:
            return {
                "mode": "normal",
                "sizing_mult": 1.0,
                "skip_low_confidence": False,
                "run_maintenance": False,
                "description": "standard operating hours",
                "source": "legacy_hardcoded",
            }


class AutopoiesisIdentity:
    """Layer 9: Autopoietic Identity — "Am I still ME?"

    Detects if the organism has drifted too far from its core identity.
    """

    def __init__(self):
        self._identity_hash = None

    def compute_identity_hash(self, params: np.ndarray) -> str:
        """Compute a hash of the organism's current parameter state."""
        if params is None or len(params) == 0:
            return "empty"
        # Use statistical fingerprint
        fingerprint = (
            f"mean={float(params.mean()):.4f}|"
            f"std={float(params.std()):.4f}|"
            f"norm={float(np.linalg.norm(params)):.4f}|"
            f"dim={len(params)}"
        )
        return fingerprint

    def check_identity(self, current_params: np.ndarray,
                       original_params: np.ndarray) -> Dict:
        """Check if organism has drifted from its identity."""
        if original_params is None or current_params is None:
            return {"drifted": False, "distance": 0.0}

        min_len = min(len(current_params), len(original_params))
        if min_len == 0:
            return {"drifted": False, "distance": 0.0}

        diff = current_params[:min_len] - original_params[:min_len]
        distance = float(np.linalg.norm(diff)) / (min_len ** 0.5)

        # Cosine similarity
        dot = np.dot(current_params[:min_len], original_params[:min_len])
        norm_curr = np.linalg.norm(current_params[:min_len])
        norm_orig = np.linalg.norm(original_params[:min_len])
        cosine_sim = dot / (norm_curr * norm_orig + 1e-10)

        drifted = distance > 2.0 or cosine_sim < 0.7

        return {
            "drifted": drifted,
            "distance": round(distance, 4),
            "cosine_similarity": round(float(cosine_sim), 4),
            "identity_hash": self.compute_identity_hash(current_params),
        }


class MetaController:
    """Layer 13: Meta-Controller — Orchestrates all lifecycle layers."""

    def __init__(self):
        self.active_inference = ActiveInferenceCore()
        self.morphogenesis = MorphogenesisEngine()
        self.criticality = CriticalityMonitor()
        self.danger_theory = DangerTheoryImmunity()
        self.hormesis = HormesisEngine()
        # Phase 27 Task 19 audit fix: share the circadian singleton with the
        # scheduler so Process S / mode transitions are observed consistently.
        self.circadian = get_circadian()
        self.autopoiesis = AutopoiesisIdentity()
        self._tick_count = 0
        init_db()

    def lifecycle_tick(self, market_state: Dict) -> Dict:
        """Execute one lifecycle tick — called hourly by scheduler.

        Runs all applicable lifecycle layers and returns combined decisions.
        """
        self._tick_count += 1
        hour_utc = datetime.now(timezone.utc).hour
        decisions = {}

        # Layer 11: Circadian rhythm
        time_mod = self.circadian.get_time_modulation(hour_utc)
        decisions["circadian"] = time_mod

        # Layer 6: Danger assessment
        danger = self.danger_theory.assess_danger(market_state)
        decisions["danger"] = danger

        # Layer 7: Hormesis
        stress_level = market_state.get("market_stress", 0.5)
        recent_pnl = market_state.get("recent_pnl_list", [])
        hormesis = self.hormesis.compute_adaptation(stress_level, recent_pnl)
        decisions["hormesis"] = hormesis

        # Layer 3: Criticality check
        param_changes = market_state.get("param_changes", [])
        crit = self.criticality.compute_criticality(param_changes)
        crit_adj = self.criticality.get_adjustment(crit)
        decisions["criticality"] = {"value": crit, "adjustment": crit_adj}

        # Layer 1: Active Inference — rank actions by EFE if candidates available
        action_candidates = market_state.get("action_candidates", [])
        if action_candidates:
            ranked = self.active_inference.compute_efe(action_candidates, market_state)
            decisions["active_inference"] = {
                "top_action": ranked[0][0] if ranked else None,
                "top_efe": ranked[0][1] if ranked else 0,
            }

        # Layer 2: Morphogenesis — check for new synapse suggestions
        modules_active = market_state.get("modules_active", [])
        pnl = market_state.get("last_pnl", 0)
        if len(modules_active) >= 2 and pnl != 0:
            for i in range(len(modules_active)):
                for j in range(i + 1, len(modules_active)):
                    self.morphogenesis.log_coactivation(
                        modules_active[i], modules_active[j], pnl
                    )
        new_synapses = self.morphogenesis.suggest_new_synapses()
        prune_synapses = self.morphogenesis.suggest_pruning()
        decisions["morphogenesis"] = {
            "new_synapse_suggestions": len(new_synapses),
            "prune_suggestions": len(prune_synapses),
        }

        # Layer 9: Autopoiesis — identity drift check (weekly, not every tick)
        if self._tick_count % 168 == 0:  # Every ~7 days (168 hours)
            current_params = market_state.get("organism_params")
            original_params = market_state.get("original_params")
            if current_params is not None and original_params is not None:
                identity = self.autopoiesis.check_identity(
                    np.array(current_params), np.array(original_params)
                )
                decisions["autopoiesis"] = identity
                if identity.get("drifted"):
                    logger.warning(f"[Lifecycle] Identity drift detected! "
                                 f"distance={identity['distance']}, "
                                 f"cos_sim={identity['cosine_similarity']}")

        # Compute final sizing multiplier from all layers
        sizing_mult = 1.0
        sizing_mult *= time_mod.get("sizing_mult", 1.0)

        if danger["response"] == "DEFENSIVE":
            sizing_mult *= 0.3
        elif danger["response"] == "CAUTIOUS":
            sizing_mult *= 0.6

        if hormesis["response"] == "toxic":
            sizing_mult *= hormesis["adaptation_rate"]

        sizing_mult *= crit_adj.get("learning_rate_mult", 1.0)

        decisions["final_sizing_mult"] = round(sizing_mult, 3)
        decisions["tick"] = self._tick_count
        decisions["hour_utc"] = hour_utc

        # Persist decisions
        self._persist_tick(decisions)

        # Deposit to pheromone for strategy consumption
        self._publish_decisions(decisions)

        return decisions

    def _persist_tick(self, decisions: Dict):
        """Log lifecycle tick to hormone_history using CANONICAL hormone
        semantics (cortisol=1.0 calm, lower stressed — neural_organism.py:620-628)
        and the real organism_health fraction [0..1].

        The retired writer stored `danger_level` (0=calm, 1=panic) straight
        into `cortisol` and dumped the `final_sizing_mult` (range ~0.5..1.3)
        into `organism_health`. Two writers with opposing conventions on the
        same log turned every downstream consumer into a lie. The canonical
        inputs are owned by the NeuralOrganism singleton — we read them, and
        park the lifecycle-specific telemetry in its own columns.
        """
        try:
            # Lazy import to dodge the circular: autonomous_lifecycle is imported
            # by scheduler long before neural_organism's module-level state exists.
            try:
                from neural_organism import get_organism
                org = get_organism()
                h = org.hormones
                org_health = org.get_organism_health()  # clamped [0..1]
                cortisol = float(h.cortisol)
                dopamine = float(h.dopamine)
                serotonin = float(h.serotonin)
                adrenaline = float(h.adrenaline)
                market_stress = float(getattr(h, "_stress", 0.0))
            except Exception:
                # Organism not yet booted — record a neutral "I don't know"
                # row under canonical convention (1.0 = calm).
                cortisol = 1.0
                dopamine = 1.0
                serotonin = 1.0
                adrenaline = 1.0
                org_health = 0.5
                market_stress = 0.0

            danger_level = float(decisions.get("danger", {}).get("danger_level", 0.0))
            sizing_mult = float(decisions.get("final_sizing_mult", 1.0))

            execute_with_retry(
                """INSERT INTO hormone_history
                   (cortisol, dopamine, serotonin, adrenaline,
                    market_stress, organism_health, trigger_event,
                    sizing_mult, danger_level)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    cortisol,
                    dopamine,
                    serotonin,
                    adrenaline,
                    market_stress,
                    org_health,
                    f"lifecycle_tick_{decisions.get('tick', 0)}",
                    sizing_mult,
                    danger_level,
                ),
                max_retries=3,
            )
        except Exception as e:
            logger.debug(f"[Lifecycle] Failed to persist tick: {e}")

    def _publish_decisions(self, decisions: Dict):
        """Publish lifecycle decisions to pheromone field."""
        try:
            from pheromone_field import get_pheromone_field
            pfield = get_pheromone_field()
            pfield.deposit("lifecycle", "lifecycle_state", {
                "sizing_mult": decisions["final_sizing_mult"],
                "danger_level": decisions.get("danger", {}).get("danger_level", 0),
                "danger_response": decisions.get("danger", {}).get("response", "NORMAL"),
                "circadian_mode": decisions.get("circadian", {}).get("mode", "normal"),
                "hormesis": decisions.get("hormesis", {}).get("response", "comfortable"),
                "criticality": decisions.get("criticality", {}).get("value", 0.5),
                "tick": decisions["tick"],
            })
        except Exception as e:
            logger.debug(f"[Lifecycle] Failed to publish: {e}")

    def get_status(self) -> Dict:
        return {
            "tick_count": self._tick_count,
            "danger_precision": self.danger_theory.get_precision(),
            "morphogenesis_suggestions": len(self.morphogenesis.suggest_new_synapses()),
            "morphogenesis_pruning": len(self.morphogenesis.suggest_pruning()),
        }

    def print_status(self):
        status = self.get_status()
        print(f"\n{'='*55}")
        print(f"  AUTONOMOUS LIFECYCLE STATUS")
        print(f"{'='*55}")
        print(f"  Ticks: {status['tick_count']}")
        print(f"  Danger precision: {status['danger_precision']:.1%}")
        print(f"  New synapse suggestions: {status['morphogenesis_suggestions']}")
        print(f"  Pruning suggestions: {status['morphogenesis_pruning']}")
        print(f"{'='*55}\n")


# Singleton
_lifecycle_instance = None

def get_lifecycle() -> MetaController:
    global _lifecycle_instance
    if _lifecycle_instance is None:
        _lifecycle_instance = MetaController()
    return _lifecycle_instance


# Phase 27 Task 19 audit fix: shared CircadianRhythm singleton. Scheduler
# (`_sleep_wake_tick`) and MetaController (`lifecycle_tick`) were each
# creating their OWN CircadianRhythm instance, so Process S evolved in the
# scheduler's copy but MetaController's `get_time_modulation()` read from a
# never-updated twin. Both now pull from `get_circadian()` so the sleep
# pressure and mode state are consistent across the organism.
_circadian_instance: Optional["CircadianRhythm"] = None

def get_circadian() -> "CircadianRhythm":
    global _circadian_instance
    if _circadian_instance is None:
        _circadian_instance = CircadianRhythm()
    return _circadian_instance
