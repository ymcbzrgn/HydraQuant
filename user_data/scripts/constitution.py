"""
constitution.py — Phase 26 Sprint 2, Task 10C

Constitution: Unbreakable rules that NO module can override.
Latency Guard: Per-module ms budget with automatic degradation.

Constitution enforces:
  - max_drawdown: 25%
  - max_single_position: 3%
  - max_leverage: 5x
  - max_portfolio_heat: 10%
  - kill switches: adrenaline freeze, OOM protection, loss streak freeze

Latency Guard enforces:
  - Tier-0: <1ms (constitution checks)
  - Tier-1: <100ms (RL inference, hormones)
  - Tier-2: <5s (TTM, ensemble)
  - Tier-3: <60s (world model, MADAM)
  - Tier-4: unbounded (dream engine, neuroevolution)

Override hierarchy (top overrides bottom, NEVER reverse):
  1. Constitution (hardcoded, unchangeable)
  2. Prefrontal cortex (learns, can add rules)
  3. Hormonal modulation (dynamic, bounded by constitution)
  4. RL policy (optimizes within hormonal bounds)
  5. Individual neurons (fine-tune within RL policy)

Integration:
  - Called by AIFreqtradeSizer before every trade
  - Called by scheduler before every module execution
  - Logs violations to organism_audit table
"""

import os
import sys
import json
import logging
import time as _time
import functools
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Callable, Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
logger = logging.getLogger("constitution")

from ai_config import AI_DB_PATH
from db import get_db_connection, execute_with_retry, init_db

# ===========================================================================
# Constitution — Unbreakable Laws
# ===========================================================================

CONSTITUTION = {
    "version": "1.0",
    "safety_limits": {
        "max_drawdown_pct": 25.0,
        "max_single_position_pct": 3.0,
        "max_leverage": 5.0,
        "max_portfolio_heat_pct": 10.0,
        "atr_leverage_product_max": 8.0,
    },
    "kill_switches": {
        "adrenaline_freeze_stress": 0.85,
        "oom_threshold_mb": 500,
        "consecutive_loss_freeze": 5,
        "consecutive_loss_freeze_hours": 24,
    },
    "audit_requirements": [
        "feature_snapshot", "uncertainty_bounds", "hormone_state",
        "regime_classification", "active_overrides", "module_contributions",
    ],
    "override_hierarchy": [
        "constitution",        # Level 1 — NEVER overridden
        "prefrontal_cortex",   # Level 2
        "hormonal_modulation", # Level 3
        "rl_policy",           # Level 4
        "individual_neurons",  # Level 5
    ],
}


class ConstitutionEnforcer:
    """Enforces unbreakable trading rules."""

    def __init__(self):
        self._violation_count = 0
        self._freeze_until = None
        init_db()

    def check_trade(self, trade_params: Dict) -> Dict:
        """Check a proposed trade against constitutional limits.

        Args:
            trade_params: {
                "sizing_pct": float (% of portfolio),
                "leverage": float,
                "portfolio_drawdown_pct": float,
                "portfolio_heat_pct": float,
                "atr_pct": float,
                "consecutive_losses": int,
                "market_stress": float,
                "free_ram_mb": float (optional),
            }

        Returns:
            {"allowed": bool, "violations": [...], "clamped_params": {...}}
        """
        limits = CONSTITUTION["safety_limits"]
        kills = CONSTITUTION["kill_switches"]
        violations = []
        clamped = {}

        # Check freeze
        if self._freeze_until and datetime.now(timezone.utc) < self._freeze_until:
            remaining = (self._freeze_until - datetime.now(timezone.utc)).total_seconds() / 3600
            return {
                "allowed": False,
                "violations": [f"FREEZE active ({remaining:.1f}h remaining)"],
                "clamped_params": {},
                "freeze": True,
            }

        # 1. Max drawdown
        dd = trade_params.get("portfolio_drawdown_pct", 0)
        if dd > limits["max_drawdown_pct"]:
            violations.append(f"drawdown {dd:.1f}% > max {limits['max_drawdown_pct']}%")

        # 2. Max single position
        sizing = trade_params.get("sizing_pct", 0)
        if sizing > limits["max_single_position_pct"]:
            clamped["sizing_pct"] = limits["max_single_position_pct"]
            violations.append(f"position {sizing:.1f}% clamped to {limits['max_single_position_pct']}%")

        # 3. Max leverage
        leverage = trade_params.get("leverage", 1.0)
        if leverage > limits["max_leverage"]:
            clamped["leverage"] = limits["max_leverage"]
            violations.append(f"leverage {leverage:.1f}x clamped to {limits['max_leverage']}x")

        # 4. Max portfolio heat
        heat = trade_params.get("portfolio_heat_pct", 0)
        if heat > limits["max_portfolio_heat_pct"]:
            violations.append(f"portfolio heat {heat:.1f}% > max {limits['max_portfolio_heat_pct']}%")

        # 5. ATR × leverage product
        atr = trade_params.get("atr_pct", 0)
        lev = clamped.get("leverage", leverage)
        if atr * lev > limits["atr_leverage_product_max"]:
            max_lev = limits["atr_leverage_product_max"] / max(atr, 0.01)
            clamped["leverage"] = min(clamped.get("leverage", lev), max_lev)
            violations.append(f"ATR×leverage {atr*lev:.1f}% > max {limits['atr_leverage_product_max']}%")

        # Kill switches
        # 6. Adrenaline freeze
        stress = trade_params.get("market_stress", 0)
        if stress > kills["adrenaline_freeze_stress"]:
            violations.append(f"KILL: stress {stress:.2f} > freeze threshold {kills['adrenaline_freeze_stress']}")

        # 7. OOM protection
        free_ram = trade_params.get("free_ram_mb")
        if free_ram is not None and free_ram < kills["oom_threshold_mb"]:
            violations.append(f"KILL: free RAM {free_ram:.0f}MB < {kills['oom_threshold_mb']}MB")

        # 8. Consecutive loss freeze
        losses = trade_params.get("consecutive_losses", 0)
        if losses >= kills["consecutive_loss_freeze"]:
            freeze_hours = kills["consecutive_loss_freeze_hours"]
            self._freeze_until = datetime.now(timezone.utc) + timedelta(hours=freeze_hours)
            violations.append(f"KILL: {losses} consecutive losses → {freeze_hours}h freeze activated")

        # Determine if trade is allowed
        blocking = [v for v in violations if "KILL" in v or "drawdown" in v.lower() or "heat" in v.lower()]
        allowed = len(blocking) == 0

        # Log violations
        if violations:
            self._violation_count += len(violations)
            self._log_violations(violations, trade_params)

        return {
            "allowed": allowed,
            "violations": violations,
            "clamped_params": clamped,
            "freeze": self._freeze_until is not None and datetime.now(timezone.utc) < self._freeze_until,
        }

    def clamp_action(self, action: Dict) -> Dict:
        """Clamp RL action outputs to constitutional limits.

        Called by RL agents to ensure actions respect constitution.
        """
        limits = CONSTITUTION["safety_limits"]

        clamped = action.copy()
        if "sizing_pct" in clamped:
            clamped["sizing_pct"] = min(clamped["sizing_pct"], limits["max_single_position_pct"])
        if "leverage" in clamped:
            clamped["leverage"] = min(clamped["leverage"], limits["max_leverage"])

        return clamped

    def _log_violations(self, violations: List[str], params: Dict):
        """Log constitutional violations to organism_audit."""
        try:
            execute_with_retry(
                """INSERT INTO organism_audit
                   (timestamp, event_type, details_json)
                   VALUES (datetime('now'), 'constitution_violation', ?)""",
                (json.dumps({
                    "violations": violations,
                    "params": {k: v for k, v in params.items()
                              if isinstance(v, (int, float, str, bool))},
                }),),
                max_retries=3,
            )
        except Exception:
            pass

    def get_limits(self) -> Dict:
        """Get current constitutional limits."""
        return CONSTITUTION["safety_limits"].copy()

    def get_status(self) -> Dict:
        return {
            "version": CONSTITUTION["version"],
            "violation_count": self._violation_count,
            "freeze_active": self._freeze_until is not None and datetime.now(timezone.utc) < self._freeze_until,
            "freeze_until": self._freeze_until.isoformat() if self._freeze_until else None,
            "limits": CONSTITUTION["safety_limits"],
        }


# ===========================================================================
# Latency Guard — Per-module ms budget
# ===========================================================================

LATENCY_TIERS = {
    "tier_0": {"budget_ms": 1, "name": "Hard RT", "on_exceed": "inline"},
    "tier_1": {"budget_ms": 100, "name": "Soft RT", "on_exceed": "fallback"},
    "tier_2": {"budget_ms": 5000, "name": "Near RT", "on_exceed": "stale_cache"},
    "tier_3": {"budget_ms": 60000, "name": "Background", "on_exceed": "async"},
    "tier_4": {"budget_ms": float("inf"), "name": "Offline", "on_exceed": "none"},
}

# Module → tier mapping
MODULE_TIERS = {
    "constitution": "tier_0",
    "sizing_clamp": "tier_0",
    "rl_inference": "tier_1",
    "hormone_compute": "tier_1",
    "get_param": "tier_1",
    "ood_check": "tier_1",
    "ttm_perception": "tier_2",
    "ensemble_prediction": "tier_2",
    "conformal_interval": "tier_2",
    "chart_features": "tier_2",
    "catboost_predict": "tier_2",
    "world_model_imagine": "tier_3",
    "causal_query": "tier_3",
    "madam_debate": "tier_3",
    "rag_retrieval": "tier_3",
    "dream_engine": "tier_4",
    "sleep_consolidation": "tier_4",
    "neuroevolution": "tier_4",
    "ablation_league": "tier_4",
}


class LatencyGuard:
    """Per-module latency budget enforcement."""

    def __init__(self):
        self._violations: Dict[str, int] = {}
        self._total_calls: Dict[str, int] = {}
        self._total_latency: Dict[str, float] = {}

    def with_guard(self, func: Callable, module_name: str,
                   fallback: Any = None, tier: str = None) -> Any:
        """Execute function within latency budget.

        If function exceeds budget:
          - tier_1: return fallback value
          - tier_2: log warning, return result anyway
          - tier_3: should be async (warning only)
        """
        if tier is None:
            tier = MODULE_TIERS.get(module_name, "tier_2")

        budget_ms = LATENCY_TIERS[tier]["budget_ms"]

        start = _time.monotonic()
        try:
            result = func()
            elapsed_ms = (_time.monotonic() - start) * 1000

            # Track metrics
            self._total_calls[module_name] = self._total_calls.get(module_name, 0) + 1
            self._total_latency[module_name] = self._total_latency.get(module_name, 0) + elapsed_ms

            if elapsed_ms > budget_ms:
                self._violations[module_name] = self._violations.get(module_name, 0) + 1
                logger.warning(f"[LatencyGuard] {module_name} took {elapsed_ms:.0f}ms > "
                             f"{budget_ms}ms budget ({tier})")

                if tier == "tier_1" and fallback is not None:
                    return fallback  # Return fallback for tier-1 violations

            return result

        except Exception as e:
            elapsed_ms = (_time.monotonic() - start) * 1000
            logger.warning(f"[LatencyGuard] {module_name} failed after {elapsed_ms:.0f}ms: {e}")
            if fallback is not None:
                return fallback
            raise

    def guard(self, module_name: str, tier: str = None, fallback: Any = None):
        """Decorator version of with_guard."""
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                return self.with_guard(
                    lambda: func(*args, **kwargs),
                    module_name, fallback, tier,
                )
            return wrapper
        return decorator

    def get_stats(self) -> Dict:
        """Get latency statistics per module."""
        stats = {}
        for module in set(list(self._total_calls.keys()) + list(self._violations.keys())):
            calls = self._total_calls.get(module, 0)
            total_ms = self._total_latency.get(module, 0)
            violations = self._violations.get(module, 0)
            stats[module] = {
                "calls": calls,
                "avg_ms": round(total_ms / max(calls, 1), 1),
                "violations": violations,
                "violation_rate": round(violations / max(calls, 1), 3),
                "tier": MODULE_TIERS.get(module, "unknown"),
            }
        return stats

    def print_stats(self):
        stats = self.get_stats()
        print(f"\n{'='*65}")
        print(f"  LATENCY GUARD STATS")
        print(f"{'='*65}")
        print(f"  {'Module':<25} | {'Tier':<7} | {'Calls':>5} | {'Avg ms':>7} | {'Violations':>10}")
        print(f"  {'-'*62}")
        for module, s in sorted(stats.items(), key=lambda x: x[1]["violations"], reverse=True):
            print(f"  {module:<25} | {s['tier']:<7} | {s['calls']:>5} | "
                  f"{s['avg_ms']:>6.1f} | {s['violations']:>10}")
        print(f"{'='*65}\n")


# Singletons
_constitution_instance = None
_latency_guard_instance = None

def get_constitution() -> ConstitutionEnforcer:
    global _constitution_instance
    if _constitution_instance is None:
        _constitution_instance = ConstitutionEnforcer()
    return _constitution_instance

def get_latency_guard() -> LatencyGuard:
    global _latency_guard_instance
    if _latency_guard_instance is None:
        _latency_guard_instance = LatencyGuard()
    return _latency_guard_instance
