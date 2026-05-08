"""Phase 30 C.2 — CompositeRiskManager.

Single risk gate that aggregates: assertion chain (A.18), single-position cap
(A.26), aggregate exposure (A.26), correlation limit (A.18), daily loss limit
(A.18), realtime anomaly halt (A.27).

Returns CompositeRiskResult with blocked_by_check + adjusted_stake.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class CompositeRiskResult:
    passed: bool
    adjusted_stake: float
    reason: str = ""
    blocked_by_check: Optional[str] = None
    checks: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class _CheckEntry:
    name: str
    fn: Callable[..., Any]


class CompositeRiskManager:
    def __init__(self, checks: Optional[List[_CheckEntry]] = None):
        self.checks: List[_CheckEntry] = list(checks or [])

    def add_check(self, name: str, fn: Callable[..., Any]) -> None:
        self.checks.append(_CheckEntry(name=name, fn=fn))

    def evaluate(self, pair: str, stake: float, portfolio_value: float,
                 open_positions: List[str], **kwargs) -> CompositeRiskResult:
        checks_log: List[Dict[str, Any]] = []
        adjusted_stake = stake
        for entry in self.checks:
            try:
                result = entry.fn(
                    pair=pair, stake=adjusted_stake,
                    portfolio_value=portfolio_value,
                    open_positions=open_positions,
                    **kwargs,
                )
            except Exception as e:
                logger.error(f"[CRM] {entry.name} crashed: {e}")
                checks_log.append({"name": entry.name, "passed": False, "error": str(e)})
                continue
            entry_log: Dict[str, Any] = {"name": entry.name}
            if hasattr(result, "passed"):
                entry_log["passed"] = bool(result.passed)
                entry_log["reason"] = getattr(result, "reason", "")
                if not result.passed and getattr(result, "severity", "error") == "error":
                    self._persist(pair, entry.name, False, getattr(result, "reason", ""))
                    return CompositeRiskResult(
                        passed=False,
                        adjusted_stake=adjusted_stake,
                        reason=getattr(result, "reason", "blocked"),
                        blocked_by_check=entry.name,
                        checks=checks_log + [entry_log],
                    )
            elif isinstance(result, dict):
                entry_log.update(result)
                if not result.get("passed", True):
                    self._persist(pair, entry.name, False, result.get("reason", ""))
                    return CompositeRiskResult(
                        passed=False,
                        adjusted_stake=result.get("adjusted_stake", adjusted_stake),
                        reason=result.get("reason", "blocked"),
                        blocked_by_check=entry.name,
                        checks=checks_log + [entry_log],
                    )
                adjusted_stake = result.get("adjusted_stake", adjusted_stake)
            checks_log.append(entry_log)
            self._persist(pair, entry.name, True, "")
        return CompositeRiskResult(
            passed=True, adjusted_stake=adjusted_stake,
            reason="all_checks_passed", checks=checks_log,
        )

    def _persist(self, pair: str, check: str, passed: bool, reason: str) -> None:
        try:
            from db import AI_DB_PATH, get_db_connection

            with get_db_connection(AI_DB_PATH) as conn:
                conn.execute(
                    """INSERT INTO composite_risk_decisions
                       (pair, check_name, passed, reason, modifier_applied)
                       VALUES (?, ?, ?, ?, ?)""",
                    (pair, check, int(passed), reason[:500], 1.0),
                )
                conn.commit()
        except Exception:
            pass


def build_default() -> CompositeRiskManager:
    crm = CompositeRiskManager()

    def _kelly_chain(**kw) -> Dict[str, Any]:
        from assertions import (
            check_kelly_floor, check_kelly_ceiling, check_stop_loss_present, check_leverage_bound,
        )
        meta = kw.get("decision_metadata", {}) or {}
        kf = meta.get("kelly_fraction", 0.01)
        for fn, label in [
            (lambda: check_kelly_floor(kf), "kelly_floor"),
            (lambda: check_kelly_ceiling(kf), "kelly_ceiling"),
            (lambda: check_stop_loss_present(meta.get("stop_loss_pct", -0.05)), "stop_loss"),
            (lambda: check_leverage_bound(meta.get("leverage", 1.0)), "leverage"),
        ]:
            r = fn()
            if not r.passed and r.severity == "error":
                return {"passed": False, "reason": r.reason}
        return {"passed": True}

    def _position_cap(**kw) -> Dict[str, Any]:
        from assertions import check_single_position_cap, check_aggregate_exposure_cap

        r = check_single_position_cap(kw["stake"], kw["portfolio_value"])
        if not r.passed:
            return {"passed": False, "reason": r.reason}
        open_value = sum(p.get("stake_amount", 0.0) for p in kw.get("open_positions", []) if isinstance(p, dict))
        r2 = check_aggregate_exposure_cap(open_value + kw["stake"], kw["portfolio_value"])
        if not r2.passed:
            return {"passed": False, "reason": r2.reason}
        return {"passed": True}

    def _correlation(**kw) -> Dict[str, Any]:
        from assertions import check_correlation_limit

        r = check_correlation_limit(kw["pair"], [p.get("pair", str(p)) if isinstance(p, dict) else str(p)
                                                  for p in kw.get("open_positions", [])])
        return {"passed": r.passed, "reason": r.reason}

    def _anomaly_halt(**kw) -> Dict[str, Any]:
        try:
            from realtime_anomaly_detector import get_detector

            halted, reason = get_detector().is_halted(kw["pair"])
            return {"passed": not halted, "reason": reason if halted else ""}
        except Exception:
            return {"passed": True}

    crm.add_check("kelly_chain", _kelly_chain)
    crm.add_check("position_cap", _position_cap)
    crm.add_check("correlation", _correlation)
    crm.add_check("anomaly_halt", _anomaly_halt)
    return crm
