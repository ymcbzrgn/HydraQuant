"""Phase 30 C.1 — SignalController.

Owns: signal aggregation, evidence engine call, agent_pool debate, calibrator.
Returns: proceed=true/false with confidence-driven stake_amount.
"""
from __future__ import annotations

import logging
from typing import Optional

from .base import BaseController, ControllerContext, ControllerDecision

logger = logging.getLogger(__name__)


class SignalController(BaseController):
    name = "signal"

    def __init__(self, evidence_engine=None, agent_pool=None, calibrator=None):
        self.evidence_engine = evidence_engine
        self.agent_pool = agent_pool
        self.calibrator = calibrator

    def decide(self, ctx: ControllerContext) -> ControllerDecision:
        confidence = 0.5
        try:
            if self.evidence_engine is not None:
                ev = self.evidence_engine.score(ctx.pair, regime=ctx.market_regime)
                confidence = float(ev.get("confidence", 0.5))
        except Exception as e:
            logger.warning(f"[SignalController] evidence: {e}")

        veto_reason: Optional[str] = None
        try:
            if self.agent_pool is not None and hasattr(self.agent_pool, "run_debate_minimal"):
                votes = self.agent_pool.run_debate_minimal(ctx.pair, ctx.side)
                from four_state_veto import aggregate

                veto = aggregate(votes)
                if veto.decision != "allow":
                    veto_reason = f"agent_veto:{veto.decision}"
        except Exception as e:
            logger.warning(f"[SignalController] agent_pool: {e}")

        if veto_reason:
            return ControllerDecision(proceed=False, reason=veto_reason, blocked_by=veto_reason)

        try:
            if self.calibrator is not None:
                from calibrator_health_check import health_report

                rep = health_report()
                if not rep["recommend_bypass"]:
                    confidence = float(self.calibrator.calibrate(confidence))
        except Exception:
            pass

        if confidence < 0.5:
            return ControllerDecision(proceed=False, reason="low_confidence",
                                      blocked_by="confidence_below_0.5",
                                      metadata={"confidence": confidence})
        stake = ctx.portfolio_value * 0.02 * (confidence - 0.5) * 2  # simple proportional
        return ControllerDecision(proceed=True, stake_amount=max(0.0, stake),
                                  reason="signal_ok", metadata={"confidence": confidence})
