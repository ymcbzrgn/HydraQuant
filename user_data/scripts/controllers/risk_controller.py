"""Phase 30 C.1/C.2 — RiskController glue with CompositeRiskManager."""
from __future__ import annotations

import logging
from typing import Optional

from .base import BaseController, ControllerContext, ControllerDecision

logger = logging.getLogger(__name__)


class RiskController(BaseController):
    name = "risk"

    def __init__(self, composite_risk_manager=None):
        self.crm = composite_risk_manager
        if self.crm is None:
            try:
                from composite_risk_manager import build_default

                self.crm = build_default()
            except Exception:
                self.crm = None

    def decide(self, ctx: ControllerContext) -> ControllerDecision:
        if self.crm is None:
            return ControllerDecision(proceed=True, reason="no_crm",
                                      stake_amount=ctx.metadata.get("proposed_stake", 0.0))
        proposed_stake = float(ctx.metadata.get("proposed_stake", 0.0))
        result = self.crm.evaluate(
            pair=ctx.pair,
            stake=proposed_stake,
            portfolio_value=ctx.portfolio_value,
            open_positions=ctx.open_positions,
        )
        if not result.passed:
            return ControllerDecision(
                proceed=False,
                reason=result.reason or "risk_blocked",
                blocked_by=result.blocked_by_check,
                metadata={"checks": result.checks},
            )
        return ControllerDecision(
            proceed=True,
            stake_amount=result.adjusted_stake,
            reason="risk_ok",
            metadata={"checks": result.checks},
        )
