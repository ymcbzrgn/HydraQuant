"""Phase 30 C.1 — TimingController (cerebellum + anomaly halt + funding window)."""
from __future__ import annotations

import logging

from .base import BaseController, ControllerContext, ControllerDecision

logger = logging.getLogger(__name__)


class TimingController(BaseController):
    name = "timing"

    def decide(self, ctx: ControllerContext) -> ControllerDecision:
        try:
            from realtime_anomaly_detector import get_detector

            halted, reason = get_detector().is_halted(ctx.pair)
            if halted:
                return ControllerDecision(proceed=False, reason=reason, blocked_by="anomaly_halt")
        except Exception:
            pass

        mult = 1.0
        try:
            from cerebellum_timing import CerebellumTiming

            mult = float(CerebellumTiming().get_timing_multiplier(pair=ctx.pair))
        except Exception:
            pass

        adjusted = float(ctx.metadata.get("proposed_stake", 0.0)) * mult
        return ControllerDecision(
            proceed=True,
            stake_amount=adjusted,
            reason=f"timing_ok_mult={mult:.2f}",
            metadata={"timing_multiplier": mult},
        )
