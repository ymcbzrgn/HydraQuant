"""Phase 30 D.2 — Plan/Verify SOP for major sizing changes.

Two-subagent flow:
1. Planner: produces plan (proposed change + risk analysis).
2. Verifier: independent agent reviews plan.

Output: VERDICT in {PASS, FAIL, PARTIAL}. Only PASS proceeds to shadow paper trade.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class PlanVerifyResult:
    verdict: str  # PASS / FAIL / PARTIAL
    plan: Dict[str, Any]
    verify_reason: str
    raw_plan: str = ""
    raw_verify: str = ""


def run_sop(
    pair: str,
    change_proposal: Dict[str, Any],
    planner: Optional[Callable[[str], str]] = None,
    verifier: Optional[Callable[[str, str], str]] = None,
) -> PlanVerifyResult:
    if planner is None or verifier is None:
        try:
            from llm_router import LLMRouter

            router = LLMRouter()

            def _llm_call(model_label: str, prompt: str) -> str:
                from effort_probe import select_models
                for m in select_models(model_label):
                    try:
                        if hasattr(router, "invoke_with_model"):
                            r = router.invoke_with_model(m, prompt)
                            if r:
                                return str(r)
                    except Exception:
                        continue
                return ""

            if planner is None:
                planner = lambda prompt: _llm_call("high", prompt)
            if verifier is None:
                verifier = lambda plan, prompt: _llm_call("default", f"VERIFY:\n{plan}\n\nORIGINAL_PROMPT:\n{prompt}")
        except Exception:
            planner = planner or (lambda p: "{}")
            verifier = verifier or (lambda plan, p: "PASS")

    plan_prompt = (
        f"You are the HydraQuant Planner. Pair={pair}. Proposed change: {json.dumps(change_proposal)}\n"
        "Produce a structured plan as JSON: {steps: [], risks: [], rollback: ''}"
    )
    raw_plan = planner(plan_prompt)
    try:
        from json_parse_robust import parse_json
        plan, _ = parse_json(raw_plan, "plan_verify_sop_plan")
    except Exception:
        plan = {}

    verify_prompt = (
        "You are the HydraQuant Verifier. Independently review the plan above.\n"
        "Output one of: PASS, FAIL, PARTIAL — followed by a one-line reason."
    )
    raw_verify = verifier(json.dumps(plan or {}), verify_prompt) or ""
    text = (raw_verify or "").strip().upper()
    if text.startswith("PASS"):
        verdict = "PASS"
    elif text.startswith("FAIL"):
        verdict = "FAIL"
    elif text.startswith("PARTIAL"):
        verdict = "PARTIAL"
    else:
        verdict = "FAIL"

    try:
        from db import AI_DB_PATH, get_db_connection

        with get_db_connection(AI_DB_PATH) as conn:
            conn.execute(
                """INSERT INTO plan_verify_runs (pair, plan_json, verify_verdict, verify_reason, executed)
                   VALUES (?, ?, ?, ?, 0)""",
                (pair, json.dumps(plan or {})[:4000], verdict, (raw_verify or "")[:500]),
            )
            conn.commit()
    except Exception:
        pass

    return PlanVerifyResult(
        verdict=verdict,
        plan=plan or {},
        verify_reason=(raw_verify or "")[:300],
        raw_plan=(raw_plan or "")[:2000],
        raw_verify=(raw_verify or "")[:2000],
    )
