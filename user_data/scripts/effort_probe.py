"""Phase 30 B.15 — Effort probe cascade.

3-tier role cascade for LLM tasks:
- coord (heaviest, slowest, smartest)   -> e.g. claude-opus / gemini-2.5-pro
- pool  (mid, fast, balanced)           -> e.g. gemini-2.5-flash / llama-70b
- probe (cheapest, fastest, retry-safe) -> e.g. llama-8b-instant / haiku

Selection by `effort_label` PARAM:
- "high"    -> coord
- "default" -> pool
- "low"     -> probe

Cascade fallback: coord fail/timeout -> pool -> probe.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

CASCADES: Dict[str, List[str]] = {
    "coord": [
        "models/gemini-2.5-pro",
        "models/gemini-3.1-pro-preview",
        "claude-opus-4-7",
    ],
    "pool": [
        "models/gemini-2.5-flash",
        "llama-3.3-70b-versatile",
        "claude-sonnet-4-6",
    ],
    "probe": [
        "llama-3.1-8b-instant",
        "models/gemini-2.5-flash-lite",
        "claude-haiku-4-5-20251001",
    ],
}


def select_models(effort_label: str = "default") -> List[str]:
    label = (effort_label or "default").lower()
    if label in ("high", "coord"):
        return CASCADES["coord"] + CASCADES["pool"][:1]
    if label in ("low", "probe", "fast"):
        return CASCADES["probe"]
    return CASCADES["pool"] + CASCADES["probe"][:1]


def cascade_invoke(prompt: str, effort_label: str = "default", max_attempts: int = 3,
                   invoke_fn=None) -> Optional[Dict[str, Any]]:
    """Try models in cascade until one succeeds or attempts exhausted.

    invoke_fn(model, prompt) -> {"text": str, ...} or raises.
    """
    if invoke_fn is None:
        try:
            from llm_router import LLMRouter

            router = LLMRouter()

            def _default(model, prompt):
                return router.invoke_with_model(model, prompt) if hasattr(router, "invoke_with_model") else None
            invoke_fn = _default
        except Exception:
            return None
    models = select_models(effort_label)[:max_attempts]
    for m in models:
        try:
            result = invoke_fn(m, prompt)
            if result:
                return {"model": m, "result": result}
        except Exception as e:
            logger.warning(f"[EffortProbe] {m} failed: {e} — cascading")
            continue
    return None
