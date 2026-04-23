"""EK Sprint 2026-04-23 (EK.2.1): query-context → LinUCB feature vector.

LinUCB routes an LLM call based on a normalised [0,1] feature representation
of the call's context (task type, prompt length, JSON requirement, regime
volatility, hour of day). Keeping the extractor in a dedicated module means
`ModelSlot.linucb_score` / `linucb_update` stay zero-dep w.r.t. the rest of
the router, and the same function can be used to pre-compute features in
retroactive feedback paths.
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np


TASK_TYPE_MAP = {
    "coordinator_debate":   0,
    "agent_pool_r1":        1,
    "agent_pool_r2":        2,
    "rlaif_judge":          3,
    "hypothesis_cycle":     4,
    "multimodal_inference": 5,
    "post_trade_court":     6,
    "default":              7,
}
N_TASKS = len(TASK_TYPE_MAP)
FEATURE_DIM = 5


def extract_features(ctx: Dict[str, Any]) -> np.ndarray:
    """Return a ``FEATURE_DIM``-vector in ``[0, 1]`` for LinUCB.

    Expected keys (all optional, defaults preserve the exploration prior):
      * ``task``         — one of TASK_TYPE_MAP keys. Falls back to 'default'.
      * ``prompt_len``   — approximate character count of the outbound prompt.
      * ``needs_json``   — True if the caller enforces a structured reply.
      * ``regime_vol``   — regime volatility proxy in ``[0, 1]`` (VPIN,
        ATR z-score, etc.). Clamped to stay inside the range.
      * ``hour_utc``     — integer hour-of-day, used as a circadian feature.
    """
    task_id = TASK_TYPE_MAP.get(ctx.get("task", "default"),
                                TASK_TYPE_MAP["default"])
    prompt_len = max(0, int(ctx.get("prompt_len", 0) or 0))
    needs_json = bool(ctx.get("needs_json", False))
    try:
        regime_vol = float(ctx.get("regime_vol", 0.5) or 0.5)
    except (TypeError, ValueError):
        regime_vol = 0.5
    try:
        hour_utc = int(ctx.get("hour_utc", 12) or 12)
    except (TypeError, ValueError):
        hour_utc = 12

    return np.array([
        task_id / max(N_TASKS - 1, 1),
        min(prompt_len / 8000.0, 1.0),
        1.0 if needs_json else 0.0,
        float(np.clip(regime_vol, 0.0, 1.0)),
        (hour_utc % 24) / 24.0,
    ], dtype=np.float64)
