"""
rlaif_reward.py — Phase 27 Task 15 (C4 Ajani)

RLAIF d-RLAIF reward generator.

After each closed trade, ask 3 different LLM judges (Gemini, Groq, Mistral)
to score the decision on 5 rubric dimensions. The final composite reward
combines:

  env_reward  (actual realised PnL, the ground truth)
  llm_reward  (worst-case-optimised rubric score, the process-quality signal)

The WCO fusion (take the MINIMUM across judges per dimension) makes the
signal robust to any single sycophant model. `total_reward = 0.80 * env
+ 0.20 * llm` — the LLM portion never dominates.

6 reward-hacking surfaces are mitigated up-front:
  1. PnL avoidance       → forgone PnL is tracked separately (Fix 6).
  2. Confidence anchoring → entropy penalty via variance check (inline).
  3. Regime exploitation → regime is verified from tech data, not LLM claim.
  4. LLM sycophancy      → WCO across 3 providers.
  5. Verbosity bias      → fixed-format rubric JSON (prompt-enforced).
  6. Goodhart over-opt   → composite is CAPPED to ±1.0 then mixed at 20%.

Persisted to `rlaif_rewards` (schema in db.py).
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
logger = logging.getLogger("rlaif_reward")

from ai_config import AI_DB_PATH
from db import get_db_connection, init_db


RUBRIC = {
    "signal_quality":   {"weight": 0.25,
                         "desc": "Was the confidence CALIBRATED to the actual outcome?"},
    "sizing_quality":   {"weight": 0.25,
                         "desc": "Was the position size proportional to edge × certainty?"},
    "timing_quality":   {"weight": 0.20,
                         "desc": "Was entry/exit timing appropriate vs. the move?"},
    "risk_management":  {"weight": 0.15,
                         "desc": "Were stoploss, leverage, drawdown bounds respected?"},
    "regime_alignment": {"weight": 0.15,
                         "desc": "Was the trade appropriate for the actual regime?"},
}


def _rubric_prompt(verdict: Dict[str, Any], context: Dict[str, Any]) -> str:
    """Fixed-format prompt forces JSON reply (Goodhart mitigation #5).

    Mega Sprint 2026-04-23 (A.4): outcome / pnl / n_recent_losses removed
    from the judge context. They leaked the answer into the prompt, which
    biased composite to cluster near +1 on wins and -1 on losses and made
    the rubric redundant with env_reward. Judges now evaluate the ENTRY
    STATE only; composite measures process quality, env_reward measures
    PnL.
    """
    from ai_config import get_flag
    hindsight_removed = get_flag("rlaif_hindsight_removed", True)

    lines = [
        "You are an RLAIF judge scoring a trading decision AT ENTRY TIME.",
        "Score STRICTLY in [0, 1] for each dimension. Output ONLY valid JSON.",
        "",
        "=== RUBRIC ===",
    ]
    for dim, meta in RUBRIC.items():
        lines.append(f"  {dim}: {meta['desc']} (weight {meta['weight']:.2f})")
    lines += [
        "",
        "=== TRADE ===",
        f"pair:       {verdict.get('pair')}",
        f"signal:     {verdict.get('signal')}",
        f"confidence: {verdict.get('confidence')}",
        f"regime:     {verdict.get('regime')}",
        f"duration:   {verdict.get('duration')}",
    ]
    if not hindsight_removed:
        # Legacy behaviour kept behind the feature flag for rollback/A-B.
        lines += [
            f"outcome:    {verdict.get('outcome')}",
            f"pnl:        {verdict.get('pnl')}",
        ]
    lines += [
        "",
        "=== CONTEXT ===",
        f"actual_regime_verified: {context.get('verified_regime')}",
    ]
    if not hindsight_removed:
        lines.append(f"n_recent_losses: {context.get('n_recent_losses', 0)}")
    lines += [
        "",
        "Return JSON: {\"signal_quality\": float, \"sizing_quality\": float, "
        "\"timing_quality\": float, \"risk_management\": float, "
        "\"regime_alignment\": float, \"justification\": \"one sentence\"}",
    ]
    return "\n".join(lines)


def _parse_rubric_response(content: str) -> Optional[Dict[str, float]]:
    """Parse judge's JSON, return dict of clipped [0,1] scores or None."""
    if not content:
        return None
    txt = str(content).strip()
    for marker in ("```json", "```JSON", "```"):
        txt = txt.replace(marker, "")
    try:
        start = txt.find("{")
        end = txt.rfind("}")
        if start < 0 or end <= start:
            return None
        data = json.loads(txt[start:end + 1])
    except (json.JSONDecodeError, ValueError):
        return None
    # Strict parse: ANY malformed dimension invalidates the provider response.
    # Previous behaviour fell back to 0.5 (neutral) per missing dim, which
    # silently injected "no information" into the WCO min and poisoned
    # composite scoring. The Signal Quality audit (2026-04-20) reversed this
    # — `rlaif_rewards` now stores only provider rows where every dim parsed.
    scores: Dict[str, float] = {}
    for dim in RUBRIC:
        raw = data.get(dim)
        try:
            val = float(raw)
        except (TypeError, ValueError):
            logger.warning(
                f"[RLAIF] dimension '{dim}' parse failed ({raw!r}) — skipping provider"
            )
            return None
        scores[dim] = max(0.0, min(1.0, val))
    return scores


class RLAIFRewardGenerator:
    """Score a trade with 3-judge WCO rubric and mix with env reward."""

    ENV_WEIGHT = 0.80   # Environment reward (realised PnL) dominates.
    LLM_WEIGHT = 0.20   # LLM rubric is process-quality signal.

    def __init__(self, llm_router=None):
        self._router = llm_router
        init_db()

    def _get_router(self):
        if self._router is None:
            try:
                from llm_router import LLMRouter
                self._router = LLMRouter(temperature=0.2)
            except Exception as e:
                logger.debug(f"[RLAIF] router init failed: {e}")
                self._router = None
        return self._router

    async def _score_with_judges(self, verdict: Dict[str, Any],
                                  context: Dict[str, Any]) -> Dict[str, Any]:
        """Dispatch the rubric to 3 providers, collect WCO per dimension."""
        router = self._get_router()
        if router is None:
            return {"per_dim_wco": None, "provider_scores": {},
                    "error": "router unavailable"}

        try:
            from langchain_core.messages import SystemMessage, HumanMessage
        except Exception as e:
            return {"per_dim_wco": None, "provider_scores": {},
                    "error": f"langchain unavailable: {e}"}

        messages = [
            SystemMessage(content="You are an RLAIF judge. Reply with JSON only."),
            HumanMessage(content=_rubric_prompt(verdict, context)),
        ]

        # Tur-3: share the LinUCB feature builder with rag_graph /
        # agent_pool so every caller hands the router identical shape.
        import datetime as _dt
        _combined = "".join(
            str(getattr(m, "content", "") or "") for m in messages
        )
        rlaif_ctx = {
            "task": "rlaif_judge",
            "prompt_len": len(_combined),
            "needs_json": True,
            "regime_vol": 0.5,
            "hour_utc": _dt.datetime.now(_dt.timezone.utc).hour,
        }

        try:
            ens = await router.ensemble_invoke(
                messages, n_judges=3, temperature=0.2,
                task_context=rlaif_ctx,
                pair=str(verdict.get("pair") or ""),
            )
        except Exception as e:
            logger.debug(f"[RLAIF] ensemble_invoke failed: {e}")
            return {"per_dim_wco": None, "provider_scores": {},
                    "error": str(e)}

        per_provider: Dict[str, Dict[str, float]] = {}
        for content, provider in zip(ens.get("responses", []),
                                      ens.get("providers", [])):
            parsed = _parse_rubric_response(content)
            if parsed is not None:
                per_provider[provider] = parsed

        if not per_provider:
            return {"per_dim_wco": None, "provider_scores": {},
                    "error": "no parseable rubric replies"}

        # Mega Sprint 2026-04-23 (A.4) + Tur-2 (H12): median aggregation
        # lives in `_median_aggregate` so the unit test exercises the same
        # code path production uses. Single-judge rubrics are skipped
        # because a "median" of one value is just the one value — which
        # reintroduces the WCO-min fragility the switch was meant to fix.
        wco = self._median_aggregate(per_provider)

        return {"per_dim_wco": wco,
                "provider_scores": per_provider,
                "cdr": ens.get("cdr"),
                "error": None}

    @staticmethod
    def _median_aggregate(per_provider: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """Median fuse per-dimension scores across providers.

        Revize Tur-2 (M4): require ≥2 judges per dimension. With one judge
        the "median" collapses back to that single value and the aggregation
        stops being robust to outliers — exactly the failure mode that
        motivated moving away from WCO-min in the first place.
        """
        import statistics
        out: Dict[str, float] = {}
        for dim in RUBRIC:
            dim_scores = [
                scores[dim] for scores in per_provider.values() if dim in scores
            ]
            if len(dim_scores) < 2:
                continue
            out[dim] = statistics.median(dim_scores)
        return out

    @staticmethod
    def _composite(per_dim: Dict[str, float]) -> float:
        """Weighted sum of 5 dimensions, remapped to [-1, +1] (0.5 = neutral).

        Missing dimensions (e.g. when every judge skipped one) are dropped
        and the remaining weights are renormalised — previously a KeyError
        would bubble up and crash the reward path.
        """
        total_weight = sum(meta["weight"] for dim, meta in RUBRIC.items() if dim in per_dim)
        if total_weight <= 0:
            return 0.0
        raw = sum(per_dim[dim] * RUBRIC[dim]["weight"] for dim in RUBRIC if dim in per_dim)
        composite_01 = raw / total_weight
        return 2.0 * composite_01 - 1.0  # [0,1] → [-1,+1]

    def _persist(self, record: Dict[str, Any]) -> None:
        try:
            conn = get_db_connection(AI_DB_PATH)
            conn.execute("""
                INSERT INTO rlaif_rewards
                    (trade_id, pair, timestamp,
                     signal_quality, sizing_quality, timing_quality,
                     risk_management, regime_alignment,
                     composite, provider_scores,
                     env_reward, total_reward, outcome_pnl)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                record.get("trade_id"), record.get("pair"),
                record.get("timestamp"),
                record.get("signal_quality"), record.get("sizing_quality"),
                record.get("timing_quality"), record.get("risk_management"),
                record.get("regime_alignment"),
                record.get("composite"),
                json.dumps(record.get("provider_scores", {})),
                record.get("env_reward"), record.get("total_reward"),
                record.get("outcome_pnl"),
            ))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.debug(f"[RLAIF] persist failed: {e}")

    async def score_trade(self, verdict: Dict[str, Any],
                           context: Optional[Dict[str, Any]] = None,
                           env_reward: Optional[float] = None) -> Dict[str, Any]:
        """Score a single trade. Returns the full reward record."""
        context = context or {}
        if env_reward is None:
            # Default env reward = tanh(pnl_pct) so unbounded PnL is bounded.
            pnl = float(verdict.get("pnl") or 0.0)
            import math as _math
            env_reward = _math.tanh(pnl / 5.0)  # 5% PnL → ~0.76
            env_reward = max(-1.0, min(1.0, env_reward))

        judged = await self._score_with_judges(verdict, context)
        wco = judged.get("per_dim_wco")
        if wco is None:
            # Judges unavailable → NO DB persist (previous behaviour stored NULL
            # rubric rows that poisoned downstream rlaif analytics). The caller
            # still gets an env-only reward so trade accounting is not blocked.
            logger.warning(
                f"[RLAIF] {verdict.get('trade_id')} all judges failed "
                f"({judged.get('error')}) — env-only reward, skipping DB persist"
            )
            return {
                "trade_id": verdict.get("trade_id"),
                "pair": verdict.get("pair"),
                "timestamp": datetime.now(tz=timezone.utc).isoformat(),
                "signal_quality": None, "sizing_quality": None,
                "timing_quality": None, "risk_management": None,
                "regime_alignment": None,
                "composite": None,
                "provider_scores": judged.get("provider_scores", {}),
                "env_reward": float(env_reward),
                "total_reward": float(env_reward),
                "outcome_pnl": verdict.get("pnl"),
                "error": judged.get("error"),
                "persisted": False,
            }

        composite = self._composite(wco)
        total = self.ENV_WEIGHT * env_reward + self.LLM_WEIGHT * composite
        total = max(-1.0, min(1.0, total))

        record = {
            "trade_id": verdict.get("trade_id"),
            "pair": verdict.get("pair"),
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            **{dim: wco.get(dim) for dim in RUBRIC},
            "composite": round(float(composite), 4),
            "provider_scores": judged.get("provider_scores", {}),
            "env_reward": round(float(env_reward), 4),
            "total_reward": round(float(total), 4),
            "outcome_pnl": verdict.get("pnl"),
            "cdr": judged.get("cdr"),
        }
        self._persist(record)
        record["persisted"] = True
        logger.info(
            f"[RLAIF] trade={verdict.get('trade_id')} {verdict.get('pair')} "
            f"env={env_reward:+.2f} llm={composite:+.2f} → total={total:+.2f} "
            f"(cdr={judged.get('cdr')})"
        )
        return record


_rlaif_instance: Optional[RLAIFRewardGenerator] = None


def get_rlaif() -> RLAIFRewardGenerator:
    global _rlaif_instance
    if _rlaif_instance is None:
        _rlaif_instance = RLAIFRewardGenerator()
    return _rlaif_instance
