"""
hypothesis_generator.py — Phase 27 Task 16 (G2 Ajani)

LLM Strategy Researcher — weekly hypothesis loop.

Once per week, gather last-cycle evidence (PostTradeCourt verdicts, Ablation
League table, Forgone PnL digest) and ask an LLM to propose 3-5 parameter
tweaks. Each hypothesis must change EXACTLY one parameter, name a mechanism,
and provide a falsification condition.

6-Gate validation (runs AFTER LLM generation, BEFORE deploy):
  1. Complexity  — single-parameter only (reject multi-param).
  2. Novelty     — no duplicate hypothesis_id in last 90 days.
  3. Walk-forward — in-sample Sharpe placeholder (stubbed — wire to real
                    backtester in Sprint 3B task 16b).
  4. Deflated Sharpe Ratio — corrects for multiple-hypothesis testing
                              (López de Prado 2014).
  5. OOS holdout — 20% tail (stub — same wiring point as gate 3).
  6. Shadow      — gets deployed to shadow_period_sharpe tracking.

We only call the LLM when the router has daily quota headroom (checks
rpd_status via router), and we cap to 5 hypotheses per cycle so free-tier
usage stays sane.

FINSABER guardrail (KDD 2026): LLMs are TERRIBLE at generating trading
SIGNALS but can be useful for PARAMETER refinement. This loop keeps the LLM
in the parameter role only — never directly opens a position.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
logger = logging.getLogger("hypothesis_generator")

from ai_config import AI_DB_PATH
from db import get_db_connection, init_db


MAX_HYPOTHESES_PER_CYCLE = 5          # hard cap — 1 cycle / week
LOOKBACK_DAYS_FOR_CONTEXT = 14
NOVELTY_LOOKBACK_DAYS = 90


def _hypothesis_id(parameter: str, proposed_value: float) -> str:
    """Stable ID — same param / value pair always hashes the same."""
    raw = f"{parameter}:{round(float(proposed_value), 6)}".encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:12]


def _prompt_template(context: Dict[str, Any]) -> str:
    """Weekly hypothesis prompt. JSON-only reply enforced."""
    lines = [
        "You are a quantitative strategy researcher.",
        "Propose parameter tweaks for a live crypto trading bot. Each hypothesis",
        "must change EXACTLY ONE parameter, state the mechanism, and give a",
        "falsification condition (what empirically disproves this hypothesis).",
        "",
        "RULES:",
        "  - NO new signals. Only adjust EXISTING parameters.",
        "  - 3 to 5 hypotheses. STRICT JSON output.",
        "  - Small, testable tweaks (e.g. a 10-15% change), not radical redesigns.",
        "",
        "=== LAST 14 DAYS ===",
        f"n_trades: {context.get('n_trades', 0)}",
        f"win_rate: {context.get('win_rate', 0.0):.2f}",
        f"avg_pnl:  {context.get('avg_pnl_pct', 0.0):+.2f}%",
        f"avg_hold: {context.get('avg_hold_hours', 0.0):.1f}h",
        "",
        "=== FORGONE DIGEST ===",
        f"missed_alpha: {context.get('forgone_alpha_pct', 0.0):+.2f}%",
        f"top_missed:   {context.get('top_missed_pair', 'n/a')}",
        "",
        "=== RECENT VERDICTS ===",
    ]
    verdicts = context.get("recent_verdicts") or []
    for v in verdicts[:5]:
        lines.append(f"  {v}")
    lines += [
        "",
        "Return JSON:",
        "{\"hypotheses\": [",
        "  {\"parameter\": str, \"current_value\": float, \"proposed_value\": float,",
        "   \"mechanism\": \"short one sentence\", \"falsification\": \"condition\",",
        "   \"affected_pairs\": [\"BTC/USDT:USDT\", ...] or \"_any\",",
        "   \"expected_impact\": \"what you expect to see if right\"}",
        "]}",
    ]
    return "\n".join(lines)


def _parse_hypotheses(content: str) -> List[Dict[str, Any]]:
    """Parse and SHAPE-VALIDATE hypothesis list. Returns [] if malformed."""
    if not content:
        return []
    txt = str(content).strip()
    for marker in ("```json", "```JSON", "```"):
        txt = txt.replace(marker, "")
    try:
        start = txt.find("{")
        end = txt.rfind("}")
        if start < 0 or end <= start:
            return []
        data = json.loads(txt[start:end + 1])
    except (json.JSONDecodeError, ValueError):
        return []
    raw_list = data.get("hypotheses")
    if not isinstance(raw_list, list):
        return []
    clean: List[Dict[str, Any]] = []
    for item in raw_list:
        if not isinstance(item, dict):
            continue
        param = str(item.get("parameter", "")).strip()
        if not param:
            continue
        try:
            current_value = float(item.get("current_value"))
            proposed_value = float(item.get("proposed_value"))
        except (TypeError, ValueError):
            continue
        affected = item.get("affected_pairs") or "_any"
        if isinstance(affected, list):
            affected = ",".join(str(p) for p in affected[:20])
        clean.append({
            "parameter": param,
            "current_value": current_value,
            "proposed_value": proposed_value,
            "mechanism": str(item.get("mechanism", ""))[:500],
            "falsification": str(item.get("falsification", ""))[:500],
            "affected_pairs": str(affected)[:500],
            "expected_impact": str(item.get("expected_impact", ""))[:500],
        })
    return clean[:MAX_HYPOTHESES_PER_CYCLE]


def _deflated_sharpe(sharpe: float, n_hypotheses: int) -> float:
    """López de Prado (2014) — DSR correction for multiple testing.

    Simplified form: DSR = Sharpe · (1 − γ·log(N)) where γ is a regime-dependent
    haircut. We use γ=0.15 which is conservative but keeps positive hypotheses
    from being penalised to zero at small N.
    """
    if n_hypotheses <= 1:
        return float(sharpe)
    haircut = 1.0 - 0.15 * math.log(max(n_hypotheses, 2))
    return float(sharpe) * max(0.0, haircut)


class LLMStrategyResearcher:
    """Weekly LLM hypothesis loop with 6-gate validation."""

    def __init__(self, llm_router=None):
        self._router = llm_router
        init_db()

    def _get_router(self):
        if self._router is None:
            try:
                from llm_router import LLMRouter
                self._router = LLMRouter(temperature=0.3)
            except Exception as e:
                logger.debug(f"[Researcher] router init failed: {e}")
                self._router = None
        return self._router

    # ─── Context gathering ─────────────────────────────────────────────
    def _gather_context(self) -> Dict[str, Any]:
        ctx: Dict[str, Any] = {
            "n_trades": 0, "win_rate": 0.0,
            "avg_pnl_pct": 0.0, "avg_hold_hours": 0.0,
            "forgone_alpha_pct": 0.0,
            "top_missed_pair": "n/a",
            "recent_verdicts": [],
        }
        try:
            conn = get_db_connection(AI_DB_PATH)
            rows = conn.execute("""
                SELECT outcome_pnl, outcome_duration FROM ai_decisions
                WHERE outcome_pnl IS NOT NULL
                  AND timestamp > datetime('now', ? || ' days')
            """, (f"-{LOOKBACK_DAYS_FOR_CONTEXT}",)).fetchall()
            if rows:
                n = len(rows)
                wins = sum(1 for r in rows if (r["outcome_pnl"] or 0) > 0)
                total_pnl = sum(float(r["outcome_pnl"] or 0) for r in rows)
                total_dur = sum(float(r["outcome_duration"] or 0) for r in rows)
                ctx.update({
                    "n_trades": n,
                    "win_rate": wins / max(n, 1),
                    "avg_pnl_pct": total_pnl / max(n, 1),
                    "avg_hold_hours": (total_dur / max(n, 1)) / 3600.0,
                })
            # Forgone alpha digest
            try:
                f = conn.execute("""
                    SELECT SUM(forgone_pnl) AS total, pair, COUNT(*) AS n
                    FROM forgone_profit
                    WHERE was_executed = 0 AND forgone_pnl IS NOT NULL
                      AND signal_time > datetime('now', '-14 days')
                    GROUP BY pair
                    ORDER BY total DESC LIMIT 1
                """).fetchone()
                if f:
                    ctx["forgone_alpha_pct"] = float(f["total"] or 0.0)
                    ctx["top_missed_pair"] = f["pair"] or "n/a"
            except Exception:
                pass
            # Recent verdicts
            try:
                verdicts = conn.execute("""
                    SELECT details_json FROM organism_audit
                    WHERE event_type = 'trade_verdict'
                      AND timestamp > datetime('now', '-14 days')
                    ORDER BY id DESC LIMIT 5
                """).fetchall()
                ctx["recent_verdicts"] = [
                    json.loads(r["details_json"]).get("blame", {}).get("root_cause", "")
                    for r in verdicts if r["details_json"]
                ][:5]
            except Exception:
                ctx["recent_verdicts"] = []
            conn.close()
        except Exception as e:
            logger.debug(f"[Researcher] gather_context failed: {e}")
        return ctx

    # ─── 6-gate validation ─────────────────────────────────────────────
    def _gate_complexity(self, hyp: Dict[str, Any]) -> bool:
        return bool(hyp.get("parameter")) and hyp["current_value"] != hyp["proposed_value"]

    def _gate_novelty(self, hyp: Dict[str, Any]) -> bool:
        """Reject if identical (parameter, proposed_value) was already tested
        within the last NOVELTY_LOOKBACK_DAYS."""
        try:
            conn = get_db_connection(AI_DB_PATH)
            hyp_id = _hypothesis_id(hyp["parameter"], hyp["proposed_value"])
            row = conn.execute("""
                SELECT 1 FROM hypothesis_history
                WHERE hypothesis_id = ?
                  AND created_at > datetime('now', ? || ' days')
            """, (hyp_id, f"-{NOVELTY_LOOKBACK_DAYS}")).fetchone()
            conn.close()
            return row is None
        except Exception:
            return True  # on error, treat as novel

    @staticmethod
    def _gate_walk_forward(hyp: Dict[str, Any]) -> Optional[float]:
        """Stub — returns a neutral Sharpe (0.0). A live walk-forward
        backtester is wired in as Sprint 3B task 16b; for now we persist the
        hypothesis but mark it PENDING_BACKTEST so the shadow period is the
        effective validation window."""
        return 0.0

    @staticmethod
    def _gate_oos(hyp: Dict[str, Any]) -> Optional[float]:
        """Stub — matching task 16b wiring point for OOS holdout."""
        return 0.0

    # ─── Persistence ───────────────────────────────────────────────────
    def _persist(self, hyp: Dict[str, Any], is_sharpe: float,
                 oos_sharpe: float, dsr: float,
                 n_in_batch: int,
                 validation: str) -> None:
        hyp_id = _hypothesis_id(hyp["parameter"], hyp["proposed_value"])
        try:
            conn = get_db_connection(AI_DB_PATH)
            conn.execute("""
                INSERT OR IGNORE INTO hypothesis_history
                    (hypothesis_id, parameter, current_value, proposed_value,
                     mechanism, falsification, affected_pairs,
                     is_sharpe, oos_sharpe, deflated_sharpe,
                     n_hypotheses_in_batch, validation_result, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                hyp_id, hyp["parameter"],
                float(hyp["current_value"]), float(hyp["proposed_value"]),
                hyp["mechanism"], hyp["falsification"], hyp["affected_pairs"],
                float(is_sharpe), float(oos_sharpe), float(dsr),
                int(n_in_batch), validation,
                datetime.now(tz=timezone.utc).isoformat(),
            ))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.debug(f"[Researcher] persist failed: {e}")

    # ─── Weekly cycle ──────────────────────────────────────────────────
    async def run_cycle(self) -> Dict[str, Any]:
        """One weekly hypothesis generation pass. Returns summary."""
        router = self._get_router()
        if router is None:
            return {"generated": 0, "accepted": 0, "error": "router unavailable"}

        # RPD headroom check — skip if any critical provider is near its cap.
        try:
            status = router.rpd_status()
            tight = [s for s in status if not s["within_budget"]]
            if len(tight) > len(status) * 0.5:
                logger.info(f"[Researcher] RPD tight on {len(tight)} slots — skipping cycle")
                return {"generated": 0, "accepted": 0, "error": "rpd tight"}
        except Exception:
            pass

        ctx = self._gather_context()
        if ctx["n_trades"] < 5:
            return {"generated": 0, "accepted": 0, "error": "insufficient trade history"}

        try:
            from langchain_core.messages import HumanMessage, SystemMessage
        except Exception as e:
            return {"generated": 0, "accepted": 0, "error": f"langchain missing: {e}"}

        messages = [
            SystemMessage(content="You are a quantitative strategy researcher. "
                                  "Reply with STRICT JSON only."),
            HumanMessage(content=_prompt_template(ctx)),
        ]
        try:
            response = await router.ainvoke(messages, temperature=0.3, priority="medium")
            content = str(getattr(response, "content", ""))
        except Exception as e:
            logger.warning(f"[Researcher] LLM call failed: {e}")
            return {"generated": 0, "accepted": 0, "error": str(e)}

        hypotheses = _parse_hypotheses(content)
        n_generated = len(hypotheses)
        if n_generated == 0:
            return {"generated": 0, "accepted": 0, "error": "no valid hypotheses parsed"}

        accepted = 0
        for hyp in hypotheses:
            if not self._gate_complexity(hyp):
                self._persist(hyp, 0.0, 0.0, 0.0, n_generated, "REJECT_COMPLEXITY")
                continue
            if not self._gate_novelty(hyp):
                self._persist(hyp, 0.0, 0.0, 0.0, n_generated, "REJECT_NOVELTY")
                continue
            is_sharpe = self._gate_walk_forward(hyp) or 0.0
            oos_sharpe = self._gate_oos(hyp) or 0.0
            dsr = _deflated_sharpe(is_sharpe, n_generated)
            # Gates 3-5 are stubs; we persist as PENDING so a human or
            # Sprint 3B backtester can graduate these to DEPLOY.
            self._persist(hyp, is_sharpe, oos_sharpe, dsr, n_generated, "PENDING_BACKTEST")
            accepted += 1

        logger.info(f"[Researcher] cycle done: generated={n_generated}, accepted={accepted}")
        return {"generated": n_generated, "accepted": accepted,
                "rejected": n_generated - accepted,
                "forgone_context": ctx}


_researcher_instance: Optional[LLMStrategyResearcher] = None


def get_researcher() -> LLMStrategyResearcher:
    global _researcher_instance
    if _researcher_instance is None:
        _researcher_instance = LLMStrategyResearcher()
    return _researcher_instance
