"""
Phase 20: Agent Pool — MiroFish-inspired adaptive multi-agent trading debate.

Instead of fixed 5 agents (MADAM), select agents BY REGIME and weight BY TRACK RECORD.
Each agent has memory of past performance — good agents get more influence, bad ones fade.

Architecture:
  - 7 agent types with specialized system prompts
  - 2 ALWAYS included: DevilsAdvocate + EvidenceValidator
  - 2 selected by regime + performance history
  - Multi-round debate: Position → Cross-examination → Final
  - Post-trade: update agent track records

MiroFish Patterns Used:
  - Agent personality profiles (from oasis_profile_generator.py)
  - Track record-weighted influence (from simulation_config_generator.py influence_weight)
  - Multi-round interaction (from OASIS simulation rounds)
"""

import os
import sys
import json
import sqlite3
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone

sys.path.append(os.path.dirname(__file__))

from ai_config import AI_DB_PATH
from db import get_connection, get_db_connection

# Phase 24: Neural Organism — adaptive parameters
try:
    from neural_organism import _p
except ImportError:
    def _p(param_id, fallback=0.5, regime="_global"):
        return fallback

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# AGENT REGISTRY — 7 specialized agent types
# ═══════════════════════════════════════════════════════════════

AGENT_REGISTRY = {
    "TrendFollower": {
        "best_regimes": ["trending_bull", "trending_bear"],
        "rag_keywords": "trend momentum EMA ADX continuation breakout higher-highs",
        "rag_event_types": ["trend_reversal", "breakout"],
        "system_prompt": (
            "You are TrendFollower — a trend-following trading agent. "
            "You advocate for entering trades IN THE DIRECTION of the established trend. "
            "Your weapons: EMA alignment, ADX strength, price momentum, trend continuation patterns. "
            "WEAKNESS: You perform poorly in ranging/choppy markets — acknowledge this honestly. "
            "If the regime is ranging, reduce your conviction significantly."
        ),
    },
    "MeanReverter": {
        "best_regimes": ["ranging"],
        "rag_keywords": "mean reversion range oversold overbought Bollinger RSI extremes",
        "rag_event_types": ["range_bound", "reversal"],
        "system_prompt": (
            "You are MeanReverter — a mean-reversion trading agent. "
            "You advocate for fading extreme moves and trading AGAINST the crowd at extremes. "
            "Your weapons: RSI extremes, Bollinger Band touches, F&G extreme readings. "
            "WEAKNESS: You get destroyed in trending markets — trend is not your friend. "
            "If the trend is strong (ADX>30), reduce your conviction significantly."
        ),
    },
    "MomentumRider": {
        "best_regimes": ["trending_bull"],
        "rag_keywords": "momentum acceleration MACD histogram volume confirmation higher-timeframe",
        "rag_event_types": ["momentum_surge", "breakout"],
        "system_prompt": (
            "You are MomentumRider — a momentum-based trading agent. "
            "You advocate for joining ACCELERATING momentum, not just existing trends. "
            "Your weapons: RSI>50 momentum zone (2.8x better than oversold per research), "
            "increasing MACD histogram, volume confirmation, higher-timeframe alignment. "
            "WEAKNESS: Momentum can reverse suddenly — always identify the exit trigger."
        ),
    },
    "FundingContrarian": {
        "best_regimes": ["high_volatility", "ranging"],
        "rag_keywords": "funding rate squeeze long short ratio liquidation open interest",
        "rag_event_types": ["funding_extreme", "liquidation_cascade"],
        "system_prompt": (
            "You are FundingContrarian — a contrarian agent that fades crowded trades. "
            "You advocate for positions OPPOSITE to extreme funding rates and L/S ratios. "
            "Your weapons: extreme funding rate (>0.05%), crowded L/S ratios, F&G extremes. "
            "RESEARCH BACKING: Funding rate is the most reliable microstructure signal. "
            "WEAKNESS: The crowd can be right for extended periods in strong trends — "
            "don't fight a steamroller. Confirm with price action before going contrarian."
        ),
    },
    "RiskMinimizer": {
        "best_regimes": ["high_volatility", "transitional"],
        "rag_keywords": "risk volatility drawdown liquidation ATR crash VIX preservation",
        "rag_event_types": ["flash_crash", "volatility_spike"],
        "system_prompt": (
            "You are RiskMinimizer — a risk-first agent that prioritizes capital preservation. "
            "You advocate for SMALLER positions or NEUTRAL when risk is elevated. "
            "Your weapons: ATR volatility, historical max drawdown from backtests, liquidation risk, "
            "regime uncertainty, high VIX, recent crash history. "
            "WEAKNESS: If you always say 'don't trade', you're useless. Only activate when risk is "
            "genuinely above average. In calm markets, step aside and let others decide."
        ),
    },
    "DevilsAdvocate": {
        "best_regimes": ["*"],  # ALWAYS included
        "rag_keywords": "counter-argument contrarian bearish-case bullish-case divergence failure",
        "rag_event_types": ["reversal", "regime_shift"],
        "system_prompt": (
            "You are DevilsAdvocate — your ONLY job is to argue AGAINST the majority. "
            "If most agents say BULLISH → construct the STRONGEST bearish argument. "
            "If most agents say BEARISH → construct the STRONGEST bullish argument. "
            "If NEUTRAL → argue for the most EXTREME position to stress-test the consensus. "
            "You are NOT trying to be right. You are trying to EXPOSE WEAK ARGUMENTS. "
            "If you cannot find a strong counter-argument, SAY SO — 'I cannot find a compelling "
            "counter to the majority view, which increases my confidence in their position.'"
        ),
    },
    "EvidenceValidator": {
        "best_regimes": ["*"],  # ALWAYS included
        "rag_keywords": "factcheck verification data-quality indicator-reading raw-numbers",
        "rag_event_types": [],
        "system_prompt": (
            "You are EvidenceValidator — your ONLY job is to FACT-CHECK other agents' claims. "
            "You have the Evidence Engine FactSheet with RAW NUMBERS. "
            "If an agent says 'RSI is oversold' but FactSheet shows RSI=55 → CALL THEM OUT. "
            "If an agent claims 'strong momentum' but MACD histogram is negative → FLAG IT. "
            "Rate the OVERALL EVIDENCE QUALITY: how many claims are verified vs unverified? "
            "Your verdict: EVIDENCE_STRONG (>80% verified), EVIDENCE_MIXED (50-80%), "
            "EVIDENCE_WEAK (<50% verified). This directly affects final confidence."
        ),
    },
    "MacroCorrelator": {
        "best_regimes": ["*"],  # Always relevant — macro affects all regimes
        "rag_keywords": "macro DXY VIX treasury yields correlation risk-on risk-off FOMC CPI",
        "rag_event_types": ["fomc", "cpi_release", "fed_decision"],
        "system_prompt": (
            "You are MacroCorrelator — you analyze cross-asset macro correlations. "
            "Your weapons: DXY-BTC correlation (21-27x stronger than Gold-BTC per research), "
            "S&P 500 risk-on/off signals, VIX fear gauge, US Treasury yields, Gold as safe haven. "
            "DXY falling + VIX falling = risk-on environment = BULLISH for crypto. "
            "DXY rising + VIX spiking = risk-off = BEARISH for crypto. "
            "CRITICAL: Crypto doesn't trade in a vacuum. Every major quant fund has a macro desk. "
            "If macro says risk-off but technicals say bullish → the macro signal is usually stronger "
            "on 4H+ timeframes. On 1H, technicals can diverge temporarily. "
            "WEAKNESS: Macro is slow-moving. Don't over-weight for short-term trades."
        ),
    },
    "TemporalAnalyst": {
        "best_regimes": ["ranging", "transitional"],
        "rag_keywords": "seasonality day-of-week session hour-of-day expiry unlock options",
        "rag_event_types": ["token_unlock", "options_expiry", "halving"],
        "system_prompt": (
            "You are TemporalAnalyst — you analyze time-based patterns and seasonality. "
            "Your weapons: Day-of-week effects (crypto tends to dip Sunday-Monday, "
            "rally Tuesday-Wednesday per multiple studies). Hour-of-day patterns (Asian session "
            "vs European vs US session). Monthly seasonality (historically BTC stronger in Q4). "
            "Options expiry dates (last Friday of month = max pain magnet). "
            "ALSO: Check if we're near a known event: FOMC meeting, CPI release, "
            "BTC halving anniversary, major token unlock. Events override seasonality. "
            "WEAKNESS: Seasonality is WEAK alpha — never use alone. Only as a tiebreaker "
            "when other signals are ambiguous. If momentum and evidence agree, ignore seasonality."
        ),
    },
    # Phase 27 Task 23 (I2 Ajani): adversarial self-play. ExploiterAgent
    # actively tries to construct a scenario where the majority position
    # would LOSE money; DefenderAgent responds with the specific safeguard
    # that prevents that loss. Both are ALWAYS included (meta role).
    "ExploiterAgent": {
        "best_regimes": ["*"],
        "rag_keywords": "failure stop-hunt whipsaw liquidity-grab reversal trap",
        "rag_event_types": ["flash_crash", "stop_hunt", "reversal"],
        "system_prompt": (
            "You are ExploiterAgent — your ONLY job is to PROPOSE a SPECIFIC "
            "adversarial scenario under which the current majority position "
            "would LOSE money. Attack the reasoning, not the agents. "
            "Output format: one CONCRETE scenario (e.g. 'BTC -3% in next "
            "15min as funding-rate extreme triggers long-squeeze'), one "
            "TARGET weakness ('over-reliance on trend signal in high-funding'), "
            "and an estimated PREDICTED LOSS in percent of position. "
            "If no credible exploit exists, say so — that's a STRONG signal."
        ),
    },
    "DefenderAgent": {
        "best_regimes": ["*"],
        "rag_keywords": "stoploss hedge trailing safeguard circuit-breaker defensive",
        "rag_event_types": ["flash_crash", "volatility_spike"],
        "system_prompt": (
            "You are DefenderAgent — your ONLY job is to RESPOND to the "
            "ExploiterAgent's scenario with the specific safeguard that "
            "would protect the position. Be concrete: cite the exact "
            "stoploss level, sizing reduction, or exit rule that disarms "
            "the exploit. If NO defense exists, say so EXPLICITLY — that "
            "means the exploit is real and sizing should shrink."
        ),
    },
    "ReflectionAgent": {
        "best_regimes": ["*"],  # Always relevant — meta-learning
        "rag_keywords": "performance history accuracy win-rate past-mistakes retrospective",
        "rag_event_types": [],
        "system_prompt": (
            "You are ReflectionAgent — you analyze PAST MISTAKES and SUCCESSES from agent history. "
            "Before every debate, you review: What did our agents predict last time for this pair? "
            "Were they right or wrong? What patterns emerge from recent performance? "
            "Your weapons: agent_performance table (win rates per agent per regime), "
            "agent_memory table (what each agent said and what actually happened). "
            "KEY INSIGHT: If TrendFollower has been wrong 4 times in a row on this pair, "
            "their current opinion should carry LESS weight. If FundingContrarian has been "
            "right 7/10 times this week, their opinion is more valuable. "
            "You don't generate a trading direction. Instead, you provide a META-ANALYSIS: "
            "which agents to trust more today, which to trust less, and what lessons from "
            "recent trades should inform the current decision. "
            "CRITICAL: You are the MEMORY of the team. Without you, agents make the same "
            "mistakes repeatedly. With you, the team learns and improves."
        ),
    },
}


# ═══════════════════════════════════════════════════════════════
# Phase 27 Fix 2C (J4): Argument pattern extraction for quality scoring
# ═══════════════════════════════════════════════════════════════
# Bucket each agent's key_argument into one of these canonical patterns (by regex)
# so that the quality table can accumulate win-rate + avg PnL per (agent, pattern, regime).
ARGUMENT_PATTERNS = {
    "adx_trend":       r"\badx\s*[><=]\s*\d+|\btrend.*strong|\bstrong.*trend",
    "rsi_oversold":    r"\brsi\b.*(oversold|below\s*30|<\s*30)",
    "rsi_overbought":  r"\brsi\b.*(overbought|above\s*70|>\s*70)",
    "funding_extreme": r"\bfunding\b.*(extreme|>\s*0?\.0[5-9]|squeeze)",
    "macd_signal":     r"\bmacd\b.*(cross|histogram|diverg)",
    "volume_anomaly":  r"\bvolume\b.*(spike|anomal|surge|confirm|diverg)",
    "support_level":   r"\b(support|resistance)\b.*(level|zone|broken|held)",
    "ema_alignment":   r"\bema\b.*(align|cross|above|below|stacked)",
    "fng_extreme":     r"\b(fear|greed|f&g|fng)\b.*(extreme|panic|euphoria|<\s*\d+|>\s*\d+)",
    "momentum_strong": r"\bmomentum\b.*(strong|accel|zone|building)",
    "macro_risk_off":  r"\b(dxy|vix|risk-off|treasury)\b",
    "regime_mismatch": r"\bregime\b.*(mismatch|wrong|against)",
    "liquidation":     r"\b(liquidation|cascade|long/short|l/s)\b",
}


class AgentPool:
    """
    Adaptive multi-agent trading debate system.
    Selects agents by regime, weights by track record, runs multi-round debate.
    """

    def __init__(self, db_path: str = AI_DB_PATH, llm_router=None):
        self.db_path = db_path
        self._llm = llm_router
        self._init_tables()
        # Phase 28: Grafeo integration for agent relationship graph
        self._graph_store = None
        try:
            from graph_store import get_graph_store
            self._graph_store = get_graph_store()
        except Exception:
            pass

    def _get_conn(self):
        conn = get_db_connection(self.db_path)
        return conn

    # ═══════════════════════════════════════════════════════════
    # Phase 27 Fix 2A/2C helpers — context injection for agents
    # ═══════════════════════════════════════════════════════════

    def _get_reflection_context(self, pair: str, regime: str,
                                 agents: List[str]) -> str:
        """Fix 2A (J1): Real DB memory block for ReflectionAgent's R3 prompt.

        Pulls 30-day performance per agent + 7-day memory rows for this pair so
        R3 can generate meta-analysis from REAL numbers instead of hallucinating.
        """
        try:
            conn = self._get_conn()
            lines: List[str] = []

            # Per-agent performance in this regime (last 30 days)
            lines.append("=== AGENT PERFORMANCE (last 30 days, regime={}) ===".format(regime))
            any_perf = False
            for name in agents:
                if name == "ReflectionAgent":
                    continue
                row = conn.execute("""
                    SELECT COUNT(*) as total,
                           SUM(CASE WHEN was_correct THEN 1 ELSE 0 END) as wins,
                           ROUND(AVG(outcome_pnl), 2) as avg_pnl
                    FROM agent_performance
                    WHERE agent_type = ? AND regime = ?
                      AND timestamp > datetime('now', '-30 days')
                """, (name, regime)).fetchone()
                total = row["total"] or 0
                wins = row["wins"] or 0
                avg = row["avg_pnl"] or 0.0
                wr = (wins / total * 100.0) if total > 0 else 0.0
                if total > 0:
                    any_perf = True
                lines.append(
                    f"  {name}: {total} signals, {wr:.0f}% WR, avg PnL {avg:+.2f}%"
                )
            if not any_perf:
                lines.append("  (no recent performance rows — ReflectionAgent must "
                             "acknowledge limited history and defer to live factsheet.)")

            # Recent memory rows for this pair (last 7 days)
            rows = conn.execute("""
                SELECT agent_type, signal, strength, key_argument,
                       final_outcome_pnl
                FROM agent_memory
                WHERE pair = ? AND timestamp > datetime('now', '-7 days')
                ORDER BY timestamp DESC LIMIT 20
            """, (pair,)).fetchall()

            if rows:
                lines.append(f"\n=== MEMORY: {pair} (last 7 days) ===")
                for r in rows:
                    outcome = (f"→ {r['final_outcome_pnl']:+.2f}%"
                               if r['final_outcome_pnl'] is not None
                               else "→ PENDING")
                    arg = (r['key_argument'] or "")[:80]
                    strength = r['strength'] if r['strength'] is not None else 0.0
                    lines.append(
                        f"  {r['agent_type']}: {r['signal']}(str={strength:.2f}) "
                        f"{outcome} | {arg}"
                    )
            conn.close()
            return "\n".join(lines)
        except Exception as e:
            logger.debug(f"[AgentPool:R3] Reflection context failed: {e}")
            return "Historical data unavailable this cycle."

    def _get_argument_quality(self, agent_type: str, regime: str) -> str:
        """Fix 2C (J4): Best/worst argument pattern for this agent+regime.

        Injected into R1 prompt so the agent knows which of their historical
        reasoning templates have actually worked.
        """
        try:
            conn = self._get_conn()
            rows = conn.execute("""
                SELECT argument_pattern, times_used, times_correct,
                       avg_pnl_when_used, quality_score
                FROM argument_quality
                WHERE agent_type = ? AND regime = ? AND times_used >= 3
                ORDER BY quality_score DESC
            """, (agent_type, regime)).fetchall()
            conn.close()

            if not rows or len(rows) == 0:
                return ""  # not enough history yet

            best = rows[0]
            best_wr = (best["times_correct"] / best["times_used"] * 100.0
                       if best["times_used"] > 0 else 0.0)
            parts = [f"  BEST in {regime}: '{best['argument_pattern']}' "
                     f"({best_wr:.0f}% acc, {best['times_used']} uses, "
                     f"avg PnL {best['avg_pnl_when_used']:+.2f}%)"]

            if len(rows) > 1:
                worst = rows[-1]
                worst_wr = (worst["times_correct"] / worst["times_used"] * 100.0
                            if worst["times_used"] > 0 else 0.0)
                parts.append(
                    f"  WORST in {regime}: '{worst['argument_pattern']}' "
                    f"({worst_wr:.0f}% acc, {worst['times_used']} uses — DO NOT rely on)"
                )
            return "\n".join(parts)
        except Exception as e:
            logger.debug(f"[AgentPool:R1] Argument quality fetch failed: {e}")
            return ""

    def _extract_argument_pattern(self, argument: str) -> Optional[str]:
        """Fix 2C: Map a free-form key_argument to one of ARGUMENT_PATTERNS."""
        if not argument:
            return None
        import re as _re
        text = argument.lower()
        for name, pattern in ARGUMENT_PATTERNS.items():
            try:
                if _re.search(pattern, text):
                    return name
            except _re.error:
                continue
        return None

    def _update_argument_quality(self, agent_type: str, pattern: str,
                                  regime: str, was_correct: bool,
                                  outcome_pnl: float) -> None:
        """Fix 2C: Upsert argument_quality row after an outcome is known."""
        if not pattern:
            return
        try:
            conn = self._get_conn()
            row = conn.execute("""
                SELECT times_used, times_correct, avg_pnl_when_used
                FROM argument_quality
                WHERE agent_type = ? AND argument_pattern = ? AND regime = ?
            """, (agent_type, pattern, regime)).fetchone()

            if row is None:
                used = 1
                correct = 1 if was_correct else 0
                avg_pnl = float(outcome_pnl)
            else:
                used = (row["times_used"] or 0) + 1
                correct = (row["times_correct"] or 0) + (1 if was_correct else 0)
                prev_avg = row["avg_pnl_when_used"] or 0.0
                avg_pnl = (prev_avg * (used - 1) + float(outcome_pnl)) / used

            quality_score = (correct / used) if used > 0 else 0.5
            from datetime import datetime as _dt, timezone as _tz
            now = _dt.now(tz=_tz.utc).isoformat()

            conn.execute("""
                INSERT INTO argument_quality
                    (agent_type, argument_pattern, regime, times_used,
                     times_correct, avg_pnl_when_used, quality_score, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(agent_type, argument_pattern, regime) DO UPDATE SET
                    times_used = excluded.times_used,
                    times_correct = excluded.times_correct,
                    avg_pnl_when_used = excluded.avg_pnl_when_used,
                    quality_score = excluded.quality_score,
                    updated_at = excluded.updated_at
            """, (agent_type, pattern, regime, used, correct, avg_pnl,
                  quality_score, now))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.debug(f"[AgentPool:ArgQuality] Upsert failed: {e}")

    def _init_tables(self):
        """Ensure agent tables exist (idempotent)."""
        try:
            conn = self._get_conn()
            conn.execute("""CREATE TABLE IF NOT EXISTS agent_memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT, agent_type TEXT NOT NULL,
                pair TEXT NOT NULL, regime TEXT, signal TEXT NOT NULL, strength REAL,
                key_argument TEXT, evidence_engine_confidence REAL,
                final_outcome_pnl REAL, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)""")
            conn.execute("""CREATE TABLE IF NOT EXISTS agent_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT, agent_type TEXT NOT NULL,
                pair TEXT NOT NULL, regime TEXT, signal TEXT NOT NULL,
                outcome_pnl REAL, was_correct BOOLEAN,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)""")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_agent_mem_type ON agent_memory(agent_type, regime)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_agent_perf ON agent_performance(agent_type, regime)")
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"[AgentPool:Init] Table init failed: {e}")

    # ═══════════════════════════════════════════════════════════
    # AGENT SELECTION
    # ═══════════════════════════════════════════════════════════

    def select_agents(self, regime: str, n_variable: int = 3) -> List[str]:
        """Select agents: 2 fixed (DevilsAdvocate, EvidenceValidator) + n_variable by regime+performance.
        With 10 agents, we select 5 total (2 fixed + 3 variable)."""
        selected = ["DevilsAdvocate", "EvidenceValidator"]

        candidates = []
        for name, config in AGENT_REGISTRY.items():
            if name in selected:
                continue
            regimes = config["best_regimes"]
            if "*" in regimes or regime in regimes:
                # Score = regime match bonus + historical performance
                perf = self._get_agent_performance(name, regime)
                win_rate = perf.get("win_rate", 0.50)
                n_signals = perf.get("n_signals", 0)
                # Performance score: favors proven agents but gives newcomers a chance
                perf_score = (win_rate * _p("agent.perf_wr_weight", 0.60) +
                             min(n_signals / _p("agent.perf_exp_normalizer", 50), 1.0) * _p("agent.perf_exp_weight", 0.40))
                candidates.append((name, perf_score))

        candidates.sort(key=lambda x: x[1], reverse=True)
        selected.extend([c[0] for c in candidates[:n_variable]])

        logger.info(f"[AgentPool:Select] Regime={regime} → agents: {selected}")
        return selected

    def _get_agent_performance(self, agent_type: str, regime: str = None) -> Dict:
        """Get historical performance stats for an agent."""
        try:
            conn = self._get_conn()
            if regime:
                rows = conn.execute("""
                    SELECT COUNT(*) as total,
                           SUM(CASE WHEN was_correct THEN 1 ELSE 0 END) as correct,
                           AVG(outcome_pnl) as avg_pnl
                    FROM agent_performance
                    WHERE agent_type = ? AND regime = ?
                """, (agent_type, regime)).fetchone()
            else:
                rows = conn.execute("""
                    SELECT COUNT(*) as total,
                           SUM(CASE WHEN was_correct THEN 1 ELSE 0 END) as correct,
                           AVG(outcome_pnl) as avg_pnl
                    FROM agent_performance
                    WHERE agent_type = ?
                """, (agent_type,)).fetchone()
            conn.close()

            total = rows["total"] or 0
            correct = rows["correct"] or 0
            return {
                "n_signals": total,
                "win_rate": correct / total if total > 0 else 0.50,
                "avg_pnl": float(rows["avg_pnl"]) if rows["avg_pnl"] else 0.0,
            }
        except Exception:
            return {"n_signals": 0, "win_rate": 0.50, "avg_pnl": 0.0}

    # ═══════════════════════════════════════════════════════════
    # MULTI-ROUND DEBATE
    # ═══════════════════════════════════════════════════════════

    def run_debate(self, pair: str, evidence_factsheet: str, regime: str,
                   tech_data: dict, llm=None) -> Dict[str, Any]:
        """
        Run multi-round debate among selected agents.
        Returns: {signal, confidence, reasoning, agent_votes, source}
        """
        from langchain_core.messages import HumanMessage, SystemMessage

        llm_to_use = llm or self._llm
        if not llm_to_use:
            logger.warning(
                "[AgentPool:Debate] No LLM available — returning None "
                "(caller decides fallback; no NEUTRAL pollution)"
            )
            return None

        agents = self.select_agents(regime)
        positions = {}

        # ── Round 1: Each agent states position ──
        for agent_name in agents:
            try:
                perf = self._get_agent_performance(agent_name, regime)
                perf_context = (
                    f"\nYour track record in {regime} regime: "
                    f"{perf['n_signals']} signals, {perf['win_rate']:.0%} win rate, "
                    f"avg P&L {perf['avg_pnl']:+.2f}%. "
                    f"{'Adjust your conviction based on where you historically perform well.' if perf['n_signals'] > 10 else 'No significant history yet — be humble in your conviction.'}"
                )

                # Phase 27 Fix 2C: inject argument-quality feedback so the agent
                # biases toward reasoning patterns that have actually worked.
                arg_feedback = self._get_argument_quality(agent_name, regime)
                arg_block = f"\n\nYOUR ARGUMENT QUALITY HISTORY:\n{arg_feedback}\n" if arg_feedback else ""

                # Agentic RAG: agent can request retrieval by including [RETRIEVE: type]
                retrieval_hint = (
                    "\n\nYou can request evidence by including these tags in your response:\n"
                    "[RETRIEVE: news] — recent crypto news about this pair\n"
                    "[RETRIEVE: events] — similar historical events\n"
                    "[RETRIEVE: patterns] — statistical pattern matching from backtests\n"
                    "Include the tag where you need evidence. It will be resolved before Round 2.\n"
                )

                prompt = (
                    f"Analyze {pair} for a trading decision.\n\n"
                    f"EVIDENCE ENGINE FACTSHEET (verified data — you MUST reference this):\n"
                    f"{evidence_factsheet}\n\n"
                    f"Current regime: {regime}\n"
                    f"{perf_context}"
                    f"{arg_block}"
                    f"{retrieval_hint}\n\n"
                    f"Respond in this EXACT JSON format (no other text):\n"
                    f'{{"direction": "BULLISH" or "BEARISH" or "NEUTRAL", '
                    f'"strength": 0.0 to 1.0, '
                    f'"key_argument": "your strongest point with data citation", '
                    f'"key_risk": "biggest risk to your position"}}'
                )

                response = llm_to_use.invoke(
                    [SystemMessage(content=AGENT_REGISTRY[agent_name]["system_prompt"]),
                     HumanMessage(content=prompt)],
                    temperature=0.4, priority="high"
                )

                # Agentic RAG: process any retrieval requests in response.
                # Phase 27 Fix 2B: agent_name drives agent-specific RAG keywords.
                raw_content = response.content
                raw_content = self._process_retrieval_requests(raw_content, pair, agent_name)

                parsed = self._parse_agent_response(raw_content)
                if parsed is None:
                    logger.warning(
                        f"[AgentPool:R1] {agent_name} parse returned None — "
                        f"skipping (no NEUTRAL pollution)"
                    )
                    continue
                positions[agent_name] = parsed
                logger.info(f"[AgentPool:R1] {agent_name} → {parsed['direction']} "
                           f"str={parsed['strength']:.2f}")

            except Exception as e:
                logger.warning(
                    f"[AgentPool:R1] {agent_name} failed "
                    f"({type(e).__name__}): {e} — skipping (no NEUTRAL pollution)"
                )
                continue

        # ── Round 2: Devil's Advocate cross-examination ──
        majority_dir = self._compute_majority(positions)
        da_challenge = positions.get("DevilsAdvocate", {}).get("key_argument", "No challenge")

        for agent_name in agents:
            if agent_name in ("DevilsAdvocate", "EvidenceValidator"):
                continue  # They already did their job in R1
            try:
                prompt_r2 = (
                    f"Round 2: DevilsAdvocate challenges the {majority_dir} consensus:\n"
                    f'"{da_challenge}"\n\n'
                    f"Do you REVISE your position or DEFEND it? Respond in JSON:\n"
                    f'{{"revised_direction": "BULLISH"/"BEARISH"/"NEUTRAL", '
                    f'"revised_strength": 0.0-1.0, '
                    f'"rebuttal": "your response to the challenge"}}'
                )

                response = llm_to_use.invoke(
                    [SystemMessage(content=AGENT_REGISTRY[agent_name]["system_prompt"]),
                     HumanMessage(content=prompt_r2)],
                    temperature=0.3, priority="medium"
                )

                r2_parsed = self._parse_round2_response(response.content)
                positions[agent_name]["round2"] = r2_parsed
                logger.info(f"[AgentPool:R2] {agent_name} → "
                           f"{'REVISED' if r2_parsed.get('revised_direction') != positions[agent_name].get('direction') else 'DEFENDED'}")

            except Exception as e:
                logger.debug(f"[AgentPool:R2] {agent_name} R2 failed: {e}")

        # ── Round 2b (Phase 27 Task 23): Adversarial self-play ──
        # ExploiterAgent proposes a specific loss scenario; DefenderAgent
        # responds with the safeguard that neutralises it. Both responses
        # are persisted to `exploit_archive` so the nightly regression job
        # can re-probe the strategy against every historical exploit.
        if "ExploiterAgent" in agents and "DefenderAgent" in agents:
            try:
                majority = self._compute_majority(positions)
                exploit_prompt = (
                    f"Round 2b — Adversarial probe for {pair} (regime={regime}).\n"
                    f"Current majority position: {majority}.\n"
                    f"Propose the SPECIFIC scenario where this position loses money. "
                    f"Reply JSON only: "
                    f'{{"scenario": "concrete 1-sentence attack", '
                    f'"target_weakness": "one phrase", '
                    f'"predicted_loss_pct": number}}'
                )
                exploit_response = llm_to_use.invoke(
                    [SystemMessage(content=AGENT_REGISTRY["ExploiterAgent"]["system_prompt"]),
                     HumanMessage(content=exploit_prompt)],
                    temperature=0.4, priority="medium",
                )
                exploit_parsed = self._parse_exploit_response(exploit_response.content)
                defender_prompt = (
                    f"Round 2b — Defender response.\n"
                    f"Exploit scenario: {exploit_parsed.get('scenario', 'none')}\n"
                    f"Target weakness: {exploit_parsed.get('target_weakness', 'none')}\n"
                    f"Respond JSON only: "
                    f'{{"defense": "concrete safeguard or NONE", '
                    f'"neutralises": true or false}}'
                )
                defender_response = llm_to_use.invoke(
                    [SystemMessage(content=AGENT_REGISTRY["DefenderAgent"]["system_prompt"]),
                     HumanMessage(content=defender_prompt)],
                    temperature=0.3, priority="medium",
                )
                defender_parsed = self._parse_defender_response(defender_response.content)

                positions["ExploiterAgent"] = {
                    **positions.get("ExploiterAgent", {}),
                    "round2b": {**exploit_parsed, **defender_parsed},
                }
                # Persist — always. was_validated_by_outcome is NULL until
                # trade closes and post-trade court maps outcome back.
                self._archive_exploit(pair, regime, exploit_parsed, defender_parsed)
                logger.info(
                    f"[AgentPool:R2b] exploit='{exploit_parsed.get('scenario','')[:60]}' "
                    f"defended={defender_parsed.get('neutralises', False)}"
                )
            except Exception as e:
                logger.debug(f"[AgentPool:R2b] adversarial round skipped: {e}")

        # ── Round 3: ReflectionAgent meta-analysis + final positions ──
        # ReflectionAgent synthesizes what happened in R1+R2 and provides meta-guidance
        if "ReflectionAgent" in agents:
            try:
                r1_summary = "; ".join(f"{n}: {p.get('direction', '?')}({p.get('strength', 0):.0%})"
                                       for n, p in positions.items() if n not in ("ReflectionAgent",))
                r2_revisions = [n for n, p in positions.items()
                               if p.get("round2", {}).get("revised_direction")
                               and p["round2"]["revised_direction"] != p.get("direction")]

                # Phase 27 Fix 2A: REAL DB memory — stop making R3 hallucinate.
                reflection_history = self._get_reflection_context(pair, regime, agents)

                prompt_r3 = (
                    f"Round 3 — META-ANALYSIS for {pair}:\n"
                    f"Round 1 positions: {r1_summary}\n"
                    f"Round 2 revisions: {', '.join(r2_revisions) if r2_revisions else 'None — all defended'}\n\n"
                    f"HISTORICAL DATA (from agent_performance + agent_memory tables — REAL, not invented):\n"
                    f"{reflection_history}\n\n"
                    f"As ReflectionAgent, ground your meta-analysis in the numbers above. JSON only:\n"
                    f'{{"trust_most": "agent name with best recent track record", '
                    f'"trust_least": "agent name with worst recent track record", '
                    f'"meta_insight": "key lesson from recent agent performance", '
                    f'"confidence_modifier": -0.10 to +0.10}}'
                )

                response = llm_to_use.invoke(
                    [SystemMessage(content=AGENT_REGISTRY["ReflectionAgent"]["system_prompt"]),
                     HumanMessage(content=prompt_r3)],
                    temperature=0.2, priority="medium"
                )
                r3_parsed = self._parse_round2_response(response.content)
                positions["ReflectionAgent"]["round3"] = r3_parsed
                logger.info(f"[AgentPool:R3] ReflectionAgent meta-analysis complete")

            except Exception as e:
                logger.debug(f"[AgentPool:R3] ReflectionAgent failed: {e}")

        # ── Weighted Synthesis ──
        result = self._weighted_synthesis(pair, positions, regime, evidence_factsheet)
        if result is None:
            logger.info(
                f"[AgentPool:Debate] {pair} synthesis returned None — "
                f"no side-effects recorded, caller decides fallback"
            )
            return None

        # ── Record agent memories ──
        self._record_agent_memories(pair, regime, positions, result.get("confidence", 0))

        # ── Phase 27 Fix 2D: MAGMA graph memory — argued/persuaded/resisted edges ──
        self._record_debate_graph(pair, regime, positions)

        # ── Phase 27 Fix 2E: pheromone deposit for downstream modules ──
        self._deposit_debate_pheromones(pair, regime, positions, result)

        return result

    def _process_retrieval_requests(self, response_text: str, pair: str,
                                     agent_name: str = "") -> str:
        """Agentic RAG: parse [RETRIEVE: X] tags and inject retrieval results.

        Phase 27 Fix 2B (J3): queries are PREFIXED with agent-specific keywords
        from AGENT_REGISTRY[agent_name]['rag_keywords'] so TrendFollower actually
        pulls trend-related news instead of the same generic 'pair latest analysis'
        every agent was receiving.
        """
        import re
        pattern = re.compile(r'\[RETRIEVE:\s*(\w+)\]')
        matches = pattern.findall(response_text)
        if not matches:
            return response_text

        agent_kw = AGENT_REGISTRY.get(agent_name, {}).get("rag_keywords", "") if agent_name else ""
        agent_event_types = AGENT_REGISTRY.get(agent_name, {}).get("rag_event_types", []) if agent_name else []

        for source in matches[:3]:  # Max 3 retrievals per agent
            retrieved = ""
            try:
                if source == "news":
                    from hybrid_retriever import HybridRetriever
                    r = HybridRetriever()
                    query = f"{pair} {agent_kw}".strip() if agent_kw else f"{pair} latest analysis"
                    results = r.search(query, top_k=3)
                    retrieved = "\n".join(
                        doc.get("text", "")[:150] for doc in results[:3]
                    ) if results else "No news found."

                elif source == "events":
                    from hybrid_retriever import HybridRetriever
                    from streaming_rag import detect_event_type
                    r = HybridRetriever()
                    event = detect_event_type(pair)
                    # Phase 27 Fix 2B: prefer agent's own event types when generic detection is blank
                    if event == "general" and agent_event_types:
                        event = agent_event_types[0]
                    if event != "general":
                        results = r.search_similar_events(event, top_k=3)
                        retrieved = "\n".join(
                            doc.get("text", "")[:150] for doc in results[:3]
                        ) if results else "No historical events found."
                    else:
                        retrieved = "No specific event detected."

                elif source == "patterns":
                    from pattern_stat_store import PatternStatStore
                    store = PatternStatStore()
                    stats = store.query(pair=pair)
                    if stats and not stats.get("insufficient_data"):
                        retrieved = (
                            f"Win rate: {stats.get('win_rate', 0):.0%}, "
                            f"Avg PnL: {stats.get('avg_profit_pct', 0):+.2f}%, "
                            f"Trades: {stats.get('matching_trades', 0)}"
                        )
                    else:
                        retrieved = "Insufficient pattern data."

            except Exception as e:
                retrieved = f"Retrieval failed: {e}"

            response_text = response_text.replace(
                f"[RETRIEVE: {source}]",
                f"\n[Retrieved {source}]: {retrieved}\n"
            )
            logger.info(f"[AgentPool:AgenticRAG] Retrieved {source} for {pair} "
                       f"(agent={agent_name or 'unknown'})")

        return response_text

    def _compute_majority(self, positions: Dict) -> str:
        """Determine majority direction from agent positions."""
        bull, bear = 0, 0
        for name, pos in positions.items():
            if name in ("DevilsAdvocate", "EvidenceValidator"):
                continue
            d = pos.get("direction")
            s = pos.get("strength")
            if not d or s is None:
                continue
            if d == "BULLISH":
                bull += s
            elif d == "BEARISH":
                bear += s

        if bull > bear and bull > 0.3:
            return "BULLISH"
        elif bear > bull and bear > 0.3:
            return "BEARISH"
        return "NEUTRAL"

    def _weighted_synthesis(self, pair: str, positions: Dict, regime: str,
                           evidence_factsheet: str) -> Dict:
        """Combine agent positions weighted by track record."""
        bull_score = 0.0
        bear_score = 0.0
        total_weight = 0.0
        agent_votes = {}

        for name, pos in positions.items():
            perf = self._get_agent_performance(name, regime)
            # Weight = base 1.0 × performance modifier
            weight = 1.0
            if perf["n_signals"] >= 10:
                weight = _p("agent.vote_weight_base", 0.8) + (perf["win_rate"] * _p("agent.vote_weight_scale", 0.4))

            direction = pos.get("direction")
            if not direction:
                continue
            # Use round2 revised direction if available
            r2 = pos.get("round2", {})
            if r2.get("revised_direction"):
                direction = r2["revised_direction"]

            strength_raw = pos.get("strength")
            if strength_raw is None:
                continue
            strength = float(strength_raw)
            if r2.get("revised_strength") is not None:
                strength = float(r2["revised_strength"])

            if direction == "BULLISH":
                bull_score += strength * weight
            elif direction == "BEARISH":
                bear_score += strength * weight
            total_weight += weight

            agent_votes[name] = {
                "direction": direction,
                "strength": round(strength, 2),
                "weight": round(weight, 2),
                "win_rate": round(perf["win_rate"], 2),
            }

        # Normalize
        if total_weight > 0:
            bull_norm = bull_score / total_weight
            bear_norm = bear_score / total_weight
        else:
            bull_norm = bear_norm = 0.0

        # Signal determination
        if bull_norm > bear_norm and bull_norm > 0.30:
            signal = "BULLISH"
            confidence = min(bull_norm, 0.85)
        elif bear_norm > bull_norm and bear_norm > 0.30:
            signal = "BEARISH"
            confidence = min(bear_norm, 0.85)
        else:
            signal = "NEUTRAL"
            confidence = max(bull_norm, bear_norm) * 0.5

        # EvidenceValidator verdict adjusts confidence
        ev = positions.get("EvidenceValidator", {})
        ev_arg = ev.get("key_argument", "").lower()
        if "evidence_weak" in ev_arg or "weak" in ev_arg:
            confidence *= 0.85
            logger.info(f"[AgentPool:Synth] EvidenceValidator flagged WEAK evidence → -15%")
        elif "evidence_strong" in ev_arg or "strong" in ev_arg:
            confidence *= 1.05
            logger.info(f"[AgentPool:Synth] EvidenceValidator confirmed STRONG evidence → +5%")

        # Phase 27 Fix 2A: apply ReflectionAgent R3 confidence_modifier (was IGNORED).
        # Clamped to ±0.10 so a single meta-agent can't veto the group.
        r3 = positions.get("ReflectionAgent", {}).get("round3", {}) or {}
        try:
            raw_modifier = r3.get("confidence_modifier", 0)
            r3_modifier = max(-0.10, min(0.10, float(raw_modifier or 0)))
        except (TypeError, ValueError):
            r3_modifier = 0.0
        if r3_modifier != 0.0:
            confidence += r3_modifier
            logger.info(f"[AgentPool:Synth] R3 modifier {r3_modifier:+.2f} → conf={confidence:.4f}")

        confidence = round(min(confidence, 0.85), 4)
        if confidence < 0.10:
            logger.info(
                f"[AgentPool:Synthesis] {pair} confidence too low ({confidence:.4f}) "
                f"after R1+R2+R3 — returning None (no pollution)"
            )
            return None

        # Build reasoning
        reasoning_parts = [f"[AgentPool] {pair} {signal} conf={confidence:.2f}"]
        for name, vote in agent_votes.items():
            reasoning_parts.append(f"{name}: {vote['direction']}({vote['strength']:.0%}) w={vote['weight']:.2f}")
        reasoning = " | ".join(reasoning_parts)

        logger.info(f"[AgentPool:Synthesis] {pair}: {signal} conf={confidence:.2f} "
                   f"(bull={bull_norm:.2f}, bear={bear_norm:.2f})")

        return {
            "signal": signal,
            "confidence": confidence,
            "reasoning": reasoning,
            "agent_votes": agent_votes,
            "source": "AGENT_POOL",
        }

    # ═══════════════════════════════════════════════════════════
    # RESPONSE PARSING
    # ═══════════════════════════════════════════════════════════

    def _parse_agent_response(self, content) -> Dict:
        """Parse agent LLM response into structured dict."""
        if isinstance(content, list):
            content = " ".join([b.get("text", "") for b in content if isinstance(b, dict) and "text" in b])
        content = str(content).strip()

        # Try JSON parse
        import re
        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
        content = content.replace("```json", "").replace("```", "").strip()

        parsed = self._extract_agent_fields(content)
        if parsed is not None:
            return parsed

        brace_start = content.find('{')
        brace_end = content.rfind('}')
        if brace_start >= 0 and brace_end > brace_start:
            parsed = self._extract_agent_fields(content[brace_start:brace_end + 1])
            if parsed is not None:
                return parsed

        logger.warning(
            f"[AgentPool] unparseable R1 response (len={len(content)}) — "
            f"returning None to skip agent"
        )
        return None

    @staticmethod
    def _extract_agent_fields(raw: str) -> Optional[Dict]:
        """Strictly parse an R1 agent JSON response.

        Returns None when the payload is not valid JSON, or when `direction`
        or `strength` are missing. No NEUTRAL/0.0 defaults — callers MUST skip
        an agent whose output cannot be validated, otherwise polluted rows
        poison agent_memory / weighted synthesis downstream.
        """
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            return None
        if not isinstance(data, dict):
            return None

        direction = data.get("direction")
        strength = data.get("strength")
        if not direction or strength is None:
            return None
        try:
            strength_f = float(strength)
        except (TypeError, ValueError):
            return None

        return {
            "direction": str(direction).upper(),
            "strength": min(max(strength_f, 0.0), 1.0),
            "key_argument": str(data.get("key_argument", ""))[:500],
            "key_risk": str(data.get("key_risk", ""))[:500],
        }

    def _parse_round2_response(self, content) -> Dict:
        """Parse Round 2 response."""
        if isinstance(content, list):
            content = " ".join([b.get("text", "") for b in content if isinstance(b, dict) and "text" in b])
        content = str(content).strip()

        import re
        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
        content = content.replace("```json", "").replace("```", "").strip()

        try:
            brace_start = content.find('{')
            brace_end = content.rfind('}')
            if brace_start >= 0 and brace_end > brace_start:
                data = json.loads(content[brace_start:brace_end + 1])
                return {
                    "revised_direction": data.get("revised_direction", "").upper() or None,
                    "revised_strength": float(data.get("revised_strength", 0.5)) if data.get("revised_strength") is not None else None,
                    "rebuttal": str(data.get("rebuttal", ""))[:500],
                }
        except Exception:
            pass

        return {"revised_direction": None, "revised_strength": None, "rebuttal": content[:200]}

    # ═══════════════════════════════════════════════════════════
    # MEMORY & TRACK RECORD
    # ═══════════════════════════════════════════════════════════

    # ═══════════════════════════════════════════════════════════
    # Phase 27 Fix 2D / 2E — MAGMA graph + pheromone outputs
    # ═══════════════════════════════════════════════════════════

    # ═══════════════════════════════════════════════════════════
    # Phase 27 Task 23 — Adversarial self-play helpers
    # ═══════════════════════════════════════════════════════════

    @staticmethod
    def _parse_exploit_response(content) -> Dict[str, Any]:
        """Parse ExploiterAgent's JSON reply (safe fallback on failure)."""
        if isinstance(content, list):
            content = " ".join(
                b.get("text", "") for b in content if isinstance(b, dict) and "text" in b
            )
        txt = str(content).replace("```json", "").replace("```", "").strip()
        try:
            import re as _re
            txt = _re.sub(r"<think>.*?</think>", "", txt, flags=_re.DOTALL)
            s, e = txt.find("{"), txt.rfind("}")
            if s >= 0 and e > s:
                data = json.loads(txt[s:e + 1])
                return {
                    "scenario": str(data.get("scenario", ""))[:400],
                    "target_weakness": str(data.get("target_weakness", ""))[:200],
                    "predicted_loss_pct": float(data.get("predicted_loss_pct", 0) or 0),
                }
        except Exception:
            pass
        return {"scenario": "", "target_weakness": "", "predicted_loss_pct": 0.0}

    @staticmethod
    def _parse_defender_response(content) -> Dict[str, Any]:
        if isinstance(content, list):
            content = " ".join(
                b.get("text", "") for b in content if isinstance(b, dict) and "text" in b
            )
        txt = str(content).replace("```json", "").replace("```", "").strip()
        try:
            import re as _re
            txt = _re.sub(r"<think>.*?</think>", "", txt, flags=_re.DOTALL)
            s, e = txt.find("{"), txt.rfind("}")
            if s >= 0 and e > s:
                data = json.loads(txt[s:e + 1])
                defense = str(data.get("defense", ""))[:400]
                neutralises = bool(data.get("neutralises", False))
                if defense.upper().strip() in ("NONE", "N/A", ""):
                    neutralises = False
                return {"defense": defense, "neutralises": neutralises}
        except Exception:
            pass
        return {"defense": "", "neutralises": False}

    def _archive_exploit(self, pair: str, regime: str,
                          exploit: Dict[str, Any], defense: Dict[str, Any]) -> None:
        """Persist to exploit_archive. TTL = 30 days so old exploits age out."""
        try:
            from datetime import datetime, timezone, timedelta
            now = datetime.now(tz=timezone.utc)
            ttl = now + timedelta(days=30)
            conn = self._get_conn()
            conn.execute("""
                INSERT INTO exploit_archive
                    (pair, regime, exploit_scenario, target_weakness,
                     predicted_loss, was_defended, defense_description,
                     was_validated_by_outcome, created_at, ttl_expiry)
                VALUES (?, ?, ?, ?, ?, ?, ?, NULL, ?, ?)
            """, (
                pair, regime,
                exploit.get("scenario", ""),
                exploit.get("target_weakness", ""),
                float(exploit.get("predicted_loss_pct", 0) or 0),
                1 if defense.get("neutralises") else 0,
                defense.get("defense", ""),
                now.isoformat(), ttl.isoformat(),
            ))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.debug(f"[AgentPool:Exploit] archive failed: {e}")

    def _record_debate_graph(self, pair: str, regime: str, positions: Dict) -> None:
        """Fix 2D (J2): Write `argued_*`, `persuaded`, `resisted` edges so the
        graph_store's add_agent_interaction API is actually exercised (was dead
        code before — magma_edges had 0 debate edges)."""
        if not self._graph_store:
            return
        try:
            pair_node = pair.lower().replace("/", "_").replace(":", "_")
            from datetime import datetime as _dt
            debate_id = f"{pair_node}_{_dt.now().strftime('%Y%m%d_%H%M')}"

            # R1: every agent's declared direction becomes an edge (skip meta-agents)
            for name, pos in positions.items():
                if name == "ReflectionAgent":
                    continue
                raw_dir = pos.get("direction")
                raw_str = pos.get("strength")
                if not raw_dir or raw_str is None:
                    continue
                direction = str(raw_dir).lower()
                strength = float(raw_str)
                try:
                    self._graph_store.add_edge(
                        "entity", name.lower(),
                        f"argued_{direction}", pair_node,
                        weight=strength,
                        metadata={
                            "key_argument": (pos.get("key_argument", "") or "")[:200],
                            "regime": regime,
                            "debate_id": debate_id,
                        },
                    )
                except Exception as e:
                    logger.debug(f"[AgentPool:Graph] argued edge {name} failed: {e}")

            # R2: was the agent persuaded by DevilsAdvocate or did they resist?
            for name, pos in positions.items():
                if name in ("DevilsAdvocate", "EvidenceValidator", "ReflectionAgent"):
                    continue
                r2 = pos.get("round2", {}) or {}
                revised = r2.get("revised_direction")
                if not revised:
                    continue
                try:
                    if revised != pos.get("direction"):
                        # Persuaded: devilsadvocate → name
                        self._graph_store.add_agent_interaction(
                            "devilsadvocate", name.lower(),
                            interaction_type="persuaded", weight=1.0,
                        )
                    else:
                        # Defended their position against DA — mutual resistance.
                        self._graph_store.add_agent_interaction(
                            name.lower(), "devilsadvocate",
                            interaction_type="resisted", weight=1.0,
                        )
                except Exception as e:
                    logger.debug(f"[AgentPool:Graph] R2 edge {name} failed: {e}")
        except Exception as e:
            logger.debug(f"[AgentPool:Graph] debate graph failed: {e}")

    def _deposit_debate_pheromones(self, pair: str, regime: str,
                                     positions: Dict, result: Dict) -> None:
        """Fix 2E (J5): Publish debate outcome to the pheromone field so
        downstream modules (sizing, organism) can read CONSENSUS / DISSENT."""
        try:
            from pheromone_field import get_pheromone_field, PheromoneField
            pfield = get_pheromone_field()

            pfield.deposit(
                "agent_pool", PheromoneField.SIGNAL_AGENT_CONSENSUS,
                {
                    "signal": result.get("signal", "NEUTRAL"),
                    "confidence": float(result.get("confidence", 0.0)),
                    "n_agents": len(positions),
                    "pair": pair,
                    "regime": regime,
                },
                half_life=120.0,  # 2 minutes — visible to the next perception cycle
            )

            # Detect a meaningful two-sided split (Fix 2E dissent signal)
            bull_s = sum(float(p.get("strength", 0) or 0)
                         for name, p in positions.items()
                         if p.get("direction") == "BULLISH"
                         and name not in ("DevilsAdvocate", "EvidenceValidator"))
            bear_s = sum(float(p.get("strength", 0) or 0)
                         for name, p in positions.items()
                         if p.get("direction") == "BEARISH"
                         and name not in ("DevilsAdvocate", "EvidenceValidator"))
            if bull_s > 0.3 and bear_s > 0.3:
                pfield.deposit(
                    "agent_pool", PheromoneField.SIGNAL_AGENT_DISSENT,
                    {
                        "bull_strength": round(bull_s, 3),
                        "bear_strength": round(bear_s, 3),
                        "pair": pair,
                        "regime": regime,
                    },
                    half_life=120.0,
                )
                logger.info(
                    f"[AgentPool:Pheromone] dissent deposited — bull={bull_s:.2f}, bear={bear_s:.2f}"
                )
        except Exception as e:
            logger.debug(f"[AgentPool:Pheromone] deposit failed: {e}")

    def _record_agent_memories(self, pair: str, regime: str, positions: Dict,
                                evidence_confidence: float):
        """Record what each agent said (for later outcome matching).

        Guards (prevents NEUTRAL/0.0 pollution in agent_memory):
          1. Missing direction or strength → skip that agent row.
          2. Synthesis evidence_confidence <= 0.01 → skip the whole write
             (synthesis had no real signal; storing it poisons track records).
          3. R1 fallback `NEUTRAL` + strength 0 → skip that row.
        """
        if evidence_confidence is None or evidence_confidence <= 0.01:
            logger.info(
                f"[AgentPool:Memory] {pair} skip — evidence_confidence "
                f"{evidence_confidence} too low, no real signal to record"
            )
            return

        inserted = 0
        skipped = 0
        try:
            conn = self._get_conn()
            for agent_name, pos in positions.items():
                direction = pos.get("direction")
                strength_raw = pos.get("strength")
                if not direction or strength_raw is None:
                    skipped += 1
                    continue
                try:
                    strength = float(strength_raw)
                except (TypeError, ValueError):
                    skipped += 1
                    continue

                # Phase 27 audit fix: R2 revised_direction AND revised_strength must
                # BOTH be taken into account — previously strength was pinned to R1.
                r2 = pos.get("round2", {}) or {}
                if r2.get("revised_direction"):
                    direction = r2["revised_direction"]
                if r2.get("revised_strength") is not None:
                    try:
                        strength = float(r2["revised_strength"])
                    except (TypeError, ValueError):
                        pass

                if direction == "NEUTRAL" and strength <= 0.0:
                    skipped += 1
                    continue

                conn.execute("""
                    INSERT INTO agent_memory
                    (agent_type, pair, regime, signal, strength, key_argument, evidence_engine_confidence)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (agent_name, pair, regime, direction,
                      strength,
                      pos.get("key_argument", "")[:500],
                      evidence_confidence))
                inserted += 1
            conn.commit()
            conn.close()
            logger.debug(
                f"[AgentPool:Memory] {pair} inserted={inserted} skipped={skipped}"
            )
        except Exception as e:
            logger.debug(f"[AgentPool:Memory] {pair} recording failed: {e}")

    def record_trade_outcome(self, pair: str, outcome_pnl: float,
                              regime: str = None, signal: str = None):
        """
        Called from strategy confirm_trade_exit.
        Updates agent_performance based on what each agent predicted vs actual outcome.
        Also records EvidenceEngine outcome even when no agent debate occurred.
        """
        try:
            conn = self._get_conn()
            # Get recent agent memories for this pair (Phase 27: also fetch key_argument
            # so we can update argument_quality with the outcome).
            rows = conn.execute("""
                SELECT agent_type, signal, strength, key_argument FROM agent_memory
                WHERE pair = ? AND timestamp > datetime('now', '-6 hours')
                ORDER BY timestamp DESC LIMIT 10
            """, (pair,)).fetchall()

            updated = 0
            for row in rows:
                agent_signal = row["signal"]
                if signal:
                    was_correct = (agent_signal == signal and outcome_pnl > 0) or \
                                  (agent_signal != signal and agent_signal == "NEUTRAL" and outcome_pnl < 0)
                else:
                    was_correct = (outcome_pnl > 0 and agent_signal in ("BULLISH", "BEARISH")) or \
                                  (outcome_pnl < 0 and agent_signal == "NEUTRAL")

                conn.execute("""
                    INSERT INTO agent_performance
                    (agent_type, pair, regime, signal, outcome_pnl, was_correct)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (row["agent_type"], pair, regime, row["signal"],
                      outcome_pnl, was_correct))
                updated += 1

                # Phase 27 Fix 2C: update argument_quality for the pattern this
                # agent leaned on, so the R1 feedback loop learns which rationales
                # actually win money in each regime.
                pattern = self._extract_argument_pattern(row["key_argument"] or "")
                if pattern:
                    self._update_argument_quality(
                        row["agent_type"], pattern, regime or "_global",
                        bool(was_correct), float(outcome_pnl)
                    )

            # Update memory records with outcome
            if rows:
                conn.execute("""
                    UPDATE agent_memory SET final_outcome_pnl = ?
                    WHERE id IN (
                        SELECT id FROM agent_memory
                        WHERE pair = ? AND final_outcome_pnl IS NULL
                        ORDER BY timestamp DESC LIMIT ?
                    )
                """, (outcome_pnl, pair, len(rows)))

            # Always record EvidenceEngine outcome — even without agent debate
            # This ensures performance tracking works from day 1
            if updated == 0 and signal:
                was_correct = outcome_pnl > 0  # Trade was profitable = signal was correct
                conn.execute("""
                    INSERT INTO agent_performance
                    (agent_type, pair, regime, signal, outcome_pnl, was_correct)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, ("EvidenceEngine", pair, regime, signal, outcome_pnl, was_correct))
                updated = 1

            conn.commit()
            conn.close()

            # Phase 28: Record agent→pair relationship in Grafeo
            if self._graph_store and updated > 0:
                try:
                    for row in rows:
                        self._graph_store.add_edge(
                            "entity", row["agent_type"].lower(), "traded",
                            pair.lower().replace("/", "_"),
                            weight=abs(outcome_pnl) / 10.0,
                            metadata={"pnl": outcome_pnl, "signal": row["signal"],
                                      "regime": regime or "unknown"})
                except Exception:
                    pass

            logger.info(f"[AgentPool:Outcome] {pair} → {outcome_pnl:+.2f}%, "
                       f"updated {updated} agent records")
        except Exception as e:
            logger.warning(f"[AgentPool:Outcome] {pair} update failed: {e}")

    def rebalance_weights(self):
        """
        Weekly job: Log agent performance summary.
        Bad agents naturally lose weight through _get_agent_performance win_rate.
        No explicit weight manipulation needed — the selection score handles it.
        """
        try:
            conn = self._get_conn()
            rows = conn.execute("""
                SELECT agent_type, regime,
                       COUNT(*) as n,
                       SUM(CASE WHEN was_correct THEN 1 ELSE 0 END) as correct,
                       AVG(outcome_pnl) as avg_pnl
                FROM agent_performance
                WHERE timestamp > datetime('now', '-30 days')
                GROUP BY agent_type, regime
                ORDER BY agent_type, regime
            """).fetchall()
            conn.close()

            if not rows:
                logger.info("[AgentPool:Rebalance] No performance data yet.")
                return

            for r in rows:
                wr = (r["correct"] / r["n"] * 100) if r["n"] > 0 else 0
                logger.info(f"[AgentPool:Rebalance] {r['agent_type']} ({r['regime']}): "
                           f"{r['n']} signals, {wr:.0f}% win rate, avg_pnl={r['avg_pnl']:+.2f}%")

        except Exception as e:
            logger.error(f"[AgentPool:Rebalance] Failed: {e}")

    def get_performance_summary(self) -> List[Dict]:
        """Get performance stats for all agents (for API endpoint)."""
        try:
            conn = self._get_conn()
            rows = conn.execute("""
                SELECT agent_type, regime,
                       COUNT(*) as n_signals,
                       SUM(CASE WHEN was_correct THEN 1 ELSE 0 END) as correct,
                       AVG(outcome_pnl) as avg_pnl
                FROM agent_performance
                GROUP BY agent_type, regime
                ORDER BY agent_type
            """).fetchall()
            conn.close()
            return [dict(r) for r in rows]
        except Exception:
            return []
