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

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# AGENT REGISTRY — 7 specialized agent types
# ═══════════════════════════════════════════════════════════════

AGENT_REGISTRY = {
    "TrendFollower": {
        "best_regimes": ["trending_bull", "trending_bear"],
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
    "ReflectionAgent": {
        "best_regimes": ["*"],  # Always relevant — meta-learning
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


class AgentPool:
    """
    Adaptive multi-agent trading debate system.
    Selects agents by regime, weights by track record, runs multi-round debate.
    """

    def __init__(self, db_path: str = AI_DB_PATH, llm_router=None):
        self.db_path = db_path
        self._llm = llm_router
        self._init_tables()

    def _get_conn(self):
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_tables(self):
        """Create agent memory and performance tables."""
        try:
            conn = self._get_conn()
            conn.execute("""
                CREATE TABLE IF NOT EXISTS agent_memory (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    agent_type TEXT NOT NULL,
                    pair TEXT NOT NULL,
                    regime TEXT,
                    signal TEXT NOT NULL,
                    strength REAL,
                    key_argument TEXT,
                    evidence_engine_confidence REAL,
                    final_outcome_pnl REAL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_agent_mem_type "
                        "ON agent_memory(agent_type, regime)")

            conn.execute("""
                CREATE TABLE IF NOT EXISTS agent_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    agent_type TEXT NOT NULL,
                    pair TEXT NOT NULL,
                    regime TEXT,
                    signal TEXT NOT NULL,
                    outcome_pnl REAL,
                    was_correct BOOLEAN,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_agent_perf "
                        "ON agent_performance(agent_type, regime)")
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"[AgentPool:Init] Table creation failed: {e}")

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
                perf_score = win_rate * 0.60 + min(n_signals / 50, 1.0) * 0.40
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
            logger.warning("[AgentPool:Debate] No LLM available. Returning empty.")
            return {"signal": "NEUTRAL", "confidence": 0.0,
                    "reasoning": "No LLM available for agent debate", "source": "AGENT_POOL"}

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

                # Agentic RAG: process any retrieval requests in response
                raw_content = response.content
                raw_content = self._process_retrieval_requests(raw_content, pair)

                parsed = self._parse_agent_response(raw_content)
                positions[agent_name] = parsed
                logger.info(f"[AgentPool:R1] {agent_name} → {parsed.get('direction', '?')} "
                           f"str={parsed.get('strength', 0):.2f}")

            except Exception as e:
                logger.warning(f"[AgentPool:R1] {agent_name} failed: {e}")
                positions[agent_name] = {
                    "direction": "NEUTRAL", "strength": 0.0,
                    "key_argument": f"Agent failed: {e}", "key_risk": "Agent unavailable"
                }

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

        # ── Round 3: ReflectionAgent meta-analysis + final positions ──
        # ReflectionAgent synthesizes what happened in R1+R2 and provides meta-guidance
        if "ReflectionAgent" in agents:
            try:
                r1_summary = "; ".join(f"{n}: {p.get('direction', '?')}({p.get('strength', 0):.0%})"
                                       for n, p in positions.items() if n not in ("ReflectionAgent",))
                r2_revisions = [n for n, p in positions.items()
                               if p.get("round2", {}).get("revised_direction")
                               and p["round2"]["revised_direction"] != p.get("direction")]

                prompt_r3 = (
                    f"Round 3 — META-ANALYSIS for {pair}:\n"
                    f"Round 1 positions: {r1_summary}\n"
                    f"Round 2 revisions: {', '.join(r2_revisions) if r2_revisions else 'None — all defended'}\n\n"
                    f"As ReflectionAgent, provide your meta-analysis in JSON:\n"
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

        # ── Record agent memories ──
        self._record_agent_memories(pair, regime, positions, result.get("confidence", 0))

        return result

    def _process_retrieval_requests(self, response_text: str, pair: str) -> str:
        """Agentic RAG: parse [RETRIEVE: X] tags and inject retrieval results."""
        import re
        pattern = re.compile(r'\[RETRIEVE:\s*(\w+)\]')
        matches = pattern.findall(response_text)
        if not matches:
            return response_text

        for source in matches[:3]:  # Max 3 retrievals per agent
            retrieved = ""
            try:
                if source == "news":
                    from hybrid_retriever import HybridRetriever
                    r = HybridRetriever()
                    results = r.search(f"{pair} latest analysis", top_k=3)
                    retrieved = "\n".join(
                        doc.get("text", "")[:150] for doc in results[:3]
                    ) if results else "No news found."

                elif source == "events":
                    from hybrid_retriever import HybridRetriever
                    from streaming_rag import detect_event_type
                    r = HybridRetriever()
                    event = detect_event_type(pair)
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
            logger.info(f"[AgentPool:AgenticRAG] Retrieved {source} for {pair}")

        return response_text

    def _compute_majority(self, positions: Dict) -> str:
        """Determine majority direction from agent positions."""
        bull, bear = 0, 0
        for name, pos in positions.items():
            if name in ("DevilsAdvocate", "EvidenceValidator"):
                continue
            d = pos.get("direction", "NEUTRAL")
            s = pos.get("strength", 0)
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
                weight = 0.8 + (perf["win_rate"] * 0.4)  # Range: 0.8-1.2

            direction = pos.get("direction", "NEUTRAL")
            # Use round2 revised direction if available
            r2 = pos.get("round2", {})
            if r2.get("revised_direction"):
                direction = r2["revised_direction"]

            strength = float(pos.get("strength", 0.5))
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

        confidence = round(max(0.01, min(confidence, 0.85)), 4)

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

        try:
            data = json.loads(content)
            return {
                "direction": data.get("direction", "NEUTRAL").upper(),
                "strength": min(max(float(data.get("strength", 0.5)), 0.0), 1.0),
                "key_argument": str(data.get("key_argument", ""))[:500],
                "key_risk": str(data.get("key_risk", ""))[:500],
            }
        except (json.JSONDecodeError, ValueError):
            pass

        # Brace extraction fallback
        brace_start = content.find('{')
        brace_end = content.rfind('}')
        if brace_start >= 0 and brace_end > brace_start:
            try:
                data = json.loads(content[brace_start:brace_end + 1])
                return {
                    "direction": data.get("direction", "NEUTRAL").upper(),
                    "strength": min(max(float(data.get("strength", 0.5)), 0.0), 1.0),
                    "key_argument": str(data.get("key_argument", ""))[:500],
                    "key_risk": str(data.get("key_risk", ""))[:500],
                }
            except Exception:
                pass

        # Keyword fallback
        lower = content.lower()
        if "bullish" in lower:
            return {"direction": "BULLISH", "strength": 0.5, "key_argument": content[:200], "key_risk": "Parse failed"}
        elif "bearish" in lower:
            return {"direction": "BEARISH", "strength": 0.5, "key_argument": content[:200], "key_risk": "Parse failed"}
        return {"direction": "NEUTRAL", "strength": 0.3, "key_argument": content[:200], "key_risk": "Parse failed"}

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

    def _record_agent_memories(self, pair: str, regime: str, positions: Dict,
                                evidence_confidence: float):
        """Record what each agent said (for later outcome matching)."""
        try:
            conn = self._get_conn()
            for agent_name, pos in positions.items():
                direction = pos.get("direction", "NEUTRAL")
                r2 = pos.get("round2", {})
                if r2.get("revised_direction"):
                    direction = r2["revised_direction"]

                conn.execute("""
                    INSERT INTO agent_memory
                    (agent_type, pair, regime, signal, strength, key_argument, evidence_engine_confidence)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (agent_name, pair, regime, direction,
                      pos.get("strength", 0.5),
                      pos.get("key_argument", "")[:500],
                      evidence_confidence))
            conn.commit()
            conn.close()
            logger.debug(f"[AgentPool:Memory] {pair} {len(positions)} agent memories recorded")
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
            # Get recent agent memories for this pair
            rows = conn.execute("""
                SELECT agent_type, signal, strength FROM agent_memory
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
