import logging
from typing import List, Dict, Any, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from ai_config import AI_DB_PATH
from llm_router import LLMRouter

logger = logging.getLogger(__name__)

class CoTRAG:
    """
    Phase 16 - CoT-RAG (Chain-of-Thought RAG)
    Provides a structured, multi-step reasoning protocol where each step
    is backed by specifically retrieved evidence.
    """
    
    REASONING_STEPS = [
        {"step": "market_state", "label": "Market Regime", "search_query_hint": "crypto market trend regime volatility ADX"},
        {"step": "technical", "label": "Technical Analysis", "search_query_hint": "{pair} technical indicators RSI MACD EMA support resistance volume"},
        {"step": "sentiment", "label": "Sentiment & News", "search_query_hint": "{pair} sentiment Fear Greed CryptoBERT news catalyst"},
        {"step": "risk", "label": "Risk Assessment", "search_query_hint": "{pair} risk volatility downside tail risk correlation"},
        {"step": "decision", "label": "Final Synthesis", "search_query_hint": None}  # Synthesis, no new retrieval
    ]

    def __init__(self, llm_router: Optional[LLMRouter] = None, retriever=None):
        self.router = llm_router or LLMRouter(temperature=0.2)
        self.retriever = retriever

    def reason_step_by_step(self, pair: str, query: str) -> dict:
        """
        Executes a 5-step Chain of Thought reasoning process.
        Returns the steps, final decision, and an audit trail.
        """
        logger.info(f"[CoT-RAG] Initiating structured reasoning for {pair}...")
        
        executed_steps = []
        cumulative_context = ""
        
        for step_config in self.REASONING_STEPS:
            step_result = self._execute_step(step_config, pair, query, cumulative_context)
            executed_steps.append(step_result)
            
            # Append to cumulative history for the next step to reference
            cumulative_context += f"\\n[{step_config['label']}] Analysis: {step_result['analysis']}"

        # The last step is the decision step
        decision_step = executed_steps[-1]
        
        # We parse the decision step for signal and confidence
        # Since it's free-text, we rely on basic heuristic parsing if not strictly JSON formatted
        signal = "NEUTRAL"
        analysis_upper = decision_step["analysis"].upper()
        if "BULLISH" in analysis_upper:
            signal = "BULLISH"
        elif "BEARISH" in analysis_upper:
            signal = "BEARISH"

        # Dynamic confidence from step agreement instead of hardcoded 0.85
        agreement_count = 0
        for step in executed_steps[:-1]:  # Exclude decision step
            step_text = step["analysis"].upper()
            if signal == "BULLISH" and "BULLISH" in step_text:
                agreement_count += 1
            elif signal == "BEARISH" and "BEARISH" in step_text:
                agreement_count += 1
            elif signal == "NEUTRAL" and "NEUTRAL" in step_text:
                agreement_count += 1
        # 4 analysis steps: 0 agree=0.45, 1=0.52, 2=0.58, 3=0.65, 4=0.72
        confidence = 0.45 + (agreement_count * 0.07)

        return {
            "pair": pair,
            "steps": executed_steps,
            "final_decision": signal,
            "confidence": round(confidence, 2),
            "reasoning_chain": cumulative_context.strip()
        }

    def _execute_step(self, step_config: dict, pair: str, base_query: str, previous_context: str) -> dict:
        """Executes a single reasoning step with dedicated retrieval if applicable."""
        step_id = step_config["step"]
        label = step_config["label"]
        hint_template = step_config["search_query_hint"]
        
        evidence_text = ""
        evidence_count = 0
        
        # 1. Retrieve evidence if needed
        if hint_template and self.retriever:
            search_query = hint_template.replace("{pair}", pair)
            try:
                # Retrieve specific context for this step
                results = self.retriever.search(query=search_query, top_k=3)
                evidence_count = len(results)
                docs = [r['text'] if isinstance(r, dict) and 'text' in r else str(r) for r in results]
                evidence_text = "\\n".join(docs)
                logger.info(f"[CoT-RAG] Step '{label}' retrieved {evidence_count} evidence blocks.")
            except Exception as e:
                logger.error(f"[CoT-RAG] Retrieval failed for step '{label}': {e}")
                
        # 2. Formulate specific prompt
        if step_id == "decision":
            prompt = (
                f"You are the Final Synthesis AI. Based on the 4 previous reasoning steps:\\n{previous_context}\\n\\n"
                f"TASK: Provide a definitive trading decision for {pair}.\\n"
                f"1. Weigh the evidence from each step. Which steps AGREE and which CONTRADICT?\\n"
                f"2. If 3+ steps agree on direction, follow the majority.\\n"
                f"3. If steps are split or contradictory, lean NEUTRAL.\\n"
                f"4. State clearly: BULLISH, BEARISH, or NEUTRAL.\\n"
                f"5. Summarize in exactly 2 sentences, citing which steps supported the decision."
            )
        else:
            step_instructions = {
                "market_state": "Classify the market REGIME: TRENDING (ADX>25), RANGING (ADX<20), or HIGH_VOLATILITY (ATR>2x avg). Cite specific data.",
                "technical": "Analyze key indicators (RSI, MACD, EMAs, BBands) with EXACT values from evidence. State TECHNICAL LEAN: BULLISH/BEARISH/NEUTRAL.",
                "sentiment": "Assess crowd sentiment (Fear & Greed, CryptoBERT, news tone). Include contrarian analysis. State SENTIMENT LEAN.",
                "risk": "Identify the TOP 2 risks to a directional trade. What could go wrong? What's the invalidation level? Rate RISK LEVEL: LOW/MEDIUM/HIGH."
            }
            instruction = step_instructions.get(step_id, f"Analyze {label} for {pair}.")

            prompt = (
                f"You are analyzing {label} for {pair}.\\n"
                f"SPECIFIC TASK: {instruction}\\n\\n"
                f"Previous Steps:\\n{previous_context if previous_context else 'None'}\\n\\n"
                f"Fresh Evidence:\\n{evidence_text if evidence_text else 'None'}\\n\\n"
                f"RULES:\\n"
                f"- Cite SPECIFIC data points from the evidence [SOURCE: value]\\n"
                f"- If evidence is insufficient, say 'DATA UNAVAILABLE' — do NOT guess\\n"
                f"- Keep to 2-3 sentences. Do NOT give a final trading signal."
            )

        # 3. Request generation
        try:
            response = self.router.invoke([
                SystemMessage(content=f"You are a specialized crypto analyst executing the '{label}' step of a structured reasoning chain. Be precise, cite data, and never hallucinate."),
                HumanMessage(content=prompt)
            ], priority="medium")
            analysis = str(response.content).strip()
        except Exception as e:
            logger.error(f"[CoT-RAG] LLM Execution failed for '{label}': {e}")
            analysis = f"Error generating analysis for {label}."
            
        return {
            "step": step_id,
            "label": label,
            "analysis": analysis,
            "evidence_count": evidence_count
        }
