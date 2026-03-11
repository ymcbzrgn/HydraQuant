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
        {"step": "market_state", "label": "Piyasa Durumu", "search_query_hint": "current market trend"},
        {"step": "technical", "label": "Teknik Analiz", "search_query_hint": "{pair} technical indicators RSI MACD support resistance"},
        {"step": "sentiment", "label": "Sentiment", "search_query_hint": "{pair} news sentiment fear and greed"},
        {"step": "risk", "label": "Risk Değerlendirme", "search_query_hint": "{pair} risk volatility downside"},
        {"step": "decision", "label": "Final Karar", "search_query_hint": None}  # Synthesis, no new retrieval
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
        if "BULLISH" in decision_step["analysis"].upper():
            signal = "BULLISH"
        elif "BEARISH" in decision_step["analysis"].upper():
            signal = "BEARISH"
            
        return {
            "pair": pair,
            "steps": executed_steps,
            "final_decision": signal,
            "confidence": 0.85, # In a full prompt we'd extract this dynamically
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
                f"You are the Final Synthesis AI. Based on the previous reasoning steps:\\n{previous_context}\\n\\n"
                f"Provide a definitive trading decision for {pair}. "
                f"State clearly if it is BULLISH, BEARISH, or NEUTRAL, and summarize your reasoning in 2 sentences."
            )
        else:
            prompt = (
                f"You are an AI analyst focusing on {label} for {pair}.\\n"
                f"Base user query: {base_query}\\n\\n"
                f"Previous Step Context:\\n{previous_context if previous_context else 'None'}\\n\\n"
                f"Fresh Evidence Retrieved:\\n{evidence_text if evidence_text else 'None'}\\n\\n"
                f"Provide a focused 1-2 sentence analysis specifically for {label}. Do not give a final trading signal here."
            )

        # 3. Request generation
        try:
            response = self.router.invoke([
                SystemMessage(content=f"You are executing step: {label}."),
                HumanMessage(content=prompt)
            ])
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
