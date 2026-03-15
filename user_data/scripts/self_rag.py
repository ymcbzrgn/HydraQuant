import logging
import re
from typing import Dict, Any, List
from langchain_core.messages import HumanMessage, SystemMessage
from llm_router import LLMRouter
from json_utils import extract_json_strict

logger = logging.getLogger(__name__)

class SelfRAG:
    """
    Implements Self-RAG logic:
    1. Fast rule-based retrieval gating (decides if RAG is even needed).
    2. LLM-as-a-judge critique of the generated response (faithfulness, relevance).
    """
    
    def __init__(self, router: LLMRouter = None):
        self.router = router or LLMRouter()

    def should_retrieve(self, query: str, context: dict) -> bool:
        """
        Fast rule-based check to decide if retrieval is needed.
        Returns False if retrieval should be skipped, True if it should happen.
        """
        query_lower = query.lower()
        
        # Rule 1: Direct price/market data questions that shouldn't search news
        price_keywords = ["fiyat", "price", "ne kadar", "kaç dolar", "current price", "ticker"]
        if any(kw in query_lower for kw in price_keywords):
            logger.info("Self-RAG: Query is price-related. Skipping RAG.")
            return False
            
        # Rule 2: Technical indicator calculations that only need OHLCV
        tech_keywords = ["rsi", "macd", "bb", "bollinger", "ema", "sma", "support", "resistance", "hesapla", "calculate"]
        if any(kw in query_lower for kw in tech_keywords):
            # If it's specifically about technical *analysis* or computing an indicator
            logger.info("Self-RAG: Query assumes technical indicators. Skipping RAG.")
            return False
            
        # Default: retrieve
        return True

    def self_critique(self, query: str, response: str, evidence: List[str]) -> Dict[str, Any]:
        """
        Evaluates the generated response using an LLM.
        Checks for faithfulness to evidence and relevance to the query.
        """
        evidence_text = "\n".join(evidence) if evidence else "No evidence provided."
        
        prompt = f"""Evaluate this response against the provided evidence.

Query: {query}
Response: {response}
Evidence:
{evidence_text}

SCORING CRITERIA (0.0 to 1.0):

1. faithfulness: Is the response FULLY supported by evidence?
   - 1.0: Every claim traces back to a specific evidence passage
   - 0.7: Most claims supported, 1-2 minor unsupported statements
   - 0.5: Mixed — some supported, some clearly hallucinated
   - 0.3: Mostly unsupported or contradicts evidence
   - 0.0: Completely fabricated, ignores evidence entirely
   RED FLAGS (auto-reduce to <0.4): Fabricated numbers, phantom events, claims contradicting evidence

2. relevance: Does the response DIRECTLY answer the query?
   - 1.0: Directly and completely answers what was asked
   - 0.7: Answers the query but includes unnecessary tangents
   - 0.5: Partially answers — addresses the topic but misses the specific question
   - 0.0: Completely off-topic

3. confidence: YOUR confidence in this evaluation (meta-score)
   - High (0.8-1.0): Evidence is clear, easy to verify claims
   - Medium (0.5-0.7): Some ambiguity in evidence, harder to verify
   - Low (0.0-0.4): Evidence is sparse, evaluation is uncertain

Output EXACTLY a valid JSON object:
{{"faithfulness": 0.XX, "relevance": 0.XX, "confidence": 0.XX}}"""
        try:
            critique_response = self.router.invoke([
                SystemMessage(content="You are a strict RAG quality evaluator. Your ENTIRE response must be a single valid JSON object. No text before or after."),
                HumanMessage(content=prompt)
            ])
            critique_str = critique_response.content if hasattr(critique_response, "content") else str(critique_response)

            if not str(critique_str).strip():
                logger.warning("[Self-RAG] Empty response from LLM. Failing open.")
                return {"faithfulness": 1.0, "relevance": 1.0, "confidence": 1.0, "passed": True}

            metrics = extract_json_strict(str(critique_str), required_keys=["faithfulness"])
            if metrics is None:
                logger.warning(f"[Self-RAG] JSON extraction failed. Failing open. Raw: {str(critique_str)[:200]}")
                return {"faithfulness": 1.0, "relevance": 1.0, "confidence": 1.0, "passed": True}
            
            f_score = float(metrics.get("faithfulness", 0.0))
            r_score = float(metrics.get("relevance", 0.0))
            c_score = float(metrics.get("confidence", 0.0))
            
            # Condition for rejecting the response
            passed = f_score >= 0.5 
            if not passed:
                logger.warning(f"Self-RAG Critique FAILED! Faithfulness: {f_score} < 0.5")
                
            return {
                "faithfulness": f_score,
                "relevance": r_score,
                "confidence": c_score,
                "passed": passed
            }
        except Exception as e:
            logger.error(f"Error during self_critique: {e}")
            # If critique fails, fail open (assume passed) to not block the pipeline
            return {"faithfulness": 1.0, "relevance": 1.0, "confidence": 1.0, "passed": True}
