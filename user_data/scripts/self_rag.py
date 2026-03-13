import logging
import re
from typing import Dict, Any, List
from langchain_core.messages import HumanMessage, SystemMessage
from llm_router import LLMRouter

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
        
        prompt = f"""
        You are a strict evaluator. Assess the following response based on the query and evidence.
        
        Query: {query}
        Response: {response}
        Evidence:
        {evidence_text}
        
        Score the following metrics from 0.0 to 1.0 (strict float values):
        1. faithfulness: Is the response fully supported by the provided evidence? (If it hallucinates or says things not in evidence, give a low score. If no evidence was provided, assess if it accurately reflects general knowledge without hallucinating facts).
        2. relevance: Does the response directly answer the query?
        3. confidence: Overall confidence in this assessment.
        
        Output EXACTLY a valid JSON object in this format:
        {{
            "faithfulness": 0.9,
            "relevance": 0.95,
            "confidence": 0.9
        }}
        """
        try:
            critique_response = self.router.invoke([
                SystemMessage(content="You are a strict JSON evaluator. Output ONLY valid JSON."),
                HumanMessage(content=prompt)
            ])
            critique_str = critique_response.content if hasattr(critique_response, "content") else str(critique_response)
            critique_str = re.sub(r'<think>.*?</think>', '', str(critique_str), flags=re.DOTALL)
            critique_str = critique_str.replace("```json", "").replace("```", "").strip()

            if not critique_str:
                logger.warning("[Self-RAG] Empty response from LLM. Failing open.")
                return {"faithfulness": 1.0, "relevance": 1.0, "confidence": 1.0, "passed": True}

            import json
            metrics = json.loads(critique_str)
            
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
