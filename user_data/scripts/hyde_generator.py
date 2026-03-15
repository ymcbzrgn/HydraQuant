"""
Phase 3.5: HyDE (Hypothetical Document Embeddings)
When normal retrieval confidence is low, generates a hypothetical answer
and uses it as the query — dramatically improving semantic similarity matching.

Paper: "Precise Zero-Shot Dense Retrieval without Relevance Labels" (Gao et al., 2022)
"""

import os
import sys
import logging
from typing import List, Dict, Any, Optional

sys.path.append(os.path.dirname(__file__))

from llm_router import LLMRouter
from langchain_core.messages import SystemMessage, HumanMessage

logger = logging.getLogger(__name__)

HYDE_SYSTEM_PROMPT = """You are a senior crypto market analyst writing a brief factual research note.
Given a question about crypto markets, write a SHORT (2-3 sentence) hypothetical answer
as if you had access to all current market data and perfect information.

RULES:
1. Write ONLY the answer — no disclaimers, no "I think", no hedging, no preamble.
2. Be SPECIFIC with numbers, price levels, indicator values, and dates (even if hypothetical).
3. Use DOMAIN-SPECIFIC terminology: "RSI oversold at 28", "testing the 200-day EMA at $62K", "Fear & Greed at 25".
4. Include KEYWORD-DENSE phrases that would appear in real market analysis documents (these improve retrieval).
5. NEVER say "I don't know" — always generate a plausible hypothetical document.

GOOD EXAMPLE:
Question: "What is the current BTC outlook?"
Answer: "Bitcoin is trading at $67,500 after bouncing from the 200-day EMA support at $62,000. RSI at 58 is neutral with bullish MACD crossover on the daily. Next resistance at $72,000 (previous local high), with Fear & Greed at 55 (Neutral)."

BAD EXAMPLE:
"I think Bitcoin might go up or down depending on market conditions." (Vague, no data, useless for retrieval)"""


class HyDEGenerator:
    """
    Hypothetical Document Embeddings: generates a fake-but-plausible answer
    to a query, then searches for documents similar to that answer.
    """

    def __init__(self, router: Optional[LLMRouter] = None):
        self.router = router or LLMRouter(temperature=0.5, request_timeout=10)

    def generate_hypothetical(self, query: str) -> str:
        """
        Generate a hypothetical answer document for the query.
        
        Example:
          Query: "BTC ne olacak?"
          HyDE:  "BTC teknik olarak RSI oversold bölgesinde, destek 95K'da tuttu,
                  hacim artışı var, kısa vadede 100K test edilecek."
        """
        messages = [
            SystemMessage(content=HYDE_SYSTEM_PROMPT),
            HumanMessage(content=f"Question: {query}")
        ]

        try:
            response = self.router.invoke(messages)
            hypothetical = str(response.content).strip()
            logger.info(f"[HyDE] Generated hypothetical ({len(hypothetical)} chars) for: '{query[:50]}'")
            return hypothetical
        except Exception as e:
            logger.warning(f"[HyDE] Generation failed: {e}. Using original query.")
            return query

    def hyde_search(
        self,
        query: str,
        retriever,
        crag_evaluator=None,
        confidence_threshold: float = 0.7,
    ) -> List[Dict[str, Any]]:
        """
        HyDE-enhanced search pipeline:
        1. Normal search
        2. CRAG evaluate → confidence score
        3. If confidence < threshold → generate hypothetical → search again
        4. Merge results (original + HyDE), deduplicate

        Args:
            query: The search query
            retriever: HybridRetriever instance
            crag_evaluator: Optional CRAGEvaluator for quality check
            confidence_threshold: Below this, trigger HyDE

        Returns:
            List of retrieved documents
        """
        # Step 1: Normal search
        normal_results = retriever.search(query, top_k=10)

        if not crag_evaluator:
            return normal_results

        # Step 2: Evaluate retrieval quality
        verdict, score, reason = crag_evaluator.evaluate_retrieval(query, normal_results)

        if score >= confidence_threshold:
            logger.info(f"[HyDE] Normal retrieval sufficient (score={score:.2f}). Skipping HyDE.")
            return normal_results

        # Step 3: Generate hypothetical and re-search
        logger.info(f"[HyDE] Low confidence ({score:.2f}). Generating hypothetical document...")
        hypothetical = self.generate_hypothetical(query)

        # Search using hypothetical as query
        hyde_results = retriever.search(hypothetical, top_k=10)

        # Step 4: Merge + deduplicate (HyDE results first, then normal)
        seen_ids = set()
        merged = []

        for result in hyde_results + normal_results:
            doc_id = result.get('id', result.get('text', '')[:50])
            if doc_id not in seen_ids:
                seen_ids.add(doc_id)
                merged.append(result)

        logger.info(f"[HyDE] Merged {len(merged)} unique docs (normal={len(normal_results)}, hyde={len(hyde_results)})")
        return merged
