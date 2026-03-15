"""
Phase 3.4: RAG-Fusion (Multi-Query Retrieval)
Generates multiple search perspectives from a single query, retrieves for each,
then fuses results with Reciprocal Rank Fusion.

Paper: "RAG-Fusion: A New Approach" (Raudaschl, 2023)
"""

import os
import sys
import logging
from typing import List, Dict, Any, Optional

sys.path.append(os.path.dirname(__file__))

from llm_router import LLMRouter
from langchain_core.messages import SystemMessage, HumanMessage

logger = logging.getLogger(__name__)

QUERY_GEN_SYSTEM_PROMPT = """You are a multi-perspective query generator for a crypto trading RAG system.
Generate exactly {n} DIFFERENT search queries that approach the topic from different angles.

PERSPECTIVE CATEGORIES (use a different one for each query):
1. Technical: price action, indicators, chart patterns (e.g., "BTC RSI MACD support resistance levels")
2. Sentiment: crowd psychology, fear/greed, social media (e.g., "Bitcoin sentiment Fear Greed Index today")
3. Fundamental: news, adoption, development, regulation (e.g., "Bitcoin ETF SEC regulatory news 2026")
4. On-chain: whale activity, exchange flows, mining (e.g., "BTC whale accumulation exchange outflows")
5. Macro: Fed rates, DXY, inflation, cross-market (e.g., "Federal Reserve crypto impact correlation")

RULES:
- Each query MUST target a DIFFERENT perspective from the list above
- Keep queries concise (5-10 words each)
- Include the coin/pair name in each query
- Output ONE query per line, no numbering, no explanation, no markdown
- Do NOT repeat the original query"""


class RAGFusion:
    """
    RAG-Fusion: generates N diverse sub-queries from one query,
    retrieves for each, then fuses results with RRF.
    """

    def __init__(self, router: Optional[LLMRouter] = None):
        self.router = router or LLMRouter(temperature=0.7, request_timeout=10)

    def generate_queries(self, original_query: str, n: int = 3) -> List[str]:
        """
        Generate N diverse search queries from the original.

        Example:
          Input:  "BTC ne olacak?"
          Output: [
            "BTC teknik analiz RSI MACD görünümü",
            "Bitcoin haberleri son 24 saat sentiment",
            "BTC on-chain whale hareketleri"
          ]
        """
        prompt = QUERY_GEN_SYSTEM_PROMPT.format(n=n)
        messages = [
            SystemMessage(content=prompt),
            HumanMessage(content=f"Original query: {original_query}")
        ]

        try:
            response = self.router.invoke(messages)
            raw = str(response.content).strip()
            queries = [q.strip() for q in raw.split('\n') if q.strip()]

            # Always include original query as first perspective
            all_queries = [original_query] + queries[:n]
            logger.info(f"[RAG-Fusion] Generated {len(all_queries)} queries from: '{original_query[:50]}'")
            return all_queries

        except Exception as e:
            logger.warning(f"[RAG-Fusion] Query generation failed: {e}. Using original only.")
            return [original_query]

    def fused_search(
        self,
        query: str,
        retriever,
        n_queries: int = 3,
        top_k_per_query: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Multi-query retrieval + RRF fusion.

        Args:
            query: Original search query
            retriever: HybridRetriever instance
            n_queries: Number of sub-queries to generate
            top_k_per_query: Results per sub-query

        Returns:
            Fused and deduplicated results
        """
        queries = self.generate_queries(query, n=n_queries)

        # Retrieve for each sub-query
        all_id_lists = []
        all_docs_by_id = {}

        for sub_query in queries:
            results = retriever.search(sub_query, top_k=top_k_per_query)
            id_list = []
            for r in results:
                doc_id = r.get('id', r.get('text', '')[:50])
                id_list.append(doc_id)
                if doc_id not in all_docs_by_id:
                    all_docs_by_id[doc_id] = r
            all_id_lists.append(id_list)

        # RRF fusion across all sub-query results
        fused_ids = self._rrf_fusion(all_id_lists)

        # Reconstruct final document list
        fused_docs = []
        for doc_id in fused_ids:
            if doc_id in all_docs_by_id:
                fused_docs.append(all_docs_by_id[doc_id])

        logger.info(f"[RAG-Fusion] Fused {len(fused_docs)} unique docs from {len(queries)} sub-queries.")
        return fused_docs

    def _rrf_fusion(self, results_lists: List[List[str]], k: int = 60) -> List[str]:
        """Reciprocal Rank Fusion across multiple ranked lists."""
        rrf_scores: Dict[str, float] = {}
        for ranked_list in results_lists:
            for rank, doc_id in enumerate(ranked_list):
                if doc_id not in rrf_scores:
                    rrf_scores[doc_id] = 0.0
                rrf_scores[doc_id] += 1.0 / (k + rank + 1)

        sorted_fused = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return [doc_id for doc_id, score in sorted_fused]
