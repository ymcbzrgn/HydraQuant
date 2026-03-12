"""
Phase 3.3: Adaptive RAG — Query Routing
Routes queries to the optimal retrieval pipeline based on complexity.

Classification:
  - SIMPLE:  "BTC fiyatı ne?" → single retrieval, no reranking
  - MEDIUM:  "BTC teknik görünümü" → full hybrid search + reranking
  - COMPLEX: "Fed kararı + BTC korelasyonu" → multi-hop + CRAG correction
  - NO_RAG:  "Moving average nasıl hesaplanır?" → LLM-only, skip retrieval
"""

import os
import sys
import logging
import json
from typing import List, Dict, Any, Optional, Literal

sys.path.append(os.path.dirname(__file__))

from llm_router import LLMRouter
from langchain_core.messages import SystemMessage, HumanMessage
from hyde_generator import HyDEGenerator
from rag_fusion import RAGFusion
from self_rag import SelfRAG
from flare_retriever import FLARERetriever
from speculative_rag import SpeculativeRAG

logger = logging.getLogger(__name__)

CLASSIFY_SYSTEM_PROMPT = """You are a query complexity classifier for a crypto trading RAG system.
Classify the query into exactly ONE category:

SIMPLE — Fact lookup, single entity, e.g. "BTC price", "ETH market cap"
MEDIUM — Multi-factor analysis, e.g. "BTC technical outlook", "ETH support/resistance levels"
COMPLEX — Multi-hop reasoning, cross-domain, e.g. "How will Fed rate decision affect BTC/ETH correlation?"
NO_RAG — General knowledge that doesn't need retrieval, e.g. "What is a moving average?", "How does RSI work?"

Output ONLY the category name. No explanation. One word."""


class AdaptiveQueryRouter:
    """
    Adaptive RAG: Routes queries to the optimal pipeline.
    Saves API costs on simple queries, applies full power on complex ones.
    """

    def __init__(self, router: Optional[LLMRouter] = None):
        self.router = router or LLMRouter(temperature=0.1, request_timeout=10)
        # Keyword-based fast classification (avoids LLM call for obvious cases)
        self._simple_keywords = ["price", "fiyat", "market cap", "volume", "hacim"]
        self._no_rag_keywords = ["nedir", "nasıl", "what is", "how does", "explain", "definition"]
        # Phase 3.5: HyDE for COMPLEX queries with low CRAG confidence
        self.hyde = HyDEGenerator(router=self.router)
        # Phase 3.4: RAG-Fusion for MEDIUM queries
        self.rag_fusion = RAGFusion(router=self.router)
        self.self_rag = SelfRAG(router=self.router)
        # Phase 16: FLARE for dynamic generation augmentation
        self.flare = FLARERetriever(llm_router=self.router)
        # Phase 16: Speculative RAG for multi-draft generation
        self.speculative_rag = SpeculativeRAG(llm_router=self.router)

    def classify(self, query: str) -> str:
        """
        Classify query complexity: SIMPLE / MEDIUM / COMPLEX / NO_RAG.
        Uses keyword heuristics first, falls back to LLM for ambiguous cases.
        """
        query_lower = query.lower().strip()

        # Phase 9: Self-RAG Pre-check
        if not self.self_rag.should_retrieve(query, context={}):
            logger.info(f"[AdaptiveRAG] SelfRAG Rule → NO_RAG")
            return "NO_RAG"

        # Fast path: keyword heuristics (no LLM cost)
        if any(kw in query_lower for kw in self._no_rag_keywords) and not any(kw in query_lower for kw in ["news", "current", "today", "latest"]):
            logger.info(f"[AdaptiveRAG] Fast classify → NO_RAG: '{query[:50]}'")
            return "NO_RAG"

        if any(kw in query_lower for kw in self._simple_keywords) and len(query_lower.split()) <= 5:
            logger.info(f"[AdaptiveRAG] Fast classify → SIMPLE: '{query[:50]}'")
            return "SIMPLE"

        # Detect multi-hop: multiple entities or cross-domain indicators
        cross_domain_signals = ["+", "correlation", "korelasyon", "impact", "effect", "affect", "cause"]
        if any(sig in query_lower for sig in cross_domain_signals):
            logger.info(f"[AdaptiveRAG] Fast classify → COMPLEX: '{query[:50]}'")
            return "COMPLEX"

        # Slow path: LLM classification for ambiguous queries
        try:
            messages = [
                SystemMessage(content=CLASSIFY_SYSTEM_PROMPT),
                HumanMessage(content=f"Query: {query}")
            ]
            response = self.router.invoke(messages)
            category = str(response.content).strip().upper()

            if category in ("SIMPLE", "MEDIUM", "COMPLEX", "NO_RAG"):
                logger.info(f"[AdaptiveRAG] LLM classify → {category}: '{query[:50]}'")
                return category
            else:
                logger.warning(f"[AdaptiveRAG] LLM returned unexpected: '{category}'. Defaulting to MEDIUM.")
                return "MEDIUM"

        except Exception as e:
            logger.warning(f"[AdaptiveRAG] Classification failed: {e}. Defaulting to MEDIUM.")
            return "MEDIUM"

    def route(
        self,
        query: str,
        retriever,
        crag_evaluator=None,
    ) -> List[Dict[str, Any]]:
        """
        Route query to the optimal retrieval pipeline.

        Args:
            query: The search query
            retriever: HybridRetriever instance (must have .search() method)
            crag_evaluator: Optional CRAGEvaluator for COMPLEX queries

        Returns:
            List of retrieved documents
        """
        complexity = self.classify(query)

        if complexity == "NO_RAG":
            # Skip retrieval entirely — LLM can answer from training data
            logger.info("[AdaptiveRAG] NO_RAG: Skipping retrieval.")
            return []

        elif complexity == "SIMPLE":
            # Lightweight: fewer results, no CRAG overhead
            results = retriever.search(query, top_k=3)
            logger.info(f"[AdaptiveRAG] SIMPLE: Retrieved {len(results)} docs (top_k=3).")
            return results

        elif complexity == "COMPLEX":
            # Full power: CRAG correction + HyDE fallback
            results = self.hyde.hyde_search(
                query=query,
                retriever=retriever,
                crag_evaluator=crag_evaluator,
                confidence_threshold=0.7
            )
            logger.info(f"[AdaptiveRAG] COMPLEX: Retrieved {len(results)} docs (CRAG + HyDE).")
            return results

        else:  # MEDIUM (default)
            # RAG-Fusion: multi-query for broader coverage
            results = self.rag_fusion.fused_search(query, retriever, n_queries=3, top_k_per_query=5)
            logger.info(f"[AdaptiveRAG] MEDIUM: Retrieved {len(results)} docs (RAG-Fusion 3 queries).")
            return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    router = AdaptiveQueryRouter()

    test_queries = [
        "BTC price",
        "What is a moving average?",
        "BTC/USDT technical analysis support resistance",
        "Fed interest rate decision + BTC correlation impact",
    ]

    for q in test_queries:
        category = router.classify(q)
        print(f"  '{q}' → {category}")
