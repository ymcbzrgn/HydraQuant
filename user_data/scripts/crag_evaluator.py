"""
Phase 3.2: Corrective RAG (CRAG) Evaluator
Highest-ROI missing RAG technique from ROADMAP.

After retrieval, CRAG evaluates whether the retrieved documents actually answer
the query. Based on a relevance score:
  - CORRECT  (>0.8): Use documents as-is
  - AMBIGUOUS (0.4-0.8): Rewrite query and retry retrieval
  - INCORRECT (<0.4): Fall back to web search (DuckDuckGo)
"""

import os
import sys
import logging
import json
import re
from typing import List, Dict, Any, Optional, Tuple

sys.path.append(os.path.dirname(__file__))

from llm_router import LLMRouter
from langchain_core.messages import SystemMessage, HumanMessage

logger = logging.getLogger(__name__)

RELEVANCE_SYSTEM_PROMPT = """You are a strict relevance judge for a crypto trading AI.
Given a QUERY and a list of RETRIEVED DOCUMENTS, score how well the documents answer the query.

Output ONLY a JSON object:
{
    "relevance_score": <float 0.0 to 1.0>,
    "reason": "<one sentence explanation>"
}

Scoring guide:
- 1.0: Documents directly and completely answer the query
- 0.8: Documents mostly answer, minor gaps
- 0.5: Partially relevant, significant gaps
- 0.2: Tangentially related at best
- 0.0: Completely irrelevant

No markdown. No backticks. ONLY raw JSON."""

REWRITE_SYSTEM_PROMPT = """You are a query rewriter for a crypto trading RAG system.
The original query produced AMBIGUOUS retrieval results.
Rewrite the query to be more specific and targeted.

Output ONLY the rewritten query string. No quotes, no explanation."""


class CRAGEvaluator:
    """
    Corrective RAG: Evaluates retrieval quality and takes corrective action.
    Prevents the LLM from reasoning over irrelevant context.
    """

    def __init__(self, router: Optional[LLMRouter] = None):
        self.router = router or LLMRouter(temperature=0.1, request_timeout=30)

    def evaluate_retrieval(
        self,
        query: str,
        retrieved_docs: List[Dict[str, Any]],
    ) -> Tuple[str, float, str]:
        """
        Evaluate whether retrieved documents are relevant to the query.

        Args:
            query: The original search query
            retrieved_docs: List of dicts with at least 'text' key

        Returns:
            Tuple of (verdict: str, score: float, reason: str)
            verdict is one of: CORRECT, AMBIGUOUS, INCORRECT
        """
        if not retrieved_docs:
            return ("INCORRECT", 0.0, "No documents retrieved.")

        # Build condensed doc summaries for the LLM
        doc_summaries = []
        for i, doc in enumerate(retrieved_docs[:5]):  # Max 5 docs for cost
            text = doc.get('text', str(doc))[:500]  # Truncate long docs
            doc_summaries.append(f"[Doc {i+1}]: {text}")

        docs_text = "\n\n".join(doc_summaries)

        prompt = f"""QUERY: {query}

RETRIEVED DOCUMENTS:
{docs_text}

Score the relevance of these documents to the query."""

        messages = [
            SystemMessage(content=RELEVANCE_SYSTEM_PROMPT),
            HumanMessage(content=prompt)
        ]

        try:
            response = self.router.invoke(messages)
            content = str(response.content).strip()
            content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
            content = content.replace("```json", "").replace("```", "").strip()
            if not content:
                logger.warning("[CRAG] Empty LLM response. Failing open as CORRECT.")
                return ("CORRECT", 0.5, "Empty LLM response — fail-open")
            result = json.loads(content)

            score = float(result.get("relevance_score", 0.0))
            reason = result.get("reason", "No reason provided.")

            if score >= 0.8:
                verdict = "CORRECT"
            elif score >= 0.4:
                verdict = "AMBIGUOUS"
            else:
                verdict = "INCORRECT"

            logger.info(f"[CRAG] Query: '{query[:50]}...' → {verdict} (score={score:.2f})")
            return (verdict, score, reason)

        except Exception as e:
            logger.error(f"[CRAG] Evaluation failed: {e}")
            # Fail-open: assume CORRECT to avoid blocking the pipeline
            return ("CORRECT", 0.5, f"Evaluation failed: {e}")

    def rewrite_query(self, original_query: str) -> str:
        """Rewrite an ambiguous query to improve retrieval quality."""
        messages = [
            SystemMessage(content=REWRITE_SYSTEM_PROMPT),
            HumanMessage(content=f"Original query: {original_query}")
        ]

        try:
            response = self.router.invoke(messages)
            rewritten = str(response.content).strip().strip('"').strip("'")
            logger.info(f"[CRAG] Query rewrite: '{original_query[:40]}...' → '{rewritten[:40]}...'")
            return rewritten
        except Exception as e:
            logger.warning(f"[CRAG] Query rewrite failed: {e}. Using original.")
            return original_query

    def web_search_fallback(self, query: str, max_results: int = 3) -> List[Dict[str, Any]]:
        """
        DuckDuckGo web search fallback when retrieval fails completely.
        Returns list of dicts with 'text' and 'source' keys.
        """
        try:
            from ddgs import DDGS
            results = []
            with DDGS() as ddgs:
                for r in ddgs.text(query, max_results=max_results):
                    results.append({
                        "text": f"{r.get('title', '')}: {r.get('body', '')}",
                        "source": r.get('href', 'duckduckgo'),
                        "type": "web_fallback"
                    })
            logger.info(f"[CRAG] Web fallback returned {len(results)} results for: '{query[:50]}...'")
            return results

        except ImportError:
            logger.warning("[CRAG] duckduckgo_search not installed. Skipping web fallback.")
            return []
        except Exception as e:
            logger.error(f"[CRAG] Web search fallback failed: {e}")
            return []

    def corrective_retrieve(
        self,
        query: str,
        retrieved_docs: List[Dict[str, Any]],
        retriever_fn=None,
    ) -> List[Dict[str, Any]]:
        """
        Full CRAG pipeline: evaluate → correct if needed → return final docs.

        Args:
            query: The search query
            retrieved_docs: Initially retrieved documents
            retriever_fn: Optional callback to re-retrieve with rewritten query
                         Signature: (query: str) -> List[Dict]

        Returns:
            Final list of documents to send to the LLM
        """
        verdict, score, reason = self.evaluate_retrieval(query, retrieved_docs)

        if verdict == "CORRECT":
            return retrieved_docs

        elif verdict == "AMBIGUOUS":
            # Try query rewrite + re-retrieval
            rewritten = self.rewrite_query(query)
            if retriever_fn and rewritten != query:
                new_docs = retriever_fn(rewritten)
                if new_docs:
                    # Re-evaluate the new results
                    new_verdict, new_score, _ = self.evaluate_retrieval(rewritten, new_docs)
                    if new_score > score:
                        logger.info(f"[CRAG] Rewritten query improved score: {score:.2f} → {new_score:.2f}")
                        return new_docs

            # If re-retrieval didn't help, supplement with web search
            web_results = self.web_search_fallback(query)
            return retrieved_docs + web_results

        else:  # INCORRECT
            logger.warning(f"[CRAG] Retrieval INCORRECT (score={score:.2f}). Falling back to web search.")
            web_results = self.web_search_fallback(query)
            if web_results:
                return web_results
            # Last resort: return original docs with a warning
            logger.warning("[CRAG] Web fallback also empty. Using original docs as last resort.")
            return retrieved_docs


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    evaluator = CRAGEvaluator()

    # Test with fake documents
    test_query = "What is the current Bitcoin Fear & Greed Index?"
    test_docs = [
        {"text": "Bitcoin price reached $67,500 today after Federal Reserve comments."},
        {"text": "The Fear & Greed Index is currently at 72, indicating Greed sentiment."},
    ]

    verdict, score, reason = evaluator.evaluate_retrieval(test_query, test_docs)
    print(f"Verdict: {verdict}, Score: {score:.2f}, Reason: {reason}")
