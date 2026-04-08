"""
Phase 23: Jina Reranker v3 API — replaces local ColBERT model server.
Same interface as old ColBERTReranker so hybrid_retriever.py calls are unchanged.

Jina Reranker v3 (0.6B param, SOTA BEIR 61.94 nDCG@10, 131K context)
replaces jinaai/jina-colbert-v2 (local, 2GB RAM, CPU-slow).

Fallback chain: Jina v3 API → legacy ColBERT model server (if archived server running)
"""

import logging
import time
import threading
import httpx
from typing import List, Dict

from ai_config import MODEL_SERVER_URL, JINA_API_KEYS, JINA_API_URL, JINA_RERANK_MODEL

logger = logging.getLogger(__name__)


class ColBERTReranker:
    """Jina Reranker v3 API with legacy ColBERT fallback.
    Class name kept as ColBERTReranker for backward compatibility with hybrid_retriever.py.
    """
    _http_client = None
    _jina_last_fail = 0.0
    _colbert_last_fail = 0.0
    _JINA_COOLDOWN_SECS = 15
    _COLBERT_COOLDOWN_SECS = 60
    _jina_key_index = 0
    _jina_key_lock = threading.Lock()

    def __init__(self):
        if ColBERTReranker._http_client is None:
            ColBERTReranker._http_client = httpx.Client(timeout=30)
            if JINA_API_KEYS:
                logger.info(f"[Reranker] Jina Reranker v3 initialized ({len(JINA_API_KEYS)} keys)")
            else:
                logger.warning("[Reranker] No Jina API keys — will try legacy ColBERT only")

    def _next_jina_key(self) -> str | None:
        if not JINA_API_KEYS:
            return None
        with ColBERTReranker._jina_key_lock:
            idx = ColBERTReranker._jina_key_index % len(JINA_API_KEYS)
            ColBERTReranker._jina_key_index = (idx + 1) % len(JINA_API_KEYS)
            return JINA_API_KEYS[idx]

    def rerank(self, query: str, documents: List[Dict], top_k: int = 5) -> List[Dict]:
        """Rerank documents using Jina Reranker v3 API, fallback to legacy ColBERT."""
        if not documents:
            return []

        # Extract text from document dicts
        doc_texts = []
        for doc in documents:
            text = doc.get("text", doc.get("content", ""))
            doc_texts.append(text if text else "")

        # Try 1: Jina Reranker v3 API
        result = self._jina_rerank(query, doc_texts, documents, top_k)
        if result is not None:
            return result

        # Try 2: Legacy ColBERT model server (emergency)
        result = self._colbert_rerank(query, doc_texts, documents, top_k)
        if result is not None:
            return result

        # Both failed → return original order
        return documents[:top_k]

    def _jina_rerank(self, query: str, doc_texts: List[str], documents: List[Dict], top_k: int):
        """Jina Reranker v3 API call."""
        now = time.time()
        if now - ColBERTReranker._jina_last_fail < self._JINA_COOLDOWN_SECS:
            return None

        for _ in range(len(JINA_API_KEYS) if JINA_API_KEYS else 0):
            key = self._next_jina_key()
            if not key:
                break
            try:
                resp = self._http_client.post(
                    f"{JINA_API_URL}/rerank",
                    headers={"Authorization": f"Bearer {key}",
                             "Content-Type": "application/json"},
                    json={"model": JINA_RERANK_MODEL,
                          "query": query,
                          "documents": doc_texts,
                          "top_n": top_k},
                    timeout=15
                )
                if resp.status_code == 200:
                    results = resp.json().get("results", [])
                    return self._map_jina_results(results, documents)
                elif resp.status_code in (402, 429):
                    logger.warning(f"[Reranker] Jina key ...{key[-6:]} {resp.status_code}, trying next")
                    continue
                else:
                    logger.warning(f"[Reranker] Jina API {resp.status_code}: {resp.text[:100]}")
            except Exception as e:
                logger.warning(f"[Reranker] Jina API error: {e}")

        ColBERTReranker._jina_last_fail = now
        return None

    def _map_jina_results(self, jina_results: list, documents: List[Dict]) -> List[Dict]:
        """Map Jina API results back to original document dicts with normalized scores."""
        scored_docs = []
        for jr in jina_results:
            idx = jr.get("index", 0)
            if 0 <= idx < len(documents):
                scored_doc = documents[idx].copy()
                scored_doc["colbert_score"] = jr.get("relevance_score", 0.0)
                scored_docs.append(scored_doc)

        # Normalize scores to 0-1 range (same interface as old ColBERT)
        if scored_docs:
            scores = [d["colbert_score"] for d in scored_docs]
            max_s, min_s = max(scores), min(scores)
            range_s = max_s - min_s if max_s > min_s else 1.0
            for doc in scored_docs:
                doc["colbert_normalized"] = (doc["colbert_score"] - min_s) / range_s

        return scored_docs

    def _colbert_rerank(self, query: str, doc_texts: List[str], documents: List[Dict], top_k: int):
        """LEGACY: ColBERT via model server HTTP. Emergency fallback only."""
        now = time.time()
        if now - ColBERTReranker._colbert_last_fail < self._COLBERT_COOLDOWN_SECS:
            return None

        try:
            resp = self._http_client.post(
                f"{MODEL_SERVER_URL}/rerank/colbert",
                json={"query": query, "documents": doc_texts, "top_k": top_k},
                timeout=120
            )
            resp.raise_for_status()
            server_results = resp.json().get("results", [])
            if not server_results:
                return None

            scored_docs = []
            for sr in server_results:
                idx = sr.get("index", 0)
                if 0 <= idx < len(documents):
                    scored_doc = documents[idx].copy()
                    scored_doc["colbert_score"] = sr.get("score", 0.0)
                    scored_docs.append(scored_doc)

            if scored_docs:
                scores = [d["colbert_score"] for d in scored_docs]
                max_s, min_s = max(scores), min(scores)
                range_s = max_s - min_s if max_s > min_s else 1.0
                for doc in scored_docs:
                    doc["colbert_normalized"] = (doc["colbert_score"] - min_s) / range_s

            return scored_docs
        except Exception as e:
            ColBERTReranker._colbert_last_fail = now
            logger.warning(f"[Reranker] Legacy ColBERT server failed: {e}")
            return None
