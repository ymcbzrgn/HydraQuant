import logging
import time
import httpx
from typing import List, Dict

from ai_config import MODEL_SERVER_URL

logger = logging.getLogger(__name__)


class ColBERTReranker:
    """ColBERTv2 Late Interaction Reranker via model server HTTP.
    Replaces in-process torch/transformers loading with HTTP calls to
    the centralized model server at MODEL_SERVER_URL/rerank/colbert.
    """
    _http_client = None
    _last_fail = 0.0
    _COOLDOWN_SECS = 60  # skip calls for 60s after failure

    def __init__(self):
        if ColBERTReranker._http_client is None:
            ColBERTReranker._http_client = httpx.Client(timeout=10)
            logger.info(f"[ColBERT] HTTP client initialized → {MODEL_SERVER_URL}/rerank/colbert")

    def rerank(self, query: str, documents: List[Dict], top_k: int = 5) -> List[Dict]:
        """Token-level late interaction scoring via model server."""
        if not documents:
            return []

        # Circuit breaker: skip if server failed recently
        now = time.time()
        if now - ColBERTReranker._last_fail < self._COOLDOWN_SECS:
            return documents[:top_k]

        # Extract text from document dicts
        doc_texts = []
        for doc in documents:
            text = doc.get("text", doc.get("content", ""))
            doc_texts.append(text if text else "")

        try:
            resp = self._http_client.post(
                f"{MODEL_SERVER_URL}/rerank/colbert",
                json={"query": query, "documents": doc_texts, "top_k": top_k}
            )
            resp.raise_for_status()
            server_results = resp.json().get("results", [])

            if not server_results:
                return documents[:top_k]

            # Map server scores back to original document dicts
            scored_docs = []
            for sr in server_results:
                idx = sr.get("index", 0)
                if 0 <= idx < len(documents):
                    scored_doc = documents[idx].copy()
                    scored_doc["colbert_score"] = sr.get("score", 0.0)
                    scored_docs.append(scored_doc)

            # Normalize colbert scores between 0 and 1 for ensemble purposes
            if scored_docs:
                max_score = max(doc.get("colbert_score", 0.0) for doc in scored_docs)
                min_score = min(doc.get("colbert_score", 0.0) for doc in scored_docs)
                range_score = max_score - min_score if max_score > min_score else 1.0

                for doc in scored_docs:
                    if "colbert_score" in doc:
                        doc["colbert_normalized"] = (doc["colbert_score"] - min_score) / range_score
                    else:
                        doc["colbert_normalized"] = 0.0

            return scored_docs

        except Exception as e:
            ColBERTReranker._last_fail = now
            logger.warning(f"[ColBERT] Server call failed: {e}. Returning original order.")
            return documents[:top_k]
