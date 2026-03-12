import sqlite3
import logging
import json
import numpy as np
import time
import os
from datetime import datetime, timezone
from typing import Optional, Tuple
from google import genai

logger = logging.getLogger(__name__)

from ai_config import AI_DB_PATH as DB_PATH

class SemanticCache:
    """
    Caches LLM responses using semantic similarity of the query.
    Saves API costs by reusing recent identical or highly similar queries.
    """
    def __init__(self, db_path=None, similarity_threshold=0.92):
        import ai_config
        self.db_path = db_path if db_path is not None else ai_config.AI_DB_PATH
        self.similarity_threshold = similarity_threshold
        # Cache genai client — was creating new httpx client per _get_embedding() call
        self._genai_client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
        self._init_db()

    def _init_db(self):
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS semantic_cache (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        query_text TEXT NOT NULL,
                        query_embedding BLOB NOT NULL,
                        response TEXT NOT NULL,
                        pair TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        ttl_seconds INTEGER DEFAULT 300
                    )
                """)
                # Startup sanitization: purge poisoned entries with low/null confidence
                cursor.execute("""
                    DELETE FROM semantic_cache
                    WHERE json_extract(response, '$.confidence') < 0.3
                       OR json_extract(response, '$.confidence') IS NULL
                """)
                purged = cursor.rowcount
                if purged > 0:
                    logger.warning(f"[Cache Startup] Purged {purged} poisoned cache entries (confidence < 0.3 or null)")
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to init semantic_cache table: {e}")

    # Embedding model failover list (same as rag_embedding.py)
    _EMBEDDING_MODELS = [
        {"name": "gemini-embedding-001", "dims": 768},
        {"name": "gemini-embedding-2-preview", "dims": 768},
    ]

    def _get_embedding(self, text: str) -> Optional[np.ndarray]:
        for model_cfg in self._EMBEDDING_MODELS:
            try:
                kwargs = {"model": model_cfg["name"], "contents": text}
                if model_cfg.get("dims"):
                    kwargs["config"] = {"output_dimensionality": model_cfg["dims"]}
                result = self._genai_client.models.embed_content(**kwargs)
                emb = np.array(result.embeddings[0].values, dtype=np.float32)
                return emb
            except Exception as e:
                logger.warning(f"Embedding {model_cfg['name']} failed for cache: {e}")
                continue
        logger.error("All embedding models failed for semantic cache")
        return None

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def get(self, query: str, pair: Optional[str] = None) -> Optional[str]:
        """Retrieve a cached response if a highly similar query exists and is not expired."""
        query_emb = self._get_embedding(query)
        if query_emb is None:
            return None

        # Clean up expired entries first
        self.cleanup_expired()

        best_match_response = None
        highest_sim = 0.0

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                if pair:
                    cursor.execute("SELECT query_embedding, response FROM semantic_cache WHERE pair = ?", (pair,))
                else:
                    cursor.execute("SELECT query_embedding, response FROM semantic_cache WHERE pair IS NULL OR pair = ''")
                
                rows = cursor.fetchall()
                for emb_blob, response in rows:
                    if emb_blob:
                        cached_emb = np.frombuffer(emb_blob, dtype=np.float32)
                        if cached_emb.shape == query_emb.shape:
                            sim = self._cosine_similarity(query_emb, cached_emb)
                            if sim >= self.similarity_threshold and sim > highest_sim:
                                highest_sim = sim
                                best_match_response = response
                                
            if best_match_response:
                # Reject cached results with low confidence (poisoned cache entries)
                try:
                    cached_data = json.loads(best_match_response)
                    confidence_val = cached_data.get("confidence")
                    cached_conf = float(confidence_val) if confidence_val is not None else 0.0
                    if cached_conf < 0.3:
                        logger.warning(f"Semantic Cache Hit REJECTED — cached confidence too low ({cached_conf:.2f}). Forcing fresh pipeline.")
                        return None
                except (json.JSONDecodeError, ValueError, TypeError):
                    pass  # Non-JSON cache entry, return as-is
                logger.info(f"Semantic Cache Hit! Similarity: {highest_sim:.4f}")
                return best_match_response
        except Exception as e:
            logger.error(f"Error accessing semantic cache: {e}")

        return None

    def put(self, query: str, response: str, pair: Optional[str] = None, ttl: int = 300):
        """Store a response in the cache. Rejects low-confidence results to prevent cache poisoning."""
        # Never cache low-confidence results — they poison future lookups
        try:
            resp_data = json.loads(response)
            confidence_val = resp_data.get("confidence")
            resp_conf = float(confidence_val) if confidence_val is not None else 0.0
            if resp_conf < 0.3:
                logger.warning(f"Semantic Cache PUT REJECTED — confidence too low ({resp_conf:.2f}). Not caching to prevent poisoning.")
                return
        except (json.JSONDecodeError, ValueError, TypeError):
            pass  # Non-JSON response, allow caching

        query_emb = self._get_embedding(query)
        if query_emb is None:
            return

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO semantic_cache 
                    (query_text, query_embedding, response, pair, ttl_seconds) 
                    VALUES (?, ?, ?, ?, ?)
                """, (query, query_emb.tobytes(), response, pair, ttl))
                conn.commit()
                logger.info(f"Stored response in semantic cache for query: '{query[:30]}...' (TTL: {ttl}s)")
        except Exception as e:
            logger.error(f"Error writing to semantic cache: {e}")

    def invalidate(self, pair: Optional[str] = None):
        """Invalidate cache entries for a specific pair, or all entries."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                if pair:
                    cursor.execute("DELETE FROM semantic_cache WHERE pair = ?", (pair,))
                else:
                    cursor.execute("DELETE FROM semantic_cache")
                conn.commit()
                logger.info(f"Invalidated semantic cache (pair: {pair})")
        except Exception as e:
            logger.error(f"Error invalidating semantic cache: {e}")

    def cleanup_expired(self):
        """Remove expired entries based on TTL."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                # Use basic datetime comparison
                cursor.execute("""
                    DELETE FROM semantic_cache 
                    WHERE (strftime('%s', 'now') - strftime('%s', created_at)) > ttl_seconds
                """)
                deleted = cursor.rowcount
                conn.commit()
                if deleted > 0:
                    logger.debug(f"Cleaned up {deleted} expired semantic cache entries.")
        except Exception as e:
            logger.error(f"Error cleaning up expired cache: {e}")
