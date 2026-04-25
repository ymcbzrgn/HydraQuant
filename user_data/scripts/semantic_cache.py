import sqlite3
import logging
import json
import numpy as np
import time
import os
from datetime import datetime, timezone
from typing import Optional, Tuple
try:
    from google import genai
    _GENAI_AVAILABLE = True
except Exception as _e:
    logging.getLogger(__name__).warning(f"[SemanticCache] google.genai unavailable: {_e}. Cache embedding disabled.")
    genai = None
    _GENAI_AVAILABLE = False

logger = logging.getLogger(__name__)

from ai_config import AI_DB_PATH as DB_PATH
from db import get_db_connection, execute_with_retry


def _load_gemini_keys() -> list:
    """Load all Gemini API keys from env."""
    keys = []
    keys_str = os.environ.get("GEMINI_API_KEYS", "")
    if keys_str:
        keys.extend([k.strip() for k in keys_str.split(",") if k.strip()])
    single = os.environ.get("GEMINI_API_KEY")
    if single and single not in keys:
        keys.append(single)
    for i in range(1, 11):
        k = os.environ.get(f"GEMINI_API_KEY_{i}")
        if k and k not in keys:
            keys.append(k)
    return keys


class SemanticCache:
    """
    Caches LLM responses using semantic similarity of the query.
    Saves API costs by reusing recent identical or highly similar queries.

    Task 10 additions (2026-04-24):
      * `_hits` / `_misses` / `_puts` / `_rejects` atomic counters the
        scheduler drains into `cache_health_log` for rolling hit-rate
        observability. Previously `llm_calls.cache_hit` was a dead
        column — 6,548 rows, all 0, because `cache_hit=True` was never
        passed. The observer ledger side-steps that column entirely.
      * `similarity_distribution` deque — every get() records the
        best-match similarity even on miss so similarity_probe() can
        self-tune the threshold.
      * `_embedding_model_name` — the model actually used to embed
        incoming queries. put() stamps the row with the model; get()
        only compares embeddings produced by the same model so a Jina
        migration or Gemini version bump doesn't silently turn the
        whole cache into noise.
    """
    def __init__(self, db_path=None, similarity_threshold=0.92):
        import ai_config
        import threading
        from collections import deque
        self.db_path = db_path if db_path is not None else ai_config.AI_DB_PATH
        self.similarity_threshold = similarity_threshold
        # Create genai clients for all available keys (store key suffix for logging)
        all_keys = _load_gemini_keys()
        self._genai_clients = []  # list of (key_suffix, client) tuples
        for key in all_keys:
            try:
                self._genai_clients.append((key[-6:], genai.Client(api_key=key)))
            except Exception:
                pass
        if not self._genai_clients:
            logger.warning("[SemanticCache] No Gemini API keys available. Cache embedding disabled.")
        else:
            logger.info(f"[SemanticCache] Initialized {len(self._genai_clients)} Gemini clients for embedding")
        self._client_idx = 0
        # Task 10 observability counters.
        self._counter_lock = threading.Lock()
        self._hits = 0
        self._misses = 0
        self._puts = 0
        self._rejects = 0
        self._invalidations = 0
        self._similarity_samples = deque(maxlen=500)
        self._embedding_model_name: Optional[str] = None
        self._init_db()

    def _init_db(self):
        try:
            with get_db_connection(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS semantic_cache (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        query_text TEXT NOT NULL,
                        query_embedding BLOB NOT NULL,
                        response TEXT NOT NULL,
                        pair TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        ttl_seconds INTEGER DEFAULT 300,
                        embedding_model TEXT
                    )
                """)
                # Task 10: embedding_model column for cross-model isolation.
                try:
                    cursor.execute("ALTER TABLE semantic_cache ADD COLUMN embedding_model TEXT")
                except sqlite3.OperationalError:
                    pass
                # Task 10: cache_health_log — hit/miss/put/reject counters
                # the scheduler drains every 10 min so the organism can
                # observe its own cache.
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS cache_health_log (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        hits INTEGER, misses INTEGER, puts INTEGER,
                        rejects INTEGER, invalidations INTEGER,
                        hit_rate REAL, median_similarity REAL,
                        threshold REAL
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
        if not self._genai_clients:
            return None
        # Try each client (key) with each model — round-robin rotation
        for _ in range(len(self._genai_clients)):
            key_suffix, client = self._genai_clients[self._client_idx % len(self._genai_clients)]
            self._client_idx = (self._client_idx + 1) % len(self._genai_clients)
            for model_cfg in self._EMBEDDING_MODELS:
                try:
                    kwargs = {"model": model_cfg["name"], "contents": text}
                    if model_cfg.get("dims"):
                        kwargs["config"] = {"output_dimensionality": model_cfg["dims"]}
                    result = client.models.embed_content(**kwargs)
                    emb = np.array(result.embeddings[0].values, dtype=np.float32)
                    # Task 10: remember which model produced the embedding
                    # so get() can filter by model fingerprint and put()
                    # can stamp the row. Without this, a Gemini version
                    # bump silently turned every cached row into a
                    # different vector space than the current query.
                    self._embedding_model_name = model_cfg["name"]
                    return emb
                except Exception as e:
                    err_str = str(e).lower()
                    if '429' in err_str or 'quota' in err_str:
                        logger.warning(f"[SemanticCache] 429 on key ...{key_suffix} — rotating")
                        break  # Try next key
                    logger.debug(f"[SemanticCache] {model_cfg['name']} key ...{key_suffix} failed: {e}")
                    continue
        logger.warning("[SemanticCache] All embedding keys exhausted")
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
        model_fingerprint = self._embedding_model_name

        # Clean up expired entries first
        self.cleanup_expired()

        best_match_response = None
        highest_sim = 0.0

        try:
            with get_db_connection(self.db_path) as conn:
                cursor = conn.cursor()
                # Task 10: cross-model isolation. Rows whose embedding_model
                # differs from the current fingerprint live in a different
                # vector space; including them in cosine similarity compares
                # apples to ducks. Legacy rows with NULL embedding_model
                # fall through for backwards compatibility.
                if pair:
                    cursor.execute(
                        "SELECT query_embedding, response, embedding_model "
                        "FROM semantic_cache WHERE pair = ?",
                        (pair,),
                    )
                else:
                    cursor.execute(
                        "SELECT query_embedding, response, embedding_model "
                        "FROM semantic_cache WHERE pair IS NULL OR pair = ''"
                    )

                rows = cursor.fetchall()
                for row in rows:
                    emb_blob = row[0]
                    response = row[1]
                    row_model = row[2] if len(row) > 2 else None
                    if not emb_blob:
                        continue
                    if row_model and model_fingerprint and row_model != model_fingerprint:
                        continue  # different embedding model, skip
                    cached_emb = np.frombuffer(emb_blob, dtype=np.float32)
                    if cached_emb.shape == query_emb.shape:
                        sim = self._cosine_similarity(query_emb, cached_emb)
                        # Track the best-match similarity across the whole
                        # scan, even misses — drives similarity_probe().
                        if sim > highest_sim:
                            highest_sim = sim
                        if sim >= self.similarity_threshold and sim >= highest_sim:
                            best_match_response = response

            if highest_sim > 0:
                with self._counter_lock:
                    self._similarity_samples.append(float(highest_sim))

            if best_match_response:
                # Reject cached results with low confidence (poisoned cache entries)
                try:
                    cached_data = json.loads(best_match_response)
                    confidence_val = cached_data.get("confidence")
                    cached_conf = float(confidence_val) if confidence_val is not None else 0.0
                    if cached_conf < 0.3:
                        logger.warning(f"Semantic Cache Hit REJECTED — cached confidence too low ({cached_conf:.2f}). Forcing fresh pipeline.")
                        with self._counter_lock:
                            self._rejects += 1
                            self._misses += 1
                        return None
                except (json.JSONDecodeError, ValueError, TypeError):
                    pass  # Non-JSON cache entry, return as-is
                logger.info(f"Semantic Cache Hit! Similarity: {highest_sim:.4f}")
                with self._counter_lock:
                    self._hits += 1
                return best_match_response
        except Exception as e:
            logger.error(f"Error accessing semantic cache: {e}")

        with self._counter_lock:
            self._misses += 1
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
            execute_with_retry(
                """INSERT INTO semantic_cache
                   (query_text, query_embedding, response, pair, ttl_seconds, embedding_model)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (query, query_emb.tobytes(), response, pair, ttl,
                 self._embedding_model_name),
                max_retries=5,
                db_path=self.db_path,
            )
            logger.info(f"Stored response in semantic cache for query: '{query[:30]}...' (TTL: {ttl}s)")
            with self._counter_lock:
                self._puts += 1
        except Exception as e:
            logger.error(f"Error writing to semantic cache: {e}")

    def invalidate(self, pair: Optional[str] = None):
        """Invalidate cache entries for a specific pair, or all entries."""
        try:
            if pair:
                execute_with_retry(
                    "DELETE FROM semantic_cache WHERE pair = ?",
                    (pair,),
                    max_retries=5,
                    db_path=self.db_path,
                )
            else:
                execute_with_retry(
                    "DELETE FROM semantic_cache",
                    max_retries=5,
                    db_path=self.db_path,
                )
            logger.info(f"Invalidated semantic cache (pair: {pair})")
            with self._counter_lock:
                self._invalidations += 1
        except Exception as e:
            logger.error(f"Error invalidating semantic cache: {e}")

    def health_snapshot(self) -> dict:
        """Drained snapshot of cache observability counters. After read
        the counters reset so each window stays independent. Scheduler
        calls this every 10 min and persists to `cache_health_log`.

        Prefer `health_peek_and_commit` if you need atomic INSERT-or-rollback
        semantics (the rag_graph drain daemon does — losing a window's
        counters because of a transient DB lock is exactly the bug E1
        was meant to eliminate).
        """
        with self._counter_lock:
            hits = self._hits
            misses = self._misses
            puts = self._puts
            rejects = self._rejects
            invalidations = self._invalidations
            sims = list(self._similarity_samples)
            self._hits = 0
            self._misses = 0
            self._puts = 0
            self._rejects = 0
            self._invalidations = 0
            self._similarity_samples.clear()
        total = hits + misses
        hit_rate = (hits / total) if total > 0 else 0.0
        median_sim = float(np.median(sims)) if sims else 0.0
        return {
            "hits": hits, "misses": misses, "puts": puts,
            "rejects": rejects, "invalidations": invalidations,
            "hit_rate": hit_rate, "median_similarity": median_sim,
            "threshold": self.similarity_threshold,
        }

    def health_peek(self) -> dict:
        """Non-destructive read of the current counter window — used by
        callers that want snapshot-then-commit semantics. Returns the
        same shape as health_snapshot but does NOT reset counters.
        """
        with self._counter_lock:
            hits = self._hits
            misses = self._misses
            puts = self._puts
            rejects = self._rejects
            invalidations = self._invalidations
            sims = list(self._similarity_samples)
        total = hits + misses
        hit_rate = (hits / total) if total > 0 else 0.0
        median_sim = float(np.median(sims)) if sims else 0.0
        return {
            "hits": hits, "misses": misses, "puts": puts,
            "rejects": rejects, "invalidations": invalidations,
            "hit_rate": hit_rate, "median_similarity": median_sim,
            "threshold": self.similarity_threshold,
        }

    def health_commit_reset(self, peeked: dict) -> None:
        """Atomically subtract a previously-peeked window from the
        live counters. Combined with `health_peek`, this lets a drain
        caller persist the snapshot to DB and only THEN consume it —
        if the persist failed we never lose data. New increments that
        landed between peek and commit are preserved.
        """
        with self._counter_lock:
            self._hits = max(0, self._hits - int(peeked.get("hits", 0)))
            self._misses = max(0, self._misses - int(peeked.get("misses", 0)))
            self._puts = max(0, self._puts - int(peeked.get("puts", 0)))
            self._rejects = max(0, self._rejects - int(peeked.get("rejects", 0)))
            self._invalidations = max(
                0, self._invalidations - int(peeked.get("invalidations", 0))
            )
            # Similarity samples — drop the OLDEST N samples (the ones
            # we observed in the peek). deque-pop semantics make this O(N).
            n_drop = min(len(self._similarity_samples),
                         len(peeked.get("_sim_keys", [])) if isinstance(peeked.get("_sim_keys"), list) else 0)
            for _ in range(n_drop):
                try:
                    self._similarity_samples.popleft()
                except Exception:
                    break

    def similarity_probe(self) -> Optional[float]:
        """Read-only peek at the running median similarity. If systematically
        below the threshold, the caller can loosen the threshold — signals
        that the semantic distance of real queries drifted."""
        with self._counter_lock:
            if not self._similarity_samples:
                return None
            return float(np.median(self._similarity_samples))

    def cleanup_expired(self):
        """Remove expired entries based on TTL."""
        try:
            cursor = execute_with_retry(
                """DELETE FROM semantic_cache
                   WHERE (strftime('%s', 'now') - strftime('%s', created_at)) > ttl_seconds""",
                max_retries=5,
                db_path=self.db_path,
            )
            if cursor and cursor.rowcount > 0:
                logger.debug(f"Cleaned up {cursor.rowcount} expired semantic cache entries.")
        except Exception as e:
            logger.error(f"Error cleaning up expired cache: {e}")


# ─── Task 22: singleton accessor ────────────────────────────────────────────
# rag_graph's module-level `_semantic_cache` was the only real in-process
# instance carrying the hit/miss counters. When the scheduler drain job
# instantiated a NEW `SemanticCache()` every 10 min, it read the fresh
# instance's all-zero counters — making cache_health_log a lie.
# This accessor lets both rag_graph and scheduler bind to the same
# instance (per `db_path` so tests with tmp_path still isolate).

import ai_config as _ai_config_for_cache

_semantic_cache_instances: dict = {}


def get_semantic_cache(db_path: Optional[str] = None,
                       similarity_threshold: float = 0.92) -> "SemanticCache":
    """Return the per-db_path SemanticCache singleton. The first caller
    per path owns the counters; every subsequent caller shares them."""
    key = db_path if db_path is not None else _ai_config_for_cache.AI_DB_PATH
    inst = _semantic_cache_instances.get(key)
    if inst is None:
        inst = SemanticCache(db_path=key, similarity_threshold=similarity_threshold)
        _semantic_cache_instances[key] = inst
    return inst
