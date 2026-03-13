import os
import sqlite3
import hashlib
import json
import logging
import time
import threading
import numpy as np
from dotenv import load_dotenv
from google import genai
from sentence_transformers import SentenceTransformer

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))

logger = logging.getLogger(__name__)

# Constants
from ai_config import AI_DB_PATH as DB_PATH


def _load_all_gemini_keys() -> list:
    """Load all Gemini API keys from env (same logic as llm_router.py)."""
    keys = []
    # Comma-separated GEMINI_API_KEYS
    keys_str = os.environ.get("GEMINI_API_KEYS", "")
    if keys_str:
        keys.extend([k.strip() for k in keys_str.split(",") if k.strip()])
    # Single GEMINI_API_KEY
    single = os.environ.get("GEMINI_API_KEY")
    if single and single not in keys:
        keys.append(single)
    # Numbered GEMINI_API_KEY_1 through GEMINI_API_KEY_10
    for i in range(1, 11):
        k = os.environ.get(f"GEMINI_API_KEY_{i}")
        if k and k not in keys:
            keys.append(k)
    return keys


class DualEmbeddingPipeline:
    # Failover list: try in order, skip on error
    GEMINI_EMBEDDING_MODELS = [
        {"name": "gemini-embedding-001", "dims": 768},
        {"name": "gemini-embedding-2-preview", "dims": 768},
    ]

    # Class-level singletons
    _bge_model = None
    _genai_clients = None  # list of (api_key, genai.Client) tuples
    _key_index = 0  # round-robin index
    _key_lock = threading.Lock()
    _key_cooldowns = {}  # api_key -> cooldown_until timestamp

    KEY_COOLDOWN_SECS = 120  # 2 min cooldown on 429

    def __init__(self):
        all_keys = _load_all_gemini_keys()
        if not all_keys:
            logger.warning("[Embedding] No Gemini API keys found. Will use BGE-only mode.")

        # Create genai clients for all keys (singleton, shared across instances)
        if DualEmbeddingPipeline._genai_clients is None:
            DualEmbeddingPipeline._genai_clients = []
            for key in all_keys:
                try:
                    client = genai.Client(api_key=key)
                    DualEmbeddingPipeline._genai_clients.append((key, client))
                except Exception as e:
                    logger.warning(f"[Embedding] Failed to create client for key ...{key[-6:]}: {e}")
            logger.info(f"[Embedding] Initialized {len(DualEmbeddingPipeline._genai_clients)} Gemini embedding clients")

        # BGE-Financial local model (always available)
        if DualEmbeddingPipeline._bge_model is None:
            logger.info("Loading local BGE-Financial embedding model (one-time)...")
            DualEmbeddingPipeline._bge_model = SentenceTransformer("philschmid/bge-base-financial-matryoshka")
        self.local_model = DualEmbeddingPipeline._bge_model

    def _get_db_connection(self):
        conn = sqlite3.connect(DB_PATH, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _hash_text(self, text: str) -> str:
        return hashlib.sha256(text.encode('utf-8')).hexdigest()

    def _next_client(self):
        """Thread-safe round-robin key selection, skipping cooled-down keys."""
        clients = DualEmbeddingPipeline._genai_clients or []
        if not clients:
            return None, None

        now = time.time()
        with DualEmbeddingPipeline._key_lock:
            # Try all keys starting from current index
            for _ in range(len(clients)):
                idx = DualEmbeddingPipeline._key_index % len(clients)
                DualEmbeddingPipeline._key_index = (idx + 1) % len(clients)
                key, client = clients[idx]

                cooldown_until = DualEmbeddingPipeline._key_cooldowns.get(key, 0)
                if now >= cooldown_until:
                    return key, client

        logger.warning("[Embedding] All Gemini keys are in cooldown")
        return None, None

    def _penalize_key(self, key: str):
        """Put a key in cooldown after a 429/quota error."""
        DualEmbeddingPipeline._key_cooldowns[key] = time.time() + self.KEY_COOLDOWN_SECS
        logger.warning(f"[Embedding] Key ...{key[-6:]} penalized for {self.KEY_COOLDOWN_SECS}s")

    def _gemini_embed(self, text: str):
        """Try all keys × all models. Returns embedding list or None."""
        clients = DualEmbeddingPipeline._genai_clients or []
        if not clients:
            return None

        # Try up to len(clients) different keys
        for _ in range(len(clients)):
            key, client = self._next_client()
            if client is None:
                break

            for model_cfg in self.GEMINI_EMBEDDING_MODELS:
                try:
                    kwargs = {"model": model_cfg["name"], "contents": text}
                    if model_cfg.get("dims"):
                        kwargs["config"] = {"output_dimensionality": model_cfg["dims"]}
                    result = client.models.embed_content(**kwargs)

                    if hasattr(result, 'embeddings') and result.embeddings:
                        emb = result.embeddings[0].values
                    elif isinstance(result, dict) and 'embedding' in result:
                        emb = result['embedding']
                    else:
                        emb = result

                    logger.debug(f"[Embedding] Gemini OK via key ...{key[-6:]} model={model_cfg['name']}")
                    return emb
                except Exception as e:
                    err_str = str(e).lower()
                    if '429' in err_str or 'resource_exhausted' in err_str or 'quota' in err_str:
                        self._penalize_key(key)
                        break  # Try next key
                    logger.debug(f"[Embedding] {model_cfg['name']} with key ...{key[-6:]} failed: {e}")
                    continue  # Try next model on same key

        return None

    def get_embeddings(self, text: str) -> dict:
        """
        Returns both Gemini and BGE embeddings for a given text.
        Checks SQLite cache first. Falls back to BGE-only if Gemini exhausted.
        """
        text_hash = self._hash_text(text)

        # 1. Check Cache
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT gemini_embedding, bge_embedding FROM embedding_cache WHERE text_hash = ?",
                    (text_hash,)
                )
                row = cursor.fetchone()

                if row and row['gemini_embedding'] and row['bge_embedding']:
                    gemini_emb = json.loads(row['gemini_embedding'])
                    bge_emb = json.loads(row['bge_embedding'])
                    return {
                        "gemini": gemini_emb,
                        "bge": bge_emb,
                        "cached": True
                    }
        except Exception as e:
            logger.error(f"Error reading from embedding cache: {e}")

        # 2. Generate Embeddings (Cache Miss)
        logger.debug("Cache miss. Generating Dual Embeddings...")

        # BGE-Financial (always available, local)
        bge_emb = self.local_model.encode(text, normalize_embeddings=True).tolist()

        # Gemini (with key rotation + fallback)
        gemini_emb = self._gemini_embed(text)

        if gemini_emb is None:
            # All Gemini keys exhausted → use BGE-only mode
            logger.debug("[Embedding] All Gemini keys exhausted. Using BGE-only embedding.")
            gemini_emb = bge_emb  # Use BGE as Gemini substitute

        # 3. Save to Cache
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """INSERT OR REPLACE INTO embedding_cache
                    (text_hash, text_content, gemini_embedding, bge_embedding)
                    VALUES (?, ?, ?, ?)""",
                    (text_hash, text, json.dumps(gemini_emb), json.dumps(bge_emb))
                )
                conn.commit()
        except Exception as e:
            logger.error(f"Error writing to embedding cache: {e}")

        return {
            "gemini": gemini_emb,
            "bge": bge_emb,
            "cached": False
        }


# Quick local test execution
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    pipeline = DualEmbeddingPipeline()

    keys = _load_all_gemini_keys()
    logger.info(f"Loaded {len(keys)} Gemini API keys for embedding rotation")

    test_text = "Bitcoin price surged past $60,000 following the Fed rate cut."

    logger.info("First run (Uncached)...")
    res1 = pipeline.get_embeddings(test_text)
    if res1:
        logger.info(f"Cached? {res1['cached']} | Gemini dim: {len(res1['gemini'])} | BGE dim: {len(res1['bge'])}")

    logger.info("Second run (Cached)...")
    res2 = pipeline.get_embeddings(test_text)
    if res2:
        logger.info(f"Cached? {res2['cached']} | Gemini dim: {len(res2['gemini'])} | BGE dim: {len(res2['bge'])}")
