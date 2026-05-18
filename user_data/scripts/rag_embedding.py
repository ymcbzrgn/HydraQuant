import os
import sqlite3
import hashlib
import json
import logging
import time
import threading
import httpx
from dotenv import load_dotenv

# --- Resilient imports: embedding works even if one backend is missing ---
# Catch ALL exceptions (not just ImportError) because google.genai can fail with
# RecursionError, RuntimeError, or other non-import errors during protobuf/grpc init.
_GENAI_AVAILABLE = False
try:
    from google import genai
    _GENAI_AVAILABLE = True
except Exception as _e:
    logging.getLogger(__name__).warning(f"[Embedding] google.genai unavailable ({type(_e).__name__}): {_e}. Gemini embedding disabled — using BGE server only.")

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))

logger = logging.getLogger(__name__)

# Constants
from ai_config import AI_DB_PATH as DB_PATH, MODEL_SERVER_URL
from ai_config import JINA_API_KEYS, JINA_API_URL, JINA_EMBED_MODEL, JINA_EMBED_DIM
from db import get_connection, get_db_connection


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
    _genai_clients = None  # list of (api_key, genai.Client) tuples
    _key_index = 0  # round-robin index
    _key_lock = threading.Lock()
    _key_cooldowns = {}  # api_key -> cooldown_until timestamp

    # Jina Embedding API (Phase 23: replaces BGE model server)
    _http_client = None
    _jina_available = False
    _jina_checked = False
    _jina_last_fail = 0.0
    _jina_key_index = 0
    _jina_key_lock = threading.Lock()
    _JINA_COOLDOWN_SECS = 30  # shorter than BGE — API recovers faster

    # Legacy BGE model server (emergency fallback only)
    _bge_server_checked = False
    _bge_server_active = False
    _bge_last_fail = 0.0
    _BGE_COOLDOWN_SECS = 60

    # Phase 6 (2026-05-18): embedding cache moved from ai_data.sqlite to LanceDB
    _lance_gemini = None
    _lance_bge = None
    _lance_lock = threading.Lock()

    KEY_COOLDOWN_SECS = 120  # 2 min cooldown on Gemini 429

    def __init__(self):
        all_keys = _load_all_gemini_keys()

        # Create genai clients for all keys (singleton, shared across instances)
        if _GENAI_AVAILABLE and DualEmbeddingPipeline._genai_clients is None:
            if not all_keys:
                logger.warning("[Embedding] No Gemini API keys found. Will use BGE-only mode.")
            DualEmbeddingPipeline._genai_clients = []
            for key in all_keys:
                try:
                    client = genai.Client(api_key=key)
                    DualEmbeddingPipeline._genai_clients.append((key, client))
                except Exception as e:
                    logger.warning(f"[Embedding] Failed to create client for key ...{key[-6:]}: {e}")
            logger.info(f"[Embedding] Initialized {len(DualEmbeddingPipeline._genai_clients)} Gemini embedding clients")
        elif not _GENAI_AVAILABLE:
            if DualEmbeddingPipeline._genai_clients is None:
                DualEmbeddingPipeline._genai_clients = []
            logger.info("[Embedding] Gemini SDK unavailable — skipping API embedding init.")

        # HTTP client (shared for Jina API + legacy model server)
        if DualEmbeddingPipeline._http_client is None:
            DualEmbeddingPipeline._http_client = httpx.Client(timeout=30)

        # Phase 23: Jina Embedding API (primary, replaces BGE model server)
        if not DualEmbeddingPipeline._jina_checked:
            DualEmbeddingPipeline._jina_checked = True
            if JINA_API_KEYS:
                try:
                    resp = DualEmbeddingPipeline._http_client.post(
                        f"{JINA_API_URL}/embeddings",
                        headers={"Authorization": f"Bearer {JINA_API_KEYS[0]}",
                                 "Content-Type": "application/json"},
                        json={"model": JINA_EMBED_MODEL, "input": ["health check"],
                              "dimensions": JINA_EMBED_DIM, "task": "retrieval.passage"},
                        timeout=10
                    )
                    if resp.status_code == 200:
                        dim = len(resp.json()["data"][0]["embedding"])
                        DualEmbeddingPipeline._jina_available = True
                        logger.info(f"[Embedding] Jina API OK: model={JINA_EMBED_MODEL}, dim={dim}, keys={len(JINA_API_KEYS)}")
                    else:
                        logger.warning(f"[Embedding] Jina API returned {resp.status_code}: {resp.text[:200]}")
                except Exception as e:
                    logger.warning(f"[Embedding] Jina API unavailable: {e}. Will use legacy BGE fallback.")

        # Legacy BGE model server (emergency fallback — only if Jina is down)
        if not DualEmbeddingPipeline._jina_available and not DualEmbeddingPipeline._bge_server_checked:
            DualEmbeddingPipeline._bge_server_checked = True
            try:
                resp = DualEmbeddingPipeline._http_client.get(f"{MODEL_SERVER_URL}/health", timeout=5)
                health = resp.json()
                DualEmbeddingPipeline._bge_server_active = health.get("bge") == "active"
                logger.info(f"[Embedding] Legacy model server health: {health}")
            except Exception:
                DualEmbeddingPipeline._bge_server_active = False

        # Log active mode
        has_gemini = bool(DualEmbeddingPipeline._genai_clients)
        has_jina = DualEmbeddingPipeline._jina_available
        has_bge = DualEmbeddingPipeline._bge_server_active
        if has_gemini and has_jina:
            logger.info("[Embedding] Mode: DUAL API (Gemini + Jina) — Phase 23 active")
        elif has_gemini and has_bge:
            logger.info("[Embedding] Mode: DUAL LEGACY (Gemini + BGE model server)")
        elif has_gemini:
            logger.info("[Embedding] Mode: GEMINI-ONLY")
        elif has_jina:
            logger.info("[Embedding] Mode: JINA-ONLY")
        else:
            logger.error("[Embedding] Mode: NONE — no embedding backend available!")

    def _get_db_connection(self):
        conn = get_db_connection()
        return conn

    def _get_lance_cache_tables(self):
        """Phase 6 (2026-05-18): embedding cache backed by LanceDB instead of the
        ai_data.sqlite embedding_cache table. Eliminates JSON-BLOB write lock
        contention against ai_data.sqlite and shrinks that DB by ~1 GB.
        Returns (gemini_table, bge_table) or (None, None) if LanceDB unavailable."""
        if DualEmbeddingPipeline._lance_gemini is None:
            with DualEmbeddingPipeline._lance_lock:
                if DualEmbeddingPipeline._lance_gemini is None:
                    try:
                        from lance_store import get_lance_table
                        DualEmbeddingPipeline._lance_gemini = get_lance_table(
                            "embedding_cache_gemini", dim=768)
                        DualEmbeddingPipeline._lance_bge = get_lance_table(
                            "embedding_cache_bge", dim=768)
                    except Exception as e:
                        logger.error(f"[Embedding] LanceDB cache unavailable: {e}")
                        return None, None
        return DualEmbeddingPipeline._lance_gemini, DualEmbeddingPipeline._lance_bge

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
        """Try all keys x all models. Returns embedding list or None."""
        if not _GENAI_AVAILABLE:
            return None
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

                    logger.info(f"[Embedding] Gemini OK via key ...{key[-6:]} model={model_cfg['name']}")
                    return emb
                except Exception as e:
                    err_str = str(e).lower()
                    if '429' in err_str or 'resource_exhausted' in err_str or 'quota' in err_str:
                        self._penalize_key(key)
                        logger.warning(f"[Embedding] 429 on key ...{key[-6:]} model={model_cfg['name']} — rotating to next key")
                        break  # Try next key
                    logger.warning(f"[Embedding] {model_cfg['name']} with key ...{key[-6:]} failed: {e}")
                    continue  # Try next model on same key

        logger.warning(f"[Embedding] All {len(clients)} Gemini keys exhausted or in cooldown. Falling back to BGE.")
        return None

    def _next_jina_key(self) -> str | None:
        """Round-robin Jina API key selection."""
        if not JINA_API_KEYS:
            return None
        with DualEmbeddingPipeline._jina_key_lock:
            idx = DualEmbeddingPipeline._jina_key_index % len(JINA_API_KEYS)
            DualEmbeddingPipeline._jina_key_index = (idx + 1) % len(JINA_API_KEYS)
            return JINA_API_KEYS[idx]

    def _jina_embed(self, text: str):
        """Get Jina Embedding v3 via API. Returns list or None. Phase 23 primary."""
        if not DualEmbeddingPipeline._jina_available:
            return None

        # Circuit breaker
        now = time.time()
        if now - DualEmbeddingPipeline._jina_last_fail < self._JINA_COOLDOWN_SECS:
            return self._bge_embed(text)  # Fallback to legacy BGE

        # Try all keys
        for _ in range(len(JINA_API_KEYS)):
            key = self._next_jina_key()
            if not key:
                break
            try:
                resp = DualEmbeddingPipeline._http_client.post(
                    f"{JINA_API_URL}/embeddings",
                    headers={"Authorization": f"Bearer {key}",
                             "Content-Type": "application/json"},
                    json={"model": JINA_EMBED_MODEL,
                          "input": [text],
                          "dimensions": JINA_EMBED_DIM,
                          "task": "retrieval.passage"},
                    timeout=15
                )
                if resp.status_code == 200:
                    data = resp.json()
                    emb = data["data"][0]["embedding"]
                    if emb and len(emb) > 0:
                        return emb
                elif resp.status_code == 402:
                    logger.warning(f"[Embedding] Jina key ...{key[-6:]} token exhausted, trying next")
                    continue
                elif resp.status_code == 429:
                    logger.warning(f"[Embedding] Jina key ...{key[-6:]} rate limited, trying next")
                    continue
                else:
                    logger.warning(f"[Embedding] Jina API {resp.status_code}: {resp.text[:100]}")
            except Exception as e:
                logger.warning(f"[Embedding] Jina API error with key ...{key[-6:]}: {e}")

        # All Jina keys failed → circuit breaker + legacy BGE fallback
        DualEmbeddingPipeline._jina_last_fail = now
        logger.warning("[Embedding] All Jina keys failed. Falling back to legacy BGE.")
        return self._bge_embed(text)

    def _bge_embed(self, text: str):
        """LEGACY: BGE-Financial via model server HTTP. Emergency fallback only."""
        now = time.time()
        if now - DualEmbeddingPipeline._bge_last_fail < self._BGE_COOLDOWN_SECS:
            return None
        if not DualEmbeddingPipeline._bge_server_active:
            return None

        try:
            resp = DualEmbeddingPipeline._http_client.post(
                f"{MODEL_SERVER_URL}/embed/bge",
                json={"texts": [text]},
                timeout=60
            )
            resp.raise_for_status()
            embeddings = resp.json().get("embeddings", [])
            if embeddings and len(embeddings[0]) > 0:
                return embeddings[0]
            return None
        except Exception as e:
            DualEmbeddingPipeline._bge_last_fail = now
            logger.warning(f"[Embedding] Legacy BGE server call failed: {e}")
            return None

    def get_embeddings(self, text: str) -> dict:
        """
        Returns both Gemini and BGE embeddings for a given text.
        Checks LanceDB cache first. Falls back gracefully if either backend unavailable.
        """
        text_hash = self._hash_text(text)

        # 1. Check LanceDB cache (Phase 6: replaced ai_data.sqlite embedding_cache)
        gemini_table, bge_table = self._get_lance_cache_tables()
        if gemini_table is not None and bge_table is not None:
            try:
                g = gemini_table.get(ids=[text_hash])
                b = bge_table.get(ids=[text_hash])
                if g["ids"] and b["ids"] and g["embeddings"] and b["embeddings"]:
                    return {
                        "gemini": list(g["embeddings"][0]),
                        "bge": list(b["embeddings"][0]),
                        "cached": True
                    }
            except Exception as e:
                logger.error(f"Error reading from LanceDB embedding cache: {e}")

        # 2. Generate Embeddings (Cache Miss)
        logger.debug("Cache miss. Generating Dual Embeddings...")

        # Phase 23: Jina Embedding v3 (primary) → legacy BGE (fallback)
        bge_emb = self._jina_embed(text)

        # Gemini API (with key rotation + fallback)
        gemini_emb = None
        if _GENAI_AVAILABLE:
            gemini_emb = self._gemini_embed(text)

        # Fallback logic: Gemini can fill BGE slot (same 768-dim), but NOT vice versa.
        # BGE-Financial is domain-specific — copying Gemini to BGE slot would pollute
        # the BGE collection with wrong semantic space and may cause dimension mismatch.
        if gemini_emb is None and bge_emb is not None:
            logger.debug("[Embedding] Gemini unavailable. Using BGE for both slots.")
            gemini_emb = bge_emb
        elif bge_emb is None and gemini_emb is not None:
            logger.debug("[Embedding] BGE unavailable. Returning Gemini only (BGE slot empty).")
            bge_emb = []  # Don't copy Gemini -> avoids dimension/semantic mismatch in BGE collection
        elif gemini_emb is None and bge_emb is None:
            logger.error("[Embedding] BOTH backends failed! Cannot generate embeddings.")
            return {"gemini": [], "bge": [], "cached": False}

        # 3. Save to LanceDB cache (Phase 6: replaced ai_data.sqlite embedding_cache).
        # delete-then-add gives upsert semantics; the LanceTable schema is a
        # fixed-size float32[768] vector, so non-768 / empty embeddings are skipped.
        if gemini_table is not None and bge_table is not None:
            try:
                if gemini_emb and len(gemini_emb) == 768:
                    gemini_table.delete(ids=[text_hash])
                    gemini_table.add(ids=[text_hash], embeddings=[gemini_emb], documents=[text])
                if bge_emb and len(bge_emb) == 768:
                    bge_table.delete(ids=[text_hash])
                    bge_table.add(ids=[text_hash], embeddings=[bge_emb], documents=[text])
            except Exception as e:
                logger.error(f"Error writing to LanceDB embedding cache: {e}")

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
