import os
import threading

# ── NumPy 2.x Compatibility Shim ─────────────────────────────────────
# numpy 2.0 removed np.matrix. yfinance 1.2.0 (via pandas internals)
# still uses it, causing "module 'numpy' has no attribute 'matrix'" errors.
# This shim restores the attribute using np.asmatrix (still available).
# Applied here because ai_config.py is imported by ALL AI modules.
try:
    import numpy as _np
    if not hasattr(_np, 'matrix'):
        _np.matrix = _np.asmatrix
except ImportError:
    pass

# Base paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Database Path for SQLite (FTS5, Logging, Settings)
AI_DB_PATH = os.environ.get(
    "AI_DB_PATH",
    os.path.join(BASE_DIR, "db", "ai_data.sqlite")
)

# Model Server (ARCHIVED — Jina API replaced BGE/ColBERT, FlashRank moved in-process)
# Kept for emergency ColBERT fallback if ALL APIs are down for extended period
MODEL_SERVER_URL = os.environ.get("MODEL_SERVER_URL", "http://127.0.0.1:8895")

# ── Jina AI API (Phase 23: Dual API Embedding + Reranker) ──────────
# Comma-separated keys in JINA_API_KEY env var for round-robin rotation
JINA_API_KEYS = [
    k.strip() for k in
    os.environ.get("JINA_API_KEY", "").split(",")
    if k.strip()
]
JINA_API_URL = os.environ.get("JINA_API_URL", "https://api.jina.ai/v1")
JINA_EMBED_MODEL = os.environ.get("JINA_EMBED_MODEL", "jina-embeddings-v3")
JINA_EMBED_DIM = int(os.environ.get("JINA_EMBED_DIM", "768"))  # ChromaDB uyumu: 768
JINA_RERANK_MODEL = os.environ.get("JINA_RERANK_MODEL", "jina-reranker-v3")  # v3 SOTA, v2'den üstün

# ChromaDB Persist Directory for Embeddings
CHROMA_PERSIST_DIR = os.environ.get(
    "CHROMA_PERSIST_DIR",
    os.path.join(BASE_DIR, "vectordb")
)

# ── ChromaDB Singleton ──────────────────────────────────────────────
# All modules MUST use get_chroma_client() instead of PersistentClient() directly.
# Prevents SQLite lock contention from multiple PersistentClient instances.
_chroma_client = None
_chroma_lock = threading.Lock()


def get_chroma_client():
    """Thread-safe singleton ChromaDB PersistentClient."""
    global _chroma_client
    if _chroma_client is None:
        with _chroma_lock:
            if _chroma_client is None:
                import chromadb
                os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)
                _chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    return _chroma_client
