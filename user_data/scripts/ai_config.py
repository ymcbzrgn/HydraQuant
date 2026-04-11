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
JINA_EMBED_DIM = int(os.environ.get("JINA_EMBED_DIM", "768"))  # LanceDB dim: 768
JINA_RERANK_MODEL = os.environ.get("JINA_RERANK_MODEL", "jina-reranker-v3")  # v3 SOTA, v2'den üstün

# ── Vector DB Paths ──────────────────────────────────────────────
# Phase 28: ChromaDB replaced by LanceDB
CHROMA_PERSIST_DIR = os.environ.get(
    "CHROMA_PERSIST_DIR",
    os.path.join(BASE_DIR, "vectordb")
)  # LEGACY — kept for migration script reference only

LANCE_DB_DIR = os.environ.get(
    "LANCE_DB_DIR",
    os.path.join(BASE_DIR, "db", "lancedb")
)

# ── LanceDB Singleton (Phase 28) ──────────────────────────────────
# All modules MUST use get_lance_store() from lance_store.py.
# Legacy get_chroma_client() still works for migration period.

_chroma_client = None
_chroma_lock = threading.Lock()


def get_chroma_client():
    """LEGACY: ChromaDB client. Use get_lance_store() for new code."""
    global _chroma_client
    if _chroma_client is None:
        with _chroma_lock:
            if _chroma_client is None:
                try:
                    import chromadb
                    os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)
                    _chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
                except ImportError:
                    raise ImportError(
                        "chromadb not installed. Phase 28 migration complete — "
                        "use get_lance_store() from lance_store.py instead."
                    )
    return _chroma_client


def get_lance_store():
    """Phase 28: Thread-safe singleton LanceDB store. Preferred over get_chroma_client()."""
    from lance_store import get_lance_store as _get
    return _get()
