import os
import threading

# Base paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Database Path for SQLite (FTS5, Logging, Settings)
AI_DB_PATH = os.environ.get(
    "AI_DB_PATH",
    os.path.join(BASE_DIR, "db", "ai_data.sqlite")
)

# Model Server (BGE, ColBERT, FlashRank served via HTTP)
MODEL_SERVER_URL = os.environ.get("MODEL_SERVER_URL", "http://127.0.0.1:8895")

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
