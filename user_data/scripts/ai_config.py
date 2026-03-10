import os

# Base paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Database Path for SQLite (FTS5, Logging, Settings)
AI_DB_PATH = os.environ.get(
    "AI_DB_PATH",
    os.path.join(BASE_DIR, "db", "ai_data.sqlite")
)

# ChromaDB Persist Directory for Embeddings
CHROMA_PERSIST_DIR = os.environ.get(
    "CHROMA_PERSIST_DIR",
    os.path.join(BASE_DIR, "vectordb")
)
