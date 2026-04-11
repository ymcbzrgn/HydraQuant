import logging
import os
import sys

logger = logging.getLogger(__name__)

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
from ai_config import LANCE_DB_DIR

def init_lancedb():
    """Initializes the LanceDB tables for Hybrid RAG (Phase 28)."""
    from lance_store import get_lance_store

    store = get_lance_store()

    tables = [
        "crypto_news",        # 180-day rolling window of news (Gemini embeddings)
        "crypto_news_bge",    # Same news, Jina/BGE embeddings
        "market_reports",     # Deep dive research & RAPTOR trees
        "trade_history",      # Past trades and rationale
    ]

    for table_name in tables:
        store.get_or_create_table(name=table_name, dim=768, metric="cosine")
        logger.info(f"Initialized LanceDB table: {table_name}")

    logger.info(f"LanceDB setup complete ({len(tables)} tables).")

# Legacy alias
init_chromadb = init_lancedb

def download_rag_models():
    """Pre-downloads embedding models and rerankers so offline capability is preserved."""
    try:
        from sentence_transformers import SentenceTransformer
        logger.info("Downloading BGE-Base-Financial-Matryoshka (Dual Embedding 2/2)...")
        SentenceTransformer("philschmid/bge-base-financial-matryoshka")

        logger.info("Downloading all-MiniLM-L6-v2 (Fallback)...")
        SentenceTransformer("all-MiniLM-L6-v2")
    except ImportError as e:
        logger.warning(f"Could not download embedding models: {e}")

    try:
        from flashrank import Ranker
        logger.info("Downloading FlashRank models...")
        Ranker()
    except ImportError as e:
        logger.warning(f"Could not download FlashRank models: {e}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    os.makedirs(LANCE_DB_DIR, exist_ok=True)

    init_lancedb()
    download_rag_models()
