import chromadb
from chromadb.config import Settings
import logging
import os
import sys

logger = logging.getLogger(__name__)

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VECTOR_DB_DIR = os.path.join(BASE_DIR, "vectordb")

def init_chromadb():
    """Initializes the persistent ChromaDB collections for Hybrid RAG."""
    os.makedirs(VECTOR_DB_DIR, exist_ok=True)
    
    # Initialize Persistent ChromaDB Client
    client = chromadb.PersistentClient(path=VECTOR_DB_DIR)
    
    collections = [
        "crypto_news",        # 180-day rolling window of news
        "market_reports",     # Deep dive research & RAPTOR trees
        "trade_history"       # Past trades and rationale
    ]
    
    for coll_name in collections:
        # We use cosine similarity as standard for financial embedding models
        client.get_or_create_collection(
            name=coll_name,
            metadata={"hnsw:space": "cosine"}
        )
        logger.info(f"Initialized ChromaDB collection: {coll_name}")

    logger.info("ChromaDB architecture setup complete.")
    
def download_rag_models():
    """Pre-downloads embedding models and rerankers so offline capability is preserved."""
    try:
        from sentence_transformers import SentenceTransformer
        logger.info("Downloading BGE-Base-Financial-Matryoshka (Dual Embedding 2/2)...")
        # Lightweight local financial model
        SentenceTransformer("philschmid/bge-base-financial-matryoshka")
        
        logger.info("Downloading all-MiniLM-L6-v2 (Fallback)...")
        SentenceTransformer("all-MiniLM-L6-v2")
    except ImportError as e:
        logger.warning(f"Could not download embedding models: {e}")

    try:
        from flashrank import Ranker
        logger.info("Downloading FlashRank models...")
        # Automatically downloads the tiny default cross-encoder to a local cache
        Ranker()
    except ImportError as e:
        logger.warning(f"Could not download FlashRank models: {e}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Add empty .gitignore so vectordb stays out of github
    os.makedirs(VECTOR_DB_DIR, exist_ok=True)
    with open(os.path.join(VECTOR_DB_DIR, ".gitignore"), "w") as f:
        f.write("*\n!.gitignore\n")
        
    init_chromadb()
    download_rag_models()
