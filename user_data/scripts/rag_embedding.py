import os
import sqlite3
import hashlib
import json
import logging
import numpy as np
from dotenv import load_dotenv
from google import genai
from sentence_transformers import SentenceTransformer

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))

logger = logging.getLogger(__name__)

# Constants
from ai_config import AI_DB_PATH as DB_PATH
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

class DualEmbeddingPipeline:
    # Failover list: try in order, skip on error
    # gemini-embedding-001: GA, MTEB #1, 3072 native → Matryoshka to 768 for ChromaDB compat
    # gemini-embedding-2-preview: Newest multimodal preview
    GEMINI_EMBEDDING_MODELS = [
        {"name": "gemini-embedding-001", "dims": 768},
        {"name": "gemini-embedding-2-preview", "dims": 768},
    ]

    # Class-level singletons: loaded ONCE, shared across all instances
    _bge_model = None
    _genai_client = None

    def __init__(self):
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY is not set in .env")

        # genai Client singleton — prevents httpx connection pool leak
        if DualEmbeddingPipeline._genai_client is None:
            DualEmbeddingPipeline._genai_client = genai.Client(api_key=GEMINI_API_KEY)
        self.client = DualEmbeddingPipeline._genai_client

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
        
    def get_embeddings(self, text: str) -> dict:
        """
        Returns both Gemini and BGE embeddings for a given text.
        Checks SQLite cache first. Uncached texts are processed and then cached.
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
                    # Decode from JSON BLOB
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
        
        try:
            # Gemini Embedding with failover routing across available models
            gemini_emb = None
            for model_cfg in self.GEMINI_EMBEDDING_MODELS:
                try:
                    kwargs = {"model": model_cfg["name"], "contents": text}
                    if model_cfg.get("dims"):
                        kwargs["config"] = {"output_dimensionality": model_cfg["dims"]}
                    gemini_result = self.client.models.embed_content(**kwargs)

                    if hasattr(gemini_result, 'embeddings') and gemini_result.embeddings:
                        gemini_emb = gemini_result.embeddings[0].values
                    elif isinstance(gemini_result, dict) and 'embedding' in gemini_result:
                        gemini_emb = gemini_result['embedding']
                    else:
                        gemini_emb = gemini_result
                    logger.debug(f"Embedding via {model_cfg['name']} OK (dim={len(gemini_emb) if isinstance(gemini_emb, list) else '?'})")
                    break  # Success — stop trying
                except Exception as emb_err:
                    logger.warning(f"Embedding model {model_cfg['name']} failed: {emb_err}")
                    continue

            if gemini_emb is None:
                raise ValueError("All Gemini embedding models failed")
            
            # BGE-Financial Embedding (Financial Specific terms)
            bge_emb = self.local_model.encode(text, normalize_embeddings=True).tolist()
            
            # 3. Save to Cache
            try:
                with self._get_db_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        """
                        INSERT OR REPLACE INTO embedding_cache 
                        (text_hash, text_content, gemini_embedding, bge_embedding) 
                        VALUES (?, ?, ?, ?)
                        """,
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
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            return None

# Quick local test execution
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    pipeline = DualEmbeddingPipeline()
    test_text = "Bitcoin price surged past $60,000 following the Fed rate cut."
    
    logger.info("First run (Uncached)...")
    res1 = pipeline.get_embeddings(test_text)
    logger.info(f"Cached? {res1['cached']} | Gemini dim: {len(res1['gemini'])} | BGE dim: {len(res1['bge'])}")
    
    logger.info("Second run (Cached)...")
    res2 = pipeline.get_embeddings(test_text)
    logger.info(f"Cached? {res2['cached']} | Gemini dim: {len(res2['gemini'])} | BGE dim: {len(res2['bge'])}")
