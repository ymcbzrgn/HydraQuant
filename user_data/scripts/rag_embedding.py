import os
import sqlite3
import hashlib
import json
import logging
import numpy as np
from dotenv import load_dotenv
import google.generativeai as genai
from sentence_transformers import SentenceTransformer

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))

logger = logging.getLogger(__name__)

# Constants
DB_PATH = os.path.join(os.path.dirname(__file__), "..", "db", "ai_data.sqlite")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

class DualEmbeddingPipeline:
    def __init__(self):
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY is not set in .env")
            
        genai.configure(api_key=GEMINI_API_KEY)
        self.gemini_model = "models/gemini-embedding-001"
        
        logger.info("Loading local BGE-Financial embedding model...")
        self.local_model = SentenceTransformer("philschmid/bge-base-financial-matryoshka")
        
    def _get_db_connection(self):
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
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
            # Gemini Embedding (Semantic, General)
            # Freqtrade uses the newer model text-embedding-004
            gemini_result = genai.embed_content(
                model=self.gemini_model,
                content=text,
                task_type="retrieval_document"
            )
            gemini_emb = gemini_result['embedding']
            
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
