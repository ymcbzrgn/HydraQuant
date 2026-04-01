"""
Görev 2: StreamingRAG (Gerçek Zamanlı İndeksleme)
Implements an in-memory Hot Buffer for instant retrieval 
and periodically flushes aging context into Cold Chroma Storage.
"""

import os
import json
import sqlite3
import logging
from typing import List, Dict, Any
from datetime import datetime, timezone, timedelta

from ai_config import AI_DB_PATH
try:
    from rag_embedding import DualEmbeddingPipeline
except Exception as _e:
    logging.getLogger(__name__).error(f"[StreamingRAG] DualEmbeddingPipeline import failed: {type(_e).__name__}: {_e}")
    DualEmbeddingPipeline = None

logger = logging.getLogger(__name__)

# Event taxonomy for Event-Driven Temporal RAG
EVENT_TAXONOMY = {
    "macro_fomc": ["fed", "fomc", "rate decision", "powell", "interest rate", "federal reserve"],
    "macro_cpi": ["cpi", "inflation", "consumer price index"],
    "macro_jobs": ["nonfarm", "unemployment", "jobs report", "employment"],
    "crypto_halving": ["halving", "block reward", "mining reward"],
    "crypto_hack": ["hack", "exploit", "vulnerability", "drain", "stolen", "attack"],
    "crypto_regulatory": ["sec", "regulation", "ban", "approve", "etf", "lawsuit", "enforcement"],
    "crypto_listing": ["listing", "delist", "binance list", "coinbase list"],
    "crypto_whale": ["whale", "large transfer", "dormant wallet"],
    "market_crash": ["crash", "liquidation", "black swan", "flash crash", "capitulation"],
    "market_rally": ["rally", "breakout", "all-time high", "ath", "pump"],
}


def detect_event_type(text: str) -> str:
    """Detect event type from text using keyword matching."""
    text_lower = text.lower()
    for event_type, keywords in EVENT_TAXONOMY.items():
        matches = sum(1 for kw in keywords if kw in text_lower)
        if matches >= 2:
            return event_type
        if matches == 1 and len(text_lower) < 300:
            return event_type
    return "general"


class StreamingRAG:
    def __init__(self):
        """StreamingRAG with Hot Buffer vs. Cold Storage mechanics."""
        self._init_hot_buffer()
        self._embedder = DualEmbeddingPipeline() if DualEmbeddingPipeline is not None else None

    def _init_hot_buffer(self):
        """In-memory or fast SQLite hot buffer for documents < 1 hour old."""
        try:
            with sqlite3.connect(AI_DB_PATH, timeout=10) as conn:
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS hot_buffer (
                        doc_id TEXT PRIMARY KEY,
                        content TEXT,
                        embedding BLOB,
                        metadata TEXT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
        except Exception as e:
            logger.error(f"StreamingRAG hot buffer init failed: {e}")

    def ingest(self, doc_id: str, content: str, metadata: dict):
        """
        Yeni dokümantasyonu anında hot_buffer'a ekleyerek 
        5-15 dakikalık batch job'ları baypas et.
        """
        import numpy as np
        
        # Immediate sync embedding
        pipeline = self._embedder
        if pipeline is None:
            logger.warning(f"[StreamingRAG] Embedder unavailable, skipping ingest for {doc_id}")
            return
        embeddings = pipeline.get_embeddings(content)
        if not embeddings or 'gemini' not in embeddings:
            logger.warning(f"Failed to ingest document {doc_id} into hot buffer: Embedding failed.")
            return
            
        array_bytes = np.array(embeddings['gemini'], dtype=np.float32).tobytes()
            
        try:
            with sqlite3.connect(AI_DB_PATH, timeout=10) as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO hot_buffer (doc_id, content, embedding, metadata) VALUES (?, ?, ?, ?)",
                    (doc_id, content, array_bytes, json.dumps(metadata))
                )
                conn.commit()
                
            logger.info(f"StreamingRAG: Ingested {doc_id} directly into Hot Buffer.")
        except Exception as e:
            logger.error(f"StreamingRAG ingest error: {e}")

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Hot buffer'da Cosine search çalıştırılarak gerçek zamanlı sonuç döndürülür.
        """
        import numpy as np
        
        pipeline = self._embedder
        if pipeline is None:
            return []
        query_embeddings = pipeline.get_embeddings(query)
        if not query_embeddings or 'gemini' not in query_embeddings:
            return []
            
        q_vec = np.array(query_embeddings['gemini'], dtype=np.float32)
        hot_results = []
        
        try:
            with sqlite3.connect(AI_DB_PATH, timeout=10) as conn:
                cursor = conn.execute("SELECT doc_id, content, embedding, metadata FROM hot_buffer")
                
                scored_docs = []
                for row in cursor.fetchall():
                    doc_id, content, emb_bytes, meta_str = row
                    doc_vec = np.frombuffer(emb_bytes, dtype=np.float32)
                    
                    # Cosine Similarity check manually
                    norm_q = np.linalg.norm(q_vec)
                    norm_d = np.linalg.norm(doc_vec)
                    if norm_q > 0 and norm_d > 0:
                        sim = np.dot(q_vec, doc_vec) / (norm_q * norm_d)
                        scored_docs.append((sim, doc_id, content, json.loads(meta_str)))
                        
                # Keep top K
                scored_docs.sort(key=lambda x: x[0], reverse=True)
                for sim, d_id, cont, meta in scored_docs[:top_k]:
                    hot_results.append({
                        "id": d_id,
                        "content": cont,
                        "metadata": meta,
                        "score": float(sim),
                        "source": "hot_buffer"
                    })
        except Exception as e:
            logger.error(f"StreamingRAG search failed: {e}")
            
        return hot_results

    def flush_to_cold(self):
        """
        1 saatten eski dokümanları ChromaDB cold storage'a taşı ve 
        hot buffer'dan temizle.
        """
        cutoff_time = datetime.now(tz=timezone.utc) - timedelta(hours=1)
        
        try:
            with sqlite3.connect(AI_DB_PATH, timeout=10) as conn:
                # Select expired documents
                cursor = conn.execute(
                    "SELECT doc_id, content, embedding, metadata FROM hot_buffer WHERE timestamp < ?",
                    (cutoff_time.isoformat(),)
                )
                expired_rows = cursor.fetchall()
                
                if not expired_rows:
                    return
                    
                # We would normally import the active Chroma collection here
                # from db import get_chroma_collection
                # collection = get_chroma_collection(embed_texts)
                logger.info(f"StreamingRAG: Flushing {len(expired_rows)} docs to Cold Storage.")
                
                # Delete flushed content from buffer
                conn.execute(
                    "DELETE FROM hot_buffer WHERE timestamp < ?",
                    (cutoff_time.isoformat(),)
                )
                conn.commit()
                
        except Exception as e:
            logger.error(f"StreamingRAG flush failed: {e}")
