import os
import sqlite3
import logging
import chromadb
from flashrank import Ranker, RerankRequest
from typing import List, Dict, Any

from rag_embedding import DualEmbeddingPipeline

logger = logging.getLogger(__name__)

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "db", "ai_data.sqlite")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VECTOR_DB_DIR = os.path.join(BASE_DIR, "vectordb")

class HybridRetriever:
    """
    Implements Hybrid Search combining:
    1. Dense Search (ChromaDB with Gemini + BGE Matryoshka)
    2. Sparse Search (BM25 keyword search)
    3. Reranking (FlashRank Cross-Encoder)
    """
    
    def __init__(self, collection_name: str = "crypto_news"):
        self.chroma_client = chromadb.PersistentClient(path=VECTOR_DB_DIR)
        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name, 
            metadata={"hnsw:space": "cosine"}
        )
        self.embedder = DualEmbeddingPipeline()
        self.reranker = Ranker(model_name="ms-marco-TinyBERT-L-2-v2", cache_dir=os.path.join(VECTOR_DB_DIR, "flashrank_cache"))
        
    def _get_db_connection(self):
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        return conn

    def add_documents(self, documents: List[str], metadatas: List[Dict[str, Any]], ids: List[str]):
        """Embeds and adds documents to ChromaDB and updates SQLite FTS5 BM25 index."""
        gemini_embeddings = []
        # We index using the primary semantic model (Gemini) for ChromaDB
        # BGE embeddings can be stored in metadata for dual-retrieval
        for doc in documents:
            embs = self.embedder.get_embeddings(doc)
            gemini_embeddings.append(embs['gemini'])
            
        self.collection.add(
            ids=ids,
            embeddings=gemini_embeddings,
            documents=documents,
            metadatas=metadatas
        )
        
        # Add to FTS5 SQLite index
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                for doc_id, doc_text in zip(ids, documents):
                    # FTS5 doesn't natively support INSERT OR REPLACE gracefully without rowid,
                    # so we delete the existing doc_id if it exists, then insert.
                    cursor.execute("DELETE FROM bm25_index WHERE doc_id = ?", (doc_id,))
                    cursor.execute(
                        "INSERT INTO bm25_index (doc_id, content) VALUES (?, ?)", 
                        (doc_id, doc_text)
                    )
                conn.commit()
            logger.info(f"Added {len(documents)} docs to Chroma DB & SQLite FTS5.")
        except Exception as e:
            logger.error(f"Error adding to SQLite FTS5 BM25: {e}")

    def reciprocal_rank_fusion(self, results_lists: List[List[str]], k=60) -> List[str]:
        """Calculates RRF score to combine multiple ranked lists."""
        rrf_scores = {}
        for ranked_list in results_lists:
            for rank, doc_id in enumerate(ranked_list):
                if doc_id not in rrf_scores:
                    rrf_scores[doc_id] = 0.0
                rrf_scores[doc_id] += 1.0 / (k + rank + 1)
                
        # Sort desc by RRF score
        sorted_fused = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return [doc_id for doc_id, score in sorted_fused]

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Executes the full Hybrid Search:
        1. Query -> SQLite FTS5 (BM25) -> Top 30
        2. Query -> Chroma (Dense) -> Top 30
        3. RRF Fusion -> Top 20
        4. FlashRank -> Top K
        """
        # 1. Sparse Search (SQLite FTS5 BM25)
        bm25_top_ids = []
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                # FTS5 expects double quotes around phrases if they contain spaces
                sanitized_query = query.replace('"', '').replace("'", "")
                # Wrap the query in double quotes for exact or phrase matching
                fts_query = f'"{sanitized_query}"'
                
                cursor.execute(
                    "SELECT doc_id FROM bm25_index WHERE bm25_index MATCH ? ORDER BY rank LIMIT 30", 
                    (fts_query,)
                )
                rows = cursor.fetchall()
                bm25_top_ids = [row['doc_id'] for row in rows]
        except Exception as e:
            logger.error(f"SQLite FTS5 Search failed: {e}")

        # 2. Dense Search (ChromaDB utilizing Gemini vectors)
        query_embs = self.embedder.get_embeddings(query)
        dense_top_ids = []
        
        # We need to know corpus size to avoid ChromaDB querying errors
        collection_count = self.collection.count()
        if collection_count > 0:
            dense_results = self.collection.query(
                query_embeddings=[query_embs['gemini']],
                n_results=min(30, collection_count)
            )
            dense_top_ids = dense_results['ids'][0] if dense_results['ids'] else []

        # 3. Reciprocal Rank Fusion (RRF)
        fused_ids = self.reciprocal_rank_fusion([bm25_top_ids, dense_top_ids])
        fused_top_20 = fused_ids[:20]

        # Fetch actual documents for generating reranking payloads
        passages = []
        if fused_top_20:
            fetched = self.collection.get(ids=fused_top_20)
            if fetched and fetched['documents']:
                for i, doc_id in enumerate(fetched['ids']):
                    passages.append({
                        "id": doc_id,
                        "text": fetched['documents'][i],
                        "meta": fetched['metadatas'][i] if fetched['metadatas'] else {}
                    })

        if not passages:
            return []

        # 4. Cross-Encoder Reranking (FlashRank)
        rerank_request = RerankRequest(query=query, passages=passages)
        reranked_results = self.reranker.rerank(rerank_request)
        
        # Return final top_k
        return reranked_results[:top_k]

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    retriever = HybridRetriever()
    
    # Fake indexing test (We drop the table and chroma for clean test)
    # Be careful, this test adds docs on every run
    if retriever.collection.count() < 4:
        logger.info("Indexing fake documents for test...")
        fake_docs = [
            "Bitcoin is a decentralized cryptocurrency created in 2009.",
            "Tether (USDT) is a stablecoin pegged to the US Dollar.",
            "Federal Reserve cut interest rates today, sparking crypto rallies.",
            "Ethereum smart contracts power the DeFi ecosystem."
        ]
        fake_ids = [f"doc_{i}" for i in range(len(fake_docs))]
        fake_metas = [{"source": "test"} for _ in fake_docs]
        retriever.add_documents(fake_docs, fake_metas, fake_ids)
        
    logger.info("Testing Hybrid Search...")
    results = retriever.search("What is the effect of Fed rates on crypto?", top_k=2)
    for i, res in enumerate(results):
        logger.info(f"Rank {i+1} (Score: {res['score']:.4f}): {res['text']}")
