import os
import math
import sqlite3
import logging
from datetime import datetime, timezone
from typing import List, Dict, Any

from db import get_db_connection
try:
    from rag_embedding import DualEmbeddingPipeline
except ImportError as _imp_err:
    import logging as _lg
    _lg.getLogger(__name__).error(f"[IMPORT] DualEmbeddingPipeline failed: {_imp_err}. Embedding disabled.")
    DualEmbeddingPipeline = None
from rag_chunker import ContentChunker

# Phase 14 & 15: StreamingRAG, RAPTOR, MAGMA
from streaming_rag import StreamingRAG
from raptor_tree import RAPTORTree
from magma_memory import MAGMAMemory
from memo_rag import MemoRAG
from ai_config import AI_DB_PATH, get_chroma_client

logger = logging.getLogger(__name__)

from ai_config import AI_DB_PATH as DB_PATH, CHROMA_PERSIST_DIR as VECTOR_DB_DIR

class HybridRetriever:
    """
    Implements Hybrid Search combining:
    1. Dense Search (ChromaDB with Gemini + BGE Matryoshka)
    2. Sparse Search (BM25 keyword search)
    3. Reranking (FlashRank Cross-Encoder)
    """
    
    def __init__(self, collection_name: str = "crypto_news", llm_router=None):
        self._llm_router = llm_router  # Pass-through for sub-components
        self.chroma_client = get_chroma_client()
        # Primary: Gemini embeddings (general semantic, 768-dim)
        # embedding_function=None: we provide pre-computed embeddings via add(embeddings=...)
        # Without this, ChromaDB defaults to its own 384-dim DefaultEmbeddingFunction,
        # causing dimension mismatch when our 768-dim vectors are inserted.
        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
            embedding_function=None
        )
        # Secondary: BGE-Financial embeddings (domain-specific, 768-dim)
        self.bge_collection = self.chroma_client.get_or_create_collection(
            name=f"{collection_name}_bge",
            metadata={"hnsw:space": "cosine"},
            embedding_function=None
        )
        self.embedder = DualEmbeddingPipeline() if DualEmbeddingPipeline is not None else None
        if self.embedder is None:
            logger.error("[HybridRetriever] DualEmbeddingPipeline unavailable. Search will be degraded.")
        try:
            from flashrank import Ranker, RerankRequest
            self.reranker = Ranker(model_name="ms-marco-TinyBERT-L-2-v2", cache_dir=os.path.join(VECTOR_DB_DIR, "flashrank_cache"))
            self._RerankRequest = RerankRequest
        except ImportError:
            logger.warning("FlashRank not found, disabling FlashRank component.")
            self.reranker = None
            self._RerankRequest = None
            
        try:
            from colbert_reranker import ColBERTReranker
            self.colbert_reranker = ColBERTReranker()
        except Exception as e:
            logger.warning(f"ColBERT disabled: {e}")
            self.colbert_reranker = None

        try:
            from binary_quantizer import BinaryQuantizer
            self.binary_quantizer = BinaryQuantizer()
        except ImportError:
            logger.warning("BinaryQuantizer not available, disabling binary pre-filter.")
            self.binary_quantizer = None
            
        # Phase 14 Instantiations — pass shared router to avoid duplicate LLMRouter inits
        self.streaming_rag = StreamingRAG()
        self.raptor = RAPTORTree(llm_router=self._llm_router)
        self.magma = MAGMAMemory()
        self.memorag = MemoRAG(llm_router=self._llm_router)

    _db_tables_ensured = False

    def _get_db_connection(self):
        conn = sqlite3.connect(DB_PATH, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")

        # Create binary_embeddings table once per process, not every connection
        if not HybridRetriever._db_tables_ensured:
            conn.execute('''CREATE TABLE IF NOT EXISTS binary_embeddings (
                doc_id TEXT PRIMARY KEY,
                packed_bge BLOB
            )''')
            HybridRetriever._db_tables_ensured = True
        return conn

    def add_documents(self, documents: List[str], metadatas: List[Dict[str, Any]], ids: List[str]) -> int:
        """Embeds and adds documents to ChromaDB collections + SQLite FTS5.
        Returns number of successfully embedded documents (0 = FTS5-only, no vectors)."""
        embedded_count = 0

        # --- Phase 1: Generate embeddings (if embedder available) ---
        gemini_embeddings = []
        bge_embeddings = []
        valid_indices = []  # Track which docs got valid embeddings

        if self.embedder is not None:
            for i, doc in enumerate(documents):
                try:
                    embs = self.embedder.get_embeddings(doc)
                    g = embs.get('gemini', [])
                    b = embs.get('bge', [])
                    if g and b:  # Both must be non-empty for ChromaDB
                        gemini_embeddings.append(g)
                        bge_embeddings.append(b)
                        valid_indices.append(i)
                    else:
                        logger.warning(f"[HybridRetriever] Empty embedding for doc {ids[i]}, skipping ChromaDB.")
                except Exception as e:
                    logger.warning(f"[HybridRetriever] Embedding failed for doc {ids[i]}: {e}")
        else:
            logger.warning("[HybridRetriever] Embedder unavailable — documents will be added to FTS5 only (keyword search).")

        # --- Phase 2: Store vectors in ChromaDB (only valid embeddings) ---
        if valid_indices:
            valid_ids = [ids[i] for i in valid_indices]
            valid_docs = [documents[i] for i in valid_indices]
            valid_metas = [metadatas[i] for i in valid_indices]

            try:
                self.collection.add(
                    ids=valid_ids,
                    embeddings=gemini_embeddings,
                    documents=valid_docs,
                    metadatas=valid_metas
                )
                self.bge_collection.add(
                    ids=valid_ids,
                    embeddings=bge_embeddings,
                    documents=valid_docs,
                    metadatas=valid_metas
                )
                embedded_count = len(valid_indices)
            except Exception as e:
                logger.error(f"[HybridRetriever] ChromaDB add failed: {e}")

        # --- Phase 3: ALWAYS add to FTS5 (no embeddings needed) + Binary BGE ---
        # O(1) lookup for binary BGE phase
        _valid_set = set(valid_indices)
        _valid_pos = {orig_i: pos for pos, orig_i in enumerate(valid_indices)}
        try:
            packed_bges = None
            if valid_indices and hasattr(self, 'binary_quantizer') and self.binary_quantizer:
                import numpy as np
                packed_bges = self.binary_quantizer.binarize_and_pack(np.array(bge_embeddings))

            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                for i, (doc_id, doc_text) in enumerate(zip(ids, documents)):
                    cursor.execute("DELETE FROM bm25_index WHERE doc_id = ?", (doc_id,))
                    cursor.execute(
                        "INSERT INTO bm25_index (doc_id, content) VALUES (?, ?)",
                        (doc_id, doc_text)
                    )

                    # Binary BGE only for docs that have valid embeddings
                    if packed_bges is not None and i in _valid_set:
                        cursor.execute(
                            "INSERT OR REPLACE INTO binary_embeddings (doc_id, packed_bge) VALUES (?, ?)",
                            (doc_id, packed_bges[_valid_pos[i]].tobytes())
                        )
                conn.commit()

            if embedded_count > 0:
                logger.info(f"Added {len(documents)} docs to FTS5 + {embedded_count} to ChromaDB.")
            else:
                logger.info(f"Added {len(documents)} docs to FTS5 only (keyword search). ChromaDB: 0 (embedder {'unavailable' if self.embedder is None else 'failed'}).")
        except Exception as e:
            logger.error(f"Error adding to SQLite FTS5 / Binary BGE: {e}")

        return embedded_count

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
        1. MemoRAG Draft -> Expands the original query with global conceptual context
        2. Query -> SQLite FTS5 (BM25) -> Top 300 candidates
        3. Binary Quantization -> Pre-filter BM25 Top 300 to Top 30 via Hamming distance on BGE
        4. Query -> Chroma (Dense Gemini) -> Top 30
        5. RRF Fusion -> Top 20
        6. FlashRank + ColBERT -> Top K
        """
        # Phase 15: Generate MemoRAG Global Draft Context
        original_query = query
        if self.memorag:
            try:
                draft = self.memorag.generate_draft(query)
                if draft and draft != query:
                    query = f"{query} | Context Draft: {draft}"
                    logger.info("MemoRAG injected global draft into search query.")
            except Exception as e:
                logger.warning(f"MemoRAG draft failed: {e}")

        query_embs = None
        if self.embedder is not None:
            try:
                query_embs = self.embedder.get_embeddings(query)
                # If embedder returned empty vectors, treat as unavailable
                if query_embs and not query_embs.get("gemini"):
                    query_embs = None
            except Exception as e:
                logger.error(f"[HybridRetriever] Embedding query failed: {e}")
                query_embs = None

        if query_embs is None:
            logger.warning("[HybridRetriever] Embedder unavailable — falling back to BM25-only search.")

        # 1. Sparse Search (SQLite FTS5 BM25) - Widen the funnel
        bm25_top_ids = []
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                # Build OR query from individual terms for high recall.
                # Rerankers downstream handle precision.
                # Strip ALL non-alphanumeric chars to prevent FTS5 syntax errors.
                # Coin pairs like "0G/USDT:USDT" contain / and : which are FTS5 operators.
                # Also prevents ( ) * + - AND OR NOT from breaking MATCH queries.
                sanitized = "".join(c if c.isalnum() else " " for c in original_query)
                terms = [t for t in sanitized.split() if len(t) > 1]
                if terms:
                    fts_query = " OR ".join(terms)
                else:
                    # Fallback: use any single-char terms if no multi-char terms exist
                    terms = [t for t in sanitized.split() if t]
                    fts_query = " OR ".join(terms) if terms else "crypto"

                cursor.execute(
                    "SELECT doc_id FROM bm25_index WHERE bm25_index MATCH ? ORDER BY rank LIMIT 300",
                    (fts_query,)
                )
                rows = cursor.fetchall()
                bm25_top_ids = [row['doc_id'] for row in rows]
                
                # 2. Pre-filter BM25 results with Binary Quantization (BGE Hamming Distance)
                # Instead of hitting Chroma with float BGE, we do an ultra-fast local binary filter
                if bm25_top_ids and query_embs and hasattr(self, 'binary_quantizer') and self.binary_quantizer:
                    import numpy as np
                    q_bin = self.binary_quantizer.binarize_and_pack(np.array(query_embs['bge']))
                    placeholders = ",".join(["?"] * len(bm25_top_ids))
                    cursor.execute(f"SELECT doc_id, packed_bge FROM binary_embeddings WHERE doc_id IN ({placeholders})", bm25_top_ids)
                    bin_rows = cursor.fetchall()
                    
                    if bin_rows:
                        doc_ids = [r['doc_id'] for r in bin_rows]
                        doc_blobs = [np.frombuffer(r['packed_bge'], dtype=np.uint8) for r in bin_rows]
                        doc_packed = np.array(doc_blobs)
                        
                        distances = self.binary_quantizer.hamming_distance(q_bin, doc_packed)
                        
                        # Sort by smallest hamming distance (most similar)
                        scored_bin = list(zip(doc_ids, distances))
                        scored_bin.sort(key=lambda x: x[1])
                        # Replace BM25 vast list with tightly dense-filtered top 30
                        bm25_top_ids = [x[0] for x in scored_bin[:30]]
                else:
                    # Fallback if no binary quantizer
                    bm25_top_ids = bm25_top_ids[:30]
                    
        except Exception as e:
            logger.error(f"SQLite FTS5 / Binary Search failed: {e}")

        # 3. Dense Search — Gemini embeddings (only if embedder available)
        dense_gemini_ids = []

        if query_embs:
            collection_count = self.collection.count()
            if collection_count > 0:
                dense_results = self.collection.query(
                    query_embeddings=[query_embs['gemini']],
                    n_results=min(30, collection_count)
                )
                dense_gemini_ids = dense_results['ids'][0] if dense_results['ids'] else []

        # (BGE float search is bypassed completely by the binary quantizer pre-filter above)
        dense_bge_ids = []

        # 3. Reciprocal Rank Fusion (3-way: BM25 + Gemini Dense + BGE Dense)
        fused_ids = self.reciprocal_rank_fusion([bm25_top_ids, dense_gemini_ids, dense_bge_ids])
        fused_top_20 = fused_ids[:20]

        # Fetch actual documents for generating reranking payloads
        passages = []
        found_ids = set()
        if fused_top_20:
            # Primary: fetch from ChromaDB (has metadata + parent text)
            try:
                fetched = self.collection.get(ids=fused_top_20, include=["documents", "metadatas"])
                if fetched and fetched['documents']:
                    for i, doc_id in enumerate(fetched['ids']):
                        meta = fetched['metadatas'][i] if fetched['metadatas'] else {}
                        child_text = fetched['documents'][i]

                        # Parent-Child Retrieval: return FULL parent text for child chunks
                        if meta.get('type') == 'news_child' and meta.get('parent_text'):
                            display_text = meta['parent_text']
                        else:
                            display_text = child_text

                        passages.append({
                            "id": doc_id,
                            "text": display_text,
                            "meta": meta
                        })
                        found_ids.add(doc_id)
            except Exception as e:
                logger.warning(f"[HybridRetriever] ChromaDB fetch failed: {e}")

            # Fallback: fetch missing docs from FTS5 (for FTS5-only docs without embeddings)
            missing_ids = [did for did in fused_top_20 if did not in found_ids]
            if missing_ids:
                try:
                    count_before = len(found_ids)
                    with self._get_db_connection() as conn:
                        cursor = conn.cursor()
                        placeholders = ",".join(["?"] * len(missing_ids))
                        cursor.execute(
                            f"SELECT doc_id, content FROM bm25_index WHERE doc_id IN ({placeholders})",
                            missing_ids
                        )
                        for row in cursor.fetchall():
                            passages.append({
                                "id": row['doc_id'],
                                "text": row['content'],
                                "meta": {"source": "fts5_fallback"}
                            })
                            found_ids.add(row['doc_id'])
                    recovered = len(found_ids) - count_before
                    if recovered > 0:
                        logger.info(f"[HybridRetriever] Recovered {recovered}/{len(missing_ids)} docs from FTS5 fallback.")
                except Exception as e:
                    logger.warning(f"[HybridRetriever] FTS5 text fallback failed: {e}")

        # Phase 14: StreamingRAG Integration (Boost recent hot memory)
        try:
            hot_docs = self.streaming_rag.search(query, top_k=3)
            # Add directly to passages avoiding RRF decay
            for hd in hot_docs:
                # Add unique identifier preventing duplicate reranker issues
                passages.append({
                    "id": f"hot_{hd['id']}",
                    "text": hd['content'],
                    "meta": hd.get('metadata', {})
                })
                logger.info(f"StreamRAG injected '{hd['id']}' [Score: {hd['score']:.2f}]")
        except Exception as e:
            logger.error(f"StreamingRAG Search error: {e}")
            
        # Phase 14: RAPTOR Hierarchy Injection 
        try:
            raptor_summaries = self.raptor.query(query, tree_or_db=True)
            for rs in raptor_summaries:
                passages.append({
                    "id": rs["id"],
                    "text": rs["text"],
                    "meta": {"type": "raptor_summary", "level": rs["level"]}
                })
        except Exception as e:
            logger.error(f"RAPTOR Search error: {e}")
            
        # Phase 15: MAGMA Graph Context Extraction
        try:
            # Send the generic query into MAGMA memory searching all 4 graphs
            magma_edges = self.magma.query(query, max_hops=1)
            if magma_edges:
                edge_strings = [f"{e['source']} --[{e['relation']}]--> {e['target']}" for e in magma_edges[:5]]
                passages.append({
                    "id": f"magma_context_{hash(query)}",
                    "text": "MAGMA Multi-Graph Connections: " + "; ".join(edge_strings),
                    "meta": {"type": "magma_context"}
                })
                logger.info(f"MAGMA added {len(edge_strings)} high-weight memory nodes to passages.")
        except Exception as e:
            logger.error(f"MAGMA Search error: {e}")

        if not passages:
            return []

        # Phase 3.15: Temporal Decay — penalize old news before reranking
        passages = self._apply_temporal_decay(passages)

        # 4. Multi-Reranker Ensemble
        flashrank_results = []
        if self.reranker and self._RerankRequest:
            rerank_request = self._RerankRequest(query=query, passages=passages)
            flashrank_results = self.reranker.rerank(rerank_request)
            if flashrank_results:
                max_score = max(float(doc.get("score", 0.0)) for doc in flashrank_results)
                min_score = min(float(doc.get("score", 0.0)) for doc in flashrank_results)
                range_score = max_score - min_score if max_score > min_score else 1.0
                for doc in flashrank_results:
                    doc["flashrank_normalized"] = (float(doc.get("score", 0.0)) - min_score) / range_score
                    
        colbert_results = []
        if self.colbert_reranker:
            # We want to score all candidates, so top_k is len(passages)
            colbert_results = self.colbert_reranker.rerank(query, passages, top_k=len(passages))
            
        final_results = self._ensemble_rerank(passages, flashrank_results, colbert_results)
        
        # Return final top_k
        return final_results[:top_k]

    def _ensemble_rerank(self, base_passages, flashrank_results, colbert_results, alpha=0.5):
        """Combines FlashRank and ColBERT normalized scores."""
        flash_dict = {doc['id']: doc for doc in flashrank_results}
        colbert_dict = {doc['id']: doc for doc in colbert_results}
        
        ensemble_results = []
        for doc in base_passages:
            doc_id = doc['id']
            f_norm = flash_dict.get(doc_id, {}).get("flashrank_normalized", 0.0)
            c_norm = colbert_dict.get(doc_id, {}).get("colbert_normalized", 0.0)
            
            if not flash_dict and colbert_dict:
                final_score = c_norm
            elif flash_dict and not colbert_dict:
                final_score = f_norm
            elif not flash_dict and not colbert_dict:
                final_score = float(doc.get("score", 0.0))
            else:
                final_score = alpha * f_norm + (1 - alpha) * c_norm
                
            doc_copy = doc.copy()
            doc_copy["ensemble_score"] = final_score
            doc_copy["score"] = final_score # Used later for standard sorting if needed
            ensemble_results.append(doc_copy)
            
        ensemble_results.sort(key=lambda x: x["ensemble_score"], reverse=True)
        return ensemble_results

    def _apply_temporal_decay(
        self,
        results: List[Dict[str, Any]],
        half_life_days: float = 7.0,
        alpha: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Apply temporal decay to search results.
        Formula: score = alpha * relevance + (1-alpha) * 0.5^(age/half_life)
        
        - 1-hour-old news: decay ≈ 1.0 → score barely reduced
        - 7-day-old news: decay = 0.5 → 30% penalty
        - 30-day-old news: decay ≈ 0.05 → ~28.5% penalty
        - 90-day-old news: decay ≈ 0.0002 → killed
        """
        now = datetime.now(tz=timezone.utc)
        
        for result in results:
            meta = result.get('meta', {})
            pub_date_str = meta.get('published_at') or meta.get('date') or meta.get('timestamp')
            
            if pub_date_str:
                try:
                    pub_date = datetime.fromisoformat(str(pub_date_str).replace('Z', '+00:00'))
                    if pub_date.tzinfo is None:
                        pub_date = pub_date.replace(tzinfo=timezone.utc)
                    age_days = (now - pub_date).total_seconds() / 86400.0
                    decay = math.pow(0.5, age_days / half_life_days)
                except (ValueError, TypeError):
                    decay = 0.5  # Unknown date → moderate penalty
            else:
                decay = 0.5  # No date metadata → moderate penalty
            
            original_score = float(result.get('score', 1.0))
            result['score'] = alpha * original_score + (1 - alpha) * decay
        
        # Re-sort by decayed score (highest first)
        results.sort(key=lambda x: float(x.get('score', 0)), reverse=True)
        
        return results

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
