import os
import math
import sqlite3
import logging
import time
from datetime import datetime, timezone
from typing import List, Dict, Any

from db import get_db_connection, get_connection
try:
    from rag_embedding import DualEmbeddingPipeline
except Exception as _imp_err:
    import logging as _lg
    _lg.getLogger(__name__).error(f"[IMPORT] DualEmbeddingPipeline failed: {type(_imp_err).__name__}: {_imp_err}. Embedding disabled.")
    DualEmbeddingPipeline = None
from rag_chunker import ContentChunker

# Phase 14 & 15: StreamingRAG, RAPTOR, MAGMA
from streaming_rag import StreamingRAG
from raptor_tree import RAPTORTree
from magma_memory import MAGMAMemory
from memo_rag import MemoRAG
from ai_config import AI_DB_PATH
from lance_store import get_lance_store, get_lance_table

logger = logging.getLogger(__name__)

from ai_config import AI_DB_PATH as DB_PATH

# Phase 24: Neural Organism — adaptive parameters
try:
    from neural_organism import _p
except ImportError:
    def _p(param_id, fallback=0.5, regime="_global"):
        return fallback

class HybridRetriever:
    """
    Implements Hybrid Search combining:
    1. Dense Search (LanceDB with Gemini + Jina/BGE embeddings)
    2. Sparse Search (BM25 keyword search)
    3. Reranking (FlashRank Cross-Encoder)
    """
    
    def __init__(self, collection_name: str = "crypto_news", llm_router=None):
        self._llm_router = llm_router  # Pass-through for sub-components
        # Phase 28: LanceDB replaces ChromaDB
        self._lance_store = get_lance_store()
        # Primary: Gemini embeddings (general semantic, 768-dim)
        self.collection = self._lance_store.get_or_create_table(
            name=collection_name, dim=768, metric="cosine"
        )
        # Secondary: Jina/BGE embeddings (domain-specific, 768-dim)
        self.bge_collection = self._lance_store.get_or_create_table(
            name=f"{collection_name}_bge", dim=768, metric="cosine"
        )
        self.embedder = DualEmbeddingPipeline() if DualEmbeddingPipeline is not None else None
        if self.embedder is None:
            logger.error("[HybridRetriever] DualEmbeddingPipeline unavailable. Search will be degraded.")

        # Phase 23: FlashRank in-process (200MB, always warm) + Jina Reranker v3 API
        self._flashrank_last_fail = 0.0
        self._flashrank_available = False
        self._flashrank_ranker = None
        self.colbert_reranker = None  # Now actually Jina Reranker v3 (class kept for compat)

        # FlashRank: load in-process (no model server needed)
        try:
            from flashrank import Ranker, RerankRequest as FlashRankRerankRequest
            self._flashrank_ranker = Ranker()  # default model — specific name gives 404 on HuggingFace
            self._FlashRankRerankRequest = FlashRankRerankRequest
            self._flashrank_available = True
            logger.info("[HybridRetriever] FlashRank loaded in-process (~200MB)")
        except Exception as e:
            self._FlashRankRerankRequest = None
            logger.warning(f"[HybridRetriever] FlashRank import failed: {e}. Reranking = Jina only.")

        # Jina Reranker v3 (ColBERTReranker class now wraps Jina API)
        try:
            from colbert_reranker import ColBERTReranker
            self.colbert_reranker = ColBERTReranker()
            logger.info("[HybridRetriever] Jina Reranker v3 initialized (via ColBERTReranker wrapper)")
        except Exception as e:
            logger.warning(f"[HybridRetriever] Jina Reranker init failed: {e}")

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
        conn = get_db_connection()

        # Create binary_embeddings table once per process, not every connection
        if not HybridRetriever._db_tables_ensured:
            HybridRetriever._db_tables_ensured = True
        return conn

    def add_documents(self, documents: List[str], metadatas: List[Dict[str, Any]], ids: List[str]) -> int:
        """Embeds and adds documents to LanceDB tables + SQLite FTS5.
        Returns number of successfully embedded documents (0 = FTS5-only, no vectors)."""
        embedded_count = 0

        # --- Phase 1: Generate embeddings (if embedder available) ---
        # Track Gemini and BGE independently — BGE may be unavailable while Gemini works fine
        gemini_embeddings = []
        bge_embeddings = []
        gemini_valid_indices = []
        bge_valid_indices = []

        if self.embedder is not None:
            for i, doc in enumerate(documents):
                try:
                    embs = self.embedder.get_embeddings(doc)
                    g = embs.get('gemini', [])
                    b = embs.get('bge', [])
                    if g:
                        gemini_embeddings.append(g)
                        gemini_valid_indices.append(i)
                    if b:
                        bge_embeddings.append(b)
                        bge_valid_indices.append(i)
                    if not g:
                        logger.warning(f"[HybridRetriever] Empty Gemini embedding for doc {ids[i]}, skipping.")
                except Exception as e:
                    logger.warning(f"[HybridRetriever] Embedding failed for doc {ids[i]}: {e}")
        else:
            logger.warning("[HybridRetriever] Embedder unavailable — documents will be added to FTS5 only (keyword search).")

        # --- Phase 2: Store vectors in LanceDB (separate try/except per table) ---
        # Primary: Gemini collection — drives embedded_count
        if gemini_valid_indices:
            g_ids = [ids[i] for i in gemini_valid_indices]
            g_docs = [documents[i] for i in gemini_valid_indices]
            g_metas = [metadatas[i] for i in gemini_valid_indices]
            try:
                self.collection.add(
                    ids=g_ids, embeddings=gemini_embeddings,
                    documents=g_docs, metadatas=g_metas
                )
                embedded_count = len(gemini_valid_indices)
            except Exception as e:
                if 'dimension' in str(e).lower() or 'schema' in str(e).lower():
                    logger.warning(f"[HybridRetriever] Gemini table dimension mismatch — recreating: {e}")
                    try:
                        cname = self.collection.name
                        self._lance_store.delete_table(cname)
                        self.collection = self._lance_store.get_or_create_table(
                            name=cname, dim=768, metric="cosine"
                        )
                        self.collection.add(
                            ids=g_ids, embeddings=gemini_embeddings,
                            documents=g_docs, metadatas=g_metas
                        )
                        embedded_count = len(gemini_valid_indices)
                        logger.info(f"[HybridRetriever] Gemini table recreated — {embedded_count} docs added.")
                    except Exception as retry_e:
                        logger.error(f"[HybridRetriever] Gemini table retry failed: {retry_e}")
                else:
                    logger.error(f"[HybridRetriever] LanceDB Gemini add failed: {e}")

        # Secondary: BGE collection — independent, failure does NOT affect embedded_count
        if bge_valid_indices:
            b_ids = [ids[i] for i in bge_valid_indices]
            b_docs = [documents[i] for i in bge_valid_indices]
            b_metas = [metadatas[i] for i in bge_valid_indices]
            try:
                self.bge_collection.add(
                    ids=b_ids, embeddings=bge_embeddings,
                    documents=b_docs, metadatas=b_metas
                )
            except Exception as e:
                if 'dimension' in str(e).lower() or 'schema' in str(e).lower():
                    logger.warning(f"[HybridRetriever] BGE table dimension mismatch — recreating: {e}")
                    try:
                        bname = self.bge_collection.name
                        self._lance_store.delete_table(bname)
                        self.bge_collection = self._lance_store.get_or_create_table(
                            name=bname, dim=768, metric="cosine"
                        )
                        self.bge_collection.add(
                            ids=b_ids, embeddings=bge_embeddings,
                            documents=b_docs, metadatas=b_metas
                        )
                        logger.info(f"[HybridRetriever] BGE table recreated — {len(bge_valid_indices)} docs added.")
                    except Exception as retry_e:
                        logger.error(f"[HybridRetriever] BGE table retry failed: {retry_e}")
                else:
                    logger.error(f"[HybridRetriever] LanceDB BGE add failed: {e}")

        # --- Phase 3: ALWAYS add to FTS5 (no embeddings needed) + Binary BGE ---
        # O(1) lookup for binary BGE phase (uses bge_valid_indices, not gemini)
        _bge_set = set(bge_valid_indices)
        _bge_pos = {orig_i: pos for pos, orig_i in enumerate(bge_valid_indices)}
        try:
            packed_bges = None
            if bge_valid_indices and hasattr(self, 'binary_quantizer') and self.binary_quantizer:
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

                    # Binary BGE only for docs that have valid BGE embeddings
                    if packed_bges is not None and i in _bge_set:
                        cursor.execute(
                            "INSERT OR REPLACE INTO binary_embeddings (doc_id, packed_bge) VALUES (?, ?)",
                            (doc_id, packed_bges[_bge_pos[i]].tobytes())
                        )
                conn.commit()

            if embedded_count > 0:
                logger.info(f"Added {len(documents)} docs to FTS5 + {embedded_count} to LanceDB.")
            else:
                logger.info(f"Added {len(documents)} docs to FTS5 only (keyword search). ChromaDB: 0 (embedder {'unavailable' if self.embedder is None else 'failed'}).")
        except Exception as e:
            logger.error(f"Error adding to SQLite FTS5 / Binary BGE: {e}")

        return embedded_count

    def reciprocal_rank_fusion(self, results_lists: List[List[str]], k=None) -> List[str]:
        """Calculates RRF score to combine multiple ranked lists (Phase 24: adaptive k)."""
        if k is None:
            k = int(_p("retriever.rrf_k", 60))
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
        4a. Query -> Chroma (Dense Gemini) -> Top 30
        4b. Query -> Chroma (Dense BGE-Financial) -> Top 30 (complementary finance-domain search)
        5. RRF Fusion (3-way: BM25 + Gemini + BGE) -> Top 20
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
                # FTS5 reserved keywords — must be excluded from query terms
                _fts5_reserved = {"AND", "OR", "NOT", "NEAR"}
                terms = [t for t in sanitized.split() if len(t) > 1 and t.upper() not in _fts5_reserved]
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
                bge_emb = query_embs.get('bge') if query_embs else None
                has_bge = bge_emb is not None and hasattr(bge_emb, '__len__') and len(bge_emb) > 0
                if bm25_top_ids and has_bge and hasattr(self, 'binary_quantizer') and self.binary_quantizer:
                    import numpy as np
                    q_bin = self.binary_quantizer.binarize_and_pack(np.array(bge_emb))
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
                dense_results = self.collection.search(
                    query_embedding=query_embs['gemini'],
                    n_results=min(30, collection_count)
                )
                dense_gemini_ids = dense_results['ids'][0] if dense_results['ids'] else []

        # 3b. Dense Search — BGE-Financial embeddings (complementary to binary pre-filter)
        # Binary quantization refines BM25 candidates (same pool, better ranking).
        # BGE dense search finds candidates BM25 missed entirely (different pool).
        # These are complementary — true 3-way RRF needs both.
        dense_bge_ids = []
        if query_embs and query_embs.get('bge'):
            try:
                bge_count = self.bge_collection.count()
                if bge_count > 0:
                    bge_results = self.bge_collection.search(
                        query_embedding=query_embs['bge'],
                        n_results=min(30, bge_count)
                    )
                    dense_bge_ids = bge_results['ids'][0] if bge_results['ids'] else []
            except Exception as e:
                logger.warning(f"[HybridRetriever] BGE dense search failed (non-critical): {e}")

        # 3. Reciprocal Rank Fusion (3-way: BM25 + Gemini Dense + BGE Dense)
        fused_ids = self.reciprocal_rank_fusion([bm25_top_ids, dense_gemini_ids, dense_bge_ids])
        fused_top_20 = fused_ids[:20]

        # Fetch actual documents for generating reranking payloads
        passages = []
        found_ids = set()
        if fused_top_20:
            # Primary: fetch from LanceDB (has metadata + parent text)
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
                logger.warning(f"[HybridRetriever] LanceDB fetch failed: {e}")

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

        # 4. Multi-Reranker Ensemble (Phase 23: FlashRank in-process + Jina v3 API)
        flashrank_results = []
        if self._flashrank_available and self._flashrank_ranker and (time.time() - self._flashrank_last_fail >= 60):
            try:
                # FlashRank in-process: RerankRequest API (same as model_server used)
                doc_texts = [p.get("text", p.get("content", "")) for p in passages]
                flash_passages = [{"id": i, "text": t} for i, t in enumerate(doc_texts)]
                flash_req = self._FlashRankRerankRequest(query=query, passages=flash_passages)
                flash_output = self._flashrank_ranker.rerank(flash_req)
                for item in flash_output:
                    idx = item.get("id", 0) if isinstance(item, dict) else getattr(item, "id", 0)
                    score = item.get("score", 0.0) if isinstance(item, dict) else getattr(item, "score", 0.0)
                    if isinstance(idx, str):
                        idx = int(idx)
                    if 0 <= idx < len(passages):
                        scored = passages[idx].copy()
                        scored["score"] = score
                        flashrank_results.append(scored)
                if flashrank_results:
                    max_score = max(float(doc.get("score", 0.0)) for doc in flashrank_results)
                    min_score = min(float(doc.get("score", 0.0)) for doc in flashrank_results)
                    range_score = max_score - min_score if max_score > min_score else 1.0
                    for doc in flashrank_results:
                        doc["flashrank_normalized"] = (float(doc.get("score", 0.0)) - min_score) / range_score
            except Exception as e:
                self._flashrank_last_fail = time.time()
                logger.warning(f"[HybridRetriever] FlashRank in-process call failed: {e}")
                    
        colbert_results = []
        if self.colbert_reranker:
            # We want to score all candidates, so top_k is len(passages)
            colbert_results = self.colbert_reranker.rerank(query, passages, top_k=len(passages))
            
        final_results = self._ensemble_rerank(passages, flashrank_results, colbert_results)

        # Phase 27 Item 14: graph-augmented retrieval. Pull top GAM-RAG winning
        # patterns for the CURRENT pheromone regime (not a query-text guess —
        # regime_classifier deposits SIGNAL_REGIME on every cycle) and merge
        # them at the FRONT of the result list (de-duplicated by id).
        try:
            regime_hint = "trending_bull"  # safe fallback when pheromone empty
            try:
                from pheromone_field import get_pheromone_field
                regime_ph = get_pheromone_field().read("market_regime")
                if isinstance(regime_ph, str) and regime_ph.strip():
                    regime_hint = regime_ph.strip()
                elif isinstance(regime_ph, dict):
                    cand = regime_ph.get("regime") or regime_ph.get("value")
                    if isinstance(cand, str) and cand.strip():
                        regime_hint = cand.strip()
            except Exception:
                pass
            from gam_rag import GamRAG
            gam = GamRAG()
            gam_docs = gam.retrieve_past_wisdom(
                current_regime=regime_hint,
                pair=original_query[:24],
                k=2,
            )
            if gam_docs:
                gam_id_set = {f"gam:{i}" for i in range(len(gam_docs))}
                gam_passages = [
                    {"id": f"gam:{i}", "text": str(doc),
                     "meta": {"source": "gam_rag", "regime": regime_hint}}
                    for i, doc in enumerate(gam_docs)
                ]
                # De-duplicate (no overlap normally, but defensive).
                existing_ids = {p["id"] for p in final_results}
                fresh = [p for p in gam_passages if p["id"] not in existing_ids]
                if fresh:
                    final_results = fresh + final_results
                    logger.info(f"[GAM-RAG] Injected {len(fresh)} winning-pattern passages "
                                f"into hybrid search (regime={regime_hint})")
        except Exception as e:
            logger.debug(f"[GAM-RAG] hybrid integration skipped: {e}")

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
        half_life_days: float = None,
        alpha: float = None
    ) -> List[Dict[str, Any]]:
        """
        Apply temporal decay to search results.
        Formula: score = alpha * relevance + (1-alpha) * 0.5^(age/half_life)
        
        - 1-hour-old news: decay ≈ 1.0 → score barely reduced
        - 7-day-old news: decay = 0.5 → 30% penalty
        - 30-day-old news: decay ≈ 0.05 → ~28.5% penalty
        - 90-day-old news: decay ≈ 0.0002 → killed
        """
        # Phase 24: adaptive temporal decay parameters
        if half_life_days is None:
            half_life_days = _p("retriever.temporal_half_life", 7.0)
        if alpha is None:
            alpha = _p("retriever.temporal_alpha", 0.7)
        unknown_decay = _p("retriever.unknown_date_decay", 0.5)

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
                    decay = unknown_decay
            else:
                decay = unknown_decay

            original_score = float(result.get('score', 1.0))
            result['score'] = alpha * original_score + (1 - alpha) * decay
        
        # Re-sort by decayed score (highest first)
        results.sort(key=lambda x: float(x.get('score', 0)), reverse=True)
        
        return results

    # ═══ RAG GUARANTEE: Event search + Regime filter + Chunk boost ═══

    def search_similar_events(self, event_type: str, top_k: int = 10) -> list:
        """Event-Driven Temporal RAG: find historical chunks with same event type."""
        if not event_type or event_type == "general":
            return []
        try:
            query_text = f"Historical impact of {event_type.replace('_', ' ')} on cryptocurrency prices"
            # Phase 28: LanceDB needs embedding, not text. Use embedder if available.
            query_embs = self.embedder.get_embeddings(query_text) if self.embedder else None
            if query_embs and query_embs.get('gemini'):
                results = self.collection.search(
                    query_embedding=query_embs['gemini'],
                    n_results=top_k,
                    where={"event_type": event_type},
                )
            else:
                results = {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}
            if not results or not results.get("documents") or not results["documents"][0]:
                return []
            return [
                {"text": doc, "metadata": meta, "score": 1.0 / (1.0 + dist)}
                for doc, meta, dist in zip(
                    results["documents"][0],
                    results["metadatas"][0],
                    results["distances"][0]
                )
            ]
        except Exception as e:
            logger.debug(f"[EventRAG] Event search failed: {e}")
            return []

    def _build_regime_filter(self, current_regime: str = None) -> dict:
        """Regime-Aware Filter: only retrieve chunks from compatible regimes."""
        if not current_regime:
            return {}
        try:
            return {
                "$or": [
                    {"market_regime": current_regime},
                    {"market_regime": "transitional"},
                ]
            }
        except Exception:
            return {}

    def _apply_chunk_boost(self, results: list) -> list:
        """Outcome-Based Chunk Scoring: boost/penalize chunks based on trade PnL history."""
        try:
            from rag_feedback import RAGFeedbackLoop
            feedback = RAGFeedbackLoop()
            for doc in results:
                doc_id = doc.get("id", "")
                if doc_id:
                    boost = feedback.get_chunk_boost(doc_id)
                    doc["score"] = float(doc.get("score", 1.0)) * boost
            results.sort(key=lambda x: float(x.get("score", 0)), reverse=True)
        except Exception:
            pass  # Feedback unavailable = no boost (neutral)
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
