import os
import sys
import time
import logging
from typing import List, Dict
import sqlite3
import subprocess
import threading

# Ensure local imports work regardless of execution directory
sys.path.append(os.path.dirname(__file__))

# Data Fetchers
from rss_fetcher import fetch_rss_feeds
from fng_fetcher import fetch_fng

from db import get_db_connection
from sentiment_analyzer import analyze_unscored_news
from rag_chunker import ContentChunker
from hybrid_retriever import HybridRetriever
from entity_extractor import KnowledgeGraphManager

# Phase 14 Integrations
from streaming_rag import StreamingRAG
from raptor_tree import RAPTORTree
from cryptopanic_fetcher import CryptoPanicFetcher
from alphavantage_fetcher import AlphaVantageFetcher
from magma_memory import MAGMAMemory
from memo_rag import MemoRAG

logger = logging.getLogger(__name__)

class DataPipeline:
    """
    Phase 4.2: Data Ingestion Pipeline & NLP Pre-processing
    Orchestrates the continuous flow of information from external APIs to the RAG memory.
    Replaces the legacy subprocess orchestrator with direct function calls where possible.
    """
    def __init__(self):
        self.retriever = HybridRetriever(collection_name="crypto_news")
        self.streaming_rag = StreamingRAG()
        self.raptor = RAPTORTree()
        self.crypto_panic = CryptoPanicFetcher()
        self.alpha_vantage = AlphaVantageFetcher()
        self.magma = MAGMAMemory()
        self.memorag = MemoRAG()
        
    def run_pipeline(self):
        """Executes a single pass of the entire ingestion pipeline."""
        logger.info("---| RUNNING DATA PIPELINE |---")
        
        # 1. Fetch Raw Data
        logger.info("[Step 1] Fetching external data...")
        try:
            fetch_fng()
            new_articles_count = fetch_rss_feeds()
            
            # Phase 14: Community & Pre-computed Sentiment Fetchers
            crypto_panic_news = self.crypto_panic.fetch(limit=10)
            alpha_news = self.alpha_vantage.fetch_news_sentiment()
            
            # We would typically inject these into the db or memory, 
            # for Phase 14 we log integration metrics
            logger.info(f"Fetched {len(crypto_panic_news)} CryptoPanic posts and {len(alpha_news)} from AlphaVantage.")
            
        except Exception as e:
            logger.error(f"Failed during data fetch: {e}")
            new_articles_count = 0
            
        logger.info(f"Fetch completed. New RSS Articles: {new_articles_count}")
        
        # 2. Analyze Sentiment (CryptoBERT/FinBERT)
        logger.info("[Step 2] Processing sentiment for unscored news...")
        try:
            analyze_unscored_news()
        except Exception as e:
            logger.error(f"Failed during sentiment analysis: {e}")
            
        # 3. Vectorization & RAG Insertion
        logger.info("[Step 3] Vectorizing and embedding unprocessed news into ChromaDB...")
        self._embed_unprocessed_news()
        
        logger.info("---| PIPELINE PASS COMPLETE |---")

    def _embed_unprocessed_news(self):
        """
        Finds database articles that haven't been pushed to ChromaDB,
        chunks them, generates embeddings, and saves them.
        """
        conn = get_db_connection()
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        
        # Check if column exists, create if not (Evolutionary schema update)
        try:
            c.execute("SELECT is_embedded FROM market_news LIMIT 1")
        except sqlite3.OperationalError:
            logger.warning("Adding 'is_embedded' column to market_news table.")
            c.execute("ALTER TABLE market_news ADD COLUMN is_embedded BOOLEAN DEFAULT 0")
            conn.commit()

        # Fetch articles that have sentiment scored but not yet embedded
        c.execute("""
            SELECT id, title, summary, source, sentiment_score, published_at, url 
            FROM market_news 
            WHERE is_embedded = 0 AND sentiment_score IS NOT NULL
            LIMIT 50
        """)
        articles = c.fetchall()
        
        if not articles:
            logger.info("No new articles to embed.")
            conn.close()
            return
            
        logger.info(f"Embedding {len(articles)} articles...")
        
        docs_to_insert = []
        metadatas_to_insert = []
        ids_to_insert = []
        successful_db_ids = []
        
        # Phase 5.1: Initialize Knowledge Graph Extractor
        kg = KnowledgeGraphManager()
        
        # Pre-compute regime and cache it (same for all chunks in this batch)
        current_regime = self._get_current_btc_regime()

        for article in articles:
            # Convert sqlite3.Row to dict (Row has [] access but no .get())
            article = dict(article)
            text_content = f"Title: {article['title']}\n\nSummary: {article['summary']}"
            # Detect event type from title + summary
            event_type = self._detect_event_type(
                article['title'] + ' ' + (article.get('summary') or ''))
            # Rich contextual prompt — pass full article dict (Anthropic-style enhanced)
            doc_summary = article  # Pass full article dict for rich context
            
            # Use Parent-Child chunking if text is very long, otherwise Recursive.
            if len(text_content) > 1000:
                chunks = ContentChunker.chunk_parent_child(text_content, parent_size=600, child_size=200)
                for i, chunk_dict in enumerate(chunks):
                    doc_id = f"news_{article['id']}_chunk_{i}"
                    # Phase 3.1: Contextual Chunking — prepend doc context for better retrieval
                    contextual_text = ContentChunker.construct_contextual_prompt(
                        chunk_dict['child_text'], doc_summary
                    )
                    docs_to_insert.append(contextual_text)
                    metadatas_to_insert.append({
                        "source": article['source'],
                        "published_at": str(article['published_at']),
                        "sentiment_score": float(article['sentiment_score']) if article['sentiment_score'] else 0.0,
                        "type": "news_child",
                        "url": article['url'],
                        "parent_text": chunk_dict['parent_text'],
                        "market_regime": current_regime,
                        "event_type": event_type,
                    })
                    ids_to_insert.append(doc_id)
            else:
                chunks = ContentChunker.chunk_recursive(text_content, chunk_size=512, chunk_overlap=50)
                for i, chunk_text in enumerate(chunks):
                    doc_id = f"news_{article['id']}_chunk_{i}"
                    # Phase 3.1: Contextual Chunking — prepend doc context
                    contextual_text = ContentChunker.construct_contextual_prompt(
                        chunk_text, doc_summary
                    )
                    docs_to_insert.append(contextual_text)
                    metadatas_to_insert.append({
                        "source": article['source'],
                        "published_at": str(article['published_at']),
                        "sentiment_score": float(article['sentiment_score']) if article['sentiment_score'] else 0.0,
                        "type": "news",
                        "url": article['url'],
                        "parent_text": text_content,
                        "market_regime": current_regime,
                        "event_type": event_type,
                    })
                    ids_to_insert.append(doc_id)
            
            successful_db_ids.append(article['id'])
            
        # Push to Vector DB
        try:
            embedded_count = 0
            if docs_to_insert:
                embedded_count = self.retriever.add_documents(
                    documents=docs_to_insert,
                    metadatas=metadatas_to_insert,
                    ids=ids_to_insert
                ) or 0

            # Flag as embedded ONLY if vectors were actually stored in ChromaDB.
            # If only FTS5 was written (embedded_count=0), do NOT flag — next cycle
            # will retry embedding when the embedder is available. FTS5 inserts are
            # idempotent (DELETE + INSERT) so retries are safe.
            if successful_db_ids and embedded_count > 0:
                format_strings = ','.join(['?'] * len(successful_db_ids))
                c.execute(f"UPDATE market_news SET is_embedded = 1 WHERE id IN ({format_strings})", tuple(successful_db_ids))
                conn.commit()
                logger.info(f"Successfully vectorized and flagged {len(successful_db_ids)} root articles ({embedded_count} chunks to ChromaDB).")
            elif successful_db_ids:
                logger.warning(f"Added {len(docs_to_insert)} chunks to FTS5 only. NOT flagging is_embedded — will retry when embedder is available.")
                
            # Phase 14: StreamingRAG Ingestion
            for metadata, content, d_id in zip(metadatas_to_insert, docs_to_insert, ids_to_insert):
                self.streaming_rag.ingest(doc_id=d_id, content=content, metadata=metadata)
                
            # Phase 14: Building RAPTOR Summary Tree locally in background
            if docs_to_insert:
                raptor_chunks = [{"id": r_id, "text": txt, "metadata": meta} 
                                 for r_id, txt, meta in zip(ids_to_insert, docs_to_insert, metadatas_to_insert)]
                # Bounded by 10 to not exhaust API constraints immediately upon bulk loads
                self.raptor.build_tree(raptor_chunks[:10])
                
            # Phase 15: Update MemoRAG Global Summary
            if docs_to_insert:
                # Passing raw texts allows LLM to continuously shrink new info into the global context
                self.memorag.update_global_memory(docs_to_insert)

            # Phase 5.1 & Phase 15: Extract Entities and build MAGMA Knowledge Graph
            # Entity extraction is non-critical enrichment. Skip if LLM providers are exhausted
            # to preserve quota for signal generation (the critical path).
            _skip_entities = False
            try:
                if hasattr(kg, 'router') and hasattr(kg.router, 'is_any_provider_available'):
                    if not kg.router.is_any_provider_available():
                        logger.warning("[DataPipeline] All LLM providers exhausted — skipping entity extraction to preserve quota.")
                        _skip_entities = True
            except Exception:
                pass

            entity_failures = 0
            for article in articles:
                if _skip_entities or entity_failures >= 3:
                    break  # Stop trying after 3 consecutive failures or if providers exhausted
                if article['id'] in successful_db_ids:
                    try:
                        extracted = kg.extract_from_text(article['summary'], source_reference=article['url'])
                        if extracted and isinstance(extracted, dict) and 'entities' in extracted:
                            entities = extracted['entities']
                            entity_failures = 0  # Reset on success
                            logger.info(f"Extracted {len(entities)} entities for art_{article['id']}")

                            # Phase 15: MAGMA Entity Graph
                            if isinstance(entities, list):
                                for i in range(len(entities)):
                                    for j in range(i + 1, len(entities)):
                                        if isinstance(entities[i], dict) and isinstance(entities[j], dict):
                                            source_e = entities[i].get('name', 'UNKNOWN')
                                            target_e = entities[j].get('name', 'UNKNOWN')
                                            if source_e != 'UNKNOWN' and target_e != 'UNKNOWN':
                                                self.magma.add_edge("entity", source_e, "correlates", target_e, metadata={"source": article['url']})
                                                self.magma.add_edge("entity", target_e, "correlates", source_e, metadata={"source": article['url']})
                        else:
                            entity_failures += 1
                    except Exception as e:
                        entity_failures += 1
                        logger.warning(f"Entity extraction failed for art_{article['id']}: {e}")
                        if entity_failures >= 3:
                            logger.warning("[DataPipeline] 3 consecutive entity extraction failures — skipping remaining articles.")
                
        except Exception as e:
            logger.error(f"Failed to push embeddings to DB: {e}")
            
        finally:
            conn.close()

    # ═══ RAG GUARANTEE: Helper methods for regime + event metadata ═══

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

    def _detect_event_type(self, text: str) -> str:
        """Detect event type from news text using keyword matching."""
        text_lower = text.lower()
        for event_type, keywords in self.EVENT_TAXONOMY.items():
            matches = sum(1 for kw in keywords if kw in text_lower)
            if matches >= 2:
                return event_type
            if matches == 1 and len(text_lower) < 300:
                return event_type
        return "general"

    def _get_current_btc_regime(self) -> str:
        """Get current BTC regime from evidence_audit_log."""
        try:
            from ai_config import AI_DB_PATH
            conn = sqlite3.connect(AI_DB_PATH, timeout=10)
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT regime FROM evidence_audit_log WHERE pair LIKE 'BTC%' "
                "ORDER BY timestamp DESC LIMIT 1"
            ).fetchone()
            conn.close()
            return row["regime"] if row else "transitional"
        except Exception:
            return "transitional"


def start_sse_stream():
    """Run the streaming script indefinitely with auto-restart on failure."""
    while True:
        logger.info("Starting SSE Stream processor...")
        script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "crypto_cv_stream.py")
        try:
            subprocess.run([sys.executable, script_path], check=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"Stream interrupted. Restarting in 10s... {e}")
            time.sleep(10)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    import argparse
    
    parser = argparse.ArgumentParser(description="Freqtrade AI Data Ingestion Pipeline")
    parser.add_argument("--once", action="store_true", help="Run the pipeline exactly once and exit")
    parser.add_argument("--interval", type=int, default=300, help="Interval in seconds between runs (default: 5 mins)")
    parser.add_argument("--scheduled", action="store_true", help="Use APScheduler instead of while/sleep loop")
    args = parser.parse_args()
    
    # Start the SSE thread (Real-Time Background)
    if not args.once:
        sse_thread = threading.Thread(target=start_sse_stream, daemon=True)
        sse_thread.start()
    
    pipeline = DataPipeline()
    
    if args.once:
        pipeline.run_pipeline()
    elif args.scheduled:
        # Phase 4.5: APScheduler mode — proper job scheduling
        from scheduler import PipelineScheduler
        sched = PipelineScheduler()
        if sched.start():
            logger.info("APScheduler mode active. Jobs:")
            for job in sched.get_job_info():
                logger.info(f"  {job['name']}: {job['trigger']} → next: {job['next_run']}")
            try:
                while True:
                    time.sleep(60)
            except KeyboardInterrupt:
                sched.stop()
        else:
            logger.error("APScheduler failed. Install: pip install apscheduler")
    else:
        # Legacy mode: while/sleep loop
        logger.info(f"Starting continuous data pipeline (Interval: {args.interval}s)")
        while True:
            pipeline.run_pipeline()
            logger.info(f"Sleeping for {args.interval} seconds...")
            time.sleep(args.interval)

