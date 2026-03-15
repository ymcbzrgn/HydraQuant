import os
import sqlite3
import uuid
import logging
from typing import Dict, Any, List
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from ai_config import AI_DB_PATH as DB_PATH, CHROMA_PERSIST_DIR as CHROMA_PATH, get_chroma_client

class GamRAG:
    """
    Phase 6.1: Gain-Adaptive Memory RAG (GAM-RAG)
    A persistent memory store that explicitly indexes AI reasoning
    from structurally highly-successful past trades.
    When future similar regimes arise, the Agent uses this memory.
    """

    def __init__(self, db_path: str = DB_PATH, chroma_path: str = CHROMA_PATH):
        self.db_path = db_path
        self._ensure_paths(chroma_path)

        # Use singleton ChromaDB client
        self.chroma_client = get_chroma_client()
        # GAM-RAG uses ChromaDB's built-in embedding (query_texts=, add(documents=...))
        # so we do NOT set embedding_function=None here (unlike hybrid_retriever which
        # provides pre-computed 768-dim embeddings). This collection uses the default
        # all-MiniLM-L6-v2 (384-dim) which is separate from the 768-dim crypto_news collection.
        self.gam_collection = self.chroma_client.get_or_create_collection(
            name="successful_trade_patterns",
            metadata={"hnsw:space": "cosine"}
        )

    def _ensure_paths(self, chroma_path: str):
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        os.makedirs(chroma_path, exist_ok=True)

    def _get_db_connection(self):
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def get_winning_patterns(self, min_pnl: float = 2.0) -> List[Dict[str, Any]]:
        """Fetch historical trades that yielded a massive profit and haven't been 'memorized' yet."""
        try:
            with self._get_db_connection() as conn:
                c = conn.cursor()
                
                # Check if gam_memorized column exists, auto-migrate if missing
                c.execute("PRAGMA table_info(ai_decisions)")
                columns = [col['name'] for col in c.fetchall()]
                if 'gam_memorized' not in columns:
                    c.execute('ALTER TABLE ai_decisions ADD COLUMN gam_memorized INTEGER DEFAULT 0')
                    conn.commit()

                c.execute('''
                    SELECT id, pair, timestamp, signal_type, confidence, 
                           reasoning_summary, regime, outcome_pnl, outcome_duration
                    FROM ai_decisions
                    WHERE outcome_pnl >= ? 
                      AND gam_memorized = 0
                    ORDER BY timestamp ASC LIMIT 50
                ''', (min_pnl,))
                
                rows = c.fetchall()
                return [dict(row) for row in rows]
        except Exception as e:
            logger.error(f"Failed to fetch winning patterns: {e}")
            return []

    def memorize_trade(self, trade: Dict[str, Any]) -> bool:
        """
        [Bidirectional RAG Write-Back]
        Strips the winning trade logic and stores it into ChromaDB as an embeddable string.
        """
        # Create the embedded "Lesson Document"
        memory_document = (
            f"HISTORICAL WIN: In a {trade['regime'].upper()} regime, "
            f"predicting a {trade['signal_type']} on {trade['pair']} yielded a +{trade['outcome_pnl']:.2f}% profit. "
            f"The logic that caused this win: {trade['reasoning_summary']}"
        )
        
        metadata = {
            "source": "ai_decisions",
            "trade_id": trade['id'],
            "pair": trade['pair'],
            "pnl": trade['outcome_pnl'],
            "regime": trade['regime'],
            "timestamp": trade['timestamp']
        }
        
        doc_id = f"gam_mem_{trade['id']}_{uuid.uuid4().hex[:6]}"
        
        try:
            logger.info(f"Writing Bidirectional Memory for Trade ID {trade['id']} (PnL: +{trade['outcome_pnl']:.2f}%)")
            self.gam_collection.add(
                documents=[memory_document],
                metadatas=[metadata],
                ids=[doc_id]
            )
            
            # Mark the trade as memorized in SQLite
            with self._get_db_connection() as conn:
                c = conn.cursor()
                c.execute('''
                    UPDATE ai_decisions
                    SET gam_memorized = 1
                    WHERE id = ?
                ''', (trade['id'],))
                conn.commit()
                
            return True
            
        except Exception as e:
            logger.error(f"Failed to write Bidirectional Memory for ID {trade['id']}: {e}")
            return False

    def retrieve_past_wisdom(self, current_regime: str, pair: str, k: int = 2) -> List[str]:
        """
        Called by the Multi-Agent LLMs before making a decision to ask 
        'What logical pattern historically worked best in this exact regime?'
        """
        query_text = f"What logic has historically won in {current_regime} for {pair}?"
        
        try:
            results = self.gam_collection.query(
                query_texts=[query_text],
                n_results=k,
                # Filter by similar contextual regimes if possible
                where={"regime": current_regime}
            )
            
            # `results['documents'][0]` because query_texts is a list of length 1
            docs = results.get('documents', [[]])[0]
            return docs
            
        except Exception as e:
            logger.warning(f"Failed to retrieve past GAM wisdom: {e}")
            return []

    def process_new_memories(self):
        """Scans the DB for all new highly profitable trades and embeds them."""
        logger.info("Scanning for new Gain-Adaptive Memories (GAM)...")
        winners = self.get_winning_patterns()
        
        if not winners:
            logger.info("No un-memorized highly profitable trades found.")
            return
            
        success_count = 0
        for trade in winners:
            if self.memorize_trade(trade):
                success_count += 1
                
        logger.info(f"GAM-RAG Sync Complete. Memorized {success_count}/{len(winners)} new winning patterns into ChromaDB.")

if __name__ == "__main__":
    gam = GamRAG()
    gam.process_new_memories()
    
    # Test retrieval
    wisdom = gam.retrieve_past_wisdom(current_regime="bull", pair="BTC/USDT")
    if wisdom:
        print("\n=== RETRIEVED PAST WISDOM ===")
        for w in wisdom:
            print(f"- {w}")
