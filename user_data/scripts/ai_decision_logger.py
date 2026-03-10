import sqlite3
import os
import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

from ai_config import AI_DB_PATH as DB_PATH

class AIDecisionLogger:
    """
    Autonomy Scaffolding Module 3.5.1
    Logs all decisions, confidence levels, and reasoning summaries
    from the Gemini/Language Models into the local SQLite database.
    This enables post-trade analysis, Kelly position sizing, and trust curve fitting.
    """
    
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        # Ensure DB is initialized (useful if running in isolated test)
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._ensure_schema_up_to_date()
        
    def _ensure_schema_up_to_date(self):
        """Phase 6.1: Gracefully migrate old databases to contain outcome columns."""
        with self._get_db_connection() as conn:
            c = conn.cursor()
            
            # Create base table if it doesn't exist
            c.execute('''
                CREATE TABLE IF NOT EXISTS ai_decisions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    pair TEXT,
                    signal_type TEXT,
                    confidence REAL,
                    position_size REAL,
                    entry_price REAL,
                    model_used TEXT,
                    rag_context_ids TEXT,
                    reasoning_summary TEXT,
                    regime TEXT,
                    trust_score_at_decision REAL
                )
            ''')
            
            # Migrate missing Phase 6 columns
            c.execute("PRAGMA table_info(ai_decisions)")
            columns = [col['name'] for col in c.fetchall()]
            
            if 'outcome_pnl' not in columns:
                c.execute('ALTER TABLE ai_decisions ADD COLUMN outcome_pnl REAL')
            if 'outcome_duration' not in columns:
                c.execute('ALTER TABLE ai_decisions ADD COLUMN outcome_duration REAL')
            if '_status_cache' not in columns:
                c.execute('ALTER TABLE ai_decisions ADD COLUMN _status_cache TEXT')
                
            conn.commit()
        
    def _get_db_connection(self):
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn
        
    def log_decision(
        self,
        pair: str,
        signal_type: str,
        confidence: float,
        reasoning_summary: str,
        model_used: str = "gemini-2.5-flash",
        rag_context_ids: Optional[List[str]] = None,
        position_size: Optional[float] = None,
        entry_price: Optional[float] = None,
        regime: str = "neutral",
        trust_score_at_decision: float = 0.5
    ) -> Optional[int]:
        """
        Records an AI decision into the database.
        
        Args:
            pair: Trading pair (e.g., 'BTC/USDT')
            signal_type: Decision (e.g., 'BULL', 'BEAR', 'NEUTRAL')
            confidence: LLM's confidence score (0.01 to 1.00)
            reasoning_summary: A brief summary of why to take this trade
            model_used: ID of the LLM provider used
            rag_context_ids: Document/news IDs used to make exactly this decision 
            position_size: Fractional stake amount calculated (handled by sizing engine later)
            entry_price: The spot price at the time of decision
            regime: Current market environment (e.g., 'bull', 'bear', 'ranging')
            trust_score_at_decision: Global AI trust level at the time of prediction
            
        Returns:
            int: The unique database `id` (trade_id) of the logged decision, or None if failed.
        """
        
        # Serialize Context IDs to JSON string for SQLite storage
        context_str = json.dumps(rag_context_ids) if rag_context_ids else "[]"
        
        try:
            with self._get_db_connection() as conn:
                c = conn.cursor()
                
                c.execute('''
                    INSERT INTO ai_decisions (
                        pair, signal_type, confidence, position_size, entry_price, 
                        model_used, rag_context_ids, reasoning_summary, regime, 
                        trust_score_at_decision
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    pair, signal_type, confidence, position_size, entry_price,
                    model_used, context_str, reasoning_summary, regime,
                    trust_score_at_decision
                ))
                
                # We used to reject confidence < 0.35, but now we let the position_sizer math 
                # assign "dust" fractions and trade them to accurately measure all LLM theories.
                    
                conn.commit()
                last_row_id: int = c.lastrowid
            
            logger.info(f"Successfully logged {signal_type} decision for {pair} (Conf: {confidence:.2f}) -> ID: {last_row_id}")
            return last_row_id
            
        except Exception as e:
            logger.error(f"Failed to log AI decision for {pair}: {e}")
            return None
            
    def update_trade_outcome(
        self,
        decision_id: int,
        pnl_percent: float,
        duration_minutes: float,
        status: str = "closed"
    ) -> bool:
        """
        [Phase 6.1] Binds the final trade outcome to the original AI decision.
        Freqtrade hooks call this when a trade exits.
        
        Args:
            decision_id: The returning `id` from log_decision()
            pnl_percent: The final Net Profit/Loss percentage (e.g. 2.45 or -1.2)
            duration_minutes: How long the trade was held
            status: e.g. "closed_win", "closed_loss", "stop_loss"
        """
        try:
            with self._get_db_connection() as conn:
                c = conn.cursor()
                # outcome_pnl & outcome_duration fields were provisioned in Phase 3.5 schema
                c.execute('''
                    UPDATE ai_decisions
                    SET outcome_pnl = ?, outcome_duration = ?, _status_cache = ?
                    WHERE id = ?
                ''', (pnl_percent, duration_minutes, status, decision_id))
                
                # If no rows were updated, ID is invalid
                if c.rowcount == 0:
                     logger.warning(f"Could not bind outcome to decision_id {decision_id}. ID not found.")
                     return False
                     
                conn.commit()
            
            logger.info(f"Successfully bound Trade Outcome [{pnl_percent:.2f}%] to Decision ID {decision_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update trade outcome for ID {decision_id}: {e}")
            return False
            
    def get_recent_decisions(self, pair: Optional[str] = None, limit: int = 10, include_outcomes: bool = False) -> List[Dict[str, Any]]:
        """Retrieves recent decisions for analysis."""
        try:
            with self._get_db_connection() as conn:
                c = conn.cursor()
                
                query = "SELECT * FROM ai_decisions"
                params = []
                filters = []
                
                if pair:
                    filters.append("pair = ?")
                    params.append(pair)
                    
                if include_outcomes:
                    filters.append("outcome_pnl IS NOT NULL")
                    
                if filters:
                    query += " WHERE " + " AND ".join(filters)
                    
                query += " ORDER BY timestamp DESC LIMIT ?"
                
                # Execute with all parameters flattened
                c.execute(query, tuple(params) + (limit,))
                rows = c.fetchall()
                
                # Convert rows to dict
                result = []
                for row in rows:
                    r_dict = dict(row)
                    r_dict['rag_context_ids'] = json.loads(r_dict.get('rag_context_ids', '[]'))
                    result.append(r_dict)
                    
                return result
                
        except Exception as e:
            logger.error(f"Failed to retrieve decisions: {e}")
            return []

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    logger_test = AIDecisionLogger()
    
    # Test Logging
    logger.info("Testing Autonomy Decision Logger PnL Binding...")
    trade_id_1 = logger_test.log_decision(
        pair="BTC/USDT",
        signal_type="BULL",
        confidence=0.85,
        reasoning_summary="Strong inflow indicators + SEC case settlement.",
        model_used="models/gemini-2.5-flash",
        rag_context_ids=["doc_123", "doc_456"],
        regime="bull",
        trust_score_at_decision=0.75
    )
    
    # Test Forgone profit log boundary
    trade_id_2 = logger_test.log_decision(
        pair="ETH/USDT",
        signal_type="BEAR",
        confidence=0.25, # Too low, will trigger forgone profit logic
        reasoning_summary="Mild bearish divergences but not fully confirmed.",
        model_used="models/gemini-2.5-flash"
    )
    
    # Wait for the fake trade to "process", then bind an outcome!
    if trade_id_1:
         logger_test.update_trade_outcome(trade_id_1, pnl_percent=4.25, duration_minutes=120)
    
    if trade_id_2:
         logger_test.update_trade_outcome(trade_id_2, pnl_percent=-1.15, duration_minutes=45)
    
    print("\n\n=== RECENT CLOSED TRADES IN DB ===")
    history = logger_test.get_recent_decisions(limit=2, include_outcomes=True)
    for h in history:
        outcome_str = f"PnL: {h.get('outcome_pnl', 'N/A')}%"
        print(f"[{h['timestamp']}] {h['pair']} -> {h['signal_type']} (Conf: {h['confidence']:.2f}) | {outcome_str}")
