import sqlite3
import json
import logging
from typing import Dict, Any, List, Optional
from langchain_core.messages import SystemMessage, HumanMessage

from ai_config import AI_DB_PATH
from llm_router import LLMRouter

logger = logging.getLogger(__name__)

class BidirectionalRAG:
    """
    Phase 15 - Bidirectional RAG.
    Creates a feedback loop from trading outcomes (PnL) back into the knowledge base.
    Evaluates decisions when trades close, generates "lessons learned", and
    injects these directly into the Chroma or Hybrid Search corpus.
    """
    def __init__(self, db_path: str = AI_DB_PATH, llm_router: Optional[LLMRouter] = None):
        self.db_path = db_path
        self.router = llm_router or LLMRouter()
        self._init_db()

    def _init_db(self):
        """Initializes the bidirectional learning log table."""
        with sqlite3.connect(self.db_path) as conn:
            c = conn.cursor()
            c.execute('''
                CREATE TABLE IF NOT EXISTS ai_lessons (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    decision_id INTEGER,
                    pair TEXT,
                    signal TEXT,
                    outcome_pnl REAL,
                    lesson_text TEXT,
                    is_embedded BOOLEAN DEFAULT 0,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            conn.commit()

    def evaluate_trade_outcome(self, decision_id: int, pair: str, signal: str, outcome_pnl: float, reasoning: str) -> Optional[str]:
        """
        Uses LLM to evaluate why a trade succeeded or failed based on its reasoning.
        Returns a generated "lesson" string to be stored and later embedded.
        """
        prompt = [
            SystemMessage(content=(
                "You are a quantitative trade evaluator performing post-mortem analysis.\n\n"
                "RULES:\n"
                "1. Be SPECIFIC — cite indicator values, confidence levels, and timing from the reasoning.\n"
                "2. Distinguish between PROCESS errors (bad reasoning) and OUTCOME noise (good reasoning, bad luck).\n"
                "3. For LOSSES: identify the PRIMARY structural failure (hallucination, regime mismatch, timing, sizing).\n"
                "4. For WINS: identify which specific pattern/signal was validated and should be weighted more in future.\n"
                "5. Output ONLY the lesson in 2-3 sentences. Use format: 'LESSON [pair]: [finding]'\n"
                "6. Make the lesson ACTIONABLE — it will be embedded into the knowledge base for future decisions."
            )),
            HumanMessage(content=(
                f"Pair: {pair}\n"
                f"Signal Given: {signal}\n"
                f"Original Reasoning: {reasoning}\n"
                f"Actual Outcome PnL: {outcome_pnl}%\n\n"
                f"Was this a PROCESS error or OUTCOME noise? Extract the actionable lesson:"
            ))
        ]
        
        try:
            response = self.router.invoke(prompt, temperature=0.1, priority="low")
            lesson = str(response.content).strip()
            
            # Save the lesson
            with sqlite3.connect(self.db_path) as conn:
                c = conn.cursor()
                c.execute('''
                    INSERT INTO ai_lessons (decision_id, pair, signal, outcome_pnl, lesson_text)
                    VALUES (?, ?, ?, ?, ?)
                ''', (decision_id, pair, signal, outcome_pnl, lesson))
                conn.commit()
                
            logger.info(f"Generated Bidirectional RAG lesson for {pair} (PnL: {outcome_pnl}%).")
            return lesson
            
        except Exception as e:
            logger.error(f"Failed to generate trade evaluation lesson: {e}")
            return None

    def get_unembedded_lessons(self) -> List[Dict[str, Any]]:
        """Retrieves lessons that need to be written back to the vector DB."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("SELECT * FROM ai_lessons WHERE is_embedded = 0").fetchall()
            return [dict(r) for r in rows]

    def mark_lessons_embedded(self, lesson_ids: List[int]):
        """Flags lessons as having been written back to Chroma via HybridRetriever."""
        if not lesson_ids:
            return
        with sqlite3.connect(self.db_path) as conn:
            placeholders = ",".join(["?"] * len(lesson_ids))
            conn.execute(f"UPDATE ai_lessons SET is_embedded = 1 WHERE id IN ({placeholders})", lesson_ids)
            conn.commit()
