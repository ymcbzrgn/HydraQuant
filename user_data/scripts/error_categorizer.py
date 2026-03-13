import os
import sqlite3
import json
import logging
from typing import Dict, Any, List
from llm_router import LLMRouter
from langchain_core.messages import SystemMessage, HumanMessage
from ai_config import AI_DB_PATH as DB_PATH

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# System Prompt to instruct the Coordinator on how to grade its own failures.
ERROR_ANALYSIS_SYSTEM_PROMPT = """
ROLE: You are the Chief Risk Officer for a quantitative hedge fund.
OBJECTIVE: We are conducting an autonomous post-mortem on a LOSING trade.
Your job is to read the AI's *Original Reasoning* and the *Actual PnL Outcome*, and ruthlessly categorize the core structural failure into EXACTLY ONE of the following tags:

1. [HALLUCINATION]: The AI relied on fake news, fabricated data, or misread the indicators.
2. [REGIME_MISMATCH]: The logic was sound for a Bull market, but the actual regime was Bear/Sideways.
3. [TIMING_ERROR]: The market eventually did what the AI predicted, but the entry was premature or too late.
4. [POSITION_SIZING]: The AI's conviction was mismatched with the risk (e.g. 95% confidence on a highly volatile pair).
5. [CORRELATION_ERROR]: The AI opened this trade while already overexposed to similar assets (e.g. Longing BTC, ETH, and SOL simultaneously).
6. [MARKET_NOISE]: The AI made the mathematically correct, highest-EV decision. However, an unpredictable black swan or random noise caused the loss. Do not blame the AI if the logic was flawless.

OUTPUT FORMAT: You MUST return a strict JSON object with exactly two keys:
{
    "error_category": "<ONE_OF_THE_6_TAGS>",
    "explanation": "<A concise 2-sentence explanation of why you chose this tag>"
}
No markdown formatting, no backticks, ONLY raw JSON.
"""

class ErrorCategorizer:
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self.router = LLMRouter(temperature=0.1, request_timeout=30)
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._ensure_columns()

    def _get_db_connection(self):
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _ensure_columns(self):
        """Ensure the error_category and error_explanation columns exist for Phase 6."""
        with self._get_db_connection() as conn:
            c = conn.cursor()
            c.execute("PRAGMA table_info(ai_decisions)")
            columns = [col['name'] for col in c.fetchall()]
            
            if 'error_category' not in columns:
                c.execute('ALTER TABLE ai_decisions ADD COLUMN error_category TEXT')
            if 'error_explanation' not in columns:
                c.execute('ALTER TABLE ai_decisions ADD COLUMN error_explanation TEXT')
            conn.commit()

    def get_unclassified_losses(self) -> List[Dict[str, Any]]:
        """Fetch all trades that closed in a loss AND have not been categorized yet."""
        try:
            with self._get_db_connection() as conn:
                c = conn.cursor()
                c.execute('''
                    SELECT id, pair, timestamp, signal_type, confidence, 
                           reasoning_summary, regime, outcome_pnl, outcome_duration
                    FROM ai_decisions
                    WHERE outcome_pnl < 0.0 
                      AND error_category IS NULL
                    ORDER BY timestamp ASC LIMIT 50
                ''')
                rows = c.fetchall()
                return [dict(row) for row in rows]
        except Exception as e:
            logger.error(f"Failed to fetch unclassified losses: {e}")
            return []

    def classify_loss(self, trade: Dict[str, Any]) -> bool:
        """Ping the LLM to classify a single loss, then write it to the DB."""
        prompt = f"""
        TRADE POST-MORTEM DATA:
        Pair: {trade['pair']}
        Predicted Signal: {trade['signal_type']} with Confidence {trade['confidence']}
        Market Regime at Entry: {trade['regime']}
        
        Original AI Reasoning:
        "{trade['reasoning_summary']}"
        
        Actual Outcome:
        Trade exited with a LOSS of {trade['outcome_pnl']}% after {trade['outcome_duration']} minutes.
        
        Based on the above, classify the error strictly matching the System Prompt guidelines.
        """
        
        messages = [
            SystemMessage(content=ERROR_ANALYSIS_SYSTEM_PROMPT),
            HumanMessage(content=prompt)
        ]
        
        try:
            logger.info(f"Pinging LLM to categorize loss on {trade['pair']} (ID: {trade['id']})...")
            # Invoke the Round-Robin failover architecture constructed in Phase 5.3
            response = self.router.invoke(messages)
            # Handle LangChain output formatting differences (List vs String)
            content = response.content
            if isinstance(content, list):
                # Sometimes models return lists of message blocks
                content = " ".join([str(c) for c in content])
            
            content_str = str(content).strip()

            # Clean possible markdown JSON wrappers gracefully
            content_str = content_str.replace("```json", "").replace("```", "").strip()

            if not content_str:
                logger.warning(f"[ErrorCategorizer] Empty LLM response for trade {trade['id']}. Skipping.")
                return False

            result = json.loads(content_str)
            category = result.get("error_category", "UNKNOWN")
            explanation = result.get("explanation", "Failed to parse explanation.")
            
            with self._get_db_connection() as conn:
                c = conn.cursor()
                c.execute('''
                    UPDATE ai_decisions
                    SET error_category = ?, error_explanation = ?
                    WHERE id = ?
                ''', (category, explanation, trade['id']))
                conn.commit()
                
            logger.info(f"Classified ID {trade['id']} as {category}.")
            return True
            
        except json.JSONDecodeError as e:
            logger.error(f"LLM did not return valid JSON for trade {trade['id']}: {e}\nRaw Output: {content}")
            return False
        except Exception as e:
            logger.error(f"Routing/API Error during classification for ID {trade['id']}: {e}")
            return False

    def run_batch_classification(self):
        """Scans the DB for all new losing trades and categorizes them sequentially."""
        logger.info("Starting Phase 6.1 Batch Error Categorization...")
        losses = self.get_unclassified_losses()
        
        if not losses:
            logger.info("No unclassified loss-making trades found. System is clean!")
            return
            
        logger.info(f"Found {len(losses)} unclassified losses. Beginning post-mortem analysis...")
        success_count = 0
        
        for trade in losses:
            if self.classify_loss(trade):
                success_count += 1
                
        logger.info(f"Batch completed. Categorized {success_count}/{len(losses)} trades.")

if __name__ == "__main__":
    categorizer = ErrorCategorizer()
    categorizer.run_batch_classification()
