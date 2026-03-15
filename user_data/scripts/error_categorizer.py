import os
import re
import sqlite3
import json
import logging
from typing import Dict, Any, List
from llm_router import LLMRouter
from langchain_core.messages import SystemMessage, HumanMessage
from ai_config import AI_DB_PATH as DB_PATH
from json_utils import extract_json_strict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# System Prompt to instruct the Coordinator on how to grade its own failures.
ERROR_ANALYSIS_SYSTEM_PROMPT = """IDENTITY: You are the Chief Risk Officer conducting an autonomous post-mortem on a LOSING trade.
Your job: ruthlessly identify the ROOT CAUSE of failure — not symptoms, not excuses.

CONSTITUTIONAL RULES:
1. Choose EXACTLY ONE category. If multiple apply, pick the PRIMARY cause (the one that, if fixed, would have prevented the loss).
2. Your explanation must reference SPECIFIC data from the trade (confidence level, regime, indicator values cited in reasoning).
3. Do NOT blame "the market" unless the logic was genuinely flawless (category 6 only).
4. Your ENTIRE response MUST be a single valid JSON object. Nothing else.

ERROR CATEGORIES:
1. [HALLUCINATION]: AI cited fake news, fabricated indicator values, or misread data (e.g., claimed "RSI oversold" when RSI was 55).
2. [REGIME_MISMATCH]: Logic was correct for one regime but the actual regime was different (e.g., trend-following in a ranging market).
3. [TIMING_ERROR]: Direction was eventually correct but entry was premature (>24h early) or too late (after the move).
4. [POSITION_SIZING]: Conviction was mismatched with risk — high confidence on volatile/uncertain setup, or low confidence but oversized.
5. [CORRELATION_ERROR]: Overexposed to correlated assets (e.g., long BTC+ETH+SOL = 3x the same bet).
6. [MARKET_NOISE]: The decision was mathematically correct, highest-EV play. Loss was caused by unpredictable event/noise. Do NOT use this as a catch-all — only when logic was genuinely sound.

EXAMPLES:
Input: BULLISH signal, confidence 0.82, reasoning "RSI oversold at 28", actual RSI was 52 → {"error_category":"HALLUCINATION","explanation":"AI claimed RSI was oversold at 28 but actual RSI was 52. The entire bull case was built on fabricated indicator data."}
Input: BULLISH signal, confidence 0.65, ranging market, loss -3% → {"error_category":"REGIME_MISMATCH","explanation":"AI applied trend-following logic in a ranging market (ADX=15). Breakout signal failed as price reverted to range midpoint."}
Input: BEARISH signal, confidence 0.60, market dropped 2 days later → {"error_category":"TIMING_ERROR","explanation":"Bear thesis was correct but entry was 48h early. Stop loss hit before the predicted move materialized."}

OUTPUT FORMAT:
{"error_category":"<ONE_OF_THE_6_TAGS>","explanation":"<2-sentence explanation referencing specific trade data>"}"""

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
            response = self.router.invoke(messages, priority="low")
            # Handle LangChain output formatting differences (List vs String)
            content = response.content
            if isinstance(content, list):
                # Sometimes models return lists of message blocks
                content = " ".join([str(c) for c in content])
            
            content_str = str(content).strip()

            if not content_str:
                logger.warning(f"[ErrorCategorizer] Empty LLM response for trade {trade['id']}. Skipping.")
                return False

            result = extract_json_strict(content_str, required_keys=["error_category"])
            if result is None:
                logger.error(f"[ErrorCategorizer] Failed to extract JSON for trade {trade['id']}. Raw: {content_str[:300]}")
                return False
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
