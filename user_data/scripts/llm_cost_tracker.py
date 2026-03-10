import sqlite3
import logging
from datetime import datetime, timezone
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

from ai_config import AI_DB_PATH as DB_PATH

class LLMCostTracker:
    """
    Tracks token usage and calculates costs for LLM API calls.
    Maintains a daily budget to prevent massive unexpected API bills.
    """
    
    # Cost per 1M tokens (Input, Output) in USD
    COSTS_PER_1M = {
        "gemini-2.5-flash": {"input": 0.15, "output": 0.60},
        "gemini-2.5-flash-lite": {"input": 0.075, "output": 0.30},
        "gemini-2.5-pro": {"input": 1.25, "output": 10.00},
        "llama-3.3-70b-versatile": {"input": 0.59, "output": 0.79},
        "models/text-embedding-004": {"input": 0.0, "output": 0.0}, # Minimal/free for low volume
    }

    def __init__(self, db_path=None):
        import ai_config
        self.db_path = db_path if db_path is not None else ai_config.AI_DB_PATH
        self._init_db()

    def _init_db(self):
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS llm_calls (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        model TEXT NOT NULL,
                        provider TEXT NOT NULL,
                        agent_name TEXT,
                        input_tokens INTEGER DEFAULT 0,
                        output_tokens INTEGER DEFAULT 0,
                        cost_usd REAL DEFAULT 0.0,
                        latency_ms REAL DEFAULT 0.0,
                        status TEXT DEFAULT 'success',
                        cache_hit BOOLEAN DEFAULT 0,
                        trading_pair TEXT
                    )
                """)
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to init llm_calls table: {e}")

    def calculate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        # Normalize model name for lookup
        model_key = model.lower()
        if "flash-lite" in model_key or "lite" in model_key:
            model_key = "gemini-2.5-flash-lite"
        elif "pro" in model_key:
            model_key = "gemini-2.5-pro"
        elif "llama" in model_key:
            model_key = "llama-3.3-70b-versatile"
        else:
            # Default to flash pricing if unknown
            model_key = "gemini-2.5-flash"
            
        costs = self.COSTS_PER_1M.get(model_key, self.COSTS_PER_1M["gemini-2.5-flash"])
        input_cost = (input_tokens / 1_000_000) * costs["input"]
        output_cost = (output_tokens / 1_000_000) * costs["output"]
        return input_cost + output_cost

    def log_call(self, model: str, provider: str, input_tokens: int, output_tokens: int,
                 cost_usd: float, latency_ms: float, agent_name: str = "", 
                 cache_hit: bool = False, pair: str = "", status: str = "success"):
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO llm_calls 
                    (model, provider, agent_name, input_tokens, output_tokens, cost_usd, latency_ms, status, cache_hit, trading_pair)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (model, provider, agent_name, input_tokens, output_tokens, cost_usd, latency_ms, status, int(cache_hit), pair))
                conn.commit()
        except Exception as e:
            logger.error(f"Error logging LLM call: {e}")

    def get_daily_cost(self, target_date: Optional[str] = None) -> float:
        """Get the total cost for a specific date (YYYY-MM-DD). Defaults to today."""
        if target_date is None:
            target_date = datetime.now(timezone.utc).strftime('%Y-%m-%d')
            
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT SUM(cost_usd) FROM llm_calls 
                    WHERE date(timestamp) = ?
                """, (target_date,))
                row = cursor.fetchone()
                return float(row[0]) if row and row[0] is not None else 0.0
        except Exception as e:
            logger.error(f"Error getting daily cost: {e}")
            return 0.0

    def get_daily_summary(self, target_date: Optional[str] = None) -> Dict[str, Any]:
        """Get an aggregate summary of calls, tokens, and latency for a date."""
        if target_date is None:
            target_date = datetime.now(timezone.utc).strftime('%Y-%m-%d')
            
        summary = {"total_cost": 0.0, "total_calls": 0, "models": {}}
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT model, COUNT(*), SUM(input_tokens), SUM(output_tokens), SUM(cost_usd), AVG(latency_ms)
                    FROM llm_calls 
                    WHERE date(timestamp) = ?
                    GROUP BY model
                """, (target_date,))
                
                rows = cursor.fetchall()
                for row in rows:
                    model, calls, in_tok, out_tok, cost, avg_lat = row
                    summary["models"][model] = {
                        "calls": calls,
                        "input_tokens": in_tok,
                        "output_tokens": out_tok,
                        "cost_usd": cost,
                        "avg_latency_ms": avg_lat
                    }
                    summary["total_calls"] += calls
                    summary["total_cost"] += (cost or 0.0)
                    
        except Exception as e:
            logger.error(f"Error getting daily summary: {e}")
            
        return summary

    def check_budget(self, daily_limit_usd: float = 5.0) -> bool:
        """Check if we are under the daily budget."""
        daily_cost = self.get_daily_cost()
        if daily_cost >= daily_limit_usd:
            logger.warning(f"LLM DAILY BUDGET EXCEEDED! Current: ${daily_cost:.4f}, Limit: ${daily_limit_usd:.4f}")
            try:
                from telegram_notifier import AITelegramNotifier
                AITelegramNotifier().send_alert(f"Daily API budget exceeded (${daily_cost:.2f} / ${daily_limit_usd:.2f})", level="CRITICAL")
            except Exception:
                pass
            return False
        return True
