"""
analytics_engine.py — Phase 28: DuckDB Analytics Engine

SQLite'tan 10-100x hizli OLAP sorgulari icin DuckDB wrapper.
SQLite verilerini ATTACH ile okur — veri kopyalamaya gerek yok.

Kullanim:
    from analytics_engine import get_analytics
    engine = get_analytics()
    # Rolling Sharpe
    result = engine.rolling_sharpe("BTC/USDT:USDT", window=30)
    # Regime performance
    result = engine.regime_performance()
    # Pattern correlation matrix
    result = engine.pattern_correlation("BTC/USDT:USDT")
    # Sprint 2: Replay buffer analytics
    result = engine.replay_buffer_sample(batch_size=64)
"""

import os
import logging
import threading
from typing import Dict, List, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

try:
    import duckdb
    DUCKDB_AVAILABLE = True
except ImportError:
    DUCKDB_AVAILABLE = False
    logger.warning("[Analytics] duckdb not installed. pip install duckdb")

from ai_config import AI_DB_PATH

# DuckDB file path
DUCKDB_PATH = os.environ.get(
    "DUCKDB_PATH",
    os.path.join(os.path.dirname(AI_DB_PATH), "analytics.duckdb")
)


class AnalyticsEngine:
    """DuckDB-based analytics engine. ATTACHes SQLite for zero-copy reads."""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        if not DUCKDB_AVAILABLE:
            raise ImportError("duckdb not installed. Run: pip install duckdb")

        os.makedirs(os.path.dirname(DUCKDB_PATH), exist_ok=True)
        self._conn = duckdb.connect(DUCKDB_PATH)

        # Attach SQLite for reading AI data
        try:
            self._conn.execute("INSTALL sqlite; LOAD sqlite;")
        except Exception:
            pass  # Already installed

        try:
            self._conn.execute(f"ATTACH '{AI_DB_PATH}' AS ai (TYPE sqlite, READ_ONLY)")
            logger.info(f"[Analytics] DuckDB attached SQLite: {AI_DB_PATH}")
        except Exception as e:
            if "already attached" not in str(e).lower():
                logger.warning(f"[Analytics] SQLite attach failed: {e}")

        self._init_tables()
        self._initialized = True
        logger.info(f"[Analytics] DuckDB ready at {DUCKDB_PATH}")

    def _init_tables(self):
        """Create DuckDB-native tables for Sprint 2 data."""
        # Sprint 2: RL replay buffer (columnar, fast batch sampling)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS rl_episodes (
                episode_id INTEGER, step INTEGER,
                state BLOB, action BLOB,
                reward DOUBLE, next_state BLOB,
                done BOOLEAN DEFAULT false,
                source VARCHAR DEFAULT 'live',
                regime VARCHAR, pair VARCHAR,
                ts TIMESTAMP DEFAULT current_timestamp
            )
        """)

        # Sprint 2: World model state snapshots
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS wm_states (
                pair VARCHAR, timeframe VARCHAR DEFAULT '1h',
                ts TIMESTAMP, ttm_embedding BLOB,
                chart_features BLOB, regime VARCHAR,
                fng INTEGER, hormones BLOB
            )
        """)

        # Sprint 2: Dream scenarios
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS dream_trajectories (
                session_id VARCHAR, trajectory_idx INTEGER,
                step INTEGER, state BLOB,
                event_type VARCHAR, reward DOUBLE,
                passed_filter BOOLEAN DEFAULT false,
                ts TIMESTAMP DEFAULT current_timestamp
            )
        """)

        # Sprint 2: Organ performance time series
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS organ_perf (
                organ VARCHAR, regime VARCHAR,
                metric VARCHAR, value DOUBLE,
                sample_size INTEGER,
                ts TIMESTAMP DEFAULT current_timestamp
            )
        """)

        # Sprint 2: Hormone history
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS hormone_ts (
                cortisol DOUBLE, dopamine DOUBLE,
                serotonin DOUBLE, adrenaline DOUBLE,
                market_stress DOUBLE, organism_health DOUBLE,
                trigger_event VARCHAR,
                ts TIMESTAMP DEFAULT current_timestamp
            )
        """)

    # ─── OLAP Analytics (read from attached SQLite) ─────────────────

    def rolling_sharpe(self, pair: Optional[str] = None, window: int = 30) -> List[Dict]:
        """Rolling Sharpe ratio over trade outcomes."""
        try:
            where = f"WHERE pair = '{pair}'" if pair else ""
            result = self._conn.execute(f"""
                SELECT
                    pair,
                    timestamp,
                    outcome_pnl,
                    AVG(outcome_pnl) OVER w AS avg_pnl,
                    STDDEV(outcome_pnl) OVER w AS std_pnl,
                    CASE WHEN STDDEV(outcome_pnl) OVER w > 0
                         THEN AVG(outcome_pnl) OVER w / STDDEV(outcome_pnl) OVER w
                         ELSE 0 END AS sharpe
                FROM ai.ai_decisions
                {where}
                AND outcome_pnl IS NOT NULL
                WINDOW w AS (PARTITION BY pair ORDER BY timestamp
                             ROWS BETWEEN {window-1} PRECEDING AND CURRENT ROW)
                ORDER BY timestamp DESC
                LIMIT 100
            """).fetchall()
            columns = ["pair", "timestamp", "pnl", "avg_pnl", "std_pnl", "sharpe"]
            return [dict(zip(columns, row)) for row in result]
        except Exception as e:
            logger.error(f"[Analytics] rolling_sharpe failed: {e}")
            return []

    def regime_performance(self) -> List[Dict]:
        """Performance breakdown by regime."""
        try:
            result = self._conn.execute("""
                SELECT
                    regime,
                    COUNT(*) AS trades,
                    SUM(CASE WHEN outcome_pnl > 0 THEN 1 ELSE 0 END) AS wins,
                    ROUND(AVG(outcome_pnl), 4) AS avg_pnl,
                    ROUND(SUM(outcome_pnl), 2) AS total_pnl,
                    ROUND(STDDEV(outcome_pnl), 4) AS pnl_std,
                    ROUND(MAX(outcome_pnl), 4) AS best_trade,
                    ROUND(MIN(outcome_pnl), 4) AS worst_trade
                FROM ai.ai_decisions
                WHERE outcome_pnl IS NOT NULL AND regime IS NOT NULL
                GROUP BY regime
                ORDER BY total_pnl DESC
            """).fetchall()
            columns = ["regime", "trades", "wins", "avg_pnl", "total_pnl",
                        "pnl_std", "best_trade", "worst_trade"]
            return [dict(zip(columns, row)) for row in result]
        except Exception as e:
            logger.error(f"[Analytics] regime_performance failed: {e}")
            return []

    def hourly_performance(self) -> List[Dict]:
        """Performance by hour of day (for Cerebellum)."""
        try:
            result = self._conn.execute("""
                SELECT
                    EXTRACT(HOUR FROM CAST(timestamp AS TIMESTAMP)) AS hour,
                    COUNT(*) AS trades,
                    SUM(CASE WHEN outcome_pnl > 0 THEN 1 ELSE 0 END) AS wins,
                    ROUND(AVG(outcome_pnl), 4) AS avg_pnl
                FROM ai.ai_decisions
                WHERE outcome_pnl IS NOT NULL
                GROUP BY hour
                ORDER BY hour
            """).fetchall()
            columns = ["hour", "trades", "wins", "avg_pnl"]
            return [dict(zip(columns, row)) for row in result]
        except Exception as e:
            logger.error(f"[Analytics] hourly_performance failed: {e}")
            return []

    def agent_leaderboard(self) -> List[Dict]:
        """Agent performance leaderboard."""
        try:
            result = self._conn.execute("""
                SELECT
                    agent_type,
                    COUNT(*) AS trades,
                    SUM(CASE WHEN was_correct THEN 1 ELSE 0 END) AS correct,
                    ROUND(AVG(outcome_pnl), 4) AS avg_pnl,
                    ROUND(SUM(outcome_pnl), 2) AS total_pnl
                FROM ai.agent_performance
                GROUP BY agent_type
                ORDER BY total_pnl DESC
            """).fetchall()
            columns = ["agent_type", "trades", "correct", "avg_pnl", "total_pnl"]
            return [dict(zip(columns, row)) for row in result]
        except Exception as e:
            logger.error(f"[Analytics] agent_leaderboard failed: {e}")
            return []

    def pattern_win_rate_by_features(self, pair: Optional[str] = None) -> List[Dict]:
        """Win rate breakdown by pattern features (RSI bucket, regime, etc.)."""
        try:
            where = f"WHERE pair = '{pair}'" if pair else ""
            result = self._conn.execute(f"""
                SELECT
                    rsi_bucket, regime, direction,
                    COUNT(*) AS trades,
                    ROUND(AVG(CASE WHEN profit_pct > 0 THEN 1.0 ELSE 0.0 END), 3) AS win_rate,
                    ROUND(AVG(profit_pct), 4) AS avg_pnl
                FROM ai.pattern_trades
                {where}
                GROUP BY rsi_bucket, regime, direction
                HAVING trades >= 5
                ORDER BY win_rate DESC
            """).fetchall()
            columns = ["rsi_bucket", "regime", "direction", "trades", "win_rate", "avg_pnl"]
            return [dict(zip(columns, row)) for row in result]
        except Exception as e:
            logger.error(f"[Analytics] pattern_win_rate failed: {e}")
            return []

    def forgone_pnl_analysis(self) -> List[Dict]:
        """Analyze forgone profits — what did we miss by not trading?"""
        try:
            result = self._conn.execute("""
                SELECT
                    pair,
                    COUNT(*) AS missed_trades,
                    ROUND(SUM(forgone_pnl), 2) AS total_forgone,
                    ROUND(AVG(confidence), 3) AS avg_confidence,
                    ROUND(AVG(forgone_pnl), 4) AS avg_forgone_pnl
                FROM ai.forgone_profit
                WHERE was_executed = 0 AND forgone_pnl IS NOT NULL
                GROUP BY pair
                ORDER BY total_forgone DESC
            """).fetchall()
            columns = ["pair", "missed_trades", "total_forgone", "avg_confidence", "avg_forgone_pnl"]
            return [dict(zip(columns, row)) for row in result]
        except Exception as e:
            logger.error(f"[Analytics] forgone_pnl_analysis failed: {e}")
            return []

    def llm_cost_summary(self, days: int = 7) -> Dict[str, Any]:
        """LLM API cost summary for last N days."""
        try:
            result = self._conn.execute(f"""
                SELECT
                    model,
                    COUNT(*) AS calls,
                    SUM(prompt_tokens) AS total_prompt_tokens,
                    SUM(completion_tokens) AS total_completion_tokens,
                    ROUND(SUM(cost_usd), 4) AS total_cost,
                    ROUND(AVG(latency_ms), 0) AS avg_latency_ms
                FROM ai.llm_calls
                WHERE timestamp > current_timestamp - INTERVAL '{days} days'
                GROUP BY model
                ORDER BY total_cost DESC
            """).fetchall()
            columns = ["model", "calls", "prompt_tokens", "completion_tokens", "total_cost", "avg_latency_ms"]
            rows = [dict(zip(columns, row)) for row in result]
            return {
                "period_days": days,
                "models": rows,
                "total_cost": sum(r["total_cost"] or 0 for r in rows),
                "total_calls": sum(r["calls"] or 0 for r in rows),
            }
        except Exception as e:
            logger.error(f"[Analytics] llm_cost_summary failed: {e}")
            return {"period_days": days, "models": [], "total_cost": 0, "total_calls": 0}

    # ─── Sprint 2: Replay Buffer Analytics ──────────────────────────

    def insert_episode(self, episode_id: int, step: int, state: bytes,
                       action: bytes, reward: float, next_state: bytes = None,
                       done: bool = False, source: str = "live",
                       regime: str = None, pair: str = None):
        """Insert a step into the RL replay buffer."""
        self._conn.execute("""
            INSERT INTO rl_episodes (episode_id, step, state, action, reward,
                                      next_state, done, source, regime, pair)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (episode_id, step, state, action, reward, next_state, done, source, regime, pair))

    def sample_replay_buffer(self, batch_size: int = 64,
                              source: str = None) -> List[tuple]:
        """Random batch sample from replay buffer (for RL training)."""
        try:
            where = f"WHERE source = '{source}'" if source else ""
            result = self._conn.execute(f"""
                SELECT episode_id, step, state, action, reward, next_state, done
                FROM rl_episodes {where}
                USING SAMPLE {batch_size}
            """).fetchall()
            return result
        except Exception as e:
            logger.error(f"[Analytics] replay sample failed: {e}")
            return []

    def replay_buffer_stats(self) -> Dict[str, Any]:
        """Replay buffer statistics."""
        try:
            result = self._conn.execute("""
                SELECT
                    source,
                    COUNT(*) AS steps,
                    COUNT(DISTINCT episode_id) AS episodes,
                    ROUND(AVG(reward), 4) AS avg_reward,
                    ROUND(MAX(reward), 4) AS max_reward,
                    ROUND(MIN(reward), 4) AS min_reward
                FROM rl_episodes
                GROUP BY source
            """).fetchall()
            columns = ["source", "steps", "episodes", "avg_reward", "max_reward", "min_reward"]
            return {"buffers": [dict(zip(columns, row)) for row in result]}
        except Exception as e:
            return {"buffers": []}

    # ─── Sprint 2: World Model States ───────────────────────────────

    def insert_world_state(self, pair: str, timestamp: str,
                            ttm_embedding: bytes = None,
                            chart_features: bytes = None,
                            regime: str = None, fng: int = None,
                            hormones: bytes = None, timeframe: str = "1h"):
        """Store a world model state snapshot."""
        self._conn.execute("""
            INSERT INTO wm_states (pair, timeframe, ts, ttm_embedding,
                                    chart_features, regime, fng, hormones)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (pair, timeframe, timestamp, ttm_embedding, chart_features,
              regime, fng, hormones))

    def insert_hormone_snapshot(self, cortisol: float, dopamine: float,
                                 serotonin: float, adrenaline: float,
                                 market_stress: float = 0, organism_health: float = 0.5,
                                 trigger: str = "trade_close"):
        """Store hormone history snapshot."""
        self._conn.execute("""
            INSERT INTO hormone_ts (cortisol, dopamine, serotonin, adrenaline,
                                     market_stress, organism_health, trigger_event)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (cortisol, dopamine, serotonin, adrenaline, market_stress,
              organism_health, trigger))

    # ─── Stats ──────────────────────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        """Return engine statistics."""
        stats = {"path": DUCKDB_PATH, "tables": {}}
        try:
            tables = self._conn.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_schema='main'"
            ).fetchall()
            for (name,) in tables:
                count = self._conn.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0]
                stats["tables"][name] = count
        except Exception:
            pass
        return stats

    def close(self):
        """Close DuckDB connection."""
        try:
            self._conn.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_analytics = None
_analytics_lock = threading.Lock()


def get_analytics() -> AnalyticsEngine:
    """Thread-safe singleton AnalyticsEngine accessor."""
    global _analytics
    if _analytics is None:
        with _analytics_lock:
            if _analytics is None:
                _analytics = AnalyticsEngine()
    return _analytics
