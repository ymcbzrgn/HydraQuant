"""
RAG Feedback Loop: Trade PnL → Chunk quality score update.
Self-improving retrieval — profitable trade chunks get boosted,
losing trade chunks get penalized. Over time, retrieval quality
automatically improves as the system learns from outcomes.

Tables:
  chunk_quality_scores: per-chunk quality score (0.1-1.0)
  trade_chunk_map: which chunks were used in which trade
  chunk_quality_flags: flagged chunks for review
"""

import sqlite3
import logging
from typing import List, Optional

import os
import sys
sys.path.append(os.path.dirname(__file__))

from ai_config import AI_DB_PATH

logger = logging.getLogger(__name__)


class RAGFeedbackLoop:
    """Closed-loop feedback: Trade outcomes improve retrieval quality."""

    def __init__(self, db_path: str = AI_DB_PATH):
        self.db_path = db_path
        self._init_db()

    def _get_conn(self):
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=30000")
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        try:
            with self._get_conn() as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS chunk_quality_scores (
                        chunk_id TEXT PRIMARY KEY,
                        quality_score REAL DEFAULT 0.5,
                        trade_count INTEGER DEFAULT 0,
                        win_count INTEGER DEFAULT 0,
                        total_pnl REAL DEFAULT 0.0,
                        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS trade_chunk_map (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        trade_id INTEGER,
                        chunk_id TEXT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS chunk_quality_flags (
                        chunk_id TEXT PRIMARY KEY,
                        flag TEXT,
                        flagged_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                conn.commit()
        except Exception as e:
            logger.error(f"[RAGFeedback] Init failed: {e}")

    def record_trade_chunks(self, trade_id: int, chunk_ids: List[str]):
        """Record which chunks were used for a trade decision."""
        if not chunk_ids:
            return
        try:
            with self._get_conn() as conn:
                for cid in chunk_ids:
                    conn.execute(
                        "INSERT INTO trade_chunk_map (trade_id, chunk_id) VALUES (?, ?)",
                        (trade_id, cid)
                    )
                conn.commit()
                logger.debug(f"[RAGFeedback] Recorded {len(chunk_ids)} chunks for trade {trade_id}")
        except Exception as e:
            logger.debug(f"[RAGFeedback] Record failed: {e}")

    def update_scores_for_trade(self, trade_id: int, pnl_pct: float):
        """Update chunk quality scores based on trade outcome."""
        try:
            with self._get_conn() as conn:
                rows = conn.execute(
                    "SELECT chunk_id FROM trade_chunk_map WHERE trade_id = ?",
                    (trade_id,)
                ).fetchall()

                if not rows:
                    return

                is_win = pnl_pct > 0
                # Asymmetric update: reward wins faster than penalizing losses
                # Prevents good chunks from being killed by one bad trade
                delta = 0.05 if is_win else -0.03

                for row in rows:
                    cid = row["chunk_id"]
                    conn.execute("""
                        INSERT INTO chunk_quality_scores
                            (chunk_id, quality_score, trade_count, win_count, total_pnl)
                        VALUES (?, 0.5 + ?, 1, ?, ?)
                        ON CONFLICT(chunk_id) DO UPDATE SET
                            quality_score = MIN(1.0, MAX(0.1, quality_score + ?)),
                            trade_count = trade_count + 1,
                            win_count = win_count + ?,
                            total_pnl = total_pnl + ?,
                            updated_at = datetime('now')
                    """, (cid, delta, 1 if is_win else 0, pnl_pct,
                          delta, 1 if is_win else 0, pnl_pct))

                conn.commit()
                logger.info(f"[RAGFeedback] Trade {trade_id} PnL={pnl_pct:+.2f}% → "
                           f"updated {len(rows)} chunks ({'WIN' if is_win else 'LOSS'})")
        except Exception as e:
            logger.debug(f"[RAGFeedback] Score update failed: {e}")

    def get_chunk_boost(self, chunk_id: str) -> float:
        """Get boost factor for retrieval reranking (0.7 - 1.3)."""
        try:
            with self._get_conn() as conn:
                row = conn.execute(
                    "SELECT quality_score FROM chunk_quality_scores WHERE chunk_id = ?",
                    (chunk_id,)
                ).fetchone()
                if row:
                    return 0.7 + 0.6 * float(row["quality_score"])
                return 1.0  # Unknown chunk = neutral
        except Exception:
            return 1.0

    def get_summary(self) -> dict:
        """Get feedback loop summary stats."""
        try:
            with self._get_conn() as conn:
                row = conn.execute("""
                    SELECT COUNT(*) as total, AVG(quality_score) as avg_score,
                           SUM(trade_count) as total_trades, SUM(win_count) as total_wins
                    FROM chunk_quality_scores
                """).fetchone()
                return dict(row) if row else {}
        except Exception:
            return {}
