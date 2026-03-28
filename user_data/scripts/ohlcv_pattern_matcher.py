"""
OHLCV Pattern Matcher — Real time-series similarity search.

Instead of bucket-based k-NN (RSI=oversold → match), this embeds actual
OHLCV candle sequences as numeric fingerprints and finds the most similar
historical patterns using Euclidean distance.

"Bu candle pattern 2024-Mart'taki patterne %89 benziyor, o zaman +3.5% yükseldi."

Fingerprint = 20 normalized returns + 6 indicator features = 26-dim vector.
Stored in SQLite, compared at query time (~10ms for 10K patterns).
"""

import os
import sys
import json
import math
import sqlite3
import logging
from typing import Dict, Any, Optional, List, Tuple

sys.path.append(os.path.dirname(__file__))

from ai_config import AI_DB_PATH

logger = logging.getLogger(__name__)

FINGERPRINT_DIM = 26  # 20 returns + RSI + MACD + ADX + volume_ratio + ATR_ratio + FNG


class OHLCVPatternMatcher:
    """
    Stores and matches OHLCV candle pattern fingerprints.
    Each fingerprint captures: last 20 candle returns + indicator snapshot.
    """

    def __init__(self, db_path: str = AI_DB_PATH):
        self.db_path = db_path
        self._init_db()

    def _get_conn(self):
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_db(self):
        try:
            with self._get_conn() as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS ohlcv_patterns (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        pair TEXT NOT NULL,
                        timeframe TEXT DEFAULT '1h',
                        timestamp TEXT,
                        fingerprint TEXT NOT NULL,
                        outcome_1h REAL,
                        outcome_4h REAL,
                        outcome_24h REAL,
                        direction TEXT,
                        indicators_json TEXT,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                conn.execute("CREATE INDEX IF NOT EXISTS idx_ohlcv_pair ON ohlcv_patterns(pair)")
                conn.commit()
        except Exception as e:
            logger.error(f"[OHLCVMatcher] DB init failed: {e}")

    # ── Fingerprint Creation ──────────────────────────────────

    @staticmethod
    def compute_fingerprint(closes: List[float], indicators: Dict[str, float] = None,
                            window: int = 20) -> List[float]:
        """
        Compute a fingerprint from OHLCV close prices + indicators.

        Args:
            closes: List of close prices (at least window+1 values).
            indicators: Optional dict with rsi, macd_hist, adx, volume_ratio, atr_ratio, fng.
            window: Number of returns to include (default 20).

        Returns:
            List of floats (fingerprint vector, length = window + 6).
        """
        if len(closes) < window + 1:
            return []

        # Normalized returns (percentage change, clamped to ±10%)
        recent = closes[-(window + 1):]
        returns = []
        for i in range(1, len(recent)):
            if recent[i - 1] > 0:
                ret = (recent[i] - recent[i - 1]) / recent[i - 1]
                ret = max(-0.10, min(0.10, ret))  # Clamp to ±10%
                returns.append(round(ret, 6))
            else:
                returns.append(0.0)

        # Indicator features (normalized to 0-1 range)
        ind = indicators or {}
        rsi_norm = ind.get("rsi", 50) / 100.0
        macd_norm = max(-1, min(1, ind.get("macd_hist", 0) / 2.0)) * 0.5 + 0.5  # [-2,2] → [0,1]
        adx_norm = min(ind.get("adx", 25) / 50.0, 1.0)
        vol_norm = min(ind.get("volume_ratio", 1.0) / 3.0, 1.0)
        atr_norm = min(ind.get("atr_ratio", 1.0) / 3.0, 1.0)
        fng_norm = ind.get("fng", 50) / 100.0

        fingerprint = returns + [rsi_norm, macd_norm, adx_norm, vol_norm, atr_norm, fng_norm]
        return fingerprint

    # ── Storage ──────────────────────────────────────────────

    def store_pattern(self, pair: str, fingerprint: List[float], timestamp: str = "",
                      outcome_1h: float = None, outcome_4h: float = None,
                      outcome_24h: float = None, direction: str = None,
                      timeframe: str = "1h", indicators: Dict = None):
        """Store a single OHLCV pattern with its outcomes."""
        if not fingerprint or len(fingerprint) < 10:
            return

        try:
            with self._get_conn() as conn:
                conn.execute("""
                    INSERT INTO ohlcv_patterns
                    (pair, timeframe, timestamp, fingerprint, outcome_1h, outcome_4h,
                     outcome_24h, direction, indicators_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    pair, timeframe, timestamp, json.dumps(fingerprint),
                    outcome_1h, outcome_4h, outcome_24h, direction,
                    json.dumps(indicators) if indicators else None
                ))
                conn.commit()
        except Exception as e:
            logger.error(f"[OHLCVMatcher] Store failed: {e}")

    def store_batch(self, patterns: List[Dict]):
        """Store multiple patterns efficiently."""
        try:
            with self._get_conn() as conn:
                for p in patterns:
                    fp = p.get("fingerprint", [])
                    if not fp or len(fp) < 10:
                        continue
                    conn.execute("""
                        INSERT INTO ohlcv_patterns
                        (pair, timeframe, timestamp, fingerprint, outcome_1h, outcome_4h,
                         outcome_24h, direction, indicators_json)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        p.get("pair"), p.get("timeframe", "1h"), p.get("timestamp", ""),
                        json.dumps(fp), p.get("outcome_1h"), p.get("outcome_4h"),
                        p.get("outcome_24h"), p.get("direction"),
                        json.dumps(p.get("indicators")) if p.get("indicators") else None
                    ))
                conn.commit()
                logger.info(f"[OHLCVMatcher] Stored {len(patterns)} patterns")
        except Exception as e:
            logger.error(f"[OHLCVMatcher] Batch store failed: {e}")

    # ── Similarity Search ──────────────────────────────────────

    def find_similar(self, query_fingerprint: List[float], k: int = 10,
                     pair: Optional[str] = None) -> Dict[str, Any]:
        """
        Find k most similar historical OHLCV patterns.

        Returns:
            {
                "matches": [{"distance": float, "pair": str, "outcome_1h": float, ...}],
                "predicted_1h": float (weighted avg outcome),
                "predicted_4h": float,
                "predicted_24h": float,
                "confidence": float (inverse of avg distance),
                "total_patterns": int,
            }
        """
        if not query_fingerprint or len(query_fingerprint) < 10:
            return {"matches": [], "error": "Invalid fingerprint"}

        try:
            with self._get_conn() as conn:
                conditions = ["1=1"]
                params = []
                if pair:
                    conditions.append("pair = ?")
                    params.append(pair)

                rows = conn.execute(f"""
                    SELECT pair, timestamp, fingerprint, outcome_1h, outcome_4h,
                           outcome_24h, direction
                    FROM ohlcv_patterns
                    WHERE {' AND '.join(conditions)}
                """, params).fetchall()

                if len(rows) < k:
                    return {"matches": [], "insufficient_data": True, "total_patterns": len(rows)}

                # Compute distances
                scored = []
                for row in rows:
                    try:
                        stored_fp = json.loads(row["fingerprint"])
                        dist = self._euclidean_distance(query_fingerprint, stored_fp)
                        scored.append((dist, dict(row)))
                    except (json.JSONDecodeError, TypeError):
                        continue

                scored.sort(key=lambda x: x[0])
                top_k = scored[:k]

                # Weighted predictions (closer patterns get more weight)
                matches = []
                weights = []
                outcomes_1h = []
                outcomes_4h = []
                outcomes_24h = []

                for dist, row in top_k:
                    weight = 1.0 / (dist + 0.001)  # Inverse distance weighting
                    weights.append(weight)

                    match_info = {
                        "distance": round(dist, 4),
                        "similarity": round(max(0, 1.0 - dist * 5), 2),  # Scale to 0-1
                        "pair": row["pair"],
                        "timestamp": row["timestamp"],
                        "direction": row["direction"],
                    }

                    if row["outcome_1h"] is not None:
                        outcomes_1h.append((row["outcome_1h"], weight))
                        match_info["outcome_1h"] = round(row["outcome_1h"], 3)
                    if row["outcome_4h"] is not None:
                        outcomes_4h.append((row["outcome_4h"], weight))
                        match_info["outcome_4h"] = round(row["outcome_4h"], 3)
                    if row["outcome_24h"] is not None:
                        outcomes_24h.append((row["outcome_24h"], weight))
                        match_info["outcome_24h"] = round(row["outcome_24h"], 3)

                    matches.append(match_info)

                def weighted_avg(pairs):
                    if not pairs:
                        return None
                    total_w = sum(w for _, w in pairs)
                    if total_w == 0:
                        return None
                    return round(sum(v * w for v, w in pairs) / total_w, 3)

                avg_dist = sum(d for d, _ in top_k) / len(top_k) if top_k else 1.0
                confidence = round(max(0, min(1, 1.0 - avg_dist * 3)), 2)

                result = {
                    "matches": matches,
                    "predicted_1h": weighted_avg(outcomes_1h),
                    "predicted_4h": weighted_avg(outcomes_4h),
                    "predicted_24h": weighted_avg(outcomes_24h),
                    "confidence": confidence,
                    "avg_distance": round(avg_dist, 4),
                    "total_patterns": len(rows),
                }

                logger.info(f"[OHLCVMatcher] Found {k} similar patterns (avg_dist={avg_dist:.4f}, "
                            f"confidence={confidence:.2f}, pred_4h={result['predicted_4h']})")
                return result

        except Exception as e:
            logger.error(f"[OHLCVMatcher] Search failed: {e}")
            return {"matches": [], "error": str(e)}

    @staticmethod
    def _euclidean_distance(a: List[float], b: List[float]) -> float:
        """Compute Euclidean distance between two vectors."""
        min_len = min(len(a), len(b))
        return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(min_len)))

    def format_for_prompt(self, query_fingerprint: List[float], k: int = 10,
                          pair: Optional[str] = None) -> str:
        """Format pattern matching results for LLM prompt injection."""
        result = self.find_similar(query_fingerprint, k=k, pair=pair)

        if result.get("insufficient_data") or not result.get("matches"):
            return ""

        conf = result["confidence"]
        pred_1h = result.get("predicted_1h")
        pred_4h = result.get("predicted_4h")
        pred_24h = result.get("predicted_24h")

        lines = [
            f"=== OHLCV PATTERN MATCH ({len(result['matches'])} similar candle patterns) ===",
            f"Pattern Confidence: {conf:.0%} | Avg Distance: {result['avg_distance']:.4f}",
        ]

        preds = []
        if pred_1h is not None:
            preds.append(f"1h: {pred_1h:+.2f}%")
        if pred_4h is not None:
            preds.append(f"4h: {pred_4h:+.2f}%")
        if pred_24h is not None:
            preds.append(f"24h: {pred_24h:+.2f}%")
        if preds:
            lines.append(f"Predicted Outcomes: {' | '.join(preds)}")

        # Show top 3
        for i, m in enumerate(result["matches"][:3], 1):
            parts = [f"#{i} {m['pair']}"]
            if m.get("outcome_4h") is not None:
                parts.append(f"→ {m['outcome_4h']:+.2f}% (4h)")
            parts.append(f"sim={m['similarity']:.0%}")
            lines.append(f"  {'  '.join(parts)}")

        lines.append("Candle pattern similarity is a LEADING indicator. High similarity + positive outcomes = bullish.")
        return "\n".join(lines)

    def get_total_patterns(self) -> int:
        """Return total stored patterns."""
        try:
            with self._get_conn() as conn:
                row = conn.execute("SELECT COUNT(*) as cnt FROM ohlcv_patterns").fetchone()
                return row["cnt"] if row else 0
        except Exception:
            return 0
