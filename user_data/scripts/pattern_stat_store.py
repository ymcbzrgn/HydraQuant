"""
PatternStatStore: Queryable statistics engine for backtest trade patterns.

Instead of embedding documents, this stores STATISTICS about what happened
historically under specific conditions. The RAG pipeline queries this
for empirical evidence to ground LLM reasoning.

Example query:
    stats = store.query(pair="BTC/USDT", rsi_bucket="oversold", regime="bull")
    → {"matching_trades": 47, "win_rate": 0.72, "avg_profit_pct": 3.2, ...}
"""

import os
import sys
import sqlite3
import logging
from typing import Dict, Any, Optional, List

sys.path.append(os.path.dirname(__file__))

from ai_config import AI_DB_PATH

logger = logging.getLogger(__name__)


class PatternStatStore:
    """
    SQLite-based statistical pattern store.
    Populated from freqtrade backtest results, queried during signal generation.
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
        """Create the pattern_trades table for storing backtest results."""
        try:
            with self._get_conn() as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS pattern_trades (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        pair TEXT NOT NULL,
                        strategy TEXT,
                        direction TEXT NOT NULL,
                        entry_date TEXT,
                        exit_date TEXT,
                        profit_pct REAL NOT NULL,
                        duration_hours REAL,
                        exit_reason TEXT,
                        regime TEXT DEFAULT 'unknown',
                        rsi_bucket TEXT,
                        macd_signal TEXT,
                        ema_alignment TEXT,
                        adx_bucket TEXT,
                        volume_bucket TEXT,
                        fng_bucket TEXT,
                        atr_bucket TEXT,
                        entry_price REAL,
                        indicators_json TEXT,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_pattern_pair_regime
                    ON pattern_trades(pair, regime)
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_pattern_conditions
                    ON pattern_trades(rsi_bucket, macd_signal, regime)
                """)
                conn.commit()
        except Exception as e:
            logger.error(f"[PatternStatStore] DB init failed: {e}")

    # ── Ingestion ──────────────────────────────────────────────

    def ingest_trade(self, trade: Dict[str, Any]):
        """
        Ingest a single backtest trade with its technical context.
        Called by BacktestEmbedder for each trade in backtest results.
        """
        try:
            with self._get_conn() as conn:
                conn.execute("""
                    INSERT INTO pattern_trades
                    (pair, strategy, direction, entry_date, exit_date, profit_pct,
                     duration_hours, exit_reason, regime, rsi_bucket, macd_signal,
                     ema_alignment, adx_bucket, volume_bucket, fng_bucket, atr_bucket,
                     entry_price, indicators_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    trade.get('pair'),
                    trade.get('strategy'),
                    trade.get('direction', 'long'),
                    trade.get('entry_date'),
                    trade.get('exit_date'),
                    trade.get('profit_pct', 0.0),
                    trade.get('duration_hours'),
                    trade.get('exit_reason'),
                    trade.get('regime', 'unknown'),
                    trade.get('rsi_bucket'),
                    trade.get('macd_signal'),
                    trade.get('ema_alignment'),
                    trade.get('adx_bucket'),
                    trade.get('volume_bucket'),
                    trade.get('fng_bucket'),
                    trade.get('atr_bucket'),
                    trade.get('entry_price'),
                    trade.get('indicators_json'),
                ))
                conn.commit()
        except Exception as e:
            logger.error(f"[PatternStatStore] Ingest failed: {e}")

    def ingest_batch(self, trades: List[Dict[str, Any]]):
        """Ingest multiple trades efficiently with duplicate detection.
        Deduplication: same pair + entry_date + exit_date + profit_pct = duplicate."""
        try:
            with self._get_conn() as conn:
                inserted = 0
                skipped = 0
                for trade in trades:
                    # Duplicate check: same pair + entry_date + profit_pct
                    entry_date = trade.get('entry_date', '')
                    pair = trade.get('pair', '')
                    profit = trade.get('profit_pct', 0.0)
                    if entry_date and pair:
                        existing = conn.execute(
                            "SELECT id FROM pattern_trades WHERE pair=? AND entry_date=? AND profit_pct=? LIMIT 1",
                            (pair, entry_date, profit)
                        ).fetchone()
                        if existing:
                            skipped += 1
                            continue

                    conn.execute("""
                        INSERT INTO pattern_trades
                        (pair, strategy, direction, entry_date, exit_date, profit_pct,
                         duration_hours, exit_reason, regime, rsi_bucket, macd_signal,
                         ema_alignment, adx_bucket, volume_bucket, fng_bucket, atr_bucket,
                         entry_price, indicators_json)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        trade.get('pair'),
                        trade.get('strategy'),
                        trade.get('direction', 'long'),
                        trade.get('entry_date'),
                        trade.get('exit_date'),
                        trade.get('profit_pct', 0.0),
                        trade.get('duration_hours'),
                        trade.get('exit_reason'),
                        trade.get('regime', 'unknown'),
                        trade.get('rsi_bucket'),
                        trade.get('macd_signal'),
                        trade.get('ema_alignment'),
                        trade.get('adx_bucket'),
                        trade.get('volume_bucket'),
                        trade.get('fng_bucket'),
                        trade.get('atr_bucket'),
                        trade.get('entry_price'),
                        trade.get('indicators_json'),
                    ))
                    inserted += 1
                conn.commit()
                logger.info(f"[PatternStatStore] Ingested {inserted} trades ({skipped} duplicates skipped).")
        except Exception as e:
            logger.error(f"[PatternStatStore] Batch ingest failed: {e}")

    # ── Querying ──────────────────────────────────────────────

    def query(self, pair: Optional[str] = None, regime: Optional[str] = None,
              rsi_bucket: Optional[str] = None, macd_signal: Optional[str] = None,
              ema_alignment: Optional[str] = None, direction: Optional[str] = None,
              min_trades: int = 10) -> Dict[str, Any]:
        """
        Query historical pattern statistics with flexible filtering.

        Returns:
            {
                "matching_trades": int,
                "win_rate": float,
                "avg_profit_pct": float,
                "avg_duration_hours": float,
                "max_drawdown_pct": float,
                "profit_factor": float,
                "regime_breakdown": {regime: {"n": int, "win_rate": float}},
                "exit_reason_dist": {reason: count},
            }
        """
        conditions = []
        params = []

        if pair:
            conditions.append("pair = ?")
            params.append(pair)
        if regime:
            conditions.append("regime = ?")
            params.append(regime)
        if rsi_bucket:
            conditions.append("rsi_bucket = ?")
            params.append(rsi_bucket)
        if macd_signal:
            conditions.append("macd_signal = ?")
            params.append(macd_signal)
        if ema_alignment:
            conditions.append("ema_alignment = ?")
            params.append(ema_alignment)
        if direction:
            conditions.append("direction = ?")
            params.append(direction)

        where_clause = " AND ".join(conditions) if conditions else "1=1"

        try:
            with self._get_conn() as conn:
                # Main statistics
                rows = conn.execute(f"""
                    SELECT profit_pct, duration_hours, exit_reason, regime
                    FROM pattern_trades
                    WHERE {where_clause}
                """, params).fetchall()

                if len(rows) < min_trades:
                    return {"matching_trades": len(rows), "insufficient_data": True}

                # Graduated confidence: more trades = more reliable results
                # Bailey & Lopez de Prado: min 30 for basic stats, 100+ for reliable
                data_quality = "low" if len(rows) < 30 else ("medium" if len(rows) < 100 else "high")

                profits = [float(r['profit_pct']) for r in rows]
                wins = [p for p in profits if p > 0]
                losses = [p for p in profits if p <= 0]

                durations = [float(r['duration_hours']) for r in rows if r['duration_hours']]

                # Regime breakdown
                regime_data = {}
                for r in rows:
                    rg = r['regime'] or 'unknown'
                    if rg not in regime_data:
                        regime_data[rg] = {"total": 0, "wins": 0}
                    regime_data[rg]["total"] += 1
                    if float(r['profit_pct']) > 0:
                        regime_data[rg]["wins"] += 1

                regime_breakdown = {
                    rg: {"n": d["total"], "win_rate": round(d["wins"] / d["total"], 3) if d["total"] > 0 else 0}
                    for rg, d in regime_data.items()
                }

                # Exit reason distribution
                exit_reasons = {}
                for r in rows:
                    reason = r['exit_reason'] or 'unknown'
                    exit_reasons[reason] = exit_reasons.get(reason, 0) + 1

                gross_profit = sum(wins) if wins else 0
                gross_loss = abs(sum(losses)) if losses else 0.001

                return {
                    "matching_trades": len(rows),
                    "data_quality": data_quality,
                    "win_rate": round(len(wins) / len(rows), 3) if rows else 0,
                    "avg_profit_pct": round(sum(profits) / len(profits), 3) if profits else 0,
                    "avg_duration_hours": round(sum(durations) / len(durations), 1) if durations else 0,
                    "max_drawdown_pct": round(min(profits), 2) if profits else 0,
                    "best_trade_pct": round(max(profits), 2) if profits else 0,
                    "profit_factor": round(gross_profit / gross_loss, 2),
                    "regime_breakdown": regime_breakdown,
                    "exit_reason_dist": exit_reasons,
                }

        except Exception as e:
            logger.error(f"[PatternStatStore] Query failed: {e}")
            return {"matching_trades": 0, "error": str(e)}

    def get_pair_summary(self, pair: str) -> Dict[str, Any]:
        """Get overall statistics for a specific pair across all conditions."""
        return self.query(pair=pair, min_trades=10)

    def get_regime_stats(self, pair: str, regime: str) -> Dict[str, Any]:
        """Get statistics for a pair in a specific regime."""
        return self.query(pair=pair, regime=regime, min_trades=10)

    # ── Level 4: Temporal k-NN ──────────────────────────────

    # Bucket → numeric feature mapping (normalized 0-1)
    _BUCKET_NUMERICS = {
        "rsi_bucket": {"oversold": 0.15, "low": 0.35, "neutral": 0.50, "high": 0.65, "overbought": 0.85},
        "macd_signal": {"strong_bearish": 0.10, "weak_bearish": 0.35, "weak_bullish": 0.65, "strong_bullish": 0.90},
        "regime": {"ranging": 0.20, "trending_bear": 0.30, "transitional": 0.45, "high_volatility": 0.50, "trending_bull": 0.75},
        "volume_bucket": {"low": 0.20, "normal": 0.50, "high": 0.80},
        "fng_bucket": {"extreme_fear": 0.10, "fear": 0.30, "neutral": 0.50, "greed": 0.70, "extreme_greed": 0.90},
        "ema_alignment": {"full_bearish": 0.15, "below_200": 0.35, "above_200": 0.65, "full_bullish": 0.85},
    }

    def _bucket_to_vector(self, row) -> List[float]:
        """Convert a trade's bucket columns into a numeric feature vector."""
        vec = []
        for col, mapping in self._BUCKET_NUMERICS.items():
            val = row[col] if col in row.keys() else None
            vec.append(mapping.get(val, 0.5))  # Unknown → 0.5 (neutral)
        return vec

    def temporal_knn(self, current_features: Dict[str, str], k: int = 0,
                     pair: Optional[str] = None) -> Dict[str, Any]:
        """
        Level 4: Find k most similar historical trade conditions using feature distance.

        k=0 means adaptive: k = max(5, int(sqrt(N))) — scales with data size.
        Uses Mahalanobis distance when N>30 (academic consensus for finance),
        falls back to Euclidean when N<=30 (covariance matrix would be singular).

        References:
          - k=sqrt(N) rule: Lall & Sharma (1996)
          - Mahalanobis for finance: De Maesschalck et al. (2000)

        current_features: {"rsi_bucket": "oversold", "regime": "trending_bull", ...}
        Returns: {"k_neighbors": [...], "knn_win_rate": float, "knn_avg_pnl": float}
        """
        import math

        # Build current feature vector
        current_vec = []
        for col, mapping in self._BUCKET_NUMERICS.items():
            val = current_features.get(col)
            current_vec.append(mapping.get(val, 0.5))

        try:
            with self._get_conn() as conn:
                conditions = ["1=1"]
                params = []
                if pair:
                    conditions.append("pair = ?")
                    params.append(pair)

                rows = conn.execute(f"""
                    SELECT pair, direction, profit_pct, duration_hours, exit_reason,
                           rsi_bucket, macd_signal, regime, volume_bucket, fng_bucket, ema_alignment
                    FROM pattern_trades
                    WHERE {' AND '.join(conditions)}
                """, params).fetchall()

                # Adaptive k: sqrt(N), clamped to [5, 50]
                n = len(rows)
                if k <= 0:
                    k = max(5, min(50, int(math.sqrt(n))))

                if n < k:
                    return {"k_neighbors": [], "insufficient_data": True, "total_candidates": n}

                # Build all feature vectors for distance computation
                all_vecs = []
                for row in rows:
                    all_vecs.append(self._bucket_to_vector(row))

                # Mahalanobis distance when N>30, Euclidean otherwise
                # Mahalanobis accounts for feature correlations (academic consensus)
                use_mahalanobis = n > 30
                inv_cov = None

                if use_mahalanobis:
                    try:
                        import numpy as np
                        mat = np.array(all_vecs)
                        cov = np.cov(mat, rowvar=False)
                        # Regularize: add small diagonal to prevent singularity
                        cov += np.eye(cov.shape[0]) * 1e-6
                        inv_cov = np.linalg.inv(cov)
                    except Exception:
                        use_mahalanobis = False  # Fallback to Euclidean

                # Compute distances
                scored = []
                for i, row in enumerate(rows):
                    row_vec = all_vecs[i]
                    if use_mahalanobis and inv_cov is not None:
                        import numpy as np
                        diff = np.array(current_vec) - np.array(row_vec)
                        dist = float(np.sqrt(diff @ inv_cov @ diff))
                    else:
                        # Euclidean distance
                        dist = sum((a - b) ** 2 for a, b in zip(current_vec, row_vec)) ** 0.5
                    scored.append((dist, dict(row)))

                # Sort by distance (closest first)
                scored.sort(key=lambda x: x[0])
                neighbors = scored[:k]

                profits = [n[1]["profit_pct"] for n in neighbors]
                wins = sum(1 for p in profits if p > 0)

                result = {
                    "k_neighbors": [
                        {"distance": round(d, 3), "pair": n["pair"], "profit_pct": n["profit_pct"],
                         "direction": n["direction"], "regime": n["regime"], "exit_reason": n["exit_reason"]}
                        for d, n in neighbors
                    ],
                    "knn_win_rate": round(wins / len(neighbors), 3) if neighbors else 0,
                    "knn_avg_pnl": round(sum(profits) / len(profits), 3) if profits else 0,
                    "knn_best": round(max(profits), 2) if profits else 0,
                    "knn_worst": round(min(profits), 2) if profits else 0,
                    "avg_distance": round(sum(d for d, _ in neighbors) / len(neighbors), 3) if neighbors else 0,
                    "total_candidates": len(rows),
                }

                logger.info(f"[PatternStatStore:kNN] k={k}, win_rate={result['knn_win_rate']:.0%}, "
                            f"avg_pnl={result['knn_avg_pnl']:+.2f}%, avg_dist={result['avg_distance']:.3f}")
                return result

        except Exception as e:
            logger.error(f"[PatternStatStore:kNN] Query failed: {e}")
            return {"k_neighbors": [], "error": str(e)}

    def format_knn_for_prompt(self, current_features: Dict[str, str], k: int = 10,
                               pair: Optional[str] = None) -> str:
        """Format k-NN results as prompt-injectable text."""
        result = self.temporal_knn(current_features, k=k, pair=pair)
        if result.get("insufficient_data") or not result.get("k_neighbors"):
            return ""

        wr = result["knn_win_rate"]
        avg = result["knn_avg_pnl"]
        best = result["knn_best"]
        worst = result["knn_worst"]
        dist = result["avg_distance"]

        lines = [
            f"=== TEMPORAL k-NN: {k} MOST SIMILAR HISTORICAL STATES ===",
            f"Win Rate: {wr:.0%} | Avg P&L: {avg:+.2f}% | Best: {best:+.2f}% | Worst: {worst:+.2f}%",
            f"Avg Feature Distance: {dist:.3f} (lower = more similar)",
        ]

        # Show top 3 closest neighbors
        for i, n in enumerate(result["k_neighbors"][:3], 1):
            lines.append(f"  #{i} {n['pair']} {n['direction']} → {n['profit_pct']:+.2f}% ({n['exit_reason']}) [dist={n['distance']:.3f}]")

        lines.append("These are the historically most similar market states. Weight your signal accordingly.")
        return "\n".join(lines)

    # ── Level 4: Multi-Strategy Ensemble ──────────────────

    def ensemble_vote(self, pair: Optional[str] = None, regime: Optional[str] = None,
                      rsi_bucket: Optional[str] = None, min_trades_per_strategy: int = 10) -> Dict[str, Any]:
        """
        Level 4: For given conditions, check what MULTIPLE strategies did historically.
        Returns per-strategy results + consensus.
        """
        try:
            with self._get_conn() as conn:
                conditions = ["strategy IS NOT NULL AND strategy != 'unknown'"]
                params = []
                if pair:
                    conditions.append("pair = ?")
                    params.append(pair)
                if regime:
                    conditions.append("regime = ?")
                    params.append(regime)
                if rsi_bucket:
                    conditions.append("rsi_bucket = ?")
                    params.append(rsi_bucket)

                rows = conn.execute(f"""
                    SELECT strategy, direction, profit_pct
                    FROM pattern_trades
                    WHERE {' AND '.join(conditions)}
                """, params).fetchall()

                if not rows:
                    return {"strategies": {}, "consensus": "NEUTRAL", "consensus_strength": 0}

                # Group by strategy
                from collections import defaultdict
                strat_data = defaultdict(list)
                for r in rows:
                    strat_data[r["strategy"]].append({
                        "direction": r["direction"],
                        "profit_pct": r["profit_pct"]
                    })

                strategies = {}
                votes_long = 0
                votes_short = 0
                total_strategies = 0

                for strat, trades in strat_data.items():
                    if len(trades) < min_trades_per_strategy:
                        continue

                    total_strategies += 1
                    profits = [t["profit_pct"] for t in trades]
                    wins = sum(1 for p in profits if p > 0)
                    wr = wins / len(profits)
                    avg = sum(profits) / len(profits)

                    # Strategy "votes" based on its historical performance
                    long_trades = [t for t in trades if t["direction"] == "long"]
                    short_trades = [t for t in trades if t["direction"] == "short"]

                    long_wr = sum(1 for t in long_trades if t["profit_pct"] > 0) / max(len(long_trades), 1)
                    short_wr = sum(1 for t in short_trades if t["profit_pct"] > 0) / max(len(short_trades), 1)

                    if long_wr > short_wr and long_wr > 0.50:
                        votes_long += 1
                    elif short_wr > long_wr and short_wr > 0.50:
                        votes_short += 1

                    strategies[strat] = {
                        "n": len(trades), "win_rate": round(wr, 3), "avg_pnl": round(avg, 2),
                        "long_wr": round(long_wr, 3), "short_wr": round(short_wr, 3),
                    }

                # Determine consensus
                if total_strategies == 0:
                    consensus = "NEUTRAL"
                    strength = 0
                elif votes_long > votes_short:
                    consensus = "LONG"
                    strength = round(votes_long / total_strategies, 2)
                elif votes_short > votes_long:
                    consensus = "SHORT"
                    strength = round(votes_short / total_strategies, 2)
                else:
                    consensus = "NEUTRAL"
                    strength = 0

                logger.info(f"[PatternStatStore:Ensemble] {total_strategies} strategies → {consensus} "
                            f"(strength={strength}, long={votes_long}, short={votes_short})")

                return {
                    "strategies": strategies,
                    "consensus": consensus,
                    "consensus_strength": strength,
                    "votes_long": votes_long,
                    "votes_short": votes_short,
                    "total_strategies": total_strategies,
                }

        except Exception as e:
            logger.error(f"[PatternStatStore:Ensemble] Query failed: {e}")
            return {"strategies": {}, "consensus": "NEUTRAL", "error": str(e)}

    def format_for_prompt(self, pair: str, regime: Optional[str] = None,
                          rsi_bucket: Optional[str] = None,
                          macd_signal: Optional[str] = None) -> str:
        """
        Format query results as a text block suitable for injection into LLM prompts.
        Used by rag_graph.py to inject backtest context into MADAM debate.
        """
        stats = self.query(
            pair=pair, regime=regime,
            rsi_bucket=rsi_bucket, macd_signal=macd_signal
        )

        if stats.get("insufficient_data") or stats.get("matching_trades", 0) < 3:
            return ""

        n = stats["matching_trades"]
        wr = stats["win_rate"]
        avg_pnl = stats["avg_profit_pct"]
        pf = stats["profit_factor"]
        avg_dur = stats["avg_duration_hours"]

        lines = [
            f"=== BACKTEST HISTORICAL BASELINE (n={n} trades) ===",
            f"Win Rate: {wr:.1%} | Avg P&L: {avg_pnl:+.2f}% | Profit Factor: {pf:.2f} | Avg Duration: {avg_dur:.0f}h",
        ]

        # Add regime breakdown if multiple regimes
        rb = stats.get("regime_breakdown", {})
        if len(rb) > 1:
            regime_parts = [f"{rg}: {d['win_rate']:.0%} (n={d['n']})" for rg, d in rb.items()]
            lines.append(f"By Regime: {' | '.join(regime_parts)}")

        if regime:
            regime_stats = rb.get(regime, {})
            if regime_stats:
                lines.append(f"Current Regime ({regime}): {regime_stats.get('win_rate', 0):.0%} win rate (n={regime_stats.get('n', 0)})")

        lines.append("Use these as prior probabilities. Signal should align or explain deviation.")

        return "\n".join(lines)

    # ── Bucketing Helpers ──────────────────────────────────────

    @staticmethod
    def classify_rsi(rsi: float) -> str:
        if rsi < 30:
            return "oversold"
        elif rsi < 45:
            return "low"
        elif rsi <= 55:
            return "neutral"
        elif rsi <= 70:
            return "high"
        else:
            return "overbought"

    @staticmethod
    def classify_macd(macd_hist: float) -> str:
        if macd_hist > 0.5:
            return "strong_bullish"
        elif macd_hist > 0:
            return "weak_bullish"
        elif macd_hist > -0.5:
            return "weak_bearish"
        else:
            return "strong_bearish"

    @staticmethod
    def classify_regime(adx: float, atr_ratio: float = 1.0, price: float = None, ema200: float = None) -> str:
        """
        Classify market regime. Labels MUST match RegimeClassifier output:
        trending_bull, trending_bear, ranging, high_volatility, transitional.
        """
        if atr_ratio > 2.0:
            return "high_volatility"
        elif adx < 20:
            return "ranging"
        elif adx < 25:
            return "transitional"
        elif adx >= 25:
            # Determine trend direction if price/EMA available
            if price is not None and ema200 is not None:
                return "trending_bull" if price > ema200 else "trending_bear"
            return "trending_bull"  # Optimistic default when no price data
        return "transitional"

    @staticmethod
    def classify_ema(price: float, ema20: float, ema50: float, ema200: float) -> str:
        if price > ema20 > ema50 > ema200:
            return "full_bullish"
        elif price < ema20 < ema50 < ema200:
            return "full_bearish"
        elif price > ema200:
            return "above_200"
        else:
            return "below_200"

    @staticmethod
    def classify_volume(volume_ratio: float) -> str:
        """volume_ratio = current_volume / sma_volume_20"""
        if volume_ratio > 1.5:
            return "high"
        elif volume_ratio > 0.8:
            return "normal"
        else:
            return "low"

    @staticmethod
    def classify_fng(fng: int) -> str:
        if fng < 20:
            return "extreme_fear"
        elif fng < 40:
            return "fear"
        elif fng <= 60:
            return "neutral"
        elif fng <= 80:
            return "greed"
        else:
            return "extreme_greed"

    def get_total_trades(self) -> int:
        """Return total number of pattern trades stored."""
        try:
            with self._get_conn() as conn:
                row = conn.execute("SELECT COUNT(*) as cnt FROM pattern_trades").fetchone()
                return row['cnt'] if row else 0
        except Exception:
            return 0
