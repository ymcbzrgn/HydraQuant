"""
Phase 20: Opportunity Scanner — Wide LLM-free screening of 200+ pairs.

Instead of analyzing all 40 pairs equally (wasting LLM calls on low-opportunity pairs),
scan wide and select only the best opportunities for deep analysis.

Professional quant approach: SCAN WIDE → SELECT NARROW → ANALYZE DEEP

5 Opportunity Types:
  1. momentum_breakout: EMA aligned + ADX>25 + volume>1.5x
  2. mean_reversion: RSI extreme + BB touch + F&G extreme
  3. funding_contrarian: |funding_rate| > 0.05% + L/S extreme
  4. regime_shift: ADX transitioning from <20 to >25
  5. volume_anomaly: Volume>3x avg without proportional price move

Usage:
    from opportunity_scanner import OpportunityScanner
    scanner = OpportunityScanner()
    # With strategy dataframe provider:
    results = scanner.scan_pairs(pairs, dp=self.dp, timeframe="1h", top_n=20)
    # Standalone from DB:
    results = scanner.scan_pairs_from_db(pairs, top_n=20)
"""

import os
import sys
import json
import sqlite3
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone

sys.path.append(os.path.dirname(__file__))

from ai_config import AI_DB_PATH

# Phase 24: Neural Organism — adaptive parameters
try:
    from neural_organism import _p
except ImportError:
    def _p(param_id, fallback=0.5, regime="_global"):
        return fallback

logger = logging.getLogger(__name__)


class OpportunityScanner:
    """
    LLM-free wide screening engine.
    Scores each pair on 5 opportunity types (0-100 each), returns top N.
    """

    OPPORTUNITY_TYPES = {
        "momentum_breakout": {"weight": 0.25, "description": "EMA aligned + ADX>25 + volume spike"},
        "mean_reversion": {"weight": 0.25, "description": "RSI extreme + BB touch + F&G extreme"},
        "funding_contrarian": {"weight": 0.20, "description": "Extreme funding rate + L/S ratio"},
        "regime_shift": {"weight": 0.15, "description": "ADX transitioning from ranging to trending"},
        "volume_anomaly": {"weight": 0.15, "description": "Volume>3x avg without price move"},
    }

    def __init__(self, db_path: str = AI_DB_PATH):
        self.db_path = db_path
        self._market_data = None
        self._score_cache: Dict[str, Dict] = {}  # pair → {score, timestamp}

        try:
            from market_data_fetcher import MarketDataFetcher
            self._market_data = MarketDataFetcher(db_path=db_path)
        except Exception as e:
            logger.debug(f"[OpportunityScanner:Init] MarketDataFetcher unavailable: {e}")

    # ═══════════════════════════════════════════════════════════
    # PUBLIC API
    # ═══════════════════════════════════════════════════════════

    def scan_pairs(self, pairs: List[str], dp=None, timeframe: str = "1h",
                   top_n: int = 20) -> List[Dict]:
        """
        Scan pairs using strategy's dataframe provider.
        Returns top_n pairs sorted by composite opportunity score (descending).
        """
        if not pairs:
            return []

        results = []
        fng = self._get_fear_greed()

        for pair in pairs:
            try:
                # Get analyzed dataframe from strategy
                td = {}
                if dp:
                    try:
                        df, _ = dp.get_analyzed_dataframe(pair, timeframe)
                        if df is not None and len(df) > 5:
                            td = self._extract_indicators(df, pair)
                    except Exception:
                        pass

                if not td.get("current_price"):
                    continue

                # Get derivatives data
                derivatives = None
                if self._market_data:
                    try:
                        derivatives = self._market_data.get_latest_derivatives(pair)
                    except Exception:
                        pass

                # Score all opportunity types
                scores = self._score_pair(td, derivatives, fng)
                composite = self._compute_composite(scores)

                # Find top opportunity type
                top_type = max(scores, key=scores.get) if scores else "none"

                result = {
                    "pair": pair,
                    "composite_score": round(composite, 1),
                    "top_type": top_type,
                    "momentum_score": round(scores.get("momentum_breakout", 0), 1),
                    "reversion_score": round(scores.get("mean_reversion", 0), 1),
                    "funding_score": round(scores.get("funding_contrarian", 0), 1),
                    "regime_shift_score": round(scores.get("regime_shift", 0), 1),
                    "volume_anomaly_score": round(scores.get("volume_anomaly", 0), 1),
                }
                results.append(result)

                # Cache for custom_stake_amount lookup
                self._score_cache[pair] = {
                    "score": composite,
                    "timestamp": datetime.now(tz=timezone.utc)
                }

            except Exception as e:
                logger.debug(f"[OpportunityScanner] {pair} scan failed: {e}")

        # Sort by composite score descending
        results.sort(key=lambda x: x["composite_score"], reverse=True)

        # Persist to DB
        self._persist_scores(results)

        top = results[:top_n]
        if results:
            top_summary = ", ".join(f"{r['pair']}({r['composite_score']})" for r in top[:5])
            logger.info(f"[OpportunityScanner] Scanned {len(results)} pairs → top {len(top)}: {top_summary}")

        return top

    def scan_pairs_from_db(self, pairs: List[str] = None, top_n: int = 20) -> List[Dict]:
        """
        Scan using cached data in SQLite (for scheduler job, no dp needed).
        Falls back to latest opportunity_scores if available.
        """
        try:
            conn = sqlite3.connect(self.db_path, timeout=10)
            conn.row_factory = sqlite3.Row
            if pairs:
                placeholders = ",".join("?" for _ in pairs)
                rows = conn.execute(f"""
                    SELECT * FROM opportunity_scores
                    WHERE pair IN ({placeholders})
                    AND id IN (SELECT MAX(id) FROM opportunity_scores GROUP BY pair)
                    ORDER BY composite_score DESC LIMIT ?
                """, (*pairs, top_n)).fetchall()
            else:
                rows = conn.execute("""
                    SELECT * FROM opportunity_scores
                    WHERE id IN (SELECT MAX(id) FROM opportunity_scores GROUP BY pair)
                    ORDER BY composite_score DESC LIMIT ?
                """, (top_n,)).fetchall()
            conn.close()
            return [dict(r) for r in rows]
        except Exception as e:
            logger.debug(f"[OpportunityScanner] DB scan failed: {e}")
            return []

    def get_cached_score(self, pair: str) -> Optional[float]:
        """Get latest cached composite score for a pair (used by custom_stake_amount)."""
        cached = self._score_cache.get(pair)
        if cached:
            age_seconds = (datetime.now(tz=timezone.utc) - cached["timestamp"]).total_seconds()
            if age_seconds < 3600:  # Valid for 1 hour
                return cached["score"]

        # Fallback to DB
        try:
            conn = sqlite3.connect(self.db_path, timeout=10)
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT composite_score FROM opportunity_scores "
                "WHERE pair = ? ORDER BY timestamp DESC LIMIT 1", (pair,)
            ).fetchone()
            conn.close()
            if row:
                score = float(row["composite_score"])
                self._score_cache[pair] = {
                    "score": score,
                    "timestamp": datetime.now(tz=timezone.utc)
                }
                return score
        except Exception:
            pass
        return None

    # ═══════════════════════════════════════════════════════════
    # SCORING LOGIC
    # ═══════════════════════════════════════════════════════════

    def _score_pair(self, td: dict, derivatives: Optional[Dict],
                    fng: Optional[int]) -> Dict[str, float]:
        """Score a pair on all 5 opportunity types (each 0-100)."""
        scores = {
            "momentum_breakout": self._score_momentum(td),
            "mean_reversion": self._score_reversion(td, fng),
            "funding_contrarian": self._score_funding(derivatives),
            "regime_shift": self._score_regime_shift(td),
            "volume_anomaly": self._score_volume_anomaly(td),
        }
        # Cross-sectional return for composite tiebreaker
        ret7d = td.get("return_7d_pct", 0) or 0
        scores["_return_7d_abs"] = abs(ret7d)
        return scores

    def _score_momentum(self, td: dict) -> float:
        """Momentum breakout: EMA aligned + ADX>25 + volume>1.5x."""
        score = 0.0
        price = td.get("current_price", 0)

        # EMA alignment (0-40 points)
        ema9 = td.get("ema_9")
        ema20 = td.get("ema_20")
        ema50 = td.get("ema_50")
        ema200 = td.get("ema_200")

        if price and ema9 and ema20 and ema50 and ema200:
            p, e9, e20, e50, e200 = float(price), float(ema9), float(ema20), float(ema50), float(ema200)
            if p > e9 > e20 > e50 > e200:
                score += _p("opp.momentum.ema_full_bull", 40)
            elif p > e20 > e50 > e200:
                score += _p("opp.momentum.ema_partial_1", 30)
            elif p > e50 > e200:
                score += _p("opp.momentum.ema_partial_2", 20)
            elif p > e200:
                score += _p("opp.momentum.ema_above_200", 10)
            elif p < e9 < e20 < e50 < e200:
                score += _p("opp.momentum.ema_full_bear", 35)
            elif p < e20 < e50 < e200:
                score += _p("opp.momentum.ema_partial_bear", 25)

        # ADX strength (Phase 24: adaptive)
        adx = td.get("adx_14") or td.get("adx")
        if adx:
            adx = float(adx)
            if adx > _p("opp.momentum.adx_strong", 35):
                score += _p("opp.momentum.adx_score_strong", 30)
            elif adx > 25:
                score += _p("opp.momentum.adx_score_med", 20)
            elif adx > 20:
                score += _p("opp.momentum.adx_score_low", 10)

        # Volume confirmation (Phase 24: adaptive)
        vol = td.get("volume", {}) if isinstance(td.get("volume"), dict) else {}
        vol_ratio = vol.get("ratio", 1.0)
        if isinstance(vol_ratio, (int, float)):
            if vol_ratio > _p("opp.momentum.vol_extreme", 2.0):
                score += _p("opp.momentum.vol_score_hi", 30)
            elif vol_ratio > 1.5:
                score += _p("opp.momentum.vol_score_med", 20)
            elif vol_ratio > 1.2:
                score += _p("opp.momentum.vol_score_low", 10)

        return min(100.0, score)

    def _score_reversion(self, td: dict, fng: Optional[int]) -> float:
        """Mean reversion: RSI extreme + BB touch + F&G extreme."""
        score = 0.0
        price = td.get("current_price", 0)

        # RSI extreme (0-35 points)
        rsi = td.get("rsi_14") or td.get("rsi")
        if rsi:
            rsi = float(rsi)
            if rsi < 20 or rsi > 80:
                score += _p("opp.reversion.rsi_extreme", 35)
            elif rsi < 25 or rsi > 75:
                score += _p("opp.reversion.rsi_moderate", 25)
            elif rsi < 30 or rsi > 70:
                score += _p("opp.reversion.rsi_mild", 15)

        # Bollinger Band touch (0-30 points)
        bb_lower = td.get("bb_lower")
        bb_upper = td.get("bb_upper")
        if price and bb_lower and bb_upper:
            p, bbl, bbu = float(price), float(bb_lower), float(bb_upper)
            if p <= bbl:
                score += _p("opp.reversion.bb_touch", 30)
            elif p >= bbu:
                score += _p("opp.reversion.bb_touch", 30)
            else:
                bb_range = bbu - bbl
                if bb_range > 0:
                    dist_lower = (p - bbl) / bb_range
                    dist_upper = (bbu - p) / bb_range
                    if dist_lower < 0.10 or dist_upper < 0.10:
                        score += _p("opp.reversion.bb_near", 20)

        # Fear & Greed extreme (0-35 points)
        if fng is not None:
            fng_ext = int(_p("opp.reversion.fng_extreme_thr", 15))
            if fng < fng_ext or fng > (100 - fng_ext):
                score += _p("opp.reversion.fng_extreme", 35)
            elif fng < 25 or fng > 75:
                score += _p("opp.reversion.fng_moderate", 20)
            elif fng < 30 or fng > 70:
                score += _p("opp.reversion.fng_mild", 10)

        return min(100.0, score)

    def _score_funding(self, derivatives: Optional[Dict]) -> float:
        """Funding contrarian: extreme funding rate + L/S ratio."""
        if not derivatives:
            return 0.0

        score = 0.0

        # Funding rate extremity (0-60 points)
        fr = derivatives.get("funding_rate")
        if fr is not None:
            fr = abs(float(fr))
            if fr > 0.001:
                score += _p("opp.funding.fr_very_extreme", 60)
            elif fr > 0.0005:
                score += _p("opp.funding.fr_extreme", 40)
            elif fr > 0.0003:
                score += _p("opp.funding.fr_notable", 20)

        # L/S ratio extremity (Phase 24: adaptive)
        ls = derivatives.get("long_short_ratio")
        if ls is not None:
            ls = float(ls)
            deviation = abs(ls - 1.0)
            if deviation > 0.8:
                score += _p("opp.funding.ls_very_unbalanced", 40)
            elif deviation > 0.5:
                score += _p("opp.funding.ls_unbalanced", 25)
            elif deviation > 0.3:
                score += _p("opp.funding.ls_skewed", 10)

        return min(100.0, score)

    def _score_regime_shift(self, td: dict) -> float:
        """Regime shift: ADX transitioning from ranging to trending."""
        score = 0.0

        adx = td.get("adx_14") or td.get("adx")
        if not adx:
            return 0.0

        adx = float(adx)

        # ADX in the transition zone (20-30) = potential breakout (Phase 25: adaptive scores)
        if 20 <= adx <= 30:
            score += _p("opp.regime.sweet_spot", 40)

            macd_hist = td.get("macd_histogram") or td.get("macd_hist")
            if macd_hist is not None:
                mh = float(macd_hist)
                if abs(mh) < 0.5:
                    score += _p("opp.regime.macd_cross", 20)

            price = td.get("current_price", 0)
            ema50 = td.get("ema_50") or td.get("ema50")
            if price and ema50:
                dist_pct = abs(float(price) - float(ema50)) / float(ema50) * 100
                if dist_pct < 1.0:
                    score += _p("opp.regime.ema_close", 25)
                elif dist_pct < 2.0:
                    score += _p("opp.regime.ema_near", 15)

            vol = td.get("volume", {}) if isinstance(td.get("volume"), dict) else {}
            if vol.get("trend") == "rising":
                score += _p("opp.regime.vol_rising", 15)

        elif adx < 20:
            score += _p("opp.regime.pre_transition", 10)

        return min(100.0, score)

    def _score_volume_anomaly(self, td: dict) -> float:
        """Volume anomaly: Volume>3x avg without proportional price move."""
        score = 0.0

        vol = td.get("volume", {}) if isinstance(td.get("volume"), dict) else {}
        vol_ratio = vol.get("ratio", 1.0)
        if not isinstance(vol_ratio, (int, float)):
            return 0.0

        price_change_1h = td.get("price_change_1h_pct", 0) or 0

        # High volume (Phase 25: adaptive scoring)
        if vol_ratio > 5.0:
            score += _p("opp.volume.extreme_score", 50)
        elif vol_ratio > 3.0:
            score += _p("opp.volume.high_score", 35)
        elif vol_ratio > 2.0:
            score += _p("opp.volume.moderate_score", 20)

        # Low price move despite high volume = accumulation/distribution
        if vol_ratio > 2.0 and abs(price_change_1h) < 1.0:
            score += _p("opp.volume.stealth_extreme", 50)
        elif vol_ratio > 2.0 and abs(price_change_1h) < 2.0:
            score += _p("opp.volume.stealth_high", 30)
        elif vol_ratio > 1.5 and abs(price_change_1h) < 0.5:
            score += _p("opp.volume.stealth_moderate", 20)

        return min(100.0, score)

    def _compute_composite(self, scores: Dict[str, float]) -> float:
        """Compute weighted composite score from individual opportunity scores (Phase 24: adaptive weights).
        Includes cross-sectional return as a tiebreaker (+/- up to 10 points)."""
        # Phase 24: Read weights from Neural Organism
        adaptive_weights = {
            "momentum_breakout": _p("opp.weight.momentum", 0.25),
            "mean_reversion": _p("opp.weight.reversion", 0.25),
            "funding_contrarian": _p("opp.weight.funding", 0.20),
            "regime_shift": _p("opp.weight.regime_shift", 0.15),
            "volume_anomaly": _p("opp.weight.volume", 0.15),
        }
        # Normalize weights to sum=1.0
        w_total = sum(adaptive_weights.values())
        if w_total > 0:
            adaptive_weights = {k: v / w_total for k, v in adaptive_weights.items()}

        composite = 0.0
        for opp_type in self.OPPORTUNITY_TYPES:
            composite += scores.get(opp_type, 0) * adaptive_weights.get(opp_type, 0.2)
        # Cross-sectional momentum tiebreaker: absolute 7d return adds differentiation
        ret7d = scores.get("_return_7d_abs", 0)
        if ret7d > 10:
            composite += 10  # Strong mover (>10% in 7d)
        elif ret7d > 5:
            composite += 6
        elif ret7d > 2:
            composite += 3
        return composite

    # ═══════════════════════════════════════════════════════════
    # HELPERS
    # ═══════════════════════════════════════════════════════════

    def _extract_indicators(self, df, pair: str) -> dict:
        """Extract key indicators from a pandas dataframe (minimal, for scanning).
        Handles multiple column naming conventions (ema_9 vs ema9 vs EMA_9)."""
        try:
            last = df.iloc[-1]
            prev = df.iloc[-2] if len(df) > 1 else last

            def _safe_float(row, *col_names):
                """Try multiple column name variants, return float or None."""
                for col in col_names:
                    if col in row:
                        v = row[col]
                        if v == v:  # NaN check
                            return float(v)
                return None

            td = {
                "current_price": float(last.get("close", 0)),
                "rsi_14": _safe_float(last, "rsi", "rsi_14", "RSI"),
                "adx_14": _safe_float(last, "adx", "adx_14", "ADX"),
                "macd_histogram": _safe_float(last, "macdhist", "macd_histogram", "MACD_histogram", "macd_hist"),
                "ema_9": _safe_float(last, "ema_9", "ema9", "EMA_9"),
                "ema_20": _safe_float(last, "ema_20", "ema20", "EMA_20"),
                "ema_50": _safe_float(last, "ema_50", "ema50", "EMA_50"),
                "ema_200": _safe_float(last, "ema_200", "ema200", "EMA_200"),
                "bb_lower": _safe_float(last, "bb_lower", "bb_lowerband", "BBL_20_2.0"),
                "bb_upper": _safe_float(last, "bb_upper", "bb_upperband", "BBU_20_2.0"),
                "atr_14": _safe_float(last, "atr", "atr_14", "ATR"),
                "price_change_1h_pct": 0.0,
            }

            # Volume ratio — compute from raw volume if no precomputed mean
            if "volume" in last:
                v = float(last["volume"]) if last["volume"] == last["volume"] else 0
                vm = None
                # Try precomputed mean column
                for col in ("volume_mean", "volume_sma_20", "vol_mean"):
                    if col in last and last[col] == last[col] and float(last[col]) > 0:
                        vm = float(last[col])
                        break
                # Compute from dataframe if no precomputed mean
                if vm is None and len(df) >= 20:
                    vm = float(df["volume"].iloc[-20:].mean())
                if vm and vm > 0:
                    ratio = round(v / vm, 2)
                else:
                    ratio = 1.0
                trend = "rising" if v > float(prev.get("volume", 0)) else "stable"
                td["volume"] = {"ratio": ratio, "trend": trend}
            else:
                td["volume"] = {"ratio": 1.0, "trend": "stable"}

            # Price change
            if td["current_price"] and prev.get("close"):
                prev_close = float(prev["close"])
                if prev_close > 0:
                    td["price_change_1h_pct"] = round((td["current_price"] - prev_close) / prev_close * 100, 2)

            # Cross-sectional momentum fallback: 7-day return (if enough data)
            if len(df) >= 168:  # 7 days × 24 candles
                close_7d_ago = float(df["close"].iloc[-168])
                if close_7d_ago > 0:
                    td["return_7d_pct"] = round((td["current_price"] - close_7d_ago) / close_7d_ago * 100, 2)
            elif len(df) >= 24:  # 1 day fallback
                close_1d_ago = float(df["close"].iloc[-24])
                if close_1d_ago > 0:
                    td["return_7d_pct"] = round((td["current_price"] - close_1d_ago) / close_1d_ago * 100, 2)

            return td
        except Exception as e:
            logger.debug(f"[OpportunityScanner] {pair} indicator extraction failed: {e}")
            return {}

    def _get_fear_greed(self) -> Optional[int]:
        """Read latest F&G from DB."""
        try:
            conn = sqlite3.connect(self.db_path, timeout=10)
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT value FROM fear_and_greed ORDER BY timestamp DESC LIMIT 1"
            ).fetchone()
            conn.close()
            return int(row["value"]) if row else None
        except Exception:
            return None

    def _persist_scores(self, results: List[Dict]):
        """Persist scan results to opportunity_scores table."""
        if not results:
            return
        try:
            conn = sqlite3.connect(self.db_path, timeout=10)
            conn.execute("PRAGMA journal_mode=WAL")

            # Ensure table exists
            conn.execute("""
                CREATE TABLE IF NOT EXISTS opportunity_scores (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pair TEXT NOT NULL,
                    composite_score REAL NOT NULL,
                    top_type TEXT,
                    momentum_score REAL,
                    reversion_score REAL,
                    funding_score REAL,
                    regime_shift_score REAL,
                    volume_anomaly_score REAL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_opp_pair_ts "
                        "ON opportunity_scores(pair, timestamp)")

            for r in results:
                conn.execute("""
                    INSERT INTO opportunity_scores
                    (pair, composite_score, top_type, momentum_score, reversion_score,
                     funding_score, regime_shift_score, volume_anomaly_score)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (r["pair"], r["composite_score"], r["top_type"],
                      r.get("momentum_score", 0), r.get("reversion_score", 0),
                      r.get("funding_score", 0), r.get("regime_shift_score", 0),
                      r.get("volume_anomaly_score", 0)))
            conn.commit()
            conn.close()
            logger.debug(f"[OpportunityScanner:Persist] {len(results)} scores saved")
        except Exception as e:
            logger.debug(f"[OpportunityScanner:Persist] Failed: {e}")
