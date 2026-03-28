"""
Phase 20: Cross-Pair Intelligence — Market-wide pattern detection overlay.

Instead of analyzing each pair in isolation, detect market-wide patterns:
- 7/10 top pairs bearish → market-wide weakness → reduce all confidence
- BTC just turned bullish → altcoins typically follow in 4-8h
- >70% of pairs have positive funding → market crowded long → contrarian warning

All LLM-free. Uses Evidence Engine results + derivatives data from SQLite.

Usage:
    from cross_pair_intel import CrossPairIntel
    intel = CrossPairIntel()
    bias = intel.compute_market_bias(evidence_results)
    btc_lead = intel.detect_btc_lead(btc_signal, alt_signals)
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

logger = logging.getLogger(__name__)


class CrossPairIntel:
    """
    Market-wide intelligence overlay. Detects cross-pair patterns.
    All computation is LLM-free — uses SQLite queries + simple aggregation.
    """

    def __init__(self, db_path: str = AI_DB_PATH):
        self.db_path = db_path
        self._latest: Dict[str, Any] = {}

    def _get_conn(self):
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    # ═══════════════════════════════════════════════════════════
    # PUBLIC API
    # ═══════════════════════════════════════════════════════════

    def compute_market_bias(self, evidence_results: List[Dict]) -> Dict[str, Any]:
        """
        Aggregate Evidence Engine signals from top pairs to detect market-wide bias.
        If 7/10 pairs are BEARISH → market-wide bearish → all pairs get -0.10 penalty.
        """
        if not evidence_results:
            return {"bias": "NEUTRAL", "strength": 0.0, "bullish_count": 0,
                    "bearish_count": 0, "neutral_count": 0, "total": 0}

        bullish = 0
        bearish = 0
        neutral = 0
        total_conf_bull = 0.0
        total_conf_bear = 0.0

        for r in evidence_results:
            signal = r.get("signal", "NEUTRAL")
            conf = r.get("confidence", 0)
            if signal == "BULLISH":
                bullish += 1
                total_conf_bull += conf
            elif signal == "BEARISH":
                bearish += 1
                total_conf_bear += conf
            else:
                neutral += 1

        total = len(evidence_results)

        # Determine market bias
        if total == 0:
            bias = "NEUTRAL"
            strength = 0.0
        elif bullish / total >= 0.70:
            bias = "BULLISH"
            strength = bullish / total
        elif bearish / total >= 0.70:
            bias = "BEARISH"
            strength = bearish / total
        elif bullish > bearish and bullish / total >= 0.55:
            bias = "MILD_BULLISH"
            strength = bullish / total * 0.5
        elif bearish > bullish and bearish / total >= 0.55:
            bias = "MILD_BEARISH"
            strength = bearish / total * 0.5
        else:
            bias = "NEUTRAL"
            strength = 0.0

        result = {
            "bias": bias,
            "strength": round(strength, 2),
            "bullish_count": bullish,
            "bearish_count": bearish,
            "neutral_count": neutral,
            "total": total,
            "avg_bull_confidence": round(total_conf_bull / bullish, 3) if bullish else 0,
            "avg_bear_confidence": round(total_conf_bear / bearish, 3) if bearish else 0,
        }

        logger.info(f"[CrossPairIntel:Bias] Market: {bias} (strength={strength:.0%}) "
                    f"bull={bullish}/{total}, bear={bearish}/{total}")

        self._latest["market_bias"] = result
        return result

    def detect_btc_lead(self, btc_signal: Dict, alt_signals: List[Dict]) -> Dict[str, Any]:
        """
        Detect BTC leading altcoins pattern.
        If BTC just turned BULLISH, altcoins historically follow in 4-8h.
        """
        if not btc_signal:
            return {"btc_leading": False, "expected_alt_direction": "NEUTRAL",
                    "confidence_adjustment": 0.0}

        btc_direction = btc_signal.get("signal", "NEUTRAL")
        btc_conf = btc_signal.get("confidence", 0)

        # Count how many alts agree with BTC
        agreeing = 0
        disagreeing = 0
        for alt in alt_signals:
            alt_dir = alt.get("signal", "NEUTRAL")
            if alt_dir == btc_direction:
                agreeing += 1
            elif alt_dir != "NEUTRAL" and btc_direction != "NEUTRAL":
                disagreeing += 1

        total_alts = len(alt_signals) if alt_signals else 1

        # BTC leading pattern: BTC has clear direction but alts haven't followed yet
        btc_leading = (btc_conf > 0.40 and
                       btc_direction != "NEUTRAL" and
                       disagreeing / total_alts < 0.30)  # Less than 30% disagree

        # If BTC is leading and alts haven't caught up, slight bias for alts to follow
        if btc_leading and agreeing / total_alts < 0.50:
            conf_adj = 0.05 if btc_direction == "BULLISH" else -0.05
        else:
            conf_adj = 0.0

        result = {
            "btc_leading": btc_leading,
            "btc_direction": btc_direction,
            "btc_confidence": btc_conf,
            "expected_alt_direction": btc_direction if btc_leading else "NEUTRAL",
            "alts_agreeing": agreeing,
            "alts_disagreeing": disagreeing,
            "confidence_adjustment": conf_adj,
        }

        if btc_leading:
            logger.info(f"[CrossPairIntel:BTCLead] BTC {btc_direction} (conf={btc_conf:.2f}), "
                       f"{agreeing}/{total_alts} alts agree → alt bias {conf_adj:+.2f}")

        self._latest["btc_lead"] = result
        return result

    def funding_heatmap(self, pairs: List[str] = None) -> Dict[str, Any]:
        """
        Aggregate funding rates across all pairs.
        >70% positive = market crowded long = contrarian bearish warning.
        """
        try:
            conn = self._get_conn()

            if pairs:
                placeholders = ",".join("?" for _ in pairs)
                rows = conn.execute(f"""
                    SELECT pair, funding_rate, long_short_ratio
                    FROM derivatives_data
                    WHERE pair IN ({placeholders})
                    AND id IN (SELECT MAX(id) FROM derivatives_data GROUP BY pair)
                """, tuple(pairs)).fetchall()
            else:
                rows = conn.execute("""
                    SELECT pair, funding_rate, long_short_ratio
                    FROM derivatives_data
                    WHERE id IN (SELECT MAX(id) FROM derivatives_data GROUP BY pair)
                    AND timestamp > datetime('now', '-2 hours')
                """).fetchall()

            conn.close()

            if not rows:
                return {"avg_funding": 0, "pct_positive": 0, "crowding": "neutral",
                        "pairs_count": 0, "extreme_pairs": []}

            positive = 0
            negative = 0
            total_funding = 0.0
            extreme_pairs = []

            for r in rows:
                fr = r["funding_rate"]
                if fr is None:
                    continue
                fr = float(fr)
                total_funding += fr
                if fr > 0:
                    positive += 1
                elif fr < 0:
                    negative += 1
                if abs(fr) > 0.0005:  # Extreme funding
                    extreme_pairs.append({"pair": r["pair"], "funding_rate": fr})

            n = positive + negative
            avg_funding = total_funding / n if n > 0 else 0
            pct_positive = positive / n * 100 if n > 0 else 50

            if pct_positive > 70:
                crowding = "crowded_long"
            elif pct_positive < 30:
                crowding = "crowded_short"
            else:
                crowding = "neutral"

            result = {
                "avg_funding": round(avg_funding, 6),
                "pct_positive": round(pct_positive, 1),
                "crowding": crowding,
                "pairs_count": n,
                "extreme_pairs": extreme_pairs[:5],
            }

            if crowding != "neutral":
                logger.info(f"[CrossPairIntel:Funding] Market {crowding}: "
                           f"{pct_positive:.0f}% positive, avg={avg_funding*100:.4f}%, "
                           f"{len(extreme_pairs)} extreme pairs")

            self._latest["funding_heatmap"] = result
            return result

        except Exception as e:
            logger.debug(f"[CrossPairIntel:Funding] Heatmap failed: {e}")
            return {"avg_funding": 0, "pct_positive": 50, "crowding": "neutral",
                    "pairs_count": 0, "extreme_pairs": []}

    def get_confidence_overlay(self, pair: str) -> float:
        """
        Get aggregate confidence adjustment for a pair based on cross-pair intelligence.
        Returns: float adjustment to add to signal confidence (can be negative).
        """
        adjustment = 0.0

        # Market bias overlay
        bias = self._latest.get("market_bias", {})
        bias_dir = bias.get("bias", "NEUTRAL")
        bias_strength = bias.get("strength", 0)

        if bias_dir in ("BEARISH", "MILD_BEARISH"):
            adjustment -= 0.05 * bias_strength  # Market-wide weakness
        elif bias_dir in ("BULLISH", "MILD_BULLISH"):
            adjustment += 0.03 * bias_strength  # Slight boost in bullish market

        # BTC lead overlay (for altcoins only)
        btc_lead = self._latest.get("btc_lead", {})
        if not pair.upper().startswith("BTC") and btc_lead.get("btc_leading"):
            adjustment += btc_lead.get("confidence_adjustment", 0)

        # Funding crowding overlay
        funding = self._latest.get("funding_heatmap", {})
        if funding.get("crowding") == "crowded_long":
            adjustment -= 0.03  # Market-wide crowding = reduce bullish confidence
        elif funding.get("crowding") == "crowded_short":
            adjustment += 0.03

        return round(adjustment, 3)

    def get_latest(self) -> Dict[str, Any]:
        """Return latest cross-pair intelligence for API endpoint.
        Reads from DB if in-memory cache is empty (handles ephemeral instances)."""
        result = {
            "market_bias": self._latest.get("market_bias", {}),
            "btc_lead": self._latest.get("btc_lead", {}),
            "funding_heatmap": self._latest.get("funding_heatmap", {}),
            "updated_at": datetime.now(tz=timezone.utc).isoformat(),
        }

        # If in-memory is empty, try loading from DB (persisted by update())
        if not result["market_bias"] and not result["funding_heatmap"]:
            try:
                conn = self._get_conn()
                row = conn.execute(
                    "SELECT data_json, timestamp FROM cross_pair_cache "
                    "WHERE id = 1"
                ).fetchone()
                conn.close()
                if row and row["data_json"]:
                    cached = json.loads(row["data_json"])
                    result.update(cached)
                    result["updated_at"] = row["timestamp"]
                    # Also populate in-memory for subsequent calls
                    self._latest.update(cached)
            except Exception:
                pass  # Table may not exist yet

        return result

    def update(self):
        """
        Scheduler job: refresh all cross-pair intelligence.
        Called every 30 minutes.
        """
        logger.info("[CrossPairIntel:Update] Refreshing cross-pair intelligence...")

        # 1. Funding heatmap (always available from derivatives_data)
        self.funding_heatmap()

        # 2. Market bias + BTC lead detection from recent Evidence Engine audit logs
        try:
            conn = self._get_conn()
            rows = conn.execute("""
                SELECT pair, signal, confidence FROM evidence_audit_log
                WHERE id IN (SELECT MAX(id) FROM evidence_audit_log GROUP BY pair)
                AND timestamp > datetime('now', '-2 hours')
                ORDER BY confidence DESC
                LIMIT 20
            """).fetchall()
            conn.close()

            if rows:
                results = [{"pair": r["pair"], "signal": r["signal"], "confidence": r["confidence"]}
                           for r in rows]
                # Market bias from all pairs
                self.compute_market_bias(results)

                # BTC lead detection: separate BTC signal from altcoin signals
                btc_signals = [r for r in results if r["pair"].upper().startswith("BTC")]
                alt_signals = [r for r in results if not r["pair"].upper().startswith("BTC")]
                if btc_signals and alt_signals:
                    self.detect_btc_lead(btc_signals[0], alt_signals)
                    logger.info(f"[CrossPairIntel:Update] BTC lead detection: "
                               f"BTC={btc_signals[0]['signal']}, {len(alt_signals)} alts checked")
        except Exception as e:
            logger.debug(f"[CrossPairIntel:Update] Market bias/BTC lead update failed: {e}")

        # Persist to DB so other instances (API, strategy) can read it
        self._persist_latest()

        logger.info("[CrossPairIntel:Update] Refresh complete.")

    def _persist_latest(self):
        """Persist computed intelligence to SQLite for cross-instance access."""
        try:
            conn = self._get_conn()
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cross_pair_cache (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    data_json TEXT,
                    timestamp TEXT
                )
            """)
            data = {
                "market_bias": self._latest.get("market_bias", {}),
                "btc_lead": self._latest.get("btc_lead", {}),
                "funding_heatmap": self._latest.get("funding_heatmap", {}),
            }
            conn.execute("""
                INSERT OR REPLACE INTO cross_pair_cache (id, data_json, timestamp)
                VALUES (1, ?, ?)
            """, (json.dumps(data), datetime.now(tz=timezone.utc).isoformat()))
            conn.commit()
            conn.close()
            logger.debug("[CrossPairIntel:Persist] Latest data saved to DB")
        except Exception as e:
            logger.debug(f"[CrossPairIntel:Persist] Failed: {e}")
