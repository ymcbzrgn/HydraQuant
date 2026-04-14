"""
self_model.py — Phase 26 Sprint 2, Task 9A

Metacognition: The organism maintains a model of ITSELF.

4 analysis dimensions:
  1. Organ Strength/Weakness — which modules perform well?
  2. Temporal Profile — which hours/days are strong/weak?
  3. Bias Detection — overconfidence after streaks, recency bias, etc.
  4. Competence Map — pair×regime skill matrix

Integration:
  - Reads from ai_decisions (trade outcomes)
  - Writes to self_model_profile table (SQLite)
  - Used by Active Learner (9B) to find knowledge gaps
  - Used by HRL Meta-Policy (7D) for organ selection
  - Used by neural_organism for confidence modulation

Reference: Metacognition in AI trading systems (novel contribution #3)
"""

import os
import sys
import json
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
logger = logging.getLogger("self_model")

from ai_config import AI_DB_PATH
from db import get_db_connection, get_connection, execute_with_retry, init_db


class SelfModel:
    """The organism maintains a model of ITSELF — metacognition."""

    def __init__(self, db_path: str = AI_DB_PATH):
        self.db_path = db_path
        self.organ_strengths: Dict[str, float] = {}
        self.organ_weaknesses: Dict[str, float] = {}
        self.temporal_profile: Dict[str, float] = {}
        self.bias_detection: Dict[str, Any] = {}
        self.competence_map: Dict[Tuple[str, str], float] = {}
        init_db()

    # ===================================================================
    # Introspection
    # ===================================================================

    def introspect(self, lookback_days: int = 30) -> Dict[str, Any]:
        """Full self-analysis: organ strengths, temporal profile, biases, competence map."""
        conn = get_db_connection(self.db_path)
        try:
            cutoff = (datetime.now(timezone.utc) - timedelta(days=lookback_days)).isoformat()

            trades = conn.execute("""
                SELECT id, pair, signal_type, confidence, regime,
                       outcome_pnl, outcome_duration, timestamp
                FROM ai_decisions
                WHERE outcome_pnl IS NOT NULL AND timestamp >= ?
                ORDER BY timestamp ASC
            """, (cutoff,)).fetchall()

            if len(trades) < 10:
                logger.info(f"[SelfModel] Insufficient trades for introspection: {len(trades)}")
                return {"error": "insufficient data", "n_trades": len(trades)}

            trade_list = [dict(t) for t in trades]

        finally:
            conn.close()

        logger.info(f"[SelfModel] Introspecting on {len(trade_list)} trades...")

        # 1. Temporal Profile — performance by hour
        self._analyze_temporal(trade_list)

        # 2. Bias Detection
        self._detect_biases(trade_list)

        # 3. Competence Map — pair × regime
        self._build_competence_map(trade_list)

        # 4. Organ strengths (from evidence sub-scores correlation with outcomes)
        self._analyze_organs(trade_list)

        # Persist to DB
        self._persist_profiles()

        result = {
            "n_trades": len(trade_list),
            "temporal_weak_hours": [h for h, v in self.temporal_profile.items() if v < 0.35],
            "temporal_strong_hours": [h for h, v in self.temporal_profile.items() if v > 0.65],
            "biases_detected": list(self.bias_detection.keys()),
            "competence_entries": len(self.competence_map),
            "organ_strengths": self.organ_strengths,
        }

        logger.info(f"[SelfModel] Introspection complete: "
                    f"{len(result['temporal_weak_hours'])} weak hours, "
                    f"{len(result['biases_detected'])} biases, "
                    f"{result['competence_entries']} competence entries")

        return result

    def _analyze_temporal(self, trades: List[Dict]):
        """Analyze performance by hour of day and day of week."""
        hour_wins = {}
        hour_total = {}

        for trade in trades:
            ts = trade.get("timestamp", "")
            try:
                dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
                hour = dt.hour
                day = dt.strftime("%A")
            except Exception:
                continue

            won = (trade.get("outcome_pnl", 0) or 0) > 0

            # Hour
            hour_key = f"hour_{hour}"
            hour_wins[hour_key] = hour_wins.get(hour_key, 0) + (1 if won else 0)
            hour_total[hour_key] = hour_total.get(hour_key, 0) + 1

            # Day
            day_key = f"day_{day}"
            hour_wins[day_key] = hour_wins.get(day_key, 0) + (1 if won else 0)
            hour_total[day_key] = hour_total.get(day_key, 0) + 1

        for key in hour_total:
            if hour_total[key] >= 3:  # Minimum 3 trades for statistical relevance
                self.temporal_profile[key] = hour_wins[key] / hour_total[key]

    def _detect_biases(self, trades: List[Dict]):
        """Detect behavioral biases in trading patterns."""
        self.bias_detection = {}

        # 1. Overconfidence after winning streak
        streak_losses = 0
        streak_opportunities = 0
        for i in range(3, len(trades)):
            prev_3_won = all((trades[j].get("outcome_pnl", 0) or 0) > 0
                           for j in range(i - 3, i))
            if prev_3_won:
                streak_opportunities += 1
                if (trades[i].get("outcome_pnl", 0) or 0) < 0:
                    streak_losses += 1

        if streak_opportunities >= 3:
            loss_rate = streak_losses / streak_opportunities
            if loss_rate > 0.5:
                self.bias_detection["overconfidence_after_streak"] = {
                    "loss_rate_after_3wins": round(loss_rate, 3),
                    "opportunities": streak_opportunities,
                    "severity": "HIGH" if loss_rate > 0.7 else "MEDIUM",
                }

        # 2. Recency bias — are recent trades weighted more in confidence?
        if len(trades) >= 20:
            first_half_conf = np.mean([t.get("confidence", 0.5) or 0.5 for t in trades[:len(trades) // 2]])
            second_half_conf = np.mean([t.get("confidence", 0.5) or 0.5 for t in trades[len(trades) // 2:]])
            conf_drift = second_half_conf - first_half_conf
            if abs(conf_drift) > 0.1:
                self.bias_detection["confidence_drift"] = {
                    "first_half_avg": round(float(first_half_conf), 3),
                    "second_half_avg": round(float(second_half_conf), 3),
                    "drift": round(float(conf_drift), 3),
                    "direction": "increasing" if conf_drift > 0 else "decreasing",
                }

        # 3. Loss aversion — sizing reduction after losses
        post_loss_sizes = []
        post_win_sizes = []
        for i in range(1, len(trades)):
            prev_pnl = trades[i - 1].get("outcome_pnl", 0) or 0
            curr_conf = trades[i].get("confidence", 0.5) or 0.5
            if prev_pnl < 0:
                post_loss_sizes.append(curr_conf)
            elif prev_pnl > 0:
                post_win_sizes.append(curr_conf)

        if post_loss_sizes and post_win_sizes:
            avg_post_loss = np.mean(post_loss_sizes)
            avg_post_win = np.mean(post_win_sizes)
            if avg_post_loss < avg_post_win * 0.8:
                self.bias_detection["loss_aversion"] = {
                    "post_loss_avg_conf": round(float(avg_post_loss), 3),
                    "post_win_avg_conf": round(float(avg_post_win), 3),
                    "ratio": round(float(avg_post_loss / (avg_post_win + 1e-10)), 3),
                }

    def _build_competence_map(self, trades: List[Dict]):
        """Build pair × regime competence matrix."""
        self.competence_map = {}
        buckets: Dict[Tuple[str, str], List[float]] = {}

        for trade in trades:
            pair = trade.get("pair", "UNKNOWN")
            regime = trade.get("regime", "transitional") or "transitional"
            pnl = trade.get("outcome_pnl", 0) or 0

            key = (pair, regime)
            if key not in buckets:
                buckets[key] = []
            buckets[key].append(1.0 if pnl > 0 else 0.0)

        for key, outcomes in buckets.items():
            if len(outcomes) >= 3:  # Minimum 3 trades
                self.competence_map[key] = round(float(np.mean(outcomes)), 3)

    def _analyze_organs(self, trades: List[Dict]):
        """Analyze organ (evidence sub-score) correlation with outcomes."""
        # Load evidence sub-scores
        conn = get_db_connection(self.db_path)
        try:
            organ_scores: Dict[str, List[Tuple[float, float]]] = {}  # organ → [(score, pnl)]

            for trade in trades:
                pair = trade.get("pair", "")
                ts = trade.get("timestamp", "")
                pnl = trade.get("outcome_pnl", 0) or 0

                ev = conn.execute("""
                    SELECT sub_scores_json FROM evidence_audit_log
                    WHERE pair = ? AND ABS(JULIANDAY(timestamp) - JULIANDAY(?)) < 0.01
                    LIMIT 1
                """, (pair, ts)).fetchone()

                if ev and ev["sub_scores_json"]:
                    try:
                        sub = json.loads(ev["sub_scores_json"])
                        for organ, score in sub.items():
                            if organ not in organ_scores:
                                organ_scores[organ] = []
                            organ_scores[organ].append((float(score), float(pnl)))
                    except Exception:
                        pass
        finally:
            conn.close()

        # Compute correlation: high score → positive PnL = strong organ
        for organ, data in organ_scores.items():
            if len(data) < 5:
                continue

            scores = np.array([d[0] for d in data])
            pnls = np.array([d[1] for d in data])

            # Simple correlation
            if scores.std() > 1e-6 and pnls.std() > 1e-6:
                corr = float(np.corrcoef(scores, pnls)[0, 1])
            else:
                corr = 0.0

            # High score → wins = organ is accurate
            high_score_mask = scores > np.median(scores)
            if high_score_mask.sum() > 0:
                win_rate = float((pnls[high_score_mask] > 0).mean())
            else:
                win_rate = 0.5

            if win_rate > 0.55:
                self.organ_strengths[organ] = round(win_rate, 3)
            elif win_rate < 0.40:
                self.organ_weaknesses[organ] = round(win_rate, 3)

    # ===================================================================
    # Query API
    # ===================================================================

    def should_i_trade(self, pair: str, regime: str, hour: int) -> Tuple[bool, str, float]:
        """Ask the organism: should I take this trade?

        Returns (should_trade, reason, confidence_modifier).
        confidence_modifier: 0.0-1.0 multiplier on final confidence.
        """
        reasons = []
        modifier = 1.0

        # Check competence map
        competence = self.competence_map.get((pair, regime))
        if competence is not None and competence < 0.35:
            reasons.append(f"low competence in {pair}+{regime} ({competence:.0%})")
            modifier *= 0.5

        # Check temporal profile
        hour_skill = self.temporal_profile.get(f"hour_{hour}")
        if hour_skill is not None and hour_skill < 0.30:
            reasons.append(f"weak hour ({hour}:00, win_rate={hour_skill:.0%})")
            modifier *= 0.7

        # Check biases
        if "overconfidence_after_streak" in self.bias_detection:
            # Don't block, but warn
            reasons.append("overconfidence bias active")
            modifier *= 0.85

        if not reasons:
            return True, "competent", modifier

        if modifier < 0.4:
            return False, "; ".join(reasons), modifier

        return True, f"caution: {'; '.join(reasons)}", modifier

    def get_competence(self, pair: str, regime: str) -> float:
        """Get competence level for a pair+regime combination."""
        return self.competence_map.get((pair, regime), 0.5)

    def get_weak_areas(self) -> List[Dict]:
        """Get areas where the organism is weakest."""
        weak = []
        for (pair, regime), score in self.competence_map.items():
            if score < 0.4:
                weak.append({"pair": pair, "regime": regime, "competence": score})
        return sorted(weak, key=lambda x: x["competence"])

    # ===================================================================
    # Persistence
    # ===================================================================

    def _persist_profiles(self):
        """Write self-model profiles to self_model_profile table."""
        entries = []

        # Organ strengths
        for organ, score in self.organ_strengths.items():
            entries.append(("organ_strength", organ, score, None, None))
        for organ, score in self.organ_weaknesses.items():
            entries.append(("organ_weakness", organ, score, None, None))

        # Temporal profile
        for key, score in self.temporal_profile.items():
            entries.append(("temporal", key, score, None, None))

        # Competence map
        for (pair, regime), score in self.competence_map.items():
            entries.append(("competence", f"{pair}|{regime}", score, None, None))

        # Biases
        for bias_name, bias_data in self.bias_detection.items():
            severity_val = {"HIGH": 0.9, "MEDIUM": 0.5, "LOW": 0.2}.get(
                bias_data.get("severity", "MEDIUM") if isinstance(bias_data, dict) else "MEDIUM",
                0.5
            )
            entries.append(("bias", bias_name, severity_val, None, None))

        with get_connection() as conn:
            for profile_type, key, value, confidence, sample_size in entries:
                try:
                    conn.execute("""
                        INSERT OR REPLACE INTO self_model_profile
                        (profile_type, key, value, confidence, sample_size)
                        VALUES (?, ?, ?, ?, ?)
                    """, (profile_type, key, value, confidence, sample_size))
                except Exception as e:
                    logger.debug(f"[SelfModel] Persist error: {e}")
            conn.commit()

        logger.info(f"[SelfModel] Persisted {len(entries)} profile entries")

    def load_from_db(self):
        """Load self-model profiles from DB."""
        conn = get_db_connection(self.db_path)
        try:
            rows = conn.execute("""
                SELECT profile_type, key, value FROM self_model_profile
            """).fetchall()

            for row in rows:
                pt = row["profile_type"]
                key = row["key"]
                val = row["value"]

                if pt == "organ_strength":
                    self.organ_strengths[key] = val
                elif pt == "organ_weakness":
                    self.organ_weaknesses[key] = val
                elif pt == "temporal":
                    self.temporal_profile[key] = val
                elif pt == "competence":
                    parts = key.split("|")
                    if len(parts) == 2:
                        self.competence_map[(parts[0], parts[1])] = val
                elif pt == "bias":
                    self.bias_detection[key] = {"severity_value": val}

            logger.info(f"[SelfModel] Loaded {len(rows)} profile entries from DB")
        finally:
            conn.close()

    def get_status(self) -> Dict:
        return {
            "organ_strengths": len(self.organ_strengths),
            "organ_weaknesses": len(self.organ_weaknesses),
            "temporal_entries": len(self.temporal_profile),
            "competence_entries": len(self.competence_map),
            "biases_detected": len(self.bias_detection),
        }

    def print_status(self):
        status = self.get_status()
        print(f"\n{'='*55}")
        print(f"  SELF-MODEL STATUS")
        print(f"{'='*55}")
        print(f"  Strong organs: {self.organ_strengths}")
        print(f"  Weak organs:   {self.organ_weaknesses}")
        print(f"  Temporal:      {status['temporal_entries']} entries")
        print(f"  Competence:    {status['competence_entries']} pair×regime combos")
        print(f"  Biases:        {list(self.bias_detection.keys())}")
        print(f"{'='*55}\n")


# Singleton
_self_model_instance = None

def get_self_model() -> SelfModel:
    global _self_model_instance
    if _self_model_instance is None:
        _self_model_instance = SelfModel()
    return _self_model_instance
