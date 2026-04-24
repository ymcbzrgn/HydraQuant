"""
cerebellum_timing.py — Phase 26 Sprint 2, Task 11F

Cerebellum: 24-slot hour performance optimization (Process 15).

Tracks win rate and PnL by hour of day (UTC). Uses this to:
  - Reduce sizing during historically weak hours
  - Increase sizing during historically strong hours
  - Feed into self_model temporal profile

The cerebellum is the brain's timing center — it optimizes WHEN to trade.

Integration:
  - Reads from: ai_decisions (trade outcomes with timestamps)
  - Reads from: DuckDB hourly_performance() analytics
  - Writes to: pheromone_field ("cerebellum_timing")
  - Consumed by: autonomous_lifecycle (circadian layer), HydraSizer
"""

import os
import sys
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
logger = logging.getLogger("cerebellum_timing")

from ai_config import AI_DB_PATH
from db import get_db_connection, init_db

N_SLOTS = 24  # 24 hours

# Minimum trades per slot for statistical significance
MIN_TRADES_PER_SLOT = 5

# Sizing adjustment range
MIN_TIMING_MULT = 0.3   # Worst hours get 30% sizing
MAX_TIMING_MULT = 1.3   # Best hours get 130% sizing


class CerebellumTiming:
    """24-hour performance optimizer — the organism's timing center."""

    def __init__(self):
        self._hour_stats: Dict[int, Dict] = {}  # hour → {wins, losses, pnl_sum, count}
        self._sizing_multipliers: Dict[int, float] = {}
        self._last_update = None
        init_db()

    def update_from_trades(self, lookback_days: int = 30):
        """Update hourly statistics from trade history."""
        conn = get_db_connection(AI_DB_PATH)
        try:
            cutoff = (datetime.now(timezone.utc) - timedelta(days=lookback_days)).isoformat()

            rows = conn.execute("""
                SELECT timestamp, outcome_pnl
                FROM ai_decisions
                WHERE outcome_pnl IS NOT NULL AND timestamp >= ?
                ORDER BY timestamp ASC
            """, (cutoff,)).fetchall()

            # Reset stats
            self._hour_stats = {h: {"wins": 0, "losses": 0, "pnl_sum": 0.0, "count": 0}
                                for h in range(N_SLOTS)}

            for row in rows:
                try:
                    ts = str(row["timestamp"])
                    dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                    hour = dt.hour
                    pnl = row["outcome_pnl"] or 0

                    self._hour_stats[hour]["count"] += 1
                    self._hour_stats[hour]["pnl_sum"] += pnl
                    if pnl > 0:
                        self._hour_stats[hour]["wins"] += 1
                    else:
                        self._hour_stats[hour]["losses"] += 1
                except Exception:
                    continue

            # Compute sizing multipliers
            self._compute_multipliers()
            self._last_update = datetime.now(timezone.utc)

            total_trades = sum(s["count"] for s in self._hour_stats.values())
            active_hours = sum(1 for s in self._hour_stats.values()
                             if s["count"] >= MIN_TRADES_PER_SLOT)
            logger.info(f"[Cerebellum] Updated: {total_trades} trades across "
                       f"{active_hours} active hours")

        finally:
            conn.close()

    def _compute_multipliers(self):
        """Compute per-hour sizing multipliers from win rates."""
        win_rates = {}
        for hour, stats in self._hour_stats.items():
            if stats["count"] >= MIN_TRADES_PER_SLOT:
                win_rates[hour] = stats["wins"] / stats["count"]
            else:
                win_rates[hour] = 0.5  # Default for insufficient data

        if not win_rates:
            self._sizing_multipliers = {h: 1.0 for h in range(N_SLOTS)}
            return

        # Normalize: best hour gets MAX_TIMING_MULT, worst gets MIN_TIMING_MULT
        min_wr = min(win_rates.values())
        max_wr = max(win_rates.values())
        wr_range = max_wr - min_wr

        for hour in range(N_SLOTS):
            wr = win_rates.get(hour, 0.5)
            if wr_range > 0.01:
                normalized = (wr - min_wr) / wr_range
                mult = MIN_TIMING_MULT + normalized * (MAX_TIMING_MULT - MIN_TIMING_MULT)
            else:
                mult = 1.0
            self._sizing_multipliers[hour] = round(mult, 3)

    def get_timing_multiplier(self, hour_utc: int = None) -> float:
        """Get sizing multiplier for current (or specified) hour."""
        if hour_utc is None:
            hour_utc = datetime.now(timezone.utc).hour

        if not self._sizing_multipliers:
            self.update_from_trades()

        return self._sizing_multipliers.get(hour_utc, 1.0)

    def get_best_hours(self, top_n: int = 5) -> List[Dict]:
        """Get top N best performing hours."""
        if not self._hour_stats:
            self.update_from_trades()

        hours = []
        for h, stats in self._hour_stats.items():
            if stats["count"] >= MIN_TRADES_PER_SLOT:
                wr = stats["wins"] / stats["count"]
                avg_pnl = stats["pnl_sum"] / stats["count"]
                hours.append({
                    "hour_utc": h,
                    "win_rate": round(wr, 3),
                    "avg_pnl": round(avg_pnl, 4),
                    "trade_count": stats["count"],
                    "multiplier": self._sizing_multipliers.get(h, 1.0),
                })

        hours.sort(key=lambda x: x["win_rate"], reverse=True)
        return hours[:top_n]

    def get_worst_hours(self, top_n: int = 5) -> List[Dict]:
        """Get top N worst performing hours."""
        best = self.get_best_hours(top_n=24)
        best.sort(key=lambda x: x["win_rate"])
        return best[:top_n]

    def get_full_schedule(self) -> List[Dict]:
        """Get full 24-hour performance schedule."""
        if not self._hour_stats:
            self.update_from_trades()

        schedule = []
        for h in range(N_SLOTS):
            stats = self._hour_stats.get(h, {"wins": 0, "losses": 0, "pnl_sum": 0, "count": 0})
            wr = stats["wins"] / max(stats["count"], 1)
            schedule.append({
                "hour_utc": h,
                "win_rate": round(wr, 3) if stats["count"] >= MIN_TRADES_PER_SLOT else None,
                "avg_pnl": round(stats["pnl_sum"] / max(stats["count"], 1), 4),
                "trade_count": stats["count"],
                "multiplier": self._sizing_multipliers.get(h, 1.0),
                "sufficient_data": stats["count"] >= MIN_TRADES_PER_SLOT,
            })
        return schedule

    def publish_to_pheromone(self):
        """Publish current timing info to pheromone field."""
        try:
            from pheromone_field import get_pheromone_field
            hour = datetime.now(timezone.utc).hour
            mult = self.get_timing_multiplier(hour)
            best = self.get_best_hours(3)
            worst = self.get_worst_hours(3)

            pfield = get_pheromone_field()
            pfield.deposit("cerebellum", "cerebellum_timing", {
                "current_hour": hour,
                "current_multiplier": mult,
                "best_hours": [h["hour_utc"] for h in best],
                "worst_hours": [h["hour_utc"] for h in worst],
            })
        except Exception:
            pass

    def print_schedule(self):
        """Print full 24-hour schedule."""
        schedule = self.get_full_schedule()
        print(f"\n{'='*60}")
        print(f"  CEREBELLUM 24-HOUR SCHEDULE")
        print(f"{'='*60}")
        print(f"  {'Hour':>4} | {'Win Rate':>8} | {'Avg PnL':>8} | {'Trades':>6} | {'Mult':>5}")
        print(f"  {'-'*45}")
        for s in schedule:
            wr_str = f"{s['win_rate']:.0%}" if s['win_rate'] is not None else "  N/A"
            print(f"  {s['hour_utc']:>4}h | {wr_str:>8} | {s['avg_pnl']:>7.3f} | "
                  f"{s['trade_count']:>6} | {s['multiplier']:>5.2f}")
        print(f"{'='*60}\n")


# Singleton
_cerebellum_instance = None

def get_cerebellum() -> CerebellumTiming:
    global _cerebellum_instance
    if _cerebellum_instance is None:
        _cerebellum_instance = CerebellumTiming()
    return _cerebellum_instance
