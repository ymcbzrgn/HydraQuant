"""
Phase 3.5.4: Autonomy Level Management (L0-L5)
Manages the bot's autonomy level based on track record.

Trade-First Philosophy: Default=TRADE, not Default=BLOCK.
Confidence modulates SIZE, never PERMISSION. Even L0 trades with minimum size.
ONE hard constraint: max_position_cap (3% portfolio).

Levels:
  L0: Nano-live — Kelly=0.03, max $10/trade (minimum viable trade, full logging)
  L1: Micro-live — Kelly=0.07, max $25/trade
  L2: Small-live — Kelly=0.15, max $75/trade
  L3: Cautious-live — Kelly=0.30, max $200/trade
  L4: Standard-live — Kelly=0.50, no cap
  L5: Full-auto — Kelly=0.75, no cap
"""

import os
import sys
import sqlite3
import logging
from datetime import datetime, timezone
from typing import Optional

sys.path.append(os.path.dirname(__file__))

logger = logging.getLogger(__name__)

from ai_config import AI_DB_PATH as DB_PATH

# Phase 24: Neural Organism — adaptive parameters
try:
    from neural_organism import _p
except ImportError:
    def _p(param_id, fallback=0.5, regime="_global"):
        return fallback

# Kelly fraction mapping per autonomy level (Phase 24: reads from Neural Organism)
# Trade-First: EVERY level trades. Size grows with trust, never zero.
def _get_kelly_fractions():
    return {
        0: _p("autonomy.kelly_l0", 0.03),
        1: _p("autonomy.kelly_l1", 0.07),
        2: _p("autonomy.kelly_l2", 0.15),
        3: _p("autonomy.kelly_l3", 0.30),
        4: _p("autonomy.kelly_l4", 0.50),
        5: _p("autonomy.kelly_l5", 0.75),
    }
KELLY_FRACTIONS = {0: 0.03, 1: 0.07, 2: 0.15, 3: 0.30, 4: 0.50, 5: 0.75}  # fallback

# Promotion criteria: (min_trades, min_sharpe, max_dd_pct, min_days)
PROMOTION_CRITERIA = {
    0: (20, 0.0, 100.0, 3),      # L0→L1: 20 real nano trades, 3 days (fast bootstrap)
    1: (50, 0.0, 100.0, 7),      # L1→L2: 50 trades, 7 days
    2: (100, 0.5, 15.0, 30),     # L2→L3: 100 trades, Sharpe>0.5, DD<15%, 30 days
    3: (200, 0.8, 10.0, 60),     # L3→L4: 200 trades, Sharpe>0.8, DD<10%, 60 days
    4: (500, 1.0, 8.0, 90),      # L4→L5: 500 trades, Sharpe>1.0, DD<8%, 90 days
}


class AutonomyManager:
    """
    Manages the AI trading bot's autonomy level (L0-L5).
    Promotes based on sustained performance, demotes on drawdown.
    """

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._ensure_table()
        self.current_level = self._load_level()

    def _get_conn(self):
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _ensure_table(self):
        with self._get_conn() as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS autonomy_state (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    level INTEGER DEFAULT 0,
                    promoted_at TEXT,
                    total_trades INTEGER DEFAULT 0,
                    sharpe_estimate REAL DEFAULT 0.0,
                    max_drawdown_pct REAL DEFAULT 0.0,
                    days_at_level INTEGER DEFAULT 0,
                    updated_at TEXT
                )
            ''')
            # Ensure exactly one row exists
            row = conn.execute("SELECT COUNT(*) FROM autonomy_state").fetchone()
            if row[0] == 0:
                conn.execute(
                    "INSERT INTO autonomy_state (id, level, promoted_at, updated_at) VALUES (1, 0, ?, ?)",
                    (datetime.now(tz=timezone.utc).isoformat(), datetime.now(tz=timezone.utc).isoformat())
                )
            conn.commit()

    def _load_level(self) -> int:
        with self._get_conn() as conn:
            row = conn.execute("SELECT level FROM autonomy_state WHERE id = 1").fetchone()
            return int(row['level']) if row else 0

    def get_level(self) -> int:
        """Get the current autonomy level (0-5)."""
        return self.current_level

    def get_kelly_fraction(self) -> float:
        """Get the Kelly fraction for the current autonomy level (Phase 24: adaptive)."""
        adaptive = _get_kelly_fractions()
        return adaptive.get(self.current_level, KELLY_FRACTIONS.get(self.current_level, 0.03))

    def get_max_stake(self, portfolio_value: float = 0.0) -> Optional[float]:
        """
        Get max stake limit for current level, scaled to portfolio size.
        If portfolio_value given, returns percentage-based cap.
        Fallback to fixed minimums if portfolio unknown.
        """
        # Percentage of portfolio per level (scales with account size)
        pct_limits = {0: 0.01, 1: 0.025, 2: 0.05, 3: 0.10}
        # Absolute minimums (floor, never below these even on tiny accounts)
        abs_minimums = {0: 10.0, 1: 25.0, 2: 75.0, 3: 200.0}

        if self.current_level >= 4:
            return None  # L4/L5: no cap

        pct = pct_limits.get(self.current_level, 0.03)
        abs_min = abs_minimums.get(self.current_level, 10.0)

        if portfolio_value > 0:
            return max(portfolio_value * pct, abs_min)
        return abs_min

    def update_metrics(
        self,
        total_trades: int,
        sharpe: float,
        max_dd_pct: float,
        days_at_level: int
    ):
        """Update performance metrics used for promotion decisions."""
        with self._get_conn() as conn:
            conn.execute(
                """UPDATE autonomy_state
                   SET total_trades = ?, sharpe_estimate = ?, max_drawdown_pct = ?,
                       days_at_level = ?, updated_at = ?
                   WHERE id = 1""",
                (total_trades, sharpe, max_dd_pct, days_at_level,
                 datetime.now(tz=timezone.utc).isoformat())
            )
            conn.commit()

    def check_promotion(
        self,
        total_trades: int,
        sharpe: float,
        max_dd_pct: float,
        days_at_level: int
    ) -> bool:
        """Check if promotion criteria are met for the current level."""
        if self.current_level >= 5:
            return False  # Already at max

        criteria = PROMOTION_CRITERIA.get(self.current_level, (999, 9.0, 0.0, 365))
        min_trades, min_sharpe, max_dd, min_days = criteria

        if (total_trades >= min_trades and
            sharpe >= min_sharpe and
            max_dd_pct <= max_dd and
            days_at_level >= min_days):

            self._promote()
            return True

        return False

    def _promote(self):
        """Promote to next autonomy level (with DB-level guard against duplicates)."""
        # Re-read DB level to prevent stale-instance duplicate promotions
        db_level = self._load_level()
        if db_level != self.current_level:
            self.current_level = db_level
            logger.info(f"[Autonomy] Stale instance detected, synced to DB level L{db_level}")
            return

        new_level = min(5, self.current_level + 1)
        with self._get_conn() as conn:
            # Atomic: only promote if DB still at expected level
            cursor = conn.execute(
                "UPDATE autonomy_state SET level = ?, promoted_at = ?, days_at_level = 0, updated_at = ? WHERE id = 1 AND level = ?",
                (new_level, datetime.now(tz=timezone.utc).isoformat(), datetime.now(tz=timezone.utc).isoformat(), self.current_level)
            )
            conn.commit()
            if cursor.rowcount == 0:
                logger.info(f"[Autonomy] Promotion skipped — DB level already changed")
                self.current_level = self._load_level()
                return

        logger.info(f"[Autonomy] PROMOTED: L{self.current_level} → L{new_level}")
        try:
            from telegram_notifier import AITelegramNotifier
            AITelegramNotifier().send_alert(f"Autonomy PROMOTED L{self.current_level} → L{new_level}. Win rate & Sharpe sustained.", level="INFO")
        except Exception:
            pass

        self.current_level = new_level

    def check_demotion(self, daily_loss_pct: float, weekly_loss_pct: float) -> bool:
        """
        Check if demotion should occur.
        - Daily loss > 3% → scale positions to 25%
        - Weekly loss > 5% → drop one level
        """
        if self.current_level <= 0:
            return False

        if weekly_loss_pct > 5.0:
            self._demote()
            return True

        return False

    def _demote(self):
        """Demote one autonomy level (with DB-level guard against duplicates)."""
        db_level = self._load_level()
        if db_level != self.current_level:
            self.current_level = db_level
            logger.info(f"[Autonomy] Stale instance detected, synced to DB level L{db_level}")
            return

        new_level = max(0, self.current_level - 1)
        with self._get_conn() as conn:
            cursor = conn.execute(
                "UPDATE autonomy_state SET level = ?, days_at_level = 0, updated_at = ? WHERE id = 1 AND level = ?",
                (new_level, datetime.now(tz=timezone.utc).isoformat(), self.current_level)
            )
            conn.commit()
            if cursor.rowcount == 0:
                logger.info(f"[Autonomy] Demotion skipped — DB level already changed")
                self.current_level = self._load_level()
                return

        logger.warning(f"[Autonomy] DEMOTED: L{self.current_level} → L{new_level}")
        try:
            from telegram_notifier import AITelegramNotifier
            AITelegramNotifier().send_alert(f"Autonomy DEMOTED L{self.current_level} → L{new_level} (Weekly loss > 5%)", level="WARNING")
        except Exception:
            pass

        self.current_level = new_level

    def should_scale_down(self, daily_loss_pct: float) -> float:
        """
        Returns position scale factor based on daily loss.
        Normal = 1.0, Daily loss > 3% = 0.25 (emergency brake).
        """
        if daily_loss_pct > 3.0:
            logger.warning(f"[Autonomy] Emergency brake: daily loss {daily_loss_pct:.2f}% > 3%. Scaling to 25%.")
            return 0.25
        return 1.0
