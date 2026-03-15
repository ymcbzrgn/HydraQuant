"""
Phase 3.5.3: Risk Budget System (Dynamic VaR)
Manages daily risk budget based on portfolio value and asset volatility.

Core principle: Don't BLOCK trades — SHRINK them when budget is low.
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


class RiskBudgetManager:
    """
    Controls daily risk exposure using a VaR-inspired budget.
    Every trade consumes budget = position_size * volatility * (1/confidence).
    Budget resets daily at 00:00 UTC.
    """

    def __init__(
        self,
        portfolio_value: float = 10000.0,
        daily_var_pct: float = 0.50,
        db_path: str = DB_PATH,
    ):
        self.portfolio_value = portfolio_value
        self.daily_var_pct = daily_var_pct
        self.db_path = db_path
        self._multiplier = 1.0  # Adjusted weekly based on P&L

        # Initialize from DB or start fresh
        self._ensure_table()
        self._load_state()

    def _get_conn(self):
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _ensure_table(self):
        with self._get_conn() as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS risk_budget (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT NOT NULL,
                    initial_budget REAL NOT NULL,
                    consumed REAL DEFAULT 0.0,
                    multiplier REAL DEFAULT 1.0,
                    updated_at TEXT
                )
            ''')
            conn.commit()

    def _load_state(self):
        """Load today's budget or create one."""
        today = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT * FROM risk_budget WHERE date = ?", (today,)
            ).fetchone()

            if row:
                self._consumed = float(row['consumed'])
                self._multiplier = float(row['multiplier'])
                self._today = today
            else:
                # New day — reset budget
                initial = self.portfolio_value * self.daily_var_pct * self._multiplier
                conn.execute(
                    "INSERT INTO risk_budget (date, initial_budget, consumed, multiplier, updated_at) VALUES (?, ?, 0.0, ?, ?)",
                    (today, initial, self._multiplier, datetime.now(tz=timezone.utc).isoformat())
                )
                conn.commit()
                self._consumed = 0.0
                self._today = today

    @property
    def daily_budget(self) -> float:
        """Total daily risk budget (portfolio * VaR% * multiplier)."""
        return self.portfolio_value * self.daily_var_pct * self._multiplier

    def remaining_budget(self) -> float:
        """How much risk budget is left today."""
        return max(0.0, self.daily_budget - self._consumed)

    def budget_utilization(self) -> float:
        """Fraction of budget consumed (0.0 to 1.0+)."""
        if self.daily_budget <= 0:
            return 1.0
        return self._consumed / self.daily_budget

    def consume_budget(
        self,
        position_size: float,
        asset_volatility: float,
        confidence: float,
    ) -> float:
        """
        Consume risk budget for a trade.

        Args:
            position_size: Dollar value of the position
            asset_volatility: Asset's recent volatility (e.g. 0.03 = 3%)
            confidence: AI's confidence in the trade (0.01 to 1.0)

        Returns:
            Remaining budget after this trade
        """
        safe_confidence = max(confidence, 0.01)  # Prevent division by zero
        consumption = position_size * asset_volatility * (1.0 / safe_confidence)
        self._consumed += consumption

        # Persist to DB
        today = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
        try:
            with self._get_conn() as conn:
                conn.execute(
                    "UPDATE risk_budget SET consumed = ?, updated_at = ? WHERE date = ?",
                    (self._consumed, datetime.now(tz=timezone.utc).isoformat(), today)
                )
                conn.commit()
        except Exception as e:
            logger.error(f"[RiskBudget] Failed to persist budget: {e}")

        remaining = self.remaining_budget()
        logger.info(
            f"[RiskBudget] Consumed ${consumption:.2f} "
            f"(pos=${position_size:.0f}, vol={asset_volatility:.3f}, conf={confidence:.2f}). "
            f"Remaining: ${remaining:.2f}/{self.daily_budget:.2f}"
        )
        return remaining

    def scale_position(self, proposed_stake: float) -> float:
        """
        Scale a proposed position based on remaining budget.
        If >75% budget used, start shrinking. If 100% used, shrink to 10% (never block).
        """
        utilization = self.budget_utilization()

        if utilization < 0.75:
            return proposed_stake  # Full position
        elif utilization < 1.0:
            # Linear scale-down from 100% to 25% between 75%-100% utilization
            fraction = 1.0 - 3.0 * (utilization - 0.75)  # 1.0 → 0.25
            return proposed_stake * max(fraction, 0.25)
        else:
            # Budget exceeded — shrink to 10% (dust position, never block)
            return proposed_stake * 0.10

    def weekly_adjust(self, weekly_pnl_pct: float):
        """
        Adjust next week's budget multiplier based on P&L.
        Profitable week → increase budget (max 2.0x).
        Losing week → decrease budget (min 0.5x).
        """
        if weekly_pnl_pct > 0:
            self._multiplier = min(2.0, self._multiplier * 1.1)
        elif weekly_pnl_pct < -2.0:
            self._multiplier = max(0.5, self._multiplier * 0.8)
        else:
            self._multiplier = max(0.5, self._multiplier * 0.95)

        logger.info(f"[RiskBudget] Weekly adjust: PnL={weekly_pnl_pct:.2f}%, new multiplier={self._multiplier:.2f}")

    def update_portfolio_value(self, real_balance: float):
        """
        Sync portfolio_value with real exchange balance.
        Called from AIFreqtradeSizer on every trade to keep budget proportional to actual account.
        """
        if real_balance <= 0:
            return
        old_value = self.portfolio_value
        self.portfolio_value = real_balance
        if abs(old_value - real_balance) > 1.0:
            logger.info(f"[RiskBudget] Portfolio synced: ${old_value:.2f} → ${real_balance:.2f} "
                        f"(budget: ${self.daily_budget:.2f})")

    def reset_daily(self):
        """Force reset the daily budget (normally auto-resets via _load_state)."""
        self._consumed = 0.0
        today = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
        try:
            with self._get_conn() as conn:
                initial = self.daily_budget
                conn.execute(
                    "INSERT OR REPLACE INTO risk_budget (date, initial_budget, consumed, multiplier, updated_at) VALUES (?, ?, 0.0, ?, ?)",
                    (today, initial, self._multiplier, datetime.now(tz=timezone.utc).isoformat())
                )
                conn.commit()
        except Exception as e:
            logger.error(f"[RiskBudget] Reset failed: {e}")

        logger.info(f"[RiskBudget] Daily reset. New budget: ${self.daily_budget:.2f}")
