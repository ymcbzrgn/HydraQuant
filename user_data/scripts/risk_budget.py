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
from db import get_db_connection

# Phase 24: Neural Organism — adaptive parameters
try:
    from neural_organism import _p
except ImportError:
    def _p(param_id, fallback=0.5, regime="_global"):
        return fallback


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

        # Live risk-state the constitution guard reads (HydraSizer:1608-1609).
        # Without real writers these getattr()-default-0 lookups made the
        # max_portfolio_heat_pct / max_drawdown_pct branches in constitution
        # permanently inert — the guard thought the account was always calm.
        self._peak_balance: float = portfolio_value
        self._current_drawdown_pct: float = 0.0
        self._portfolio_heat_pct: float = 0.0

        # Initialize from DB or start fresh
        self._ensure_table()
        self._load_state()

    def _get_conn(self):
        conn = get_db_connection(self.db_path)
        return conn

    def _ensure_table(self):
        with self._get_conn() as conn:
            # Boot-order self-heal: the retired schema
            # (id PRIMARY KEY CHECK(id=1), daily_var_limit/current_usage/…)
            # cannot be upgraded via ADD COLUMN because the CHECK(id=1)
            # constraint prevents INSERTs with auto-ids. If that schema is
            # detected, move it aside to risk_budget_legacy_backup and let
            # the canonical CREATE TABLE below take over. The retired
            # columns have no canonical counterpart, so no row-level data
            # transfer is meaningful — the backup preserves raw bytes for
            # forensic inspection.
            existing = {r[1] for r in conn.execute("PRAGMA table_info(risk_budget)").fetchall()}
            if existing and "date" not in existing:
                logger.warning(
                    "[RiskBudget] Legacy schema detected (cols=%s) — renaming to "
                    "risk_budget_legacy_backup so canonical schema can take over.",
                    sorted(existing),
                )
                try:
                    conn.execute("DROP TABLE IF EXISTS risk_budget_legacy_backup")
                    conn.execute("ALTER TABLE risk_budget RENAME TO risk_budget_legacy_backup")
                except sqlite3.OperationalError as e:
                    logger.error("[RiskBudget] Legacy rename failed: %s", e)

            conn.execute('''
                CREATE TABLE IF NOT EXISTS risk_budget (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT NOT NULL,
                    initial_budget REAL NOT NULL,
                    consumed REAL DEFAULT 0.0,
                    multiplier REAL DEFAULT 1.0,
                    updated_at TEXT,
                    UNIQUE(date)
                )
            ''')
            # Defensive ADD COLUMN safety-net — if any older canonical-ish
            # install predates one of these columns, add it in place.
            for _col, _typ in [
                ("date", "TEXT"),
                ("initial_budget", "REAL"),
                ("consumed", "REAL DEFAULT 0.0"),
                ("multiplier", "REAL DEFAULT 1.0"),
            ]:
                try:
                    conn.execute(f"ALTER TABLE risk_budget ADD COLUMN {_col} {_typ}")
                except sqlite3.OperationalError:
                    pass
            # Self-healing: if a pre-UNIQUE install produced duplicate rows
            # (observed in production for 2026-04-11), dedup keeping the row
            # with the largest id (latest consumed value), then enforce UNIQUE.
            try:
                conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS ux_risk_budget_date ON risk_budget(date)")
            except sqlite3.IntegrityError:
                conn.execute(
                    "DELETE FROM risk_budget WHERE id NOT IN "
                    "(SELECT MAX(id) FROM risk_budget WHERE date IS NOT NULL GROUP BY date)"
                )
                try:
                    conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS ux_risk_budget_date ON risk_budget(date)")
                except sqlite3.IntegrityError:
                    logger.warning("[RiskBudget] UNIQUE(date) index still blocked — duplicate NULL dates may exist")
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
                        Reported for auditability; NOT applied as a divisor
                        to avoid double-penalising low-conf trades — position
                        sizing already scales stake by confidence^1.5 upstream
                        (position_sizer.py:172-181). Re-dividing here turned
                        consumed into a non-dollar risk-score that crossed
                        initial_budget by day 3 in production.

        Returns:
            Remaining budget after this trade
        """
        # 1-sigma dollar VaR proxy: expected absolute move of the position
        # over the volatility horizon. This keeps the budget unit in USD.
        consumption = max(0.0, float(position_size)) * max(0.0, float(asset_volatility))
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
        Cap position based on remaining budget (Phase 21: cap, not multiplier).
        Small stakes pass through unchanged. Only large stakes get capped.
        Trade-First: never block, never crush already-small stakes.
        """
        remaining = self.remaining_budget()
        if remaining <= 0:
            # Budget exhausted — allow dust trades only (Phase 24: adaptive)
            dust_frac = _p("risk.dust_fraction", 0.01)
            dust_min = _p("risk.dust_min_usd", 1.0)
            return min(proposed_stake, max(self.daily_budget * dust_frac, dust_min))

        # Cap at fraction of remaining budget (Phase 24: adaptive)
        budget_cap = remaining * _p("risk.budget_cap_fraction", 0.25)
        return min(proposed_stake, budget_cap)

    def weekly_adjust(self, weekly_pnl_pct: float):
        """
        Adjust next week's budget multiplier based on P&L.
        Profitable week → increase budget (max 2.0x).
        Losing week → decrease budget (min 0.5x).

        Task 24: the adjustment now PERSISTS. The earlier implementation
        mutated `self._multiplier` in memory only; the scheduler's weekly
        cron constructs a fresh RiskBudgetManager instance, so the
        updated multiplier was lost as soon as this function returned
        and every Monday's `_load_state` read the stale DB value of 1.0.
        """
        if weekly_pnl_pct > 0:
            self._multiplier = min(_p("risk.weekly_mult_max", 2.0),
                                   self._multiplier * _p("risk.weekly_win_mult", 1.1))
        elif weekly_pnl_pct < -2.0:
            self._multiplier = max(_p("risk.weekly_mult_min", 0.5),
                                   self._multiplier * _p("risk.weekly_loss_mult", 0.8))
        else:
            self._multiplier = max(_p("risk.weekly_mult_min", 0.5),
                                   self._multiplier * 0.95)

        logger.info(f"[RiskBudget] Weekly adjust: PnL={weekly_pnl_pct:.2f}%, new multiplier={self._multiplier:.2f}")

        # Persist to today's row so the next strategy/scheduler cycle
        # reads the updated multiplier via _load_state. Fire-and-forget —
        # a DB outage here is logged but non-fatal (the in-memory value
        # is correct for the lifetime of this instance).
        today = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
        try:
            with self._get_conn() as conn:
                cursor = conn.execute(
                    "UPDATE risk_budget SET multiplier = ?, updated_at = ? WHERE date = ?",
                    (self._multiplier,
                     datetime.now(tz=timezone.utc).isoformat(),
                     today),
                )
                if cursor.rowcount == 0:
                    # Today's row doesn't exist yet (weekly cron fires
                    # before the first daily_reset) — INSERT OR REPLACE.
                    conn.execute(
                        "INSERT OR REPLACE INTO risk_budget "
                        "(date, initial_budget, consumed, multiplier, updated_at) "
                        "VALUES (?, ?, 0.0, ?, ?)",
                        (today, self.daily_budget, self._multiplier,
                         datetime.now(tz=timezone.utc).isoformat()),
                    )
                conn.commit()
        except Exception as e:
            logger.error(f"[RiskBudget] Weekly persist failed: {e}")

    def reload_multiplier_from_db(self) -> bool:
        """Hot-reload self._multiplier from the most-recent DB row. Used
        by long-lived instances (HydraSizer's self.risk_budget) so that
        a scheduler-side weekly_adjust becomes visible without a
        strategy restart. Returns True if the value changed.
        """
        try:
            today = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
            with self._get_conn() as conn:
                row = conn.execute(
                    "SELECT multiplier FROM risk_budget WHERE date = ?",
                    (today,),
                ).fetchone()
            if row and row["multiplier"] is not None:
                new_mult = float(row["multiplier"])
                if abs(new_mult - self._multiplier) > 1e-6:
                    old = self._multiplier
                    self._multiplier = new_mult
                    logger.info(
                        f"[RiskBudget] multiplier reloaded {old:.3f} → {new_mult:.3f}"
                    )
                    return True
        except Exception as e:
            logger.debug(f"[RiskBudget] reload_multiplier_from_db failed: {e}")
        return False

    def update_portfolio_value(
        self,
        real_balance: float,
        in_trades_usd: Optional[float] = None,
    ):
        """
        Sync portfolio_value with real exchange balance.
        Called from HydraSizer on every trade to keep budget proportional to
        actual account. Also maintains the live drawdown + heat gauges the
        constitution reads via getattr() on this instance.

        Args:
            real_balance: total account equity in quote currency.
            in_trades_usd: notional currently tied up in open positions. When
                omitted the heat gauge is left at its previous value — pass
                0.0 to reset heat when no open positions exist.
        """
        if real_balance <= 0:
            return
        old_value = self.portfolio_value
        self.portfolio_value = real_balance

        # Peak tracking for drawdown. Allostatic: peak monotonically rises
        # with equity, never shrinks — a recovering account stays "in
        # drawdown" until it surpasses its prior high.
        if real_balance > self._peak_balance:
            self._peak_balance = real_balance
        if self._peak_balance > 0:
            dd = (self._peak_balance - real_balance) / self._peak_balance
            self._current_drawdown_pct = max(0.0, dd * 100.0)
        else:
            self._current_drawdown_pct = 0.0

        # Heat = notional-in-flight as fraction of total equity. Cap at 100%
        # because leveraged positions can report in_trades > equity on paper.
        if in_trades_usd is not None and real_balance > 0:
            heat = float(in_trades_usd) / float(real_balance)
            self._portfolio_heat_pct = max(0.0, min(heat * 100.0, 100.0))

        if abs(old_value - real_balance) > 1.0:
            logger.info(
                f"[RiskBudget] Portfolio synced: ${old_value:.2f} → ${real_balance:.2f} "
                f"(budget: ${self.daily_budget:.2f}, peak: ${self._peak_balance:.2f}, "
                f"dd={self._current_drawdown_pct:.2f}%, heat={self._portfolio_heat_pct:.2f}%)"
            )

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
