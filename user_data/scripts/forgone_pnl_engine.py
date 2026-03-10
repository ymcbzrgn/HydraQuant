"""
Phase 3.5 / Phase 6: Forgone P&L Engine
"TEK EN ÖNEMLİ diagnostik metrik" — ROADMAP satır 1036-1039

Every AI signal that is NOT executed as a real trade gets paper-traded here.
After a configurable window (e.g. 4 hours), we check the market price and
calculate what we WOULD have made or lost. This is the only way to measure
whether guardrails are creating or destroying value.
"""

import os
import sqlite3
import logging
from typing import Optional, List, Dict, Any
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

from ai_config import AI_DB_PATH as DB_PATH


class ForgonePnLEngine:
    """
    Tracks every AI signal — executed or not — and calculates
    what the outcome WOULD have been to quantify guardrail value.
    """

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._ensure_table()

    def _get_db_connection(self):
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_table(self):
        """Auto-migrate the forgone_profit table to include all required columns."""
        with self._get_db_connection() as conn:
            c = conn.cursor()
            c.execute('''
                CREATE TABLE IF NOT EXISTS forgone_profit (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pair TEXT NOT NULL,
                    signal_type TEXT,
                    signal_time DATETIME DEFAULT CURRENT_TIMESTAMP,
                    confidence REAL,
                    entry_price REAL,
                    was_executed BOOLEAN DEFAULT 0,
                    exit_price REAL,
                    forgone_pnl REAL,
                    resolved_at DATETIME
                )
            ''')
            # Migrate old tables missing new columns
            c.execute("PRAGMA table_info(forgone_profit)")
            columns = [col['name'] for col in c.fetchall()]
            for col_name, col_type in [
                ('signal_type', 'TEXT'),
                ('entry_price', 'REAL'),
                ('exit_price', 'REAL'),
                ('resolved_at', 'DATETIME'),
            ]:
                if col_name not in columns:
                    c.execute(f'ALTER TABLE forgone_profit ADD COLUMN {col_name} {col_type}')
            conn.commit()

    def log_forgone_signal(
        self,
        pair: str,
        signal_type: str,
        confidence: float,
        entry_price: float,
        was_executed: bool = False,
    ) -> Optional[int]:
        """
        Called for EVERY AI signal, whether executed or rejected.
        If was_executed=False, the signal becomes a "paper trade" that we resolve later.
        If was_executed=True, we log it for completeness but the real P&L comes from Freqtrade.

        Returns:
            int: The row ID of the forgone_profit entry, or None on failure.
        """
        try:
            with self._get_db_connection() as conn:
                c = conn.cursor()
                c.execute('''
                    INSERT INTO forgone_profit (pair, signal_type, confidence, entry_price, was_executed)
                    VALUES (?, ?, ?, ?, ?)
                ''', (pair, signal_type, confidence, entry_price, 1 if was_executed else 0))
                conn.commit()
                row_id: int = c.lastrowid

            executed_str = "EXECUTED" if was_executed else "FORGONE (paper trade)"
            logger.info(f"[Forgone P&L] Logged {signal_type} signal for {pair} @ ${entry_price:.2f} "
                        f"(Conf: {confidence:.2f}) — {executed_str} — ID: {row_id}")
            return row_id

        except Exception as e:
            logger.error(f"[Forgone P&L] Failed to log signal for {pair}: {e}")
            return None

    def resolve_forgone_trade(self, forgone_id: int, exit_price: float) -> bool:
        """
        Called after the paper-trade window expires (e.g. 4 hours later).
        Calculates the P&L that WOULD have happened.

        For BULL signals: pnl = ((exit - entry) / entry) * 100
        For BEAR signals: pnl = ((entry - exit) / entry) * 100
        """
        try:
            with self._get_db_connection() as conn:
                c = conn.cursor()
                c.execute("SELECT pair, signal_type, entry_price FROM forgone_profit WHERE id = ?", (forgone_id,))
                row = c.fetchone()

                if not row:
                    logger.warning(f"[Forgone P&L] ID {forgone_id} not found.")
                    return False

                entry_price = row['entry_price']
                signal_type = row['signal_type']

                if signal_type == "BULL":
                    pnl = ((exit_price - entry_price) / entry_price) * 100
                elif signal_type == "BEAR":
                    pnl = ((entry_price - exit_price) / entry_price) * 100
                else:
                    pnl = 0.0

                c.execute('''
                    UPDATE forgone_profit
                    SET exit_price = ?, forgone_pnl = ?, resolved_at = ?
                    WHERE id = ?
                ''', (exit_price, round(pnl, 4), datetime.now(tz=timezone.utc).isoformat(), forgone_id))
                conn.commit()

            logger.info(f"[Forgone P&L] Resolved ID {forgone_id}: {signal_type} on {row['pair']} "
                        f"Entry=${entry_price:.2f} Exit=${exit_price:.2f} -> PnL={pnl:.2f}%")
            return True

        except Exception as e:
            logger.error(f"[Forgone P&L] Failed to resolve ID {forgone_id}: {e}")
            return False

    def weekly_summary(self) -> Dict[str, Any]:
        """
        Generates the weekly Forgone P&L report.
        Answers: "How much value did our guardrails create or destroy this week?"
        """
        try:
            with self._get_db_connection() as conn:
                c = conn.cursor()

                # Forgone (not executed) trades — resolved
                c.execute('''
                    SELECT COUNT(*) as cnt, 
                           COALESCE(SUM(forgone_pnl), 0) as total_pnl,
                           COALESCE(AVG(forgone_pnl), 0) as avg_pnl,
                           COALESCE(AVG(confidence), 0) as avg_confidence
                    FROM forgone_profit
                    WHERE was_executed = 0 AND forgone_pnl IS NOT NULL
                      AND resolved_at >= datetime('now', '-7 days')
                ''')
                forgone = dict(c.fetchone())

                # Executed trades — for comparison
                c.execute('''
                    SELECT COUNT(*) as cnt,
                           COALESCE(SUM(forgone_pnl), 0) as total_pnl,
                           COALESCE(AVG(forgone_pnl), 0) as avg_pnl,
                           COALESCE(AVG(confidence), 0) as avg_confidence
                    FROM forgone_profit
                    WHERE was_executed = 1 AND forgone_pnl IS NOT NULL
                      AND resolved_at >= datetime('now', '-7 days')
                ''')
                executed = dict(c.fetchone())

            summary = {
                "period": "last_7_days",
                "forgone_trades": {
                    "count": forgone['cnt'],
                    "total_pnl_pct": round(forgone['total_pnl'], 2),
                    "avg_pnl_pct": round(forgone['avg_pnl'], 2),
                    "avg_confidence": round(forgone['avg_confidence'], 2),
                },
                "executed_trades": {
                    "count": executed['cnt'],
                    "total_pnl_pct": round(executed['total_pnl'], 2),
                    "avg_pnl_pct": round(executed['avg_pnl'], 2),
                    "avg_confidence": round(executed['avg_confidence'], 2),
                },
                "guardrail_value": round(executed['total_pnl'] - forgone['total_pnl'], 2),
                "verdict": ""
            }

            if forgone['total_pnl'] > 0:
                summary["verdict"] = (
                    f"GUARDRAILS DESTROYED VALUE: Skipped signals would have earned +{forgone['total_pnl']:.2f}%. "
                    f"Consider relaxing confidence thresholds."
                )
            elif forgone['total_pnl'] < 0:
                summary["verdict"] = (
                    f"GUARDRAILS SAVED MONEY: Skipped signals would have lost {forgone['total_pnl']:.2f}%. "
                    f"Current risk management is working."
                )
            else:
                summary["verdict"] = "No forgone trades resolved this week."

            logger.info(f"[Forgone P&L Weekly] {summary['verdict']}")
            return summary

        except Exception as e:
            logger.error(f"[Forgone P&L] Weekly summary failed: {e}")
            return {"error": str(e)}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    engine = ForgonePnLEngine()

    # Simulate: AI gives a BULL signal for BTC at $67,500, but it was NOT executed
    fid1 = engine.log_forgone_signal("BTC/USDT", "BULL", 0.45, 67500.00, was_executed=False)
    # Simulate: AI gives a BEAR signal for ETH at $3,200, and it WAS executed
    fid2 = engine.log_forgone_signal("ETH/USDT", "BEAR", 0.82, 3200.00, was_executed=True)

    # 4 hours later, resolve them with market prices
    if fid1:
        engine.resolve_forgone_trade(fid1, exit_price=68200.00)  # BTC went up -> forgone profit
    if fid2:
        engine.resolve_forgone_trade(fid2, exit_price=3050.00)   # ETH went down -> executed win

    # Print weekly summary
    import json
    report = engine.weekly_summary()
    print("\n=== FORGONE P&L WEEKLY REPORT ===")
    print(json.dumps(report, indent=2))
