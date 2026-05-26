"""Phase 30 B.16 — Hourly KPI rollup CSV + Grafana feed.

Aggregates per-hour KPIs from trades + ai_decisions + llm_calls + price_anomaly_events
into kpi_rollup_hourly. Optional CSV dump for Grafana CSV plugin.

Scheduler: hourly cron at HH:01 UTC.
"""
from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_CSV_PATH = Path(__file__).parent.parent / "data" / "kpi_rollup_hourly.csv"


def _resolve_trades_db_path() -> str:
    """Trades live in tradesv3*.sqlite (freqtrade's own ledger), NOT ai_data.sqlite.

    2026-05-26: previous code ran `SELECT … FROM trades` against ai_data.sqlite
    and silently failed every hour with "no such table: trades", burning one
    pool connection per cron tick. Now we resolve the active trades DB
    (TR-DRY first, then production) and ATTACH it for cross-DB joins.
    """
    import os
    candidates = [
        "/root/freqtrade/user_data/tradesv3_tr_dry.sqlite",
        "/root/freqtrade/user_data/tradesv3.sqlite",
    ]
    for p in candidates:
        if os.path.exists(p) and os.path.getsize(p) > 0:
            return p
    return candidates[0]


def rollup_last_hour() -> Dict[str, Any]:
    """Compute and persist current-hour bucket.

    Reads `trades` from the freqtrade trades DB via ATTACH; reads `llm_calls`
    and `price_anomaly_events` from ai_data.sqlite (their home); writes
    kpi_rollup_hourly back to ai_data.sqlite.
    """
    try:
        from db import AI_DB_PATH, get_db_connection
        trades_db = _resolve_trades_db_path()

        with get_db_connection(AI_DB_PATH) as conn:
            # ATTACH trades DB read-only so we can join across DBs in one connection.
            conn.execute(f"ATTACH DATABASE ? AS td", (trades_db,))
            try:
                row = conn.execute(
                    """SELECT COUNT(*),
                              SUM(CASE WHEN close_profit > 0 THEN 1 ELSE 0 END),
                              SUM(CASE WHEN close_profit < 0 THEN 1 ELSE 0 END),
                              COALESCE(SUM(close_profit_abs), 0)
                       FROM td.trades
                       WHERE close_date >= datetime('now', '-1 hour')
                         AND close_profit IS NOT NULL"""
                ).fetchone()
                n_trades, wins, losses, pnl_sum = row
            finally:
                try:
                    conn.execute("DETACH DATABASE td")
                except Exception:
                    pass

            row2 = conn.execute(
                """SELECT COUNT(*), COALESCE(AVG(latency_ms), 0)
                   FROM llm_calls
                   WHERE timestamp >= datetime('now', '-1 hour')"""
            ).fetchone()
            n_llm, avg_lat = row2

            row3 = conn.execute(
                """SELECT COUNT(*) FROM price_anomaly_events
                   WHERE ts >= datetime('now', '-1 hour')"""
            ).fetchone()
            n_anom = row3[0] if row3 else 0

            from datetime import datetime, timezone
            bucket = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:00")

            conn.execute(
                """INSERT OR REPLACE INTO kpi_rollup_hourly
                   (hour_bucket, n_trades, wins, losses, pnl_sum,
                    n_llm_calls, avg_latency_ms, n_anomalies)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (bucket, int(n_trades or 0), int(wins or 0), int(losses or 0),
                 float(pnl_sum or 0), int(n_llm or 0), float(avg_lat or 0), int(n_anom or 0)),
            )
            conn.commit()

            return {
                "hour": bucket,
                "n_trades": int(n_trades or 0),
                "wins": int(wins or 0),
                "losses": int(losses or 0),
                "pnl_sum": float(pnl_sum or 0),
                "n_llm_calls": int(n_llm or 0),
                "avg_latency_ms": float(avg_lat or 0),
                "n_anomalies": int(n_anom or 0),
            }
    except Exception as e:
        logger.error(f"[KPIRollup] failed: {e}")
        return {}


def export_csv(path: Optional[Path] = None, last_n_hours: int = 168) -> Optional[Path]:
    path = Path(path) if path else DEFAULT_CSV_PATH
    try:
        from db import AI_DB_PATH, get_db_connection

        with get_db_connection(AI_DB_PATH) as conn:
            rows = conn.execute(
                f"""SELECT hour_bucket, n_trades, wins, losses, pnl_sum,
                          n_llm_calls, avg_latency_ms, n_anomalies
                   FROM kpi_rollup_hourly
                   ORDER BY hour_bucket DESC
                   LIMIT {int(last_n_hours)}"""
            ).fetchall()
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as fp:
            writer = csv.writer(fp)
            writer.writerow(["hour_bucket", "n_trades", "wins", "losses", "pnl_sum",
                             "n_llm_calls", "avg_latency_ms", "n_anomalies"])
            for r in reversed(rows):
                writer.writerow(r)
        return path
    except Exception as e:
        logger.error(f"[KPIRollup] csv export failed: {e}")
        return None
