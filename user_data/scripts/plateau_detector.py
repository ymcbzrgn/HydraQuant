"""Phase 30 A.23 — Plateau detection on trade winrate / pnl.

Detects when a metric flatlines (std% < threshold) over rolling N days.
Plateau is a signal that the system isn't learning anymore — emit warn,
optionally trigger:
- Forced agent exploration boost
- LinUCB epsilon increase
- Operator alert
"""
from __future__ import annotations

import logging
import statistics
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def detect_winrate_plateau(window_days: int = 14, std_pct_threshold: float = 0.02) -> Dict[str, Any]:
    """Computes per-day winrate over the window, returns plateau verdict."""
    try:
        from db import AI_DB_PATH, get_db_connection

        with get_db_connection(AI_DB_PATH) as conn:
            rows = conn.execute(
                f"""SELECT DATE(close_date) AS day,
                          COUNT(*) AS n,
                          SUM(CASE WHEN close_profit > 0 THEN 1 ELSE 0 END) AS wins
                   FROM trades
                   WHERE close_date >= datetime('now', '-{int(window_days)} days')
                     AND close_profit IS NOT NULL
                   GROUP BY DATE(close_date)
                   ORDER BY day"""
            ).fetchall()
    except Exception as e:
        logger.error(f"[PlateauDetector] db read failed: {e}")
        return {"plateau": False, "error": str(e)}

    if len(rows) < 5:
        return {"plateau": False, "reason": "insufficient_data", "n_days": len(rows)}

    rates: List[float] = [(r[2] / r[1]) if r[1] else 0.0 for r in rows]
    if not rates:
        return {"plateau": False, "reason": "no_rates"}
    mean = sum(rates) / len(rates)
    std = statistics.stdev(rates) if len(rates) > 1 else 0.0
    std_pct = (std / mean) if mean > 0 else 0.0

    plateau = std_pct < std_pct_threshold and len(rates) >= 7
    severity = "warn" if plateau else "info"

    out = {
        "plateau": plateau,
        "metric": "winrate",
        "window_days": window_days,
        "n_days": len(rates),
        "mean": round(mean, 4),
        "std": round(std, 4),
        "std_pct": round(std_pct, 4),
        "threshold": std_pct_threshold,
        "severity": severity,
    }
    try:
        from db import AI_DB_PATH, get_db_connection
        with get_db_connection(AI_DB_PATH) as conn:
            conn.execute(
                """INSERT INTO plateau_events
                   (metric, window_days, std_pct, mean, severity)
                   VALUES (?, ?, ?, ?, ?)""",
                ("winrate", window_days, std_pct, mean, severity),
            )
            conn.commit()
    except Exception:
        pass
    if plateau:
        try:
            from severity_router import emit
            emit(kind="plateau.winrate", severity="warn",
                 message=f"Winrate plateau: std_pct={std_pct:.2%} over {window_days}d",
                 payload=out)
        except Exception:
            pass
    return out
