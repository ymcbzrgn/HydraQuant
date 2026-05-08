"""Autonomy promotion diagnostic — Phase 30 A.29.

Live state: autonomy_state.level=0 since 2026-03-12 (57+ days stuck) despite
1579 trades. This module:
1. Reads autonomy_state + last 30d trades.
2. Computes promotion eligibility (n_trades, sharpe, dd, winrate).
3. Persists snapshot to autonomy_diagnostics tablo.
4. Telegrams CRITICAL via severity_router if level=0 stuck >14d.

Scheduler hook: daily 03:00 UTC.
"""
from __future__ import annotations

import logging
import statistics
from datetime import datetime, timezone
from typing import Any, Dict

logger = logging.getLogger(__name__)

PROMOTION_CRITERIA = {
    "level_0_to_1": {
        "min_n_trades_30d": 30,
        "min_sharpe_30d": 0.5,
        "max_drawdown_30d": 0.15,
        "min_winrate_30d": 0.55,
    },
    "level_1_to_2": {
        "min_n_trades_30d": 60,
        "min_sharpe_30d": 1.0,
        "max_drawdown_30d": 0.10,
        "min_winrate_30d": 0.58,
    },
    "level_2_to_3": {
        "min_n_trades_30d": 120,
        "min_sharpe_30d": 1.3,
        "max_drawdown_30d": 0.08,
        "min_winrate_30d": 0.60,
    },
}

STUCK_THRESHOLD_DAYS = 14


def _approx_sharpe(pnls):
    if not pnls or len(pnls) < 2:
        return 0.0
    mean = sum(pnls) / len(pnls)
    std = statistics.stdev(pnls)
    return (mean / std) if std > 0 else 0.0


def run_diagnostic() -> Dict[str, Any]:
    """Returns diagnostic dict; persists to DB; logs CRITICAL if stuck."""
    from db import get_db_connection, AI_DB_PATH

    with get_db_connection(AI_DB_PATH) as conn:
        cur = conn.execute(
            "SELECT level, last_promoted_at FROM autonomy_state WHERE id=1"
        )
        row = cur.fetchone()
        if not row:
            logger.error("[AutonomyDiagnostic] No autonomy_state row")
            return {"error": "no_state"}

        level = int(row[0]) if row[0] is not None else 0
        last_promoted_raw = row[1] or ""
        try:
            last_dt = datetime.fromisoformat(str(last_promoted_raw).replace("Z", "+00:00"))
            days_stuck = (datetime.now(timezone.utc) - last_dt).days
        except Exception:
            days_stuck = 999

        cur = conn.execute(
            """SELECT COUNT(*),
                      AVG(close_profit),
                      SUM(CASE WHEN close_profit > 0 THEN 1 ELSE 0 END),
                      MIN(close_profit)
               FROM trades
               WHERE close_date >= datetime('now', '-30 days')
                 AND close_profit IS NOT NULL"""
        )
        n, mean_pnl, wins, worst = cur.fetchone()
        n = int(n or 0)
        wins = int(wins or 0)
        winrate = (wins / n) if n else 0.0

        cur = conn.execute(
            """SELECT close_profit FROM trades
               WHERE close_date >= datetime('now', '-30 days')
                 AND close_profit IS NOT NULL"""
        )
        pnls = [float(r[0]) for r in cur.fetchall()]
        sharpe_approx = _approx_sharpe(pnls)
        worst = float(worst or 0)

        criteria_key = f"level_{level}_to_{level+1}"
        crit = PROMOTION_CRITERIA.get(criteria_key, {})
        eligible = (
            n >= crit.get("min_n_trades_30d", 999)
            and sharpe_approx >= crit.get("min_sharpe_30d", 99)
            and abs(worst) <= crit.get("max_drawdown_30d", 0)
            and winrate >= crit.get("min_winrate_30d", 1.0)
        )

        report: Dict[str, Any] = {
            "current_level": level,
            "days_stuck": days_stuck,
            "n_trades_30d": n,
            "winrate_30d": round(winrate, 4),
            "sharpe_approx_30d": round(sharpe_approx, 4),
            "worst_drawdown_30d": round(worst, 4),
            "criteria": crit,
            "eligible_for_promotion": bool(eligible),
        }

        conn.execute(
            """INSERT INTO autonomy_diagnostics
               (level, days_stuck, n_trades_30d, winrate_30d, sharpe_approx_30d,
                worst_drawdown_30d, eligible)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (level, days_stuck, n, winrate, sharpe_approx, worst, int(eligible)),
        )
        conn.commit()

        if level == 0 and days_stuck > STUCK_THRESHOLD_DAYS:
            msg = (
                f"[AutonomyDiagnostic] STUCK at level 0 for {days_stuck} days; "
                f"eligible={eligible}; n={n} winrate={winrate:.2%} sharpe={sharpe_approx:.2f}"
            )
            logger.warning(msg)
            try:
                from severity_router import emit  # type: ignore

                emit(
                    kind="autonomy.stuck",
                    severity="critical",
                    message=msg,
                    payload=report,
                )
            except Exception:
                pass

        logger.info(f"[AutonomyDiagnostic] {report}")
        return report


def maybe_promote_if_eligible(dry_run: bool = True) -> Dict[str, Any]:
    """If diagnostic says eligible, optionally promote.

    Default dry_run=True — manuel approve gerekli. Production'da
    `autonomy.auto_promote` flag PARAM_REGISTRY ile yonetilir.
    """
    report = run_diagnostic()
    if not report.get("eligible_for_promotion"):
        return {"action": "noop", **report}
    if dry_run:
        return {"action": "would_promote", **report}

    from db import get_db_connection, AI_DB_PATH

    with get_db_connection(AI_DB_PATH) as conn:
        new_level = int(report["current_level"]) + 1
        conn.execute(
            "UPDATE autonomy_state SET level=?, last_promoted_at=? WHERE id=1",
            (new_level, datetime.now(timezone.utc).isoformat()),
        )
        conn.commit()
    logger.info(f"[AutonomyDiagnostic] PROMOTED level -> {new_level}")
    return {"action": "promoted", "new_level": new_level, **report}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print(run_diagnostic())
