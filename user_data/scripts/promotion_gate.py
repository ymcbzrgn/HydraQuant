"""Phase 30 D.9 — Real-capital promotion gate (8-condition hard gate).

Gates real capital activation. All 8 must pass over the rolling window:
1. PnL > 0
2. Sharpe > 1.0
3. Max DD < 10%
4. 0 liquidations
5. n_trades >= 30
6. winrate >= 55%
7. LinUCB convergence (variance < 0.1)
8. autonomy_state level >= 1

Manual approve required even when gate passes — module only reports.
"""
from __future__ import annotations

import json
import logging
import statistics
from dataclasses import dataclass, field
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


@dataclass
class GateResult:
    passed: bool
    eligibility_pct: float
    blocked_by: List[str]
    metrics: Dict[str, Any] = field(default_factory=dict)


def evaluate_gate(window_days: int = 14) -> GateResult:
    """Phase 30 D.9 — `trades` lives in tradesv3.sqlite (freqtrade DB), other state
    in ai_data.sqlite. Read both."""
    try:
        import os, sqlite3
        from pathlib import Path
        from db import AI_DB_PATH, get_db_connection

        # Locate tradesv3.sqlite (sibling of ai_data.sqlite or env override)
        _trades_db = os.environ.get("TRADES_DB_PATH")
        if not _trades_db:
            _candidate = Path(AI_DB_PATH).parent.parent / "tradesv3.sqlite"
            _trades_db = str(_candidate)
        n = wins = liquid = 0
        pnl = worst = mean_p = 0.0
        pnls = []
        try:
            with sqlite3.connect(_trades_db) as tdb:
                row = tdb.execute(
                    f"""SELECT COUNT(*),
                              COALESCE(SUM(close_profit_abs), 0),
                              SUM(CASE WHEN close_profit > 0 THEN 1 ELSE 0 END),
                              SUM(CASE WHEN close_profit < -0.95 THEN 1 ELSE 0 END),
                              MIN(close_profit),
                              AVG(close_profit)
                       FROM trades
                       WHERE close_date >= datetime('now', '-{int(window_days)} days')
                         AND close_profit IS NOT NULL"""
                ).fetchone()
                n, pnl, wins, liquid, worst, mean_p = row
                n = int(n or 0); wins = int(wins or 0)
                cur = tdb.execute(
                    f"""SELECT close_profit FROM trades
                       WHERE close_date >= datetime('now', '-{int(window_days)} days')
                         AND close_profit IS NOT NULL"""
                )
                pnls = [float(r[0]) for r in cur.fetchall()]
        except Exception as _trades_e:
            logger.error(f"[D.9] tradesv3.sqlite read failed: {_trades_e}")

        winrate = (wins / n) if n else 0.0
        sharpe = 0.0
        if pnls and len(pnls) > 1:
            std = statistics.stdev(pnls)
            sharpe = ((mean_p or 0) / std) if std > 0 else 0.0

        with get_db_connection(AI_DB_PATH) as conn:
            cur = conn.execute("SELECT level FROM autonomy_state WHERE id=1")
            r = cur.fetchone()
            autonomy_level = int(r[0]) if r and r[0] is not None else 0
            try:
                cur = conn.execute("SELECT AVG(reward_variance) FROM linucb_state")
                r = cur.fetchone()
                linucb_var = float(r[0]) if r and r[0] is not None else 99.0
            except Exception:
                linucb_var = 99.0

            max_dd = abs(worst or 0)

            gates = {
                "pnl_positive": (pnl or 0) > 0,
                "sharpe_above_1": sharpe > 1.0,
                "max_dd_below_10pct": max_dd < 0.10,
                "no_liquidations": (liquid or 0) == 0,
                "min_trades_30": n >= 30,
                "winrate_above_55": winrate >= 0.55,
                "linucb_converged": linucb_var < 0.1,
                "autonomy_level_1plus": autonomy_level >= 1,
            }
            blocked = [k for k, v in gates.items() if not v]
            metrics = {
                "n_trades": n, "pnl_usdt": pnl, "winrate": winrate,
                "sharpe": sharpe, "max_dd": max_dd, "n_liquid": liquid,
                "autonomy_level": autonomy_level, "linucb_var": linucb_var,
            }

            conn.execute(
                """INSERT INTO promotion_gate_history
                   (eligibility_pct, passed, blocked_by, metrics_json)
                   VALUES (?, ?, ?, ?)""",
                (sum(gates.values()) / len(gates), int(all(gates.values())),
                 json.dumps(blocked), json.dumps(metrics, default=str)),
            )
            conn.commit()
            return GateResult(
                passed=all(gates.values()),
                eligibility_pct=sum(gates.values()) / len(gates),
                blocked_by=blocked,
                metrics=metrics,
            )
    except Exception as e:
        logger.error(f"[D.9] evaluate failed: {e}")
        return GateResult(False, 0.0, ["error"], {"error": str(e)})


def weekly_summary() -> str:
    r = evaluate_gate(window_days=14)
    status = "READY" if r.passed else f"BLOCKED ({len(r.blocked_by)})"
    msg = (
        f"REAL-CAPITAL GATE (14d)\n"
        f"Status: {status}\n"
        f"Eligibility: {r.eligibility_pct:.0%}\n"
        f"Blocked by: {', '.join(r.blocked_by) if r.blocked_by else '-'}\n"
        f"Metrics: PnL={r.metrics.get('pnl_usdt', 0):+.2f} USDT | "
        f"WR={r.metrics.get('winrate', 0):.1%} | Sharpe={r.metrics.get('sharpe', 0):.2f}"
    )
    try:
        from severity_router import emit
        sev = "info" if r.passed else "warn"
        emit(kind="promotion_gate.weekly", severity=sev, message=msg, payload=r.metrics)
    except Exception:
        pass
    return msg
