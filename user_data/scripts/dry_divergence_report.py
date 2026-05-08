"""Phase 30 B.19 — TR-DRY vs Testnet 8-metric divergence report.

Compares two parallel dry bots (Bybit Futures testnet vs Bybit.tr DRY-RUN).
Used as input to D.9 real-capital promotion gate.

Scheduler: daily cron at 23:50 UTC -> Telegram daily summary.
"""
from __future__ import annotations

import logging
import sqlite3
from collections import Counter
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

DEFAULT_TESTNET_DB = "/root/freqtrade/user_data/tradesv3.sqlite"
DEFAULT_TR_DRY_DB = "/root/freqtrade/user_data/tradesv3_tr_dry.sqlite"


def _metrics_for(db_path: str, since_hours: int = 24) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "n_trades": 0, "win_rate": 0.0, "avg_pnl_pct": 0.0,
        "avg_hold_min": 0.0, "avg_stake_usdt": 0.0, "n_liquidations": 0,
        "exit_reasons": {},
    }
    try:
        with sqlite3.connect(db_path) as conn:
            cur = conn.execute(
                f"""SELECT COUNT(*),
                          SUM(CASE WHEN close_profit > 0 THEN 1 ELSE 0 END),
                          AVG(close_profit),
                          AVG((julianday(close_date)-julianday(open_date))*24*60),
                          AVG(stake_amount),
                          SUM(CASE WHEN close_profit < -0.95 THEN 1 ELSE 0 END)
                   FROM trades
                   WHERE open_date >= datetime('now', '-{int(since_hours)} hours')
                     AND close_date IS NOT NULL"""
            )
            n, wins, avg_pnl, hold, stake, liquid = cur.fetchone()
            n = int(n or 0)
            wins = int(wins or 0)
            out.update({
                "n_trades": n,
                "win_rate": (wins / n) if n else 0.0,
                "avg_pnl_pct": float((avg_pnl or 0)) * 100,
                "avg_hold_min": float(hold or 0),
                "avg_stake_usdt": float(stake or 0),
                "n_liquidations": int(liquid or 0),
            })
            cur = conn.execute(
                f"""SELECT exit_reason, COUNT(*) FROM trades
                   WHERE open_date >= datetime('now', '-{int(since_hours)} hours')
                     AND close_date IS NOT NULL
                   GROUP BY exit_reason ORDER BY 2 DESC LIMIT 5"""
            )
            out["exit_reasons"] = {r[0] or "unknown": int(r[1]) for r in cur.fetchall()}
    except Exception as e:
        logger.error(f"[DryDiv] metrics_for {db_path} failed: {e}")
    return out


def divergence_report(
    testnet_db: str = DEFAULT_TESTNET_DB,
    tr_dry_db: str = DEFAULT_TR_DRY_DB,
    since_hours: int = 24,
) -> Dict[str, Any]:
    testnet = _metrics_for(testnet_db, since_hours)
    tr_dry = _metrics_for(tr_dry_db, since_hours)
    divergence = {
        "trade_count_ratio": (testnet["n_trades"] / max(tr_dry["n_trades"], 1)),
        "win_rate_delta": testnet["win_rate"] - tr_dry["win_rate"],
        "pnl_delta_pct": testnet["avg_pnl_pct"] - tr_dry["avg_pnl_pct"],
        "stake_delta_usdt": testnet["avg_stake_usdt"] - tr_dry["avg_stake_usdt"],
        "liquidation_delta": testnet["n_liquidations"] - tr_dry["n_liquidations"],
    }
    return {"testnet": testnet, "tr_dry": tr_dry, "divergence": divergence}


def daily_telegram_summary() -> str:
    r = divergence_report()
    lines = [
        "DRY DIVERGENCE 24h",
        f"Testnet:  {r['testnet']['n_trades']} trade WR={r['testnet']['win_rate']:.1%} "
        f"avg_pnl={r['testnet']['avg_pnl_pct']:+.2f}% liquid={r['testnet']['n_liquidations']}",
        f"TR-DRY:   {r['tr_dry']['n_trades']} trade WR={r['tr_dry']['win_rate']:.1%} "
        f"avg_pnl={r['tr_dry']['avg_pnl_pct']:+.2f}% liquid={r['tr_dry']['n_liquidations']}",
        f"deltaWR={r['divergence']['win_rate_delta']:+.1%} "
        f"deltaPnL={r['divergence']['pnl_delta_pct']:+.2f}% "
        f"deltaLiq={r['divergence']['liquidation_delta']}",
    ]
    msg = "\n".join(lines)
    try:
        from severity_router import emit
        sev = "warn" if abs(r['divergence']['win_rate_delta']) > 0.20 else "info"
        emit(kind="dry_divergence.daily", severity=sev, message=msg, payload=r)
    except Exception:
        pass
    return msg
