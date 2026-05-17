"""Phase 30 A.2 — Calibrator health check + bypass-restore plan.

Reads confidence_calibrator state, tracks Brier score over rolling window,
and recommends/applies bypass restoration.

Phase 1 (THIS sprint): bypass condition revised — only bypass when
trade_count_30d < n_threshold. Otherwise use real calibrator.

Phase 2 (1 week observation): default bypass=0.0; full restoration.
"""
from __future__ import annotations

import logging
import os
import sqlite3
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)


def _resolve_trades_db() -> str:
    """2026-05-18 fix: `trades` table lives in tradesv3.sqlite (freqtrade DB),
    not ai_data.sqlite. Mirror the resolution pattern from promotion_gate.py.
    TRADES_DB_PATH env var overrides for TR-DRY paper-trade bot.
    """
    env_override = os.environ.get("TRADES_DB_PATH")
    if env_override:
        return env_override
    try:
        from db import AI_DB_PATH
        return str(Path(AI_DB_PATH).parent.parent / "tradesv3.sqlite")
    except Exception:
        return "/root/freqtrade/user_data/tradesv3.sqlite"


def _trade_count_30d() -> int:
    db_path = _resolve_trades_db()
    try:
        conn = sqlite3.connect(db_path, timeout=10.0)
        try:
            row = conn.execute(
                """SELECT COUNT(*) FROM trades
                   WHERE close_date >= datetime('now', '-30 days')
                     AND close_profit IS NOT NULL"""
            ).fetchone()
            return int(row[0] or 0)
        finally:
            conn.close()
    except Exception as e:
        logger.error(f"[CalibratorHealth] count failed ({db_path}): {e}")
        return 0


def _brier_score_30d() -> float:
    """Compute Brier (squared error) of confidence vs realized win/loss."""
    try:
        from db import AI_DB_PATH, get_db_connection

        with get_db_connection(AI_DB_PATH) as conn:
            rows = conn.execute(
                """SELECT confidence, outcome_pnl FROM ai_decisions
                   WHERE outcome_pnl IS NOT NULL
                     AND timestamp >= datetime('now', '-30 days')
                   LIMIT 5000"""
            ).fetchall()
    except Exception as e:
        logger.error(f"[CalibratorHealth] brier failed: {e}")
        return 1.0
    if not rows:
        return 1.0
    total = 0.0
    n = 0
    for conf, pnl in rows:
        if conf is None or pnl is None:
            continue
        win = 1.0 if pnl > 0 else 0.0
        total += (float(conf) - win) ** 2
        n += 1
    return total / n if n else 1.0


def health_report() -> Dict[str, Any]:
    n = _trade_count_30d()
    brier = _brier_score_30d()
    cold_start = n < 50  # bypass only while bootstrapping
    well_calibrated = brier < 0.18  # benchmark: random=0.25, perfect=0
    recommend_bypass = cold_start
    return {
        "trade_count_30d": n,
        "brier_score_30d": round(brier, 4),
        "cold_start": cold_start,
        "well_calibrated": well_calibrated,
        "recommend_bypass": recommend_bypass,
        "phase": "1_conditional" if cold_start else "2_real",
    }


def apply_recommendation(dry_run: bool = True) -> Dict[str, Any]:
    """Update PARAM_REGISTRY calibrator.bypass live (or dry-run)."""
    rep = health_report()
    target = 1.0 if rep["recommend_bypass"] else 0.0
    if dry_run:
        return {**rep, "action": f"would_set_bypass_{target}"}
    try:
        from neural_organism import PARAM_REGISTRY  # type: ignore

        if "calibrator.bypass" in PARAM_REGISTRY:
            PARAM_REGISTRY["calibrator.bypass"]["default"] = target
        else:
            PARAM_REGISTRY["calibrator.bypass"] = {
                "organ": "calibrator",
                "default": target,
                "min": 0.0,
                "max": 1.0,
            }
        return {**rep, "action": f"set_bypass_{target}"}
    except Exception as e:
        return {**rep, "action": "failed", "error": str(e)}
