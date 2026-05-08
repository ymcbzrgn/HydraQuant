"""Phase 30 B.17 — Supervised Fine-Tuning export pipeline.

Exports closed-trade outcomes as JSONL training data with tag namespace
(`hydra.trade.v1.{regime}.{outcome}`) for foundation model self-distillation.

Scheduler: daily cron at 04:00 UTC.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_OUT_DIR = Path(__file__).parent.parent / "data" / "sft_export"


def _outcome_tag(close_profit: float) -> str:
    if close_profit is None:
        return "unknown"
    if close_profit > 0.005:
        return "win"
    if close_profit < -0.005:
        return "loss"
    return "neutral"


def build_dataset(window_days: int = 30, out_dir: Optional[Path] = None) -> Optional[Path]:
    out_dir = Path(out_dir) if out_dir else DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
    out_path = out_dir / f"sft_trades_{ts}.jsonl"

    try:
        from db import AI_DB_PATH, get_db_connection

        with get_db_connection(AI_DB_PATH) as conn:
            rows = conn.execute(
                f"""SELECT t.id, t.pair, t.is_short, t.open_rate, t.close_rate,
                          t.close_profit, t.exit_reason, t.open_date, t.close_date,
                          d.confidence, d.regime, d.reasoning_summary
                   FROM trades t
                   LEFT JOIN ai_decisions d
                     ON d.pair = t.pair
                     AND DATETIME(d.timestamp) BETWEEN DATETIME(t.open_date, '-1 hour')
                                                  AND DATETIME(t.open_date, '+30 minutes')
                   WHERE t.close_date >= datetime('now', '-{int(window_days)} days')
                     AND t.close_profit IS NOT NULL"""
            ).fetchall()
    except Exception as e:
        logger.error(f"[SFT] db: {e}")
        return None

    n_written = 0
    with open(out_path, "w", encoding="utf-8") as fp:
        for (tid, pair, is_short, op, cp, profit, exit_reason,
             od, cd, conf, regime, summary) in rows:
            tag = f"hydra.trade.v1.{regime or 'unknown'}.{_outcome_tag(profit)}"
            example: Dict[str, Any] = {
                "tag": tag,
                "trade_id": tid,
                "pair": pair,
                "side": "short" if is_short else "long",
                "open_rate": op,
                "close_rate": cp,
                "close_profit": profit,
                "exit_reason": exit_reason,
                "open_date": od,
                "close_date": cd,
                "ai_confidence": conf,
                "regime_at_entry": regime,
                "reasoning_summary": (summary or "")[:1000],
            }
            fp.write(json.dumps(example, default=str) + "\n")
            n_written += 1
    logger.info(f"[SFT] wrote {n_written} examples to {out_path}")
    return out_path
