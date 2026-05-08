"""Phase 30 D.6 — Foundation model self-distillation skeleton.

Pipeline (training executes OUT OF SESSION; multi-day):
1. Pull 3 years of trade outcomes + agent_performance + evidence_audit_log.
2. Build SFT dataset (B.17) with reasoning chains and outcomes.
3. Fine-tune small base model (e.g. Llama-3.2-3B) on dataset.
4. Eval on hold-out walk-forward backtest (C.3).
5. Promote checkpoint -> production via D.1 shadow Kelly cycle.

This module provides:
- prepare_distillation_dataset(window_days=1095)
- launch_training(config) — writes job spec for a separate training process
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

JOBS_DIR = Path(__file__).parent.parent / "models" / "self_distill"


@dataclass
class DistillationConfig:
    base_model: str = "meta-llama/Llama-3.2-3B-Instruct"
    n_epochs: int = 2
    learning_rate: float = 5e-6
    batch_size: int = 8
    seq_len: int = 4096
    window_days: int = 1095
    use_lora: bool = True


def prepare_dataset(cfg: DistillationConfig) -> Optional[Path]:
    out_dir = JOBS_DIR / f"dataset_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "train.jsonl"
    n = 0
    try:
        from db import AI_DB_PATH, get_db_connection

        with get_db_connection(AI_DB_PATH) as conn, open(jsonl_path, "w") as fp:
            cur = conn.execute(
                f"""SELECT d.pair, d.signal_type, d.confidence, d.regime,
                          d.reasoning_summary, t.close_profit, t.exit_reason
                   FROM ai_decisions d
                   LEFT JOIN trades t
                      ON t.pair = d.pair
                     AND DATETIME(d.timestamp) BETWEEN DATETIME(t.open_date, '-30 minutes')
                                                  AND DATETIME(t.open_date, '+30 minutes')
                   WHERE d.timestamp >= datetime('now', '-{int(cfg.window_days)} days')
                     AND d.reasoning_summary IS NOT NULL"""
            )
            for pair, sig, conf, regime, reasoning, profit, exit_r in cur:
                example = {
                    "instruction": f"Pair {pair} regime {regime}. Provide signal reasoning.",
                    "input": "",
                    "output": (reasoning or "")[:4000],
                    "label_signal": sig,
                    "label_confidence": conf,
                    "label_outcome_pnl": profit,
                    "label_exit_reason": exit_r,
                }
                fp.write(json.dumps(example, default=str) + "\n")
                n += 1
    except Exception as e:
        logger.error(f"[D.6] dataset prep failed: {e}")
        return None
    (out_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
    logger.info(f"[D.6] {n} examples -> {jsonl_path}")
    return jsonl_path


def launch_training(cfg: DistillationConfig) -> Optional[Path]:
    dataset = prepare_dataset(cfg)
    if not dataset:
        return None
    job_dir = JOBS_DIR / f"job_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    job_dir.mkdir(parents=True, exist_ok=True)
    spec = {
        "config": asdict(cfg),
        "dataset_path": str(dataset),
        "status": "queued_for_external_runner",
    }
    (job_dir / "spec.json").write_text(json.dumps(spec, indent=2))
    (job_dir / "STATUS.txt").write_text(
        "PREPARED — actual training (estimate 60-120 days) runs in a dedicated GPU process.\n"
        "Operator: schedule 'python -m foundation_models.distill_runner --spec=spec.json'\n"
        "on a GPU host with the dataset rsynced."
    )
    return job_dir
