"""
decision_transformer.py — Phase 27 Task 21 (C2 Ajani)

Decision Transformer pipeline (GPT-2 + LoRA).

Per the alpha doc we treat trades as a (return-to-go, state, action) sequence
and fine-tune GPT-2 via LoRA (rank 16, ~900K trainable params). Inference is
deferred — right now this module ONLY builds the data pipeline and the
training infrastructure so the scheduler Sunday job can start producing
checkpoints. Once we have a model with positive out-of-sample validation,
Task 21b wires it into the sizing pipeline.

ALPHA doc §Task 21 / Teorik Temeller C2 covers the theory. CPU feasibility
(no GPU on hydra): 2-10h training / ~600MB peak RAM for 900K trainable
params — well within our Sunday-night LoRA window.

Integration surfaces exposed today:
  build_corpus()        — stream decision_contract-shaped rows from
                          `ai_decisions` into (return-to-go, state, action)
                          tuples; used by the scheduler training job and by
                          future inference callers.
  train_lora()          — STUB that records "pending_impl" + sample count
                          so the scheduler reports progress. Full LoRA wiring
                          lives in Sprint 3B task 21b.
  latest_checkpoint()   — enumerate `user_data/models/dt_*.pt` sidecars so
                          a future `predict()` knows which file to load.

Deliberately NO inference hook into sizing here — that connection is gated
on a validated OOS Sharpe improvement.
"""

from __future__ import annotations

import json
import logging
import math
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
logger = logging.getLogger("decision_transformer")

from ai_config import AI_DB_PATH
from db import get_db_connection, init_db


MIN_TRADES_FOR_TRAIN = 500      # ALPHA doc C2: 500 = minimum, 1289 = current.
LORA_RANK = 16
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


def _ensure_model_dir() -> str:
    os.makedirs(MODEL_DIR, exist_ok=True)
    return MODEL_DIR


def _tokenize_state(row: Dict[str, Any]) -> Dict[str, Any]:
    """Decision-contract-shaped trimming of an ai_decisions row.

    The full state would include chart features, pheromone snapshot, hormones,
    etc. — for the LoRA budget we project down to the fields the contract
    actually captures today. This is intentionally CONSERVATIVE: we can expand
    the state vector after the model trains well on a minimal view.
    """
    return {
        "confidence": float(row.get("confidence") or 0.5),
        "trust": float(row.get("trust_score_at_decision") or 0.5),
        "regime": (row.get("regime") or "_global")[:16],
        "model": (row.get("model_used") or "evidence")[:24],
    }


def _action_from_signal(signal_type: Optional[str]) -> int:
    """Map signal string → discrete action token (BULLISH=1, BEARISH=2, else 0)."""
    if not signal_type:
        return 0
    up = signal_type.upper()
    if up == "BULLISH":
        return 1
    if up == "BEARISH":
        return 2
    return 0


def build_corpus(min_trades: int = MIN_TRADES_FOR_TRAIN) -> Dict[str, Any]:
    """Pull resolved trades and emit Decision-Transformer training tuples.

    Each tuple is (return_to_go, state_dict, action_int). The corpus is
    organised oldest→newest so we can compute return-to-go in one pass.
    """
    try:
        conn = get_db_connection(AI_DB_PATH)
        rows = conn.execute("""
            SELECT id, timestamp, pair, signal_type, confidence,
                   trust_score_at_decision, regime, outcome_pnl,
                   model_used
            FROM ai_decisions
            WHERE outcome_pnl IS NOT NULL
            ORDER BY timestamp ASC
        """).fetchall()
        conn.close()
    except Exception as e:
        return {"status": "error", "reason": f"db: {e}", "n_tuples": 0}

    if len(rows) < min_trades:
        return {
            "status": "insufficient_data",
            "n_trades": len(rows),
            "min_trades": min_trades,
        }

    # Return-to-go: suffix sum of outcome_pnl — at each step, "what is the
    # total reward I'm going to earn from here onward?" (Chen et al. 2021).
    pnls = [float(r["outcome_pnl"] or 0.0) for r in rows]
    suffix = [0.0] * len(pnls)
    acc = 0.0
    for i in range(len(pnls) - 1, -1, -1):
        acc += pnls[i]
        suffix[i] = acc

    corpus: List[Dict[str, Any]] = []
    for rtg, row in zip(suffix, rows):
        corpus.append({
            "return_to_go": round(rtg, 4),
            "state": _tokenize_state(dict(row)),
            "action": _action_from_signal(row["signal_type"]),
            "reward": float(row["outcome_pnl"] or 0.0),
            "pair": row["pair"],
            "timestamp": row["timestamp"],
        })
    return {"status": "ok", "n_tuples": len(corpus), "corpus_preview": corpus[:2]}


def train_lora(corpus: Optional[List[Dict[str, Any]]] = None,
               epochs: int = 3,
               lr: float = 1e-4) -> Dict[str, Any]:
    """Fine-tune GPT-2 124M with LoRA rank=16 on the trade corpus.

    STUB — writes a pending sidecar so the scheduler logs forward progress.
    Full implementation is Sprint 3B task 21b (requires `peft` + `transformers`
    on the server, which is heavy for free-tier hydra).
    """
    if corpus is None:
        result = build_corpus()
        if result["status"] != "ok":
            return result
        n = result["n_tuples"]
    else:
        n = len(corpus)

    if n < MIN_TRADES_FOR_TRAIN:
        return {"status": "insufficient_data", "n_tuples": n}

    # Fine-tune body intentionally unimplemented — exposes the interface so
    # scheduler + future predict() know the flag location without shipping
    # a half-trained model to production.
    sidecar = os.path.join(_ensure_model_dir(), "dt_lora_pending.flag")
    try:
        with open(sidecar, "w") as f:
            f.write(
                f"pending_impl\nn_tuples={n}\nepochs={epochs}\n"
                f"lr={lr}\nlora_rank={LORA_RANK}\n"
                f"created_at={datetime.now(tz=timezone.utc).isoformat()}\n"
            )
    except Exception:
        pass

    return {
        "status": "pending_impl",
        "n_tuples": n,
        "epochs": epochs,
        "lora_rank": LORA_RANK,
        "message": "LoRA training body planned Sprint 3B task 21b",
    }


def latest_checkpoint() -> Optional[str]:
    """Return the freshest `dt_*.pt` checkpoint path or None."""
    try:
        _ensure_model_dir()
        candidates = sorted(
            (os.path.join(MODEL_DIR, f) for f in os.listdir(MODEL_DIR)
             if f.startswith("dt_") and f.endswith(".pt")),
            key=lambda p: os.path.getmtime(p),
            reverse=True,
        )
        return candidates[0] if candidates else None
    except Exception:
        return None


# Module-level singleton is not needed — the pipeline is stateless. We expose
# `scheduled_cycle()` as the single entrypoint the scheduler calls.
def scheduled_cycle() -> Dict[str, Any]:
    """Entrypoint used by scheduler `_dt_training_cycle` (Sunday nights)."""
    init_db()
    corpus = build_corpus()
    if corpus["status"] != "ok":
        return corpus
    return train_lora()
