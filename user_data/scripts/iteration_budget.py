"""Phase 30 B.13 — Iteration budget per HydraSizer parent / child run.

Each run_id (typically pair-cycle) gets:
- parent_max_iters: top-level decision iterations
- child_max_iters: per-agent retries
- token_budget: cap on total LLM tokens

Breach -> abort the run with structured error, persist budget_breached flag.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Dict, Optional

logger = logging.getLogger(__name__)

DEFAULT_PARENT_MAX = 6
DEFAULT_CHILD_MAX = 3
DEFAULT_TOKEN_BUDGET = 50_000


@dataclass
class _State:
    run_id: str
    parent_iters: int = 0
    child_iters: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    parent_max: int = DEFAULT_PARENT_MAX
    child_max: int = DEFAULT_CHILD_MAX
    token_budget: int = DEFAULT_TOKEN_BUDGET
    breached: bool = False
    breach_reason: str = ""


_RUNS: Dict[str, _State] = {}
_LOCK = threading.RLock()


def begin(run_id: str, parent_max: int = DEFAULT_PARENT_MAX,
          child_max: int = DEFAULT_CHILD_MAX,
          token_budget: int = DEFAULT_TOKEN_BUDGET) -> None:
    with _LOCK:
        _RUNS[run_id] = _State(run_id=run_id, parent_max=parent_max,
                               child_max=child_max, token_budget=token_budget)


def increment_parent(run_id: str) -> bool:
    with _LOCK:
        s = _RUNS.get(run_id)
        if not s:
            return True
        s.parent_iters += 1
        if s.parent_iters > s.parent_max:
            s.breached = True
            s.breach_reason = "parent_max_exceeded"
            _persist(s)
            return False
        _persist(s)
    return True


def increment_child(run_id: str) -> bool:
    with _LOCK:
        s = _RUNS.get(run_id)
        if not s:
            return True
        s.child_iters += 1
        if s.child_iters > s.child_max * max(1, s.parent_iters):
            s.breached = True
            s.breach_reason = "child_max_exceeded"
            _persist(s)
            return False
        _persist(s)
    return True


def add_tokens(run_id: str, tin: int, tout: int) -> bool:
    with _LOCK:
        s = _RUNS.get(run_id)
        if not s:
            return True
        s.tokens_in += int(tin or 0)
        s.tokens_out += int(tout or 0)
        if s.tokens_in + s.tokens_out > s.token_budget:
            s.breached = True
            s.breach_reason = "token_budget_exceeded"
            _persist(s)
            return False
        _persist(s)
    return True


def _persist(s: _State) -> None:
    try:
        from db import AI_DB_PATH, get_db_connection

        with get_db_connection(AI_DB_PATH) as conn:
            conn.execute(
                """INSERT INTO iteration_budget_state
                   (run_id, parent_iters, child_iters, tokens_in, tokens_out, budget_breached, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, datetime('now'))
                   ON CONFLICT(run_id) DO UPDATE SET
                       parent_iters=excluded.parent_iters,
                       child_iters=excluded.child_iters,
                       tokens_in=excluded.tokens_in,
                       tokens_out=excluded.tokens_out,
                       budget_breached=excluded.budget_breached,
                       updated_at=datetime('now')""",
                (s.run_id, s.parent_iters, s.child_iters, s.tokens_in, s.tokens_out, int(s.breached)),
            )
            conn.commit()
    except Exception:
        pass


def end(run_id: str) -> Optional[Dict]:
    with _LOCK:
        s = _RUNS.pop(run_id, None)
    if not s:
        return None
    return {
        "run_id": s.run_id,
        "parent_iters": s.parent_iters,
        "child_iters": s.child_iters,
        "tokens_total": s.tokens_in + s.tokens_out,
        "breached": s.breached,
        "breach_reason": s.breach_reason,
    }


def reset_for_tests() -> None:
    with _LOCK:
        _RUNS.clear()
