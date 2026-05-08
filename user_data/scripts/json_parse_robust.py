"""Phase 30 A.14 — 3-step JSON parse with failure tolerance.

Step 1: json.loads
Step 2: Strip code fence + retry
Step 3: Tolerant repair (single-quote -> double, trailing commas, common LLM artifacts)

Failures persisted to parse_failures tablo for offline analysis.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Optional, Tuple

logger = logging.getLogger(__name__)

CODE_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```", re.IGNORECASE)
TRAILING_COMMA_RE = re.compile(r",(\s*[}\]])")
SINGLE_QUOTE_KEY_RE = re.compile(r"'([^']+)'\s*:")
SINGLE_QUOTE_STR_RE = re.compile(r":\s*'([^']*)'")


def _record_failure(source: str, raw: str, error: str, recovery: str = "", recovered: bool = False) -> None:
    try:
        from db import AI_DB_PATH, get_db_connection

        with get_db_connection(AI_DB_PATH) as conn:
            conn.execute(
                """INSERT INTO parse_failures
                   (source, raw_text, error_text, recovery_method, recovered)
                   VALUES (?, ?, ?, ?, ?)""",
                (source, raw[:4000], error[:1000], recovery, int(recovered)),
            )
            conn.commit()
    except Exception:
        pass


def _step1(s: str) -> Any:
    return json.loads(s)


def _step2(s: str) -> Any:
    m = CODE_FENCE_RE.search(s)
    inner = m.group(1) if m else s
    inner = inner.strip()
    return json.loads(inner)


def _step3(s: str) -> Any:
    inner = s
    m = CODE_FENCE_RE.search(inner)
    if m:
        inner = m.group(1)
    inner = inner.strip()
    inner = TRAILING_COMMA_RE.sub(r"\1", inner)
    inner = SINGLE_QUOTE_KEY_RE.sub(r'"\1":', inner)
    inner = SINGLE_QUOTE_STR_RE.sub(r': "\1"', inner)
    inner = inner.replace("\n", " ").replace("\r", "")
    return json.loads(inner)


def parse_json(text: str, source: str = "unknown") -> Tuple[Optional[Any], str]:
    """Returns (parsed_object, recovery_method).

    recovery_method in {"step1", "step2", "step3", "failed"}.
    On failure, parsed_object is None and parse_failures row is written.
    """
    if text is None:
        _record_failure(source, "", "input is None")
        return None, "failed"
    text = text.strip()
    last_err = ""
    for step_fn, label in [(_step1, "step1"), (_step2, "step2"), (_step3, "step3")]:
        try:
            return step_fn(text), label
        except Exception as e:
            last_err = f"{label}: {e}"
    logger.warning(f"[json_parse_robust] {source} all 3 steps failed: {last_err}")
    _record_failure(source, text, last_err, recovery="failed", recovered=False)
    return None, "failed"
