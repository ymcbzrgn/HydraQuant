"""Phase 30 A.11 — tee_and_hint raw response capture.

When LLM returns invalid/unparseable output, tee the raw response to disk
under user_data/data/raw_llm/<date>/<agent>_<ts>.txt for offline replay.
Hints (last 200 chars) attached to exception logs for debugging.
"""
from __future__ import annotations

import logging
import os
import threading
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)
_LOCK = threading.RLock()
_BASE = Path(__file__).parent.parent / "data" / "raw_llm"


def _ensure_dir(d: Path) -> None:
    d.mkdir(parents=True, exist_ok=True)


def tee(agent_or_source: str, raw_text: str, kind: str = "response") -> str:
    if not raw_text:
        return ""
    now = datetime.now(timezone.utc)
    day = now.strftime("%Y%m%d")
    ts = now.strftime("%H%M%S_%f")
    safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in agent_or_source)[:64]
    out_dir = _BASE / day
    with _LOCK:
        _ensure_dir(out_dir)
        path = out_dir / f"{safe}_{kind}_{ts}.txt"
        try:
            path.write_text(raw_text, encoding="utf-8", errors="replace")
        except Exception as e:
            logger.error(f"[TeeLogger] write failed {path}: {e}")
            return ""
    return str(path)


def hint(raw_text: str, n: int = 200) -> str:
    if not raw_text:
        return ""
    return raw_text[-n:].replace("\n", "\\n")


def cleanup_old(days: int = 14) -> int:
    if not _BASE.exists():
        return 0
    cutoff = datetime.now(timezone.utc).timestamp() - days * 86400
    n = 0
    for d in _BASE.iterdir():
        if not d.is_dir():
            continue
        try:
            for f in d.iterdir():
                if f.stat().st_mtime < cutoff:
                    f.unlink()
                    n += 1
            if not any(d.iterdir()):
                d.rmdir()
        except Exception:
            continue
    return n
