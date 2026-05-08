"""Phase 30 A.20 — Append-only JSONL scratchpad per scheduler job.

Each job_id gets its own file user_data/data/scratchpad/<job_id>_<ts>.jsonl.
Append-only: fsync on close. Index row in scratchpad_jobs tablo.
"""
from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)
_BASE = Path(__file__).parent.parent / "data" / "scratchpad"
_LOCK = threading.RLock()


class Scratchpad:
    """Per-job append-only JSONL writer."""

    def __init__(self, job_id: str):
        self.job_id = job_id
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in job_id)[:64]
        with _LOCK:
            _BASE.mkdir(parents=True, exist_ok=True)
            self.path = _BASE / f"{safe}_{ts}.jsonl"
        self.line_count = 0
        self.fp = open(self.path, "a", encoding="utf-8")
        self._db_register("started")

    def append(self, record: Dict[str, Any]) -> None:
        try:
            line = json.dumps(record, default=str, ensure_ascii=False)
            self.fp.write(line + "\n")
            self.line_count += 1
        except Exception as e:
            logger.error(f"[Scratchpad] append failed {self.job_id}: {e}")

    def close(self, status: str = "completed") -> None:
        try:
            self.fp.flush()
            import os
            os.fsync(self.fp.fileno())
            self.fp.close()
        except Exception:
            pass
        self._db_register(status, end=True)

    def _db_register(self, status: str, end: bool = False) -> None:
        try:
            from db import AI_DB_PATH, get_db_connection

            with get_db_connection(AI_DB_PATH) as conn:
                if end:
                    conn.execute(
                        """UPDATE scratchpad_jobs
                           SET ended_at=datetime('now'), status=?, line_count=?
                           WHERE scratchpad_path=?""",
                        (status, self.line_count, str(self.path)),
                    )
                else:
                    conn.execute(
                        """INSERT INTO scratchpad_jobs
                           (job_id, scratchpad_path, status)
                           VALUES (?, ?, ?)""",
                        (self.job_id, str(self.path), status),
                    )
                conn.commit()
        except Exception:
            pass


def wrap_job(job_fn, job_id: str):
    """Decorator-style wrapper that gives a Scratchpad to job_fn(scratchpad, *args)."""
    def wrapped(*args, **kwargs):
        sp = Scratchpad(job_id)
        try:
            sp.append({"event": "start", "args_repr": str(args)[:300]})
            result = job_fn(sp, *args, **kwargs)
            sp.append({"event": "end", "ok": True})
            sp.close("completed")
            return result
        except Exception as e:
            sp.append({"event": "exception", "error": str(e)[:500]})
            sp.close("failed")
            raise

    return wrapped


def cleanup_old(days: int = 30) -> int:
    if not _BASE.exists():
        return 0
    import time
    cutoff = time.time() - days * 86400
    n = 0
    for f in _BASE.glob("*.jsonl"):
        try:
            if f.stat().st_mtime < cutoff:
                f.unlink()
                n += 1
        except Exception:
            continue
    return n
