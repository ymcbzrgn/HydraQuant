"""Phase 30 A.4 — Tool result disk persist + 2K preview.

Large LLM tool outputs (RAG retrievals, embedding queries, full agent debate
transcripts) are persisted to user_data/data/tool_results/<date>/<tool>_<ts>.txt.
A 2K-char preview + path is written to `tool_results` SQLite table for fast
indexing.
"""
from __future__ import annotations

import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)
_BASE = Path(__file__).parent.parent / "data" / "tool_results"
_LOCK = threading.RLock()
PREVIEW_LEN = 2000


def store(tool_name: str, full_text: str, kind: str = "result") -> Optional[str]:
    if not full_text:
        return None
    now = datetime.now(timezone.utc)
    day = now.strftime("%Y%m%d")
    ts = now.strftime("%H%M%S_%f")
    safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in tool_name)[:64]
    out_dir = _BASE / day
    with _LOCK:
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{safe}_{kind}_{ts}.txt"
        try:
            path.write_text(full_text, encoding="utf-8", errors="replace")
        except Exception as e:
            logger.error(f"[ToolResultStore] write failed: {e}")
            return None
    preview = full_text[:PREVIEW_LEN]
    try:
        from db import AI_DB_PATH, get_db_connection

        with get_db_connection(AI_DB_PATH) as conn:
            conn.execute(
                """INSERT INTO tool_results (tool_name, preview, full_path, byte_size, kind)
                   VALUES (?, ?, ?, ?, ?)""",
                (tool_name, preview, str(path), len(full_text.encode("utf-8")), kind),
            )
            conn.commit()
    except Exception:
        pass
    return str(path)


def cleanup_old(days: int = 21) -> int:
    if not _BASE.exists():
        return 0
    import time
    cutoff = time.time() - days * 86400
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
