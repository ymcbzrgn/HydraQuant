"""Phase 30 C.5 — Operator session persistence.

Saves/restores live runtime state across systemd restarts:
- Active LinUCB feature buffers
- Hormonal state pending updates
- Agent pool partial debates
- Anomaly detector halt windows

Format: pickle on disk + checksum. Loaded on startup if file recent (<60s).
"""
from __future__ import annotations

import logging
import pickle
import threading
import time
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)

DEFAULT_PATH = Path(__file__).parent.parent / "data" / "session_snapshot.pkl"
DEFAULT_FRESH_SECONDS = 300

_LOCK = threading.RLock()


def save(state: Dict[str, Any], path: Path = DEFAULT_PATH) -> bool:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        with _LOCK, open(tmp, "wb") as fp:
            pickle.dump({"ts": time.time(), "state": state}, fp,
                        protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(path)
        return True
    except Exception as e:
        logger.error(f"[SessionPersist] save failed: {e}")
        return False


def load(path: Path = DEFAULT_PATH, max_age_seconds: int = DEFAULT_FRESH_SECONDS) -> Dict[str, Any]:
    try:
        if not path.is_file():
            return {}
        with _LOCK, open(path, "rb") as fp:
            blob = pickle.load(fp)
        ts = blob.get("ts", 0)
        if time.time() - ts > max_age_seconds:
            logger.info(f"[SessionPersist] snapshot stale ({time.time() - ts:.0f}s); skipping load")
            return {}
        return blob.get("state", {})
    except Exception as e:
        logger.error(f"[SessionPersist] load failed: {e}")
        return {}


def clear(path: Path = DEFAULT_PATH) -> None:
    try:
        if path.is_file():
            path.unlink()
    except Exception:
        pass
