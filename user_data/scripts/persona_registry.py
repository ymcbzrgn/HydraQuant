"""Phase 30 C.12 — Persona JSON registry for agent_pool.

Loads persona configs from user_data/data/personas/*.json and exposes them
to agent_pool as runtime-pluggable agents. Hot-reload supported (per-call file
mtime check).
"""
from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

PERSONA_DIR = Path(__file__).parent.parent / "data" / "personas"
_CACHE: Dict[str, Dict[str, Any]] = {}
_MTIMES: Dict[str, float] = {}
_LOCK = threading.RLock()


def _load_one(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(path.read_text())
    except Exception as e:
        logger.error(f"[PersonaRegistry] {path}: {e}")
        return None


def reload() -> int:
    if not PERSONA_DIR.is_dir():
        return 0
    n = 0
    with _LOCK:
        for f in PERSONA_DIR.glob("*.json"):
            try:
                mt = f.stat().st_mtime
            except Exception:
                continue
            if _MTIMES.get(str(f)) == mt:
                continue
            persona = _load_one(f)
            if not persona:
                continue
            aid = persona.get("agent_id") or f.stem
            _CACHE[aid] = persona
            _MTIMES[str(f)] = mt
            n += 1
    return n


def get(agent_id: str) -> Optional[Dict[str, Any]]:
    reload()
    return _CACHE.get(agent_id)


def list_all() -> List[Dict[str, Any]]:
    reload()
    return list(_CACHE.values())
