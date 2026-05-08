"""Phase 30 A.25 — Workflow event bus.

Pub-sub for trade lifecycle events. Subscribers:
- DB persist (workflow_events table)
- Telegram notifier (filtered by kind)
- Telemetry single (B.18)
- Idle scheduler activity reporter (A.22)

Decouples HydraSizer.confirm_trade_entry / custom_exit from notification logic.
"""
from __future__ import annotations

import json
import logging
import threading
from collections import defaultdict
from typing import Any, Callable, Dict, List

logger = logging.getLogger(__name__)

EVENT_KINDS = (
    "trade.opened",
    "trade.closed",
    "trade.modified",
    "trade.rejected_pre_check",
    "trade.exit_warning",
    "agent.veto",
    "agent.recovered",
    "system.alert",
)

_SUBSCRIBERS: Dict[str, List[Callable]] = defaultdict(list)
_LOCK = threading.RLock()


def subscribe(kind: str, callback: Callable[[Dict[str, Any]], None]) -> None:
    with _LOCK:
        _SUBSCRIBERS[kind].append(callback)


def emit(kind: str, payload: Dict[str, Any]) -> None:
    payload = dict(payload or {})
    payload["kind"] = kind
    try:
        from db import AI_DB_PATH, get_db_connection

        with get_db_connection(AI_DB_PATH) as conn:
            conn.execute(
                """INSERT INTO workflow_events (kind, payload_json, consumer)
                   VALUES (?, ?, ?)""",
                (kind, json.dumps(payload, default=str), "bus"),
            )
            conn.commit()
    except Exception as e:
        logger.error(f"[EventBus] DB persist failed: {e}")

    try:
        from idle_aware_scheduling import get_idle_scheduler
        get_idle_scheduler().report_activity(kind)
    except Exception:
        pass

    with _LOCK:
        cbs = list(_SUBSCRIBERS.get(kind, []))
    for cb in cbs:
        try:
            cb(payload)
        except Exception as e:
            logger.error(f"[EventBus] subscriber failed for {kind}: {e}")


def install_default_subscribers() -> None:
    """Wire severity_router + telegram for system.alert + trade.* events."""
    try:
        from severity_router import emit as severity_emit

        def _on_alert(p: Dict[str, Any]) -> None:
            severity_emit(
                kind=p.get("kind", "trade.event"),
                severity=p.get("severity", "info"),
                message=p.get("message", ""),
                payload=p,
            )

        for k in ("trade.opened", "trade.closed", "trade.rejected_pre_check", "system.alert"):
            subscribe(k, _on_alert)
    except Exception as e:
        logger.error(f"[EventBus] install default subscribers: {e}")


def reset_for_tests() -> None:
    with _LOCK:
        _SUBSCRIBERS.clear()
