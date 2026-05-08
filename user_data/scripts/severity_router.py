"""Phase 30 A.19 — Severity-aware reporting router.

Single emit() entrypoint dispatches by severity:
- critical -> Telegram + log + DB persist
- error    -> log + DB persist
- warn     -> log + DB persist (sampled)
- info     -> log only (DEBUG-level if quiet hours)
- debug    -> log only

Used by autonomy_diagnostic, restart_monitor, anomaly_detector,
doom_loop, parse_failures, etc.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

SEVERITY_LEVELS = ("debug", "info", "warn", "error", "critical")
_SAMPLE_RATE = {"warn": 1.0, "info": 1.0}
_LAST_SEND: Dict[str, float] = {}
_DEDUP_WINDOW_SEC = 300  # critical/error per (kind, payload-hash) dedup
_LOCK = threading.RLock()


def _persist(kind: str, severity: str, source: str, payload: Optional[Dict[str, Any]]) -> None:
    try:
        from db import AI_DB_PATH, get_db_connection

        with get_db_connection(AI_DB_PATH) as conn:
            conn.execute(
                """INSERT INTO telemetry_events
                   (kind, severity, source_module, payload_json)
                   VALUES (?, ?, ?, ?)""",
                (kind, severity, source, json.dumps(payload or {}, default=str)),
            )
            conn.commit()
    except Exception as e:
        logger.error(f"[SeverityRouter] DB persist failed: {e}")


def _telegram(kind: str, message: str, payload: Optional[Dict[str, Any]]) -> None:
    """Phase 30 A.19 — Critical alert dispatcher.

    Uses telegram_notifier.AITelegramNotifier().send_alert() — the canonical
    entrypoint. Earlier wiring referenced non-existent send_message_sync/notify
    symbols and silently swallowed every CRITICAL event (Audit Finding #1).
    """
    try:
        from telegram_notifier import AITelegramNotifier

        text = f"[{kind.upper()}] {message}"
        if payload:
            preview = ", ".join(f"{k}={v}" for k, v in list(payload.items())[:5])
            text += f"\n{preview}"
        AITelegramNotifier().send_alert(text, level="CRITICAL")
    except Exception as e:
        logger.error(f"[SeverityRouter] Telegram dispatch failed: {e}")


def emit(
    kind: str,
    severity: str,
    message: str = "",
    payload: Optional[Dict[str, Any]] = None,
    source_module: str = "",
) -> None:
    """Route a single event by severity."""
    severity = (severity or "info").lower()
    if severity not in SEVERITY_LEVELS:
        severity = "info"

    log_msg = f"[{kind}] {message}"
    log_fn = {
        "debug": logger.debug,
        "info": logger.info,
        "warn": logger.warning,
        "error": logger.error,
        "critical": logger.critical,
    }.get(severity, logger.info)
    log_fn(log_msg)

    if severity in ("debug",):
        return

    dedup_key = f"{kind}|{severity}|{message}"
    with _LOCK:
        last = _LAST_SEND.get(dedup_key, 0.0)
        now = time.time()
        if severity in ("critical", "error") and (now - last) < _DEDUP_WINDOW_SEC:
            return
        _LAST_SEND[dedup_key] = now

    _persist(kind, severity, source_module, payload)

    if severity == "critical":
        _telegram(kind, message, payload)


def reset_dedup_for_tests() -> None:
    with _LOCK:
        _LAST_SEND.clear()
