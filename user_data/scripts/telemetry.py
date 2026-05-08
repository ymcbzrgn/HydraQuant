"""Phase 30 B.18 — Telemetry single module.

Replaces scattered evidence_audit_log + agent_performance + organism_audit
ad-hoc inserts with a single record_*(kind=...) entry. Internally dispatches
to the legacy tables AND writes telemetry_events for unified queries.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def record(
    kind: str,
    severity: str = "info",
    source_module: str = "",
    payload: Optional[Dict[str, Any]] = None,
    legacy_dispatch: bool = True,
) -> None:
    """Single canonical entry for all telemetry."""
    payload = payload or {}
    try:
        from db import AI_DB_PATH, get_db_connection

        with get_db_connection(AI_DB_PATH) as conn:
            conn.execute(
                """INSERT INTO telemetry_events
                   (kind, severity, source_module, payload_json)
                   VALUES (?, ?, ?, ?)""",
                (kind, severity, source_module, json.dumps(payload, default=str)),
            )
            if legacy_dispatch:
                _dispatch_legacy(conn, kind, source_module, payload)
            conn.commit()
    except Exception as e:
        logger.error(f"[Telemetry] record failed: {e}")


def _dispatch_legacy(conn, kind: str, source: str, payload: Dict[str, Any]) -> None:
    try:
        if kind.startswith("evidence."):
            conn.execute(
                """INSERT INTO evidence_audit_log
                   (pair, score_json, regime, timestamp)
                   VALUES (?, ?, ?, datetime('now'))""",
                (payload.get("pair", ""), json.dumps(payload, default=str),
                 payload.get("regime", "")),
            )
        elif kind.startswith("agent."):
            conn.execute(
                """INSERT INTO agent_performance
                   (agent_type, regime, score, timestamp)
                   VALUES (?, ?, ?, datetime('now'))""",
                (payload.get("agent_type", source), payload.get("regime", "any"),
                 float(payload.get("score", 0.0))),
            )
        elif kind.startswith("organism."):
            conn.execute(
                """INSERT INTO organism_audit
                   (event_type, details_json, timestamp)
                   VALUES (?, ?, datetime('now'))""",
                (kind, json.dumps(payload, default=str)),
            )
    except Exception:
        pass


def query(kind_prefix: str = "", since_hours: int = 24, limit: int = 200) -> list:
    try:
        from db import AI_DB_PATH, get_db_connection

        with get_db_connection(AI_DB_PATH) as conn:
            sql = (
                "SELECT kind, severity, source_module, payload_json, ts "
                "FROM telemetry_events "
                "WHERE ts >= datetime('now', ?)"
            )
            args = [f"-{int(since_hours)} hours"]
            if kind_prefix:
                sql += " AND kind LIKE ?"
                args.append(kind_prefix + "%")
            sql += f" ORDER BY id DESC LIMIT {int(limit)}"
            return [
                {"kind": r[0], "severity": r[1], "source": r[2],
                 "payload": json.loads(r[3] or "{}"), "ts": r[4]}
                for r in conn.execute(sql, args).fetchall()
            ]
    except Exception as e:
        logger.error(f"[Telemetry] query: {e}")
        return []
