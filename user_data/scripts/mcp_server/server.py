"""Phase 30 C.8 — HydraQuant MCP control plane.

Exposes HydraQuant DB + state as MCP tools:
- query_trades(filters) -> recent trades
- query_decisions(pair, since) -> ai_decisions
- query_telemetry(kind_prefix) -> telemetry_events
- get_kpi_rollup(hours) -> kpi_rollup_hourly
- get_promotion_status() -> D.9 gate snapshot
- emit_workflow_event(kind, payload) -> trade_event_emitter

Implementation: stdlib HTTP server (no external mcp dependency required).
Operator-side LLM clients (Claude Code, Cursor, etc.) connect via MCP protocol
or direct HTTP for ad-hoc queries.
"""
from __future__ import annotations

import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Callable, Dict

logger = logging.getLogger(__name__)


def _query_trades(filters: Dict[str, Any]) -> Any:
    import sqlite3
    pair = filters.get("pair")
    limit = int(filters.get("limit", 50))
    sql = "SELECT id, pair, is_short, open_date, close_date, close_profit FROM trades"
    args: list = []
    if pair:
        sql += " WHERE pair = ?"
        args.append(pair)
    sql += f" ORDER BY id DESC LIMIT {limit}"
    try:
        with sqlite3.connect("/root/freqtrade/user_data/tradesv3.sqlite") as conn:
            return [list(r) for r in conn.execute(sql, args).fetchall()]
    except Exception as e:
        return {"error": str(e)}


def _query_decisions(filters: Dict[str, Any]) -> Any:
    from db import AI_DB_PATH, get_db_connection
    pair = filters.get("pair")
    since_hours = int(filters.get("since_hours", 24))
    with get_db_connection(AI_DB_PATH) as conn:
        sql = ("SELECT id, pair, signal_type, confidence, regime, outcome_pnl, timestamp "
               "FROM ai_decisions WHERE timestamp >= datetime('now', ?)")
        args: list = [f"-{since_hours} hours"]
        if pair:
            sql += " AND pair = ?"
            args.append(pair)
        sql += " ORDER BY id DESC LIMIT 100"
        return [list(r) for r in conn.execute(sql, args).fetchall()]


def _query_telemetry(filters: Dict[str, Any]) -> Any:
    from telemetry import query
    return query(kind_prefix=filters.get("kind_prefix", ""),
                 since_hours=int(filters.get("since_hours", 24)),
                 limit=int(filters.get("limit", 200)))


def _kpi_rollup(filters: Dict[str, Any]) -> Any:
    from db import AI_DB_PATH, get_db_connection
    hours = int(filters.get("hours", 24))
    with get_db_connection(AI_DB_PATH) as conn:
        return [list(r) for r in conn.execute(
            f"""SELECT hour_bucket, n_trades, wins, losses, pnl_sum, n_anomalies
               FROM kpi_rollup_hourly ORDER BY hour_bucket DESC LIMIT {hours}"""
        ).fetchall()]


def _promotion_status(filters: Dict[str, Any]) -> Any:
    from promotion_gate import evaluate_gate
    r = evaluate_gate(window_days=int(filters.get("window_days", 14)))
    return {"passed": r.passed, "eligibility_pct": r.eligibility_pct,
            "blocked_by": r.blocked_by, "metrics": r.metrics}


def _emit_event(filters: Dict[str, Any]) -> Any:
    from trade_event_emitter import emit
    emit(filters.get("kind", "external.event"), filters.get("payload", {}))
    return {"ok": True}


TOOLS: Dict[str, Callable[[Dict[str, Any]], Any]] = {
    "query_trades": _query_trades,
    "query_decisions": _query_decisions,
    "query_telemetry": _query_telemetry,
    "kpi_rollup": _kpi_rollup,
    "promotion_status": _promotion_status,
    "emit_event": _emit_event,
}


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        logger.debug("[MCP] " + fmt, *args)

    def do_GET(self):
        if self.path == "/tools":
            body = json.dumps({"tools": list(TOOLS.keys())}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_response(404)
        self.end_headers()

    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", "0"))
            blob = self.rfile.read(length)
            req = json.loads(blob.decode() or "{}")
            tool = req.get("tool", "")
            params = req.get("params", {}) or {}
            fn = TOOLS.get(tool)
            if fn is None:
                self._json(404, {"error": f"unknown tool {tool}"})
                return
            result = fn(params)
            self._json(200, {"result": result})
        except Exception as e:
            self._json(500, {"error": str(e)})

    def _json(self, status: int, payload: Dict[str, Any]) -> None:
        body = json.dumps(payload, default=str).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def serve(host: str = "127.0.0.1", port: int = 8893) -> None:
    server = HTTPServer((host, port), _Handler)
    logger.info(f"[MCP] serving on http://{host}:{port}")
    server.serve_forever()


def serve_in_background(host: str = "127.0.0.1", port: int = 8893) -> threading.Thread:
    t = threading.Thread(target=serve, args=(host, port), daemon=True, name="mcp-server")
    t.start()
    return t


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    serve()
