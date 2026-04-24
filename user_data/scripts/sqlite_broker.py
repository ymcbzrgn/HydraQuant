"""
sqlite_broker.py — Single-writer ZMQ broker for ai_data.sqlite.

Rationale: five HydraQuant processes (freqtrade, scheduler, rag,
models, ai-api) keep independent 4-connection pools against the same
WAL-mode SQLite file. SQLite serialises writers at the file level so
the 20 fd's spend most of their contention time in `database is
locked` retries — 1,950 retry-log lines in 17h of production. The
graph DB already solved the exact same problem via `_GrafeoBroker`
(graph_store.py:59-303): scheduler owns the file, everybody else
talks ZMQ REQ/REP. This module ports that template to ai_data.sqlite.

Protocol (JSON over ipc://):
    {"action": "ping"}                               → {"ok": True, "pong": True}
    {"action": "stats"}                              → {"ok": True, "writes": int, "batch": int}
    {"action": "write", "sql": str, "params": list,
     "want_lastrowid": bool}                         → {"ok": True, "lastrowid": int|None, "rowcount": int}
    {"action": "batch", "stmts": [{"sql":..., "params":...}]} → atomic txn
    {"action": "executemany", "sql": str,
     "seq": [list, list, ...]}                        → {"ok": True, "rowcount": int}

Client-side fallback: when the broker is unavailable the client
returns `{"ok": False, "error": "broker_unavailable"}` and the caller
sinks the write into the local `hot_writes` spool. A drain job
replays spool rows when the broker returns. Nothing is lost — the
organism keeps writing even when its own nervous system is cut.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import sys
import threading
import time
from typing import Any, Dict, List, Optional

sys.path.append(os.path.dirname(__file__))

logger = logging.getLogger(__name__)

try:
    import zmq
    ZMQ_AVAILABLE = True
except ImportError:
    ZMQ_AVAILABLE = False
    logger.warning("[SqliteBroker] pyzmq not installed — broker & client disabled")

from ai_config import AI_DB_PATH

WRITE_ENDPOINT = os.environ.get(
    "SQLITE_WRITE_ENDPOINT",
    "ipc:///tmp/sqlite_write_hydra.sock",
)


def _ensure_spool_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """CREATE TABLE IF NOT EXISTS hot_writes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sql_stmt TEXT NOT NULL,
            params_json TEXT,
            target_table TEXT,
            priority INTEGER DEFAULT 5,
            retry_count INTEGER DEFAULT 0,
            state TEXT DEFAULT 'pending',
            last_error TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            committed_at TEXT
        )"""
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS ix_hot_writes_state_pri "
        "ON hot_writes(state, priority DESC, id)"
    )


# ─── Broker ─────────────────────────────────────────────────────────────────

class _SqliteWriteBroker:
    """ZMQ REP broker that owns the ai_data.sqlite WRITE side. Hosted by
    the scheduler process (one instance per host)."""

    def __init__(self, db_path: str = AI_DB_PATH):
        self._db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._write_lock = threading.Lock()
        self._stats = {"writes": 0, "batch": 0, "errors": 0}

    def start(self) -> None:
        if self._running:
            return
        if not ZMQ_AVAILABLE:
            logger.warning("[SqliteBroker] pyzmq unavailable, broker skipped")
            return
        sock_path = WRITE_ENDPOINT.replace("ipc://", "")
        if os.path.exists(sock_path):
            try:
                os.unlink(sock_path)
            except OSError:
                pass
        try:
            self._conn = sqlite3.connect(
                self._db_path, timeout=30.0, check_same_thread=False,
            )
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA busy_timeout=30000")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.execute("PRAGMA wal_autocheckpoint=1000")
            _ensure_spool_schema(self._conn)
            self._conn.commit()
        except Exception as e:
            logger.error(f"[SqliteBroker] open failed: {e}")
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._serve, daemon=True, name="sqlite-write-broker")
        self._thread.start()
        logger.info(f"[SqliteBroker] listening on {WRITE_ENDPOINT}")

    def _serve(self) -> None:
        ctx = zmq.Context()
        socket = ctx.socket(zmq.REP)
        socket.setsockopt(zmq.RCVTIMEO, 5000)
        socket.bind(WRITE_ENDPOINT)
        while self._running:
            try:
                msg = socket.recv_string()
                req = json.loads(msg)
                resp = self._handle(req)
                socket.send_string(json.dumps(resp, default=str))
            except zmq.Again:
                continue
            except Exception as e:
                try:
                    socket.send_string(json.dumps({"ok": False, "error": str(e)}))
                except Exception:
                    pass
        socket.close()
        ctx.term()
        if self._conn:
            try:
                self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            except Exception:
                pass
            try:
                self._conn.close()
            except Exception:
                pass
        logger.info("[SqliteBroker] stopped")

    def _handle(self, req: Dict[str, Any]) -> Dict[str, Any]:
        action = req.get("action", "")
        try:
            if action == "ping":
                return {"ok": True, "pong": True}
            if action == "stats":
                return {"ok": True, **self._stats}
            if action == "write":
                return self._do_write(req)
            if action == "batch":
                return self._do_batch(req)
            if action == "executemany":
                return self._do_executemany(req)
            return {"ok": False, "error": f"unknown action: {action}"}
        except Exception as e:
            self._stats["errors"] += 1
            return {"ok": False, "error": str(e)}

    def _do_write(self, req: Dict[str, Any]) -> Dict[str, Any]:
        sql = req.get("sql") or ""
        params = req.get("params") or []
        want_last = bool(req.get("want_lastrowid", False))
        with self._write_lock:
            cur = self._conn.execute(sql, params)
            self._conn.commit()
            self._stats["writes"] += 1
            resp = {"ok": True, "rowcount": cur.rowcount}
            if want_last:
                resp["lastrowid"] = cur.lastrowid
            return resp

    def _do_batch(self, req: Dict[str, Any]) -> Dict[str, Any]:
        stmts = req.get("stmts") or []
        with self._write_lock:
            try:
                for s in stmts:
                    self._conn.execute(s.get("sql") or "", s.get("params") or [])
                self._conn.commit()
                self._stats["batch"] += 1
                return {"ok": True, "n_stmts": len(stmts)}
            except Exception as e:
                try:
                    self._conn.rollback()
                except Exception:
                    pass
                raise

    def _do_executemany(self, req: Dict[str, Any]) -> Dict[str, Any]:
        sql = req.get("sql") or ""
        seq = req.get("seq") or []
        with self._write_lock:
            cur = self._conn.executemany(sql, seq)
            self._conn.commit()
            self._stats["writes"] += 1
            return {"ok": True, "rowcount": cur.rowcount}

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=10)


_broker = _SqliteWriteBroker()


def start_sqlite_broker() -> None:
    """Scheduler-side entry. Safe to call more than once."""
    _broker.start()


def stop_sqlite_broker() -> None:
    _broker.stop()


# ─── Client ─────────────────────────────────────────────────────────────────

class WriteClient:
    """Process-local REQ client. Thread-safe (send_lock); auto-recovers
    from socket errors by tearing the socket down and reconnecting."""

    _instance: Optional["WriteClient"] = None
    _singleton_lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._singleton_lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        self._send_lock = threading.Lock()
        self._ctx = None
        self._socket = None
        self._alive: Optional[bool] = None
        self._last_probe = 0.0
        self._initialized = True
        if not ZMQ_AVAILABLE:
            logger.warning("[SqliteBroker:Client] pyzmq unavailable — fallback mode")
            return
        try:
            self._open_socket()
        except Exception as e:
            logger.debug(f"[SqliteBroker:Client] open_socket failed: {e}")

    def _open_socket(self) -> None:
        self._ctx = zmq.Context.instance()
        self._socket = self._ctx.socket(zmq.REQ)
        self._socket.setsockopt(zmq.RCVTIMEO, 10000)
        self._socket.setsockopt(zmq.SNDTIMEO, 5000)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.connect(WRITE_ENDPOINT)

    def _reset_socket(self) -> None:
        try:
            if self._socket is not None:
                self._socket.close()
        except Exception:
            pass
        self._socket = None
        try:
            self._open_socket()
        except Exception:
            self._socket = None

    def is_alive(self, cache_seconds: float = 30.0) -> bool:
        """Cached liveness probe. Cheap enough to call on the hot path."""
        if not ZMQ_AVAILABLE:
            return False
        now = time.time()
        if self._alive is not None and (now - self._last_probe) < cache_seconds:
            return self._alive
        self._last_probe = now
        resp = self._request({"action": "ping"})
        self._alive = bool(resp.get("ok") and resp.get("pong"))
        return self._alive

    def _request(self, req: Dict[str, Any]) -> Dict[str, Any]:
        if self._socket is None:
            return {"ok": False, "error": "broker_unavailable"}
        with self._send_lock:
            try:
                self._socket.send_string(json.dumps(req, default=str))
                resp = json.loads(self._socket.recv_string())
                return resp
            except zmq.Again:
                self._reset_socket()
                return {"ok": False, "error": "timeout"}
            except Exception as e:
                self._reset_socket()
                return {"ok": False, "error": str(e)}

    # ─ Public API ────────────────────────────────────────────────────────

    def write(self, sql: str, params: Optional[list] = None,
              want_lastrowid: bool = False) -> Dict[str, Any]:
        return self._request({
            "action": "write",
            "sql": sql,
            "params": list(params or []),
            "want_lastrowid": want_lastrowid,
        })

    def batch(self, stmts: List[Dict[str, Any]]) -> Dict[str, Any]:
        return self._request({"action": "batch", "stmts": stmts})

    def executemany(self, sql: str, seq: List[list]) -> Dict[str, Any]:
        return self._request({
            "action": "executemany",
            "sql": sql,
            "seq": [list(x) for x in seq],
        })


_client: Optional[WriteClient] = None


def get_write_client() -> WriteClient:
    global _client
    if _client is None:
        _client = WriteClient()
    return _client


# ─── Dead-letter spool helpers ──────────────────────────────────────────────

def spool_write(sql: str, params: Optional[list] = None,
                target_table: str = "", priority: int = 5,
                db_path: str = AI_DB_PATH) -> bool:
    """Persist a write to hot_writes when the broker is unreachable.
    Drain job (scheduler._sqlite_spool_drain_tick) replays pending rows
    when the broker reappears. Returns True on successful enqueue."""
    try:
        # Use a direct connection — if the shared pool is locked the
        # spool itself should still be writable via a fresh fd with WAL
        # + 30s busy_timeout.
        conn = sqlite3.connect(db_path, timeout=30.0, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=30000")
        _ensure_spool_schema(conn)
        conn.execute(
            "INSERT INTO hot_writes (sql_stmt, params_json, target_table, priority) "
            "VALUES (?, ?, ?, ?)",
            (sql, json.dumps(list(params or [])), target_table or "", int(priority)),
        )
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        logger.error(f"[SqliteBroker:Spool] write failed: {e}")
        return False


MAX_SPOOL_RETRIES = 20


def drain_spool(batch_size: int = 50, db_path: str = AI_DB_PATH) -> int:
    """Replay up to `batch_size` pending rows via the broker. Returns
    the number of successfully drained rows.

    Task 28.5: cap retry_count at MAX_SPOOL_RETRIES; rows that keep
    failing transition to state='dead' so they stop hogging each drain
    cycle. Protects against a bad INSERT (constraint violation on an
    unrecoverable row) being replayed every 60s forever.
    """
    client = get_write_client()
    if not client.is_alive():
        return 0
    drained = 0
    try:
        conn = sqlite3.connect(db_path, timeout=30.0, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        _ensure_spool_schema(conn)
        rows = conn.execute(
            "SELECT id, sql_stmt, params_json, retry_count FROM hot_writes "
            "WHERE state = 'pending' AND retry_count < ? "
            "ORDER BY priority DESC, id LIMIT ?",
            (MAX_SPOOL_RETRIES, int(batch_size)),
        ).fetchall()
        for row in rows:
            try:
                params = json.loads(row["params_json"] or "[]")
            except Exception:
                params = []
            resp = client.write(row["sql_stmt"], params=params)
            if resp.get("ok"):
                conn.execute(
                    "UPDATE hot_writes SET state='committed', "
                    "committed_at=CURRENT_TIMESTAMP WHERE id = ?",
                    (row["id"],),
                )
                drained += 1
            else:
                new_retry = int(row["retry_count"] or 0) + 1
                if new_retry >= MAX_SPOOL_RETRIES:
                    conn.execute(
                        "UPDATE hot_writes SET retry_count = ?, "
                        "state = 'dead', last_error = ? WHERE id = ?",
                        (new_retry, str(resp.get("error", ""))[:200], row["id"]),
                    )
                    logger.warning(
                        f"[SqliteBroker:Spool] row {row['id']} DEAD after "
                        f"{new_retry} retries — last_error={resp.get('error')}"
                    )
                else:
                    conn.execute(
                        "UPDATE hot_writes SET retry_count = ?, "
                        "last_error = ? WHERE id = ?",
                        (new_retry, str(resp.get("error", ""))[:200], row["id"]),
                    )
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"[SqliteBroker:Spool] drain failed: {e}")
    return drained
