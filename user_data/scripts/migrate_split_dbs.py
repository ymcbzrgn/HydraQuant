"""FAZ 4 split-DB migration — direct copy + drop (no VIEW/TRIGGER).

Moves 12 hot tables from ai_data.sqlite to 3 topic-specific files. The
broken VIEW+TRIGGER transparent approach in migrate_transparent_v2.py is
NOT used here because SQLite does not allow normal views to reference
ATTACH'd database objects (confirmed 2026-05-26).

After this script runs, callers MUST schema-qualify every reference to a
moved table: `llm.llm_calls`, `obs.protection_logs`, `pat.ohlcv_patterns`,
etc. The default-pool _create_connection in db.py ATTACHes the split DBs
under those aliases so qualified queries resolve transparently.

Safety:
  - Idempotent. If a table already lives in the split DB and is absent
    from main, the script reports it and skips.
  - Verifies src_count == dst_count BEFORE dropping main.
  - Backs up ai_data.sqlite caller-side BEFORE invocation. This script
    does NOT take a backup itself.
  - Run with all 5 services STOPPED.
"""
from __future__ import annotations

import os
import sqlite3
import sys
import time

BASE_DIR = "/root/freqtrade/user_data/db"
AI_DB = os.path.join(BASE_DIR, "ai_data.sqlite")
LLM_DB = os.path.join(BASE_DIR, "llm_data.sqlite")
OBS_DB = os.path.join(BASE_DIR, "observability.sqlite")
PAT_DB = os.path.join(BASE_DIR, "patterns.sqlite")

PLAN = [
    (LLM_DB, "llm", ["llm_calls", "llm_response_cache", "llm_dead_models"]),
    (OBS_DB, "obs", [
        "telemetry_events", "system_metrics", "protection_logs",
        "evidence_audit_log", "rag_endpoint_latency",
    ]),
    (PAT_DB, "pat", [
        "ohlcv_patterns", "counterfactual_results", "forgone_profit",
        "pattern_trades",
    ]),
]


def _p(msg: str) -> None:
    sys.stdout.write(msg + "\n")
    sys.stdout.flush()


def _ensure_split_db(path: str) -> None:
    c = sqlite3.connect(path, timeout=60)
    try:
        c.execute("PRAGMA journal_mode=WAL")
        c.execute("PRAGMA synchronous=NORMAL")
        c.commit()
    finally:
        c.close()


def _table_exists(conn, name, schema="main"):
    row = conn.execute(
        f"SELECT 1 FROM {schema}.sqlite_master WHERE type='table' AND name=?",
        (name,),
    ).fetchone()
    return row is not None


def _row_count(conn, fq):
    return conn.execute(f"SELECT COUNT(*) FROM {fq}").fetchone()[0]


def _get_create(conn, name, schema="main"):
    row = conn.execute(
        f"SELECT sql FROM {schema}.sqlite_master WHERE type='table' AND name=?",
        (name,),
    ).fetchone()
    return row[0] if row else None


def _get_indexes(conn, name, schema="main"):
    rows = conn.execute(
        f"SELECT name, sql FROM {schema}.sqlite_master "
        f"WHERE type='index' AND tbl_name=? AND sql IS NOT NULL",
        (name,),
    ).fetchall()
    return [(r[0], r[1]) for r in rows]


def _qualify_create(ddl, alias):
    head = "CREATE TABLE"
    if not ddl.strip().upper().startswith(head):
        return ddl
    rest = ddl.strip()[len(head):].lstrip()
    if rest.upper().startswith("IF NOT EXISTS"):
        rest = rest[len("IF NOT EXISTS"):].lstrip()
    return f"CREATE TABLE IF NOT EXISTS {alias}.{rest}"


def _qualify_index(ddl, alias):
    upper = ddl.strip().upper()
    if upper.startswith("CREATE UNIQUE INDEX"):
        prefix = "CREATE UNIQUE INDEX"
    elif upper.startswith("CREATE INDEX"):
        prefix = "CREATE INDEX"
    else:
        return ddl
    rest = ddl.strip()[len(prefix):].lstrip()
    if rest.upper().startswith("IF NOT EXISTS"):
        rest = rest[len("IF NOT EXISTS"):].lstrip()
    return f"{prefix} IF NOT EXISTS {alias}.{rest}"


def main() -> int:
    if not os.path.exists(AI_DB):
        _p(f"FATAL: source missing: {AI_DB}")
        return 1

    for split_path, _alias, _tables in PLAN:
        _ensure_split_db(split_path)
        _p(f"target ready: {split_path}")
    _p("")

    src = sqlite3.connect(AI_DB, timeout=180)
    src.execute("PRAGMA busy_timeout=180000")
    src.row_factory = sqlite3.Row

    for split_path, alias, _tables in PLAN:
        src.execute(f"ATTACH DATABASE ? AS {alias}", (split_path,))
        _p(f"attached {alias} -> {split_path}")
    _p("")

    overall_ok = True
    try:
        for split_path, alias, tables in PLAN:
            for table in tables:
                _p(f"=== {table} -> {alias} ===")
                src_present = _table_exists(src, table, "main")
                dst_present = _table_exists(src, table, alias)

                if not src_present and dst_present:
                    n = _row_count(src, f"{alias}.{table}")
                    _p(f"  already moved (dst={n}, src absent) — skip")
                    continue
                if not src_present and not dst_present:
                    _p(f"  WARN: table missing on both sides — skip")
                    continue

                src_n = _row_count(src, f"main.{table}")

                if dst_present:
                    dst_n = _row_count(src, f"{alias}.{table}")
                    if dst_n != 0:
                        _p(f"  ABORT: dst has {dst_n} rows (src {src_n}) — manual review")
                        overall_ok = False
                        continue
                else:
                    ddl = _get_create(src, table, "main")
                    if not ddl:
                        _p(f"  ABORT: cannot read CREATE TABLE for {table}")
                        overall_ok = False
                        continue
                    src.execute(_qualify_create(ddl, alias))

                idx_list = _get_indexes(src, table, "main")
                for idx_name, idx_sql in idx_list:
                    try:
                        src.execute(_qualify_index(idx_sql, alias))
                    except Exception as e:
                        _p(f"    idx {idx_name} skip: {e}")

                t0 = time.time()
                src.execute(f"INSERT INTO {alias}.{table} SELECT * FROM main.{table}")
                src.commit()
                dst_n = _row_count(src, f"{alias}.{table}")
                dt = time.time() - t0
                _p(f"  src={src_n} -> {alias}.{table}={dst_n}  ({dt:.1f}s)")

                if dst_n != src_n:
                    _p(f"  ABORT: count mismatch — leaving main.{table} in place")
                    overall_ok = False
                    continue

                src.execute(f"DROP TABLE main.{table}")
                src.commit()
                _p(f"  main.{table} DROPPED")
    finally:
        for split_path, alias, _tables in PLAN:
            try:
                src.execute(f"DETACH DATABASE {alias}")
            except Exception:
                pass

    if overall_ok:
        _p("")
        _p("VACUUM ai_data.sqlite ...")
        t0 = time.time()
        src.execute("VACUUM")
        src.commit()
        _p(f"  done ({time.time()-t0:.1f}s)")

    src.close()

    _p("")
    _p("=== Final sizes ===")
    for p in (AI_DB, LLM_DB, OBS_DB, PAT_DB):
        if os.path.exists(p):
            sz = os.path.getsize(p) / (1024 * 1024)
            _p(f"  {os.path.basename(p):<26} {sz:>8.1f} MB")

    return 0 if overall_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
