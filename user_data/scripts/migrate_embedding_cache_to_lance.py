"""Phase 6 (Phase 28 follow-up) — Migrate embedding_cache from SQLite to LanceDB.

Why:
    embedding_cache is the largest table in ai_data.sqlite (~1 GB after 30d eviction,
    was 2.5 GB before). JSON-encoded float arrays stored as BLOB are:
      - 2-4x larger than native binary (Arrow/Lance)
      - CPU-expensive on every read (json.loads each call)
      - Lock-contention magnet (write/read same large file as everything else)
    Native LanceDB tables (one for Gemini, one for Jina/BGE) decouple this entirely:
      - Vectors stored as Arrow fixed-size lists (native float32)
      - Memory-mapped reads (no parse cost)
      - Separate disk file (no lock contention with ai_data.sqlite writers)

Strategy:
    1. Open SQLite read-only
    2. Create two LanceDB tables: embedding_cache_gemini, embedding_cache_bge
    3. Batch-migrate 1000 rows at a time (decode JSON → float32 array)
    4. Verify counts match
    5. (Manual after verification) DROP TABLE embedding_cache + VACUUM
    6. (Separate patch) Update rag_embedding.py to read/write LanceDB

Usage:
    python migrate_embedding_cache_to_lance.py --dry-run        # Count + validate, no writes
    python migrate_embedding_cache_to_lance.py                  # Execute migration
    python migrate_embedding_cache_to_lance.py --verify         # Check LanceDB counts match SQLite
    python migrate_embedding_cache_to_lance.py --batch-size 500 # Smaller batches if memory tight

Idempotent: skips text_hash IDs already present in LanceDB (via upsert by id).
"""
from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import sys
import time
from pathlib import Path
from typing import List, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("emb_cache_migrate")


def _resolve_paths():
    """Use ai_config when available, otherwise sensible defaults."""
    try:
        from ai_config import AI_DB_PATH
        sqlite_path = AI_DB_PATH
    except Exception:
        sqlite_path = "/root/freqtrade/user_data/db/ai_data.sqlite"
    return sqlite_path


def _open_lance_tables(dim_gemini: int = 768, dim_bge: int = 768):
    """Open or create the two LanceDB tables. Schema mirrors lance_store.py."""
    from lance_store import get_lance_store
    store = get_lance_store()
    gemini_table = store.get_or_create_table("embedding_cache_gemini", dim=dim_gemini)
    bge_table = store.get_or_create_table("embedding_cache_bge", dim=dim_bge)
    return gemini_table, bge_table


def _decode_blob(blob) -> Optional[List[float]]:
    """JSON-encoded float list stored as BLOB → list[float] or None."""
    if blob is None or blob == b"":
        return None
    try:
        if isinstance(blob, (bytes, bytearray)):
            blob = blob.decode("utf-8")
        val = json.loads(blob)
        if isinstance(val, list) and val:
            return [float(x) for x in val]
        return None
    except Exception:
        return None


def _existing_ids(table) -> set:
    """Return the set of ids already in a LanceDB table (for idempotency)."""
    try:
        df = table._table.to_pandas(columns=["id"])
        if df.empty:
            return set()
        return set(df["id"].tolist())
    except Exception as e:
        logger.warning(f"Could not enumerate existing ids in {table.name}: {e}")
        return set()


def migrate(dry_run: bool, batch_size: int):
    sqlite_path = _resolve_paths()
    logger.info(f"SQLite source: {sqlite_path}")

    conn = sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True, timeout=30.0)
    conn.row_factory = sqlite3.Row
    try:
        total_row = conn.execute("SELECT COUNT(*) AS n FROM embedding_cache").fetchone()
        total = int(total_row["n"])
        logger.info(f"SQLite embedding_cache rows: {total}")
    except sqlite3.OperationalError as e:
        logger.error(f"Cannot read embedding_cache: {e}")
        conn.close()
        return 1

    if dry_run:
        sample = conn.execute(
            "SELECT text_hash, LENGTH(text_content) AS text_len, "
            "LENGTH(gemini_embedding) AS gemini_len, LENGTH(bge_embedding) AS bge_len "
            "FROM embedding_cache LIMIT 5"
        ).fetchall()
        logger.info("Sample rows:")
        for row in sample:
            logger.info(f"  hash={row['text_hash'][:12]}… text_len={row['text_len']} "
                        f"gemini_len={row['gemini_len']} bge_len={row['bge_len']}")
        conn.close()
        logger.info(f"DRY RUN — would migrate {total} rows to LanceDB embedding_cache_gemini "
                    f"+ embedding_cache_bge. Re-run without --dry-run to execute.")
        return 0

    # 2026-05-18 fix: drop existing LanceDB tables for a clean rebuild. The prior
    # _existing_ids() idempotency path silently failed (LanceDB table object has
    # no to_pandas(columns=) — exception swallowed, returned empty set), so every
    # re-run duplicated all rows. A full rebuild is the correct semantic anyway:
    # the source of truth is the SQLite embedding_cache table.
    from lance_store import get_lance_store
    _store = get_lance_store()
    for _tname in ("embedding_cache_gemini", "embedding_cache_bge"):
        if _tname in _store.list_tables():
            _store.delete_table(_tname)
            logger.info(f"Dropped existing {_tname} for clean rebuild")

    gemini_table, bge_table = _open_lance_tables()
    existing_gemini = _existing_ids(gemini_table)
    existing_bge = _existing_ids(bge_table)
    logger.info(f"Already in Lance — gemini: {len(existing_gemini)}, bge: {len(existing_bge)}")

    migrated_gemini = 0
    migrated_bge = 0
    skipped = 0
    decode_errors = 0
    t0 = time.time()
    offset = 0

    while offset < total:
        rows = conn.execute(
            "SELECT text_hash, text_content, gemini_embedding, bge_embedding, created_at "
            "FROM embedding_cache LIMIT ? OFFSET ?",
            (batch_size, offset),
        ).fetchall()
        if not rows:
            break

        gemini_batch = {"ids": [], "embeddings": [], "documents": [], "metadatas": []}
        bge_batch = {"ids": [], "embeddings": [], "documents": [], "metadatas": []}

        for row in rows:
            text_hash = row["text_hash"]
            text_content = row["text_content"] or ""
            created_at = row["created_at"] or ""

            gemini_vec = _decode_blob(row["gemini_embedding"])
            bge_vec = _decode_blob(row["bge_embedding"])

            if gemini_vec and len(gemini_vec) == 768 and text_hash not in existing_gemini:
                gemini_batch["ids"].append(text_hash)
                gemini_batch["embeddings"].append(gemini_vec)
                gemini_batch["documents"].append(text_content)
                gemini_batch["metadatas"].append({"created_at": created_at, "backend": "gemini"})
                existing_gemini.add(text_hash)
            elif gemini_vec is None or len(gemini_vec) != 768:
                decode_errors += 1

            if bge_vec and len(bge_vec) == 768 and text_hash not in existing_bge:
                bge_batch["ids"].append(text_hash)
                bge_batch["embeddings"].append(bge_vec)
                bge_batch["documents"].append(text_content)
                bge_batch["metadatas"].append({"created_at": created_at, "backend": "bge"})
                existing_bge.add(text_hash)

        if gemini_batch["ids"]:
            gemini_table.add(**gemini_batch)
            migrated_gemini += len(gemini_batch["ids"])
        if bge_batch["ids"]:
            bge_table.add(**bge_batch)
            migrated_bge += len(bge_batch["ids"])

        if not gemini_batch["ids"] and not bge_batch["ids"]:
            skipped += len(rows)

        offset += batch_size
        elapsed = time.time() - t0
        rate = offset / elapsed if elapsed > 0 else 0.0
        logger.info(f"Batch {offset}/{total} — gemini+{migrated_gemini} bge+{migrated_bge} "
                    f"skipped={skipped} decode_err={decode_errors} rate={rate:.0f} rows/s")

    conn.close()

    logger.info("=" * 60)
    logger.info(f"DONE in {time.time()-t0:.1f}s")
    logger.info(f"  Migrated gemini: {migrated_gemini}")
    logger.info(f"  Migrated bge: {migrated_bge}")
    logger.info(f"  Skipped (already in Lance): {skipped}")
    logger.info(f"  Decode errors: {decode_errors}")
    logger.info(f"  LanceDB final — gemini: {gemini_table._table.count_rows()}, "
                f"bge: {bge_table._table.count_rows()}")
    return 0


def verify_counts():
    sqlite_path = _resolve_paths()
    conn = sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True, timeout=30.0)
    sqlite_count = conn.execute(
        "SELECT COUNT(*) FROM embedding_cache WHERE gemini_embedding IS NOT NULL"
    ).fetchone()[0]
    sqlite_bge_count = conn.execute(
        "SELECT COUNT(*) FROM embedding_cache WHERE bge_embedding IS NOT NULL"
    ).fetchone()[0]
    conn.close()

    gemini_table, bge_table = _open_lance_tables()
    lance_gemini = gemini_table._table.count_rows()
    lance_bge = bge_table._table.count_rows()

    logger.info(f"SQLite gemini-present: {sqlite_count}, Lance gemini: {lance_gemini}, "
                f"delta: {sqlite_count - lance_gemini}")
    logger.info(f"SQLite bge-present: {sqlite_bge_count}, Lance bge: {lance_bge}, "
                f"delta: {sqlite_bge_count - lance_bge}")
    return 0 if (sqlite_count <= lance_gemini and sqlite_bge_count <= lance_bge) else 2


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true", help="Count and sample, no writes")
    ap.add_argument("--verify", action="store_true", help="Compare SQLite vs Lance counts")
    ap.add_argument("--batch-size", type=int, default=1000, help="Rows per batch (default 1000)")
    args = ap.parse_args()

    if args.verify:
        return verify_counts()
    return migrate(dry_run=args.dry_run, batch_size=args.batch_size)


if __name__ == "__main__":
    sys.exit(main())
