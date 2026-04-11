"""
Phase 28: ChromaDB → LanceDB Migration Script

Tum ChromaDB collection'larini LanceDB'ye tasir.
Embedding model degismez — mevcut vektorler oldugu gibi kopyalanir.

Kullanim:
    python migrate_chroma_to_lance.py [--dry-run] [--batch-size 500]

Guvenlik:
    - ChromaDB verisini SILMEZ (shadow mode)
    - Her batch sonrasi progress loglar
    - ID parity test: kaynak ve hedef doc sayisi eslesir
    - --dry-run ile once kuru calistirma yapilabilir
"""

import os
import sys
import json
import time
import logging
import argparse
from typing import Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
logger = logging.getLogger("migration")

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)


def get_chroma_collections():
    """ChromaDB'deki tum collection'lari ve doc sayilarini dondur."""
    import chromadb

    # Try both known paths
    for path in [
        os.path.join(SCRIPT_DIR, '..', 'vectordb'),
        os.path.join(SCRIPT_DIR, '..', 'db', 'vectordb'),
    ]:
        abs_path = os.path.abspath(path)
        if os.path.exists(abs_path):
            try:
                client = chromadb.PersistentClient(path=abs_path)
                colls = client.list_collections()
                if colls:
                    return client, colls, abs_path
            except Exception:
                continue
    return None, [], ""


def migrate_collection(chroma_coll, lance_table, batch_size: int = 500,
                       dry_run: bool = False):
    """Tek bir collection'i ChromaDB'den LanceDB'ye tasir."""
    total = chroma_coll.count()
    if total == 0:
        logger.info(f"  {chroma_coll.name}: 0 docs, skipping")
        return 0

    logger.info(f"  {chroma_coll.name}: {total} docs to migrate (batch_size={batch_size})")

    migrated = 0
    offset = 0

    while offset < total:
        # ChromaDB get() with offset/limit
        try:
            batch = chroma_coll.get(
                limit=batch_size,
                offset=offset,
                include=["embeddings", "documents", "metadatas"]
            )
        except Exception as e:
            logger.error(f"  ChromaDB get failed at offset={offset}: {e}")
            break

        ids = batch.get("ids", [])
        embeddings = batch.get("embeddings", [])
        documents = batch.get("documents", [])
        metadatas = batch.get("metadatas", [])

        if not ids:
            break

        if dry_run:
            logger.info(f"  [DRY-RUN] Would migrate {len(ids)} docs (offset={offset})")
        else:
            try:
                lance_table.add(
                    ids=ids,
                    embeddings=embeddings if embeddings else None,
                    documents=documents if documents else None,
                    metadatas=metadatas if metadatas else None,
                )
            except Exception as e:
                logger.error(f"  LanceDB add failed at offset={offset}: {e}")
                # Try one-by-one for this batch
                for i in range(len(ids)):
                    try:
                        lance_table.add(
                            ids=[ids[i]],
                            embeddings=[embeddings[i]] if embeddings else None,
                            documents=[documents[i]] if documents else None,
                            metadatas=[metadatas[i]] if metadatas else None,
                        )
                    except Exception as e2:
                        logger.warning(f"  Skipped {ids[i]}: {e2}")

        migrated += len(ids)
        offset += batch_size

        if migrated % 2000 == 0 or migrated == total:
            logger.info(f"  Progress: {migrated}/{total} ({100*migrated/total:.0f}%)")

    return migrated


def verify_parity(chroma_coll, lance_table) -> bool:
    """ID parity check: ChromaDB count == LanceDB count."""
    chroma_count = chroma_coll.count()
    lance_count = lance_table.count()
    match = chroma_count == lance_count
    status = "MATCH" if match else "MISMATCH"
    logger.info(f"  Parity {status}: ChromaDB={chroma_count}, LanceDB={lance_count}")
    return match


def main():
    parser = argparse.ArgumentParser(description="Migrate ChromaDB → LanceDB")
    parser.add_argument("--dry-run", action="store_true", help="Kuru calistirma, veri yazmaz")
    parser.add_argument("--batch-size", type=int, default=500, help="Batch buyuklugu")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Phase 28: ChromaDB → LanceDB Migration")
    logger.info("=" * 60)

    # Step 1: Connect to ChromaDB
    client, collections, chroma_path = get_chroma_collections()
    if not collections:
        logger.info("No ChromaDB collections found. Nothing to migrate.")
        return

    logger.info(f"ChromaDB path: {chroma_path}")
    logger.info(f"Collections: {len(collections)}")
    for c in collections:
        logger.info(f"  - {c.name}: {c.count()} docs")

    # Step 2: Connect to LanceDB
    if not args.dry_run:
        from lance_store import get_lance_store
        store = get_lance_store()
        logger.info(f"LanceDB path: {store.get_stats()['db_dir']}")

    # Step 3: Migrate each collection
    total_migrated = 0
    all_parity = True
    start = time.time()

    for chroma_coll in collections:
        name = chroma_coll.name
        count = chroma_coll.count()
        if count == 0:
            logger.info(f"  {name}: empty, skipping")
            continue

        # Determine embedding dimension from first doc
        try:
            sample = chroma_coll.get(limit=1, include=["embeddings"])
            dim = len(sample["embeddings"][0]) if sample["embeddings"] else 768
        except Exception:
            dim = 768

        logger.info(f"\nMigrating: {name} ({count} docs, dim={dim})")

        if args.dry_run:
            lance_table = None
        else:
            lance_table = store.get_or_create_table(name, dim=dim)

        migrated = migrate_collection(
            chroma_coll, lance_table,
            batch_size=args.batch_size,
            dry_run=args.dry_run,
        )
        total_migrated += migrated

        # Parity check
        if not args.dry_run and lance_table:
            if not verify_parity(chroma_coll, lance_table):
                all_parity = False

    elapsed = time.time() - start
    logger.info(f"\n{'='*60}")
    logger.info(f"Migration {'DRY-RUN ' if args.dry_run else ''}complete:")
    logger.info(f"  Collections: {len(collections)}")
    logger.info(f"  Total docs: {total_migrated}")
    logger.info(f"  Parity: {'ALL MATCH' if all_parity else 'SOME MISMATCH'}")
    logger.info(f"  Time: {elapsed:.1f}s")
    logger.info(f"{'='*60}")

    if not all_parity:
        logger.warning("PARITY MISMATCH detected! Do NOT remove ChromaDB yet.")
        sys.exit(1)


if __name__ == "__main__":
    main()
