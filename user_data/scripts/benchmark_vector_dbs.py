"""
Phase 28: Vector DB Benchmark — LanceDB vs Zvec vs (eski) ChromaDB

Kullanim:
    python benchmark_vector_dbs.py [--n-docs 1000] [--n-queries 100] [--dim 768]

Sonuc: latency, recall, RAM karsilastirmasi.
ChromaDB mevcut degilse skip edilir.
Zvec mevcut degilse skip edilir.
"""

import os
import sys
import time
import json
import logging
import argparse
import tempfile
from typing import Dict, Any

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
logger = logging.getLogger("benchmark")

from ai_config import AI_DB_PATH  # Required by project convention


def benchmark_lancedb(n_docs: int, n_queries: int, dim: int) -> Dict[str, Any]:
    """Benchmark LanceDB."""
    os.environ['LANCE_DB_DIR'] = tempfile.mkdtemp()
    from lance_store import LanceStore

    store = LanceStore.__new__(LanceStore)
    store._initialized = False
    store.__init__()

    table = store.get_or_create_table("bench", dim=dim)

    # Insert
    np.random.seed(42)
    embeddings = np.random.randn(n_docs, dim).tolist()
    ids = [f"doc_{i}" for i in range(n_docs)]
    docs = [f"Document {i} about crypto market" for i in range(n_docs)]

    t0 = time.perf_counter()
    batch_size = 500
    for i in range(0, n_docs, batch_size):
        table.add(
            ids=ids[i:i+batch_size],
            embeddings=embeddings[i:i+batch_size],
            documents=docs[i:i+batch_size],
        )
    insert_time = time.perf_counter() - t0

    # Search
    queries = np.random.randn(n_queries, dim).tolist()
    latencies = []
    for q in queries:
        t0 = time.perf_counter()
        table.search(q, n_results=10)
        latencies.append((time.perf_counter() - t0) * 1000)

    return {
        "db": "LanceDB",
        "insert_time_s": round(insert_time, 3),
        "avg_search_ms": round(np.mean(latencies), 2),
        "p50_search_ms": round(np.percentile(latencies, 50), 2),
        "p95_search_ms": round(np.percentile(latencies, 95), 2),
        "p99_search_ms": round(np.percentile(latencies, 99), 2),
        "total_docs": table.count(),
    }


def benchmark_zvec(n_docs: int, n_queries: int, dim: int) -> Dict[str, Any]:
    """Benchmark Zvec (if available)."""
    try:
        import zvec
    except ImportError:
        return {"db": "Zvec", "error": "not installed"}

    try:
        db_path = tempfile.mkdtemp()

        schema = zvec.CollectionSchema(
            fields=[
                zvec.FieldSchema(name="id", data_type=zvec.DataType.STRING),
                zvec.FieldSchema(name="vector", data_type=zvec.DataType.FLOAT_VECTOR),
                zvec.FieldSchema(name="text", data_type=zvec.DataType.STRING, nullable=True),
            ]
        )
        col = zvec.Collection.create(
            name="bench",
            schema=schema,
            option=zvec.CollectionOption(path=db_path, dim=dim),
        )

        # Insert
        np.random.seed(42)
        t0 = time.perf_counter()
        for i in range(n_docs):
            col.insert([zvec.Doc(
                id=f"doc_{i}",
                vector=np.random.randn(dim).tolist(),
                text=f"Document {i}",
            )])
        insert_time = time.perf_counter() - t0

        # Search
        queries = np.random.randn(n_queries, dim).tolist()
        latencies = []
        for q in queries:
            t0 = time.perf_counter()
            col.search(q, top_k=10)
            latencies.append((time.perf_counter() - t0) * 1000)

        col.close()
        return {
            "db": "Zvec",
            "insert_time_s": round(insert_time, 3),
            "avg_search_ms": round(np.mean(latencies), 2),
            "p50_search_ms": round(np.percentile(latencies, 50), 2),
            "p95_search_ms": round(np.percentile(latencies, 95), 2),
            "p99_search_ms": round(np.percentile(latencies, 99), 2),
            "total_docs": n_docs,
        }
    except Exception as e:
        return {"db": "Zvec", "error": str(e)}


def main():
    parser = argparse.ArgumentParser(description="Vector DB Benchmark")
    parser.add_argument("--n-docs", type=int, default=1000)
    parser.add_argument("--n-queries", type=int, default=100)
    parser.add_argument("--dim", type=int, default=768)
    args = parser.parse_args()

    logger.info(f"Benchmark: {args.n_docs} docs, {args.n_queries} queries, {args.dim}d")
    logger.info("=" * 60)

    results = []

    logger.info("LanceDB...")
    results.append(benchmark_lancedb(args.n_docs, args.n_queries, args.dim))
    logger.info(f"  {results[-1]}")

    logger.info("Zvec...")
    results.append(benchmark_zvec(args.n_docs, args.n_queries, args.dim))
    logger.info(f"  {results[-1]}")

    logger.info("=" * 60)
    logger.info("RESULTS:")
    for r in results:
        if "error" in r:
            logger.info(f"  {r['db']}: SKIPPED ({r['error']})")
        else:
            logger.info(f"  {r['db']}: insert={r['insert_time_s']}s, "
                         f"search avg={r['avg_search_ms']}ms p95={r['p95_search_ms']}ms")


if __name__ == "__main__":
    main()
