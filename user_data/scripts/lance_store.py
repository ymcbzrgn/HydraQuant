"""
lance_store.py — Phase 28: LanceDB Vector Store

ChromaDB'yi replace eden LanceDB wrapper. Ayni API yuzeyini sunar
ama altta LanceDB (Apache 2.0, disk-based, memory-mapped) kullanir.

ChromaDB'den farklar:
- Disk-based: RAM'de HNSW tutmaz, memory-mapped I/O
- Pre-filter: metadata filtreleme ANN sirasinda (post-filter degil)
- Lance format: Parquet'ten 2000x hizli random access
- Versioned: time travel destegi

Kullanim:
    from lance_store import get_lance_store
    store = get_lance_store()
    table = store.get_or_create_table("crypto_news", dim=768)
    table.add(ids=["doc1"], embeddings=[[0.1, ...]], documents=["text"], metadatas=[{...}])
    results = table.search(query_embedding=[0.1, ...], n_results=10)
"""

import os
import logging
import threading
import time
import uuid
from typing import List, Dict, Any, Optional, Union
from datetime import datetime

import numpy as np

logger = logging.getLogger(__name__)

try:
    import lancedb
    import pyarrow as pa
    LANCE_AVAILABLE = True
except ImportError:
    LANCE_AVAILABLE = False
    logger.warning("[LanceStore] lancedb not installed. pip install lancedb")

from ai_config import AI_DB_PATH

# Lance DB path — same directory as AI SQLite, subfolder 'lancedb'
LANCE_DB_DIR = os.environ.get(
    "LANCE_DB_DIR",
    os.path.join(os.path.dirname(AI_DB_PATH), "lancedb")
)


class LanceTable:
    """Wrapper around a single LanceDB table with ChromaDB-compatible API."""

    def __init__(self, table, table_name: str, dim: int, metric: str = "cosine"):
        self._table = table
        self.name = table_name
        self._dim = dim
        self._metric = metric

    @staticmethod
    def _to_float_list(vec) -> List[float]:
        """Convert any embedding format (numpy, list, tuple) to plain float list."""
        if hasattr(vec, 'tolist'):
            return vec.tolist()  # numpy array
        if isinstance(vec, (list, tuple)):
            return [float(x) for x in vec]
        return list(vec)

    def add(self, ids: List[str], embeddings: Optional[List[List[float]]] = None,
            documents: Optional[List[str]] = None,
            metadatas: Optional[List[Dict[str, Any]]] = None):
        """Add documents with embeddings to the table.

        Matches ChromaDB's add() signature.
        Metadata stored as JSON string (LanceDB is schema-strict, ChromaDB is not).
        """
        import json

        if not ids:
            return

        records = []
        for i, doc_id in enumerate(ids):
            meta = metadatas[i] if metadatas is not None and i < len(metadatas) else {}
            record = {
                "id": doc_id,
                "vector": self._to_float_list(embeddings[i]) if embeddings is not None and i < len(embeddings) else [0.0] * self._dim,
                "document": documents[i] if documents is not None and i < len(documents) else "",
                "metadata_json": json.dumps(meta or {}, ensure_ascii=False),
                "created_at": datetime.utcnow().isoformat(),
                # Sprint 2 Trinity fields
                "rl_relevance": 0.0,
                "rl_update_count": 0,
                "avg_trade_pnl": 0.0,
            }
            records.append(record)

        try:
            self._table.add(records)
        except Exception as e:
            logger.error(f"[LanceStore] add failed for {self.name}: {e}")
            raise

    def search(self, query_embedding: List[float], n_results: int = 10,
               where: Optional[Dict[str, Any]] = None) -> Dict[str, List]:
        """Search by embedding vector. Returns ChromaDB-compatible result format.

        Returns:
            {"ids": [[id1, id2, ...]], "documents": [[doc1, doc2, ...]],
             "distances": [[d1, d2, ...]], "metadatas": [[{}, {}, ...]]}
        """
        try:
            q = self._table.search(query_embedding).metric(self._metric).limit(n_results)

            # Apply metadata filters via JSON contains pattern
            if where:
                for k, v in where.items():
                    # LanceDB SQL filter on JSON string column
                    escaped_v = str(v).replace("'", "''")
                    q = q.where(f"metadata_json LIKE '%\"{k}\": \"{escaped_v}\"%'", prefilter=True)

            results = q.to_pandas()

            if results.empty:
                return {"ids": [[]], "documents": [[]], "distances": [[]], "metadatas": [[]]}

            import json as _json

            ids = results["id"].tolist()
            documents = results["document"].tolist() if "document" in results.columns else [""] * len(ids)
            distances = results["_distance"].tolist() if "_distance" in results.columns else [0.0] * len(ids)

            # Parse metadata from JSON column
            metadatas = []
            for _, row in results.iterrows():
                try:
                    meta = _json.loads(row.get("metadata_json", "{}") or "{}")
                except (TypeError, _json.JSONDecodeError):
                    meta = {}
                metadatas.append(meta)

            return {
                "ids": [ids],
                "documents": [documents],
                "distances": [distances],
                "metadatas": [metadatas],
            }
        except Exception as e:
            logger.error(f"[LanceStore] search failed for {self.name}: {e}")
            return {"ids": [[]], "documents": [[]], "distances": [[]], "metadatas": [[]]}

    def count(self) -> int:
        """Return number of rows in table."""
        try:
            return self._table.count_rows()
        except Exception:
            return 0

    def get(self, ids: List[str], include: Optional[List[str]] = None) -> Dict[str, List]:
        """Fetch documents by ID. ChromaDB-compatible return format.

        Returns:
            {"ids": [...], "documents": [...], "metadatas": [...], "embeddings": [...]}
        """
        import json as _json

        if not ids:
            return {"ids": [], "documents": [], "metadatas": [], "embeddings": []}

        try:
            id_list = ", ".join(f"'{i}'" for i in ids)
            results = self._table.search().where(f"id IN ({id_list})").limit(len(ids)).to_pandas()

            if results.empty:
                return {"ids": [], "documents": [], "metadatas": [], "embeddings": []}

            out_ids = results["id"].tolist()
            out_docs = results["document"].tolist() if "document" in results.columns else []
            out_metas = []
            for _, row in results.iterrows():
                try:
                    meta = _json.loads(row.get("metadata_json", "{}") or "{}")
                except (TypeError, _json.JSONDecodeError):
                    meta = {}
                out_metas.append(meta)
            out_embs = results["vector"].tolist() if "vector" in results.columns else []

            return {"ids": out_ids, "documents": out_docs, "metadatas": out_metas, "embeddings": out_embs}
        except Exception as e:
            logger.error(f"[LanceStore] get failed for {self.name}: {e}")
            return {"ids": [], "documents": [], "metadatas": [], "embeddings": []}

    def delete(self, ids: Optional[List[str]] = None, where: Optional[str] = None):
        """Delete rows by ID or filter."""
        try:
            if ids:
                id_list = ", ".join(f"'{i}'" for i in ids)
                self._table.delete(f"id IN ({id_list})")
            elif where:
                self._table.delete(where)
        except Exception as e:
            logger.error(f"[LanceStore] delete failed for {self.name}: {e}")

    def update_rl_relevance(self, doc_id: str, relevance_delta: float, trade_pnl: float):
        """Update RL feedback fields for Trinity (Sprint 2, Task 10B)."""
        try:
            self._table.update(
                where=f"id = '{doc_id}'",
                values={
                    "rl_relevance": f"rl_relevance + {relevance_delta}",
                    "rl_update_count": "rl_update_count + 1",
                    "avg_trade_pnl": f"(avg_trade_pnl * rl_update_count + {trade_pnl}) / (rl_update_count + 1)",
                }
            )
        except Exception as e:
            logger.debug(f"[LanceStore] rl_relevance update failed for {doc_id}: {e}")


class LanceStore:
    """Thread-safe singleton LanceDB connection manager.

    Replaces ChromaDB's PersistentClient. All modules use get_lance_store() instead.
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        if not LANCE_AVAILABLE:
            raise ImportError("lancedb not installed. Run: pip install lancedb")
        os.makedirs(LANCE_DB_DIR, exist_ok=True)
        self._db = lancedb.connect(LANCE_DB_DIR)
        self._tables: Dict[str, LanceTable] = {}
        self._initialized = True
        logger.info(f"[LanceStore] Connected to {LANCE_DB_DIR}")

    def get_or_create_table(self, name: str, dim: int = 768,
                            metric: str = "cosine") -> LanceTable:
        """Get existing table or create new one. Thread-safe.

        Matches ChromaDB's get_or_create_collection() pattern.
        """
        if name in self._tables:
            return self._tables[name]

        with self._lock:
            if name in self._tables:
                return self._tables[name]

            existing_tables = self._db.table_names()
            if name in existing_tables:
                table = self._db.open_table(name)
                logger.info(f"[LanceStore] Opened existing table: {name} ({table.count_rows()} rows)")
            else:
                # Create with schema
                schema = pa.schema([
                    pa.field("id", pa.string()),
                    pa.field("vector", pa.list_(pa.float32(), dim)),
                    pa.field("document", pa.string()),
                    pa.field("metadata_json", pa.string()),
                    pa.field("created_at", pa.string()),
                    pa.field("rl_relevance", pa.float32()),
                    pa.field("rl_update_count", pa.int32()),
                    pa.field("avg_trade_pnl", pa.float32()),
                ])
                table = self._db.create_table(name, schema=schema)
                logger.info(f"[LanceStore] Created new table: {name} (dim={dim})")

            wrapper = LanceTable(table, name, dim, metric)
            self._tables[name] = wrapper
            return wrapper

    def delete_table(self, name: str):
        """Delete a table entirely."""
        try:
            self._db.drop_table(name)
            self._tables.pop(name, None)
            logger.info(f"[LanceStore] Deleted table: {name}")
        except Exception as e:
            logger.error(f"[LanceStore] Failed to delete table {name}: {e}")

    def list_tables(self) -> List[str]:
        """List all table names."""
        return self._db.table_names()

    def get_stats(self) -> Dict[str, Any]:
        """Return store statistics."""
        tables = self._db.table_names()
        stats = {"db_dir": LANCE_DB_DIR, "table_count": len(tables), "tables": {}}
        for name in tables:
            try:
                t = self._db.open_table(name)
                stats["tables"][name] = t.count_rows()
            except Exception:
                stats["tables"][name] = -1
        return stats


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------

_lance_store = None
_lance_store_lock = threading.Lock()


def get_lance_store() -> LanceStore:
    """Thread-safe singleton LanceStore accessor.

    Replaces get_chroma_client() from ai_config.py.
    """
    global _lance_store
    if _lance_store is None:
        with _lance_store_lock:
            if _lance_store is None:
                _lance_store = LanceStore()
    return _lance_store


# ---------------------------------------------------------------------------
# Convenience: get_or_create shortcut
# ---------------------------------------------------------------------------

def get_lance_table(name: str, dim: int = 768) -> LanceTable:
    """Shortcut: get_lance_store().get_or_create_table(name, dim)."""
    return get_lance_store().get_or_create_table(name, dim)
