"""
MAGMA: Multi-Graph Agentic Memory

Phase 28: Now uses Grafeo graph database instead of SQLite magma_edges table.
API is backward-compatible — add_edge(), traverse(), query(), prune(), get_stats().

Organizes information across 4 orthogonal graphs:
1. Semantic: Meaning relationships (oversold -> reversal)
2. Temporal: Event sequencing (Fed decision -> market reaction)
3. Causal: Cause-and-effect (rate hike -> dollar strength)
4. Entity: Asset correlations (BTC -> ETH)
"""

import logging
import os
import sys
from typing import List, Dict, Any

sys.path.append(os.path.dirname(__file__))

logger = logging.getLogger(__name__)

try:
    from graph_store import get_graph_store, VALID_GRAPH_TYPES
    GRAFEO_AVAILABLE = True
except ImportError:
    GRAFEO_AVAILABLE = False
    VALID_GRAPH_TYPES = {"semantic", "temporal", "causal", "entity"}

# Fallback: SQLite-based MAGMA (used only if Grafeo unavailable)
from ai_config import AI_DB_PATH
from db import get_db_connection


class MAGMAMemory:
    """Multi-Graph Agentic Memory — now backed by Grafeo graph DB."""

    VALID_GRAPHS = VALID_GRAPH_TYPES

    def __init__(self, db_path=AI_DB_PATH):
        self.db_path = db_path
        # Use Grafeo only for production DB path. Test/custom paths use SQLite fallback.
        self._use_grafeo = GRAFEO_AVAILABLE and (db_path == AI_DB_PATH)
        if self._use_grafeo:
            try:
                self._gs = get_graph_store()
                logger.info(f"[MAGMA] Using Grafeo backend (nodes={self._gs.get_stats()['node_count']})")
            except Exception as e:
                logger.warning(f"[MAGMA] Grafeo unavailable, falling back to SQLite: {e}")
                self._use_grafeo = False
                self._init_sqlite_fallback()
        else:
            self._init_sqlite_fallback()

    def _init_sqlite_fallback(self):
        """SQLite fallback — legacy behavior."""
        try:
            with get_db_connection(self.db_path) as conn:
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS magma_edges (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        graph_type TEXT NOT NULL, source TEXT NOT NULL,
                        relation TEXT NOT NULL, target TEXT NOT NULL,
                        weight REAL DEFAULT 1.0,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        metadata TEXT DEFAULT '{}',
                        UNIQUE(graph_type, source, relation, target))''')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_magma_graphs ON magma_edges(graph_type, source)')
                conn.commit()
        except Exception as e:
            logger.error(f"MAGMA SQLite fallback init failed: {e}")

    def add_edge(self, graph_type: str, source: str, relation: str,
                 target: str, metadata: dict = None) -> bool:
        """Add edge with Hebbian reinforcement. Grafeo primary, SQLite fallback."""
        if graph_type not in self.VALID_GRAPHS:
            logger.error(f"Invalid MAGMA graph type: {graph_type}")
            return False

        if self._use_grafeo:
            return self._gs.add_edge(graph_type, source, relation, target,
                                      weight=1.0, metadata=metadata)

        # SQLite fallback
        import json
        meta_str = json.dumps(metadata) if metadata else "{}"
        try:
            with get_db_connection(self.db_path) as conn:
                conn.execute('''
                    INSERT INTO magma_edges (graph_type, source, relation, target, weight, timestamp, metadata)
                    VALUES (?, ?, ?, ?, 1.0, CURRENT_TIMESTAMP, ?)
                    ON CONFLICT(graph_type, source, relation, target)
                    DO UPDATE SET weight = weight + 0.1, timestamp = CURRENT_TIMESTAMP
                ''', (graph_type, source.lower(), relation.lower(), target.lower(), meta_str))
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"MAGMA add_edge failed: {e}")
            return False

    def traverse(self, start_node: str, graph_type: str, max_hops: int = 2) -> List[Dict[str, Any]]:
        """BFS traversal from start_node."""
        if graph_type not in self.VALID_GRAPHS:
            return []

        if self._use_grafeo:
            results = self._gs.traverse(start_node, graph_type, max_hops)
            # Convert to legacy format
            return [{"source": r.get("from_node", ""), "target": r.get("to_node", ""),
                      "graph_type": graph_type, "relation": "", "weight": 1.0, "metadata": "{}"}
                    for r in results]

        # SQLite fallback
        start_node = start_node.lower()
        visited = set()
        queue = [(start_node, 0)]
        found_edges = []
        try:
            with get_db_connection(self.db_path) as conn:
                cursor = conn.cursor()
                while queue:
                    current_node, current_hop = queue.pop(0)
                    if current_hop >= max_hops or current_node in visited:
                        continue
                    visited.add(current_node)
                    cursor.execute('''
                        SELECT source, relation, target, weight, metadata, graph_type
                        FROM magma_edges WHERE graph_type = ? AND source = ?
                        ORDER BY weight DESC LIMIT 20
                    ''', (graph_type, current_node))
                    for row in cursor.fetchall():
                        found_edges.append(dict(row))
                        if row['target'] not in visited:
                            queue.append((row['target'], current_hop + 1))
            return found_edges
        except Exception as e:
            logger.error(f"MAGMA traversal failed: {e}")
            return []

    def query(self, question: str, graph_types: List[str] = None, max_hops: int = 2) -> List[Dict[str, Any]]:
        """Extract concepts from question and traverse relevant graphs."""
        if graph_types is None:
            graph_types = list(self.VALID_GRAPHS)

        if self._use_grafeo:
            results = self._gs.query(question, max_hops=max_hops)
            return results[:50]

        # SQLite fallback — keyword extraction + BFS
        words = set([w.lower().strip(',.?!"') for w in question.split() if len(w) > 3])
        starting_nodes = []
        try:
            with get_db_connection(self.db_path) as conn:
                cursor = conn.cursor()
                if words:
                    placeholders = ','.join(['?'] * len(words))
                    cursor.execute(f"SELECT DISTINCT source FROM magma_edges WHERE source IN ({placeholders})", list(words))
                    starting_nodes = [r[0] for r in cursor.fetchall()]
                if not starting_nodes:
                    for word in words:
                        cursor.execute("SELECT DISTINCT source FROM magma_edges WHERE source LIKE ? LIMIT 3", (f"%{word}%",))
                        starting_nodes.extend([r[0] for r in cursor.fetchall()])
        except Exception:
            pass

        all_edges = []
        for node in set(starting_nodes):
            for g_type in graph_types:
                if g_type in self.VALID_GRAPHS:
                    all_edges.extend(self.traverse(node, g_type, max_hops))

        # Deduplicate
        unique = []
        seen = set()
        for e in all_edges:
            sig = (e.get('graph_type', ''), e.get('source', ''), e.get('relation', ''), e.get('target', ''))
            if sig not in seen:
                seen.add(sig)
                unique.append(e)
        unique.sort(key=lambda x: x.get('weight', 0), reverse=True)
        return unique[:50]

    def prune(self, min_weight: float = 0.5, max_age_days: int = 180):
        """Remove weak/old edges."""
        if self._use_grafeo:
            # Grafeo doesn't have time-based pruning yet — log only
            logger.info("[MAGMA] Grafeo pruning: not yet implemented (edges persist)")
            return 0

        try:
            with get_db_connection(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "DELETE FROM magma_edges WHERE weight < ? AND timestamp < datetime('now', ?)",
                    (min_weight, f"-{max_age_days} days"))
                deleted = cursor.rowcount
                conn.commit()
                logger.info(f"MAGMA Pruned: {deleted} weak/old edges removed.")
                return deleted
        except Exception as e:
            logger.error(f"MAGMA pruning failed: {e}")
            return 0

    def get_stats(self) -> dict:
        """Return graph statistics."""
        if self._use_grafeo:
            gs_stats = self._gs.get_stats()
            return {"total": {"nodes": gs_stats["node_count"], "edges": gs_stats["edge_count"]},
                    "backend": "grafeo"}

        stats = {g: {"nodes": 0, "edges": 0} for g in self.VALID_GRAPHS}
        try:
            with get_db_connection(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT graph_type, COUNT(*) FROM magma_edges GROUP BY graph_type")
                for row in cursor.fetchall():
                    if row[0] in stats:
                        stats[row[0]]["edges"] = row[1]
                cursor.execute('''
                    SELECT graph_type, COUNT(DISTINCT node) FROM (
                        SELECT graph_type, source as node FROM magma_edges
                        UNION SELECT graph_type, target as node FROM magma_edges
                    ) GROUP BY graph_type''')
                for row in cursor.fetchall():
                    if row[0] in stats:
                        stats[row[0]]["nodes"] = row[1]
        except Exception as e:
            logger.error(f"MAGMA stats failed: {e}")
        stats["backend"] = "sqlite"
        return stats
