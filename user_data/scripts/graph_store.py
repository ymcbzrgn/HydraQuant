"""
graph_store.py — Phase 28: Grafeo Graph Database Store

MAGMA graph, agent networks, causal graphs, cross-pair intel icin
Grafeo (Apache 2.0, Rust-native, Cypher+GQL) wrapper.

Kullanim:
    from graph_store import get_graph_store
    gs = get_graph_store()
    gs.add_edge("semantic", "rsi", "indicates", "oversold", weight=0.8)
    results = gs.cypher("MATCH (a)-[r]->(b) WHERE a.name = 'rsi' RETURN b.name, r.weight")
    neighbors = gs.get_neighbors("rsi", max_hops=2)
"""

import os
import json
import logging
import threading
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)

try:
    import grafeo
    GRAFEO_AVAILABLE = True
except ImportError:
    GRAFEO_AVAILABLE = False
    logger.warning("[GraphStore] grafeo not installed. pip install grafeo")

from ai_config import AI_DB_PATH

# Graph DB path
GRAPH_DB_DIR = os.environ.get(
    "GRAPH_DB_DIR",
    os.path.join(os.path.dirname(AI_DB_PATH), "graphdb")
)
GRAPH_DB_PATH = os.path.join(GRAPH_DB_DIR, "hydra.grafeo")

# Valid graph types (matching MAGMA's original 4 types)
VALID_GRAPH_TYPES = {"semantic", "temporal", "causal", "entity"}


class GraphStore:
    """Thread-safe Grafeo graph database wrapper.

    Replaces SQLite-based magma_edges table with a proper graph DB.
    Supports Cypher queries, multi-hop traversal, and vector search.
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
        if not GRAFEO_AVAILABLE:
            raise ImportError("grafeo not installed. Run: pip install grafeo")
        os.makedirs(GRAPH_DB_DIR, exist_ok=True)
        self._write_lock = threading.Lock()
        self._is_writer = False

        # Try exclusive write access first
        try:
            self._db = grafeo.GrafeoDB.open(GRAPH_DB_PATH)
            self._is_writer = True
            logger.info(f"[GraphStore] WRITER mode: {GRAPH_DB_PATH} "
                         f"(nodes={self._db.node_count}, edges={self._db.edge_count})")
        except Exception as e:
            if "locked" in str(e).lower():
                # Another process has write lock — open read-only in memory
                try:
                    self._db = grafeo.GrafeoDB.open_in_memory(GRAPH_DB_PATH)
                    logger.info(f"[GraphStore] READER mode (in-memory): {GRAPH_DB_PATH} "
                                 f"(nodes={self._db.node_count}, edges={self._db.edge_count})")
                except Exception as e2:
                    # DB file doesn't exist yet — create in-memory empty
                    self._db = grafeo.GrafeoDB()
                    logger.warning(f"[GraphStore] EMPTY in-memory mode (no DB file yet)")
            else:
                raise

        self._initialized = True

    # ─── MAGMA-Compatible API ───────────────────────────────────────

    def add_edge(self, graph_type: str, source: str, relation: str,
                 target: str, weight: float = 1.0,
                 metadata: Optional[Dict] = None) -> bool:
        """Add or strengthen an edge (Hebbian learning).

        MAGMA-compatible: same signature as MAGMAMemory.add_edge().
        If edge exists, weight increases by 0.1 (Hebbian reinforcement).
        """
        if graph_type not in VALID_GRAPH_TYPES:
            logger.warning(f"[GraphStore] Invalid graph_type: {graph_type}")
            return False

        source_lower = source.lower().strip()
        target_lower = target.lower().strip()

        if not self._is_writer:
            # Reader mode — can't write, log and skip
            logger.debug(f"[GraphStore] Skipped add_edge (reader mode): {source}->{target}")
            return False

        with self._write_lock:
            try:
                # Find or create source node
                src_node = self._find_or_create_node(source_lower, graph_type)
                tgt_node = self._find_or_create_node(target_lower, graph_type)

                # Check if edge already exists
                existing = self._find_edge(src_node.id, tgt_node.id, relation)
                if existing:
                    # Hebbian reinforcement: increase weight
                    old_weight = 1.0
                    try:
                        # Get current weight via Cypher (safest way)
                        w_result = self._db.execute_cypher(
                            f"MATCH ()-[r]->() WHERE id(r) = {existing.id} RETURN r.weight AS w"
                        )
                        if w_result:
                            old_weight = w_result[0].get("w", 1.0)
                    except Exception:
                        pass
                    new_weight = min(old_weight + 0.1, 10.0)
                    self._db.set_edge_property(existing.id, "weight", new_weight)
                    self._db.set_edge_property(existing.id, "timestamp",
                                                datetime.utcnow().isoformat())
                    return True

                # Create new edge
                props = {
                    "weight": weight,
                    "graph_type": graph_type,
                    "relation": relation,
                    "timestamp": datetime.utcnow().isoformat(),
                }
                if metadata:
                    props["metadata_json"] = json.dumps(metadata, ensure_ascii=False)

                self._db.create_edge(src_node.id, tgt_node.id, relation, props)
                return True

            except Exception as e:
                logger.error(f"[GraphStore] add_edge failed: {e}")
                return False

    def query(self, query_text: str, graph_type: str = "semantic",
              max_hops: int = 1) -> List[Dict]:
        """BFS traversal from matching nodes. MAGMA-compatible."""
        try:
            # Find nodes matching query
            cypher = f"""
                MATCH (n)-[r*1..{max_hops}]->(m)
                WHERE n.name CONTAINS '{query_text.lower()}'
                AND r[0].graph_type = '{graph_type}'
                RETURN n.name AS source, m.name AS target,
                       r[0].relation AS relation, r[0].weight AS weight
                LIMIT 20
            """
            results = self._db.execute_cypher(cypher)
            return [dict(r) for r in results] if results else []
        except Exception as e:
            logger.debug(f"[GraphStore] query failed: {e}")
            return []

    def traverse(self, start_node: str, graph_type: str = "semantic",
                 max_hops: int = 2) -> List[Dict]:
        """Multi-hop traversal from a starting node."""
        try:
            cypher = f"""
                MATCH path = (a)-[*1..{max_hops}]->(b)
                WHERE a.name = '{start_node.lower()}'
                RETURN a.name AS from_node, b.name AS to_node,
                       length(path) AS hops
                LIMIT 50
            """
            results = self._db.execute_cypher(cypher)
            return [dict(r) for r in results] if results else []
        except Exception as e:
            logger.debug(f"[GraphStore] traverse failed: {e}")
            return []

    # ─── Cypher Query API ───────────────────────────────────────────

    def cypher(self, query: str) -> List[Dict]:
        """Execute raw Cypher query. Returns list of result dicts."""
        try:
            results = self._db.execute_cypher(query)
            return [dict(r) for r in results] if results else []
        except Exception as e:
            logger.error(f"[GraphStore] Cypher failed: {e}")
            return []

    # ─── Sprint 2: Causal Graph API ─────────────────────────────────

    def add_causal_edge(self, source_var: str, target_var: str,
                        strength: float, time_lag: int = 0,
                        p_value: float = 0.0, regime: str = "_global"):
        """Add a causal discovery edge (Tigramite PCMCI+ output)."""
        return self.add_edge(
            graph_type="causal",
            source=source_var,
            relation="causes",
            target=target_var,
            weight=strength,
            metadata={
                "time_lag": time_lag,
                "p_value": p_value,
                "regime": regime,
                "method": "PCMCI+",
                "discovered_at": datetime.utcnow().isoformat(),
            }
        )

    def get_causal_parents(self, target_var: str, regime: str = "_global") -> List[Dict]:
        """Find all variables that causally influence target_var."""
        return self.cypher(f"""
            MATCH (a)-[r:causes]->(b)
            WHERE b.name = '{target_var.lower()}'
            RETURN a.name AS parent, r.weight AS strength,
                   r.metadata_json AS meta
            ORDER BY r.weight DESC
        """)

    def get_causal_children(self, source_var: str) -> List[Dict]:
        """Find all variables causally influenced by source_var."""
        return self.cypher(f"""
            MATCH (a)-[r:causes]->(b)
            WHERE a.name = '{source_var.lower()}'
            RETURN b.name AS child, r.weight AS strength,
                   r.metadata_json AS meta
            ORDER BY r.weight DESC
        """)

    # ─── Sprint 2: RL Agent Graph API ───────────────────────────────

    def add_agent_node(self, agent_type: str, organ: str,
                       properties: Optional[Dict] = None):
        """Create or update an RL agent node."""
        with self._write_lock:
            props = {"name": agent_type.lower(), "organ": organ}
            if properties:
                props.update(properties)
            return self._find_or_create_node(agent_type.lower(), "entity",
                                              labels=["Agent"], extra_props=props)

    def add_agent_interaction(self, agent_a: str, agent_b: str,
                               interaction_type: str = "coordinates",
                               weight: float = 1.0):
        """Record interaction between two RL agents."""
        return self.add_edge("entity", agent_a, interaction_type, agent_b, weight)

    # ─── Sprint 2: GNN Export ───────────────────────────────────────

    def export_for_gnn(self, graph_type: Optional[str] = None) -> Dict[str, Any]:
        """Export graph as PyTorch Geometric-compatible format.

        Returns:
            {"node_ids": [...], "node_features": [[...], ...],
             "edge_index": [[src...], [tgt...]], "edge_attr": [[...], ...]}
        """
        try:
            # Get all nodes
            if graph_type:
                nodes_result = self.cypher(f"""
                    MATCH (n)-[r]-() WHERE r.graph_type = '{graph_type}'
                    RETURN DISTINCT n.name AS name, labels(n) AS labels
                """)
            else:
                nodes_result = self.cypher("MATCH (n) RETURN n.name AS name, labels(n) AS labels")

            if not nodes_result:
                return {"node_ids": [], "node_features": [], "edge_index": [[], []], "edge_attr": []}

            # Build node index
            node_names = list(set(r.get("name", "") for r in nodes_result))
            node_to_idx = {name: i for i, name in enumerate(node_names)}

            # Get all edges
            if graph_type:
                edges_result = self.cypher(f"""
                    MATCH (a)-[r]->(b) WHERE r.graph_type = '{graph_type}'
                    RETURN a.name AS src, b.name AS tgt, r.weight AS weight, r.relation AS rel
                """)
            else:
                edges_result = self.cypher("""
                    MATCH (a)-[r]->(b)
                    RETURN a.name AS src, b.name AS tgt, r.weight AS weight, r.relation AS rel
                """)

            sources = []
            targets = []
            edge_weights = []
            for e in edges_result:
                src_name = e.get("src", "")
                tgt_name = e.get("tgt", "")
                if src_name in node_to_idx and tgt_name in node_to_idx:
                    sources.append(node_to_idx[src_name])
                    targets.append(node_to_idx[tgt_name])
                    edge_weights.append(e.get("weight", 1.0))

            return {
                "node_ids": node_names,
                "node_features": [],  # Populated by GNN module with actual features
                "edge_index": [sources, targets],
                "edge_attr": edge_weights,
            }
        except Exception as e:
            logger.error(f"[GraphStore] GNN export failed: {e}")
            return {"node_ids": [], "node_features": [], "edge_index": [[], []], "edge_attr": []}

    # ─── Internal Helpers ───────────────────────────────────────────

    def _find_or_create_node(self, name: str, graph_type: str,
                              labels: Optional[List[str]] = None,
                              extra_props: Optional[Dict] = None):
        """Find existing node by name or create new one."""
        # Try Cypher find first — use id(n) to get integer node ID
        try:
            escaped = name.replace("'", "\\'")
            result = self._db.execute_cypher(
                f"MATCH (n) WHERE n.name = '{escaped}' RETURN id(n) AS nid LIMIT 1"
            )
            if result and len(result) > 0:
                nid = result[0].get("nid")
                if nid is not None:
                    return self._db.get_node(nid)
        except Exception:
            pass

        # Create new node
        node_labels = labels or [graph_type.capitalize()]
        props = {"name": name, "graph_type": graph_type,
                 "created_at": datetime.utcnow().isoformat()}
        if extra_props:
            props.update(extra_props)
        return self._db.create_node(node_labels, props)

    def _find_edge(self, src_id: int, tgt_id: int, relation: str):
        """Find existing edge between two nodes with given relation."""
        try:
            result = self._db.execute_cypher(f"""
                MATCH (a)-[r:{relation}]->(b)
                WHERE id(a) = {src_id} AND id(b) = {tgt_id}
                RETURN id(r) AS rid LIMIT 1
            """)
            if result and len(result) > 0:
                rid = result[0].get("rid")
                if rid is not None:
                    return self._db.get_edge(rid)
        except Exception:
            pass
        return None

    # ─── Stats & Management ─────────────────────────────────────────

    def save(self):
        """Persist graph to disk via WAL checkpoint (writer only)."""
        if not self._is_writer:
            return
        try:
            self._db.wal_checkpoint()
        except Exception:
            pass

    def get_stats(self) -> Dict[str, Any]:
        """Return graph statistics."""
        try:
            mem = self._db.memory_usage() if callable(self._db.memory_usage) else self._db.memory_usage
        except Exception:
            mem = "N/A"
        return {
            "path": GRAPH_DB_PATH,
            "mode": "writer" if self._is_writer else "reader",
            "node_count": self._db.node_count,
            "edge_count": self._db.edge_count,
            "memory_usage": mem,
        }

    def close(self):
        """Save and close the graph database."""
        try:
            self._db.wal_checkpoint()
        except Exception:
            pass
        self._db.close()


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------

_graph_store = None
_graph_store_lock = threading.Lock()


def get_graph_store() -> GraphStore:
    """Thread-safe singleton GraphStore accessor."""
    global _graph_store
    if _graph_store is None:
        with _graph_store_lock:
            if _graph_store is None:
                _graph_store = GraphStore()
    return _graph_store
