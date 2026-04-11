"""
graph_store.py — Phase 28: Grafeo Graph Database Store (ZeroMQ IPC)

Multi-process mimarisi:
- Scheduler process = BROKER (Grafeo DB'yi exclusive acip ZMQ REP dinler)
- RAG + Bot process = CLIENT (ZMQ REQ ile sorgu gonderir)
- IPC endpoint: Unix domain socket (0.1ms latency)

Kullanim:
    # Broker tarafinda (scheduler'da otomatik baslar):
    from graph_store import start_graph_broker
    start_graph_broker()  # background thread

    # Client tarafinda (herhangi bir process):
    from graph_store import get_graph_store
    gs = get_graph_store()
    gs.add_edge("semantic", "rsi", "indicates", "oversold")
    results = gs.cypher("MATCH (n) RETURN n.name LIMIT 10")
"""

import os
import json
import logging
import threading
import time
from typing import List, Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

try:
    import grafeo
    GRAFEO_AVAILABLE = True
except ImportError:
    GRAFEO_AVAILABLE = False

try:
    import zmq
    ZMQ_AVAILABLE = True
except ImportError:
    ZMQ_AVAILABLE = False
    logger.warning("[GraphStore] pyzmq not installed. pip install pyzmq")

from ai_config import AI_DB_PATH

GRAPH_DB_DIR = os.environ.get(
    "GRAPH_DB_DIR",
    os.path.join(os.path.dirname(AI_DB_PATH), "graphdb")
)
GRAPH_DB_PATH = os.path.join(GRAPH_DB_DIR, "hydra.grafeo")
ZMQ_ENDPOINT = os.environ.get("GRAFEO_ZMQ_ENDPOINT", "ipc:///tmp/grafeo_hydra.sock")

VALID_GRAPH_TYPES = {"semantic", "temporal", "causal", "entity"}


# ═══════════════════════════════════════════════════════════════════
# BROKER — runs inside scheduler process (single writer)
# ═══════════════════════════════════════════════════════════════════

class _GrafeoBroker:
    """ZMQ REP broker that holds the exclusive Grafeo lock."""

    def __init__(self):
        self._db = None
        self._running = False
        self._thread = None

    def start(self):
        """Start broker in background thread."""
        if self._running:
            return
        if not GRAFEO_AVAILABLE or not ZMQ_AVAILABLE:
            logger.warning("[GraphBroker] grafeo or pyzmq not available, skipping")
            return

        os.makedirs(GRAPH_DB_DIR, exist_ok=True)

        # Clean stale socket
        sock_path = ZMQ_ENDPOINT.replace("ipc://", "")
        if os.path.exists(sock_path):
            try:
                os.unlink(sock_path)
            except OSError:
                pass

        try:
            self._db = grafeo.GrafeoDB.open(GRAPH_DB_PATH)
            logger.info(f"[GraphBroker] Grafeo opened: nodes={self._db.node_count}, edges={self._db.edge_count}")
        except Exception as e:
            logger.error(f"[GraphBroker] Cannot open Grafeo: {e}")
            return

        self._running = True
        self._thread = threading.Thread(target=self._serve, daemon=True, name="grafeo-broker")
        self._thread.start()
        logger.info(f"[GraphBroker] Listening on {ZMQ_ENDPOINT}")

    def _serve(self):
        """REP loop — receives JSON requests, returns JSON responses."""
        ctx = zmq.Context()
        socket = ctx.socket(zmq.REP)
        socket.setsockopt(zmq.RCVTIMEO, 5000)  # 5s timeout for graceful shutdown
        socket.bind(ZMQ_ENDPOINT)

        while self._running:
            try:
                msg = socket.recv_string()
                req = json.loads(msg)
                resp = self._handle(req)
                socket.send_string(json.dumps(resp, default=str))
            except zmq.Again:
                continue  # timeout, check _running
            except Exception as e:
                try:
                    socket.send_string(json.dumps({"ok": False, "error": str(e)}))
                except Exception:
                    pass

        socket.close()
        ctx.term()
        if self._db:
            try:
                self._db.wal_checkpoint()
            except Exception:
                pass
            self._db.close()
        logger.info("[GraphBroker] Stopped")

    def _handle(self, req: dict) -> dict:
        """Route request to appropriate handler."""
        action = req.get("action", "")
        try:
            if action == "add_edge":
                return self._do_add_edge(req)
            elif action == "cypher":
                return self._do_cypher(req)
            elif action == "add_causal_edge":
                return self._do_add_causal_edge(req)
            elif action == "add_agent_node":
                return self._do_add_agent_node(req)
            elif action == "export_gnn":
                return self._do_export_gnn(req)
            elif action == "stats":
                return self._do_stats()
            elif action == "ping":
                return {"ok": True, "pong": True}
            else:
                return {"ok": False, "error": f"Unknown action: {action}"}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def _find_or_create_node(self, name: str, graph_type: str, labels=None, props=None):
        escaped = name.replace("'", "\\'")
        try:
            result = self._db.execute_cypher(
                f"MATCH (n) WHERE n.name = '{escaped}' RETURN id(n) AS nid LIMIT 1"
            )
            if result and len(result) > 0:
                nid = result[0].get("nid")
                if nid is not None:
                    return self._db.get_node(nid)
        except Exception:
            pass
        node_labels = labels or [graph_type.capitalize()]
        node_props = {"name": name, "graph_type": graph_type,
                      "created_at": datetime.utcnow().isoformat()}
        if props:
            node_props.update(props)
        return self._db.create_node(node_labels, node_props)

    def _find_edge(self, src_id, tgt_id, relation):
        try:
            result = self._db.execute_cypher(
                f"MATCH (a)-[r:{relation}]->(b) WHERE id(a) = {src_id} AND id(b) = {tgt_id} "
                f"RETURN id(r) AS rid LIMIT 1"
            )
            if result and len(result) > 0:
                rid = result[0].get("rid")
                if rid is not None:
                    return self._db.get_edge(rid)
        except Exception:
            pass
        return None

    def _do_add_edge(self, req):
        gt = req["graph_type"]
        src = req["source"].lower().strip()
        tgt = req["target"].lower().strip()
        rel = req["relation"]
        weight = req.get("weight", 1.0)
        metadata = req.get("metadata")

        if gt not in VALID_GRAPH_TYPES:
            return {"ok": False, "error": f"Invalid graph_type: {gt}"}

        src_node = self._find_or_create_node(src, gt)
        tgt_node = self._find_or_create_node(tgt, gt)

        existing = self._find_edge(src_node.id, tgt_node.id, rel)
        if existing:
            # Hebbian reinforcement
            old_weight = 1.0
            try:
                w_result = self._db.execute_cypher(
                    f"MATCH ()-[r]->() WHERE id(r) = {existing.id} RETURN r.weight AS w"
                )
                if w_result:
                    old_weight = w_result[0].get("w", 1.0)
            except Exception:
                pass
            new_weight = min(old_weight + 0.1, 10.0)
            self._db.set_edge_property(existing.id, "weight", new_weight)
            self._db.set_edge_property(existing.id, "timestamp", datetime.utcnow().isoformat())
            return {"ok": True, "hebbian": True, "weight": new_weight}

        props = {"weight": weight, "graph_type": gt, "relation": rel,
                 "timestamp": datetime.utcnow().isoformat()}
        if metadata:
            props["metadata_json"] = json.dumps(metadata, ensure_ascii=False)
        self._db.create_edge(src_node.id, tgt_node.id, rel, props)
        return {"ok": True, "created": True}

    def _do_cypher(self, req):
        query = req["query"]
        results = self._db.execute_cypher(query)
        rows = [dict(r) for r in results] if results else []
        return {"ok": True, "rows": rows, "count": len(rows)}

    def _do_add_causal_edge(self, req):
        req["graph_type"] = "causal"
        req["relation"] = "causes"
        req["metadata"] = {
            "time_lag": req.get("time_lag", 0),
            "p_value": req.get("p_value", 0.0),
            "regime": req.get("regime", "_global"),
            "method": "PCMCI+",
            "discovered_at": datetime.utcnow().isoformat(),
        }
        return self._do_add_edge(req)

    def _do_add_agent_node(self, req):
        name = req["agent_type"].lower()
        self._find_or_create_node(name, "entity", labels=["Agent"],
                                   props={"organ": req.get("organ", "")})
        return {"ok": True}

    def _do_export_gnn(self, req):
        graph_type = req.get("graph_type")
        try:
            if graph_type:
                nodes_r = self._db.execute_cypher(
                    f"MATCH (n)-[r]-() WHERE r.graph_type = '{graph_type}' "
                    f"RETURN DISTINCT n.name AS name"
                )
            else:
                nodes_r = self._db.execute_cypher("MATCH (n) RETURN n.name AS name")

            node_names = list(set(r.get("name", "") for r in (nodes_r or [])))
            node_to_idx = {n: i for i, n in enumerate(node_names)}

            if graph_type:
                edges_r = self._db.execute_cypher(
                    f"MATCH (a)-[r]->(b) WHERE r.graph_type = '{graph_type}' "
                    f"RETURN a.name AS src, b.name AS tgt, r.weight AS weight"
                )
            else:
                edges_r = self._db.execute_cypher(
                    "MATCH (a)-[r]->(b) RETURN a.name AS src, b.name AS tgt, r.weight AS weight"
                )

            sources, targets, weights = [], [], []
            for e in (edges_r or []):
                s, t = e.get("src", ""), e.get("tgt", "")
                if s in node_to_idx and t in node_to_idx:
                    sources.append(node_to_idx[s])
                    targets.append(node_to_idx[t])
                    weights.append(e.get("weight", 1.0))

            return {"ok": True, "node_ids": node_names, "edge_index": [sources, targets],
                    "edge_attr": weights}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def _do_stats(self):
        try:
            mem = self._db.memory_usage() if callable(self._db.memory_usage) else self._db.memory_usage
        except Exception:
            mem = "N/A"
        return {"ok": True, "path": GRAPH_DB_PATH, "mode": "broker",
                "node_count": self._db.node_count, "edge_count": self._db.edge_count,
                "memory_usage": mem}

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=10)


_broker = _GrafeoBroker()


def start_graph_broker():
    """Start the Grafeo ZMQ broker. Call from scheduler process only."""
    _broker.start()


def stop_graph_broker():
    """Stop the broker."""
    _broker.stop()


# ═══════════════════════════════════════════════════════════════════
# CLIENT — used by all processes (bot, rag, scheduler)
# ═══════════════════════════════════════════════════════════════════

class GraphStore:
    """ZMQ client that sends requests to the Grafeo broker."""

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
        if not ZMQ_AVAILABLE:
            raise ImportError("pyzmq not installed. Run: pip install pyzmq")
        self._ctx = zmq.Context()
        self._socket = self._ctx.socket(zmq.REQ)
        self._socket.setsockopt(zmq.RCVTIMEO, 10000)  # 10s timeout
        self._socket.setsockopt(zmq.SNDTIMEO, 5000)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.connect(ZMQ_ENDPOINT)
        self._send_lock = threading.Lock()
        self._initialized = True

        # Check broker alive
        try:
            resp = self._request({"action": "ping"})
            if resp.get("pong"):
                stats = self._request({"action": "stats"})
                logger.info(f"[GraphStore] Connected to broker: "
                             f"nodes={stats.get('node_count', '?')}, edges={stats.get('edge_count', '?')}")
            else:
                logger.warning("[GraphStore] Broker not responding — graph operations will fail gracefully")
        except Exception:
            logger.warning("[GraphStore] Broker not available — graph operations will fail gracefully")

    def _request(self, req: dict) -> dict:
        """Send request to broker, return response."""
        with self._send_lock:
            try:
                self._socket.send_string(json.dumps(req, default=str))
                resp = json.loads(self._socket.recv_string())
                return resp
            except zmq.Again:
                return {"ok": False, "error": "timeout"}
            except Exception as e:
                # Socket may be in bad state — recreate
                try:
                    self._socket.close()
                    self._socket = self._ctx.socket(zmq.REQ)
                    self._socket.setsockopt(zmq.RCVTIMEO, 10000)
                    self._socket.setsockopt(zmq.SNDTIMEO, 5000)
                    self._socket.setsockopt(zmq.LINGER, 0)
                    self._socket.connect(ZMQ_ENDPOINT)
                except Exception:
                    pass
                return {"ok": False, "error": str(e)}

    # ─── MAGMA-Compatible API ───────────────────────────────────

    def add_edge(self, graph_type: str, source: str, relation: str,
                 target: str, weight: float = 1.0,
                 metadata: Optional[Dict] = None) -> bool:
        resp = self._request({
            "action": "add_edge", "graph_type": graph_type,
            "source": source, "relation": relation, "target": target,
            "weight": weight, "metadata": metadata,
        })
        return resp.get("ok", False)

    def cypher(self, query: str) -> List[Dict]:
        resp = self._request({"action": "cypher", "query": query})
        return resp.get("rows", [])

    def query(self, query_text: str, graph_type: str = "semantic",
              max_hops: int = 1) -> List[Dict]:
        cypher_q = (
            f"MATCH (n)-[r*1..{max_hops}]->(m) "
            f"WHERE n.name CONTAINS '{query_text.lower()}' "
            f"RETURN n.name AS source, m.name AS target LIMIT 20"
        )
        return self.cypher(cypher_q)

    def traverse(self, start_node: str, graph_type: str = "semantic",
                 max_hops: int = 2) -> List[Dict]:
        cypher_q = (
            f"MATCH path = (a)-[*1..{max_hops}]->(b) "
            f"WHERE a.name = '{start_node.lower()}' "
            f"RETURN a.name AS from_node, b.name AS to_node LIMIT 50"
        )
        return self.cypher(cypher_q)

    # ─── Causal API ─────────────────────────────────────────────

    def add_causal_edge(self, source_var: str, target_var: str,
                        strength: float, time_lag: int = 0,
                        p_value: float = 0.0, regime: str = "_global"):
        resp = self._request({
            "action": "add_causal_edge", "source": source_var, "target": target_var,
            "weight": strength, "time_lag": time_lag, "p_value": p_value, "regime": regime,
        })
        return resp.get("ok", False)

    def get_causal_parents(self, target_var: str, regime: str = "_global") -> List[Dict]:
        return self.cypher(
            f"MATCH (a)-[r:causes]->(b) WHERE b.name = '{target_var.lower()}' "
            f"RETURN a.name AS parent, r.weight AS strength, r.metadata_json AS meta "
            f"ORDER BY r.weight DESC"
        )

    def get_causal_children(self, source_var: str) -> List[Dict]:
        return self.cypher(
            f"MATCH (a)-[r:causes]->(b) WHERE a.name = '{source_var.lower()}' "
            f"RETURN b.name AS child, r.weight AS strength ORDER BY r.weight DESC"
        )

    # ─── Agent API ──────────────────────────────────────────────

    def add_agent_node(self, agent_type: str, organ: str, properties=None):
        return self._request({
            "action": "add_agent_node", "agent_type": agent_type, "organ": organ,
        })

    def add_agent_interaction(self, agent_a: str, agent_b: str,
                               interaction_type: str = "coordinates", weight: float = 1.0):
        return self.add_edge("entity", agent_a, interaction_type, agent_b, weight)

    # ─── GNN Export ─────────────────────────────────────────────

    def export_for_gnn(self, graph_type: Optional[str] = None) -> Dict:
        resp = self._request({"action": "export_gnn", "graph_type": graph_type})
        if resp.get("ok"):
            return {"node_ids": resp["node_ids"], "node_features": [],
                    "edge_index": resp["edge_index"], "edge_attr": resp["edge_attr"]}
        return {"node_ids": [], "node_features": [], "edge_index": [[], []], "edge_attr": []}

    # ─── Stats ──────────────────────────────────────────────────

    def get_stats(self) -> Dict:
        resp = self._request({"action": "stats"})
        if resp.get("ok"):
            return resp
        return {"path": GRAPH_DB_PATH, "mode": "disconnected",
                "node_count": 0, "edge_count": 0}

    def save(self):
        pass  # Broker handles persistence

    def close(self):
        try:
            self._socket.close()
            self._ctx.term()
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════════
# Singleton
# ═══════════════════════════════════════════════════════════════════

_graph_store = None
_graph_store_lock = threading.Lock()


def get_graph_store() -> GraphStore:
    global _graph_store
    if _graph_store is None:
        with _graph_store_lock:
            if _graph_store is None:
                _graph_store = GraphStore()
    return _graph_store
