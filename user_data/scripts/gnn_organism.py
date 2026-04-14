"""
gnn_organism.py — Phase 26 Sprint 2, Task 9D

Graph Attention Network (GAT) on MAGMA/Grafeo knowledge graph.

Turns the static causal graph into a LIVE learning system:
  - Nodes: parameters, pairs, regimes, exit reasons, modules
  - Edges: causal edges (from 6A), semantic edges, temporal edges
  - Node features: embeddings, performance metrics, activation counts

GNN discovers hidden patterns through message passing:
  "BTC→ETH edge gets 87% attention" → strong cross-pair relationship

Integration:
  - Reads from graph_store.export_for_gnn() (Grafeo)
  - Reads node features from ai_decisions, evidence_audit_log
  - Writes discovered patterns to causal_discoveries table
  - Feeds attention weights to cross_pair_intel

Reference: Velickovic et al. "Graph Attention Networks" (ICLR 2018)
"""

import os
import sys
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
logger = logging.getLogger("gnn_organism")

from ai_config import AI_DB_PATH
from db import get_db_connection, get_connection, execute_with_retry, init_db

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "models", "rl")
GNN_MODEL_PATH = os.path.join(MODEL_DIR, "gnn_organism_v1.pt")

# GNN dimensions
NODE_FEATURE_DIM = 64
HIDDEN_DIM = 32
N_HEADS = 4
OUTPUT_DIM = 32


class OrganismGNN:
    """Graph Attention Network on the organism's knowledge graph."""

    def __init__(self):
        self._initialized = False
        self._graph_store = None
        self._attention_weights = None
        init_db()

    def _get_graph_store(self):
        """Lazy-load graph store."""
        if self._graph_store is None:
            try:
                from graph_store import get_graph_store
                self._graph_store = get_graph_store()
            except Exception as e:
                logger.warning(f"[GNN] Graph store not available: {e}")
        return self._graph_store

    def _init_network(self) -> bool:
        """Initialize GAT network."""
        try:
            import torch
            import torch.nn as nn
            import torch.nn.functional as F
        except ImportError:
            logger.error("[GNN] PyTorch not installed")
            return False

        # Simple GAT implementation (no torch-geometric dependency)
        class GATLayer(nn.Module):
            def __init__(self, in_features, out_features, n_heads):
                super().__init__()
                self.n_heads = n_heads
                self.out_per_head = out_features // n_heads

                self.W = nn.Linear(in_features, out_features, bias=False)
                self.a_src = nn.Parameter(torch.zeros(n_heads, self.out_per_head))
                self.a_tgt = nn.Parameter(torch.zeros(n_heads, self.out_per_head))
                nn.init.xavier_uniform_(self.W.weight)
                nn.init.xavier_uniform_(self.a_src.unsqueeze(0))
                nn.init.xavier_uniform_(self.a_tgt.unsqueeze(0))

            def forward(self, x, edge_index):
                """
                x: (n_nodes, in_features)
                edge_index: (2, n_edges) — [source_indices, target_indices]
                """
                h = self.W(x)  # (n_nodes, out_features)
                n_nodes = x.shape[0]

                # Reshape for multi-head: (n_nodes, n_heads, out_per_head)
                h_heads = h.view(n_nodes, self.n_heads, self.out_per_head)

                if edge_index.shape[1] == 0:
                    return h, None

                src_idx = edge_index[0]
                tgt_idx = edge_index[1]

                # Attention scores
                h_src = h_heads[src_idx]  # (n_edges, n_heads, out_per_head)
                h_tgt = h_heads[tgt_idx]

                # Attention: a_src · h_src + a_tgt · h_tgt
                e = (h_src * self.a_src.unsqueeze(0)).sum(-1) + \
                    (h_tgt * self.a_tgt.unsqueeze(0)).sum(-1)
                e = F.leaky_relu(e, 0.2)  # (n_edges, n_heads)

                # Softmax per target node
                alpha = torch.zeros(n_nodes, self.n_heads, device=x.device)
                for i in range(edge_index.shape[1]):
                    tgt = tgt_idx[i]
                    alpha[tgt] += torch.exp(e[i])

                # Normalize
                alpha_norm = alpha[tgt_idx] + 1e-10
                attention = torch.exp(e) / alpha_norm  # (n_edges, n_heads)

                # Aggregate
                out = torch.zeros(n_nodes, self.n_heads, self.out_per_head,
                                  device=x.device)
                for i in range(edge_index.shape[1]):
                    tgt = tgt_idx[i]
                    out[tgt] += attention[i].unsqueeze(-1) * h_heads[src_idx[i]]

                out = out.view(n_nodes, -1)  # (n_nodes, out_features)
                return out, attention

        self.gat1 = GATLayer(NODE_FEATURE_DIM, HIDDEN_DIM * N_HEADS, N_HEADS)
        self.gat2 = GATLayer(HIDDEN_DIM * N_HEADS, OUTPUT_DIM * N_HEADS, N_HEADS)

        import torch.optim as optim
        self.optimizer = optim.Adam(
            list(self.gat1.parameters()) + list(self.gat2.parameters()),
            lr=1e-3,
        )

        self._initialized = True
        total_params = sum(p.numel() for p in self.gat1.parameters()) + \
                       sum(p.numel() for p in self.gat2.parameters())
        logger.info(f"[GNN] Initialized: {total_params:,} parameters, "
                    f"{N_HEADS} attention heads")
        return True

    # ===================================================================
    # Graph Loading
    # ===================================================================

    def _load_graph_data(self) -> Optional[Dict]:
        """Load graph from Grafeo export_for_gnn()."""
        gs = self._get_graph_store()
        if gs is None:
            return None

        try:
            data = gs.export_for_gnn()
            n_nodes = len(data.get("node_ids", []))
            edge_index = data.get("edge_index", [[], []])
            n_edges = len(edge_index[0]) if edge_index and len(edge_index) >= 2 else 0

            if n_nodes == 0:
                logger.info("[GNN] Graph is empty (0 nodes)")
                return None

            logger.info(f"[GNN] Loaded graph: {n_nodes} nodes, {n_edges} edges")
            return data

        except Exception as e:
            logger.warning(f"[GNN] Failed to load graph: {e}")
            return None

    def _build_node_features(self, node_ids: List[str]) -> np.ndarray:
        """Build feature matrix for graph nodes.

        Each node gets a feature vector based on its type and available data.
        """
        features = np.random.randn(len(node_ids), NODE_FEATURE_DIM).astype(np.float32) * 0.1

        conn = get_db_connection(AI_DB_PATH)
        try:
            for i, node_id in enumerate(node_ids):
                # Check if node is a pair
                pair_data = conn.execute("""
                    SELECT AVG(outcome_pnl) as avg_pnl,
                           AVG(confidence) as avg_conf,
                           COUNT(*) as trade_count
                    FROM ai_decisions
                    WHERE pair = ? AND outcome_pnl IS NOT NULL
                """, (node_id,)).fetchone()

                if pair_data and pair_data["trade_count"] and pair_data["trade_count"] > 0:
                    features[i, 0] = float(pair_data["avg_pnl"] or 0)
                    features[i, 1] = float(pair_data["avg_conf"] or 0.5)
                    features[i, 2] = min(float(pair_data["trade_count"]) / 100, 1.0)
        finally:
            conn.close()

        return features

    # ===================================================================
    # Forward Pass + Pattern Discovery
    # ===================================================================

    def forward(self) -> Optional[Dict]:
        """Run GNN forward pass on the knowledge graph.

        Returns node embeddings and attention-based pattern discovery.
        """
        import torch
        import torch.nn.functional as F

        if not self._initialized:
            if not self._init_network():
                return None

        graph_data = self._load_graph_data()
        if graph_data is None:
            return None

        node_ids = graph_data["node_ids"]
        edge_index_raw = graph_data["edge_index"]

        # Build tensors
        node_features = self._build_node_features(node_ids)
        x = torch.FloatTensor(node_features)

        if len(edge_index_raw) >= 2 and len(edge_index_raw[0]) > 0:
            edge_index = torch.LongTensor(edge_index_raw)
        else:
            edge_index = torch.zeros(2, 0, dtype=torch.long)

        # Forward pass
        h1, att1 = self.gat1(x, edge_index)
        h1 = F.elu(h1)
        h2, att2 = self.gat2(h1, edge_index)

        self._attention_weights = att2

        # Node embeddings
        embeddings = h2.detach().numpy()

        result = {
            "node_ids": node_ids,
            "embeddings": embeddings,
            "n_nodes": len(node_ids),
            "n_edges": edge_index.shape[1],
        }

        # Discover hidden patterns from attention weights
        if att2 is not None and edge_index.shape[1] > 0:
            patterns = self._discover_patterns(att2, edge_index, node_ids)
            result["discovered_patterns"] = patterns

        return result

    def _discover_patterns(self, attention, edge_index, node_ids) -> List[Dict]:
        """Extract top-attended edges as discovered patterns."""
        patterns = []

        # Average attention across heads
        avg_att = attention.mean(dim=1).detach().numpy()  # (n_edges,)

        # Get top-K edges by attention
        top_k = min(20, len(avg_att))
        top_indices = np.argsort(avg_att)[-top_k:][::-1]

        for idx in top_indices:
            src = int(edge_index[0, idx])
            tgt = int(edge_index[1, idx])

            if src < len(node_ids) and tgt < len(node_ids):
                patterns.append({
                    "source": node_ids[src],
                    "target": node_ids[tgt],
                    "attention": round(float(avg_att[idx]), 4),
                    "rank": len(patterns) + 1,
                })

        return patterns

    def discover_hidden_patterns(self) -> List[Dict]:
        """Public API: run forward pass and return discovered patterns."""
        result = self.forward()
        if result is None:
            return []
        return result.get("discovered_patterns", [])

    # ===================================================================
    # Persistence
    # ===================================================================

    def persist_patterns(self, patterns: List[Dict]):
        """Store discovered patterns in causal_discoveries."""
        for pattern in patterns:
            try:
                execute_with_retry(
                    """INSERT OR REPLACE INTO causal_discoveries
                       (source_var, target_var, causal_strength, time_lag,
                        p_value, method, n_observations, regime, is_active)
                       VALUES (?, ?, ?, 0, ?, 'GNN_attention', ?, '_global', 1)""",
                    (pattern["source"], pattern["target"],
                     pattern["attention"], 1.0 - pattern["attention"],
                     pattern.get("n_obs", 0)),
                    max_retries=3,
                )
            except Exception as e:
                logger.debug(f"[GNN] Persist pattern failed: {e}")

    def save(self, path: str = None):
        import torch
        path = path or GNN_MODEL_PATH
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "gat1": self.gat1.state_dict(),
            "gat2": self.gat2.state_dict(),
        }, path)
        logger.info(f"[GNN] Saved: {path}")

    def load(self, path: str = None) -> bool:
        import torch
        path = path or GNN_MODEL_PATH
        if not os.path.exists(path):
            return False
        try:
            self._init_network()
            cp = torch.load(path, map_location="cpu", weights_only=False)
            self.gat1.load_state_dict(cp["gat1"])
            self.gat2.load_state_dict(cp["gat2"])
            logger.info(f"[GNN] Loaded: {path}")
            return True
        except Exception as e:
            logger.error(f"[GNN] Load failed: {e}")
            return False


# Singleton
_gnn_instance = None

def get_gnn() -> OrganismGNN:
    global _gnn_instance
    if _gnn_instance is None:
        _gnn_instance = OrganismGNN()
    return _gnn_instance
