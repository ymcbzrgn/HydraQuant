import sqlite3
import logging
import os
import sys
from typing import List, Dict, Any

sys.path.append(os.path.dirname(__file__))
from ai_config import AI_DB_PATH

logger = logging.getLogger(__name__)

class MAGMAMemory:
    """
    MAGMA: Multi-Graph Agentic Memory
    Organizes information across 4 orthogonal graphs to provide distinct perspectives:
    1. Semantic: Meaning relationships (oversold -> reversal)
    2. Temporal: Event sequencing (Fed decision -> market reaction)
    3. Causal: Cause-and-effect (rate hike -> dollar strength)
    4. Entity: Asset correlations (BTC -> ETH)
    """
    
    VALID_GRAPHS = {"semantic", "temporal", "causal", "entity"}
    
    def __init__(self, db_path=AI_DB_PATH):
        self.db_path = db_path
        self._init_db()
        
    def _init_db(self):
        """Initialize the SQLite tables for MAGMA edges."""
        try:
            with sqlite3.connect(self.db_path, timeout=30) as conn:
                cursor = conn.cursor()
                # Use adjacency list representation for the graphs
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS magma_edges (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        graph_type TEXT NOT NULL,
                        source TEXT NOT NULL,
                        relation TEXT NOT NULL,
                        target TEXT NOT NULL,
                        weight REAL DEFAULT 1.0,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        metadata TEXT DEFAULT '{}',
                        UNIQUE(graph_type, source, relation, target)
                    )
                ''')
                # Explicit index for fast O(1) adjacency lookups during traversal
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_magma_graphs ON magma_edges(graph_type, source)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_magma_pruning ON magma_edges(timestamp, weight)')
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to initialize MAGMA database: {e}")

    def add_edge(self, graph_type: str, source: str, relation: str, target: str, metadata: dict = None) -> bool:
        """
        Add an edge to one of the 4 graph types.
        Uses Hebbian learning semantics: if the edge already exists, increment its weight.
        """
        if graph_type not in self.VALID_GRAPHS:
            logger.error(f"Invalid MAGMA graph type: {graph_type}")
            return False
            
        import json
        meta_str = json.dumps(metadata) if metadata else "{}"
        
        try:
            with sqlite3.connect(self.db_path, timeout=30) as conn:
                cursor = conn.cursor()
                # Upsert logic to handle duplicate edges by increasing weight
                cursor.execute('''
                    INSERT INTO magma_edges (graph_type, source, relation, target, weight, timestamp, metadata)
                    VALUES (?, ?, ?, ?, 1.0, CURRENT_TIMESTAMP, ?)
                    ON CONFLICT(graph_type, source, relation, target) 
                    DO UPDATE SET 
                        weight = weight + 0.1,
                        timestamp = CURRENT_TIMESTAMP
                ''', (graph_type, source.lower(), relation.lower(), target.lower(), meta_str))
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Failed to add MAGMA edge: {e}")
            return False

    def traverse(self, start_node: str, graph_type: str, max_hops: int = 2) -> List[Dict[str, Any]]:
        """
        Traverse a specific graph type starting from a given node using BFS.
        Returns all edges encountered up to max_hops.
        """
        if graph_type not in self.VALID_GRAPHS:
            return []
            
        start_node = start_node.lower()
        visited = set()
        queue = [(start_node, 0)]
        found_edges = []
        
        try:
            with sqlite3.connect(self.db_path, timeout=30) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                while queue:
                    current_node, current_hop = queue.pop(0)
                    
                    if current_hop >= max_hops:
                        continue
                        
                    if current_node in visited:
                        continue
                        
                    visited.add(current_node)
                    
                    # O(1) lookup via index
                    cursor.execute('''
                        SELECT source, relation, target, weight, metadata, graph_type
                        FROM magma_edges
                        WHERE graph_type = ? AND source = ?
                        ORDER BY weight DESC LIMIT 20
                    ''', (graph_type, current_node))
                    
                    neighbors = cursor.fetchall()
                    
                    for row in neighbors:
                        # Append the edge
                        edge_dict = dict(row)
                        found_edges.append(edge_dict)
                        # Queue the target for next hop
                        target = row['target']
                        if target not in visited:
                            queue.append((target, current_hop + 1))
                            
            return found_edges
        except Exception as e:
            logger.error(f"MAGMA traversal failed: {e}")
            return []

    def query(self, question: str, graph_types: List[str] = None, max_hops: int = 2) -> List[Dict[str, Any]]:
        """
        Extract concepts from a question and traverse relevant graphs.
        Uses a lightweight keyword extraction (MVP) to find starting points.
        """
        if graph_types is None:
            graph_types = list(self.VALID_GRAPHS)
            
        # 1. Simple concept extraction (would normally use NLP/LLM, keeping MVP here)
        # For MVP, we look up existing nodes that intersect with words in the question
        words = set([w.lower().strip(',.?!"') for w in question.split() if len(w) > 3])
        
        starting_nodes = []
        try:
            with sqlite3.connect(self.db_path, timeout=30) as conn:
                cursor = conn.cursor()
                # Find any source nodes that match our words
                placeholders = ','.join(['?']*len(words))
                if placeholders:
                    cursor.execute(f"SELECT DISTINCT source FROM magma_edges WHERE source IN ({placeholders})", list(words))
                    starting_nodes = [r[0] for r in cursor.fetchall()]
        except Exception as e:
            logger.error(f"MAGMA query node extraction failed: {e}")
            
        # Fallback if no exact word match: just traverse from prominent nodes?
        # In a real setup, LLM extracts entities. Here we just return empty if no match.
        if not starting_nodes:
            # Let's try partial matching for at least one critical entity
            try:
                with sqlite3.connect(self.db_path, timeout=30) as conn:
                    cursor = conn.cursor()
                    for word in words:
                        cursor.execute("SELECT DISTINCT source FROM magma_edges WHERE source LIKE ? LIMIT 3", (f"%{word}%",))
                        matches = [r[0] for r in cursor.fetchall()]
                        starting_nodes.extend(matches)
            except Exception:
                pass
                
        starting_nodes = list(set(starting_nodes))
        
        # 2. Traverse graphs from found nodes
        all_edges = []
        for node in starting_nodes:
            for g_type in graph_types:
                if g_type in self.VALID_GRAPHS:
                    edges = self.traverse(node, g_type, max_hops)
                    all_edges.extend(edges)
                    
        # Remove direct duplicates while preserving order
        unique_edges = []
        seen = set()
        for e in all_edges:
            sig = (e['graph_type'], e['source'], e['relation'], e['target'])
            if sig not in seen:
                seen.add(sig)
                unique_edges.append(e)
                
        # Sort by weight descending
        unique_edges.sort(key=lambda x: x['weight'], reverse=True)
        return unique_edges[:50]  # Cap context size

    def prune(self, min_weight: float = 0.5, max_age_days: int = 180):
        """Clean up weak and old connections to keep the memory graph sharp."""
        try:
            with sqlite3.connect(self.db_path, timeout=30) as conn:
                cursor = conn.cursor()
                # Delete edges that haven't been reinforced (low weight) AND are old
                cursor.execute(
                    "DELETE FROM magma_edges WHERE weight < ? AND timestamp < datetime('now', ?)",
                    (min_weight, f"-{max_age_days} days")
                )
                deleted = cursor.rowcount
                conn.commit()
                logger.info(f"MAGMA Memory Pruned: Removed {deleted} weak/old edges.")
                return deleted
        except Exception as e:
            logger.error(f"MAGMA string pruning failed: {e}")
            return 0

    def get_stats(self) -> dict:
        """Return graph node/edge counts."""
        stats = {g: {"nodes": 0, "edges": 0} for g in self.VALID_GRAPHS}
        try:
            with sqlite3.connect(self.db_path, timeout=30) as conn:
                cursor = conn.cursor()
                # Count edges per graph type
                cursor.execute("SELECT graph_type, COUNT(*) FROM magma_edges GROUP BY graph_type")
                for row in cursor.fetchall():
                    g_type, count = row
                    if g_type in stats:
                        stats[g_type]["edges"] = count
                
                # Count unique nodes (sources + targets)
                cursor.execute('''
                    SELECT graph_type, COUNT(DISTINCT node) FROM (
                        SELECT graph_type, source as node FROM magma_edges
                        UNION
                        SELECT graph_type, target as node FROM magma_edges
                    ) GROUP BY graph_type
                ''')
                for row in cursor.fetchall():
                    g_type, count = row
                    if g_type in stats:
                        stats[g_type]["nodes"] = count
                        
        except Exception as e:
            logger.error(f"MAGMA stats failed: {e}")
        return stats
