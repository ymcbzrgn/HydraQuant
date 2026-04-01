"""
GraphRAG — Microsoft-inspired knowledge graph community summarization.

Builds on existing kg_entities + kg_relationships tables.
Adds community detection (connected components) + LLM-generated summaries.
Provides local (entity neighborhood) and global (community summary) retrieval.

Different from MAGMA (temporal edge graph) and GAM-RAG (entity graph retrieval):
GraphRAG adds community-level summarization for global questions like
"What's the overall DeFi ecosystem sentiment?" or "Which sectors are correlated?"
"""

import sqlite3
import json
import logging
from typing import List, Dict, Optional

import os
import sys
sys.path.append(os.path.dirname(__file__))

from ai_config import AI_DB_PATH

logger = logging.getLogger(__name__)


class GraphRAG:
    """Community-based graph retrieval with LLM summarization."""

    def __init__(self, db_path: str = AI_DB_PATH):
        self.db_path = db_path
        self._init_db()

    def _get_conn(self):
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=30000")
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        try:
            with self._get_conn() as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS graph_communities (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        community_id INTEGER,
                        member_entities TEXT,
                        summary TEXT,
                        level INTEGER DEFAULT 0,
                        entity_count INTEGER DEFAULT 0,
                        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                conn.commit()
        except Exception as e:
            logger.error(f"[GraphRAG] Init failed: {e}")

    def build_communities(self) -> List[List[str]]:
        """Detect communities using connected components on kg_relationships."""
        try:
            with self._get_conn() as conn:
                entities = conn.execute("SELECT name FROM kg_entities").fetchall()
                relationships = conn.execute(
                    "SELECT source_entity, target_entity FROM kg_relationships"
                ).fetchall()

            if not entities:
                logger.info("[GraphRAG] No entities in knowledge graph")
                return []

            # Build adjacency list
            adj = {}
            for r in relationships:
                src, tgt = r["source_entity"], r["target_entity"]
                adj.setdefault(src, set()).add(tgt)
                adj.setdefault(tgt, set()).add(src)

            # BFS connected components
            visited = set()
            communities = []
            for e in [e["name"] for e in entities]:
                if e in visited:
                    continue
                component = []
                queue = [e]
                while queue:
                    node = queue.pop(0)
                    if node in visited:
                        continue
                    visited.add(node)
                    component.append(node)
                    queue.extend(adj.get(node, set()) - visited)
                if len(component) >= 2:
                    communities.append(component)

            logger.info(f"[GraphRAG] Found {len(communities)} communities "
                       f"from {len(entities)} entities")
            return communities

        except Exception as e:
            logger.error(f"[GraphRAG] Community detection failed: {e}")
            return []

    def summarize_communities(self, communities: List[List[str]], llm_router=None):
        """Generate LLM summaries for each community."""
        if not communities:
            return

        try:
            with self._get_conn() as conn:
                conn.execute("DELETE FROM graph_communities")

                for i, members in enumerate(communities[:50]):
                    members_str = ", ".join(members[:20])

                    summary = f"Community of {len(members)} entities: {members_str}"
                    if llm_router:
                        try:
                            from langchain_core.messages import HumanMessage
                            prompt = (
                                f"In 2-3 sentences, describe the trading relationship between "
                                f"these crypto entities for a trader:\n{members_str}"
                            )
                            response = llm_router.invoke(
                                [HumanMessage(content=prompt)],
                                temperature=0.3, priority="low"
                            )
                            if hasattr(response, 'content') and response.content:
                                summary = response.content
                        except Exception:
                            pass

                    conn.execute(
                        "INSERT INTO graph_communities "
                        "(community_id, member_entities, summary, level, entity_count) "
                        "VALUES (?, ?, ?, ?, ?)",
                        (i, json.dumps(members[:20]), summary, 0, len(members))
                    )
                conn.commit()

            logger.info(f"[GraphRAG] Summarized {min(len(communities), 50)} communities")

        except Exception as e:
            logger.error(f"[GraphRAG] Summarization failed: {e}")

    def query_local(self, entity: str, max_hops: int = 1) -> List[Dict]:
        """Get 1-hop neighbors of an entity from knowledge graph."""
        try:
            with self._get_conn() as conn:
                rows = conn.execute("""
                    SELECT source_entity, target_entity, relation_type, weight
                    FROM kg_relationships
                    WHERE source_entity = ? OR target_entity = ?
                    ORDER BY weight DESC LIMIT 20
                """, (entity, entity)).fetchall()
            return [dict(r) for r in rows]
        except Exception:
            return []

    def query_global(self, query: str = None, top_k: int = 5) -> List[Dict]:
        """Get most relevant community summaries."""
        try:
            with self._get_conn() as conn:
                if query:
                    # Simple keyword matching on summary
                    keywords = query.lower().split()[:5]
                    rows = conn.execute("""
                        SELECT community_id, summary, member_entities, entity_count
                        FROM graph_communities
                        ORDER BY entity_count DESC
                        LIMIT ?
                    """, (top_k,)).fetchall()
                else:
                    rows = conn.execute("""
                        SELECT community_id, summary, member_entities, entity_count
                        FROM graph_communities
                        ORDER BY entity_count DESC
                        LIMIT ?
                    """, (top_k,)).fetchall()
            return [dict(r) for r in rows]
        except Exception:
            return []

    def get_stats(self) -> dict:
        """Get graph stats."""
        try:
            with self._get_conn() as conn:
                communities = conn.execute(
                    "SELECT COUNT(*) as n FROM graph_communities"
                ).fetchone()
                entities = conn.execute(
                    "SELECT COUNT(*) as n FROM kg_entities"
                ).fetchone()
                relationships = conn.execute(
                    "SELECT COUNT(*) as n FROM kg_relationships"
                ).fetchone()
            return {
                "communities": communities["n"] if communities else 0,
                "entities": entities["n"] if entities else 0,
                "relationships": relationships["n"] if relationships else 0,
            }
        except Exception:
            return {}
