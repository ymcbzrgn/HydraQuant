"""
Görev 1: RAPTOR Enhanced (Hiyerarşik Özet Ağacı)
Recursive Abstractive Processing for Tree-Organized Retrieval
"""

import os
import json
import logging
import sqlite3
from typing import List, Dict, Any

from ai_config import AI_DB_PATH
from llm_router import LLMRouter
from langchain_core.messages import HumanMessage

logger = logging.getLogger(__name__)

class RAPTORTree:
    def __init__(self, llm_router: LLMRouter = None):
        """
        RAPTOR ağacı oluşturucu.
        Level 0: Orijinal chunklar (leaf)
        Level 1: Her 5-10 chunk'ın özeti
        Level 2: Level 1 özetlerinin meta-özeti
        """
        self.router = llm_router or LLMRouter()
        self._init_db()

    def _init_db(self):
        """RAPTOR node'larını FTS5 SQLite üzerinde tut."""
        try:
            with sqlite3.connect(AI_DB_PATH, timeout=10) as conn:
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS raptor_nodes (
                        node_id TEXT PRIMARY KEY,
                        level INTEGER,
                        content TEXT,
                        children_ids TEXT,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                # FTS5 Virtual Table for semantic-like keyword routing if embeddings aren't used
                conn.execute('''
                    CREATE VIRTUAL TABLE IF NOT EXISTS raptor_fts 
                    USING fts5(node_id UNINDEXED, level UNINDEXED, content)
                ''')
        except Exception as e:
            logger.error(f"RAPTOR DB initialization failed: {e}")

    def build_tree(self, chunks: List[Dict[str, Any]], cluster_size: int = 5) -> Dict[str, List[Dict[str, Any]]]:
        """
        Leaf chunk'lardan 3 seviyeli bir özet ağacı çıkar.
        chunks: [{'id': 'c1', 'text': '...', 'metadata': {...}}, ...]
        """
        if not chunks:
            return {"level_0": [], "level_1": [], "level_2": []}
            
        # Level 0 (Leaves)
        level_0 = chunks
        
        # Level 1 (Cluster summaries)
        level_1 = self._build_next_level(level_0, cluster_size, 1)
        
        # Level 2 (Meta-summaries)
        level_2 = self._build_next_level(level_1, cluster_size, 2)
        
        self._persist_tree(level_0, 0)
        self._persist_tree(level_1, 1)
        self._persist_tree(level_2, 2)
        
        return {
            "level_0": level_0,
            "level_1": level_1,
            "level_2": level_2
        }

    def _build_next_level(self, lower_level_nodes: List[Dict[str, Any]], cluster_size: int, target_level: int) -> List[Dict[str, Any]]:
        """Alt seviyedeki node'ları kümeleyip LLM ile özetlerini çıkarır."""
        next_level = []
        
        # Basit ardışık kümeleme (Gerçekte embeddings ile K-Means de yapılabilir ama LLM token sınırları için ardışık yeterli)
        for i in range(0, len(lower_level_nodes), cluster_size):
            cluster = lower_level_nodes[i:i + cluster_size]
            cluster_texts = [node.get('text', '') for node in cluster]
            cluster_ids = [node.get('id', str(idx)) for idx, node in enumerate(cluster)]
            
            summary = self._summarize_cluster(cluster_texts, target_level)
            
            node_id = f"lvl{target_level}_cluster_{i}"
            next_level.append({
                "id": node_id,
                "text": summary,
                "children": cluster_ids
            })
            
        return next_level

    def _summarize_cluster(self, texts: List[str], level: int) -> str:
        """LLM kullanarak küme metinlerini özetler."""
        combined_text = "\n---\n".join(texts)
        level_name = "Meta-Summary" if level == 2 else "Cluster Summary"
        
        prompt = f"""
As an expert crypto market analyst, create a concise {level_name} capturing the core themes, 
sentiment, and factual data points from the following linked texts. 
Maintain specific metrics if they appear across multiple texts.

Texts:
{combined_text}

Provide only the summary text without any preamble.
"""
        response = self.router.invoke([HumanMessage(content=prompt)])
        return str(response.content).strip()

    def _persist_tree(self, nodes: List[Dict[str, Any]], level: int):
        """Ağaç node'larını DB'ye kaydet."""
        try:
            with sqlite3.connect(AI_DB_PATH, timeout=10) as conn:
                for node in nodes:
                    node_id = node.get('id', f"lvl{level}_unknown")
                    content = node.get('text', '')
                    children = json.dumps(node.get('children', []))
                    
                    conn.execute(
                        "INSERT OR IGNORE INTO raptor_nodes (node_id, level, content, children_ids) VALUES (?, ?, ?, ?)",
                        (node_id, level, content, children)
                    )
                    conn.execute(
                        "INSERT OR IGNORE INTO raptor_fts (node_id, level, content) VALUES (?, ?, ?)",
                        (node_id, level, content)
                    )
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to persist RAPTOR level {level}: {e}")

    def query(self, question: str, tree_or_db: bool = True) -> List[Dict[str, Any]]:
        """
        Sorunun karmaşıklığına göre doğru seviyeden bilgi çeker. 
        RAG pipeline'ına ek bağlam olarak döner.
        """
        # Determine query abstraction level
        abstraction_prompt = f"""
Are you asking about a specific metric/event (Leaf), a sector/asset theme (Cluster), or the general broad market trend (Meta)?
Question: "{question}"
Reply exactly with one word: LEAF, CLUSTER, or META.
"""
        response = self.router.invoke([HumanMessage(content=abstraction_prompt)])
        classifier = str(response.content).strip().upper()
        
        if "META" in classifier:
            target_level = 2
        elif "CLUSTER" in classifier:
            target_level = 1
        else:
            target_level = 0
            
        # SQLite FTS Search over the targeted level
        results = []
        try:
            with sqlite3.connect(AI_DB_PATH, timeout=10) as conn:
                # Basic FTS5 match using term extraction (simplified)
                # In production raptor, we'd embed the question and do cosine against level X
                # For this implementation, we extract the top keywords or just return the highest level if META
                
                if target_level == 2:
                    # Return all level 2 meta summaries (usually very few)
                    cursor = conn.execute("SELECT node_id, content FROM raptor_nodes WHERE level = 2 ORDER BY created_at DESC LIMIT 3")
                else:
                    # Sanitize question for FTS5
                    clean_q = "".join(c if c.isalnum() else " " for c in question).strip()
                    cursor = conn.execute(f"SELECT node_id, content FROM raptor_fts WHERE level = ? AND content MATCH ? ORDER BY rank LIMIT 3", (target_level, clean_q))
                    
                for row in cursor.fetchall():
                    results.append({"id": row[0], "text": row[1], "level": target_level})
                    
        except Exception as e:
            logger.error(f"RAPTOR query failed: {e}")
            
        return results
