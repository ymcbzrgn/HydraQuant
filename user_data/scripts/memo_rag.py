import sqlite3
import logging
import os
import sys
import json
from typing import List, Dict, Any

sys.path.append(os.path.dirname(__file__))
from ai_config import AI_DB_PATH
from llm_router import LLMRouter

logger = logging.getLogger(__name__)

class MemoRAG:
    """
    MemoRAG: Global Corpus Memory & Draft Generation
    Maintains a highly compressed, global "understanding" of the entire database.
    Instead of searching raw fragments immediately, it generates a 'draft' answer 
    from its global memory, which is then used to guide the actual retrieval.
    """
    
    def __init__(self, db_path=AI_DB_PATH, llm_router: LLMRouter = None):
        self.db_path = db_path
        self.router = llm_router if llm_router else LLMRouter(temperature=0.2)
        self._init_db()
        
    def _init_db(self):
        """Initialize SQLite tables for global memory blocks."""
        try:
            with sqlite3.connect(self.db_path, timeout=30) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS memorag_global (
                        id INTEGER PRIMARY KEY DEFAULT 1,
                        summary TEXT,
                        last_processed_doc_id TEXT,
                        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        CHECK (id = 1)
                    )
                ''')
                
                # Singletons need an initial row
                cursor.execute("INSERT OR IGNORE INTO memorag_global (id, summary) VALUES (1, 'Initial empty global memory.')")
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to initialize MemoRAG database: {e}")

    def get_global_memory(self) -> str:
        """Fetch the current globally compressed corpus summary."""
        try:
            with sqlite3.connect(self.db_path, timeout=30) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT summary FROM memorag_global WHERE id = 1")
                row = cursor.fetchone()
                return row[0] if row else ""
        except Exception as e:
            logger.error(f"Failed to read MemoRAG global memory: {e}")
            return ""

    def update_global_memory(self, new_texts: List[str]):
        """
        Incrementally update the global memory by feeding it new documents.
        This compresses the new texts INTO the existing global memory.
        """
        if not new_texts:
            return
            
        current_memory = self.get_global_memory()
        
        # Combine texts (keep it relatively small to avoid token limits, limit to top 5)
        new_batch = "\\n\\n---\\n\\n".join(new_texts[:5])
        
        prompt = f"""Update the Global Memory by integrating new information.

CURRENT GLOBAL MEMORY:
{current_memory}

NEW INFORMATION:
{new_batch}

COMPRESSION RULES:
1. Keep under 500 words. Every word must earn its place.
2. PRIORITIZE (in order): macroeconomic regime changes > major technical level breaks > dominant narratives > institutional flows > regulatory developments
3. DISCARD: duplicate info already in memory, ephemeral intraday noise, vague/unattributed claims
4. PRESERVE: specific numbers (prices, indicator values, dates), causal relationships, regime classifications
5. UPDATE: if new info contradicts existing memory, update to reflect the LATEST state
6. TIMESTAMP: include approximate dates for major events (e.g., "As of March 2026, BTC...")
7. Return ONLY the updated compressed memory — no preamble, no explanation, no markdown."""

        try:
            from langchain_core.messages import SystemMessage, HumanMessage
            response = self.router.invoke([
                SystemMessage(content="You are a Global Memory Compressor for a crypto trading system. Maintain a concise, accurate, timestamped summary of market state. Prioritize actionable information over narrative. NEVER fabricate data — only compress what's provided."),
                HumanMessage(content=prompt)
            ], priority="low")
            
            content_raw = response.content
            if isinstance(content_raw, list):
                content_raw = " ".join([b.get("text", "") for b in content_raw if "text" in b])
                
            updated_memory = content_raw.strip()
            
            with sqlite3.connect(self.db_path, timeout=30) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "UPDATE memorag_global SET summary = ?, updated_at = CURRENT_TIMESTAMP WHERE id = 1",
                    (updated_memory,)
                )
                conn.commit()
                logger.info(f"MemoRAG global memory updated successfully. Size: {len(updated_memory)} chars.")
        except Exception as e:
            logger.error(f"Failed to update MemoRAG global memory: {e}")

    def generate_draft(self, query: str) -> str:
        """
        Generate a theoretical draft answer based purely on the global compressed memory.
        This draft is used to improve dense retrieval (similar to HyDE but grounded in real global context).
        """
        global_memory = self.get_global_memory()
        
        if len(global_memory) < 50:
            # Not enough memory built up yet to provide a useful draft
            return query
            
        prompt = f"""Generate a keyword-dense draft answer using ONLY the global memory below.

GLOBAL MEMORY:
{global_memory}

QUERY: "{query}"

RULES:
1. Use ONLY information from the global memory above. Do NOT add external knowledge.
2. Be SPECIFIC: include price levels, indicator values, dates, and entity names from the memory.
3. Include KEYWORD-DENSE phrases that would appear in real market analysis documents (to improve retrieval).
4. If the memory doesn't contain relevant information, output the original query unchanged.
5. Keep it concise: 2-3 sentences maximum.
6. This draft will be used as a search query — optimize for retrieval relevance, not human readability."""

        try:
            from langchain_core.messages import SystemMessage, HumanMessage
            response = self.router.invoke([
                SystemMessage(content="You generate keyword-dense draft answers for retrieval augmentation. Use ONLY the provided global memory — NEVER fabricate data. If memory lacks relevant info, return the original query."),
                HumanMessage(content=prompt)
            ], priority="low")
            
            content_raw = response.content
            if isinstance(content_raw, list):
                content_raw = " ".join([b.get("text", "") for b in content_raw if "text" in b])
                
            draft = content_raw.strip()
            return draft
        except Exception as e:
            logger.error(f"Failed to generate MemoRAG draft: {e}")
            return query  # Fallback to original query
