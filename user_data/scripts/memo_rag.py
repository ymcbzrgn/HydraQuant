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
        
        prompt = f"""You are maintaining a highly compressed GLOBAL MEMORY of a cryptocurrency market database.
Current Global Memory:
{current_memory}

New Information to Integrate:
{new_batch}

Task: Update the Global Memory to include the new information. 
Keep it under 500 words. Focus on macroeconomic shifts, major technical levels, dominant narratives, and structural changes.
Discard noise, duplicate information, or highly localized ephemeral price actions.
Return ONLY the updated compressed memory."""

        try:
            from langchain_core.messages import SystemMessage, HumanMessage
            response = self.router.invoke([
                SystemMessage(content="You are a Global Memory Compressor."),
                HumanMessage(content=prompt)
            ])
            
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
            
        prompt = f"""You have access to a compressed global memory of the entire cryptocurrency market database.
Global Memory:
{global_memory}

User Query: "{query}"

Task: Using ONLY your global memory, generate a preliminary "draft" answer to the query. 
This draft will be used by a search engine to find the actual detailed documents.
Keep the draft concise, factual, and dense with keywords that should appear in the real answer.
If the global memory doesn't contain the answer, output the query itself."""

        try:
            from langchain_core.messages import SystemMessage, HumanMessage
            response = self.router.invoke([
                SystemMessage(content="You are a Draft Generator."),
                HumanMessage(content=prompt)
            ])
            
            content_raw = response.content
            if isinstance(content_raw, list):
                content_raw = " ".join([b.get("text", "") for b in content_raw if "text" in b])
                
            draft = content_raw.strip()
            return draft
        except Exception as e:
            logger.error(f"Failed to generate MemoRAG draft: {e}")
            return query  # Fallback to original query
