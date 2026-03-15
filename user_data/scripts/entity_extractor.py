import os
import sqlite3
import logging
import json
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))
logger = logging.getLogger(__name__)

from ai_config import AI_DB_PATH as DB_PATH

class Relationship(BaseModel):
    source: str = Field(description="The source entity (e.g., Bitcoin, SEC, Federal Reserve)")
    target: str = Field(description="The target entity (e.g., Coinbase, Inflation, Market)")
    relation: str = Field(description="The relationship description (e.g., regulates, impacts, partnered_with)")

class EntityExtractionResult(BaseModel):
    entities: List[str] = Field(description="List of key entities in the text")
    relationships: List[Relationship] = Field(description="List of relationships between entities")

class KnowledgeGraphManager:
    """
    LazyGraphRAG module: Extracts entities and relationships from text
    using an LLM and stores them in SQLite for on-demand graph traversals.
    Uses LLMRouter for unified rate limit tracking, multi-provider failover,
    and cost tracking instead of direct Gemini/Groq SDK calls.
    """
    def __init__(self, llm_router=None):
        from llm_router import LLMRouter
        self.router = llm_router if llm_router is not None else LLMRouter(temperature=0)
        self.parser = JsonOutputParser(pydantic_object=EntityExtractionResult)
        self._init_db_tables()

    def _get_db_connection(self):
        conn = sqlite3.connect(DB_PATH, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_db_tables(self):
        with self._get_db_connection() as conn:
            c = conn.cursor()
            c.execute('''
                CREATE TABLE IF NOT EXISTS kg_entities (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE NOT NULL
                )
            ''')
            c.execute('''
                CREATE TABLE IF NOT EXISTS kg_relationships (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_id INTEGER NOT NULL,
                    target_id INTEGER NOT NULL,
                    relation TEXT NOT NULL,
                    document_reference TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(source_id) REFERENCES kg_entities(id),
                    FOREIGN KEY(target_id) REFERENCES kg_entities(id),
                    UNIQUE(source_id, target_id, relation)
                )
            ''')
            conn.commit()

    def extract_from_text(self, text: str, source_reference: str = "unknown"):
        """Extracts and persists knowledge graph components."""
        prompt = f"""You are an expert financial Knowledge Graph extractor specializing in crypto markets.

TASK: Extract entities and relationships from the text below.

ENTITY TYPES (extract these):
- COIN/TOKEN: Bitcoin, Ethereum, SOL, etc. (use standard tickers: BTC, ETH, SOL)
- PERSON: CEO names, founders, analysts, politicians
- INSTITUTION: SEC, Federal Reserve, Binance, BlackRock, etc.
- EVENT: ETF approval, halving, rate decision, hack, partnership announcement

RELATIONSHIP TYPES (use standardized verbs):
- regulates, is_regulated_by
- impacts, is_impacted_by
- partners_with
- invests_in, is_invested_by
- competes_with
- correlates_with
- causes, is_caused_by

RULES:
1. Only extract entities EXPLICITLY mentioned in the text. Do NOT infer or hallucinate.
2. Normalize entity names: "Bitcoin" → "BTC", "Ethereum" → "ETH"
3. Keep relationships directional and specific: "SEC regulates Coinbase" not "SEC and Coinbase are related"
4. If no clear entities/relationships exist, return {{"entities": [], "relationships": []}}

Format output according to this schema:
{self.parser.get_format_instructions()}

Text:
{text}
"""
        
        try:
            response = self.router.invoke(
                [
                    SystemMessage(content="You are a financial Knowledge Graph extractor. Return ONLY valid JSON."),
                    HumanMessage(content=prompt)
                ],
                temperature=0,
                priority="low",       # Entity extraction is non-critical
                max_wall_time=30.0     # Don't burn 90s on entity extraction
            )

            # Normalize content: Gemini v1 may return list of content blocks
            content = response.content if hasattr(response, "content") else str(response)
            if isinstance(content, list):
                content = " ".join(
                    b.get("text", "") if isinstance(b, dict) else str(b)
                    for b in content
                )
            content = str(content).strip()

            # Try LangChain parser first, fallback to robust extraction
            try:
                extracted_data = self.parser.parse(content)
            except Exception as parse_err:
                logger.warning(f"LangChain parser failed: {parse_err}. Trying robust JSON extraction...")
                from json_utils import extract_json
                extracted_data = extract_json(content)
                if extracted_data is None:
                    logger.error(f"Robust extraction also failed. Raw: {content[:300]}")
                    return None
                # Ensure required structure
                if "entities" not in extracted_data:
                    extracted_data["entities"] = []
                if "relationships" not in extracted_data:
                    extracted_data["relationships"] = []
            self._save_to_db(extracted_data, source_reference)
            return extracted_data
        except Exception as e:
            logger.error(f"Failed to extract entities: {e}")
            return None

    def _save_to_db(self, data: dict, source_reference: str):
        with self._get_db_connection() as conn:
            c = conn.cursor()
            
            # Helper to get or create entity
            def get_or_create_entity(name: str) -> int:
                name = name.strip().lower()
                c.execute("SELECT id FROM kg_entities WHERE name = ?", (name,))
                row = c.fetchone()
                if row:
                    return row['id']
                c.execute("INSERT INTO kg_entities (name) VALUES (?)", (name,))
                return c.lastrowid
            
            # Insert relationships
            for rel in data.get('relationships', []):
                source_id = get_or_create_entity(rel['source'])
                target_id = get_or_create_entity(rel['target'])
                relation_text = rel['relation'].strip().lower()
                
                try:
                    c.execute("""
                        INSERT OR IGNORE INTO kg_relationships 
                        (source_id, target_id, relation, document_reference) 
                        VALUES (?, ?, ?, ?)
                    """, (source_id, target_id, relation_text, source_reference))
                except sqlite3.Error as e:
                    logger.warning(f"DB Error saving relationship: {e}")
            
            conn.commit()
            logger.info(f"Saved {len(data.get('relationships', []))} relationships to DB.")

    def query_entity_network(self, entity_name: str, depth: int = 1):
        """
        LazyGraph Traversal: Finds relationships connected to an entity.
        Returns a human-readable list of relations.
        """
        name = entity_name.strip().lower()
        with self._get_db_connection() as conn:
            c = conn.cursor()
            
            # Simplistic 1-hop query for LazyGraphRAG
            c.execute("""
                SELECT e1.name as source_name, r.relation, e2.name as target_name 
                FROM kg_relationships r
                JOIN kg_entities e1 ON r.source_id = e1.id
                JOIN kg_entities e2 ON r.target_id = e2.id
                WHERE e1.name = ? OR e2.name = ?
            """, (name, name))
            
            rows = c.fetchall()
            
            results = []
            for r in rows:
                results.append(f"{r['source_name']} -> {r['relation']} -> {r['target_name']}")
                
            return results

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    kg = KnowledgeGraphManager()
    
    test_text = "The SEC today announced charges against Coinbase for operating as an unregistered securities exchange. Following the news, Bitcoin dropped by 5% as market panic ensued."
    
    logger.info("Extracting graph from text...")
    res = kg.extract_from_text(test_text, source_reference="news_article_xyz")
    
    import json
    print(json.dumps(res, indent=2))
    
    logger.info("\nQuerying network for 'SEC'...")
    network = kg.query_entity_network("SEC")
    for link in network:
        print(link)
