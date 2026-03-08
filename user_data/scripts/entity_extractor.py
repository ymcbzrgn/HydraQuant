import os
import sqlite3
import logging
import json
from typing import List, Dict, Any
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq

load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))
logger = logging.getLogger(__name__)

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "db", "ai_data.sqlite")

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
    """
    def __init__(self):
        # We use a fast, structured output LLM (Gemini) with Groq (Llama-3) as a fallback
        # in case we hit Gemini API rate limits during bulk processing.
        gemini_key = os.environ.get("GEMINI_API_KEY")
        groq_key = os.environ.get("GROQ_API_KEY")
        
        self.llm = ChatGoogleGenerativeAI(
            model="models/gemini-2.5-flash", 
            api_key=gemini_key, 
            temperature=0
        ) if gemini_key else None
        
        self.fallback_llm = ChatGroq(
            model="llama-3.1-8b-instant",
            api_key=groq_key,
            temperature=0
        ) if groq_key else None
        
        if not self.llm and not self.fallback_llm:
            logger.warning("No API keys found for Entity Extraction. Models will fail.")
        self.parser = JsonOutputParser(pydantic_object=EntityExtractionResult)
        self._init_db_tables()

    def _get_db_connection(self):
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
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
        prompt = f"""You are an expert financial Knowledge Graph extractor.
Given the following text, extract the key entities (coins, people, institutions, events) 
and the relationships between them.

Format your output strictly according to the following JSON schema:
{self.parser.get_format_instructions()}

Text:
{text}
"""
        
        try:
            if self.llm:
                try:
                    response = self.llm.invoke([HumanMessage(content=prompt)])
                except Exception as e:
                    logger.warning(f"Gemini API limit or failure: {e}. Attempting Groq Fallback...")
                    if self.fallback_llm:
                        response = self.fallback_llm.invoke([HumanMessage(content=prompt)])
                    else:
                        raise e
            elif self.fallback_llm:
                response = self.fallback_llm.invoke([HumanMessage(content=prompt)])
            else:
                return None
                
            extracted_data = self.parser.parse(response.content)
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
