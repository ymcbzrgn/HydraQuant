import sqlite3
import os
import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

from ai_config import AI_DB_PATH as DB_PATH

def get_db_connection():
    """Returns a connection to the AI SQLite database with safe concurrency settings."""
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")  # Allow concurrent reads + writes
    return conn

def init_db():
    """Initializes the database schema for the AI data pipeline."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("PRAGMA journal_mode=WAL")  # WAL mode for concurrent access
    
    # Haber ve duygu analizi verilerinin tutulacagi tablo
    c.execute('''
        CREATE TABLE IF NOT EXISTS market_news (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source TEXT NOT NULL,
            title TEXT NOT NULL,
            summary TEXT,
            url TEXT UNIQUE,
            published_at DATETIME,
            sentiment_score REAL,
            raw_data TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Fear & Greed Endeksi tablosu
    c.execute('''
        CREATE TABLE IF NOT EXISTS fear_and_greed (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            value INTEGER NOT NULL,
            classification TEXT,
            timestamp DATETIME UNIQUE,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # AI Karar loglari tablosu
    c.execute('''
        CREATE TABLE IF NOT EXISTS ai_decisions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            pair TEXT NOT NULL,
            signal_type TEXT,
            confidence REAL,
            position_size REAL,
            entry_price REAL,
            model_used TEXT,
            rag_context_ids TEXT,
            reasoning_summary TEXT,
            regime TEXT,
            trust_score_at_decision REAL,
            outcome_pnl REAL,
            outcome_duration INTEGER
        )
    ''')

    # Forgone Profit (paper trade engine) loglari The system will use this to track rejected signals
    c.execute('''
        CREATE TABLE IF NOT EXISTS forgone_profit (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pair TEXT NOT NULL,
            signal_type TEXT,
            signal_time DATETIME DEFAULT CURRENT_TIMESTAMP,
            confidence REAL,
            entry_price REAL,
            was_executed BOOLEAN DEFAULT 0,
            exit_price REAL,
            forgone_pnl REAL,
            resolved_at DATETIME
        )
    ''')

    # Embedding Cache tablosu (API maliyetlerini ve suresini azaltmak icin)
    c.execute('''
        CREATE TABLE IF NOT EXISTS embedding_cache (
            text_hash TEXT PRIMARY KEY,
            text_content TEXT NOT NULL,
            gemini_embedding BLOB,
            bge_embedding BLOB,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # FTS5 Virtual Table for Persistent BM25 Search
    c.execute('''
        CREATE VIRTUAL TABLE IF NOT EXISTS bm25_index USING fts5(
            doc_id UNINDEXED,
            content,
            tokenize = 'porter unicode61'
        )
    ''')

    # Create indices
    c.execute('CREATE INDEX IF NOT EXISTS idx_market_news_published ON market_news(published_at)')
    c.execute('CREATE INDEX IF NOT EXISTS idx_ai_decisions_pair ON ai_decisions(pair)')
    
    conn.commit()
    conn.close()
    logger.info(f"Database initialized at {DB_PATH}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    init_db()
