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

    # Portfolio State: Gerçek bakiyeyi SQLite'a persist et (scheduler/API okusun)
    c.execute('''
        CREATE TABLE IF NOT EXISTS portfolio_state (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            stake_currency TEXT DEFAULT 'USDT',
            total_balance REAL DEFAULT 0.0,
            free_balance REAL DEFAULT 0.0,
            in_trades REAL DEFAULT 0.0,
            assets_json TEXT DEFAULT '{}',
            updated_at TEXT
        )
    ''')

    # Hypothetical Portfolio: "$100 ile başlasaydın şuan ne olurdu?" tracking
    c.execute('''
        CREATE TABLE IF NOT EXISTS hypothetical_portfolio (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            trade_pair TEXT NOT NULL,
            trade_pnl_pct REAL NOT NULL,
            balance_before REAL NOT NULL,
            balance_after REAL NOT NULL,
            trade_closed_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Coin-level rolling sentiment (from coin_sentiment_aggregator.py)
    c.execute('''
        CREATE TABLE IF NOT EXISTS coin_sentiment_rolling (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            coin TEXT NOT NULL,
            sentiment_1h REAL DEFAULT 0,
            sentiment_4h REAL DEFAULT 0,
            sentiment_24h REAL DEFAULT 0,
            news_count_24h INTEGER DEFAULT 0
        )
    ''')

    # Ensure market_news has columns added after initial schema
    for col, typedef in [
        ("title_hash", "TEXT"),
        ("is_embedded", "BOOLEAN DEFAULT 0"),
    ]:
        try:
            c.execute(f"ALTER TABLE market_news ADD COLUMN {col} {typedef}")
        except sqlite3.OperationalError:
            pass  # column already exists

    # Phase 18: Signal Health Tracking (AI vs Fallback vs Voting ratio)
    c.execute('''
        CREATE TABLE IF NOT EXISTS signal_health (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            pair TEXT NOT NULL,
            signal_source TEXT NOT NULL,
            signal_type TEXT,
            confidence REAL,
            latency_ms REAL
        )
    ''')

    # Phase 19 Level 3: Market data tables (derivatives, DeFi, macro)
    c.execute('''
        CREATE TABLE IF NOT EXISTS derivatives_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pair TEXT NOT NULL,
            open_interest_usd REAL,
            funding_rate REAL,
            long_short_ratio REAL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS macro_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            metric_name TEXT NOT NULL,
            value REAL NOT NULL,
            prev_value REAL,
            change_pct REAL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS defi_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            metric_name TEXT NOT NULL,
            value REAL NOT NULL,
            change_pct REAL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # OHLCV Pattern Matcher (candle-sequence similarity search)
    c.execute('''
        CREATE TABLE IF NOT EXISTS ohlcv_patterns (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pair TEXT NOT NULL,
            timeframe TEXT DEFAULT '1h',
            timestamp TEXT,
            fingerprint TEXT NOT NULL,
            outcome_1h REAL,
            outcome_4h REAL,
            outcome_24h REAL,
            direction TEXT,
            indicators_json TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    c.execute('CREATE INDEX IF NOT EXISTS idx_ohlcv_pair ON ohlcv_patterns(pair)')

    # Phase 19 Level 3: Google Trends search interest
    c.execute('''
        CREATE TABLE IF NOT EXISTS search_trends (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            keyword TEXT NOT NULL,
            interest_score INTEGER,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Phase 20: Agent Pool — agent decisions memory (MiroFish-inspired)
    c.execute('''
        CREATE TABLE IF NOT EXISTS agent_memory (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            agent_type TEXT NOT NULL,
            pair TEXT NOT NULL,
            regime TEXT,
            signal TEXT NOT NULL,
            strength REAL,
            key_argument TEXT,
            evidence_engine_confidence REAL,
            final_outcome_pnl REAL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Phase 20: Agent Pool — historical performance tracking
    c.execute('''
        CREATE TABLE IF NOT EXISTS agent_performance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            agent_type TEXT NOT NULL,
            pair TEXT NOT NULL,
            regime TEXT,
            signal TEXT NOT NULL,
            outcome_pnl REAL,
            was_correct BOOLEAN,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Phase 20: Opportunity Scanner — cached scan results
    c.execute('''
        CREATE TABLE IF NOT EXISTS opportunity_scores (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pair TEXT NOT NULL,
            composite_score REAL NOT NULL,
            top_type TEXT,
            momentum_score REAL,
            reversion_score REAL,
            funding_score REAL,
            regime_shift_score REAL,
            volume_anomaly_score REAL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Phase 20: Evidence Engine — structured audit log
    c.execute('''
        CREATE TABLE IF NOT EXISTS evidence_audit_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pair TEXT NOT NULL,
            signal TEXT NOT NULL,
            confidence REAL NOT NULL,
            sub_scores_json TEXT,
            contradictions_json TEXT,
            evidence_sources_json TEXT,
            regime TEXT,
            max_confidence_cap REAL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Phase 20: Cross-Pair Intelligence cache (single-row table)
    c.execute('''
        CREATE TABLE IF NOT EXISTS cross_pair_cache (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            data_json TEXT,
            timestamp TEXT
        )
    ''')

    # Create indices
    c.execute('CREATE INDEX IF NOT EXISTS idx_market_news_published ON market_news(published_at)')
    c.execute('CREATE INDEX IF NOT EXISTS idx_ai_decisions_pair ON ai_decisions(pair)')
    c.execute('CREATE INDEX IF NOT EXISTS idx_sentiment_rolling_coin ON coin_sentiment_rolling(coin, timestamp)')
    c.execute('CREATE INDEX IF NOT EXISTS idx_signal_health_ts ON signal_health(timestamp)')
    c.execute('CREATE INDEX IF NOT EXISTS idx_deriv_pair_ts ON derivatives_data(pair, timestamp)')
    c.execute('CREATE INDEX IF NOT EXISTS idx_macro_name_ts ON macro_data(metric_name, timestamp)')
    c.execute('CREATE INDEX IF NOT EXISTS idx_defi_name_ts ON defi_data(metric_name, timestamp)')
    c.execute('CREATE INDEX IF NOT EXISTS idx_trends_kw_ts ON search_trends(keyword, timestamp)')
    # Phase 20 indices
    c.execute('CREATE INDEX IF NOT EXISTS idx_agent_mem_type ON agent_memory(agent_type, regime)')
    c.execute('CREATE INDEX IF NOT EXISTS idx_agent_perf ON agent_performance(agent_type, regime)')
    c.execute('CREATE INDEX IF NOT EXISTS idx_opp_pair_ts ON opportunity_scores(pair, timestamp)')
    c.execute('CREATE INDEX IF NOT EXISTS idx_evidence_pair_ts ON evidence_audit_log(pair, timestamp)')

    conn.commit()
    conn.close()
    logger.info(f"Database initialized at {DB_PATH}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    init_db()
