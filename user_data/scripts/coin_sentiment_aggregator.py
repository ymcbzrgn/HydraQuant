import sqlite3
import logging
import os
import sys
import pandas as pd

sys.path.append(os.path.dirname(__file__))
from db import get_db_connection

logger = logging.getLogger(__name__)

# Basic mapping of crypto keywords to tickers (MVP level)
COIN_MAPPINGS = {
    "BTC": ["bitcoin", "btc", "satoshi"],
    "ETH": ["ethereum", "eth", "vitalik"],
    "SOL": ["solana", "sol"],
    "XRP": ["ripple", "xrp"],
    "ADA": ["cardano", "ada"],
    "DOGE": ["dogecoin", "doge"],
    "BNB": ["binance coin", "bnb"],
}

# Phase 7: Tier Weighting - Give higher weight to established crypto news sites
TIER_WEIGHTS = {
    "coindesk.com": 1.0, "cointelegraph.com": 1.0, "decrypt.co": 1.0, "theblock.co": 1.0,
    "cryptoslate.com": 0.8, "cryptopotato.com": 0.8, "cryptonews.com": 0.8,
    "chaingpt.org": 0.6, "finance.yahoo.com": 0.6
}

def _weighted_mean(df_subset: pd.DataFrame) -> float:
    """Calculates weighted sentiment mean based on news source tier."""
    if df_subset.empty:
        return 0.0
    weights = df_subset['source'].map(lambda s: TIER_WEIGHTS.get(str(s).lower(), 0.5))
    if weights.sum() == 0:
        return 0.0
    return float((df_subset['sentiment_score'] * weights).sum() / weights.sum())

def init_sentiment_table():
    conn = get_db_connection()
    c = conn.cursor()
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
    conn.commit()
    conn.close()

def compute_rolling_sentiment():
    """Calculates taking the mean sentiment per coin over specific time windows."""
    init_sentiment_table()
    conn = get_db_connection()
    c = conn.cursor()
    
    # Needs to extract coin mentions. In SQLite, we can pull the recent 24h ones into pandas
    # Added "source" to query for Tier Weighting
    query = """
    SELECT id, title, summary, source, published_at, sentiment_score
    FROM market_news 
    WHERE sentiment_score IS NOT NULL 
    AND published_at >= datetime('now', '-24 hours')
    """
    df = pd.read_sql_query(query, conn, parse_dates=['published_at'])
    
    if df.empty:
        logger.info("No scored news found in the last 24h to aggregate.")
        conn.close()
        return
        
    # Ensure timezone-naive comparison for simplicity
    df['published_at'] = df['published_at'].dt.tz_localize(None)
    now = pd.Timestamp.utcnow().tz_localize(None)
    
    inserts = []
    
    for ticker, keywords in COIN_MAPPINGS.items():
        # Filter articles that mention the coin (case insensitive)
        pattern = '|'.join([fr"\b{kw}\b" for kw in keywords])
        # combine title and summary for search
        df['text'] = df['title'].fillna('') + ' ' + df['summary'].fillna('')
        coin_df = df[df['text'].str.contains(pattern, case=False, na=False, regex=True)]
        
        if coin_df.empty:
            continue
            
        # Time windows
        mask_1h = coin_df['published_at'] >= (now - pd.Timedelta(hours=1))
        mask_4h = coin_df['published_at'] >= (now - pd.Timedelta(hours=4))
        
        sent_1h = _weighted_mean(coin_df[mask_1h])
        sent_4h = _weighted_mean(coin_df[mask_4h])
        sent_24h = _weighted_mean(coin_df)
        count_24h = int(len(coin_df))
        
        inserts.append((ticker, sent_1h, sent_4h, sent_24h, count_24h))
        logger.debug(f"{ticker} rolling sentiment: 1h({sent_1h:.2f}), 4h({sent_4h:.2f}), 24h({sent_24h:.2f})")
        
    if inserts:
        c.executemany('''
            INSERT INTO coin_sentiment_rolling (coin, sentiment_1h, sentiment_4h, sentiment_24h, news_count_24h)
            VALUES (?, ?, ?, ?, ?)
        ''', inserts)
        conn.commit()
        logger.info(f"Updated rolling sentiment for {len(inserts)} coins.")
        
    conn.close()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    compute_rolling_sentiment()
