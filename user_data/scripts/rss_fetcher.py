import feedparser
import logging
import sqlite3
import hashlib
import re
from datetime import datetime
from time import mktime
import sys
import os

# Ensure the DB module is in path
sys.path.append(os.path.dirname(__file__))
from db import get_db_connection

logger = logging.getLogger(__name__)

# TIER 1 and TIER 2 RSS feeds explicitly defined from ROADMAP.md
# Expanded to include Macro and Secondary Crypto sites for maximum Brain data
RSS_FEEDS = {
    # TIER 1 - Crypto Prime
    "CoinDesk": "https://www.coindesk.com/arc/outboundfeeds/rss/?outputType=xml",
    "CoinTelegraph_All": "https://cointelegraph.com/rss",
    "Decrypt": "https://decrypt.co/feed",
    "The_Block": "https://www.theblock.co/rss.xml",
    
    # TIER 2 - Crypto Secondary
    "CryptoSlate": "https://cryptoslate.com/feed/",
    "CryptoPotato": "https://cryptopotato.com/feed/",
    "CryptoNews": "https://cryptonews.com/news/feed/",
    "Bitcoin_Magazine": "https://bitcoinmagazine.com/feed",
    "DailyHodl": "https://dailyhodl.com/feed/",
    "UToday": "https://u.today/rss",
    "CoinJournal": "https://coinjournal.net/news/feed/",
    
    # TIER 3 - Macro / Traditional Finance
    "CNBC_Finance": "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=10000664",
    "Fox_Business": "https://moxie.foxbusiness.com/google-publisher/markets.xml",
    "WSJ_Markets": "https://feeds.a.dj.com/rss/RSSMarketsMain.xml",
    "Investing_Com": "https://www.investing.com/rss/news_301.rss"
}

def parse_date(entry):
    """Attempts to parse the publication date from an RSS entry."""
    if hasattr(entry, 'published_parsed') and entry.published_parsed:
        return datetime.fromtimestamp(mktime(entry.published_parsed))
    return datetime.utcnow()

# Phase 7: Title Hash Deduplication
def title_hash(title: str) -> str:
    """Normalize et ve hash'le — benzer başlıkları yakala"""
    title = re.sub(r'[^\w\s]', '', title)
    normalized = title.lower().strip()
    # Kısa kelimeleri kaldır (a, the, is...)
    words = [w for w in normalized.split() if len(w) > 3]
    return hashlib.md5(" ".join(words).encode()).hexdigest()

def fetch_rss_feeds():
    """Fetches articles from RSS feeds and stores them if they don't exist."""
    conn = get_db_connection()
    c = conn.cursor()
    new_articles = 0
    
    # Phase 7: Schema upgrade for title deduplication
    try:
        c.execute("ALTER TABLE market_news ADD COLUMN title_hash TEXT")
    except sqlite3.OperationalError:
        pass  # Column likely already exists
        
    try:
        c.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_market_news_title_hash ON market_news(title_hash)")
    except sqlite3.OperationalError:
        pass
    
    for source_name, url in RSS_FEEDS.items():
        logger.info(f"Fetching RSS feed from {source_name}...")
        try:
            feed = feedparser.parse(url)
            
            for entry in feed.entries:
                # Basic dedup based on URL which is marked UNIQUE in DB
                title = entry.get('title', '')
                link = entry.get('link', '')
                summary = entry.get('summary', '') or entry.get('description', '')
                
                # Omit empty titles
                if not title or not link:
                    continue
                    
                pub_date = parse_date(entry)
                thash = title_hash(title)
                
                try:
                    c.execute('''
                        INSERT OR IGNORE INTO market_news (source, title, summary, url, published_at, title_hash)
                        VALUES (?, ?, ?, ?, ?, ?)
                    ''', (source_name, title, summary, link, pub_date, thash))
                    if c.rowcount > 0:
                        new_articles += 1
                except sqlite3.IntegrityError:
                    # Item already exists in the DB based on UNIQUE URL or title_hash
                    continue
                    
        except Exception as e:
            logger.error(f"Error fetching {source_name}: {e}")
            
    conn.commit()
    conn.close()
    logger.info(f"Finished RSS fetch. Inserted {new_articles} new articles.")
    return new_articles

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fetch_rss_feeds()
