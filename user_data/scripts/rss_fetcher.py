import feedparser
import logging
import sqlite3
from datetime import datetime
from time import mktime
import sys
import os

# Ensure the DB module is in path
sys.path.append(os.path.dirname(__file__))
from db import get_db_connection

logger = logging.getLogger(__name__)

# TIER 1 and TIER 2 RSS feeds explicitly defined from ROADMAP.md
RSS_FEEDS = {
    # TIER 1
    "CoinDesk": "https://www.coindesk.com/arc/outboundfeeds/rss/?outputType=xml",
    "CoinTelegraph_All": "https://cointelegraph.com/rss",
    "Decrypt": "https://decrypt.co/feed",
    "The_Block": "https://www.theblock.co/rss.xml",
    # TIER 2
    "CryptoSlate": "https://cryptoslate.com/feed/",
    "CryptoPotato": "https://cryptopotato.com/feed/",
    "CryptoNews": "https://cryptonews.com/news/feed/",
    "Bitcoin_Magazine": "https://bitcoinmagazine.com/feed"
}

def parse_date(entry):
    """Attempts to parse the publication date from an RSS entry."""
    if hasattr(entry, 'published_parsed') and entry.published_parsed:
        return datetime.fromtimestamp(mktime(entry.published_parsed))
    return datetime.utcnow()

def fetch_rss_feeds():
    """Fetches articles from RSS feeds and stores them if they don't exist."""
    conn = get_db_connection()
    c = conn.cursor()
    new_articles = 0
    
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
                
                try:
                    c.execute('''
                        INSERT INTO market_news (source, title, summary, url, published_at)
                        VALUES (?, ?, ?, ?, ?)
                    ''', (source_name, title, summary, link, pub_date))
                    new_articles += 1
                except sqlite3.IntegrityError:
                    # Item already exists in the DB based on UNIQUE URL
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
