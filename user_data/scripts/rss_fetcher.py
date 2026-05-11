import feedparser
import logging
import socket
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

# Revize Tur-2 (H10): setdefaulttimeout is a process-global, so the
# mega-sprint module-level call was silently forcing a 30s timeout on
# every unrelated SQLite / HTTP call the process made for the rest of
# its lifetime. Scoped to fetch_rss_feeds() via context manager below.

# TIER 1 and TIER 2 RSS feeds explicitly defined from ROADMAP.md
# Expanded to include Macro and Secondary Crypto sites for maximum Brain data
RSS_FEEDS = {
    # TIER 1 - Crypto Prime (en güvenilir, en hızlı)
    "CoinDesk": "https://www.coindesk.com/arc/outboundfeeds/rss/?outputType=xml",
    "CoinTelegraph_All": "https://cointelegraph.com/rss",
    "Decrypt": "https://decrypt.co/feed",
    "The_Block": "https://www.theblock.co/rss.xml",

    # TIER 2 - Crypto Secondary (iyi kapsama, farklı perspektifler)
    "CryptoSlate": "https://cryptoslate.com/feed/",
    "CryptoPotato": "https://cryptopotato.com/feed/",
    "CryptoNews": "https://cryptonews.com/news/feed/",
    "Bitcoin_Magazine": "https://bitcoinmagazine.com/feed",
    "DailyHodl": "https://dailyhodl.com/feed/",
    "UToday": "https://u.today/rss",
    "CoinJournal": "https://coinjournal.net/news/feed/",

    # TIER 2.5 - CryptoPanic Replacement (API key gerektirmeyen, ücretsiz RSS)
    # Bu feed'ler CryptoPanic'in 404 dönmesiyle eklendi (Nisan 2026)
    "NewsBTC": "https://www.newsbtc.com/feed/",
    "Bitcoinist": "https://bitcoinist.com/feed/",
    "BeInCrypto": "https://beincrypto.com/feed/",
    "Blockworks": "https://blockworks.co/feed",
    "CoinGecko_Blog": "https://blog.coingecko.com/rss/",
    "Messari": "https://messari.io/rss",
    "DeFiLlama_News": "https://feed.defillama.com/",

    # TIER 3 - Macro / Traditional Finance
    "CNBC_Finance": "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=10000664",
    "Fox_Business": "https://moxie.foxbusiness.com/google-publisher/markets.xml",
    "WSJ_Markets": "https://feeds.a.dj.com/rss/RSSMarketsMain.xml",
    "Investing_Com": "https://www.investing.com/rss/news_301.rss",

    # TIER 3.5 - DeFi / On-Chain Specific
    "Rekt_News": "https://rekt.news/feed.xml",
    "Week_In_Ethereum": "https://weekinethereumnews.com/feed/",
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
    return hashlib.sha256(" ".join(words).encode()).hexdigest()

def fetch_rss_feeds():
    """Fetches articles from RSS feeds and stores them if they don't exist."""
    prev_timeout = socket.getdefaulttimeout()
    socket.setdefaulttimeout(30)
    try:
        return _fetch_rss_feeds_inner()
    finally:
        socket.setdefaulttimeout(prev_timeout)


def _fetch_rss_feeds_inner():
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
    
    # ═══ PHASE 30 A.15 — News cluster (Jaccard 24h) singleton ═══
    try:
        import sys as _phase30_sys
        if "_phase30_news_cluster" not in globals():
            from news_cluster import NewsCluster as _phase30_NC
            globals()["_phase30_news_cluster"] = _phase30_NC()
    except Exception:
        pass
    _phase30_cluster = globals().get("_phase30_news_cluster")

    for source_name, url in RSS_FEEDS.items():
        logger.info(f"Fetching RSS feed from {source_name}...")
        feed = None
        try:
            # ═══ PHASE 30 A.21 — Browser UA rotation for fetch ═══
            try:
                import requests as _phase30_req
                from scripts.browser_ua import random_headers as _phase30_ua, is_access_denied as _phase30_blocked
                _phase30_resp = _phase30_req.get(url, headers=_phase30_ua(), timeout=30)
                _denied, _kind = _phase30_blocked(_phase30_resp.status_code, _phase30_resp.text[:5000])
                if _denied:
                    logger.warning(f"[Phase30:A.21] {source_name} access denied: {_kind}; skipping")
                    continue
                feed = feedparser.parse(_phase30_resp.text)
            except Exception:
                feed = feedparser.parse(url)

            for entry in feed.entries:
                title = entry.get('title', '')
                link = entry.get('link', '')
                summary = entry.get('summary', '') or entry.get('description', '')

                if not title or not link:
                    continue

                # ═══ PHASE 30 A.13 + A.15 + A.16 — News tag/cluster/threat ═══
                try:
                    if _phase30_cluster is not None and _phase30_cluster.is_duplicate(title):
                        continue
                except Exception:
                    pass

                pub_date = parse_date(entry)
                thash = title_hash(title)

                try:
                    c.execute('''
                        INSERT OR IGNORE INTO market_news (source, title, summary, url, published_at, title_hash)
                        VALUES (?, ?, ?, ?, ?, ?)
                    ''', (source_name, title, summary, link, pub_date, thash))
                    if c.rowcount > 0:
                        new_articles += 1
                        _phase30_news_id = c.lastrowid
                        try:
                            if _phase30_cluster is not None:
                                _phase30_cluster.assign(_phase30_news_id, title)
                        except Exception:
                            pass
                        try:
                            from news_ai_tagger import classify_headline as _phase30_tag
                            _phase30_tag(title, summary, use_llm=False)
                        except Exception:
                            pass
                        try:
                            from threat_classifier import classify as _phase30_threat
                            _phase30_threat(title, summary, news_id=_phase30_news_id)
                        except Exception:
                            pass
                except sqlite3.IntegrityError:
                    continue

        except Exception as e:
            logger.error(f"Error fetching {source_name}: {e}")
        finally:
            # Phase 23: feedparser XML DOM leak fix — release DOM immediately
            # feedparser holds entire XML tree with reference cycles that delay GC
            del feed

    conn.commit()
    conn.close()

    # Phase 23: force GC after all feeds processed — collect any remaining DOM fragments
    import gc
    gc.collect()

    logger.info(f"Finished RSS fetch. Inserted {new_articles} new articles.")
    return new_articles

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fetch_rss_feeds()
