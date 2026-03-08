import requests
import sqlite3
import logging
from datetime import datetime
import sys
import os
from dotenv import load_dotenv

sys.path.append(os.path.dirname(__file__))
from db import get_db_connection

logger = logging.getLogger(__name__)

# Load API keys from .env
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))
CRYPTOCOMPARE_API_KEY = os.getenv("CRYPTOCOMPARE_API_KEY")

URL = "https://min-api.cryptocompare.com/data/v2/news/?lang=EN"

def fetch_cryptocompare_news():
    """Fetches news from CryptoCompare and stores them."""
    headers = {}
    if CRYPTOCOMPARE_API_KEY:
        headers["authorization"] = f"Apikey {CRYPTOCOMPARE_API_KEY}"
    else:
        logger.warning("CRYPTOCOMPARE_API_KEY is not set. Using rate-limited anonymous access.")
        
    try:
        response = requests.get(URL, headers=headers)
        response.raise_for_status()
        data = response.json()
        
        if data and data.get("Data"):
            conn = get_db_connection()
            c = conn.cursor()
            new_articles = 0
            
            for item in data["Data"]:
                source_name = f"CryptoCompare - {item.get('source', 'Unknown')}"
                title = item.get("title", "")
                summary = item.get("body", "")
                link = item.get("url", "")
                
                # CryptoCompare provides unix timestamps
                pub_date = datetime.fromtimestamp(int(item.get("published_on")))
                
                try:
                    c.execute('''
                        INSERT INTO market_news (source, title, summary, url, published_at)
                        VALUES (?, ?, ?, ?, ?)
                    ''', (source_name, title, summary, link, pub_date))
                    new_articles += 1
                except sqlite3.IntegrityError:
                    # Item already exists
                    continue
                    
            conn.commit()
            conn.close()
            logger.info(f"Finished CryptoCompare fetch. Inserted {new_articles} new articles.")
            
    except Exception as e:
        logger.error(f"Error fetching CryptoCompare News: {e}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fetch_cryptocompare_news()
