import requests
import sqlite3
import logging
from datetime import datetime
import sys
import os

sys.path.append(os.path.dirname(__file__))
from db import get_db_connection

logger = logging.getLogger(__name__)

URL = "https://api.alternative.me/fng/?limit=1"

def fetch_fng():
    """Fetches the Fear and Greed index and stores it in the DB."""
    try:
        response = requests.get(URL)
        response.raise_for_status()
        data = response.json()
        
        if data and "data" in data and len(data["data"]) > 0:
            latest = data["data"][0]
            value = int(latest["value"])
            classification = latest["value_classification"]
            
            # API date is returned as string timestamp
            timestamp = datetime.fromtimestamp(int(latest["timestamp"]))
            
            conn = get_db_connection()
            c = conn.cursor()
            
            try:
                c.execute('''
                    INSERT INTO fear_and_greed (value, classification, timestamp)
                    VALUES (?, ?, ?)
                ''', (value, classification, timestamp))
                conn.commit()
                logger.info(f"Inserted F&G index: {value} ({classification}) for {timestamp}")
            except sqlite3.IntegrityError:
                logger.info(f"F&G index for {timestamp} already exists.")
            finally:
                conn.close()
                
    except Exception as e:
        logger.error(f"Error fetching Fear and Greed Index: {e}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fetch_fng()
