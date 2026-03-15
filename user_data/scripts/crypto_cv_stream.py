import json
import sseclient
import requests
import sqlite3
import logging
from datetime import datetime
import sys
import os

sys.path.append(os.path.dirname(__file__))
from db import get_db_connection

logger = logging.getLogger(__name__)

URL = "https://cryptocurrency.cv/api/ai/sentiment/stream"

def stream_market_news():
    """Listens to the cryptocurrency.cv SSE stream and stores news/sentiment."""
    logger.info("Starting SSE connection to cryptocurrency.cv...")
    response = None
    try:
        headers = {
            'Accept': 'text/event-stream',
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(URL, stream=True, headers=headers)
        response.raise_for_status()
        client = sseclient.SSEClient(response)

        for event in client.events():
            if event.event == "message" and event.data:
                conn = None
                try:
                    data = json.loads(event.data)

                    source_name = f"cv - {data.get('source', 'Unknown')}"
                    title = data.get("title", "")
                    summary = data.get("summary", "")
                    link = data.get("url", "")
                    sentiment_score = float(data.get("sentiment_score", 0.0))

                    # Assuming an ISO 8601 string or similar timestamp. In case missing use utcnow
                    pub_date_str = data.get("published_at")
                    if pub_date_str:
                        try:
                            # Tries common formats
                            pub_date = datetime.fromisoformat(pub_date_str.replace("Z", "+00:00"))
                        except ValueError:
                            pub_date = datetime.utcnow()
                    else:
                        pub_date = datetime.utcnow()

                    conn = get_db_connection()
                    c = conn.cursor()

                    c.execute('''
                        INSERT INTO market_news (source, title, summary, url, published_at, sentiment_score, raw_data)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    ''', (source_name, title, summary, link, pub_date, sentiment_score, json.dumps(data)))

                    conn.commit()
                    logger.info(f"Received from stream: {title} (Sentiment: {sentiment_score})")

                except sqlite3.IntegrityError:
                    # Item already exists
                    pass
                except Exception as parse_e:
                    logger.error(f"Error parsing SSE event data: {parse_e}")
                finally:
                    if conn is not None:
                        conn.close()

    except Exception as e:
        logger.error(f"Error connecting to cryptocurrency.cv stream: {e}")
    finally:
        # Close the streaming response to release the socket fd
        if response is not None:
            try:
                response.close()
            except Exception:
                pass

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    stream_market_news()
