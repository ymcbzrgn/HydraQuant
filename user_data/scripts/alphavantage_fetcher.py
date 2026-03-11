"""
Görev 4: Alpha Vantage News Fetcher
Pre-computed news sentiment baseline fetching.
Max 25/day free limit (≈1 request/hour).
"""

import os
import requests
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class AlphaVantageFetcher:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.environ.get("ALPHAVANTAGE_API_KEY")
        self.base_url = "https://www.alphavantage.co/query"

    def fetch_news_sentiment(self, tickers: List[str] = ["CRYPTO:BTC", "CRYPTO:ETH"]) -> List[Dict[str, Any]]:
        """
        Alpha Vantage NEWS_SENTIMENT endpointini çağırarak önceden hesaplanmış
        overall_sentiment_score değerlerini al.
        """
        if not self.api_key:
            logger.warning("ALPHAVANTAGE_API_KEY eksik. Veri çekilmeyecek.")
            return []

        results = []
        try:
            # We join multiple tickers if the API supports it, else we fetch for BTC typically.
            # AlphaVantage accepts comma separated tickers up to a limit.
            params = {
                "function": "NEWS_SENTIMENT",
                "tickers": ",".join(tickers),
                "apikey": self.api_key,
                "limit": 25,
                "sort": "LATEST"
            }
            
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            # Note: rate limit response keys usually contain 'Information' instead of 'feed' 
            # if the max daily cap is exceeded. We should handle it gracefully.
            if "Information" in data and "rate limit" in data["Information"].lower():
                logger.warning("AlphaVantage daily rate limit exceeded.")
                return []
                
            feed = data.get("feed", [])
            for article in feed:
                sentiment_score = article.get("overall_sentiment_score", 0.0)
                
                results.append({
                    "title": article.get("title", ""),
                    "url": article.get("url", ""),
                    "source": article.get("source", ""),
                    "published_at": article.get("time_published", ""),
                    "av_sentiment_score": float(sentiment_score) if sentiment_score else 0.0,
                    "sentiment_label": article.get("overall_sentiment_label", "Neutral")
                })
                
            logger.info(f"AlphaVantage: {len(results)} news items processed.")
            return results
            
        except requests.exceptions.HTTPError as he:
            logger.error(f"AlphaVantage HTTP Error during fetch: {he}")
            return []
        except Exception as e:
            logger.error(f"AlphaVantage unexpected extraction error: {e}")
            return []
