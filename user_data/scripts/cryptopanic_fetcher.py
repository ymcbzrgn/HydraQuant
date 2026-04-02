"""
Görev 3: CryptoPanic Fetcher
Community sentiment votes (-1 to +1) from CryptoPanic.
Max 100/day free limit (≈4 requests/hour).
"""

import os
import requests
import logging
from typing import List, Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)

class CryptoPanicFetcher:
    # Class-level flags survive garbage collection of instances
    _disabled = False
    _404_logged = False

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.environ.get("CRYPTOPANIC_API_KEY")
        self.base_url = "https://cryptopanic.com/api/v1/posts/"

    def fetch(self, currencies: List[str] = ["BTC", "ETH"], limit: int = 20) -> List[Dict[str, Any]]:
        """
        CryptoPanic API'den haberleri ve oyları (votes) çek.
        """
        if self._disabled:
            return []
        if not self.api_key:
            logger.warning("[CryptoPanic] API key missing. Set CRYPTOPANIC_API_KEY in .env")
            return []

        results = []
        try:
            params = {
                "auth_token": self.api_key,
                "currencies": ",".join(currencies),
                "public": "true"
            }

            response = requests.get(self.base_url, params=params, timeout=10)

            # Handle specific HTTP errors gracefully
            if response.status_code == 404:
                if not self._404_logged:
                    logger.warning("[CryptoPanic] API returned 404 — token may be expired or endpoint changed. "
                                   "Disabling until next restart. Check CRYPTOPANIC_API_KEY in .env")
                    self._404_logged = True
                self._disabled = True
                return []
            if response.status_code == 429:
                logger.info("[CryptoPanic] Rate limited (429). Skipping this cycle.")
                return []

            response.raise_for_status()

            data = response.json()
            posts = data.get("results", [])[:limit]

            for post in posts:
                votes = post.get("votes", {})
                sentiment_score = self.calculate_crowd_sentiment(votes)

                results.append({
                    "title": post.get("title", ""),
                    "url": post.get("url", ""),
                    "source": post.get("source", {}).get("title", ""),
                    "published_at": post.get("published_at", ""),
                    "sentiment_score": sentiment_score,
                    "votes": votes
                })

            logger.info(f"[CryptoPanic] {len(results)} posts fetched for {currencies}")
            return results

        except requests.exceptions.HTTPError as e:
            # Mask auth token in error log
            err_msg = str(e).replace(self.api_key, "***MASKED***") if self.api_key else str(e)
            logger.error(f"[CryptoPanic] HTTP error: {err_msg}")
            return []
        except Exception as e:
            logger.error(f"[CryptoPanic] Fetch failed: {type(e).__name__}: {e}")
            return []

    def calculate_crowd_sentiment(self, votes: dict) -> float:
        """
        Community vote'lardan sentiment skoru hesapla (-1.0 to +1.0).
        (positive + liked - negative) / total
        """
        pos = votes.get("positive", 0)
        like = votes.get("liked", 0)
        imp = votes.get("important", 0)
        neg = votes.get("negative", 0)
        dis = votes.get("disliked", 0)

        total = pos + like + imp + neg + dis
        if total == 0:
            return 0.0
            
        # Slightly weight 'important' positively if no clear direction
        weighted_score = (pos + like + (imp * 0.2) - neg - dis) / total
        
        # Clamp bounds
        return max(-1.0, min(1.0, weighted_score))
