"""Phase 30 A.15 — News clustering with Jaccard similarity (24h window).

Same news repeated across CryptoPanic + RSS + CryptoCompare flooded the
sentiment pipeline. This module dedupes via Jaccard token-set similarity
in a sliding 24h window. Persists cluster_key, jaccard_score to news_clusters.
"""
from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

DEFAULT_THRESHOLD = 0.6
WORD_RE = re.compile(r"[a-z0-9]{3,}")
STOPWORDS = {"the", "and", "for", "with", "from", "this", "that", "are", "was",
             "have", "has", "but", "not", "you", "all", "can", "will", "now",
             "one", "two", "via", "into", "off"}


def tokens(text: str) -> Set[str]:
    return {w for w in WORD_RE.findall((text or "").lower()) if w not in STOPWORDS}


def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def _content_hash(headline: str) -> str:
    return hashlib.sha1((headline or "").lower().encode()).hexdigest()[:16]


@dataclass
class ClusterMember:
    news_id: int
    headline: str
    tokens: Set[str]
    cluster_key: str
    score: float


class NewsCluster:
    def __init__(self, threshold: float = DEFAULT_THRESHOLD, window_hours: int = 24):
        self.threshold = float(threshold)
        self.window = timedelta(hours=int(window_hours))
        self._members: List[Tuple[datetime, ClusterMember]] = []

    def assign(self, news_id: int, headline: str, ts: Optional[datetime] = None) -> Dict:
        ts = ts or datetime.now(timezone.utc)
        cutoff = ts - self.window
        self._members = [(t, m) for (t, m) in self._members if t >= cutoff]

        toks = tokens(headline)
        best_key: Optional[str] = None
        best_score = 0.0
        for _, m in self._members:
            s = jaccard(toks, m.tokens)
            if s >= self.threshold and s > best_score:
                best_score = s
                best_key = m.cluster_key

        if best_key is None:
            best_key = _content_hash(headline)
            best_score = 1.0
        member = ClusterMember(news_id=news_id, headline=headline,
                               tokens=toks, cluster_key=best_key, score=best_score)
        self._members.append((ts, member))

        size = sum(1 for _, m in self._members if m.cluster_key == best_key)
        out = {"cluster_key": best_key, "jaccard_score": float(round(best_score, 3)),
               "cluster_size": size}
        try:
            from db import AI_DB_PATH, get_db_connection

            with get_db_connection(AI_DB_PATH) as conn:
                conn.execute(
                    """INSERT INTO news_clusters
                       (cluster_key, news_id, jaccard_score, cluster_size)
                       VALUES (?, ?, ?, ?)""",
                    (best_key, news_id, best_score, size),
                )
                conn.commit()
        except Exception:
            pass
        return out

    def is_duplicate(self, headline: str) -> bool:
        toks = tokens(headline)
        for _, m in self._members:
            if jaccard(toks, m.tokens) >= self.threshold:
                return True
        return False
