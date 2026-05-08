"""Phase 30 A.21 — Browser-UA rotation + HTML access-denied detect.

Used by rss_fetcher and any external HTTP scraper. Rotates among realistic
desktop UAs; detects 403/Cloudflare/HTML-block responses and logs them so
operator can swap source.
"""
from __future__ import annotations

import logging
import random
import re
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

USER_AGENTS: List[str] = [
    "Mozilla/5.0 (X11; Linux x86_64; rv:123.0) Gecko/20100101 Firefox/123.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_5) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.5 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:122.0) Gecko/20100101 Firefox/122.0",
]

ACCESS_DENIED_RE = re.compile(
    r"(access\s*denied|cloudflare|cf-ray|<title>403|<title>access denied|just a moment)",
    re.IGNORECASE,
)


def random_headers() -> Dict[str, str]:
    return {
        "User-Agent": random.choice(USER_AGENTS),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate",
        "DNT": "1",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
    }


def is_access_denied(status: int, body: str) -> Tuple[bool, str]:
    if status == 403:
        return True, "http_403"
    if status == 429:
        return True, "http_429"
    if status == 503:
        return True, "http_503"
    if body and ACCESS_DENIED_RE.search(body[:5000]):
        return True, "html_marker"
    return False, ""
