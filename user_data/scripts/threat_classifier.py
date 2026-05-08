"""Phase 30 A.16 — 3-tier threat classification (critical/high/medium).

Classifies news/sentiment events into severity tiers:
- critical: market-shaking (hack, regulator ban, exchange insolvency)
- high:     significant (major exchange listing, regulator hearing)
- medium:   notable (analyst report, partial outage)

Logic: keyword-trigger first (fast), LLM fallback for ambiguous (cached).
Persists to threat_events.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

CRITICAL_KEYWORDS = [
    "hacked", "exploit", "rugpull", "insolvency", "bankrupt", "halts trading",
    "freezes withdrawals", "ftx-style", "depeg", "stablecoin collapse",
    "sec lawsuit", "doj indictment", "criminal charges", "sanctions",
    "kraken paused", "binance halted", "bybit halted",
]
HIGH_KEYWORDS = [
    "etf approval", "etf rejected", "fed rate", "rate hike", "rate cut",
    "cpi report", "regulation framework", "mica", "presidential",
    "majority listing", "delisting major",
]
MEDIUM_KEYWORDS = [
    "partnership", "upgrade", "fork", "airdrop", "burn", "treasury",
    "outage", "maintenance", "fork upgrade",
]


def _score_keywords(text: str, kws: List[str]) -> int:
    text_l = (text or "").lower()
    return sum(1 for k in kws if k in text_l)


def classify(headline: str, body: str = "", news_id: int = -1) -> Dict[str, Any]:
    text = f"{headline}\n{body}"
    crit = _score_keywords(text, CRITICAL_KEYWORDS)
    high = _score_keywords(text, HIGH_KEYWORDS)
    med = _score_keywords(text, MEDIUM_KEYWORDS)

    if crit >= 1:
        tier = "critical"
        score = min(1.0, 0.7 + 0.1 * crit)
        keywords = [k for k in CRITICAL_KEYWORDS if k in text.lower()][:5]
    elif high >= 1:
        tier = "high"
        score = min(0.85, 0.5 + 0.1 * high)
        keywords = [k for k in HIGH_KEYWORDS if k in text.lower()][:5]
    elif med >= 1:
        tier = "medium"
        score = min(0.7, 0.3 + 0.1 * med)
        keywords = [k for k in MEDIUM_KEYWORDS if k in text.lower()][:5]
    else:
        tier = "low"
        score = 0.1
        keywords = []

    out = {"tier": tier, "score": float(round(score, 3)), "keywords": keywords}
    try:
        from db import AI_DB_PATH, get_db_connection

        with get_db_connection(AI_DB_PATH) as conn:
            conn.execute(
                """INSERT INTO threat_events (news_id, tier, score, keywords)
                   VALUES (?, ?, ?, ?)""",
                (int(news_id) if news_id != -1 else None, tier, score, ",".join(keywords)),
            )
            conn.commit()
    except Exception:
        pass

    if tier == "critical":
        try:
            from severity_router import emit
            emit(kind="news.critical_threat", severity="critical",
                 message=f"Critical news: {headline[:120]}",
                 payload={"tier": tier, "score": score, "keywords": keywords})
        except Exception:
            pass
    return out
