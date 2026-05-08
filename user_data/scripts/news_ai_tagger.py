"""Phase 30 A.13 — AI tag-extract -> classify -> hash-cache pipeline.

Pipeline:
1. Extract tags from headline (LLM cheap model, JSON output).
2. Classify against trader_interests.txt (trader-controlled).
3. Cache by content hash to avoid re-classification.

Returns relevance_score [0, 1] + matched_interests list.
"""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

INTERESTS_PATH = Path(__file__).parent.parent / "data" / "trader_interests.txt"


def load_interests() -> List[str]:
    if not INTERESTS_PATH.is_file():
        return []
    out = []
    for line in INTERESTS_PATH.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            out.append(line.lower())
    return out


def _content_hash(headline: str, body: str = "") -> str:
    h = hashlib.sha256()
    h.update((headline or "").encode("utf-8"))
    h.update(b"|")
    h.update((body or "").encode("utf-8"))
    return h.hexdigest()


def _cached_classification(content_hash: str) -> Optional[Dict]:
    try:
        from llm_response_cache import get_cached
        cache_key = f"news_tag_{content_hash}"
        result = get_cached(cache_key, ttl_seconds=24 * 3600)
        if result:
            import json
            return json.loads(result[0])
    except Exception:
        return None
    return None


def _put_cache(content_hash: str, result: Dict) -> None:
    try:
        from llm_response_cache import put_cached
        import json
        put_cached(f"news_tag_{content_hash}", "interest-classifier", json.dumps(result), 0, 0)
    except Exception:
        pass


def _heuristic_match(text: str, interests: List[str]) -> List[str]:
    text_l = (text or "").lower()
    matched: List[str] = []
    for interest in interests:
        words = [w for w in interest.split() if len(w) > 3]
        if not words:
            continue
        hits = sum(1 for w in words if w in text_l)
        if hits >= max(1, len(words) // 2):
            matched.append(interest)
    return matched


def classify_headline(
    headline: str,
    body: str = "",
    use_llm: bool = True,
) -> Dict:
    """Returns {relevance_score, matched_interests, source}."""
    interests = load_interests()
    content_hash = _content_hash(headline, body)
    cached = _cached_classification(content_hash)
    if cached:
        return {**cached, "source": "cache"}

    matched_heur = _heuristic_match(f"{headline} {body}", interests)
    relevance = min(1.0, len(matched_heur) / max(1, min(3, len(interests))))
    out: Dict = {
        "relevance_score": float(round(relevance, 3)),
        "matched_interests": matched_heur,
        "source": "heuristic",
    }

    if use_llm and not matched_heur:
        try:
            llm_match = _llm_classify(headline, body, interests)
            if llm_match:
                out["matched_interests"] = llm_match
                out["relevance_score"] = float(min(1.0, len(llm_match) / 3.0))
                out["source"] = "llm"
        except Exception as e:
            logger.debug(f"[NewsAITagger] LLM classify failed: {e}")

    _put_cache(content_hash, out)
    return out


def _llm_classify(headline: str, body: str, interests: List[str]) -> List[str]:
    try:
        from llm_router import LLMRouter
    except Exception:
        return []
    interests_block = "\n".join(f"- {i}" for i in interests[:30])
    prompt = (
        "Trader interests:\n"
        f"{interests_block}\n\n"
        "Match headline against above interests. Return JSON list of matched interest strings only.\n"
        f"Headline: {headline}\nBody: {body[:300]}\n\nJSON:"
    )
    try:
        router = LLMRouter()
        response_text = router.invoke_simple(prompt, max_tokens=128) if hasattr(router, "invoke_simple") else ""
    except Exception:
        return []
    try:
        from json_parse_robust import parse_json
        parsed, _ = parse_json(response_text or "[]", "news_ai_tagger")
        if isinstance(parsed, list):
            return [str(x).lower() for x in parsed if isinstance(x, str)]
    except Exception:
        pass
    return []
