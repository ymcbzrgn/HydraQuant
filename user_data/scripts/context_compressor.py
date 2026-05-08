"""Phase 30 B.10 — 5-step context compression.

Pipeline reduces a long debate / RAG context to under target tokens:
1. Strip <think> blocks (A.9).
2. Drop near-duplicate paragraphs (Jaccard > 0.85).
3. Keep top-K by inverse-rank importance (markers like "BLOCKED", "CRITICAL").
4. LLM auxiliary summarize remaining to bullet form (cheap model).
5. Final length cap.

Used when a chain of thought exceeds prompt_caching breakpoints.
"""
from __future__ import annotations

import logging
import re
from typing import List, Optional

logger = logging.getLogger(__name__)

DEFAULT_TARGET_CHARS = 6000
PRIORITY_MARKERS = ("BLOCKED", "CRITICAL", "DENY", "ALERT", "ERROR", "VIOLATION", "LIQUIDATION")


def _step1_strip_think(text: str) -> str:
    try:
        from think_scrubber import scrub
        return scrub(text)
    except Exception:
        return text


def _step2_dedupe(paragraphs: List[str], threshold: float = 0.85) -> List[str]:
    out: List[str] = []
    seen_tokens: List[set] = []
    word_re = re.compile(r"[a-z0-9]{3,}")
    for para in paragraphs:
        toks = set(word_re.findall(para.lower()))
        is_dup = False
        for prev in seen_tokens:
            if not toks or not prev:
                continue
            inter = len(toks & prev)
            union = len(toks | prev)
            if union and inter / union >= threshold:
                is_dup = True
                break
        if not is_dup:
            out.append(para)
            seen_tokens.append(toks)
    return out


def _step3_priority_filter(paragraphs: List[str], top_k: int) -> List[str]:
    scored = []
    for p in paragraphs:
        score = sum(1 for m in PRIORITY_MARKERS if m in p) + (len(p) / 1000.0)
        scored.append((score, p))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [p for _, p in scored[:top_k]]


def _step4_llm_summarize(paragraphs: List[str], model: Optional[str] = None) -> List[str]:
    text = "\n\n".join(paragraphs)
    if len(text) < 4000:
        return paragraphs
    try:
        from llm_router import LLMRouter

        router = LLMRouter()
        prompt = (
            "Summarize the following debate/context to bullet points. Preserve all "
            "BLOCKED/CRITICAL/DENY markers verbatim. Output as plain bullets:\n\n" + text
        )
        out = router.invoke_simple(prompt, max_tokens=512) if hasattr(router, "invoke_simple") else None
        if out:
            return [out]
    except Exception:
        pass
    return paragraphs


def _step5_cap(text: str, target_chars: int) -> str:
    if len(text) <= target_chars:
        return text
    return text[: target_chars - 64] + "\n[... compressed_truncated ...]"


def compress(
    text: str,
    target_chars: int = DEFAULT_TARGET_CHARS,
    use_llm: bool = False,
    top_k: int = 30,
) -> str:
    if not text:
        return ""
    text = _step1_strip_think(text)
    paragraphs = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not paragraphs:
        return _step5_cap(text, target_chars)
    paragraphs = _step2_dedupe(paragraphs)
    if sum(len(p) for p in paragraphs) > target_chars:
        paragraphs = _step3_priority_filter(paragraphs, top_k=top_k)
    if use_llm and sum(len(p) for p in paragraphs) > target_chars:
        paragraphs = _step4_llm_summarize(paragraphs)
    out = "\n\n".join(paragraphs)
    return _step5_cap(out, target_chars)
