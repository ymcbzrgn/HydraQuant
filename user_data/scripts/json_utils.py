"""
Shared JSON extraction utility for LLM responses.
Prevents wasted LLM calls by robustly extracting JSON from text-wrapped responses.

3-tier extraction:
  Tier A: Direct json.loads() on full response
  Tier B: Brace extraction — find first { and last } (strips surrounding text)
  Tier C: Regex fallback — extract individual JSON fields

Usage:
    from json_utils import extract_json
    result = extract_json(llm_response_text)  # Returns dict or None
"""

import json
import re
import logging

logger = logging.getLogger(__name__)


def extract_json(text) -> dict | None:
    """
    Robustly extract a JSON object from LLM response text.
    Handles: raw JSON, JSON wrapped in text, markdown code fences, <think> tags,
    and Gemini v1 content blocks [{'type': 'text', 'text': '...'}].
    Returns parsed dict or None if all tiers fail.
    """
    # Pre-process: unwrap Gemini v1 content blocks if passed as list
    if isinstance(text, list):
        parts = []
        for block in text:
            if isinstance(block, dict) and "text" in block:
                parts.append(block["text"])
            elif isinstance(block, str):
                parts.append(block)
            else:
                parts.append(str(block))
        text = " ".join(parts)

    if not text or not isinstance(text, str):
        return None

    # Pre-clean: remove <think> blocks and markdown fences
    cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    cleaned = cleaned.replace("```json", "").replace("```", "").strip()

    if not cleaned:
        return None

    # Tier A: Direct parse (entire response is valid JSON)
    try:
        result = json.loads(cleaned)
        if isinstance(result, dict):
            return result
        if isinstance(result, list):
            return {"_array": result}  # Wrap arrays for uniform return type
    except (json.JSONDecodeError, ValueError):
        pass

    # Tier B: Brace extraction (LLM added text before/after JSON)
    brace_start = cleaned.find('{')
    brace_end = cleaned.rfind('}')
    if brace_start >= 0 and brace_end > brace_start:
        candidate = cleaned[brace_start:brace_end + 1]
        try:
            result = json.loads(candidate)
            if isinstance(result, dict):
                logger.debug("[json_utils] Extracted JSON via brace extraction")
                return result
        except (json.JSONDecodeError, ValueError):
            pass

    # Tier C: Try to find a JSON array (for sentiment_analyzer batch responses)
    bracket_start = cleaned.find('[')
    bracket_end = cleaned.rfind(']')
    if bracket_start >= 0 and bracket_end > bracket_start:
        candidate = cleaned[bracket_start:bracket_end + 1]
        try:
            result = json.loads(candidate)
            if isinstance(result, list):
                logger.debug("[json_utils] Extracted JSON array via bracket extraction")
                return {"_array": result}
        except (json.JSONDecodeError, ValueError):
            pass

    logger.warning(f"[json_utils] All extraction tiers failed. Raw text: {cleaned[:200]}")
    return None


def extract_json_array(text: str) -> list | None:
    """
    Extract a JSON array from LLM response text.
    Specifically designed for batch responses like sentiment scoring.
    Returns list or None.
    """
    result = extract_json(text)
    if result is None:
        return None
    if "_array" in result:
        return result["_array"]
    # If it's a dict, not an array
    return None


def extract_json_strict(text: str, required_keys: list[str] = None) -> dict | None:
    """
    Extract JSON and verify it contains required keys.
    Returns None if any required key is missing.
    """
    result = extract_json(text)
    if result is None:
        return None
    if required_keys:
        missing = [k for k in required_keys if k not in result]
        if missing:
            logger.warning(f"[json_utils] Extracted JSON missing required keys: {missing}")
            return None
    return result
