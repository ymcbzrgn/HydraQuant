"""Phase 30 A.9 — LLM thinking-block scrubber.

Strips reasoning-mode artifacts (<think>, <thinking>, <reasoning>, ```thinking)
from user-facing responses before Telegram/log emission. Also handles raw
DeepSeek/Qwen "<think>...let me reason..." prefixes that leaked into
ai_lessons rows in production.
"""
from __future__ import annotations

import re
from typing import Optional

THINK_PATTERNS = [
    re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE),
    re.compile(r"<thinking>.*?</thinking>", re.DOTALL | re.IGNORECASE),
    re.compile(r"<reasoning>.*?</reasoning>", re.DOTALL | re.IGNORECASE),
    re.compile(r"```thinking.*?```", re.DOTALL | re.IGNORECASE),
    re.compile(r"```think\b.*?```", re.DOTALL | re.IGNORECASE),
]

PREFIX_PATTERNS = [
    re.compile(r"^\s*let me think.*?(?=\n\n|\Z)", re.DOTALL | re.IGNORECASE | re.MULTILINE),
    re.compile(r"^\s*thinking step by step.*?(?=\n\n|\Z)", re.DOTALL | re.IGNORECASE | re.MULTILINE),
    re.compile(r"^\s*okay,? let'?s tackle this.*?(?=\n\n|\Z)", re.DOTALL | re.IGNORECASE | re.MULTILINE),
]

UNCLOSED_THINK_RE = re.compile(r"<think>(?!.*</think>).*", re.DOTALL | re.IGNORECASE)


def scrub(text: Optional[str], preserve_markers: bool = False) -> str:
    if not text:
        return text or ""
    for pat in THINK_PATTERNS:
        text = pat.sub("[THINK_REDACTED]" if preserve_markers else "", text)
    for pat in PREFIX_PATTERNS:
        text = pat.sub("", text)
    text = UNCLOSED_THINK_RE.sub("[THINK_UNCLOSED_REDACTED]" if preserve_markers else "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def has_thinking(text: Optional[str]) -> bool:
    if not text:
        return False
    for pat in THINK_PATTERNS + PREFIX_PATTERNS:
        if pat.search(text):
            return True
    return bool(UNCLOSED_THINK_RE.search(text))
