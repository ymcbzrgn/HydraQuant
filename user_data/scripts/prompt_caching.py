"""Phase 30 B.14 — Anthropic prompt caching helper (system + 3 breakpoints).

Wraps Anthropic API messages with `cache_control` blocks at:
- System prompt (large, stable)
- Tool definitions
- Recent user context (if > 1024 tokens)

For non-Anthropic models, returns input unchanged.
"""
from __future__ import annotations

from typing import Any, Dict, List

ANTHROPIC_MODEL_PREFIXES = ("claude-",)
MIN_CACHE_TOKENS = 1024


def _is_anthropic(model: str) -> bool:
    return any(model.startswith(p) for p in ANTHROPIC_MODEL_PREFIXES)


def annotate_messages(model: str, system: str, messages: List[Dict[str, Any]],
                      tools: List[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Returns kwargs ready to pass to Anthropic SDK `messages.create`.

    For non-Anthropic models, returns plain dict {system, messages, tools}.
    """
    if not _is_anthropic(model):
        return {"system": system, "messages": messages, "tools": tools or []}

    sys_block: List[Dict[str, Any]] = []
    if system and len(system) >= MIN_CACHE_TOKENS:
        sys_block = [{
            "type": "text",
            "text": system,
            "cache_control": {"type": "ephemeral"},
        }]
    elif system:
        sys_block = [{"type": "text", "text": system}]

    annotated_tools: List[Dict[str, Any]] = []
    if tools:
        last_idx = len(tools) - 1
        for i, t in enumerate(tools):
            tt = dict(t)
            if i == last_idx:
                tt["cache_control"] = {"type": "ephemeral"}
            annotated_tools.append(tt)

    annotated_messages: List[Dict[str, Any]] = []
    for i, m in enumerate(messages):
        mm = dict(m)
        content = mm.get("content")
        if i == len(messages) - 1 and isinstance(content, str) and len(content) >= MIN_CACHE_TOKENS:
            mm["content"] = [{
                "type": "text",
                "text": content,
                "cache_control": {"type": "ephemeral"},
            }]
        annotated_messages.append(mm)

    return {
        "system": sys_block,
        "messages": annotated_messages,
        "tools": annotated_tools,
    }
