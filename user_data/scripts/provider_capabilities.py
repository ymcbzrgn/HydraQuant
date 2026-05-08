"""Phase 30 B.6 — Provider capabilities matrix.

Per-model: context_window, supports_json, supports_tools, costs.
Used by llm_router pre-check (skip prompts that exceed window or need
unsupported features).
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

DEFAULT_CAPABILITIES: Dict[str, Dict[str, Any]] = {
    "models/gemini-2.5-flash":         {"context_window": 1_000_000, "supports_json": 1, "supports_tools": 1, "cost_per_1m_in": 0.0, "cost_per_1m_out": 0.0},
    "models/gemini-2.5-pro":           {"context_window": 1_000_000, "supports_json": 1, "supports_tools": 1, "cost_per_1m_in": 0.0, "cost_per_1m_out": 0.0},
    "models/gemini-2.5-flash-lite":    {"context_window": 1_000_000, "supports_json": 1, "supports_tools": 1, "cost_per_1m_in": 0.0, "cost_per_1m_out": 0.0},
    "models/gemini-3.1-flash-lite":    {"context_window": 1_000_000, "supports_json": 1, "supports_tools": 1, "cost_per_1m_in": 0.0, "cost_per_1m_out": 0.0},
    "models/gemini-3.1-pro-preview":   {"context_window": 1_000_000, "supports_json": 1, "supports_tools": 1, "cost_per_1m_in": 0.0, "cost_per_1m_out": 0.0},
    "llama-3.1-8b-instant":            {"context_window": 131_072, "supports_json": 1, "supports_tools": 1, "cost_per_1m_in": 0.0, "cost_per_1m_out": 0.0},
    "llama-3.3-70b-versatile":         {"context_window": 32_768, "supports_json": 1, "supports_tools": 1, "cost_per_1m_in": 0.0, "cost_per_1m_out": 0.0},
    "qwen/qwen3-32b":                  {"context_window": 131_072, "supports_json": 1, "supports_tools": 0, "cost_per_1m_in": 0.0, "cost_per_1m_out": 0.0},
    "meta-llama/llama-4-scout-17b-16e-instruct": {"context_window": 131_072, "supports_json": 1, "supports_tools": 1, "cost_per_1m_in": 0.0, "cost_per_1m_out": 0.0},
    "mistral-small-latest":            {"context_window": 32_768, "supports_json": 1, "supports_tools": 1, "cost_per_1m_in": 0.0, "cost_per_1m_out": 0.0},
    "mistral-large-latest":            {"context_window": 32_768, "supports_json": 1, "supports_tools": 1, "cost_per_1m_in": 0.0, "cost_per_1m_out": 0.0},
    "openai/gpt-oss-20b":              {"context_window": 32_768, "supports_json": 1, "supports_tools": 1, "cost_per_1m_in": 0.0, "cost_per_1m_out": 0.0},
    "openai/gpt-oss-120b":             {"context_window": 32_768, "supports_json": 1, "supports_tools": 1, "cost_per_1m_in": 0.0, "cost_per_1m_out": 0.0},
    "claude-opus-4-7":                 {"context_window": 1_000_000, "supports_json": 1, "supports_tools": 1, "cost_per_1m_in": 15.0, "cost_per_1m_out": 75.0},
    "claude-sonnet-4-6":               {"context_window": 200_000,   "supports_json": 1, "supports_tools": 1, "cost_per_1m_in": 3.0, "cost_per_1m_out": 15.0},
    "claude-haiku-4-5-20251001":       {"context_window": 200_000,   "supports_json": 1, "supports_tools": 1, "cost_per_1m_in": 0.8, "cost_per_1m_out": 4.0},
}


def init_default() -> None:
    """Populate provider_capabilities table with defaults if empty."""
    try:
        from db import AI_DB_PATH, get_db_connection

        with get_db_connection(AI_DB_PATH) as conn:
            for model, caps in DEFAULT_CAPABILITIES.items():
                conn.execute(
                    """INSERT OR IGNORE INTO provider_capabilities
                       (model, context_window, supports_json, supports_tools,
                        cost_per_1m_in, cost_per_1m_out, current_concurrency, target_concurrency)
                       VALUES (?, ?, ?, ?, ?, ?, 2, 2)""",
                    (model, caps["context_window"], caps["supports_json"], caps["supports_tools"],
                     caps["cost_per_1m_in"], caps["cost_per_1m_out"]),
                )
            conn.commit()
    except Exception as e:
        logger.error(f"[Capabilities] init: {e}")


def lookup(model: str) -> Optional[Dict[str, Any]]:
    try:
        from db import AI_DB_PATH, get_db_connection

        with get_db_connection(AI_DB_PATH) as conn:
            row = conn.execute(
                """SELECT context_window, supports_json, supports_tools,
                          cost_per_1m_in, cost_per_1m_out, current_concurrency
                   FROM provider_capabilities WHERE model=?""",
                (model,),
            ).fetchone()
            if row:
                return dict(zip(
                    ("context_window", "supports_json", "supports_tools",
                     "cost_per_1m_in", "cost_per_1m_out", "current_concurrency"),
                    row,
                ))
    except Exception:
        pass
    return DEFAULT_CAPABILITIES.get(model)


def can_handle(model: str, prompt_tokens: int, needs_json: bool = False, needs_tools: bool = False) -> bool:
    caps = lookup(model) or {}
    if int(caps.get("context_window", 0)) < prompt_tokens + 256:
        return False
    if needs_json and not int(caps.get("supports_json", 0)):
        return False
    if needs_tools and not int(caps.get("supports_tools", 0)):
        return False
    return True
