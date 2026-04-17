"""
trade_language.py — Phase 27 Task 25 (I5 Ajani)

Treat the sequence of closed trades as a LANGUAGE and mine its patterns.

Three pattern-mining strategies (ALPHA doc §Task 25 graduated pipeline):
  1. N-gram frequency + chi-squared significance (p < 0.001)
  2. Sequential pattern mining via PrefixSpan (optional dependency)
  3. Conditional-probability grammar rules: P(outcome | pattern)

All discoveries are persisted to `sequence_patterns` so the MADAM coordinator
and sizing pipeline can consult them later (e.g. "BULL_HIGH_WIN, BULL_HIGH_WIN,
BEAR_LOW" has historically preceded 68% bear outcomes → next trade sizing
should shrink).

Scheduler runs this weekly (Sunday 08:00 UTC, after CatBoost retrain + OOD
refit so the labels the tokeniser consumes are fresh).
"""

from __future__ import annotations

import json
import logging
import math
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
logger = logging.getLogger("trade_language")

from ai_config import AI_DB_PATH
from db import get_db_connection, init_db


# ──────────────────────────────────────────────────────────────
# Tokeniser
# ──────────────────────────────────────────────────────────────

def _confidence_bucket(conf: float) -> str:
    if conf >= 0.65:
        return "HIGH"
    if conf >= 0.45:
        return "MED"
    return "LOW"


def tokenize_trade(row: Dict[str, Any]) -> str:
    """Decision-contract style token — direction × confidence × regime × outcome.

    The vocabulary is bounded by design: 3 directions × 3 conf buckets × 5
    regimes (trim) × 2 outcomes = 90 max — plenty of distinctness for
    statistical tests, few enough for N-gram chi-squared to have power.
    """
    signal = (row.get("signal_type") or "NEUT").upper()[:4]
    conf_b = _confidence_bucket(float(row.get("confidence") or 0.5))
    regime = (row.get("regime") or "UNK")[:4].upper()
    pnl = row.get("outcome_pnl")
    outcome = "WIN" if (pnl is not None and pnl > 0) else "LOSS"
    return f"{signal}_{conf_b}_{regime}_{outcome}"


def load_trade_corpus(min_trades: int = 100) -> List[Dict[str, Any]]:
    """Pull resolved trades oldest→newest."""
    try:
        conn = get_db_connection(AI_DB_PATH)
        rows = conn.execute("""
            SELECT signal_type, confidence, regime, outcome_pnl, timestamp
            FROM ai_decisions
            WHERE outcome_pnl IS NOT NULL
            ORDER BY timestamp ASC
        """).fetchall()
        conn.close()
    except Exception as e:
        logger.debug(f"[TradeLang] load corpus failed: {e}")
        return []
    if len(rows) < min_trades:
        return []
    return [dict(r) for r in rows]


# ──────────────────────────────────────────────────────────────
# N-gram frequency + chi-squared significance
# ──────────────────────────────────────────────────────────────

def find_ngram_patterns(tokens: List[str], n: int = 3,
                         min_chi2: float = 10.83) -> List[Tuple[Tuple[str, ...], int, float]]:
    """Return n-grams with chi-squared > `min_chi2` (p < 0.001 @ df=1)."""
    if len(tokens) < n + 1:
        return []
    ngrams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n)]
    counts = Counter(ngrams)
    total = len(ngrams)
    vocab = len(set(tokens))
    if vocab == 0:
        return []
    expected = total / max(vocab ** n, 1)
    significant: List[Tuple[Tuple[str, ...], int, float]] = []
    for pattern, observed in counts.most_common(200):
        if expected <= 0:
            continue
        chi2 = ((observed - expected) ** 2) / expected
        if chi2 > min_chi2:
            significant.append((pattern, observed, round(chi2, 2)))
    return significant


# ──────────────────────────────────────────────────────────────
# Sequential pattern mining (PrefixSpan — optional)
# ──────────────────────────────────────────────────────────────

def find_prefixspan_patterns(token_sequences: List[List[str]],
                              min_support: int = 5) -> List[Tuple[Tuple[str, ...], int]]:
    """Use `prefixspan` library if installed; else empty list.

    Returns a list of (pattern_tuple, support) sorted by support descending.
    """
    try:
        from prefixspan import PrefixSpan  # type: ignore
    except Exception:
        return []
    try:
        ps = PrefixSpan(token_sequences)
        ps.minlen = 2
        ps.maxlen = 5
        raw = ps.frequent(min_support)
        out: List[Tuple[Tuple[str, ...], int]] = []
        for support, pat in raw:
            if len(pat) >= 2:
                out.append((tuple(pat), int(support)))
        out.sort(key=lambda x: -x[1])
        return out[:50]
    except Exception as e:
        logger.debug(f"[TradeLang] prefixspan failed: {e}")
        return []


# ──────────────────────────────────────────────────────────────
# Conditional-probability grammar rules
# ──────────────────────────────────────────────────────────────

def grammar_rules(tokens: List[str], context_len: int = 2) -> Dict[str, Dict[str, float]]:
    """Build P(next_outcome | context) conditional probability tables.

    `context` = last `context_len` tokens BEFORE a pair of (direction, outcome).
    Returns a dict: `context_key → {outcome_token: probability}` with only
    rules that have ≥ 5 observations retained.
    """
    if len(tokens) < context_len + 1:
        return {}
    bucket: Dict[Tuple[str, ...], Counter] = defaultdict(Counter)
    for i in range(context_len, len(tokens)):
        ctx = tuple(tokens[i - context_len:i])
        outcome = tokens[i]
        bucket[ctx][outcome] += 1
    rules: Dict[str, Dict[str, float]] = {}
    for ctx, outcomes in bucket.items():
        total = sum(outcomes.values())
        if total < 5:
            continue
        rules[",".join(ctx)] = {
            out: round(c / total, 4) for out, c in outcomes.items()
        }
    return rules


# ──────────────────────────────────────────────────────────────
# Persistence
# ──────────────────────────────────────────────────────────────

def _persist_pattern(pattern: str, n_gram_size: int, occurrences: int,
                      expected: float, chi2_score: float,
                      next_dist: Dict[str, float],
                      regime: Optional[str] = None) -> None:
    """Upsert a pattern row. Post-audit fix: the Phase 26 INSERT variant
    flooded `sequence_patterns` with a fresh row every Sunday; switching to
    ON CONFLICT(pattern, regime) DO UPDATE keeps one canonical row per
    pattern and lets `occurrences` / `chi2_score` evolve over time.
    """
    try:
        conn = get_db_connection(AI_DB_PATH)
        conn.execute("""
            INSERT INTO sequence_patterns
                (pattern, n_gram_size, occurrences, expected_occurrences,
                 chi2_score, p_value, next_outcome_distribution,
                 regime, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(pattern, regime) DO UPDATE SET
                n_gram_size = excluded.n_gram_size,
                occurrences = excluded.occurrences,
                expected_occurrences = excluded.expected_occurrences,
                chi2_score = excluded.chi2_score,
                p_value = excluded.p_value,
                next_outcome_distribution = excluded.next_outcome_distribution,
                updated_at = excluded.updated_at
        """, (
            pattern, n_gram_size, occurrences,
            float(expected), float(chi2_score),
            0.001 if chi2_score > 10.83 else 0.05,
            json.dumps(next_dist),
            regime or "_any",
            datetime.now(tz=timezone.utc).isoformat(),
        ))
        conn.commit()
        conn.close()
    except Exception as e:
        logger.debug(f"[TradeLang] persist failed: {e}")


def weekly_cycle(min_trades: int = 100) -> Dict[str, Any]:
    """Scheduler entrypoint — mine patterns, persist significant ones."""
    init_db()
    rows = load_trade_corpus(min_trades=min_trades)
    if not rows:
        return {"status": "insufficient_data", "n_trades": 0,
                "min_trades": min_trades}
    tokens = [tokenize_trade(r) for r in rows]

    n_gram_results = find_ngram_patterns(tokens, n=3)
    rules = grammar_rules(tokens, context_len=2)
    prefixspan_patterns = find_prefixspan_patterns([tokens])

    # Persist chi-squared-significant trigrams.
    total_ngrams = max(len(tokens) - 2, 1)
    vocab = max(len(set(tokens)), 1)
    expected = total_ngrams / (vocab ** 3)
    persisted = 0
    for pattern, occ, chi2 in n_gram_results[:50]:
        pattern_str = ",".join(pattern)
        next_dist = rules.get(",".join(pattern[:2]), {})
        _persist_pattern(pattern_str, n_gram_size=3, occurrences=occ,
                         expected=expected, chi2_score=chi2,
                         next_dist=next_dist)
        persisted += 1

    # Persist PrefixSpan discoveries (context_len doesn't apply).
    for pattern, support in prefixspan_patterns[:25]:
        pattern_str = ",".join(pattern)
        _persist_pattern(pattern_str, n_gram_size=len(pattern),
                         occurrences=support, expected=0.0, chi2_score=0.0,
                         next_dist={})
        persisted += 1

    logger.info(
        f"[TradeLang] weekly cycle: n_trades={len(rows)}, "
        f"ngrams_sig={len(n_gram_results)}, rules={len(rules)}, "
        f"prefixspan={len(prefixspan_patterns)}, persisted={persisted}"
    )
    return {
        "status": "ok",
        "n_trades": len(rows),
        "ngrams_significant": len(n_gram_results),
        "grammar_rules": len(rules),
        "prefixspan_patterns": len(prefixspan_patterns),
        "persisted": persisted,
    }
