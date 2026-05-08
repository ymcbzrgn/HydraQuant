"""Phase 30 B.11 (2/2) — Maximal Marginal Relevance diversification.

Standard MMR: at each step pick item maximizing
    lambda * relevance(q, d) - (1-lambda) * max_sim(d, selected)

Used after RAG retrieval to avoid 10 near-duplicate news items dominating
context. Scores in [0,1]; cosine similarity over token sets.
"""
from __future__ import annotations

import re
from typing import Callable, List, Optional, Sequence, Set, Tuple

WORD_RE = re.compile(r"[a-z0-9]{3,}")


def _toks(text: str) -> Set[str]:
    return set(WORD_RE.findall((text or "").lower()))


def _cos_jaccard(a: Set[str], b: Set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def mmr_select(
    query: str,
    candidates: Sequence[str],
    relevance_scores: Optional[Sequence[float]] = None,
    k: int = 5,
    lam: float = 0.7,
    sim_fn: Optional[Callable[[Set[str], Set[str]], float]] = None,
) -> List[int]:
    """Returns indices of selected candidates."""
    if not candidates:
        return []
    sim_fn = sim_fn or _cos_jaccard
    q_tokens = _toks(query)
    cand_tokens = [_toks(c) for c in candidates]
    if relevance_scores is None:
        relevance_scores = [sim_fn(q_tokens, t) for t in cand_tokens]
    selected: List[int] = []
    candidate_idx = list(range(len(candidates)))
    while len(selected) < min(k, len(candidates)):
        best_idx = -1
        best_score = -1.0
        for i in candidate_idx:
            if i in selected:
                continue
            rel = float(relevance_scores[i])
            redundancy = 0.0
            if selected:
                redundancy = max(sim_fn(cand_tokens[i], cand_tokens[j]) for j in selected)
            score = lam * rel - (1 - lam) * redundancy
            if score > best_score:
                best_score = score
                best_idx = i
        if best_idx < 0:
            break
        selected.append(best_idx)
    return selected
