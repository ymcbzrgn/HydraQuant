"""Phase 30 A.12 — 4-State veto system (ALLOW / PASS / DENY / ASK).

Replaces binary agent_pool veto with weighted 4-state aggregation:
- ALLOW: agent actively endorses (weight=+1)
- PASS:  agent abstains (weight=0, no signal)
- DENY:  agent vetoes (weight=-1)
- ASK:   agent requests human review (weight=-0.5, telegram alert)

Aggregation uses EarnedTrust per-agent weight from agent_pool registry.
DENY threshold (default sum-weighted < -0.5) blocks the trade.
ASK any -> Telegram alert + auto-fallback to safer side.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

ALLOW = "ALLOW"
PASS_ = "PASS"
DENY = "DENY"
ASK = "ASK"

VOTE_WEIGHT = {ALLOW: +1.0, PASS_: 0.0, DENY: -1.0, ASK: -0.5}


@dataclass
class Vote:
    agent_name: str
    state: str  # one of ALLOW/PASS/DENY/ASK
    confidence: float = 1.0  # 0..1
    reason: str = ""


@dataclass
class VetoResult:
    decision: str  # "allow", "deny", "ask_human"
    aggregate_score: float
    n_allow: int
    n_pass: int
    n_deny: int
    n_ask: int
    blocked_by: List[str] = field(default_factory=list)
    asked_by: List[str] = field(default_factory=list)


def aggregate(
    votes: List[Vote],
    trust_weights: Optional[Dict[str, float]] = None,
    deny_threshold: float = -0.5,
    ask_one_blocks: bool = True,
) -> VetoResult:
    trust_weights = trust_weights or {}
    score = 0.0
    n_allow = n_pass = n_deny = n_ask = 0
    blocked_by: List[str] = []
    asked_by: List[str] = []
    for v in votes:
        w = trust_weights.get(v.agent_name, 1.0)
        s = v.state.upper()
        contribution = VOTE_WEIGHT.get(s, 0.0) * w * max(0.0, min(1.0, v.confidence))
        score += contribution
        if s == ALLOW:
            n_allow += 1
        elif s == PASS_:
            n_pass += 1
        elif s == DENY:
            n_deny += 1
            blocked_by.append(v.agent_name)
        elif s == ASK:
            n_ask += 1
            asked_by.append(v.agent_name)
    decision = "allow"
    if n_ask > 0 and ask_one_blocks:
        decision = "ask_human"
    elif score < deny_threshold:
        decision = "deny"
    return VetoResult(
        decision=decision,
        aggregate_score=score,
        n_allow=n_allow,
        n_pass=n_pass,
        n_deny=n_deny,
        n_ask=n_ask,
        blocked_by=blocked_by,
        asked_by=asked_by,
    )


def should_proceed(result: VetoResult) -> Tuple[bool, str]:
    if result.decision == "allow":
        return True, "allow"
    return False, result.decision
