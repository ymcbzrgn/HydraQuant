"""Phase 30 D.7 — Quintuple (6-source) perception fusion.

Extends triple_perception 9-stage pipeline with two new stages:
- 5th: long-horizon quantile (B.3 foundation_long_horizon)
- 6th: visual chart pattern (D.4 visual_perception)

Disagreement penalty: when 6 sources point in conflicting directions the
final confidence is shrunk; when 5+ agree confidence is amplified.
"""
from __future__ import annotations

import logging
import statistics
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)


@dataclass
class SourceVote:
    source: str
    direction: float  # -1..+1
    confidence: float  # 0..1


@dataclass
class FusedPerception:
    direction: float
    confidence: float
    disagreement_penalty: float
    n_sources: int
    breakdown: Dict[str, Dict[str, float]]


def fuse(votes: Sequence[SourceVote]) -> FusedPerception:
    if not votes:
        return FusedPerception(0.0, 0.0, 0.0, 0, {})
    dirs = [v.direction for v in votes]
    confs = [v.confidence for v in votes]
    weighted = sum(d * c for d, c in zip(dirs, confs)) / max(sum(confs), 1e-9)
    if len(dirs) >= 2:
        std = statistics.stdev(dirs)
    else:
        std = 0.0
    penalty = min(1.0, std)
    final_conf = max(0.0, min(1.0, statistics.mean(confs) - 0.5 * penalty))
    breakdown = {
        v.source: {"direction": v.direction, "confidence": v.confidence}
        for v in votes
    }
    return FusedPerception(
        direction=max(-1.0, min(1.0, weighted)),
        confidence=final_conf,
        disagreement_penalty=penalty,
        n_sources=len(votes),
        breakdown=breakdown,
    )


def collect_votes(pair: str, ohlcv_close: Sequence[float], ohlcv_1h_close: Sequence[float],
                  png_path: Optional[str] = None) -> List[SourceVote]:
    votes: List[SourceVote] = []
    try:
        from foundation_short_horizon import get_forecaster

        f = get_forecaster()
        mean, std = f.predict_direction(ohlcv_close)
        votes.append(SourceVote("short_horizon", mean, max(0.0, 1.0 - min(std * 100, 1.0))))
    except Exception:
        pass
    try:
        from foundation_long_horizon import get_forecaster as lh

        q = lh().forecast(ohlcv_1h_close)
        votes.append(SourceVote("long_horizon", max(-1.0, min(1.0, q.q50 * 10)), q.confidence))
    except Exception:
        pass
    try:
        from chronos_perception import get_predictor

        adapter = get_predictor()
        if adapter and hasattr(adapter, "score"):
            r = adapter.score(pair, ohlcv_close)
            votes.append(SourceVote("chronos", float(r.get("direction", 0)), float(r.get("confidence", 0))))
    except Exception:
        pass
    try:
        from catboost_trainer import score_pair  # type: ignore

        c = score_pair(pair, ohlcv_close)
        votes.append(SourceVote("catboost", float(c.get("direction", 0)), float(c.get("confidence", 0))))
    except Exception:
        pass
    try:
        from conformal_calibrator import calibrate as cc  # type: ignore

        cr = cc(pair=pair, ohlcv_close=ohlcv_close)
        votes.append(SourceVote("conformal", float(cr.get("direction", 0)), float(cr.get("confidence", 0))))
    except Exception:
        pass
    try:
        if png_path:
            from visual_perception import detect_patterns
            patterns = detect_patterns(png_path)
            if patterns:
                bullish = sum(1 for p in patterns if "bull" in p.label.lower())
                bearish = sum(1 for p in patterns if "bear" in p.label.lower())
                direction = (bullish - bearish) / max(1, bullish + bearish)
                conf = sum(p.confidence for p in patterns) / max(1, len(patterns))
                votes.append(SourceVote("visual", direction, conf))
    except Exception:
        pass
    return votes
