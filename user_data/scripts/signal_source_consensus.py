"""
signal_source_consensus.py — Multi-source agreement → confidence override.

Sprint 2026-05-02 — Bot critically evaluates its own RAG outputs.

Problem this solves:
  Production observation: 60% of TR-DRY signals returned NEUTRAL conf=0.13.
  Root cause: RAG service has a default-ish 0.225 raw confidence when its
  internal Triple Perception fluctuates near 0.5. Bot blindly inherited
  that default and passed it to the trade gate.

  But during the same observation window, OTHER signal sources (Triple
  Perception itself, Multi-TF agreement, VPIN, funding-rate edge, Hurst
  regime) often had clear directional opinions. The bot wasn't synthesizing
  them — it was just trusting RAG.

This module turns multi-source disagreement into a CONVICTION OVERRIDE:
  • When 4+ of 6 sources agree directionally and RAG returned NEUTRAL/weak,
    BUMP the final confidence to the consensus level
  • When sources DISAGREE wildly, KEEP the low confidence (genuine
    uncertainty)
  • Detects RAG bias: tracks the rolling 100-sample distribution of RAG
    raw confidence values; if "0.225 ± 0.05" appears in > 50% of recent
    samples, flag RAG as biased and reduce its consensus weight

Sources weighted (default — neuron-tunable via PARAM_REGISTRY):
  • RAG ENSEMBLE (the ai_decision)               weight 0.25
  • Triple Perception (1H/4H/1D fusion)          weight 0.25
  • Multi-TF Agreement (4h + daily trend match)  weight 0.15
  • VPIN order flow toxicity                     weight 0.10
  • Funding rate alignment                        weight 0.10
  • Hurst regime structural fit                   weight 0.15

Public API:
  • compute_consensus(pair, ai_decision, dataframe) → dict
      Returns:
        consensus_signal:  BULLISH / BEARISH / NEUTRAL
        consensus_conf:    [0..1]
        agreement:         number of aligned sources (0..6)
        sources_active:    sources that had a non-neutral opinion
        rag_bias_active:   bool — RAG flagged as biased
        action:            "override" / "use_rag" / "hold"
  • record_rag_observation(raw_conf, signal) → None
      Feeds the rolling RAG distribution tracker. Called from
      _get_ai_signal in HydraSizer.

When `action == "override"`, the strategy should USE consensus_signal +
consensus_conf as the final decision instead of ai_decision's values.

PARAM_REGISTRY:
  envelope.consensus.min_agreement_to_override
  envelope.consensus.rag_bias_window
  envelope.consensus.rag_bias_mode_pct
  envelope.consensus.weights.* (six per-source weights)
"""
from __future__ import annotations

import logging
import threading
import time
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# Module-level state
# Audit 2026-05-02 #15: maxlen=600 covers PARAM_REGISTRY.rag_bias_window
# max value 500 plus safety margin. Was 200 — neuron tuning to >200
# silently lost samples.
_RAG_HISTORY: Deque[Tuple[float, float, str]] = deque(maxlen=600)  # (ts, raw_conf, signal)
_LOCK = threading.Lock()


def _params() -> Dict[str, float]:
    try:
        from neural_organism import _p
        return {
            "min_agreement": int(_p("envelope.consensus.min_agreement_to_override", 4)),
            "rag_bias_window": int(_p("envelope.consensus.rag_bias_window", 100)),
            "rag_bias_mode_pct": float(_p("envelope.consensus.rag_bias_mode_pct", 0.50)),
            "rag_bias_tolerance": float(_p("envelope.consensus.rag_bias_tolerance", 0.05)),
            "weight_rag":     float(_p("envelope.consensus.weight.rag", 0.25)),
            "weight_tp":      float(_p("envelope.consensus.weight.tp", 0.25)),
            "weight_mtf":     float(_p("envelope.consensus.weight.mtf", 0.15)),
            "weight_vpin":    float(_p("envelope.consensus.weight.vpin", 0.10)),
            "weight_funding": float(_p("envelope.consensus.weight.funding", 0.10)),
            "weight_hurst":   float(_p("envelope.consensus.weight.hurst", 0.15)),
            # Audit 2026-05-02 #5/#6/#7 fixes
            "directional_threshold": float(_p("envelope.consensus.directional_threshold", 0.10)),
            "rag_weak_conf_floor":   float(_p("envelope.consensus.rag_weak_conf_floor", 0.30)),
            "rag_weak_dir_floor":    float(_p("envelope.consensus.rag_weak_dir_floor", 0.50)),
            "funding_conf_scaler":   float(_p("envelope.consensus.funding_conf_scaler", 1000.0)),
            "hurst_trending_floor":  float(_p("envelope.consensus.hurst_trending_floor", 0.55)),
            "hurst_mr_ceiling":      float(_p("envelope.consensus.hurst_mean_revert_ceiling", 0.45)),
            "hurst_conf_normalizer": float(_p("envelope.consensus.hurst_conf_normalizer", 0.30)),
        }
    except Exception:
        return {
            "min_agreement": 4,
            "rag_bias_window": 100,
            "rag_bias_mode_pct": 0.50,
            "rag_bias_tolerance": 0.05,
            "weight_rag": 0.25, "weight_tp": 0.25, "weight_mtf": 0.15,
            "weight_vpin": 0.10, "weight_funding": 0.10, "weight_hurst": 0.15,
            "directional_threshold": 0.10,
            "rag_weak_conf_floor": 0.30,
            "rag_weak_dir_floor": 0.50,
            "funding_conf_scaler": 1000.0,
            "hurst_trending_floor": 0.55,
            "hurst_mr_ceiling": 0.45,
            "hurst_conf_normalizer": 0.30,
        }


def record_rag_observation(raw_conf: float, signal: str) -> None:
    """Add a raw RAG output to the rolling distribution tracker.
    Called by HydraSizer._get_ai_signal after every RAG response.
    """
    with _LOCK:
        _RAG_HISTORY.append((time.time(), float(raw_conf), str(signal)))


def detect_rag_bias() -> Dict[str, Any]:
    """Returns whether RAG appears to be stuck on a default value.

    Method: histogram the last `rag_bias_window` observations into 0.05-
    wide buckets. If any bucket holds > rag_bias_mode_pct of the mass,
    RAG is biased toward that value.
    """
    cfg = _params()
    with _LOCK:
        history = list(_RAG_HISTORY)[-int(cfg["rag_bias_window"]):]
    if len(history) < 5:
        # Sprint 2026-05-04: cold-start floor lowered from 20→5
        # so bias-detector engages quickly after restart, preventing
        # RAG-default-0.225 from dominating consensus weighting.
        return {"biased": False, "n": len(history), "reason": "insufficient_data"}
    values = [h[1] for h in history]
    # Bucketize by tolerance
    tol = cfg["rag_bias_tolerance"]
    buckets: Dict[float, int] = {}
    for v in values:
        # Round to nearest tolerance grid
        bucket = round(v / tol) * tol
        buckets[bucket] = buckets.get(bucket, 0) + 1
    n = len(values)
    top_bucket, top_count = max(buckets.items(), key=lambda x: x[1])
    top_pct = top_count / n
    biased = top_pct > cfg["rag_bias_mode_pct"]
    return {
        "biased": biased,
        "n": n,
        "mode_value": round(top_bucket, 4),
        "mode_pct": round(top_pct, 3),
        "threshold": cfg["rag_bias_mode_pct"],
    }


def _direction_to_score(signal: str) -> float:
    """Map BULLISH/BEARISH/NEUTRAL to numeric direction [-1..+1]."""
    sig = (signal or "").upper()
    if sig in ("BULL", "BULLISH", "LONG"):
        return 1.0
    if sig in ("BEAR", "BEARISH", "SHORT"):
        return -1.0
    return 0.0


def _gather_sources(pair: str, ai_decision: Dict, dataframe) -> Dict[str, Dict]:
    """Collect each source's directional opinion + confidence.

    Returns a dict: source_name → {direction: -1/0/+1, conf: [0..1], reason: str}
    """
    sources: Dict[str, Dict] = {}

    # 1. RAG ENSEMBLE (the ai_decision)
    rag_dir = _direction_to_score(ai_decision.get("signal", "NEUTRAL"))
    rag_conf = float(ai_decision.get("confidence", 0.0) or 0.0)
    sources["rag"] = {"direction": rag_dir, "conf": rag_conf,
                      "reason": ai_decision.get("source", "RAG")}

    # 2. Triple Perception
    tp = ai_decision.get("perception") or {}
    tp_signal = tp.get("signal", "NEUTRAL") if isinstance(tp, dict) else "NEUTRAL"
    tp_conf = float(tp.get("confidence", 0.0) or 0.0) if isinstance(tp, dict) else 0.0
    sources["tp"] = {"direction": _direction_to_score(tp_signal),
                     "conf": tp_conf, "reason": "TriplePerception"}

    # 3. Multi-TF agreement
    # Audit 2026-05-02 #8 fix: PREFER the canonical MTF state from
    # ai_decision["mtf_state"] (populated by HydraSizer.populate_entry_
    # trend using _compute_higher_timeframe — which RESAMPLES to true
    # 4h/daily candles). Only fall back to 1h-EMA approximation when
    # the canonical state is missing (cold start before MTF gate ran).
    try:
        mtf_state = ai_decision.get("mtf_state") if isinstance(ai_decision, dict) else None
        trend_4h = None
        trend_daily = None
        if isinstance(mtf_state, dict):
            trend_4h = mtf_state.get("trend_4h")
            trend_daily = mtf_state.get("trend_daily")
        if trend_4h is None and trend_daily is None:
            # Fallback to 1h-EMA proxy ONLY when canonical state is missing
            last = dataframe.iloc[-1].squeeze() if dataframe is not None and len(dataframe) else None
            if last is not None:
                close = float(last.get("close") or 0)
                ema200 = float(last.get("ema_200") or 0)
                ema50 = float(last.get("ema_50") or 0)
                if close > 0 and ema200 > 0:
                    trend_daily = "bullish" if close > ema200 else "bearish"
                if close > 0 and ema50 > 0:
                    trend_4h = "bullish" if close > ema50 else "bearish"
        mtf_dir = 0.0
        mtf_conf = 0.0
        agree = 0
        for t in (trend_4h, trend_daily):
            if t == "bullish":
                mtf_dir += 1
                agree += 1
            elif t == "bearish":
                mtf_dir -= 1
                agree += 1
        if agree > 0:
            mtf_dir = 1.0 if mtf_dir > 0 else (-1.0 if mtf_dir < 0 else 0.0)
            mtf_conf = abs(mtf_dir) * (agree / 2.0)
        source_label = "canonical" if isinstance(mtf_state, dict) else "ema-proxy"
        sources["mtf"] = {"direction": mtf_dir, "conf": mtf_conf,
                          "reason": f"MTF[{source_label}] 4h={trend_4h} daily={trend_daily}"}
    except Exception:
        sources["mtf"] = {"direction": 0.0, "conf": 0.0, "reason": "MTF-unavailable"}

    # 4. VPIN order flow (low VPIN = clean → bot can act on directional view;
    # high VPIN = toxic → the bot should NOT trade direction; we mirror ai_dir
    # but with reduced confidence)
    try:
        from order_flow import get_order_flow
        of = get_order_flow()
        vpin_hist = of._vpin_history.get(pair) if hasattr(of, "_vpin_history") else None
        vpin_now = float(vpin_hist[-1]) if vpin_hist else 0.5
        # When VPIN is low (calm), VPIN tends to support the prevailing
        # direction (no informed reversal pending). Use rag_dir × (1-vpin)
        # as a directional vote.
        vpin_support = (1.0 - min(1.0, max(0.0, vpin_now)))
        sources["vpin"] = {"direction": rag_dir * (1.0 if vpin_support > 0.5 else 0.0),
                           "conf": vpin_support,
                           "reason": f"VPIN={vpin_now:.2f}"}
    except Exception:
        sources["vpin"] = {"direction": 0.0, "conf": 0.0, "reason": "VPIN-unavailable"}

    # 5. Funding rate alignment
    try:
        from db import get_db_connection
        base_pair = pair.split(":")[0]
        with get_db_connection() as conn:
            row = conn.execute(
                "SELECT funding_rate FROM derivatives_data "
                "WHERE pair = ? ORDER BY timestamp DESC LIMIT 1",
                (base_pair,)
            ).fetchone()
        if row and row["funding_rate"] is not None:
            fr = float(row["funding_rate"])
            # Negative funding → shorts get paid → SHORT favored
            # Positive funding → longs paid for nothing → SHORT favored
            # Audit 2026-05-02 #6 fix: scaler is PARAM_REGISTRY-driven so
            # neurons can tune the response curve to the venue's typical
            # funding magnitudes.
            cfg_funding = _params()
            funding_dir = -1.0 if fr > 0 else 1.0
            funding_conf = min(1.0, abs(fr) * cfg_funding["funding_conf_scaler"])
            sources["funding"] = {"direction": funding_dir, "conf": funding_conf,
                                  "reason": f"funding={fr:+.4%}"}
        else:
            sources["funding"] = {"direction": 0.0, "conf": 0.0,
                                  "reason": "funding-unavailable"}
    except Exception:
        sources["funding"] = {"direction": 0.0, "conf": 0.0,
                              "reason": "funding-error"}

    # 6. Hurst regime structural fit
    try:
        from pheromone_field import get_pheromone_field
        h_state = get_pheromone_field().read(f"hurst_3vote::{pair}")
        if isinstance(h_state, dict):
            hurst = float(h_state.get("hurst", 0.5))
            # Audit 2026-05-02 #5 fix: thresholds + normalizer come from
            # PARAM_REGISTRY (envelope.consensus.hurst_*), no longer
            # duplicated as 0.55/0.45/0.30 magic numbers.
            cfg_hurst = _params()
            tr_floor = cfg_hurst["hurst_trending_floor"]
            mr_ceiling = cfg_hurst["hurst_mr_ceiling"]
            normalizer = max(0.01, cfg_hurst["hurst_conf_normalizer"])
            if hurst > tr_floor:
                sources["hurst"] = {"direction": rag_dir,
                                    "conf": min(1.0, (hurst - tr_floor) / normalizer),
                                    "reason": f"Hurst trending {hurst:.2f}"}
            elif hurst < mr_ceiling:
                sources["hurst"] = {"direction": -rag_dir,
                                    "conf": min(1.0, (mr_ceiling - hurst) / normalizer),
                                    "reason": f"Hurst mean-revert {hurst:.2f}"}
            else:
                sources["hurst"] = {"direction": 0.0, "conf": 0.0,
                                    "reason": f"Hurst random {hurst:.2f}"}
        else:
            sources["hurst"] = {"direction": 0.0, "conf": 0.0,
                                "reason": "Hurst-unavailable"}
    except Exception:
        sources["hurst"] = {"direction": 0.0, "conf": 0.0,
                            "reason": "Hurst-error"}

    return sources


def compute_consensus(pair: str, ai_decision: Dict,
                      dataframe=None) -> Dict[str, Any]:
    """Run multi-source consensus and decide whether to override RAG.

    Returns:
      consensus_signal:    "BULLISH" / "BEARISH" / "NEUTRAL"
      consensus_conf:      [0..1]
      agreement:           int — sources aligned with consensus direction
      sources_active:      int — sources with non-zero direction vote
      rag_bias_active:     bool
      action:              "override" / "use_rag" / "hold"
      breakdown:           dict — full per-source detail (audit trail)
    """
    cfg = _params()
    sources = _gather_sources(pair, ai_decision, dataframe)
    bias_state = detect_rag_bias()
    rag_biased = bias_state.get("biased", False)

    # Per-source weights — when RAG is biased, redistribute its weight
    # proportionally to the others.
    weights = {
        "rag": cfg["weight_rag"],
        "tp": cfg["weight_tp"],
        "mtf": cfg["weight_mtf"],
        "vpin": cfg["weight_vpin"],
        "funding": cfg["weight_funding"],
        "hurst": cfg["weight_hurst"],
    }
    if rag_biased:
        rag_w = weights["rag"]
        weights["rag"] = 0.10  # demoted to minimal voice
        # redistribute (rag_w - 0.10) across other sources proportionally
        diff = rag_w - 0.10
        others = {k: v for k, v in weights.items() if k != "rag"}
        total_others = sum(others.values()) or 1.0
        for k in others:
            weights[k] += diff * (others[k] / total_others)

    # Compute weighted directional sum + agreement count
    weighted_dir = 0.0
    weighted_conf = 0.0
    agreement = 0
    sources_active = 0
    for name, info in sources.items():
        d = info["direction"]
        c = info["conf"]
        w = weights[name]
        weighted_dir += d * c * w
        weighted_conf += c * w
        if abs(d) > 0.5:
            sources_active += 1

    # Determine consensus signal — Audit 2026-05-02 #7 fix: threshold from
    # PARAM_REGISTRY so neurons can tune sensitivity. Higher threshold =
    # need stronger consensus to declare direction.
    dir_thr = cfg["directional_threshold"]
    if weighted_dir > dir_thr:
        consensus_signal = "BULLISH"
        consensus_dir_score = 1.0
    elif weighted_dir < -dir_thr:
        consensus_signal = "BEARISH"
        consensus_dir_score = -1.0
    else:
        consensus_signal = "NEUTRAL"
        consensus_dir_score = 0.0

    # Count sources that ALIGN with consensus direction
    if consensus_dir_score != 0:
        agreement = sum(
            1 for info in sources.values()
            if info["direction"] * consensus_dir_score > 0
        )

    # Decide action — Audit 2026-05-02 #7 fix: floors are PARAM_REGISTRY
    rag_dir = sources["rag"]["direction"]
    rag_conf = sources["rag"]["conf"]
    rag_weak = (
        rag_conf < cfg["rag_weak_conf_floor"]
        or abs(rag_dir) < cfg["rag_weak_dir_floor"]
    )

    if (rag_weak and consensus_signal != "NEUTRAL"
            and agreement >= cfg["min_agreement"]):
        action = "override"
    elif rag_biased and consensus_signal != "NEUTRAL" and agreement >= 3:
        action = "override"
    else:
        action = "use_rag"

    return {
        "consensus_signal": consensus_signal,
        "consensus_conf": round(min(0.95, max(0.0, weighted_conf)), 4),
        "agreement": agreement,
        "sources_active": sources_active,
        "rag_bias_active": rag_biased,
        "action": action,
        "breakdown": {
            name: {
                "direction": round(info["direction"], 2),
                "conf": round(info["conf"], 3),
                "weight": round(weights[name], 3),
                "reason": info["reason"],
            }
            for name, info in sources.items()
        },
        "bias_state": bias_state,
    }
