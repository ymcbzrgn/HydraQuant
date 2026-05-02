"""
adaptive_exit_urgency.py — Bot's smart maker/taker exit decision.

Sprint 2026-05-02 — Bot evaluates urgency on every exit, dynamically.

Problem this solves:
  Production observation (TR-DRY): NAORIS short hit ROI target +3% at
  09:03 UTC, but Freqtrade placed the exit as a maker limit. The limit
  sat unfilled until 11:48 UTC (2h45m), then trailing stop activated and
  the trade closed at +0.0% — the +3% gain evaporated waiting for a
  rebate worth 0.07%.

  Static maker-only is wrong for exits with realized profit. Static
  taker is wrong for low-vol mean-reversion exits. The right answer is
  PER-TRADE urgency-based adaptive routing.

How urgency is computed (no hardcoded thresholds):
  urgency = w1 × normalized_profit
          + w2 × normalized_volatility
          + w3 × normalized_age
          + w4 × regime_bonus
          + w5 × signal_reversal_penalty

  • normalized_profit: current_profit / target_roi (clipped 0..2)
  • normalized_volatility: ATR / 20-bar mean ATR (clipped 0..2)
  • normalized_age: (hours_held / 12.0) clipped 0..2
                    older positions → more urgent (carry cost)
  • regime_bonus: high_volatility → +0.3, ranging → -0.2, trending → 0
  • signal_reversal_penalty: if AI now signals OPPOSITE direction → +0.5

  urgency >= taker_threshold (default 1.0) → use TAKER (immediate fill)
  urgency <  threshold                     → use MAKER (capture rebate)

Public API:
  • compute_urgency(trade, current_profit, last_candle, ai_signal) → dict
  • should_use_taker_for_exit(...) → bool

Used by HydraSizer.custom_exit_price.

PARAM_REGISTRY:
  envelope.exit_urgency.taker_threshold
  envelope.exit_urgency.weight_profit
  envelope.exit_urgency.weight_volatility
  envelope.exit_urgency.weight_age
  envelope.exit_urgency.weight_regime
  envelope.exit_urgency.weight_reversal
  envelope.exit_urgency.target_roi_baseline
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def _params() -> Dict[str, float]:
    try:
        from neural_organism import _p
        return {
            "taker_threshold":     float(_p("envelope.exit_urgency.taker_threshold", 0.5)),
            "weight_profit":       float(_p("envelope.exit_urgency.weight_profit", 0.40)),
            "weight_volatility":   float(_p("envelope.exit_urgency.weight_volatility", 0.20)),
            "weight_age":          float(_p("envelope.exit_urgency.weight_age", 0.15)),
            "weight_regime":       float(_p("envelope.exit_urgency.weight_regime", 0.10)),
            "weight_reversal":     float(_p("envelope.exit_urgency.weight_reversal", 0.15)),
            "target_roi_baseline": float(_p("envelope.exit_urgency.target_roi_baseline", 0.045)),
            "max_age_hours":       float(_p("envelope.exit_urgency.max_age_hours", 12.0)),
        }
    except Exception:
        return {
            "taker_threshold": 0.5,
            "weight_profit": 0.40, "weight_volatility": 0.20,
            "weight_age": 0.15, "weight_regime": 0.10, "weight_reversal": 0.15,
            "target_roi_baseline": 0.045,
            "max_age_hours": 12.0,
        }


def compute_urgency(trade, current_profit: float,
                    last_candle: Optional[Dict] = None,
                    ai_signal: Optional[str] = None,
                    regime: Optional[str] = None,
                    current_time=None) -> Dict[str, Any]:
    """Compute urgency score and component breakdown for an exit decision.

    Returns dict:
      urgency_score:      [0..2+] — higher = more urgent (taker)
      should_use_taker:   bool
      action_reason:      str — explanation of dominant factor
      breakdown:          per-component values + weights
    """
    cfg = _params()

    # 1. Profit factor: current_profit normalized by target ROI baseline
    target_roi = cfg["target_roi_baseline"]
    profit_norm = abs(float(current_profit or 0.0)) / max(0.001, target_roi)
    profit_norm = min(2.0, profit_norm)
    # Only contributes positively when profit is at/above target
    profit_factor = profit_norm if current_profit >= 0 else 0.0

    # 2. Volatility factor: current ATR / rolling-mean ATR
    vol_factor = 0.0
    try:
        if last_candle is not None:
            atr = float(last_candle.get("atr") or 0.0)
            atr_avg = float(last_candle.get("atr_sma") or last_candle.get("atr_avg") or atr)
            if atr_avg > 0:
                vol_factor = min(2.0, atr / atr_avg)
                # Center on 1.0 so calm markets contribute 0
                vol_factor = max(0.0, vol_factor - 1.0)
    except Exception:
        vol_factor = 0.0

    # 3. Age factor: hours_held / max_age_hours (carry cost grows with time)
    # Audit 2026-05-02 #14 fix: when current_time is None, fall back to
    # datetime.now(UTC) instead of silently treating the trade as young.
    # A 24h-old position should NOT appear fresh just because a caller
    # forgot to pass current_time.
    age_factor = 0.0
    try:
        if trade.open_date_utc is not None:
            if current_time is None:
                from datetime import datetime as _dt, timezone as _tz
                current_time = _dt.now(tz=_tz.utc)
                logger.debug(
                    "[ExitUrgency] current_time was None — using utcnow() fallback"
                )
            hours_held = (current_time - trade.open_date_utc).total_seconds() / 3600.0
            age_factor = min(2.0, hours_held / cfg["max_age_hours"])
    except Exception:
        age_factor = 0.0

    # 4. Regime bonus: high_volatility favors taker, ranging favors maker
    regime_factor = 0.0
    if regime == "high_volatility":
        regime_factor = 0.5
    elif regime == "trending_bull" or regime == "trending_bear":
        regime_factor = 0.0  # neutral
    elif regime == "ranging":
        regime_factor = -0.3  # patient maker
    elif regime == "illiquid":
        regime_factor = -0.5  # don't market-order an illiquid book

    # 5. Reversal penalty: AI now signaling against the position
    reversal_factor = 0.0
    if ai_signal:
        sig = str(ai_signal).upper()
        is_short = bool(getattr(trade, "is_short", False))
        if is_short and sig in ("BULL", "BULLISH", "LONG"):
            reversal_factor = 1.0
        elif (not is_short) and sig in ("BEAR", "BEARISH", "SHORT"):
            reversal_factor = 1.0

    # Weighted urgency
    urgency = (
        cfg["weight_profit"] * profit_factor
        + cfg["weight_volatility"] * vol_factor
        + cfg["weight_age"] * age_factor
        + cfg["weight_regime"] * regime_factor
        + cfg["weight_reversal"] * reversal_factor
    )
    # Bound urgency to [-0.5, 3] (regime can push slightly negative)
    urgency = max(-0.5, min(3.0, urgency))
    threshold = cfg["taker_threshold"]
    use_taker = urgency >= threshold

    # Determine the dominant component for logging
    components = {
        "profit": cfg["weight_profit"] * profit_factor,
        "volatility": cfg["weight_volatility"] * vol_factor,
        "age": cfg["weight_age"] * age_factor,
        "regime": cfg["weight_regime"] * regime_factor,
        "reversal": cfg["weight_reversal"] * reversal_factor,
    }
    dominant = max(components.items(), key=lambda x: abs(x[1]))[0]

    return {
        "urgency_score": round(urgency, 3),
        "threshold": threshold,
        "should_use_taker": use_taker,
        "dominant_factor": dominant,
        "action_reason": (
            f"taker (dominant: {dominant})" if use_taker
            else f"maker (dominant: {dominant}, urgency<{threshold})"
        ),
        "breakdown": {
            "profit_norm": round(profit_norm, 3),
            "vol_factor": round(vol_factor, 3),
            "age_factor": round(age_factor, 3),
            "regime_factor": regime_factor,
            "reversal_factor": reversal_factor,
            "weighted_components": {k: round(v, 4) for k, v in components.items()},
        },
    }


def should_use_taker_for_exit(trade, current_profit: float,
                              last_candle: Optional[Dict] = None,
                              ai_signal: Optional[str] = None,
                              regime: Optional[str] = None,
                              current_time=None) -> bool:
    """Convenience boolean wrapper. True → strategy should price exit
    as a taker (cross the spread for immediate fill). False → maker."""
    info = compute_urgency(
        trade, current_profit, last_candle, ai_signal, regime, current_time
    )
    return bool(info.get("should_use_taker", False))
