"""
Phase 19: Market Regime Classifier

Classifies current market conditions into regimes for regime-conditional RAG filtering.
Uses ADX + EMA alignment + ATR volatility (rule-based MVP).
Can be upgraded to HMM (hmmlearn) when enough data accumulates.

Regimes:
  - trending_bull: ADX>25 + price > EMA200 (strong uptrend)
  - trending_bear: ADX>25 + price < EMA200 (strong downtrend)
  - ranging:       ADX<20 (sideways chop)
  - high_volatility: ATR > 2x average (volatile, direction unclear)
  - transitional:  ADX 20-25 (regime changing)
"""

import logging
from typing import Dict, Any, Optional

# Phase 24: Neural Organism — adaptive parameters
try:
    from neural_organism import _p
except ImportError:
    def _p(param_id, fallback=0.5, regime="_global"):
        return fallback

logger = logging.getLogger(__name__)


class RegimeClassifier:
    """
    Rule-based regime classification from technical indicators.
    Singleton-safe, no database needed.
    """

    # Regime labels
    TRENDING_BULL = "trending_bull"
    TRENDING_BEAR = "trending_bear"
    RANGING = "ranging"
    HIGH_VOLATILITY = "high_volatility"
    TRANSITIONAL = "transitional"

    ALL_REGIMES = {TRENDING_BULL, TRENDING_BEAR, RANGING, HIGH_VOLATILITY, TRANSITIONAL}

    @staticmethod
    def classify(tech_data: Dict[str, Any]) -> str:
        """
        Classify market regime from technical indicator data.

        Expected keys in tech_data:
            - adx (float): Average Directional Index
            - atr (float): Average True Range
            - atr_sma (float): ATR 20-period SMA (for volatility ratio)
            - price or current_price (float): Current price
            - ema200 (float): 200-period EMA
            - ema20 (float): 20-period EMA (optional, for trend confirmation)

        Returns: regime string
        """
        adx = tech_data.get("adx") or tech_data.get("adx_14")
        atr = tech_data.get("atr") or tech_data.get("atr_14")
        atr_sma = tech_data.get("atr_sma") or tech_data.get("atr_avg")
        price = tech_data.get("price") or tech_data.get("current_price") or tech_data.get("close")
        ema200 = tech_data.get("ema200") or tech_data.get("ema_200")
        ema20 = tech_data.get("ema20") or tech_data.get("ema_20")

        # Default if no data
        if adx is None:
            return RegimeClassifier.TRANSITIONAL

        adx = float(adx)

        # Step 1: Check high volatility first (overrides everything)
        if atr and atr_sma and float(atr_sma) > 0:
            atr_ratio = float(atr) / float(atr_sma)
            if atr_ratio > _p("regime.atr_high_vol", 2.0):
                logger.info(f"[RegimeClassifier] HIGH_VOLATILITY: ATR ratio {atr_ratio:.2f}x")
                return RegimeClassifier.HIGH_VOLATILITY

        # Step 2: Ranging market (Phase 24: adaptive ADX threshold)
        if adx < _p("regime.adx_ranging", 20):
            logger.info(f"[RegimeClassifier] RANGING: ADX={adx:.1f}")
            return RegimeClassifier.RANGING

        # Step 3: Transitional
        if adx < _p("regime.adx_trending", 25):
            logger.info(f"[RegimeClassifier] TRANSITIONAL: ADX={adx:.1f}")
            return RegimeClassifier.TRANSITIONAL

        # Step 4: Trending — determine direction
        if price is not None and ema200 is not None:
            price = float(price)
            ema200 = float(ema200)
            if price > ema200:
                # Confirm with EMA20 if available
                if ema20 and float(ema20) > ema200:
                    logger.info(f"[RegimeClassifier] TRENDING_BULL: ADX={adx:.1f}, price>{ema200:.0f}, EMA20>{ema200:.0f}")
                else:
                    logger.info(f"[RegimeClassifier] TRENDING_BULL: ADX={adx:.1f}, price>{ema200:.0f}")
                return RegimeClassifier.TRENDING_BULL
            else:
                logger.info(f"[RegimeClassifier] TRENDING_BEAR: ADX={adx:.1f}, price<{ema200:.0f}")
                return RegimeClassifier.TRENDING_BEAR

        # ADX>25 but no price/EMA data — generic trending
        logger.info(f"[RegimeClassifier] TRENDING (generic): ADX={adx:.1f}, no price/EMA data")
        return RegimeClassifier.TRENDING_BULL  # Optimistic default

    @staticmethod
    def get_regime_description(regime: str) -> str:
        """Human-readable regime description for LLM prompts."""
        descriptions = {
            RegimeClassifier.TRENDING_BULL: "Strong bullish trend (ADX>25, price above EMA200). Trend-following strategies favored.",
            RegimeClassifier.TRENDING_BEAR: "Strong bearish trend (ADX>25, price below EMA200). Short or cash positions favored.",
            RegimeClassifier.RANGING: "Ranging/sideways market (ADX<20). Mean-reversion strategies favored, signals unreliable.",
            RegimeClassifier.HIGH_VOLATILITY: "High volatility regime (ATR>2x average). Extreme caution, reduce position sizes.",
            RegimeClassifier.TRANSITIONAL: "Transitional regime (ADX 20-25). Regime may be changing, lower confidence.",
        }
        return descriptions.get(regime, f"Unknown regime: {regime}")

    @staticmethod
    def get_confidence_modifier(regime: str) -> float:
        """
        Returns a multiplier for confidence adjustment based on regime.
        Ranging and volatile regimes reduce confidence.
        """
        modifiers = {
            RegimeClassifier.TRENDING_BULL: 1.0,
            RegimeClassifier.TRENDING_BEAR: 1.0,
            RegimeClassifier.RANGING: _p("regime.mod_ranging", 0.80),
            RegimeClassifier.HIGH_VOLATILITY: _p("regime.mod_high_vol", 0.75),
            RegimeClassifier.TRANSITIONAL: 0.90,
        }
        return modifiers.get(regime, 0.90)


# ═══════════════════════════════════════════════════════════════
# Phase 27 Task 12 (B5 Ajani): 4-Layer Regime Detection
# ═══════════════════════════════════════════════════════════════

class _BOCPDState:
    """Lightweight per-pair online-changepoint state.

    Tracks rolling mean + variance so the Score-Driven BOCPD heuristic can
    flag distribution drift without pulling in a full `bayesian-changepoint`
    dependency. We estimate the z-score of the latest observation against
    the rolling mean; large absolute z-scores push down the expected residual
    time to the next changepoint.
    """

    def __init__(self, window: int = 60):
        from collections import deque as _deque
        self._window = window
        self._buf = _deque(maxlen=window)

    def update(self, returns) -> Dict[str, float]:
        import math as _math
        import numpy as _np
        # Accept scalar or iterable
        if returns is None:
            return {"median_residual_time": 24.0, "score_magnitude": 0.0}
        try:
            arr = _np.asarray(list(returns), dtype=float)
            if arr.size == 0:
                return {"median_residual_time": 24.0, "score_magnitude": 0.0}
            latest = float(arr[-1])
            for val in arr:
                self._buf.append(float(val))
        except TypeError:
            latest = float(returns)
            self._buf.append(latest)

        buf = list(self._buf)
        if len(buf) < 5:
            return {"median_residual_time": 24.0, "score_magnitude": 0.0}
        mean = float(_np.mean(buf))
        std = float(_np.std(buf)) or 1e-8
        score_magnitude = abs(latest - mean) / std

        # Residual time heuristic: high surprise → fewer hours until drift becomes
        # detectable, low surprise → more hours. Clamped to a realistic 1-48h range.
        residual_time = max(1.0, min(48.0, 24.0 * _math.exp(-score_magnitude / 2.0)))
        return {
            "median_residual_time": residual_time,
            "score_magnitude": score_magnitude,
        }


class FourLayerRegimeDetector:
    """Phase 27 Task 12: ensemble regime detector with 4 complementary layers.

    Lead time ranking (fastest → slowest):
        Layer 0 — VPIN toxicity composite (1–3h lead, from order_flow.py)
        Layer 1 — Score-Driven BOCPD (2–6h lead, rolling return surprise)
        Layer 2 — Causal edge instability (4–12h lead, causal_engine PCMCI+)
        Layer 3 — ADX + EMA (confirmation only, legacy RegimeClassifier)

    Output is persisted to `regime_layers` so post-mortems can reconstruct the
    layer agreement at decision time and so sizing is reproducible.
    """

    def __init__(self):
        self._bocpd: Dict[str, _BOCPDState] = {}

    def _get_bocpd(self, pair: str) -> _BOCPDState:
        st = self._bocpd.get(pair)
        if st is None:
            st = _BOCPDState()
            self._bocpd[pair] = st
        return st

    def _layer0_vpin(self, pair: str) -> Dict[str, Any]:
        """Read VPIN / composite toxicity from the OrderFlowAnalyzer (Grup 3)."""
        try:
            from order_flow import get_order_flow
            of = get_order_flow()
            # Grup 3 stores histories per-pair — use the freshest composite.
            vpin_hist = of._vpin_history.get(pair)
            latest_vpin = float(vpin_hist[-1]) if vpin_hist and len(vpin_hist) else 0.0
            # Percentile-rank against own history
            alert = False
            if vpin_hist and len(vpin_hist) >= 10:
                from order_flow import _percentile_rank
                pct = _percentile_rank(latest_vpin, list(vpin_hist))
                alert = pct > 0.80  # top-20% toxic
            return {"alert": bool(alert), "value": round(latest_vpin, 4)}
        except Exception as e:
            logger.debug(f"[Regime:Layer0] VPIN read failed ({pair}): {e}")
            return {"alert": False, "value": 0.0}

    def _layer1_bocpd(self, pair: str, tech_data: Dict[str, Any]) -> Dict[str, Any]:
        """Score-Driven BOCPD on recent returns."""
        closes = tech_data.get("recent_closes") or []
        if not closes or len(closes) < 2:
            return {"alert": False, "residual_time": 24.0, "score_magnitude": 0.0}
        import numpy as _np
        arr = _np.asarray(closes, dtype=float)
        rets = _np.diff(_np.log(arr[arr > 0])) if (arr > 0).all() else _np.array([0.0])
        state = self._get_bocpd(pair)
        bocpd = state.update(rets)
        residual_time = float(bocpd.get("median_residual_time", 24.0))
        score_magnitude = float(bocpd.get("score_magnitude", 0.0))
        alert = bool(residual_time < 8.0 or score_magnitude > 2.0)
        return {
            "alert": alert,
            "residual_time": round(residual_time, 2),
            "score_magnitude": round(score_magnitude, 4),
        }

    def _layer2_causal(self, pair: str, regime_hint: Optional[str] = None) -> Dict[str, Any]:
        """Causal edge instability index.

        `causal_discoveries` has no `pair` column (PCMCI+ fits across the whole
        market by design — edges are market-wide structural claims). To make
        the signal as pair-relevant as we can, we filter by the pair's CURRENT
        regime when known: the edges in "trending_bull" have different stability
        profiles than "ranging", and that is the dimension causal_discoveries
        actually supports.
        """
        try:
            import numpy as _np
            from db import get_db_connection
            conn = get_db_connection()
            # Audit fix (2026-04-19): only honour TRUE PCMCI+ edges. Pre-fix,
            # gnn_organism wrote attention patterns with method='GNN_attention'
            # into the same table, polluting the instability index with
            # news-entity co-occurrence noise.
            if regime_hint:
                rows = conn.execute("""
                    SELECT source_var, target_var, causal_strength, regime
                    FROM causal_discoveries
                    WHERE discovered_at > datetime('now', '-14 days')
                      AND regime = ?
                      AND method = 'PCMCI+'
                """, (regime_hint,)).fetchall()
            else:
                rows = conn.execute("""
                    SELECT source_var, target_var, causal_strength, regime
                    FROM causal_discoveries
                    WHERE discovered_at > datetime('now', '-14 days')
                      AND method = 'PCMCI+'
                """).fetchall()
            conn.close()
            if not rows or len(rows) < 2:
                return {"alert": False, "instability": 0.0,
                        "regime_filter": regime_hint or "_any"}
            strengths = _np.asarray(
                [float(r["causal_strength"] or 0.0) for r in rows]
            )
            mean_abs = float(_np.mean(_np.abs(strengths))) + 1e-8
            instability = float(_np.std(strengths)) / mean_abs
            alert = instability > 0.25
            return {
                "alert": bool(alert),
                "instability": round(instability, 4),
                "regime_filter": regime_hint or "_any",
                "n_edges": len(rows),
            }
        except Exception as e:
            logger.debug(f"[Regime:Layer2] causal read failed: {e}")
            return {"alert": False, "instability": 0.0,
                    "regime_filter": regime_hint or "_any"}

    def detect(self, pair: str, tech_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run all 4 layers and fuse them into a single regime + sizing modifier.

        Classify ADX regime FIRST so Layer 2 can filter causal edges by the
        pair's current regime (causal_discoveries has no pair column; regime
        is the closest approximation to per-pair relevance).
        """
        l3_regime = RegimeClassifier.classify(tech_data)
        l0 = self._layer0_vpin(pair)
        l1 = self._layer1_bocpd(pair, tech_data)
        l2 = self._layer2_causal(pair, regime_hint=l3_regime)

        alert_count = int(l0["alert"]) + int(l1["alert"]) + int(l2["alert"])
        if alert_count == 0:
            prob, status = 0.10, "STABLE"
        elif alert_count == 1:
            prob, status = 0.50, "MICROSTRUCTURE_ANOMALY"
        elif alert_count == 2:
            prob, status = 0.75, "REGIME_CHANGE_LIKELY"
        else:
            prob, status = 0.95, "REGIME_CHANGE_IMMINENT"

        sizing_modifier = max(0.3, 1.0 - prob * 0.5)

        result = {
            "pair": pair,
            "current_regime": l3_regime,
            "regime_change_prob": round(prob, 3),
            "status": status,
            "sizing_modifier": round(sizing_modifier, 3),
            "layers": {
                "L0_vpin": l0,
                "L1_bocpd": l1,
                "L2_causal": l2,
                "L3_adx": {"regime": l3_regime},
            },
        }
        self._persist(pair, result)
        return result

    def _persist(self, pair: str, result: Dict[str, Any]) -> None:
        try:
            from datetime import datetime as _dt, timezone as _tz
            from db import get_db_connection
            conn = get_db_connection()
            layers = result["layers"]
            conn.execute("""
                INSERT OR REPLACE INTO regime_layers
                    (pair, timestamp,
                     layer0_vpin, layer0_alert,
                     layer1_bocpd_residual, layer1_alert,
                     layer2_causal_instability, layer2_alert,
                     layer3_adx_regime,
                     regime_change_prob, sizing_modifier, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                pair, _dt.now(tz=_tz.utc).isoformat(),
                float(layers["L0_vpin"]["value"]), int(bool(layers["L0_vpin"]["alert"])),
                float(layers["L1_bocpd"].get("residual_time", 0.0)), int(bool(layers["L1_bocpd"]["alert"])),
                float(layers["L2_causal"]["instability"]), int(bool(layers["L2_causal"]["alert"])),
                layers["L3_adx"]["regime"],
                float(result["regime_change_prob"]),
                float(result["sizing_modifier"]),
                result["status"],
            ))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.debug(f"[Regime:Layer] persist failed for {pair}: {e}")


_four_layer: Optional["FourLayerRegimeDetector"] = None


def get_regime_detector() -> "FourLayerRegimeDetector":
    """Singleton accessor for the 4-layer detector."""
    global _four_layer
    if _four_layer is None:
        _four_layer = FourLayerRegimeDetector()
    return _four_layer
