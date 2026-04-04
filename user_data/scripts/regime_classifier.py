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
