"""
market_maker_mode.py — Phase 26 Sprint 2, Task 11D

Regime-Based Dual Trading: Market Making (ranging) ↔ Signal Following (trending).

When ADX < 20 (ranging): Switch to market making mode
  - GLFT formulas (no terminal time — suits 24/7 crypto)
  - Bar Portion (BP) signal for directional skew
  - Cortisol → gamma modulation (hormonal risk aversion)

When ADX > 25 (trending): Standard signal following mode (existing system)

Reference: Stoikov Jan 2025 (SSRN 5066176) — 45.84% return, 0.78 Sharpe, 3.94% max DD
           Lalor & Swishchuk 2025 — Hormonal gamma modulation

Integration:
  - Reads from: chart_features (ADX), hormone_state (cortisol), lob_encoder, order_flow
  - Writes quotes via: custom_entry_price() / custom_exit_price() in strategy
  - Publishes to: pheromone_field ("mm_state")
"""

import os
import sys
import logging
import math
from typing import Dict, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
logger = logging.getLogger("market_maker_mode")

# GLFT parameters
GAMMA_BASE = 0.1        # Base risk aversion
K_BASE = 1.5            # Order arrival rate
A_BASE = 0.1            # Market impact
DELTA_BASE = 0.001      # Tick size as fraction of price

# Bar Portion EMA
BP_EMA_PERIOD = 14

# ADX regime thresholds
ADX_MM_THRESHOLD = 20      # Below: market making
ADX_TREND_THRESHOLD = 25   # Above: trend following
# Between: transitional — reduce MM exposure

# Bybit VIP 0 fees
MAKER_FEE_PCT = 0.020
MIN_PROFITABLE_SPREAD_PCT = MAKER_FEE_PCT * 2.5  # Need 2.5x fee for profitability


class MarketMakerMode:
    """Hormonal market maker with GLFT quoting and BP skew."""

    def __init__(self):
        self._bp_ema: Dict[str, float] = {}
        self._position: Dict[str, float] = {}  # Current inventory per pair
        self._mode: Dict[str, str] = {}  # "mm" | "trend" | "transitional"

    def detect_mode(self, adx: float, pair: str = "") -> str:
        """Detect current regime mode from ADX."""
        if adx < ADX_MM_THRESHOLD:
            mode = "mm"
        elif adx > ADX_TREND_THRESHOLD:
            mode = "trend"
        else:
            mode = "transitional"

        self._mode[pair] = mode
        return mode

    def compute_bar_portion(self, open_price: float, high: float,
                            low: float, close: float) -> float:
        """Compute Bar Portion (BP) = (Close - Open) / (High - Low).

        BP → +1.0 = strong bullish, BP → -1.0 = strong bearish, BP ≈ 0 = doji.
        """
        hl_range = high - low
        if hl_range < 1e-10:
            return 0.0
        return (close - open_price) / hl_range

    def update_bp_ema(self, pair: str, bp: float, period: int = BP_EMA_PERIOD):
        """Update exponential moving average of Bar Portion."""
        alpha = 2.0 / (period + 1)
        if pair not in self._bp_ema:
            self._bp_ema[pair] = bp
        else:
            self._bp_ema[pair] = alpha * bp + (1 - alpha) * self._bp_ema[pair]

    def compute_quotes(self, mid_price: float, volatility: float,
                       pair: str, cortisol: float = 0.5,
                       position: float = 0.0) -> Dict:
        """Compute bid/ask quotes using GLFT formulas with hormonal gamma.

        Args:
            mid_price: Current mid price
            volatility: Recent price volatility (e.g., ATR as fraction)
            pair: Trading pair
            cortisol: Organism stress hormone [0, 1]
            position: Current inventory position

        Returns:
            {"bid": float, "ask": float, "half_spread": float,
             "skew": float, "gamma_effective": float, "mode": str}
        """
        mode = self._mode.get(pair, "trend")

        if mode == "trend":
            # In trending mode, don't provide MM quotes
            return {
                "bid": 0, "ask": 0, "mode": "trend",
                "message": "trending regime — use signal following",
            }

        # Hormonal gamma modulation (Lalor & Swishchuk 2025)
        gamma = GAMMA_BASE * (2.0 - cortisol)  # High stress → wider spread
        k = K_BASE
        delta = DELTA_BASE

        # GLFT formulas (no terminal time)
        if gamma * delta > 0 and k > 0:
            c1 = (1.0 / (gamma * delta)) * math.log(1 + gamma * delta / k)
            inner = gamma / (2 * A_BASE * delta * k)
            exponent = k / (gamma * delta) + 1
            c2 = math.sqrt(inner * ((1 + gamma * delta / k) ** exponent))
        else:
            c1 = 0.001
            c2 = 0.001

        half_spread = c1 + (delta / 2.0) * c2 * volatility

        # Bar Portion directional skew
        bp_ema = self._bp_ema.get(pair, 0.0)
        skew = c2 * volatility * bp_ema

        # Inventory skew (mean-revert inventory)
        alpha_inventory = 0.001 * (2.0 - cortisol)
        inventory_skew = alpha_inventory * position

        # Reservation price
        reservation = mid_price - skew * position - inventory_skew

        # Final quotes
        bid = reservation - half_spread
        ask = reservation + half_spread

        # Ensure minimum profitable spread
        min_spread = mid_price * MIN_PROFITABLE_SPREAD_PCT / 100
        actual_spread = ask - bid
        if actual_spread < min_spread:
            expansion = (min_spread - actual_spread) / 2
            bid -= expansion
            ask += expansion

        # Transitional mode: widen spread
        if mode == "transitional":
            spread_mult = 1.5
            mid = (bid + ask) / 2
            half = (ask - bid) / 2 * spread_mult
            bid = mid - half
            ask = mid + half

        return {
            "bid": round(bid, 8),
            "ask": round(ask, 8),
            "half_spread": round(float(half_spread), 8),
            "half_spread_pct": round(float(half_spread / mid_price * 100), 6),
            "skew": round(float(skew), 8),
            "bp_ema": round(float(bp_ema), 4),
            "gamma_effective": round(float(gamma), 4),
            "reservation_price": round(float(reservation), 8),
            "mode": mode,
            "cortisol": cortisol,
        }

    def update_position(self, pair: str, delta: float):
        """Update inventory position for a pair."""
        self._position[pair] = self._position.get(pair, 0) + delta

    def get_position(self, pair: str) -> float:
        return self._position.get(pair, 0)

    def should_use_mm(self, pair: str, adx: float, spread_pct: float) -> Tuple[bool, str]:
        """Decide if market making should be used for this pair.

        Returns (use_mm: bool, reason: str)
        """
        mode = self.detect_mode(adx, pair)

        if mode == "trend":
            return False, f"ADX={adx:.1f} > {ADX_TREND_THRESHOLD} (trending)"

        if spread_pct < MIN_PROFITABLE_SPREAD_PCT:
            return False, f"spread {spread_pct:.4f}% < min profitable {MIN_PROFITABLE_SPREAD_PCT:.3f}%"

        if mode == "mm":
            return True, f"ADX={adx:.1f} < {ADX_MM_THRESHOLD} (ranging)"

        # Transitional
        return True, f"ADX={adx:.1f} (transitional, reduced exposure)"

    def publish_to_pheromone(self, quotes: Dict, pair: str):
        """Publish MM state to pheromone field."""
        try:
            from pheromone_field import get_pheromone_field
            pfield = get_pheromone_field()
            pfield.deposit("market_maker", "mm_state", {
                "pair": pair,
                "mode": quotes.get("mode", "trend"),
                "half_spread_pct": quotes.get("half_spread_pct", 0),
                "gamma": quotes.get("gamma_effective", 0),
                "bp_ema": quotes.get("bp_ema", 0),
            })
        except Exception:
            pass

    def get_status(self) -> Dict:
        return {
            "modes": dict(self._mode),
            "positions": {k: round(v, 4) for k, v in self._position.items()},
            "bp_emas": {k: round(v, 4) for k, v in self._bp_ema.items()},
        }


# Singleton
_mm_instance = None

def get_market_maker() -> MarketMakerMode:
    global _mm_instance
    if _mm_instance is None:
        _mm_instance = MarketMakerMode()
    return _mm_instance
