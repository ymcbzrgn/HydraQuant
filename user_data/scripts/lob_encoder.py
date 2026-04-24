"""
lob_encoder.py — Phase 26 Sprint 2, Task 11B

Level-2 Order Book Encoder — Perceives market microstructure.

Features extracted:
  - Bid/Ask depth ratio (top 5/10/20 levels)
  - Microprice deviation from midprice
  - Spread percentile (rolling 1h)
  - Book pressure acceleration (d/dt of imbalance)
  - LOB embedding (32-dim) for fusion with other modalities

Integration:
  - Reads from: Bybit WebSocket orderbook data (via freqtrade exchange)
  - Writes to: pheromone_field ("lob_state")
  - Consumed by: multimodal_encoder (10A), market_maker_mode (11D), slippage_forecaster (11E)
"""

import os
import sys
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
from collections import deque

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
logger = logging.getLogger("lob_encoder")

LOB_EMBEDDING_DIM = 32
SPREAD_HISTORY_SIZE = 60  # 1 hour at 1-minute intervals


class LOBEncoder:
    """Level-2 Order Book perception and encoding."""

    def __init__(self):
        self._spread_history: Dict[str, deque] = {}
        self._imbalance_history: Dict[str, deque] = {}

    def encode(self, orderbook: Dict, pair: str = "") -> Dict:
        """Encode orderbook state into features.

        Args:
            orderbook: {"bids": [[price, size], ...], "asks": [[price, size], ...]}

        Returns:
            {"lob_embedding": np.array(32), "imbalance_score": float,
             "spread_regime": str, "microprice": float, ...}
        """
        bids = orderbook.get("bids", [])
        asks = orderbook.get("asks", [])

        if not bids or not asks:
            return self._empty_result()

        # Parse levels
        bid_prices = np.array([float(b[0]) for b in bids[:20]])
        bid_sizes = np.array([float(b[1]) for b in bids[:20]])
        ask_prices = np.array([float(a[0]) for a in asks[:20]])
        ask_sizes = np.array([float(a[1]) for a in asks[:20]])

        best_bid = bid_prices[0]
        best_ask = ask_prices[0]
        mid_price = (best_bid + best_ask) / 2

        # Spread
        spread = best_ask - best_bid
        spread_pct = spread / mid_price * 100 if mid_price > 0 else 0

        # Depth ratios
        bid_depth_5 = bid_sizes[:5].sum()
        ask_depth_5 = ask_sizes[:5].sum()
        bid_depth_10 = bid_sizes[:10].sum()
        ask_depth_10 = ask_sizes[:10].sum()
        bid_depth_20 = bid_sizes.sum()
        ask_depth_20 = ask_sizes.sum()

        depth_ratio_5 = bid_depth_5 / (ask_depth_5 + 1e-10)
        depth_ratio_10 = bid_depth_10 / (ask_depth_10 + 1e-10)
        depth_ratio_20 = bid_depth_20 / (ask_depth_20 + 1e-10)

        # Imbalance score [-1, +1]: positive = buy pressure
        imbalance = (bid_depth_5 - ask_depth_5) / (bid_depth_5 + ask_depth_5 + 1e-10)

        # Microprice
        microprice = (bid_sizes[0] * ask_prices[0] + ask_sizes[0] * bid_prices[0]) / \
                     (bid_sizes[0] + ask_sizes[0] + 1e-10)
        microprice_deviation = (microprice - mid_price) / mid_price * 10000  # in bps

        # Spread percentile (rolling)
        if pair not in self._spread_history:
            self._spread_history[pair] = deque(maxlen=SPREAD_HISTORY_SIZE)
        self._spread_history[pair].append(spread_pct)

        spread_history = list(self._spread_history[pair])
        if len(spread_history) > 5:
            spread_pctile = sum(1 for s in spread_history if s <= spread_pct) / len(spread_history)
        else:
            spread_pctile = 0.5

        # Spread regime
        if spread_pctile > 0.8:
            spread_regime = "wide"
        elif spread_pctile < 0.2:
            spread_regime = "tight"
        else:
            spread_regime = "normal"

        # Book pressure acceleration
        if pair not in self._imbalance_history:
            self._imbalance_history[pair] = deque(maxlen=10)
        self._imbalance_history[pair].append(imbalance)

        imb_hist = list(self._imbalance_history[pair])
        if len(imb_hist) >= 3:
            pressure_accel = imb_hist[-1] - imb_hist[-2]
        else:
            pressure_accel = 0.0

        # LOB Embedding (32-dim feature vector)
        lob_features = np.zeros(LOB_EMBEDDING_DIM, dtype=np.float32)
        lob_features[0] = imbalance
        lob_features[1] = depth_ratio_5 - 1.0  # Centered around 0
        lob_features[2] = depth_ratio_10 - 1.0
        lob_features[3] = depth_ratio_20 - 1.0
        lob_features[4] = microprice_deviation / 10  # Normalized
        lob_features[5] = spread_pct * 100  # Scale up
        lob_features[6] = spread_pctile
        lob_features[7] = pressure_accel
        lob_features[8] = np.log1p(bid_depth_5)
        lob_features[9] = np.log1p(ask_depth_5)
        lob_features[10] = np.log1p(bid_depth_20)
        lob_features[11] = np.log1p(ask_depth_20)

        # Weighted price levels (depth-weighted average distance from mid)
        if len(bid_prices) > 1:
            bid_wavg = np.average(mid_price - bid_prices[:10], weights=bid_sizes[:10] + 1e-10)
            ask_wavg = np.average(ask_prices[:10] - mid_price, weights=ask_sizes[:10] + 1e-10)
            lob_features[12] = bid_wavg / mid_price * 10000
            lob_features[13] = ask_wavg / mid_price * 10000

        return {
            "lob_embedding": lob_features,
            "imbalance_score": round(float(imbalance), 4),
            "depth_ratio_5": round(float(depth_ratio_5), 4),
            "depth_ratio_10": round(float(depth_ratio_10), 4),
            "depth_ratio_20": round(float(depth_ratio_20), 4),
            "microprice": round(float(microprice), 8),
            "microprice_deviation_bps": round(float(microprice_deviation), 2),
            "spread_pct": round(float(spread_pct), 6),
            "spread_percentile": round(float(spread_pctile), 3),
            "spread_regime": spread_regime,
            "pressure_acceleration": round(float(pressure_accel), 4),
            "mid_price": round(float(mid_price), 8),
            "best_bid": round(float(best_bid), 8),
            "best_ask": round(float(best_ask), 8),
        }

    def encode_and_publish(self, orderbook: Dict, pair: str = "") -> Dict:
        """Encode the orderbook AND deposit the result to the pheromone
        field so downstream consumers (regime_classifier spread_regime,
        api_ai dashboard) actually see live LOB state. Task 14 wire-up:
        the old `publish_to_pheromone` had no caller, so `lob_state`
        deposits were always absent and the ILLIQUID regime rule
        (regime_classifier.py step-0) had nothing to read.
        """
        result = self.encode(orderbook, pair=pair)
        try:
            self.publish_to_pheromone(result, pair)
        except Exception:
            pass
        return result

    def _empty_result(self) -> Dict:
        return {
            "lob_embedding": np.zeros(LOB_EMBEDDING_DIM, dtype=np.float32),
            "imbalance_score": 0.0,
            "depth_ratio_5": 1.0,
            "spread_pct": 0.0,
            "spread_regime": "unknown",
            "microprice_deviation_bps": 0.0,
            "pressure_acceleration": 0.0,
        }

    def publish_to_pheromone(self, result: Dict, pair: str):
        """Publish LOB state to pheromone field."""
        try:
            from pheromone_field import get_pheromone_field
            pfield = get_pheromone_field()
            pfield.deposit("lob_encoder", "lob_state", {
                "pair": pair,
                "imbalance": result["imbalance_score"],
                "spread_regime": result["spread_regime"],
                "pressure_accel": result["pressure_acceleration"],
                "microprice_dev": result.get("microprice_deviation_bps", 0),
            })
        except Exception:
            pass


# Singleton
_lob_instance = None

def get_lob_encoder() -> LOBEncoder:
    global _lob_instance
    if _lob_instance is None:
        _lob_instance = LOBEncoder()
    return _lob_instance
