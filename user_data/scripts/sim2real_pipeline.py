"""
sim2real_pipeline.py — Phase 26 Sprint 2, Task 12F

Sim2Real Transfer — Domain randomization for robust deployment.

Adds noise to backtest simulations to close the sim-real gap:
  - Slippage noise: ±0-10 bps random
  - Fee noise: ±20% of nominal fee
  - Spread noise: 1-5x random widening
  - Volume noise: ±30% random scaling
  - Latency noise: 0-500ms random delay effect

This prevents overfitting to perfect-execution backtests.
Models trained with domain randomization are more robust in production.

Integration:
  - Applied during: backtest_label_generator (13B) data generation
  - Applied during: rl_environment (7A) episode generation
  - Consumed by: CatBoost training, IQL pre-training
"""

import os
import sys
import logging
from typing import Dict, Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
logger = logging.getLogger("sim2real_pipeline")

# Randomization ranges
RANDOMIZATION_CONFIG = {
    "slippage_bps": {"min": 0, "max": 10, "distribution": "uniform"},
    "fee_multiplier": {"min": 0.8, "max": 1.2, "distribution": "uniform"},
    "spread_multiplier": {"min": 1.0, "max": 5.0, "distribution": "log_uniform"},
    "volume_multiplier": {"min": 0.7, "max": 1.3, "distribution": "uniform"},
    "latency_effect_pct": {"min": 0, "max": 0.05, "distribution": "exponential"},
}


class Sim2RealPipeline:
    """Domain randomization for sim-to-real transfer."""

    def __init__(self, seed: int = None):
        self._rng = np.random.default_rng(seed)
        self._randomization_level = 1.0  # 0=none, 1=full

    def set_level(self, level: float):
        """Set randomization level (0.0-1.0). 0=no noise, 1=full noise."""
        self._randomization_level = np.clip(level, 0.0, 1.0)

    def randomize_trade(self, trade: Dict) -> Dict:
        """Apply domain randomization to a backtest trade.

        Modifies PnL to account for realistic execution noise.
        """
        if self._randomization_level == 0:
            return trade

        randomized = trade.copy()
        pnl = trade.get("profit_ratio", 0) * 100  # Convert to percentage

        # 1. Slippage noise (always negative impact)
        slippage_bps = self._rng.uniform(
            RANDOMIZATION_CONFIG["slippage_bps"]["min"],
            RANDOMIZATION_CONFIG["slippage_bps"]["max"],
        ) * self._randomization_level
        slippage_pct = slippage_bps / 100  # bps to percentage
        pnl -= slippage_pct

        # 2. Fee noise
        fee_mult = self._rng.uniform(
            RANDOMIZATION_CONFIG["fee_multiplier"]["min"],
            RANDOMIZATION_CONFIG["fee_multiplier"]["max"],
        )
        nominal_fee = 0.04  # Typical round-trip fee (0.02% × 2)
        fee_noise = nominal_fee * (fee_mult - 1.0) * self._randomization_level
        pnl -= fee_noise

        # 3. Spread noise (mostly negative, worse fills)
        spread_mult = self._rng.uniform(1.0, RANDOMIZATION_CONFIG["spread_multiplier"]["max"])
        spread_impact = 0.01 * (spread_mult - 1.0) * self._randomization_level
        pnl -= spread_impact

        # 4. Latency effect (small random price movement during execution)
        latency_pct = self._rng.exponential(0.01) * self._randomization_level
        latency_direction = self._rng.choice([-1, 1])
        pnl += latency_direction * min(latency_pct, 0.05)

        randomized["profit_ratio"] = pnl / 100  # Back to ratio
        randomized["randomized"] = True
        randomized["noise_applied"] = {
            "slippage_bps": round(slippage_bps, 2),
            "fee_mult": round(fee_mult, 3),
            "spread_mult": round(spread_mult, 3),
            "latency_pct": round(latency_pct, 4),
        }

        return randomized

    def randomize_state(self, state: np.ndarray) -> np.ndarray:
        """Apply observation noise to RL state vector.

        Simulates sensor noise in perception modules.
        """
        if self._randomization_level == 0:
            return state

        noise_scale = 0.01 * self._randomization_level
        noise = self._rng.normal(0, noise_scale, size=state.shape).astype(np.float32)
        return state + noise

    def randomize_reward(self, reward: float) -> float:
        """Apply noise to RL reward signal."""
        if self._randomization_level == 0:
            return reward

        noise = self._rng.normal(0, abs(reward) * 0.1 * self._randomization_level)
        return reward + noise

    def get_curriculum_level(self, n_episodes: int) -> float:
        """Get curriculum-based randomization level.

        Starts with no noise, gradually increases:
          0-100 episodes: level 0.0 (clean)
          100-500: level 0.3 (mild)
          500-2000: level 0.6 (moderate)
          2000+: level 1.0 (full)
        """
        if n_episodes < 100:
            return 0.0
        elif n_episodes < 500:
            return 0.3
        elif n_episodes < 2000:
            return 0.6
        else:
            return 1.0


# Singleton
_sim2real_instance = None

def get_sim2real(seed: int = None) -> Sim2RealPipeline:
    global _sim2real_instance
    if _sim2real_instance is None:
        _sim2real_instance = Sim2RealPipeline(seed=seed)
    return _sim2real_instance
