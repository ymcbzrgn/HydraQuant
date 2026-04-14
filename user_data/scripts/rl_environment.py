"""
rl_environment.py — Phase 26 Sprint 2, Task 7A

Custom Gymnasium environment for trading parameter optimization.

State space (335 dim):
  - TTM embedding: 64 dims
  - Chart features: 193 dims
  - Chronos quantiles: 3 dims (P10, P50, P90)
  - Deep ensemble: 1 dim (variance)
  - Conformal: 2 dims (interval width, coverage)
  - OOD: 1 dim (distance)
  - Hormones: 4 dims (cortisol, dopamine, serotonin, adrenaline)
  - Neuron params: ~50 dims (top neuron values)
  - F&G: 1 dim
  - Funding rate: 1 dim
  - Regime: 6 dims (one-hot)
  - Crowd: 3 dims (long_short_ratio, open_interest_norm, social)
  - Evidence sub-scores: 6 dims

Action space (continuous):
  - sizing_multiplier: [0.1, 3.0]
  - confidence_threshold: [0.2, 0.9]
  - stoploss_multiplier: [0.5, 5.0]
  - leverage: [1.0, 5.0]

Reward:
  pnl - λ_dd * drawdown - λ_risk * var_breach + λ_sharpe * Δsharpe

Usage:
    env = TradingEnv()
    obs, info = env.reset()
    action = agent.predict(obs)
    obs, reward, done, truncated, info = env.step(action)
"""

import os
import sys
import json
import logging
from typing import Dict, List, Tuple, Optional, Any

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
logger = logging.getLogger("rl_environment")

try:
    import gymnasium as gym
    from gymnasium import spaces
    HAS_GYMNASIUM = True
except ImportError:
    try:
        import gym
        from gym import spaces
        HAS_GYMNASIUM = True
    except ImportError:
        HAS_GYMNASIUM = False
        logger.warning("[RLEnv] gymnasium/gym not installed — env will not work at runtime")
        # Provide stubs for import-time safety
        class spaces:
            @staticmethod
            def Box(*a, **kw): return None
            @staticmethod
            def Dict(*a, **kw): return None

from ai_config import AI_DB_PATH
from db import get_db_connection, init_db

# ===========================================================================
# Constants
# ===========================================================================

# State dimensions
DIM_TTM = 64
DIM_CHART = 193
DIM_CHRONOS = 3
DIM_ENSEMBLE = 1
DIM_CONFORMAL = 2
DIM_OOD = 1
DIM_HORMONES = 4
DIM_NEURONS = 50
DIM_FNG = 1
DIM_FUNDING = 1
DIM_REGIME = 6
DIM_CROWD = 3
DIM_EVIDENCE = 6

TOTAL_STATE_DIM = (DIM_TTM + DIM_CHART + DIM_CHRONOS + DIM_ENSEMBLE +
                   DIM_CONFORMAL + DIM_OOD + DIM_HORMONES + DIM_NEURONS +
                   DIM_FNG + DIM_FUNDING + DIM_REGIME + DIM_CROWD + DIM_EVIDENCE)

# Action dimensions
DIM_ACTION = 4  # sizing_mult, confidence_thresh, stoploss_mult, leverage

# Reward coefficients
LAMBDA_DD = 2.0      # Drawdown penalty
LAMBDA_RISK = 1.5    # VaR breach penalty
LAMBDA_SHARPE = 0.5  # Sharpe improvement bonus

# Regime one-hot mapping
REGIME_MAP = {
    "trending_bull": 0, "trending_bear": 1, "ranging": 2,
    "transitional": 3, "high_volatility": 4, "unknown": 5,
}

# Safe RL constraints (CMDP)
CONSTRAINTS = {
    "max_drawdown": 0.25,
    "max_single_position": 0.03,
    "portfolio_heat": 0.10,
}


class TradingEnv:
    """Custom trading environment for RL parameter optimization.

    Compatible with gymnasium API but does not require gym as hard dependency.
    Can be wrapped with gym.make() if gymnasium is available.
    """

    def __init__(self, db_path: str = AI_DB_PATH,
                 episode_length: int = 100,
                 initial_balance: float = 10000.0):
        self.db_path = db_path
        self.episode_length = episode_length
        self.initial_balance = initial_balance

        # Action bounds (always available, regardless of gym)
        self._action_low = np.array([0.1, 0.2, 0.5, 1.0], dtype=np.float32)
        self._action_high = np.array([3.0, 0.9, 5.0, 5.0], dtype=np.float32)

        # State and action spaces (gym-compatible if available)
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0,
            shape=(TOTAL_STATE_DIM,),
            dtype=np.float32,
        ) if HAS_GYMNASIUM else None
        self.action_space = spaces.Box(
            low=self._action_low,
            high=self._action_high,
            dtype=np.float32,
        ) if HAS_GYMNASIUM else None

        # Episode state
        self._step = 0
        self._balance = initial_balance
        self._peak_balance = initial_balance
        self._pnl_history = []
        self._sharpe_prev = 0.0
        self._trade_data = None
        self._current_idx = 0

        # Load historical data for replay
        self._episodes_data = None

        init_db()

    # ===================================================================
    # Data Loading
    # ===================================================================

    def _load_episode_data(self) -> Optional[List[Dict]]:
        """Load historical trade data for environment replay."""
        if self._episodes_data is not None:
            return self._episodes_data

        conn = get_db_connection(self.db_path)
        try:
            rows = conn.execute("""
                SELECT d.pair, d.signal_type, d.confidence, d.regime,
                       d.outcome_pnl, d.outcome_duration, d.timestamp,
                       d.trust_score_at_decision, d.position_size,
                       e.sub_scores_json, e.max_confidence_cap
                FROM ai_decisions d
                LEFT JOIN evidence_audit_log e
                    ON d.pair = e.pair
                    AND ABS(JULIANDAY(d.timestamp) - JULIANDAY(e.timestamp)) < 0.01
                WHERE d.outcome_pnl IS NOT NULL
                ORDER BY d.timestamp ASC
            """).fetchall()

            if not rows:
                return None

            data = []
            for row in rows:
                sub_scores = {}
                if row["sub_scores_json"]:
                    try:
                        sub_scores = json.loads(row["sub_scores_json"])
                    except Exception:
                        pass

                data.append({
                    "pair": row["pair"],
                    "signal_type": row["signal_type"],
                    "confidence": row["confidence"] or 0.5,
                    "regime": row["regime"] or "transitional",
                    "outcome_pnl": row["outcome_pnl"],
                    "trust_score": row["trust_score_at_decision"] or 0.5,
                    "position_size": row["position_size"] or 1.0,
                    "max_cap": row["max_confidence_cap"] or 0.35,
                    "sub_trend": sub_scores.get("trend", sub_scores.get("technical", 0.5)),
                    "sub_momentum": sub_scores.get("momentum", sub_scores.get("mom", 0.5)),
                    "sub_crowd": sub_scores.get("crowd", sub_scores.get("sentiment", 0.5)),
                    "sub_risk": sub_scores.get("risk", 0.5),
                    "sub_macro": sub_scores.get("macro", 0.5),
                    "sub_evidence": sub_scores.get("evidence", sub_scores.get("evid", 0.5)),
                })

            self._episodes_data = data
            logger.info(f"[RLEnv] Loaded {len(data)} historical trades for replay")
            return data

        finally:
            conn.close()

    # ===================================================================
    # State Construction
    # ===================================================================

    def _build_state(self, trade_data: Dict) -> np.ndarray:
        """Build 335-dimensional state vector from trade data.

        In production, this pulls from live perception modules.
        In training (replay), we reconstruct from stored data.
        """
        state = np.zeros(TOTAL_STATE_DIM, dtype=np.float32)
        offset = 0

        # TTM embedding (64d) — zeros in replay, live from triple_perception
        offset += DIM_TTM

        # Chart features (193d) — zeros in replay, live from chart_features
        offset += DIM_CHART

        # Chronos quantiles (3d)
        offset += DIM_CHRONOS

        # Ensemble variance (1d)
        offset += DIM_ENSEMBLE

        # Conformal (2d) — interval width, coverage
        offset += DIM_CONFORMAL

        # OOD distance (1d)
        offset += DIM_OOD

        # Hormones (4d)
        state[offset] = 0.5     # cortisol (default)
        state[offset + 1] = 0.5  # dopamine
        state[offset + 2] = 0.5  # serotonin
        state[offset + 3] = 0.5  # adrenaline
        offset += DIM_HORMONES

        # Neuron params (50d) — zeros in replay
        offset += DIM_NEURONS

        # Fear & Greed (1d) — normalized 0-1
        state[offset] = 0.5
        offset += DIM_FNG

        # Funding rate (1d)
        state[offset] = 0.0
        offset += DIM_FUNDING

        # Regime one-hot (6d)
        regime = trade_data.get("regime", "transitional")
        regime_idx = REGIME_MAP.get(regime, 3)
        state[offset + regime_idx] = 1.0
        offset += DIM_REGIME

        # Crowd (3d) — long_short_ratio, OI, social
        state[offset] = trade_data.get("sub_crowd", 0.5)
        offset += DIM_CROWD

        # Evidence sub-scores (6d)
        state[offset] = trade_data.get("sub_trend", 0.5)
        state[offset + 1] = trade_data.get("sub_momentum", 0.5)
        state[offset + 2] = trade_data.get("sub_crowd", 0.5)
        state[offset + 3] = trade_data.get("sub_risk", 0.5)
        state[offset + 4] = trade_data.get("sub_macro", 0.5)
        state[offset + 5] = trade_data.get("sub_evidence", 0.5)
        offset += DIM_EVIDENCE

        return state

    def build_live_state(self, perception_result: Dict,
                         chart_feats: Dict[str, float],
                         hormones: Dict[str, float],
                         fng_value: float,
                         funding_rate: float,
                         regime: str,
                         evidence_scores: Dict[str, float],
                         neuron_params: Optional[np.ndarray] = None) -> np.ndarray:
        """Build state vector from live perception modules.

        This is the production API — called during live trading.
        """
        state = np.zeros(TOTAL_STATE_DIM, dtype=np.float32)
        offset = 0

        # TTM embedding (64d)
        ttm = perception_result.get("ttm_embedding")
        if ttm is not None:
            ttm_arr = np.array(ttm, dtype=np.float32)[:DIM_TTM]
            state[offset:offset + len(ttm_arr)] = ttm_arr
        offset += DIM_TTM

        # Chart features (193d)
        if chart_feats:
            feat_keys = sorted(chart_feats.keys())
            for i, k in enumerate(feat_keys[:DIM_CHART]):
                val = chart_feats[k]
                if val is not None and np.isfinite(val):
                    state[offset + i] = float(val)
        offset += DIM_CHART

        # Chronos quantiles (3d)
        state[offset] = perception_result.get("chronos_p10", 0.0)
        state[offset + 1] = perception_result.get("chronos_p50", 0.0)
        state[offset + 2] = perception_result.get("chronos_p90", 0.0)
        offset += DIM_CHRONOS

        # Ensemble variance (1d)
        state[offset] = perception_result.get("ensemble_variance", 0.0)
        offset += DIM_ENSEMBLE

        # Conformal (2d)
        state[offset] = perception_result.get("chronos_interval_width", 0.0)
        state[offset + 1] = perception_result.get("conformal_coverage", 0.9)
        offset += DIM_CONFORMAL

        # OOD (1d)
        state[offset] = perception_result.get("ood_distance", 0.0)
        offset += DIM_OOD

        # Hormones (4d)
        state[offset] = hormones.get("cortisol", 0.5)
        state[offset + 1] = hormones.get("dopamine", 0.5)
        state[offset + 2] = hormones.get("serotonin", 0.5)
        state[offset + 3] = hormones.get("adrenaline", 0.5)
        offset += DIM_HORMONES

        # Neuron params (50d)
        if neuron_params is not None:
            n = min(len(neuron_params), DIM_NEURONS)
            state[offset:offset + n] = neuron_params[:n]
        offset += DIM_NEURONS

        # F&G (1d)
        state[offset] = fng_value / 100.0 if fng_value > 1 else fng_value
        offset += DIM_FNG

        # Funding rate (1d)
        state[offset] = np.clip(funding_rate * 1000, -5, 5)  # Scale funding rate
        offset += DIM_FUNDING

        # Regime one-hot (6d)
        regime_idx = REGIME_MAP.get(regime, 3)
        state[offset + regime_idx] = 1.0
        offset += DIM_REGIME

        # Crowd (3d)
        state[offset] = evidence_scores.get("crowd", 0.5)
        state[offset + 1] = evidence_scores.get("open_interest_norm", 0.0)
        state[offset + 2] = evidence_scores.get("social", 0.5)
        offset += DIM_CROWD

        # Evidence sub-scores (6d)
        state[offset] = evidence_scores.get("trend", evidence_scores.get("technical", 0.5))
        state[offset + 1] = evidence_scores.get("momentum", evidence_scores.get("mom", 0.5))
        state[offset + 2] = evidence_scores.get("crowd", evidence_scores.get("sentiment", 0.5))
        state[offset + 3] = evidence_scores.get("risk", 0.5)
        state[offset + 4] = evidence_scores.get("macro", 0.5)
        state[offset + 5] = evidence_scores.get("evidence", evidence_scores.get("evid", 0.5))
        offset += DIM_EVIDENCE

        return state

    # ===================================================================
    # Reward Computation
    # ===================================================================

    def _compute_reward(self, pnl: float, action: np.ndarray) -> float:
        """Compute shaped reward with risk penalties.

        reward = pnl - λ_dd * drawdown - λ_risk * var_breach + λ_sharpe * Δsharpe
        """
        # Track balance
        self._balance += pnl
        self._peak_balance = max(self._peak_balance, self._balance)
        self._pnl_history.append(pnl)

        # Drawdown penalty
        drawdown = (self._peak_balance - self._balance) / (self._peak_balance + 1e-10)
        dd_penalty = LAMBDA_DD * max(0, drawdown - 0.05)  # Penalty above 5% DD

        # VaR breach penalty (position too large)
        sizing_mult = action[0]
        if sizing_mult > 2.0:
            risk_penalty = LAMBDA_RISK * (sizing_mult - 2.0)
        else:
            risk_penalty = 0.0

        # Sharpe improvement bonus
        if len(self._pnl_history) >= 10:
            returns = np.array(self._pnl_history[-20:])
            sharpe = returns.mean() / (returns.std() + 1e-10) * np.sqrt(252)
            sharpe_delta = sharpe - self._sharpe_prev
            self._sharpe_prev = sharpe
            sharpe_bonus = LAMBDA_SHARPE * max(0, sharpe_delta)
        else:
            sharpe_bonus = 0.0

        reward = pnl - dd_penalty - risk_penalty + sharpe_bonus

        return float(reward)

    # ===================================================================
    # Gymnasium API
    # ===================================================================

    def reset(self, seed=None) -> Tuple[np.ndarray, Dict]:
        """Reset environment to start a new episode."""
        self._step = 0
        self._balance = self.initial_balance
        self._peak_balance = self.initial_balance
        self._pnl_history = []
        self._sharpe_prev = 0.0

        # Load data if not loaded
        data = self._load_episode_data()
        if data and len(data) > self.episode_length:
            # Random starting point for episode
            rng = np.random.default_rng(seed)
            max_start = len(data) - self.episode_length
            self._current_idx = rng.integers(0, max_start)
        else:
            self._current_idx = 0

        trade = data[self._current_idx] if data else {}
        obs = self._build_state(trade)
        info = {"pair": trade.get("pair", ""), "step": 0}

        return obs, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step: apply action, get reward, advance to next trade.

        Action: [sizing_mult, confidence_thresh, stoploss_mult, leverage]
        """
        # Clip action to valid range
        action = np.clip(action, self._action_low, self._action_high)
        sizing_mult = action[0]
        conf_thresh = action[1]
        stoploss_mult = action[2]
        leverage = action[3]

        data = self._load_episode_data()
        if not data:
            return np.zeros(TOTAL_STATE_DIM, dtype=np.float32), 0.0, True, False, {}

        # Get current trade
        trade_idx = self._current_idx + self._step
        if trade_idx >= len(data):
            return np.zeros(TOTAL_STATE_DIM, dtype=np.float32), 0.0, True, False, {}

        trade = data[trade_idx]
        original_pnl = trade["outcome_pnl"]
        original_conf = trade["confidence"]

        # Apply action effects to PnL
        # 1. If confidence below threshold, skip trade (PnL = 0)
        if original_conf < conf_thresh:
            modified_pnl = 0.0
            trade_taken = False
        else:
            # 2. Scale PnL by sizing multiplier and leverage
            modified_pnl = original_pnl * sizing_mult * leverage

            # 3. Apply stoploss cap
            stoploss_pct = stoploss_mult * 1.0  # Base stoploss is 1%
            if modified_pnl < 0 and abs(modified_pnl) > stoploss_pct:
                modified_pnl = -stoploss_pct

            trade_taken = True

        # Compute reward
        reward = self._compute_reward(modified_pnl, action)

        # Advance step
        self._step += 1
        done = self._step >= self.episode_length
        truncated = (trade_idx + 1) >= len(data)

        # Next observation
        if not done and not truncated:
            next_trade = data[trade_idx + 1]
            next_obs = self._build_state(next_trade)
        else:
            next_obs = np.zeros(TOTAL_STATE_DIM, dtype=np.float32)

        info = {
            "pair": trade["pair"],
            "step": self._step,
            "original_pnl": original_pnl,
            "modified_pnl": modified_pnl,
            "trade_taken": trade_taken,
            "balance": self._balance,
            "drawdown": (self._peak_balance - self._balance) / (self._peak_balance + 1e-10),
            "action": {
                "sizing_mult": float(sizing_mult),
                "confidence_thresh": float(conf_thresh),
                "stoploss_mult": float(stoploss_mult),
                "leverage": float(leverage),
            },
            "constraint_violations": self._check_constraints(action),
        }

        return next_obs, reward, done, truncated, info

    def _check_constraints(self, action: np.ndarray) -> Dict[str, bool]:
        """Check CMDP safety constraints."""
        drawdown = (self._peak_balance - self._balance) / (self._peak_balance + 1e-10)
        sizing = action[0] * action[3]  # sizing_mult * leverage

        return {
            "max_drawdown_violated": drawdown > CONSTRAINTS["max_drawdown"],
            "max_position_violated": sizing > CONSTRAINTS["max_single_position"] * 100,
            "portfolio_heat_violated": False,  # Requires portfolio-level tracking
        }

    # ===================================================================
    # Episode Recording (for DuckDB replay buffer)
    # ===================================================================

    def record_episode(self, episode_id: int, states: List[np.ndarray],
                       actions: List[np.ndarray], rewards: List[float],
                       source: str = "backtest") -> int:
        """Record a completed episode to DuckDB replay buffer.

        Returns number of steps recorded.
        """
        try:
            from analytics_engine import get_analytics
            engine = get_analytics()

            for step_idx in range(len(states)):
                next_state = states[step_idx + 1] if step_idx + 1 < len(states) else None
                done = step_idx == len(states) - 1

                engine.insert_episode(
                    episode_id=episode_id,
                    step=step_idx,
                    state=states[step_idx].tobytes(),
                    action=actions[step_idx].tobytes(),
                    reward=rewards[step_idx],
                    next_state=next_state.tobytes() if next_state is not None else None,
                    done=done,
                    source=source,
                )

            return len(states)

        except Exception as e:
            logger.error(f"[RLEnv] Failed to record episode: {e}")
            return 0

    # ===================================================================
    # Utility
    # ===================================================================

    def get_state_dim(self) -> int:
        return TOTAL_STATE_DIM

    def get_action_dim(self) -> int:
        return DIM_ACTION

    def get_action_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._action_low, self._action_high


# ===========================================================================
# Convenience: generate episodes from backtest data
# ===========================================================================

def generate_offline_episodes(n_episodes: int = 100,
                               episode_length: int = 50,
                               source: str = "backtest") -> int:
    """Generate offline episodes for IQL pre-training.

    Uses random actions to explore the action space.
    Records to DuckDB replay buffer.
    """
    env = TradingEnv(episode_length=episode_length)
    total_steps = 0

    for ep_id in range(n_episodes):
        obs, info = env.reset(seed=ep_id)

        states = [obs]
        actions = []
        rewards = []

        for _ in range(episode_length):
            # Random action (exploration)
            action = np.random.uniform(
                env._action_low,
                env._action_high,
            ).astype(np.float32)

            next_obs, reward, done, truncated, info = env.step(action)

            actions.append(action)
            rewards.append(reward)
            states.append(next_obs)

            if done or truncated:
                break

        recorded = env.record_episode(ep_id, states[:-1], actions, rewards, source=source)
        total_steps += recorded

        if (ep_id + 1) % 10 == 0:
            logger.info(f"[RLEnv] Generated {ep_id + 1}/{n_episodes} episodes "
                       f"({total_steps} total steps)")

    logger.info(f"[RLEnv] Generated {n_episodes} episodes, {total_steps} total steps")
    return total_steps
