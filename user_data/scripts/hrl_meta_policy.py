"""
hrl_meta_policy.py — Phase 26 Sprint 2, Task 7D

Hierarchical RL Meta-Policy using PPO.

The meta-policy decides WHICH organ agent should be activated:
  - Organ 1: Sizing Agent (SAC) — adjusts position sizing params
  - Organ 2: Confidence Agent (SAC) — adjusts confidence thresholds
  - Organ 3: Defense Agent (SAC) — adjusts stoploss/risk params
  - Organ 4: Timing Agent (SAC) — adjusts entry/exit timing
  - Organ 5: Memory Agent (SAC) — adjusts learning rate/memory params

Meta-observation: aggregated state + organ performance metrics
Meta-action: discrete selection of which organ to activate + resource allocation

Inspired by Hi-DARTS (2025): 25.17% return, 0.75 Sharpe.

Usage:
    meta = HRLMetaPolicy()
    organ_idx, budget = meta.select_organ(state, organ_performances)
    action = organ_agents[organ_idx].predict(state)

Reference: Pateria et al. "Hierarchical RL: A Survey" (ACM 2021)
"""

import os
import sys
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
logger = logging.getLogger("hrl_meta_policy")

from ai_config import AI_DB_PATH
from db import execute_with_retry, init_db

# ===========================================================================
# Constants
# ===========================================================================

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "models", "rl")
HRL_MODEL_PATH = os.path.join(MODEL_DIR, "hrl_meta_v1.pt")

# Organ definitions
ORGANS = [
    {"name": "sizing", "description": "Position sizing optimization",
     "action_indices": [0], "trigger": "pnl_deviation"},
    {"name": "confidence", "description": "Confidence threshold tuning",
     "action_indices": [1], "trigger": "signal_quality_change"},
    {"name": "defense", "description": "Stoploss and risk parameters",
     "action_indices": [2], "trigger": "drawdown_increase"},
    {"name": "timing", "description": "Entry/exit timing optimization",
     "action_indices": [3], "trigger": "hour_performance_change"},
    {"name": "memory", "description": "Learning rate and memory params",
     "action_indices": [], "trigger": "regime_change"},
]

N_ORGANS = len(ORGANS)

# Meta-state: organ performance (5 organs × 4 metrics) + market context (10)
META_STATE_DIM = N_ORGANS * 4 + 10
META_ACTION_DIM = N_ORGANS  # Softmax over organs (which to activate)

HRL_CONFIG = {
    "hidden_dim": 128,
    "learning_rate": 1e-3,
    "discount": 0.99,
    "gae_lambda": 0.95,
    "clip_ratio": 0.2,     # PPO clip
    "value_coef": 0.5,
    "entropy_coef": 0.01,  # Encourage exploration of organ selection
    "max_grad_norm": 0.5,
    "n_update_epochs": 4,
    "batch_size": 32,
    "rollout_length": 64,
}


class OrganPerformanceTracker:
    """Track performance metrics for each organ agent."""

    def __init__(self):
        self._metrics = {organ["name"]: {
            "recent_pnl": [],
            "avg_reward": 0.0,
            "activation_count": 0,
            "last_activated": None,
            "win_rate": 0.5,
        } for organ in ORGANS}

    def update(self, organ_name: str, reward: float, win: bool):
        """Update organ performance after its action is applied."""
        if organ_name not in self._metrics:
            return

        m = self._metrics[organ_name]
        m["recent_pnl"].append(reward)
        if len(m["recent_pnl"]) > 50:
            m["recent_pnl"] = m["recent_pnl"][-50:]

        m["avg_reward"] = np.mean(m["recent_pnl"])
        m["activation_count"] += 1
        m["last_activated"] = datetime.utcnow().isoformat()
        m["win_rate"] = (m["win_rate"] * 0.95 + (1.0 if win else 0.0) * 0.05)

    def get_state_vector(self) -> np.ndarray:
        """Get organ performance as state vector (N_ORGANS × 4)."""
        state = np.zeros(N_ORGANS * 4, dtype=np.float32)
        for i, organ in enumerate(ORGANS):
            m = self._metrics.get(organ["name"], {})
            base = i * 4
            state[base] = m.get("avg_reward", 0.0)
            state[base + 1] = m.get("win_rate", 0.5)
            state[base + 2] = min(m.get("activation_count", 0) / 100, 1.0)
            state[base + 3] = len(m.get("recent_pnl", [])) / 50
        return state

    def get_metrics(self) -> Dict:
        return {k: {kk: vv for kk, vv in v.items() if kk != "recent_pnl"}
                for k, v in self._metrics.items()}


class HRLMetaPolicy:
    """PPO-based hierarchical meta-policy for organ selection."""

    def __init__(self, config: Dict = None):
        self.config = config or HRL_CONFIG
        self._initialized = False
        self.organ_tracker = OrganPerformanceTracker()

        # Rollout buffer
        self._rollout = {
            "states": [], "actions": [], "rewards": [],
            "values": [], "log_probs": [], "dones": [],
        }

        init_db()

    def _init_networks(self):
        """Initialize PPO actor-critic networks."""
        try:
            import torch
            import torch.nn as nn
        except ImportError:
            logger.error("[HRL] PyTorch not installed")
            return False

        hidden = self.config["hidden_dim"]

        # Actor: state → organ probabilities (softmax)
        self.actor = nn.Sequential(
            nn.Linear(META_STATE_DIM, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, N_ORGANS),
        )

        # Critic: state → value
        self.critic = nn.Sequential(
            nn.Linear(META_STATE_DIM, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

        import torch.optim as optim
        self.optimizer = optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=self.config["learning_rate"],
        )

        self._initialized = True
        return True

    # ===================================================================
    # Organ Selection API
    # ===================================================================

    def build_meta_state(self, market_context: Dict[str, float]) -> np.ndarray:
        """Build meta-state from organ performances + market context."""
        state = np.zeros(META_STATE_DIM, dtype=np.float32)

        # Organ performances (N_ORGANS × 4)
        organ_state = self.organ_tracker.get_state_vector()
        state[:len(organ_state)] = organ_state

        # Market context (10 dims)
        offset = N_ORGANS * 4
        ctx_keys = ["drawdown", "sharpe_recent", "volatility", "regime_stability",
                     "fng_value", "pnl_deviation", "signal_quality",
                     "hour_performance", "trade_frequency", "confidence_avg"]
        for i, key in enumerate(ctx_keys):
            if i + offset < META_STATE_DIM:
                state[offset + i] = market_context.get(key, 0.0)

        return state

    def select_organ(self, meta_state: np.ndarray,
                     deterministic: bool = False) -> Tuple[int, float]:
        """Select which organ to activate.

        Returns: (organ_index, confidence_score)
        """
        import torch

        if not self._initialized:
            if not self._init_networks():
                # Fallback: round-robin
                return 0, 0.5

        with torch.no_grad():
            s = torch.FloatTensor(meta_state).unsqueeze(0)
            logits = self.actor(s)
            probs = torch.softmax(logits, dim=1)[0]

            if deterministic:
                organ_idx = probs.argmax().item()
            else:
                dist = torch.distributions.Categorical(probs)
                organ_idx = dist.sample().item()

            confidence = probs[organ_idx].item()

        return organ_idx, confidence

    def select_organ_with_trigger(self, meta_state: np.ndarray,
                                   triggers: Dict[str, float]) -> Tuple[int, float]:
        """Select organ based on both PPO policy and trigger signals.

        Triggers override PPO when their signal is strong enough.
        """
        # Check triggers first
        trigger_scores = {}
        for i, organ in enumerate(ORGANS):
            trigger_key = organ["trigger"]
            trigger_val = triggers.get(trigger_key, 0.0)
            trigger_scores[i] = trigger_val

        # If any trigger is very strong (> 0.8), use it directly
        max_trigger_idx = max(trigger_scores, key=trigger_scores.get)
        if trigger_scores[max_trigger_idx] > 0.8:
            return max_trigger_idx, trigger_scores[max_trigger_idx]

        # Otherwise, use PPO policy
        return self.select_organ(meta_state)

    # ===================================================================
    # PPO Training
    # ===================================================================

    def add_experience(self, state: np.ndarray, action: int,
                       reward: float, value: float, log_prob: float,
                       done: bool):
        """Add experience to rollout buffer."""
        self._rollout["states"].append(state)
        self._rollout["actions"].append(action)
        self._rollout["rewards"].append(reward)
        self._rollout["values"].append(value)
        self._rollout["log_probs"].append(log_prob)
        self._rollout["dones"].append(done)

        if len(self._rollout["states"]) >= self.config["rollout_length"]:
            self.train_ppo()

    def train_ppo(self) -> Optional[Dict[str, float]]:
        """Train PPO on collected rollout."""
        import torch
        import torch.nn.functional as F

        if not self._initialized or len(self._rollout["states"]) < 16:
            self._rollout = {k: [] for k in self._rollout}
            return None

        # Convert to tensors
        states = torch.FloatTensor(np.array(self._rollout["states"]))
        actions = torch.LongTensor(self._rollout["actions"])
        rewards = torch.FloatTensor(self._rollout["rewards"])
        old_values = torch.FloatTensor(self._rollout["values"])
        old_log_probs = torch.FloatTensor(self._rollout["log_probs"])
        dones = torch.FloatTensor(self._rollout["dones"])

        T = len(rewards)
        discount = self.config["discount"]
        gae_lambda = self.config["gae_lambda"]

        # GAE (Generalized Advantage Estimation)
        advantages = torch.zeros(T)
        gae = 0
        for t in reversed(range(T)):
            if t == T - 1:
                next_value = 0
            else:
                next_value = old_values[t + 1]
            delta = rewards[t] + discount * (1 - dones[t]) * next_value - old_values[t]
            gae = delta + discount * gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae

        returns = advantages + old_values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO update epochs
        metrics = {"policy_loss": 0, "value_loss": 0, "entropy": 0}

        for _ in range(self.config["n_update_epochs"]):
            logits = self.actor(states)
            probs = torch.softmax(logits, dim=1)
            dist = torch.distributions.Categorical(probs)
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()

            values = self.critic(states).squeeze()

            # PPO clipped objective
            ratio = torch.exp(new_log_probs - old_log_probs)
            clip = self.config["clip_ratio"]
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - clip, 1 + clip) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = F.mse_loss(values, returns)

            loss = (policy_loss +
                    self.config["value_coef"] * value_loss -
                    self.config["entropy_coef"] * entropy)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.actor.parameters()) + list(self.critic.parameters()),
                self.config["max_grad_norm"]
            )
            self.optimizer.step()

            metrics["policy_loss"] += policy_loss.item()
            metrics["value_loss"] += value_loss.item()
            metrics["entropy"] += entropy.item()

        n_epochs = self.config["n_update_epochs"]
        metrics = {k: v / n_epochs for k, v in metrics.items()}

        # Clear rollout
        self._rollout = {k: [] for k in self._rollout}

        logger.debug(f"[HRL] PPO update: policy_loss={metrics['policy_loss']:.4f}, "
                    f"value_loss={metrics['value_loss']:.4f}, "
                    f"entropy={metrics['entropy']:.4f}")

        return metrics

    # ===================================================================
    # Save / Load
    # ===================================================================

    def save(self, path: str = None):
        import torch
        path = path or HRL_MODEL_PATH
        os.makedirs(os.path.dirname(path), exist_ok=True)

        checkpoint = {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "config": self.config,
            "organ_metrics": self.organ_tracker.get_metrics(),
            "saved_at": datetime.utcnow().isoformat(),
        }
        torch.save(checkpoint, path)
        logger.info(f"[HRL] Model saved: {path}")

    def load(self, path: str = None) -> bool:
        import torch
        path = path or HRL_MODEL_PATH
        if not os.path.exists(path):
            return False

        try:
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            self._init_networks()
            self.actor.load_state_dict(checkpoint["actor"])
            self.critic.load_state_dict(checkpoint["critic"])
            logger.info(f"[HRL] Model loaded: {path}")
            return True
        except Exception as e:
            logger.error(f"[HRL] Failed to load: {e}")
            return False

    def get_organ_info(self, organ_idx: int) -> Dict:
        """Get info about a specific organ."""
        if 0 <= organ_idx < N_ORGANS:
            return ORGANS[organ_idx]
        return {}

    def print_status(self):
        """Print organ performance status."""
        print("\n" + "=" * 60)
        print("  HRL META-POLICY STATUS")
        print("=" * 60)
        metrics = self.organ_tracker.get_metrics()
        print(f"\n  {'Organ':<15} | {'Avg Reward':>10} | {'Win Rate':>8} | {'Activations':>11}")
        print("  " + "-" * 55)
        for organ in ORGANS:
            m = metrics.get(organ["name"], {})
            print(f"  {organ['name']:<15} | "
                  f"{m.get('avg_reward', 0):>10.4f} | "
                  f"{m.get('win_rate', 0.5):>7.1%} | "
                  f"{m.get('activation_count', 0):>11}")
        print(f"\n  Model exists: {os.path.exists(HRL_MODEL_PATH)}")
        print("=" * 60 + "\n")

    # ─── Phase 27 Item 11: organ-weight accessor for sizing pipeline ──
    def get_organ_weight(self, organ_name: str) -> float:
        """Read the meta-policy's current weight for a named organ.

        Used by HydraSizer.custom_stake_amount via the CAAT multiplier
        so sizing can respect "is the IQL motor or SAC motor in charge right
        now?" decisions. Returns 1.0 (neutral) when the organ is unknown or
        the policy hasn't trained yet.
        """
        try:
            stats = self.organ_tracker.get_metrics() if hasattr(self.organ_tracker, "get_metrics") else {}
            if organ_name in stats:
                metric = stats[organ_name]
                if isinstance(metric, dict):
                    val = float(metric.get("mean", 1.0))
                else:
                    val = float(metric)
                return max(0.5, min(1.5, val))
        except Exception:
            pass
        return 1.0

    # ─── Phase 27 Item 11: motor selector — IQL vs SAC ──
    def select_motor(self, regime: Optional[str] = None) -> str:
        """Return which RL motor should drive the next decision."""
        try:
            stats = self.organ_tracker.get_metrics() if hasattr(self.organ_tracker, "get_metrics") else {}
            iql_score = float(stats.get("iql", {}).get("mean", 0.0)) if isinstance(stats.get("iql"), dict) else float(stats.get("iql", 0.0) or 0.0)
            sac_score = float(stats.get("sac", {}).get("mean", 0.0)) if isinstance(stats.get("sac"), dict) else float(stats.get("sac", 0.0) or 0.0)
            return "sac" if sac_score > iql_score + 0.05 else "iql"
        except Exception:
            return "iql"


# Singleton
_hrl_instance = None


def get_meta_policy() -> HRLMetaPolicy:
    global _hrl_instance
    if _hrl_instance is None:
        _hrl_instance = HRLMetaPolicy()
    return _hrl_instance
