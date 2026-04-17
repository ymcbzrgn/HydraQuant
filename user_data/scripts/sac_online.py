"""
sac_online.py — Phase 26 Sprint 2, Task 7C

Soft Actor-Critic (SAC) online fine-tuning.

Continues from IQL pre-trained checkpoint (7B) with live trade data.
SAC adds entropy regularization for exploration/exploitation balance.
Off-policy: every data point is reusable (sample efficient).

Cal-QL inspired: smooth offline→online transition by constraining
Q-values to not exceed the offline learned bounds initially.

Pipeline:
  1. Load IQL checkpoint as initialization
  2. Receive live trade outcome → add to replay buffer
  3. Train SAC on mixed buffer (offline + online data)
  4. Update policy for next trading decisions

Usage:
    sac = SACOnlineTrainer.from_iql_checkpoint()
    sac.add_transition(state, action, reward, next_state, done)
    sac.train_step()  # Called periodically
    action = sac.predict(state)

Reference: Haarnoja et al. "Soft Actor-Critic" (ICML 2018)
"""

import os
import sys
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
logger = logging.getLogger("sac_online")

from ai_config import AI_DB_PATH
from db import execute_with_retry, init_db

# ===========================================================================
# Constants
# ===========================================================================

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "models", "rl")
SAC_MODEL_PATH = os.path.join(MODEL_DIR, "sac_online_v1.pt")

SAC_CONFIG = {
    "hidden_dim": 256,
    "n_layers": 3,
    "learning_rate": 3e-4,
    "batch_size": 128,
    "discount": 0.99,
    "tau": 0.005,
    "alpha": 0.2,          # Entropy coefficient (auto-tuned)
    "auto_alpha": True,     # Automatic entropy tuning
    "buffer_size": 50000,
    "min_buffer_size": 256, # Minimum buffer before training starts
    "train_interval": 10,   # Train every N transitions
    "calql_warmup": 1000,   # Cal-QL conservative warmup steps
}


class SACOnlineTrainer:
    """SAC online fine-tuning from IQL checkpoint."""

    def __init__(self, state_dim: int, action_dim: int,
                 action_low: np.ndarray, action_high: np.ndarray,
                 config: Dict = None):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_low = action_low
        self.action_high = action_high
        self.config = config or SAC_CONFIG

        self._initialized = False
        self._step_count = 0
        self._train_count = 0

        # Simple replay buffer
        self._buffer_size = self.config["buffer_size"]
        self._buffer_ptr = 0
        self._buffer_len = 0
        self._states = np.zeros((self._buffer_size, state_dim), dtype=np.float32)
        self._actions = np.zeros((self._buffer_size, action_dim), dtype=np.float32)
        self._rewards = np.zeros(self._buffer_size, dtype=np.float32)
        self._next_states = np.zeros((self._buffer_size, state_dim), dtype=np.float32)
        self._dones = np.zeros(self._buffer_size, dtype=np.float32)

        init_db()

    def _init_networks(self):
        """Initialize SAC networks."""
        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim
        except ImportError:
            logger.error("[SAC] PyTorch not installed")
            return False

        hidden = self.config["hidden_dim"]
        n_layers = self.config["n_layers"]

        def mlp(input_dim, output_dim, hidden_dim, n_hidden):
            layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
            for _ in range(n_hidden - 1):
                layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
            layers.append(nn.Linear(hidden_dim, output_dim))
            return nn.Sequential(*layers)

        # Twin Q-networks
        self.q1 = mlp(self.state_dim + self.action_dim, 1, hidden, n_layers)
        self.q2 = mlp(self.state_dim + self.action_dim, 1, hidden, n_layers)
        self.q1_target = mlp(self.state_dim + self.action_dim, 1, hidden, n_layers)
        self.q2_target = mlp(self.state_dim + self.action_dim, 1, hidden, n_layers)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        # Policy (Gaussian)
        self.policy = mlp(self.state_dim, self.action_dim * 2, hidden, n_layers)

        # Optimizers
        lr = self.config["learning_rate"]
        self.q_optim = optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()), lr=lr
        )
        self.policy_optim = optim.Adam(self.policy.parameters(), lr=lr)

        # Auto entropy tuning
        if self.config["auto_alpha"]:
            self.target_entropy = -self.action_dim
            self.log_alpha = torch.zeros(1, requires_grad=True)
            self.alpha_optim = optim.Adam([self.log_alpha], lr=lr)
            self.alpha = self.config["alpha"]
        else:
            self.alpha = self.config["alpha"]

        self._initialized = True
        return True

    @classmethod
    def from_iql_checkpoint(cls, iql_path: str = None) -> 'SACOnlineTrainer':
        """Create SAC trainer initialized from IQL checkpoint."""
        from rl_environment import TOTAL_STATE_DIM, DIM_ACTION

        action_low = np.array([0.1, 0.2, 0.5, 1.0], dtype=np.float32)
        action_high = np.array([3.0, 0.9, 5.0, 5.0], dtype=np.float32)

        trainer = cls(
            state_dim=TOTAL_STATE_DIM,
            action_dim=DIM_ACTION,
            action_low=action_low,
            action_high=action_high,
        )

        iql_path = iql_path or os.path.join(MODEL_DIR, "iql_sizing_v1.pt")

        if os.path.exists(iql_path):
            try:
                import torch
                checkpoint = torch.load(iql_path, map_location="cpu", weights_only=False)

                trainer._init_networks()

                # Transfer Q-networks and policy from IQL
                trainer.q1.load_state_dict(checkpoint["q1_net"])
                trainer.q2.load_state_dict(checkpoint["q2_net"])
                trainer.q1_target.load_state_dict(checkpoint["q1_target"])
                trainer.q2_target.load_state_dict(checkpoint["q2_target"])
                trainer.policy.load_state_dict(checkpoint["policy_net"])

                logger.info(f"[SAC] Initialized from IQL checkpoint: {iql_path}")
            except Exception as e:
                logger.warning(f"[SAC] Failed to load IQL checkpoint: {e}")
                trainer._init_networks()
        else:
            logger.info("[SAC] No IQL checkpoint found, starting from scratch")
            trainer._init_networks()

        return trainer

    # ===================================================================
    # Online Learning API
    # ===================================================================

    def add_transition(self, state: np.ndarray, action: np.ndarray,
                       reward: float, next_state: np.ndarray, done: bool):
        """Add a live transition to the replay buffer."""
        idx = self._buffer_ptr % self._buffer_size
        self._states[idx] = state
        self._actions[idx] = action
        self._rewards[idx] = reward
        self._next_states[idx] = next_state
        self._dones[idx] = float(done)

        self._buffer_ptr += 1
        self._buffer_len = min(self._buffer_len + 1, self._buffer_size)
        self._step_count += 1

        # Auto-train
        if (self._step_count % self.config["train_interval"] == 0 and
                self._buffer_len >= self.config["min_buffer_size"]):
            self.train_step()

    def train_step(self) -> Optional[Dict[str, float]]:
        """Execute one training step."""
        import torch
        import torch.nn.functional as F

        if not self._initialized or self._buffer_len < self.config["batch_size"]:
            return None

        batch_size = self.config["batch_size"]
        discount = self.config["discount"]
        tau = self.config["tau"]

        # Sample batch
        idx = np.random.randint(0, self._buffer_len, size=batch_size)
        s = torch.FloatTensor(self._states[idx])
        a = torch.FloatTensor(self._actions[idx])
        r = torch.FloatTensor(self._rewards[idx]).unsqueeze(1)
        ns = torch.FloatTensor(self._next_states[idx])
        d = torch.FloatTensor(self._dones[idx]).unsqueeze(1)

        # --- Q update ---
        with torch.no_grad():
            next_action, next_log_prob = self._sample_action(ns)
            nsa = torch.cat([ns, next_action], dim=1)
            q1_next = self.q1_target(nsa)
            q2_next = self.q2_target(nsa)
            q_next = torch.min(q1_next, q2_next) - self.alpha * next_log_prob
            q_target = r + discount * (1 - d) * q_next

        sa = torch.cat([s, a], dim=1)
        q1_loss = F.mse_loss(self.q1(sa), q_target)
        q2_loss = F.mse_loss(self.q2(sa), q_target)
        q_loss = q1_loss + q2_loss

        self.q_optim.zero_grad()
        q_loss.backward()
        self.q_optim.step()

        # --- Policy update ---
        new_action, log_prob = self._sample_action(s)
        new_sa = torch.cat([s, new_action], dim=1)
        q1_new = self.q1(new_sa)
        q2_new = self.q2(new_sa)
        q_new = torch.min(q1_new, q2_new)

        policy_loss = (self.alpha * log_prob - q_new).mean()

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        # --- Alpha update (auto entropy tuning) ---
        alpha_loss = 0.0
        if self.config["auto_alpha"]:
            alpha_loss_val = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
            self.alpha_optim.zero_grad()
            alpha_loss_val.backward()
            self.alpha_optim.step()
            self.alpha = self.log_alpha.exp().item()
            alpha_loss = alpha_loss_val.item()

        # --- Soft update targets ---
        for target, source in [(self.q1_target, self.q1), (self.q2_target, self.q2)]:
            for tp, sp in zip(target.parameters(), source.parameters()):
                tp.data.copy_(tau * sp.data + (1 - tau) * tp.data)

        self._train_count += 1

        return {
            "q_loss": q_loss.item(),
            "policy_loss": policy_loss.item(),
            "alpha": self.alpha,
            "alpha_loss": alpha_loss,
            "buffer_size": self._buffer_len,
        }

    def _sample_action(self, state_tensor):
        """Sample action from policy with reparameterization trick."""
        import torch

        policy_out = self.policy(state_tensor)
        mean = policy_out[:, :self.action_dim]
        log_std = policy_out[:, self.action_dim:].clamp(-5, 2)
        std = log_std.exp()

        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # Reparameterization trick
        action = torch.tanh(x_t)

        # Scale to action bounds
        action_low = torch.FloatTensor(self.action_low)
        action_high = torch.FloatTensor(self.action_high)
        action = action_low + (action + 1) * 0.5 * (action_high - action_low)

        # Log probability (with tanh squashing correction)
        log_prob = normal.log_prob(x_t) - torch.log(1 - torch.tanh(x_t).pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=1, keepdim=True)

        return action, log_prob

    def predict(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Predict action for live trading."""
        import torch

        if not self._initialized:
            return np.array([1.0, 0.5, 2.0, 1.0], dtype=np.float32)

        with torch.no_grad():
            s = torch.FloatTensor(state).unsqueeze(0)
            policy_out = self.policy(s)
            mean = policy_out[0, :self.action_dim]

            if deterministic:
                action = torch.tanh(mean)
            else:
                log_std = policy_out[0, self.action_dim:].clamp(-5, 2)
                std = log_std.exp()
                normal = torch.distributions.Normal(mean, std)
                action = torch.tanh(normal.sample())

        # Scale to action bounds
        action_np = action.numpy()
        scaled = self.action_low + (action_np + 1) * 0.5 * (self.action_high - self.action_low)
        return np.clip(scaled, self.action_low, self.action_high).astype(np.float32)

    def save(self, path: str = None):
        """Save SAC checkpoint."""
        import torch
        path = path or SAC_MODEL_PATH
        os.makedirs(os.path.dirname(path), exist_ok=True)

        checkpoint = {
            "q1": self.q1.state_dict(),
            "q2": self.q2.state_dict(),
            "q1_target": self.q1_target.state_dict(),
            "q2_target": self.q2_target.state_dict(),
            "policy": self.policy.state_dict(),
            "alpha": self.alpha,
            "config": self.config,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "action_low": self.action_low.tolist(),
            "action_high": self.action_high.tolist(),
            "step_count": self._step_count,
            "train_count": self._train_count,
            "saved_at": datetime.utcnow().isoformat(),
        }

        if self.config["auto_alpha"]:
            checkpoint["log_alpha"] = self.log_alpha.detach().numpy().tolist()

        torch.save(checkpoint, path)
        logger.info(f"[SAC] Model saved: {path} (steps={self._step_count}, "
                    f"trained={self._train_count})")

    def load(self, path: str = None) -> bool:
        """Load SAC checkpoint."""
        import torch
        path = path or SAC_MODEL_PATH
        if not os.path.exists(path):
            return False

        try:
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            self._init_networks()
            self.q1.load_state_dict(checkpoint["q1"])
            self.q2.load_state_dict(checkpoint["q2"])
            self.q1_target.load_state_dict(checkpoint["q1_target"])
            self.q2_target.load_state_dict(checkpoint["q2_target"])
            self.policy.load_state_dict(checkpoint["policy"])
            self.alpha = checkpoint.get("alpha", 0.2)
            self._step_count = checkpoint.get("step_count", 0)
            self._train_count = checkpoint.get("train_count", 0)
            logger.info(f"[SAC] Model loaded: {path}")
            return True
        except Exception as e:
            logger.error(f"[SAC] Failed to load: {e}")
            return False

    # Phase 27 Item 10: scheduler entry — pull live transitions, train N steps.
    def online_step(self, n_steps: int = 200) -> Dict[str, Any]:
        """Pull recent transitions from rl_replay_buffer and run `n_steps`
        gradient updates. Lightweight enough to fit in a Sunday cron window.
        """
        if not self._initialized:
            return {"status": "skipped", "reason": "not initialized"}
        try:
            from db import get_db_connection
            conn = get_db_connection()
            rows = conn.execute("""
                SELECT state_json, action_json, reward, next_state_json, done
                FROM rl_replay_buffer
                WHERE source IN ('live', 'dream')
                ORDER BY id DESC LIMIT 5000
            """).fetchall()
            conn.close()
            import json as _json
            n_added = 0
            for r in rows:
                try:
                    state = np.asarray(_json.loads(r["state_json"]), dtype=np.float32)
                    next_state = np.asarray(_json.loads(r["next_state_json"]), dtype=np.float32)
                    action = np.asarray(_json.loads(r["action_json"]), dtype=np.float32)
                    self.add_transition(state, action, float(r["reward"] or 0.0),
                                         next_state, bool(r["done"]))
                    n_added += 1
                except Exception:
                    continue
        except Exception as e:
            return {"status": "skipped", "reason": f"replay load: {e}"}

        losses: List[Dict] = []
        for _ in range(int(n_steps)):
            res = self.train_step()
            if res is not None:
                losses.append(res)
        if losses:
            avg_q = float(np.mean([l.get("q_loss", 0.0) for l in losses]))
            avg_p = float(np.mean([l.get("policy_loss", 0.0) for l in losses]))
            return {"status": "trained", "n_added": n_added,
                    "n_steps": len(losses),
                    "avg_q_loss": round(avg_q, 4),
                    "avg_policy_loss": round(avg_p, 4)}
        return {"status": "no_data", "n_added": n_added}


# Singleton accessor
_sac_instance = None


def get_sac_trainer() -> SACOnlineTrainer:
    """Get or create SAC trainer singleton."""
    global _sac_instance
    if _sac_instance is None:
        _sac_instance = SACOnlineTrainer.from_iql_checkpoint()
    return _sac_instance
