"""
world_model.py — Phase 26 Sprint 2, Task 8C

JEPA-inspired RSSM World Model — "What if?" imagination engine.

Architecture:
  Encoder: state features (335d from rl_environment) → latent z (64d)
  Recurrent: GRU(128) — temporal dynamics
  Stochastic: Gaussian(32d) — uncertainty in dynamics
  Predictor: MLP → predict z_next + reward

1000 rollouts × 3-5 steps = ~1 second on CPU.
Short-horizon rollout prevents compounding error accumulation.

Integration:
  - Reads from DuckDB world_model_states table
  - Outputs rollouts for Dream Engine (8D)
  - Stores rollouts in world_model_rollouts table
  - Used by RL agents for planning

Reference: LeCun "A Path Towards Autonomous Machine Intelligence" (2022) — JEPA
           Hafner et al. "Dream to Control" DreamerV3 (2023) — RSSM

Usage:
    wm = WorldModel()
    wm.train(replay_buffer)
    rollouts = wm.imagine(current_state, n_rollouts=1000, horizon=5)
"""

import os
import sys
import json
import logging
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
logger = logging.getLogger("world_model")

from ai_config import AI_DB_PATH
from db import get_db_connection, get_connection, execute_with_retry, init_db
from rl_environment import TOTAL_STATE_DIM

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "models", "rl")
WM_MODEL_PATH = os.path.join(MODEL_DIR, "world_model_v1.pt")

# World model dimensions
LATENT_DIM = 64       # z dimension
GRU_HIDDEN = 128      # Recurrent state dimension
STOCHASTIC_DIM = 32   # Gaussian noise dimension
INPUT_DIM = TOTAL_STATE_DIM  # From rl_environment (currently 335)

# Imagination parameters
DEFAULT_HORIZON = 5    # Short-horizon to avoid compounding error
DEFAULT_ROLLOUTS = 1000
MAX_ROLLOUTS = 5000

# Event types for dream injection
EVENT_TYPES = [
    "normal",           # Normal market continuation
    "flash_crash",      # Sudden -10% drop
    "whale_buy",        # Large buy pressure
    "news_shock",       # Unexpected news
    "regime_shift",     # Regime transition
    "liquidity_dry",    # Liquidity evaporation
    "funding_spike",    # Extreme funding rate
    "squeeze",          # Short/long squeeze
]


class WorldModel:
    """JEPA-inspired RSSM world model for trading imagination."""

    def __init__(self):
        self._initialized = False
        self._train_step = 0
        self._gru_hidden = None
        init_db()

    def _init_networks(self) -> bool:
        """Initialize PyTorch networks."""
        try:
            import torch
            import torch.nn as nn
        except ImportError:
            logger.error("[WorldModel] PyTorch not installed")
            return False

        # Encoder: input (335d) → latent (64d)
        self.encoder = nn.Sequential(
            nn.Linear(INPUT_DIM, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, LATENT_DIM),
        )

        # GRU: recurrent dynamics
        self.gru = nn.GRU(
            input_size=LATENT_DIM + STOCHASTIC_DIM,
            hidden_size=GRU_HIDDEN,
            batch_first=True,
        )

        # Stochastic: h → (mu, logvar) for Gaussian sampling
        self.stochastic_mu = nn.Linear(GRU_HIDDEN, STOCHASTIC_DIM)
        self.stochastic_logvar = nn.Linear(GRU_HIDDEN, STOCHASTIC_DIM)

        # Predictor: h + z_stoch → z_next + reward
        self.predictor = nn.Sequential(
            nn.Linear(GRU_HIDDEN + STOCHASTIC_DIM, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
        )
        self.z_pred_head = nn.Linear(64, LATENT_DIM)
        self.reward_head = nn.Linear(64, 1)

        # Decoder: z → reconstructed observation (for training signal)
        self.decoder = nn.Sequential(
            nn.Linear(LATENT_DIM, 128), nn.ReLU(),
            nn.Linear(128, 256), nn.ReLU(),
            nn.Linear(256, INPUT_DIM),
        )

        # Optimizer
        import torch.optim as optim
        all_params = (
            list(self.encoder.parameters()) +
            list(self.gru.parameters()) +
            list(self.stochastic_mu.parameters()) +
            list(self.stochastic_logvar.parameters()) +
            list(self.predictor.parameters()) +
            list(self.z_pred_head.parameters()) +
            list(self.reward_head.parameters()) +
            list(self.decoder.parameters())
        )
        self.optimizer = optim.Adam(all_params, lr=3e-4)

        self._initialized = True
        total_params = sum(p.numel() for p in all_params)
        logger.info(f"[WorldModel] Initialized: {total_params:,} parameters")
        return True

    def _reparameterize(self, mu, logvar):
        """Reparameterization trick for Gaussian sampling."""
        import torch
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    # ===================================================================
    # Training
    # ===================================================================

    def train_step(self, states: np.ndarray, next_states: np.ndarray,
                   rewards: np.ndarray) -> Dict[str, float]:
        """Train one step on a batch of transitions.

        Args:
            states: (batch, INPUT_DIM) current states
            next_states: (batch, INPUT_DIM) next states
            rewards: (batch,) rewards
        """
        import torch
        import torch.nn.functional as F

        if not self._initialized:
            if not self._init_networks():
                return {"error": "not initialized"}

        s = torch.FloatTensor(states)
        ns = torch.FloatTensor(next_states)
        r = torch.FloatTensor(rewards).unsqueeze(1)

        # Encode current and next states
        z = self.encoder(s)
        z_next_true = self.encoder(ns)

        # Stochastic component
        batch_size = s.shape[0]
        h = torch.zeros(1, batch_size, GRU_HIDDEN)
        z_stoch_input = torch.randn(batch_size, STOCHASTIC_DIM)

        # GRU forward
        gru_input = torch.cat([z, z_stoch_input], dim=1).unsqueeze(1)
        gru_out, h_next = self.gru(gru_input, h)
        h_squeezed = gru_out.squeeze(1)

        # Stochastic latent
        mu = self.stochastic_mu(h_squeezed)
        logvar = self.stochastic_logvar(h_squeezed)
        z_stoch = self._reparameterize(mu, logvar)

        # Predict next state and reward
        pred_input = torch.cat([h_squeezed, z_stoch], dim=1)
        pred_features = self.predictor(pred_input)
        z_next_pred = self.z_pred_head(pred_features)
        reward_pred = self.reward_head(pred_features)

        # Decode for reconstruction loss
        s_recon = self.decoder(z)

        # Losses
        # 1. Next-state prediction (JEPA loss in latent space)
        pred_loss = F.mse_loss(z_next_pred, z_next_true.detach())

        # 2. Reward prediction
        reward_loss = F.mse_loss(reward_pred, r)

        # 3. Reconstruction (auxiliary signal)
        recon_loss = F.mse_loss(s_recon, s) * 0.1  # Lower weight

        # 4. KL divergence (regularize stochastic component)
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp()) * 0.01

        total_loss = pred_loss + reward_loss + recon_loss + kl_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), 1.0)
        self.optimizer.step()

        self._train_step += 1

        return {
            "total_loss": total_loss.item(),
            "pred_loss": pred_loss.item(),
            "reward_loss": reward_loss.item(),
            "recon_loss": recon_loss.item(),
            "kl_loss": kl_loss.item(),
        }

    def train_from_buffer(self, n_epochs: int = 50,
                          batch_size: int = 64) -> Dict:
        """Train from DuckDB replay buffer or SQLite."""
        from iql_pretrain import ReplayBuffer
        from rl_environment import TOTAL_STATE_DIM, DIM_ACTION

        buffer = ReplayBuffer(state_dim=TOTAL_STATE_DIM, action_dim=DIM_ACTION)
        loaded = buffer.load_from_duckdb(TOTAL_STATE_DIM, DIM_ACTION)
        if loaded == 0:
            loaded = buffer.load_from_sqlite(TOTAL_STATE_DIM, DIM_ACTION)

        if loaded < batch_size:
            return {"error": f"insufficient data: {loaded} < {batch_size}"}

        logger.info(f"[WorldModel] Training: {n_epochs} epochs, {loaded} transitions")

        all_metrics = []
        for epoch in range(n_epochs):
            batch = buffer.sample(min(batch_size, buffer.size))
            metrics = self.train_step(
                batch["states"], batch["next_states"], batch["rewards"]
            )
            all_metrics.append(metrics)

            if (epoch + 1) % 10 == 0:
                avg = {k: np.mean([m[k] for m in all_metrics[-10:]]) for k in metrics}
                logger.info(f"[WorldModel] Epoch {epoch+1}/{n_epochs}: "
                           f"total={avg['total_loss']:.4f}, "
                           f"pred={avg['pred_loss']:.4f}")

        final = {k: np.mean([m[k] for m in all_metrics[-10:]]) for k in all_metrics[-1]}
        logger.info(f"[WorldModel] Training complete: {final}")
        return final

    # ===================================================================
    # Imagination (Rollouts)
    # ===================================================================

    def imagine(self, initial_state: np.ndarray,
                n_rollouts: int = DEFAULT_ROLLOUTS,
                horizon: int = DEFAULT_HORIZON,
                event_injection: bool = True) -> List[Dict]:
        """Imagine future trajectories from current state.

        Returns list of rollout results, each containing:
          - trajectory: list of (state_embedding, reward, event)
          - total_reward: sum of predicted rewards
          - event_sequence: injected events
        """
        import torch

        if not self._initialized:
            if not self._init_networks():
                return []

        n_rollouts = min(n_rollouts, MAX_ROLLOUTS)
        rollouts = []

        with torch.no_grad():
            s = torch.FloatTensor(initial_state).unsqueeze(0)
            z_init = self.encoder(s)

            for r_idx in range(n_rollouts):
                trajectory = []
                z = z_init.clone()
                h = torch.zeros(1, 1, GRU_HIDDEN)
                total_reward = 0.0
                events = []

                for step in range(horizon):
                    # Optional event injection
                    if event_injection:
                        event = np.random.choice(EVENT_TYPES,
                                                  p=[0.5, 0.08, 0.08, 0.08, 0.08,
                                                     0.06, 0.06, 0.06])
                    else:
                        event = "normal"
                    events.append(event)

                    # Add event perturbation to latent
                    z_perturbed = self._apply_event(z, event)

                    # Stochastic sampling
                    z_stoch = torch.randn(1, STOCHASTIC_DIM)

                    # GRU forward
                    gru_input = torch.cat([z_perturbed, z_stoch], dim=1).unsqueeze(1)
                    gru_out, h = self.gru(gru_input, h)
                    h_squeezed = gru_out.squeeze(1)

                    # Predict
                    mu = self.stochastic_mu(h_squeezed)
                    logvar = self.stochastic_logvar(h_squeezed)
                    z_stoch_sampled = self._reparameterize(mu, logvar)

                    pred_input = torch.cat([h_squeezed, z_stoch_sampled], dim=1)
                    pred_features = self.predictor(pred_input)
                    z_next = self.z_pred_head(pred_features)
                    reward = self.reward_head(pred_features).item()

                    trajectory.append({
                        "z": z_next.numpy().flatten().tolist(),
                        "reward": reward,
                        "event": event,
                    })

                    total_reward += reward
                    z = z_next

                rollouts.append({
                    "trajectory": trajectory,
                    "total_reward": total_reward,
                    "events": events,
                    "horizon": horizon,
                })

        # Sort by total reward (best rollouts first)
        rollouts.sort(key=lambda x: x["total_reward"], reverse=True)

        logger.debug(f"[WorldModel] Imagined {n_rollouts} rollouts, "
                    f"best={rollouts[0]['total_reward']:.3f}, "
                    f"worst={rollouts[-1]['total_reward']:.3f}")

        return rollouts

    def _apply_event(self, z, event: str):
        """Apply event perturbation to latent state."""
        import torch

        if event == "normal":
            return z
        elif event == "flash_crash":
            noise = torch.randn_like(z) * 2.0
            noise[0, :8] -= 3.0  # Strong negative shift
            return z + noise
        elif event == "whale_buy":
            noise = torch.randn_like(z) * 1.0
            noise[0, :8] += 2.0  # Positive shift
            return z + noise
        elif event == "regime_shift":
            noise = torch.randn_like(z) * 1.5
            return z + noise
        elif event == "liquidity_dry":
            noise = torch.randn_like(z) * 0.5
            noise[0, 8:16] -= 2.0  # Volatility increase
            return z + noise
        else:
            noise = torch.randn_like(z) * 0.5
            return z + noise

    def sample_random_state(self) -> np.ndarray:
        """Sample a random initial state for dream generation."""
        import torch
        with torch.no_grad():
            z = torch.randn(1, LATENT_DIM)
        return z.numpy().flatten()

    # ===================================================================
    # Persistence
    # ===================================================================

    def save(self, path: str = None):
        import torch
        path = path or WM_MODEL_PATH
        os.makedirs(os.path.dirname(path), exist_ok=True)

        checkpoint = {
            "encoder": self.encoder.state_dict(),
            "gru": self.gru.state_dict(),
            "stochastic_mu": self.stochastic_mu.state_dict(),
            "stochastic_logvar": self.stochastic_logvar.state_dict(),
            "predictor": self.predictor.state_dict(),
            "z_pred_head": self.z_pred_head.state_dict(),
            "reward_head": self.reward_head.state_dict(),
            "decoder": self.decoder.state_dict(),
            "train_step": self._train_step,
            "saved_at": datetime.utcnow().isoformat(),
        }
        torch.save(checkpoint, path)
        logger.info(f"[WorldModel] Saved: {path}")

    def load(self, path: str = None) -> bool:
        import torch
        path = path or WM_MODEL_PATH
        if not os.path.exists(path):
            return False

        try:
            self._init_networks()
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            self.encoder.load_state_dict(checkpoint["encoder"])
            self.gru.load_state_dict(checkpoint["gru"])
            self.stochastic_mu.load_state_dict(checkpoint["stochastic_mu"])
            self.stochastic_logvar.load_state_dict(checkpoint["stochastic_logvar"])
            self.predictor.load_state_dict(checkpoint["predictor"])
            self.z_pred_head.load_state_dict(checkpoint["z_pred_head"])
            self.reward_head.load_state_dict(checkpoint["reward_head"])
            self.decoder.load_state_dict(checkpoint["decoder"])
            self._train_step = checkpoint.get("train_step", 0)
            logger.info(f"[WorldModel] Loaded: {path} (step={self._train_step})")
            return True
        except Exception as e:
            logger.error(f"[WorldModel] Load failed: {e}")
            return False

    def persist_rollouts(self, rollouts: List[Dict], run_id: str = None):
        """Save rollout results to SQLite for analysis."""
        if not rollouts:
            return

        run_id = run_id or str(uuid.uuid4())[:8]
        with get_connection() as conn:
            for r_idx, rollout in enumerate(rollouts[:100]):  # Cap at 100
                for step, traj in enumerate(rollout["trajectory"]):
                    try:
                        conn.execute("""
                            INSERT INTO world_model_rollouts
                            (run_id, rollout_idx, step, predicted_reward,
                             predicted_regime, event_injected)
                            VALUES (?, ?, ?, ?, ?, ?)
                        """, (run_id, r_idx, step, traj["reward"],
                              None, traj["event"]))
                    except Exception:
                        pass
            conn.commit()

        logger.info(f"[WorldModel] Persisted {min(len(rollouts), 100)} rollouts "
                    f"(run_id={run_id})")


# Singleton
_wm_instance = None

def get_world_model() -> WorldModel:
    global _wm_instance
    if _wm_instance is None:
        _wm_instance = WorldModel()
    return _wm_instance
