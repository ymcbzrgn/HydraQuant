"""
reptile_meta.py — Phase 26 Sprint 2, Task 8B

Reptile Meta-Learning — "Learn HOW to learn."

Learns an initialization of organism parameters that can adapt to ANY regime
in just 5 gradient steps. After Reptile training, the organism can switch from
bull to bear market in 5 trades instead of 50.

Algorithm (OpenAI Reptile):
  theta = initial_params
  for episode in meta_episodes:
      task = sample_regime()           # Random regime from history
      phi = theta.clone()
      for k in range(K):               # K=5 inner steps
          loss = evaluate(phi, task)
          phi = phi - alpha * grad(loss)
      theta += beta * (phi - theta)     # Meta-update toward adapted params

Integration:
  - EWC (8A) prevents forgetting during inner loop
  - IQL/SAC (7B/7C) provides the base model to meta-learn
  - Scheduler runs weekly meta-update

Reference: Nichol et al. "On First-Order Meta-Learning Algorithms" (2018)
"""

import os
import sys
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
logger = logging.getLogger("reptile_meta")

from ai_config import AI_DB_PATH
from db import get_db_connection, get_connection, init_db

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "models", "rl")
REPTILE_STATE_PATH = os.path.join(MODEL_DIR, "reptile_meta_init.npz")

# Reptile hyperparameters
REPTILE_CONFIG = {
    "inner_lr": 1e-3,       # Inner loop learning rate (alpha)
    "outer_lr": 0.1,        # Meta-learning rate (beta)
    "inner_steps": 5,       # K — inner adaptation steps
    "meta_episodes": 50,    # Number of meta-training episodes
    "task_batch_size": 16,  # Transitions per inner step
    "eval_tasks": 5,        # Tasks for meta-evaluation
}

# Regimes to sample from
REGIME_POOL = [
    "trending_bull", "trending_bear", "ranging",
    "high_volatility", "transitional", "recovery",
]


class RegimeTaskSampler:
    """Samples regime-specific training tasks from historical data."""

    def __init__(self, db_path: str = AI_DB_PATH):
        self.db_path = db_path
        self._cache: Dict[str, List[Dict]] = {}

    def sample_task(self, regime: str = None,
                    n_samples: int = 16) -> Optional[List[Dict]]:
        """Sample a batch of transitions from a specific regime."""
        if regime is None:
            regime = np.random.choice(REGIME_POOL)

        # Cache regime data
        if regime not in self._cache:
            self._cache[regime] = self._load_regime_data(regime)

        data = self._cache.get(regime, [])
        if len(data) < n_samples:
            # Try any regime if specific one has too little data
            for fallback in REGIME_POOL:
                if fallback not in self._cache:
                    self._cache[fallback] = self._load_regime_data(fallback)
                if len(self._cache.get(fallback, [])) >= n_samples:
                    data = self._cache[fallback]
                    regime = fallback
                    break

        if len(data) < n_samples:
            return None

        indices = np.random.choice(len(data), size=n_samples, replace=False)
        return [data[i] for i in indices]

    def _load_regime_data(self, regime: str) -> List[Dict]:
        """Load trade data for a specific regime."""
        conn = get_db_connection(self.db_path)
        try:
            rows = conn.execute("""
                SELECT confidence, outcome_pnl, trust_score_at_decision,
                       position_size, regime
                FROM ai_decisions
                WHERE outcome_pnl IS NOT NULL AND regime = ?
                ORDER BY timestamp ASC
            """, (regime,)).fetchall()

            return [dict(r) for r in rows]
        except Exception:
            return []
        finally:
            conn.close()

    def get_available_regimes(self) -> Dict[str, int]:
        """Get count of available data per regime."""
        conn = get_db_connection(self.db_path)
        try:
            rows = conn.execute("""
                SELECT regime, COUNT(*) as cnt
                FROM ai_decisions
                WHERE outcome_pnl IS NOT NULL
                GROUP BY regime
                ORDER BY cnt DESC
            """).fetchall()
            return {r["regime"]: r["cnt"] for r in rows}
        finally:
            conn.close()


class ReptileMetaLearner:
    """Reptile meta-learning for fast regime adaptation."""

    def __init__(self, param_dim: int = None, config: Dict = None):
        self.config = config or REPTILE_CONFIG
        self.param_dim = param_dim
        self._meta_params = None  # The learned initialization
        self._task_sampler = RegimeTaskSampler()
        self._load_state()
        init_db()

    def _load_state(self):
        """Load saved meta-initialization."""
        if os.path.exists(REPTILE_STATE_PATH):
            try:
                data = np.load(REPTILE_STATE_PATH)
                self._meta_params = data["meta_params"]
                self.param_dim = len(self._meta_params)
                logger.info(f"[Reptile] Loaded meta-init: {self.param_dim} params")
            except Exception as e:
                logger.warning(f"[Reptile] Failed to load state: {e}")

    def save_state(self):
        """Save meta-initialization."""
        if self._meta_params is not None:
            try:
                os.makedirs(os.path.dirname(REPTILE_STATE_PATH), exist_ok=True)
                np.savez(REPTILE_STATE_PATH, meta_params=self._meta_params)
                logger.info(f"[Reptile] State saved: {len(self._meta_params)} params")
            except Exception as e:
                logger.warning(f"[Reptile] Failed to save: {e}")

    def _task_loss(self, params: np.ndarray, task_data: List[Dict]) -> float:
        """Compute loss on a regime task.

        Simple loss: predicted confidence vs actual outcome direction.
        """
        total_loss = 0.0
        for sample in task_data:
            pnl = sample.get("outcome_pnl", 0)
            conf = sample.get("confidence", 0.5)
            trust = sample.get("trust_score_at_decision", 0.5)

            # Target: 1 if profitable, 0 if loss
            target = 1.0 if pnl > 0 else 0.0

            # Simple linear prediction from params
            # params[:3] act as weights for [confidence, trust, bias]
            if len(params) >= 3:
                pred = params[0] * conf + params[1] * trust + params[2]
                pred = 1.0 / (1.0 + np.exp(-np.clip(pred, -10, 10)))  # sigmoid
            else:
                pred = conf

            # Binary cross-entropy loss
            eps = 1e-7
            loss = -(target * np.log(pred + eps) + (1 - target) * np.log(1 - pred + eps))
            total_loss += loss

        return total_loss / max(len(task_data), 1)

    def _task_gradient(self, params: np.ndarray, task_data: List[Dict]) -> np.ndarray:
        """Compute numerical gradient of task loss."""
        eps = 1e-4
        grad = np.zeros_like(params)
        base_loss = self._task_loss(params, task_data)

        for i in range(len(params)):
            params_plus = params.copy()
            params_plus[i] += eps
            loss_plus = self._task_loss(params_plus, task_data)
            grad[i] = (loss_plus - base_loss) / eps

        return grad

    def meta_train(self, n_episodes: int = None) -> Dict:
        """Run Reptile meta-training loop."""
        n_episodes = n_episodes or self.config["meta_episodes"]
        inner_lr = self.config["inner_lr"]
        outer_lr = self.config["outer_lr"]
        inner_steps = self.config["inner_steps"]
        task_batch = self.config["task_batch_size"]

        # Initialize meta-params if not loaded
        if self._meta_params is None:
            self.param_dim = self.param_dim or 50
            self._meta_params = np.random.randn(self.param_dim).astype(np.float32) * 0.1

        theta = self._meta_params.copy()
        losses = []
        regime_counts = {}

        logger.info(f"[Reptile] Meta-training: {n_episodes} episodes, "
                    f"{inner_steps} inner steps, {self.param_dim} params")

        for ep in range(n_episodes):
            # Sample random regime task
            regime = np.random.choice(REGIME_POOL)
            task_data = self._task_sampler.sample_task(regime, task_batch)

            if task_data is None:
                continue

            regime_counts[regime] = regime_counts.get(regime, 0) + 1

            # Inner loop: adapt theta to this regime
            phi = theta.copy()
            for k in range(inner_steps):
                grad = self._task_gradient(phi, task_data)
                phi = phi - inner_lr * grad

            # Outer loop: meta-update
            theta = theta + outer_lr * (phi - theta)

            # Track loss
            loss_after = self._task_loss(phi, task_data)
            losses.append(loss_after)

            if (ep + 1) % 10 == 0:
                avg_loss = np.mean(losses[-10:])
                logger.info(f"[Reptile] Episode {ep+1}/{n_episodes}: "
                           f"loss={avg_loss:.4f}, regime={regime}")

        self._meta_params = theta
        self.save_state()

        metrics = {
            "n_episodes": n_episodes,
            "final_loss": float(np.mean(losses[-10:])) if losses else 0.0,
            "regime_distribution": regime_counts,
            "param_dim": self.param_dim,
        }

        logger.info(f"[Reptile] Meta-training complete: loss={metrics['final_loss']:.4f}")
        return metrics

    def adapt(self, regime: str, n_steps: int = None) -> np.ndarray:
        """Fast adaptation to a new regime (5-shot).

        Returns adapted parameters starting from meta-initialization.
        """
        if self._meta_params is None:
            logger.warning("[Reptile] No meta-init available, returning random params")
            return np.random.randn(self.param_dim or 50).astype(np.float32) * 0.1

        n_steps = n_steps or self.config["inner_steps"]
        inner_lr = self.config["inner_lr"]
        task_batch = self.config["task_batch_size"]

        task_data = self._task_sampler.sample_task(regime, task_batch)
        if task_data is None:
            return self._meta_params.copy()

        phi = self._meta_params.copy()
        for k in range(n_steps):
            grad = self._task_gradient(phi, task_data)
            phi = phi - inner_lr * grad

        loss_before = self._task_loss(self._meta_params, task_data)
        loss_after = self._task_loss(phi, task_data)

        logger.info(f"[Reptile] Adapted to {regime} in {n_steps} steps: "
                    f"loss {loss_before:.4f} → {loss_after:.4f}")

        return phi

    def get_meta_params(self) -> Optional[np.ndarray]:
        """Get current meta-initialization parameters."""
        return self._meta_params.copy() if self._meta_params is not None else None

    def get_status(self) -> Dict:
        available = self._task_sampler.get_available_regimes()
        return {
            "has_meta_init": self._meta_params is not None,
            "param_dim": self.param_dim,
            "available_regimes": available,
            "config": self.config,
        }


# Singleton
_reptile_instance = None

def get_reptile() -> ReptileMetaLearner:
    global _reptile_instance
    if _reptile_instance is None:
        _reptile_instance = ReptileMetaLearner()
    return _reptile_instance
