"""
iql_pretrain.py — Phase 26 Sprint 2, Task 7B

Implicit Q-Learning (IQL) offline pre-training.

IQL is safe for offline RL because it never queries OOD (out-of-distribution) actions.
Key insight: learns V(s) and Q(s,a) separately, uses expectile regression to avoid
querying actions not in the dataset.

Pipeline:
  1. Load episodes from DuckDB rl_episodes table (filled by 13B backtest pipeline)
  2. Build replay buffer (states, actions, rewards, next_states, dones)
  3. Train IQL model (V-network + Q-network + policy)
  4. Save checkpoint to user_data/models/rl/iql_sizing_v1.pt
  5. Log training run to rl_checkpoints table

Reference: Kostrikov et al. "Offline Reinforcement Learning with Implicit Q-Learning" (ICLR 2022)

Usage:
    python iql_pretrain.py --train                    # Train from replay buffer
    python iql_pretrain.py --generate-episodes 200    # Generate episodes first
    python iql_pretrain.py --status                   # Show training status
"""

import os
import sys
import json
import logging
import argparse
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
logger = logging.getLogger("iql_pretrain")

from ai_config import AI_DB_PATH
from db import get_db_connection, get_connection, execute_with_retry, init_db

# ===========================================================================
# Constants
# ===========================================================================

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "models", "rl")
IQL_MODEL_PATH = os.path.join(MODEL_DIR, "iql_sizing_v1.pt")

# IQL Hyperparameters
IQL_CONFIG = {
    "hidden_dim": 256,
    "n_layers": 3,
    "learning_rate": 3e-4,
    "batch_size": 256,
    "discount": 0.99,
    "tau": 0.005,       # Target network update rate
    "expectile": 0.7,   # IQL expectile (higher = more optimistic)
    "temperature": 3.0, # AWR temperature for policy extraction
    "n_epochs": 100,
    "eval_interval": 10,
}


class ReplayBuffer:
    """Simple replay buffer for offline RL training."""

    def __init__(self, state_dim: int, action_dim: int, max_size: int = 100000):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.states = np.zeros((max_size, state_dim), dtype=np.float32)
        self.actions = np.zeros((max_size, action_dim), dtype=np.float32)
        self.rewards = np.zeros(max_size, dtype=np.float32)
        self.next_states = np.zeros((max_size, state_dim), dtype=np.float32)
        self.dones = np.zeros(max_size, dtype=np.float32)

    def add(self, state, action, reward, next_state, done):
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = float(done)

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        idx = np.random.randint(0, self.size, size=batch_size)
        return {
            "states": self.states[idx],
            "actions": self.actions[idx],
            "rewards": self.rewards[idx],
            "next_states": self.next_states[idx],
            "dones": self.dones[idx],
        }

    def load_from_duckdb(self, state_dim: int, action_dim: int) -> int:
        """Load episodes from DuckDB replay buffer."""
        try:
            from analytics_engine import get_analytics
            engine = get_analytics()
            stats = engine.replay_buffer_stats()

            total_steps = sum(b.get("steps", 0) for b in stats.get("buffers", []))
            if total_steps == 0:
                logger.warning("[IQL] No episodes in DuckDB replay buffer")
                return 0

            # Sample all data (up to max_size)
            samples = engine.sample_replay_buffer(batch_size=min(total_steps, self.max_size))

            loaded = 0
            for sample in samples:
                try:
                    state = np.frombuffer(sample[2], dtype=np.float32)[:state_dim]
                    action = np.frombuffer(sample[3], dtype=np.float32)[:action_dim]
                    reward = float(sample[4])
                    next_state = (np.frombuffer(sample[5], dtype=np.float32)[:state_dim]
                                 if sample[5] else np.zeros(state_dim, dtype=np.float32))
                    done = bool(sample[6])

                    if len(state) == state_dim and len(action) == action_dim:
                        self.add(state, action, reward, next_state, done)
                        loaded += 1
                except Exception:
                    continue

            logger.info(f"[IQL] Loaded {loaded} transitions from DuckDB")
            return loaded

        except Exception as e:
            logger.error(f"[IQL] Failed to load from DuckDB: {e}")
            return 0

    def load_from_sqlite(self, state_dim: int, action_dim: int) -> int:
        """Fallback: load from SQLite rl_replay_buffer table."""
        conn = get_db_connection(AI_DB_PATH)
        try:
            rows = conn.execute("""
                SELECT state_json, action_json, reward, done
                FROM rl_replay_buffer
                ORDER BY episode_id, step
                LIMIT ?
            """, (self.max_size,)).fetchall()

            loaded = 0
            for i, row in enumerate(rows):
                try:
                    state = np.array(json.loads(row["state_json"]), dtype=np.float32)[:state_dim]
                    action = np.array(json.loads(row["action_json"]), dtype=np.float32)[:action_dim]
                    reward = float(row["reward"])
                    done = bool(row["done"])

                    # Next state from next row
                    if i + 1 < len(rows) and not done:
                        next_state = np.array(
                            json.loads(rows[i + 1]["state_json"]), dtype=np.float32
                        )[:state_dim]
                    else:
                        next_state = np.zeros(state_dim, dtype=np.float32)

                    if len(state) >= state_dim:
                        self.add(state[:state_dim], action[:action_dim],
                                reward, next_state, done)
                        loaded += 1
                except Exception:
                    continue

            logger.info(f"[IQL] Loaded {loaded} transitions from SQLite")
            return loaded

        finally:
            conn.close()

    def load_from_world_model_states(self, state_dim: int, action_dim: int) -> int:
        """Bootstrap transitions from `world_model_states` snapshots.

        Neither DuckDB nor the SQLite `rl_replay_buffer` is populated in a
        fresh install — dream_engine is the only writer, and dream_engine
        needs a trained world_model first, so the system deadlocks at
        `insufficient data: 0 < 64` forever. This fallback reads consecutive
        perception snapshots persisted by TriplePerception._persist_world_model_state
        and synthesises (state, zero-action, zero-reward, next_state, done)
        tuples so WorldModel has something to learn dynamics from. No
        counterfactual labels are introduced — rewards stay at 0.0, which
        the world model ignores (only the state transition loss matters).
        """
        conn = get_db_connection(AI_DB_PATH)
        try:
            rows = conn.execute("""
                SELECT pair, timeframe, timestamp, ttm_embedding
                FROM world_model_states
                WHERE ttm_embedding IS NOT NULL
                ORDER BY pair, timeframe, timestamp
                LIMIT ?
            """, (self.max_size,)).fetchall()

            if not rows:
                return 0

            loaded = 0
            by_key: Dict[str, List] = {}
            for row in rows:
                key = f"{row['pair']}|{row['timeframe']}"
                by_key.setdefault(key, []).append(row)

            for key, seq in by_key.items():
                if len(seq) < 2:
                    continue
                for i in range(len(seq) - 1):
                    try:
                        state_vec = np.frombuffer(seq[i]["ttm_embedding"], dtype=np.float32)
                        next_vec = np.frombuffer(seq[i + 1]["ttm_embedding"], dtype=np.float32)
                    except Exception:
                        continue

                    state = np.zeros(state_dim, dtype=np.float32)
                    next_state = np.zeros(state_dim, dtype=np.float32)
                    state[: min(len(state_vec), state_dim)] = state_vec[: state_dim]
                    next_state[: min(len(next_vec), state_dim)] = next_vec[: state_dim]

                    self.add(
                        state,
                        np.zeros(action_dim, dtype=np.float32),
                        0.0,
                        next_state,
                        i == len(seq) - 2,
                    )
                    loaded += 1

            logger.info(
                f"[IQL] Bootstrapped {loaded} transitions from world_model_states"
            )
            return loaded

        finally:
            conn.close()


class IQLTrainer:
    """Implicit Q-Learning trainer using PyTorch."""

    def __init__(self, state_dim: int, action_dim: int,
                 action_low: np.ndarray, action_high: np.ndarray,
                 config: Dict = None):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_low = action_low
        self.action_high = action_high
        self.config = config or IQL_CONFIG

        self._model = None
        self._initialized = False

    def _init_networks(self):
        """Initialize IQL networks (V, Q, Policy)."""
        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim
        except ImportError:
            logger.error("[IQL] PyTorch not installed. Run: pip install torch")
            return False

        hidden = self.config["hidden_dim"]
        n_layers = self.config["n_layers"]

        # Build MLP helper
        def mlp(input_dim, output_dim, hidden_dim, n_hidden):
            layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
            for _ in range(n_hidden - 1):
                layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
            layers.append(nn.Linear(hidden_dim, output_dim))
            return nn.Sequential(*layers)

        # V-network: s → V(s)
        self.v_net = mlp(self.state_dim, 1, hidden, n_layers)

        # Q-networks (twin): (s, a) → Q(s, a)
        self.q1_net = mlp(self.state_dim + self.action_dim, 1, hidden, n_layers)
        self.q2_net = mlp(self.state_dim + self.action_dim, 1, hidden, n_layers)

        # Target Q-networks
        self.q1_target = mlp(self.state_dim + self.action_dim, 1, hidden, n_layers)
        self.q2_target = mlp(self.state_dim + self.action_dim, 1, hidden, n_layers)
        self.q1_target.load_state_dict(self.q1_net.state_dict())
        self.q2_target.load_state_dict(self.q2_net.state_dict())

        # Policy network: s → a (Gaussian)
        self.policy_net = mlp(self.state_dim, self.action_dim * 2, hidden, n_layers)

        # Optimizers
        lr = self.config["learning_rate"]
        self.v_optim = optim.Adam(self.v_net.parameters(), lr=lr)
        self.q_optim = optim.Adam(
            list(self.q1_net.parameters()) + list(self.q2_net.parameters()), lr=lr
        )
        self.policy_optim = optim.Adam(self.policy_net.parameters(), lr=lr)

        self._initialized = True
        logger.info(f"[IQL] Networks initialized: state_dim={self.state_dim}, "
                    f"action_dim={self.action_dim}, hidden={hidden}")
        return True

    def train(self, replay_buffer: ReplayBuffer, n_epochs: int = None) -> Dict:
        """Train IQL on offline data."""
        import torch
        import torch.nn.functional as F

        if not self._initialized:
            if not self._init_networks():
                return {"error": "PyTorch not available"}

        if replay_buffer.size < self.config["batch_size"]:
            return {"error": f"Insufficient data: {replay_buffer.size} < {self.config['batch_size']}"}

        n_epochs = n_epochs or self.config["n_epochs"]
        batch_size = self.config["batch_size"]
        discount = self.config["discount"]
        tau = self.config["tau"]
        expectile = self.config["expectile"]
        temperature = self.config["temperature"]

        steps_per_epoch = max(replay_buffer.size // batch_size, 1)
        total_steps = n_epochs * steps_per_epoch

        metrics = {
            "v_losses": [], "q_losses": [], "policy_losses": [],
            "avg_reward": [], "epochs": n_epochs,
        }

        logger.info(f"[IQL] Training: {n_epochs} epochs, {steps_per_epoch} steps/epoch, "
                    f"{replay_buffer.size} transitions")

        for epoch in range(n_epochs):
            epoch_v_loss = 0.0
            epoch_q_loss = 0.0
            epoch_p_loss = 0.0

            for step in range(steps_per_epoch):
                batch = replay_buffer.sample(batch_size)

                s = torch.FloatTensor(batch["states"])
                a = torch.FloatTensor(batch["actions"])
                r = torch.FloatTensor(batch["rewards"]).unsqueeze(1)
                ns = torch.FloatTensor(batch["next_states"])
                d = torch.FloatTensor(batch["dones"]).unsqueeze(1)

                sa = torch.cat([s, a], dim=1)

                # --- V-network update (expectile regression) ---
                with torch.no_grad():
                    q1 = self.q1_target(sa)
                    q2 = self.q2_target(sa)
                    q_min = torch.min(q1, q2)

                v = self.v_net(s)
                diff = q_min - v
                # Expectile loss: asymmetric L2
                weight = torch.where(diff > 0, expectile, 1 - expectile)
                v_loss = (weight * diff.pow(2)).mean()

                self.v_optim.zero_grad()
                v_loss.backward()
                self.v_optim.step()

                # --- Q-network update ---
                with torch.no_grad():
                    v_next = self.v_net(ns)
                    q_target = r + discount * (1 - d) * v_next

                q1_pred = self.q1_net(sa)
                q2_pred = self.q2_net(sa)
                q_loss = F.mse_loss(q1_pred, q_target) + F.mse_loss(q2_pred, q_target)

                self.q_optim.zero_grad()
                q_loss.backward()
                self.q_optim.step()

                # --- Policy update (AWR — Advantage Weighted Regression) ---
                with torch.no_grad():
                    q1_val = self.q1_target(sa)
                    q2_val = self.q2_target(sa)
                    q_min_val = torch.min(q1_val, q2_val)
                    v_val = self.v_net(s)
                    advantage = q_min_val - v_val
                    weights = torch.exp(temperature * advantage)
                    weights = torch.clamp(weights, max=100.0)

                policy_out = self.policy_net(s)
                mean = policy_out[:, :self.action_dim]
                log_std = policy_out[:, self.action_dim:].clamp(-5, 2)
                std = log_std.exp()

                # Log probability of dataset actions under current policy
                dist = torch.distributions.Normal(mean, std)
                log_prob = dist.log_prob(a).sum(dim=1, keepdim=True)

                policy_loss = -(weights * log_prob).mean()

                self.policy_optim.zero_grad()
                policy_loss.backward()
                self.policy_optim.step()

                # --- Soft update target networks ---
                for target, source in [
                    (self.q1_target, self.q1_net),
                    (self.q2_target, self.q2_net),
                ]:
                    for tp, sp in zip(target.parameters(), source.parameters()):
                        tp.data.copy_(tau * sp.data + (1 - tau) * tp.data)

                epoch_v_loss += v_loss.item()
                epoch_q_loss += q_loss.item()
                epoch_p_loss += policy_loss.item()

            # Epoch metrics
            avg_v = epoch_v_loss / steps_per_epoch
            avg_q = epoch_q_loss / steps_per_epoch
            avg_p = epoch_p_loss / steps_per_epoch
            metrics["v_losses"].append(avg_v)
            metrics["q_losses"].append(avg_q)
            metrics["policy_losses"].append(avg_p)

            if (epoch + 1) % self.config["eval_interval"] == 0:
                logger.info(f"[IQL] Epoch {epoch + 1}/{n_epochs}: "
                           f"V_loss={avg_v:.4f}, Q_loss={avg_q:.4f}, "
                           f"Policy_loss={avg_p:.4f}")

        metrics["final_v_loss"] = metrics["v_losses"][-1]
        metrics["final_q_loss"] = metrics["q_losses"][-1]
        metrics["final_policy_loss"] = metrics["policy_losses"][-1]

        logger.info(f"[IQL] Training complete: V={metrics['final_v_loss']:.4f}, "
                    f"Q={metrics['final_q_loss']:.4f}, P={metrics['final_policy_loss']:.4f}")

        return metrics

    def predict(self, state: np.ndarray) -> np.ndarray:
        """Predict action from current policy."""
        import torch

        if not self._initialized:
            # Return default action
            return np.array([1.0, 0.5, 2.0, 1.0], dtype=np.float32)

        with torch.no_grad():
            s = torch.FloatTensor(state).unsqueeze(0)
            policy_out = self.policy_net(s)
            mean = policy_out[0, :self.action_dim].numpy()

        # Clip to action bounds
        action = np.clip(mean, self.action_low, self.action_high)
        return action.astype(np.float32)

    def save(self, path: str = None):
        """Save model checkpoint."""
        import torch

        if not self._initialized:
            return

        path = path or IQL_MODEL_PATH
        os.makedirs(os.path.dirname(path), exist_ok=True)

        checkpoint = {
            "v_net": self.v_net.state_dict(),
            "q1_net": self.q1_net.state_dict(),
            "q2_net": self.q2_net.state_dict(),
            "q1_target": self.q1_target.state_dict(),
            "q2_target": self.q2_target.state_dict(),
            "policy_net": self.policy_net.state_dict(),
            "config": self.config,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "action_low": self.action_low.tolist(),
            "action_high": self.action_high.tolist(),
            "saved_at": datetime.utcnow().isoformat(),
        }

        torch.save(checkpoint, path)
        logger.info(f"[IQL] Model saved: {path}")

    def load(self, path: str = None) -> bool:
        """Load model checkpoint."""
        import torch

        path = path or IQL_MODEL_PATH
        if not os.path.exists(path):
            logger.warning(f"[IQL] No checkpoint found at {path}")
            return False

        try:
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            self.state_dim = checkpoint["state_dim"]
            self.action_dim = checkpoint["action_dim"]
            self.action_low = np.array(checkpoint["action_low"])
            self.action_high = np.array(checkpoint["action_high"])
            self.config = checkpoint.get("config", IQL_CONFIG)

            self._init_networks()

            self.v_net.load_state_dict(checkpoint["v_net"])
            self.q1_net.load_state_dict(checkpoint["q1_net"])
            self.q2_net.load_state_dict(checkpoint["q2_net"])
            self.q1_target.load_state_dict(checkpoint["q1_target"])
            self.q2_target.load_state_dict(checkpoint["q2_target"])
            self.policy_net.load_state_dict(checkpoint["policy_net"])

            logger.info(f"[IQL] Model loaded from {path}")
            return True

        except Exception as e:
            logger.error(f"[IQL] Failed to load checkpoint: {e}")
            return False

    def log_training_run(self, metrics: Dict):
        """Log training run to rl_checkpoints table.

        Note: reward_mean/reward_std/sharpe columns store training loss metrics
        (q_loss/v_loss/policy_loss) as no real reward is available during offline training.
        """
        try:
            # Compute actual reward stats from replay buffer if available
            avg_reward = metrics.get("avg_reward", None)
            if avg_reward is None:
                avg_reward = -metrics.get("final_q_loss", 0)  # Negate loss as proxy

            execute_with_retry(
                """INSERT INTO rl_checkpoints
                   (model_name, organ, version, path,
                    training_episodes, reward_mean, reward_std,
                    sharpe, is_active)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1)""",
                ("iql_sizing", "sizing", 1, IQL_MODEL_PATH,
                 metrics.get("epochs", 0),
                 avg_reward,
                 metrics.get("final_v_loss", 0),
                 0.0),  # Sharpe not available during offline training
                max_retries=5
            )
            logger.info("[IQL] Training run logged to rl_checkpoints")
        except Exception as e:
            logger.warning(f"[IQL] Failed to log training run: {e}")


# ===========================================================================
# Pipeline Runner
# ===========================================================================

def run_iql_training(generate_episodes: int = 0,
                     n_epochs: int = None) -> Dict:
    """Run full IQL training pipeline."""
    from rl_environment import TradingEnv, generate_offline_episodes, TOTAL_STATE_DIM, DIM_ACTION

    init_db()

    # Step 1: Generate episodes if requested
    if generate_episodes > 0:
        logger.info(f"[IQL] Generating {generate_episodes} episodes...")
        generate_offline_episodes(n_episodes=generate_episodes, source="backtest")

    # Step 2: Load replay buffer
    env = TradingEnv()
    buffer = ReplayBuffer(state_dim=TOTAL_STATE_DIM, action_dim=DIM_ACTION)

    # Try DuckDB first, then SQLite
    loaded = buffer.load_from_duckdb(TOTAL_STATE_DIM, DIM_ACTION)
    if loaded == 0:
        loaded = buffer.load_from_sqlite(TOTAL_STATE_DIM, DIM_ACTION)

    if loaded < IQL_CONFIG["batch_size"]:
        logger.error(f"[IQL] Insufficient data: {loaded} < {IQL_CONFIG['batch_size']}")
        return {"error": "insufficient data", "loaded": loaded}

    # Step 3: Train
    action_low, action_high = env.get_action_bounds()
    trainer = IQLTrainer(
        state_dim=TOTAL_STATE_DIM,
        action_dim=DIM_ACTION,
        action_low=action_low,
        action_high=action_high,
    )

    metrics = trainer.train(buffer, n_epochs=n_epochs)

    if "error" in metrics:
        return metrics

    # Step 4: Save
    trainer.save()
    trainer.log_training_run(metrics)

    return metrics


# ===========================================================================
# CLI
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description="IQL Pre-training — Phase 26 Sprint 2 (7B)")
    parser.add_argument("--train", action="store_true", help="Train IQL from replay buffer")
    parser.add_argument("--generate-episodes", type=int, default=0,
                       help="Generate N episodes before training")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs")
    parser.add_argument("--status", action="store_true", help="Show training status")

    args = parser.parse_args()

    if args.status:
        conn = get_db_connection(AI_DB_PATH)
        try:
            rows = conn.execute("""
                SELECT * FROM rl_checkpoints
                WHERE model_name = 'iql_sizing'
                ORDER BY created_at DESC LIMIT 5
            """).fetchall()
            print("\nIQL Training Runs:")
            if rows:
                for r in rows:
                    print(f"  v{r['version']} | {r['path']} | "
                          f"episodes={r['training_episodes']} | {r['created_at']}")
            else:
                print("  No training runs yet.")
            print(f"\nModel exists: {os.path.exists(IQL_MODEL_PATH)}")
        finally:
            conn.close()
        return

    if args.train or args.generate_episodes > 0:
        metrics = run_iql_training(
            generate_episodes=args.generate_episodes,
            n_epochs=args.epochs,
        )
        if "error" not in metrics:
            print(f"\nIQL training complete:")
            print(f"  V_loss: {metrics.get('final_v_loss', 'N/A'):.4f}")
            print(f"  Q_loss: {metrics.get('final_q_loss', 'N/A'):.4f}")
            print(f"  Policy_loss: {metrics.get('final_policy_loss', 'N/A'):.4f}")
            print(f"  Model: {IQL_MODEL_PATH}")
        else:
            print(f"Error: {metrics['error']}")
        return

    parser.print_help()


if __name__ == "__main__":
    main()
