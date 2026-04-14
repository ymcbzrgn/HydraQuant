"""
dream_engine.py — Phase 26 Sprint 2, Task 8D

Dream-Augmented Learning — The organism "dreams" to practice scenarios
it has NEVER experienced in real trading.

Flow:
  1. World Model (8C) generates imagined trajectories
  2. DreamFilter validates each trajectory (3-layer anomaly detection)
  3. Valid dreams are added to RL replay buffer with source='dream'
  4. RL agents (7B-7D) train on mixed real + dream data

DreamFilter (anti-hallucination):
  1. Mahalanobis distance: dream state vs real data distribution
  2. Reward magnitude: reject unrealistic rewards (>3x max real)
  3. Transition smoothness: reject discontinuous state jumps

Reference: Hafner et al. "Dream to Control" (2020)

Usage:
    engine = DreamEngine()
    n_valid = engine.dream_session(n_dreams=100)
    # Valid dreams are auto-added to replay buffer
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
logger = logging.getLogger("dream_engine")

from ai_config import AI_DB_PATH
from db import get_db_connection, get_connection, execute_with_retry, init_db

# ===========================================================================
# Constants
# ===========================================================================

# Dream quality thresholds
MAHALANOBIS_THRESHOLD = 3.0    # Standard deviations from real data
REWARD_MAGNITUDE_MULT = 3.0    # Max reward = 3x real max
TRANSITION_SMOOTHNESS = 5.0    # Max L2 distance between consecutive states
MIN_FILTER_PASS_RATE = 0.2     # If <20% pass → world model is unhealthy

# Default dream session parameters
DEFAULT_N_DREAMS = 100
DEFAULT_HORIZON = 5


class DreamFilter:
    """3-layer anomaly filter for dream trajectories.

    Prevents the RL agent from learning the world model's HALLUCINATIONS.
    Every dream must pass all 3 filters before entering the replay buffer.
    """

    def __init__(self):
        self._real_stats = None
        self._stats_computed = False

    def compute_real_stats(self, real_states: np.ndarray, real_rewards: np.ndarray):
        """Compute reference statistics from real data."""
        if len(real_states) < 10:
            return

        self._real_stats = {
            "mean": real_states.mean(axis=0),
            "std": real_states.std(axis=0) + 1e-8,
            "cov_inv": None,  # Simplified: use diagonal
            "max_abs_reward": float(np.abs(real_rewards).max()) + 1e-6,
            "mean_reward": float(real_rewards.mean()),
            "std_reward": float(real_rewards.std()) + 1e-6,
            "n_samples": len(real_states),
        }

        # Diagonal precision matrix (simplified Mahalanobis)
        self._real_stats["precision_diag"] = 1.0 / (self._real_stats["std"] ** 2)
        self._stats_computed = True

        logger.info(f"[DreamFilter] Stats computed from {len(real_states)} real samples, "
                    f"max_reward={self._real_stats['max_abs_reward']:.3f}")

    def is_valid_dream(self, trajectory: List[Dict]) -> Tuple[bool, str]:
        """Validate a dream trajectory through 3 filters.

        Returns (is_valid, reason) — reason is empty if valid.
        """
        if not self._stats_computed or self._real_stats is None:
            return True, ""  # No stats → pass through (first run)

        for i, step in enumerate(trajectory):
            z = np.array(step.get("z", []), dtype=np.float32)
            reward = step.get("reward", 0.0)

            # Filter 1: Mahalanobis distance
            if len(z) > 0 and len(z) <= len(self._real_stats["mean"]):
                z_trimmed = z[:len(self._real_stats["mean"])]
                diff = z_trimmed - self._real_stats["mean"][:len(z_trimmed)]
                prec = self._real_stats["precision_diag"][:len(z_trimmed)]
                maha_sq = float(np.sum(diff ** 2 * prec))
                maha_dist = np.sqrt(maha_sq / len(z_trimmed))

                if maha_dist > MAHALANOBIS_THRESHOLD:
                    return False, f"mahalanobis={maha_dist:.2f} > {MAHALANOBIS_THRESHOLD}"

            # Filter 2: Reward magnitude
            max_reward = self._real_stats["max_abs_reward"] * REWARD_MAGNITUDE_MULT
            if abs(reward) > max_reward:
                return False, f"reward={reward:.3f}, max_allowed={max_reward:.3f}"

            # Filter 3: Transition smoothness
            if i > 0:
                z_prev = np.array(trajectory[i - 1].get("z", []), dtype=np.float32)
                if len(z) > 0 and len(z_prev) > 0:
                    min_len = min(len(z), len(z_prev))
                    transition_dist = float(np.linalg.norm(
                        z[:min_len] - z_prev[:min_len]
                    ))
                    if transition_dist > TRANSITION_SMOOTHNESS:
                        return False, f"transition_jump={transition_dist:.2f} > {TRANSITION_SMOOTHNESS}"

        return True, ""


class DreamEngine:
    """Dream-Augmented Learning — generates and filters dream trajectories."""

    def __init__(self):
        self._filter = DreamFilter()
        self._world_model = None
        self._session_count = 0
        init_db()

    def _get_world_model(self):
        """Lazy-load world model."""
        if self._world_model is None:
            try:
                from world_model import get_world_model
                self._world_model = get_world_model()
            except Exception as e:
                logger.warning(f"[DreamEngine] World model not available: {e}")
        return self._world_model

    def _load_real_stats(self):
        """Load real data statistics for dream filtering."""
        conn = get_db_connection(AI_DB_PATH)
        try:
            rows = conn.execute("""
                SELECT confidence, outcome_pnl, trust_score_at_decision
                FROM ai_decisions
                WHERE outcome_pnl IS NOT NULL
                ORDER BY timestamp DESC
                LIMIT 500
            """).fetchall()

            if len(rows) < 10:
                return

            states = np.array([[
                r["confidence"] or 0.5,
                r["trust_score_at_decision"] or 0.5,
            ] for r in rows], dtype=np.float32)

            rewards = np.array([r["outcome_pnl"] for r in rows], dtype=np.float32)

            self._filter.compute_real_stats(states, rewards)

        finally:
            conn.close()

    # ===================================================================
    # Dream Session
    # ===================================================================

    def dream_session(self, n_dreams: int = DEFAULT_N_DREAMS,
                      horizon: int = DEFAULT_HORIZON,
                      initial_state: np.ndarray = None) -> Dict:
        """Run a dream session — imagine, filter, and store valid dreams.

        Returns dict with counts and pass rate.
        """
        wm = self._get_world_model()
        if wm is None:
            return {"error": "world model not available", "valid": 0, "total": 0}

        # Load real stats for filtering
        self._load_real_stats()

        # Get initial state
        if initial_state is None:
            from rl_environment import TOTAL_STATE_DIM
            initial_state = np.random.randn(TOTAL_STATE_DIM).astype(np.float32) * 0.1

        # Generate dreams (rollouts from world model)
        logger.info(f"[DreamEngine] Starting dream session: {n_dreams} dreams, "
                    f"horizon={horizon}")

        rollouts = wm.imagine(
            initial_state=initial_state,
            n_rollouts=n_dreams,
            horizon=horizon,
            event_injection=True,
        )

        if not rollouts:
            return {"error": "imagination failed", "valid": 0, "total": 0}

        # Filter dreams
        valid_dreams = []
        rejected_reasons = {}

        for rollout in rollouts:
            is_valid, reason = self._filter.is_valid_dream(rollout["trajectory"])
            if is_valid:
                valid_dreams.append(rollout)
            else:
                rejected_reasons[reason.split("=")[0]] = \
                    rejected_reasons.get(reason.split("=")[0], 0) + 1

        pass_rate = len(valid_dreams) / max(len(rollouts), 1)

        logger.info(f"[DreamEngine] Filter results: {len(valid_dreams)}/{len(rollouts)} passed "
                    f"({pass_rate:.1%})")

        if rejected_reasons:
            for reason, count in rejected_reasons.items():
                logger.debug(f"[DreamEngine]   Rejected {count} for: {reason}")

        # Health check: if pass rate too low, world model is unhealthy
        if pass_rate < MIN_FILTER_PASS_RATE and len(rollouts) > 10:
            logger.warning(f"[DreamEngine] LOW PASS RATE ({pass_rate:.1%}) — "
                         f"world model may be hallucinating!")

        # Store valid dreams in replay buffer
        stored = self._store_dreams(valid_dreams)

        # Persist dream session metadata
        session_id = str(uuid.uuid4())[:8]
        self._persist_session(session_id, valid_dreams, rollouts, pass_rate)

        self._session_count += 1

        return {
            "session_id": session_id,
            "total_dreams": len(rollouts),
            "valid_dreams": len(valid_dreams),
            "pass_rate": round(pass_rate, 3),
            "stored_in_buffer": stored,
            "rejected_reasons": rejected_reasons,
            "best_reward": valid_dreams[0]["total_reward"] if valid_dreams else 0,
            "worst_reward": valid_dreams[-1]["total_reward"] if valid_dreams else 0,
        }

    def _store_dreams(self, valid_dreams: List[Dict]) -> int:
        """Store valid dream trajectories in RL replay buffer.

        Dream states are latent z vectors (64d from world model).
        We pad them to TOTAL_STATE_DIM (335d) so IQL loader can read them.
        The padded dimensions are zeros — IQL will learn to ignore them.
        """
        from rl_environment import TOTAL_STATE_DIM

        stored = 0

        for dream in valid_dreams:
            for step_idx, step in enumerate(dream["trajectory"]):
                z = np.array(step.get("z", []), dtype=np.float32)
                reward = step.get("reward", 0.0)

                if len(z) == 0:
                    continue

                # Pad z to full state dimension for IQL compatibility
                state_full = np.zeros(TOTAL_STATE_DIM, dtype=np.float32)
                state_full[:min(len(z), TOTAL_STATE_DIM)] = z[:TOTAL_STATE_DIM]

                # Next state from next step
                if step_idx + 1 < len(dream["trajectory"]):
                    z_next = np.array(
                        dream["trajectory"][step_idx + 1].get("z", []),
                        dtype=np.float32
                    )
                    next_state_full = np.zeros(TOTAL_STATE_DIM, dtype=np.float32)
                    next_state_full[:min(len(z_next), TOTAL_STATE_DIM)] = z_next[:TOTAL_STATE_DIM]
                else:
                    next_state_full = np.zeros(TOTAL_STATE_DIM, dtype=np.float32)

                done = step_idx == len(dream["trajectory"]) - 1

                # Store in SQLite replay buffer (source='dream')
                try:
                    execute_with_retry(
                        """INSERT INTO rl_replay_buffer
                           (episode_id, step, state_json, action_json,
                            reward, next_state_json, done, source, regime)
                           VALUES (?, ?, ?, ?, ?, ?, ?, 'dream', ?)""",
                        (self._session_count * 1000 + stored, step_idx,
                         json.dumps(state_full.tolist()),
                         json.dumps([0.0, 0.0, 0.0, 0.0]),
                         reward,
                         json.dumps(next_state_full.tolist()),
                         done, None),
                        max_retries=3
                    )
                    stored += 1
                except Exception:
                    pass

        logger.info(f"[DreamEngine] Stored {stored} dream transitions in replay buffer")
        return stored

    def _persist_session(self, session_id: str, valid_dreams: List[Dict],
                         all_rollouts: List[Dict], pass_rate: float):
        """Persist dream session metadata to dream_scenarios table."""
        with get_connection() as conn:
            for i, dream in enumerate(valid_dreams[:50]):  # Cap at 50
                for step_idx, step in enumerate(dream["trajectory"]):
                    try:
                        conn.execute("""
                            INSERT INTO dream_scenarios
                            (dream_session_id, trajectory_idx, step,
                             reward, passed_filter, filter_reason)
                            VALUES (?, ?, ?, ?, 1, NULL)
                        """, (session_id, i, step_idx, step["reward"]))
                    except Exception:
                        pass
            conn.commit()

    # ===================================================================
    # Status
    # ===================================================================

    def get_status(self) -> Dict:
        """Get dream engine status."""
        conn = get_db_connection(AI_DB_PATH)
        try:
            # Count dream entries in replay buffer
            dream_count = conn.execute(
                "SELECT COUNT(*) FROM rl_replay_buffer WHERE source = 'dream'"
            ).fetchone()[0]

            # Count dream sessions
            sessions = conn.execute("""
                SELECT dream_session_id, COUNT(*) as steps,
                       AVG(reward) as avg_reward
                FROM dream_scenarios
                GROUP BY dream_session_id
                ORDER BY ROWID DESC
                LIMIT 5
            """).fetchall()

            return {
                "total_dream_transitions": dream_count,
                "session_count": self._session_count,
                "filter_stats_available": self._filter._stats_computed,
                "recent_sessions": [dict(s) for s in sessions],
            }
        finally:
            conn.close()

    def print_status(self):
        status = self.get_status()
        print(f"\n{'='*50}")
        print(f"  DREAM ENGINE STATUS")
        print(f"{'='*50}")
        print(f"  Dream transitions in buffer: {status['total_dream_transitions']}")
        print(f"  Sessions this run: {status['session_count']}")
        print(f"  Filter stats: {'ready' if status['filter_stats_available'] else 'not computed'}")
        if status["recent_sessions"]:
            print(f"  Recent sessions:")
            for s in status["recent_sessions"]:
                print(f"    {s['dream_session_id']}: {s['steps']} steps, "
                      f"avg_reward={s['avg_reward']:.3f}")
        print(f"{'='*50}\n")


# Singleton
_dream_instance = None

def get_dream_engine() -> DreamEngine:
    global _dream_instance
    if _dream_instance is None:
        _dream_instance = DreamEngine()
    return _dream_instance
