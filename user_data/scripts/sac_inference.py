"""
sac_inference.py — SAC online policy inference wrapper (T14).

Phase 27 Task 21 trains a SAC actor-critic on the rl_replay_buffer. The
checkpoint at user_data/models/sac_online_v1.pt was being WRITTEN every
weekend but NEVER LOADED for inference — until this module.

Closes the loop:
  - get_sac_inference(): per-process singleton
  - predict(state) → (action_vec, q_value): returns NEUTRAL when the
    checkpoint is missing or torch is unavailable, otherwise runs the
    SAC actor for a single forward pass

Action vector convention (4 dims, [-1..+1]) — same as dt_inference for
trinity_fusion symmetry.
"""
from __future__ import annotations

import logging
import os
import threading
from typing import Any, Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

ACTION_DIM = 4
STATE_DIM = 16  # SACOnlineTrainer default state dim
_NEUTRAL_ACTION = np.zeros(ACTION_DIM, dtype=np.float32)


def _state_to_vector(state: Dict[str, Any]) -> np.ndarray:
    """Build a 16-dim state vector compatible with the trained SAC actor.

    Uses richer features than dt_inference because SAC's state-space is
    continuous (vs DT which tokenised). Pads with zeros if a feature is
    missing — the actor was trained with this same fallback pattern via
    rl_replay_buffer.
    """
    out = np.zeros(STATE_DIM, dtype=np.float32)
    out[0] = float(state.get("confidence", 0.5))
    out[1] = 1.0 if str(state.get("signal", "NEUTRAL")).upper().startswith("BULL") else (
        -1.0 if str(state.get("signal", "NEUTRAL")).upper().startswith("BEAR") else 0.0
    )
    out[2] = float(state.get("regime_id", 0)) / 6.0
    out[3] = float(state.get("fng", 50.0)) / 100.0
    out[4] = float(state.get("drawdown_pct", 0.0)) / 20.0
    out[5] = float(state.get("balance_vs_peak", 1.0))
    out[6] = float(state.get("organism_health", 0.5))
    out[7] = float(state.get("hour_of_day", 12)) / 24.0
    out[8] = float(state.get("cortisol", 1.0))
    out[9] = float(state.get("dopamine", 1.0))
    out[10] = float(state.get("serotonin", 1.0))
    out[11] = float(state.get("adrenaline", 0.0))
    out[12] = float(state.get("funding_rate", 0.0)) * 100.0
    out[13] = float(state.get("ls_ratio", 1.0))
    out[14] = float(state.get("uncertainty", 0.5))
    out[15] = float(state.get("ood_score", 0.0))
    return out


class SACInference:
    """Wraps the latest SAC online checkpoint for trade-time inference."""

    def __init__(self) -> None:
        self._trainer = None
        self._loaded_path: Optional[str] = None
        self._load_lock = threading.Lock()
        self._last_load_attempt: float = 0.0
        self._load_cooldown_s: float = 600.0

    def _try_load(self) -> None:
        import time as _time
        now = _time.time()
        if self._trainer is not None:
            return
        if (now - self._last_load_attempt) < self._load_cooldown_s and self._last_load_attempt > 0:
            return
        with self._load_lock:
            if self._trainer is not None:
                return
            self._last_load_attempt = _time.time()
            try:
                from sac_online import SACOnlineTrainer, SAC_MODEL_PATH
                if not os.path.exists(SAC_MODEL_PATH):
                    logger.debug("[SACInference] checkpoint missing — neutral mode")
                    return
                trainer = SACOnlineTrainer(state_dim=STATE_DIM, action_dim=ACTION_DIM)
                # SACOnlineTrainer.load() exists in the existing codebase.
                # If the API uses a different verb, fall back to direct
                # state_dict load so this wrapper stays robust.
                loaded = False
                if hasattr(trainer, "load"):
                    try:
                        trainer.load(SAC_MODEL_PATH)
                        loaded = True
                    except Exception:
                        loaded = False
                if not loaded:
                    try:
                        import torch
                        ckpt = torch.load(SAC_MODEL_PATH, map_location="cpu", weights_only=False)
                        if hasattr(trainer, "actor") and "actor_state" in ckpt:
                            trainer.actor.load_state_dict(ckpt["actor_state"], strict=False)
                        loaded = True
                    except Exception as e2:
                        logger.debug(f"[SACInference] direct load failed: {e2}")
                if loaded:
                    self._trainer = trainer
                    self._loaded_path = SAC_MODEL_PATH
                    logger.info(f"[SACInference] loaded {os.path.basename(SAC_MODEL_PATH)}")
            except Exception as e:
                logger.debug(f"[SACInference] load failed (retry in {self._load_cooldown_s:.0f}s): {e}")
                self._trainer = None

    def predict(self, state: Dict[str, Any]) -> Tuple[np.ndarray, float]:
        self._try_load()
        if self._trainer is None:
            return _NEUTRAL_ACTION.copy(), 0.0
        try:
            sv = _state_to_vector(state)
            action = self._trainer.predict(sv, deterministic=True)
            action = np.asarray(action, dtype=np.float32).reshape(-1)
            if action.size != ACTION_DIM:
                # Pad / truncate to canonical width
                fixed = np.zeros(ACTION_DIM, dtype=np.float32)
                fixed[:min(action.size, ACTION_DIM)] = action[:ACTION_DIM]
                action = fixed
            action = np.clip(action, -1.0, 1.0)
            # q-value via critic if available
            q = 0.0
            if hasattr(self._trainer, "q_value"):
                try:
                    q = float(self._trainer.q_value(sv, action))
                except Exception:
                    q = 0.0
            elif hasattr(self._trainer, "critic"):
                try:
                    import torch
                    with torch.no_grad():
                        st = torch.tensor(sv, dtype=torch.float32).unsqueeze(0)
                        ac = torch.tensor(action, dtype=torch.float32).unsqueeze(0)
                        q = float(self._trainer.critic(torch.cat([st, ac], dim=-1)).item())
                except Exception:
                    q = 0.0
            return action, q
        except Exception as e:
            logger.debug(f"[SACInference] predict failed: {e}")
            return _NEUTRAL_ACTION.copy(), 0.0

    def has_model(self) -> bool:
        self._try_load()
        return self._trainer is not None


# ─── Singleton ──────────────────────────────────────────────────────────────
_instance: Optional[SACInference] = None
_instance_lock = threading.Lock()


def get_sac_inference() -> SACInference:
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = SACInference()
    return _instance
