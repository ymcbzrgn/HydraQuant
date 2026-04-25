"""
dt_inference.py — Decision Transformer inference wrapper (T13).

Phase 27 trained DT LoRA checkpoints land in user_data/models/dt_lora_<ts>.pt
weekly via scheduler. Until this module existed, those checkpoints were
WRITE-ONLY — inference path was missing and trinity_fusion stayed dead.

This wrapper closes the loop:
  - load_latest_checkpoint(): finds and loads the freshest dt_lora_*.pt
  - predict(state): runs the model and returns a (action_vec, q_value)
    pair compatible with trinity_fusion.update_rl_action()
  - get_dt_inference(): per-process singleton

Failure semantics: if peft / transformers / checkpoints are missing the
predictor returns a NEUTRAL action (zeros) with q_value=0.0 — callers can
treat that as "no opinion" and not pollute the fusion.

Action vector convention (4 dims, [-1..+1]):
  [0] sizing_multiplier  in [-1, +1]  → mapped to 0.5..1.5 by consumer
  [1] confidence_threshold delta in [-0.2, +0.2]
  [2] side_bias  in [-1, +1]  (-1 short, +1 long)
  [3] hold_horizon hint in [-1, +1]
"""
from __future__ import annotations

import logging
import os
import threading
from typing import Any, Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

ACTION_DIM = 4
_NEUTRAL_ACTION = np.zeros(ACTION_DIM, dtype=np.float32)


def _state_to_vector(state: Dict[str, Any]) -> np.ndarray:
    """Compact state vector from HydraSizer's signal context.

    The trained DT consumed (return-to-go, state, action) tuples where
    state was a tokenised `_tokenize_state` of the ai_decisions row.
    For inference we need the same shape — we approximate with a fixed
    8-dim numerical vector that mirrors the most informative columns.
    """
    out = np.zeros(8, dtype=np.float32)
    out[0] = float(state.get("confidence", 0.5))
    out[1] = 1.0 if str(state.get("signal", "NEUTRAL")).upper().startswith("BULL") else (
        -1.0 if str(state.get("signal", "NEUTRAL")).upper().startswith("BEAR") else 0.0
    )
    out[2] = float(state.get("regime_id", 0)) / 6.0  # 6 regimes
    out[3] = float(state.get("fng", 50.0)) / 100.0
    out[4] = float(state.get("drawdown_pct", 0.0)) / 20.0
    out[5] = float(state.get("balance_vs_peak", 1.0))
    out[6] = float(state.get("organism_health", 0.5))
    out[7] = float(state.get("hour_of_day", 12)) / 24.0
    return out


class DTInference:
    """Wraps the latest DT LoRA checkpoint for online inference."""

    def __init__(self) -> None:
        self._model = None
        self._tokenizer = None
        self._loaded_checkpoint: Optional[str] = None
        self._load_lock = threading.Lock()
        self._last_load_attempt: float = 0.0
        self._load_cooldown_s: float = 600.0  # don't retry failed load for 10 min

    def _try_load(self) -> None:
        """Attempt to load the latest DT LoRA checkpoint.

        Idempotent under contention. Cooldown protects against repeated
        load-storms when transformers/peft is unavailable.
        """
        import time as _time
        now = _time.time()
        if self._model is not None:
            return
        if (now - self._last_load_attempt) < self._load_cooldown_s and self._last_load_attempt > 0:
            return
        with self._load_lock:
            if self._model is not None:
                return
            self._last_load_attempt = _time.time()
            try:
                from decision_transformer import latest_checkpoint
                ckpt = latest_checkpoint()
                if not ckpt:
                    logger.debug("[DTInference] no checkpoint yet — neutral mode")
                    return
                # GPT-2 + LoRA load. We import lazily to avoid heavy deps
                # at module import time.
                import torch
                from transformers import GPT2LMHeadModel, GPT2Tokenizer
                base_model = GPT2LMHeadModel.from_pretrained("gpt2")
                tok = GPT2Tokenizer.from_pretrained("gpt2")
                if tok.pad_token is None:
                    tok.pad_token = tok.eos_token
                state = torch.load(ckpt, map_location="cpu", weights_only=False)
                # PEFT LoRA path
                try:
                    from peft import PeftModel, LoraConfig, get_peft_model
                    cfg = state.get("lora_config")
                    if cfg is not None:
                        if isinstance(cfg, dict):
                            cfg = LoraConfig(**cfg)
                        base_model = get_peft_model(base_model, cfg)
                    if "lora_state" in state:
                        base_model.load_state_dict(state["lora_state"], strict=False)
                except Exception:
                    # Head-only fallback path (decision_transformer head_state)
                    if "head_state" in state:
                        base_model.lm_head.load_state_dict(state["head_state"], strict=False)
                base_model.eval()
                self._model = base_model
                self._tokenizer = tok
                self._loaded_checkpoint = ckpt
                logger.info(f"[DTInference] loaded {os.path.basename(ckpt)}")
            except Exception as e:
                logger.debug(f"[DTInference] load failed (will retry in {self._load_cooldown_s:.0f}s): {e}")
                self._model = None
                self._tokenizer = None

    def predict(self, state: Dict[str, Any]) -> Tuple[np.ndarray, float]:
        """Run DT on `state` and return (action_vec, q_value).

        FIX-C3 (2026-04-25): the previous decode (topk(softmax) → centered)
        produced rank-preserved structured noise — always [+,+,-,-] with
        magnitudes <0.05, regardless of state. Audit caught this. Until
        a proper action-regression head is trained, return NEUTRAL — sending
        noise into trinity_fusion would be worse than no signal.

        When `_action_head` is detected on the loaded checkpoint (future
        sprint adds it), inference produces a real action vector.
        """
        self._try_load()
        if self._model is None or self._tokenizer is None:
            return _NEUTRAL_ACTION.copy(), 0.0
        # No dedicated action-regression head yet → honest neutral.
        # When `decision_transformer.py` ships a real action head the
        # check below activates.
        if not getattr(self._model, "_action_head", None):
            return _NEUTRAL_ACTION.copy(), 0.0
        try:
            import torch
            sv = _state_to_vector(state)
            txt = " ".join(f"{v:.4f}" for v in sv) + " ACTION"
            ids = self._tokenizer(txt, return_tensors="pt", truncation=True,
                                   max_length=128).input_ids
            with torch.no_grad():
                hidden = self._model.transformer(ids).last_hidden_state[0, -1, :]
                action_logits = self._model._action_head(hidden)
                action = torch.tanh(action_logits).cpu().numpy().astype(np.float32)
                if action.size != ACTION_DIM:
                    fixed = np.zeros(ACTION_DIM, dtype=np.float32)
                    fixed[:min(action.size, ACTION_DIM)] = action[:ACTION_DIM]
                    action = fixed
                q = float(np.linalg.norm(action) / np.sqrt(ACTION_DIM))
            return action, q
        except Exception as e:
            logger.debug(f"[DTInference] predict failed: {e}")
            return _NEUTRAL_ACTION.copy(), 0.0

    def has_useful_signal(self) -> bool:
        """True only when both model AND action_head are loaded."""
        self._try_load()
        return (
            self._model is not None
            and getattr(self._model, "_action_head", None) is not None
        )

    def has_model(self) -> bool:
        self._try_load()
        return self._model is not None


# ─── Singleton ──────────────────────────────────────────────────────────────
_instance: Optional[DTInference] = None
_instance_lock = threading.Lock()


def get_dt_inference() -> DTInference:
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = DTInference()
    return _instance
