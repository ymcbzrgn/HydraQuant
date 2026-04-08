"""
Phase 26: Deep Ensemble — 5-Model Uncertainty Estimation

Gold standard for uncertainty in finance (CFA Research Foundation 2025).
5 independently trained small MLPs → variance of predictions = uncertainty.

Key principle: models that AGREE = low uncertainty → larger position.
Models that DISAGREE = high uncertainty → smaller position.

KRİTİK BULGU: Epistemic/Aleatoric ayrımı GÜVENİLMEZ (NeurIPS 2024).
→ TOPLAM belirsizliği kullan, ayrıştırmaya çalışma.
"""

import logging
import os
import json
import numpy as np
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    from neural_organism import _p
except ImportError:
    def _p(param_id, fallback=0.5, regime="_global"):
        return fallback


class SmallMLP:
    """Tiny feed-forward network for ensemble member.
    NumPy-only implementation — no PyTorch needed for inference.
    Weights stored as JSON for easy persistence.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64, output_dim: int = 1, seed: int = 0):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Local RNG — does NOT pollute global numpy random state
        rng = np.random.default_rng(seed)

        # Xavier initialization
        scale1 = np.sqrt(2.0 / (input_dim + hidden_dim))
        scale2 = np.sqrt(2.0 / (hidden_dim + output_dim))

        self.w1 = rng.standard_normal((input_dim, hidden_dim)) * scale1
        self.b1 = np.zeros(hidden_dim)
        self.w2 = rng.standard_normal((hidden_dim, output_dim)) * scale2
        self.b2 = np.zeros(output_dim)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Forward pass: input → hidden (ReLU) → output."""
        h = np.maximum(0, x @ self.w1 + self.b1)  # ReLU
        return h @ self.w2 + self.b2

    def train_step(self, x: np.ndarray, y: np.ndarray, lr: float = 0.001) -> float:
        """Single SGD step with MSE loss. Returns loss value."""
        # Forward
        h = x @ self.w1 + self.b1
        h_relu = np.maximum(0, h)
        pred = h_relu @ self.w2 + self.b2

        # Loss
        error = pred - y.reshape(-1, self.output_dim)
        loss = float(np.mean(error ** 2))

        # Backward
        d_output = error * 2 / len(x)
        d_w2 = h_relu.T @ d_output
        d_b2 = d_output.mean(axis=0)

        d_hidden = d_output @ self.w2.T
        d_hidden[h <= 0] = 0  # ReLU gradient

        d_w1 = x.T @ d_hidden
        d_b1 = d_hidden.mean(axis=0)

        # Update
        self.w1 -= lr * d_w1
        self.b1 -= lr * d_b1
        self.w2 -= lr * d_w2
        self.b2 -= lr * d_b2

        return loss

    def to_dict(self) -> dict:
        return {
            "w1": self.w1.tolist(), "b1": self.b1.tolist(),
            "w2": self.w2.tolist(), "b2": self.b2.tolist(),
            "input_dim": self.input_dim, "hidden_dim": self.hidden_dim,
            "output_dim": self.output_dim,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SmallMLP":
        mlp = cls(d["input_dim"], d["hidden_dim"], d["output_dim"])
        mlp.w1 = np.array(d["w1"])
        mlp.b1 = np.array(d["b1"])
        mlp.w2 = np.array(d["w2"])
        mlp.b2 = np.array(d["b2"])
        return mlp


class DeepEnsemble:
    """5-model deep ensemble for uncertainty estimation.

    Each model is independently initialized and trained on bootstrapped data.
    Prediction = mean of 5 models.
    Uncertainty = variance of 5 predictions.

    Uncertainty → sizing multiplier:
        sizing = 1.0 / (1.0 + uncertainty_scale * variance)
    """

    def __init__(self, n_models: int = 5, input_dim: int = 50, hidden_dim: int = 64):
        self.n_models = n_models
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.models: List[SmallMLP] = [
            SmallMLP(input_dim, hidden_dim, seed=i * 42) for i in range(n_models)
        ]
        self._fitted = False
        self._state_path = os.path.join(
            os.path.dirname(__file__), "..", "models", "deep_ensemble_state.json"
        )
        self._load_state()

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 100,
        lr: float = 0.001,
        bootstrap: bool = True,
    ) -> Dict[str, float]:
        """Train all ensemble members.

        Args:
            X: Feature matrix (n_samples, input_dim)
            y: Target vector (n_samples,)
            epochs: Training epochs per model
            bootstrap: Use bootstrap sampling (different data per model = diversity)

        Returns:
            {"avg_loss": float, "losses": list}
        """
        if len(X) < 20:
            logger.warning("[Ensemble] Not enough data to train (<20)")
            return {"avg_loss": float("inf"), "losses": []}

        # Normalize features
        self._feature_mean = X.mean(axis=0)
        self._feature_std = X.std(axis=0) + 1e-8
        X_norm = (X - self._feature_mean) / self._feature_std

        # Normalize target
        self._target_mean = y.mean()
        self._target_std = y.std() + 1e-8
        y_norm = (y - self._target_mean) / self._target_std

        losses = []
        for i, model in enumerate(self.models):
            if bootstrap:
                # Bootstrap sample (with replacement) — each model sees different data
                indices = np.random.choice(len(X_norm), size=len(X_norm), replace=True)
                X_boot = X_norm[indices]
                y_boot = y_norm[indices]
            else:
                X_boot = X_norm
                y_boot = y_norm

            final_loss = 0.0
            for epoch in range(epochs):
                final_loss = model.train_step(X_boot, y_boot, lr=lr)

            losses.append(final_loss)
            logger.debug(f"[Ensemble] Model {i}: final_loss={final_loss:.6f}")

        self._fitted = True
        avg_loss = float(np.mean(losses))
        self._save_state()
        logger.info(f"[Ensemble] Trained {self.n_models} models, avg_loss={avg_loss:.6f}")
        return {"avg_loss": avg_loss, "losses": losses}

    def predict_with_uncertainty(self, features: np.ndarray) -> Dict[str, float]:
        """Predict with uncertainty estimation.

        Args:
            features: Feature vector (input_dim,) or (1, input_dim)

        Returns:
            {
                "mean": float — ensemble mean prediction,
                "variance": float — ensemble variance (= uncertainty),
                "std": float — ensemble standard deviation,
                "predictions": list — individual model predictions,
                "sizing_multiplier": float — uncertainty → position size scaling,
            }
        """
        if not self._fitted:
            return {
                "mean": 0.0, "variance": 0.0, "std": 0.0,
                "predictions": [], "sizing_multiplier": 1.0,
            }

        if features.ndim == 1:
            features = features.reshape(1, -1)

        # Pad or truncate to input_dim
        if features.shape[1] < self.input_dim:
            features = np.pad(features, ((0, 0), (0, self.input_dim - features.shape[1])))
        elif features.shape[1] > self.input_dim:
            features = features[:, :self.input_dim]

        # Normalize
        if hasattr(self, '_feature_mean'):
            features = (features - self._feature_mean) / self._feature_std

        # Collect predictions from all models
        preds = []
        for model in self.models:
            pred = model.predict(features)[0, 0]
            preds.append(float(pred))

        # De-normalize predictions
        if hasattr(self, '_target_mean'):
            preds = [p * self._target_std + self._target_mean for p in preds]

        mean_pred = float(np.mean(preds))
        variance = float(np.var(preds))
        std = float(np.std(preds))

        # Sizing multiplier from uncertainty
        uncertainty_scale = _p("ensemble.uncertainty_scale", 10.0)
        sizing_multiplier = 1.0 / (1.0 + uncertainty_scale * variance)
        sizing_multiplier = max(sizing_multiplier, 0.1)  # floor at 10%

        return {
            "mean": round(mean_pred, 6),
            "variance": round(variance, 6),
            "std": round(std, 6),
            "predictions": [round(p, 6) for p in preds],
            "sizing_multiplier": round(float(sizing_multiplier), 4),
        }

    def _save_state(self) -> None:
        try:
            os.makedirs(os.path.dirname(self._state_path), exist_ok=True)
            state = {
                "n_models": self.n_models,
                "input_dim": self.input_dim,
                "hidden_dim": self.hidden_dim,
                "models": [m.to_dict() for m in self.models],
                "feature_mean": self._feature_mean.tolist() if hasattr(self, '_feature_mean') else None,
                "feature_std": self._feature_std.tolist() if hasattr(self, '_feature_std') else None,
                "target_mean": self._target_mean if hasattr(self, '_target_mean') else 0.0,
                "target_std": self._target_std if hasattr(self, '_target_std') else 1.0,
            }
            with open(self._state_path, "w") as f:
                json.dump(state, f)
        except Exception as e:
            logger.debug(f"[Ensemble] Save failed: {e}")

    def _load_state(self) -> None:
        try:
            if os.path.exists(self._state_path):
                with open(self._state_path) as f:
                    state = json.load(f)
                self.models = [SmallMLP.from_dict(d) for d in state["models"]]
                self.n_models = len(self.models)
                # Sync input_dim from saved state to avoid dimension mismatch
                if self.models:
                    self.input_dim = self.models[0].input_dim
                    self.hidden_dim = self.models[0].hidden_dim
                if state.get("feature_mean"):
                    self._feature_mean = np.array(state["feature_mean"])
                    self._feature_std = np.array(state["feature_std"])
                    self._target_mean = state.get("target_mean", 0.0)
                    self._target_std = state.get("target_std", 1.0)
                    self._fitted = True
                    logger.info(f"[Ensemble] Loaded {self.n_models} models from {self._state_path}")
        except Exception as e:
            logger.debug(f"[Ensemble] No saved state: {e}")
