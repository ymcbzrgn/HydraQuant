"""
Phase 26: CatBoost Classifier for FreqAI

Classification variant for directional prediction (bullish/bearish/neutral).
Native CatBoost probability calibration — NO Platt scaling needed.

Shares all Phase 26 enhancements with CatBoostRegressor:
  - Feature noise injection (training-serving skew mitigation)
  - Native embedding_features (TTM, path signatures)
  - Ordered boosting (target leakage prevention)
"""
import logging
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier as _CatBoostClassifier, Pool

from freqtrade.freqai.base_models.BaseClassifierModel import BaseClassifierModel
from freqtrade.freqai.data_kitchen import FreqaiDataKitchen


logger = logging.getLogger(__name__)


class CatBoostClassifier(BaseClassifierModel):
    """
    CatBoost classification model for FreqAI.
    Output: class probabilities (natively well-calibrated — no Platt scaling needed).

    Key for Phase 26 Dual-Axis Calibration (Novel Contribution #6):
    CatBoost probability = P(win) → sizing
    CQR interval = uncertainty range → sizing modifier
    Together: sizing = P(win) × (1 / interval_width)
    """

    def fit(self, data_dictionary: dict, dk: FreqaiDataKitchen, **kwargs) -> Any:
        X_train = data_dictionary["train_features"]
        y_train = data_dictionary["train_labels"].to_numpy()[:, 0]
        train_weights = data_dictionary["train_weights"]

        # Test set
        if self.freqai_info.get("data_split_parameters", {}).get("test_size", 0.1) == 0:
            eval_set = None
        else:
            eval_set = Pool(
                data_dictionary["test_features"],
                label=data_dictionary["test_labels"].to_numpy()[:, 0],
                weight=data_dictionary["test_weights"],
            )

        # Feature noise injection
        noise_config = self.freqai_info.get("feature_noise_injection", {})
        if noise_config.get("enabled", False):
            noise_pct = noise_config.get("noise_pct", 0.01)
            n_copies = noise_config.get("augmentation_copies", 3)
            X_train, y_train, train_weights = self._inject_noise(
                X_train, y_train, train_weights, noise_pct, n_copies
            )
            logger.info(
                f"[CatBoostCls] Noise injection: {n_copies} copies "
                f"→ {len(X_train)} training samples"
            )

        # Embedding features
        embedding_prefix = self.freqai_info.get("embedding_features_prefix", "")
        embedding_features_indices = None
        if embedding_prefix:
            embedding_cols = [
                i for i, col in enumerate(X_train.columns)
                if col.startswith(embedding_prefix)
            ]
            if embedding_cols:
                embedding_features_indices = embedding_cols

        # Categorical features
        cat_features = [
            i for i, col in enumerate(X_train.columns)
            if X_train[col].dtype == "object" or X_train[col].dtype.name == "category"
        ]

        # Model params
        model_params = self.model_training_parameters.copy()
        model_params.setdefault("iterations", 1000)
        model_params.setdefault("learning_rate", 0.05)
        model_params.setdefault("depth", 6)
        model_params.setdefault("l2_leaf_reg", 3.0)
        model_params.setdefault("random_seed", 42)
        model_params.setdefault("verbose", 100)
        model_params.setdefault("task_type", "CPU")
        model_params.setdefault("auto_class_weights", "Balanced")  # handle class imbalance

        early_stopping = model_params.pop("early_stopping_rounds", 50)
        if eval_set is not None:
            model_params["early_stopping_rounds"] = early_stopping

        model = _CatBoostClassifier(**model_params)

        train_pool = Pool(
            X_train,
            label=y_train,
            weight=train_weights,
            cat_features=cat_features if cat_features else None,
            embedding_features=embedding_features_indices,
        )

        init_model = self.get_init_model(dk.pair)

        model.fit(
            train_pool,
            eval_set=eval_set,
            init_model=init_model,
            log_cout=logging.getLogger("catboost_internal"),
        )

        # Log feature importance
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            feature_names = X_train.columns.tolist()
            top_10 = sorted(
                zip(feature_names, importances), key=lambda x: x[1], reverse=True
            )[:10]
            logger.info(
                f"[CatBoostCls] Top 10 features: "
                + ", ".join(f"{name}={imp:.1f}" for name, imp in top_10)
            )

        return model

    @staticmethod
    def _inject_noise(X, y, weights, noise_pct=0.01, n_copies=3):
        """Same noise injection as CatBoostRegressor — prevents threshold overfitting."""
        augmented_X = [X]
        augmented_y = [y]
        augmented_w = [weights]

        numeric_cols = X.select_dtypes(include=[np.number]).columns

        for _ in range(n_copies):
            noisy = X.copy()
            noise = np.random.normal(1.0, noise_pct, (len(X), len(numeric_cols)))
            noisy[numeric_cols] = noisy[numeric_cols].values * noise
            augmented_X.append(noisy)
            augmented_y.append(y.copy())
            augmented_w.append(weights.copy())

        return (
            pd.concat(augmented_X, ignore_index=True),
            np.concatenate(augmented_y),
            pd.concat(augmented_w, ignore_index=True),
        )
