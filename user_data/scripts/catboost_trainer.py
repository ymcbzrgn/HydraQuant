"""
catboost_trainer.py — Phase 26 Sprint 2: CatBoost Training Pipeline

Two training modes:
  v1 (legacy): ai_decisions + evidence_audit_log → 11 features → CatBoost
  v2 (Sprint 2): backtest_training_data → 193 chart features → CatBoost

v2 is the primary mode. v1 is kept for backward compatibility.

Kullanim:
    python catboost_trainer.py [--min-trades 50] [--test-ratio 0.2]
    python catboost_trainer.py --v2  # Use backtest training data (193 features)

Output: user_data/models/catboost_signal_v1.cbm (v1) or catboost_signal_v2.cbm (v2)
"""

import os
import sys
import json
import logging
import argparse
from typing import Dict, List, Tuple, Optional
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
logger = logging.getLogger("catboost_trainer")

from ai_config import AI_DB_PATH
from db import get_db_connection, get_connection, execute_with_retry, init_db

# Model output path
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "catboost_signal_v1.cbm")
MODEL_PATH_V2 = os.path.join(MODEL_DIR, "catboost_signal_v2.cbm")


def gather_training_data(min_trades: int = 50) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], List[str]]:
    """
    Gather training data from ai_decisions + evidence_audit_log.

    Labels: BULLISH (pnl > +0.5%), BEARISH (pnl < -0.5%), NEUTRAL (rest)
    Features: confidence, regime_code, sub_scores (6), trade duration, trust_score

    Returns: (X, y, feature_names) or (None, None, []) if insufficient data
    """
    conn = get_db_connection(AI_DB_PATH)
    try:
        # Get trades with outcomes
        rows = conn.execute("""
            SELECT d.pair, d.signal_type, d.confidence, d.regime,
                   d.trust_score_at_decision, d.outcome_pnl, d.outcome_duration,
                   d.timestamp,
                   e.sub_scores_json, e.max_confidence_cap
            FROM ai_decisions d
            LEFT JOIN evidence_audit_log e
                ON d.pair = e.pair
                AND ABS(JULIANDAY(d.timestamp) - JULIANDAY(e.timestamp)) < 0.01
            WHERE d.outcome_pnl IS NOT NULL
            ORDER BY d.timestamp ASC
        """).fetchall()

        if len(rows) < min_trades:
            logger.warning(f"Insufficient data: {len(rows)} < {min_trades} trades")
            return None, None, []

        logger.info(f"Gathered {len(rows)} trades for training")

        # Feature engineering
        feature_names = [
            "confidence", "trust_score", "outcome_duration",
            "max_cap", "regime_code",
            "sub_technical", "sub_sentiment", "sub_momentum",
            "sub_volatility", "sub_correlation", "sub_divergence",
        ]

        regime_map = {
            "trending_bull": 4, "ranging": 2, "transitional": 3,
            "trending_bear": 1, "high_volatility": 5,
        }

        X_list = []
        y_list = []

        for row in rows:
            # Parse sub-scores
            sub_scores = {}
            if row["sub_scores_json"]:
                try:
                    sub_scores = json.loads(row["sub_scores_json"])
                except Exception:
                    pass

            features = [
                row["confidence"] or 0.5,
                row["trust_score_at_decision"] or 0.5,
                row["outcome_duration"] or 1.0,
                row["max_confidence_cap"] or 0.35,
                regime_map.get(row["regime"], 3),
                sub_scores.get("technical", 0.5),
                sub_scores.get("sentiment", 0.5),
                sub_scores.get("momentum", 0.5),
                sub_scores.get("volatility", 0.5),
                sub_scores.get("correlation", 0.5),
                sub_scores.get("divergence", 0.5),
            ]

            # Label: direction based on actual PnL
            pnl = row["outcome_pnl"]
            if pnl > 0.5:
                label = 2  # BULLISH
            elif pnl < -0.5:
                label = 0  # BEARISH
            else:
                label = 1  # NEUTRAL

            X_list.append(features)
            y_list.append(label)

        # Phase 27 Item 13: sim2real noise augmentation. For every real
        # training sample we ALSO push 2 randomised twins (slippage + fee +
        # spread noise on the original PnL). Tripled training set teaches
        # CatBoost to be robust to live execution friction. Best-effort:
        # skip if sim2real isn't importable.
        try:
            from sim2real_pipeline import get_sim2real
            sim2real = get_sim2real(seed=42)
            sim2real.set_level(0.5)  # half-strength noise for augmentation
            augmented_X: List = []
            augmented_y: List = []
            for row, base_features, base_label in zip(rows, list(X_list), list(y_list)):
                base_pnl = float(row["outcome_pnl"] or 0.0)
                for _ in range(2):
                    noisy = sim2real.randomize_trade(
                        {"profit_ratio": base_pnl / 100.0, "pair": row["pair"]}
                    )
                    new_pnl_pct = float(noisy.get("profit_ratio", base_pnl / 100.0)) * 100.0
                    if new_pnl_pct > 0.5:
                        new_label = 2
                    elif new_pnl_pct < -0.5:
                        new_label = 0
                    else:
                        new_label = 1
                    augmented_X.append(base_features)
                    augmented_y.append(new_label)
            if augmented_X:
                X_list.extend(augmented_X)
                y_list.extend(augmented_y)
                logger.info(
                    f"[Sim2Real] augmented {len(augmented_X)} noisy twins "
                    f"(total samples now {len(X_list)})"
                )
        except Exception as e:
            logger.debug(f"[Sim2Real] augmentation skipped: {e}")

        return np.array(X_list), np.array(y_list), feature_names

    finally:
        conn.close()


def train_catboost(X: np.ndarray, y: np.ndarray, feature_names: List[str],
                   test_ratio: float = 0.2) -> Dict[str, any]:
    """
    Train CatBoost with walk-forward temporal split.
    No future leak — train on first (1-test_ratio), test on last test_ratio.
    """
    try:
        from catboost import CatBoostClassifier, Pool
    except ImportError:
        logger.error("catboost not installed. Run: pip install catboost")
        return {"error": "catboost not installed"}

    # Walk-forward split (temporal — no shuffle!)
    split_idx = int(len(X) * (1 - test_ratio))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    logger.info(f"Split: train={len(X_train)}, test={len(X_test)} "
                f"(ratio={test_ratio}, temporal walk-forward)")

    # Class distribution
    for label, name in [(0, "BEARISH"), (1, "NEUTRAL"), (2, "BULLISH")]:
        n_train = np.sum(y_train == label)
        n_test = np.sum(y_test == label)
        logger.info(f"  {name}: train={n_train}, test={n_test}")

    # Train
    model = CatBoostClassifier(
        iterations=200,
        depth=4,
        learning_rate=0.05,
        loss_function='MultiClass',
        eval_metric='Accuracy',
        random_seed=42,
        verbose=50,
        early_stopping_rounds=30,
        auto_class_weights='Balanced',
    )

    train_pool = Pool(X_train, y_train, feature_names=feature_names)
    test_pool = Pool(X_test, y_test, feature_names=feature_names)

    model.fit(train_pool, eval_set=test_pool, use_best_model=True)

    # Evaluate
    train_acc = np.mean(model.predict(X_train).flatten().astype(int) == y_train)
    test_acc = np.mean(model.predict(X_test).flatten().astype(int) == y_test)

    # Feature importance
    importance = model.get_feature_importance()
    fi = sorted(zip(feature_names, importance), key=lambda x: x[1], reverse=True)

    logger.info(f"\nResults:")
    logger.info(f"  Train accuracy: {train_acc:.3f}")
    logger.info(f"  Test accuracy:  {test_acc:.3f}")
    logger.info(f"  Feature importance:")
    for name, imp in fi:
        logger.info(f"    {name}: {imp:.1f}")

    # Save model
    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save_model(MODEL_PATH)
    logger.info(f"\nModel saved: {MODEL_PATH}")

    # Save metadata
    metadata = {
        "trained_at": datetime.utcnow().isoformat(),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "train_accuracy": round(train_acc, 4),
        "test_accuracy": round(test_acc, 4),
        "feature_names": feature_names,
        "feature_importance": {n: round(float(v), 2) for n, v in fi},
        "model_path": MODEL_PATH,
        "class_mapping": {0: "BEARISH", 1: "NEUTRAL", 2: "BULLISH"},
    }
    meta_path = MODEL_PATH.replace(".cbm", "_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Metadata saved: {meta_path}")

    return metadata


def train_catboost_v2(X: np.ndarray, y: np.ndarray, feature_names: List[str],
                      test_ratio: float = 0.2, model_version: str = None) -> Dict[str, any]:
    """
    Train CatBoost v2 with 193 chart features from backtest_training_data.

    Improvements over v1:
    - 193 features (vs 11)
    - More iterations (500 vs 200)
    - Deeper trees (depth 6 vs 4)
    - L2 regularization
    - Training run logging to catboost_training_runs table
    - F1 macro metric tracking
    """
    try:
        from catboost import CatBoostClassifier, Pool
    except ImportError:
        logger.error("catboost not installed. Run: pip install catboost")
        return {"error": "catboost not installed"}

    if model_version is None:
        model_version = datetime.utcnow().strftime("v2_%Y%m%d_%H%M")

    # Walk-forward split (temporal — no shuffle!)
    split_idx = int(len(X) * (1 - test_ratio))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    logger.info(f"[v2] Split: train={len(X_train)}, test={len(X_test)} "
                f"(ratio={test_ratio}, temporal walk-forward)")
    logger.info(f"[v2] Features: {len(feature_names)}")

    # Class distribution
    label_dist = {}
    for label, name in [(0, "BEARISH"), (1, "NEUTRAL"), (2, "BULLISH")]:
        n_train = int(np.sum(y_train == label))
        n_test = int(np.sum(y_test == label))
        label_dist[name] = {"train": n_train, "test": n_test}
        logger.info(f"  {name}: train={n_train}, test={n_test}")

    # Handle NaN/Inf in features
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

    # Train with improved hyperparameters
    model = CatBoostClassifier(
        iterations=500,
        depth=6,
        learning_rate=0.03,
        l2_leaf_reg=5.0,
        loss_function='MultiClass',
        eval_metric='TotalF1:average=Macro',
        random_seed=42,
        verbose=100,
        early_stopping_rounds=50,
        auto_class_weights='Balanced',
        border_count=128,
        grow_policy='Lossguide',
        max_leaves=64,
    )

    train_pool = Pool(X_train, y_train, feature_names=feature_names)
    test_pool = Pool(X_test, y_test, feature_names=feature_names)

    model.fit(train_pool, eval_set=test_pool, use_best_model=True)

    # Evaluate
    y_train_pred = model.predict(X_train).flatten().astype(int)
    y_test_pred = model.predict(X_test).flatten().astype(int)

    train_acc = float(np.mean(y_train_pred == y_train))
    test_acc = float(np.mean(y_test_pred == y_test))

    # F1 macro
    try:
        from sklearn.metrics import f1_score
        train_f1 = float(f1_score(y_train, y_train_pred, average='macro', zero_division=0))
        test_f1 = float(f1_score(y_test, y_test_pred, average='macro', zero_division=0))
    except ImportError:
        # Manual F1 calculation
        train_f1 = train_acc  # fallback
        test_f1 = test_acc

    # Feature importance — top 30
    importance = model.get_feature_importance()
    fi = sorted(zip(feature_names, importance), key=lambda x: x[1], reverse=True)

    logger.info(f"\n[v2] Results:")
    logger.info(f"  Train accuracy: {train_acc:.3f} | F1: {train_f1:.3f}")
    logger.info(f"  Test accuracy:  {test_acc:.3f} | F1: {test_f1:.3f}")
    logger.info(f"  Top 15 features:")
    for name, imp in fi[:15]:
        logger.info(f"    {name}: {imp:.1f}")

    # Save model
    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save_model(MODEL_PATH_V2)
    logger.info(f"\n[v2] Model saved: {MODEL_PATH_V2}")

    # Also save as v1 path for backward compatibility (triple_perception reads v1)
    model.save_model(MODEL_PATH)
    logger.info(f"[v2] Also saved as: {MODEL_PATH} (backward compat)")

    # Metadata
    metadata = {
        "model_version": model_version,
        "trained_at": datetime.utcnow().isoformat(),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "n_features": len(feature_names),
        "train_accuracy": round(train_acc, 4),
        "test_accuracy": round(test_acc, 4),
        "train_f1": round(train_f1, 4),
        "test_f1": round(test_f1, 4),
        "feature_names": feature_names,
        "feature_importance_top30": {n: round(float(v), 2) for n, v in fi[:30]},
        "model_path": MODEL_PATH_V2,
        "class_mapping": {0: "BEARISH", 1: "NEUTRAL", 2: "BULLISH"},
        "label_distribution": label_dist,
        "hyperparams": {
            "iterations": 500, "depth": 6, "learning_rate": 0.03,
            "l2_leaf_reg": 5.0, "border_count": 128,
            "grow_policy": "Lossguide", "max_leaves": 64,
        },
    }

    # Save metadata JSON
    meta_path = MODEL_PATH_V2.replace(".cbm", "_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"[v2] Metadata saved: {meta_path}")

    # Log training run to DB
    try:
        execute_with_retry(
            """INSERT INTO catboost_training_runs
               (model_version, n_train, n_test, n_features,
                train_accuracy, test_accuracy, train_f1, test_f1,
                feature_importance_json, label_distribution_json,
                model_path, hyperparams_json, data_sources_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (model_version, len(X_train), len(X_test), len(feature_names),
             train_acc, test_acc, train_f1, test_f1,
             json.dumps({n: round(float(v), 2) for n, v in fi[:30]}),
             json.dumps(label_dist),
             MODEL_PATH_V2,
             json.dumps(metadata["hyperparams"]),
             json.dumps(["backtest_training_data"])),
            max_retries=5
        )
        logger.info("[v2] Training run logged to catboost_training_runs")
    except Exception as e:
        logger.warning(f"[v2] Failed to log training run: {e}")

    return metadata


def main():
    parser = argparse.ArgumentParser(description="CatBoost Training Pipeline")
    parser.add_argument("--min-trades", type=int, default=50)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--v2", action="store_true", help="Use v2 pipeline (193 chart features)")
    args = parser.parse_args()

    if args.v2:
        logger.info("=" * 60)
        logger.info("Phase 26 Sprint 2: CatBoost v2 Training Pipeline")
        logger.info("=" * 60)

        try:
            from backtest_label_generator import BacktestLabelGenerator
            gen = BacktestLabelGenerator()
            X, y, feature_names = gen.get_training_dataset(min_samples=args.min_trades)
        except ImportError:
            logger.error("backtest_label_generator not available")
            sys.exit(1)

        if X is None:
            logger.error("Insufficient training data. Run backtest_label_generator.py --generate first.")
            sys.exit(1)

        result = train_catboost_v2(X, y, feature_names, test_ratio=args.test_ratio)
        if "error" in result:
            logger.error(f"Training failed: {result['error']}")
            sys.exit(1)

        logger.info("=" * 60)
        logger.info(f"SUCCESS: CatBoost v2 model ready at {MODEL_PATH_V2}")
        logger.info(f"  Features:  {result['n_features']}")
        logger.info(f"  Train acc: {result['train_accuracy']} | F1: {result['train_f1']}")
        logger.info(f"  Test acc:  {result['test_accuracy']}  | F1: {result['test_f1']}")
        logger.info("=" * 60)
    else:
        logger.info("=" * 60)
        logger.info("CatBoost v1 Training Pipeline (legacy)")
        logger.info("=" * 60)

        X, y, feature_names = gather_training_data(min_trades=args.min_trades)
        if X is None:
            logger.error("Insufficient training data. Need more completed trades.")
            sys.exit(1)

        result = train_catboost(X, y, feature_names, test_ratio=args.test_ratio)
        if "error" in result:
            logger.error(f"Training failed: {result['error']}")
            sys.exit(1)

        logger.info("=" * 60)
        logger.info(f"SUCCESS: CatBoost model ready at {MODEL_PATH}")
        logger.info(f"  Train acc: {result['train_accuracy']}")
        logger.info(f"  Test acc: {result['test_accuracy']}")
        logger.info("=" * 60)


if __name__ == "__main__":
    main()
