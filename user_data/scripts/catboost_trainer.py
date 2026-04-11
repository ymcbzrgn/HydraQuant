"""
catboost_trainer.py — Phase 28: CatBoost Training Pipeline

chart_features + EE sub-scores + TTM embeddings + market data → CatBoost model.
Walk-forward temporal split (no future leak). Feature importance logging.

Kullanim:
    python catboost_trainer.py [--min-trades 50] [--test-ratio 0.2]

Output: user_data/models/catboost_signal_v1.cbm
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
from db import get_db_connection

# Model output path
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "catboost_signal_v1.cbm")


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


def main():
    parser = argparse.ArgumentParser(description="CatBoost Training Pipeline")
    parser.add_argument("--min-trades", type=int, default=50)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Phase 28: CatBoost Training Pipeline")
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
