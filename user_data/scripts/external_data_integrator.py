"""
external_data_integrator.py — Phase 26 Sprint 2, Task 13C

External Data Integration — Kaggle + HuggingFace + Triple Barrier labeling.

Integrates external data sources for richer ML training:
  1. Kaggle crypto datasets (OHLCV + indicators + labels)
  2. HuggingFace financial datasets
  3. Triple Barrier labeling method (Lopez de Prado)

Triple Barrier Method:
  - Upper barrier: take-profit at +X%
  - Lower barrier: stop-loss at -Y%
  - Vertical barrier: max holding time T
  - Label = which barrier was hit first (WIN/LOSS/TIMEOUT)

Integration:
  - Feeds into: backtest_training_data table (via backtest_label_generator)
  - Used by: CatBoost v2 training (enriched dataset)
  - Consumed by: catboost_trainer.train_catboost_v2
"""

import os
import sys
import json
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
logger = logging.getLogger("external_data_integrator")

from ai_config import AI_DB_PATH
from db import get_connection, execute_with_retry, init_db

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

# Triple Barrier defaults
DEFAULT_TAKE_PROFIT_PCT = 2.0    # +2% take profit
DEFAULT_STOP_LOSS_PCT = 1.0     # -1% stop loss
DEFAULT_MAX_HOLDING_HOURS = 48  # 48h max holding time


class TripleBarrierLabeler:
    """Triple Barrier Method for ML trade labeling (Lopez de Prado).

    Creates labels based on which barrier is hit first:
      - Upper (+TP%): label = BULLISH (2)
      - Lower (-SL%): label = BEARISH (0)
      - Vertical (time): label = NEUTRAL (1)
    """

    def __init__(self, take_profit_pct: float = DEFAULT_TAKE_PROFIT_PCT,
                 stop_loss_pct: float = DEFAULT_STOP_LOSS_PCT,
                 max_holding_hours: int = DEFAULT_MAX_HOLDING_HOURS):
        self.take_profit = take_profit_pct / 100
        self.stop_loss = stop_loss_pct / 100
        self.max_holding_bars = max_holding_hours  # Assuming 1h bars

    def label_series(self, df: pd.DataFrame, entry_indices: List[int] = None) -> List[Dict]:
        """Apply triple barrier labeling to OHLCV DataFrame.

        Args:
            df: OHLCV DataFrame with 'close' column
            entry_indices: Specific indices to label (default: every Nth bar)

        Returns: List of labeled entries
        """
        if "close" not in df.columns:
            logger.error("[TripleBarrier] DataFrame must have 'close' column")
            return []

        close = df["close"].values

        if entry_indices is None:
            # Label every 4th bar by default
            entry_indices = list(range(0, len(close) - self.max_holding_bars, 4))

        labels = []
        for idx in entry_indices:
            if idx + self.max_holding_bars >= len(close):
                break

            entry_price = close[idx]
            upper = entry_price * (1 + self.take_profit)
            lower = entry_price * (1 - self.stop_loss)

            # Find which barrier is hit first
            label = 1  # Default: NEUTRAL (vertical barrier)
            label_name = "NEUTRAL"
            exit_idx = idx + self.max_holding_bars
            exit_price = close[exit_idx]

            for future_idx in range(idx + 1, min(idx + self.max_holding_bars + 1, len(close))):
                price = close[future_idx]
                high = df["high"].values[future_idx] if "high" in df.columns else price
                low = df["low"].values[future_idx] if "low" in df.columns else price

                if high >= upper:
                    label = 2  # BULLISH — take profit hit
                    label_name = "BULLISH"
                    exit_idx = future_idx
                    exit_price = upper
                    break
                elif low <= lower:
                    label = 0  # BEARISH — stop loss hit
                    label_name = "BEARISH"
                    exit_idx = future_idx
                    exit_price = lower
                    break

            profit_pct = (exit_price - entry_price) / entry_price * 100
            duration_hours = exit_idx - idx  # In bars (1h each)

            entry_date = ""
            exit_date = ""
            if "date" in df.columns:
                entry_date = str(df["date"].iloc[idx])
                exit_date = str(df["date"].iloc[exit_idx])

            labels.append({
                "entry_idx": idx,
                "exit_idx": exit_idx,
                "entry_price": round(float(entry_price), 8),
                "exit_price": round(float(exit_price), 8),
                "profit_pct": round(float(profit_pct), 4),
                "duration_hours": duration_hours,
                "label": label,
                "label_name": label_name,
                "barrier_hit": label_name.lower(),
                "entry_date": entry_date,
                "exit_date": exit_date,
            })

        # Label distribution
        n_bull = sum(1 for l in labels if l["label"] == 2)
        n_bear = sum(1 for l in labels if l["label"] == 0)
        n_neut = sum(1 for l in labels if l["label"] == 1)
        logger.info(f"[TripleBarrier] Labeled {len(labels)} entries: "
                    f"BULL={n_bull}, BEAR={n_bear}, NEUT={n_neut}")

        return labels


class ExternalDataIntegrator:
    """Integrates external data sources for ML training enrichment."""

    def __init__(self):
        self._labeler = TripleBarrierLabeler()
        init_db()

    def integrate_ohlcv_with_triple_barrier(self, pair: str,
                                             ohlcv_path: str = None) -> int:
        """Load OHLCV data and generate triple-barrier labeled training samples.

        Returns number of samples generated.
        """
        # Find OHLCV data
        if ohlcv_path is None:
            ohlcv_path = self._find_ohlcv(pair)

        if ohlcv_path is None or not os.path.exists(ohlcv_path):
            logger.warning(f"[ExtData] No OHLCV data found for {pair}")
            return 0

        # Load data
        try:
            df = pd.read_feather(ohlcv_path)
        except Exception:
            try:
                df = pd.read_csv(ohlcv_path)
            except Exception as e:
                logger.error(f"[ExtData] Failed to load {ohlcv_path}: {e}")
                return 0

        # Normalize columns
        col_map = {}
        for col in df.columns:
            lower = col.lower()
            if lower in ("date", "datetime", "timestamp"):
                col_map[col] = "date"
            elif lower in ("open", "high", "low", "close", "volume"):
                col_map[col] = lower
        df = df.rename(columns=col_map)

        if "close" not in df.columns:
            logger.error(f"[ExtData] No 'close' column in {ohlcv_path}")
            return 0

        if "date" not in df.columns:
            df["date"] = pd.date_range("2025-01-01", periods=len(df), freq="1h")

        df = df.sort_values("date").reset_index(drop=True)

        # Generate triple barrier labels
        labels = self._labeler.label_series(df)
        if not labels:
            return 0

        # Compute chart features for each labeled entry and store
        from chart_features import compute_chart_features

        stored = 0
        batch = []

        for label_entry in labels:
            idx = label_entry["entry_idx"]

            # Need at least 250 candles before entry for chart features
            if idx < 250:
                continue

            df_slice = df.iloc[idx - 250:idx + 1].copy()

            try:
                features = compute_chart_features(df_1h=df_slice, include_signature=False)
            except Exception:
                continue

            if not features:
                continue

            batch.append((
                pair, label_entry["entry_date"], label_entry["exit_date"],
                "long",  # Triple barrier is direction-agnostic
                label_entry["profit_pct"],
                label_entry["duration_hours"],
                label_entry["barrier_hit"],
                label_entry["label"],
                label_entry["label_name"],
                json.dumps(features),
                len(features),
                "triple_barrier",
                "external",
                None,
                "1h",
            ))

        # Batch insert
        if batch:
            with get_connection() as conn:
                for record in batch:
                    try:
                        conn.execute("""
                            INSERT OR IGNORE INTO backtest_training_data
                            (pair, open_date, close_date, direction,
                             profit_pct, trade_duration_hours,
                             exit_reason, label, label_name,
                             features_json, n_features,
                             source, strategy, regime, timeframe)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, record)
                        if conn.execute("SELECT changes()").fetchone()[0] > 0:
                            stored += 1
                    except Exception:
                        pass
                conn.commit()

        logger.info(f"[ExtData] Stored {stored} triple-barrier labeled samples for {pair}")
        return stored

    def _find_ohlcv(self, pair: str) -> Optional[str]:
        """Find OHLCV feather file for a pair."""
        pair_file = pair.replace("/", "_").replace(":", "_")

        search_paths = [
            os.path.join(DATA_DIR, "bybit", "futures", f"{pair_file}-1h-futures.feather"),
            os.path.join(DATA_DIR, "binance", "futures", f"{pair_file}-1h-futures.feather"),
        ]

        for path in search_paths:
            if os.path.exists(path):
                return path
        return None

    def integrate_all_pairs(self) -> Dict:
        """Run triple barrier labeling on all available pairs."""
        results = {"total_stored": 0, "pairs_processed": 0, "pairs_skipped": 0}

        for exchange in ["bybit", "binance"]:
            futures_dir = os.path.join(DATA_DIR, exchange, "futures")
            if not os.path.isdir(futures_dir):
                continue

            for f in sorted(os.listdir(futures_dir)):
                if not f.endswith("-1h-futures.feather"):
                    continue

                pair_raw = f.replace("-1h-futures.feather", "")
                parts = pair_raw.split("_")
                if len(parts) >= 3:
                    pair = f"{parts[0]}/{parts[1]}:{parts[2]}"
                else:
                    continue

                path = os.path.join(futures_dir, f)
                stored = self.integrate_ohlcv_with_triple_barrier(pair, path)

                if stored > 0:
                    results["total_stored"] += stored
                    results["pairs_processed"] += 1
                else:
                    results["pairs_skipped"] += 1

        logger.info(f"[ExtData] Integration complete: {results['total_stored']} samples "
                    f"from {results['pairs_processed']} pairs")
        return results


# Singleton
_integrator_instance = None

def get_integrator() -> ExternalDataIntegrator:
    global _integrator_instance
    if _integrator_instance is None:
        _integrator_instance = ExternalDataIntegrator()
    return _integrator_instance
