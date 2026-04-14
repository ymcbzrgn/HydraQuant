"""
backtest_label_generator.py — Phase 26 Sprint 2, Task 13B

End-to-end pipeline: Backtest results → chart_features (193d) → labeled training data → CatBoost.

Solves the cold-start problem: CatBoost was trained on only 382 live trades with 11 features (%50 acc).
This pipeline generates 5000-20000 labeled samples from backtests with 193 chart features.

Flow:
  1. Parse backtest ZIP/JSON files (reuses backtest_embedder)
  2. For each trade's open_date, load OHLCV and compute 193 chart features
  3. Apply Triple Barrier labeling: WIN (+1%), LOSE (-0.5%), NEUTRAL
  4. Write to SQLite backtest_training_data table
  5. Optionally enrich with live trade data from ai_decisions
  6. Trigger CatBoost retraining with the enriched dataset

Usage:
    python backtest_label_generator.py --generate          # Generate labels from backtests
    python backtest_label_generator.py --generate --retrain  # Generate + retrain CatBoost
    python backtest_label_generator.py --status             # Show dataset stats
    python backtest_label_generator.py --enrich-live        # Add live trades to training set
    python backtest_label_generator.py --run-backtest       # Run fresh backtest first, then generate

Output: backtest_training_data table in ai_data.sqlite
"""

import os
import sys
import json
import logging
import glob
import zipfile
import argparse
import subprocess
import tempfile
import time as _time
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
logger = logging.getLogger("backtest_label_generator")

from ai_config import AI_DB_PATH
from db import get_db_connection, get_connection, execute_with_retry, init_db
from chart_features import compute_chart_features

# ===========================================================================
# Constants
# ===========================================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKTEST_RESULTS_DIR = os.path.join(BASE_DIR, "backtest_results")
OHLCV_DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")

# Triple Barrier thresholds (percentage)
WIN_THRESHOLD = 1.0      # > +1% = WIN
LOSE_THRESHOLD = -0.5    # < -0.5% = LOSE
# Between = NEUTRAL

# Label mapping
LABEL_MAP = {0: "BEARISH", 1: "NEUTRAL", 2: "BULLISH"}

# Minimum candles needed before trade entry for chart_features
MIN_CANDLES_BEFORE = 250  # chart_features needs ~200 candles lookback


class BacktestLabelGenerator:
    """Generates labeled training data from backtest results + chart features."""

    def __init__(self, db_path: str = AI_DB_PATH):
        self.db_path = db_path
        self._ohlcv_cache: Dict[str, pd.DataFrame] = {}
        init_db()

    # ===================================================================
    # OHLCV Loading
    # ===================================================================

    def _find_ohlcv_file(self, pair: str, timeframe: str = "1h") -> Optional[str]:
        """Find OHLCV feather file for a pair across exchanges."""
        # Normalize pair: BTC/USDT:USDT → BTC_USDT_USDT
        pair_file = pair.replace("/", "_").replace(":", "_")

        # Search patterns — bybit futures first, then binance
        search_paths = [
            os.path.join(OHLCV_DATA_DIR, "bybit", "futures", f"{pair_file}-{timeframe}-futures.feather"),
            os.path.join(OHLCV_DATA_DIR, "bybit", "futures", f"{pair_file}-{timeframe}.feather"),
            os.path.join(OHLCV_DATA_DIR, "binance", "futures", f"{pair_file}-{timeframe}-futures.feather"),
            os.path.join(OHLCV_DATA_DIR, "binance", "futures", f"{pair_file}-{timeframe}.feather"),
        ]

        # Also try without USDT suffix: BTC_USDT → BTC_USDT_USDT
        if not pair_file.endswith("_USDT"):
            pair_file2 = pair_file + "_USDT"
            search_paths.extend([
                os.path.join(OHLCV_DATA_DIR, "bybit", "futures", f"{pair_file2}-{timeframe}-futures.feather"),
                os.path.join(OHLCV_DATA_DIR, "binance", "futures", f"{pair_file2}-{timeframe}-futures.feather"),
            ])

        for path in search_paths:
            if os.path.exists(path):
                return path

        return None

    def _load_ohlcv(self, pair: str, timeframe: str = "1h") -> Optional[pd.DataFrame]:
        """Load OHLCV data for a pair, with caching."""
        cache_key = f"{pair}_{timeframe}"
        if cache_key in self._ohlcv_cache:
            return self._ohlcv_cache[cache_key]

        path = self._find_ohlcv_file(pair, timeframe)
        if path is None:
            logger.debug(f"[LabelGen] No OHLCV data found for {pair} ({timeframe})")
            return None

        try:
            df = pd.read_feather(path)
            # Normalize column names
            col_map = {}
            for col in df.columns:
                lower = col.lower()
                if lower == "date" or lower == "datetime" or lower == "timestamp":
                    col_map[col] = "date"
                elif lower in ("open", "high", "low", "close", "volume"):
                    col_map[col] = lower

            df = df.rename(columns=col_map)

            if "date" not in df.columns:
                # First column is usually date
                df = df.rename(columns={df.columns[0]: "date"})

            df["date"] = pd.to_datetime(df["date"], utc=True)
            df = df.sort_values("date").reset_index(drop=True)

            self._ohlcv_cache[cache_key] = df
            logger.debug(f"[LabelGen] Loaded {len(df)} candles for {pair} ({timeframe})")
            return df

        except Exception as e:
            logger.warning(f"[LabelGen] Failed to load OHLCV for {pair}: {e}")
            return None

    def _get_candles_before(self, df: pd.DataFrame, timestamp_ms: int,
                           n_candles: int = MIN_CANDLES_BEFORE) -> Optional[pd.DataFrame]:
        """Get N candles ending at or before the given timestamp."""
        ts = pd.Timestamp(timestamp_ms, unit="ms", tz="UTC")

        # Find the index where date <= ts
        mask = df["date"] <= ts
        valid_idx = df.index[mask]

        if len(valid_idx) < n_candles:
            return None

        end_idx = valid_idx[-1] + 1  # inclusive
        start_idx = end_idx - n_candles

        return df.iloc[start_idx:end_idx].copy()

    # ===================================================================
    # Backtest Parsing
    # ===================================================================

    def _extract_trades_from_zip(self, zip_path: str) -> Tuple[List[Dict], str]:
        """Extract trades from a backtest ZIP file."""
        trades = []
        strategy = "unknown"

        try:
            with zipfile.ZipFile(zip_path, "r") as z:
                for name in z.namelist():
                    if name.endswith(".json") and "config" not in name and "_AIFreq" not in name:
                        with z.open(name) as f:
                            data = json.load(f)
                            if "strategy" in data:
                                for strat_name, strat_data in data["strategy"].items():
                                    strategy = strat_name
                                    trades.extend(strat_data.get("trades", []))
        except Exception as e:
            logger.error(f"[LabelGen] Failed to read ZIP {zip_path}: {e}")

        return trades, strategy

    def _extract_trades_from_json(self, json_path: str) -> Tuple[List[Dict], str]:
        """Extract trades from a backtest JSON file."""
        trades = []
        strategy = "unknown"

        try:
            with open(json_path, "r") as f:
                data = json.load(f)
                if "strategy" in data:
                    for strat_name, strat_data in data["strategy"].items():
                        strategy = strat_name
                        trades.extend(strat_data.get("trades", []))
        except Exception as e:
            logger.error(f"[LabelGen] Failed to read JSON {json_path}: {e}")

        return trades, strategy

    # ===================================================================
    # Feature Computation + Labeling
    # ===================================================================

    def _compute_features_for_trade(self, trade: Dict) -> Optional[Dict[str, float]]:
        """Compute 193 chart features at the trade's entry point."""
        pair = trade.get("pair", "")
        open_date = trade.get("open_date", "")

        if not pair or not open_date:
            return None

        # Load 1h OHLCV
        df_1h = self._load_ohlcv(pair, "1h")
        if df_1h is None:
            return None

        # Convert open_date to timestamp
        try:
            if isinstance(open_date, str):
                ts = pd.Timestamp(open_date, tz="UTC")
            else:
                ts = pd.Timestamp(open_date, unit="ms", tz="UTC")
            ts_ms = int(ts.timestamp() * 1000)
        except Exception:
            return None

        # Get candles before trade entry
        df_slice = self._get_candles_before(df_1h, ts_ms, MIN_CANDLES_BEFORE)
        if df_slice is None:
            logger.debug(f"[LabelGen] Insufficient candles for {pair} at {open_date}")
            return None

        # Compute 193 chart features
        try:
            features = compute_chart_features(
                df_1h=df_slice,
                df_4h=None,  # Could load 4h data too, but 1h is sufficient
                df_1d=None,
                include_signature=True,
            )
            return features
        except Exception as e:
            logger.warning(f"[LabelGen] chart_features failed for {pair}: {e}")
            return None

    def _label_trade(self, trade: Dict) -> Tuple[int, str]:
        """Apply Triple Barrier labeling to a trade."""
        profit_pct = trade.get("profit_ratio", 0) * 100  # Convert ratio to percentage

        if profit_pct >= WIN_THRESHOLD:
            return 2, "BULLISH"
        elif profit_pct <= LOSE_THRESHOLD:
            return 0, "BEARISH"
        else:
            return 1, "NEUTRAL"

    # ===================================================================
    # Data Generation
    # ===================================================================

    def generate_from_backtests(self, results_dir: str = None,
                                max_trades: int = 50000) -> int:
        """Generate labeled training data from all backtest result files.

        Returns number of new training samples generated.
        """
        if results_dir is None:
            results_dir = BACKTEST_RESULTS_DIR

        if not os.path.isdir(results_dir):
            logger.error(f"[LabelGen] Results directory not found: {results_dir}")
            return 0

        # Find all backtest files
        zip_files = sorted(glob.glob(os.path.join(results_dir, "*.zip")))
        json_files = sorted(glob.glob(os.path.join(results_dir, "*.json")))
        # Exclude meta files and config files
        json_files = [f for f in json_files if not f.endswith(".meta.json")
                      and "last_result" not in f]

        all_files = zip_files + json_files
        if not all_files:
            logger.warning(f"[LabelGen] No backtest files found in {results_dir}")
            return 0

        logger.info(f"[LabelGen] Found {len(all_files)} backtest files")

        total_generated = 0
        total_skipped_no_features = 0
        total_skipped_duplicate = 0

        for file_path in all_files:
            if total_generated >= max_trades:
                break

            # Extract trades
            if file_path.endswith(".zip"):
                trades, strategy = self._extract_trades_from_zip(file_path)
            else:
                trades, strategy = self._extract_trades_from_json(file_path)

            if not trades:
                continue

            logger.info(f"[LabelGen] Processing {len(trades)} trades from {os.path.basename(file_path)} ({strategy})")

            batch = []
            for trade in trades:
                if total_generated + len(batch) >= max_trades:
                    break

                # Compute features
                features = self._compute_features_for_trade(trade)
                if features is None:
                    total_skipped_no_features += 1
                    continue

                # Label
                label, label_name = self._label_trade(trade)

                # Build record
                pair = trade.get("pair", "UNKNOWN")
                open_date = trade.get("open_date", "")
                close_date = trade.get("close_date", "")
                profit_pct = trade.get("profit_ratio", 0) * 100
                duration_mins = trade.get("trade_duration", 0)
                duration_hours = duration_mins / 60 if duration_mins else None
                exit_reason = trade.get("exit_reason", "unknown")
                is_short = trade.get("is_short", False)
                direction = "short" if is_short else "long"

                batch.append((
                    pair, open_date, close_date, direction,
                    round(profit_pct, 4), duration_hours,
                    exit_reason, label, label_name,
                    json.dumps(features), len(features),
                    "backtest", strategy, None, "1h"
                ))

            # Batch insert
            if batch:
                inserted = self._batch_insert(batch)
                total_generated += inserted
                total_skipped_duplicate += len(batch) - inserted
                logger.info(f"[LabelGen]   → {inserted} new, {len(batch) - inserted} duplicates")

        # Clear OHLCV cache to free memory
        self._ohlcv_cache.clear()

        logger.info(f"[LabelGen] Generation complete: "
                    f"{total_generated} new samples, "
                    f"{total_skipped_no_features} skipped (no features), "
                    f"{total_skipped_duplicate} skipped (duplicates)")

        return total_generated

    def _batch_insert(self, batch: List[Tuple]) -> int:
        """Insert training data batch, skipping duplicates. Returns count of inserted."""
        inserted = 0
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
                        inserted += 1
                except Exception as e:
                    logger.debug(f"[LabelGen] Insert error: {e}")
            conn.commit()
        return inserted

    # ===================================================================
    # Live Trade Enrichment
    # ===================================================================

    def enrich_from_live_trades(self, min_trades: int = 20) -> int:
        """Add live trade data (from ai_decisions) to the training set.

        Live trades have actual AI confidence scores and outcomes,
        making them higher-quality training data than backtest trades.
        """
        conn = get_db_connection(self.db_path)
        try:
            # Get completed live trades with outcomes
            rows = conn.execute("""
                SELECT d.pair, d.signal_type, d.confidence, d.regime,
                       d.outcome_pnl, d.outcome_duration, d.timestamp
                FROM ai_decisions d
                WHERE d.outcome_pnl IS NOT NULL
                ORDER BY d.timestamp ASC
            """).fetchall()

            if len(rows) < min_trades:
                logger.info(f"[LabelGen] Only {len(rows)} live trades (need {min_trades})")
                return 0

            logger.info(f"[LabelGen] Enriching with {len(rows)} live trades")

            batch = []
            skipped = 0

            for row in rows:
                pair = row["pair"]
                timestamp = row["timestamp"]
                pnl = row["outcome_pnl"]

                # Compute chart features at trade entry
                features = self._compute_features_for_live_trade(pair, timestamp)
                if features is None:
                    skipped += 1
                    continue

                # Add AI-specific features
                features["live_confidence"] = row["confidence"] or 0.5
                features["live_signal_bullish"] = 1.0 if row["signal_type"] == "BULLISH" else 0.0

                # Label from actual PnL
                if pnl > WIN_THRESHOLD:
                    label, label_name = 2, "BULLISH"
                elif pnl < LOSE_THRESHOLD:
                    label, label_name = 0, "BEARISH"
                else:
                    label, label_name = 1, "NEUTRAL"

                duration = row["outcome_duration"]
                duration_hours = duration / 3600 if duration else None

                batch.append((
                    pair, timestamp, timestamp, "long",
                    round(pnl, 4), duration_hours,
                    "live", label, label_name,
                    json.dumps(features), len(features),
                    "live", "AIFreqtradeSizer",
                    row["regime"], "1h"
                ))

            if batch:
                inserted = self._batch_insert(batch)
                logger.info(f"[LabelGen] Live enrichment: {inserted} new, {skipped} skipped")
                return inserted

            return 0

        finally:
            conn.close()

    def _compute_features_for_live_trade(self, pair: str, timestamp: str) -> Optional[Dict[str, float]]:
        """Compute chart features for a live trade entry."""
        df_1h = self._load_ohlcv(pair, "1h")
        if df_1h is None:
            return None

        try:
            ts = pd.Timestamp(timestamp, tz="UTC")
            ts_ms = int(ts.timestamp() * 1000)
        except Exception:
            return None

        df_slice = self._get_candles_before(df_1h, ts_ms, MIN_CANDLES_BEFORE)
        if df_slice is None:
            return None

        try:
            return compute_chart_features(df_1h=df_slice, include_signature=True)
        except Exception:
            return None

    # ===================================================================
    # Run Backtest (fresh data generation)
    # ===================================================================

    def run_fresh_backtest(self, pairs: List[str] = None,
                          timerange_days: int = 90,
                          strategy: str = "SampleStrategy") -> bool:
        """Run a fresh backtest to generate more trade data.

        Uses SampleStrategy (simple RSI+EMA) instead of AIFreqtradeSizer
        because AIFreqtradeSizer depends on RAG/LLM which don't work in backtest.
        This generates more diverse trades for CatBoost training.
        """
        freqtrade_bin = os.path.join(BASE_DIR, "..", ".venv", "bin", "freqtrade")
        if not os.path.exists(freqtrade_bin):
            freqtrade_bin = "freqtrade"

        # Config
        config_path = os.path.join(BASE_DIR, "..", "config_bybit_testnet_futures.json")
        if not os.path.exists(config_path):
            config_path = os.path.join(BASE_DIR, "..", "config.json")
            if not os.path.exists(config_path):
                logger.error("[LabelGen] No config file found")
                return False

        # Default pairs — all available OHLCV data
        if pairs is None:
            pairs = self._discover_available_pairs()
            if not pairs:
                pairs = ["BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT",
                         "DOGE/USDT:USDT", "ADA/USDT:USDT", "XRP/USDT:USDT",
                         "LINK/USDT:USDT", "DOT/USDT:USDT", "AAVE/USDT:USDT",
                         "BNB/USDT:USDT"]

        logger.info(f"[LabelGen] Running backtest: {len(pairs)} pairs, {timerange_days} days, {strategy}")

        # Timerange
        now = datetime.now(timezone.utc)
        start = (now - timedelta(days=timerange_days)).strftime("%Y%m%d")
        end = now.strftime("%Y%m%d")
        timerange = f"{start}-{end}"

        # Override config with StaticPairList
        override_config = {
            "pairlists": [{"method": "StaticPairList"}],
            "exchange": {"pair_whitelist": pairs},
            "dry_run": True,
        }

        override_path = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".json", prefix="bt_label_",
                                             dir="/tmp", delete=False) as tf:
                json.dump(override_config, tf)
                override_path = tf.name

            # 1. Download data
            dl_cmd = [
                freqtrade_bin, "download-data",
                "--config", config_path,
                "--config", override_path,
                "--timerange", timerange,
                "--timeframe", "1h",
            ]
            logger.info(f"[LabelGen] Downloading data: {timerange}")
            dl_result = subprocess.run(dl_cmd, capture_output=True, text=True, timeout=900,
                                       cwd=os.path.join(BASE_DIR, ".."))
            if dl_result.returncode != 0:
                logger.warning(f"[LabelGen] Data download warning: {dl_result.stderr[:300]}")

            # 2. Run backtest with --export signals for maximum data
            bt_cmd = [
                freqtrade_bin, "backtesting",
                "--strategy", strategy,
                "--config", config_path,
                "--config", override_path,
                "--timerange", timerange,
                "--timeframe", "1h",
                "--export", "trades",
            ]
            logger.info(f"[LabelGen] Running backtest: {' '.join(bt_cmd[:6])}...")
            bt_result = subprocess.run(bt_cmd, capture_output=True, text=True, timeout=3600,
                                       cwd=os.path.join(BASE_DIR, ".."))

            if bt_result.returncode != 0:
                logger.error(f"[LabelGen] Backtest failed (rc={bt_result.returncode}): "
                           f"{bt_result.stderr[:500]}")
                return False

            logger.info("[LabelGen] Backtest complete, generating labels...")
            return True

        except subprocess.TimeoutExpired:
            logger.error("[LabelGen] Backtest timed out (60min)")
            return False
        except Exception as e:
            logger.error(f"[LabelGen] Backtest error: {e}")
            return False
        finally:
            if override_path and os.path.exists(override_path):
                try:
                    os.unlink(override_path)
                except Exception:
                    pass

    def _discover_available_pairs(self) -> List[str]:
        """Discover pairs that have OHLCV data available."""
        pairs = []
        for exchange in ["bybit", "binance"]:
            futures_dir = os.path.join(OHLCV_DATA_DIR, exchange, "futures")
            if not os.path.isdir(futures_dir):
                continue
            for f in os.listdir(futures_dir):
                if f.endswith("-1h-futures.feather"):
                    # BTC_USDT_USDT-1h-futures.feather → BTC/USDT:USDT
                    pair_raw = f.replace("-1h-futures.feather", "")
                    parts = pair_raw.split("_")
                    if len(parts) >= 3:
                        pair = f"{parts[0]}/{parts[1]}:{parts[2]}"
                        if pair not in pairs:
                            pairs.append(pair)
        logger.info(f"[LabelGen] Discovered {len(pairs)} pairs with OHLCV data")
        return pairs

    # ===================================================================
    # Training Data Export (for catboost_trainer.py)
    # ===================================================================

    def get_training_dataset(self, min_samples: int = 100,
                             sources: List[str] = None) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], List[str]]:
        """Load training data from backtest_training_data table.

        Returns (X, y, feature_names) ready for CatBoost training.
        Walk-forward ordering preserved (sorted by open_date).
        """
        conn = get_db_connection(self.db_path)
        try:
            where_clause = "1=1"
            params = []
            if sources:
                placeholders = ",".join(["?" for _ in sources])
                where_clause = f"source IN ({placeholders})"
                params = list(sources)

            rows = conn.execute(f"""
                SELECT features_json, label, n_features
                FROM backtest_training_data
                WHERE {where_clause}
                ORDER BY open_date ASC
            """, params).fetchall()

            if len(rows) < min_samples:
                logger.warning(f"[LabelGen] Insufficient data: {len(rows)} < {min_samples}")
                return None, None, []

            # Parse features — use the first row to determine feature names
            first_features = json.loads(rows[0]["features_json"])
            feature_names = sorted(first_features.keys())
            n_features = len(feature_names)

            X_list = []
            y_list = []
            skipped = 0

            for row in rows:
                try:
                    features = json.loads(row["features_json"])
                    # Extract features in consistent order
                    feature_vec = []
                    for fname in feature_names:
                        val = features.get(fname, 0.0)
                        # Handle NaN/Inf
                        if val is None or (isinstance(val, float) and (np.isnan(val) or np.isinf(val))):
                            val = 0.0
                        feature_vec.append(float(val))

                    X_list.append(feature_vec)
                    y_list.append(row["label"])
                except Exception:
                    skipped += 1
                    continue

            if skipped > 0:
                logger.info(f"[LabelGen] Skipped {skipped} rows with parsing errors")

            logger.info(f"[LabelGen] Loaded {len(X_list)} samples, {n_features} features")

            # Label distribution
            y_arr = np.array(y_list)
            for lbl, name in LABEL_MAP.items():
                count = np.sum(y_arr == lbl)
                pct = count / len(y_arr) * 100 if len(y_arr) > 0 else 0
                logger.info(f"[LabelGen]   {name}: {count} ({pct:.1f}%)")

            return np.array(X_list), y_arr, feature_names

        finally:
            conn.close()

    # ===================================================================
    # Status / Reporting
    # ===================================================================

    def get_status(self) -> Dict[str, Any]:
        """Get status of the training dataset."""
        conn = get_db_connection(self.db_path)
        try:
            # Total counts by source
            rows = conn.execute("""
                SELECT source, strategy, label_name, COUNT(*) as cnt
                FROM backtest_training_data
                GROUP BY source, strategy, label_name
                ORDER BY source, strategy, label_name
            """).fetchall()

            total = conn.execute("SELECT COUNT(*) FROM backtest_training_data").fetchone()[0]
            avg_features = conn.execute(
                "SELECT AVG(n_features) FROM backtest_training_data"
            ).fetchone()[0]

            # Date range
            date_range = conn.execute("""
                SELECT MIN(open_date), MAX(open_date)
                FROM backtest_training_data
            """).fetchone()

            # Last training run
            last_run = conn.execute("""
                SELECT * FROM catboost_training_runs
                ORDER BY trained_at DESC LIMIT 1
            """).fetchone()

            status = {
                "total_samples": total,
                "avg_features": round(avg_features, 0) if avg_features else 0,
                "date_range": (date_range[0], date_range[1]) if date_range and date_range[0] else None,
                "breakdown": [dict(r) for r in rows],
                "last_training_run": dict(last_run) if last_run else None,
            }

            return status

        finally:
            conn.close()

    def print_status(self):
        """Print a human-readable status report."""
        status = self.get_status()

        print("\n" + "=" * 60)
        print("  BACKTEST TRAINING PIPELINE STATUS")
        print("=" * 60)
        print(f"  Total samples:    {status['total_samples']}")
        print(f"  Avg features:     {status['avg_features']}")
        if status['date_range']:
            print(f"  Date range:       {status['date_range'][0]} → {status['date_range'][1]}")
        print()

        if status['breakdown']:
            print("  Source       | Strategy            | Label    | Count")
            print("  " + "-" * 56)
            for row in status['breakdown']:
                print(f"  {row['source']:<12} | {row['strategy']:<19} | {row['label_name']:<8} | {row['cnt']}")
        else:
            print("  No training data yet.")

        if status['last_training_run']:
            run = status['last_training_run']
            print(f"\n  Last training: {run.get('trained_at', 'unknown')}")
            print(f"  Model: v{run.get('model_version', '?')} — "
                  f"Train acc: {run.get('train_accuracy', 0):.3f}, "
                  f"Test acc: {run.get('test_accuracy', 0):.3f}")
        else:
            print("\n  No CatBoost training runs yet.")

        print("=" * 60 + "\n")


# ===========================================================================
# Full Pipeline Runner
# ===========================================================================

def run_full_pipeline(run_backtest: bool = False, enrich_live: bool = True,
                      retrain: bool = True, timerange_days: int = 90) -> Dict[str, Any]:
    """Run the full 13B pipeline: backtest → labels → CatBoost.

    Returns results dict with counts and accuracy metrics.
    """
    results = {"generated": 0, "live_enriched": 0, "retrain_result": None}

    gen = BacktestLabelGenerator()

    # Step 1: Optionally run fresh backtest
    if run_backtest:
        logger.info("[Pipeline] Step 1: Running fresh backtest...")
        success = gen.run_fresh_backtest(timerange_days=timerange_days)
        if not success:
            logger.warning("[Pipeline] Backtest failed, continuing with existing data...")

    # Step 2: Generate labels from all backtest results
    logger.info("[Pipeline] Step 2: Generating labels from backtest results...")
    results["generated"] = gen.generate_from_backtests()

    # Step 3: Enrich with live trade data
    if enrich_live:
        logger.info("[Pipeline] Step 3: Enriching with live trade data...")
        results["live_enriched"] = gen.enrich_from_live_trades()

    # Step 4: Retrain CatBoost
    if retrain:
        logger.info("[Pipeline] Step 4: Retraining CatBoost...")
        try:
            from catboost_trainer import train_catboost_v2
            X, y, feature_names = gen.get_training_dataset(min_samples=50)
            if X is not None:
                result = train_catboost_v2(X, y, feature_names)
                results["retrain_result"] = result
                logger.info(f"[Pipeline] CatBoost retrained: "
                          f"test_acc={result.get('test_accuracy', 0):.3f}")
            else:
                logger.warning("[Pipeline] Insufficient data for retraining")
        except ImportError:
            logger.error("[Pipeline] catboost_trainer.train_catboost_v2 not available")
        except Exception as e:
            logger.error(f"[Pipeline] Retraining failed: {e}")

    # Print status
    gen.print_status()

    return results


# ===========================================================================
# CLI
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description="Backtest Label Generator — Phase 26 Sprint 2 (13B)")
    parser.add_argument("--generate", action="store_true", help="Generate labels from existing backtests")
    parser.add_argument("--enrich-live", action="store_true", help="Add live trades to training set")
    parser.add_argument("--run-backtest", action="store_true", help="Run fresh backtest first")
    parser.add_argument("--retrain", action="store_true", help="Retrain CatBoost after generation")
    parser.add_argument("--status", action="store_true", help="Show dataset status")
    parser.add_argument("--full", action="store_true", help="Full pipeline (backtest+generate+enrich+retrain)")
    parser.add_argument("--timerange-days", type=int, default=90, help="Backtest timerange in days")
    parser.add_argument("--strategy", type=str, default="SampleStrategy", help="Strategy for backtest")

    args = parser.parse_args()

    if args.status:
        gen = BacktestLabelGenerator()
        gen.print_status()
        return

    if args.full:
        results = run_full_pipeline(
            run_backtest=True,
            enrich_live=True,
            retrain=True,
            timerange_days=args.timerange_days,
        )
        print(f"\nPipeline results: {json.dumps(results, indent=2, default=str)}")
        return

    if args.generate or args.enrich_live or args.run_backtest:
        gen = BacktestLabelGenerator()

        if args.run_backtest:
            gen.run_fresh_backtest(
                timerange_days=args.timerange_days,
                strategy=args.strategy,
            )

        if args.generate:
            count = gen.generate_from_backtests()
            print(f"Generated {count} training samples")

        if args.enrich_live:
            count = gen.enrich_from_live_trades()
            print(f"Enriched {count} live trade samples")

        if args.retrain:
            try:
                from catboost_trainer import train_catboost_v2
                X, y, fnames = gen.get_training_dataset(min_samples=50)
                if X is not None:
                    result = train_catboost_v2(X, y, fnames)
                    print(f"Retrained: test_acc={result.get('test_accuracy', 0):.3f}")
            except Exception as e:
                print(f"Retrain failed: {e}")

        gen.print_status()
        return

    parser.print_help()


if __name__ == "__main__":
    main()
