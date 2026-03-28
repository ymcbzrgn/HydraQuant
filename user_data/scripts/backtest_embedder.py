"""
Phase 19: BacktestEmbedder — Feed 2 years of backtest data into RAG pipeline.

Processes freqtrade backtest results into:
1. PatternStatStore (queryable statistics for evidence-based signals)
2. ChromaDB (vector-searchable trade lessons for RAG retrieval)

Usage:
    # Process a specific backtest ZIP:
    python backtest_embedder.py --file user_data/backtest_results/backtest-result-2026-03-17.zip

    # Process all unprocessed backtests:
    python backtest_embedder.py --all

    # Process with indicator enrichment (requires OHLCV data in user_data/data/):
    python backtest_embedder.py --all --enrich
"""

import os
import sys
import json
import zipfile
import sqlite3
import logging
import argparse
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple

sys.path.append(os.path.dirname(__file__))

from ai_config import AI_DB_PATH
from pattern_stat_store import PatternStatStore

logger = logging.getLogger(__name__)


class BacktestEmbedder:
    """
    Processes freqtrade backtest results into the AI learning pipeline.
    Works with or without OHLCV indicator enrichment.
    """

    def __init__(self, db_path: str = AI_DB_PATH):
        self.db_path = db_path
        self.stat_store = PatternStatStore(db_path=db_path)
        self._init_tracking_db()
        self._ohlcv_cache: Dict[str, Any] = {}

    def _get_conn(self):
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_tracking_db(self):
        """Track which backtest files have been processed."""
        try:
            with self._get_conn() as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS backtest_processed (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        file_path TEXT UNIQUE,
                        strategy TEXT,
                        num_trades INTEGER,
                        timerange TEXT,
                        processed_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                conn.commit()
        except Exception as e:
            logger.error(f"[BacktestEmbedder] Tracking DB init failed: {e}")

    def _is_processed(self, file_path: str) -> bool:
        """Check if a backtest file has already been processed."""
        try:
            with self._get_conn() as conn:
                row = conn.execute(
                    "SELECT id FROM backtest_processed WHERE file_path = ?",
                    (file_path,)
                ).fetchone()
                return row is not None
        except Exception:
            return False

    def _mark_processed(self, file_path: str, strategy: str, num_trades: int, timerange: str = ""):
        """Mark a backtest file as processed."""
        try:
            with self._get_conn() as conn:
                conn.execute(
                    "INSERT OR IGNORE INTO backtest_processed (file_path, strategy, num_trades, timerange) VALUES (?, ?, ?, ?)",
                    (file_path, strategy, num_trades, timerange)
                )
                conn.commit()
        except Exception as e:
            logger.error(f"[BacktestEmbedder] Mark processed failed: {e}")

    # ── Backtest Parsing ──────────────────────────────────────

    def _find_trades_in_data(self, data: dict) -> Tuple[List[Dict], str]:
        """
        Recursively find trades list in freqtrade backtest JSON data.

        Freqtrade backtest formats vary:
          Format A: {"strategy": {"StratName": {"trades": [...]}}}
          Format B: {"StratName": {"trades": [...]}}
          Format C: {"trades": [...]}
          Format D: [trade, trade, ...]  (flat list)

        This method handles all of them.
        """
        if isinstance(data, list):
            return data, "unknown"

        if not isinstance(data, dict):
            return [], "unknown"

        # Direct trades key at top level (Format C)
        if 'trades' in data and isinstance(data['trades'], list):
            return data['trades'], data.get('strategy_name', 'unknown')

        # Iterate top-level keys
        for key, value in data.items():
            if not isinstance(value, dict):
                continue

            # Format B: {"StratName": {"trades": [...]}}
            if 'trades' in value and isinstance(value['trades'], list):
                return value['trades'], key

            # Format A: {"strategy": {"StratName": {"trades": [...]}}}
            # Go one level deeper
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, dict) and 'trades' in sub_value:
                    if isinstance(sub_value['trades'], list):
                        return sub_value['trades'], sub_key

        return [], "unknown"

    def extract_trades_from_zip(self, zip_path: str) -> Tuple[List[Dict], str]:
        """
        Extract trades from a freqtrade backtest ZIP file.
        Returns (trades_list, strategy_name).
        """
        try:
            with zipfile.ZipFile(zip_path, 'r') as zf:
                # Find the main JSON results file (skip meta/config/metadata files)
                json_files = [f for f in zf.namelist()
                              if f.endswith('.json')
                              and 'meta' not in f.lower()
                              and 'config' not in f.lower()]
                if not json_files:
                    logger.error(f"No JSON results found in {zip_path}")
                    return [], "unknown"

                with zf.open(json_files[0]) as jf:
                    data = json.load(jf)

                trades, strategy = self._find_trades_in_data(data)
                if trades:
                    logger.info(f"[BacktestEmbedder] ZIP {zip_path}: found {len(trades)} trades for {strategy}")
                return trades, strategy

        except Exception as e:
            logger.error(f"[BacktestEmbedder] ZIP extraction failed: {e}")

        return [], "unknown"

    def extract_trades_from_json(self, json_path: str) -> Tuple[List[Dict], str]:
        """Extract trades from a plain JSON file (non-ZIP)."""
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)

            return self._find_trades_in_data(data)

        except Exception as e:
            logger.error(f"[BacktestEmbedder] JSON extraction failed: {e}")

        return [], "unknown"

    # ── Indicator Enrichment ──────────────────────────────────

    def _load_ohlcv(self, pair: str, timeframe: str = "1h") -> Optional[Any]:
        """
        Load OHLCV data from freqtrade's data directory.
        Returns a pandas DataFrame or None.
        """
        cache_key = f"{pair}_{timeframe}"
        if cache_key in self._ohlcv_cache:
            return self._ohlcv_cache[cache_key]

        try:
            import pandas as pd

            # freqtrade stores data as: user_data/data/exchange/pair-timeframe.feather or .json
            base_dir = os.path.join(os.path.dirname(os.path.dirname(self.db_path)), "data")

            # Try multiple exchange directories and naming conventions
            # Freqtrade futures data: BTC_USDT_USDT-1h-futures.feather (in futures/ subdir)
            # Freqtrade spot data: BTC_USDT-1h.feather (in root or spot/ subdir)
            for exchange in ['bybit', 'binance', 'binanceus']:
                exchange_dir = os.path.join(base_dir, exchange)
                if not os.path.isdir(exchange_dir):
                    continue

                pair_clean = pair.split(":")[0]  # "BTC/USDT:USDT" → "BTC/USDT"
                pair_file = pair_clean.replace('/', '_')
                pair_file_settle = pair_file + "_USDT"  # futures settle coin

                for subdir in ['futures', 'spot', '']:
                    for name in [
                        f"{pair_file_settle}-{timeframe}-futures",  # BTC_USDT_USDT-1h-futures
                        f"{pair_file}-{timeframe}-futures",         # BTC_USDT-1h-futures
                        f"{pair_file}-{timeframe}",                 # BTC_USDT-1h
                        f"{pair_file_settle}-{timeframe}",          # BTC_USDT_USDT-1h
                    ]:
                        for ext in ['.feather', '.json']:
                            if subdir:
                                fpath = os.path.join(exchange_dir, subdir, f"{name}{ext}")
                            else:
                                fpath = os.path.join(exchange_dir, f"{name}{ext}")
                            if os.path.exists(fpath):
                                if ext == '.feather':
                                    df = pd.read_feather(fpath)
                                else:
                                    df = pd.read_json(fpath)
                                self._ohlcv_cache[cache_key] = df
                                logger.info(f"[BacktestEmbedder] Loaded OHLCV: {fpath} ({len(df)} candles)")
                                return df

        except Exception as e:
            logger.debug(f"[BacktestEmbedder] OHLCV load for {pair} failed: {e}")

        return None

    def _compute_indicators_at(self, df, timestamp_ms: int) -> Dict[str, Any]:
        """
        Compute basic indicators on OHLCV dataframe and return values at given timestamp.
        Uses simple pandas operations (no TA-Lib dependency required).
        """
        try:
            import pandas as pd
            import numpy as np

            if df is None or len(df) < 200:
                return {}

            # Ensure date column
            if 'date' in df.columns:
                if not pd.api.types.is_datetime64_any_dtype(df['date']):
                    df['date'] = pd.to_datetime(df['date'], utc=True)
            elif 'open_time' in df.columns:
                df['date'] = pd.to_datetime(df['open_time'], unit='ms', utc=True)
            else:
                return {}

            target_dt = pd.Timestamp(timestamp_ms, unit='ms', tz='UTC')

            # Find the candle at or just before the entry
            mask = df['date'] <= target_dt
            if mask.sum() < 50:
                return {}

            idx = mask.sum() - 1
            window = df.iloc[max(0, idx - 200):idx + 1].copy()

            close = window['close']
            high = window['high']
            low = window['low']
            volume = window['volume'] if 'volume' in window.columns else None

            result = {}

            # RSI (14)
            delta = close.diff()
            gain = delta.clip(lower=0).rolling(14).mean()
            loss = (-delta.clip(upper=0)).rolling(14).mean()
            rs = gain / loss.replace(0, np.nan)
            rsi = 100 - (100 / (1 + rs))
            if len(rsi.dropna()) > 0:
                result['rsi'] = round(float(rsi.iloc[-1]), 1)

            # MACD (12, 26, 9)
            ema12 = close.ewm(span=12).mean()
            ema26 = close.ewm(span=26).mean()
            macd_line = ema12 - ema26
            signal_line = macd_line.ewm(span=9).mean()
            macd_hist = macd_line - signal_line
            if len(macd_hist.dropna()) > 0:
                result['macd_hist'] = round(float(macd_hist.iloc[-1]), 4)

            # EMAs
            for period in [20, 50, 200]:
                ema = close.ewm(span=period).mean()
                if len(ema.dropna()) > 0:
                    result[f'ema{period}'] = round(float(ema.iloc[-1]), 2)

            # ATR (14)
            tr = pd.concat([
                high - low,
                (high - close.shift()).abs(),
                (low - close.shift()).abs()
            ], axis=1).max(axis=1)
            atr = tr.rolling(14).mean()
            if len(atr.dropna()) > 0:
                result['atr'] = round(float(atr.iloc[-1]), 2)

            # ADX (14)
            plus_dm = high.diff().clip(lower=0)
            minus_dm = (-low.diff()).clip(lower=0)
            plus_di = 100 * (plus_dm.rolling(14).mean() / atr.replace(0, np.nan))
            minus_di = 100 * (minus_dm.rolling(14).mean() / atr.replace(0, np.nan))
            dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
            adx = dx.rolling(14).mean()
            if len(adx.dropna()) > 0:
                result['adx'] = round(float(adx.iloc[-1]), 1)

            # Volume ratio
            if volume is not None and len(volume) > 20:
                vol_sma = volume.rolling(20).mean()
                if vol_sma.iloc[-1] > 0:
                    result['volume_ratio'] = round(float(volume.iloc[-1] / vol_sma.iloc[-1]), 2)

            # Current price
            result['price'] = round(float(close.iloc[-1]), 2)

            return result

        except Exception as e:
            logger.debug(f"[BacktestEmbedder] Indicator computation failed: {e}")
            return {}

    # ── Trade Classification ──────────────────────────────────

    def classify_trade(self, trade: Dict, indicators: Dict = None) -> Dict[str, Any]:
        """
        Classify a backtest trade into PatternStatStore buckets.
        Works with or without indicator enrichment.
        """
        pair = trade.get('pair', 'UNKNOWN')
        profit_ratio = trade.get('profit_ratio', 0)
        profit_pct = profit_ratio * 100  # Convert to percentage
        duration_mins = trade.get('trade_duration', 0)
        duration_hours = duration_mins / 60 if duration_mins else None
        is_short = trade.get('is_short', False)
        direction = 'short' if is_short else 'long'

        entry = {
            'pair': pair,
            'strategy': trade.get('_strategy', 'unknown'),
            'direction': direction,
            'entry_date': trade.get('open_date', ''),
            'exit_date': trade.get('close_date', ''),
            'profit_pct': round(profit_pct, 3),
            'duration_hours': round(duration_hours, 2) if duration_hours else None,
            'exit_reason': trade.get('exit_reason', 'unknown'),
            'entry_price': trade.get('open_rate', 0),
        }

        # Indicator-based classification (if available)
        ind = indicators or {}

        if 'rsi' in ind:
            entry['rsi_bucket'] = PatternStatStore.classify_rsi(ind['rsi'])
        if 'macd_hist' in ind:
            entry['macd_signal'] = PatternStatStore.classify_macd(ind['macd_hist'])
        if 'adx' in ind:
            atr_ratio = ind.get('atr', 0) / ind.get('price', 1) * 100 if ind.get('price') else 1.0
            entry['regime'] = PatternStatStore.classify_regime(
                ind['adx'], atr_ratio,
                price=ind.get('price'), ema200=ind.get('ema200')
            )
            entry['adx_bucket'] = 'trending' if ind['adx'] > 25 else 'ranging' if ind['adx'] < 20 else 'transitional'
        if 'volume_ratio' in ind:
            entry['volume_bucket'] = PatternStatStore.classify_volume(ind['volume_ratio'])
        if all(k in ind for k in ['price', 'ema20', 'ema50', 'ema200']):
            entry['ema_alignment'] = PatternStatStore.classify_ema(
                ind['price'], ind['ema20'], ind['ema50'], ind['ema200']
            )

        # Store raw indicators as JSON for future analysis
        if ind:
            entry['indicators_json'] = json.dumps(ind)

        return entry

    # ── Lesson Generation ─────────────────────────────────────

    def generate_lesson_text(self, trade: Dict, indicators: Dict = None, strategy: str = "") -> str:
        """
        Generate a rich text lesson from a backtest trade.
        This text gets embedded into ChromaDB for RAG retrieval.
        """
        pair = trade.get('pair', 'UNKNOWN')
        profit_pct = trade.get('profit_ratio', 0) * 100
        duration_mins = trade.get('trade_duration', 0)
        duration_str = f"{duration_mins / 60:.1f}h" if duration_mins else "unknown"
        is_short = trade.get('is_short', False)
        direction = "SHORT" if is_short else "LONG"
        exit_reason = trade.get('exit_reason', 'unknown')
        open_rate = trade.get('open_rate', 0)
        close_rate = trade.get('close_rate', 0)
        leverage = trade.get('leverage', 1.0)
        open_date = trade.get('open_date', '')[:10]  # Just the date part
        outcome = "WIN" if profit_pct > 0 else "LOSS"

        lines = [
            f"HISTORICAL TRADE: {pair} {direction} | {open_date} | {outcome}",
            f"Entry: ${open_rate:.2f} → Exit: ${close_rate:.2f} | P&L: {profit_pct:+.2f}% | Duration: {duration_str}",
            f"Exit Reason: {exit_reason} | Leverage: {leverage:.1f}x | Strategy: {strategy}",
        ]

        # Add indicator context if available
        ind = indicators or {}
        if ind:
            ind_parts = []
            if 'rsi' in ind:
                bucket = PatternStatStore.classify_rsi(ind['rsi'])
                ind_parts.append(f"RSI={ind['rsi']:.0f} ({bucket})")
            if 'macd_hist' in ind:
                signal = PatternStatStore.classify_macd(ind['macd_hist'])
                ind_parts.append(f"MACD={signal}")
            if 'adx' in ind:
                ind_parts.append(f"ADX={ind['adx']:.0f}")
            if 'ema_alignment' in ind:
                ind_parts.append(f"EMA={ind.get('ema_alignment', '')}")
            if 'volume_ratio' in ind:
                ind_parts.append(f"Vol={ind['volume_ratio']:.1f}x")

            if ind_parts:
                lines.append(f"Indicators at Entry: {' | '.join(ind_parts)}")

            if 'adx' in ind:
                regime = PatternStatStore.classify_regime(ind['adx'])
                lines.append(f"Market Regime: {regime}")

        # Generate actionable lesson based on outcome
        if profit_pct > 5:
            lines.append(f"LESSON: Strong {direction} win in {pair}. This setup produced exceptional returns ({profit_pct:+.2f}%). "
                         f"Pattern worth tracking for future signals in similar conditions.")
        elif profit_pct > 0:
            lines.append(f"LESSON: Modest {direction} win in {pair} ({profit_pct:+.2f}%). "
                         f"Exit via {exit_reason} after {duration_str}.")
        elif profit_pct > -2:
            lines.append(f"LESSON: Small {direction} loss in {pair} ({profit_pct:+.2f}%). "
                         f"Likely noise rather than structural failure. Review if pattern repeats.")
        else:
            lines.append(f"LESSON: Significant {direction} loss in {pair} ({profit_pct:+.2f}%). "
                         f"Exit via {exit_reason}. Identify what changed after entry to cause reversal.")

        return "\n".join(lines)

    # ── Main Processing ───────────────────────────────────────

    def process_file(self, file_path: str, enrich: bool = False) -> int:
        """
        Process a single backtest results file (ZIP or JSON).
        Returns number of trades processed.
        """
        if self._is_processed(file_path):
            logger.info(f"[BacktestEmbedder] Already processed: {file_path}")
            return 0

        # Extract trades
        if file_path.endswith('.zip'):
            trades, strategy = self.extract_trades_from_zip(file_path)
        elif file_path.endswith('.json'):
            trades, strategy = self.extract_trades_from_json(file_path)
        else:
            logger.error(f"[BacktestEmbedder] Unsupported file format: {file_path}")
            return 0

        if not trades:
            logger.warning(f"[BacktestEmbedder] No trades found in {file_path}")
            return 0

        logger.info(f"[BacktestEmbedder] Processing {len(trades)} trades from {strategy} ({file_path})")

        # Process each trade
        stat_entries = []
        lesson_docs = []
        lesson_metas = []
        lesson_ids = []

        for i, trade in enumerate(trades):
            pair = trade.get('pair', 'UNKNOWN')

            # Optional: enrich with indicators from OHLCV data
            indicators = {}
            if enrich:
                ts = trade.get('open_timestamp')
                if ts:
                    ohlcv_df = self._load_ohlcv(pair)
                    if ohlcv_df is not None:
                        indicators = self._compute_indicators_at(ohlcv_df, ts)

            # Classify for PatternStatStore
            classified = self.classify_trade(trade, indicators)
            classified['strategy'] = strategy
            stat_entries.append(classified)

            # Generate lesson for ChromaDB
            lesson_text = self.generate_lesson_text(trade, indicators, strategy)
            lesson_docs.append(lesson_text)
            lesson_metas.append({
                "type": "backtest_lesson",
                "pair": pair,
                "strategy": strategy,
                "direction": "short" if trade.get('is_short') else "long",
                "profit_pct": round(trade.get('profit_ratio', 0) * 100, 2),
                "exit_reason": trade.get('exit_reason', ''),
                "source": "backtest_embedder",
            })
            lesson_ids.append(f"bt_{strategy}_{i}_{pair.replace('/', '_')}")

        # Batch ingest into PatternStatStore
        self.stat_store.ingest_batch(stat_entries)
        logger.info(f"[BacktestEmbedder] Ingested {len(stat_entries)} trades into PatternStatStore")

        # Batch embed into ChromaDB
        embedded_count = self._embed_lessons(lesson_docs, lesson_metas, lesson_ids)
        logger.info(f"[BacktestEmbedder] Embedded {embedded_count} lessons into ChromaDB")

        # Phase 19: Extract MAGMA causal edges from backtest patterns
        magma_count = self._extract_magma_edges(stat_entries)
        if magma_count > 0:
            logger.info(f"[BacktestEmbedder] Extracted {magma_count} MAGMA causal edges from backtest patterns")

        # Level 4: Build RAPTOR trade hierarchy (3-level abstraction tree)
        raptor_nodes = self._build_raptor_hierarchy(lesson_docs, strategy)
        if raptor_nodes > 0:
            logger.info(f"[BacktestEmbedder:RAPTOR] Built hierarchy with {raptor_nodes} nodes from {len(lesson_docs)} lessons")

        # Mark as processed
        self._mark_processed(file_path, strategy, len(trades))

        total = len(stat_entries)
        logger.info(f"[BacktestEmbedder] Complete: {total} trades → {len(stat_entries)} stats + {embedded_count} lessons + {magma_count} MAGMA edges")
        return total

    def _extract_magma_edges(self, classified_trades: List[Dict]) -> int:
        """
        Phase 19: Extract causal relationships from backtest patterns and store in MAGMA.
        Example edges:
          causal: "rsi_oversold" → "bounce" (weight from win rate)
          causal: "macd_bullish_cross" → "rally" (weight from profit)
          entity: "BTC/USDT" → "mean_reversion_works" (if high win rate in ranging)
        """
        try:
            from magma_memory import MAGMAMemory
            magma = MAGMAMemory()
        except Exception as e:
            logger.debug(f"[BacktestEmbedder] MAGMA unavailable: {e}")
            return 0

        edge_count = 0

        # Aggregate by (pair, rsi_bucket, direction) to find patterns with enough data
        from collections import defaultdict
        pattern_groups = defaultdict(list)
        for t in classified_trades:
            key = (t.get('pair', ''), t.get('rsi_bucket', ''), t.get('direction', ''))
            if key[1]:  # Only if we have RSI bucket
                pattern_groups[key].append(t.get('profit_pct', 0))

        for (pair, rsi_bucket, direction), profits in pattern_groups.items():
            if len(profits) < 3:
                continue

            wins = sum(1 for p in profits if p > 0)
            win_rate = wins / len(profits)
            avg_profit = sum(profits) / len(profits)

            # Causal edge: RSI condition → outcome
            if win_rate > 0.55:
                outcome = "bounce" if direction == "long" else "drop"
                magma.add_edge("causal", rsi_bucket, f"leads_to_{outcome}", pair,
                               metadata={"win_rate": win_rate, "n": len(profits), "avg_pnl": avg_profit})
                edge_count += 1
                logger.info(f"[BacktestEmbedder:MAGMA] causal: {rsi_bucket} → {outcome} for {pair} "
                            f"(wr={win_rate:.0%}, n={len(profits)})")

            # Entity edge: pair → strategy effectiveness
            if win_rate > 0.60:
                magma.add_edge("entity", pair.lower().replace("/", "_"), f"profitable_{direction}",
                               rsi_bucket, metadata={"win_rate": win_rate, "n": len(profits)})
                edge_count += 1

        # Aggregate by (rsi_bucket, macd_signal) for combined pattern edges
        combo_groups = defaultdict(list)
        for t in classified_trades:
            rsi = t.get('rsi_bucket', '')
            macd = t.get('macd_signal', '')
            if rsi and macd:
                combo_groups[(rsi, macd)].append(t.get('profit_pct', 0))

        for (rsi, macd), profits in combo_groups.items():
            if len(profits) < 5:
                continue
            wins = sum(1 for p in profits if p > 0)
            win_rate = wins / len(profits)
            if win_rate > 0.55:
                magma.add_edge("causal", f"{rsi}+{macd}", "profitable_setup", f"wr_{win_rate:.0%}",
                               metadata={"n": len(profits), "avg_pnl": sum(profits) / len(profits)})
                edge_count += 1
                logger.info(f"[BacktestEmbedder:MAGMA] combo: {rsi}+{macd} → profitable "
                            f"(wr={win_rate:.0%}, n={len(profits)})")

        return edge_count

    def _build_raptor_hierarchy(self, lesson_docs: List[str], strategy: str) -> int:
        """
        Level 4: Build RAPTOR 3-level abstraction tree from backtest lessons.
        L0: Individual trade lessons
        L1: Pair-level summaries (every 5-10 trades)
        L2: Strategy-level meta-insights
        """
        if len(lesson_docs) < 5:
            logger.debug("[BacktestEmbedder:RAPTOR] Too few lessons for hierarchy (<5)")
            return 0

        try:
            from raptor_tree import RAPTORTree
            raptor = RAPTORTree()

            # Convert lesson texts to RAPTOR chunk format
            chunks = [
                {"id": f"bt_{strategy}_{i}", "text": doc, "metadata": {"strategy": strategy}}
                for i, doc in enumerate(lesson_docs)
            ]

            tree = raptor.build_tree(chunks, cluster_size=min(10, max(5, len(chunks) // 3)))

            total_nodes = sum(len(v) for v in tree.values())
            logger.info(f"[BacktestEmbedder:RAPTOR] Tree built: L0={len(tree.get('level_0', []))}, "
                        f"L1={len(tree.get('level_1', []))}, L2={len(tree.get('level_2', []))}")
            return total_nodes

        except Exception as e:
            logger.warning(f"[BacktestEmbedder:RAPTOR] Hierarchy build failed: {e}")
            return 0

    def _embed_lessons(self, docs: List[str], metas: List[Dict], ids: List[str]) -> int:
        """Embed lesson documents into ChromaDB via HybridRetriever."""
        try:
            from hybrid_retriever import HybridRetriever
            retriever = HybridRetriever(collection_name="crypto_news")
            retriever.add_documents(documents=docs, metadatas=metas, ids=ids)
            return len(docs)
        except Exception as e:
            logger.error(f"[BacktestEmbedder] ChromaDB embedding failed: {e}")
            return 0

    def process_all(self, results_dir: str = None, enrich: bool = False) -> int:
        """
        Process all unprocessed backtest files in the results directory.
        Returns total trades processed.
        """
        if results_dir is None:
            results_dir = os.path.join(
                os.path.dirname(os.path.dirname(self.db_path)),
                "backtest_results"
            )

        if not os.path.isdir(results_dir):
            logger.warning(f"[BacktestEmbedder] Results directory not found: {results_dir}")
            return 0

        total = 0
        for fname in sorted(os.listdir(results_dir)):
            fpath = os.path.join(results_dir, fname)
            if fname.startswith('.') or '.meta' in fname or 'config' in fname.lower():
                continue
            if fname.endswith('.zip') or fname.endswith('.json'):
                try:
                    count = self.process_file(fpath, enrich=enrich)
                    total += count
                except Exception as e:
                    logger.error(f"[BacktestEmbedder] Failed to process {fpath}: {e}")

        logger.info(f"[BacktestEmbedder] Total: {total} trades processed from {results_dir}")
        return total

    def get_processing_history(self) -> List[Dict]:
        """Return list of previously processed backtest files."""
        try:
            with self._get_conn() as conn:
                rows = conn.execute(
                    "SELECT file_path, strategy, num_trades, timerange, processed_at "
                    "FROM backtest_processed ORDER BY processed_at DESC"
                ).fetchall()
                return [dict(r) for r in rows]
        except Exception:
            return []


# ── CLI Entry Point ───────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="BacktestEmbedder: Feed backtest data into RAG pipeline")
    parser.add_argument("--file", type=str, help="Process a specific backtest ZIP or JSON file")
    parser.add_argument("--all", action="store_true", help="Process all unprocessed files in backtest_results/")
    parser.add_argument("--dir", type=str, help="Custom backtest results directory")
    parser.add_argument("--enrich", action="store_true", help="Enrich with OHLCV indicators (requires data files)")
    parser.add_argument("--history", action="store_true", help="Show processing history")
    parser.add_argument("--stats", action="store_true", help="Show PatternStatStore statistics")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

    embedder = BacktestEmbedder()

    if args.history:
        history = embedder.get_processing_history()
        if not history:
            print("No backtests processed yet.")
        else:
            for h in history:
                print(f"  {h['processed_at']}: {h['strategy']} ({h['num_trades']} trades) — {h['file_path']}")
        return

    if args.stats:
        store = PatternStatStore()
        total = store.get_total_trades()
        print(f"PatternStatStore: {total} total trades")
        if total > 0:
            # Show per-pair stats
            try:
                with store._get_conn() as conn:
                    rows = conn.execute("""
                        SELECT pair, COUNT(*) as n,
                               ROUND(AVG(CASE WHEN profit_pct > 0 THEN 1.0 ELSE 0.0 END), 3) as win_rate,
                               ROUND(AVG(profit_pct), 2) as avg_pnl
                        FROM pattern_trades
                        GROUP BY pair
                        ORDER BY n DESC
                        LIMIT 20
                    """).fetchall()
                    print(f"\n{'Pair':<15} {'Trades':>7} {'Win Rate':>10} {'Avg P&L':>10}")
                    print("-" * 45)
                    for r in rows:
                        print(f"{r['pair']:<15} {r['n']:>7} {r['win_rate']:>9.1%} {r['avg_pnl']:>9.2f}%")
            except Exception as e:
                print(f"Error: {e}")
        return

    if args.file:
        count = embedder.process_file(args.file, enrich=args.enrich)
        print(f"Processed {count} trades from {args.file}")
    elif args.all:
        count = embedder.process_all(results_dir=args.dir, enrich=args.enrich)
        print(f"Total: {count} trades processed")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
