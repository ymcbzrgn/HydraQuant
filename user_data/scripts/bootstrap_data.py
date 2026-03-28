"""
Bootstrap Data Loader — Populate the RAG pipeline with historical data.

Run this ONCE after deployment to fill PatternStatStore, ChromaDB, MAGMA,
and OHLCV patterns with backtest results and historical OHLCV data.

Usage:
    # Load all backtests + generate OHLCV patterns from downloaded data:
    python bootstrap_data.py --all

    # Only load backtests:
    python bootstrap_data.py --backtests

    # Only generate OHLCV patterns from existing OHLCV data files:
    python bootstrap_data.py --ohlcv --pairs BTC/USDT,ETH/USDT

    # Show status:
    python bootstrap_data.py --status
"""

import os
import sys
import json
import logging
import argparse

sys.path.append(os.path.dirname(__file__))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))

from ai_config import AI_DB_PATH

logger = logging.getLogger(__name__)


def load_backtests(results_dir: str = None, enrich: bool = True):
    """Load all backtest results into PatternStatStore + ChromaDB + MAGMA."""
    from backtest_embedder import BacktestEmbedder

    embedder = BacktestEmbedder()
    count = embedder.process_all(results_dir=results_dir, enrich=enrich)
    print(f"Loaded {count} trades from backtests.")
    return count


def generate_ohlcv_patterns(pairs: list = None, timeframe: str = "1h",
                            data_dir: str = None):
    """
    Generate OHLCV fingerprints from downloaded price data.
    Slides a 21-candle window across historical data and stores patterns
    with their 1h/4h/24h outcomes.
    """
    from ohlcv_pattern_matcher import OHLCVPatternMatcher

    matcher = OHLCVPatternMatcher()

    if pairs is None:
        pairs = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]

    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(os.path.dirname(AI_DB_PATH)), "data")

    total_patterns = 0

    for pair in pairs:
        pair_file = pair.replace("/", "_")
        # Futures pairs use ":USDT" suffix → "BTC/USDT:USDT" → "BTC_USDT_USDT"
        pair_file_futures = pair_file.replace(":", "_") if ":" in pair else pair_file

        # Try multiple exchange directories and filename patterns
        # Freqtrade saves futures data as: BTC_USDT_USDT-1h-futures.feather
        # Spot data as: BTC_USDT-1h.feather
        df = None
        for exchange in ["bybit", "binance", "binanceus"]:
            for subdir in ["futures", "spot", ""]:
                for name_variant in [
                    f"{pair_file_futures}-{timeframe}-futures",  # Futures format: BTC_USDT_USDT-1h-futures
                    f"{pair_file}-{timeframe}-futures",          # Futures without settle: BTC_USDT-1h-futures
                    f"{pair_file}-{timeframe}",                  # Spot format: BTC_USDT-1h
                    f"{pair_file_futures}-{timeframe}",          # Futures name, no suffix
                ]:
                    for ext in [".feather", ".json"]:
                        if subdir:
                            fpath = os.path.join(data_dir, exchange, subdir, f"{name_variant}{ext}")
                        else:
                            fpath = os.path.join(data_dir, exchange, f"{name_variant}{ext}")
                        if os.path.exists(fpath):
                            try:
                                import pandas as pd
                                if ext == ".feather":
                                    df = pd.read_feather(fpath)
                                else:
                                    df = pd.read_json(fpath)
                                print(f"Loaded {len(df)} candles from {fpath}")
                                break
                            except Exception as e:
                                print(f"Failed to load {fpath}: {e}")
                    if df is not None:
                        break
                if df is not None:
                    break
            if df is not None:
                break

        if df is None or len(df) < 50:
            print(f"No OHLCV data found for {pair}, skipping.")
            continue

        # Ensure columns
        if "close" not in df.columns:
            if "Close" in df.columns:
                df = df.rename(columns={"Close": "close", "High": "high", "Low": "low",
                                        "Open": "open", "Volume": "volume"})

        closes = df["close"].tolist()

        # Compute basic indicators for fingerprint enrichment
        import numpy as np
        rsi_period = 14
        delta = df["close"].diff()
        gain = delta.clip(lower=0).rolling(rsi_period).mean()
        loss = (-delta.clip(upper=0)).rolling(rsi_period).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi_series = 100 - (100 / (1 + rs))

        ema12 = df["close"].ewm(span=12).mean()
        ema26 = df["close"].ewm(span=26).mean()
        macd_hist = (ema12 - ema26) - (ema12 - ema26).ewm(span=9).mean()

        # Slide window: every 4 candles (to avoid excessive overlap)
        window = 20
        patterns = []

        for i in range(window + 1, len(df) - 24, 4):
            segment_closes = closes[i - window - 1:i + 1]

            # Indicators at this point
            indicators = {
                "rsi": float(rsi_series.iloc[i]) if not np.isnan(rsi_series.iloc[i]) else 50,
                "macd_hist": float(macd_hist.iloc[i]) if not np.isnan(macd_hist.iloc[i]) else 0,
            }

            fp = OHLCVPatternMatcher.compute_fingerprint(segment_closes, indicators)
            if not fp:
                continue

            # Outcomes
            current_price = closes[i]
            outcome_1h = ((closes[min(i + 1, len(closes) - 1)] - current_price) / current_price * 100) if i + 1 < len(closes) else None
            outcome_4h = ((closes[min(i + 4, len(closes) - 1)] - current_price) / current_price * 100) if i + 4 < len(closes) else None
            outcome_24h = ((closes[min(i + 24, len(closes) - 1)] - current_price) / current_price * 100) if i + 24 < len(closes) else None

            timestamp = ""
            if "date" in df.columns:
                timestamp = str(df["date"].iloc[i])

            patterns.append({
                "pair": pair,
                "timeframe": timeframe,
                "timestamp": timestamp,
                "fingerprint": fp,
                "outcome_1h": round(outcome_1h, 4) if outcome_1h is not None else None,
                "outcome_4h": round(outcome_4h, 4) if outcome_4h is not None else None,
                "outcome_24h": round(outcome_24h, 4) if outcome_24h is not None else None,
                "indicators": indicators,
            })

        if patterns:
            matcher.store_batch(patterns)
            print(f"{pair}: {len(patterns)} OHLCV patterns generated and stored.")
            total_patterns += len(patterns)

    print(f"Total: {total_patterns} OHLCV patterns. Matcher now has {matcher.get_total_patterns()} total.")
    return total_patterns


def show_status():
    """Show current data status across all stores."""
    print("═══ DATA STATUS ═══\n")

    try:
        from pattern_stat_store import PatternStatStore
        pss = PatternStatStore()
        print(f"PatternStatStore: {pss.get_total_trades()} trades")
    except Exception as e:
        print(f"PatternStatStore: ERROR ({e})")

    try:
        from ohlcv_pattern_matcher import OHLCVPatternMatcher
        matcher = OHLCVPatternMatcher()
        print(f"OHLCV Patterns:   {matcher.get_total_patterns()} patterns")
    except Exception as e:
        print(f"OHLCV Patterns:   ERROR ({e})")

    try:
        from magma_memory import MAGMAMemory
        magma = MAGMAMemory()
        stats = magma.get_stats()
        total_edges = sum(v["edges"] for v in stats.values())
        edge_detail = ", ".join(f"{k}:{v['edges']}" for k, v in stats.items())
        print(f"MAGMA Edges:      {total_edges} total ({edge_detail})")
    except Exception as e:
        print(f"MAGMA Edges:      ERROR ({e})")

    try:
        from backtest_embedder import BacktestEmbedder
        embedder = BacktestEmbedder()
        history = embedder.get_processing_history()
        print(f"Backtests Done:   {len(history)} files processed")
    except Exception as e:
        print(f"Backtests Done:   ERROR ({e})")

    try:
        from confidence_calibrator import ConfidenceCalibrator
        cal = ConfidenceCalibrator()
        brier = cal.brier_score()
        print(f"Calibrator Brier: {brier:.4f}" if brier >= 0 else "Calibrator:       No data yet")
    except Exception as e:
        print(f"Calibrator:       ERROR ({e})")

    try:
        from market_data_fetcher import MarketDataFetcher
        mdf = MarketDataFetcher()
        d = mdf.get_latest_derivatives("BTC/USDT")
        defi = mdf.get_latest_defi()
        macro = mdf.get_latest_macro()
        print(f"Market Data:      Derivatives={'YES' if d else 'NO'}, DeFi={'YES' if defi else 'NO'}, Macro={'YES' if macro else 'NO'}")
    except Exception as e:
        print(f"Market Data:      ERROR ({e})")


def main():
    parser = argparse.ArgumentParser(description="Bootstrap data into RAG pipeline")
    parser.add_argument("--all", action="store_true", help="Load backtests + generate OHLCV patterns")
    parser.add_argument("--backtests", action="store_true", help="Load backtest results only")
    parser.add_argument("--ohlcv", action="store_true", help="Generate OHLCV patterns from data files")
    parser.add_argument("--pairs", type=str, default="BTC/USDT,ETH/USDT,SOL/USDT",
                        help="Comma-separated pairs for OHLCV generation")
    parser.add_argument("--status", action="store_true", help="Show data status")
    parser.add_argument("--dir", type=str, help="Custom backtest results directory")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

    if args.status:
        show_status()
    elif args.all:
        load_backtests(results_dir=args.dir)
        pairs = [p.strip() for p in args.pairs.split(",")]
        generate_ohlcv_patterns(pairs=pairs)
        print("\n")
        show_status()
    elif args.backtests:
        load_backtests(results_dir=args.dir)
    elif args.ohlcv:
        pairs = [p.strip() for p in args.pairs.split(",")]
        generate_ohlcv_patterns(pairs=pairs)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
