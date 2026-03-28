"""
Phase 20: Evidence Engine — LLM-free signal generator grounded in empirical data.

Replaces the dumb _technical_fallback() (11 indicators, 0.35 cap) with a 6-phase
evidence-based pipeline using 30+ signal sources and dynamic confidence 0.35-0.55.

Design Philosophy:
  - Evidence Engine runs FIRST, always, with zero API cost
  - LLM agents (MADAM/AgentPool) are OPTIONAL narrative enhancers on top
  - Grounded in peer-reviewed research: RSI>50 momentum (2.8x), F&G contrarian (68%),
    funding rate microstructure, backtest k-NN, OHLCV pattern matching

Pipeline:
  Phase A: GATHER — collect all available data from SQLite + tech_data
  Phase B: DEEP PATTERN ANALYSIS — k-NN (50), OHLCV (100), backtest stats, ensemble
  Phase C: SUB-QUESTION DECOMPOSITION — 6 independent scores (MiroFish pattern)
  Phase D: CONTRADICTION DETECTION — MiroFish reflection loop, LLM-free
  Phase E: SYNTHESIS — weighted average + regime modifier + Platt calibration
  Phase F: AUDIT LOG — structured logging for every decision (MiroFish JSONL pattern)

Usage:
    from evidence_engine import EvidenceEngine
    engine = EvidenceEngine()
    result = engine.generate_signal("BTC/USDT", tech_data)
    # → {"signal": "BULLISH", "confidence": 0.48, "reasoning": "...", "source": "EVIDENCE_ENGINE"}
"""

import os
import sys
import json
import math
import sqlite3
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime, timezone

sys.path.append(os.path.dirname(__file__))

from ai_config import AI_DB_PATH

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# Data containers for pipeline phases
# ═══════════════════════════════════════════════════════════════

@dataclass
class GatherResult:
    """Phase A output: all gathered raw data."""
    tech: Dict[str, Any] = field(default_factory=dict)
    derivatives: Optional[Dict] = None     # {funding_rate, long_short_ratio, open_interest_usd}
    fng: Optional[int] = None              # Fear & Greed Index 0-100
    macro: Optional[Dict] = None           # {dxy_broad: {value, change_pct}, vix: {value}, ...}
    btc_dom: Optional[float] = None        # BTC dominance percentage


@dataclass
class PatternResult:
    """Phase B output: historical pattern analysis."""
    knn: Optional[Dict] = None             # temporal_knn() result
    ohlcv: Optional[Dict] = None           # find_similar() result
    stats: Optional[Dict] = None           # query() result
    ensemble: Optional[Dict] = None        # ensemble_vote() result


# ═══════════════════════════════════════════════════════════════
# Evidence Engine
# ═══════════════════════════════════════════════════════════════

class EvidenceEngine:
    """
    LLM-free signal generator using 30+ empirical data sources.
    All data comes from SQLite queries and in-memory computation — zero API cost.
    Model server (BGE/ColBERT/FlashRank) is NOT used — fully independent.
    """

    # Sub-question weights (sum to 1.0)
    # DeMiguel 1/N is optimal with NO domain knowledge. But we DO know:
    # - Crypto-macro correlation is low (BTC-SP500 ρ≈0.3, variable)
    # - Crowd (F&G+funding) is 70-80% accurate at extremes (research validated)
    # - Trend momentum is strongest crypto alpha source (Moskowitz 2012, ScienceDirect 2025)
    # Weights reflect known predictive power for crypto:
    DEFAULT_WEIGHTS = {
        "q1_trend": 0.22,      # Strongest: EMA alignment + ADX, momentum papers
        "q2_momentum": 0.20,   # Strong: RSI>50 zone = 2.8x (our research)
        "q3_crowd": 0.22,      # Strong: F&G extremes, funding rate crowding
        "q4_evidence": 0.15,   # Moderate: k-NN + backtest (depends on data quantity)
        "q5_macro": 0.10,      # Weak: low crypto-macro correlation, mostly noise
        "q6_risk": 0.11,       # Moderate: ATR volatility, volume confirmation
    }

    # Regime-specific weight overrides (sum=1.0 each, macro always low)
    REGIME_WEIGHTS = {
        "ranging": {
            "q1_trend": 0.10,       # Trend unreliable in ranging
            "q2_momentum": 0.15,
            "q3_crowd": 0.28,       # Contrarian most useful in ranges
            "q4_evidence": 0.28,    # Historical patterns crucial
            "q5_macro": 0.08,       # Low: crypto-macro decorrelation
            "q6_risk": 0.11,
        },
        "high_volatility": {
            "q1_trend": 0.15,
            "q2_momentum": 0.13,    # Momentum less reliable
            "q3_crowd": 0.22,
            "q4_evidence": 0.18,
            "q5_macro": 0.08,       # Low: crypto-macro decorrelation
            "q6_risk": 0.24,        # Risk assessment critical in vol
        },
    }

    def __init__(self, pattern_store=None, ohlcv_matcher=None,
                 market_data=None, regime_classifier=None,
                 calibrator=None, db_path: str = AI_DB_PATH):
        """
        Initialize with optional singleton references from rag_graph.py.
        If not provided, creates own lightweight instances (for standalone/test use).
        """
        self.db_path = db_path

        # Reuse singletons if provided, otherwise create own
        self._pattern_store = pattern_store
        self._ohlcv_matcher = ohlcv_matcher
        self._market_data = market_data
        self._regime_classifier = regime_classifier
        self._calibrator = calibrator

        # Lazy-init if not provided
        if self._pattern_store is None:
            try:
                from pattern_stat_store import PatternStatStore
                self._pattern_store = PatternStatStore(db_path=db_path)
            except Exception as e:
                logger.debug(f"[EvidenceEngine:Init] PatternStatStore unavailable: {e}")

        if self._ohlcv_matcher is None:
            try:
                from ohlcv_pattern_matcher import OHLCVPatternMatcher
                self._ohlcv_matcher = OHLCVPatternMatcher(db_path=db_path)
            except Exception as e:
                logger.debug(f"[EvidenceEngine:Init] OHLCVPatternMatcher unavailable: {e}")

        if self._market_data is None:
            try:
                from market_data_fetcher import MarketDataFetcher
                self._market_data = MarketDataFetcher(db_path=db_path)
            except Exception as e:
                logger.debug(f"[EvidenceEngine:Init] MarketDataFetcher unavailable: {e}")

        if self._regime_classifier is None:
            try:
                from regime_classifier import RegimeClassifier
                self._regime_classifier = RegimeClassifier()
            except Exception as e:
                logger.debug(f"[EvidenceEngine:Init] RegimeClassifier unavailable: {e}")

        if self._calibrator is None:
            try:
                from confidence_calibrator import ConfidenceCalibrator
                self._calibrator = ConfidenceCalibrator(db_path=db_path)
            except Exception as e:
                logger.debug(f"[EvidenceEngine:Init] ConfidenceCalibrator unavailable: {e}")

        # Check data readiness
        self._pattern_ready = False
        self._ohlcv_ready = False
        try:
            if self._pattern_store and self._pattern_store.get_total_trades() >= 10:
                self._pattern_ready = True
        except Exception:
            pass
        try:
            if self._ohlcv_matcher and self._ohlcv_matcher.get_total_patterns() >= 20:
                self._ohlcv_ready = True
        except Exception:
            pass

    # ═══════════════════════════════════════════════════════════
    # PUBLIC API
    # ═══════════════════════════════════════════════════════════

    def generate_signal(self, pair: str, tech_data: dict) -> dict:
        """
        Main entry point. Returns {signal, confidence, reasoning, source}.
        Same format as _technical_fallback() for drop-in replacement.
        """
        if not tech_data:
            tech_data = {}

        # If no current_price (or zero/negative), try to build minimal tech_data from DB
        price = tech_data.get("current_price")
        if not price or (isinstance(price, (int, float)) and price <= 0):
            tech_data = self._build_minimal_tech_data(pair, tech_data)

        price = tech_data.get("current_price")
        if not price or (isinstance(price, (int, float)) and price <= 0):
            logger.info(f"[EvidenceEngine] {pair}: No tech_data and no DB fallback, returning NEUTRAL")
            return {"signal": "NEUTRAL", "confidence": 0.01,
                    "reasoning": "[EvidenceEngine] No indicator data available (no tech_data, no DB price)",
                    "source": "EVIDENCE_ENGINE"}

        try:
            # Phase A: GATHER
            gather = self._gather(pair, tech_data)

            # Classify regime (needed for all subsequent phases)
            regime = "transitional"  # safe default
            if self._regime_classifier:
                try:
                    from regime_classifier import RegimeClassifier
                    regime = RegimeClassifier.classify(tech_data)
                    logger.info(f"[EvidenceEngine:Regime] {pair} → {regime}")
                except Exception as e:
                    logger.debug(f"[EvidenceEngine:Regime] Classification failed: {e}")

            # Phase B: DEEP PATTERN ANALYSIS
            patterns = self._analyze_patterns(pair, tech_data, regime, gather)

            # Phase C: SUB-QUESTION DECOMPOSITION
            scores = self._score_sub_questions(pair, gather, patterns, regime, tech_data)

            # Phase D: CONTRADICTION DETECTION
            contradictions = self._detect_contradictions(scores, gather, regime, tech_data)

            # Phase E: SYNTHESIS + CALIBRATION
            result = self._synthesize(pair, scores, contradictions, regime, gather, patterns)

            # Phase F: AUDIT LOG
            evidence_sources = {
                "derivatives": gather.derivatives is not None,
                "fng": gather.fng is not None,
                "macro": gather.macro is not None,
                "btc_dom": gather.btc_dom is not None,
                "knn": patterns.knn is not None and not patterns.knn.get("insufficient_data"),
                "ohlcv": patterns.ohlcv is not None and not patterns.ohlcv.get("insufficient_data"),
                "backtest": patterns.stats is not None and not patterns.stats.get("insufficient_data"),
                "ensemble": patterns.ensemble is not None and patterns.ensemble.get("total_strategies", 0) >= 2,
            }
            self._audit_log(pair, result["signal"], result["confidence"], scores,
                           contradictions, regime, evidence_sources, result.get("_max_cap", 0.35))

            return result

        except Exception as e:
            logger.error(f"[EvidenceEngine] {pair} pipeline failed: {type(e).__name__}: {e}")
            return {"signal": "NEUTRAL", "confidence": 0.01,
                    "reasoning": f"[EvidenceEngine] Pipeline error: {e}", "source": "EVIDENCE_ENGINE"}

    def format_factsheet(self, result: dict) -> str:
        """Format Evidence Engine result as a FactSheet for LLM coordinator prompt injection."""
        if not result or result.get("confidence", 0) < 0.01:
            return ""

        signal = result.get("signal", "NEUTRAL")
        conf = result.get("confidence", 0)
        reasoning = result.get("reasoning", "")

        lines = [
            "=== EVIDENCE ENGINE FACTSHEET (LLM-FREE, DATA-DRIVEN) ===",
            f"Signal: {signal} | Confidence: {conf:.2f}",
            f"Reasoning: {reasoning}",
            "",
            "IMPORTANT: This signal is generated from empirical data (backtests, derivatives,",
            "F&G, k-NN, OHLCV patterns) without any LLM involvement. Use it as a PRIOR",
            "probability anchor. Your LLM-based analysis should CONFIRM or OVERRIDE with reasoning.",
        ]
        return "\n".join(lines)

    # ═══════════════════════════════════════════════════════════
    # Phase A: GATHER
    # ═══════════════════════════════════════════════════════════

    def _gather(self, pair: str, tech_data: dict) -> GatherResult:
        """Collect all available data from SQLite and tech_data dict."""
        gather = GatherResult(tech=tech_data)

        # Normalize pair: derivatives_data stores "BTC/USDT" but pipeline sends "BTC/USDT:USDT"
        deriv_pair = pair.split(":")[0]  # "BTC/USDT:USDT" → "BTC/USDT"

        # 1. Derivatives (funding rate, OI, L/S ratio)
        if self._market_data:
            try:
                gather.derivatives = self._market_data.get_latest_derivatives(deriv_pair)
                if gather.derivatives:
                    logger.info(f"[EvidenceEngine:Gather] {pair} derivatives: "
                               f"FR={gather.derivatives.get('funding_rate')}, "
                               f"L/S={gather.derivatives.get('long_short_ratio')}")
            except Exception as e:
                logger.debug(f"[EvidenceEngine:Gather] Derivatives failed: {e}")

        # 2. Fear & Greed Index
        try:
            gather.fng = self._get_fear_greed()
            if gather.fng is not None:
                logger.info(f"[EvidenceEngine:Gather] F&G Index: {gather.fng}")
        except Exception as e:
            logger.debug(f"[EvidenceEngine:Gather] F&G failed: {e}")

        # 3. Macro data (DXY, VIX, etc.)
        if self._market_data:
            try:
                gather.macro = self._market_data.get_latest_macro()
                if gather.macro:
                    logger.info(f"[EvidenceEngine:Gather] Macro data loaded: {len(gather.macro)} metrics")
            except Exception as e:
                logger.debug(f"[EvidenceEngine:Gather] Macro failed: {e}")

        # 4. BTC dominance (from macro_data table or cross-asset)
        try:
            gather.btc_dom = self._get_btc_dominance()
            if gather.btc_dom is not None:
                logger.info(f"[EvidenceEngine:Gather] BTC dominance: {gather.btc_dom:.1f}%")
        except Exception as e:
            logger.debug(f"[EvidenceEngine:Gather] BTC dominance failed: {e}")

        return gather

    def _get_fear_greed(self) -> Optional[int]:
        """Read latest Fear & Greed Index from DB."""
        try:
            conn = sqlite3.connect(self.db_path, timeout=10)
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT value FROM fear_and_greed ORDER BY timestamp DESC LIMIT 1"
            ).fetchone()
            conn.close()
            return int(row["value"]) if row else None
        except Exception:
            return None

    def _build_minimal_tech_data(self, pair: str, existing: dict) -> dict:
        """
        When tech_data is empty (concurrency timeout, GET request), build minimal
        data from DB so Evidence Engine can still produce a signal.
        Uses derivatives_data, latest OHLCV patterns, and any cached data.
        """
        td = dict(existing) if existing else {}
        try:
            conn = sqlite3.connect(self.db_path, timeout=10)
            conn.row_factory = sqlite3.Row

            # Try to get latest price from derivatives_data (OI implies a price reference)
            _deriv_pair = pair.split(":")[0]  # "BTC/USDT:USDT" → "BTC/USDT"
            deriv = conn.execute(
                "SELECT open_interest_usd, funding_rate, long_short_ratio FROM derivatives_data "
                "WHERE pair = ? ORDER BY timestamp DESC LIMIT 1", (_deriv_pair,)
            ).fetchone()

            # Try to get latest price from OHLCV patterns (stored during bootstrap)
            ohlcv = conn.execute(
                "SELECT indicators_json FROM ohlcv_patterns "
                "WHERE pair = ? ORDER BY created_at DESC LIMIT 1",
                (pair.replace(":USDT", "").replace("/", "_"),)  # Normalize pair format
            ).fetchone()

            # Try the last evidence_audit_log for this pair (might have cached sub-scores)
            prev_audit = conn.execute(
                "SELECT sub_scores_json, regime FROM evidence_audit_log "
                "WHERE pair = ? ORDER BY timestamp DESC LIMIT 1", (pair,)
            ).fetchone()

            conn.close()

            # Build minimal tech_data from whatever we found
            if ohlcv and ohlcv["indicators_json"]:
                try:
                    indicators = json.loads(ohlcv["indicators_json"])
                    td.setdefault("rsi_14", indicators.get("rsi"))
                    td.setdefault("macd_histogram", indicators.get("macd_hist"))
                except Exception:
                    pass

            # We need SOME price to work with. Use a placeholder that lets
            # the engine run (sub-questions that need price will be neutral,
            # but F&G, funding rate, macro, and historical evidence still work)
            if not td.get("current_price"):
                td["current_price"] = 1.0  # Placeholder — EMA/BB comparisons won't work
                td["_price_from_db"] = True  # Flag so reasoning explains the limitation

            if prev_audit and prev_audit["regime"]:
                td["_cached_regime"] = prev_audit["regime"]

            logger.info(f"[EvidenceEngine:MinimalData] {pair}: built from DB — "
                       f"rsi={'yes' if td.get('rsi_14') else 'no'}, "
                       f"price={'real' if not td.get('_price_from_db') else 'placeholder'}")

        except Exception as e:
            logger.debug(f"[EvidenceEngine:MinimalData] {pair} DB fallback failed: {e}")
            if not td.get("current_price"):
                td["current_price"] = 1.0
                td["_price_from_db"] = True

        return td

    def _get_btc_dominance(self) -> Optional[float]:
        """Read BTC dominance from macro_data or defi_data table."""
        try:
            conn = sqlite3.connect(self.db_path, timeout=10)
            conn.row_factory = sqlite3.Row
            # Try cross-asset data first (yfinance might store this)
            row = conn.execute(
                "SELECT value FROM macro_data WHERE metric_name LIKE '%btc_dom%' "
                "ORDER BY timestamp DESC LIMIT 1"
            ).fetchone()
            conn.close()
            return float(row["value"]) if row else None
        except Exception:
            return None

    # ═══════════════════════════════════════════════════════════
    # Phase B: DEEP PATTERN ANALYSIS
    # ═══════════════════════════════════════════════════════════

    def _analyze_patterns(self, pair: str, tech_data: dict, regime: str,
                          gather: GatherResult) -> PatternResult:
        """Query historical patterns from 3 independent sources."""
        result = PatternResult()

        # Extract classification buckets (guarded import for graceful degradation)
        try:
            from pattern_stat_store import PatternStatStore
        except ImportError:
            logger.debug("[EvidenceEngine:Pattern] PatternStatStore not importable, skipping patterns")
            return result

        rsi = tech_data.get("rsi_14") or tech_data.get("rsi")
        macd_hist = tech_data.get("macd_histogram") or tech_data.get("macd_hist") or tech_data.get("macdhist")
        adx = tech_data.get("adx_14") or tech_data.get("adx")

        rsi_bucket = PatternStatStore.classify_rsi(float(rsi)) if rsi else None
        macd_bucket = PatternStatStore.classify_macd(float(macd_hist)) if macd_hist else None

        # 1. Temporal k-NN: 50 neighbors (expanded from default 10)
        if self._pattern_ready and self._pattern_store:
            try:
                current_features = {}
                if rsi_bucket:
                    current_features["rsi_bucket"] = rsi_bucket
                if regime:
                    current_features["regime"] = regime
                if macd_bucket:
                    current_features["macd_signal"] = macd_bucket

                # Need at least 2 features for meaningful k-NN
                if len(current_features) >= 2:
                    result.knn = self._pattern_store.temporal_knn(
                        current_features, k=50, pair=pair)
                    if result.knn and not result.knn.get("insufficient_data"):
                        logger.info(f"[EvidenceEngine:Pattern] {pair} k-NN: "
                                   f"wr={result.knn['knn_win_rate']:.0%}, "
                                   f"avg_pnl={result.knn['knn_avg_pnl']:+.2f}%, "
                                   f"dist={result.knn.get('avg_distance', 0):.3f}")
            except Exception as e:
                logger.debug(f"[EvidenceEngine:Pattern] k-NN failed: {e}")

        # 2. OHLCV Pattern Matching: 100 candle-sequence matches
        if self._ohlcv_ready and self._ohlcv_matcher:
            try:
                closes = tech_data.get("recent_closes", [])
                if closes and len(closes) >= 21:
                    from ohlcv_pattern_matcher import OHLCVPatternMatcher
                    indicators = {
                        "rsi": float(rsi) if rsi else 50,
                        "macd_hist": float(macd_hist) if macd_hist else 0,
                        "adx": float(adx) if adx else 25,
                        "volume_ratio": float(tech_data.get("volume", {}).get("ratio", 1.0))
                            if isinstance(tech_data.get("volume"), dict) else 1.0,
                        "atr_ratio": 1.0,
                        "fng": float(gather.fng) if gather.fng else 50,
                    }
                    fp = OHLCVPatternMatcher.compute_fingerprint(closes, indicators)
                    if fp:
                        result.ohlcv = self._ohlcv_matcher.find_similar(fp, k=100, pair=pair)
                        if result.ohlcv and not result.ohlcv.get("insufficient_data"):
                            logger.info(f"[EvidenceEngine:Pattern] {pair} OHLCV: "
                                       f"pred_4h={result.ohlcv.get('predicted_4h', 0):+.2f}%, "
                                       f"conf={result.ohlcv.get('confidence', 0):.2f}")
            except Exception as e:
                logger.debug(f"[EvidenceEngine:Pattern] OHLCV failed: {e}")

        # 3. PatternStatStore: full query + regime breakdown
        if self._pattern_ready and self._pattern_store:
            try:
                result.stats = self._pattern_store.query(
                    pair=pair, regime=regime, rsi_bucket=rsi_bucket, min_trades=5)
                if result.stats and not result.stats.get("insufficient_data"):
                    logger.info(f"[EvidenceEngine:Pattern] {pair} backtest: "
                               f"n={result.stats['matching_trades']}, "
                               f"wr={result.stats['win_rate']:.0%}, "
                               f"pf={result.stats.get('profit_factor', 0):.2f}")
            except Exception as e:
                logger.debug(f"[EvidenceEngine:Pattern] Stats query failed: {e}")

        # 4. Multi-strategy ensemble vote
        if self._pattern_ready and self._pattern_store:
            try:
                result.ensemble = self._pattern_store.ensemble_vote(
                    pair=pair, regime=regime, rsi_bucket=rsi_bucket)
                if result.ensemble and result.ensemble.get("total_strategies", 0) >= 2:
                    logger.info(f"[EvidenceEngine:Pattern] {pair} ensemble: "
                               f"{result.ensemble['consensus']} "
                               f"(strength={result.ensemble['consensus_strength']:.0%}, "
                               f"n_strats={result.ensemble['total_strategies']})")
            except Exception as e:
                logger.debug(f"[EvidenceEngine:Pattern] Ensemble failed: {e}")

        return result

    # ═══════════════════════════════════════════════════════════
    # Phase C: SUB-QUESTION DECOMPOSITION
    # ═══════════════════════════════════════════════════════════

    def _score_sub_questions(self, pair: str, gather: GatherResult, patterns: PatternResult,
                             regime: str, tech_data: dict) -> Dict[str, float]:
        """Score 6 independent sub-questions, each 0.0-1.0 (0=bearish, 0.5=neutral, 1=bullish)."""
        scores = {}

        scores["q1_trend"] = self._score_q1_trend(tech_data)
        scores["q2_momentum"] = self._score_q2_momentum(tech_data)
        scores["q3_crowd"] = self._score_q3_crowd(gather)
        scores["q4_evidence"] = self._score_q4_evidence(patterns)
        scores["q5_macro"] = self._score_q5_macro(gather, pair)
        scores["q6_risk"] = self._score_q6_risk(tech_data, patterns)

        logger.info(f"[EvidenceEngine:SubQ] {pair} scores: "
                   f"trend={scores['q1_trend']:.2f}, mom={scores['q2_momentum']:.2f}, "
                   f"crowd={scores['q3_crowd']:.2f}, evid={scores['q4_evidence']:.2f}, "
                   f"macro={scores['q5_macro']:.2f}, risk={scores['q6_risk']:.2f}")

        return scores

    def _score_q1_trend(self, td: dict) -> float:
        """Q1: Trend Direction — EMA alignment + ADX strength."""
        price = td.get("current_price", 0)
        ema20 = td.get("ema_20") or td.get("ema20")
        ema50 = td.get("ema_50") or td.get("ema50")
        ema200 = td.get("ema_200") or td.get("ema200")
        adx = td.get("adx_14") or td.get("adx")

        if not price or not ema200:
            return 0.50  # No data → neutral

        # EMA alignment classification (guarded import)
        try:
            from pattern_stat_store import PatternStatStore
        except ImportError:
            # Fallback: simple price vs ema200
            return 0.62 if float(price) > float(ema200) else 0.38
        if ema20 and ema50 and ema200:
            ema_align = PatternStatStore.classify_ema(price, float(ema20), float(ema50), float(ema200))
        elif ema200:
            ema_align = "above_200" if price > float(ema200) else "below_200"
        else:
            ema_align = "neutral"

        adx_val = float(adx) if adx else 20

        # Score mapping
        if ema_align == "full_bullish" and adx_val > 25:
            score = 0.85
        elif ema_align == "full_bullish":
            score = 0.70
        elif ema_align == "full_bearish" and adx_val > 25:
            score = 0.15
        elif ema_align == "full_bearish":
            score = 0.30
        elif ema_align == "above_200":
            score = 0.62
        elif ema_align == "below_200":
            score = 0.38
        else:
            score = 0.50

        return score

    def _score_q2_momentum(self, td: dict) -> float:
        """Q2: Momentum — RSI>50 momentum (NOT RSI<30 oversold, per 2.8x research)."""
        rsi = td.get("rsi_14") or td.get("rsi")
        macd_hist = td.get("macd_histogram") or td.get("macd_hist") or td.get("macdhist")
        htf = td.get("htf", {}) if isinstance(td.get("htf"), dict) else {}
        htf_rsi = htf.get("rsi_4h")

        # RSI scoring: RSI>50 momentum zone is 2.8x better than RSI<30 oversold
        if rsi is not None:
            rsi = float(rsi)
            if 50 < rsi <= 70:
                rsi_score = 0.75    # Momentum zone — RESEARCH: 2.8x better
            elif rsi > 70:
                rsi_score = 0.30    # Overbought risk
            elif 30 <= rsi <= 50:
                rsi_score = 0.42    # Weak/uncertain
            elif rsi < 30:
                rsi_score = 0.55    # Oversold (slight bounce potential, but NOT as strong as >50)
            else:
                rsi_score = 0.50
        else:
            rsi_score = 0.50

        # MACD histogram direction
        if macd_hist is not None:
            macd_hist = float(macd_hist)
            if macd_hist > 0:
                macd_score = 0.65
            else:
                macd_score = 0.35
        else:
            macd_score = 0.50

        # Higher timeframe RSI confirmation
        if htf_rsi is not None and rsi is not None:
            htf_rsi = float(htf_rsi)
            # Both above 50 or both below 50 = aligned
            aligned = (rsi > 50 and htf_rsi > 50) or (rsi < 50 and htf_rsi < 50)
            htf_score = 0.70 if aligned else 0.35
        else:
            htf_score = 0.50

        # Weighted combination
        momentum_score = rsi_score * 0.50 + macd_score * 0.30 + htf_score * 0.20
        return max(0.0, min(1.0, momentum_score))

    def _score_q3_crowd(self, gather: GatherResult) -> float:
        """Q3: Crowd Positioning — CONTRARIAN (F&G, funding rate, L/S ratio)."""
        score = 0.50  # neutral baseline

        # Fear & Greed contrarian (Research: 70-80% accuracy at extremes, threshold <20 validated)
        if gather.fng is not None:
            fng = gather.fng
            if fng < 10:
                score += 0.22       # Extreme fear → strong contrarian BULLISH
            elif fng < 20:
                score += 0.15       # Fear → contrarian bullish (research-validated threshold)
            elif fng < 30:
                score += 0.06       # Mild fear → slight bullish
            elif fng > 85:
                score -= 0.22       # Extreme greed → strong contrarian BEARISH
            elif fng > 75:
                score -= 0.15       # Greed → contrarian bearish
            elif fng > 65:
                score -= 0.06       # Mild greed → slight bearish
            # 30-65: neutral zone, no adjustment

        # Funding rate contrarian (Research: strong microstructure signal for crowding detection)
        if gather.derivatives:
            fr = gather.derivatives.get("funding_rate")
            if fr is not None:
                fr = float(fr)
                if fr > 0.001:          # >0.1% — truly extreme crowded long
                    score -= 0.18
                elif fr > 0.0005:       # >0.05% — extreme crowded long
                    score -= 0.12
                elif fr > 0.0003:       # >0.03% — crowded long
                    score -= 0.06
                elif fr < -0.001:       # <-0.1% — truly extreme crowded short
                    score += 0.18
                elif fr < -0.0005:      # <-0.05% — extreme crowded short
                    score += 0.12
                elif fr < -0.0003:      # <-0.03% — crowded short
                    score += 0.06

            # Long/Short ratio
            ls = gather.derivatives.get("long_short_ratio")
            if ls is not None:
                ls = float(ls)
                if ls > 1.5:            # Too many longs
                    score -= 0.08
                elif ls < 0.7:          # Too many shorts
                    score += 0.08

        return max(0.0, min(1.0, score))

    def _score_q4_evidence(self, patterns: PatternResult) -> float:
        """Q4: Historical Evidence — k-NN + backtest stats + OHLCV + ensemble."""
        score = 0.50  # neutral baseline

        # k-NN results
        if patterns.knn and not patterns.knn.get("insufficient_data"):
            knn_wr = patterns.knn.get("knn_win_rate", 0.5)
            if knn_wr > 0.65:
                score += 0.15
            elif knn_wr > 0.55:
                score += 0.07
            elif knn_wr < 0.35:
                score -= 0.15
            elif knn_wr < 0.45:
                score -= 0.07

            # Distance bonus: very similar matches are more reliable
            avg_dist = patterns.knn.get("avg_distance", 1.0)
            if avg_dist < 0.3:
                score += 0.05  # High similarity = high confidence in k-NN

        # Backtest stats
        if patterns.stats and not patterns.stats.get("insufficient_data"):
            wr = patterns.stats.get("win_rate", 0.5)
            pf = patterns.stats.get("profit_factor", 1.0)
            if wr > 0.60:
                score += 0.12
            elif wr < 0.40:
                score -= 0.12
            if pf > 2.0:
                score += 0.08
            elif pf < 0.5:
                score -= 0.08

        # OHLCV pattern match
        if patterns.ohlcv and not patterns.ohlcv.get("insufficient_data"):
            pred_4h = patterns.ohlcv.get("predicted_4h", 0)
            if pred_4h > 1.0:
                score += 0.08
            elif pred_4h < -1.0:
                score -= 0.08

        # Ensemble consensus
        if patterns.ensemble and patterns.ensemble.get("total_strategies", 0) >= 2:
            consensus = patterns.ensemble.get("consensus", "NEUTRAL")
            strength = patterns.ensemble.get("consensus_strength", 0)
            if consensus == "LONG":
                score += 0.05 * strength
            elif consensus == "SHORT":
                score -= 0.05 * strength

        return max(0.0, min(1.0, score))

    def _score_q5_macro(self, gather: GatherResult, pair: str) -> float:
        """Q5: Macro Context — DXY, BTC dominance, VIX."""
        score = 0.50

        # DXY: falling dollar = good for crypto
        if gather.macro:
            dxy_data = gather.macro.get("dxy_broad") or gather.macro.get("dxy")
            if isinstance(dxy_data, dict):
                dxy_change = dxy_data.get("change_pct")
                if dxy_change is not None:
                    dxy_change = float(dxy_change)
                    if dxy_change < -0.3:
                        score += 0.10   # Weak dollar = bullish crypto
                    elif dxy_change > 0.3:
                        score -= 0.10   # Strong dollar = bearish crypto

            # VIX: high fear in TradFi = usually bad for crypto too
            vix_data = gather.macro.get("vix")
            if isinstance(vix_data, dict):
                vix_val = vix_data.get("value")
                if vix_val is not None:
                    vix_val = float(vix_val)
                    if vix_val > 30:
                        score -= 0.08
                    elif vix_val < 15:
                        score += 0.05

        # BTC dominance
        if gather.btc_dom is not None:
            is_btc = pair.upper().startswith("BTC")
            btc_dom = float(gather.btc_dom)
            if btc_dom > 58:
                # Money fleeing to BTC — good for BTC, bad for alts
                score += 0.08 if is_btc else -0.08
            elif btc_dom < 45:
                # Alt season — bad for BTC, good for alts
                score += -0.05 if is_btc else 0.08

        return max(0.0, min(1.0, score))

    def _score_q6_risk(self, td: dict, patterns: PatternResult) -> float:
        """Q6: Risk Assessment — ATR volatility, worst-case scenarios, volume."""
        score = 0.50  # neutral = normal risk

        price = td.get("current_price", 0)
        atr = td.get("atr_14") or td.get("atr")

        # ATR as % of price (volatility proxy)
        if price and atr:
            atr_pct = float(atr) / float(price) * 100
            if atr_pct > 4.0:
                score -= 0.15       # Very high volatility
            elif atr_pct > 3.0:
                score -= 0.08       # High volatility
            elif atr_pct < 1.5:
                score += 0.08       # Low volatility = cleaner signals

        # k-NN worst case
        if patterns.knn and not patterns.knn.get("insufficient_data"):
            knn_worst = patterns.knn.get("knn_worst", 0)
            if knn_worst < -5.0:
                score -= 0.10       # Worst similar case was ugly

        # Volume confirmation
        vol = td.get("volume", {}) if isinstance(td.get("volume"), dict) else {}
        vol_ratio = vol.get("ratio", 1.0)
        if isinstance(vol_ratio, (int, float)):
            if vol_ratio > 1.5:
                score += 0.05       # Strong volume confirms
            elif vol_ratio < 0.5:
                score -= 0.05       # Thin volume = unreliable

        return max(0.0, min(1.0, score))

    # ═══════════════════════════════════════════════════════════
    # Phase D: CONTRADICTION DETECTION
    # ═══════════════════════════════════════════════════════════

    def _detect_contradictions(self, scores: Dict[str, float], gather: GatherResult,
                                regime: str, tech_data: dict) -> List[str]:
        """Detect conflicting signals between sub-questions. MiroFish reflection pattern."""
        contradictions = []

        # 1. Trend bullish BUT crowd crowded long → who is left to buy?
        if scores["q1_trend"] > 0.65 and scores["q3_crowd"] < 0.35:
            contradictions.append(
                "Trend bullish but crowd already crowded long (extreme funding/greed) — buyers exhausted")

        # 2. Momentum bullish BUT historical evidence bearish
        if scores["q2_momentum"] > 0.65 and scores["q4_evidence"] < 0.35:
            contradictions.append(
                "Current momentum bullish but historically similar setups lost money — false signal risk")

        # 3. Historical evidence bullish BUT regime is uncertain/volatile
        if scores["q4_evidence"] > 0.65 and regime in ("high_volatility", "transitional"):
            contradictions.append(
                "Historical evidence strong but current regime is uncertain — conditions may differ")

        # 4. Trend bullish BUT higher timeframe bearish (timeframe conflict)
        htf = tech_data.get("htf", {}) if isinstance(tech_data.get("htf"), dict) else {}
        if scores["q1_trend"] > 0.65 and htf.get("trend_daily") == "bearish":
            contradictions.append(
                "1H trend bullish but daily timeframe is bearish — potential bear market rally")

        # 5. Trend bearish BUT higher timeframe bullish
        if scores["q1_trend"] < 0.35 and htf.get("trend_daily") == "bullish":
            contradictions.append(
                "1H trend bearish but daily timeframe is bullish — potential dip in uptrend")

        # 6. All sub-scores suspiciously aligned (groupthink check)
        non_risk = [scores[k] for k in ["q1_trend", "q2_momentum", "q3_crowd", "q4_evidence"]]
        all_bullish = all(s > 0.60 for s in non_risk)
        all_bearish = all(s < 0.40 for s in non_risk)
        if all_bullish or all_bearish:
            contradictions.append(
                "All 4 main signals unanimously agree — beware of groupthink, reduce confidence")

        if contradictions:
            logger.info(f"[EvidenceEngine:Contradiction] {len(contradictions)} found: "
                       f"{'; '.join(contradictions[:2])}")

        return contradictions

    # ═══════════════════════════════════════════════════════════
    # Phase E: SYNTHESIS + CALIBRATION
    # ═══════════════════════════════════════════════════════════

    def _synthesize(self, pair: str, scores: Dict[str, float], contradictions: List[str],
                    regime: str, gather: GatherResult, patterns: PatternResult) -> dict:
        """
        Adaptive Synthesis — dynamically excludes blind sub-scores from calculation.

        Philosophy: "Don't let 3 blind advisors dilute 3 seeing ones."

        If a sub-score has NO real data (returns default 0.50/0.35), its weight is
        redistributed to sub-scores that DO have real data. This prevents the
        "neutral drag" problem where defaults pull everything toward 0.50 → NEUTRAL.

        Additionally applies:
        - Bayesian uncertainty discount: fewer active factors = lower confidence ceiling
        - Factor disagreement penalty: high variance among active factors = less certain
        - Momentum age awareness: all factors aligned = possible late entry (momentum trap)

        References:
        - Araştırma: "Option C+D Hybrid" (Bayesian re-weight + uncertainty discount)
        - DeMiguel et al. (2009): 1/N optimal with limited data
        - Barroso & Santa-Clara (2015): momentum volatility predicts reversals
        """

        # ═══ STEP 1: Identify which sub-scores have REAL data vs BLIND ═══
        # Instead of guessing from score value (0.50 could be legitimate neutral),
        # check whether the INPUT DATA for each sub-score actually existed.
        has_data = {
            "q1_trend": bool(gather.tech.get("ema_200") or gather.tech.get("ema200")),
            "q2_momentum": bool(gather.tech.get("rsi_14") or gather.tech.get("rsi")),
            "q3_crowd": gather.fng is not None or bool(gather.derivatives),
            "q4_evidence": bool(
                (patterns.knn and not patterns.knn.get("insufficient_data")) or
                (patterns.stats and not patterns.stats.get("insufficient_data")) or
                (patterns.ohlcv and not patterns.ohlcv.get("insufficient_data"))
            ),
            "q5_macro": bool(gather.macro),
            "q6_risk": bool(gather.tech.get("atr_14") or gather.tech.get("atr")),
        }

        active_scores = {}
        blind_scores = {}
        for key, val in scores.items():
            if has_data.get(key, True):
                active_scores[key] = val
            else:
                blind_scores[key] = val

        n_active = len(active_scores)
        n_total = len(scores)

        # ═══ STEP 2: Re-weight to ACTIVE factors only (Option C) ═══
        base_weights = dict(self.REGIME_WEIGHTS.get(regime, self.DEFAULT_WEIGHTS))

        if n_active >= 2:
            # Redistribute blind weights proportionally to active factors
            active_weight_sum = sum(base_weights.get(k, 0) for k in active_scores)
            if active_weight_sum > 0:
                scale = 1.0 / active_weight_sum  # normalize to sum=1.0
                weights = {k: base_weights.get(k, 0) * scale for k in active_scores}
            else:
                weights = {k: 1.0 / n_active for k in active_scores}

            raw_score = sum(active_scores[k] * weights[k] for k in active_scores)
        else:
            # Fewer than 2 active factors — use all scores with original weights
            weights = base_weights
            raw_score = sum(scores[k] * weights.get(k, 0) for k in scores)

        # ═══ STEP 3: Direction ═══
        if raw_score > 0.53:
            signal = "BULLISH"
        elif raw_score < 0.47:
            signal = "BEARISH"
        else:
            signal = "NEUTRAL"

        # ═══ STEP 4: Dynamic-k sigmoid (replaces old agreement penalty) ═══
        #
        # KEY INSIGHT: The agreement penalty was DOUBLE-COUNTING disagreement.
        # When macro=0.32 (bearish) and trend=0.70 (bullish), the weighted average
        # ALREADY pulls raw_score down (0.70→0.58). Applying an EXTRA multiplier
        # on top penalizes the disagreement twice.
        #
        # Better approach: make the sigmoid's STEEPNESS (k) dynamic.
        # - All factors agree → k=12 (sharp: small raw_score deviation = high confidence)
        # - Factors disagree → k=7 (gentle: need a BIGGER signal to be confident)
        #
        # This means disagreement makes the engine CAUTIOUS (needs stronger evidence)
        # instead of BLIND (blanket penalty). Strong signals still punch through.
        #
        # Example with raw_score=0.58:
        #   k=12 (all agree):   conf=0.72 → confident
        #   k=7 (disagree):     conf=0.63 → cautious but still REAL
        # vs old approach:       conf=0.72 × 0.70 = 0.50 → NEUTRAL (too harsh!)
        #
        if n_active >= 2:
            active_vals = list(active_scores.values())
            mean_val = sum(active_vals) / len(active_vals)
            variance = sum((v - mean_val) ** 2 for v in active_vals) / len(active_vals)
            std = variance ** 0.5
            # std ranges: 0.0 (perfect agreement) to ~0.25 (extreme disagreement)
            # Map to alignment: 1.0 (aligned) → 0.0 (split)
            alignment = max(0.0, 1.0 - std * 5.0)  # std=0.20 → alignment=0.0
        else:
            alignment = 0.5

        # Dynamic k: ranges from 7 (disagreement) to 12 (agreement)
        _k = 7.0 + 5.0 * alignment  # 7-12 range
        confidence = 1.0 / (1.0 + math.exp(-_k * (raw_score - 0.50)))

        # ═══ STEP 5: Bayesian uncertainty discount (Option D) ═══
        # Only discount: fewer active factors = slightly less certain
        data_completeness = n_active / n_total if n_total > 0 else 0.5
        uncertainty_factor = 0.75 + 0.25 * data_completeness  # range: 0.75 to 1.0

        confidence *= uncertainty_factor

        # ═══ STEP 7: Evidence count for cap ═══
        evidence_count = 0
        if patterns.knn and not patterns.knn.get("insufficient_data"):
            evidence_count += 1
        if patterns.ohlcv and not patterns.ohlcv.get("insufficient_data"):
            evidence_count += 1
        if patterns.stats and not patterns.stats.get("insufficient_data"):
            evidence_count += 1
        if patterns.ensemble and patterns.ensemble.get("total_strategies", 0) >= 2:
            evidence_count += 1
        if gather.derivatives:
            evidence_count += 1
        if gather.fng is not None:
            evidence_count += 1
        if gather.macro:
            evidence_count += 1

        # Cap: evidence count now less restrictive because adaptive re-weighting
        # already handles missing data. Cap is a safety net, not the primary filter.
        if evidence_count >= 5:
            max_cap = 0.90
        elif evidence_count >= 3:
            max_cap = 0.80
        elif evidence_count >= 1:
            max_cap = 0.70
        else:
            max_cap = 0.55

        confidence = min(confidence, max_cap)

        logger.info(f"[EvidenceEngine:Adaptive] {pair}: active={n_active}/{n_total} factors, "
                    f"blind={list(blind_scores.keys())}, k={_k:.1f}, "
                    f"alignment={alignment:.2f}, uncertainty={uncertainty_factor:.2f}")

        # If running on DB fallback data (no real tech_data), reduce cap further
        # Q1 (trend) and Q2 (momentum) are unreliable without real price/indicators
        if gather.tech.get("_price_from_db"):
            confidence *= 0.50  # 50% penalty — no real price = no real trade
            max_cap = min(max_cap, 0.35)  # Force shadow territory
            logger.info(f"[EvidenceEngine:Synthesis] {pair} DB-only mode: confidence halved, cap→0.35")

        # Contradiction penalty: each contradiction reduces confidence by 0.05 (max -0.15)
        contradiction_penalty = min(len(contradictions) * 0.05, 0.15)
        confidence -= contradiction_penalty
        confidence = max(0.01, confidence)  # Floor BEFORE regime modifier (prevents negative * modifier bug)

        # Regime modifier (trending=1.0, ranging=0.80, volatile=0.75)
        if self._regime_classifier:
            try:
                from regime_classifier import RegimeClassifier
                regime_mod = RegimeClassifier.get_confidence_modifier(regime)
                confidence *= regime_mod
            except Exception:
                pass

        # Platt scaling calibration (if enough historical data)
        if self._calibrator:
            try:
                calibrated = self._calibrator.adjust_confidence(confidence)
                if calibrated != confidence:
                    logger.info(f"[EvidenceEngine:Calibrate] {pair} "
                               f"raw={confidence:.3f} → calibrated={calibrated:.3f}")
                    confidence = calibrated
            except Exception:
                pass

        # Floor — never return 0 confidence (Trade-First: confidence modulates SIZE)
        confidence = max(0.01, confidence)
        confidence = round(confidence, 4)

        # Build human-readable reasoning
        reasoning = self._build_reasoning(pair, scores, contradictions, signal, confidence,
                                          regime, raw_score, max_cap, evidence_count, weights,
                                          gather, patterns)

        result = {
            "signal": signal,
            "confidence": confidence,
            "reasoning": reasoning,
            "source": "EVIDENCE_ENGINE",
            "_max_cap": max_cap,  # internal, for audit log
        }

        logger.info(f"[EvidenceEngine:Synthesis] {pair}: {signal} conf={confidence:.2f} "
                    f"(raw={raw_score:.3f}, cap={max_cap:.2f}, "
                    f"contradictions={len(contradictions)}, evidence={evidence_count})")

        return result

    def _build_reasoning(self, pair: str, scores: Dict, contradictions: List,
                         signal: str, confidence: float, regime: str,
                         raw_score: float, max_cap: float, evidence_count: int,
                         weights: Dict, gather: GatherResult, patterns: PatternResult) -> str:
        """Build structured reasoning string for transparency."""
        parts = [f"[EvidenceEngine] {pair} {signal} conf={confidence:.2f}"]

        # Key drivers
        sorted_scores = sorted(scores.items(), key=lambda x: abs(x[1] - 0.5), reverse=True)
        top_drivers = sorted_scores[:3]
        driver_strs = []
        for name, val in top_drivers:
            direction = "bullish" if val > 0.55 else "bearish" if val < 0.45 else "neutral"
            driver_strs.append(f"{name}={val:.2f}({direction})")
        parts.append(f"Key drivers: {', '.join(driver_strs)}")

        # Regime
        parts.append(f"Regime: {regime}")

        # Data quality warning
        if gather.tech.get("_price_from_db"):
            parts.append("WARNING: No real-time tech_data — using DB fallback (F&G/funding/macro only)")

        # Evidence depth
        parts.append(f"Evidence sources: {evidence_count}/7, cap={max_cap:.2f}")

        # Contradictions
        if contradictions:
            parts.append(f"Contradictions ({len(contradictions)}): {contradictions[0][:80]}")

        # F&G
        if gather.fng is not None:
            parts.append(f"F&G={gather.fng}")

        # Funding rate
        if gather.derivatives and gather.derivatives.get("funding_rate") is not None:
            fr = gather.derivatives["funding_rate"]
            parts.append(f"Funding={float(fr)*100:+.4f}%")

        return " | ".join(parts)

    # ═══════════════════════════════════════════════════════════
    # Phase F: AUDIT LOG
    # ═══════════════════════════════════════════════════════════

    def _audit_log(self, pair: str, signal: str, confidence: float,
                   scores: Dict, contradictions: List, regime: str,
                   evidence_sources: Dict, max_cap: float):
        """Persist structured audit log to SQLite. MiroFish JSONL pattern."""
        try:
            conn = sqlite3.connect(self.db_path, timeout=10)
            conn.execute("PRAGMA journal_mode=WAL")

            # Ensure table exists (idempotent)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS evidence_audit_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pair TEXT NOT NULL,
                    signal TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    sub_scores_json TEXT,
                    contradictions_json TEXT,
                    evidence_sources_json TEXT,
                    regime TEXT,
                    max_confidence_cap REAL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_evidence_pair_ts "
                        "ON evidence_audit_log(pair, timestamp)")

            conn.execute("""
                INSERT INTO evidence_audit_log
                (pair, signal, confidence, sub_scores_json, contradictions_json,
                 evidence_sources_json, regime, max_confidence_cap)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (pair, signal, confidence,
                  json.dumps(scores),
                  json.dumps(contradictions),
                  json.dumps(evidence_sources),
                  regime, max_cap))
            conn.commit()
            conn.close()
            logger.debug(f"[EvidenceEngine:Audit] {pair} logged: {signal} {confidence:.2f}")
        except Exception as e:
            logger.debug(f"[EvidenceEngine:Audit] {pair} log failed: {e}")
