import os
import logging
import json
import re
import sys

from typing import List, Dict, Any, Literal
from typing_extensions import TypedDict
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.tools.ddg_search import DuckDuckGoSearchRun
from langgraph.graph import StateGraph, START, END

# Ensure local imports work
sys.path.append(os.path.dirname(__file__))

from hybrid_retriever import HybridRetriever
from ai_decision_logger import AIDecisionLogger
from llm_router import LLMRouter
from crag_evaluator import CRAGEvaluator
from adaptive_router import AdaptiveQueryRouter
from rag_fusion import RAGFusion
from entity_extractor import KnowledgeGraphManager
from magma_memory import MAGMAMemory
from semantic_cache import SemanticCache
from self_rag import SelfRAG
from cot_rag import CoTRAG
from speculative_rag import SpeculativeRAG

# Load Env
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))
logger = logging.getLogger(__name__)

# Phase 18: Signal health tracking (in-memory counters, lightweight)
_signal_stats = {"ai": 0, "fallback": 0, "voting": 0, "timeout": 0, "total": 0}

# =============================================================================
# MODULE-LEVEL SINGLETONS — created ONCE at import/startup, reused forever.
# This is THE memory leak fix: no more per-request instantiation.
# =============================================================================

# SINGLE LLM Router — per-call temperature via llm.invoke(msgs, temperature=0.7)
# Old system: 7 NEW LLMRouter per request → unbounded memory leak (19,600 instances/hr)
# New system: 1 at startup → fixed forever. bind() creates zero-cost wrapper per call.
llm = LLMRouter(temperature=0.1)
web_search_tool = DuckDuckGoSearchRun()

# Single persistent retriever instance (holds ColBERT, BGE, FlashRank, ChromaDB)
retriever = HybridRetriever()

# Phase 3.2: Corrective RAG — evaluates + fixes bad retrieval
crag = CRAGEvaluator(router=llm)

# Phase 3.3: Adaptive RAG — routes queries to optimal pipeline
adaptive_router = AdaptiveQueryRouter(router=llm)

# Phase 3.4: RAG-Fusion for multi-perspective retrieval
rag_fusion = RAGFusion(router=llm)

# Persistent Logger for AI Decisions (Phase 3.5.1)
decision_logger = AIDecisionLogger()

# Phase 9: Semantic Cache + Self-RAG (were leaking per get_trading_signal call)
_semantic_cache = SemanticCache()
_self_rag = SelfRAG(router=llm)

# Phase 15: MAGMA + KG — optional, graceful degradation if init fails
try:
    _magma = MAGMAMemory()
except Exception as e:
    logger.error(f"[INIT] MAGMAMemory failed to initialize: {e}. MAGMA context disabled.")
    _magma = None

try:
    _kg = KnowledgeGraphManager()
except Exception as e:
    logger.error(f"[INIT] KnowledgeGraphManager failed to initialize: {e}. KG context disabled.")
    _kg = None

# Phase 16: CoT-RAG + Speculative RAG — optional, graceful degradation if init fails
try:
    _cot_rag = CoTRAG(llm_router=llm, retriever=retriever)
except Exception as e:
    logger.error(f"[INIT] CoTRAG failed to initialize: {e}. CoT-RAG disabled.")
    _cot_rag = None

try:
    _spec_rag = SpeculativeRAG(llm_router=llm, retriever=retriever)
except Exception as e:
    logger.error(f"[INIT] SpeculativeRAG failed to initialize: {e}. Speculative RAG disabled.")
    _spec_rag = None

# --- Graph State Definition ---
class GraphState(TypedDict):
    """
    State dictionary for the LangGraph Multi-Agent RAG Brain.
    Phase 5.2: Extended with bull/bear researcher outputs for MADAM debate.
    Phase 17: Extended with technical_data (real OHLCV + indicators from strategy).
    """
    pair: str
    documents: List[str]
    technical_data: Dict[str, Any]
    technical_analysis: str
    sentiment_analysis: str
    news_analysis: str
    bull_case: str
    bear_case: str
    signal: str
    confidence: float
    reasoning: str

# --- Phase 17: Technical Data Formatters ---

def _format_technical_data(pair: str, td: dict) -> str:
    """Format comprehensive technical data into a structured text block for LLM."""
    lines = [f"=== LIVE TECHNICAL DATA for {pair} ==="]
    price = td.get("current_price", 0)

    # Price overview
    lines.append(f"Current Price: ${price:,.2f}")
    for key, label in [("price_change_1h_pct", "1H"), ("price_change_4h_pct", "4H"),
                        ("price_change_24h_pct", "24H"), ("price_change_7d_pct", "7D")]:
        if td.get(key) is not None:
            lines.append(f"  {label} Change: {td[key]:+.2f}%")

    # Trend (Moving Averages)
    lines.append("\n--- TREND (Moving Averages) ---")
    for key in ["ema_9", "ema_20", "ema_50", "ema_200", "sma_50", "sma_200"]:
        if td.get(key) is not None:
            label = key.upper().replace("_", " ")
            vs_price = ((price - td[key]) / td[key] * 100) if td[key] > 0 else 0
            lines.append(f"{label}: ${td[key]:,.2f} (price {'above' if vs_price > 0 else 'below'} by {abs(vs_price):.1f}%)")

    # Multi-Timeframe
    htf = td.get("htf", {})
    if htf:
        lines.append("\n--- MULTI-TIMEFRAME ---")
        if htf.get("rsi_4h") is not None:
            lines.append(f"4H RSI: {htf['rsi_4h']:.1f} | 4H Trend: {htf.get('trend_4h', '?')}")
        if htf.get("ema_20_4h") is not None:
            lines.append(f"4H EMA20: ${htf['ema_20_4h']:,.2f}")
        if htf.get("rsi_daily") is not None:
            lines.append(f"Daily RSI: {htf['rsi_daily']:.1f} | Daily Trend: {htf.get('trend_daily', '?')}")
        if htf.get("ema_50_daily") is not None:
            lines.append(f"Daily EMA50: ${htf['ema_50_daily']:,.2f}")

    # Momentum
    lines.append("\n--- MOMENTUM ---")
    if td.get("rsi_14") is not None:
        rsi = td["rsi_14"]
        rsi_label = "OVERSOLD" if rsi < 30 else "OVERBOUGHT" if rsi > 70 else "neutral"
        lines.append(f"RSI(14): {rsi:.1f} ({rsi_label})")
    if td.get("macd") is not None:
        hist = td.get('macd_histogram', 0)
        lines.append(f"MACD: Line={td['macd']:.4f} Signal={td.get('macd_signal', 0):.4f} Hist={hist:+.4f} ({'bullish' if hist > 0 else 'bearish'} momentum)")
    if td.get("adx_14") is not None:
        adx = td["adx_14"]
        lines.append(f"ADX(14): {adx:.1f} ({'STRONG trend' if adx > 25 else 'WEAK/ranging'})")

    # Volatility
    lines.append("\n--- VOLATILITY ---")
    if td.get("atr_14") is not None:
        atr_pct = (td["atr_14"] / price * 100) if price > 0 else 0
        lines.append(f"ATR(14): ${td['atr_14']:,.2f} ({atr_pct:.2f}% of price)")
    if td.get("bb_upper") is not None and td.get("bb_lower") is not None:
        bb_mid = td.get("bb_mid", (td["bb_upper"] + td["bb_lower"]) / 2)
        bb_width = ((td["bb_upper"] - td["bb_lower"]) / bb_mid * 100) if bb_mid > 0 else 0
        lines.append(f"Bollinger: [${td['bb_lower']:,.2f} — ${bb_mid:,.2f} — ${td['bb_upper']:,.2f}] Width={bb_width:.1f}%")

    # Key Levels
    levels = td.get("levels", {})
    if levels:
        lines.append("\n--- KEY LEVELS ---")
        for label in ["24h", "7d", "30d"]:
            h, l = levels.get(f"high_{label}"), levels.get(f"low_{label}")
            if h and l:
                rng = ((h - l) / l * 100) if l > 0 else 0
                lines.append(f"{label} Range: ${l:,.2f} — ${h:,.2f} ({rng:.1f}%)")
        if levels.get("support"):
            lines.append(f"Support Levels: {', '.join(f'${s:,.2f}' for s in levels['support'])}")
        if levels.get("resistance"):
            lines.append(f"Resistance Levels: {', '.join(f'${r:,.2f}' for r in levels['resistance'])}")
        fib = levels.get("fibonacci", {})
        if fib:
            lines.append(f"Fibonacci (from ${fib.get('swing_low', 0):,.2f} to ${fib.get('swing_high', 0):,.2f}):")
            for lvl in ["fib_236", "fib_382", "fib_500", "fib_618", "fib_786"]:
                if fib.get(lvl):
                    pct = lvl.replace("fib_", "").replace("0", "").strip()
                    lines.append(f"  {pct[:-1]}.{pct[-1]}%: ${fib[lvl]:,.2f}")
        pivot = levels.get("pivot", {})
        if pivot:
            lines.append(f"Pivot: S2=${pivot.get('s2', 0):,.2f} S1=${pivot.get('s1', 0):,.2f} PP=${pivot.get('pp', 0):,.2f} R1=${pivot.get('r1', 0):,.2f} R2=${pivot.get('r2', 0):,.2f}")

    # Volume Analysis
    vol = td.get("volume", {})
    if vol and vol.get("avg_20"):
        lines.append(f"\n--- VOLUME ---")
        lines.append(f"Current: {vol['current']:,.0f} | 20-Avg: {vol['avg_20']:,.0f} | Ratio: {vol.get('ratio', 0):.2f}x ({'above' if vol.get('ratio', 0) > 1 else 'below'} average)")
        if vol.get("trend"):
            lines.append(f"Volume Trend: {vol['trend']} ({vol.get('trend_pct', 0):+.1f}%)")

    # Candlestick Patterns
    patterns = td.get("patterns", [])
    if patterns:
        lines.append(f"\n--- CANDLESTICK PATTERNS ---")
        lines.append(f"Detected: {', '.join(patterns)}")

    # Daily Summaries (7 days)
    daily = td.get("daily_summaries", [])
    if daily:
        lines.append(f"\n--- DAILY SUMMARIES (last {len(daily)} days) ---")
        for d in daily:
            o, c = d.get("open", 0), d.get("close", 0)
            chg = ((c - o) / o * 100) if o > 0 else 0
            lines.append(f"  {d.get('date', '?')}: O=${o:,.2f} C=${c:,.2f} ({chg:+.1f}%) H=${d.get('high', 0):,.2f} L=${d.get('low', 0):,.2f}")

    # Last 24 candles (compact: just last 12 to save tokens, full 24 available)
    candles = td.get("last_candles", [])
    if candles:
        show = candles[-12:]  # Last 12h for detail, AI has daily summaries for context
        lines.append(f"\n--- LAST {len(show)} HOURLY CANDLES ---")
        for c in show:
            t = c.get("time", "?")
            o, cl = c.get("open", 0), c.get("close", 0)
            direction = "▲" if cl >= o else "▼"
            body_pct = abs(cl - o) / o * 100 if o > 0 else 0
            lines.append(f"  {t}: ${cl:,.2f} {direction}{body_pct:.1f}% V={c.get('volume', 0):,.0f}")

    return "\n".join(lines)


def _format_tech_summary_compact(td: dict) -> str:
    """Compact tech summary for Bull/Bear researcher context — includes multi-timeframe."""
    if not td or not td.get("current_price"):
        return ""
    parts = []
    parts.append(f"Price: ${td['current_price']:,.2f} (1H: {td.get('price_change_1h_pct', 0):+.1f}%, 24H: {td.get('price_change_24h_pct', 0):+.1f}%, 7D: {td.get('price_change_7d_pct', 0):+.1f}%)")

    indicators = []
    if td.get("rsi_14") is not None:
        indicators.append(f"RSI={td['rsi_14']:.0f}")
    if td.get("macd_histogram") is not None:
        indicators.append(f"MACD_hist={td['macd_histogram']:+.4f}")
    if td.get("adx_14") is not None:
        indicators.append(f"ADX={td['adx_14']:.0f}")
    if indicators:
        parts.append("1H Indicators: " + ", ".join(indicators))

    # Multi-timeframe
    htf = td.get("htf", {})
    htf_parts = []
    if htf.get("rsi_4h") is not None:
        htf_parts.append(f"4H RSI={htf['rsi_4h']:.0f}")
    if htf.get("trend_4h"):
        htf_parts.append(f"4H trend={htf['trend_4h']}")
    if htf.get("rsi_daily") is not None:
        htf_parts.append(f"Daily RSI={htf['rsi_daily']:.0f}")
    if htf.get("trend_daily"):
        htf_parts.append(f"Daily trend={htf['trend_daily']}")
    if htf_parts:
        parts.append("Multi-TF: " + ", ".join(htf_parts))

    # Key levels
    levels = td.get("levels", {})
    if levels.get("support"):
        parts.append(f"Support: {', '.join(f'${s:,.0f}' for s in levels['support'][:2])}")
    if levels.get("resistance"):
        parts.append(f"Resistance: {', '.join(f'${r:,.0f}' for r in levels['resistance'][:2])}")

    # Patterns
    patterns = td.get("patterns", [])
    if patterns:
        parts.append(f"Candle Pattern: {', '.join(patterns)}")

    # Volume
    vol = td.get("volume", {})
    if vol.get("ratio"):
        parts.append(f"Volume: {vol['ratio']:.1f}x avg ({vol.get('trend', '?')})")

    return "\n".join(parts)


def _format_tech_for_coordinator(td: dict) -> str:
    """Fact-check block for coordinator — raw numbers to verify agent claims."""
    if not td or not td.get("current_price"):
        return ""
    p = td["current_price"]
    lines = ["[RAW TECHNICAL INDICATORS — Use these to FACT-CHECK agent claims]"]

    # Price
    parts = [f"Price: ${p:,.2f}"]
    for key, label in [("price_change_1h_pct", "1H"), ("price_change_24h_pct", "24H"), ("price_change_7d_pct", "7D")]:
        if td.get(key) is not None:
            parts.append(f"{label}: {td[key]:+.1f}%")
    lines.append(" | ".join(parts))

    # Momentum
    mom = []
    if td.get("rsi_14") is not None:
        mom.append(f"RSI={td['rsi_14']:.0f}")
    if td.get("macd_histogram") is not None:
        mom.append(f"MACD_hist={td['macd_histogram']:+.4f}")
    if td.get("adx_14") is not None:
        mom.append(f"ADX={td['adx_14']:.0f}")
    if mom:
        lines.append("Momentum: " + " | ".join(mom))

    # Multi-timeframe
    htf = td.get("htf", {})
    if htf:
        htf_parts = []
        if htf.get("trend_4h"):
            htf_parts.append(f"4H={htf['trend_4h']}(RSI={htf.get('rsi_4h', '?')})")
        if htf.get("trend_daily"):
            htf_parts.append(f"Daily={htf['trend_daily']}(RSI={htf.get('rsi_daily', '?')})")
        if htf_parts:
            lines.append("Multi-TF: " + " | ".join(htf_parts))

    # Moving averages
    ma_parts = []
    for key in ["ema_9", "ema_20", "ema_50", "ema_200", "sma_200"]:
        if td.get(key) is not None:
            ma_parts.append(f"{key.upper()}=${td[key]:,.0f}")
    if ma_parts:
        lines.append("MAs: " + " > ".join(ma_parts))

    # Key levels
    levels = td.get("levels", {})
    if levels.get("support"):
        lines.append(f"S/R: Support={[f'${s:,.0f}' for s in levels['support']]} Resistance={[f'${r:,.0f}' for r in levels.get('resistance', [])]}")

    # Volume + Patterns
    vol = td.get("volume", {})
    patterns = td.get("patterns", [])
    extras = []
    if vol.get("ratio"):
        extras.append(f"Vol={vol['ratio']:.1f}x avg")
    if patterns:
        extras.append(f"Patterns: {', '.join(patterns)}")
    if extras:
        lines.append(" | ".join(extras))

    return "\n".join(lines)


# --- Phase 18: Fallback Functions (when LLM is exhausted) ---

def _technical_fallback(tech_data: dict) -> dict:
    """
    Rule-based technical scoring using Phase 17 indicators.
    Used when ALL LLMs are exhausted. Max confidence cap: 0.35.
    """
    if not tech_data or not tech_data.get("current_price"):
        return {"signal": "NEUTRAL", "confidence": 0.01,
                "reasoning": "[Technical Fallback] No indicator data available", "source": "FALLBACK"}

    score = 0.0
    reasons = []
    price = tech_data["current_price"]

    # RSI (weight 0.15)
    rsi = tech_data.get("rsi_14")
    if rsi is not None:
        if rsi < 30:
            score += 0.15
            reasons.append(f"RSI={rsi:.0f}(oversold)")
        elif rsi > 70:
            score -= 0.15
            reasons.append(f"RSI={rsi:.0f}(overbought)")
        elif rsi < 45:
            score += 0.05
        elif rsi > 55:
            score -= 0.05

    # MACD histogram (weight 0.15)
    macd_hist = tech_data.get("macd_histogram")
    if macd_hist is not None:
        if macd_hist > 0:
            score += 0.15
            reasons.append(f"MACD_hist={macd_hist:+.4f}(bull)")
        else:
            score -= 0.15
            reasons.append(f"MACD_hist={macd_hist:+.4f}(bear)")

    # EMA cross 9/20 (weight 0.12)
    ema_9 = tech_data.get("ema_9")
    ema_20 = tech_data.get("ema_20")
    if ema_9 is not None and ema_20 is not None:
        if ema_9 > ema_20:
            score += 0.12
            reasons.append("EMA9>20(bull)")
        else:
            score -= 0.12
            reasons.append("EMA9<20(bear)")

    # EMA alignment 20/50/200 (weight 0.10)
    ema_50 = tech_data.get("ema_50")
    ema_200 = tech_data.get("ema_200")
    if ema_20 is not None and ema_50 is not None and ema_200 is not None:
        if ema_20 > ema_50 > ema_200:
            score += 0.10
            reasons.append("EMA_golden")
        elif ema_200 > ema_50 > ema_20:
            score -= 0.10
            reasons.append("EMA_death")

    # Bollinger Bands (weight 0.08)
    bb_lower = tech_data.get("bb_lower")
    bb_upper = tech_data.get("bb_upper")
    if bb_lower is not None and price <= bb_lower:
        score += 0.08
        reasons.append("below_BB")
    elif bb_upper is not None and price >= bb_upper:
        score -= 0.08
        reasons.append("above_BB")

    # ADX trend strength (weight 0.07)
    adx = tech_data.get("adx_14")
    if adx is not None:
        if adx < 20:
            score *= 0.7  # ranging market: reduce conviction
            reasons.append(f"ADX={adx:.0f}(ranging)")
        elif adx > 25:
            reasons.append(f"ADX={adx:.0f}(trend)")

    # Volume ratio (weight 0.07)
    vol = tech_data.get("volume", {})
    vol_ratio = vol.get("ratio", 1.0) if isinstance(vol, dict) else 1.0
    if vol_ratio > 1.2:
        # Volume confirms the current direction
        score *= 1.1
        reasons.append(f"Vol={vol_ratio:.1f}x(confirms)")
    elif vol_ratio < 0.8:
        score *= 0.85
        reasons.append(f"Vol={vol_ratio:.1f}x(weak)")

    # Multi-timeframe: 4H RSI (weight 0.08)
    htf = tech_data.get("htf", {})
    rsi_4h = htf.get("rsi_4h") if isinstance(htf, dict) else None
    if rsi_4h is not None:
        if rsi_4h < 40:
            score += 0.08
            reasons.append(f"4H_RSI={rsi_4h:.0f}(bull)")
        elif rsi_4h > 60:
            score -= 0.08
            reasons.append(f"4H_RSI={rsi_4h:.0f}(bear)")

    # Multi-timeframe: Daily trend (weight 0.08)
    trend_daily = htf.get("trend_daily") if isinstance(htf, dict) else None
    if trend_daily == "bullish":
        score += 0.08
        reasons.append("daily_trend=bull")
    elif trend_daily == "bearish":
        score -= 0.08
        reasons.append("daily_trend=bear")

    # S/R proximity (weight 0.05)
    levels = tech_data.get("levels", {})
    if isinstance(levels, dict):
        supports = levels.get("support", [])
        resistances = levels.get("resistance", [])
        if supports and isinstance(supports, list) and len(supports) > 0:
            nearest_sup = min(supports, key=lambda s: abs(price - s) if isinstance(s, (int, float)) else 999999)
            if isinstance(nearest_sup, (int, float)) and price > 0:
                dist_pct = (price - nearest_sup) / price * 100
                if 0 < dist_pct < 2:
                    score += 0.05
                    reasons.append("near_support")
        if resistances and isinstance(resistances, list) and len(resistances) > 0:
            nearest_res = min(resistances, key=lambda r: abs(price - r) if isinstance(r, (int, float)) else 999999)
            if isinstance(nearest_res, (int, float)) and price > 0:
                dist_pct = (nearest_res - price) / price * 100
                if 0 < dist_pct < 2:
                    score -= 0.05
                    reasons.append("near_resistance")

    # Candlestick patterns (weight 0.05)
    patterns = tech_data.get("patterns", [])
    if isinstance(patterns, list):
        for p in patterns:
            if "bullish" in str(p).lower():
                score += 0.025
                reasons.append(str(p))
            elif "bearish" in str(p).lower():
                score -= 0.025
                reasons.append(str(p))

    # Determine signal
    if score > 0.10:
        signal = "BULLISH"
    elif score < -0.10:
        signal = "BEARISH"
    else:
        signal = "NEUTRAL"

    # Confidence: scale and cap at 0.35
    confidence = min(abs(score) * 0.50, 0.35)
    confidence = round(confidence, 2)

    reason_str = f"[Technical Fallback] {signal}: {'; '.join(reasons[:6])}. Score: {score:+.2f}"

    return {"signal": signal, "confidence": confidence, "reasoning": reason_str, "source": "FALLBACK"}


def _voting_fallback(tech_text: str, sent_text: str, news_text: str) -> dict:
    """
    Parse 3 unbiased agent outputs (Technical, Sentiment, News) for direction keywords and vote.
    Bull/Bear researchers are excluded — they're biased by design (one always argues bullish, other bearish).
    Used when coordinator LLM is completely exhausted.
    """
    bullish_kw = ["bullish", "upward", "support", "accumulation", "oversold", "bounce", "recovery", "uptrend", "buy"]
    bearish_kw = ["bearish", "downward", "resistance", "distribution", "overbought", "rejection", "decline", "downtrend", "sell"]

    bull_votes = 0
    bear_votes = 0

    for agent_text in [tech_text, sent_text, news_text]:
        text_lower = str(agent_text).lower()
        b_count = sum(1 for kw in bullish_kw if kw in text_lower)
        s_count = sum(1 for kw in bearish_kw if kw in text_lower)
        if b_count > s_count:
            bull_votes += 1
        elif s_count > b_count:
            bear_votes += 1
        # tie or both zero = no vote (neutral)

    majority = max(bull_votes, bear_votes)
    if bull_votes >= 2 and bull_votes > bear_votes:
        signal = "BULLISH"
    elif bear_votes >= 2 and bear_votes > bull_votes:
        signal = "BEARISH"
    else:
        signal = "NEUTRAL"

    # Confidence: (majority/3) * 0.40, capped at 0.30
    if signal != "NEUTRAL" and majority >= 2:
        confidence = min(round((majority / 3) * 0.40, 2), 0.30)
    else:
        confidence = 0.0

    return {
        "signal": signal,
        "confidence": confidence,
        "reasoning": f"[Voting Fallback] {signal} by {majority}/3 agent vote (bull={bull_votes}, bear={bear_votes}). Coordinator LLM unavailable.",
        "source": "VOTING"
    }


def _record_signal_health(pair: str, source: str, signal_type: str, confidence: float):
    """Helper: record a signal to the signal_health SQLite table."""
    conn = None
    try:
        from db import get_db_connection
        conn = get_db_connection()
        conn.execute(
            "INSERT INTO signal_health (pair, signal_source, signal_type, confidence) VALUES (?, ?, ?, ?)",
            (pair, source, signal_type, confidence)
        )
        conn.commit()
    except Exception as e:
        logger.debug(f"[SignalHealth] Failed to record: {e}")
    finally:
        if conn:
            conn.close()


# --- Parallel Analyst Nodes ---

def analyze_technical(state: GraphState):
    """Analyzes technical indicators — uses REAL data from strategy when available, DuckDuckGo fallback."""
    logger.info("---[NODE] TECHNICAL ANALYST---")
    pair = state.get("pair", "BTC/USDT")
    tech_data = state.get("technical_data") or {}

    # Phase 17: Use REAL indicator data from strategy if available
    if tech_data and tech_data.get("current_price"):
        tech_context = _format_technical_data(pair, tech_data)
        search_res = ""
        logger.info(f"[Technical] Using REAL indicator data for {pair} (price=${tech_data['current_price']:,.2f})")
    else:
        # Fallback: DuckDuckGo (legacy, when no data from strategy)
        logger.warning(f"[Technical] No real indicator data for {pair}. Falling back to web search.")
        try:
            search_res = web_search_tool.invoke(f"{pair} current technical analysis RSI MACD support resistance price prediction")
        except Exception as e:
            logger.warning(f"Technical Search Failed: {e}")
            search_res = "Unable to fetch live technical data."
        tech_context = ""

    # Phase 15: MAGMA Graph Context (Semantic + Causal)
    magma_context = ""
    if _magma is not None:
        try:
            semantic_edges = _magma.query(f"{pair} tech analysis", graph_types=["semantic", "causal"], max_hops=1)
            if semantic_edges:
                ext_nodes = [f"{e['source']} -> {e['relation']} -> {e['target']}" for e in semantic_edges[:10]]
                magma_context = "\nMAGMA Historic Technical Correlations:\n" + "\n".join(ext_nodes)
        except Exception as e:
            logger.warning(f"MAGMA Technical lookup failed: {e}")

    # Agent DNA: Technical Analyst (FinCoT + Show Your Work + Numeric Grounding + Multi-Timeframe)
    TECH_SYSTEM = """IDENTITY: You are a senior quantitative Technical Analyst at a systematic crypto trading desk.
Your methodology combines momentum, trend, volatility, and volume analysis using EXACT indicator values.

CONSTITUTIONAL RULES:
1. ONLY cite indicator values that appear in the provided data. NEVER fabricate support/resistance levels, RSI values, or any numbers.
2. Every claim MUST include [SOURCE: INDICATOR_NAME=VALUE] tag — e.g., [SOURCE: RSI_14=45.2].
3. If an indicator is missing from the data, say "DATA UNAVAILABLE" — do NOT estimate or guess.
4. NEVER provide a final trading signal or recommend a trade. You provide TECHNICAL PERSPECTIVE only.
5. Analyze what the indicators ARE showing, not what you WISH they showed. No confirmation bias.

YOU DO NOT: predict exact prices, guarantee outcomes, provide position sizing, or make final trade decisions.
YOU DO: read indicators precisely, identify trend/momentum/volatility regimes, detect divergences, and flag contradictions between timeframes.

BAD OUTPUT EXAMPLE: "RSI looks oversold and MACD is turning bullish, suggesting a bounce is likely." (No exact values, no source tags, speculative)
GOOD OUTPUT EXAMPLE: "RSI_14 at 32.4 [SOURCE: RSI_14=32.4] is in oversold territory (<35). However, MACD histogram at -0.0023 [SOURCE: MACD_HIST=-0.0023] shows no bullish crossover yet — momentum remains bearish. Contradiction: price holding above EMA_200 [SOURCE: EMA_200=$62,450] despite bearish momentum. TECHNICAL LEAN: NEUTRAL (oversold but no reversal confirmation).\""""

    if tech_context:
        prompt = f"""Analyze these LIVE technical indicators for {pair} using the FinCoT (Financial Chain-of-Thought) method:

=== RAW INDICATOR DATA ===
{tech_context}
{magma_context}

=== FinCoT 5-STEP ANALYSIS (complete ALL steps) ===

STEP 1 — TREND IDENTIFICATION:
Examine EMA/SMA alignment and price position relative to key MAs.
- Price vs EMA9/EMA20/EMA50/EMA200: above or below? Aligned or tangled?
- What is the ADX value? (>25 = trending, <20 = ranging)
- Verdict: STRONG_UPTREND / WEAK_UPTREND / RANGING / WEAK_DOWNTREND / STRONG_DOWNTREND

STEP 2 — MOMENTUM ASSESSMENT:
- RSI_14 value: oversold (<30), neutral (30-70), overbought (>70)?
- MACD line vs signal line: bullish or bearish crossover? Histogram direction?
- Any RSI or MACD DIVERGENCE vs price? (bullish div = price lower low, RSI higher low)

STEP 3 — VOLATILITY REGIME:
- ATR as % of price: low (<1.5%), normal (1.5-3%), high (>3%)?
- Bollinger Band width: narrowing (squeeze) or expanding (breakout)?
- Current price position within BBands: near upper/middle/lower?

STEP 4 — KEY LEVELS:
- Nearest support: from EMA_200, BB_lower, or recent swing low
- Nearest resistance: from EMA_50, BB_upper, or recent swing high
- ONLY cite levels that exist in the data. Do NOT fabricate levels.

STEP 5 — MULTI-TIMEFRAME CHECK (if HTF data available):
- Does the higher timeframe CONFIRM or CONTRADICT the current timeframe?
- If HTF is bearish but current TF is bullish = likely bear market rally (lower confidence)
- TIMEFRAME_ALIGNMENT: ALIGNED / PARTIALLY_ALIGNED / CONTRADICTING

=== OUTPUT FORMAT ===
Synthesize all 5 steps into a dense 3-4 sentence analysis. Cite [SOURCE: INDICATOR=VALUE] for every claim.
End with: TECHNICAL LEAN: BULLISH / BEARISH / NEUTRAL
NEVER provide a final trading signal. ONLY your technical perspective."""
    else:
        prompt = f"""Analyze available market data for {pair}:
{search_res}
{magma_context}

Provide a dense 3-4 sentence technical analysis citing specific data points where available.
End with: TECHNICAL LEAN: BULLISH / BEARISH / NEUTRAL.
NEVER provide a final trading signal. ONLY your technical perspective."""

    try:
        response = llm.invoke([SystemMessage(content=TECH_SYSTEM), HumanMessage(content=prompt)], priority="high")
        content_raw = response.content
        if isinstance(content_raw, list):
            content_raw = " ".join([b.get("text", "") for b in content_raw if "text" in b])
        return {"technical_analysis": content_raw.strip()}
    except Exception as e:
        logger.error(f"[NODE] Technical Analyst LLM invoke failed: {e}")
        return {"technical_analysis": f"Technical analysis unavailable (LLM error: {type(e).__name__})"}


def analyze_sentiment(state: GraphState):
    """Retrieves and analyzes the latest DB fear/greed and CryptoBERT sentiment."""
    logger.info("---[NODE] SENTIMENT ANALYST---")
    pair = state.get("pair", "BTC/USDT")
    
    # ===== LIVE DATA: Query real sentiment from ai_data.sqlite =====
    import sqlite3
    from ai_config import AI_DB_PATH as db_path
    db_context_parts = []
    
    conn = None
    try:
        conn = sqlite3.connect(db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()

        # Fear & Greed Index (live from fng_fetcher.py)
        c.execute("SELECT value, classification FROM fear_and_greed ORDER BY timestamp DESC LIMIT 1")
        fng_row = c.fetchone()
        if fng_row:
            db_context_parts.append(f"Fear & Greed Index: {fng_row['value']} ({fng_row['classification']}).")
        else:
            db_context_parts.append("Fear & Greed Index: Data unavailable.")

        # CryptoBERT rolling sentiment (live from coin_sentiment_aggregator.py)
        base_coin = pair.split("/")[0]
        c.execute("SELECT sentiment_1h, sentiment_4h, sentiment_24h FROM coin_sentiment_rolling WHERE coin = ? ORDER BY timestamp DESC LIMIT 1", (base_coin,))
        sent_row = c.fetchone()
        if sent_row:
            s1h = sent_row['sentiment_1h']
            s4h = sent_row['sentiment_4h']
            s24h = sent_row['sentiment_24h']
            db_context_parts.append(f"CryptoBERT rolling sentiment for {base_coin}: 1H={s1h:.2f}, 4H={s4h:.2f}, 24H={s24h:.2f}.")
        else:
            db_context_parts.append(f"CryptoBERT rolling sentiment for {base_coin}: No data available yet.")
    except Exception as e:
        logger.warning(f"Sentiment DB query failed: {e}. Falling back to neutral context.")
        db_context_parts.append("Sentiment data temporarily unavailable. Assume NEUTRAL baseline.")
    finally:
        if conn:
            conn.close()
    
    db_context = " ".join(db_context_parts)
    # ===== END LIVE DATA =====
    
    # Agent DNA: Sentiment Analyst (Temporal Grounding + Contrarian + Numeric Grounding)
    SENT_SYSTEM = """IDENTITY: You are a Behavioral Economics & Crypto Sentiment Analyst specializing in crowd psychology, fear/greed cycles, and contrarian signals.
Your edge: understanding when the crowd is RIGHT (trend continuation) vs WRONG (reversal imminent).

CONSTITUTIONAL RULES:
1. ALWAYS cite exact sentiment values with [SOURCE: METRIC=VALUE] — e.g., [SOURCE: F&G=25], [SOURCE: SENT_1H=-0.35].
2. Compare MULTIPLE timeframes of sentiment (1H vs 4H vs 24H) to detect SHIFTS, not just snapshots.
3. ALWAYS include contrarian analysis: "When Fear & Greed is at X, historically the market tends to Y."
4. Distinguish between LEADING sentiment (fear/greed shifts BEFORE price moves) and LAGGING sentiment (reacting to price).
5. If sentiment data is unavailable or stale (>6h old), explicitly state: "WARNING: Stale/missing sentiment data. Reducing weight."

YOU DO NOT: predict prices, make final trade decisions, or assume sentiment alone drives markets.
YOU DO: assess crowd positioning, identify extreme readings, detect sentiment divergence from price, and flag contrarian setups.

BAD OUTPUT EXAMPLE: "Sentiment is neutral." (No data, no values, no contrarian analysis)
GOOD OUTPUT EXAMPLE: "Fear & Greed at 25 (Extreme Fear) [SOURCE: F&G=25] historically signals oversold conditions — contrarian bullish. However, CryptoBERT 1H sentiment at -0.42 [SOURCE: SENT_1H=-0.42] is DETERIORATING vs 4H at -0.18 [SOURCE: SENT_4H=-0.18], suggesting fear is ACCELERATING, not stabilizing. Contrarian buy signal is PREMATURE until 1H sentiment stops declining. SENTIMENT LEAN: NEUTRAL (extreme fear but still deteriorating).\""""

    prompt = f"""Analyze the current sentiment metrics for {pair}:

=== LIVE SENTIMENT DATA ===
{db_context}

=== STRUCTURED ANALYSIS (complete ALL sections) ===

SECTION A — FEAR & GREED ASSESSMENT:
- Current F&G value and classification
- Historical context: what typically happens at this level?
- Is this a CONTRARIAN signal? (F&G < 20 = contrarian bullish, F&G > 80 = contrarian bearish)

SECTION B — SENTIMENT MOMENTUM (if CryptoBERT data available):
- Compare 1H vs 4H vs 24H sentiment scores
- Is sentiment IMPROVING (1H > 4H > 24H), DETERIORATING (1H < 4H < 24H), or MIXED?
- Sentiment direction matters MORE than absolute level

SECTION C — CONTRARIAN VERDICT:
- Is the crowd positioned too far in one direction?
- Smart money typically fades extreme crowd positioning
- But contrarian signals need PRICE CONFIRMATION to be actionable

Synthesize into 3-4 sentences with [SOURCE: METRIC=VALUE] citations.
End with: SENTIMENT LEAN: BULLISH / BEARISH / NEUTRAL
NEVER provide a final trading signal. ONLY your sentiment perspective."""

    try:
        response = llm.invoke([SystemMessage(content=SENT_SYSTEM), HumanMessage(content=prompt)], temperature=0.4, priority="medium")
        content_raw = response.content
        if isinstance(content_raw, list):
            content_raw = " ".join([b.get("text", "") for b in content_raw if "text" in b])
        return {"sentiment_analysis": content_raw.strip()}
    except Exception as e:
        logger.error(f"[NODE] Sentiment Analyst LLM invoke failed: {e}")
        return {"sentiment_analysis": f"Sentiment analysis unavailable (LLM error: {type(e).__name__})"}


def analyze_news(state: GraphState):
    """Retrieves and reads the latest semantic/BM25 news chunks from ChromaDB."""
    logger.info("---[NODE] NEWS & MACRO ANALYST---")
    pair = state.get("pair", "BTC/USDT")
    
    # Phase 3.3: Adaptive RAG — route query to optimal pipeline
    query = f"{pair} macro fundamentals exact news ETF"
    corrected_results = adaptive_router.route(
        query=query,
        retriever=retriever,
        crag_evaluator=crag
    )
    
    documents = [res.get("text", str(res)) for res in corrected_results[:5]]
    
    # Phase 5.1: Knowledge Graph Traversal
    base_coin = pair.split("/")[0]
    network_links = []

    if _kg is not None:
        try:
            # Query for ticker and full name (e.g., BTC and Bitcoin)
            network_links = _kg.query_entity_network(base_coin)

            # Add common full names for major coins to enrich graph hits
            coin_map = {"BTC": "Bitcoin", "ETH": "Ethereum", "SOL": "Solana", "XRP": "Ripple"}
            if base_coin in coin_map:
                network_links.extend(_kg.query_entity_network(coin_map[base_coin]))
        except Exception as e:
            logger.warning(f"KG entity network lookup failed: {e}")
        
    kg_context = "\n".join(list(set(network_links))) # Deduplicate
    
    if kg_context:
        documents.append(f"--- KNOWLEDGE GRAPH RELATIONS ---\n{kg_context}")
    
    context = "\n\n".join(documents)
    
    # Agent DNA: News & Macro Analyst (Signal Decay + Source Attribution + Impact Assessment)
    NEWS_SYSTEM = """IDENTITY: You are a Crypto Fundamental & Macroeconomic Analyst specializing in news impact assessment and information decay.
Your edge: distinguishing between CATALYST news (creates new price movement) vs NARRATIVE news (already priced in) vs NOISE (irrelevant).

CONSTITUTIONAL RULES:
1. Every news claim MUST include [SOURCE: NEWS_ITEM] tag referencing which retrieved document supports it.
2. Assess FRESHNESS of each news item: FRESH (<2h), STALE (2-6h), COLD (6-24h), DEAD (>24h). Weight accordingly.
3. Negative news decays FASTER than positive (market overreacts to fear, then reverts). Apply 1.5x decay to negative items.
4. If NO news is retrieved or ALL news is STALE, explicitly state: "No fresh catalysts. Signal based on stale information."
5. NEVER fabricate news events. If the documents don't mention something, it didn't happen.
6. Distinguish FACT (reported event) from INTERPRETATION (market implication). Label each clearly.

YOU DO NOT: predict prices, make trade decisions, or assume all news is equally impactful.
YOU DO: triage news by freshness and impact, identify already-priced-in information, detect second-order effects, and assess remaining tradeable alpha.

BAD OUTPUT EXAMPLE: "Recent positive news suggests bullish momentum." (No source, no freshness, no specifics)
GOOD OUTPUT EXAMPLE: "SEC ETF approval delay [SOURCE: Doc 2] is 18h old (COLD) — likely 80% priced in given yesterday's -3% reaction. However, whale accumulation report [SOURCE: Doc 4] is 2h old (FRESH) and shows 12,000 BTC moved to cold storage — this is a CATALYST with potential upside not yet fully reflected. Net news impact: mildly BULLISH but dominated by one fresh data point. FUNDAMENTAL LEAN: BULLISH (weak, single catalyst).\""""

    prompt = f"""Analyze these retrieved news documents and Knowledge Graph relations for {pair}:

=== RETRIEVED DOCUMENTS ===
{context}

=== NEWS IMPACT ANALYSIS (complete ALL sections) ===

SECTION A — NEWS TRIAGE:
For each document, classify:
- CATALYST (creates new price movement, not yet priced in)
- NARRATIVE (supports existing trend, mostly priced in)
- NOISE (irrelevant to {pair} price action)
- Freshness: FRESH / STALE / COLD / DEAD

SECTION B — IMPACT ASSESSMENT:
- What is the NET direction of fresh catalysts? (Bullish / Bearish / Mixed)
- Is there any single HIGH-IMPACT event that dominates?
- Second-order effects: does this news affect correlated assets?

SECTION C — PRICED-IN ANALYSIS:
- How much of this news is already reflected in current price?
- Tradeable alpha remaining: HIGH / MEDIUM / LOW / NONE

Synthesize into 3-4 sentences with [SOURCE: Doc N] citations.
End with: FUNDAMENTAL LEAN: BULLISH / BEARISH / NEUTRAL
Focus only on news impact. NEVER provide a final trading signal."""

    try:
        response = llm.invoke([SystemMessage(content=NEWS_SYSTEM), HumanMessage(content=prompt)], temperature=0.3, priority="medium")
        content_raw = response.content
        if isinstance(content_raw, list):
            content_raw = " ".join([b.get("text", "") for b in content_raw if "text" in b])
        return {"news_analysis": content_raw.strip(), "documents": documents}
    except Exception as e:
        logger.error(f"[NODE] News Analyst LLM invoke failed: {e}")
        return {"news_analysis": f"News analysis unavailable (LLM error: {type(e).__name__})", "documents": documents}


# --- Phase 5.2: Bull/Bear Researcher Nodes ---

def research_bullish(state: GraphState):
    """Bull Researcher: Collects and advocates for bullish evidence."""
    logger.info("---[NODE] BULL RESEARCHER---")
    pair = state.get("pair", "BTC/USDT")
    tech_data = state.get("technical_data") or {}

    # RAG-Fusion: search for bullish signals from multiple angles
    bull_results = rag_fusion.fused_search(
        f"{pair} bullish signals support RSI oversold accumulation",
        retriever, n_queries=3, top_k_per_query=5
    )
    bull_context = "\n".join([r.get("text", str(r)) for r in bull_results[:5]])

    # Phase 17: Add real technical data for data-backed arguments
    tech_summary = _format_tech_summary_compact(tech_data)
    tech_block = f"\n\nREAL-TIME INDICATORS:\n{tech_summary}" if tech_summary else ""

    # Agent DNA: Bull Researcher (Advocacy + Contrarian Self-Validation + Anti-Confirmation-Bias)
    BULL_SYSTEM = """IDENTITY: You are the BULL RESEARCHER — a dedicated advocate for the long/bullish case.
Your role is to find and present the STRONGEST possible bullish argument, backed by specific data.
You are an ADVOCATE, not a balanced analyst. The Bear Researcher handles the other side.

CONSTITUTIONAL RULES:
1. Every bullish argument MUST cite a specific indicator value or data point with [SOURCE: INDICATOR=VALUE].
2. NEVER fabricate support levels, volume data, or on-chain metrics. Only cite what's in the provided data.
3. Be HONEST about BULL_STRENGTH — if the evidence is weak, give a low score. A 0.90 with thin evidence destroys your credibility.
4. You MUST acknowledge your WEAKEST point — what's the biggest risk to the bull case?
5. Anti-Confirmation-Bias: For EVERY supporting argument, also note ONE contradicting data point from the same data.

CALIBRATION:
- BULL_STRENGTH 0.2-0.3: Very weak bull case, barely any supporting evidence
- BULL_STRENGTH 0.4-0.5: Mild bullish lean, mixed signals, several contradictions
- BULL_STRENGTH 0.6-0.7: Solid bullish case, multiple confirming signals, manageable risks
- BULL_STRENGTH 0.8-0.9: Strong bullish case, 3+ independent signals converge, risks are low
- BULL_STRENGTH > 0.9: Virtually never. Reserve for textbook setups with overwhelming multi-TF alignment.

YOU DO NOT: make final trade decisions, ignore contrary evidence, or inflate your strength score.
YOU DO: advocate for long positions, cite specific data, acknowledge weaknesses honestly."""

    prompt = f"""Build the STRONGEST possible bullish case for {pair}.

=== EVIDENCE ===
{bull_context}{tech_block}

=== STRUCTURED BULL CASE (complete ALL sections) ===

SECTION A — TECHNICAL BULL SIGNALS:
- Cite specific indicators that support a long position [SOURCE: INDICATOR=VALUE]
- RSI oversold? Support levels holding? Volume confirmation?

SECTION B — SENTIMENT/MACRO TAILWINDS:
- Any contrarian buy signals? Fear extremes? Positive news catalysts?

SECTION C — WEAKEST POINT (MANDATORY):
- What is the single biggest threat to this bull case?
- What specific price level or event would INVALIDATE the thesis?

SECTION D — ANTI-CONFIRMATION CHECK:
- For each KEY_ARGUMENT, note one contradicting data point

=== OUTPUT ===
- BULL_STRENGTH: 0.0-1.0 (calibrated per the scale above — DO NOT default to 0.7-0.8)
- KEY_ARGUMENTS: 2-3 strongest bullish points with [SOURCE] citations
- WEAKEST_POINT: The biggest risk
- INVALIDATION: Specific price or event that kills the bull case"""

    try:
        response = llm.invoke([SystemMessage(content=BULL_SYSTEM), HumanMessage(content=prompt)], temperature=0.3, priority="high")
        content_raw = response.content
        if isinstance(content_raw, list):
            content_raw = " ".join([b.get("text", "") for b in content_raw if "text" in b])
        return {"bull_case": content_raw.strip()}
    except Exception as e:
        logger.error(f"[NODE] Bull Researcher LLM invoke failed: {e}")
        return {"bull_case": f"Bull case unavailable (LLM error: {type(e).__name__})"}


def research_bearish(state: GraphState):
    """Bear Researcher: Collects and advocates for bearish evidence."""
    logger.info("---[NODE] BEAR RESEARCHER---")
    pair = state.get("pair", "BTC/USDT")
    tech_data = state.get("technical_data") or {}

    # RAG-Fusion: search for bearish signals from multiple angles
    bear_results = rag_fusion.fused_search(
        f"{pair} bearish signals resistance rejection death cross distribution",
        retriever, n_queries=3, top_k_per_query=5
    )
    bear_context = "\n".join([r.get("text", str(r)) for r in bear_results[:5]])

    # Phase 17: Add real technical data for data-backed arguments
    tech_summary = _format_tech_summary_compact(tech_data)
    tech_block = f"\n\nREAL-TIME INDICATORS:\n{tech_summary}" if tech_summary else ""

    # Agent DNA: Bear Researcher (Advocacy + Contrarian Self-Validation + Black Swan Awareness)
    BEAR_SYSTEM = """IDENTITY: You are the BEAR RESEARCHER — a dedicated advocate for the short/bearish case.
Your role is to find and present the STRONGEST possible bearish argument, backed by specific data.
You are an ADVOCATE, not a balanced analyst. The Bull Researcher handles the other side.

CONSTITUTIONAL RULES:
1. Every bearish argument MUST cite a specific indicator value or data point with [SOURCE: INDICATOR=VALUE].
2. NEVER fabricate resistance levels, volume data, or on-chain metrics. Only cite what's in the provided data.
3. Be HONEST about BEAR_STRENGTH — if the evidence is weak, give a low score. Inflating destroys credibility.
4. You MUST acknowledge your WEAKEST point — what's the biggest risk to the bear case? (e.g., a bear trap scenario)
5. Include BLACK SWAN AWARENESS: what tail risk exists that the data doesn't show? (regulatory, exchange hack, macro shock)
6. Anti-Confirmation-Bias: For EVERY supporting argument, also note ONE contradicting data point from the same data.

CALIBRATION:
- BEAR_STRENGTH 0.2-0.3: Very weak bear case, barely any supporting evidence
- BEAR_STRENGTH 0.4-0.5: Mild bearish lean, mixed signals
- BEAR_STRENGTH 0.6-0.7: Solid bearish case, multiple confirming signals
- BEAR_STRENGTH 0.8-0.9: Strong bearish case, 3+ independent signals converge
- BEAR_STRENGTH > 0.9: Virtually never. Reserve for textbook breakdowns.

YOU DO NOT: make final trade decisions, ignore contrary evidence, or inflate your strength score.
YOU DO: advocate for short/caution, cite specific data, acknowledge weaknesses honestly, consider tail risks."""

    prompt = f"""Build the STRONGEST possible bearish case for {pair}.

=== EVIDENCE ===
{bear_context}{tech_block}

=== STRUCTURED BEAR CASE (complete ALL sections) ===

SECTION A — TECHNICAL BEAR SIGNALS:
- Cite specific indicators that support a short/cautious position [SOURCE: INDICATOR=VALUE]
- Resistance rejection? Death cross? Declining volume? Bearish divergence?

SECTION B — SENTIMENT/MACRO HEADWINDS:
- Negative news catalysts? Fear rising? Macro risks (Fed, regulation)?

SECTION C — WEAKEST POINT (MANDATORY):
- What is the single biggest threat to this bear case?
- Historical scenario where this exact bear setup was a bear TRAP?

SECTION D — TAIL RISK CHECK:
- Any black swan risk that could accelerate the downside? (exchange hack, regulatory action, protocol exploit)
- Any black swan risk that could INVALIDATE the bear case? (ETF approval, whale buy, positive macro surprise)

=== OUTPUT ===
- BEAR_STRENGTH: 0.0-1.0 (calibrated per the scale above — DO NOT default to 0.7-0.8)
- KEY_ARGUMENTS: 2-3 strongest bearish points with [SOURCE] citations
- WEAKEST_POINT: The biggest risk to the bear thesis
- INVALIDATION: Specific price or event that kills the bear case"""

    try:
        response = llm.invoke([SystemMessage(content=BEAR_SYSTEM), HumanMessage(content=prompt)], temperature=0.3, priority="high")
        content_raw = response.content
        if isinstance(content_raw, list):
            content_raw = " ".join([b.get("text", "") for b in content_raw if "text" in b])
        return {"bear_case": content_raw.strip()}
    except Exception as e:
        logger.error(f"[NODE] Bear Researcher LLM invoke failed: {e}")
        return {"bear_case": f"Bear case unavailable (LLM error: {type(e).__name__})"}


def coordinator_debate(state: GraphState):
    """MADAM-RAG: Synthesizes all 5 agent reports via structured bull vs bear debate."""
    logger.info("---[NODE] MASTER COORDINATOR DEBATE (MADAM-RAG)---")
    pair = state.get("pair", "UNKNOWN")
    tech = state.get("technical_analysis", "No TA")
    sent = state.get("sentiment_analysis", "No Sentiment")
    news = state.get("news_analysis", "No News")
    bull = state.get("bull_case", "No bull case")
    bear = state.get("bear_case", "No bear case")
    tech_data = state.get("technical_data") or {}

    # Phase 17: Raw indicator block for coordinator cross-referencing
    raw_indicators = _format_tech_for_coordinator(tech_data)
    raw_block = f"\n\n{raw_indicators}" if raw_indicators else ""

    prompt = f"""You are the Master Coordinator (Executive AI) for a quantitative trading firm trading {pair}.

=== AGENT REPORTS ===

<technical_analyst>
{tech}
</technical_analyst>

<sentiment_analyst>
{sent}
</sentiment_analyst>

<news_analyst>
{news}
</news_analyst>

<bull_researcher>
{bull}
</bull_researcher>

<bear_researcher>
{bear}
</bear_researcher>
{raw_block}

=== MARKET REGIME CLASSIFICATION (Determine FIRST) ===
Using the RAW INDICATORS above, classify the regime:
- TRENDING: ADX>25, price consistently above/below EMA20 → signals are more reliable, follow the trend
- RANGING: ADX<20, price oscillating around EMA20 → signals are UNRELIABLE, bias toward NEUTRAL, reduce ALL confidence by 0.15
- HIGH_VOLATILITY: ATR>2x average → signals are unpredictable, cap confidence at 0.60

=== STRUCTURED DEBATE (complete ALL steps) ===

STEP 1 — EVIDENCE CROSS-CHECK:
Compare Bull vs Bear evidence strength. Which side has MORE concrete, data-backed arguments?
Cross-reference with the 3 analyst reports. Do technicals/sentiment/news support bull or bear?
CRITICAL: Use the RAW INDICATORS above to VERIFY agent claims — if an agent claims "RSI oversold" but RSI is actually 55, PENALIZE that agent's credibility.

STEP 2 — CONTRADICTION DETECTION:
Identify contradictions between agents. If 2+ agents contradict each other, confidence MUST decrease.
If ALL agents agree unanimously — be SKEPTICAL (groupthink). Reduce confidence by 0.10.

STEP 3 — PRE-MORTEM:
Assume your proposed signal FAILED and the trade lost 5% in 24h. Working backwards:
- What market condition changed that you underweighted?
- Which agent's evidence did you OVER-rely on?
If the pre-mortem reveals 2+ plausible failure modes, REDUCE confidence by 0.10.

STEP 4 — STEELMAN THE LOSING SIDE:
If leaning BULLISH: construct the STRONGEST bearish argument the Bear Researcher MISSED.
If leaning BEARISH: construct the STRONGEST bullish argument the Bull Researcher MISSED.
If the steelmanned counter-argument is strong (would change a reasonable person's mind), reduce confidence by 0.10.

STEP 5 — CONFIDENCE DECOMPOSITION:
Rate each 0.0 to 1.0:
- data_quality: How fresh/complete is the data? (weight: 0.25)
- signal_strength: How many independent indicators agree? (weight: 0.35)
- regime_confidence: How clear is the market regime? (weight: 0.25)
- analyst_agreement: How much do 5 agents agree? (weight: 0.15)
FINAL_CONFIDENCE = weighted average, then apply regime/pre-mortem/steelman adjustments.

=== CONFIDENCE CALIBRATION SCALE ===
0.50 = Coin flip. You would NOT bet your own money.
0.55 = Slight lean. One extra indicator supporting.
0.60 = Mild conviction. Multiple indicators agree but no strong catalyst.
0.65 = Moderate. Clear trend + sentiment alignment.
0.70 = Strong. 3+ independent signals converge — you MUST cite all 3.
0.75 = Very strong. Multi-timeframe alignment + volume confirmation.
0.80 = Extreme. Once-a-month textbook setup.
0.85+ = Almost never. Overwhelming evidence. Your MEDIAN should be 0.55-0.62.

=== CALIBRATION EXAMPLES ===
RSI=48, MACD slight positive, one EMA cross, neutral sentiment → BULLISH 0.52
RSI=38, MACD turning, EMA200 bounce, Fear&Greed=25 → BULLISH 0.63
RSI=32, MACD divergence, EMA200+volume, Fear=15, whale accumulation → BULLISH 0.74
Everything mixed, no edge → NEUTRAL 0.52

=== RESPONSE FORMAT (valid JSON ONLY, no markdown, no text outside JSON) ===
{{
   "signal": "BULLISH" | "BEARISH" | "NEUTRAL",
   "confidence": 0.00 to 1.00,
   "reasoning": "2-3 sentence synthesis. Cite which agents agreed/disagreed and which raw indicators confirmed. Mention regime and any confidence adjustments applied."
}}"""

    # Phase 18: Advanced RAG Auto-Toggle — only enable expensive stages if ChromaDB has enough docs
    _chroma_doc_count = 0
    try:
        _chroma_doc_count = retriever.collection.count()
    except Exception:
        pass
    _advanced_rag_enabled = _chroma_doc_count >= 100

    if not _advanced_rag_enabled:
        logger.info(f"[RAG] Advanced stages disabled (ChromaDB: {_chroma_doc_count} docs < 100 threshold)")

    # Phase 16: CoT-RAG Integration (only if advanced RAG enabled)
    if _advanced_rag_enabled and _cot_rag is not None:
        try:
            cot_results = _cot_rag.reason_step_by_step(pair=pair, query=prompt)
            cot_reasoning = cot_results.get("reasoning_chain", "")
            if cot_reasoning:
                prompt += f"\n\n[CoT-RAG 5-Step Deep Analysis]:\n{cot_reasoning}\nEnsure the final JSON decision factors in these evidence-backed step deductions."
        except Exception as e:
            logger.error(f"[CoT-RAG] Master Coordinator execution failed: {e}")

    # Complexity classification (always — cheap, needed for routing decisions)
    complexity = adaptive_router.classify(f"trading decision cross correlation {pair}")

    # SpeculativeRAG + FLARE — only if advanced RAG enabled AND COMPLEX query
    if _advanced_rag_enabled and complexity == "COMPLEX" and _spec_rag is not None:
        logger.info("[MADAM-RAG] COMPLEX flow: Using Speculative RAG to draft scenarios.")
        try:
            spec_result = _spec_rag.draft_and_verify(query=prompt, num_drafts=3)
            best_scenario = spec_result.get("best_draft", "")
            if best_scenario:
                prompt += f"\n\n[Speculative RAG Best Draft Scenario]:\n{best_scenario}"
        except Exception as e:
            logger.error(f"[Speculative RAG] Draft and verify failed: {e}")

        logger.info("[MADAM-RAG] COMPLEX flow: Using FLARE to verify reasoning before final JSON generation.")
        try:
            flare_context = f"Tech: {tech}\nSent: {sent}\nNews: {news}\nBull: {bull}\nBear: {bear}"
            flare_query = f"Synthesize a trading decision analysis for {pair} considering all evidence."
            adaptive_router.flare.retriever = retriever
            flare_res = adaptive_router.flare.generate_with_active_retrieval(query=flare_query, context=flare_context)
            reasoning_draft = flare_res.get("analysis", "")
            prompt += f"\n\nUse this verified FLARE reasoning as a base: {reasoning_draft}"
        except Exception as e:
            logger.error(f"[FLARE] Active retrieval failed: {e}. Proceeding without FLARE.")

    # --- Phase 18: 4-Tier Coordinator LLM with Degradation Chain ---
    # Tier 1: Primary LLM call
    # Tier 2: Retry with lower temperature
    # Tier 3: 3-agent voting fallback
    # Tier 4: Pure technical indicator fallback
    # Agent DNA: Master Coordinator (Debate Synthesis + Anti-Hallucination + Confidence Calibration)
    COORD_SYSTEM = """IDENTITY: You are the Master Coordinator (Executive AI) for a systematic crypto trading desk.
You synthesize reports from 5 specialist agents into a single, well-calibrated trading signal.

CONSTITUTIONAL RULES:
1. Your ENTIRE response MUST be a single valid JSON object. NOTHING else — no text before, no text after, no markdown.
2. Cross-check EVERY agent claim against the RAW INDICATORS. If an agent cites a number that contradicts the raw data, that agent's report is UNRELIABLE.
3. Confidence scores MUST follow the calibration scale provided. Your MEDIAN across all signals should be 0.55-0.62. If you're consistently above 0.70, you are overconfident.
4. NEVER exceed 0.85 confidence. Even textbook setups have tail risk.
5. If evidence is insufficient or contradictory, DEFAULT TO NEUTRAL with confidence 0.50-0.55. This is not a failure — it's intellectual honesty.
6. Count INDEPENDENT data sources, not repeated mentions. If Bull and Sentiment both cite "positive news," that's ONE source counted twice.
7. NEUTRAL signals MUST have confidence below 0.55.

BASE RATE ANCHORS (internalize before deciding):
- Only 30-40% of technical breakouts succeed. Most are false breakouts.
- Only 25-35% of oversold RSI bounces lead to meaningful recovery.
- News-driven pumps: 60-70% retrace within 24h.
- Multi-timeframe alignment raises success to 55-65%.

YOU DO NOT: always agree with the majority, inflate confidence, cite fabricated data, or output anything except JSON.
YOU DO: synthesize evidence, detect contradictions, apply calibration, penalize unverified claims, and produce well-calibrated signals."""

    coordinator_msgs = [SystemMessage(content=COORD_SYSTEM), HumanMessage(content=prompt)]

    def _parse_coordinator_response(content_raw):
        """Parse coordinator LLM response into (signal, conf, reason) or None on failure.
        Uses 3-tier extraction: direct JSON → brace extraction → regex fallback.
        This prevents wasting LLM calls when the model returns valid JSON wrapped in text."""
        if isinstance(content_raw, list):
            content_raw = " ".join([b.get("text", "") for b in content_raw if "text" in b])
        raw_content = re.sub(r'<think>.*?</think>', '', content_raw, flags=re.DOTALL)
        raw_content = raw_content.replace("```json", "").replace("```", "").strip()
        if not raw_content:
            return None

        # Tier A: Direct JSON parse (entire response is valid JSON)
        try:
            data = json.loads(raw_content)
            return (data.get("signal", "NEUTRAL"), float(data.get("confidence", 0.0)), data.get("reasoning", ""))
        except (json.JSONDecodeError, ValueError):
            pass

        # Tier B: Extract JSON object between first { and last } (LLM added text around JSON)
        brace_start = raw_content.find('{')
        brace_end = raw_content.rfind('}')
        if brace_start >= 0 and brace_end > brace_start:
            try:
                data = json.loads(raw_content[brace_start:brace_end + 1])
                logger.info("[Coordinator] JSON extracted via brace extraction (text around JSON stripped)")
                return (data.get("signal", "NEUTRAL"), float(data.get("confidence", 0.0)), data.get("reasoning", ""))
            except (json.JSONDecodeError, ValueError):
                pass

        # Tier C: Regex fallback — find signal/confidence even in malformed JSON
        signal_match = re.search(r'"signal"\s*:\s*"(BULLISH|BEARISH|NEUTRAL)"', raw_content, re.IGNORECASE)
        conf_match = re.search(r'"confidence"\s*:\s*([\d.]+)', raw_content)
        reason_match = re.search(r'"reasoning"\s*:\s*"([^"]*)"', raw_content)
        if signal_match and conf_match:
            signal = signal_match.group(1).upper()
            conf = min(float(conf_match.group(1)), 1.0)
            reason = reason_match.group(1) if reason_match else "Parsed via regex fallback"
            logger.info(f"[Coordinator] JSON extracted via regex fallback: {signal} {conf}")
            return (signal, conf, reason)

        logger.error(f"Failed to parse Coordinator JSON (all 3 tiers). Raw: {raw_content[:300]}")
        return None

    # Tier 1: Primary LLM call (temperature=0.7)
    parsed = None
    try:
        response = llm.invoke(coordinator_msgs, temperature=0.7, priority="critical")
        parsed = _parse_coordinator_response(response.content)
    except Exception as e:
        logger.error(f"[NODE] Coordinator primary LLM failed: {type(e).__name__}: {e}")

    # Tier 2: Retry with lower temperature + stricter JSON-only prompt
    if parsed is None:
        logger.warning("[NODE] Coordinator: Primary failed/unparseable. Retrying with temperature=0.3 + strict JSON prompt...")
        try:
            strict_msgs = [
                SystemMessage(content="""You are a JSON-only API endpoint. You receive trading analysis and return a single JSON object.
RULES:
1. Your ENTIRE response is a JSON object. No text before. No text after. No markdown. No code fences.
2. If you cannot decide, return: {"signal":"NEUTRAL","confidence":0.50,"reasoning":"Insufficient evidence for directional call."}
3. Confidence range: 0.00 to 0.85. NEVER exceed 0.85.
4. Signal must be exactly one of: "BULLISH", "BEARISH", "NEUTRAL"."""),
                HumanMessage(content=prompt + "\n\nRESPOND WITH ONLY THE JSON OBJECT. Example format:\n{\"signal\":\"NEUTRAL\",\"confidence\":0.52,\"reasoning\":\"Mixed signals across agents.\"}")
            ]
            response = llm.invoke(strict_msgs, temperature=0.3, priority="critical")
            parsed = _parse_coordinator_response(response.content)
        except Exception as e:
            logger.error(f"[NODE] Coordinator retry also failed: {type(e).__name__}: {e}")

    # Tier 1/2 success
    if parsed is not None:
        signal, conf, reason = parsed
        return {"signal": signal, "confidence": conf, "reasoning": reason, "source": "AI"}

    # Tier 3: 3-Agent Voting Fallback (tech, sent, news — unbiased agents only)
    logger.warning("[NODE] Coordinator: Both LLM calls failed. Using VOTING FALLBACK.")
    vote_result = _voting_fallback(tech, sent, news)
    if vote_result["confidence"] > 0:
        return vote_result

    # Tier 4: Technical Fallback (pure indicator scoring)
    # Always called — even with empty tech_data, returns minimum 0.01 confidence
    logger.warning("[NODE] Coordinator: Voting inconclusive. Using TECHNICAL FALLBACK.")
    return _technical_fallback(tech_data)



# --- Graph Construction (Multi-Agent DAG) ---
workflow = StateGraph(GraphState)

# Define nodes (5 parallel analysts + 1 coordinator)
workflow.add_node("analyze_technical", analyze_technical)
workflow.add_node("analyze_sentiment", analyze_sentiment)
workflow.add_node("analyze_news", analyze_news)
workflow.add_node("research_bullish", research_bullish)
workflow.add_node("research_bearish", research_bearish)
workflow.add_node("coordinator_debate", coordinator_debate)

# Define edges (5 Parallel Agents)
workflow.add_edge(START, "analyze_technical")
workflow.add_edge(START, "analyze_sentiment")
workflow.add_edge(START, "analyze_news")
workflow.add_edge(START, "research_bullish")
workflow.add_edge(START, "research_bearish")

# All 5 parallel nodes converge to the MADAM Coordinator Debate
workflow.add_edge(
    ["analyze_technical", "analyze_sentiment", "analyze_news", "research_bullish", "research_bearish"],
    "coordinator_debate"
)

workflow.add_edge("coordinator_debate", END)

# Compile graph
rag_bot = workflow.compile()

def get_trading_signal(pair: str, technical_data: dict = None) -> dict:
    """Entry point for Freqtrade to request a trading decision from the Analyst Team."""
    logger.info(f"Initiating Multi-Agent Analyst Team for {pair}{'  [with LIVE indicators]' if technical_data else ''}...")
    
    # Phase 9: Semantic Cache (module-level singleton)
    query_str = f"trading signal analysis for {pair}"

    # a. Check Cache
    cached_response_str = _semantic_cache.get(query=query_str, pair=pair)
    if cached_response_str:
        try:
            logger.info("[Semantic Cache] Reusing cached decision for pair.")
            cached_result = json.loads(cached_response_str)
            # Phase 18: Track cache hits in signal stats
            _signal_stats["total"] += 1
            _signal_stats["ai"] += 1  # cache = previous AI result
            _record_signal_health(pair, "CACHE", cached_result.get("signal", "NEUTRAL"), cached_result.get("confidence", 0.0))
            return cached_result
        except Exception as e:
            logger.error(f"Failed to parse cached response: {e}")

    # b. Rule-based Retrieval Gating Check
    # Trade-First: NEVER block a signal. If retrieval is skipped, proceed with reduced confidence.
    # The position sizer will modulate size accordingly — confidence controls SIZE, not PERMISSION.
    retrieval_gated = not _self_rag.should_retrieve(query_str, {})
    if retrieval_gated:
        logger.info(f"[RAG] Retrieval gated for '{query_str[:40]}...' — proceeding with LLM-only analysis at reduced confidence.")
    
    # Initialize state
    inputs = {
        "pair": pair,
        "documents": [],
        "technical_data": technical_data or {},
        "technical_analysis": "",
        "sentiment_analysis": "",
        "news_analysis": "",
        "bull_case": "",
        "bear_case": "",
        "signal": "",
        "confidence": 0.0,
        "reasoning": ""
    }
    
    # d. Retry logic with Self-RAG Critique
    max_retries = 1
    final_output = {}
    signal = "NEUTRAL"
    confidence = 0.0
    reasoning = ""
    
    for attempt in range(max_retries + 1):
        try:
            for output in rag_bot.stream(inputs):
                for key, value in output.items():
                    # Accumulate outputs from all nodes — don't overwrite partial results
                    # If coordinator crashes, we still have analyst data for diagnostics
                    final_output.update(value)
        except Exception as e:
            logger.error(f"[GRAPH] rag_bot.stream() crashed on attempt {attempt+1}: {type(e).__name__}: {e}")
            reasoning = f"Graph execution error: {type(e).__name__}: {e}"

        signal = final_output.get("signal", "NEUTRAL") if final_output else "NEUTRAL"
        confidence = final_output.get("confidence", 0.0) if final_output else 0.0
        if not reasoning:
            reasoning = final_output.get("reasoning", "") if final_output else ""

        # Self-RAG Critique
        critique = _self_rag.self_critique(
            query=query_str,
            response=f"Signal: {signal}. Reasoning: {reasoning}",
            evidence=[final_output.get("technical_analysis", ""), final_output.get("news_analysis", "")]
        )

        if critique["passed"] or attempt == max_retries:
            if not critique["passed"]:
                logger.warning(f"[Self-RAG] Output failed critique, but max retries reached. Proceeding.")
            else:
                logger.info(f"[Self-RAG] Response critique PASSED. Quality verified.")
            break

        logger.warning(f"[Self-RAG] Output failed critique. Retrying pipeline. Attempt {attempt+1}/{max_retries}")

    # Log the decision persistently in Phase 3.5.1 Logger
    decision_logger.log_decision(
        pair=pair,
        signal_type=signal,
        confidence=confidence,
        reasoning_summary=reasoning,
        regime="MULTI_AGENT_PHASE_5"
    )
    
    result_dict = {
        "signal": signal,
        "confidence": confidence,
        "reasoning": reasoning
    }

    # Phase 18: Signal health tracking + Telegram alert
    signal_source = final_output.get("source", "AI") if final_output else "TIMEOUT"
    _record_signal_health(pair, signal_source, signal, confidence)

    _signal_stats["total"] += 1
    if signal_source in ("AI", "CACHE"):
        _signal_stats["ai"] += 1
    elif signal_source == "VOTING":
        _signal_stats["voting"] += 1
    else:
        _signal_stats["fallback"] += 1

    # Check AI ratio every 10 signals — alert if below 50%
    if _signal_stats["total"] % 10 == 0 and _signal_stats["total"] > 0:
        ai_ratio = _signal_stats["ai"] / _signal_stats["total"] * 100
        if ai_ratio < 50.0:
            try:
                from telegram_notifier import AITelegramNotifier
                AITelegramNotifier().send_alert(
                    f"AI sinyal orani duestu: {ai_ratio:.0f}% "
                    f"(AI:{_signal_stats['ai']}, Voting:{_signal_stats['voting']}, "
                    f"Fallback:{_signal_stats['fallback']} / {_signal_stats['total']} total)",
                    level="WARNING", cooldown_secs=3600
                )
            except Exception:
                pass

    # e. Put to Semantic Cache (SemanticCache.put already rejects confidence < 0.3)
    _semantic_cache.put(query=query_str, response=json.dumps(result_dict), pair=pair)
    if confidence < 0.3:
        logger.warning(f"[Signal] Low confidence result ({confidence:.2f}) for {pair} — NOT cached, will re-analyze next time.")

    return result_dict

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

_signal_executor = ThreadPoolExecutor(max_workers=4)

def get_trading_signal_with_timeout(pair: str, timeout_seconds: int = 45, technical_data: dict = None) -> dict:
    """Wraps get_trading_signal with a thread-based timeout (uvicorn-safe)."""
    try:
        future = _signal_executor.submit(get_trading_signal, pair, technical_data)
        return future.result(timeout=timeout_seconds)
    except FuturesTimeoutError:
        logger.warning(f"[TIMEOUT] Pipeline for {pair} exceeded {timeout_seconds}s. Using technical fallback...")
        result = _technical_fallback(technical_data or {})
        _record_signal_health(pair, "FALLBACK", result["signal"], result["confidence"])
        _signal_stats["total"] += 1
        _signal_stats["fallback"] += 1
        return result
    except Exception as e:
        logger.error(f"[ERROR] Pipeline for {pair} failed: {e}")
        result = _technical_fallback(technical_data or {})
        _record_signal_health(pair, "FALLBACK", result["signal"], result["confidence"])
        _signal_stats["total"] += 1
        _signal_stats["fallback"] += 1
        return result

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", type=str, default="BTC/USDT", help="Pair to analyze")
    parser.add_argument("--serve", action="store_true", help="Run as HTTP service (models loaded once)")
    parser.add_argument("--port", type=int, default=8891, help="Port for HTTP service")
    args = parser.parse_args()

    if args.serve:
        from fastapi import FastAPI, Request
        from fastapi.middleware.cors import CORSMiddleware
        import uvicorn
        import gc
        import threading

        # Periodic GC daemon — forces garbage collection every 5 min
        # Prevents glibc memory fragmentation from making RSS grow indefinitely
        def _gc_daemon():
            import time as _time
            while True:
                _time.sleep(300)
                collected = gc.collect()
                if collected:
                    logger.info(f"[GC] Collected {collected} objects")

        gc_thread = threading.Thread(target=_gc_daemon, daemon=True)
        gc_thread.start()

        serve_app = FastAPI(title="RAG Signal Service")
        serve_app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"],
        )

        @serve_app.get("/signal/{pair:path}")
        def signal_endpoint(pair: str):
            return get_trading_signal_with_timeout(pair, timeout_seconds=45)

        @serve_app.post("/signal/{pair:path}")
        async def signal_endpoint_post(pair: str, request: Request):
            """Phase 17: POST endpoint accepts technical_data from strategy."""
            try:
                body = await request.json()
            except Exception:
                body = {}
            technical_data = body.get("technical_data")
            return get_trading_signal_with_timeout(pair, timeout_seconds=45, technical_data=technical_data)

        @serve_app.get("/health")
        def health():
            return {
                "status": "online",
                "models_loaded": True,
                "colbert": "active" if retriever.colbert_reranker else "disabled",
                "flashrank": "active" if retriever.reranker else "disabled",
            }

        @serve_app.get("/signal-health")
        def signal_health_endpoint():
            """Phase 18: Signal source distribution for last 24 hours."""
            conn = None
            try:
                from db import get_db_connection
                conn = get_db_connection()
                rows = conn.execute("""
                    SELECT signal_source, COUNT(*) as cnt
                    FROM signal_health
                    WHERE timestamp > datetime('now', '-24 hours')
                    GROUP BY signal_source
                """).fetchall()

                total = sum(r['cnt'] for r in rows)
                dist = {r['signal_source']: r['cnt'] for r in rows}
                ai_count = dist.get('AI', 0) + dist.get('CACHE', 0)
                ai_ratio = (ai_count / total * 100) if total > 0 else 100.0

                return {
                    "total_signals_24h": total,
                    "distribution": dist,
                    "ai_ratio_pct": round(ai_ratio, 1),
                    "healthy": ai_ratio >= 50.0,
                    "in_memory_stats": dict(_signal_stats)
                }
            except Exception as e:
                return {"error": str(e), "in_memory_stats": dict(_signal_stats)}
            finally:
                if conn:
                    conn.close()

        logger.info(f"RAG Signal Service starting on port {args.port}")
        logger.info(f"Models loaded: ColBERT={'active' if retriever.colbert_reranker else 'disabled'}, "
                     f"FlashRank={'active' if retriever.reranker else 'disabled'}")
        uvicorn.run(serve_app, host="0.0.0.0", port=args.port)
    else:
        result = get_trading_signal(args.pair)
        print("\n--- JSON OUTPUT ---")
        print(json.dumps(result))
