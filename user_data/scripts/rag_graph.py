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

    if tech_context:
        prompt = f"""You are a master Crypto Technical Analyst with access to REAL-TIME indicator data.

Analyze these LIVE technical indicators for {pair}:
{tech_context}
{magma_context}

Based on these REAL numbers, provide a dense 2-3 sentence technical analysis:
1. Identify the current trend (bullish/bearish/ranging) from EMA/SMA alignment and price position
2. Assess momentum (RSI overbought/oversold, MACD cross direction, ADX strength)
3. Note volatility regime (ATR%, Bollinger Band width) and key levels (support/resistance from BB & EMAs)
4. Analyze the last candle patterns (momentum, reversals, volume confirmation)

State whether the technicals lean BULLISH, BEARISH, or NEUTRAL.
NEVER provide a final trading signal. ONLY provide your technical perspective."""
    else:
        prompt = f"""You are a master Crypto Technical Analyst.
Analyze these current technical indicators and market search results for {pair}:
{search_res}
{magma_context}

Provide a dense, 2-3 sentence technical analysis. State whether the technicals lean BULLISH, BEARISH, or NEUTRAL.
NEVER provide a final trading signal. ONLY provide your technical perspective."""

    try:
        response = llm.invoke([SystemMessage(content="You are a Technical Analyst."), HumanMessage(content=prompt)])
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
        
        conn.close()
    except Exception as e:
        logger.warning(f"Sentiment DB query failed: {e}. Falling back to neutral context.")
        db_context_parts.append("Sentiment data temporarily unavailable. Assume NEUTRAL baseline.")
    
    db_context = " ".join(db_context_parts)
    # ===== END LIVE DATA =====
    
    prompt = f"""You are a Behavioral Economics & Crypto Sentiment Analyst. 
Analyze the current sentiment metrics for {pair}:
{db_context}

Provide a 2-3 sentence psychological market analysis. State whether the crowd sentiment leans BULLISH, BEARISH, or NEUTRAL.
NEVER provide a final trading signal. ONLY provide your sentiment perspective."""

    try:
        response = llm.invoke([SystemMessage(content="You are a Sentiment Analyst."), HumanMessage(content=prompt)], temperature=0.4)
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
    
    prompt = f"""You are a Crypto Fundamental & Macroeconomic Analyst. 
Analyze these retrieved recent news documents and Knowledge Graph relations for {pair}:
{context}

Provide a dense, 2-3 sentence fundamental analysis on the news. State whether the fundamentals lean BULLISH, BEARISH, or NEUTRAL. 
Focus only on news impact. NEVER provide a final trading signal."""

    try:
        response = llm.invoke([SystemMessage(content="You are a Fundamental News Analyst."), HumanMessage(content=prompt)], temperature=0.3)
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

    prompt = f"""You are a BULL RESEARCHER for {pair}. Your job is to build the STRONGEST possible bullish case.

Relevant evidence:
{bull_context}{tech_block}

Build your bullish argument covering:
1. Technical strength: RSI oversold bounce, support levels holding, volume increases
2. Sentiment tailwinds: positive news, Fear & Greed trending toward Greed
3. On-chain: whale accumulation, exchange outflows, hodler behavior

Use the REAL indicator values above to support your arguments with specific numbers (e.g., "RSI at 35 is approaching oversold territory").

Output format:
- BULL_STRENGTH: 0.0-1.0 (how strong is the bullish case?)
- KEY_ARGUMENTS: 2-3 strongest bullish points

Be an advocate. Find the BEST bullish evidence, but be honest about weakness."""
    
    try:
        response = llm.invoke([SystemMessage(content="You are the Bull Researcher."), HumanMessage(content=prompt)], temperature=0.3)
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

    prompt = f"""You are a BEAR RESEARCHER for {pair}. Your job is to build the STRONGEST possible bearish case.

Relevant evidence:
{bear_context}{tech_block}

Build your bearish argument covering:
1. Technical weakness: resistance rejection, death cross, declining volume
2. Sentiment headwinds: negative news, Fear & Greed trending toward Fear
3. On-chain: whale distribution, exchange inflows, miner selling

Use the REAL indicator values above to support your arguments with specific numbers (e.g., "MACD histogram at -0.0045 confirms bearish momentum").

Output format:
- BEAR_STRENGTH: 0.0-1.0 (how strong is the bearish case?)
- KEY_ARGUMENTS: 2-3 strongest bearish points

Be an advocate. Find the BEST bearish evidence, but be honest about weakness."""
    
    try:
        response = llm.invoke([SystemMessage(content="You are the Bear Researcher."), HumanMessage(content=prompt)], temperature=0.3)
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
You have received reports from your 5-agent analyst team:

[TECHNICAL ANALYST]:
{tech}

[SENTIMENT ANALYST]:
{sent}

[NEWS & MACRO ANALYST]:
{news}

[BULL RESEARCHER — Advocacy for LONG]:
{bull}

[BEAR RESEARCHER — Advocacy for SHORT]:
{bear}{raw_block}

CONDUCT A STRUCTURED DEBATE:
1. Compare Bull vs Bear evidence strength. Which side has MORE concrete, data-backed arguments?
2. Cross-reference with the 3 analyst reports. Do technicals/sentiment/news support bull or bear?
3. Use the RAW TECHNICAL INDICATORS above to VERIFY agent claims — if an agent says "RSI is oversold" but RSI is actually 55, penalize that agent's credibility.
4. Identify any CONTRADICTIONS between agents.
5. Make your final decision based on the WEIGHT OF EVIDENCE, not on any single agent.
6. Confidence = bull_strength / (bull_strength + bear_strength), adjusted by analyst agreement and indicator confirmation.

Respond in valid JSON ONLY, no markdown:
{{
   "signal": "BULLISH" | "BEARISH" | "NEUTRAL",
   "confidence": 0.00 to 1.00,
   "reasoning": "2-sentence synthesis of the debate outcome. Mention which agents agreed/disagreed."
}}"""

    # Phase 16: CoT-RAG Integration
    if _cot_rag is not None:
        try:
            cot_results = _cot_rag.reason_step_by_step(pair=pair, query=prompt)
            cot_reasoning = cot_results.get("reasoning_chain", "")
            if cot_reasoning:
                prompt += f"\n\n[CoT-RAG 5-Step Deep Analysis]:\n{cot_reasoning}\nEnsure the final JSON decision factors in these evidence-backed step deductions."
        except Exception as e:
            logger.error(f"[CoT-RAG] Master Coordinator execution failed: {e}")
    complexity = adaptive_router.classify(f"trading decision cross correlation {pair}")
    if complexity == "COMPLEX" and _spec_rag is not None:
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

    # Final LLM call — the coordinator's own synthesis. Must succeed for a real signal.
    try:
        response = llm.invoke([SystemMessage(content="You are the Master Coordinator."), HumanMessage(content=prompt)], temperature=0.7)
        content_raw = response.content
        if isinstance(content_raw, list):
            content_raw = " ".join([b.get("text", "") for b in content_raw if "text" in b])

        # Strip thinking model tags (DeepSeek R1, Qwen reasoning, etc.)
        raw_content = re.sub(r'<think>.*?</think>', '', content_raw, flags=re.DOTALL)
        raw_content = raw_content.replace("```json", "").replace("```", "").strip()

        if not raw_content:
            logger.warning("[NODE] Coordinator received empty LLM response. Defaulting to NEUTRAL.")
            signal, conf, reason = "NEUTRAL", 0.0, "Empty LLM response from coordinator"
            return {"signal": signal, "confidence": conf, "reasoning": reason}

        try:
            data = json.loads(raw_content)
            signal = data.get("signal", "NEUTRAL")
            conf = float(data.get("confidence", 0.0))
            reason = data.get("reasoning", "")
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Failed to parse Coordinator JSON: {e}. Raw: {raw_content[:200]}")
            signal, conf, reason = "NEUTRAL", 0.0, f"JSON Parsing Failure: {raw_content[:100]}"
    except Exception as e:
        logger.error(f"[NODE] Coordinator LLM invoke FAILED: {type(e).__name__}: {e}")
        signal, conf, reason = "NEUTRAL", 0.0, f"Coordinator LLM error: {type(e).__name__}: {e}"

    return {"signal": signal, "confidence": conf, "reasoning": reason}



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
            return json.loads(cached_response_str)
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
        logger.warning(f"[TIMEOUT] Pipeline for {pair} exceeded {timeout_seconds}s. Returning NEUTRAL.")
        return {"signal": "NEUTRAL", "confidence": 0.0, "reasoning": "Pipeline timeout"}
    except Exception as e:
        logger.error(f"[ERROR] Pipeline for {pair} failed: {e}")
        return {"signal": "NEUTRAL", "confidence": 0.0, "reasoning": f"Pipeline error: {e}"}

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

        logger.info(f"RAG Signal Service starting on port {args.port}")
        logger.info(f"Models loaded: ColBERT={'active' if retriever.colbert_reranker else 'disabled'}, "
                     f"FlashRank={'active' if retriever.reranker else 'disabled'}")
        uvicorn.run(serve_app, host="0.0.0.0", port=args.port)
    else:
        result = get_trading_signal(args.pair)
        print("\n--- JSON OUTPUT ---")
        print(json.dumps(result))
