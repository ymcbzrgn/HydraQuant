import os
import logging
import json
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

# Load Env
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))
logger = logging.getLogger(__name__)

# Replace raw Gemini with the zero-latency failover Router
llm = LLMRouter(temperature=0)
web_search_tool = DuckDuckGoSearchRun()

# Single persistent retriever instance
retriever = HybridRetriever()

# Phase 3.2: Corrective RAG — evaluates + fixes bad retrieval
crag = CRAGEvaluator(router=llm)

# Phase 3.3: Adaptive RAG — routes queries to optimal pipeline
adaptive_router = AdaptiveQueryRouter(router=llm)

# Phase 3.4: RAG-Fusion for multi-perspective retrieval
rag_fusion = RAGFusion(router=llm)

# Persistent Logger for AI Decisions (Phase 3.5.1)
decision_logger = AIDecisionLogger()

# --- Graph State Definition ---
class GraphState(TypedDict):
    """
    State dictionary for the LangGraph Multi-Agent RAG Brain.
    Phase 5.2: Extended with bull/bear researcher outputs for MADAM debate.
    """
    pair: str
    documents: List[str]
    technical_analysis: str
    sentiment_analysis: str
    news_analysis: str
    bull_case: str
    bear_case: str
    signal: str
    confidence: float
    reasoning: str

# --- Parallel Analyst Nodes ---

def analyze_technical(state: GraphState):
    """Fetches and analyzes real-time technical indicators."""
    logger.info("---[NODE] TECHNICAL ANALYST---")
    pair = state.get("pair", "BTC/USDT")
    
    # Agent-Specific LLM (Zero Emotion, High Reliability)
    tech_llm = LLMRouter(temperature=0.1, request_timeout=15)
    
    # In a full production env, this would call ccxt or ta-lib. 
    # For now, we use a tool-augmented web search to get current TA context.
    try:
        from langchain_community.tools.ddg_search import DuckDuckGoSearchRun
        search = DuckDuckGoSearchRun()
        search_res = search.invoke(f"{pair} current technical analysis RSI MACD support resistance price prediction")
    except Exception as e:
        logger.warning(f"Technical Search Failed: {e}")
        search_res = "Unable to fetch live technical data."
        
    # Phase 15: MAGMA Graph Context (Semantic + Causal)
    magma_context = ""
    try:
        magma = MAGMAMemory()
        semantic_edges = magma.query(f"{pair} tech analysis", graph_types=["semantic", "causal"], max_hops=1)
        if semantic_edges:
            ext_nodes = [f"{e['source']} -> {e['relation']} -> {e['target']}" for e in semantic_edges[:10]]
            magma_context = "\\nMAGMA Historic Technical Correlations:\\n" + "\\n".join(ext_nodes)
    except Exception as e:
        logger.warning(f"MAGMA Technical lookup failed: {e}")

    prompt = f"""You are a master Crypto Technical Analyst. 
Analyze these current technical indicators and market search results for {pair}:
{search_res}
{magma_context}

Provide a dense, 2-3 sentence technical analysis. State whether the technicals lean BULLISH, BEARISH, or NEUTRAL.
NEVER provide a final trading signal. ONLY provide your technical perspective."""

    response = tech_llm.invoke([SystemMessage(content="You are a Technical Analyst."), HumanMessage(content=prompt)])
    
    content_raw = response.content
    if isinstance(content_raw, list):
        content_raw = " ".join([b.get("text", "") for b in content_raw if "text" in b])
        
    return {"technical_analysis": content_raw.strip()}


def analyze_sentiment(state: GraphState):
    """Retrieves and analyzes the latest DB fear/greed and CryptoBERT sentiment."""
    logger.info("---[NODE] SENTIMENT ANALYST---")
    pair = state.get("pair", "BTC/USDT")
    
    # Agent-Specific LLM (Understands human emotion and market hype)
    sent_llm = LLMRouter(temperature=0.4, request_timeout=15)
    
    # ===== LIVE DATA: Query real sentiment from ai_data.sqlite =====
    import sqlite3
    from ai_config import AI_DB_PATH as db_path
    db_context_parts = []
    
    try:
        conn = sqlite3.connect(db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        
        # Fear & Greed Index (live from fng_fetcher.py)
        c.execute("SELECT value, value_classification FROM fear_and_greed ORDER BY timestamp DESC LIMIT 1")
        fng_row = c.fetchone()
        if fng_row:
            db_context_parts.append(f"Fear & Greed Index: {fng_row['value']} ({fng_row['value_classification']}).")
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

    response = sent_llm.invoke([SystemMessage(content="You are a Sentiment Analyst."), HumanMessage(content=prompt)])
    
    content_raw = response.content
    if isinstance(content_raw, list):
        content_raw = " ".join([b.get("text", "") for b in content_raw if "text" in b])
        
    return {"sentiment_analysis": content_raw.strip()}


def analyze_news(state: GraphState):
    """Retrieves and reads the latest semantic/BM25 news chunks from ChromaDB."""
    logger.info("---[NODE] NEWS & MACRO ANALYST---")
    pair = state.get("pair", "BTC/USDT")
    
    # Agent-Specific LLM (Factual News Extraction)
    news_llm = LLMRouter(temperature=0.3, request_timeout=15)
    
    # Phase 3.3: Adaptive RAG — route query to optimal pipeline
    query = f"{pair} macro fundamentals exact news ETF"
    corrected_results = adaptive_router.route(
        query=query,
        retriever=retriever,
        crag_evaluator=crag
    )
    
    documents = [res.get("text", str(res)) for res in corrected_results[:5]]
    
    # Phase 5.1: Knowledge Graph Traversal
    kg = KnowledgeGraphManager()
    base_coin = pair.split("/")[0]
    
    # Query for ticker and full name (e.g., BTC and Bitcoin)
    network_links = kg.query_entity_network(base_coin)
    
    # Add common full names for major coins to enrich graph hits
    coin_map = {"BTC": "Bitcoin", "ETH": "Ethereum", "SOL": "Solana", "XRP": "Ripple"}
    if base_coin in coin_map:
        network_links.extend(kg.query_entity_network(coin_map[base_coin]))
        
    kg_context = "\n".join(list(set(network_links))) # Deduplicate
    
    if kg_context:
        documents.append(f"--- KNOWLEDGE GRAPH RELATIONS ---\n{kg_context}")
    
    context = "\n\n".join(documents)
    
    prompt = f"""You are a Crypto Fundamental & Macroeconomic Analyst. 
Analyze these retrieved recent news documents and Knowledge Graph relations for {pair}:
{context}

Provide a dense, 2-3 sentence fundamental analysis on the news. State whether the fundamentals lean BULLISH, BEARISH, or NEUTRAL. 
Focus only on news impact. NEVER provide a final trading signal."""

    response = news_llm.invoke([SystemMessage(content="You are a Fundamental News Analyst."), HumanMessage(content=prompt)])
    
    content_raw = response.content
    if isinstance(content_raw, list):
        content_raw = " ".join([b.get("text", "") for b in content_raw if "text" in b])
        
    return {"news_analysis": content_raw.strip(), "documents": documents}


# --- Phase 5.2: Bull/Bear Researcher Nodes ---

def research_bullish(state: GraphState):
    """Bull Researcher: Collects and advocates for bullish evidence."""
    logger.info("---[NODE] BULL RESEARCHER---")
    pair = state.get("pair", "BTC/USDT")
    
    bull_llm = LLMRouter(temperature=0.3, request_timeout=15)
    
    # RAG-Fusion: search for bullish signals from multiple angles
    bull_results = rag_fusion.fused_search(
        f"{pair} bullish signals support RSI oversold accumulation",
        retriever, n_queries=3, top_k_per_query=5
    )
    bull_context = "\n".join([r.get("text", str(r)) for r in bull_results[:5]])
    
    prompt = f"""You are a BULL RESEARCHER for {pair}. Your job is to build the STRONGEST possible bullish case.

Relevant evidence:
{bull_context}

Build your bullish argument covering:
1. Technical strength: RSI oversold bounce, support levels holding, volume increases
2. Sentiment tailwinds: positive news, Fear & Greed trending toward Greed
3. On-chain: whale accumulation, exchange outflows, hodler behavior

Output format:
- BULL_STRENGTH: 0.0-1.0 (how strong is the bullish case?)
- KEY_ARGUMENTS: 2-3 strongest bullish points

Be an advocate. Find the BEST bullish evidence, but be honest about weakness."""
    
    response = bull_llm.invoke([SystemMessage(content="You are the Bull Researcher."), HumanMessage(content=prompt)])
    content_raw = response.content
    if isinstance(content_raw, list):
        content_raw = " ".join([b.get("text", "") for b in content_raw if "text" in b])
    
    return {"bull_case": content_raw.strip()}


def research_bearish(state: GraphState):
    """Bear Researcher: Collects and advocates for bearish evidence."""
    logger.info("---[NODE] BEAR RESEARCHER---")
    pair = state.get("pair", "BTC/USDT")
    
    bear_llm = LLMRouter(temperature=0.3, request_timeout=15)
    
    # RAG-Fusion: search for bearish signals from multiple angles
    bear_results = rag_fusion.fused_search(
        f"{pair} bearish signals resistance rejection death cross distribution",
        retriever, n_queries=3, top_k_per_query=5
    )
    bear_context = "\n".join([r.get("text", str(r)) for r in bear_results[:5]])
    
    prompt = f"""You are a BEAR RESEARCHER for {pair}. Your job is to build the STRONGEST possible bearish case.

Relevant evidence:
{bear_context}

Build your bearish argument covering:
1. Technical weakness: resistance rejection, death cross, declining volume
2. Sentiment headwinds: negative news, Fear & Greed trending toward Fear
3. On-chain: whale distribution, exchange inflows, miner selling

Output format:
- BEAR_STRENGTH: 0.0-1.0 (how strong is the bearish case?)
- KEY_ARGUMENTS: 2-3 strongest bearish points

Be an advocate. Find the BEST bearish evidence, but be honest about weakness."""
    
    response = bear_llm.invoke([SystemMessage(content="You are the Bear Researcher."), HumanMessage(content=prompt)])
    content_raw = response.content
    if isinstance(content_raw, list):
        content_raw = " ".join([b.get("text", "") for b in content_raw if "text" in b])
    
    return {"bear_case": content_raw.strip()}


def coordinator_debate(state: GraphState):
    """MADAM-RAG: Synthesizes all 5 agent reports via structured bull vs bear debate."""
    logger.info("---[NODE] MASTER COORDINATOR DEBATE (MADAM-RAG)---")
    pair = state.get("pair", "UNKNOWN")
    tech = state.get("technical_analysis", "No TA")
    sent = state.get("sentiment_analysis", "No Sentiment")
    news = state.get("news_analysis", "No News")
    bull = state.get("bull_case", "No bull case")
    bear = state.get("bear_case", "No bear case")
    
    exec_llm = LLMRouter(temperature=0.7, request_timeout=25)
    
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
{bear}

CONDUCT A STRUCTURED DEBATE:
1. Compare Bull vs Bear evidence strength. Which side has MORE concrete, data-backed arguments?
2. Cross-reference with the 3 analyst reports. Do technicals/sentiment/news support bull or bear?
3. Identify any CONTRADICTIONS between agents.
4. Make your final decision based on the WEIGHT OF EVIDENCE, not on any single agent.
5. Confidence = bull_strength / (bull_strength + bear_strength), adjusted by analyst agreement.

        
Respond in valid JSON ONLY, no markdown:
{{
   "signal": "BULLISH" | "BEARISH" | "NEUTRAL",
   "confidence": 0.00 to 1.00,
   "reasoning": "2-sentence synthesis of the debate outcome. Mention which agents agreed/disagreed."
}}"""

    # Phase 16: CoT-RAG Integration
    try:
        from cot_rag import CoTRAG
        cot_rag = CoTRAG(llm_router=exec_llm, retriever=retriever)
        cot_results = cot_rag.reason_step_by_step(pair=pair, query=prompt)
        cot_reasoning = cot_results.get("reasoning_chain", "")
        if cot_reasoning:
            prompt += f"\n\n[CoT-RAG 5-Step Deep Analysis]:\n{cot_reasoning}\nEnsure the final JSON decision factors in these evidence-backed step deductions."
    except Exception as e:
        logger.error(f"[CoT-RAG] Master Coordinator execution failed: {e}")
    complexity = adaptive_router.classify(f"trading decision cross correlation {pair}")
    if complexity == "COMPLEX":
        logger.info("[MADAM-RAG] COMPLEX flow: Using Speculative RAG to draft scenarios.")
        from speculative_rag import SpeculativeRAG
        spec_rag = SpeculativeRAG(llm_router=exec_llm, retriever=retriever)
        spec_result = spec_rag.draft_and_verify(query=prompt, num_drafts=3)
        best_scenario = spec_result.get("best_draft", "")
        if best_scenario:
            prompt += f"\n\n[Speculative RAG Best Draft Scenario]:\n{best_scenario}"
            
        logger.info("[MADAM-RAG] COMPLEX flow: Using FLARE to verify reasoning before final JSON generation.")
        flare_context = f"Tech: {tech}\nSent: {sent}\nNews: {news}\nBull: {bull}\nBear: {bear}"
        flare_query = f"Synthesize a trading decision analysis for {pair} considering all evidence."
        adaptive_router.flare.retriever = retriever
        flare_res = adaptive_router.flare.generate_with_active_retrieval(query=flare_query, context=flare_context)
        reasoning_draft = flare_res.get("analysis", "")
        
        json_prompt = prompt + f"\n\nUse this verified FLARE reasoning as a base: {reasoning_draft}"
        response = exec_llm.invoke([SystemMessage(content="You are the Master Coordinator."), HumanMessage(content=json_prompt)])
    else:
        response = exec_llm.invoke([SystemMessage(content="You are the Master Coordinator."), HumanMessage(content=prompt)])

    content_raw = response.content
    if isinstance(content_raw, list):
        content_raw = " ".join([b.get("text", "") for b in content_raw if "text" in b])
        
    raw_content = content_raw.replace("```json", "").replace("```", "").strip()
    
    try:
        data = json.loads(raw_content)
        signal = data.get("signal", "NEUTRAL")
        conf = float(data.get("confidence", 0.0))
        reason = data.get("reasoning", "")
    except Exception as e:
        logger.error(f"Failed to parse Coordinator JSON: {e}")
        signal, conf, reason = "NEUTRAL", 0.0, "JSON Parsing Failure."
        
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

def get_trading_signal(pair: str) -> dict:
    """Entry point for Freqtrade to request a trading decision from the Analyst Team."""
    logger.info(f"Initiating Multi-Agent Analyst Team for {pair}...")
    
    # Phase 9: Semantic Cache
    from semantic_cache import SemanticCache
    from self_rag import SelfRAG
    import json
    
    cache = SemanticCache()
    query_str = f"trading signal analysis for {pair}"
    
    # a. Check Cache
    cached_response_str = cache.get(query=query_str, pair=pair)
    if cached_response_str:
        try:
            logger.info("[Semantic Cache] Reusing cached decision for pair.")
            return json.loads(cached_response_str)
        except Exception as e:
            logger.error(f"Failed to parse cached response: {e}")

    self_rag = SelfRAG()
    
    # b. Rule-based Retrieval Gating Check
    # Trade-First: NEVER block a signal. If retrieval is skipped, proceed with reduced confidence.
    # The position sizer will modulate size accordingly — confidence controls SIZE, not PERMISSION.
    retrieval_gated = not self_rag.should_retrieve(query_str, {})
    if retrieval_gated:
        logger.info(f"[RAG] Retrieval gated for '{query_str[:40]}...' — proceeding with LLM-only analysis at reduced confidence.")
    
    # Initialize state
    inputs = {
        "pair": pair,
        "documents": [],
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
        for output in rag_bot.stream(inputs):
            for key, value in output.items():
                final_output = value
                
        signal = final_output.get("signal", "NEUTRAL") if final_output else "NEUTRAL"
        confidence = final_output.get("confidence", 0.0) if final_output else 0.0
        reasoning = final_output.get("reasoning", "") if final_output else ""
        
        # Self-RAG Critique
        critique = self_rag.self_critique(
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
    
    # e. Put directly to Semantic Cache
    cache.put(query=query_str, response=json.dumps(result_dict), pair=pair)
    
    return result_dict

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Allow CLI execution for testing or Freqtrade subprocess call
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", type=str, default="BTC/USDT", help="Pair to analyze")
    args = parser.parse_args()
    
    result = get_trading_signal(args.pair)
    # Print clean JSON to stdout so subprocess can read it
    print("\n--- JSON OUTPUT ---")
    print(json.dumps(result))
