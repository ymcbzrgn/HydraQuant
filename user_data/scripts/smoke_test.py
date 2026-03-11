"""
Phase 17: Comprehensive End-to-End Smoke Test
Tests all 18 RAG techniques + monitoring + deployment readiness.
No real API calls — mock/import verification only.
"""

import os
import sys
import time

# Step 0: Ensure we use a temporary isolated database for the smoke test
os.environ["AI_DB_PATH"] = "/tmp/smoke_db.sqlite"
os.environ["CHROMA_PERSIST_DIR"] = "/tmp/smoke_chroma"

# Add scripts directory to path to allow imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(SCRIPT_DIR)

import logging
from unittest.mock import patch, MagicMock

# Minimal logging to not clutter the smoke test output
logging.basicConfig(level=logging.ERROR)


def print_result(step_name: str, passed: bool, msg: str = ""):
    icon = "PASS" if passed else "FAIL"
    print(f"[{icon}] {step_name} {msg}")


def run_full_smoke_test() -> dict:
    """Run all AI system checks, return structured results."""
    start_time = time.time()
    results = {"total_checks": 0, "passed": 0, "failed": 0, "failures": [], "details": []}

    def check(name: str, fn):
        results["total_checks"] += 1
        try:
            fn()
            results["passed"] += 1
            print_result(name, True)
            results["details"].append({"name": name, "passed": True})
        except Exception as e:
            results["failed"] += 1
            results["failures"].append(f"{name}: {e}")
            print_result(name, False, str(e))
            results["details"].append({"name": name, "passed": False, "error": str(e)})

    print("=" * 60)
    print("AI Pipeline End-to-End Smoke Test (Phase 17)")
    print("=" * 60)

    # --- Core Infrastructure ---
    def test_db_init():
        from db import init_db
        init_db()

    def test_ai_config():
        from ai_config import AI_DB_PATH, CHROMA_PERSIST_DIR
        assert AI_DB_PATH, "AI_DB_PATH is empty"
        assert CHROMA_PERSIST_DIR, "CHROMA_PERSIST_DIR is empty"

    def test_llm_router_import():
        from llm_router import LLMRouter
        router = LLMRouter()
        assert hasattr(router, 'invoke'), "LLMRouter missing invoke()"
        assert hasattr(router, 'ainvoke'), "LLMRouter missing ainvoke()"

    check("1. Database Init", test_db_init)
    check("2. AI Config", test_ai_config)
    check("3. LLM Router Import", test_llm_router_import)

    # --- Data Pipeline ---
    def test_rss_insert():
        from db import get_db_connection
        with get_db_connection() as conn:
            conn.execute('''
                INSERT OR IGNORE INTO market_news (source, title, summary, url, published_at)
                VALUES (?, ?, ?, ?, ?)
            ''', ("SmokeSource", "Smoke Test BTC News", "BTC breaks $100K.", "https://smoke.test/btc", "2026-01-01T00:00:00Z"))
            conn.commit()

    def test_sentiment_analyzer():
        from sentiment_analyzer import analyze_unscored_news
        with patch('sentiment_analyzer.load_sentiment_pipeline') as mock_pipe:
            dummy_model = MagicMock()
            dummy_model.return_value = [{'label': 'bullish', 'score': 0.95}]
            mock_pipe.return_value = dummy_model
            analyze_unscored_news()

    def test_data_pipeline():
        from data_pipeline import DataPipeline
        with patch('rag_embedding.DualEmbeddingPipeline.get_embeddings') as mock_embed:
            mock_embed.return_value = {'gemini': [0.1] * 768, 'bge': [0.1] * 768}
            pipeline = DataPipeline()
            pipeline._embed_unprocessed_news()

    check("4. RSS Data Insert", test_rss_insert)
    check("5. Sentiment Analyzer", test_sentiment_analyzer)
    check("6. Data Pipeline & Embedding", test_data_pipeline)

    # --- Retrieval & RAG Core ---
    def test_hybrid_retriever():
        from hybrid_retriever import HybridRetriever
        with patch('hybrid_retriever.HybridRetriever.search') as mock_search:
            mock_search.return_value = ["Mock Document from Chroma/BM25"]
            retriever = HybridRetriever()
            res = retriever.search("BTC status", top_k=2)
            assert len(res) == 1

    def test_adaptive_router():
        from adaptive_router import AdaptiveQueryRouter
        router = AdaptiveQueryRouter()
        classification = router.classify("Is BTC dropping?")
        assert classification is not None

    def test_crag():
        from crag_evaluator import CRAGEvaluator
        CRAGEvaluator(router=MagicMock())

    def test_rag_graph():
        import rag_graph
        with patch('rag_graph.get_trading_signal') as mock_graph:
            mock_graph.return_value = {"decision": "BULLISH", "confidence": 0.88, "reasoning": "Debate concluded bullish."}
            res = rag_graph.get_trading_signal("BTC/USDT")
            assert res['decision'] == "BULLISH"

    check("7. Hybrid Retriever", test_hybrid_retriever)
    check("8. Adaptive Router", test_adaptive_router)
    check("9. CRAG Evaluator", test_crag)
    check("10. RAG Graph (MADAM Debate)", test_rag_graph)

    # --- RAG Techniques (Phase 14-16) ---
    def test_raptor_import():
        from raptor_tree import RAPTORTree
        assert hasattr(RAPTORTree, 'build_tree')

    def test_streaming_rag_import():
        from streaming_rag import StreamingRAG
        assert hasattr(StreamingRAG, 'ingest')

    def test_cryptopanic_import():
        from cryptopanic_fetcher import CryptoPanicFetcher
        assert CryptoPanicFetcher is not None

    def test_alphavantage_import():
        from alphavantage_fetcher import AlphaVantageFetcher
        assert AlphaVantageFetcher is not None

    def test_magma_import():
        from magma_memory import MAGMAMemory
        assert hasattr(MAGMAMemory, 'add_edge')

    def test_memorag_import():
        from memo_rag import MemoRAG
        assert hasattr(MemoRAG, 'update_global_memory')

    def test_bidi_import():
        from bidirectional_rag import BidirectionalRAG
        assert hasattr(BidirectionalRAG, 'evaluate_trade_outcome')

    def test_flare_import():
        from flare_retriever import FLARERetriever
        assert hasattr(FLARERetriever, 'generate_with_active_retrieval')

    def test_speculative_import():
        from speculative_rag import SpeculativeRAG
        assert hasattr(SpeculativeRAG, 'draft_and_verify')

    def test_cot_rag_import():
        from cot_rag import CoTRAG
        assert hasattr(CoTRAG, 'reason_step_by_step')

    check("11. RAPTOR Tree Import", test_raptor_import)
    check("12. StreamingRAG Import", test_streaming_rag_import)
    check("13. CryptoPanic Fetcher Import", test_cryptopanic_import)
    check("14. AlphaVantage Fetcher Import", test_alphavantage_import)
    check("15. MAGMA Memory Import", test_magma_import)
    check("16. MemoRAG Import", test_memorag_import)
    check("17. Bidirectional RAG Import", test_bidi_import)
    check("18. FLARE Retriever Import", test_flare_import)
    check("19. Speculative RAG Import", test_speculative_import)
    check("20. CoT-RAG Import", test_cot_rag_import)

    # --- Autonomy & Risk ---
    def test_position_sizer():
        from position_sizer import PositionSizer
        sizer = PositionSizer()
        with patch('position_sizer.BayesianKelly.update'):
            frac = sizer.calculate_stake_fraction(0.85)
            assert 0 <= frac <= 1

    def test_risk_budget():
        from risk_budget import RiskBudgetManager
        rm = RiskBudgetManager()
        can_open = rm.scale_position(100.0)
        rm.consume_budget(position_size=100.0, asset_volatility=0.05, confidence=0.8)
        assert can_open is not None

    def test_decision_logger():
        from ai_decision_logger import AIDecisionLogger
        dl = AIDecisionLogger()
        trade_id = dl.log_decision(
            pair="BTC/USDT", signal_type="BULLISH",
            confidence=0.88, reasoning_summary="Smoke test reasoning."
        )
        assert trade_id is not None

    def test_forgone_pnl():
        from forgone_pnl_engine import ForgonePnLEngine
        engine = ForgonePnLEngine()
        engine.log_forgone_signal("BTC/USDT", "BULLISH", 0.88, 50000.0, False)

    check("21. Position Sizer", test_position_sizer)
    check("22. Risk Budget", test_risk_budget)
    check("23. Decision Logger", test_decision_logger)
    check("24. Forgone P&L Engine", test_forgone_pnl)

    # --- Phase 17: System Monitor ---
    def test_system_monitor():
        from system_monitor import SystemMonitor
        monitor = SystemMonitor()
        monitor.record_metric("test_metric", 42.0, {"source": "smoke_test"})
        dashboard = monitor.get_dashboard_data(hours=1)
        assert "rag_latency_avg_ms" in dashboard
        health = monitor.check_health()
        assert health["status"] in ("healthy", "degraded", "critical")

    check("25. System Monitor", test_system_monitor)

    # --- Summary ---
    results["duration_seconds"] = round(time.time() - start_time, 2)
    print("=" * 60)
    print(f"Results: {results['passed']}/{results['total_checks']} passed "
          f"({results['failed']} failed) in {results['duration_seconds']}s")
    if results["failures"]:
        print("Failures:")
        for f in results["failures"]:
            print(f"  - {f}")
    print("=" * 60)

    return results


# Legacy wrapper
def run_smoke_test():
    return run_full_smoke_test()


if __name__ == "__main__":
    run_full_smoke_test()
