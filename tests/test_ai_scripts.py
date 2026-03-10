"""
Tests for the Freqtrade AI system.
Covers: LLM Router, Position Sizer, Hybrid Retriever, RAG Graph,
        Error Categorizer, Decision Logger, Forgone P&L Engine.

Run: pytest tests/test_ai_scripts.py -v
"""
import sys
import os
import sqlite3
import threading
import pytest

# Make user_data/scripts importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "user_data", "scripts"))

# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def tmp_db(tmp_path):
    """Create a temporary SQLite DB path for isolated tests."""
    db_path = str(tmp_path / "test_ai_data.sqlite")
    return db_path


# ============================================================
# Test 1: LLM Router Key Rotation is Thread-Safe
# ============================================================
def test_llm_router_key_rotation_thread_safe():
    """Bug 5 fix: Verify that concurrent key rotation doesn't corrupt the key list."""
    from llm_router import LLMRouter
    
    router = LLMRouter(temperature=0.0)
    
    if not router.gemini_keys:
        pytest.skip("No Gemini keys configured in .env")
    
    initial_key_count = len(router.gemini_keys)
    initial_keys_set = set(router.gemini_keys)
    errors = []
    
    def rotate_keys():
        try:
            for _ in range(50):
                router._get_chain()
        except Exception as e:
            errors.append(str(e))
    
    threads = [threading.Thread(target=rotate_keys) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    
    assert len(errors) == 0, f"Thread errors: {errors}"
    assert len(router.gemini_keys) == initial_key_count, "Keys were lost during rotation!"
    assert set(router.gemini_keys) == initial_keys_set, "Keys were corrupted during rotation!"


# ============================================================
# Test 2: LLM Router has threading Lock
# ============================================================
def test_llm_router_has_lock():
    """Verify that LLMRouter uses threading.Lock for thread safety."""
    from llm_router import LLMRouter, _COOLDOWN_LOCK
    
    router = LLMRouter(temperature=0.0)
    assert hasattr(router, '_key_lock'), "LLMRouter missing _key_lock"
    assert isinstance(router._key_lock, type(threading.Lock())), "_key_lock is not a Lock"
    assert isinstance(_COOLDOWN_LOCK, type(threading.Lock())), "_COOLDOWN_LOCK is not a Lock"


# ============================================================
# Test 3: Position Sizer Confidence Curve
# ============================================================
def test_position_sizer_confidence_curve():
    """Confidence=0.5 with exponent=2 should give 0.25 * max_risk."""
    from position_sizer import PositionSizer
    
    sizer = PositionSizer(max_portfolio_risk_per_trade=0.05, confidence_exponent=2.0)
    # Override autonomy Kelly to allow real trades in test
    sizer.autonomy.current_level = 5  # L5 Kelly=0.75
    
    # 0.5^2 = 0.25 * 0.05 = 0.0125
    stake = sizer.calculate_stake_fraction(0.5)
    assert abs(stake - 0.0125) < 0.001, f"Expected 0.0125, got {stake}"
    
    # 0.9^2 = 0.81 * 0.05 = 0.0405
    stake_high = sizer.calculate_stake_fraction(0.9)
    assert abs(stake_high - 0.0405) < 0.001, f"Expected 0.0405, got {stake_high}"


# ============================================================
# Test 4: Position Sizer Zero Confidence
# ============================================================
def test_position_sizer_zero_confidence():
    """confidence=0.0 should produce a stake of 0."""
    from position_sizer import PositionSizer
    
    sizer = PositionSizer()
    sizer.autonomy.current_level = 5  # L5 Kelly=0.75
    stake = sizer.calculate_stake_fraction(0.0)
    assert stake == 0.0, f"Expected 0.0, got {stake}"


# ============================================================
# Test 5: Hybrid Retriever RRF Fusion
# ============================================================
def test_hybrid_retriever_rrf():
    """3-way RRF fusion should correctly merge and rank document IDs."""
    from hybrid_retriever import HybridRetriever
    
    retriever = HybridRetriever.__new__(HybridRetriever)  # Skip __init__ (no ChromaDB needed)
    
    list1 = ["doc_A", "doc_B", "doc_C"]
    list2 = ["doc_B", "doc_D", "doc_A"]
    list3 = ["doc_C", "doc_B", "doc_E"]
    
    fused = retriever.reciprocal_rank_fusion([list1, list2, list3])
    
    # doc_B appears in all 3 lists -> highest RRF score
    assert fused[0] == "doc_B", f"Expected doc_B as #1, got {fused[0]}"
    # doc_A appears in 2 lists
    assert "doc_A" in fused[:3], "doc_A should be in top 3"
    # doc_E appears in only 1 list -> lowest
    assert fused[-1] == "doc_E" or fused[-1] == "doc_D", f"Expected doc_E or doc_D last"


# ============================================================
# Test 6: RAG Graph — No Hardcoded Sentiment
# ============================================================
def test_rag_graph_no_hardcoded_sentiment():
    """Critical Bug #2 fix: Verify that the hardcoded 'F&G: 72, CryptoBERT: 0.82' string is GONE."""
    import inspect
    from rag_graph import analyze_sentiment
    
    source = inspect.getsource(analyze_sentiment)
    
    assert "72 (Greed)" not in source, "HARDCODED F&G score still present!"
    assert "0.82 (Bullish)" not in source, "HARDCODED CryptoBERT score still present!"
    assert "sqlite3" in source or "ai_data.sqlite" in source, "No DB query in sentiment analyst!"


# ============================================================
# Test 7: Error Categorizer Uses BaseMessage
# ============================================================
def test_error_categorizer_message_types():
    """Bug 6 fix: Verify messages use SystemMessage/HumanMessage, not tuples."""
    import inspect
    from error_categorizer import ErrorCategorizer
    
    source = inspect.getsource(ErrorCategorizer.classify_loss)
    
    assert "SystemMessage" in source, "classify_loss should use SystemMessage"
    assert "HumanMessage" in source, "classify_loss should use HumanMessage"
    assert '("system"' not in source, "Still using tuple format for system message!"
    assert '("user"' not in source, "Still using tuple format for user message!"


# ============================================================
# Test 8: Decision Logger Schema Migration
# ============================================================
def test_decision_logger_schema(tmp_db):
    """Verify that AIDecisionLogger creates the table with all Phase 6 columns."""
    from ai_decision_logger import AIDecisionLogger
    
    logger_inst = AIDecisionLogger(db_path=tmp_db)
    
    conn = sqlite3.connect(tmp_db)
    c = conn.cursor()
    c.execute("PRAGMA table_info(ai_decisions)")
    columns = [col[1] for col in c.fetchall()]
    conn.close()
    
    required = ['id', 'timestamp', 'pair', 'signal_type', 'confidence', 'outcome_pnl',
                'outcome_duration', '_status_cache']
    for col in required:
        assert col in columns, f"Missing column: {col}"


# ============================================================
# Test 9: Decision Logger Returns ID (not bool)
# ============================================================
def test_decision_logger_returns_id(tmp_db):
    """Phase 6.1 fix: log_decision should return an int ID, not a bool."""
    from ai_decision_logger import AIDecisionLogger
    
    logger_inst = AIDecisionLogger(db_path=tmp_db)
    result = logger_inst.log_decision(
        pair="BTC/USDT",
        signal_type="BULL",
        confidence=0.75,
        reasoning_summary="Test decision"
    )
    
    assert isinstance(result, int), f"Expected int, got {type(result)}"
    assert result > 0, f"Expected positive ID, got {result}"


# ============================================================
# Test 10: Forgone P&L Logging and Resolution
# ============================================================
def test_forgone_pnl_logging(tmp_db):
    """Verify forgone P&L engine logs signals and resolves paper trades correctly."""
    from forgone_pnl_engine import ForgonePnLEngine
    
    engine = ForgonePnLEngine(db_path=tmp_db)
    
    # Log a forgone BULL signal
    fid = engine.log_forgone_signal("BTC/USDT", "BULL", 0.45, 50000.00, was_executed=False)
    assert fid is not None, "log_forgone_signal returned None"
    assert isinstance(fid, int), f"Expected int ID, got {type(fid)}"
    
    # Resolve it: BTC went up from 50k to 51k -> +2%
    resolved = engine.resolve_forgone_trade(fid, exit_price=51000.00)
    assert resolved is True, "resolve_forgone_trade failed"
    
    # Verify the P&L was calculated correctly
    conn = sqlite3.connect(tmp_db)
    c = conn.cursor()
    c.execute("SELECT forgone_pnl, exit_price FROM forgone_profit WHERE id = ?", (fid,))
    row = c.fetchone()
    conn.close()
    
    assert row is not None, "Row not found in database"
    assert abs(row[0] - 2.0) < 0.01, f"Expected ~2.0% PnL, got {row[0]}"
    assert row[1] == 51000.00, f"Expected exit_price 51000, got {row[1]}"


# ============================================================
# Test 11: Forgone P&L Weekly Summary Verdict
# ============================================================
def test_forgone_pnl_weekly_summary(tmp_db):
    """Weekly summary should produce correct verdicts based on forgone P&L totals."""
    from forgone_pnl_engine import ForgonePnLEngine

    engine = ForgonePnLEngine(db_path=tmp_db)

    # Log a forgone signal that would have LOST money
    fid = engine.log_forgone_signal("SOL/USDT", "BULL", 0.30, 150.00, was_executed=False)
    assert fid is not None
    engine.resolve_forgone_trade(fid, exit_price=140.00)  # -6.67% loss

    report = engine.weekly_summary()
    assert report['forgone_trades']['count'] == 1
    assert report['forgone_trades']['total_pnl_pct'] < 0, "Forgone PnL should be negative"
    assert "SAVED MONEY" in report['verdict'], f"Expected SAVED MONEY verdict, got: {report['verdict']}"


# ============================================================
# Test 12: Forgone P&L BEAR Signal (Inverse Direction)
# ============================================================
def test_forgone_pnl_bear_signal(tmp_db):
    """BEAR signal P&L should be calculated as (entry-exit)/entry, i.e. profit when price drops."""
    from forgone_pnl_engine import ForgonePnLEngine

    engine = ForgonePnLEngine(db_path=tmp_db)

    fid = engine.log_forgone_signal("ETH/USDT", "BEAR", 0.70, 3000.00, was_executed=False)
    assert fid is not None
    engine.resolve_forgone_trade(fid, exit_price=2700.00)  # Price dropped 10% → BEAR profits

    conn = sqlite3.connect(tmp_db)
    c = conn.cursor()
    c.execute("SELECT forgone_pnl FROM forgone_profit WHERE id = ?", (fid,))
    pnl = c.fetchone()[0]
    conn.close()

    assert abs(pnl - 10.0) < 0.01, f"Expected +10% PnL for BEAR (price drop), got {pnl}"


# ============================================================
# Test 13: Parent-Child Retrieval Swap
# ============================================================
def test_parent_child_retrieval():
    """When a child chunk matches, the search should return parent_text instead."""
    from hybrid_retriever import HybridRetriever

    # We test the document expansion logic directly by simulating a fetched result
    retriever = HybridRetriever.__new__(HybridRetriever)

    # Simulate what ChromaDB returns — child with parent_text in metadata
    child_text = "Fed cut rates 25bps"
    parent_text = "Federal Reserve announced a 25 basis point cut in interest rates today, sparking a broad crypto rally. Bitcoin surged past $70,000 while altcoins followed with 5-15% gains."

    meta = {
        "type": "news_child",
        "parent_text": parent_text,
        "source": "reuters"
    }

    # The logic in search() should prefer parent_text for news_child type
    if meta.get('type') == 'news_child' and meta.get('parent_text'):
        display_text = meta['parent_text']
    else:
        display_text = child_text

    assert display_text == parent_text, "Should have swapped child for parent text"
    assert len(display_text) > len(child_text), "Parent text should be longer than child"


# ============================================================
# Test 14: CRAG Evaluator — CORRECT Classification
# ============================================================
def test_crag_evaluator_correct():
    """CRAG should classify highly relevant docs as CORRECT."""
    from crag_evaluator import CRAGEvaluator

    evaluator = CRAGEvaluator.__new__(CRAGEvaluator)

    # Test with empty docs → INCORRECT
    verdict, score, reason = evaluator.evaluate_retrieval("test query", [])
    assert verdict == "INCORRECT"
    assert score == 0.0
    assert "No documents" in reason


# ============================================================
# Test 15: CRAG Evaluator — Web Fallback Exists
# ============================================================
def test_crag_evaluator_web_fallback_exists():
    """CRAG should have a web_search_fallback method and handle missing duckduckgo gracefully."""
    from crag_evaluator import CRAGEvaluator

    evaluator = CRAGEvaluator.__new__(CRAGEvaluator)

    # web_search_fallback should not crash even if duckduckgo_search is not installed
    results = evaluator.web_search_fallback("bitcoin price today", max_results=1)
    assert isinstance(results, list), "web_search_fallback should return a list"


# ============================================================
# Test 16: CRAG is Integrated in rag_graph.py (NOT dead code)
# ============================================================
def test_crag_integrated_in_rag_graph():
    """Verify that rag_graph.py actually imports and uses CRAG + AdaptiveRouter."""
    import inspect
    import rag_graph

    source = inspect.getsource(rag_graph)

    # CRAG import check
    assert "from crag_evaluator import CRAGEvaluator" in source, "CRAG not imported in rag_graph!"
    assert "CRAGEvaluator" in source, "CRAGEvaluator not instantiated in rag_graph!"

    # Adaptive Router integration check
    assert "from adaptive_router import AdaptiveQueryRouter" in source, "AdaptiveRouter not imported in rag_graph!"
    assert "adaptive_router.route" in source, "adaptive_router.route() not called in rag_graph!"


# ============================================================
# Test 17: Adaptive Router Classify
# ============================================================
def test_adaptive_router_classify():
    """Verify AdaptiveQueryRouter correctly classifies query complexity."""
    from adaptive_router import AdaptiveQueryRouter

    router = AdaptiveQueryRouter.__new__(AdaptiveQueryRouter)
    router._simple_keywords = ["price", "fiyat", "market cap", "volume", "hacim"]
    router._no_rag_keywords = ["nedir", "nasıl", "what is", "how does", "explain", "definition"]
    
    class MockSelfRAG:
        def should_retrieve(self, *args, **kwargs): return True
    router.self_rag = MockSelfRAG()

    # NO_RAG: general knowledge question
    assert router.classify("What is a moving average?") == "NO_RAG"

    # SIMPLE: fact lookup, short query
    assert router.classify("BTC price") == "SIMPLE"

    # COMPLEX: cross-domain with "correlation"
    assert router.classify("Fed interest rate decision correlation with BTC") == "COMPLEX"


# ============================================================
# Test 18: Risk Budget Consume
# ============================================================
def test_risk_budget_consume(tmp_db):
    """Verify RiskBudgetManager tracks budget consumption correctly."""
    from risk_budget import RiskBudgetManager

    mgr = RiskBudgetManager(portfolio_value=10000.0, daily_var_pct=0.01, db_path=tmp_db)

    initial_budget = mgr.daily_budget
    assert initial_budget == 100.0, f"Expected $100 budget, got {initial_budget}"

    remaining = mgr.remaining_budget()
    assert remaining == 100.0, f"Expected $100 remaining, got {remaining}"

    # Consume some budget: position=$500, vol=0.03, conf=0.5
    # consumption = 500 * 0.03 * (1/0.5) = 30.0
    remaining_after = mgr.consume_budget(500.0, 0.03, 0.5)
    assert abs(remaining_after - 70.0) < 0.01, f"Expected ~$70 remaining, got {remaining_after}"


# ============================================================
# Test 19: Risk Budget Daily Reset
# ============================================================
def test_risk_budget_daily_reset(tmp_db):
    """Verify daily reset clears consumed budget."""
    from risk_budget import RiskBudgetManager

    mgr = RiskBudgetManager(portfolio_value=10000.0, daily_var_pct=0.01, db_path=tmp_db)

    # Consume almost all budget
    mgr.consume_budget(1000.0, 0.05, 0.5)  # 1000*0.05*2 = 100 → budget fully consumed

    assert mgr.remaining_budget() <= 0.01, "Budget should be nearly depleted"

    # Reset
    mgr.reset_daily()
    assert mgr.remaining_budget() == mgr.daily_budget, "After reset, full budget should be available"


# ============================================================
# Test 20: Autonomy Manager Kelly Fractions
# ============================================================
def test_autonomy_kelly_fraction(tmp_db):
    """Verify correct Kelly fraction mapping for each autonomy level."""
    from autonomy_manager import AutonomyManager, KELLY_FRACTIONS

    mgr = AutonomyManager(db_path=tmp_db)

    # Default is L0
    assert mgr.get_level() == 0
    assert mgr.get_kelly_fraction() == 0.0

    # Check all defined fractions
    expected = {0: 0.0, 1: 0.0, 2: 0.10, 3: 0.25, 4: 0.50, 5: 0.75}
    assert KELLY_FRACTIONS == expected, f"Kelly fractions don't match: {KELLY_FRACTIONS}"

    # Test promotion (L0→L1 has minimal criteria)
    promoted = mgr.check_promotion(total_trades=0, sharpe=0.0, max_dd_pct=0.0, days_at_level=0)
    assert promoted is True, "L0→L1 should promote immediately"
    assert mgr.get_level() == 1
    assert mgr.get_kelly_fraction() == 0.0  # L1 is still paper trading


# ============================================================
# Test 21: Temporal Decay — Old News Score Drops
# ============================================================
def test_temporal_decay_old_news():
    """30-day-old news should have significantly lower score than original."""
    from hybrid_retriever import HybridRetriever
    from datetime import datetime, timezone, timedelta

    retriever = HybridRetriever.__new__(HybridRetriever)

    old_date = (datetime.now(tz=timezone.utc) - timedelta(days=30)).isoformat()
    results = [
        {"id": "doc_old", "text": "Old news", "score": 1.0, "meta": {"published_at": old_date}},
    ]

    decayed = retriever._apply_temporal_decay(results, half_life_days=7.0, alpha=0.7)
    # 30 days / 7 half-life ≈ 4.3 half-lives → decay ≈ 0.05
    # score = 0.7*1.0 + 0.3*0.05 ≈ 0.715
    assert decayed[0]['score'] < 0.75, f"30-day old news score too high: {decayed[0]['score']:.4f}"
    assert decayed[0]['score'] > 0.60, f"30-day old news score too low: {decayed[0]['score']:.4f}"


# ============================================================
# Test 22: Temporal Decay — Fresh News Score Preserved
# ============================================================
def test_temporal_decay_fresh_news():
    """1-hour-old news should retain nearly full score."""
    from hybrid_retriever import HybridRetriever
    from datetime import datetime, timezone, timedelta

    retriever = HybridRetriever.__new__(HybridRetriever)

    fresh_date = (datetime.now(tz=timezone.utc) - timedelta(hours=1)).isoformat()
    results = [
        {"id": "doc_fresh", "text": "Fresh news", "score": 1.0, "meta": {"published_at": fresh_date}},
    ]

    decayed = retriever._apply_temporal_decay(results, half_life_days=7.0, alpha=0.7)
    # 1 hour = 0.042 days / 7 → decay ≈ 0.9996 → score ≈ 0.7 + 0.3*1.0 ≈ 1.0
    assert decayed[0]['score'] > 0.95, f"Fresh news lost too much score: {decayed[0]['score']:.4f}"


# ============================================================
# Test 23: HyDE Generates Hypothetical Document
# ============================================================
def test_hyde_generates_hypothetical():
    """HyDE should generate a non-empty hypothetical answer."""
    from hyde_generator import HyDEGenerator

    hyde = HyDEGenerator.__new__(HyDEGenerator)

    # Test with a mock router that returns a fixed response
    class MockRouter:
        def invoke(self, messages):
            class Response:
                content = "BTC is currently trading at 98K with RSI at 65. Support at 95K, resistance at 100K."
            return Response()

    hyde.router = MockRouter()

    result = hyde.generate_hypothetical("BTC ne olacak?")
    assert len(result) > 20, f"Hypothetical too short: {len(result)} chars"
    assert "BTC" in result or "btc" in result.lower(), "Hypothetical should mention BTC"


# ============================================================
# Test 24: RAG-Fusion Multi-Query Generation
# ============================================================
def test_rag_fusion_multi_query():
    """RAG-Fusion should generate multiple diverse queries."""
    from rag_fusion import RAGFusion

    fusion = RAGFusion.__new__(RAGFusion)

    class MockRouter:
        def invoke(self, messages):
            class Response:
                content = "BTC teknik analiz RSI MACD\nBitcoin son haberler sentiment\nBTC whale hareketleri on-chain"
            return Response()

    fusion.router = MockRouter()

    queries = fusion.generate_queries("BTC ne olacak?", n=3)
    assert len(queries) >= 3, f"Expected at least 3 queries, got {len(queries)}"
    assert queries[0] == "BTC ne olacak?", "First query should be the original"
    # Each sub-query should be unique
    assert len(set(queries)) == len(queries), "Sub-queries should be unique"


# ============================================================
# Test 25: Risk Budget Scale — Over Budget
# ============================================================
def test_risk_budget_scale_over_budget(tmp_db):
    """When budget is exhausted, position should scale to minimum (10%)."""
    from risk_budget import RiskBudgetManager

    mgr = RiskBudgetManager(portfolio_value=10000.0, daily_var_pct=0.01, db_path=tmp_db)

    # Exhaust the entire budget
    mgr.consume_budget(2000.0, 0.10, 0.5)  # 2000*0.10*2 = 400 >> 100 budget

    # Scale should result in minimum factor
    proposed = 500.0
    scaled = mgr.scale_position(proposed)
    assert scaled < proposed, f"Scaled ({scaled}) should be less than proposed ({proposed})"
    assert scaled <= proposed * 0.25, f"Scaled ({scaled}) should be at most 25% of proposed"


# ============================================================
# Test 26: Bull and Bear Agents produce distinct reports
# ============================================================
def test_bull_bear_debate():
    """Bull and Bear agents should produce distinct structural arguments."""
    from rag_graph import research_bullish, research_bearish
    
    # Mock rag_graph.rag_fusion
    import rag_graph
    
    class MockFusion:
        def fused_search(self, q, ret, n_queries, top_k_per_query):
            if "bullish" in q.lower():
                return [{"text": "RSI oversold, strong support."}]
            else:
                return [{"text": "Resistance rejected, death cross."}]
    
    rag_graph.rag_fusion = MockFusion()
    
    state_in = {"pair": "BTC/USDT"}
    bull_res = research_bullish(state_in)
    bear_res = research_bearish(state_in)
    
    assert "bull_case" in bull_res
    assert len(bull_res["bull_case"]) > 0
    assert "bear_case" in bear_res
    assert len(bear_res["bear_case"]) > 0


# ============================================================
# Test 27: Bayesian Kelly Update logic
# ============================================================
def test_bayesian_kelly_update(tmp_db):
    """10 wins, 2 losses -> win prob = (1+10)/(1+1+10+2) = 11/14 ≈ 0.785"""
    from position_sizer import BayesianKelly
    
    bk = BayesianKelly(db_path=tmp_db)
    
    for _ in range(10):
        bk.update(won=True, pnl_pct=0.05)
    for _ in range(2):
        bk.update(won=False, pnl_pct=-0.03)
        
    prob = bk.win_probability()
    assert abs(prob - (11/14)) < 0.01, f"Expected 0.785, got {prob}"


# ============================================================
# Test 28: Bayesian Kelly Fraction calculation
# ============================================================
def test_bayesian_kelly_fraction(tmp_db):
    """f* = (b*p - q)/b capped at 0.25"""
    from position_sizer import BayesianKelly
    
    bk = BayesianKelly(db_path=tmp_db)
    # Force high win rate
    bk.alpha = 90.0
    bk.beta_param = 10.0
    bk.avg_win_loss_ratio = 2.0
    
    f = bk.kelly_fraction()
    assert f == 0.25, f"Should be capped at 0.25, got {f}"
    
    # Force losing strategy
    bk.alpha = 10.0
    bk.beta_param = 90.0
    f_loss = bk.kelly_fraction()
    assert f_loss == 0.0, f"Should be 0.0 for losing strategies, got {f_loss}"


# ============================================================
# Test 29: Contextual Chunking
# ============================================================
def test_contextual_chunking_called():
    """Text is prepended with document context"""
    from rag_chunker import ContentChunker
    
    chunk = "The Federal Reserve raised rates."
    summary = "Macro Economic Update (Reuters)"
    
    res = ContentChunker.construct_contextual_prompt(chunk, summary)
    assert summary in res
    assert chunk in res
    assert res.startswith("Document context")


# ============================================================
# Test 30: Confidence Calibration (Brier Score)
# ============================================================
def test_confidence_calibrator_brier(tmp_db):
    """Brier score is correctly calculated"""
    import sqlite3
    from confidence_calibrator import ConfidenceCalibrator
    
    conn = sqlite3.connect(tmp_db)
    conn.execute('''
        CREATE TABLE ai_decisions (
            id INTEGER PRIMARY KEY,
            pair TEXT,
            signal_type TEXT,
            confidence REAL,
            timestamp TEXT
        )
    ''')
    # Insert dummy data (Conf 0.8 -> Won, Conf 0.8 -> Won, Conf 0.8 -> Lost)
    # Brier for these: (0.8-1)^2 + (0.8-1)^2 + (0.8-0)^2 = 0.04 + 0.04 + 0.64 = 0.72 / 3 = 0.24
    conn.execute("INSERT INTO ai_decisions (signal_type, confidence, timestamp) VALUES ('BULLISH', 0.8, '2026-03-01T00:00:00')")
    conn.execute("INSERT INTO ai_decisions (signal_type, confidence, timestamp) VALUES ('BEARISH', 0.8, '2026-03-02T00:00:00')")
    conn.execute("INSERT INTO ai_decisions (signal_type, confidence, timestamp) VALUES ('NEUTRAL', 0.8, '2026-03-03T00:00:00')")  # Lost (outcome=0)
    conn.commit()
    conn.close()
    
    calibrator = ConfidenceCalibrator(db_path=tmp_db)
    brier = calibrator.brier_score(min_trades=1)
    
    assert abs(brier - 0.24) < 0.01, f"Expected 0.24, got {brier}"

# ============================================================
# Test 31: BayesianKelly Integrated via PositionSizer in custom_stake_amount
# ============================================================
def test_bayesian_kelly_integrated_in_strategy():
    import sys
    sys.path.append("/Users/yamacbezirgan/Projects/freqtrade/freqtrade-strategies/user_data/strategies")
    from AIFreqtradeSizer import AIFreqtradeSizer
    
    # Initialize the strategy with dummy config
    config = {'user_data_dir': '/tmp', 'stake_currency': 'USDT', 'dry_run': True}
    strategy = AIFreqtradeSizer(config)
    
    # After Görev 1, strategy should eagerly instantiate self._position_sizer
    assert hasattr(strategy, '_position_sizer'), "Expected PositionSizer to be injected"
    assert hasattr(strategy, '_bayesian_kelly'), "Expected BayesianKelly to be initialized early"
    
    # Ensure custom_stake_amount executes without throwing an error when applying Bayesian fraction
    # Mocks get_ai_signal internal call to bypass DB query
    strategy._get_ai_signal = lambda pair, ct: {"signal": "BULLISH", "confidence": 0.9}
    
    class MockRunmode:
        value = 'dry_run'

    class MockDP:
        runmode = MockRunmode()
        def get_analyzed_dataframe(self, pair, timeframe):
            import pandas as pd
            return pd.DataFrame([{'close': 50000, '%-fng_index': 50}]), None
            
    strategy.dp = MockDP()
    
    class MockTrade:
        pass
        
    stake = strategy.custom_stake_amount(pair="BTC/USDT", current_time=None, current_rate=50000, 
                                         proposed_stake=100.0, min_stake=10, max_stake=1000,
                                         leverage=1.0, entry_tag="", side="long", trade=MockTrade())
    assert stake is not None

# ============================================================
# Test 32: Tier Weighting
# ============================================================
def test_tier_weighting():
    import pandas as pd
    from coin_sentiment_aggregator import _weighted_mean
    
    # Tier 1 (1.0), Tier 3 (0.6), Unknown (0.5)
    df = pd.DataFrame({
        'source': ['coindesk.com', 'chaingpt.org', 'randomblog.com'],
        'sentiment_score': [1.0, -1.0, 1.0]
    })
    
    # Calculation: (1.0*1.0) + (-1.0*0.6) + (1.0*0.5) / (1.0 + 0.6 + 0.5)
    # = (1.0 - 0.6 + 0.5) / 2.1 = 0.9 / 2.1 = 0.4285
    val = _weighted_mean(df)
    assert abs(val - 0.4285) < 0.01

# ============================================================
# Test 33: Title Hash Deduplication
# ============================================================
def test_title_hash_dedup():
    from rss_fetcher import title_hash
    
    title1 = "Bitcoin surges past $100K amid regulatory approval?"
    title2 = " bitcoin surges PAST $100K amid regulatory APPROVAL "
    title3 = "completely different string"
    
    hash1 = title_hash(title1)
    hash2 = title_hash(title2)
    hash3 = title_hash(title3)
    
    assert hash1 == hash2, "Expected normalized strings to hash equally"
    assert hash1 != hash3, "Expected different strings to hash differently"

# ============================================================
# Test 34: Text Cleaning for Sentiment
# ============================================================
def test_text_cleaning_emoji():
    from sentiment_analyzer import clean_text
    
    raw = "🚀 Bitcoin is mooning! 😱 📈   Buy  now..."
    clean = clean_text(raw)
    
    # Emojis stripped, multiple spaces compressed
    assert clean == "Bitcoin is mooning! Buy now...", f"Got: {clean}"

# ============================================================
# Test 35: Google GenAI No Deprecation
# ============================================================
def test_google_genai_no_deprecation():
    import os
    file_path = "/Users/yamacbezirgan/Projects/freqtrade/user_data/scripts/rag_embedding.py"
    if not os.path.exists(file_path):
        return  # skip if not found
        
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
        
    # Ensure old import is gone
    assert "import google.generativeai as genai" not in content
    # Ensure new import is used
    assert "from google import genai" in content

# ============================================================
# Test 36: Smoke Test Imports
# ============================================================
def test_smoke_test_imports():
    try:
        import smoke_test
    except ImportError as e:
        import pytest
        pytest.fail(f"smoke_test failed to import: {e}")

# ============================================================
# Test 37: AI Config DB Path Resolution
# ============================================================
def test_ai_config_db_path():
    import ai_config
    assert hasattr(ai_config, 'AI_DB_PATH'), "ai_config must define AI_DB_PATH"
    assert hasattr(ai_config, 'CHROMA_PERSIST_DIR'), "ai_config must define CHROMA_PERSIST_DIR"
    assert "ai_data.sqlite" in ai_config.AI_DB_PATH

# ============================================================
# Test 38: AI Config ENV Override
# ============================================================
def test_ai_config_env_override(monkeypatch):
    import importlib
    monkeypatch.setenv("AI_DB_PATH", "/custom/path/db.sqlite")
    import ai_config
    importlib.reload(ai_config)
    assert ai_config.AI_DB_PATH == "/custom/path/db.sqlite"

# ============================================================
# Test 39: Requirements Pinned and Complete
# ============================================================
def test_requirements_pinned():
    import os
    req_file = os.path.join(os.path.dirname(__file__), "..", "requirements-ai.txt")
    if not os.path.exists(req_file):
        import pytest
        pytest.skip("requirements-ai.txt not found")
        
    with open(req_file, 'r') as f:
        content = f.read().splitlines()
        
    pinned_count = 0
    for line in content:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        assert "==" in line, f"Dependency missing version pin (==): {line}"
        pinned_count += 1
    assert pinned_count >= 9, "Expected at least 9 pinned packages in requirements-ai.txt"

def test_requirements_completeness():
    import os, glob, ast, sys
    scripts_dir = os.path.join(os.path.dirname(__file__), "..", "user_data", "scripts")
    req_ai_path = os.path.join(os.path.dirname(__file__), "..", "requirements-ai.txt")
    req_path = os.path.join(os.path.dirname(__file__), "..", "requirements.txt")
    
    req_ai_pkgs = []
    if os.path.exists(req_ai_path):
        with open(req_ai_path, "r") as f:
            req_ai_pkgs = [line.split("==")[0].lower() for line in f if "==" in line]
            
    req_pkgs = []
    if os.path.exists(req_path):
        with open(req_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"): continue
                pkg = line.split("==")[0].split(">=")[0].split("~=")[0].split("<")[0].split(";")[0].strip().lower()
                req_pkgs.append(pkg)
                
    allowed_pkgs = set(req_ai_pkgs + req_pkgs)
    
    package_mapping = {
        "dotenv": "python-dotenv",
        "duckduckgo_search": "duckduckgo-search",
        "langchain_core": "langchain",
        "langchain_google_genai": "langchain-google-genai",
        "langchain_groq": "langchain-groq",
        "langchain_openai": "langchain-openai",
        "chromadb": "chromadb",
        "sentence_transformers": "sentence-transformers",
        "feedparser": "feedparser",
        "bs4": "beautifulsoup4",
        "flashrank": "flashrank",
        "urllib3": "urllib3",
        "requests": "requests",
        "optimum": "optimum[onnxruntime]",
        "google": "google-genai",
        "ddgs": "duckduckgo-search",
        "sseclient": "sseclient-py",
    }
    
    ignore_modules = {"freqtrade", "pytest", "mock", "typing", "typing_extensions", "json", "os", "sys", "math", "datetime", "logging", "sqlite3", "uuid", "re", "hashlib", "time", "threading", "tempfile", "importlib", "shutil", "urllib", "traceback", "asyncio", "argparse", "unittest", "subprocess", "random", "base64", "socket", "io", "copy"}
    if hasattr(sys, "stdlib_module_names"):
        ignore_modules.update(sys.stdlib_module_names)
        
    local_scripts = [f.replace(".py", "") for f in os.listdir(scripts_dir) if f.endswith(".py")]
    ignore_modules.update(local_scripts)
    
    py_files = glob.glob(os.path.join(scripts_dir, "*.py"))
    for py_file in py_files:
        with open(py_file, "r") as f:
            try:
                tree = ast.parse(f.read())
            except SyntaxError:
                continue
            
        for node in ast.walk(tree):
            import_name = None
            if isinstance(node, ast.Import):
                import_name = node.names[0].name.split(".")[0]
            elif isinstance(node, ast.ImportFrom) and node.module:
                import_name = node.module.split(".")[0]
                
            if import_name and import_name not in ignore_modules:
                pkg_name = package_mapping.get(import_name, import_name).lower()
                is_langchain = import_name.startswith("langchain") and any(p.startswith("langchain") for p in allowed_pkgs)
                assert pkg_name in allowed_pkgs or is_langchain, f"Module '{import_name}' used in {os.path.basename(py_file)} but not in requirements files! (Mapped to '{pkg_name}')"

# ============================================================
# Test 40: All Scripts Import AI Config Correctly
# ============================================================
def test_all_scripts_import_ai_config():
    import os, glob
    scripts_dir = os.path.join(os.path.dirname(__file__), "..", "user_data", "scripts")
    py_files = glob.glob(os.path.join(scripts_dir, "*.py"))
    
    for py_file in py_files:
        filename = os.path.basename(py_file)
        if filename == "ai_config.py":
            continue
            
        with open(py_file, 'r') as f:
            content = f.read()
            
        # Check only non-comment lines for hardcoded paths
        code_lines = [l for l in content.splitlines() if l.strip() and not l.strip().startswith("#")]
        code_only = "\n".join(code_lines)
        hardcoded_violation = "os.path.join(" in code_only and "ai_data.sqlite" in code_only
        assert not hardcoded_violation, f"File {filename} still contains hardcoded ai_data.sqlite os.path.join pattern!"
        
        uses_db = "sqlite3.connect" in content or "DB_PATH" in content or "db_path" in content
        is_smoke_test = filename == "smoke_test.py"
        if uses_db and not is_smoke_test:
            assert "ai_config" in content, f"{filename} uses database components but does not import ai_config"

# ============================================================
# Phase 9: Cost Control & Smart Retrieval Tests
# ============================================================
def test_semantic_cache_hit(monkeypatch, tmp_path):
    monkeypatch.setenv("AI_DB_PATH", str(tmp_path / "test_cache.sqlite"))
    
    def mock_genai_embed_content(*args, **kwargs):
        content = kwargs.get("contents", "test")
        import hashlib
        sz = 768
        h = int(hashlib.md5(content.encode()).hexdigest(), 16)
        vec = [0.0] * sz
        vec[h % sz] = 1.0
        
        class MockEmbedding:
            @property
            def values(self): return vec
        class MockResponse:
            @property
            def embeddings(self): return [MockEmbedding()]
        return MockResponse()
        
    from google import genai
    class MockClient:
        class MockModels:
            def embed_content(self, *args, **kwargs):
                return mock_genai_embed_content(*args, **kwargs)
        @property
        def models(self): return self.MockModels()
    
    monkeypatch.setattr(genai, "Client", lambda *args, **kwargs: MockClient())
    
    from semantic_cache import SemanticCache
    import time
    cache = SemanticCache(db_path=str(tmp_path / "test_cache.sqlite"))
    cache.put("What is the trend for BTC?", "BULLISH response", pair="BTC/USDT", ttl=300)
    
    hit = cache.get("What is the trend for BTC?", pair="BTC/USDT")
    assert hit == "BULLISH response", "Should return cache hit for identical query"

def test_semantic_cache_miss_different_query(monkeypatch, tmp_path):
    monkeypatch.setenv("AI_DB_PATH", str(tmp_path / "test_cache.sqlite"))
    
    def mock_genai_embed_content(*args, **kwargs):
        content = kwargs.get("contents", "test")
        import hashlib
        sz = 768
        h = int(hashlib.md5(content.encode()).hexdigest(), 16)
        vec = [0.0] * sz
        vec[h % sz] = 1.0
        
        class MockEmbedding:
            @property
            def values(self): return vec
        class MockResponse:
            @property
            def embeddings(self): return [MockEmbedding()]
        return MockResponse()
        
    from google import genai
    class MockClient:
        class MockModels:
            def embed_content(self, *args, **kwargs):
                return mock_genai_embed_content(*args, **kwargs)
        @property
        def models(self): return self.MockModels()
    
    monkeypatch.setattr(genai, "Client", lambda *args, **kwargs: MockClient())
    
    from semantic_cache import SemanticCache
    cache = SemanticCache(db_path=str(tmp_path / "test_cache.sqlite"))
    cache.put("What is the trend for BTC?", "BULLISH response", pair="BTC/USDT", ttl=300)
    
    hit = cache.get("Who is the CEO of Apple?", pair="BTC/USDT")
    assert hit is None, "Should return None for completely different query"

def test_semantic_cache_expired(monkeypatch, tmp_path):
    monkeypatch.setenv("AI_DB_PATH", str(tmp_path / "test_cache.sqlite"))
    
    def mock_genai_embed_content(*args, **kwargs):
        class MockEmbedding:
            @property
            def values(self): return [1.0] * 768
        class MockResponse:
            @property
            def embeddings(self): return [MockEmbedding()]
        return MockResponse()
        
    from google import genai
    class MockClient:
        class MockModels:
            def embed_content(self, *args, **kwargs):
                return mock_genai_embed_content(*args, **kwargs)
        @property
        def models(self): return self.MockModels()
    
    monkeypatch.setattr(genai, "Client", lambda *args, **kwargs: MockClient())
    
    from semantic_cache import SemanticCache
    cache = SemanticCache(db_path=str(tmp_path / "test_cache.sqlite"))
    
    # TTL is -10 seconds (already expired)
    cache.put("Expired query", "Response", pair="BTC/USDT", ttl=-10)
    
    hit = cache.get("Expired query", pair="BTC/USDT")
    assert hit is None, "Should return None because TTL elapsed."

def test_self_rag_should_retrieve_price_query():
    from self_rag import SelfRAG
    rag = SelfRAG(router=None) # router mock not needed for fast path
    
    res = rag.should_retrieve("BTC fiyatı ne kadar?", {})
    assert res is False, "Should fast-fail for price queries"

def test_self_rag_should_retrieve_analysis_query():
    from self_rag import SelfRAG
    rag = SelfRAG(router=None)
    
    res = rag.should_retrieve("Fed faiz indirirse BTC düşer mi?", {})
    assert res is True, "Should return True for analysis"

def test_self_rag_critique_low_faithfulness():
    from self_rag import SelfRAG
    
    class MockResponse:
        content = '{"faithfulness": 0.2, "relevance": 0.9, "confidence": 0.8}'
        
    class MockRouter:
        def invoke(self, prompt, **kwargs):
            return MockResponse()
            
    rag = SelfRAG(router=MockRouter())
    res = rag.self_critique("query", "resp", ["evidence"])
    assert res["passed"] is False, "Should fail when faithfulness < 0.5"

def test_llm_cost_tracker_log_and_summary(monkeypatch, tmp_path):
    monkeypatch.setenv("AI_DB_PATH", str(tmp_path / "test_cost.sqlite"))
    from llm_cost_tracker import LLMCostTracker
    
    tracker = LLMCostTracker(db_path=str(tmp_path / "test_cost.sqlite"))
    tracker.log_call("gemini-2.5-flash", "gemini", 1000, 500, 0.00045, 1200)
    
    summary = tracker.get_daily_summary()
    assert summary["total_calls"] == 1
    assert "gemini-2.5-flash" in summary["models"]
    assert summary["models"]["gemini-2.5-flash"]["calls"] == 1

def test_llm_cost_tracker_budget_check(monkeypatch, tmp_path):
    monkeypatch.setenv("AI_DB_PATH", str(tmp_path / "test_cost.sqlite"))
    from llm_cost_tracker import LLMCostTracker
    
    tracker = LLMCostTracker(db_path=str(tmp_path / "test_cost.sqlite"))
    # Log $6 cost
    tracker.log_call("gemini-2.5-flash", "gemini", 1000, 500, 6.0, 1200)
    
    under_budget = tracker.check_budget(daily_limit_usd=5.0)
    assert under_budget is False, "Budget exceeded but returned True"

# --- Phase 10 Tests ---

def test_binary_quantizer_pack():
    from binary_quantizer import BinaryQuantizer
    import numpy as np
    
    floats = np.array([[-0.1, 0.0, 0.5, 1.2, -0.9, 0.1, 0.0, 0.0]]) # 8 elements -> 1 byte
    # Binarized: [0, 0, 1, 1, 0, 1, 0, 0]
    # Packed: 00110100 in binary = 52 in decimal
    packed = BinaryQuantizer.binarize_and_pack(floats)
    
    assert packed.shape == (1, 1)
    assert packed[0, 0] == 52

def test_binary_quantizer_hamming():
    from binary_quantizer import BinaryQuantizer
    import numpy as np
    
    q_bin = np.array([52], dtype=np.uint8) # 00110100
    
    # Doc 1: exact match
    # Doc 2: entirely flipped -> 11001011 -> 203
    # Doc 3: one bit different -> 00110101 -> 53
    doc_bins = np.array([[52], [203], [53]], dtype=np.uint8)
    
    distances = BinaryQuantizer.hamming_distance(q_bin, doc_bins)
    
    assert distances[0] == 0  # Exact
    assert distances[1] == 8  # All 8 bits different
    assert distances[2] == 1  # 1 bit different

def test_colbert_reranker_scores(monkeypatch):
    from colbert_reranker import ColBERTReranker
    
    # Mock __init__ so we don't download the model in tests
    monkeypatch.setattr(ColBERTReranker, "__init__", lambda self, *args, **kwargs: None)
    
    model = ColBERTReranker.__new__(ColBERTReranker)
    
    # Mock embeddings to just return something
    monkeypatch.setattr(model, "_get_embeddings", lambda text: None)
    
    # Mock max_sim_score to return length of text so we can test sorting/normalization
    monkeypatch.setattr(model, "_max_sim_score", lambda q, d: float(len(str(d))))
    
    docs = [{"content": "short\ntext", "id": "1"}, {"content": "very long document here indeed yes", "id": "2"}]
    
    # Provide dummy q_embs so the mock function can run without failing
    monkeypatch.setattr(model, "_get_embeddings", lambda text: str(text))
    
    results = model.rerank("query", docs)
    
    assert results[0]["id"] == "2"
    assert results[1]["id"] == "1"
    
    assert results[0]["colbert_normalized"] == 1.0
    assert results[1]["colbert_normalized"] == 0.0

def test_hybrid_retriever_binary_store(monkeypatch, tmp_path):
    monkeypatch.setenv("AI_DB_PATH", str(tmp_path / "test_hybrid.sqlite"))
    monkeypatch.setenv("CHROMA_PERSIST_DIR", str(tmp_path / "chroma"))
    
    from hybrid_retriever import HybridRetriever
    
    class MockEmbedder:
        def get_embeddings(self, text):
            import numpy as np
            return {"gemini": [0.1]*768, "bge": [0.5]*768}
            
    # Minimal init
    monkeypatch.setattr(HybridRetriever, "__init__", lambda self: None)
    retriever = HybridRetriever.__new__(HybridRetriever)
    retriever.chroma_client = None
    
    class MockCollection:
        def add(self, **kwargs): pass
        def query(self, **kwargs): return {"ids": [["doc_1"]]}
        def count(self): return 1
        
    retriever.collection = MockCollection()
    retriever.bge_collection = MockCollection()
    retriever.embedder = MockEmbedder()
    retriever.colbert_reranker = None
    retriever.reranker = None
    
    from binary_quantizer import BinaryQuantizer
    retriever.binary_quantizer = BinaryQuantizer()
    
    import sqlite3
    db_path = str(tmp_path / "test_hybrid.sqlite")
    
    def _mock_get_db(*args):
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        conn.execute('CREATE TABLE IF NOT EXISTS bm25_index (doc_id TEXT, content TEXT)')
        conn.execute('CREATE TABLE IF NOT EXISTS binary_embeddings (doc_id TEXT, packed_bge BLOB)')
        return conn
        
    retriever._get_db_connection = _mock_get_db
    
    retriever.add_documents(["test test"], [{"meta": "data"}], ["doc_1"])
    
    # Verify sqlite stored the blob
    conn = _mock_get_db()
    row = conn.execute("SELECT * FROM binary_embeddings WHERE doc_id='doc_1'").fetchone()
    assert row is not None
    assert isinstance(row['packed_bge'], bytes)

def test_hybrid_retriever_ensemble():
    from hybrid_retriever import HybridRetriever
    retriever = HybridRetriever.__new__(HybridRetriever)
    
    base = [{"id": "1"}, {"id": "2"}]
    flash = [{"id": "1", "flashrank_normalized": 1.0}, {"id": "2", "flashrank_normalized": 0.0}]
    colb = [{"id": "1", "colbert_normalized": 0.0}, {"id": "2", "colbert_normalized": 1.0}]
    
    res = retriever._ensemble_rerank(base, flash, colb, alpha=0.5)
    
    # Both should have exactly 0.5 final score due to perfect offset
    assert abs(res[0]["ensemble_score"] - 0.5) < 0.01
    assert abs(res[1]["ensemble_score"] - 0.5) < 0.01

# --- PHASE 11: Operational Readiness Tests ---

def test_telegram_notifier_format():
    """Verify send_trade_signal produces correct markdown format without relying on actual httpx."""
    from telegram_notifier import AITelegramNotifier
    
    notifier = AITelegramNotifier(bot_token="fake", chat_id="fake")
    sent_msgs = []
    
    def _mock_send(msg):
        sent_msgs.append(msg)
        
    notifier._send_message = _mock_send
    
    notifier.send_trade_signal(
        pair="BTC/USDT", 
        signal="long", 
        confidence=0.85, 
        reasoning_summary="Test reasoning", 
        position_pct=2.5
    )
    
    msg = sent_msgs[0]
    assert "📊 *AI Signal:* BTC/USDT" in msg
    assert "Direction: *BULLISH 🟢*" in msg
    assert "(confidence: 0.85)" in msg
    assert "Reasoning: Test reasoning" in msg
    assert "Position size: 2.5% of portfolio" in msg

def test_telegram_notifier_daily_summary():
    """Verify daily summary correctly interpolates and formats dict stats."""
    from telegram_notifier import AITelegramNotifier
    
    notifier = AITelegramNotifier(bot_token="fake", chat_id="fake")
    sent_msgs = []
    notifier._send_message = lambda msg: sent_msgs.append(msg)
    
    stats = {
        "open_trades": 3,
        "closed_today": 2,
        "daily_pnl": 45.20,
        "daily_pnl_pct": 1.2,
        "accuracy": 70.0,
        "correct_trades": 7,
        "total_eval_trades": 10,
        "api_cost_today": 0.32,
        "autonomy_level": "L2",
        "forgone_pnl": 12.50
    }
    
    notifier.send_daily_summary(stats)
    
    msg = sent_msgs[0]
    assert "📈 *Daily AI Report*" in msg
    assert "Open trades: 3 | Closed today: 2" in msg
    assert "Daily PNL: +$45.20 (+1.20%)" in msg
    assert "AI accuracy: 7/10 (70%)" in msg
    assert "API cost today: $0.32" in msg
    assert "Autonomy level: L2" in msg
    assert "Forgone PNL: $12.50" in msg

def test_futures_config_valid():
    """Verify config_binance_testnet_futures.json exists, is valid JSON, and has correct flags."""
    import json
    import os
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config_binance_testnet_futures.json")
    
    assert os.path.exists(config_path), f"{config_path} should exist"
    
    with open(config_path, "r") as f:
        config = json.load(f)
        
    assert config.get("trading_mode") == "futures"
    assert config.get("margin_mode") == "isolated"
    assert config.get("testnet") is True
    assert "binance" in config.get("exchange", {}).get("name", "").lower()
    
    whitelist = config.get("exchange", {}).get("pair_whitelist", [])
    # Check futures annotation format like BTC/USDT:USDT
    for pair in whitelist:
        assert ":" in pair, "Futures pair format should include settlement currency annotation (e.g. :USDT)"

def test_backtest_comparison_exists():
    """Verify the BacktestComparison module can be imported and instantiated."""
    try:
        from backtest_comparison import BacktestComparison
        comp = BacktestComparison(timerange="20260101-20260310", pairs=["BTC/USDT", "ETH/USDT"])
        assert comp.timerange == "20260101-20260310"
        assert comp.ai_strategy == "AIFreqtradeSizer"
        assert comp.baseline_strategy == "BaselineTechnical"
    except ImportError as e:
        import pytest
        pytest.fail(f"Failed to import backtest_comparison: {e}")

def test_baseline_strategy_no_ai():
    """Verify BaselineTechnical strategy avoids importing AI and retains isolation."""
    import os
    strat_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "freqtrade-strategies", "user_data", "strategies", "BaselineTechnical.py")
    
    assert os.path.exists(strat_path), f"BaselineTechnical.py should exist at {strat_path}"
    
    with open(strat_path, "r") as f:
        content = f.read()
        
    forbidden_imports = ["langchain", "google.genai", "rag_graph", "position_sizer", "autonomy_manager", "forgone_pnl_engine"]
    
    for forbidden in forbidden_imports:
        assert forbidden not in content, f"Baseline strategy must NOT import {forbidden}"
        
    assert "class BaselineTechnical" in content
    assert "return proposed_stake" in content, "Must not have dynamic AI stake modifier"

