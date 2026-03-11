import pytest
import sqlite3
import os
from unittest.mock import patch, MagicMock
from bidirectional_rag import BidirectionalRAG

@pytest.fixture
def mock_bidi(tmp_path):
    db_path = tmp_path / "test_bidi.sqlite"
    # Basic router mock returning constant texts
    class MockRouter:
        def invoke(self, prompt, **kwargs):
            class MockResponse:
                content = "Trade failed due to unexpected volatility spike."
            return MockResponse()
            
    bidi = BidirectionalRAG(db_path=str(db_path), llm_router=MockRouter())
    return bidi

def test_bidi_init_empty(mock_bidi):
    # Ensure it initializes with empty schema
    lessons = mock_bidi.get_unembedded_lessons()
    assert len(lessons) == 0

def test_bidi_evaluate_trade(mock_bidi):
    # Insert mock evaluation
    lesson = mock_bidi.evaluate_trade_outcome(
        decision_id=1,
        pair="BTC/USDT",
        signal="LONG",
        outcome_pnl=-2.5,
        reasoning="Strong RSI"
    )
    
    # Should contain mocked compression response
    assert lesson == "Trade failed due to unexpected volatility spike."
    
    # Should be stored unembedded
    lessons = mock_bidi.get_unembedded_lessons()
    assert len(lessons) == 1
    assert lessons[0]['pair'] == "BTC/USDT"
    assert lessons[0]['outcome_pnl'] == -2.5
    assert int(lessons[0]['is_embedded']) == 0

def test_bidi_mark_embedded(mock_bidi):
    mock_bidi.evaluate_trade_outcome(2, "ETH/USDT", "SHORT", 1.5, "Weak MACD")
    lessons = mock_bidi.get_unembedded_lessons()
    assert len(lessons) == 1
    
    lesson_id = lessons[0]['id']
    mock_bidi.mark_lessons_embedded([lesson_id])
    
    # Verify no unembedded remain
    lessons_after = mock_bidi.get_unembedded_lessons()
    assert len(lessons_after) == 0
