import pytest
import sqlite3
import os
from unittest.mock import patch, MagicMock
from memo_rag import MemoRAG

@pytest.fixture
def mock_memorag(tmp_path):
    db_path = tmp_path / "test_memorag.sqlite"
    # Basic router mock returning constant texts
    class MockRouter:
        def invoke(self, prompt, **kwargs):
            class MockResponse:
                content = "Mocked compressed global memory summary."
            return MockResponse()
            
    memorag = MemoRAG(db_path=str(db_path), llm_router=MockRouter())
    return memorag

def test_memorag_init_empty(mock_memorag):
    # Ensure it initializes with empty schema
    ans = mock_memorag.get_global_memory()
    assert "empty" in ans.lower()

def test_memorag_update_global_memory(mock_memorag):
    # Push string arrays
    mock_memorag.update_global_memory(["Text 1", "Text 2", "Text 3"])
    
    # Should contain mocked compression response
    ans = mock_memorag.get_global_memory()
    assert ans == "Mocked compressed global memory summary."
    
def test_memorag_generate_draft_bypass(mock_memorag):
    # If memory < 50 length, return query
    with sqlite3.connect(mock_memorag.db_path) as conn:
        c = conn.cursor()
        c.execute("UPDATE memorag_global SET summary = 'short' WHERE id = 1")
        conn.commit()
    
    draft = mock_memorag.generate_draft("What is BTC?")
    assert draft == "What is BTC?"

def test_memorag_generate_draft_active(mock_memorag):
    # If memory >= 50 length, return mocked draft
    long_memory = "x" * 100
    with sqlite3.connect(mock_memorag.db_path) as conn:
        c = conn.cursor()
        c.execute("UPDATE memorag_global SET summary = ? WHERE id = 1", (long_memory,))
        conn.commit()
    
    draft = mock_memorag.generate_draft("What is BTC?")
    assert draft == "Mocked compressed global memory summary."
