import sys

file_path = "tests/test_ai_scripts.py"

with open(file_path, "r") as f:
    content = f.read()

import_target = """from magma_memory import MAGMAMemory"""

import_replacement = """from magma_memory import MAGMAMemory
from memo_rag import MemoRAG
from bidirectional_rag import BidirectionalRAG"""

content = content.replace(import_target, import_replacement, 1)

tests_append = """
@pytest.fixture
def mock_memorag(mock_db_path, mock_llm_router):
    import sqlite3
    with sqlite3.connect(mock_db_path) as conn:
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS memorag_global (
            id INTEGER PRIMARY KEY DEFAULT 1,
            summary TEXT,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )''')
        conn.commit()
    return MemoRAG(db_path=str(mock_db_path), llm_router=mock_llm_router)

def test_memorag_init_empty(mock_memorag):
    ans = mock_memorag.get_global_memory()
    assert "empty" in ans.lower()

def test_memorag_update_global_memory(mock_memorag):
    mock_memorag.update_global_memory(["Text 1"])
    ans = mock_memorag.get_global_memory()
    assert "mocked completion" in ans.lower()

def test_memorag_generate_draft_active(mock_memorag):
    import sqlite3
    long_memory = "x" * 100
    with sqlite3.connect(mock_memorag.db_path) as conn:
        c = conn.cursor()
        c.execute("UPDATE memorag_global SET summary = ? WHERE id = 1", (long_memory,))
        conn.commit()
    
    draft = mock_memorag.generate_draft("What is BTC?")
    assert "mocked completion" in draft.lower()

@pytest.fixture
def mock_bidi(mock_db_path, mock_llm_router):
    import sqlite3
    with sqlite3.connect(mock_db_path) as conn:
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS ai_lessons (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            decision_id INTEGER,
            pair TEXT,
            signal TEXT,
            outcome_pnl REAL,
            lesson_text TEXT,
            is_embedded BOOLEAN DEFAULT 0,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )''')
        conn.commit()
    return BidirectionalRAG(db_path=str(mock_db_path), llm_router=mock_llm_router)

def test_bidi_evaluate_trade(mock_bidi):
    lesson = mock_bidi.evaluate_trade_outcome(1, "BTC/USDT", "LONG", -2.5, "Strong RSI")
    assert "mocked completion" in lesson.lower()
    
    lessons = mock_bidi.get_unembedded_lessons()
    assert len(lessons) == 1
    assert lessons[0]['pair'] == "BTC/USDT"
    
def test_bidi_mark_embedded(mock_bidi):
    mock_bidi.evaluate_trade_outcome(2, "ETH/USDT", "SHORT", 1.5, "Weak MACD")
    lessons = mock_bidi.get_unembedded_lessons()
    lesson_id = lessons[0]['id']
    mock_bidi.mark_lessons_embedded([lesson_id])
    assert len(mock_bidi.get_unembedded_lessons()) == 0
"""

content += tests_append

with open(file_path, "w") as f:
    f.write(content)

