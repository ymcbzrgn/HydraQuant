import sys

file_path = "user_data/scripts/api_ai.py"

with open(file_path, "r") as f:
    content = f.read()

import_target = """from forgone_pnl_engine import ForgonePnLEngine
from memo_rag import MemoRAG"""

import_replacement = """from forgone_pnl_engine import ForgonePnLEngine
from memo_rag import MemoRAG
from bidirectional_rag import BidirectionalRAG"""

content = content.replace(import_target, import_replacement, 1)

endpoint_target = """        "uptime": "100%"
    }"""
    
endpoint_replacement = """        "uptime": "100%"
    }
    
@app.get("/api/ai/lessons")
def get_ai_lessons(limit: int = 50):
    \"\"\"Returns Bidirectional RAG trade evaluation lessons.\"\"\"
    try:
        with get_db_conn() as conn:
            rows = conn.execute(
                "SELECT id, pair, signal, outcome_pnl, lesson_text, is_embedded, timestamp "
                "FROM ai_lessons ORDER BY timestamp DESC LIMIT ?", (limit,)
            ).fetchall()
            return [dict(r) for r in rows]
    except sqlite3.OperationalError:
        return []"""

content = content.replace(endpoint_target, endpoint_replacement, 1)

with open(file_path, "w") as f:
    f.write(content)
