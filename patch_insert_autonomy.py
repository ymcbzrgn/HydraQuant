import sqlite3
from user_data.scripts.db import AI_DB_PATH
import os

try:
    with sqlite3.connect(os.environ.get('AI_DB_PATH', AI_DB_PATH)) as conn:
        conn.execute("INSERT INTO autonomy_diagnostics (level, days_stuck, n_trades_30d, winrate_30d, sharpe_approx_30d, worst_drawdown_30d) VALUES (1, 0, 0, 0.0, 0.0, 0.0)")
except Exception:
    pass
