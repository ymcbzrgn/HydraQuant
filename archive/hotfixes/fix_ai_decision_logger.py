import sys

file_path = "user_data/scripts/ai_decision_logger.py"

with open(file_path, "r") as f:
    content = f.read()

import_target = """import logging
from typing import Dict, Any, List, Optional"""

import_replacement = """import logging
from typing import Dict, Any, List, Optional
from bidirectional_rag import BidirectionalRAG"""

content = content.replace(import_target, import_replacement, 1)

init_target = """        self.db_path = db_path
        self._init_db()"""
        
init_replacement = """        self.db_path = db_path
        self.bidi_rag = BidirectionalRAG(db_path=db_path)
        self._init_db()"""
        
content = content.replace(init_target, init_replacement, 1)

eval_target = """                # If no rows were updated, ID is invalid
                if c.rowcount == 0:
                     logger.warning(f"Could not bind outcome to decision_id {decision_id}. ID not found.")
                     return False
                     
                conn.commit()
            
            logger.info(f"Successfully bound Trade Outcome [{pnl_percent:.2f}%] to Decision ID {decision_id}")
            return True"""
            
eval_replacement = """                # If no rows were updated, ID is invalid
                if c.rowcount == 0:
                     logger.warning(f"Could not bind outcome to decision_id {decision_id}. ID not found.")
                     return False
                     
                c.execute("SELECT pair, signal_type, reasoning_summary FROM ai_decisions WHERE id = ?", (decision_id,))
                decision_info = c.fetchone()
                     
                conn.commit()
            
            if decision_info:
                # Phase 15: Run Bidirectional RAG evaluation
                pair = decision_info[0]
                sig = decision_info[1]
                reasoning = decision_info[2]
                self.bidi_rag.evaluate_trade_outcome(decision_id, pair, sig, pnl_percent, reasoning)
                
            logger.info(f"Successfully bound Trade Outcome [{pnl_percent:.2f}%] to Decision ID {decision_id}")
            return True"""

content = content.replace(eval_target, eval_replacement, 1)

with open(file_path, "w") as f:
    f.write(content)
