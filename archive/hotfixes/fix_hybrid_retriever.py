import sys

file_path = "user_data/scripts/hybrid_retriever.py"

with open(file_path, "r") as f:
    content = f.read()
    
target = """        self.streaming_rag = StreamingRAG()
        self.raptor = RAPTORTree()
        self.magma = MAGMAMemory()"""

replacement = """        self.streaming_rag = StreamingRAG()
        self.raptor = RAPTORTree()
        self.magma = MAGMAMemory()
        self.memorag = MemoRAG()"""
        
content = content.replace(target, replacement)

target2 = """        self.streaming_rag = None
        self.raptor = None
        self.magma = None"""

replacement2 = """        self.streaming_rag = None
        self.raptor = None
        self.magma = None
        self.memorag = None"""
        
content = content.replace(target2, replacement2)

target3 = """from magma_memory import MAGMAMemory"""

replacement3 = """from magma_memory import MAGMAMemory
from memo_rag import MemoRAG"""

content = content.replace(target3, replacement3, 1)

target4 = """        query_embs = self.embedder.get_embeddings(query)
        
        # 1. Sparse Search (SQLite FTS5 BM25) - Widen the funnel
        bm25_top_ids = []
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                sanitized_query = query.replace('"', '').replace("'", "")"""

replacement4 = """        # Phase 15: Generate MemoRAG Global Draft Context
        original_query = query
        if self.memorag:
            draft = self.memorag.generate_draft(query)
            if draft and draft != query:
                # Merge draft context for denser embedding vector extraction
                query = f"{query} | Context Draft: {draft}"
                logger.info("MemoRAG injected global draft into search query.")

        query_embs = self.embedder.get_embeddings(query)
        
        # 1. Sparse Search (SQLite FTS5 BM25) - Widen the funnel
        bm25_top_ids = []
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                sanitized_query = original_query.replace('"', '').replace("'", "")"""
                
content = content.replace(target4, replacement4)

with open(file_path, "w") as f:
    f.write(content)
