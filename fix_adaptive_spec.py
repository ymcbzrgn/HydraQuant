import sys

file_path = "user_data/scripts/adaptive_router.py"

with open(file_path, "r") as f:
    content = f.read()

import_target = """from flare_retriever import FLARERetriever"""

import_replacement = """from flare_retriever import FLARERetriever
from speculative_rag import SpeculativeRAG"""

content = content.replace(import_target, import_replacement, 1)

init_target = """        self.flare = FLARERetriever(llm_router=self.router)"""

init_replacement = """        self.flare = FLARERetriever(llm_router=self.router)
        # Phase 16: Speculative RAG for multi-draft generation
        self.speculative_rag = SpeculativeRAG(llm_router=self.router)"""

content = content.replace(init_target, init_replacement, 1)

with open(file_path, "w") as f:
    f.write(content)

