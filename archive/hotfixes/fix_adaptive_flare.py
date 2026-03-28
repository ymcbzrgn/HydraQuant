import sys

file_path = "user_data/scripts/adaptive_router.py"

with open(file_path, "r") as f:
    content = f.read()

import_target = """from rag_fusion import RAGFusion
from self_rag import SelfRAG"""

import_replacement = """from rag_fusion import RAGFusion
from self_rag import SelfRAG
from flare_retriever import FLARERetriever"""

content = content.replace(import_target, import_replacement, 1)

init_target = """        self.rag_fusion = RAGFusion(router=self.router)
        self.self_rag = SelfRAG(router=self.router)"""

init_replacement = """        self.rag_fusion = RAGFusion(router=self.router)
        self.self_rag = SelfRAG(router=self.router)
        # Phase 16: FLARE for dynamic generation augmentation
        self.flare = FLARERetriever(llm_router=self.router)"""

content = content.replace(init_target, init_replacement, 1)

with open(file_path, "w") as f:
    f.write(content)
