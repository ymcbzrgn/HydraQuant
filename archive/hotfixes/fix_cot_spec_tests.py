import sys

file_path = "tests/test_ai_scripts.py"

with open(file_path, "r") as f:
    content = f.read()

tests_append = """
# --- Phase 16: CoT-RAG Tests ---

@pytest.fixture
def mock_cot_rag(mock_llm_router):
    from cot_rag import CoTRAG
    
    class MockCotRetriever:
        def search(self, query, top_k=3):
            return [{"text": "Mocked evidence for " + query}]
            
    cot = CoTRAG(llm_router=mock_llm_router, retriever=MockCotRetriever())
    return cot

def test_cot_rag_5_steps(mock_cot_rag):
    class StepMockRouter:
        def invoke(self, messages, **kwargs):
            class MockResponse:
                def __init__(self, text):
                    self.content = text
            prompt = str(messages[0].content) if hasattr(messages[0], 'content') else str(messages)
            if "Final Synthesis AI" in prompt:
                return MockResponse("BULLISH")
            return MockResponse("Step output.")
            
    mock_cot_rag.router = StepMockRouter()
    
    result = mock_cot_rag.reason_step_by_step("BTC/USDT", "Show me analysis")
    
    assert len(result["steps"]) == 5
    assert result["final_decision"] == "BULLISH"
    assert "Step output." in result["reasoning_chain"]
    
def test_cot_rag_evidence_per_step(mock_cot_rag):
    class EvidenceMockRouter:
        def invoke(self, messages, **kwargs):
            class MockResponse:
                def __init__(self, text):
                    self.content = text
            return MockResponse("Analysis")
            
    mock_cot_rag.router = EvidenceMockRouter()
    result = mock_cot_rag.reason_step_by_step("ETH/USDT", "query")
    
    # Check that steps that have search_query_hint used evidence
    for step in result["steps"]:
        if step["step"] != "decision":
            assert step["evidence_count"] > 0
            
# --- Phase 16: Speculative RAG Tests ---

@pytest.fixture
def mock_spec_rag(mock_llm_router):
    from speculative_rag import SpeculativeRAG
    
    class MockSpecRetriever:
        def search(self, query, top_k=15):
            return [{"text": f"Doc {i}"} for i in range(15)]
            
    spec = SpeculativeRAG(llm_router=mock_llm_router, retriever=MockSpecRetriever())
    return spec

def test_speculative_3_drafts(mock_spec_rag):
    class SpecMockRouter:
        def __init__(self):
            self.call_count = 0
            
        def invoke(self, messages, **kwargs):
            self.call_count += 1
            class MockResponse:
                def __init__(self, text):
                    self.content = text
            
            prompt = str(messages[1].content) if len(messages) > 1 and hasattr(messages[1], 'content') else str(messages)
            if "scenario analyst drafting" in prompt:
                return MockResponse(f"Draft {self.call_count}")
            elif "Verification Overlord" in prompt:
                return MockResponse("BEST_DRAFT_INDEX: 1\\nReason: It is clearly the best.")
            return MockResponse("Fallback")
            
    mock_spec_rag.router = SpecMockRouter()
    result = mock_spec_rag.draft_and_verify("Query", num_drafts=3)
    
    assert len(result["all_drafts"]) == 3
    assert result["best_draft_index"] == 1
    assert "Draft 2" in result["best_draft"]

def test_speculative_verify_picks_best(mock_spec_rag):
    class VerifyMockRouter:
        def invoke(self, messages, **kwargs):
            class MockResponse:
                def __init__(self, text):
                    self.content = text
                    
            prompt = str(messages[1].content) if len(messages) > 1 and hasattr(messages[1], 'content') else str(messages)
            if "Verification Overlord" in prompt:
                return MockResponse("BEST_DRAFT_INDEX: 2\\nReason: Draft 2 aligns perfectly.")
            return MockResponse("Draft")
            
    mock_spec_rag.router = VerifyMockRouter()
    result = mock_spec_rag.draft_and_verify("Query", num_drafts=3)
    
    assert result["best_draft_index"] == 2
    assert "Draft 2 aligns perfectly" in result["verification_reasoning"]

"""
content += tests_append

with open(file_path, "w") as f:
    f.write(content)

