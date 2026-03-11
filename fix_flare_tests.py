import sys

file_path = "tests/test_ai_scripts.py"

with open(file_path, "r") as f:
    content = f.read()

tests_append = """
# --- Phase 16: FLARE Tests ---

@pytest.fixture
def mock_flare(mock_llm_router):
    from flare_retriever import FLARERetriever
    
    class MockRetriever:
        def search(self, query, top_k=2):
            return [{"text": "Mocked extra context for " + query}]
            
    flare = FLARERetriever(llm_router=mock_llm_router, retriever=MockRetriever())
    return flare

def test_flare_active_retrieval(mock_flare):
    from flare_retriever import FLARERetriever
    
    # Mock LLM generation to return 2 sentences
    # Mock confidence to return < 0.5 for the first sentence
    
    class ActiveMockRouter:
        def __init__(self):
            self.call_count = 0
            
        def invoke(self, messages, **kwargs):
            class MockResponse:
                def __init__(self, text):
                    self.content = text
            
            prompt = str(messages[0].content) if hasattr(messages[0], 'content') else str(messages)
            if len(messages) > 1 and hasattr(messages[1], 'content'):
                prompt += str(messages[1].content)
                
            self.call_count += 1
            
            if "Generate an analytical response" in prompt:
                return MockResponse("First uncertain sentence. Second certain sentence.")
            elif "evaluating the confidence" in prompt:
                if "First uncertain" in prompt:
                    return MockResponse("0.3")
                else:
                    return MockResponse("0.9")
            elif "rewrite and improve" in prompt:
                return MockResponse("First corrected sentence.")
            elif "You are evaluating the confidence" in prompt:
                 # The exact prompt used by _assess_confidence
                 if "First uncertain" in prompt:
                     return MockResponse("0.3")
                 return MockResponse("0.9")
                
            return MockResponse("Fallback mock.")
            
    mock_flare.router = ActiveMockRouter()
    
    result = mock_flare.generate_with_active_retrieval("What is the state of the market?")
    
    assert result["retrievals_triggered"] > 0
    assert len(result["low_confidence_sentences"]) > 0
    assert "First uncertain" in result["low_confidence_sentences"][0]
    assert "First corrected" in result["analysis"]

def test_flare_high_confidence_no_retrieval(mock_flare):
    class HighConfMockRouter:
        def __init__(self):
            self.call_count = 0
            
        def invoke(self, messages, **kwargs):
            class MockResponse:
                def __init__(self, text):
                    self.content = text
            
            prompt = str(messages[0].content) if hasattr(messages[0], 'content') else str(messages)
            if len(messages) > 1 and hasattr(messages[1], 'content'):
                prompt += str(messages[1].content)
                
            self.call_count += 1
            
            if "Generate an analytical response" in prompt:
                return MockResponse("Everything is fine. No uncertainty here.")
            elif "You are evaluating the confidence" in prompt:
                return MockResponse("0.9")
                
            return MockResponse("Fallback mock.")
            
    mock_flare.router = HighConfMockRouter()
    
    result = mock_flare.generate_with_active_retrieval("Query")
    
    assert result["retrievals_triggered"] == 0
    assert len(result["low_confidence_sentences"]) == 0
    assert "Everything is fine" in result["analysis"]

"""
content += tests_append

with open(file_path, "w") as f:
    f.write(content)

