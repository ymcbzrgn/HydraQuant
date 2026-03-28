import sys

file_path = "tests/test_ai_scripts.py"

with open(file_path, "r") as f:
    content = f.read()

target = """@pytest.fixture(autouse=True)
def mock_external_apis(monkeypatch):
    \"\"\"Mock ALL external API calls to prevent tests from failing on keys/network.\"\"\""""

replacement = """@pytest.fixture(autouse=True)
def mock_external_apis(monkeypatch):
    \"\"\"Mock ALL external API calls to prevent tests from failing on keys/network.\"\"\"
    
    # Critical fix for torch.__spec__ is None when pytest tries to inspect uninitialized torch submodules
    import sys
    class DummyTorchSpec:
        name = "torch"
    
    try:
        import torch
        if not hasattr(torch, "__spec__") or torch.__spec__ is None:
            torch.__spec__ = DummyTorchSpec()
    except ImportError:
        pass
"""

content = content.replace(target, replacement, 1)

with open(file_path, "w") as f:
    f.write(content)

