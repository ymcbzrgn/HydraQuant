import os
import sys
import importlib
import pytest
from fastapi.testclient import TestClient

# Make user_data/scripts importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "user_data", "scripts"))

def test_ai_config_cors_origins_default(monkeypatch):
    """Verify that CORS_ORIGINS defaults to an empty list if env var is not set."""
    import ai_config
    monkeypatch.delenv("CORS_ORIGINS", raising=False)
    importlib.reload(ai_config)
    assert ai_config.CORS_ORIGINS == []

def test_ai_config_cors_origins_custom(monkeypatch):
    """Verify that CORS_ORIGINS reads correctly from the environment variable."""
    import ai_config
    monkeypatch.setenv("CORS_ORIGINS", "http://localhost:3000, https://app.example.com")
    importlib.reload(ai_config)
    assert ai_config.CORS_ORIGINS == ["http://localhost:3000", "https://app.example.com"]

def test_api_ai_cors_middleware(monkeypatch):
    """Verify that api_ai.app uses CORS_ORIGINS from ai_config."""
    monkeypatch.setenv("CORS_ORIGINS", "http://test-origin.com")

    import ai_config
    importlib.reload(ai_config)

    import api_ai
    importlib.reload(api_ai)

    # Check middleware
    cors_middleware = None
    for middleware in api_ai.app.user_middleware:
        if "CORSMiddleware" in str(middleware.cls):
            cors_middleware = middleware
            break

    assert cors_middleware is not None
    assert cors_middleware.options["allow_origins"] == ["http://test-origin.com"]

def test_api_ai_cors_request(monkeypatch):
    """Verify CORS response headers in api_ai."""
    monkeypatch.setenv("CORS_ORIGINS", "http://test-origin.com")

    import ai_config
    importlib.reload(ai_config)

    # We need to mock DB path for api_ai to import correctly if it's not already
    monkeypatch.setenv("AI_DB_PATH", ":memory:")

    import api_ai
    importlib.reload(api_ai)

    client = TestClient(api_ai.app)

    # Test allowed origin
    response = client.options(
        "/api/ai/status",
        headers={
            "Origin": "http://test-origin.com",
            "Access-Control-Request-Method": "GET",
        }
    )
    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") == "http://test-origin.com"

    # Test disallowed origin
    response = client.options(
        "/api/ai/status",
        headers={
            "Origin": "http://evil.com",
            "Access-Control-Request-Method": "GET",
        }
    )
    # CORSMiddleware returns 400 or just doesn't include the header if origin is not allowed
    assert response.headers.get("access-control-allow-origin") != "http://evil.com"
