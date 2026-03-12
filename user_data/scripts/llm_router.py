import os
import logging
import threading
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableWithFallbacks
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from llm_cost_tracker import LLMCostTracker

import httpx
from google.api_core import exceptions as google_exc
import openai
import groq
import time
import re

# google.genai SDK (langchain-google-genai v4+) throws its own exceptions
try:
    from google.genai import errors as genai_errors
    _GENAI_FAILOVER = (genai_errors.ClientError, genai_errors.ServerError)
except ImportError:
    _GENAI_FAILOVER = ()

# Global dictionary to store API Keys currently in Penalty Box: {api_key: unlock_timestamp}
KEY_COOLDOWNS: Dict[str, float] = {}
COOLDOWN_DURATION = 60.0  # seconds
_COOLDOWN_LOCK = threading.Lock()  # Thread-safe access to KEY_COOLDOWNS

# Module-level model discovery cache — shared across ALL LLMRouter instances
_GEMINI_MODEL_CACHE: Dict[str, Any] = {"models": None, "timestamp": 0.0}
_MODEL_CACHE_TTL = 600.0  # 10 minutes

# Per-model penalty box: "api_key:model_name" → unlock_timestamp
# Gemini quotas are PER-MODEL PER-PROJECT — one model's 429 doesn't affect others on the same key
MODEL_COOLDOWNS: Dict[str, float] = {}

# Exceptions that CANNOT be recovered by failover — bugs in OUR code
_HARD_CRASH_EXCEPTIONS = (ValueError, TypeError, KeyError, AttributeError, SyntaxError)

# Everything else = failover. This list is for penalty-box logic (rate limit detection).
FAILOVER_EXCEPTIONS = (
    *_GENAI_FAILOVER,
    google_exc.NotFound,
    google_exc.TooManyRequests,
    google_exc.ResourceExhausted,
    google_exc.ServiceUnavailable,
    google_exc.InternalServerError,
    google_exc.DeadlineExceeded,
    google_exc.Unauthenticated,
    google_exc.PermissionDenied,
    openai.RateLimitError,
    openai.InternalServerError,
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.AuthenticationError,
    openai.PermissionDeniedError,
    groq.RateLimitError,
    groq.InternalServerError,
    groq.APIConnectionError,
    groq.APITimeoutError,
    groq.AuthenticationError,
    groq.PermissionDeniedError,
    httpx.TimeoutException,
    httpx.NetworkError,
)

load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))
logger = logging.getLogger(__name__)

class LLMRouter:
    """
    Phase 5.3: Bulletproof LLM Router with Dynamic Load Balancing
    Routes requests to Gemini Flash first. If it rate-limits or fails, 
    seamlessly falls back to other Gemini keys chronologically (Round-Robin),
    then falls back to Groq Llama-3, then OpenRouter models.
    """
    
    def __init__(self, temperature: float = 0.0, request_timeout: int = 15):
        self.temperature = temperature
        self.request_timeout = request_timeout
        self.cost_tracker = LLMCostTracker()
        self._key_lock = threading.Lock()  # Thread-safe key rotation
        
        # 1. Primary Models (Provider-Internal Failover): Gemini
        # We handle multiple API keys dynamically.
        self.gemini_keys = []
        
        # Support comma-separated keys in GEMINI_API_KEYS
        keys_str = os.environ.get("GEMINI_API_KEYS", "")
        if keys_str:
            self.gemini_keys.extend([k.strip() for k in keys_str.split(",") if k.strip()])
            
        # Support GEMINI_API_KEY and numbered variants (1-10)
        single_key = os.environ.get("GEMINI_API_KEY")
        if single_key and single_key not in self.gemini_keys:
            self.gemini_keys.append(single_key)
            
        for i in range(1, 11):
            k = os.environ.get(f"GEMINI_API_KEY_{i}")
            if k and k not in self.gemini_keys:
                self.gemini_keys.append(k)
                
        # Deduplicate keys just in case
        self.gemini_keys = list(dict.fromkeys(self.gemini_keys))

        self.gemini_models_by_key = {}
        if self.gemini_keys:
            # Dynamic model discovery — fetch real available models from Gemini API
            gemini_model_names = self._discover_gemini_models(self.gemini_keys[0])

            # Group initialized models by their key
            for key in self.gemini_keys:
                models_for_key = []
                for m_name in gemini_model_names:
                    models_for_key.append(ChatGoogleGenerativeAI(
                        model=m_name,
                        api_key=key,
                        temperature=self.temperature,
                        timeout=self.request_timeout,
                        max_retries=1  # MUST be 1 not 0: SDK bug treats 0 as "use default 5 retries"
                    ))
                self.gemini_models_by_key[key] = models_for_key

            logger.info(f"Loaded {len(self.gemini_keys)} Gemini keys × {len(gemini_model_names)} models. "
                         f"Models: {gemini_model_names}")
            
        # 2. First Fallback: Groq (Llama-3.1 8B Instant)
        self.groq_key = os.environ.get("GROQ_API_KEY")
        self.fallback_1 = None
        if self.groq_key:
            self.fallback_1 = ChatGroq(
                model="llama-3.1-8b-instant",
                api_key=self.groq_key,
                temperature=self.temperature,
                timeout=self.request_timeout,
                max_retries=0
            )
            
        # 3. Second Fallback: OpenRouter (DeepSeek R1 / any available)
        self.openrouter_key = os.environ.get("OPENROUTER_API_KEY")
        self.fallback_2 = None
        if self.openrouter_key:
            # Using ChatOpenAI as OpenAI-Compatible wrapper for OpenRouter
            self.fallback_2 = ChatOpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=self.openrouter_key,
                model="meta-llama/llama-3.3-70b-instruct:free",
                temperature=self.temperature,
                timeout=self.request_timeout,
                max_retries=0
            )
        
    @staticmethod
    def _discover_gemini_models(api_key: str) -> list:
        """Discover available Gemini chat models from API. Cached for 10 minutes."""
        FALLBACK_MODELS = [
            "models/gemini-2.5-flash",
            "models/gemini-2.0-flash",
        ]

        # Return cached result if fresh
        now = time.time()
        if _GEMINI_MODEL_CACHE["models"] and (now - _GEMINI_MODEL_CACHE["timestamp"]) < _MODEL_CACHE_TTL:
            logger.info(f"Using cached model list ({len(_GEMINI_MODEL_CACHE['models'])} models)")
            return _GEMINI_MODEL_CACHE["models"]

        try:
            from google import genai
            client = genai.Client(api_key=api_key)

            discovered = []
            for m in client.models.list():
                name = m.name if hasattr(m, 'name') else str(m)
                actions = m.supported_actions if hasattr(m, 'supported_actions') else []

                if 'generateContent' not in (actions or []):
                    continue

                # Only keep actual Gemini chat models — filter out gemma, nano, robotics, tts, etc.
                # name format: "models/gemini-2.5-flash" or "models/gemini-3-pro-preview"
                model_short = name.replace("models/", "")
                if not model_short.startswith("gemini-"):
                    continue
                if any(skip in model_short for skip in ['tts', 'robotics', 'image', 'embedding', 'vision', 'audio', 'computer-use']):
                    continue

                discovered.append(name)

            if discovered:
                # Sort: flash-lite first (cheapest/fastest), then flash, then pro last
                def _model_priority(name):
                    short = name.replace("models/", "")
                    if "flash-lite" in short: return (0, short)
                    if "flash" in short:      return (1, short)
                    if "lite" in short:       return (2, short)
                    if "pro" in short:        return (3, short)
                    return (4, short)
                discovered.sort(key=_model_priority)
                # Deduplicate: gemini-X-flash and gemini-X-flash-001 share rate limits
                all_shorts = {n.replace("models/", "") for n in discovered}
                deduped = []
                for name in discovered:
                    short = name.replace("models/", "")
                    base = re.sub(r'-\d{3}$', '', short)
                    if base != short and base in all_shorts:
                        continue  # Versioned variant — base model covers it
                    deduped.append(name)
                discovered = deduped
                logger.info(f"Discovered {len(discovered)} Gemini chat models (deduped): {discovered}")
                _GEMINI_MODEL_CACHE["models"] = discovered
                _GEMINI_MODEL_CACHE["timestamp"] = now
                return discovered
            else:
                logger.warning("No Gemini chat models discovered. Using fallback list.")
                return FALLBACK_MODELS
        except Exception as e:
            logger.warning(f"Model discovery failed: {e}. Using fallback list.")
            return FALLBACK_MODELS

    def _get_chain(self) -> List[Any]:
        """Constructs and returns the active LangChain fallbacks without rate-limited keys."""
        chain = []
        current_time = time.time()
        
        if self.gemini_keys:
            # Thread-safe rotation: Move first key to the back for Round-Robin
            with self._key_lock:
                first_key = self.gemini_keys.pop(0)
                self.gemini_keys.append(first_key)
                keys_snapshot = list(self.gemini_keys)  # Work on a snapshot
            
            for key in keys_snapshot:
                # Check if this Gemini key is currently in the Penalty Box
                with _COOLDOWN_LOCK:
                    unlock_time = KEY_COOLDOWNS.get(key, 0)
                if current_time < unlock_time:
                    continue # Skip this key, it's exhausted!
                    
                chain.extend(self.gemini_models_by_key[key])
                
        if self.fallback_1:
            chain.append(self.fallback_1)
        if self.fallback_2:
            chain.append(self.fallback_2)
            
        if not chain:
            raise ValueError("All API keys are exhausted or unavailable.")
            
        return chain
            
    def _penalize_key(self, model):
        """Put a Gemini API key into the penalty box if the model has one."""
        if hasattr(model, "google_api_key"):
            failed_key = getattr(model, "google_api_key")
            if hasattr(failed_key, "get_secret_value"):
                failed_key = failed_key.get_secret_value()
            with _COOLDOWN_LOCK:
                if failed_key not in KEY_COOLDOWNS or time.time() > KEY_COOLDOWNS[failed_key]:
                    KEY_COOLDOWNS[failed_key] = time.time() + COOLDOWN_DURATION
                    logger.warning(f"Key penalized for {COOLDOWN_DURATION}s.")

    def _is_key_penalized(self, model) -> bool:
        """Check if this model's API key is in the penalty box (auth/connection errors)."""
        if hasattr(model, "google_api_key"):
            key = getattr(model, "google_api_key")
            if hasattr(key, "get_secret_value"):
                key = key.get_secret_value()
            with _COOLDOWN_LOCK:
                return time.time() < KEY_COOLDOWNS.get(key, 0)
        return False

    def _penalize_model(self, model):
        """Put a specific (key, model) combo into penalty box. Other models on same key stay active."""
        api_key = ""
        if hasattr(model, "google_api_key"):
            api_key = getattr(model, "google_api_key")
            if hasattr(api_key, "get_secret_value"):
                api_key = api_key.get_secret_value()
        m_name = getattr(model, "model_name", getattr(model, "model", "unknown"))
        penalty_key = f"{api_key}:{m_name}"
        with _COOLDOWN_LOCK:
            if penalty_key not in MODEL_COOLDOWNS or time.time() > MODEL_COOLDOWNS[penalty_key]:
                MODEL_COOLDOWNS[penalty_key] = time.time() + COOLDOWN_DURATION
                logger.warning(f"[Penalize] {m_name} penalized {COOLDOWN_DURATION}s. Other models on this key still active.")

    def _is_model_penalized(self, model) -> bool:
        """Check if this specific (key, model) combo is rate-limited."""
        api_key = ""
        if hasattr(model, "google_api_key"):
            api_key = getattr(model, "google_api_key")
            if hasattr(api_key, "get_secret_value"):
                api_key = api_key.get_secret_value()
        m_name = getattr(model, "model_name", getattr(model, "model", "unknown"))
        penalty_key = f"{api_key}:{m_name}"
        with _COOLDOWN_LOCK:
            return time.time() < MODEL_COOLDOWNS.get(penalty_key, 0)

    def invoke(self, messages: List[Any], **kwargs):
        """Synchronously invokes the custom failover loop with Cooldown tracking."""
        chain = self._get_chain()
        last_exception: Optional[Exception] = None

        for model in chain:
            # Skip if key is dead (auth/conn) or this specific model is rate-limited
            if self._is_key_penalized(model) or self._is_model_penalized(model):
                continue

            try:
                start_time = time.time()
                response = model.invoke(messages, **kwargs)
                latency_ms = (time.time() - start_time) * 1000

                in_tok = 0
                out_tok = 0
                if hasattr(response, 'usage_metadata') and response.usage_metadata:
                    in_tok = response.usage_metadata.get('input_tokens', 0)
                    out_tok = response.usage_metadata.get('output_tokens', 0)

                m_name = getattr(model, "model_name", getattr(model, "model", "unknown_model"))
                provider = "gemini" if "gemini" in m_name.lower() else ("groq" if "llama" in m_name.lower() else "openrouter")
                cost = self.cost_tracker.calculate_cost(m_name, in_tok, out_tok)
                self.cost_tracker.log_call(m_name, provider, in_tok, out_tok, cost, latency_ms)

                return response
            except _HARD_CRASH_EXCEPTIONS as e:
                logger.error(f"Code bug in LLM pipeline: {type(e).__name__}: {e}")
                raise
            except Exception as e:
                last_exception = e
                m_name = getattr(model, "model_name", getattr(model, "model", "?"))
                err_upper = str(e).upper()
                if "RESOURCE_EXHAUSTED" in err_upper or "429" in err_upper or "TOO_MANY_REQUESTS" in err_upper:
                    # Rate limit: only THIS model's quota is exhausted, other models on same key have separate quota
                    self._penalize_model(model)
                    logger.info(f"[RateLimit] {m_name} quota exhausted. Other models on same key still OK.")
                elif "UNAUTHENTICATED" in err_upper or "PERMISSION_DENIED" in err_upper:
                    # Auth error: entire key is broken
                    self._penalize_key(model)
                    logger.warning(f"[AuthError] {m_name} → {type(e).__name__}. Penalizing entire key.")
                else:
                    # Unknown/other error: penalize model only (don't kill working models on same key)
                    self._penalize_model(model)
                    logger.warning(f"[Failover] {m_name} → {type(e).__name__}: {e}. Next model...")
                continue

        logger.error(f"Complete LLM Failure (All Fallbacks Exhausted)")
        if last_exception:
            raise last_exception
        raise ValueError("No fallbacks available.")

    async def ainvoke(self, messages: List[Any], **kwargs):
        """Asynchronously invokes the custom failover loop with Cooldown tracking."""
        chain = self._get_chain()
        last_exception: Optional[Exception] = None

        for model in chain:
            # Skip if key is dead (auth/conn) or this specific model is rate-limited
            if self._is_key_penalized(model) or self._is_model_penalized(model):
                continue

            try:
                start_time = time.time()
                response = await model.ainvoke(messages, **kwargs)
                latency_ms = (time.time() - start_time) * 1000

                in_tok = 0
                out_tok = 0
                if hasattr(response, 'usage_metadata') and response.usage_metadata:
                    in_tok = response.usage_metadata.get('input_tokens', 0)
                    out_tok = response.usage_metadata.get('output_tokens', 0)

                m_name = getattr(model, "model_name", getattr(model, "model", "unknown_model"))
                provider = "gemini" if "gemini" in m_name.lower() else ("groq" if "llama" in m_name.lower() else "openrouter")
                cost = self.cost_tracker.calculate_cost(m_name, in_tok, out_tok)
                self.cost_tracker.log_call(m_name, provider, in_tok, out_tok, cost, latency_ms)

                return response
            except _HARD_CRASH_EXCEPTIONS as e:
                logger.error(f"Code bug in async LLM pipeline: {type(e).__name__}: {e}")
                raise
            except Exception as e:
                last_exception = e
                m_name = getattr(model, "model_name", getattr(model, "model", "?"))
                err_upper = str(e).upper()
                if "RESOURCE_EXHAUSTED" in err_upper or "429" in err_upper or "TOO_MANY_REQUESTS" in err_upper:
                    self._penalize_model(model)
                    logger.info(f"[RateLimit] {m_name} quota exhausted. Other models on same key still OK.")
                elif "UNAUTHENTICATED" in err_upper or "PERMISSION_DENIED" in err_upper:
                    self._penalize_key(model)
                    logger.warning(f"[AuthError] {m_name} → {type(e).__name__}. Penalizing entire key.")
                else:
                    self._penalize_model(model)
                    logger.warning(f"[Failover] {m_name} → {type(e).__name__}: {e}. Next model...")
                continue

        logger.error(f"Complete LLM Async Failure (All Fallbacks Exhausted)")
        if last_exception:
            raise last_exception
        raise ValueError("No fallbacks available.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("Testing LLM Failover Router...")
    
    router = LLMRouter()
    
    # 1. Normal Test (Should hit Gemini)
    logger.info("Testing prompt (Expect Gemini):")
    res = router.invoke([HumanMessage(content="Say 'Gemini Online' if you are gemini. Otherwise say your model name.")])
    print(res.content)
    
    # 2. Forced Failover Test
    # We deliberately break Gemini by overriding its model name to a non-existent one
    logger.info("\nSimulating Gemini API Outage (Expect Fallback to Groq/OpenRouter)...")
    if router.gemini_models_by_key:
        # Break all primary Gemini models temporarily
        original_models = {}
        for key, models in router.gemini_models_by_key.items():
            original_models[key] = []
            for m in models:
                original_models[key].append(m.model)
                m.model = "models/non-existent-broken-model"
        
        try:
            res_failover = router.invoke([HumanMessage(content="Say which alternative model you are. Start with 'I am'")])
            print("Fallback Success Answer:", res_failover.content)
        except Exception as e:
            print(f"Fallback didn't trigger correctly: {e}")
            
        # Restore for normal operation
        for key, models in router.gemini_models_by_key.items():
            for i, m in enumerate(models):
                if hasattr(m, 'model'):
                    m.model = original_models[key][i]
    else:
        logger.info("Only one LLM key is configured. Cannot test fallbacks properly.")

