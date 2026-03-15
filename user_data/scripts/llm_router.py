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
GEMINI_COOLDOWN_DURATION = 60.0  # Gemini: daily quota can take minutes to reset
FALLBACK_COOLDOWN_DURATION = 15.0  # Non-Gemini providers: 15s cooldown (was 5s, caused 1221x 429 on OpenRouter)
_COOLDOWN_LOCK = threading.Lock()  # Thread-safe access to KEY_COOLDOWNS

# Module-level model discovery cache — shared across ALL LLMRouter instances
_GEMINI_MODEL_CACHE: Dict[str, Any] = {"models": None, "timestamp": 0.0}
_OPENROUTER_MODEL_CACHE: Dict[str, Any] = {"models": None, "timestamp": 0.0}
_MODEL_CACHE_TTL = 600.0  # 10 minutes

# Per-model penalty box: "api_key:model_name" → unlock_timestamp
# Gemini quotas are PER-MODEL PER-PROJECT — one model's 429 doesn't affect others on the same key
MODEL_COOLDOWNS: Dict[str, float] = {}

# Gemini-wide circuit breaker: after N consecutive Gemini failures, skip ALL Gemini for a cooldown.
# This prevents wasting 140 API calls (10 keys × 14 models) when Gemini is globally rate-limited.
_GEMINI_CIRCUIT_OPEN_UNTIL: float = 0.0
_CIRCUIT_BREAKER_THRESHOLD = 3  # consecutive Gemini 429s to trip the breaker
_CIRCUIT_BREAKER_DURATION = 120.0  # 2 minutes — give Gemini quotas time to recover

# Exceptions that CANNOT be recovered by failover — bugs in OUR code
# NOTE: ValueError/TypeError removed — they can be triggered by legitimate empty/malformed
# API responses (e.g. json.loads("") → ValueError, float(None) → TypeError).
# These should failover to next model, not crash the pipeline.
_HARD_CRASH_EXCEPTIONS = (KeyError, AttributeError, SyntaxError)

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
    
    def __init__(self, temperature: float = 0.0, request_timeout: int = 30,
                 fallback_timeout: int = 15):
        self.temperature = temperature
        self.request_timeout = request_timeout
        self.fallback_timeout = fallback_timeout  # Shorter timeout for non-Gemini providers
        self.cost_tracker = LLMCostTracker()
        self._key_lock = threading.Lock()  # Thread-safe key rotation
        self._provider_map = {}  # id(model) → provider name for cost tracking

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
                    m = ChatGoogleGenerativeAI(
                        model=m_name,
                        api_key=key,
                        temperature=self.temperature,
                        timeout=self.request_timeout,
                        max_retries=1  # MUST be 1 not 0: SDK bug treats 0 as "use default 5 retries"
                    )
                    models_for_key.append(m)
                    self._provider_map[id(m)] = "gemini"
                self.gemini_models_by_key[key] = models_for_key

            logger.info(f"Loaded {len(self.gemini_keys)} Gemini keys × {len(gemini_model_names)} models. "
                         f"Models: {gemini_model_names}")
            
        # 2. First Fallback: Groq (multiple models — each has INDEPENDENT rate limits)
        # Round-robin across 4 models = 4× throughput on free tier (~120 RPM total)
        # Ordered: smartest → dumbest for best trade signal quality
        self.groq_key = os.environ.get("GROQ_API_KEY")
        self.groq_models = []
        self.fallback_1 = None  # Keep for backward compat checks
        if self.groq_key:
            groq_model_names = [
                "llama-3.3-70b-versatile",       # 30 RPM, 1K RPD, smartest (70B)
                "qwen/qwen3-32b",                # 60 RPM, 1K RPD (32B, strong reasoning)
                "meta-llama/llama-4-scout-17b-16e-instruct",  # 30 RPM, 1K RPD (17B MoE)
                "llama-3.1-8b-instant",         # 30 RPM, 14.4K RPD, fastest but smallest (8B)
            ]
            for m_name in groq_model_names:
                self.groq_models.append(ChatGroq(
                    model=m_name,
                    api_key=self.groq_key,
                    temperature=self.temperature,
                    timeout=self.fallback_timeout,
                    max_retries=0
                ))
            self.fallback_1 = self.groq_models[0]  # backward compat
            logger.info(f"Loaded {len(self.groq_models)} Groq models: {groq_model_names}")
            for m in self.groq_models:
                self._provider_map[id(m)] = "groq"

        # 3. Cerebras (OpenAI-compatible, 30 RPM, 1M tokens/day, ultra-fast inference)
        # Ordered: smartest → dumbest. Cerebras free tier includes Qwen3-235B, GPT-OSS-120B, Llama 3.3-70B
        self.cerebras_key = os.environ.get("CEREBRAS_API_KEY")
        self.cerebras_models = []
        if self.cerebras_key:
            cerebras_model_names = [
                "qwen3-235b-a22b",  # 235B params, strongest reasoning on Cerebras free tier
                "llama3.3-70b",     # 70B params, good general purpose
                "llama3.1-8b",      # 8B params, fastest but smallest
            ]
            for m_name in cerebras_model_names:
                m = ChatOpenAI(
                    base_url="https://api.cerebras.ai/v1",
                    api_key=self.cerebras_key,
                    model=m_name,
                    temperature=self.temperature,
                    timeout=self.fallback_timeout,
                    max_retries=0
                )
                self.cerebras_models.append(m)
                self._provider_map[id(m)] = "cerebras"
            logger.info(f"Loaded {len(self.cerebras_models)} Cerebras models: {cerebras_model_names}")

        # 4. DeepSeek (OpenAI-compatible, 5M free tokens on signup)
        self.deepseek_key = os.environ.get("DEEPSEEK_API_KEY")
        self.deepseek_models = []
        if self.deepseek_key:
            deepseek_model_names = ["deepseek-chat"]  # DeepSeek V3.2
            for m_name in deepseek_model_names:
                m = ChatOpenAI(
                    base_url="https://api.deepseek.com/v1",
                    api_key=self.deepseek_key,
                    model=m_name,
                    temperature=self.temperature,
                    timeout=self.fallback_timeout,
                    max_retries=0
                )
                self.deepseek_models.append(m)
                self._provider_map[id(m)] = "deepseek"
            logger.info(f"Loaded {len(self.deepseek_models)} DeepSeek models: {deepseek_model_names}")

        # 5. SambaNova (OpenAI-compatible, 200K tokens/day free, Llama 405B access)
        self.sambanova_key = os.environ.get("SAMBANOVA_API_KEY")
        self.sambanova_models = []
        if self.sambanova_key:
            sambanova_model_names = [
                "Meta-Llama-3.3-70B-Instruct",   # 20 RPM
                "Meta-Llama-3.1-8B-Instruct",     # 30 RPM
            ]
            for m_name in sambanova_model_names:
                m = ChatOpenAI(
                    base_url="https://api.sambanova.ai/v1",
                    api_key=self.sambanova_key,
                    model=m_name,
                    temperature=self.temperature,
                    timeout=self.fallback_timeout,
                    max_retries=0
                )
                self.sambanova_models.append(m)
                self._provider_map[id(m)] = "sambanova"
            logger.info(f"Loaded {len(self.sambanova_models)} SambaNova models: {sambanova_model_names}")

        # 6. Mistral (OpenAI-compatible, 2 RPM experiment plan, 1B tokens/month)
        # Ordered: smartest → dumbest (large = 90% frontier perf at 8x less cost)
        self.mistral_key = os.environ.get("MISTRAL_API_KEY")
        self.mistral_models = []
        if self.mistral_key:
            mistral_model_names = [
                "mistral-large-latest",    # Best quality, frontier-class
                "mistral-small-latest",    # Fast, good quality
            ]
            for m_name in mistral_model_names:
                m = ChatOpenAI(
                    base_url="https://api.mistral.ai/v1",
                    api_key=self.mistral_key,
                    model=m_name,
                    temperature=self.temperature,
                    timeout=self.fallback_timeout,
                    max_retries=0
                )
                self.mistral_models.append(m)
                self._provider_map[id(m)] = "mistral"
            logger.info(f"Loaded {len(self.mistral_models)} Mistral models: {mistral_model_names}")

        # 7. OpenRouter (dynamic free models — ultimate fallback)
        self.openrouter_key = os.environ.get("OPENROUTER_API_KEY")
        self.openrouter_models = []
        self.fallback_2 = None  # Keep for backward compat checks
        if self.openrouter_key:
            openrouter_model_names = self._discover_openrouter_free_models(self.openrouter_key)
            for m_name in openrouter_model_names:
                m = ChatOpenAI(
                    base_url="https://openrouter.ai/api/v1",
                    api_key=self.openrouter_key,
                    model=m_name,
                    temperature=self.temperature,
                    timeout=self.fallback_timeout,
                    max_retries=0
                )
                self.openrouter_models.append(m)
                self._provider_map[id(m)] = "openrouter"
            if self.openrouter_models:
                self.fallback_2 = self.openrouter_models[0]  # backward compat
            logger.info(f"Loaded {len(self.openrouter_models)} OpenRouter free models: {openrouter_model_names}")
        
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

        client = None
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
                # Deduplicate: remove aliases and variants that share rate limits
                all_shorts = {n.replace("models/", "") for n in discovered}
                deduped = []
                for name in discovered:
                    short = name.replace("models/", "")
                    # Skip versioned variants (gemini-2.5-flash-001 shares quota with gemini-2.5-flash)
                    base = re.sub(r'-\d{3}$', '', short)
                    if base != short and base in all_shorts:
                        continue
                    # Skip -latest aliases (share rate limits with the model they point to)
                    if short.endswith("-latest"):
                        continue
                    # Skip -customtools variants (same model, same rate limit bucket)
                    if "-customtools" in short:
                        continue
                    # Skip dead/shutdown models
                    if short == "gemini-3-pro-preview":  # Shut down March 9, 2026
                        continue
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
        finally:
            # Close throwaway client to prevent httpx connection pool leak
            if client and hasattr(client, '_api_client'):
                try:
                    client._api_client.close()
                except Exception:
                    pass

    @staticmethod
    def _discover_openrouter_free_models(api_key: str) -> list:
        """Discover currently free models from OpenRouter API. Cached for 10 minutes."""
        FALLBACK_MODELS = [
            "meta-llama/llama-3.3-70b-instruct:free",
            "deepseek/deepseek-chat-v3-0324:free",
            "qwen/qwen3-32b:free",
        ]

        now = time.time()
        if _OPENROUTER_MODEL_CACHE["models"] and (now - _OPENROUTER_MODEL_CACHE["timestamp"]) < _MODEL_CACHE_TTL:
            logger.info(f"Using cached OpenRouter free model list ({len(_OPENROUTER_MODEL_CACHE['models'])} models)")
            return _OPENROUTER_MODEL_CACHE["models"]

        try:
            resp = httpx.get(
                "https://openrouter.ai/api/v1/models",
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=15
            )
            resp.raise_for_status()
            data = resp.json().get("data", [])

            free_models = []
            for m in data:
                pricing = m.get("pricing", {})
                try:
                    prompt_free = float(pricing.get("prompt", "1")) == 0
                    completion_free = float(pricing.get("completion", "1")) == 0
                except (ValueError, TypeError):
                    continue
                if prompt_free and completion_free:
                    model_id = m.get("id", "")
                    if model_id:
                        free_models.append(model_id)

            if free_models:
                # Prioritize well-known, high-quality model families
                priority_keywords = ["deepseek", "llama", "qwen", "nvidia", "gemini", "mistral", "step"]
                def _sort_key(mid):
                    for i, kw in enumerate(priority_keywords):
                        if kw in mid.lower():
                            return (i, mid)
                    return (len(priority_keywords), mid)
                free_models.sort(key=_sort_key)
                free_models = free_models[:6]  # Cap at 6 to avoid excessive retries (was 12, caused 1221x 429)

                logger.info(f"Discovered {len(free_models)} free OpenRouter models: {free_models}")
                _OPENROUTER_MODEL_CACHE["models"] = free_models
                _OPENROUTER_MODEL_CACHE["timestamp"] = now
                return free_models

            logger.warning("No free models found on OpenRouter. Using fallback list.")
            return FALLBACK_MODELS
        except Exception as e:
            logger.warning(f"OpenRouter model discovery failed: {e}. Using fallback list.")
            return FALLBACK_MODELS

    def _get_chain(self) -> List[Any]:
        """Constructs and returns the active LangChain fallbacks without rate-limited keys/models."""
        chain = []
        current_time = time.time()

        # Circuit breaker: if Gemini is globally down, skip ALL Gemini models entirely
        gemini_circuit_open = current_time < _GEMINI_CIRCUIT_OPEN_UNTIL

        if self.gemini_keys and not gemini_circuit_open:
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

                # Filter out individually penalized models — don't add dead weight to chain
                active_models = [m for m in self.gemini_models_by_key[key]
                                 if not self._is_model_penalized(m)]
                chain.extend(active_models)

        elif gemini_circuit_open:
            remaining = _GEMINI_CIRCUIT_OPEN_UNTIL - current_time
            logger.info(f"[CircuitBreaker] Gemini circuit OPEN — skipping all Gemini models. "
                        f"Closes in {remaining:.0f}s.")

        # Fallback chain: ordered by model intelligence for best trade signal quality
        # DeepSeek V3.2 (685B MoE) → Mistral Large → Cerebras (Qwen3-235B+) → Groq (70B+) → SambaNova → OpenRouter
        chain.extend(self.deepseek_models)
        chain.extend(self.mistral_models)
        chain.extend(self.cerebras_models)
        chain.extend(self.groq_models)
        chain.extend(self.sambanova_models)
        chain.extend(self.openrouter_models)

        if not chain:
            raise ValueError("All API keys are exhausted or unavailable.")

        return chain
            
    def _cooldown_for(self, model) -> float:
        """Return the appropriate cooldown duration based on provider type."""
        if isinstance(model, ChatGoogleGenerativeAI):
            return GEMINI_COOLDOWN_DURATION
        return FALLBACK_COOLDOWN_DURATION

    def _penalize_key(self, model):
        """Put a Gemini API key into the penalty box if the model has one."""
        if hasattr(model, "google_api_key") and model.google_api_key is not None:
            failed_key = getattr(model, "google_api_key")
            if hasattr(failed_key, "get_secret_value"):
                failed_key = failed_key.get_secret_value()
            duration = self._cooldown_for(model)
            with _COOLDOWN_LOCK:
                if failed_key not in KEY_COOLDOWNS or time.time() > KEY_COOLDOWNS[failed_key]:
                    KEY_COOLDOWNS[failed_key] = time.time() + duration
                    logger.warning(f"Key penalized for {duration}s.")

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
        if hasattr(model, "google_api_key") and model.google_api_key is not None:
            api_key = getattr(model, "google_api_key")
            if hasattr(api_key, "get_secret_value"):
                api_key = api_key.get_secret_value()
        m_name = getattr(model, "model_name", getattr(model, "model", "unknown"))
        penalty_key = f"{api_key}:{m_name}"
        duration = self._cooldown_for(model)
        with _COOLDOWN_LOCK:
            if penalty_key not in MODEL_COOLDOWNS or time.time() > MODEL_COOLDOWNS[penalty_key]:
                MODEL_COOLDOWNS[penalty_key] = time.time() + duration
                logger.warning(f"[Penalize] {m_name} penalized {duration}s. Other models on this key still active.")

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

    def _check_key_cascade(self, model):
        """If ALL models on a key are penalized, penalize the KEY too.
        This lets _get_chain() skip the entire key instead of iterating through dead models."""
        if not hasattr(model, "google_api_key") or model.google_api_key is None:
            return
        key_val = model.google_api_key
        if hasattr(key_val, "get_secret_value"):
            key_val = key_val.get_secret_value()
        if key_val not in self.gemini_models_by_key:
            return
        now = time.time()
        with _COOLDOWN_LOCK:
            all_penalized = all(
                now < MODEL_COOLDOWNS.get(
                    f"{key_val}:{getattr(m, 'model_name', getattr(m, 'model', 'unknown'))}", 0
                )
                for m in self.gemini_models_by_key[key_val]
            )
            if all_penalized and (key_val not in KEY_COOLDOWNS or now > KEY_COOLDOWNS[key_val]):
                duration = self._cooldown_for(model)
                KEY_COOLDOWNS[key_val] = now + duration
                logger.warning(f"[KeyCascade] All models on key ...{key_val[-4:]} exhausted. "
                               f"Key penalized for {duration}s.")

    def invoke(self, messages: List[Any], temperature: Optional[float] = None,
              max_wall_time: float = 90.0, **kwargs):
        """Synchronously invokes the custom failover loop with Cooldown tracking.

        Args:
            temperature: Optional per-call temperature override. Uses model.bind()
                         which creates a lightweight wrapper (no new connections).
            max_wall_time: Maximum total seconds for the entire failover chain (default 90s).
        """
        global _GEMINI_CIRCUIT_OPEN_UNTIL
        chain = self._get_chain()
        last_exception: Optional[Exception] = None
        gemini_consecutive_fails = 0
        wall_start = time.time()

        for model in chain:
            # Wall-time budget: abort failover if total time exceeds limit
            if time.time() - wall_start > max_wall_time:
                logger.warning(f"[WallTime] Exceeded {max_wall_time}s total across failover chain. Aborting.")
                break

            # Skip if key is dead (auth/conn) or this specific model is rate-limited
            if self._is_key_penalized(model) or self._is_model_penalized(model):
                continue

            is_gemini = isinstance(model, ChatGoogleGenerativeAI)

            # Circuit breaker: too many consecutive Gemini failures in THIS call → skip remaining
            if is_gemini and gemini_consecutive_fails >= _CIRCUIT_BREAKER_THRESHOLD:
                continue

            try:
                # Per-call temperature override via LangChain bind() — thread-safe, zero allocation
                # Use generation_config for Gemini models (ChatGoogleGenerativeAI)
                # Use temperature kwarg for Groq/OpenRouter (ChatGroq/ChatOpenAI)
                if temperature is not None:
                    if isinstance(model, ChatGoogleGenerativeAI):
                        target = model.bind(generation_config={"temperature": temperature})
                    else:
                        target = model.bind(temperature=temperature)
                else:
                    target = model
                start_time = time.time()
                response = target.invoke(messages, **kwargs)
                latency_ms = (time.time() - start_time) * 1000

                # Validate response has actual content — empty responses should failover
                content = getattr(response, 'content', None)
                if content is None or (isinstance(content, str) and not content.strip()):
                    m_name = getattr(model, "model_name", getattr(model, "model", "?"))
                    logger.warning(f"[EmptyResponse] {m_name} returned empty content. Trying next model...")
                    self._penalize_model(model)
                    if is_gemini:
                        gemini_consecutive_fails += 1
                    continue

                in_tok = 0
                out_tok = 0
                if hasattr(response, 'usage_metadata') and response.usage_metadata:
                    in_tok = response.usage_metadata.get('input_tokens', 0)
                    out_tok = response.usage_metadata.get('output_tokens', 0)

                m_name = getattr(model, "model_name", getattr(model, "model", "unknown_model"))
                provider = self._provider_map.get(id(model), "unknown")
                cost = self.cost_tracker.calculate_cost(m_name, in_tok, out_tok)
                self.cost_tracker.log_call(m_name, provider, in_tok, out_tok, cost, latency_ms)

                # Success — reset circuit breaker if Gemini recovered
                if is_gemini and _GEMINI_CIRCUIT_OPEN_UNTIL > 0:
                    _GEMINI_CIRCUIT_OPEN_UNTIL = 0.0
                    logger.info("[CircuitBreaker] Gemini success — circuit CLOSED.")

                return response
            except _HARD_CRASH_EXCEPTIONS as e:
                logger.error(f"Code bug in LLM pipeline: {type(e).__name__}: {e}")
                raise
            except Exception as e:
                last_exception = e
                m_name = getattr(model, "model_name", getattr(model, "model", "?"))
                err_upper = str(e).upper()
                if "RESOURCE_EXHAUSTED" in err_upper or "429" in err_upper or "TOO_MANY_REQUESTS" in err_upper:
                    # Rate limit: only THIS model's quota is exhausted
                    self._penalize_model(model)
                    self._check_key_cascade(model)
                    logger.info(f"[RateLimit] {m_name} quota exhausted. Other models on same key still OK.")
                elif "UNAUTHENTICATED" in err_upper or "PERMISSION_DENIED" in err_upper:
                    # Auth error: entire key is broken
                    self._penalize_key(model)
                    logger.warning(f"[AuthError] {m_name} → {type(e).__name__}. Penalizing entire key.")
                else:
                    # Unknown/other error: penalize model only (don't kill working models on same key)
                    self._penalize_model(model)
                    logger.warning(f"[Failover] {m_name} → {type(e).__name__}: {e}. Next model...")

                if is_gemini:
                    gemini_consecutive_fails += 1
                    # Trip circuit breaker for ALL subsequent invoke() calls too
                    if gemini_consecutive_fails >= _CIRCUIT_BREAKER_THRESHOLD and _GEMINI_CIRCUIT_OPEN_UNTIL < time.time():
                        _GEMINI_CIRCUIT_OPEN_UNTIL = time.time() + _CIRCUIT_BREAKER_DURATION
                        logger.warning(f"[CircuitBreaker] {gemini_consecutive_fails} consecutive Gemini failures — "
                                       f"circuit OPEN for {_CIRCUIT_BREAKER_DURATION}s. Skipping to fallbacks.")
                continue

        logger.error(f"Complete LLM Failure (All Fallbacks Exhausted)")
        if last_exception:
            raise last_exception
        raise ValueError("No fallbacks available.")

    async def ainvoke(self, messages: List[Any], temperature: Optional[float] = None,
                      max_wall_time: float = 90.0, **kwargs):
        """Asynchronously invokes the custom failover loop with Cooldown tracking."""
        global _GEMINI_CIRCUIT_OPEN_UNTIL
        chain = self._get_chain()
        last_exception: Optional[Exception] = None
        gemini_consecutive_fails = 0
        wall_start = time.time()

        for model in chain:
            # Wall-time budget: abort failover if total time exceeds limit
            if time.time() - wall_start > max_wall_time:
                logger.warning(f"[WallTime] Exceeded {max_wall_time}s total across failover chain. Aborting.")
                break

            # Skip if key is dead (auth/conn) or this specific model is rate-limited
            if self._is_key_penalized(model) or self._is_model_penalized(model):
                continue

            is_gemini = isinstance(model, ChatGoogleGenerativeAI)

            # Circuit breaker: too many consecutive Gemini failures in THIS call → skip remaining
            if is_gemini and gemini_consecutive_fails >= _CIRCUIT_BREAKER_THRESHOLD:
                continue

            try:
                if temperature is not None:
                    if isinstance(model, ChatGoogleGenerativeAI):
                        target = model.bind(generation_config={"temperature": temperature})
                    else:
                        target = model.bind(temperature=temperature)
                else:
                    target = model
                start_time = time.time()
                response = await target.ainvoke(messages, **kwargs)
                latency_ms = (time.time() - start_time) * 1000

                # Validate response has actual content — empty responses should failover
                content = getattr(response, 'content', None)
                if content is None or (isinstance(content, str) and not content.strip()):
                    m_name = getattr(model, "model_name", getattr(model, "model", "?"))
                    logger.warning(f"[EmptyResponse] {m_name} returned empty content. Trying next model...")
                    self._penalize_model(model)
                    if is_gemini:
                        gemini_consecutive_fails += 1
                    continue

                in_tok = 0
                out_tok = 0
                if hasattr(response, 'usage_metadata') and response.usage_metadata:
                    in_tok = response.usage_metadata.get('input_tokens', 0)
                    out_tok = response.usage_metadata.get('output_tokens', 0)

                m_name = getattr(model, "model_name", getattr(model, "model", "unknown_model"))
                provider = self._provider_map.get(id(model), "unknown")
                cost = self.cost_tracker.calculate_cost(m_name, in_tok, out_tok)
                self.cost_tracker.log_call(m_name, provider, in_tok, out_tok, cost, latency_ms)

                # Success — reset circuit breaker if Gemini recovered
                if is_gemini and _GEMINI_CIRCUIT_OPEN_UNTIL > 0:
                    _GEMINI_CIRCUIT_OPEN_UNTIL = 0.0
                    logger.info("[CircuitBreaker] Gemini success — circuit CLOSED.")

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
                    self._check_key_cascade(model)
                    logger.info(f"[RateLimit] {m_name} quota exhausted. Other models on same key still OK.")
                elif "UNAUTHENTICATED" in err_upper or "PERMISSION_DENIED" in err_upper:
                    self._penalize_key(model)
                    logger.warning(f"[AuthError] {m_name} → {type(e).__name__}. Penalizing entire key.")
                else:
                    self._penalize_model(model)
                    logger.warning(f"[Failover] {m_name} → {type(e).__name__}: {e}. Next model...")

                if is_gemini:
                    gemini_consecutive_fails += 1
                    if gemini_consecutive_fails >= _CIRCUIT_BREAKER_THRESHOLD and _GEMINI_CIRCUIT_OPEN_UNTIL < time.time():
                        _GEMINI_CIRCUIT_OPEN_UNTIL = time.time() + _CIRCUIT_BREAKER_DURATION
                        logger.warning(f"[CircuitBreaker] {gemini_consecutive_fails} consecutive Gemini failures — "
                                       f"circuit OPEN for {_CIRCUIT_BREAKER_DURATION}s. Skipping to fallbacks.")
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

