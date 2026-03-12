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

# Global dictionary to store API Keys currently in Penalty Box: {api_key: unlock_timestamp}
KEY_COOLDOWNS: Dict[str, float] = {}
COOLDOWN_DURATION = 60.0  # seconds
_COOLDOWN_LOCK = threading.Lock()  # Thread-safe access to KEY_COOLDOWNS

# Sadece bu hatalarda (Rate Limit, Server Çokmesi, Timeout, Geçersiz Key) sistem bir sonraki yedek LLM'e atlar.
# Eğer hata '400 Bad Request' (aşırı uzun prompt, bozuk JSON) ise failover yapılmayıp anında çöker ki hatayı görelim.
FAILOVER_EXCEPTIONS = (
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
                        max_retries=0
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
        """Discover available text generation models from Gemini API at runtime."""
        FALLBACK_MODELS = [
            "models/gemini-2.5-flash",
            "models/gemini-2.0-flash",
        ]
        try:
            from google import genai
            client = genai.Client(api_key=api_key)

            discovered = []
            for m in client.models.list():
                name = m.name if hasattr(m, 'name') else str(m)
                actions = m.supported_actions if hasattr(m, 'supported_actions') else []

                # Only text generation models
                if 'generateContent' not in (actions or []):
                    continue
                # Skip non-text models
                if any(skip in name for skip in ['image', 'embedding', 'vision', 'audio', 'computer-use']):
                    continue

                discovered.append(name)

            if discovered:
                # Newest models first (3.x > 2.5 > 2.0)
                discovered.sort(reverse=True)
                logger.info(f"Discovered {len(discovered)} Gemini text models from API")
                return discovered
            else:
                logger.warning("No models discovered from API. Using fallback list.")
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
            
    def invoke(self, messages: List[Any], **kwargs):
        """Synchronously invokes the custom failover loop with Cooldown tracking."""
        chain = self._get_chain()
        last_exception: Optional[Exception] = None
        
        for model in chain:
            try:
                start_time = time.time()
                response = model.invoke(messages, **kwargs)
                latency_ms = (time.time() - start_time) * 1000
                
                # Token tracking heuristics based on Langchain structure
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
            except Exception as e:
                last_exception = e
                # Check if this error is meant to be skipped
                if isinstance(e, FAILOVER_EXCEPTIONS):
                    # Identify if it's a Gemini model and extract the precise rate-limited key
                    if hasattr(model, "google_api_key"):
                        failed_key = getattr(model, "google_api_key")
                        if hasattr(failed_key, "get_secret_value"):
                            failed_key = failed_key.get_secret_value()
                            
                        # Put the key in the Penalty Box (thread-safe)
                        with _COOLDOWN_LOCK:
                            if failed_key not in KEY_COOLDOWNS or time.time() > KEY_COOLDOWNS[failed_key]:
                                KEY_COOLDOWNS[failed_key] = time.time() + COOLDOWN_DURATION
                                logger.warning(f"Key Rate-Limited! Sniping key into Penalty Box for {COOLDOWN_DURATION}s.")
                            
                    logger.info(f"LLM Node Failed ({type(e).__name__}). Routing to next available node...")
                    continue
                else:
                    # It's a structural error (e.g. 400 Bad Request, context length, bad JSON). Crash intentionally.
                    logger.error(f"Unrecoverable LLM Error! Halting failover chain: {e}")
                    raise e
                    
        logger.error(f"Complete LLM Failure (All Fallbacks Exhausted)")
        if last_exception:
            raise last_exception
        raise ValueError("No fallbacks available.")

    async def ainvoke(self, messages: List[Any], **kwargs):
        """Asynchronously invokes the custom failover loop with Cooldown tracking."""
        chain = self._get_chain()
        last_exception: Optional[Exception] = None
        
        for model in chain:
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
            except Exception as e:
                last_exception = e
                if isinstance(e, FAILOVER_EXCEPTIONS):
                    if hasattr(model, "google_api_key"):
                        failed_key = getattr(model, "google_api_key")
                        if hasattr(failed_key, "get_secret_value"):
                            failed_key = failed_key.get_secret_value()
                            
                        with _COOLDOWN_LOCK:
                            if failed_key not in KEY_COOLDOWNS or time.time() > KEY_COOLDOWNS[failed_key]:
                                KEY_COOLDOWNS[failed_key] = time.time() + COOLDOWN_DURATION
                                logger.warning(f"Key Rate-Limited! Sniping key into Penalty Box for {COOLDOWN_DURATION}s.")
                            
                    logger.info(f"LLM Node Async Failed ({type(e).__name__}). Routing to next available node...")
                    continue
                else:
                    logger.error(f"Unrecoverable LLM Error! Halting failover chain: {e}")
                    raise e
                    
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

