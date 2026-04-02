"""
Phase 22: Self-Learning Adaptive LLM Router with Thompson Sampling

Replaces static priority failover with a quality-driven multi-armed bandit.
Each model+key combo is a "slot" with a Beta distribution that learns from outcomes.
Models that produce quality responses rise; models that fail sink — automatically.

Philosophy: Quality > Speed. A fast but dumb model is a FAILURE.
"""
import os
import re
import time
import random
import logging
import sqlite3
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from llm_cost_tracker import LLMCostTracker

import httpx
from google.api_core import exceptions as google_exc
import openai
import groq

try:
    from google.genai import errors as genai_errors
    _GENAI_FAILOVER = (genai_errors.ClientError, genai_errors.ServerError)
except ImportError:
    _GENAI_FAILOVER = ()

load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))
logger = logging.getLogger(__name__)

# Module-level model discovery cache (shared across instances)
_GEMINI_MODEL_CACHE: Dict[str, Any] = {"models": None, "timestamp": 0.0}
_OPENROUTER_MODEL_CACHE: Dict[str, Any] = {"models": None, "timestamp": 0.0}
_MODEL_CACHE_TTL = 600.0

# Exceptions that indicate OUR code bugs — raise immediately, never failover
_HARD_CRASH = (KeyError, AttributeError, SyntaxError)

# ─── RPM Limits (free tier, per model per key) ───────────────────────
RPM_LIMITS = {
    "gemini-2.5-flash-lite": 15, "gemini-3.1-flash-lite": 15,
    "gemini-2.5-flash": 10, "gemini-3-flash": 5, "gemini-3.1-flash": 5,
    "gemini-2.5-pro": 2, "gemini-3.1-pro": 2,
    "llama-3.3-70b-versatile": 30, "llama-3.1-8b-instant": 30,
    "qwen/qwen3-32b": 60, "meta-llama/llama-4-scout": 30,
    "moonshotai/kimi-k2": 30, "openai/gpt-oss-20b": 15, "openai/gpt-oss-120b": 15,
    "qwen-3-235b": 30, "llama3.1-8b": 30,
    "Meta-Llama-3.3-70B": 20, "Meta-Llama-3.1-8B": 30,
    "mistral-large": 2, "mistral-small": 2,
}

def _lookup_rpm(model_name: str) -> int:
    """Substring match RPM limit. Default 10 for unknowns."""
    for prefix, limit in RPM_LIMITS.items():
        if prefix in model_name:
            return limit
    return 10

# ─── Error Taxonomy ──────────────────────────────────────────────────
PENALTY_CONFIG = {
    "rate_limit":        {"base": 30.0, "exp": True,  "max": 300.0},
    "timeout":           {"base": 15.0, "exp": False, "max": 15.0},
    "overloaded":        {"base": 45.0, "exp": False, "max": 45.0},
    "context_overflow":  {"base": 0.0,  "exp": False, "max": 0.0},   # skip, not penalize
    "auth":              {"base": 0.0,  "exp": False, "max": 0.0},   # permanent disable
    "empty":             {"base": 30.0, "exp": False, "max": 30.0},
    "other":             {"base": 30.0, "exp": False, "max": 60.0},
}

def classify_error(e: Exception) -> str:
    """Classify exception into error taxonomy category."""
    err = str(e).upper()
    if "RESOURCE_EXHAUSTED" in err or "429" in err or "TOO_MANY_REQUESTS" in err:
        return "rate_limit"
    if "DEADLINE_EXCEEDED" in err or "504" in err or isinstance(e, (httpx.TimeoutException,)):
        return "timeout"
    if "503" in err or "SERVICE_UNAVAILABLE" in err or "UNAVAILABLE" in err:
        return "overloaded"
    if ("CONTEXT" in err and "LENGTH" in err) or ("TOKEN" in err and ("LIMIT" in err or "EXCEED" in err)):
        return "context_overflow"
    if "UNAUTHENTICATED" in err or "PERMISSION_DENIED" in err or "401" in err:
        return "auth"
    return "other"


# ─── ModelSlot ───────────────────────────────────────────────────────
@dataclass
class ModelSlot:
    """A single model+key combination with Thompson Sampling state."""
    provider: str
    model_name: str
    model_obj: Any
    api_key: str
    alpha: float = 1.0          # Beta dist success param
    beta_param: float = 1.0     # Beta dist failure param
    rpm_limit: int = 10
    rpm_window: deque = field(default_factory=lambda: deque(maxlen=120))
    penalty_until: float = 0.0
    backoff_level: int = 0
    consecutive_fails: int = 0
    total_calls: int = 0
    success_count: int = 0
    quality_pass_count: int = 0
    disabled: bool = False
    max_context: int = 1_000_000

    def sample(self, exploit: bool = False) -> float:
        """Thompson Sampling draw. exploit=True returns mean (no randomness)."""
        if exploit:
            return self.alpha / (self.alpha + self.beta_param)
        return random.betavariate(max(self.alpha, 0.01), max(self.beta_param, 0.01))

    def is_available(self, now: float) -> bool:
        if self.disabled:
            return False
        if now < self.penalty_until:
            return False
        return self._rpm_ok(now)

    def _rpm_ok(self, now: float) -> bool:
        """Check if sending would exceed 80% of RPM limit."""
        cutoff = now - 60.0
        recent = sum(1 for ts in self.rpm_window if ts > cutoff)
        return recent < self.rpm_limit * 0.8

    def record_success(self, quality: bool = True):
        self.alpha += 1.0 if quality else 0.5
        self.consecutive_fails = 0
        self.backoff_level = 0
        self.total_calls += 1
        self.success_count += 1
        if quality:
            self.quality_pass_count += 1
        self.rpm_window.append(time.time())

    def record_failure(self, error_type: str):
        self.beta_param += 1.0
        self.total_calls += 1
        self.consecutive_fails += 1
        self.rpm_window.append(time.time())

        if error_type == "auth":
            self.disabled = True
            return

        cfg = PENALTY_CONFIG.get(error_type, PENALTY_CONFIG["other"])
        if cfg["base"] <= 0:
            return  # context_overflow: no time penalty

        if cfg["exp"]:
            penalty = min(cfg["base"] * (2 ** min(self.backoff_level, 6)), cfg["max"])
            self.backoff_level += 1
        else:
            penalty = cfg["base"]

        self.penalty_until = time.time() + penalty
        if penalty >= 60:
            logger.warning(f"[Penalize] {self.model_name} penalized {penalty:.0f}s "
                           f"(type={error_type}, backoff_level={self.backoff_level})")

    @property
    def slot_id(self) -> str:
        return f"{self.provider}:{self.model_name}:{self.api_key[-4:]}"


# ─── Global Circuit Breaker (Gemini) ─────────────────────────────────
class GeminiCircuitBreaker:
    """Sliding-window circuit breaker with hysteresis to prevent flapping."""

    def __init__(self, threshold: int = 10, window_s: float = 60.0,
                 min_open_s: float = 30.0, close_after: int = 3):
        self._lock = threading.Lock()
        self._failures: deque = deque()
        self._open_until: float = 0.0
        self._consecutive_ok: int = 0
        self.threshold = threshold
        self.window_s = window_s
        self.min_open_s = min_open_s
        self.close_after = close_after

    def record_failure(self):
        now = time.time()
        with self._lock:
            self._failures.append(now)
            self._consecutive_ok = 0
            # Prune old
            cutoff = now - self.window_s
            while self._failures and self._failures[0] < cutoff:
                self._failures.popleft()
            if len(self._failures) >= self.threshold and now >= self._open_until:
                self._open_until = now + self.min_open_s
                logger.warning(f"[CircuitBreaker] {len(self._failures)} Gemini failures in {self.window_s}s — OPEN for {self.min_open_s}s")

    def record_success(self):
        with self._lock:
            self._consecutive_ok += 1
            if self._consecutive_ok >= self.close_after and time.time() >= self._open_until:
                if self._open_until > 0:
                    logger.info(f"[CircuitBreaker] {self._consecutive_ok} consecutive successes — CLOSED")
                self._open_until = 0.0
                self._failures.clear()

    def is_open(self) -> bool:
        return time.time() < self._open_until


# ─── SQLite Persistence for Thompson Sampling ────────────────────────
class SlotPersistence:
    """Persist Thompson Sampling state across restarts."""

    def __init__(self):
        try:
            from ai_config import AI_DB_PATH
            self.db_path = AI_DB_PATH
        except ImportError:
            self.db_path = os.path.join(os.path.dirname(__file__), "ai_trading.db")
        self._ensure_table()

    def _ensure_table(self):
        try:
            conn = sqlite3.connect(self.db_path, timeout=10)
            conn.execute("""CREATE TABLE IF NOT EXISTS model_slot_stats (
                slot_id TEXT PRIMARY KEY,
                alpha REAL DEFAULT 1.0,
                beta_param REAL DEFAULT 1.0,
                total_calls INTEGER DEFAULT 0,
                success_count INTEGER DEFAULT 0,
                quality_pass_count INTEGER DEFAULT 0,
                last_updated TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now'))
            )""")
            conn.commit()
            conn.close()
        except Exception as e:
            logger.debug(f"[SlotPersistence] Table init skipped: {e}")

    def load_all(self) -> Dict[str, dict]:
        try:
            conn = sqlite3.connect(self.db_path, timeout=10)
            conn.row_factory = sqlite3.Row
            rows = conn.execute("SELECT * FROM model_slot_stats").fetchall()
            conn.close()
            return {r["slot_id"]: dict(r) for r in rows}
        except Exception:
            return {}

    def save_batch(self, slots: List[ModelSlot]):
        try:
            conn = sqlite3.connect(self.db_path, timeout=10)
            for s in slots:
                conn.execute("""INSERT OR REPLACE INTO model_slot_stats
                    (slot_id, alpha, beta_param, total_calls, success_count, quality_pass_count, last_updated)
                    VALUES (?, ?, ?, ?, ?, ?, strftime('%Y-%m-%dT%H:%M:%SZ','now'))""",
                    (s.slot_id, s.alpha, s.beta_param, s.total_calls, s.success_count, s.quality_pass_count))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.debug(f"[SlotPersistence] Save failed: {e}")


# ─── Informative Priors (Cold Start) ─────────────────────────────────
def _cold_start_alpha(provider: str, model_name: str) -> float:
    """Higher alpha = more trusted initially."""
    mn = model_name.lower()
    if "pro" in mn or "70b" in mn or "235b" in mn or "120b" in mn:
        return 3.0  # Large proven models
    if "flash" in mn or "32b" in mn or "20b" in mn:
        return 2.5  # Mid-range workhorses
    if "8b" in mn or "3b" in mn or "small" in mn:
        return 2.0  # Small but fast
    return 2.0


# ═══════════════════════════════════════════════════════════════════════
#  LLMRouter — Drop-in replacement with Thompson Sampling
# ═══════════════════════════════════════════════════════════════════════
class LLMRouter:
    """
    Phase 22: Self-Learning Adaptive LLM Router
    - Thompson Sampling picks the best model based on quality history
    - Sliding window RPM tracking prevents 429s proactively
    - Error taxonomy applies different penalties per error type
    - Global circuit breaker with hysteresis prevents Gemini cascade
    - SQLite persistence survives restarts
    """

    def __init__(self, temperature: float = 0.0, request_timeout: int = 30,
                 fallback_timeout: int = 15):
        self.temperature = temperature
        self.request_timeout = request_timeout
        self.fallback_timeout = fallback_timeout
        self.cost_tracker = LLMCostTracker()
        self._state_lock = threading.Lock()
        self._provider_map: Dict[int, str] = {}

        # ── Core state ──
        self.slots: List[ModelSlot] = []
        self.gemini_circuit = GeminiCircuitBreaker()
        self._persistence = SlotPersistence()
        self._call_counter = 0
        self._last_persist = time.time()
        self._last_decay = time.time()

        # ── Build all provider models → slots ──
        self.gemini_keys: List[str] = []
        self.gemini_models_by_key: Dict[str, list] = {}

        # Collect Gemini keys
        keys_str = os.environ.get("GEMINI_API_KEYS", "")
        if keys_str:
            self.gemini_keys.extend([k.strip() for k in keys_str.split(",") if k.strip()])
        single_key = os.environ.get("GEMINI_API_KEY")
        if single_key and single_key not in self.gemini_keys:
            self.gemini_keys.append(single_key)
        for i in range(1, 11):
            k = os.environ.get(f"GEMINI_API_KEY_{i}")
            if k and k not in self.gemini_keys:
                self.gemini_keys.append(k)
        self.gemini_keys = list(dict.fromkeys(self.gemini_keys))

        # Create Gemini slots
        if self.gemini_keys:
            gemini_model_names = self._discover_gemini_models(self.gemini_keys[0])
            for key in self.gemini_keys:
                models_for_key = []
                for mn in gemini_model_names:
                    m = ChatGoogleGenerativeAI(
                        model=mn, api_key=key, temperature=self.temperature,
                        timeout=self.request_timeout, max_retries=1)
                    models_for_key.append(m)
                    self._provider_map[id(m)] = "gemini"
                    self.slots.append(ModelSlot(
                        provider="gemini", model_name=mn, model_obj=m, api_key=key,
                        rpm_limit=_lookup_rpm(mn), alpha=_cold_start_alpha("gemini", mn)))
                self.gemini_models_by_key[key] = models_for_key
            logger.info(f"Loaded {len(self.gemini_keys)} Gemini keys × {len(gemini_model_names)} models. "
                        f"Models: {gemini_model_names}")

        # Groq
        self.groq_key = os.environ.get("GROQ_API_KEY")
        self.groq_models = []
        self.fallback_1 = None
        if self.groq_key:
            for mn in ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "qwen/qwen3-32b",
                        "meta-llama/llama-4-scout-17b-16e-instruct", "moonshotai/kimi-k2-instruct",
                        "openai/gpt-oss-20b", "openai/gpt-oss-120b"]:
                m = ChatGroq(model=mn, api_key=self.groq_key, temperature=self.temperature,
                             timeout=self.fallback_timeout, max_retries=0)
                self.groq_models.append(m)
                self._provider_map[id(m)] = "groq"
                self.slots.append(ModelSlot(
                    provider="groq", model_name=mn, model_obj=m, api_key=self.groq_key,
                    rpm_limit=_lookup_rpm(mn), alpha=_cold_start_alpha("groq", mn)))
            self.fallback_1 = self.groq_models[0]
            logger.info(f"Loaded {len(self.groq_models)} Groq models")

        # Cerebras
        self.cerebras_key = os.environ.get("CEREBRAS_API_KEY")
        self.cerebras_models = []
        if self.cerebras_key:
            for mn in ["qwen-3-235b-a22b-instruct-2507", "llama3.1-8b"]:
                m = ChatOpenAI(base_url="https://api.cerebras.ai/v1", api_key=self.cerebras_key,
                               model=mn, temperature=self.temperature, timeout=self.fallback_timeout, max_retries=0)
                self.cerebras_models.append(m)
                self._provider_map[id(m)] = "cerebras"
                ctx = 8192 if "8b" in mn else 32768
                self.slots.append(ModelSlot(
                    provider="cerebras", model_name=mn, model_obj=m, api_key=self.cerebras_key,
                    rpm_limit=_lookup_rpm(mn), alpha=_cold_start_alpha("cerebras", mn), max_context=ctx))
            logger.info(f"Loaded {len(self.cerebras_models)} Cerebras models")

        # DeepSeek
        self.deepseek_key = os.environ.get("DEEPSEEK_API_KEY")
        self.deepseek_models = []
        if self.deepseek_key:
            for mn in ["deepseek-chat"]:
                m = ChatOpenAI(base_url="https://api.deepseek.com/v1", api_key=self.deepseek_key,
                               model=mn, temperature=self.temperature, timeout=self.fallback_timeout, max_retries=0)
                self.deepseek_models.append(m)
                self._provider_map[id(m)] = "deepseek"
                self.slots.append(ModelSlot(
                    provider="deepseek", model_name=mn, model_obj=m, api_key=self.deepseek_key,
                    rpm_limit=10, alpha=_cold_start_alpha("deepseek", mn)))
            logger.info(f"Loaded {len(self.deepseek_models)} DeepSeek models")

        # SambaNova
        self.sambanova_key = os.environ.get("SAMBANOVA_API_KEY")
        self.sambanova_models = []
        if self.sambanova_key:
            for mn in ["Meta-Llama-3.3-70B-Instruct", "Meta-Llama-3.1-8B-Instruct"]:
                m = ChatOpenAI(base_url="https://api.sambanova.ai/v1", api_key=self.sambanova_key,
                               model=mn, temperature=self.temperature, timeout=self.fallback_timeout, max_retries=0)
                self.sambanova_models.append(m)
                self._provider_map[id(m)] = "sambanova"
                self.slots.append(ModelSlot(
                    provider="sambanova", model_name=mn, model_obj=m, api_key=self.sambanova_key,
                    rpm_limit=_lookup_rpm(mn), alpha=_cold_start_alpha("sambanova", mn)))
            logger.info(f"Loaded {len(self.sambanova_models)} SambaNova models")

        # Mistral
        self.mistral_key = os.environ.get("MISTRAL_API_KEY")
        self.mistral_models = []
        if self.mistral_key:
            for mn in ["mistral-large-latest", "mistral-small-latest"]:
                m = ChatOpenAI(base_url="https://api.mistral.ai/v1", api_key=self.mistral_key,
                               model=mn, temperature=self.temperature, timeout=self.fallback_timeout, max_retries=0)
                self.mistral_models.append(m)
                self._provider_map[id(m)] = "mistral"
                self.slots.append(ModelSlot(
                    provider="mistral", model_name=mn, model_obj=m, api_key=self.mistral_key,
                    rpm_limit=_lookup_rpm(mn), alpha=_cold_start_alpha("mistral", mn)))
            logger.info(f"Loaded {len(self.mistral_models)} Mistral models")

        # OpenRouter
        self.openrouter_key = os.environ.get("OPENROUTER_API_KEY")
        self.openrouter_models = []
        self.fallback_2 = None
        if self.openrouter_key:
            or_names = self._discover_openrouter_free_models(self.openrouter_key)
            for mn in or_names:
                m = ChatOpenAI(base_url="https://openrouter.ai/api/v1", api_key=self.openrouter_key,
                               model=mn, temperature=self.temperature, timeout=self.fallback_timeout, max_retries=0)
                self.openrouter_models.append(m)
                self._provider_map[id(m)] = "openrouter"
                self.slots.append(ModelSlot(
                    provider="openrouter", model_name=mn, model_obj=m, api_key=self.openrouter_key,
                    rpm_limit=10, alpha=_cold_start_alpha("openrouter", mn)))
            if self.openrouter_models:
                self.fallback_2 = self.openrouter_models[0]
            logger.info(f"Loaded {len(self.openrouter_models)} OpenRouter free models")

        # ── Restore learned state from SQLite ──
        saved = self._persistence.load_all()
        restored = 0
        for slot in self.slots:
            if slot.slot_id in saved:
                d = saved[slot.slot_id]
                slot.alpha = max(d.get("alpha", 1.0), 1.0)
                slot.beta_param = max(d.get("beta_param", 1.0), 1.0)
                slot.total_calls = d.get("total_calls", 0)
                slot.success_count = d.get("success_count", 0)
                slot.quality_pass_count = d.get("quality_pass_count", 0)
                restored += 1
        if restored:
            logger.info(f"[Thompson] Restored learning state for {restored}/{len(self.slots)} slots from SQLite")

    # ── Model Discovery (unchanged from Phase 5.3) ────────────────────

    @staticmethod
    def _discover_gemini_models(api_key: str) -> list:
        """Discover available Gemini chat models from API. Cached for 10 minutes."""
        FALLBACK_MODELS = ["models/gemini-2.5-flash", "models/gemini-2.5-flash-lite-preview-06-17"]
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
                model_short = name.replace("models/", "")
                if not model_short.startswith("gemini-"):
                    continue
                if any(skip in model_short for skip in ['tts', 'robotics', 'image', 'embedding', 'vision', 'audio', 'computer-use']):
                    continue
                discovered.append(name)
            if discovered:
                def _prio(name):
                    short = name.replace("models/", "")
                    if "flash-lite" in short: return (0, short)
                    if "flash" in short: return (1, short)
                    if "lite" in short: return (2, short)
                    if "pro" in short: return (3, short)
                    return (4, short)
                discovered.sort(key=_prio)
                all_shorts = {n.replace("models/", "") for n in discovered}
                deduped = []
                for name in discovered:
                    short = name.replace("models/", "")
                    base = re.sub(r'-\d{3}$', '', short)
                    if base != short and base in all_shorts:
                        continue
                    if short.endswith("-latest") or "-customtools" in short:
                        continue
                    if short == "gemini-3-pro-preview" or short.startswith("gemini-2.0-"):
                        continue
                    _pd = re.search(r'-preview-(\d{2})-(\d{4})$', short)
                    if _pd:
                        _mo, _yr = int(_pd.group(1)), int(_pd.group(2))
                        if _yr < 2026 or (_yr == 2026 and _mo < 2):
                            continue
                    deduped.append(name)
                discovered = deduped
                logger.info(f"Discovered {len(discovered)} Gemini chat models (deduped): {discovered}")
                _GEMINI_MODEL_CACHE["models"] = discovered
                _GEMINI_MODEL_CACHE["timestamp"] = now
                return discovered
            else:
                logger.warning("No Gemini chat models discovered. Using fallback.")
                return FALLBACK_MODELS
        except Exception as e:
            logger.warning(f"Model discovery failed: {e}. Using fallback.")
            return FALLBACK_MODELS
        finally:
            if client and hasattr(client, '_api_client'):
                try:
                    client._api_client.close()
                except Exception:
                    pass

    @staticmethod
    def _discover_openrouter_free_models(api_key: str) -> list:
        """Discover currently free models from OpenRouter API. Cached for 10 minutes."""
        FALLBACK = ["meta-llama/llama-3.3-70b-instruct:free", "deepseek/deepseek-chat-v3-0324:free", "qwen/qwen3-32b:free"]
        now = time.time()
        if _OPENROUTER_MODEL_CACHE["models"] and (now - _OPENROUTER_MODEL_CACHE["timestamp"]) < _MODEL_CACHE_TTL:
            logger.info(f"Using cached OpenRouter free model list ({len(_OPENROUTER_MODEL_CACHE['models'])} models)")
            return _OPENROUTER_MODEL_CACHE["models"]
        try:
            resp = httpx.get("https://openrouter.ai/api/v1/models",
                             headers={"Authorization": f"Bearer {api_key}"}, timeout=15)
            resp.raise_for_status()
            free = []
            for m in resp.json().get("data", []):
                p = m.get("pricing", {})
                try:
                    if float(p.get("prompt", "1")) == 0 and float(p.get("completion", "1")) == 0:
                        mid = m.get("id", "")
                        if mid:
                            free.append(mid)
                except (ValueError, TypeError):
                    continue
            if free:
                kw = ["deepseek", "llama", "qwen", "nvidia", "gemini", "mistral", "step"]
                free.sort(key=lambda mid: next((i for i, k in enumerate(kw) if k in mid.lower()), len(kw)))
                free = free[:6]
                logger.info(f"Discovered {len(free)} free OpenRouter models: {free}")
                _OPENROUTER_MODEL_CACHE["models"] = free
                _OPENROUTER_MODEL_CACHE["timestamp"] = now
                return free
            logger.warning("No free OpenRouter models found. Using fallback.")
            return FALLBACK
        except Exception as e:
            logger.warning(f"OpenRouter discovery failed: {e}. Using fallback.")
            return FALLBACK

    # ── Thompson Sampling Selection ───────────────────────────────────

    def _select_slots(self, priority: Optional[str] = None,
                      estimated_tokens: int = 0) -> List[ModelSlot]:
        """Build ranked candidate list using Thompson Sampling."""
        now = time.time()

        # Hourly decay: adapt to changing conditions
        if now - self._last_decay > 3600:
            with self._state_lock:
                for s in self.slots:
                    s.alpha = max(s.alpha * 0.99, 1.0)
                    s.beta_param = max(s.beta_param * 0.99, 1.0)
                self._last_decay = now

        # Filter to available slots
        circuit_open = self.gemini_circuit.is_open()
        eligible = []
        for s in self.slots:
            if not s.is_available(now):
                continue
            if s.provider == "gemini" and circuit_open:
                continue
            if estimated_tokens > 0 and estimated_tokens > s.max_context:
                continue
            eligible.append(s)

        if not eligible:
            skipped_rpm = sum(1 for s in self.slots if not s.disabled and now >= s.penalty_until and not s._rpm_ok(now))
            skipped_penalty = sum(1 for s in self.slots if not s.disabled and now < s.penalty_until)
            logger.error(f"[SelectSlots] All {len(self.slots)} slots exhausted "
                         f"(penalty={skipped_penalty}, rpm_limit={skipped_rpm}, "
                         f"circuit={'OPEN' if circuit_open else 'closed'})")
            raise ValueError("All providers exhausted (all slots penalized or rate-limited).")

        # Score each slot via Thompson Sampling
        scored = []
        for s in eligible:
            if priority == "critical":
                score = s.sample(exploit=True)   # Best known quality
            elif priority == "low":
                score = s.sample(exploit=False)   # Pure exploration
            else:
                mean = s.alpha / (s.alpha + s.beta_param)
                samp = s.sample(exploit=False)
                score = 0.3 * mean + 0.7 * samp  # Balanced
            scored.append((score, s))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [s for _, s in scored]

    # ── Core Model Call ───────────────────────────────────────────────

    def _try_model(self, slot: ModelSlot, messages: List[Any],
                   temperature: Optional[float], **kwargs) -> Tuple[Optional[Any], Optional[str]]:
        """Try a single model. Returns (response, None) on success or (None, error_type) on failure."""
        model = slot.model_obj
        try:
            if temperature is not None:
                if isinstance(model, ChatGoogleGenerativeAI):
                    target = model.bind(generation_config={"temperature": temperature})
                else:
                    target = model.bind(temperature=temperature)
            else:
                target = model

            start = time.time()
            response = target.invoke(messages, **kwargs)
            latency_ms = (time.time() - start) * 1000

            # Validate non-empty content
            content = getattr(response, 'content', None)
            if content is None or (isinstance(content, str) and not content.strip()):
                return None, "empty"

            # Normalize Gemini list content → string
            if isinstance(content, list):
                parts = []
                for block in content:
                    if isinstance(block, dict) and "text" in block:
                        parts.append(block["text"])
                    elif isinstance(block, str):
                        parts.append(block)
                    else:
                        parts.append(str(block))
                normalized = "".join(parts)
                try:
                    response.content = normalized
                except (AttributeError, TypeError):
                    response = AIMessage(content=normalized,
                                         response_metadata=getattr(response, 'response_metadata', {}))

            # Cost tracking
            in_tok = out_tok = 0
            if hasattr(response, 'usage_metadata') and response.usage_metadata:
                in_tok = response.usage_metadata.get('input_tokens', 0)
                out_tok = response.usage_metadata.get('output_tokens', 0)
            provider = self._provider_map.get(id(model), "unknown")
            cost = self.cost_tracker.calculate_cost(slot.model_name, in_tok, out_tok, provider)
            self.cost_tracker.log_call(slot.model_name, provider, in_tok, out_tok, cost, latency_ms)

            return response, None

        except _HARD_CRASH as e:
            logger.error(f"Code bug in LLM pipeline: {type(e).__name__}: {e}")
            raise
        except Exception as e:
            error_type = classify_error(e)
            tag = "[RateLimit]" if error_type == "rate_limit" else "[Failover]"
            if error_type == "rate_limit":
                logger.info(f"{tag} {slot.model_name} quota exhausted. Other models on same key still OK.")
            elif error_type != "timeout":  # Don't spam logs for timeouts
                logger.warning(f"{tag} {slot.model_name} → {type(e).__name__}: {str(e)[:120]}. Next model...")
            return None, error_type

    # ── Main Invoke ───────────────────────────────────────────────────

    def invoke(self, messages: List[Any], temperature: Optional[float] = None,
               max_wall_time: float = 90.0, priority: Optional[str] = None, **kwargs):
        """Route LLM request using Thompson Sampling. Drop-in compatible."""
        estimated_tokens = sum(len(str(getattr(m, "content", ""))) for m in messages) // 3
        candidates = self._select_slots(priority, estimated_tokens)

        wall_start = time.time()
        last_exception = None

        for slot in candidates:
            elapsed = time.time() - wall_start
            if elapsed > max_wall_time:
                logger.warning(f"[WallTime] Exceeded {max_wall_time}s across failover chain. Aborting.")
                break

            # Re-check availability (may have been penalized during earlier iteration)
            if not slot.is_available(time.time()):
                continue

            response, error_type = self._try_model(slot, messages, temperature, **kwargs)

            if response is not None:
                with self._state_lock:
                    slot.record_success(quality=True)
                if slot.provider == "gemini":
                    self.gemini_circuit.record_success()
                self._maybe_persist()
                return response

            # Failure path
            with self._state_lock:
                slot.record_failure(error_type)
            if slot.provider == "gemini":
                self.gemini_circuit.record_failure()
            last_exception = ValueError(f"{slot.model_name}: {error_type}")

        logger.error("Complete LLM Failure (All Fallbacks Exhausted)")
        if last_exception:
            raise last_exception
        raise ValueError("No fallbacks available.")

    async def ainvoke(self, messages: List[Any], temperature: Optional[float] = None,
                      max_wall_time: float = 90.0, priority: Optional[str] = None, **kwargs):
        """Async wrapper — delegates to sync invoke via executor."""
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, lambda: self.invoke(messages, temperature=temperature,
                                      max_wall_time=max_wall_time, priority=priority, **kwargs))

    # ── Public API ────────────────────────────────────────────────────

    def is_any_provider_available(self) -> bool:
        now = time.time()
        return any(s.is_available(now) for s in self.slots)

    def report_quality(self, model_name: str, quality_pass: bool):
        """Optional: external callers report whether LLM output was actually useful."""
        with self._state_lock:
            for s in self.slots:
                if s.model_name == model_name or model_name in s.model_name:
                    if quality_pass:
                        s.alpha += 0.5
                        s.quality_pass_count += 1
                    else:
                        s.beta_param += 0.5
                    break

    def _maybe_persist(self):
        self._call_counter += 1
        now = time.time()
        if self._call_counter >= 100 or (now - self._last_persist) >= 300:
            self._persistence.save_batch(self.slots)
            self._call_counter = 0
            self._last_persist = now

    def get_slot_stats(self) -> List[dict]:
        """Return stats for all slots (for monitoring/debugging)."""
        return [{"slot_id": s.slot_id, "alpha": round(s.alpha, 2), "beta": round(s.beta_param, 2),
                 "mean": round(s.alpha / (s.alpha + s.beta_param), 3),
                 "calls": s.total_calls, "ok": s.success_count, "quality": s.quality_pass_count,
                 "disabled": s.disabled, "available": s.is_available(time.time())}
                for s in self.slots]


# ── Self-test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("Testing Self-Learning Adaptive LLM Router...")

    router = LLMRouter()
    logger.info(f"Total slots: {len(router.slots)}")
    logger.info(f"Gemini keys: {len(router.gemini_keys)}")

    # Test Thompson Sampling selection
    try:
        candidates = router._select_slots(priority="medium")
        logger.info(f"Thompson selected top 3: {[s.model_name for s in candidates[:3]]}")
    except ValueError as e:
        logger.warning(f"No slots available: {e}")

    # Test invoke
    logger.info("Testing invoke (expect any model):")
    try:
        res = router.invoke([HumanMessage(content="Say your model name in one word.")])
        logger.info(f"Response: {res.content[:100]}")
    except Exception as e:
        logger.error(f"Invoke failed: {e}")

    # Show slot stats
    for stat in router.get_slot_stats()[:5]:
        logger.info(f"  {stat}")
