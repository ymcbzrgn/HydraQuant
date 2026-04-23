"""
Phase 22: Self-Learning Adaptive LLM Router with Thompson Sampling

Replaces static priority failover with a quality-driven multi-armed bandit.
Each model+key combo is a "slot" with a Beta distribution that learns from outcomes.
Models that produce quality responses rise; models that fail sink — automatically.

Philosophy: Quality > Speed. A fast but dumb model is a FAILURE.
"""
import math
import os
import pickle
import re
import time
import random
import logging
import sqlite3
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from llm_cost_tracker import LLMCostTracker

# Phase 24: Neural Organism — adaptive parameters
try:
    from neural_organism import _p
except ImportError:
    def _p(param_id, fallback=0.5, regime="_global"):
        return fallback

import httpx
from google.api_core import exceptions as google_exc
import openai
import groq
from db import get_connection, get_db_connection

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

# Phase 27 Task 18 (G5 Ajani): daily request budgets. Free tier ceilings per
# provider — Gemini 250 RPD/key, Groq 14 400 RPD, Mistral generous. We cap
# actual usage at 80% of RPD so spikes don't blow the window entirely.
RPD_LIMITS = {
    "gemini-2.5-flash-lite": 1000, "gemini-3.1-flash-lite": 1000,
    "gemini-2.5-flash": 250,       "gemini-3-flash": 250,       "gemini-3.1-flash": 250,
    "gemini-2.5-pro": 100,         "gemini-3.1-pro": 100,
    "llama-3.3-70b-versatile": 14400, "llama-3.1-8b-instant": 14400,
    "qwen/qwen3-32b": 1000, "meta-llama/llama-4-scout": 1000,
    "moonshotai/kimi-k2": 1000, "openai/gpt-oss-20b": 1000,
    "openai/gpt-oss-120b": 1000,
    "qwen-3-235b": 1700, "llama3.1-8b": 14400,
    "Meta-Llama-3.3-70B": 14400, "Meta-Llama-3.1-8B": 14400,
    "mistral-large": 86000, "mistral-small": 86000,
}


def _lookup_rpd(model_name: str) -> int:
    """Substring match RPD ceiling. 500/day default for unknown models."""
    for prefix, limit in RPD_LIMITS.items():
        if prefix in model_name:
            return limit
    return 500


def _lookup_rpm(model_name: str) -> int:
    """Substring match RPM limit. Default 10 for unknowns."""
    for prefix, limit in RPM_LIMITS.items():
        if prefix in model_name:
            return limit
    return 10

# ─── Error Taxonomy ──────────────────────────────────────────────────
def _get_penalty_config():
    """Phase 24: Adaptive penalty config from Neural Organism."""
    return {
        "rate_limit":        {"base": _p("llm.penalty.rate_limit_base", 30.0), "exp": True,  "max": _p("llm.penalty.rate_limit_max", 300.0)},
        "timeout":           {"base": _p("llm.penalty.timeout_base", 15.0),    "exp": False, "max": _p("llm.penalty.timeout_base", 15.0)},
        "overloaded":        {"base": _p("llm.penalty.overloaded_base", 45.0), "exp": False, "max": _p("llm.penalty.overloaded_base", 45.0)},
        "context_overflow":  {"base": 0.0,  "exp": False, "max": 0.0},
        "auth":              {"base": 0.0,  "exp": False, "max": 0.0},
        "empty":             {"base": _p("llm.penalty.empty_base", 30.0),      "exp": False, "max": _p("llm.penalty.empty_base", 30.0)},
        "other":             {"base": _p("llm.penalty.empty_base", 30.0),      "exp": False, "max": 60.0},
    }
PENALTY_CONFIG = {
    "rate_limit":        {"base": 30.0, "exp": True,  "max": 300.0},
    "timeout":           {"base": 15.0, "exp": False, "max": 15.0},
    "overloaded":        {"base": 45.0, "exp": False, "max": 45.0},
    "context_overflow":  {"base": 0.0,  "exp": False, "max": 0.0},
    "auth":              {"base": 0.0,  "exp": False, "max": 0.0},
    "empty":             {"base": 30.0, "exp": False, "max": 30.0},
    "other":             {"base": 30.0, "exp": False, "max": 60.0},
}  # static fallback

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


# ─── DailyQuota ──────────────────────────────────────────────────────
# Phase 27 Task 18: per-slot daily request budget. Each call stamps a UTC
# date bucket; reset automatically on the first call of a new day.
@dataclass
class DailyQuota:
    rpd_limit: int = 500
    _bucket_date: str = ""
    _calls_today: int = 0

    def _today(self) -> str:
        from datetime import datetime as _dt, timezone as _tz
        return _dt.now(tz=_tz.utc).strftime("%Y-%m-%d")

    def stamp(self) -> None:
        today = self._today()
        if today != self._bucket_date:
            self._bucket_date = today
            self._calls_today = 0
        self._calls_today += 1

    def is_within_budget(self, reserve_pct: float = 0.2) -> bool:
        today = self._today()
        if today != self._bucket_date:
            return True  # day rolled over, quota resets on next stamp
        ceiling = int(self.rpd_limit * (1 - reserve_pct))
        return self._calls_today < ceiling

    def remaining(self) -> int:
        today = self._today()
        if today != self._bucket_date:
            return self.rpd_limit
        return max(0, self.rpd_limit - self._calls_today)


def compute_llm_reward(quality_ok: bool, latency_ms: float,
                        outcome_pnl: Optional[float] = None) -> float:
    """Collapse (quality, latency, trade outcome) into a scalar LinUCB reward.

    * quality_ok → 1.0 when the response parsed and was usable, 0 otherwise.
      A failed call is ZERO reward by definition; we never want LinUCB to
      push traffic toward a slot that is generating garbage.
    * latency → exponential decay with a 2 s characteristic time. 500 ms
      calls retain ~78% of quality_ok; 5 s calls ~8%; 10 s calls ~0.6%.
    * outcome_pnl → small retroactive nudge (1.2 if the trade that used
      this call made money, 0.8 if it lost, 1.0 if unknown/None).
    """
    r_quality = 1.0 if quality_ok else 0.0
    r_latency = math.exp(-max(0.0, float(latency_ms)) / 2000.0)
    if outcome_pnl is None:
        r_outcome = 1.0
    elif outcome_pnl > 0:
        r_outcome = 1.2
    elif outcome_pnl < 0:
        r_outcome = 0.8
    else:
        r_outcome = 1.0
    return r_quality * r_latency * r_outcome


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
    # Phase 27 Task 18 additions
    rpd_limit: int = 500
    daily_quota: Optional[DailyQuota] = None
    # Mega Sprint 2026-04-23: latency-aware selection + circuit breaker
    avg_latency_ms: float = 500.0
    p95_latency_ms: float = 5000.0
    consecutive_failures: int = 0
    blacklisted_until: float = 0.0
    # Tur-2 (L2): rolling latency sample ring for percentile computation.
    # default_factory=None so pickled/restored slots still get a buffer.
    _latency_samples: Optional[deque] = None

    # EK Sprint 2026-04-23 (EK.2.2): LinUCB contextual-bandit state. The
    # 5×5 covariance A and 5-dim reward vector b are initialised to the
    # ridge-regression identity prior so new slots start with an uninformed
    # but well-conditioned posterior.
    linucb_A: np.ndarray = field(default_factory=lambda: np.eye(5, dtype=np.float64))
    linucb_b: np.ndarray = field(default_factory=lambda: np.zeros(5, dtype=np.float64))
    linucb_n_updates: int = 0

    def __post_init__(self):
        if self.daily_quota is None:
            self.daily_quota = DailyQuota(rpd_limit=self.rpd_limit)
        if not isinstance(self.linucb_A, np.ndarray):
            self.linucb_A = np.eye(5, dtype=np.float64)
        if not isinstance(self.linucb_b, np.ndarray):
            self.linucb_b = np.zeros(5, dtype=np.float64)

    def linucb_score(self, x: np.ndarray, alpha: float = 1.5) -> float:
        """Upper-confidence-bound score: mean + alpha·sqrt(uncertainty)."""
        try:
            A_inv = np.linalg.inv(self.linucb_A)
        except np.linalg.LinAlgError:
            return 0.5
        theta = A_inv @ self.linucb_b
        mean = float(theta @ x)
        try:
            bonus = float(alpha * np.sqrt(max(float(x @ A_inv @ x), 0.0)))
        except Exception:
            bonus = 0.0
        return mean + bonus

    def linucb_update(self, x: np.ndarray, reward: float) -> None:
        """Apply a (feature, reward) observation to the posterior."""
        self.linucb_A += np.outer(x, x)
        self.linucb_b += float(reward) * x
        self.linucb_n_updates += 1

    def record_latency(self, ms: float) -> None:
        # Tur-2 (L2): EMA for the mean, rolling 100-sample window for p95.
        # The old decay-based p95 (0.98 * prev) drifted slowly and never
        # saw the actual distribution — it just chased the most recent
        # outlier. A deque is O(1) append and lets numpy compute the true
        # percentile once we have ≥20 samples.
        self.avg_latency_ms = 0.9 * self.avg_latency_ms + 0.1 * ms
        if self._latency_samples is None:
            self._latency_samples = deque(maxlen=100)
        self._latency_samples.append(float(ms))
        if len(self._latency_samples) >= 20:
            self.p95_latency_ms = float(
                np.percentile(list(self._latency_samples), 95)
            )

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
        if now < self.blacklisted_until:
            return False
        # Phase 27 Task 18: enforce daily budget with 20% reserve headroom
        if not self.daily_quota.is_within_budget(reserve_pct=0.2):
            return False
        return self._rpm_ok(now)

    def _rpm_ok(self, now: float) -> bool:
        """Check if sending would exceed 80% of RPM limit."""
        cutoff = now - 60.0
        recent = sum(1 for ts in self.rpm_window if ts > cutoff)
        return recent < self.rpm_limit * 0.8

    def record_success(self, quality: bool = True, latency_ms: Optional[float] = None,
                       task_context: Optional[Dict[str, Any]] = None,
                       outcome_pnl: Optional[float] = None):
        # Phase 27 Task 18 (G5, Sutton-Barto style): discounted Thompson Sampling.
        # Phase 26 used hourly batch `alpha *= 0.99` which made recent calls
        # invisible once total_calls grew large. Per-call discount `alpha =
        # 0.997·alpha + reward` keeps the effective window to ~230 calls so
        # the sampler tracks drift in quality across the day.
        gamma = 0.997
        reward = 1.0 if quality else 0.5
        self.alpha = gamma * self.alpha + reward
        self.alpha = max(self.alpha, 0.01)
        self.consecutive_fails = 0
        self.consecutive_failures = 0  # mega-sprint circuit reset
        self.backoff_level = 0
        self.total_calls += 1
        self.success_count += 1
        if quality:
            self.quality_pass_count += 1
        if latency_ms is not None:
            self.record_latency(float(latency_ms))
        now_ts = time.time()
        self.rpm_window.append(now_ts)
        self.daily_quota.stamp()

        # EK Sprint 2026-04-23 (EK.2.6): LinUCB update with the reward from
        # this call. Only runs when the caller passed a task_context so we
        # stay backwards-compatible with every existing call site.
        if task_context is not None:
            try:
                from llm_features import extract_features
                x = extract_features(task_context)
                reward = compute_llm_reward(
                    quality_ok=bool(quality),
                    latency_ms=float(latency_ms or self.avg_latency_ms),
                    outcome_pnl=outcome_pnl,
                )
                self.linucb_update(x, reward)
            except Exception:
                pass

    def record_failure(self, error_type: str,
                        task_context: Optional[Dict[str, Any]] = None,
                        latency_ms: Optional[float] = None):
        # Phase 27 Task 18: same discounted update for the failure arm.
        gamma = 0.997
        self.beta_param = gamma * self.beta_param + 1.0
        self.beta_param = max(self.beta_param, 0.01)
        self.total_calls += 1
        self.consecutive_fails += 1
        self.consecutive_failures += 1  # mega-sprint circuit counter
        # Mega Sprint 2026-04-23: open circuit after 5 consecutive failures.
        # Cooldown doubles per subsequent failure (15s → 30s → 60s → 120s →
        # 300s cap) so a struggling provider cannot hold the entire failover
        # chain hostage.
        if self.consecutive_failures >= 5:
            cooldown = min(300, 15 * (2 ** (self.consecutive_failures - 5)))
            self.blacklisted_until = time.time() + cooldown
            logger.warning(
                f"[CB] {self.provider}/{self.model_name} blacklisted "
                f"{cooldown}s ({error_type}, n_fails={self.consecutive_failures})"
            )
        self.rpm_window.append(time.time())
        # Phase 27 audit fix: DO NOT stamp the daily quota on rate-limit errors.
        # A 429 means the provider REJECTED the request — the call never
        # actually consumed a daily-quota slot. Stamping here would double-
        # count: RPM penalty cools the slot off the minute, but the phantom
        # RPD stamp would make us think we spent a daily credit we didn't.
        # Other error types (timeout, context_overflow, auth) DID consume
        # bandwidth so they still stamp.
        if error_type != "rate_limit":
            self.daily_quota.stamp()

        if error_type == "auth":
            self.disabled = True
            return

        _pcfg = _get_penalty_config()
        cfg = _pcfg.get(error_type, _pcfg["other"])
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

        # EK Sprint 2026-04-23 (EK.2.6): failures feed LinUCB with reward=0
        # so the bandit learns to avoid slots/contexts where this model fails.
        if task_context is not None:
            try:
                from llm_features import extract_features
                x = extract_features(task_context)
                if latency_ms is not None:
                    self.record_latency(float(latency_ms))
                self.linucb_update(x, 0.0)
            except Exception:
                pass

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
            conn = get_db_connection(self.db_path)
            conn.commit()
            conn.close()
        except Exception as e:
            logger.debug(f"[SlotPersistence] Table init skipped: {e}")

    def load_all(self) -> Dict[str, dict]:
        try:
            conn = get_db_connection(self.db_path)
            rows = conn.execute("SELECT * FROM model_slot_stats").fetchall()
            conn.close()
            return {r["slot_id"]: dict(r) for r in rows}
        except Exception:
            return {}

    def save_batch(self, slots: List[ModelSlot]):
        try:
            conn = get_db_connection(self.db_path)
            for s in slots:
                conn.execute("""INSERT OR REPLACE INTO model_slot_stats
                    (slot_id, alpha, beta_param, total_calls, success_count,
                     quality_pass_count, avg_latency_ms, p95_latency_ms, last_updated)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, strftime('%Y-%m-%dT%H:%M:%SZ','now'))""",
                    (s.slot_id, s.alpha, s.beta_param, s.total_calls, s.success_count,
                     s.quality_pass_count, s.avg_latency_ms, s.p95_latency_ms))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.debug(f"[SlotPersistence] Save failed: {e}")

    def save_linucb_batch(self, slots: List[ModelSlot]) -> int:
        """EK.2.10: persist LinUCB posterior for each slot as pickled numpy
        arrays. Returns the number of rows written."""
        written = 0
        try:
            conn = get_db_connection(self.db_path)
            for s in slots:
                try:
                    a_blob = pickle.dumps(np.asarray(s.linucb_A, dtype=np.float64))
                    b_blob = pickle.dumps(np.asarray(s.linucb_b, dtype=np.float64))
                except Exception:
                    continue
                conn.execute("""INSERT OR REPLACE INTO linucb_state
                    (slot_id, a_blob, b_blob, n_updates, last_updated)
                    VALUES (?, ?, ?, ?, strftime('%Y-%m-%dT%H:%M:%SZ','now'))""",
                    (s.slot_id, a_blob, b_blob, int(s.linucb_n_updates)))
                written += 1
            conn.commit()
            conn.close()
        except Exception as e:
            logger.debug(f"[SlotPersistence] LinUCB save failed: {e}")
        return written

    def load_linucb_all(self) -> Dict[str, Dict[str, Any]]:
        try:
            conn = get_db_connection(self.db_path)
            rows = conn.execute(
                "SELECT slot_id, a_blob, b_blob, n_updates FROM linucb_state"
            ).fetchall()
            conn.close()
        except Exception:
            return {}
        out: Dict[str, Dict[str, Any]] = {}
        for row in rows:
            try:
                out[row["slot_id"]] = {
                    "A": pickle.loads(row["a_blob"]) if row["a_blob"] else None,
                    "b": pickle.loads(row["b_blob"]) if row["b_blob"] else None,
                    "n_updates": int(row["n_updates"] or 0),
                }
            except Exception:
                continue
        return out


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
                        rpm_limit=_lookup_rpm(mn), rpd_limit=_lookup_rpd(mn),
                        alpha=_cold_start_alpha("gemini", mn)))
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
                    rpm_limit=_lookup_rpm(mn), rpd_limit=_lookup_rpd(mn),
                    alpha=_cold_start_alpha("groq", mn)))
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
                    rpm_limit=_lookup_rpm(mn), rpd_limit=_lookup_rpd(mn),
                    alpha=_cold_start_alpha("cerebras", mn), max_context=ctx))
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
                    rpm_limit=_lookup_rpm(mn), rpd_limit=_lookup_rpd(mn),
                    alpha=_cold_start_alpha("sambanova", mn)))
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
                    rpm_limit=_lookup_rpm(mn), rpd_limit=_lookup_rpd(mn),
                    alpha=_cold_start_alpha("mistral", mn)))
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

        # EK Sprint 2026-04-23 (EK.2.10): restore LinUCB posterior on startup
        # so contextual-bandit learning survives `systemctl restart`. Silent
        # no-op when the table is empty or pickle bytes are malformed.
        try:
            self.load_linucb_state()
        except Exception as e:
            logger.debug(f"[LinUCB:Persist] startup restore failed: {e}")

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

    # ── Selection (Thompson + LinUCB) ─────────────────────────────────

    def _adaptive_alpha(self) -> float:
        """Exploration bonus that shrinks as the bandit accumulates samples.

        0-499 updates → 3.0  (pure exploration while every slot is cold)
        500-1999      → linear 3.0 → 1.0 as coverage builds
        2000+         → 0.5  (stable exploit once the posterior is dense)
        """
        total_calls = sum(s.linucb_n_updates for s in self.slots)
        if total_calls < 500:
            return 3.0
        if total_calls < 2000:
            return 3.0 - (2.0 * (total_calls - 500) / 1500.0)
        return 0.5

    def _select_slots(self, priority: Optional[str] = None,
                      estimated_tokens: int = 0,
                      task_context: Optional[Dict[str, Any]] = None) -> List[ModelSlot]:
        """Build ranked candidate list.

        EK Sprint 2026-04-23 (EK.2.5): when contextual-bandit routing is
        enabled we score slots by LinUCB on the caller's task features,
        fall back to Thompson for cold-start slots (n_updates < 20), and
        finally apply the mega-sprint latency penalty as a safety belt.
        """
        now = time.time()

        # Idle-slot decay: per-call discounted TS (record_success/failure) already
        # shrinks α/β by γ=0.997 on every invocation. Slots that haven't been
        # called in the last hour get this gentle extra pull toward the prior
        # (0.995) so they don't retain stale high scores forever. Kept much
        # softer than the Phase 26 0.99 to avoid double-decay on active slots
        # that DID fire in the window.
        if now - self._last_decay > 3600:
            with self._state_lock:
                for s in self.slots:
                    s.alpha = max(s.alpha * 0.995, 1.0)
                    s.beta_param = max(s.beta_param * 0.995, 1.0)
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

        from ai_config import get_flag
        lat_enabled = get_flag("llm_router_latency_weight_enabled", True)
        bandit_enabled = get_flag("llm_contextual_bandit_enabled", True)
        LAT_CAP_MS = 5000.0

        if bandit_enabled and task_context is not None:
            from llm_features import extract_features
            x = extract_features(task_context)
            alpha = self._adaptive_alpha()
            scored = []
            for s in eligible:
                ucb = s.linucb_score(x, alpha=alpha)
                if s.linucb_n_updates < 20:
                    # Cold-start hybrid: blend with Thompson so a never-called
                    # slot still gets a sane score before its posterior builds.
                    thompson_prior = s.sample(exploit=False)
                    ucb = 0.5 * ucb + 0.5 * thompson_prior
                lat_factor = max(0.1, 1.0 - s.avg_latency_ms / 10000.0)
                scored.append((ucb * lat_factor, s))
            scored.sort(key=lambda item: item[0], reverse=True)
            return [s for _, s in scored]

        # Thompson Sampling (legacy path / fallback when flag disabled).
        scored = []
        for s in eligible:
            if priority == "critical":
                quality = s.sample(exploit=True)
            elif priority == "low":
                quality = s.sample(exploit=False)
            else:
                mean = s.alpha / (s.alpha + s.beta_param)
                quality = 0.3 * mean + 0.7 * s.sample(exploit=False)

            if lat_enabled:
                lat_factor = max(0.1, 1.0 - s.avg_latency_ms / LAT_CAP_MS)
                score = quality * (lat_factor ** 2)
            else:
                score = quality
            scored.append((score, s))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [s for _, s in scored]

    # ── Core Model Call ───────────────────────────────────────────────

    def _try_model(self, slot: ModelSlot, messages: List[Any],
                   temperature: Optional[float],
                   agent_name: str = "",
                   pair: str = "",
                   **kwargs) -> Tuple[Optional[Any], Optional[str], float]:
        """Try a single model.

        Returns (response, None, latency_ms) on success or (None, error_type,
        latency_ms) on failure. Latency is surfaced so invoke() can update the
        slot's latency EMA — previously it was computed and thrown away.

        Revize Tur-2 (C2, M7): `agent_name`/`pair` populate llm_calls for
        the retroactive LinUCB feedback path, and the duplicate `start`
        timer from the pre-sprint code is collapsed into a single top-of-
        function start so latency is measured end-to-end.
        """
        model = slot.model_obj
        start = time.time()
        try:
            if temperature is not None:
                if isinstance(model, ChatGoogleGenerativeAI):
                    target = model.bind(generation_config={"temperature": temperature})
                else:
                    target = model.bind(temperature=temperature)
            else:
                target = model

            response = target.invoke(messages, **kwargs)
            latency_ms = (time.time() - start) * 1000

            # Validate non-empty content
            content = getattr(response, 'content', None)
            if content is None or (isinstance(content, str) and not content.strip()):
                return None, "empty", latency_ms

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
            self.cost_tracker.log_call(
                slot.model_name, provider, in_tok, out_tok, cost, latency_ms,
                agent_name=agent_name or "",
                pair=pair or "",
                status="success",
            )

            return response, None, latency_ms

        except _HARD_CRASH as e:
            logger.error(f"Code bug in LLM pipeline: {type(e).__name__}: {e}")
            raise
        except Exception as e:
            latency_ms = (time.time() - start) * 1000
            error_type = classify_error(e)
            tag = "[RateLimit]" if error_type == "rate_limit" else "[Failover]"
            if error_type == "rate_limit":
                logger.info(f"{tag} {slot.model_name} quota exhausted. Other models on same key still OK.")
            elif error_type != "timeout":  # Don't spam logs for timeouts
                logger.warning(f"{tag} {slot.model_name} → {type(e).__name__}: {str(e)[:120]}. Next model...")
            try:
                provider = self._provider_map.get(id(model), "unknown")
                self.cost_tracker.log_call(
                    slot.model_name, provider, 0, 0, 0.0, latency_ms,
                    agent_name=agent_name or "",
                    pair=pair or "",
                    status="error",
                )
            except Exception:
                pass
            return None, error_type, latency_ms

    # ── Main Invoke ───────────────────────────────────────────────────

    def invoke(self, messages: List[Any], temperature: Optional[float] = None,
               max_wall_time: float = 90.0, priority: Optional[str] = None,
               task_context: Optional[Dict[str, Any]] = None,
               pair: Optional[str] = None, **kwargs):
        """Route LLM request using Thompson + LinUCB. Drop-in compatible.

        Revize Tur-2 (C2): ``pair`` now writes llm_calls.trading_pair, so
        confirm_trade_exit's retroactive LinUCB feedback can find the rows
        that actually belong to this trade. Without this the llm_calls
        table held NULL trading_pair for every row and the retro feedback
        was dead code.
        """
        if task_context is None:
            import datetime as _dt
            task_context = {
                "task": "default",
                "prompt_len": sum(
                    len(str(getattr(m, "content", "") or "")) for m in messages
                ),
                "needs_json": False,
                "regime_vol": 0.5,
                "hour_utc": _dt.datetime.now(_dt.timezone.utc).hour,
            }

        estimated_tokens = sum(len(str(getattr(m, "content", ""))) for m in messages) // 3
        candidates = self._select_slots(priority, estimated_tokens,
                                        task_context=task_context)

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

            _task_name = (task_context or {}).get("task", "") if isinstance(task_context, dict) else ""
            response, error_type, latency_ms = self._try_model(
                slot, messages, temperature,
                agent_name=_task_name or "",
                pair=pair or "",
                **kwargs,
            )

            if response is not None:
                # Phase 25: Quality check — empty/tiny responses are NOT quality passes
                content = str(getattr(response, "content", ""))
                is_quality = len(content.strip()) > 20  # Real content, not just "OK" or empty
                with self._state_lock:
                    slot.record_success(quality=is_quality, latency_ms=latency_ms,
                                         task_context=task_context)
                if slot.provider == "gemini":
                    self.gemini_circuit.record_success()
                self._maybe_persist()
                return response

            # Failure path
            with self._state_lock:
                slot.record_failure(error_type, task_context=task_context,
                                     latency_ms=latency_ms)
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

    # ── Phase 27 Task 18: query-type heuristic classifier ────────────────
    @staticmethod
    def classify_query(messages: List[Any]) -> str:
        """Route short / code / analysis / default queries to different models.

        Returns one of `simple`, `code`, `complex`, `medium` — callers can map
        this to an appropriate `priority` so the Thompson sampler is biased
        toward a model whose historical reward matches that query shape.
        """
        text = " ".join(str(getattr(m, "content", "")) for m in messages)
        length = len(text)
        lower = text.lower()
        has_code = ("```" in text) or ("def " in text) or ("```python" in text)
        analysis_keywords = ("why", "explain", "analysis", "analyze",
                              "because", "reasoning", "debate", "compare")
        if has_code:
            return "code"
        if length < 500:
            return "simple"
        if any(k in lower for k in analysis_keywords) or length > 4000:
            return "complex"
        return "medium"

    # ── Phase 27 Task 17: cross-provider LLM ensemble ────────────────────
    async def ensemble_invoke(self, messages: List[Any],
                              n_judges: int = 3,
                              temperature: Optional[float] = None,
                              task_context: Optional[Dict[str, Any]] = None,
                              pair: Optional[str] = None,
                              **kwargs) -> Dict[str, Any]:
        """Invoke `n_judges` DISTINCT providers in parallel for disagreement-
        aware fusion. Useful for RLAIF / judge-style consumers who want a
        worst-case-robust signal instead of a single lucky draw.

        Returns dict with:
          `responses`: list of raw content strings (one per provider)
          `providers`: matching provider names
          `weights`:   Beta-mean weights used for fusion
          `cdr`:       Conflict Detection Rate in [0, 1] — higher = less
                       agreement on the parsed signal direction
        """
        import asyncio as _asyncio

        now = time.time()
        # Group eligible slots by provider and take the freshest per provider
        per_provider: Dict[str, ModelSlot] = {}
        for s in self.slots:
            if not s.is_available(now):
                continue
            # Prefer higher Beta mean within provider
            existing = per_provider.get(s.provider)
            if existing is None:
                per_provider[s.provider] = s
                continue
            if s.sample(exploit=True) > existing.sample(exploit=True):
                per_provider[s.provider] = s
        if not per_provider:
            raise ValueError("ensemble_invoke: no providers available")

        selected: List[ModelSlot] = list(per_provider.values())[:n_judges]
        if not selected:
            raise ValueError("ensemble_invoke: no slots selected")

        loop = _asyncio.get_event_loop()

        _ens_task_name = (task_context or {}).get("task", "") if isinstance(task_context, dict) else ""

        async def _one(slot: ModelSlot) -> Dict[str, Any]:
            try:
                response = await loop.run_in_executor(
                    None,
                    lambda: self._try_model(
                        slot, messages, temperature,
                        agent_name=_ens_task_name or "",
                        pair=pair or "",
                        **kwargs,
                    ),
                )
                if isinstance(response, tuple):
                    if len(response) == 3:
                        resp_obj, error_type, latency_ms = response
                    else:
                        resp_obj, error_type = response
                        latency_ms = None
                else:
                    resp_obj, error_type, latency_ms = response, None, None
                if resp_obj is None:
                    with self._state_lock:
                        slot.record_failure(error_type or "other",
                                             task_context=task_context,
                                             latency_ms=latency_ms)
                    return {"slot": slot, "content": None, "error": error_type}
                content = str(getattr(resp_obj, "content", ""))
                is_quality = len(content.strip()) > 20
                with self._state_lock:
                    slot.record_success(quality=is_quality, latency_ms=latency_ms,
                                         task_context=task_context)
                return {"slot": slot, "content": content, "error": None}
            except Exception as e:
                with self._state_lock:
                    slot.record_failure("other", task_context=task_context)
                return {"slot": slot, "content": None, "error": str(e)}

        results = await _asyncio.gather(*[_one(s) for s in selected])

        responses = [r["content"] for r in results if r["content"]]
        providers = [r["slot"].provider for r in results if r["content"]]
        weights = [
            (r["slot"].alpha / (r["slot"].alpha + r["slot"].beta_param))
            for r in results if r["content"]
        ]

        # Conflict Detection Rate — inspect the first word of each response
        # as a crude "direction" token. Consumers (RLAIF) typically structure
        # the prompt so the first token is BULLISH/BEARISH/NEUTRAL.
        directions = []
        for content in responses:
            head = content.strip().split()[:1]
            if head:
                tok = head[0].upper().strip(".,:;\"'")
                if tok in ("BULLISH", "BEARISH", "NEUTRAL", "LONG", "SHORT"):
                    directions.append(tok)
        if directions:
            unique = set(directions)
            cdr = (len(unique) - 1) / max(len(directions) - 1, 1)
        else:
            cdr = 0.0

        return {
            "responses": responses,
            "providers": providers,
            "weights": weights,
            "cdr": round(float(cdr), 3),
            "n_success": len(responses),
            "n_requested": len(selected),
        }

    def rpd_status(self) -> List[Dict[str, Any]]:
        """Phase 27 Task 18: snapshot of per-slot daily quota usage."""
        out = []
        for s in self.slots:
            out.append({
                "slot_id": s.slot_id,
                "rpd_limit": s.rpd_limit,
                "remaining": s.daily_quota.remaining(),
                "calls_today": s.rpd_limit - s.daily_quota.remaining(),
                "within_budget": s.daily_quota.is_within_budget(),
            })
        return out

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
            # EK.2.10: piggyback on the existing persistence tick so LinUCB
            # state is checkpointed with the same cadence as Thompson stats.
            try:
                self._persistence.save_linucb_batch(self.slots)
            except Exception as e:
                logger.debug(f"[LinUCB:Persist] save failed: {e}")
            self._call_counter = 0
            self._last_persist = now

    def save_linucb_state(self) -> int:
        """Force an immediate persistence of LinUCB posteriors. Useful for
        tests and scheduled checkpoint jobs."""
        try:
            return self._persistence.save_linucb_batch(self.slots)
        except Exception as e:
            logger.debug(f"[LinUCB:Persist] save failed: {e}")
            return 0

    def load_linucb_state(self) -> int:
        """Restore LinUCB posteriors into this router's slots. Returns the
        number of slots that received non-trivial state."""
        restored = 0
        try:
            saved = self._persistence.load_linucb_all()
        except Exception as e:
            logger.debug(f"[LinUCB:Persist] load failed: {e}")
            return 0
        by_id = {s.slot_id: s for s in self.slots}
        for slot_id, entry in saved.items():
            slot = by_id.get(slot_id)
            if slot is None:
                continue
            A = entry.get("A")
            b = entry.get("b")
            if A is None or b is None:
                continue
            try:
                slot.linucb_A = np.asarray(A, dtype=np.float64)
                slot.linucb_b = np.asarray(b, dtype=np.float64)
                slot.linucb_n_updates = int(entry.get("n_updates") or 0)
                restored += 1
            except Exception:
                continue
        if restored:
            logger.info(f"[LinUCB:Persist] restored {restored} slot posteriors")
        return restored

    def get_slot_stats(self) -> List[dict]:
        """Return stats for all slots (for monitoring/debugging)."""
        return [{"slot_id": s.slot_id, "alpha": round(s.alpha, 2), "beta": round(s.beta_param, 2),
                 "mean": round(s.alpha / (s.alpha + s.beta_param), 3),
                 "calls": s.total_calls, "ok": s.success_count, "quality": s.quality_pass_count,
                 "disabled": s.disabled, "available": s.is_available(time.time())}
                for s in self.slots]


_router_singleton: Optional["LLMRouter"] = None
_router_lock = threading.Lock()


def get_router(**kwargs) -> "LLMRouter":
    """Process-level singleton LLMRouter. Used by retroactive feedback and
    monitoring endpoints so they don't instantiate a fresh router (which
    would load every slot stats row from SQLite again)."""
    global _router_singleton
    with _router_lock:
        if _router_singleton is None:
            _router_singleton = LLMRouter(**kwargs)
    return _router_singleton


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
