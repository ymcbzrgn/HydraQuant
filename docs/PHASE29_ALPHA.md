# Phase 29 — ALPHA (Implementation Blueprint)

**Sensory Expansion & Self-Falsification**  
**24 Nisan 2026**

> "Her task için kod + effort range + impact confidence + rollback."

---

## 0. Bu Dökümanın Kuralları (Dürüstlük Manifestosu)

1. **Impact tahminleri kategorize**: 🟢 HIGH confidence (matematiksel), 🟡 MEDIUM (literature tutarlı), 🔴 LOW (spekülatif).
2. **Effort range**: Optimistic | Realistic | Pessimistic (gün cinsinden).
3. **Rollback cost**: Trivial / Easy / Hard / Irreversible.
4. **Dependency**: Önceki sprint/task zorunlu mu.
5. **Feature flag**: `config_ai.json.runtime_flags` kontrol.
6. **Validation gate**: Bu task pass için ne görmeliyiz.

---

## 1. SPRINT 3C — BACKTEST REVIVAL (CRITICAL)

**Neden ÖNCE**: Bütün diğer sprint'lerin impact'i **measurable** olması için backtest end-to-end çalışmalı. "Umarım çalışır" gambit'ini bitir.

### Task 3C.1 — MockAISignal Deterministic Stub

**Dosyalar**:
- Yeni: `user_data/scripts/mock_ai_signal.py`
- Edit: `user_data/strategies/HydraSizer.py` (7 runmode guard)

**Mevcut kod (BROKEN in backtest)**:
```python
# HydraSizer.py:149, 815, 950, 1500, 2341, 2564
if self.dp.runmode.value not in ('dry_run', 'live'):
    return dataframe  # backtest: skip AI entirely
```

**Fix**:
```python
# mock_ai_signal.py (yeni, ~80 satır)
from typing import Dict, Any, Optional
import pandas as pd

class MockAISignal:
    """Deterministic AI signal for backtest — pure feature-based.
    
    Produces BULLISH/BEARISH/NEUTRAL from RSI + ADX + trend strength.
    No HTTP, no LLM, no scheduler. Exercises downstream sizing/exit/DCA.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.conf_floor = 0.30
        self.conf_ceil = 0.85
    
    def get_signal(self, pair: str, current_time, df: pd.DataFrame) -> Dict[str, Any]:
        """Called in backtest mode instead of _get_ai_signal HTTP."""
        if len(df) < 50:
            return self._neutral(current_time, "insufficient_data")
        
        row = df.iloc[-1]
        rsi = float(row.get('rsi', 50.0))
        adx = float(row.get('adx', 20.0))
        ema_fast = float(row.get('ema_9', 0.0))
        ema_slow = float(row.get('ema_21', 0.0))
        close = float(row.get('close', 0.0))
        
        # Trend bias from EMA cross
        if ema_fast > ema_slow and close > ema_fast:
            trend_bias = 0.2
        elif ema_fast < ema_slow and close < ema_fast:
            trend_bias = -0.2
        else:
            trend_bias = 0.0
        
        # RSI extreme + ADX strength
        if rsi < 30 and adx > 25:
            direction = "BULLISH"
            strength = (30 - rsi) / 30 + (adx - 25) / 50
        elif rsi > 70 and adx > 25:
            direction = "BEARISH"
            strength = (rsi - 70) / 30 + (adx - 25) / 50
        else:
            direction = "NEUTRAL" if abs(trend_bias) < 0.1 else ("BULLISH" if trend_bias > 0 else "BEARISH")
            strength = abs(trend_bias)
        
        confidence = max(self.conf_floor, min(self.conf_ceil, strength + 0.4))
        
        return {
            "signal": direction,
            "confidence": confidence,
            "timestamp": str(current_time),
            "reasoning": f"mock: rsi={rsi:.1f} adx={adx:.1f} trend={trend_bias:+.2f}",
            "source": "MOCK_BACKTEST",
            "sub_scores": {
                "trend": trend_bias,
                "momentum": (rsi - 50) / 50,
                "volatility": min(adx / 50, 1.0),
            },
        }
    
    def _neutral(self, ts, reason: str) -> Dict[str, Any]:
        return {"signal": "NEUTRAL", "confidence": 0.30, "timestamp": str(ts),
                "reasoning": f"mock: {reason}", "source": "MOCK_BACKTEST", "sub_scores": {}}
```

**HydraSizer edits** (7 noktada):
```python
# __init__ sonunda:
from mock_ai_signal import MockAISignal
self._mock_ai = MockAISignal(self.config) if self.config.get('runmode', '').lower() == 'backtest' else None

# _get_ai_signal replace (satır 613):
def _get_ai_signal(self, pair, current_time, dataframe):
    if self._mock_ai is not None:
        return self._mock_ai.get_signal(pair, current_time, dataframe)
    # ... mevcut HTTP path (live/dry_run)

# 6 runmode guard replace (:149, 815, 950, 1500, 2341, 2564):
# ESKİ:
if self.dp.runmode.value not in ('dry_run', 'live'):
    return dataframe
# YENİ:
if self.dp.runmode.value == 'backtest' and not self.config.get('backtest_mock_ai', True):
    return dataframe  # explicit disable
```

**Effort**: 1 | 2 | 3 gün  
**Impact**: 🟢 HIGH — matematiksel kesin (runnable yes/no)  
**Rollback**: Trivial (`backtest_mock_ai=False` config)  
**Feature flag**: `config_ai.json.runtime_flags.backtest_mock_ai_enabled` (default: true)  
**Dependency**: Yok  
**Validation gate**: `freqtrade backtesting --strategy HydraSizer --timerange 20260101-20260120 --pairs BTC/USDT:USDT` tamamlanmalı, trade sayısı > 0, PnL hesaplanmalı.

**Risk**: Mock signal gerçek LLM'den farklı olabilir → backtest sonuçları yanıltıcı. AMA hedefi daha alçak — sadece strategy pipeline'ının downstream işleyip işlemediği. Alpha gerçekliği Tier 2 ile gelir.

### Task 3C.2 — Backtest Regression Test Suite

**Dosyalar**:
- Yeni: `tests/test_backtest_pipeline.py`
- Edit: `tests/test_ai_scripts.py` (smoke test ekle)

**Mevcut**: Pytest suite runtime Python kodunu test eder ama `freqtrade backtesting` CLI'si test edilmiyor.

**Fix**:
```python
# tests/test_backtest_pipeline.py (yeni, ~150 satır)
import subprocess, json, pytest
from pathlib import Path

BT_TIMERANGE = "20260101-20260115"
BT_PAIRS = "BTC/USDT:USDT ETH/USDT:USDT"

def run_backtest(timerange=BT_TIMERANGE, pairs=BT_PAIRS, extra_args=None):
    cmd = [
        "freqtrade", "backtesting",
        "--strategy", "HydraSizer",
        "--config", "config_bybit_testnet_futures.json",
        "--timerange", timerange,
        "--pairs", *pairs.split(),
        "--export", "trades",
        "--export-filename", f"/tmp/bt_test_{timerange}.json",
    ]
    if extra_args: cmd.extend(extra_args)
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    return r.returncode, r.stdout, r.stderr

def test_backtest_runs_end_to_end():
    rc, stdout, stderr = run_backtest()
    assert rc == 0, f"backtest failed: {stderr[-1000:]}"
    assert "Total trades" in stdout, "no trades summary"

def test_backtest_generates_trades():
    rc, stdout, _ = run_backtest()
    # Parse stdout for total trade count
    import re
    m = re.search(r"Total trades\s*\│\s*(\d+)", stdout)
    assert m, "no trade count in output"
    n_trades = int(m.group(1))
    assert n_trades > 0, f"zero trades in 15 days backtest"

def test_backtest_mock_signal_distribution():
    """Ensure mock signals produce non-trivial bull/bear/neutral distribution."""
    # Export trades, count enter_tag distribution
    trades_file = Path(f"/tmp/bt_test_{BT_TIMERANGE}.json")
    if not trades_file.exists(): pytest.skip("no export")
    data = json.loads(trades_file.read_text())
    # Assert at least 2 different directions present
    ...
```

**Effort**: 0.5 | 1 | 2 gün  
**Impact**: 🟢 HIGH — her code change backtest gate'inden geçer  
**Rollback**: Trivial (skip failing tests)  
**Dependency**: 3C.1  
**Validation**: CI'da backtest test suite yeşil.

### Task 3C.3 — Runmode-Aware Scheduler Stub

**Sorun**: Backtest'te scheduler kapalı. AMA bazı fix'ler scheduler job'larına bağlı (forgone_resolver, Kelly update, argument_quality backfill). Backtest'te bu pathler sessiz.

**Fix**: Backtest modunda **inline trigger** — trade close callback'te direkt çağır:

```python
# HydraSizer.py confirm_trade_exit
def confirm_trade_exit(self, pair, trade, ...):
    # ... mevcut logic ...
    
    # Backtest: inline trigger (scheduler yok)
    if self.dp.runmode.value == 'backtest':
        from position_sizer import get_real_kelly
        get_real_kelly().update(won=(trade.close_profit_abs > 0), ...)
        # argument_quality update (varsa)
```

**Effort**: 0.5 | 1 | 1.5 gün  
**Impact**: 🟡 MEDIUM — backtest'te learning loop kısmen aktif  
**Rollback**: Trivial  
**Dependency**: 3C.1

### 3C TOPLAM

- **Effort range**: 2 | 4 | 6.5 gün
- **Milestone**: Backtest end-to-end runnable + regression test yeşil + inline learning
- **Feature flag**: `backtest_mock_ai_enabled`

---

## 2. SPRINT 3D — SENSORY EXPANSION WAVE 1

**Tema**: En yüksek ROI alpha kaynakları + temel risk guard.

### Task 3D.1 — Funding/OI Divergence Sub-Score

**Dosyalar**:
- Yeni: `user_data/scripts/funding_oi_monitor.py`
- Edit: `user_data/scripts/evidence_engine.py` (7. sub-score)
- Edit: `user_data/scripts/scheduler.py` (15dk cron job)

**Kod**:
```python
# funding_oi_monitor.py (~150 satır)
import ccxt, time, sqlite3
from typing import Dict, Tuple
import numpy as np

class FundingOIMonitor:
    def __init__(self, exchange_id="bybit"):
        self.exchange = ccxt.bybit({'options': {'defaultType': 'linear'}})
    
    def fetch(self, pair: str) -> Dict[str, float]:
        """Fetch current funding + OI 24h history."""
        symbol = pair.replace("/USDT:USDT", "USDT")
        funding_hist = self.exchange.fetchFundingRateHistory(pair, limit=24)
        oi_now = self.exchange.fetch(f"v5/market/open-interest?category=linear&symbol={symbol}&intervalTime=1h&limit=24")
        return {
            "funding_current": funding_hist[-1]["fundingRate"] if funding_hist else 0.0,
            "funding_8h_avg": np.mean([f["fundingRate"] for f in funding_hist[-3:]]) if len(funding_hist) >= 3 else 0.0,
            "oi_delta_24h_pct": (oi_now[-1] - oi_now[0]) / max(oi_now[0], 1) if len(oi_now) >= 2 else 0.0,
        }
    
    def score(self, pair: str) -> Dict[str, float]:
        """Return [-1, +1] score: +1 = crowd long crowded, -1 = crowd short crowded."""
        d = self.fetch(pair)
        funding_score = np.tanh(d["funding_8h_avg"] * 1000)  # 0.01% funding → ~tanh(0.01) = 0.01
        oi_score = np.tanh(d["oi_delta_24h_pct"] * 10)       # 10% OI increase → ~tanh(1) = 0.76
        # Divergence: high funding + high OI = crowded → sell bias for bot
        divergence = -0.6 * funding_score - 0.4 * oi_score
        return {"funding_oi_score": divergence, "raw": d}
```

**evidence_engine.py**:
```python
# 7. sub_score
funding_oi = self._funding_oi_monitor.score(pair)
sub_scores["funding_oi"] = funding_oi["funding_oi_score"]
# Weight: 0.12 (trend 0.25, momentum 0.20, volume 0.15, vol 0.15, regime 0.15, sentiment 0.10 = 1.0 → rescale)
```

**scheduler.py**:
```python
# Her 15dk pair başına güncelle (ama cron yerine strategy içinde on-demand cache yap)
# Alternatif: opportunity_scanner'a entegre (zaten pair listesi dolaşıyor)
```

**Effort**: 0.5 | 1 | 2 gün  
**Impact**: 🟡 MEDIUM — literature IR +0.2-0.6, bizim validasyon YOK. Backtest Tier 2'de replay ile ölç.  
**Rollback**: Easy (weight=0 set + flag off)  
**Feature flag**: `funding_oi_score_enabled`  
**Dependency**: 3C.1 (validation için)

**Risk**: Crypto funding rate rejime bağımlı — bull market'ta high funding normal (trend follow ile uyumlu). Bot shorts açmaya eğilimli olabilir. Regime filter gerekir.

### Task 3D.2 — UI/CDaR Circuit Breaker

**Dosyalar**:
- Yeni: `user_data/scripts/portfolio_risk_breaker.py`
- Edit: `user_data/scripts/constitution.py` (cap logic)
- Edit: `user_data/strategies/HydraSizer.py` (custom_stake_amount guard)

**Kod**:
```python
# portfolio_risk_breaker.py (~120 satır)
import numpy as np
from typing import Dict, Tuple

class PortfolioRiskBreaker:
    """Ulcer Index + Conditional Drawdown at Risk (CDaR).
    
    Triggers:
    - UI (14-day) > 90th percentile historical → halve sizing
    - CDaR_95 > 8% → max_open_trades -= 2 + halve sizing
    - Rapid drawdown (>5% in 6h) → halt new entries 30min
    """
    
    def __init__(self, db_path):
        self.db_path = db_path
    
    def compute_ui(self, equity_curve: np.ndarray, window: int = 14*24) -> float:
        """Ulcer Index = sqrt(mean(DD_i²)) over rolling window."""
        peak = np.maximum.accumulate(equity_curve[-window:])
        dd = (equity_curve[-window:] - peak) / peak
        return float(np.sqrt(np.mean(dd**2)))
    
    def compute_cdar(self, equity_curve: np.ndarray, alpha: float = 0.95, window: int = 30*24) -> float:
        """CDaR_α = mean of drawdowns exceeding VaR_α."""
        peak = np.maximum.accumulate(equity_curve[-window:])
        dd = (peak - equity_curve[-window:]) / peak  # positive drawdowns
        threshold = np.quantile(dd, alpha)
        tail = dd[dd >= threshold]
        return float(np.mean(tail)) if len(tail) else 0.0
    
    def check(self, portfolio_value: float, history: np.ndarray) -> Dict[str, Any]:
        ui = self.compute_ui(history)
        cdar = self.compute_cdar(history)
        
        ui_historical_90 = self._ui_90th_percentile()  # cached
        
        # Decision matrix
        sizing_multiplier = 1.0
        max_open_delta = 0
        halt = False
        reason = []
        
        if ui > ui_historical_90:
            sizing_multiplier *= 0.5
            reason.append(f"UI {ui:.4f} > P90 {ui_historical_90:.4f}")
        
        if cdar > 0.08:
            sizing_multiplier *= 0.5
            max_open_delta = -2
            reason.append(f"CDaR_95 {cdar:.1%} > 8%")
        
        # Rapid drawdown
        recent_dd = (max(history[-6*4:]) - history[-1]) / max(history[-6*4:]) if len(history) >= 24 else 0
        if recent_dd > 0.05:
            halt = True
            reason.append(f"rapid DD {recent_dd:.1%} in 6h")
        
        return {"sizing_multiplier": sizing_multiplier, "max_open_delta": max_open_delta,
                "halt_new": halt, "ui": ui, "cdar": cdar, "reasons": reason}
```

**Effort**: 1 | 2 | 3 gün  
**Impact**: 🟡 MEDIUM — literature MaxDD -15-40%, bizim: UI+CDaR eşikleri kalibrasyon lazım. Backtest'te ayarlanabilir.  
**Rollback**: Easy (flag off → multiplier=1.0 always)  
**Feature flag**: `portfolio_risk_breaker_enabled`  
**Dependency**: 3C (validation için)

### Task 3D.3 — Dynamic ATR Chandelier (HAR-RV)

**Dosyalar**:
- Yeni: `user_data/scripts/har_rv_forecaster.py`
- Edit: `user_data/strategies/HydraSizer.py` (custom_stoploss)

**Kod**:
```python
# har_rv_forecaster.py (~100 satır)
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

class HARRVForecaster:
    """Heterogeneous Autoregressive Realized Volatility.
    
    RV_{t+1} = β0 + β_d·RV_t + β_w·RV_{t-5:t}/5 + β_m·RV_{t-22:t}/22 + ε
    """
    
    def compute_rv(self, returns: np.ndarray, period: int = 24) -> np.ndarray:
        """Realized volatility (sum of squared intraday returns)."""
        rv = np.zeros(len(returns) - period + 1)
        for i in range(len(rv)):
            rv[i] = np.sum(returns[i:i+period]**2)
        return np.sqrt(rv)
    
    def fit_har(self, rv_series: np.ndarray) -> Dict[str, float]:
        """Fit HAR-RV linear regression."""
        if len(rv_series) < 30:
            return {"beta_0": 0, "beta_d": 0.4, "beta_w": 0.3, "beta_m": 0.2, "r2": 0.0}
        
        rv_d = rv_series[22:-1]
        rv_w = np.array([rv_series[i-5:i].mean() for i in range(22, len(rv_series)-1)])
        rv_m = np.array([rv_series[i-22:i].mean() for i in range(22, len(rv_series)-1)])
        y = rv_series[23:]
        
        X = np.column_stack([rv_d, rv_w, rv_m])
        model = LinearRegression().fit(X, y)
        return {
            "beta_0": float(model.intercept_),
            "beta_d": float(model.coef_[0]),
            "beta_w": float(model.coef_[1]),
            "beta_m": float(model.coef_[2]),
            "r2": float(model.score(X, y)),
        }
    
    def forecast_next(self, rv_series: np.ndarray, params: Dict[str, float]) -> float:
        if len(rv_series) < 22:
            return float(rv_series[-1]) if len(rv_series) else 0.01
        return (params["beta_0"]
                + params["beta_d"] * rv_series[-1]
                + params["beta_w"] * rv_series[-5:].mean()
                + params["beta_m"] * rv_series[-22:].mean())
    
    def dynamic_atr_mult(self, rv_forecast: float, rv_median: float, rv_std: float) -> float:
        """k = 1.5 + 0.5·tanh((RV_forecast - RV_median)/RV_std), range [1.0, 2.0]."""
        z = (rv_forecast - rv_median) / max(rv_std, 1e-6)
        return 1.5 + 0.5 * np.tanh(z)
```

**HydraSizer.custom_stoploss edit**:
```python
# Satır ~1275 civarı (mevcut _np chandelier_atr_* fallback)
if self._har_rv_forecaster:
    rv_params = self._rv_params_cache.get(pair)  # daily refit
    if rv_params:
        rv_hist = self._rv_hist_cache[pair]
        rv_next = self._har_rv_forecaster.forecast_next(rv_hist, rv_params)
        mult = self._har_rv_forecaster.dynamic_atr_mult(
            rv_next, np.median(rv_hist), np.std(rv_hist)
        )
else:
    # Mevcut static _np fallback
    mult = _np("strategy.chandelier_atr_med", 1.35)
```

**Effort**: 1 | 1.5 | 2 gün  
**Impact**: 🟡 MEDIUM — HAR-RV crypto papers consistent, dynamic ATR should reduce whipsaws %30-40. Bizim validasyon YOK.  
**Rollback**: Trivial (flag off → static)  
**Feature flag**: `dynamic_atr_har_rv_enabled`  
**Dependency**: 3C

### Task 3D.4 — Semantic Cache Layer

**Dosyalar**:
- Edit: `user_data/scripts/llm_router.py` (invoke wrapper)
- Edit: `user_data/scripts/semantic_cache.py` (mevcut, genişlet)

**Mevcut**: `semantic_cache.py` runtime TTL cache var, LLM router'a entegre değil.

**Fix**:
```python
# llm_router.py invoke() başında:
def invoke(self, messages, task_context=None, pair=None, **kw):
    # Semantic cache check (task-specific TTL)
    task = (task_context or {}).get("task", "default")
    if task in CACHEABLE_TASKS:  # {"rag_synthesis", "news_digest", "market_commentary"}
        cached = _semantic_cache.get(
            query=self._msg_hash(messages), 
            pair=pair,
            ttl_s=CACHE_TTL.get(task, 300),  # 5min default
            similarity_threshold=0.93,
        )
        if cached:
            self._stats["cache_hit"] += 1
            return cached
    
    # ... existing LinUCB routing ...
    response = self._try_model(...)
    
    # Cache set (same conditions)
    if task in CACHEABLE_TASKS and response:
        _semantic_cache.put(
            query=self._msg_hash(messages),
            response=response,
            pair=pair,
            task=task,
        )
    
    return response

CACHEABLE_TASKS = frozenset({"rag_synthesis", "news_digest", "market_commentary"})
CACHE_TTL = {"rag_synthesis": 1800, "news_digest": 600, "market_commentary": 300}
```

**Effort**: 1 | 1.5 | 2 gün  
**Impact**: 🟢 HIGH — deterministic. Cache hit oranı %15-30 literature, bizim pattern'e göre. Latency 0ms on hit, cost $0.  
**Rollback**: Trivial (CACHEABLE_TASKS = frozenset())  
**Feature flag**: `semantic_cache_enabled`  
**Dependency**: Yok

### 3D TOPLAM
- **Effort**: 3.5 | 6 | 9 gün
- **Milestone**: 4 fix deploy + backtest doğrulama
- **Expected composite impact**: payoff 0.66 → 0.9-1.1, LLM latency 7.5s → 5s

---

## 3. SPRINT 3E — MADAM UPGRADE

**Tema**: Kim 2025 martingale fix + MoA quality tier + FinCon CVRF.

### Task 3E.1 — Sampling+Voting Wrapper (Martingale Fix)

**Dosyalar**:
- Edit: `user_data/scripts/agent_pool.py` (`run_debate`)

**Fix**:
```python
# agent_pool.py run_debate edit:
SAMPLES_PER_AGENT = _p("agent.samples_per_agent", 3)
VOTING_TEMPERATURE = _p("agent.voting_temperature", 0.7)

def run_debate(self, pair, ...):
    # ... mevcut R1 logic ama HER AGENT N=3 kez ...
    
    positions_by_agent = {}
    for agent_name in agents:
        samples = []
        for i in range(SAMPLES_PER_AGENT):
            pos = self._run_r1_agent(agent_name, pair, temperature=VOTING_TEMPERATURE, seed=i*7)
            if pos: samples.append(pos)
        
        # Majority vote within samples
        if samples:
            direction_votes = {"BULLISH": 0, "BEARISH": 0, "NEUTRAL": 0}
            for s in samples:
                direction_votes[s.get("direction", "NEUTRAL")] += 1
            majority_dir = max(direction_votes, key=direction_votes.get)
            
            # Confidence = agreement ratio
            agreement = direction_votes[majority_dir] / len(samples)
            avg_strength = np.mean([s["strength"] for s in samples if s.get("direction") == majority_dir])
            
            positions_by_agent[agent_name] = {
                "direction": majority_dir,
                "strength": avg_strength,
                "agreement": agreement,
                "n_samples": len(samples),
            }
    
    # Cross-agent majority (existing logic)
    ...
```

**Effort**: 0.25 | 0.5 | 1 gün  
**Impact**: 🟡 MEDIUM — Kim 2025 proof martingale ama sample diversity gain qualitative. Noise reduction tahminen %15-25.  
**Rollback**: Trivial (SAMPLES_PER_AGENT=1 → current behavior)  
**Feature flag**: `madam_sampling_voting_enabled`  
**Dependency**: 3C  
**Risk**: 3x LLM call per agent → cost/latency artar. LinUCB router'ın cost-aware olması gerekli (3F'de Pareto).

### Task 3E.2 — MoA 2-Layer Tier Routing

**Dosyalar**:
- Edit: `user_data/scripts/llm_router.py` (provider tier)
- Edit: `user_data/scripts/agent_pool.py` (L1/L2 assignment)

**Fix**:
```python
# llm_router.py:
PROVIDER_TIERS = {
    "fast": {"groq", "cerebras", "sambanova"},       # L1 proposers
    "premium": {"gemini-pro", "mistral-large"},      # L2 aggregator
    "any": None,
}

def invoke(self, messages, tier="any", task_context=None, pair=None, **kw):
    candidates = self._select_slots(task_context=task_context)
    if tier != "any":
        allowed = PROVIDER_TIERS[tier]
        candidates = [s for s in candidates if s.provider in allowed]
    # ... rest
```

```python
# agent_pool.py R1 loop:
response = llm.invoke(..., tier="fast", task_context=r1_ctx)  # L1 cheap

# Coordinator synthesis:
coord_response = llm.invoke(..., tier="premium", task_context=coord_ctx)  # L2 aggregator
```

**Effort**: 0.25 | 0.5 | 1 gün  
**Impact**: 🟡 MEDIUM — Together AI AlpacaEval trading extrap, synthesis quality +8-15%.  
**Rollback**: Trivial (tier="any" everywhere)  
**Feature flag**: `moa_tier_routing_enabled`  
**Dependency**: 3D.4 (semantic cache paralel)

### Task 3E.3 — FinCon Conceptual Beliefs (CVRF)

**Dosyalar**:
- Yeni: `user_data/scripts/conceptual_beliefs.py`
- Edit: `user_data/scripts/post_trade_court.py` (belief generation)
- Edit: `user_data/scripts/agent_pool.py` (belief prefix injection)
- Yeni: `migrations/phase29_conceptual_beliefs.sql`

**Migration**:
```sql
CREATE TABLE IF NOT EXISTS conceptual_beliefs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    regime TEXT NOT NULL,
    belief_text TEXT NOT NULL,
    evidence_count INTEGER DEFAULT 1,
    last_validated_at TEXT,
    accuracy_rate REAL,
    source_trade_ids TEXT,  -- JSON array
    active BOOLEAN DEFAULT 1,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(regime, belief_text(200))
);
CREATE INDEX idx_beliefs_regime ON conceptual_beliefs(regime, active);
```

**Kod**:
```python
# conceptual_beliefs.py (~180 satır)
class ConceptualBeliefStore:
    """FinCon CVRF: post-trade reflection → regime-conditional beliefs."""
    
    def generate_from_trade(self, trade, cause_attribution, verdict):
        """LLM → 2-3 belief deltas."""
        prompt = f"""Based on this closed trade, produce 2-3 CONCEPTUAL BELIEFS 
        as short declarative statements about market behavior in regime {trade['regime']}.
        
        Trade: {trade['pair']} {trade['signal']} conf={trade['confidence']:.2f} → pnl={trade['outcome_pnl']:.2f}R
        Root cause: {cause_attribution}
        
        Output JSON: {{"beliefs": [{{"text": "...", "confidence": 0-1}}]}}
        
        Good example: "In high-VIX regime, DevilsAdvocate's volatility warnings have been 73% accurate; upweight."
        Bad example: "Markets are unpredictable."
        """
        # LLM call (tier=premium for quality)
        ...
    
    def retrieve_for_debate(self, regime: str, k: int = 5) -> List[str]:
        """Return top-k beliefs for regime, sorted by accuracy_rate DESC."""
        ...
    
    def validate(self, trade_outcome):
        """After trade, check if beliefs referenced were validated. Update accuracy_rate."""
        ...
```

**Effort**: 2 | 3 | 5 gün  
**Impact**: 🔴 LOW → 🟡 MEDIUM — FinCon paper showed improvements, trading-specific generalization belirsiz. Yavaş compounding.  
**Rollback**: Easy (flag off → retrieve empty list)  
**Feature flag**: `cvrf_beliefs_enabled`  
**Dependency**: 3E.1, 3E.2, 3F (causal attribution)

### 3E TOPLAM
- **Effort**: 2.5 | 4 | 7 gün
- **Milestone**: MADAM martingale fix + MoA tier + CVRF beliefs
- **Expected**: Signal quality iyileşme (measurable after backtest)

---

## 4. SPRINT 3F — CAUSAL INTROSPECTION

### Task 3F.1 — DoWhy-GCM Per-Trade Postmortem

**Dosyalar**:
- Edit: `user_data/scripts/post_trade_court.py` (GCM attribution)
- Yeni dependency: `pip install dowhy` (venv)

**Fix**:
```python
# post_trade_court.py edit:
from dowhy.gcm import attribute_anomalies, StructuralCausalModel

class PostTradeCourt:
    def __init__(self, causal_engine):
        self.causal_engine = causal_engine
        self._scm = None  # lazy init
    
    def _ensure_scm(self, regime: str):
        """Build StructuralCausalModel from PCMCI+ discovered graph."""
        graph = self.causal_engine.load_graph(regime)  # networkx DiGraph
        self._scm = StructuralCausalModel(graph)
        # Fit functional causal models (AdditiveNoiseModel per node)
        ...
    
    def explain_loss(self, trade_id: int) -> Dict[str, float]:
        """Shapley attribution: 'market_noise' → {'funding_spike': 0.42R, ...}."""
        trade = self._load_trade(trade_id)
        self._ensure_scm(trade['regime'])
        
        anomaly = self._build_anomaly_vector(trade)
        attributions = attribute_anomalies(
            self._scm,
            target_node='outcome_pnl',
            anomaly_samples=anomaly,
        )
        # {'funding_rate': -0.42, 'regime': -0.31, 'confidence': 0.19, ...}
        
        # Persist
        self._store_attribution(trade_id, attributions)
        return attributions
```

**Effort**: 2 | 3 | 5 gün  
**Impact**: 🟢 HIGH (deterministic replacement) — "market_noise" fallback'ı Shapley ile değiştirir. Quality qualitative.  
**Rollback**: Easy (flag off → old fallback)  
**Feature flag**: `gcm_postmortem_enabled`  
**Dependency**: 3C (validation) + causal_engine working

### Task 3F.2 — Shadow Trade Counterfactual Analysis (DML)

**Dosyalar**:
- Yeni: `user_data/scripts/shadow_dml_analyzer.py`
- Edit: `user_data/scripts/scheduler.py` (weekly job)

**Kod**:
```python
# shadow_dml_analyzer.py (~200 satır)
from econml.dml import LinearDML
import pandas as pd

class ShadowDMLAnalyzer:
    """Shadow trades as counterfactual dataset → treatment effect estimation.
    
    T (treatment) = 1 if trade fired, 0 if shadow
    Y (outcome) = realized PnL (for fired) or forgone PnL (for shadow)
    X (confounders) = {regime, confidence, funding, OI, sub_scores}
    
    θ = causal effect of FIRING vs NOT FIRING given features.
    Sizing correction: f* = f_kelly × (1 + θ_hat / baseline)
    """
    
    def estimate(self, timerange: Tuple[str, str]) -> Dict[str, float]:
        T, Y, X = self._build_dataset(timerange)
        
        dml = LinearDML(
            model_y='auto',
            model_t='auto',
            random_state=42,
        )
        dml.fit(Y, T, X=X)
        
        theta = dml.effect(X).mean()
        ci = dml.effect_interval(X, alpha=0.05)
        
        return {
            "treatment_effect": float(theta),
            "ci_lower": float(ci[0].mean()),
            "ci_upper": float(ci[1].mean()),
            "n_samples": len(Y),
        }
```

**Effort**: 3 | 5 | 8 gün  
**Impact**: 🔴 LOW → 🟡 MEDIUM — Literature güçlü ama crypto shadow data kalitesi bilinmez. Shadow label bug fix sonrası ilk validasyon.  
**Rollback**: Easy (flag off, just don't run)  
**Feature flag**: `shadow_dml_analysis_enabled`  
**Dependency**: 3C + shadow ledger populated

### 3F TOPLAM
- **Effort**: 5 | 8 | 13 gün
- **Milestone**: Per-trade Shapley + weekly DML → feedback loop kapanır

---

## 5. SPRINT 3G — SENSORY EXPANSION WAVE 2

### Task 3G.1 — On-Chain Exchange Netflow

**Dosyalar**: Yeni `user_data/scripts/onchain_flow_monitor.py` + `derivatives_data` tablosuna ekle.

Özet: BitQuery GraphQL (10K call/ay free) → top-20 CEX wallet netflow → BEAR agent evidence.

**Effort**: 2 | 3 | 5 gün  
**Impact**: 🔴 LOW — CryptoQuant anekdot, validasyon kripto periyoduna bağlı  
**Feature flag**: `onchain_netflow_enabled`  
**Dependency**: 3C + 3D.1

### Task 3G.2 — Deribit Options GEX

**Dosyalar**: Yeni `user_data/scripts/deribit_gex_calculator.py`.

Kod örneği:
```python
# Black-Scholes greeks via py-vollib
from py_vollib.black_scholes.greeks.analytical import gamma as bs_gamma

def compute_dealer_gamma(options_chain: List[Dict]) -> float:
    """Σ (OI × gamma × sign) where sign = -1 for calls (MM short), +1 for puts."""
    total = 0.0
    for opt in options_chain:
        g = bs_gamma(opt['flag'], opt['spot'], opt['strike'], opt['expiry_years'], opt['r'], opt['iv'])
        sign = -1 if opt['flag'] == 'c' else 1
        total += opt['oi'] * g * sign
    return total
```

**Effort**: 2 | 3 | 4 gün  
**Impact**: 🔴 LOW — SqueezeMetrics equity methodology, crypto GEX relatively young  
**Feature flag**: `deribit_gex_enabled`  
**Dependency**: 3C + 3D.1

### Task 3G.3 — Speculative Cascade (Groq → Gemini)

**Dosyalar**: Yeni `user_data/scripts/cascade_router.py`.

Kod: (mevcut ARGE-6 raporundan)
```python
class CascadeRouter:
    def __init__(self, linucb, draft_tasks={"rag_synthesis", "market_commentary"}):
        self.router = linucb
        self.accept_thresh = 4.0
        self.draft_ema = {}
    
    def invoke(self, msgs, task_type, priority="mid"):
        if task_type not in self.draft_tasks:
            return self.router.invoke(msgs, priority=priority)
        # Stage 1: Groq draft
        draft = self.router.invoke(msgs, priority="low", tier="fast")
        # Stage 2: verifier
        verify = self.router.invoke(
            [HumanMessage(f"Response: {draft.content[:800]}\nGrade 1-5 completeness. Digit only.")],
            priority="low", tier="fast",
        )
        score = float(verify.content.strip()[:1] or 3)
        if score >= self.accept_thresh:
            return draft
        # Stage 3: full Gemini
        return self.router.invoke(msgs, priority="high", tier="premium")
```

**Effort**: 3 | 5 | 7 gün  
**Impact**: 🟢 HIGH — matematik weighted mean latency 7.5s → 2.1s verify quality gated  
**Rollback**: Easy (draft_tasks = empty)  
**Feature flag**: `speculative_cascade_enabled`  
**Dependency**: 3D.4 (semantic cache) + 3E.2 (MoA tier)

### 3G TOPLAM
- **Effort**: 7 | 11 | 16 gün
- **Milestone**: 3 yeni alpha source + latency 7.5s → 2.1s

---

## 6. SPRINT 3H — RISK UPGRADE

### Task 3H.1 — CVaR-Conditioned Kelly

**Kod** `position_sizer.py` edit:
```python
def kelly_fraction_cvar(self, pair, regime, cvar_95, mu_hat):
    """f* = (μ - rf) / (λ · CVaR_95). λ=1 (no risk aversion), rf=0."""
    lam = _p("sizing.cvar_lambda", 1.0)
    cvar_floor = _p("sizing.cvar_floor", 0.01)  # prevent div by 0
    cvar_eff = max(cvar_95, cvar_floor)
    return max(0.0, mu_hat / (lam * cvar_eff))
```

**Effort**: 3 | 5 | 7 gün (CVaR estimator + Kelly rewrite)  
**Impact**: 🟡 MEDIUM — literature broad, crypto amplifies. Sizing robustness against tail events.  
**Feature flag**: `cvar_kelly_enabled`  
**Dependency**: 3C (validation)

### Task 3H.2 — HRP Multi-Pair Allocation

**Kod**: `PyPortfolioOpt.HRPOpt` weekly refit job.

**Effort**: 2 | 3 | 5 gün  
**Impact**: 🟡 MEDIUM — Sharpe +15-20% equity, crypto extrap  
**Feature flag**: `hrp_multipair_enabled`  
**Dependency**: 3C

### Task 3H.3 — Funding Cascade Detector

**Kod**: 4-condition pattern monitor, Telegram alert + force-exit logic.

**Effort**: 5 | 7 | 10 gün  
**Impact**: 🟡 MEDIUM (tail event insurance) — rare event protection  
**Feature flag**: `funding_cascade_detector_enabled`  
**Dependency**: 3D.1

### 3H TOPLAM
- **Effort**: 10 | 15 | 22 gün
- **Milestone**: Kelly + allocation + cascade → MaxDD -%30-40

---

## 7. SPRINT 3I — META-LEARNING

### Task 3I.1 — Online Meta-Learning (OML)

**Dosyalar**: Edit `reptile_meta.py` + BOCPD trigger.

**Effort**: 5 | 7 | 10 gün  
**Impact**: 🔴 LOW — NLP/CV results, trading transfer tahmin  
**Feature flag**: `oml_online_enabled`  
**Dependency**: 3E + 3F

### Task 3I.2 — ProtoNet New-Pair Bootstrap

**Dosyalar**: Yeni `pair_prototype_bank.py` + LanceDB integration.

**Effort**: 5 | 7 | 10 gün  
**Impact**: 🟡 MEDIUM — few-shot lit, cold-start 72h → 24h measurable  
**Feature flag**: `protonet_bootstrap_enabled`  
**Dependency**: 3C + 3F

### Task 3I.3 — PCGrad Cross-Pair Transfer

**Dosyalar**: Yeni `pcgrad.py` + correlation clusterer.

**Effort**: 7 | 10 | 14 gün  
**Impact**: 🔴 LOW — multi-task learning generic  
**Feature flag**: `pcgrad_transfer_enabled`  
**Dependency**: 3I.1

### 3I TOPLAM
- **Effort**: 17 | 24 | 34 gün (3 sprint'lik büyük scope)

---

## 8. SPRINT 3J — BACKTEST TIER 2 & 3

### Task 3J.1 — LLM Response Replay Cache (Tier 2)

**Migration**:
```sql
CREATE TABLE IF NOT EXISTS llm_response_cache (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    pair TEXT NOT NULL,
    ts_utc TEXT NOT NULL,
    prompt_hash TEXT NOT NULL,
    model_version TEXT NOT NULL,
    agent_role TEXT,
    response_json TEXT NOT NULL,
    retrieval_snapshot_id TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(pair, ts_utc, prompt_hash, model_version)
);
CREATE INDEX idx_llm_replay ON llm_response_cache(pair, ts_utc);
```

**Kod**: `llm_router.py` branch + `warm_llm_cache.py` CLI.

**Effort**: 5 | 7 | 10 gün  
**Impact**: 🟢 HIGH — deterministic replay (anything cached is replayable exactly)  
**Rollback**: Easy (flag off → live always)  
**Feature flag**: `llm_replay_cache_enabled`  
**Dependency**: 3C.1

### Task 3J.2 — CPCV Walk-Forward + PBO + DSR (Tier 3)

**Kod**: `walk_forward_runner.py` with 6-groups × C(6,2)=15 paths + embargo + Bailey 2014 PBO + Deflated Sharpe.

**Effort**: 10 | 14 | 20 gün  
**Impact**: 🟢 HIGH (statistical validity, not directly alpha)  
**Rollback**: Easy (use single-split)  
**Dependency**: 3J.1

### 3J TOPLAM
- **Effort**: 15 | 21 | 30 gün

---

## 9. Cross-Cutting Infrastructure

### Task I.1 — Global Feature Flags Management

**`config_ai.json.runtime_flags`** — yeni flag'ler ekle:
```json
{
    "runtime_flags": {
        ... mevcut 10 ...,
        "backtest_mock_ai_enabled": true,
        "funding_oi_score_enabled": false,
        "portfolio_risk_breaker_enabled": false,
        "dynamic_atr_har_rv_enabled": false,
        "semantic_cache_enabled": true,
        "madam_sampling_voting_enabled": false,
        "moa_tier_routing_enabled": false,
        "cvrf_beliefs_enabled": false,
        "gcm_postmortem_enabled": false,
        "shadow_dml_analysis_enabled": false,
        "onchain_netflow_enabled": false,
        "deribit_gex_enabled": false,
        "speculative_cascade_enabled": false,
        "cvar_kelly_enabled": false,
        "hrp_multipair_enabled": false,
        "funding_cascade_detector_enabled": false,
        "oml_online_enabled": false,
        "protonet_bootstrap_enabled": false,
        "pcgrad_transfer_enabled": false,
        "llm_replay_cache_enabled": false,
        "cpcv_wf_enabled": false
    }
}
```

Gradual rollout — her sprint sonunda ilgili flag true.

### Task I.2 — Phase 29 Monitoring Dashboard

Mevcut `deploy_health_check.py` genişlet:
```python
def check_phase29_features():
    """Check each Phase 29 feature flag active + working."""
    for flag, validator in PHASE29_FLAGS.items():
        is_on = get_flag(flag, False)
        if is_on:
            verdict, detail = validator()
            check(f"Phase29 {flag}", verdict, detail)
```

### Task I.3 — A/B Testing Infrastructure

**Yeni**: `user_data/scripts/ab_test_manager.py` — ~%10 trade'de new feature off, compare rolling Sharpe.

**Effort**: 3 | 5 | 7 gün  
**Impact**: 🟢 HIGH (statistical proof of improvement)  
**Dependency**: 3J.2 (CPCV framework)

---

## 10. Rasyonel Impact Matrix (Final, sıralı)

| Sprint | Task | Effort (realistic) | Impact | Confidence |
|--------|------|---------|--------|-----|
| 3C.1 | Backtest Mock | 2g | Runnable backtest | 🟢 HIGH |
| 3C.2 | BT regression tests | 1g | CI gate | 🟢 HIGH |
| 3C.3 | Runmode scheduler stub | 1g | Learning in BT | 🟡 MEDIUM |
| 3D.1 | Funding/OI | 1g | IR +0.2-0.6 | 🟡 MEDIUM |
| 3D.2 | UI/CDaR breaker | 2g | MaxDD -15-30% | 🟡 MEDIUM |
| 3D.3 | Dynamic ATR | 1.5g | Payoff +0.1-0.3 | 🟡 MEDIUM |
| 3D.4 | Semantic cache | 1.5g | Latency hit=0 | 🟢 HIGH |
| 3E.1 | Sampling vote | 0.5g | Noise -20% | 🟡 MEDIUM |
| 3E.2 | MoA tier | 0.5g | Quality +8-15% | 🟡 MEDIUM |
| 3E.3 | CVRF beliefs | 3g | Compounding learning | 🔴 LOW |
| 3F.1 | GCM postmortem | 3g | Shapley attribution | 🟢 HIGH |
| 3F.2 | Shadow DML | 5g | Treatment effect | 🔴 LOW |
| 3G.1 | On-chain netflow | 3g | Regime filter | 🔴 LOW |
| 3G.2 | Deribit GEX | 3g | Regime detection | 🔴 LOW |
| 3G.3 | Speculative cascade | 5g | Latency -72% | 🟢 HIGH |
| 3H.1 | CVaR-Kelly | 5g | Sizing robustness | 🟡 MEDIUM |
| 3H.2 | HRP allocation | 3g | Sharpe +15-20% | 🟡 MEDIUM |
| 3H.3 | Cascade detector | 7g | Tail insurance | 🟡 MEDIUM |
| 3I.1 | OML | 7g | Drift adaptation | 🔴 LOW |
| 3I.2 | ProtoNet | 7g | Cold-start fix | 🟡 MEDIUM |
| 3I.3 | PCGrad | 10g | Cross-pair transfer | 🔴 LOW |
| 3J.1 | LLM replay | 7g | BT determinism | 🟢 HIGH |
| 3J.2 | CPCV WF | 14g | Statistical validity | 🟢 HIGH |
| I.3 | A/B framework | 5g | Feature proof | 🟢 HIGH |

**TOPLAM Phase 29**: 100-130 gün effort (realistic), 6-8 hafta/sprint × 8 sprint → **~3 ay aktif geliştirme**

**Matematiksel kesin kazanımlar** (🟢 HIGH confidence):
1. Backtest runnable (3C) — measurement capability
2. Semantic cache (3D.4) — deterministic 0ms on hit
3. GCM attribution (3F.1) — Shapley replacement
4. Speculative cascade (3G.3) — 7.5s → 2.1s weighted mean
5. LLM replay (3J.1) — deterministic backtest replay
6. CPCV WF (3J.2) — statistically valid OOS

**İlham + dua gerektirenler** (🔴 LOW):
- CVRF beliefs (yavaş compound)
- On-chain netflow (crypto domain transfer)
- Deribit GEX (crypto options young)
- OML (NLP/CV transfer)
- PCGrad (multi-task generic)

---

## 11. Nihai Öneri (dürüst)

**Eğer Yamac iki hafta ayırabiliyorsa**: 3C + 3D + 3E tamamla. En yüksek ROI + matematik kesin fix'ler + martingale fix.

**Eğer bir ay**: + 3F + 3G.3 ekle (introspection + cascade). Latency + quality measurable jumps.

**Eğer üç ay**: Tüm Phase 29 deploy. Full sensory + self-falsification + statistical validity.

**Eğer zaman yok**: Sadece 3C (1 hafta) — her fix'in validasyon gate'i olur, sonra parçalı sprint'ler.

---

## 12. Backlog / Future (Phase 30+)

Phase 29 dışında bırakılanlar:
- Lag-Llama / Moirai foundation model ensemble (Phase 28 extension)
- Order Book Imbalance + Kyle's lambda (infra ağır, phase 30)
- Prompt compression (LLMLingua-2) — effort düşük ama ROI marjinal şu an
- FinMem 3-tier memory (mevcut magma_memory.py enough)
- Bull/Bear dialectical pair restructure (MADAM mimari refactor, ayrı sprint)

---

## 13. Kabul Kriterleri (Phase 29 Done = ?)

Phase 29 "complete" olması için:
1. ✅ Backtest `freqtrade backtesting` end-to-end runnable (regression test yeşil)
2. ✅ En az 3 alpha source yeni sub-score (funding/OI, bir opsiyonel on-chain/GEX)
3. ✅ UI/CDaR circuit breaker canlıda tetiklenmiş (en az 1 event)
4. ✅ Dynamic ATR Chandelier 1 ay çalışmış, MaxDD azalmış (yes/no)
5. ✅ MADAM sampling+voting devreye + MoA tier aktif
6. ✅ GCM postmortem tablosunda non-trivial attribution (≥100 trade)
7. ✅ CPCV walk-forward ≥1 tam run tamamlanmış (15 path)
8. ✅ Deploy health dashboard Phase 29 feature'ları doğrular
9. ✅ Documentation: Phase 29 handoff + retrospective

**Başarısızlık kriteri**:
- 6 hafta sonunda 3C dahil hiçbir task tamamlanmamışsa Phase 29 **ABORT**, phase retrospective yazıp Phase 30'a geç.

---

**Belge kapanış**: Phase 29 **uzun maraton, sprint değil**. Dürüst planla, gerçekçi beklentiyle, ölçülebilir kazanımla ilerle. Backtest revival'dan önce hiçbir fix'in kesin impact'i yok. 3C = hayat damarı.

**Yazan**: Claude Opus 4.7 max (kontrolcü rolü) + 7 gece ARGE ajanı  
**Tarih**: 24 Nisan 2026 03:45 TR  
**Onay**: Yamac — Phase 29 başlangıç tarihi + Sprint 3C commit planı
