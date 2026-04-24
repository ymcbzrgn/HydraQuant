# HydraQuant — Feature Catalog

Every shipped feature, grouped by layer. Status markers:

- ✅ **LIVE** — in production
- ⚠️ **PARTIAL** — implemented but not fully wired
- 🧪 **RESEARCH** — documented, not yet shipped

## Cognition

### Evidence Engine — ✅ LIVE
LLM-free signal generation. 6 weighted sub-scores (trend / momentum / crowd / evidence / macro / risk). Regime-adaptive weights. Dynamic-k sigmoid synthesis. Abstention on data shortage instead of neutral-signal noise. ~50 ms latency, $0 cost.

### 25 RAG Types — ✅ LIVE
Hybrid retrieval (Dense + BM25 + ColBERT + RRF), CRAG corrective retrieval, Self-RAG reflective loop, FLARE forward-looking, CoT-RAG, Speculative RAG, HyDE, RAG-Fusion, MemoRAG global memory, StreamingRAG hot/cold tiers, Bidirectional RAG lessons, MAGMA 4-graph memory, RAPTOR hierarchical, GraphRAG community, Adaptive Router 4-tier classification, Contextual Retrieval (Anthropic-style), Regime-Aware Filter, Event-Driven Temporal, Outcome-Based Chunk Scoring (PnL → quality), Agentic RAG, ColBERT v2 reranker, FlashRank reranker, Binary Quantization, Semantic Cache, RAGAS feedback loop.

### MADAM Multi-Agent Debate — ✅ LIVE
12 specialists, 4–7 selected per debate by regime. 3-round protocol: R1 parallel analysis (12s wall budget), R2 cross-examination (skipped if consensus ≥ 60%), R3 meta-synthesis. Exploiter agent runs adversarial scenarios; survivors archived in `exploit_archive`.

### LLM Router — ✅ LIVE
7 providers (Gemini / Groq / Cerebras / DeepSeek / SambaNova / Mistral / OpenRouter). 40–50 active model slots after multi-key expansion. Thompson Sampling (warm) + LinUCB contextual bandit (warm n≥20). Gemini circuit breaker with hysteresis. Retroactive reward on trade close feeds every slot that fired during the trade window.

### Triple Perception — ✅ LIVE
TTM (Tiny Time Mixer) 64-dim embedding + Chronos-Bolt quantiles (p10/p50/p90) + CatBoost v2 (193 features). Optional Kronos when `HQ_ENABLE_KRONOS=1`. Fused by CatBoost-primary with TTM×Chronos-agreement fallback. Outputs `sizing_multiplier = 1 / (1 + 5 × interval_width)`.

## Neural Organism

### 17 Brain Subsystems — ✅ LIVE
Hormones (4-hormone endocrine simulation with 24h hysteresis), Amygdala (4-tier graduated fear), Hippocampus (7-dim fingerprint episodic memory), Synapses (13 causal edges with 0.3-damped propagation), Prefrontal Cortex (5 hard executive rules), Basal Ganglia (habit consolidation), Proprioception (lifecycle phase awareness), Immune Memory (per-pair graduated bans), Credit Assigner (organ-typed × STDP weight), Cerebellum (24-slot hour-of-day timing), Predictive Model (Free Energy Principle prediction error → LR boost), Interoception (8 internal sensors), Mirror Neurons (crowd inference), Adaptive Immunity (B-cell pattern memory), Default Mode Network (idle counterfactual discovery), Sleep Consolidation (synapse decay + stale habit breaking), Neuroevolution (tournament selection with 10% genome blend).

### BCM Metaplasticity + STDP — ✅ LIVE
Neurons update with Bienenstock-Cooper-Munro sliding threshold and dead zone `|reward| < θ_m`. Parameters changed just before a trade receive more credit/blame via STDP exponential decay. Published learning rules, full implementation.

### Stigmergic Pheromone Field — ✅ LIVE
Lock-free coordination. LIF leaky integrate dynamics. MMAS `[TAU_MIN, TAU_MAX]` bounds. No RLock, no race conditions — modules deposit and decay naturally. Hormonal state (10-min half-life), fear (5-min half-life), prediction/uncertainty signals drive next-cycle organism state automatically.

### Dream Engine — ✅ LIVE (DAILY)
MBPO-style branching world-model rollouts from real state snapshots. 3-layer validity filter: Mahalanobis chi-squared 95%, reward magnitude cap (<3× real max), transition smoothness (<5.0). Valid dreams write to `rl_replay_buffer` with `source='dream'`. Runs daily 02:30 UTC in subprocess with `RLIMIT_AS=1.5 GB` hard cap.

### GNN Organism — ✅ LIVE
2-layer Graph Attention Network over Grafeo-exported causal graph. Node features from `ai_decisions` (avg_pnl, avg_conf, trade_count). `_discover_patterns` mines high-attention edges, persists to `causal_discoveries` with `method='GNN_attention'`. Scheduled weekly Saturday 03:30.

## Risk Management

### Bayesian Kelly (per-pair per-regime) — ✅ LIVE
Beta posterior with Jeffreys prior (α=β=2.0), 0.98 decay, effective window ~50 trades. Automatic reset to Jeffreys when β/α > 20 after 100+ trades. `sizing.kelly_floor_fraction = 0.015` protects against calibration corruption. 56-pair state.

### 9-Stage Sizing Pipeline — ✅ LIVE
Beta posterior → Peters vol drag → vol-of-vol shrinkage → Baker-McHale small-sample correction → trade graduation ({N<30: 0.125, <100: 0.25, <300: 0.50, else: 0.75}) → ENB scale → CAAT × DualAxis × Cerebellum × Lifecycle blend → Constitution clamp → MinStakeGuard.

### Shadow Kelly Ledger — ✅ LIVE
`bayesian_kelly_shadow_per_pair` table isolated from real sizing via whitelisted `ALLOWED_KELLY_TABLES` set and separate singletons (`get_real_kelly()` vs `get_shadow_kelly()`). Enables calibration without contaminating bets.

### Chandelier Exit — ✅ LIVE
Confidence-adaptive ATR multipliers: 1.5× (conf ≥ 0.80), 1.35× (≥ 0.60), 1.2× (else). Hurst-tuned up to 3.5× in trending markets (Hurst > 0.55) or down to 1.5× in noise (Hurst < 0.45). Cortisol-scaled: `mult *= (2.0 − cortisol)`.

### Constitution — ✅ LIVE
8 hard rules: max_drawdown 25%, max_single_position 3%, max_leverage 5×, max_portfolio_heat 10%, ATR×leverage ≤ 8, adrenaline freeze at stress ≥ 0.85, 5 consecutive losses → 24h freeze, free RAM < 500 MB → kill.

### Graduated Autonomy (L0–L5) — ✅ LIVE
Kelly fraction scaling: L0 0.03 → L5 0.75. Promotion gates per level: min trades, min Sharpe, max DD, min days. Promotion/demotion push Telegram alerts.

### Forgone Alpha Harvester (HQ-11) — ✅ LIVE
Every skipped signal logged. If `pair_thresholds.forgone_alpha_7d` consistently positive → auto-lower per-pair confidence threshold. Closed-loop self-tuning.

### Hawkes Branching-Ratio Veto (HQ-13) — ✅ LIVE
Self-exciting process on trade arrivals per pair. Veto new entries when `n = α/β ≥ 0.95`. Weekly MLE refit. 6-tier sizing modifier before hard veto.

### UI/CDaR Circuit Breaker — 🧪 Phase 29 Task 3.2
Drawdown-to-capital tracking with Under-Water and Conditional Drawdown-at-Risk stops. Protects against death-by-thousand-cuts.

## Data & Memory

### 8 Data Sources, $0/month — ✅ LIVE
Bybit public API (OI, funding, L/S ratio), DeFi Llama (TVL, stablecoin supply), CoinGecko (BTC dominance, total cap), Yahoo Finance HTTP, FRED (Treasury yields, VIX fallback), 24 RSS feeds (CoinDesk, CoinTelegraph, Decrypt, The Block, CryptoSlate, CryptoPotato …), Google Trends, cryptocurrency.cv SSE stream.

### 91-Table SQLite Schema — ✅ LIVE
Neural organism (8), trading/Kelly (9), agent decisions (9), RL/replay (5), world model (3), market intel (8), meta/evolution (9), monitoring (7), other (remainder). See [ARCHITECTURE.md](ARCHITECTURE.md) for the full grouped list.

### Grafeo Graph Database — ✅ LIVE (Phase 28)
Rust-backed, 4 graph types (semantic, temporal, causal, entity), ZMQ REP broker on `ipc:///tmp/grafeo_hydra.sock`, single-writer + multi-reader. Replaces per-agent RLocks across MADAM.

### LanceDB Vector Store — ✅ LIVE (Phase 28)
3 tables: `crypto_news.lance`, `crypto_news_bge.lance`, `successful_trade_patterns.lance`. Jina v3 + BGE 768-dim embeddings. Replaces legacy Chroma (migration complete).

### DuckDB OLAP Layer — ✅ LIVE (Phase 28)
Zero-copy attach over SQLite for analytics queries. Native tables: `rl_episodes`, world-model snapshots. Consumers: cerebellum timing, RL environment, IQL pretraining.

## Orchestration

### 66 Scheduler Jobs — ✅ LIVE
In-process APScheduler with 20 singleton modules for leak prevention. Replaces a cron + airflow stack. Jobs span 5-minute (opportunity_scanner), hourly (calibrator), daily (organism cycles, Telegram summary), and weekly (evolution, causal discovery, sleep).

### FastAPI Read Surface — ✅ LIVE
39 GET endpoints on uvicorn :8890. Swagger `/docs` enabled. Covers signals, risk, evidence, agent pool, organism, causal graph, counterfactuals, LLM routing insights, autonomy state, cost, forgone P&L, ablation, benchmark.

### Telegram Outbound Notifier — ✅ LIVE
Plain `httpx` POST to `api.telegram.org/bot{token}/sendMessage`. 4 typed senders: trade signals, daily summary (23:55 UTC), weekly summary (Sun 23:55 UTC), alerts with 6h cooldown dedup. Markdown formatted. No command surface — broadcast-only.

### Systemd + Watchdog — ✅ LIVE
`hydraquant.service.watchdog` uses `--sd-notify` heartbeat (`WatchdogSec=20`) — systemd auto-restarts if no heartbeat for 20 s.

### Deploy Health Dashboard — ✅ LIVE
`deploy_health_check.py` — 12-section verdict (systemd, OOM, signal distribution, LLM propagation, Kelly, LinUCB warmth, trade performance, argument quality, causal/dream/hypothesis, scheduler RSS, memory trend, tracebacks). Exit contract: 0 / 1 / 2.

## Frontend

### FreqUI Extension — ✅ LIVE
Vue 3 + PrimeVue + TailwindCSS 4 + Vite. 15 views total, 3 HydraQuant-exclusive: `AIAnalyticsView.vue`, `AISettingsView.vue`, `RiskDashboardView.vue`. 9 AI components: `AIDashboard`, `AISignalPanel`, `AutonomyLevel`, `ConfidenceScore`, `ForgonePnLTracker`, `ModelStatusCard`, `RiskPanel`, `SentimentDisplay`, `TradeReasoning`.

## Infrastructure

### Thompson Sampling LLM Router — ✅ LIVE
Beta posterior per `(provider, model, api_key)` slot. RPM/RPD caps respected. Persistence: pickled LinUCB numpy posterior restored on boot.

### LinUCB Contextual Bandit — ✅ LIVE (Phase 27 EK)
5-dim feature vector (`task`, `prompt_len`, `needs_json`, `regime_vol`, `hour_utc`). UCB score = `θᵀx + α√(xᵀA⁻¹x)` with α=1.5. Cold-start n<20 falls back to Thompson.

### Retroactive LLM Feedback — ✅ LIVE (Phase 27 EK)
On trade close, every LLM call logged with `trading_pair = pair` during the trade window gets a `linucb_update` nudge. `+0.1` on win, `-0.05` on loss, `0.0` on neutral.

### Gemini Circuit Breaker — ✅ LIVE
Sliding window, thread-safe deque. 10 failures / 60s → OPEN 30s; 3 consecutive successes → CLOSE. Prevents cascade failures.

### Semantic Cache — ✅ LIVE
Runtime TTL cache on query embeddings. NEUTRAL signals cached 0.9 h, others 6 h. Same pair + same regime + same hour short-circuits Evidence Engine.

## Testing & Quality

### 314 HydraQuant Tests — ✅ LIVE
`test_ai_scripts.py` (242) + `test_phase26_modules.py` (43) + `test_chart_features.py` (23) + `test_bidirectional_rag.py` (3) + `test_memorag.py` (4). Current pass rate: 240/241 (99.6%) on `test_ai_scripts.py`. Runtime: ~125s.

### Ruff + Mypy + Pre-commit — ✅ LIVE
`ruff check` + `ruff format` (line length 100). `mypy` on scripts + tests. 8 pre-commit hooks (flake8, mypy, isort, ruff, pre-commit-hooks, exif-stripper, codespell, zizmor).

### Contract Tests — ✅ LIVE
`test_rag_graph_no_hardcoded_sentiment`, `test_all_scripts_import_ai_config`, `test_all_rag_invoke_sites_propagate_pair_and_task_context`, `test_shadow_kelly_ledger_is_isolated`, `test_linucb_score_prefers_positively_rewarded_slot`. Architecture enforcement.

### CI — ✅ LIVE (partial)
GitHub Actions, 24-cell matrix (6 OS × 4 Python versions), `pytest-xdist`, random order, Codecov. Coverage currently only measures the freqtrade core layer — extending to `user_data/scripts/` is a Phase 30 goal.

## Research — Not Yet Shipped

### HQ-4 Dream Coherence Score — 🧪
Variance-ratio over world-model rollouts from same initial state. Low variance → trust; high variance → model hallucinating. Meta-cognitive model reliability metric.

### HQ-5 Causal Entropy — 🧪
Shannon entropy over PCMCI+ causal edge strengths. Low H = simple market = size up.

### HQ-7 Information Asymmetry Index — 🧪
Fraction of our causal edges that diverge from public consensus (funding/sentiment). Alpha proxy.

### HQ-8 Temporal Attention Sequences — 🧪
Transformer over hour-of-day embeddings predicting next-hour win rate conditional on recent hour sequence.

### HQ-14 SIR Sentiment Contagion — 🧪
Epidemiological SIR model over F&G velocity. `R_t > 1` detects panic wave.

### Backtest Tier 1 (Mock LLM) — 🧪 Phase 29 Task 3A
Full backtest replay with in-memory LLM mock. Unblocks every other Phase 29 fix from "umarım çalışır" ("hope it works") to measurable.

### Funding/OI Divergence — 🧪 Phase 29 Task 1.1
CCXT adapter for Bybit funding rate and open-interest velocity. 7th Evidence Engine sub-score.
