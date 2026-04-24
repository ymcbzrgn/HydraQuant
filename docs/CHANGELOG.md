# Changelog

Phase-by-phase release history. Phases are ~2–6 week sprints; each ships a themed set of capabilities.

## Phase 29 — Sensory Expansion & Self-Falsification

**Status**: Active (Apr 2026). See [PHASE29_ALPHA.md](PHASE29_ALPHA.md).

**Themes**: backtest framework revival (the "umarım çalışır" / "hope it works" gambit ends), new alpha sources (funding/OI divergence, on-chain netflow, microstructure), dynamic ATR Chandelier for payoff repair, UI/CDaR circuit breaker for death-by-thousand-cuts.

## Phase 28 — Database Evolution

**Shipped**: Apr 2026.

- **Grafeo graph database** — Rust-backed, 4 graph types (semantic / temporal / causal / entity), ZMQ single-writer broker at `ipc:///tmp/grafeo_hydra.sock`
- **LanceDB vector store** — 3 collections (crypto_news, crypto_news_bge, successful_trade_patterns); replaces legacy Chroma (migration complete)
- **DuckDB OLAP layer** — zero-copy `ATTACH` over SQLite for analytics; native tables for RL episodes and world-model snapshots

## Phase 27 — Asymmetric Alpha (Sprint 3A/3B + EK + Mega Sprint)

**Shipped**: Apr 15–23, 2026.

- **LinUCB contextual bandit** wired into LLM Router (EK.2.2) with retroactive reward on trade close
- **Shadow Kelly ledger** — `bayesian_kelly_shadow_per_pair` table isolated from real sizing via whitelisted `ALLOWED_KELLY_TABLES` set
- **Kelly hard-reset** — 56 pairs reset to Jeffreys prior after data-corruption incident
- **Tier 3.5 threshold 0.70** — fast-path Evidence Engine confidence bar raised
- **argument_quality backfill** — RLAIF learning loop activated
- **Hawkes branching-ratio veto** (HQ-13) — live in production sizing gate
- **Forgone Alpha Harvester** (HQ-11) — `pair_thresholds.forgone_alpha_7d` auto-tuning
- **ExploiterAgent** (HQ-15) — adversarial R2b debate with `exploit_archive` persistence
- **Hormonal hysteresis** (HQ-6) — peak-cortisol 24h half-life decay wired into Chandelier exit
- **MinStakeGuard rebalance** — lift tolerance 3× → 6× after BTC min-notional bottleneck study
- **Organism audit table** — event-sourced organism state transitions

## Phase 26 — CAAT / ML Organism Evolution

**Shipped**: Apr 2026. Full manifesto: [PHASE26_ML_ORGANISM_EVOLUTION.md](PHASE26_ML_ORGANISM_EVOLUTION.md).

- **15 cognitive processes** — perception, prediction, attention, working memory, long-term memory, reasoning, decision-making, action selection, learning, metacognition, self-model, mental simulation, affect, motivation, consciousness (simplified Φ)
- **33 modules shipped** across Sprint 1 (Triple Perception + chart features + OOD + CQR) and Sprint 2 (courtyard, reflection, multi-modal encoder, causal engine)
- **Triple Perception** — TTM 64-dim + Chronos-Bolt quantiles + CatBoost v2 193-feature; outputs `sizing_multiplier` consumed by strategy
- **193 chart features** — `chart_features.py` pipeline feeding CatBoost v2
- **Out-of-Distribution detector** — `ood_detector.py` on input features
- **Conformal Quantile Regression** — `cqr_wrapper.py` for uncertainty-aware forecasting
- **Self-Model introspection** — weekly scheduled meta-analysis, writes `self_model_profile`
- **Post-Trade Court** — RLAIF verdict per trade, blame assignment, root-cause categorization

## Phase 24–25 — Neural Organism

**Shipped**: Apr 2026.

- **1,788 adaptive neurons** across 298 parameters × 6 regimes
- **17 brain subsystems** — Hormones, Amygdala, Hippocampus, Synapses, PFC, Basal Ganglia, Proprioception, Immune Memory, Credit Assigner, Cerebellum, Predictive Model, Interoception, Mirror Neurons, Adaptive Immunity + SleepConsolidation + NeuroEvolution + DefaultModeNetwork
- **BCM metaplasticity + STDP** — textbook learning rules on every trade close
- **16-step update cycle** — deterministic per-trade organism mutation
- **52 scheduler jobs** — organism update cadences
- **Persistence**: `neuron_state`, `neuron_synapses`, `hormone_state`, `hormone_history`, `amygdala_state`, `hippocampus_episodes`, `immune_memory`, `immune_bcells`, `cerebellum_slots`, `evolution_population`

## Phase 23 — Jina Migration

**Shipped**: Apr 2026. Full notes: [PHASE23_JINA_MIGRATION.md](PHASE23_JINA_MIGRATION.md).

- Jina v3 embeddings (768-dim) replace legacy v2-base-en
- Jina reranker v3 replaces local ColBERT-only reranking
- API-first LLM: Gemini Flash → Groq → OpenRouter tier chain
- **3.5 GB RAM freed** on host by removing local embedding model

## Phase 21–22 — Hardening + systemd

**Shipped**: Mar–Apr 2026.

- systemd `.service` + `.service.watchdog` units with `--sd-notify` heartbeat
- `WatchdogSec=20` auto-restart on missed heartbeat
- Log rotation policy (journald default)
- Process isolation: 5 services (strategy, scheduler, rag, models, ai-api)
- Memory singletons in `scheduler.py` — 20 cache-held to prevent leaks

## Phase 20 — Hybrid Engine

**Shipped**: Mar 2026.

- **evidence_engine.py** — LLM-free 6-sub-score generator
- **agent_pool.py** — 10-agent MiroFish-inspired registry, regime-adaptive selection
- **opportunity_scanner.py** — 5-min scan of top 50 pairs with scoring

## Phase 19 — Evidence-Based RAG

**Shipped**: Mar 2026. Full notes: Phase 19 Evidence-Based RAG (archived — see `archive/docs/`).

- **PatternStatStore** — trade pattern retrieval backend
- **BacktestEmbedder** — convert backtest trades into searchable embeddings
- **5 RAG injections** into rag_graph — retrieved evidence at Decision, Risk, Sizing, Entry, Exit stages

## Earlier phases — Foundation (Feb–Mar 2026)

Phases 1–18 established the base architecture: MADAM multi-agent debate framework, Bayesian Kelly sizing, CAAT blend, Constitution guardrails, graduated autonomy L0–L5, RSS news pipeline, CryptoPanic + Fear & Greed integration, cross-pair intelligence, Telegram outbound notifier.

Phase-by-phase archive: `archive/docs/`.

## Notable corrections

- **CryptoPanic API deprecated** (~Apr 2026) — returns 404; `cryptopanic_fetcher.py` keeps class for import compat but `fetch() → []`. Sentiment flows through 24 RSS feeds + cryptocurrency.cv SSE.
- **Dream cadence** — started as weekly (heap-reclamation workaround), restored to **daily** via subprocess with `RLIMIT_AS=1.5 GB` (`dream_runner.py`).
- **Jina model version** — codebase uses `jina-embeddings-v3` (not v2-base-en as earlier docs claimed).
- **Provider count** — router has **7** providers (not 6 as earlier README claimed): Gemini / Groq / Cerebras / DeepSeek / SambaNova / Mistral / OpenRouter.

## Verified counts — Apr 2026 audit

Earlier README badges understated the system. Ground truth after audit:

| Metric | Old claim | Verified |
|---|---|---|
| AI modules | 67 | **123** |
| Lines of AI code | 28,000+ | **51,593** |
| Scheduler jobs | 33 | **66** |
| SQLite tables | 49 | **91** |
| FastAPI endpoints | 27 | **39** |
| LLM providers | 6 | **7** |
| Tests | 185 | **315** HQ-specific (240 passing on test_ai_scripts) |
| Strategy lines | 2,048 | **2,791** |
| Strategy AI-module imports | 22 | **32** |

See [README.md](../README.md) for current honest numbers.
