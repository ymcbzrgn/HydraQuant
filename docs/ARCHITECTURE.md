# HydraQuant — Architecture

A technical deep dive. Every claim here is backed by `file.py:line` that you can grep for.

## Layers (top to bottom)

```
┌──────────────────────────────────────────────────────────────────┐
│  Presentation   FreqUI (Vue 3) · Telegram · Swagger /docs        │
├──────────────────────────────────────────────────────────────────┤
│  API            FastAPI :8890 · 39 GET endpoints                 │
├──────────────────────────────────────────────────────────────────┤
│  Orchestration  APScheduler · 66 jobs · 20-singleton cache       │
├──────────────────────────────────────────────────────────────────┤
│  Cognition      Evidence Engine · RAG · MADAM · AgentPool        │
│                 LLM Router (7 providers) · Neural Organism       │
├──────────────────────────────────────────────────────────────────┤
│  Decision       Kelly · CAAT · DualAxis · Cerebellum · Lifecycle │
│                 Constitution · RiskBudget · MinStakeGuard        │
├──────────────────────────────────────────────────────────────────┤
│  Data           Grafeo · LanceDB · SQLite · DuckDB · Pheromone   │
├──────────────────────────────────────────────────────────────────┤
│  Execution      freqtrade core · CCXT · Bybit / Binance          │
└──────────────────────────────────────────────────────────────────┘
```

## Entry point — signal flow in one page

1. **Candle close** arrives via freqtrade's exchange loop.
2. `HydraSizer.populate_entry_trend` (`user_data/strategies/HydraSizer.py:810`) calls Triple Perception.
3. **Triple Perception** (`triple_perception.py:227`) runs TTM embedding (64-dim) + Chronos-Bolt quantiles (p10/p50/p90) + CatBoost (193 features) + optional Kronos — fuses via CatBoost-primary, TTM×Chronos-fallback; outputs `sizing_multiplier = 1 / (1 + 5 × interval_width)`.
4. `HydraSizer._get_ai_signal` (`:613`) POSTs `http://127.0.0.1:8891/signal/{pair}` with a 120s timeout.
5. **RAG service** enters `rag_graph.get_trading_signal_with_timeout` (`rag_graph.py:2385`).
6. **Semantic cache lookup** — same pair + same regime + same hour → hit; otherwise Evidence Engine.
7. **Evidence Engine** (`evidence_engine.py:218`) runs `_gather → _analyze_patterns → _score_sub_questions → _detect_contradictions → _synthesize`. Returns `{signal, confidence, sub_scores}` in ~50ms.
8. Organism-adaptive threshold: `rag.evidence_first_threshold` (neural organism parameter, default 0.70).
9. **Fast path**: if `ee_conf ≥ threshold` → return signal; also fire a background MADAM to enrich the cache (`_background_madam` thread, `rag_graph.py:2115`).
10. **Slow path**: acquire `_signal_semaphore(timeout=30s)` → LangGraph pipeline.
11. **LangGraph nodes** in order: `analyze_technical` (`:877`), `analyze_sentiment` (`:990`), `analyze_news` (`:1114`), `research_bullish` (`:1215`), `research_bearish` (`:1326`), `coordinator_debate` (`:1439`).
12. **AgentPool.run_debate** (`agent_pool.py:570`) selects 4–7 specialists by regime → parallel R1 (12s budget) → conditional R2 if consensus < 60% → R3 meta-synthesis.
13. Signal returns to HydraSizer.
14. **9-stage sizing pipeline** applies (see below).
15. **Guards cascade**: Constitution → OrderFlow → SelfModel → RiskBudget → AutonomyManager → FundingRate → OpportunityScanner → EqualRisk.
16. **Shadow gate**: stake < $1 → log shadow trade, return 0.0.
17. **MinStakeGuard**: forced_ratio > 6× → log shadow, return 0.0; else lift to min_stake.
18. `confirm_trade_entry` (`:1800`) — post-quantization notional check.
19. Exchange executes.
20. Trade lifecycle: `custom_stoploss` (each candle, Chandelier) + `custom_exit` + `adjust_trade_position` (staged locks, gated DCA).
21. On close, `confirm_trade_exit` (`:1883`) → `BayesianKelly.update(won, pnl_pct)` + retroactive LinUCB nudge across every LLM slot that fired during the trade window.
22. **16-step organism update cycle** runs — see [NEURAL_ORGANISM.md](NEURAL_ORGANISM.md).

## Cognitive stack — component-by-component

### Evidence Engine

6 sub-scores, weights regime-overridden. Default weights:

```python
# evidence_engine.py:101
DEFAULT_WEIGHTS = {
    "q1_trend":    0.22,   # EMA alignment × ADX
    "q2_momentum": 0.20,   # RSI > 50 zone
    "q3_crowd":    0.22,   # F&G extremes × funding × L/S (contrarian)
    "q4_evidence": 0.15,   # k-NN + backtest + OHLCV ensemble
    "q5_macro":    0.10,   # DXY, VIX, BTC dominance
    "q6_risk":     0.11,   # ATR × volume
}
```

Regime overrides (`:113`): `ranging` shifts weight to crowd + evidence; `high_volatility` shifts to `q6_risk` (0.24).

Abstention: raises `EvidenceEngineDataError` on insufficient data instead of emitting a neutral 0.01 confidence row that would poison `signal_health`, RLAIF, and calibrator downstream.

### RAG Pipeline (25 types)

All 25 implemented in `user_data/scripts/`:

- Retrieval: `hybrid_retriever.py`, `semantic_cache.py`, `binary_quantizer.py`
- Quality: `crag_evaluator.py`, `self_rag.py`, `rag_evaluator.py`
- Reasoning: `cot_rag.py`, `speculative_rag.py`, `rag_fusion.py`
- Memory: `magma_memory.py`, `streaming_rag.py`, `bidirectional_rag.py`, `memo_rag.py`
- Structure: `raptor_tree.py`, `graph_rag.py`, `adaptive_router.py`
- Context: `cot_rag.py`, `regime_classifier.py`
- Self-improving: `pattern_stat_store.py`, `entity_extractor.py`
- Reranking: delegated via LanceDB + FlashRank

Adaptive Router routes queries to the right RAG based on a 4-tier complexity classification. The Agentic RAG consumes LanceDB's three tables: `crypto_news.lance`, `crypto_news_bge.lance`, `successful_trade_patterns.lance`.

### MADAM Multi-Agent Debate

`agent_pool.py:65-250` registry of 12 specialists. `select_agents` (`:502`) picks 4 baseline by regime, optionally adds ReflectionAgent if `agent.enable_r3_reflection > 0.5`, optionally ExploiterAgent + DefenderAgent if `agent.enable_r2b_adversarial > 0.5`. Net roster: 4–7 active per debate.

```python
# agent_pool.py:607 (R1 parallel)
futures = [executor.submit(_run_r1_agent, a, ...) for a in selected]
done, _ = wait(futures, return_when=ALL_COMPLETED, timeout=12.0)

# agent_pool.py:678 (R2 conditional skip)
consensus = len(dominant_verdict) / len(r1_verdicts)
if consensus >= R2_CONSENSUS_THRESHOLD:  # 0.60
    return r1_synth  # skip R2 to save cost
```

Graph persistence: `_record_debate_graph` (`:1299`) writes to Grafeo. Pheromone deposit: `_deposit_debate_pheromones` (`:1360`).

### LLM Router (7 providers)

`llm_router.py:597-722` loads slots:

| Provider | Models | Key-rotation |
|---|---|---|
| Gemini | Flash, Pro (11 env-key slots `GEMINI_API_KEY_1..10`) | yes |
| Groq | 7 models | single |
| Cerebras | 2 models | single |
| DeepSeek | 1 | single |
| SambaNova | 2 | single |
| Mistral | 2 | single |
| OpenRouter | 25+ free endpoints | single |

Each `ModelSlot` (`:206`) carries Thompson posterior (`alpha`, `beta`) and LinUCB state (`A` 5×5 identity, `b` zero-vector, `n_updates`). `_select_slots` (`:872`) scores with LinUCB UCB when warm (n ≥ 20), Thompson when cold.

Circuit breaker (`:422`): 10 failures / 60s → OPEN 30s; 3 consecutive successes → CLOSE.

Retroactive reward: on trade close, `HydraSizer.confirm_trade_exit` (`:1963-2014`) queries `llm_calls` for every call within the trade window, computes `nudge = 0.1 if pnl > 0 else -0.05 if pnl < 0 else 0.0`, and calls `slot.linucb_update(x_approx, nudge)`.

### Neural Organism

See [NEURAL_ORGANISM.md](NEURAL_ORGANISM.md) for full detail. Key facts:
- 2,249 lines in `neural_organism.py`, 19 classes, 17 functional subsystems
- 298 parameters × 6 regimes = 1,788 neurons
- 16-step update cycle runs on every trade close (`_organism_update` via `scheduler.py`)
- Four hormones (cortisol, dopamine, serotonin, adrenaline) with 24-hour half-life hysteresis
- Persistence: `neuron_state`, `neuron_synapses`, `hormone_state`, `hormone_history`, `amygdala_state`, `hippocampus_episodes`, `immune_memory`, `immune_bcells`, `cerebellum_slots`, `evolution_population`

### Pheromone Field

`pheromone_field.py` — lock-free. Every `deposit` uses LIF leaky integration:

```python
# pheromone_field.py:143
decay_factor = 0.5 ** (dt / half_life)
accumulated_value = accumulated_value * decay_factor + added
```

Bounded by MMAS `[TAU_MIN, TAU_MAX]` constants. The pheromone field is read by the Evidence Engine, the organism cycle, and the cerebellum timing module — all independently, without locks.

## Sizing — the 9-stage pipeline

Implemented in `position_sizer.per_pair_size` (`position_sizer.py:332-440`) and then blended in `HydraSizer.custom_stake_amount`:

1. **Beta posterior** — `p = α / (α + β)` per pair per regime
2. **Raw Kelly** — `f = (b·p − q) / b` where b = payoff ratio
3. **Peters vol drag** — `f − σ² / (2·b)` (arithmetic → geometric return penalty)
4. **Vol-of-vol shrinkage** — `σ² / (σ² + var(σ))`
5. **Baker-McHale** — `1 − σ_p² / (p·q)` (small-sample correction)
6. **Trade graduation** — {N<30: 0.125, <100: 0.25, <300: 0.50, else: 0.75}
7. **ENB scale** — `min(1, ENB / n_pairs)` (effective number of bets)
8. **Blend** — `0.45×CAAT + 0.30×DualAxis + 0.10×Cerebellum + 0.15×Lifecycle`, clamped `[0.20, 1.50]`
9. **Guard cascade** — Constitution (3% cap), RiskBudget (VaR 50% daily), AutonomyManager (L0-L5 Kelly scaling), Equal-Risk, MinStakeGuard, shadow gate

`sizing.kelly_floor_fraction = 0.015` (1.5%) floor protects against calibration corruption.

## Data substrate — four stores, four access patterns

### SQLite — `ai_data.sqlite` (91 tables)

Transactional, ACID, the primary source of truth.

- **Neural organism (8)**: `neuron_state`, `neuron_synapses`, `hormone_state`, `hormone_history`, `amygdala_state`, `hippocampus_episodes`, `sleep_log`, `cerebellum_slots`
- **Trading / Kelly (9)**: `bayesian_kelly`, `bayesian_kelly_per_pair`, `bayesian_kelly_shadow_per_pair`, `pattern_trades`, `forgone_profit`, `hypothetical_portfolio`, `portfolio_state`, `risk_budget`, `pair_thresholds`
- **Agent / decisions (9)**: `ai_decisions`, `ai_lessons`, `agent_memory`, `agent_performance`, `argument_quality`, `evidence_audit_log`, `self_model_profile`, `autonomy_state`, `decision_contract`
- **RL / replay (5)**: `rl_replay_buffer`, `rl_checkpoints`, `rl_relevance_feedback`, `rlaif_rewards`, `exploit_archive`
- **World model (3)**: `world_model_states`, `world_model_rollouts`, `dream_scenarios`
- **Market intel (8)**: `market_news`, `coin_sentiment_rolling`, `fear_and_greed`, `macro_data`, `derivatives_data`, `defi_data`, `search_trends`, `ohlcv_patterns`
- **Meta / evolution (9)**: `evolution_population`, `catboost_training_runs`, `hypothesis_history`, `causal_discoveries`, `counterfactual_results`, `graph_communities`, `sequence_patterns`, `dmn_discoveries`, `organ_performance_history`
- **Monitoring (7)**: `system_metrics`, `signal_health`, `interoception_state`, `llm_calls`, `model_slot_stats`, `organism_audit`, `autopoietic_integrity`
- **Other (remainder)**: `regime_layers`, `kg_entities`, `kg_relationships`, `magma_edges`, `opportunity_scores`, `immune_bcells`, `immune_memory`, `cross_pair_cache`, `backtest_processed`, `backtest_training_data`, `hot_buffer`, `linucb_state`, `memorag_global`, FTS5 shadow tables

### LanceDB — 3 vector tables

Dense embeddings for semantic retrieval. Jina v3 (768-dim) + BGE.

- `crypto_news.lance` — news article embeddings
- `crypto_news_bge.lance` — BGE-reranked news
- `successful_trade_patterns.lance` — trade-fingerprint retrieval

### Grafeo — `user_data/db/graphdb/hydra.grafeo`

Rust-backed graph store, accessed via ZMQ REP broker on `ipc:///tmp/grafeo_hydra.sock`. Single-writer (scheduler process) + multi-reader. Four valid graph types: `semantic`, `temporal`, `causal`, `entity`.

### DuckDB — `user_data/db/analytics.duckdb`

OLAP layer attached to the same SQLite via `ATTACH '…' AS ai (TYPE sqlite, READ_ONLY)` — zero-copy analytics. Native tables: `rl_episodes`, world-model snapshots. Consumers: `cerebellum_timing`, `rl_environment`, `iql_pretrain`.

## Scheduler — 66 jobs, 20 singletons

`scheduler.py` uses `apscheduler.schedulers.background.BackgroundScheduler`. Job cadences:

- **Every 5 min**: opportunity_scanner, cross_pair_intel
- **Hourly**: calibrator update, regime_classifier
- **Daily 00:05**: trade postmortem, RLAIF
- **Daily 00:30**: cerebellum timing (24-slot update)
- **Daily 02:30**: dream session via `dream_runner.py` subprocess with `RLIMIT_AS=1.5 GB`
- **Daily 04:00**: DMN counterfactual / synapse candidate discovery
- **Daily 05:15**: organism habit consolidation
- **Daily 23:55 UTC**: Telegram daily summary
- **Weekly Sunday 03:30**: sleep consolidation (synapse decay, stale habit break)
- **Weekly Sunday 04:00**: evolution tournament
- **Weekly Saturday 02:00**: causal discovery (PCMCI+ re-run)
- **Weekly Saturday 03:30**: GNN attention-based pattern discovery
- **Weekly Saturday 03:00**: self-model introspection
- **Weekly Sunday 23:55 UTC**: Telegram weekly summary

20 singletons cache-hold to prevent leaks: `_semantic_cache`, `_streaming_rag`, `_market_data_fetcher`, `_backtest_embedder`, `_magma_memory`, `_opportunity_scanner`, `_agent_pool`, `_cross_pair_intel`, `_system_monitor`, `_telegram_notifier`, `_bidi_rag`, `_calibrator`, `_forgone_engine`, `_evidence_engine`, `_cost_tracker`, `_autonomy_manager`, `_rag_evaluator`, `_graph_rag`, `_regime_classifier`, `_risk_budget`.

## Production layout — 5 services

1. **`hydraquant-models`** — BGE + ColBERT + FlashRank model server (:8895)
2. **`hydraquant-rag`** — RAG orchestrator (:8891)
3. **`hydraquant-ai-api`** — FastAPI read-only surface (:8890, 39 endpoints)
4. **`hydraquant-scheduler`** — APScheduler with 66 jobs + 20 singletons
5. **`hydraquant`** — HydraSizer strategy via freqtrade execution core

All behind systemd with `--sd-notify` watchdog (`WatchdogSec=20`) on the strategy unit.

## Execution constraints — what the server tolerates

- 32 GB ECC RAM, 4 core Platinum, 160 GB NVMe, no GPU
- Python 3.11+ (uv.lock locked)
- Bybit futures (isolated margin, one-way mode) or Binance testnet (spot/futures)
- Memory budget: ~1.0–1.5 GB steady state
- LLM cost: $0/month (all free tiers; pay-tier upgrade paths documented but not on by default)

## See also

- [Neural Organism](NEURAL_ORGANISM.md) — brain subsystems, hormones, 16-step cycle
- [Evidence Engine](EVIDENCE_ENGINE.md) — 6 sub-scores, dynamic-k sigmoid
- [LLM Routing](LLM_ROUTING.md) — bandits, circuit breaker, retroactive reward
- [Deployment](DEPLOYMENT.md) — runbook, systemd units, health dashboard
- [Phase 29 Alpha](PHASE29_ALPHA.md) — current sprint
