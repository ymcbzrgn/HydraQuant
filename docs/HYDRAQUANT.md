# HydraQuant — Manifesto

_"Cut one head, two more shall take its place."_

## The Hydra metaphor is the architecture

In myth the Hydra is terrifying because it has no single point of failure. Sever one head and two regenerate. HydraQuant is built the same way on purpose.

- **7 LLM providers** feed a Thompson + LinUCB router. Kill Gemini — a circuit breaker opens, Groq and Cerebras keep flowing.
- **3 macro-data feeds** cascade: yfinance → Yahoo HTTP → FRED. Any two can die and the Evidence Engine still reads DXY, VIX, and Treasury yields.
- **4 memory substrates** (SQLite transactional, LanceDB vectors, Grafeo graphs, DuckDB analytics) each address a distinct access pattern.
- **5 guardrails** — Constitution, VaR budget, Amygdala, Adaptive Immunity, PFC veto — are independent, not chained.
- **Dual Kelly ledger** — real + shadow — so forgone decisions stay calibrated without contaminating live sizing.

Redundancy is not the goal. Redundancy is the _side effect_ of the goal, which is: **the system keeps thinking even when a head has been cut off**.

## Philosophy — five principles that show up everywhere

### 1. Evidence first. LLMs only when they've earned their turn.

The LLM-free Evidence Engine runs on every candle close in ~50 ms. It decomposes the trade decision into six independent sub-questions (trend, momentum, crowd, evidence, macro, risk), scores each from 30+ data feeds, and returns a confidence number. If that confidence clears the neural organism's adaptive threshold (default 0.70), the trade ships **immediately** — and a background MADAM debate enriches the cache for the next cycle.

LLMs do not dictate. They earn their turn when remaining uncertainty justifies the cost.

### 2. Sizing, not blocking.

Confidence modulates **size**, not **permission**. A 0.40-confidence signal becomes a shadow trade that the Kelly calibrator studies forever. A 0.75-confidence signal becomes a standard position. A 0.95-confidence signal, in a favorable regime, with the organism's cortisol low and the immune system quiet, becomes a larger position.

Default is TRADE. The cost of false negatives — missed alpha, stale priors, atrophied calibration — is tracked in the `forgone_profit` ledger and proves itself over time.

### 3. Learn from everything, forget nothing important.

Every closed trade runs the **16-step organism update cycle**. Pheromone pressure, hormonal modulation, amygdala fear processing, hippocampal episode storage, prediction-error-weighted BCM credit, STDP temporal credit, synaptic propagation, 24-slot cerebellum timing, basal-ganglia habit consolidation, immune B-cell pattern memory, prefrontal executive veto, organ rebalance, persistence, evolution snapshot — every one of them is a full function, not a placeholder.

Neurons that update frequently become **resistant** to change (BCM metaplasticity). Parameters that change just before a trade get **more credit or blame** (STDP). The system does not forget — it also does not calcify.

### 4. Honest telemetry, path-dependent risk.

Cortisol decays with a 24-hour half-life — and it decays from _yesterday's peak_, not from "current stress." Past stress lingers. This is path-dependent risk appetite, modeled the way biology models it.

The `hypothetical_portfolio` table tracks "what if you had started with $100 and followed every signal?" The `forgone_profit` table tracks "what would you have earned on signals you didn't take?" The `bayesian_kelly_shadow_per_pair` table tracks "what would your Kelly posterior be if you counted shadow trades as real?"

Three independent views of what the system _actually did_ vs what it _might have done_. No marketing P&L.

### 5. Originality only when it's real.

We claim 15 candidate innovations across the codebase. Of those, 7 are live in production, 6 are documented research that has not shipped, and 2 are standard techniques we would rather not dress in novel language. You will find this distinction called out explicitly in the [README](../README.md) and the [Phase 27 R&D doc](PHASE27_ARGE.md).

We do not ship an inventory of "quantum superposition probabilistic sizing" when the math is expected-value weighting. We call it expected-value weighting.

## What HydraQuant is built around

### The Neural Organism (1,788 neurons, 17 subsystems)

Every formerly-hardcoded parameter became a **neuron** — a Beta-posterior sampled on demand, learning from every trade via BCM metaplasticity and STDP temporal credit. 298 parameters × 6 regimes (trending_bull, trending_bear, ranging, high_volatility, transitional, _global) = ~1,788 neurons. The brain around them has 17 functional subsystems: Hormones, Amygdala, Hippocampus, Synapses, Prefrontal Cortex, Basal Ganglia, Proprioception, Immune Memory, Credit Assigner, Cerebellum, Predictive Model, Interoception, Mirror Neurons, Adaptive Immunity, Default Mode Network, Sleep Consolidation, Neuroevolution.

Deep dive: [NEURAL_ORGANISM.md](NEURAL_ORGANISM.md).

### The Evidence Engine (LLM-free, regime-adaptive)

Six weighted sub-scores, dynamic-k sigmoid synthesis, regime-aware weight overrides, explicit abstention on data shortage. Never fabricates a NEUTRAL signal to keep downstream pipes fed. Zero API cost, ~50 ms end-to-end.

Deep dive: [EVIDENCE_ENGINE.md](EVIDENCE_ENGINE.md).

### The LLM Router (7 providers, Thompson + LinUCB)

Seven providers, 40–50 active model slots after multi-key expansion, Thompson Sampling for exploration + LinUCB contextual bandit for context-aware selection. Gemini circuit breaker with hysteresis prevents cascade failures. Retroactive reward on trade close feeds LinUCB posteriors with the outcome of every call that fired during the trade's lifetime.

Deep dive: [LLM_ROUTING.md](LLM_ROUTING.md).

### The Pheromone Field (lock-free stigmergy)

Instead of passing messages between modules (which requires locks, ordering, and careful state), subsystems **deposit** signals into a shared pheromone field. The field uses leaky-integrate LIF dynamics with MMAS bounds and half-life decay. Every reader sees a continuously decaying summary of what has happened recently; no one blocks anyone.

Hormonal state deposits with 10-minute half-life. Fear deposits with 5-minute half-life. Prediction/uncertainty signals drive the next-cycle cortisol rise automatically — if uncertainty > 0.7, cortisol += 0.1 on the next step, no explicit call needed.

### MADAM Debate + AgentPool

Twelve specialist agents in the registry, 4–7 selected per debate based on regime. Round 1: parallel analysis in a `ThreadPoolExecutor` with 12-second wall-time budget. Round 2: cross-examination — **skipped when R1 consensus ≥ 60%** to save LLM cost. Round 3: meta-synthesis by the coordinator.

Every debate persists to `agent_memory`, `argument_quality`, and the Grafeo causal graph. Low-quality arguments become negative reward in the RLAIF loop that retrains agent selection. The Exploiter agent runs adversarial attack scenarios in R2b; survivors are archived in `exploit_archive` for replay.

### Nine-Stage Sizing Pipeline

Beta posterior → Peters volatility drag → Baker-McHale shrinkage → trade graduation → ENB diversification scaling → CAAT × DualAxis × Cerebellum × Lifecycle weighted blend → Constitution clamp → Equal-Risk cap → MinStakeGuard. Every stage has a rollback.

After Phase 28's Triple Perception, the TTM + Chronos + CatBoost ensemble produces a `sizing_multiplier` consumed **before** the confidence curve is applied, so model-uncertainty directly scales position size.

## Why a cognitive organism instead of another ML pipeline

Classical ML pipelines treat every trade as an independent supervised-learning sample. They train on Monday, predict on Tuesday, retrain next week. Knowledge does not persist between runs; the model forgets.

HydraQuant is the opposite. Every trade mutates the organism. Hippocampus remembers the setup. Amygdala remembers the loss. Immune Memory remembers the pattern. Mirror Neurons update their model of the crowd. Cerebellum refines its 24-slot hour-of-day map. Sleep consolidates. Dreams generate counterfactuals. Evolution tournament-selects the best genome when performance degrades.

The organism is the model. Training is continuous. Forgetting is deliberate (SleepConsolidation decays stale synapses, breaks calcified habits).

## The questions this answers

**"Does it use reinforcement learning?"** Yes — RLAIF on the agent debate pipeline, Bayesian Kelly as a multi-armed bandit over pairs, LinUCB contextual bandit on LLM slots, Thompson Sampling across providers. We do not train a deep RL policy end-to-end because our reward signal is sparse, noisy, and non-stationary — we instead structure the problem as a collection of bandit-style decisions with strong priors.

**"Does it use LLMs?"** Yes, but only when the Evidence Engine's uncertainty justifies the cost. Most trading decisions never touch an LLM. When they do, 7 providers are available and the router chooses based on task context, current provider health, and learned reward.

**"Does it backtest?"** Not cleanly today — this is one of Phase 29's three critical tasks. The current backtesting path is gated behind `runmode not in ('dry_run', 'live')` guards in six places, and the HTTP RAG service fails per-candle under backtest load. Task 3A of [PHASE29_ALPHA.md](PHASE29_ALPHA.md) is the "lifeline" fix — mock LLM path, adapter pattern, replayable pipeline.

**"What is the P&L?"** Currently not profitable on live testnet — this is the honest status. Phase 29 is designed to fix exactly this: dynamic ATR Chandelier for payoff-ratio repair, UI/CDaR circuit breaker to stop death-by-thousand-cuts, sampling + voting MADAM to fix the martingale property of naive debate averaging.

**"Is it open-source?"** Yes — GPL v3, same as the underlying execution framework.

## What HydraQuant is not

- It is not a "one-click crypto bot." The graduated autonomy ladder (L0 → L5) exists precisely so the system earns trust before scaling risk.
- It is not a turnkey money machine. It is a research platform with live trading loops. The PnL is honest; the docs are honest.
- It is not a freqtrade fork in spirit. It uses freqtrade as a library for exchange integration. Every decision-making layer is HydraQuant.
- It is not finished. It is Phase 29 of at least thirty planned phases. The [Roadmap](ROADMAP.md) is public.

## Getting started

- [Quick Start in README](../README.md#quick-start)
- [Architecture deep dive](ARCHITECTURE.md)
- [Neural Organism walkthrough](NEURAL_ORGANISM.md)
- [Phase 29 Alpha blueprint](PHASE29_ALPHA.md)
- [Contributing](../CONTRIBUTING.md)
