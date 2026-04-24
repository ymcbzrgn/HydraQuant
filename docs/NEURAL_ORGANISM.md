# Neural Organism

HydraQuant's most distinctive subsystem. `user_data/scripts/neural_organism.py` — 2,249 lines, 19 classes, 17 functional subsystems, 1,788 regime-expanded neurons.

This document is a verified walkthrough. Every file and line reference below has been grep-validated. No hallucination.

## Why the brain metaphor is architecture, not branding

Classical ML trading systems have one weight vector that gets retrained on a schedule. Between retrains, the model is static. Between runs, the model forgets.

The neural organism does the opposite. Every formerly-hardcoded parameter (learning rates, thresholds, risk multipliers, weights, exponents) became a **neuron** — a Beta-posterior that gets sampled on demand via Thompson draw and learns from every trade via BCM metaplasticity and STDP temporal credit. 298 parameters × 6 regimes = 1,788 neurons, all persisted, all continuously updating.

Around those neurons live 14 biologically-inspired subsystems that each do one thing the system actually needs — and do it the way biology does it, not because we read a pop-neuroscience book but because the biology happens to be a good solution to a hard non-stationary-credit-assignment problem.

## The 19 classes and 17 functional subsystems

Internally there are 19 classes in `neural_organism.py`. Functionally they group into 17 subsystems (ParamNeuron is the building block for everything else; NeuralOrganism is the orchestrator):

| Subsystem | File:Line | Biological analogue | What it does in HydraQuant |
|---|---|---|---|
| **ParamNeuron** | `:511` | Hebbian neuron + Beta posterior | `sample()` Thompson-draws the current parameter value. `nudge()` updates via BCM: `alpha += mag` on LTP, `beta += mag` on LTD. Dead zone `|reward| < θ_m`. STDP via `last_update_time`. |
| **Hormones** | `:581` | Endocrine system | cortisol · dopamine · serotonin · adrenaline. `compute()` reads F&G, drawdown, consecutive losses → `cortisol = max(0.5, 1.0 − stress × 0.4)`. `adrenaline = 0` at stress > 0.85 → PANIC. |
| **Amygdala** | `:695` | Fear center | 4-tier FEAR_TIERS: `-2% normal` → `-5% stress` → `-10% fear` → `-15% panic`. `get_current_fear()` decays as `peak × 0.5^(hours/24)`. |
| **Hippocampus** | `:738` | Episodic memory | `get_fingerprint()` — 7-dim bucketed JSON (F&G × regime × ADX × funding × streak × stress). `store_episode()` → `hippocampus_episodes`. `recall(fp, k=5)`. |
| **Synapses** | `:804` | Neural connections | 13 SEED_SYNAPSES edges (`:474`). `propagate(delta)` pulls `delta × weight × sign × 0.3` to each 1-hop downstream neuron. |
| **PrefrontalCortex** | `:829` | Executive veto | 5 non-learning rules: (1) leverage cap 5×, (2) adrenaline freeze non-essential organs, (3) Hippocampus 3/5 similar lost >3% → halve sizing, (4) info_q < 0.4 → cap confidence 0.65, (5) max_equity_loss bound. |
| **BasalGanglia** | `:884` | Habit consolidation | `check_consolidation()` — if update_count ≥ 50 AND belief_width() < 0.05 → `prior_strength += 10`. Mature neurons resist change. |
| **Proprioception** | `:909` | Lifecycle phase | `assess()` returns phase from average belief width: "learning" (LR×1.5), "maturing", "mature" (LR×0.7), "overconfident" (LR×0.5). |
| **ImmuneMemory** | `:938` | Graduated pair bans | `compute_ban()` = `base × (1 + |loss| × mult) × consec^(n-1)`, capped. Writes `immune_memory`. Strategy calls `lock_pair()` at `HydraSizer.py:2187`. |
| **CreditAssigner** | `:989` | Temporal credit | Organ-typed × STDP: `0.3 + 0.7 × exp(−hours/2)` for ∈[0,6], else 0.3. SIGNAL_ORGANS credit `pnl × conf × 0.5`; SIZING_ORGANS `pnl × stake × 0.3`; DEFENSE_ORGANS `pnl × dur_norm × 0.4`. |
| **Cerebellum** | `:1047` | Motor timing | 24-slot hour model; `get_hour_multiplier() = clamp(0.6 + 0.8 × win_rate, 0.6, 1.4)`. Persists to `cerebellum_slots`. Separate `cerebellum_timing.py` reads same data 30-day-window via DuckDB. |
| **PredictiveModel** | `:1089` | Free Energy Principle | `predict_expected_pnl(fp)` = avg last-10 episodes. `compute_prediction_error()` = `|actual − expected| / max(...)`. `get_lr_boost() = 1 + last_error` ∈ [1.0, 2.0]. |
| **Interoception** | `:1135` | 8-sensor self-monitoring | param drift velocity, belief width avg, prediction error avg, hormone stability, trade frequency, 30d win rate, data completeness, consec same-direction. Returns organism_health ∈ [0, 1] — logged only today. |
| **MirrorNeurons** | `:1195` | Crowd inference | Aggregates funding rate (±0.0005 thresholds), L/S ratio (1.2 / 0.8), OI change. `crowd_is_wrong_rate` EMA (lr=0.05). Returns `{direction, intensity, contrarian_signal}`. |
| **AdaptiveImmunity** | `:1263` | B-cell memory | `encounter_threat()` → `immune_bcells`. `check_threat()` — antibody = `1 + encounters × 0.1`, sizing_reduction = `1 / antibody`. |
| **DefaultModeNetwork** | `:1339` | Idle counterfactual | Reads worst-5 `hippocampus_episodes`, finds fingerprints where `good_similar > 0`. Discovers synapse candidates (log only — no auto-add). |
| **SleepConsolidation** | `:1400` | Replay + pruning | Replay last 50 episodes → log profitable fingerprints. Synapse decay `× 0.95`. Stale habit break: `prior_strength −= 5` when prior > 15 AND θ_m < 0.001 AND count > 100. |
| **NeuroEvolution** | `:1465` | Tournament selection | `snapshot_current()` writes `evolution_population` (params_json, fitness, generation). Keeps top N=5. `run_tournament()` — blend 90% current + 10% best when underperforming. |
| **NeuralOrganism** | `:1534` | Orchestrator | Holds all 17 subsystems. Singleton via `get_organism()` (`:2165`). Main loop `update_cycle()`. |

## The four hormones

Implemented `:581–691`. Reads:

- `cortisol` — high when drawdown + consecutive losses + low F&G. Multiplies Chandelier ATR (`mult *= (2.0 − cortisol)`). Hysteresis: tracks `_trough_cortisol` and decays from yesterday's peak with 24h half-life (`:611–627`).
- `dopamine` — reward signal. Rises on wins, decays.
- `serotonin` — mood / steady-state. Reads streak stability.
- `adrenaline` — stress. `0.0 if stress > 0.85 else 1.0`. Binary — when it hits 0, PFC freezes all non-essential organs (`:849`).

Persistence: `hormone_state` (singleton row) + `hormone_history` (every update).

## Stigmergic pheromone field

`pheromone_field.py` — 537 lines. Supplements the organism by providing lock-free module coordination.

```python
# pheromone_field.py:143 (LIF leaky integrate)
decay_factor = 0.5 ** (dt / half_life)
accumulated_value = accumulated_value * decay_factor + added
```

The organism deposits:
- `HORMONE_STATE` with half-life 600s (`:1823`)
- `FEAR_LEVEL` with half-life 300s (`:1834`)

The organism reads (step 0 of the cycle):
- `prediction`, `uncertainty`, `organism_health` (`:1791–1798`)
- If `uncertainty > 0.7` → `cortisol += 0.1` on the next step (`:1800`)

Cerebellum timing, Self-Model, and triple perception also deposit/read — no locks anywhere in the field.

## The 16-step update cycle

Runs on every trade close via `NeuralOrganism.update_cycle()` at `:1790–1935`. Entry point: `HydraSizer.confirm_trade_exit:2173–2186`.

1. **Pheromone READ** (`:1791`) — prediction, uncertainty, organism health.
2. **SENSE** (`:1808`) — Proprioception phase + LR mod.
3. **PREDICT** (`:1812`) — Hippocampus fingerprint + PredictiveModel expected PnL.
4. **MIRROR** (`:1819`) — MirrorNeurons crowd analysis.
5. **HORMONES** (`:1822`) — compute + allostasis + deposit HORMONE_STATE.
6. **FEAR** (`:1841`) — Amygdala process loss + deposit FEAR_LEVEL.
7. **MEMORY** (`:1852`) — Hippocampus store episode + recall k=5.
8. **PREDICTION ERROR** (`:1856`) — PredictiveModel compute error + LR boost.
9. **INTEROCEPTION** (`:1860`) — update 8 sensors.
10. **LEARN** (`:1866`) — `lr_mod × fear_learn × pred_lr_boost`; CreditAssigner.assign → neuron.nudge.
11. **PROPAGATE** (`:1877`) — SynapseNetwork.propagate on top-5 changed neurons.
12. **TIMING** (`:1882`) — Cerebellum record outcome (hour, won, pnl).
13. **HABITUATE** (`:1885`) — BasalGanglia check_consolidation.
14. **IMMUNE** (`:1889`) — ImmuneMemory record_loss + AdaptiveImmunity encounter_threat.
15. **VETO** (`:1901`) — AdaptiveImmunity check_threat + PrefrontalCortex evaluate.
16. **REBALANCE + PERSIST + EVOLVE** (`:1914–1921`) — ORGAN_CONSTRAINTS normalize, `_maybe_persist` batch write, NeuroEvolution snapshot every 50 trades.

## Biologically-inspired proof points

1. **BCM metaplasticity + STDP** (`:540–565`, `:1000–1006`) — textbook plasticity rules, not invented terminology. Sliding threshold θ_m + dead zone; exponential decay temporal credit.
2. **24-hour hormonal hysteresis** (`:611–627`) — tracks trough cortisol, decays "distance from calm" as `(1 − trough) × 0.5^(hours/24)`. Allostatic load, not homeostatic. Survives bot restarts (`:1577–1580`).
3. **Hippocampal fingerprint → prefrontal veto** (`:738`, `:857–866`) — deterministic 7-dim JSON key → `recall(k=5)` → PFC Rule 3: "3/5 similar lost >3% → halve sizing." Hippocampal memory feeding executive function.
4. **Free Energy → learning-rate boost** (`:1089`, `:1857`) — `lr_mod *= (1 + prediction_error)` ∈ [1, 2]. Surprise directly boosts plasticity. Friston's principle applied with honest math.
5. **Stigmergic lock-free coordination** — no RLock in hormone/pheromone paths. Modules deposit, decay naturally via LIF with MMAS bounds.

## What's fully wired vs observer-only

**Fully wired (modify trades)**:
- Amygdala — `sizing_mult` on next entry
- PrefrontalCortex — freezes neurons, caps leverage 5×, halves sizing on hippocampus warn
- ImmuneMemory — BLOCKS pair via `lock_pair()`
- AdaptiveImmunity — reduces sizing on known threats
- Hormones (cortisol) — feeds hormonal_scalar into sizing
- CreditAssigner + BCM+STDP — writes every neuron's `current_val`; consumed by 27+ modules
- Cerebellum timing — hour multiplier consumed by autonomous_lifecycle
- NeuroEvolution — 10% blend toward best when underperforming
- SleepConsolidation — decays synapses, breaks stale habits

**Observer-only (for now)**:
- Interoception — returns `organism_health`, only logged
- MirrorNeurons — `contrarian_signal` returned but no consumer reads it into sizing yet
- DefaultModeNetwork — logs counterfactuals + synapse candidates; no auto-add
- Proprioception `safety_mod` — returned but not wired

These are wiring gaps we are honest about. Phase 30 roadmap addresses them.

## How to inspect the organism at runtime

- **DB**: `sqlite3 user_data/db/ai_data.sqlite "SELECT param_id, current_val, alpha, beta_param FROM neuron_state WHERE param_id LIKE 'evidence.%' LIMIT 20"`
- **API**: `curl http://127.0.0.1:8890/api/ai/organism | jq`
- **Dashboard**: `python user_data/scripts/deploy_health_check.py` — section 8 (argument_quality learning) + section 11 (memory trend)
- **Grafeo causal graph** (if running): `grafeo-cli query --db user_data/db/graphdb/hydra.grafeo --type causal`

## Related docs

- [Architecture](ARCHITECTURE.md) — where the organism sits in the overall stack
- [Evidence Engine](EVIDENCE_ENGINE.md) — the LLM-free signal path that respects organism-adaptive thresholds
- [Phase 26 — ML Organism Evolution](PHASE26_ML_ORGANISM_EVOLUTION.md) — the CAAT manifesto that produced this subsystem
- [HydraQuant manifesto](HYDRAQUANT.md) — why the brain metaphor is load-bearing
