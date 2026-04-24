# LLM Routing

The LLM Router selects **which model to ask** for every inference call HydraQuant makes. With seven providers, ~79 slots, non-stationary quality-vs-latency tradeoffs, and free-tier rate limits to respect, the decision is harder than it looks.

File: `user_data/scripts/llm_router.py` (1,408 lines).

## The seven providers

Loaded dynamically from environment variables (`llm_router.py:597–722`):

| Provider | Base URL | Slots | Free tier |
|---|---|---|---|
| **Gemini** | via `ChatGoogleGenerativeAI` | 11 (`GEMINI_API_KEY_1..10` + primary) | Yes — generous daily quota |
| **Groq** | native `ChatGroq` | 7 models | Yes |
| **Cerebras** | `https://api.cerebras.ai/v1` via OpenAI-compat | 2 | Yes |
| **DeepSeek** | OpenAI-compatible | 1 | Yes |
| **SambaNova** | `https://api.sambanova.ai/v1` | 2 | Yes |
| **Mistral** | native | 2 | Yes |
| **OpenRouter** | OpenAI-compatible | 25+ free endpoints | Yes |

Total active slots depend on env keys — typically 40–50 in production.

## ModelSlot state (`llm_router.py:206`)

Each slot carries:

- **Thompson posterior** — `alpha`, `beta` of a Beta distribution over the success indicator (0/1 quality threshold)
- **LinUCB state** (Phase 27 EK):
  - `linucb_A` — 5×5 identity initially, becomes the feature-covariance accumulator
  - `linucb_b` — 5-dim zero initially, becomes the reward-weighted feature sum
  - `linucb_n_updates` — counter for cold-start detection

- **Rate limits**: RPM / RPD caps per provider
- **Last success / failure timestamps** — used by circuit breaker
- **Latency EMA** — rolling p50 for routing decisions

## Thompson Sampling (exploration)

For a cold slot (`n_updates < 20`), we rely on Thompson:

```python
# pseudocode
score = sample_beta(alpha=slot.alpha, beta=slot.beta)
```

Beta starts at `(2, 2)` — Jeffreys prior. Wins increment alpha, failures increment beta. Over time the posterior tightens around the true quality rate. Sampling (instead of taking the mean) introduces exploration naturally — a slot with `(4, 6)` posterior sometimes outranks one with `(7, 3)` and gets selected.

## LinUCB Contextual Bandit (exploitation + context)

Once a slot has seen 20+ calls, we switch to LinUCB (`llm_router.py:240, 252–270`):

```python
# llm_router.py:252 — UCB score
A_inv = np.linalg.inv(self.linucb_A)
theta = A_inv @ self.linucb_b
mean = float(theta @ x)
bonus = float(alpha_ucb * np.sqrt(max(float(x @ A_inv @ x), 0.0)))
return mean + bonus   # alpha_ucb = 1.5 default

# llm_router.py:266 — update
self.linucb_A += np.outer(x, x)
self.linucb_b += float(reward) * x
self.linucb_n_updates += 1
```

The **5-dim context vector `x`** comes from `llm_features.extract_features(task_context)`:

1. `task` — categorical → embedding (technical / sentiment / news / bull / bear / coordinator / pool / court)
2. `prompt_len` — log-scaled token count
3. `needs_json` — boolean (structured output request)
4. `regime_vol` — current market volatility regime index
5. `hour_utc` — cyclical encoding (sin/cos pair reduced to one bucket)

The key insight: **a slot that excels at technical_analysis at low prompt lengths during low-vol regimes** is different from the same slot on long adversarial prompts during high-vol. LinUCB learns that context.

## Cold-start fallback

`_select_slots` (`:872`) checks `n_updates` first:

```python
if slot.linucb_n_updates < 20:
    score = slot.thompson_sample()
else:
    score = slot.linucb_score(context_vector)
```

Cold slots are kept fairly competitive via Thompson so they get explored.

## Retroactive reward

This is the mechanism that makes LinUCB actually work for trading. On every trade close:

```python
# HydraSizer.py:1963–2014 (simplified)
llm_calls_during_trade = db.query("""
    SELECT * FROM llm_calls
    WHERE trading_pair = :pair
      AND ts BETWEEN :open_ts AND :close_ts
""", pair=pair, open_ts=trade.open_date, close_ts=trade.close_date)

nudge = 0.1 if pnl_pct > 0 else (-0.05 if pnl_pct < 0 else 0.0)

for call in llm_calls_during_trade:
    slot = router.get_slot(call.provider, call.model)
    x_approx = llm_features.rebuild_context(call)
    slot.linucb_update(x_approx, nudge)
```

Flag gating this behavior: `llm_contextual_bandit_retroactive_reward` (default `True`, `llm_router.py:1971`).

**Critical prerequisite**: every `invoke()` call must pass `pair=pair` as a kwarg so `llm_calls.trading_pair` is populated. Otherwise retroactive reward has nothing to match against. Contract test `test_all_rag_invoke_sites_propagate_pair_and_task_context` enforces this across the codebase.

## Gemini Circuit Breaker

Gemini has been the most reliable free provider historically — but also the most prone to temporary cascade failures (quota flipped, key revoked, region outage). The circuit breaker (`llm_router.py:422–459`) prevents retry storms:

```python
# Sliding window — thread-safe deque of timestamps
# 10 failures within 60 s → OPEN
# OPEN state for 30 s → HALF_OPEN
# 3 consecutive successes in HALF_OPEN → CLOSED
# Any failure in HALF_OPEN → back to OPEN
```

When OPEN, the router masks all Gemini slots and falls through to Groq / Cerebras / etc. Telegram alert fires on state transition. Typical recovery time: 30–60 s.

## Cost tracking

`llm_cost_tracker.py` (186 lines) logs every call:

```python
# pseudocode
cost = COSTS_PER_1M[model] × input_tokens / 1e6
     + COSTS_PER_1M[model] × output_tokens / 1e6 × 3  # output roughly 3× pricier
log_call(agent_name, provider, model, input_tok, output_tok, cost, latency_ms, cache_hit, trading_pair)
```

- Free-tier slots return cost = 0.0
- Pay-tier routes track real cost
- `check_budget(daily_limit_usd)` returns True/False — **today it's logging-only**; Phase 30 will make it enforcing

## Persistence

Slot state persists across restarts via `SlotPersistence` (`:463–526`):

- Pickled LinUCB numpy posterior to disk
- Restored on boot (`:742–748`)
- Also mirrored into SQLite `linucb_state` table (Phase 27 EK migration `EK.2.10`)

A cold bot restart never loses bandit progress.

## Semantic cache — step zero

Before any slot selection, `semantic_cache.py` checks if a near-identical query has been answered recently:

- Cache key: hash of `(normalized_query, pair, regime, hour_bucket)`
- TTL: NEUTRAL signals 0.9 h, other signals 6 h
- Returns cached response → 0 latency, 0 cost, 0 bandit update

## Typical call flow

```
invoke(messages, task_context={"task": "tech_analyst", "pair": "BTC/USDT:USDT", ...})
   │
   ▼
Semantic cache check → miss
   │
   ▼
Extract 5-dim feature vector x
   │
   ▼
For each slot in active providers:
   if n_updates < 20: score = Thompson sample
   else:              score = LinUCB UCB(x)
   │
   ▼
Select top-k slots (k=1 default, k>1 for ensemble mode)
   │
   ▼
Call LLM → latency, tokens, content
   │
   ▼
Quality threshold check (parse success, length, structured fields)
   │
   ▼
Update Thompson (alpha or beta) + LinUCB (A, b)
   │
   ▼
Log to llm_calls table with trading_pair
   │
   ▼
Return content
```

On trade close, every logged call during trade window receives retroactive LinUCB update.

## Honest limits

- The bandit is only as good as the reward signal. Trade PnL is noisy — one good trade doesn't prove the slot that fired was right, and vice versa. We use small nudges (±0.05 to ±0.1) rather than large rewards to avoid overcorrection.
- LinUCB assumes linear reward as a function of features — this is wrong for complex tasks, but empirically works as a baseline. Phase 30 research: neural contextual bandits, deep posterior sampling.
- Cold-start Thompson bias — a slot with bad luck on its first 5 calls gets `(2, 7)` posterior and starves. Mitigation: we force a minimum exploration budget per slot.

## See also

- [Architecture](ARCHITECTURE.md)
- [Evidence Engine](EVIDENCE_ENGINE.md) — the fast path that avoids the router entirely
- [PHASE27_ALPHA.md](PHASE27_ALPHA.md) — when LinUCB was added (EK.2.2)
- [PHASE29_ALPHA.md](PHASE29_ALPHA.md) — current sprint on router improvements
