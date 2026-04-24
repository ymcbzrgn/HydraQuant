# Deployment

Runbook for the 5-service production layout on a single VM.

## Hardware baseline

Verified minimum on the production host:

- 32 GB ECC RAM
- 4-core Intel Xeon Platinum
- 160 GB NVMe SSD
- No GPU (CPU-only inference)
- Ubuntu 22.04 or newer

Memory budget:

| Service | Steady-state | Peak |
|---|---|---|
| `hydraquant` (strategy + freqtrade core) | ~800 MB | 1.5 GB |
| `hydraquant-scheduler` (APScheduler + 20 singletons) | ~1.2 GB | 2.5 GB (during weekly causal discovery) |
| `hydraquant-rag` (LangGraph, MADAM, AgentPool) | ~700 MB | 1.5 GB |
| `hydraquant-models` (BGE + ColBERT + FlashRank) | ~1.8 GB | 2.2 GB |
| `hydraquant-ai-api` (FastAPI read surface) | ~150 MB | 300 MB |
| **Total** | **~4.6 GB** | **~8 GB** |

## Service topology

```
                 ┌────────────────────────────┐
                 │  systemd                   │
                 └──────────┬─────────────────┘
                            │
    ┌───────────────────────┼───────────────────────┐
    │                       │                       │
┌───▼────┐       ┌──────────▼──────────┐       ┌───▼────┐
│:8080   │       │  hydraquant         │       │:8895   │
│freqtrade│◄────►│  (HydraSizer)       │◄─────►│models  │
│REST API │       │  --sd-notify        │       │server  │
└────────┘       │  WatchdogSec=20     │       └────────┘
                 └──────────┬──────────┘
                            │ HTTP :8891
                 ┌──────────▼──────────┐
                 │  hydraquant-rag     │
                 │  (LangGraph + MADAM)│
                 └──────────┬──────────┘
                            │
                 ┌──────────▼──────────┐
                 │  hydraquant-scheduler│
                 │  66 APScheduler jobs│
                 │  20 singletons      │
                 └──────────┬──────────┘
                            │
                 ┌──────────▼──────────┐
                 │  hydraquant-ai-api  │
                 │  :8890, 39 GETs     │
                 └─────────────────────┘
```

## Systemd units

Repo ships two unit files under `docker/`:

- `hydraquant.service` — basic
- `hydraquant.service.watchdog` — with `Type=notify` + `WatchdogSec=20`

Example install (production):

```bash
sudo cp docker/hydraquant.service.watchdog /etc/systemd/system/hydraquant.service
sudo systemctl daemon-reload
sudo systemctl enable hydraquant
sudo systemctl start hydraquant
```

The other 4 units (`hydraquant-scheduler`, `hydraquant-rag`, `hydraquant-models`, `hydraquant-ai-api`) live on the production host but are not version-controlled in the repo today. Phase 30 TODO: bring them into VCS.

## Environment

`.env` at repo root:

```bash
GEMINI_API_KEY_1=...          # multi-key slots: _1 .. _10
GEMINI_API_KEY_2=...
GROQ_API_KEY=...
CEREBRAS_API_KEY=...
DEEPSEEK_API_KEY=...
SAMBANOVA_API_KEY=...
MISTRAL_API_KEY=...
OPENROUTER_API_KEY=...
JINA_API_KEY=...              # comma-separated for rotation

FRED_API_KEY=...

TELEGRAM_BOT_TOKEN=...
TELEGRAM_CHAT_ID=...

BYBIT_API_KEY=...
BYBIT_SECRET=...
# or
BINANCE_API_KEY=...
BINANCE_SECRET=...
```

## Config

- `config_ai.json` — HydraQuant internal flags (10 runtime-togglable)
- `config_bybit_testnet_futures.json` — exchange + pair whitelist + strategy-specific overrides

Runtime flags (hot-reload via mtime-checked locking):

1. `llm_router_latency_weight_enabled`
2. `agentpool_parallel_r1`
3. `agentpool_r2_conditional`
4. `rlaif_hindsight_removed`
5. `min_stake_tolerance_relaxed`
6. `risk_tight_trailing`
7. `dream_daily_subprocess`
8. `shadow_kelly_separate_ledger_enabled`
9. `llm_contextual_bandit_enabled`
10. `llm_contextual_bandit_retroactive_reward`

## Deploy procedure

**CRITICAL RULE**: after every code deploy, **all 5 services must be restarted**. Missing one (especially `hydraquant-rag`) causes stale code to serve HTTP while the strategy emits freshly-generated signals — a silent divergence we have been burned by.

```bash
# 1. Pull code
cd /root/freqtrade && git pull origin main

# 2. Install new deps if requirements changed
.venv/bin/pip install -e . && .venv/bin/pip install -r requirements-phase27.txt

# 3. Run migrations if schema changed
.venv/bin/python user_data/scripts/db.py --migrate

# 4. Restart all 5 services IN ORDER
sudo systemctl restart hydraquant-models
sudo systemctl restart hydraquant-rag
sudo systemctl restart hydraquant-scheduler
sudo systemctl restart hydraquant-ai-api
sudo systemctl restart hydraquant
sleep 5

# 5. Post-deploy smoke test
.venv/bin/python user_data/scripts/deploy_health_check.py
# Exit 0 = healthy; exit 1 = attention; exit 2 = rollback.
```

## Health check

`deploy_health_check.py` (469 lines, 12 sections):

1. **systemd services** — PID + MemoryCurrent; WARN thresholds freqtrade>2 GB, scheduler>2.5 GB
2. **OOM / restart 24h** — `journalctl` scan
3. **Signal source distribution 6h** — ENSEMBLE / COORDINATOR / AGENT_POOL ≥ 25%; EvidenceEngine < 70%
4. **LLM calls `task_name` / `pair` propagation 1h** — ensures agent_name tag propagation
5. **Bayesian Kelly state** — 56 pairs, α/β distribution
6. **LinUCB bandit learning** — > 500 updates = warm
7. **Trade performance since last deploy** — payoff ratio ≥ 1.2 + WR ≥ 50% = PASS
8. **argument_quality learning** — recent rows
9. **Causal + Dream + Hypothesis** — last-run timestamps
10. **Scheduler RSS** — per-job memory watermark
11. **Memory trend** — 24h slope
12. **Tracebacks** — benign-filtered log scan

## Monitoring

- **Telegram** — daily summary at 23:55 UTC, weekly at Sun 23:55 UTC, alerts with 6h cooldown dedup.
- **SQLite metrics** — `system_metrics`, `signal_health`, `interoception_state`, `organism_audit`, `model_slot_stats`.
- **Grafana / Prometheus** — not yet wired (Phase 30 improvement).

## Alerting

Telegram `send_alert(level, message, cooldown_secs)` with 6h default cooldown per message key. Callers:

- `autonomy_manager` — L-level promote / demote
- `scheduler` — job failures + freeze events
- `rag_graph` — RAG critical errors
- `llm_router` — circuit-breaker state changes

## Rollback

- **Code**: `git checkout <previous_sha> && restart all 5`
- **DB schema**: migrations are forward-compatible; rollback of a migration requires manual reverse SQL (tracked in `migrations/` directory)
- **Neural-organism state**: `neuron_state` snapshot is stored in `evolution_population` every 50 trades; `run_tournament()` blends 10% toward the best genome when performance degrades

## Cost model

Steady-state monthly cost: **$0**. All LLM providers run on free tier (Gemini, Groq, Cerebras, DeepSeek, SambaNova, Mistral, OpenRouter). News is RSS + cryptocurrency.cv SSE + FRED — all free. The circuit breakers and retry logic are tuned for free-tier rate limits.

If you want to enable paid upgrades for hot-path quality:

- `GEMINI_API_KEY_PAID` — routes to Gemini Pro pay-tier when budget allows
- `cost_tracker.py` enforces `$5/day` soft cap (logged, not enforced today — Phase 30 will make it enforcing)

## Known deploy gotchas

1. **Grafeo single-writer lock** — only one process may hold the exclusive lock on `hydra.grafeo`. Services must talk to the scheduler's ZMQ broker at `ipc:///tmp/grafeo_hydra.sock`. If scheduler crashes, other services see "cannot acquire grafeo lock".
2. **RAG service first-request cold start** — 10–15s on first `/signal/{pair}` after restart. Strategy's 120s HTTP timeout covers this.
3. **Model server ports** — `:8895` shared by BGE + ColBERT + FlashRank; restart affects all three.
4. **Log rotation** — systemd journal default only. Log directory `user_data/logs/` fills slowly but is not rotated by HydraQuant itself.
5. **WAL mode** — `ai_data.sqlite` runs in WAL mode. Heavy parallel writes can create `-wal` files up to several hundred MB; `VACUUM` is run via cron but not auto.

## See also

- [Architecture](ARCHITECTURE.md) — service interconnection
- [HYDRAQUANT manifesto](HYDRAQUANT.md)
- [PHASE29_ALPHA.md](PHASE29_ALPHA.md) — current sprint that will touch deploy tooling
