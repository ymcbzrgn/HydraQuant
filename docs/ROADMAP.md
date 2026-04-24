# Roadmap

## Where we are — Phase 29 Alpha (Apr 2026)

Phase 29 is active and being shipped over ~6 weeks. Full blueprint: [PHASE29_ALPHA.md](PHASE29_ALPHA.md).

Three critical tasks:

1. **Task 3A — Backtest revival (lifeline)** — current backtesting is broken because the RAG HTTP service fails per-candle under backtest load, and six `runmode not in ('dry_run', 'live')` guards kick signals into a fallback RSI+MACD path. We must ship a mock-LLM backtest tier that lets every other fix be _measurable_ instead of _hoped_.
2. **Task 1.1 — Funding/OI Divergence** — 7th Evidence Engine sub-score. CCXT adapter for Bybit funding rate and open interest velocity. HIGH confidence in measurable IR lift (literature-based).
3. **Task 3.1 — Dynamic ATR Chandelier** — regime-aware ATR multiplier (HAR-RV + Lee-Mykland) to repair payoff asymmetry (currently 0.66 — trailing stops are hemorrhaging).

## Phase 29 full task list

Priority-ordered (P0 = must-ship):

| Pri | Task | Effort (optimistic / realistic / pessimistic) | Impact confidence |
|---|---|---|---|
| P0 | Task 3A — Backtest Tier 1 (mock LLM) | 2 / 3 / 5 days | HIGH |
| P0 | Task 1.1 — Funding/OI Divergence | 1 / 2 / 3 days | HIGH (lit) |
| P0 | Task 3.1 — Dynamic ATR Chandelier | 1 / 2 / 3 days | HIGH |
| P0 | Task 3.2 — UI/CDaR Circuit Breaker | 2 / 3 / 5 days | HIGH |
| P1 | Task 4.1 — MADAM Sampling + Voting | 4 / 8 / 16 h | MED |
| P1 | Task 3.3 — Speculative Cascade (Groq draft → Gemini verify) | 1 / 2 / 4 h | MED |
| P1 | Task 2.1 — Semantic Cache (GPTCache + Jina) | 2 / 3 / 5 days | MED |
| P2 | Task 5.1 — MoA 2-layer cascade (L1 cheap × L2 premium) | 4 / 8 / 16 h | MED |
| P2 | Task 6.1 — DoWhy-GCM per-trade postmortem | 3 / 5 / 10 days | MED |
| P2 | Task 7.1 — CPCV walk-forward | 3 / 5 / 8 days | HIGH |
| P3 | Task 9.1 — HRP multi-pair allocation | 3 / 5 / 8 days | MED |

Acceptance criteria for "Phase 29 done":

1. `freqtrade backtesting --strategy HydraSizer` runs end-to-end
2. ≥ 3 new alpha sources live (funding/OI + one optional)
3. UI/CDaR circuit breaker triggered in live (≥ 1 event)
4. Dynamic ATR Chandelier shipped 1 month; MaxDD measurably reduced
5. MADAM sampling + voting live; MoA cascade active
6. GCM postmortem produces non-trivial attribution over ≥ 100 trades
7. CPCV walk-forward completed ≥ 1 full run (15 paths)
8. Deploy health dashboard validates Phase 29 features
9. Phase 29 retrospective written

Abort criterion: 6 weeks in with 3A incomplete → Phase 29 retro, move to Phase 30.

## Phase 30 — Tentative

Informed by 9-agent R&D night (Apr 2026). Full research: internal research notes (`arge_20260424_7_ajan.md` in local memory). Themes:

- **Regime switching**: HMM (Student-t emissions) + MSGARCH Kelly multiplier; Phase 30 primary focus
- **Microstructure**: OFI (Order Flow Imbalance, Cont et al. 2014), VPIN; requires Bybit WebSocket L2 depth
- **Transformer time series upgrade**: Chronos-Bolt-Small evaluation (48M params, CPU-ready, 250× faster than Chronos classic)
- **Drift detection**: `river` or `frouros` on feature distribution + label feedback → auto-retrain trigger
- **MARL deferred to 2027** — Hierarchical MAPPO offers +0.1 to +0.3 Sharpe over HRP but broken backtest (Phase 29 Task 3A) invalidates published numbers
- **HRP Phase 29 shipping first** — simpler, published, CPU-friendly; meaningful cross-pair contagion downweight

## Phase 31 — Tentative

- Frontend deep: HydraQuant-branded theme for FreqUI, HydraQuant-exclusive pages beyond the 3 current views
- SDK: Python client library against the 39-endpoint FastAPI
- Two-way Telegram: `/pause`, `/force_exit`, `/status` command handlers
- Public dashboard: Grafana + Prometheus instrumentation
- Cost budget enforcement (today logging-only)
- Phase 30 tech debt: test coverage on user_data/scripts, HypOFE property tests, 12 failing tests → 0

## Long-horizon directions (speculative, not committed)

- **Multi-account orchestration**: same HydraQuant instance driving multiple exchange accounts with per-account autonomy L-levels
- **Cross-market spill-over**: trading crypto + FX + commodities with a single causal graph
- **Alternative-data deep integration**: on-chain netflow from Glassnode-like providers, options-market GEX from Deribit, social sentiment from Farcaster / Nostr
- **Self-play reinforcement**: the Exploiter agent (HQ-15) gets upgraded to propose full alternate strategies, not just adversarial scenarios

## Release cadence

- Alpha → Beta: when Phase 29 acceptance criteria met AND a 30-day dry-run passes PnL + DD gates
- Beta → GA: when autonomy level L3 (0.25 Kelly) holds for 60 days at DD < 15%
- No fixed calendar — quality gates only

## Principles that will stay

- Evidence first. LLMs earn their turn.
- Sizing, not blocking.
- Free tier only, by default.
- Honest status markers — LIVE vs RESEARCH vs STANDARD.
- No stubs, no backward-compat hacks, no "we'll fix it next sprint."

## See also

- [PHASE29_ALPHA.md](PHASE29_ALPHA.md) — current sprint blueprint
- [PHASE29_ARGE.md](PHASE29_ARGE.md) — research notes
- [CHANGELOG.md](CHANGELOG.md) — what's already shipped
