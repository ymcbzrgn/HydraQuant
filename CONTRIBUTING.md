# Contributing to HydraQuant

Thanks for your interest in contributing.

## About

HydraQuant is an AI-native quantitative trading engine for crypto markets. Its cognitive core — 123 Python modules, 51,593 lines, 14 brain subsystems, 25 RAG types, 7-provider LLM router — is HydraQuant's own work. Exchange integration (orders, candles, persistence, CCXT bridging) is delegated to the [freqtrade](https://github.com/freqtrade/freqtrade) execution framework (GPL v3), which we use unmodified.

We ship honest. We mark research-only features as research-only. We do not dress standard techniques in novel language to look smarter.

## How to contribute

### Reporting issues

- Search [existing issues](https://github.com/ymcbzrgn/HydraQuant/issues) first.
- Reproduce on testnet before filing.
- Include log excerpts tagged `[EvidenceEngine]`, `[AgentPool]`, `[NeuralOrganism]`, `[LLMRouter]`, or `[HydraSizer]`.
- For signal-quality issues, attach the `ai_decisions` row: `SELECT * FROM ai_decisions WHERE id=…`.

### Pull requests

- Branch from `main`, PR against `main`.
- One feature or fix per PR — no mixed refactors.
- New functionality requires unit tests in `tests/test_ai_scripts.py` or `tests/test_phase26_modules.py`.
- Claim impact honestly. Prefer "expected +X IR (MEDIUM confidence, literature-based)" over "dramatically improves performance".
- No stubs, no `TODO` PRs. Unfinished work must be gated behind a runtime flag in `config_ai.json`.

### Development setup

```bash
git clone https://github.com/ymcbzrgn/HydraQuant.git hydraquant
cd hydraquant

python -m venv .venv && source .venv/bin/activate
pip install -e . && pip install -r requirements/requirements-phase27.txt

# Run the HydraQuant test suite (241 tests, ~165 s)
PYTHONPATH=user_data/scripts python -m pytest tests/test_ai_scripts.py -v --noconftest -o "addopts="

# Frontend (Vue 3 + PrimeVue + TailwindCSS 4)
cd frequi && pnpm install && pnpm run dev
```

### Project layout

```
user_data/scripts/      HydraQuant AI core — 123 modules, 51,593 lines
user_data/strategies/   HydraSizer — the single bridge to execution
user_data/db/           ai_data.sqlite · lancedb/ · graphdb/hydra.grafeo
tests/                  test_ai_scripts.py (242) · test_phase26_modules.py (43)
docs/                   HYDRAQUANT.md · ARCHITECTURE.md · PHASE*.md · assets/
docker/                 Dockerfile.ai · docker-compose.ai.yml · hydraquant.service*
frequi/                 Vue 3 frontend — 3 HydraQuant views + 9 AI components
freqtrade/              vendored execution framework (do not modify)
```

### Code style

- `ruff check` + `ruff format` (line length 100)
- `mypy user_data/scripts` (partial today; full coverage is a Phase 30 goal)
- `isort` for imports
- Install `pre-commit install` to enforce locally

### Testing philosophy

- Integration tests hit real file-backed SQLite, not `:memory:` — we've been burned by mock-vs-real divergence.
- Property-based testing (`hypothesis`) is encouraged for Kelly, sizing, risk-budget, and any invariant-carrying module.
- Every new module must ship at least one contract test (example: `test_all_rag_invoke_sites_propagate_pair_and_task_context` enforces a pipeline-wide convention).

## License

GPL v3. Contributions must be compatible with the license of the underlying framework.

## Community

Issues and discussions live on GitHub. There is no Discord yet. The Telegram bot `@freqtrade_ai_yamac_bot` broadcasts trade signals and daily P&L summaries only — it is outbound; it does not accept commands.
