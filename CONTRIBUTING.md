# Contributing to HydraQuant

Thank you for your interest in contributing to HydraQuant!

## About

HydraQuant is an AI-augmented quantitative trading engine built on top of [Freqtrade](https://github.com/freqtrade/freqtrade). It extends the base trading bot with 25,000+ lines of AI code including 18 RAG types, 10 autonomous agents, and an Evidence-First signal engine.

## How to Contribute

### Reporting Issues

- Search [existing issues](https://github.com/ymcbzrgn/HydraQuant/issues) first
- Include steps to reproduce, expected vs actual behavior
- For AI-related issues, include relevant log output with `[EvidenceEngine]` or `[Phase20]` tags

### Pull Requests

- Create your PR against the `main` branch
- New features need unit tests (run `pytest tests/test_ai_scripts.py`)
- Follow existing code style and naming conventions
- Keep PRs focused — one feature or fix per PR

### Development Setup

```bash
git clone https://github.com/ymcbzrgn/HydraQuant.git
cd HydraQuant
python -m venv .venv && source .venv/bin/activate
pip install -e . && pip install -r requirements/requirements-ai.txt
pip install -r requirements/requirements-dev.txt

# Run AI tests
PYTHONPATH=user_data/scripts python -m pytest tests/test_ai_scripts.py -v --noconftest -o "addopts="
```

### Project Structure

```
freqtrade/              # Core trading engine (upstream Freqtrade)
user_data/scripts/      # AI modules (64 files, 19,757 lines)
freqtrade-strategies/   # Trading strategies (AIFreqtradeSizer)
tests/                  # Test suite (189 tests)
requirements/           # Dependency files
docker/                 # Docker + service configs
docs/                   # Documentation + brand assets
```

## Attribution

This project is built on [Freqtrade](https://github.com/freqtrade/freqtrade) (GPL v3). All contributions must comply with the GPL v3 license.
