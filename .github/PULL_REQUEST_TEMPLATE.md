## Summary

<!-- One-line description of what this PR does and why. -->

## Changes

<!-- Bullet list of the concrete changes in this PR. -->

-
-

## Related issue

<!-- Closes #XXX / Refs #XXX -->

## Tests

- [ ] Existing HydraQuant tests pass: `PYTHONPATH=user_data/scripts python -m pytest tests/test_ai_scripts.py`
- [ ] New behavior is covered by a unit test
- [ ] Contract tests pass (if adding a new RAG invocation site, scheduler job, or strategy callback)

## Checklist

- [ ] Code follows project conventions (`ruff check`, `ruff format`, `isort`)
- [ ] No hardcoded paths — uses `ai_config` for database paths
- [ ] No stubs, no `TODO` without a tracking issue
- [ ] New dependencies added to the appropriate `requirements/*.txt`
- [ ] Documentation updated (`docs/` or README) if user-facing
