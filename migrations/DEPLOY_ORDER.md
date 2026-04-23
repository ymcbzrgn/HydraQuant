# HydraQuant Migration Deploy Order (STRICT)

Revize Tur-2 (H11): migrations must run in this exact order because a
later migration can reference columns, tables, or rows introduced by an
earlier one.

## Order (dependencies matter)

1. `rev2_model_slot_stats_alter.sql` — ALTER cols (LinUCB latency fields)
2. `rev2_audit_fields.sql` — ALTER cols (evidence_audit_log + ai_decisions)
3. `c3_regime_backfill.sql` — UPDATE rows (FIRST: ai_decisions.regime)
4. `a3_neural_organism.sql` — UPDATE rows (Tier 3.5 threshold 0.70)
5. `ek1_shadow_kelly.sql` — CREATE TABLE (shadow ledger)
6. `ek2_linucb_state.sql` — CREATE TABLE (bandit persist)
7. `b1_kelly_reset.sql` — UPDATE rows (LAST: reset after all schema work)
8. `python user_data/scripts/backfill_argument_quality.py` (AFTER sql)

Rows-first `c3` runs before `a3` only because `a3` can reference the
`regime` column value distribution in logs; functionally they are
independent. `b1` is last because it wipes the posterior — rerunning any
earlier UPDATE after `b1` would undo the prior-reset.

## Deploy command (production)

```bash
cd /root/freqtrade
for f in migrations/rev2_model_slot_stats_alter.sql \
         migrations/rev2_audit_fields.sql \
         migrations/c3_regime_backfill.sql \
         migrations/a3_neural_organism.sql \
         migrations/ek1_shadow_kelly.sql \
         migrations/ek2_linucb_state.sql \
         migrations/b1_kelly_reset.sql; do
    echo "=== $f ==="
    sqlite3 user_data/db/ai_data.sqlite < "$f" \
        || echo "WARN: $f failed (idempotent ALTER or already applied)"
done
.venv/bin/python user_data/scripts/backfill_argument_quality.py

systemctl restart freqtrade freqtrade-scheduler freqtrade-rag \
                  freqtrade-models freqtrade-ai-api
```

The `|| echo` tolerates the SQLite "duplicate column name" error that
ALTER TABLE raises when a migration is re-applied — every column addition
in this sprint is additive and idempotent-safe under that pattern.
