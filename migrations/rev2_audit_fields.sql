-- Revize Tur-2 (C1): granular post-trade court fields + agent vote snapshot.
-- evidence_audit_log gains the raw numeric inputs that the granular blame
-- classifier (volatility_burst, funding_flip, thin_liquidity, stop_hunt)
-- consumes; ai_decisions gains the frozen agent-vote JSON so the
-- opposite-consensus detector sees what the pool actually said at entry
-- time instead of reading stale agent_memory rows at autopsy time.
--
-- SQLite ALTER TABLE ADD COLUMN is idempotent-unsafe: running twice errors
-- "duplicate column name". The deploy wrapper continues on failure so a
-- partial replay is safe.

ALTER TABLE evidence_audit_log ADD COLUMN atr_ratio_at_entry REAL DEFAULT NULL;
ALTER TABLE evidence_audit_log ADD COLUMN volume_z REAL DEFAULT NULL;
ALTER TABLE evidence_audit_log ADD COLUMN funding_rate REAL DEFAULT NULL;

ALTER TABLE ai_decisions ADD COLUMN agent_votes_json TEXT DEFAULT NULL;
