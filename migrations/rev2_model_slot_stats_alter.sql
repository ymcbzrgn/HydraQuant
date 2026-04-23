-- Revize Tur-2 (C4): inject the latency columns that Mega Sprint A.1 wrote
-- into the writer without back-filling the existing table definition for
-- live DBs. Run-twice errors out ("duplicate column name") — the deploy
-- wrapper catches and continues, so partial replays stay safe.

ALTER TABLE model_slot_stats ADD COLUMN avg_latency_ms REAL DEFAULT 500.0;
ALTER TABLE model_slot_stats ADD COLUMN p95_latency_ms REAL DEFAULT 5000.0;
