-- EK Sprint 2026-04-23 (EK.2.10): LinUCB contextual-bandit state for the
-- LLM router. Each slot persists its 5×5 covariance A and 5-dim reward
-- vector b as pickled numpy arrays so scheduled restarts do not wipe the
-- learned routing posterior.

CREATE TABLE IF NOT EXISTS linucb_state (
    slot_id TEXT PRIMARY KEY,
    a_blob BLOB,
    b_blob BLOB,
    n_updates INTEGER DEFAULT 0,
    last_updated TEXT
);
