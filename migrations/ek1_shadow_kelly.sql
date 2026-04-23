-- EK Sprint 2026-04-23 (EK.1): create the parallel shadow-Kelly ledger so
-- forgone-feedback updates no longer contaminate the live sizing posterior.
-- Same schema as bayesian_kelly_per_pair, updated only by the
-- `_forgone_learn_feedback` scheduler job via `get_shadow_kelly()`.

CREATE TABLE IF NOT EXISTS bayesian_kelly_shadow_per_pair (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    pair TEXT NOT NULL,
    regime TEXT NOT NULL DEFAULT '_global',
    alpha REAL DEFAULT 2.0,
    beta_param REAL DEFAULT 2.0,
    avg_win REAL DEFAULT 0.0,
    avg_loss REAL DEFAULT 0.0,
    n_trades INTEGER DEFAULT 0,
    annual_volatility REAL,
    vol_of_vol REAL,
    last_sharpe REAL,
    updated_at TEXT,
    UNIQUE(pair, regime)
);

CREATE INDEX IF NOT EXISTS idx_bk_shadow_per_pair
    ON bayesian_kelly_shadow_per_pair(pair, regime);
