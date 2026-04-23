-- Mega Sprint 2026-04-23 (B.1.5): hard reset `bayesian_kelly_per_pair`
-- after the shadow label bug (B.1.1) corrupted every posterior with BULL/
-- BEAR rows whose shadow PnL never matched a real exchange fill. Resets
-- everything to the Jeffreys-leaning prior (α = β = 2.0, n = 0) so per-
-- pair Kelly re-learns from live trades only.

UPDATE bayesian_kelly_per_pair
SET alpha      = 2.0,
    beta_param = 2.0,
    n_trades   = 0,
    avg_win    = 0.0,
    avg_loss   = 0.0,
    updated_at = strftime('%Y-%m-%dT%H:%M:%SZ', 'now');
