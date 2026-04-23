-- Mega Sprint 2026-04-23 (C.3): backfill ai_decisions.regime.
-- 7929 rows (99.8%) stuck on the legacy 'MULTI_AGENT_PHASE_5' constant
-- meant the causal engine only ever saw one bucket, so the per-regime
-- PCMCI+ runs returned zero edges. This patch maps every legacy row to
-- the freshest regime_layers.layer3_adx_regime snapshot at or before
-- the ai_decisions timestamp, falling back to 'transitional' when no
-- snapshot is available yet.

UPDATE ai_decisions
SET regime = COALESCE(
    (SELECT rl.layer3_adx_regime
     FROM regime_layers rl
     WHERE rl.pair = ai_decisions.pair
       AND rl.layer3_adx_regime IS NOT NULL
       AND datetime(rl.timestamp) <= datetime(ai_decisions.timestamp)
     ORDER BY rl.timestamp DESC LIMIT 1),
    'transitional'
)
WHERE regime = 'MULTI_AGENT_PHASE_5';
