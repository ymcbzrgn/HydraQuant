-- Mega Sprint 2026-04-23 (A.3): Tier 3.5 EvidenceFirst threshold 0.40 → 0.70.
-- Neural Organism's _migrate_new_params only inserts NEW rows and never
-- updates bounds/defaults of existing ones, so the code bump in
-- PARAM_REGISTRY does not reach live state without this manual patch.

UPDATE neuron_state
SET current_val = 0.70,
    default_val = 0.70,
    min_bound   = 0.60,
    max_bound   = 0.85
WHERE param_id = 'rag.evidence_first_threshold';
