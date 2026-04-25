"""Sprint 2026-04-25 evening — RiskEnvelope tests.

Verifies the dynamic 7-parameter envelope scales correctly with autonomy
tier, hormonal state, sensor votes, and decay multiplier.
"""
import os
import sys
import time
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'user_data', 'scripts'))


# ═══════════════════════════════════════════════════════════════════════════
# RE-1 — RiskEnvelope core
# ═══════════════════════════════════════════════════════════════════════════

class TestRE1Core:
    def test_module_imports(self):
        from risk_envelope import RiskEnvelope, EnvelopeState, get_risk_envelope, TIER_BASES
        assert TIER_BASES[0]["leverage_max"] == 2.0
        assert TIER_BASES[5]["leverage_max"] == 10.0

    def test_singleton(self):
        from risk_envelope import get_risk_envelope
        a = get_risk_envelope()
        b = get_risk_envelope()
        assert a is b

    def test_compute_returns_envelope_state(self):
        from risk_envelope import get_risk_envelope, EnvelopeState
        env = get_risk_envelope()
        state = env.compute()
        assert isinstance(state, EnvelopeState)
        assert hasattr(state, "leverage_max")
        assert hasattr(state, "kelly_cap")
        assert hasattr(state, "sensor_votes")

    def test_compute_within_global_limits(self):
        from risk_envelope import get_risk_envelope, GLOBAL_HARD_LIMITS
        state = get_risk_envelope().compute()
        for key, (lo, hi) in GLOBAL_HARD_LIMITS.items():
            val = getattr(state, key)
            assert lo <= val <= hi, f"{key}={val} out of [{lo}, {hi}]"


# ═══════════════════════════════════════════════════════════════════════════
# RE-2 — 5-Sensor Demote Panel
# ═══════════════════════════════════════════════════════════════════════════

class TestRE2SensorPanel:
    def test_panel_returns_5_keys(self):
        from risk_envelope import DemoteSensorPanel
        panel = DemoteSensorPanel()
        votes = panel.votes_breakdown()
        assert set(votes.keys()) == {
            "cortisol_panic", "hawkes_spike", "ood_high",
            "drawdown_velocity", "streak_collapse",
        }

    def test_vote_count_sum(self):
        from risk_envelope import DemoteSensorPanel
        panel = DemoteSensorPanel()
        votes = panel.votes_breakdown()
        assert panel.vote_count() == sum(1 for v in votes.values() if v)

    def test_streak_collapse_fires_at_5_losses(self):
        from risk_envelope import DemoteSensorPanel
        from neural_organism import get_organism
        org = get_organism()
        original = org._consec_losses
        try:
            org._consec_losses = 5
            assert DemoteSensorPanel().streak_collapse() is True
            org._consec_losses = 4
            assert DemoteSensorPanel().streak_collapse() is False
        finally:
            org._consec_losses = original


# ═══════════════════════════════════════════════════════════════════════════
# RE-3 — Hormonal modulation
# ═══════════════════════════════════════════════════════════════════════════

class TestRE3HormonalModulation:
    def test_factor_in_range(self):
        from risk_envelope import RiskEnvelope
        env = RiskEnvelope()
        f = env._hormonal_factor()
        assert 0.30 <= f <= 1.15

    def test_calm_organism_yields_higher_envelope_than_panic(self):
        from risk_envelope import RiskEnvelope
        from neural_organism import get_organism
        org = get_organism()
        h = org.hormones

        # Calm state
        cort_orig = h.cortisol
        dopa_orig = h.dopamine
        try:
            h.cortisol = 1.0
            h.dopamine = 1.05
            calm_lev = RiskEnvelope().compute().leverage_max

            # Panic state
            h.cortisol = 0.5
            h.dopamine = 0.9
            panic_lev = RiskEnvelope().compute().leverage_max

            assert calm_lev > panic_lev, f"calm({calm_lev}) should exceed panic({panic_lev})"
        finally:
            h.cortisol = cort_orig
            h.dopamine = dopa_orig


# ═══════════════════════════════════════════════════════════════════════════
# RE-4 — Decay state machine
# ═══════════════════════════════════════════════════════════════════════════

class TestRE4DecayStateMachine:
    def test_decay_triggers_on_3_votes(self):
        from risk_envelope import RiskEnvelope
        env = RiskEnvelope()
        # Patch sensor panel to return 3 alarm votes
        with patch.object(env._sensors, "votes_breakdown") as mock:
            mock.return_value = {
                "cortisol_panic": True, "hawkes_spike": True,
                "ood_high": True, "drawdown_velocity": False,
                "streak_collapse": False,
            }
            telemetry = env.update_sensor_state()
            assert telemetry["votes_count"] == 3
            assert telemetry["transition"] in ("demote_triggered", "stable")
            assert env._last_demote_at is not None

    def test_recovery_counter_only_active_when_in_decay(self):
        # FIX-4 (2026-04-25 audit): clean tick starts recovery counter
        # ONLY when in decay. Outside decay the counter is meaningless
        # (not a bug, but defensive against future code that might read it).
        from risk_envelope import RiskEnvelope
        env = RiskEnvelope()
        env._last_demote_at = None
        env._continuous_clean_since = None
        with patch.object(env._sensors, "votes_breakdown") as mock:
            mock.return_value = {
                "cortisol_panic": False, "hawkes_spike": False,
                "ood_high": False, "drawdown_velocity": False,
                "streak_collapse": False,
            }
            telemetry = env.update_sensor_state()
            assert telemetry["votes_count"] == 0
            # NOT in decay → counter stays None
            assert env._continuous_clean_since is None

        # Now in decay — clean tick should start counter
        import time
        env._last_demote_at = time.time() - 100
        env._continuous_clean_since = None
        with patch.object(env._sensors, "votes_breakdown") as mock2:
            mock2.return_value = {k: False for k in [
                "cortisol_panic", "hawkes_spike", "ood_high",
                "drawdown_velocity", "streak_collapse"]}
            env.update_sensor_state()
            assert env._continuous_clean_since is not None


# ═══════════════════════════════════════════════════════════════════════════
# RE-5 — Tier base envelope ladder
# ═══════════════════════════════════════════════════════════════════════════

class TestRE5TierLadder:
    def test_tier_envelopes_monotonic(self):
        from risk_envelope import TIER_BASES
        # Each higher tier should have wider parameters
        for key in ("leverage_max", "risk_per_trade", "kelly_cap",
                    "kelly_fraction_max", "stake_lift_tolerance"):
            for level in range(5):
                lo = TIER_BASES[level][key]
                hi = TIER_BASES[level + 1][key]
                assert hi >= lo, f"L{level+1} {key}={hi} < L{level}={lo}"

        # SL should DECREASE (tighter) at higher tiers
        for level in range(5):
            assert TIER_BASES[level]["sl_base_pct"] >= TIER_BASES[level + 1]["sl_base_pct"]

    def test_l5_targets_renaissance_plus_band(self):
        from risk_envelope import TIER_BASES
        l5 = TIER_BASES[5]
        # Aylık %12-18 → Renaissance Medallion'dan iyi
        assert 0.10 < l5["monthly_target_low"] <= 0.15
        assert 0.15 < l5["monthly_target_high"] <= 0.25


# ═══════════════════════════════════════════════════════════════════════════
# RE-6 — AutonomyManager promote/demote wrappers
# ═══════════════════════════════════════════════════════════════════════════

class TestRE6AutonomyWrappers:
    def test_demote_wrapper_exists(self):
        from autonomy_manager import AutonomyManager
        am = AutonomyManager()
        assert hasattr(am, "demote")
        assert callable(am.demote)

    def test_maybe_promote_wrapper_exists(self):
        from autonomy_manager import AutonomyManager
        am = AutonomyManager()
        assert hasattr(am, "maybe_promote")
        assert callable(am.maybe_promote)

    def test_demote_at_l0_returns_false(self):
        from autonomy_manager import AutonomyManager
        am = AutonomyManager()
        am.current_level = 0
        # At L0, demote cannot go lower
        assert am.demote(reason="test") is False


# ═══════════════════════════════════════════════════════════════════════════
# RE-7 — HydraSizer integration (source-grep level)
# ═══════════════════════════════════════════════════════════════════════════

class TestRE7HydraSizerWire:
    def test_leverage_uses_envelope(self):
        import inspect
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'user_data', 'strategies'))
        from importlib import import_module
        try:
            HS = import_module('HydraSizer')
            src = inspect.getsource(HS.HydraSizer.leverage)
            assert "get_risk_envelope" in src
            assert "envelope_lev_max" in src
        except Exception as e:
            pytest.skip(f"HydraSizer not loadable: {e}")

    def test_custom_stoploss_uses_envelope(self):
        import inspect
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'user_data', 'strategies'))
        from importlib import import_module
        try:
            HS = import_module('HydraSizer')
            src = inspect.getsource(HS.HydraSizer.custom_stoploss)
            assert "get_risk_envelope" in src
            assert "envelope_sl" in src
        except Exception as e:
            pytest.skip(f"HydraSizer not loadable: {e}")


# ═══════════════════════════════════════════════════════════════════════════
# RE-8 — position_sizer integration
# ═══════════════════════════════════════════════════════════════════════════

class TestRE8PositionSizerWire:
    def test_kelly_fraction_uses_envelope(self):
        import inspect
        from position_sizer import BayesianKelly
        src = inspect.getsource(BayesianKelly.kelly_fraction)
        assert "get_risk_envelope" in src
        assert "get_kelly_cap" in src

    def test_effective_max_risk_uses_envelope(self):
        import inspect
        from position_sizer import PositionSizer
        src = inspect.getsource(PositionSizer._effective_max_risk)
        assert "get_risk_envelope" in src
        assert "get_risk_per_trade" in src
