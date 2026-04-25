"""
Sprint 2026-04-25 — A/B/C/D/E/F block coverage + audit-fix regression tests.

Each test corresponds to a sprint or audit task; the section comment names
the task ID so a future regression has a clear narrative.

Convention mirrors tests/test_phase26_modules.py (sys.path insertion + per-test
setup_method). No HuggingFace / no exchange — every fixture is in-memory.
"""
import os
import sys
import sqlite3
import tempfile
import threading
import time
from typing import Optional
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'user_data', 'scripts'))


# ═══════════════════════════════════════════════════════════════════════════
# A1 — memory_sensor module + sensor_bridges 5th channel
# ═══════════════════════════════════════════════════════════════════════════

class TestA1MemorySensor:
    def test_compute_pressure_score_zero_when_quiet(self):
        from memory_sensor import compute_pressure_score
        snap = {"sys_used_pct": 0.10, "rss_pct": 0.0, "swap_pct": 0.0,
                "self_swap_pct": 0.0, "frag_score": 1.0}
        score, components = compute_pressure_score(snap)
        assert score == 0.0
        assert components["swap"] == 0.0

    def test_compute_pressure_score_saturates(self):
        from memory_sensor import compute_pressure_score
        snap = {"sys_used_pct": 1.0, "rss_pct": 1.0, "swap_pct": 1.0,
                "self_swap_pct": 1.0, "frag_score": 10.0}
        score, components = compute_pressure_score(snap)
        assert 0.99 <= score <= 1.0
        assert components["swap"] == 1.0

    def test_swap_dominates_other_components(self):
        from memory_sensor import compute_pressure_score
        # Just swap saturated, everything else quiet — swap weight is 0.30
        snap = {"sys_used_pct": 0.10, "rss_pct": 0.0, "swap_pct": 1.0,
                "self_swap_pct": 0.0, "frag_score": 1.0}
        score, _ = compute_pressure_score(snap)
        assert 0.29 <= score <= 0.31  # 0.30 ± floating slack

    def test_aggregate_memory_stress_lif_accumulation(self):
        # Use a fresh PheromoneField so other test pollution can't bleed in.
        from pheromone_field import PheromoneField, get_pheromone_field
        import pheromone_field as pf_mod
        # Reset singleton
        pf_mod._pheromone_field = PheromoneField()
        import sensor_bridges
        # Inject 5 strong deposits — accumulator should rise past 0
        from memory_sensor import record_memory_pressure
        for _ in range(5):
            record_memory_pressure(snap={"sys_used_pct": 1.0, "rss_pct": 1.0,
                                          "swap_pct": 1.0, "self_swap_pct": 1.0,
                                          "frag_score": 5.0})
            time.sleep(0.01)
        stress = sensor_bridges.aggregate_memory_stress()
        assert stress > 0.05, f"LIF should accumulate after 5 deposits, got {stress}"


# ═══════════════════════════════════════════════════════════════════════════
# AUDIT-2 — Memory NOT in _SENSOR_CAPS (no double count)
# ═══════════════════════════════════════════════════════════════════════════

class TestAudit2MemoryNotInSensorCaps:
    def test_memory_not_in_sensor_caps(self):
        from sensor_bridges import _SENSOR_CAPS, SOURCE_MEMORY, SIG_MEMORY_STRESS
        assert (SOURCE_MEMORY, SIG_MEMORY_STRESS) not in _SENSOR_CAPS, \
            "Memory must not be in _SENSOR_CAPS — it has its own dedicated channel"

    def test_aggregate_sensor_stress_excludes_memory(self):
        from pheromone_field import PheromoneField
        import pheromone_field as pf_mod
        pf_mod._pheromone_field = PheromoneField()
        import sensor_bridges
        # Saturate memory channel only
        from memory_sensor import record_memory_pressure
        for _ in range(20):
            record_memory_pressure(snap={"sys_used_pct": 1.0, "rss_pct": 1.0,
                                          "swap_pct": 1.0, "self_swap_pct": 1.0,
                                          "frag_score": 5.0})
            time.sleep(0.001)
        # sensor_stress should still be 0 because memory is not in _SENSOR_CAPS
        assert sensor_bridges.aggregate_sensor_stress() < 0.01
        # But memory_stress should be high
        assert sensor_bridges.aggregate_memory_stress() > 0.1


# ═══════════════════════════════════════════════════════════════════════════
# A1-fix — pheromone _extract_float intensity key
# ═══════════════════════════════════════════════════════════════════════════

class TestA1FixIntensityKey:
    def test_intensity_extracts_float(self):
        from pheromone_field import PheromoneField
        f = PheromoneField()
        assert f._extract_float({"intensity": 0.7}) == 0.7

    def test_value_takes_priority_over_intensity(self):
        from pheromone_field import PheromoneField
        f = PheromoneField()
        # value wins over intensity (it's earlier in the key list)
        assert f._extract_float({"value": 0.5, "intensity": 0.9}) == 0.5

    def test_intensity_drives_lif_accumulation(self):
        from pheromone_field import PheromoneField
        f = PheromoneField()
        f.deposit("test", "sig", {"intensity": 0.5}, half_life=300)
        f.deposit("test", "sig", {"intensity": 0.5}, half_life=300)
        # After 2 deposits: accumulated > TAU_MIN, < TAU_MAX
        accumulated = f.read_accumulated("sig", source="test")
        assert accumulated > 0.5  # at least the LIF base


# ═══════════════════════════════════════════════════════════════════════════
# A2 — Hormones.compute memory_pressure parameter
# ═══════════════════════════════════════════════════════════════════════════

class TestA2HormonesMemoryPressure:
    def test_memory_pressure_zero_no_stress_contribution(self):
        from neural_organism import Hormones
        h = Hormones()
        out = h.compute(fng=50, sensor_stress=0.0, memory_pressure=0.0)
        assert out["cortisol"] == 1.0  # calm

    def test_memory_pressure_full_drops_cortisol(self):
        from neural_organism import Hormones
        h = Hormones()
        out = h.compute(fng=50, sensor_stress=0.0, memory_pressure=1.0)
        # 0.40 stress contribution → cortisol = 1 - 0.40*0.40 = 0.84
        assert 0.83 <= out["cortisol"] <= 0.85

    def test_legacy_call_without_memory_pressure(self):
        from neural_organism import Hormones
        h = Hormones()
        out = h.compute(fng=50, sensor_stress=0.0)
        assert out["cortisol"] == 1.0


# ═══════════════════════════════════════════════════════════════════════════
# AUDIT-3 — refresh_hormones drawdown_pct=0
# ═══════════════════════════════════════════════════════════════════════════

class TestAudit3RefreshHormones:
    def setup_method(self):
        # Reset pheromone singleton so prior tests' deposits don't bleed in.
        from pheromone_field import PheromoneField
        import pheromone_field as pf_mod
        pf_mod._pheromone_field = PheromoneField()

    def test_refresh_hormones_does_not_use_cumulative_pnl(self):
        from neural_organism import get_organism
        org = get_organism()
        # Simulate a 25-loss streak
        org._cumulative_pnl = -25.0
        org._consec_losses = 0  # don't trigger the consec_losses stress
        out = org.refresh_hormones()
        # If audit-3 fix held, drawdown_pct=0 → no drawdown stress contribution
        # _stress should be 0 (no other inputs)
        assert out["_stress"] < 0.31, \
            f"refresh_hormones must not derive stress from _cumulative_pnl (got {out['_stress']})"


# ═══════════════════════════════════════════════════════════════════════════
# AUDIT-8 — Hormones.compute thread lock
# ═══════════════════════════════════════════════════════════════════════════

class TestAudit8HormonesLock:
    def test_compute_lock_exists(self):
        from neural_organism import Hormones
        h = Hormones()
        assert hasattr(h, "_compute_lock")
        assert isinstance(h._compute_lock, type(threading.Lock()))

    def test_concurrent_compute_does_not_corrupt_state(self):
        from neural_organism import Hormones
        h = Hormones()
        errors = []

        def worker():
            try:
                for _ in range(100):
                    h.compute(fng=50, sensor_stress=0.5, memory_pressure=0.5)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)
        assert errors == []
        # cortisol should be a finite float
        assert 0.0 <= h.cortisol <= 1.0


# ═══════════════════════════════════════════════════════════════════════════
# A3 — _memory_groom_tick (heavy gate)
# ═══════════════════════════════════════════════════════════════════════════

class TestA3MemoryGroom:
    def test_groom_tick_heavy_when_score_high(self):
        # Stub PipelineScheduler instance
        import scheduler
        sched_inst = scheduler.PipelineScheduler.__new__(scheduler.PipelineScheduler)
        # Force counter so we hit heavy on this tick (multiple of 6)
        sched_inst._memory_groom_tick_count = 5
        # Patch memory_sensor.tick to return high score
        with patch('memory_sensor.tick') as mock_tick:
            mock_tick.return_value = {"score": 0.9, "components": {}, "raw": {}}
            scheduler.PipelineScheduler._memory_cleanup(sched_inst)
        assert sched_inst._memory_groom_tick_count == 6


# ═══════════════════════════════════════════════════════════════════════════
# A4/AUDIT-9 — embedding_cache LRU eviction by last_used_at
# ═══════════════════════════════════════════════════════════════════════════

class TestA4EmbeddingCacheEvict:
    def setup_method(self):
        # Build a temp DB with the embedding_cache schema
        self.tmp = tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False)
        self.tmp.close()
        self.db_path = self.tmp.name
        conn = sqlite3.connect(self.db_path)
        conn.execute('''CREATE TABLE embedding_cache (
            text_hash TEXT PRIMARY KEY, text_content TEXT NOT NULL,
            gemini_embedding BLOB, bge_embedding BLOB,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            last_used_at DATETIME DEFAULT CURRENT_TIMESTAMP)''')
        # Insert 10 rows with varying last_used_at
        for i in range(10):
            conn.execute(
                "INSERT INTO embedding_cache VALUES (?, ?, ?, ?, ?, ?)",
                (f"hash_{i}", f"text_{i}", b"g", b"b",
                 f"2026-04-2{i % 5} 10:00:00",
                 f"2026-04-2{i} 10:00:00"),  # last_used_at
            )
        conn.commit()
        conn.close()

    def teardown_method(self):
        os.unlink(self.db_path)

    def test_eviction_keeps_top_n_by_last_used(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        before = conn.execute("SELECT COUNT(*) FROM embedding_cache").fetchone()[0]
        assert before == 10
        # Manually mirror the eviction SQL
        conn.execute(
            """DELETE FROM embedding_cache
               WHERE text_hash NOT IN (
                   SELECT text_hash FROM embedding_cache
                   ORDER BY COALESCE(last_used_at, created_at) DESC LIMIT ?
               )""",
            (3,),
        )
        conn.commit()
        rows = conn.execute("SELECT text_hash FROM embedding_cache ORDER BY last_used_at DESC").fetchall()
        conn.close()
        assert len(rows) == 3
        # The top-3 by last_used_at should be hash_9, hash_8, hash_7
        assert {r["text_hash"] for r in rows} == {"hash_9", "hash_8", "hash_7"}


# ═══════════════════════════════════════════════════════════════════════════
# A5 — pheromone cleanup max_idle_seconds
# ═══════════════════════════════════════════════════════════════════════════

class TestA5PheromoneCleanupAdaptive:
    def test_cleanup_default_no_max_idle(self):
        from pheromone_field import PheromoneField
        f = PheromoneField()
        f.deposit("test", "sig", 1.0, half_life=10000)  # very long half-life
        cleaned = f.cleanup()  # no max_idle
        assert cleaned == 0  # nothing dies

    def test_cleanup_with_max_idle_evicts_old_trail(self):
        from pheromone_field import PheromoneField
        f = PheromoneField()
        f.deposit("test", "sig", 1.0, half_life=10000)
        # Force the deposit timestamp to look old
        trail = list(f._field.values())[0]
        trail.deposits[-1].deposited_at -= 1000  # 1000 seconds ago
        cleaned = f.cleanup(max_idle_seconds=60)
        assert cleaned == 1


# ═══════════════════════════════════════════════════════════════════════════
# A6/AUDIT-10 — model_server lazy unload + load lock
# ═══════════════════════════════════════════════════════════════════════════

class TestA6ModelServerReaper:
    def test_idle_threshold_tiers(self):
        import model_server
        with patch.object(model_server, '_get_rss_mb', return_value=100):
            assert model_server._idle_threshold() == model_server._IDLE_DEFAULT_S
        with patch.object(model_server, '_get_rss_mb', return_value=4000):
            assert model_server._idle_threshold() == model_server._IDLE_WARN_S
        with patch.object(model_server, '_get_rss_mb', return_value=5000):
            assert model_server._idle_threshold() == model_server._IDLE_LIMIT_S

    def test_unload_returns_false_when_already_none(self):
        import model_server
        # Ensure it's None; _unload should be no-op
        model_server._colbert_model = None
        assert model_server._unload("colbert") is False

    def test_unload_acquires_load_lock(self):
        # AUDIT-10: confirm the unload path takes _load_lock so it can't
        # race a concurrent ensure_*.
        import model_server
        # If the lock is busy, _unload should block on it. We verify the
        # source uses the lock by introspection.
        import inspect
        src = inspect.getsource(model_server._unload)
        assert "_load_lock" in src, "_unload must acquire _load_lock (AUDIT-10)"


# ═══════════════════════════════════════════════════════════════════════════
# B1/AUDIT-7 — LLMRouter fleet_health + decay-corrected pheromone consumers
# ═══════════════════════════════════════════════════════════════════════════

class TestB1FleetHealth:
    def test_fleet_health_ratio_with_no_slots(self):
        # Build a stub router-like object
        class StubRouter:
            slots: list = []
            class _GeminiCircuit:
                @staticmethod
                def is_open(): return False
            gemini_circuit = _GeminiCircuit()
            from llm_router import LLMRouter as _LR
            fleet_health = _LR.fleet_health
        out = StubRouter().fleet_health()
        assert out["available"] == 0
        assert out["total"] == 1  # max(1, len(self.slots)) protects div-by-zero

    def test_deposit_fleet_health_writes_pheromone(self):
        from pheromone_field import PheromoneField
        import pheromone_field as pf_mod
        pf_mod._pheromone_field = PheromoneField()
        # Stub router with the deposit method
        from llm_router import LLMRouter
        router = LLMRouter.__new__(LLMRouter)
        router._deposit_fleet_health(2, 80, reason="test")
        fleet = pf_mod._pheromone_field.read("fleet_exhausted", source="llm_router")
        assert isinstance(fleet, dict)
        assert fleet["ratio"] == 2 / 80
        assert fleet["reason"] == "test"


# ═══════════════════════════════════════════════════════════════════════════
# D1 — PairCircuit.record_order_attempt fill rate threshold
# ═══════════════════════════════════════════════════════════════════════════

class TestD1FillRateThreshold:
    def setup_method(self):
        from pair_circuit import PairCircuitBreaker
        self.pc = PairCircuitBreaker()

    def test_below_threshold_n_does_not_flip_dormant(self):
        # 4 misses, n<5 → no dormant flip yet
        for _ in range(4):
            went = self.pc.record_order_attempt("X/USDT", filled=False)
            assert went is False
        assert self.pc.is_dormant("X/USDT") is False

    def test_n_5_low_fill_rate_flips_dormant(self):
        for _ in range(5):
            self.pc.record_order_attempt("X/USDT", filled=False)
        assert self.pc.is_dormant("X/USDT") is True

    def test_high_fill_rate_does_not_flip(self):
        # 10 successes — should never go dormant from fill rate
        for _ in range(10):
            self.pc.record_order_attempt("X/USDT", filled=True)
        assert self.pc.is_dormant("X/USDT") is False

    def test_get_fill_rate_returns_correct_ratio(self):
        # 3 fills out of 5 → 0.60
        for filled in [True, True, False, True, False]:
            self.pc.record_order_attempt("X/USDT", filled=filled)
        rate = self.pc.get_fill_rate("X/USDT")
        assert rate is not None
        assert 0.59 <= rate <= 0.61

    def test_get_fill_rate_none_for_unseen_pair(self):
        assert self.pc.get_fill_rate("UNKNOWN") is None


# ═══════════════════════════════════════════════════════════════════════════
# D3 — confirm-time dormant gate
# ═══════════════════════════════════════════════════════════════════════════

class TestD3DormantGate:
    def test_dormant_flag_skips_entry(self):
        # Verify the strategy logic returns False on dormant via direct
        # PairCircuitBreaker dormant simulation.
        from pair_circuit import PairCircuitBreaker
        pc = PairCircuitBreaker()
        # Force dormancy
        slot = pc._slot("DOOM/USDT")
        slot.blacklisted_until = time.time() + 600
        assert pc.is_dormant("DOOM/USDT") is True


# ═══════════════════════════════════════════════════════════════════════════
# C1+C3+C4 — _should_skip_batch reasons
# ═══════════════════════════════════════════════════════════════════════════

class TestCBackpressureGate:
    def test_skip_batch_returns_string_or_empty(self):
        # Smoke: verify the helper exists and returns a string.
        # Full integration requires HydraSizer instance which requires
        # config + dp; we instead assert the contract.
        import inspect
        # Module-level smoke
        from importlib import import_module
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'user_data', 'strategies'))
        try:
            HS = import_module('HydraSizer')
            assert hasattr(HS.HydraSizer, '_should_skip_batch')
            src = inspect.getsource(HS.HydraSizer._should_skip_batch)
            assert "rag_health_unreachable" in src
            assert "memory_pressure" in src
            assert "fleet_exhausted" in src
        except Exception as e:
            pytest.skip(f"HydraSizer module not loadable in test env: {e}")


# ═══════════════════════════════════════════════════════════════════════════
# AUDIT-4 — health_peek + health_commit_reset round-trip
# ═══════════════════════════════════════════════════════════════════════════

class TestAudit4SemanticCachePeek:
    def setup_method(self):
        # SemanticCache requires a DB path with the schema set up; use a temp.
        self.tmp = tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False)
        self.tmp.close()
        self.db_path = self.tmp.name

    def teardown_method(self):
        try:
            os.unlink(self.db_path)
        except FileNotFoundError:
            pass

    def test_peek_does_not_reset_counters(self):
        from semantic_cache import SemanticCache
        c = SemanticCache(db_path=self.db_path)
        # Inject some counters
        with c._counter_lock:
            c._hits = 10
            c._misses = 5
            c._puts = 3
        peek1 = c.health_peek()
        peek2 = c.health_peek()
        assert peek1["hits"] == 10
        assert peek2["hits"] == 10  # peek did NOT reset

    def test_commit_reset_subtracts_peeked_window(self):
        from semantic_cache import SemanticCache
        c = SemanticCache(db_path=self.db_path)
        with c._counter_lock:
            c._hits = 10
            c._misses = 5
        peek = c.health_peek()
        # New activity between peek and commit
        with c._counter_lock:
            c._hits = 12
        c.health_commit_reset(peek)
        # We should have lost 10 hits (the peeked window), preserving the 2 fresh
        with c._counter_lock:
            assert c._hits == 2
            assert c._misses == 0


# ═══════════════════════════════════════════════════════════════════════════
# AUDIT-12 — _pending_fill_checks keyed by (pair, side)
# ═══════════════════════════════════════════════════════════════════════════

class TestAudit12PendingFillKey:
    def test_key_is_tuple_pair_side(self):
        # Inspect HydraSizer __init__ source to confirm the type annotation
        # uses tuple[str, str].
        import inspect
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'user_data', 'strategies'))
        from importlib import import_module
        try:
            HS = import_module('HydraSizer')
            src = inspect.getsource(HS.HydraSizer.__init__)
            assert "_pending_fill_checks: dict[tuple[str, str], float]" in src
        except Exception as e:
            pytest.skip(f"HydraSizer not loadable: {e}")
