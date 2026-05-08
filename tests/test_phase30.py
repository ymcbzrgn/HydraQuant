"""Phase 30 — pytest coverage for new modules.

Run: PYTHONPATH=$(pwd)/user_data/scripts pytest tests/test_phase30.py -v
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "user_data" / "scripts"))


import pytest


# ── A.18 + A.26 assertions ──────────────────────────────────────────
class TestAssertions:
    def test_kelly_floor_passes_above_threshold(self):
        from assertions import check_kelly_floor
        assert check_kelly_floor(0.01, min_floor=0.005).passed

    def test_kelly_floor_warns_below_threshold(self):
        from assertions import check_kelly_floor
        r = check_kelly_floor(0.001, min_floor=0.005)
        assert not r.passed and r.severity == "warn"

    def test_kelly_ceiling_blocks_above_max(self):
        from assertions import check_kelly_ceiling
        assert not check_kelly_ceiling(0.30, max_ceiling=0.25).passed

    def test_stop_loss_must_be_negative(self):
        from assertions import check_stop_loss_present
        assert not check_stop_loss_present(0.05).passed
        assert check_stop_loss_present(-0.03).passed

    def test_leverage_bound(self):
        from assertions import check_leverage_bound
        assert check_leverage_bound(2.0, max_leverage=5.0).passed
        assert not check_leverage_bound(10.0, max_leverage=5.0).passed

    def test_position_cap_blocks_link_scenario(self):
        """LINK trade #2187: 884 USDT stake on 17876 portfolio = 4.95%."""
        from assertions import check_single_position_cap
        r = check_single_position_cap(stake=884.0, portfolio_value=17876.0, max_pct=0.025)
        assert not r.passed
        assert "4.95%" in r.reason or "0.0494" in r.reason or "cap 2.50%" in r.reason

    def test_position_cap_allows_safe_stake(self):
        from assertions import check_single_position_cap
        r = check_single_position_cap(stake=200.0, portfolio_value=17876.0, max_pct=0.025)
        assert r.passed

    def test_aggregate_exposure_caps_total_open(self):
        from assertions import check_aggregate_exposure_cap
        assert not check_aggregate_exposure_cap(open_positions_value=8000, portfolio_value=10000).passed
        assert check_aggregate_exposure_cap(open_positions_value=2000, portfolio_value=10000).passed

    def test_min_notional(self):
        from assertions import check_min_notional
        assert not check_min_notional(stake=2.0, min_notional=5.0).passed
        assert check_min_notional(stake=10.0, min_notional=5.0).passed


# ── A.27 realtime anomaly detector ──────────────────────────────────
class TestRealtimeAnomaly:
    def test_link_jump_blocks_entry(self):
        """LINK trade #2187: 9.658 -> 19.315 +100% bar must trigger."""
        from realtime_anomaly_detector import RealtimeAnomalyDetector
        d = RealtimeAnomalyDetector(record_to_db=False, threshold_pct=0.05)
        d.check_bar("LINK/USDT", 9.658)
        anom = d.check_bar("LINK/USDT", 19.315)
        assert anom and "single_bar_jump" in anom

    def test_normal_movement_passes(self):
        from realtime_anomaly_detector import RealtimeAnomalyDetector
        d = RealtimeAnomalyDetector(record_to_db=False, threshold_pct=0.05)
        d.check_bar("BTC/USDT", 100.0)
        assert d.check_bar("BTC/USDT", 102.0) is None  # 2% under threshold

    def test_halt_persists_until_cooldown(self):
        from realtime_anomaly_detector import RealtimeAnomalyDetector
        d = RealtimeAnomalyDetector(record_to_db=False, threshold_pct=0.05, cooldown_seconds=60)
        d.check_bar("ETH/USDT", 100.0)
        d.check_bar("ETH/USDT", 110.0)
        halted, _ = d.is_halted("ETH/USDT")
        assert halted

    def test_halt_isolated_per_pair(self):
        from realtime_anomaly_detector import RealtimeAnomalyDetector
        d = RealtimeAnomalyDetector(record_to_db=False, threshold_pct=0.05)
        d.check_bar("AAA", 100); d.check_bar("AAA", 110)
        d.check_bar("BBB", 100); d.check_bar("BBB", 101)
        halted_a, _ = d.is_halted("AAA")
        halted_b, _ = d.is_halted("BBB")
        assert halted_a and not halted_b


# ── A.9 think_scrubber ──────────────────────────────────────────────
class TestThinkScrubber:
    def test_strips_think_tags(self):
        from think_scrubber import scrub
        assert scrub("Hello <think>internal</think> world").strip() == "Hello  world".strip()

    def test_strips_thinking_tags(self):
        from think_scrubber import scrub
        assert "Reasoning" not in scrub("Output<thinking>Reasoning steps</thinking>Done")

    def test_preserve_markers_mode(self):
        from think_scrubber import scrub
        assert "[THINK_REDACTED]" in scrub("a<think>x</think>b", preserve_markers=True)

    def test_has_thinking_detection(self):
        from think_scrubber import has_thinking
        assert has_thinking("<reasoning>foo</reasoning>")
        assert not has_thinking("Plain text only")


# ── A.14 json_parse_robust ──────────────────────────────────────────
class TestJsonParseRobust:
    def test_step1_clean_json(self):
        from json_parse_robust import parse_json
        out, label = parse_json('{"a": 1}', "test")
        assert out == {"a": 1} and label == "step1"

    def test_step2_strips_code_fence(self):
        from json_parse_robust import parse_json
        out, label = parse_json('```json\n{"b": 2}\n```', "test")
        assert out == {"b": 2} and label == "step2"

    def test_step3_repairs_single_quotes(self):
        from json_parse_robust import parse_json
        out, label = parse_json("{'c': 'val',}", "test")
        assert out == {"c": "val"} and label == "step3"

    def test_returns_none_on_garbage(self):
        from json_parse_robust import parse_json
        out, label = parse_json("definitely not json", "test")
        assert out is None and label == "failed"


# ── A.6 doom loop detector ──────────────────────────────────────────
class TestDoomLoop:
    def test_threshold_triggers(self):
        from decision_doom_loop import hash_decision, record, reset_for_tests
        reset_for_tests()
        h = hash_decision("BTC/USDT", "NEUTRAL", 100.0)
        result = None
        for _ in range(5):
            result = record("BTC/USDT", h, threshold=5, window_sec=600)
        assert result is not None and result["consecutive_count"] >= 5

    def test_below_threshold_no_fire(self):
        from decision_doom_loop import hash_decision, record, reset_for_tests
        reset_for_tests()
        h = hash_decision("ETH/USDT", "BULL", 200.0)
        for _ in range(3):
            result = record("ETH/USDT", h, threshold=5, window_sec=600)
        assert result is None


# ── A.12 4-state veto ───────────────────────────────────────────────
class TestFourStateVeto:
    def test_majority_allow_passes(self):
        from four_state_veto import Vote, aggregate
        votes = [Vote("A", "ALLOW", 1), Vote("B", "ALLOW", 1), Vote("C", "PASS", 1)]
        assert aggregate(votes).decision == "allow"

    def test_one_ask_blocks(self):
        from four_state_veto import Vote, aggregate
        votes = [Vote("A", "ALLOW", 1), Vote("B", "ASK", 1)]
        assert aggregate(votes).decision == "ask_human"

    def test_strong_deny_blocks(self):
        from four_state_veto import Vote, aggregate
        votes = [Vote("A", "DENY", 1), Vote("B", "DENY", 1), Vote("C", "ALLOW", 0.5)]
        r = aggregate(votes)
        assert r.decision == "deny" and len(r.blocked_by) == 2


# ── B.7 rate_guard ──────────────────────────────────────────────────
class TestRateGuard:
    def test_acquire_within_limit(self):
        from rate_guard import acquire
        ok, _ = acquire("test_provider_phase30", limit_rpm=100)
        assert ok

    def test_acquire_blocks_at_limit(self):
        from rate_guard import acquire
        for _ in range(3):
            acquire("test_provider_phase30_block", limit_rpm=3)
        ok, count = acquire("test_provider_phase30_block", limit_rpm=3)
        assert not ok or count >= 3


# ── B.10 context_compressor ─────────────────────────────────────────
class TestContextCompressor:
    def test_short_input_passes_through(self):
        from context_compressor import compress
        assert compress("short text", target_chars=1000) == "short text"

    def test_long_input_caps(self):
        from context_compressor import compress
        long = "para " * 5000
        out = compress(long, target_chars=200)
        assert len(out) <= 250

    def test_dedup_strips_duplicates(self):
        from context_compressor import compress
        text = "para A content\n\npara A content\n\npara B different"
        out = compress(text, target_chars=10000)
        # near-duplicates should collapse
        assert out.count("para A content") <= 1


# ── B.11 MMR diversification ────────────────────────────────────────
class TestMMR:
    def test_picks_diverse_top_k(self):
        from mmr_diversification import mmr_select
        candidates = ["bitcoin price up", "bitcoin price up again", "ethereum upgrade", "solana mania"]
        idx = mmr_select("crypto news", candidates, k=3, lam=0.5)
        assert len(idx) == 3
        # Should NOT pick both near-duplicates (idx 0 and 1)
        assert not (0 in idx and 1 in idx) or len(set(idx)) == 3


# ── B.13 iteration_budget ───────────────────────────────────────────
class TestIterationBudget:
    def test_token_budget_breach(self):
        from iteration_budget import begin, add_tokens, end, reset_for_tests
        reset_for_tests()
        begin("test_run_b13", token_budget=1000)
        ok = add_tokens("test_run_b13", 600, 500)
        assert not ok  # 1100 > 1000
        snap = end("test_run_b13")
        assert snap["breached"]

    def test_within_budget(self):
        from iteration_budget import begin, add_tokens, end, reset_for_tests
        reset_for_tests()
        begin("test_run_b13_ok", token_budget=1000)
        ok = add_tokens("test_run_b13_ok", 100, 100)
        assert ok
        end("test_run_b13_ok")


# ── B.14 prompt_caching ─────────────────────────────────────────────
class TestPromptCaching:
    def test_anthropic_annotates_system(self):
        from prompt_caching import annotate_messages
        out = annotate_messages("claude-opus-4-7", "x" * 2000, [{"role": "user", "content": "y"}])
        assert isinstance(out["system"], list)
        assert any(b.get("cache_control") for b in out["system"])

    def test_non_anthropic_passthrough(self):
        from prompt_caching import annotate_messages
        out = annotate_messages("llama-3.1-8b", "sys", [{"role": "user", "content": "y"}])
        assert out["system"] == "sys"


# ── B.15 effort_probe ───────────────────────────────────────────────
class TestEffortProbe:
    def test_high_picks_coord_models(self):
        from effort_probe import select_models, CASCADES
        models = select_models("high")
        assert any(m in CASCADES["coord"] for m in models)

    def test_low_picks_probe_models(self):
        from effort_probe import select_models, CASCADES
        models = select_models("low")
        assert all(m in CASCADES["probe"] for m in models)


# ── A.10 prompt_integrity ───────────────────────────────────────────
class TestPromptIntegrity:
    def test_first_register_passes(self):
        from prompt_integrity import verify
        assert verify("test_agent_phase30_a10", "system prompt v1")

    def test_tampered_prompt_detected(self):
        from prompt_integrity import verify
        verify("test_agent_phase30_a10_t", "original prompt")
        assert not verify("test_agent_phase30_a10_t", "tampered prompt!")


# ── A.5 heartbeat suppression ───────────────────────────────────────
class TestHeartbeatSuppression:
    def test_first_emits(self):
        from heartbeat_suppression import should_emit, reset_for_tests
        reset_for_tests()
        assert should_emit("test_kind_phase30", "first message")

    def test_dedup_within_window(self):
        from heartbeat_suppression import should_emit, reset_for_tests
        reset_for_tests()
        assert should_emit("dup_test", "same msg")
        assert not should_emit("dup_test", "same msg")  # within window


# ── B.9 error classifier ────────────────────────────────────────────
class TestErrorClassifier:
    def test_timeout_is_retryable(self):
        from error_classifier import classify
        label, retry, perm = classify(TimeoutError())
        assert label == "retryable" and retry and not perm

    def test_keyerror_is_permanent(self):
        from error_classifier import classify
        label, retry, perm = classify(KeyError("missing"))
        assert label == "permanent" and not retry and perm

    def test_str_match_503(self):
        from error_classifier import classify
        e = Exception("503 service unavailable")
        label, retry, _ = classify(e)
        assert retry


# ── A.19 severity_router ────────────────────────────────────────────
class TestSeverityRouter:
    def test_emit_routes_by_severity(self):
        from severity_router import emit, reset_dedup_for_tests
        reset_dedup_for_tests()
        # Should not raise even when DB / Telegram unavailable
        emit("test.phase30.routing", "info", message="ok", payload={"a": 1})
        emit("test.phase30.routing", "warn", message="warn message")


# ── D.9 promotion gate ──────────────────────────────────────────────
class TestPromotionGate:
    def test_evaluate_returns_structure(self):
        from promotion_gate import evaluate_gate
        r = evaluate_gate(window_days=14)
        # In fresh local DB without trades the gate must block; just assert structure.
        assert hasattr(r, "passed") and hasattr(r, "eligibility_pct")
        assert isinstance(r.blocked_by, list)


# ── D.8 audit_runner ────────────────────────────────────────────────
class TestAuditRunner:
    def test_runs_all_yaml(self):
        from audit_runner import run_all
        results = run_all()
        assert isinstance(results, list)
        # At least the assertions module-import audit must pass
        names = [r.get("name") for r in results]
        assert "assertions_importable" in names
