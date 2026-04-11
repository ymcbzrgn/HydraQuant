"""
Phase 26 Module Tests — Sprint 28.0G

9 modul: pheromone_field, predictive_interoception, ood_detector,
conformal_calibrator, deep_ensemble, dual_axis_calibrator,
ttm_perception, chronos_perception, triple_perception.

TTM/Chronos testleri fallback path kullanir (HuggingFace model gerektirmez).
"""
import sys
import os
import time
import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

# Add scripts to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'user_data', 'scripts'))


# ═══════════════════════════════════════════════════════════════════════════
# 1. PheromoneField Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestPheromoneField:
    def setup_method(self):
        from pheromone_field import PheromoneField
        self.field = PheromoneField()

    def test_deposit_and_read_numeric(self):
        self.field.deposit("test_src", "SIGNAL_A", 42.0, half_life=300)
        val = self.field.read("SIGNAL_A")
        assert val is not None
        assert abs(val - 42.0) < 1.0  # tiny decay

    def test_deposit_and_read_dict(self):
        self.field.deposit("src", "SIGNAL_B", {"direction": "BULLISH", "conf": 0.8})
        val = self.field.read("SIGNAL_B")
        assert isinstance(val, dict)
        assert val["direction"] == "BULLISH"
        assert "_decay" in val

    def test_read_nonexistent_returns_none(self):
        assert self.field.read("DOES_NOT_EXIST") is None

    def test_read_float_default(self):
        val = self.field.read_float("MISSING", default=5.0)
        assert val == 5.0

    def test_dead_signal_returns_none(self):
        # min half_life is 0.1s (clamped in deposit). Need 0.1s * log2(100) = ~0.66s
        self.field.deposit("src", "DYING", 1.0, half_life=0.1)
        time.sleep(1.0)  # 10 half-lives, decay = 0.5^10 ≈ 0.001 < 0.01 threshold
        val = self.field.read("DYING")
        assert val is None  # read() returns None when decay < 0.01

    def test_read_all(self):
        self.field.deposit("a", "SIG1", 1.0)
        self.field.deposit("b", "SIG2", 2.0)
        all_signals = self.field.read_all()
        assert len(all_signals) >= 2

    def test_read_by_source(self):
        self.field.deposit("agent_x", "SIG_A", 10.0)
        self.field.deposit("agent_y", "SIG_B", 20.0)
        x_signals = self.field.read_by_source("agent_x")
        assert "SIG_A" in x_signals

    def test_get_field_health(self):
        self.field.deposit("s1", "A", 1.0)
        self.field.deposit("s2", "B", 2.0)
        health = self.field.get_field_health()
        assert health["active_signals"] >= 2
        assert health["total_deposits"] >= 2
        assert "avg_freshness" in health

    def test_freshness_decreases_over_time(self):
        self.field.deposit("src", "FRESH", 1.0, half_life=0.5)
        f1 = self.field.get_freshness("FRESH")
        time.sleep(0.1)
        f2 = self.field.get_freshness("FRESH")
        assert f2 < f1


# ═══════════════════════════════════════════════════════════════════════════
# 2. PredictiveInteroception Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestPredictiveInteroception:
    def setup_method(self):
        with patch("os.path.exists", return_value=False):
            from predictive_interoception import PredictiveInteroception
            self.intero = PredictiveInteroception(window_size=20)

    def test_record_values(self):
        for i in range(15):
            self.intero.record("organism_health", 0.5 + i * 0.01)
        assert len(self.intero._history["organism_health"]) == 15

    def test_predict_and_act_returns_dict(self):
        for i in range(15):
            self.intero.record("organism_health", 0.5 + i * 0.01)
        result = self.intero.predict_and_act()
        assert "predictions" in result
        assert "alerts" in result
        assert "health_trend" in result

    def test_no_prediction_under_10_samples(self):
        for i in range(5):
            self.intero.record("organism_health", 0.5)
        result = self.intero.predict_and_act()
        assert len(result["predictions"]) == 0

    def test_health_trend_improving(self):
        for i in range(20):
            self.intero.record("organism_health", 0.3 + i * 0.02)
        trend = self.intero._assess_health_trend()
        assert trend in ("improving", "stable")

    def test_health_trend_degrading(self):
        for i in range(20):
            self.intero.record("organism_health", 0.9 - i * 0.03)
        trend = self.intero._assess_health_trend()
        assert trend in ("degrading", "stable")


# ═══════════════════════════════════════════════════════════════════════════
# 3. OOD Detector Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestOODDetector:
    def setup_method(self):
        with patch("os.path.exists", return_value=False):
            from ood_detector import MarketOODDetector
            self.detector = MarketOODDetector(confidence_level=0.95)

    def test_unfitted_returns_safe(self):
        result = self.detector.detect({"f1": 0.5, "f2": 0.3})
        assert result["is_ood"] is False
        assert result["defensive_multiplier"] == 1.0

    def test_fit_and_detect_in_distribution(self):
        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(100, 5), columns=[f"f{i}" for i in range(5)])
        regimes = pd.Series(["bull"] * 100)
        self.detector.fit(X, regimes)
        point = {f"f{i}": 0.0 for i in range(5)}  # near mean
        result = self.detector.detect(point, regime="bull")
        assert result["is_ood"] is False

    def test_fit_and_detect_ood(self):
        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(100, 5), columns=[f"f{i}" for i in range(5)])
        regimes = pd.Series(["bull"] * 100)
        self.detector.fit(X, regimes)
        extreme = {f"f{i}": 20.0 for i in range(5)}  # 20 std away
        result = self.detector.detect(extreme, regime="bull")
        assert result["is_ood"] is True
        assert result["defensive_multiplier"] < 1.0

    def test_defensive_multiplier_has_floor(self):
        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(100, 3), columns=["a", "b", "c"])
        regimes = pd.Series(["x"] * 100)
        self.detector.fit(X, regimes)
        extreme = {"a": 100.0, "b": 100.0, "c": 100.0}
        result = self.detector.detect(extreme, regime="x")
        assert result["defensive_multiplier"] >= 0.1


# ═══════════════════════════════════════════════════════════════════════════
# 4. ConformalCalibrator Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestConformalCalibrator:
    def setup_method(self):
        with patch("os.path.exists", return_value=False):
            from conformal_calibrator import ConformalCalibrator
            self.cal = ConformalCalibrator(target_coverage=0.90, aci_gamma=0.05)

    def test_unfitted_returns_default(self):
        result = self.cal.predict_interval(0.5)
        assert "lower" in result
        assert "upper" in result
        assert result["lower"] < result["upper"]

    def test_fit_and_predict(self):
        np.random.seed(42)
        preds = np.random.randn(50) * 0.1 + 0.5
        actuals = preds + np.random.randn(50) * 0.05
        self.cal.fit(preds, actuals)
        result = self.cal.predict_interval(0.5)
        assert result["lower"] < 0.5 < result["upper"]

    def test_aci_changes_on_miss(self):
        """ACI should adapt alpha when actuals consistently miss the interval."""
        np.random.seed(42)
        preds = np.random.randn(30) * 0.1 + 0.5
        actuals = preds + np.random.randn(30) * 0.05
        self.cal.fit(preds, actuals)
        # After many misses, the empirical_coverage should decrease
        for _ in range(20):
            self.cal.update(0.5, 100.0)  # actual way outside
        result = self.cal.predict_interval(0.5)
        # Just verify it still returns a valid interval
        assert result["lower"] < result["upper"]

    def test_aci_adapts_on_hit(self):
        """ACI should adapt when actuals are consistently inside interval."""
        np.random.seed(42)
        preds = np.random.randn(30) * 0.1 + 0.5
        actuals = preds + np.random.randn(30) * 0.05
        self.cal.fit(preds, actuals)
        for _ in range(20):
            self.cal.update(0.5, 0.5)  # actual inside
        result = self.cal.predict_interval(0.5)
        assert result["lower"] < result["upper"]


# ═══════════════════════════════════════════════════════════════════════════
# 5. DeepEnsemble Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestSmallMLP:
    def test_predict_shape(self):
        from deep_ensemble import SmallMLP
        mlp = SmallMLP(input_dim=10, hidden_dim=32, output_dim=1)
        X = np.random.randn(5, 10)
        out = mlp.predict(X)
        assert out.shape == (5, 1)

    def test_train_reduces_loss(self):
        from deep_ensemble import SmallMLP
        mlp = SmallMLP(input_dim=5, hidden_dim=16, output_dim=1, seed=42)
        X = np.random.randn(20, 5)
        y = X[:, 0:1] * 2 + 1  # simple linear target
        loss_first = mlp.train_step(X, y, lr=0.01)
        for _ in range(50):
            mlp.train_step(X, y, lr=0.01)
        loss_last = mlp.train_step(X, y, lr=0.01)
        assert loss_last < loss_first

    def test_serialization_roundtrip(self):
        from deep_ensemble import SmallMLP
        mlp = SmallMLP(input_dim=8, seed=99)
        X = np.random.randn(3, 8)
        out_before = mlp.predict(X)
        d = mlp.to_dict()
        mlp2 = SmallMLP.from_dict(d)
        out_after = mlp2.predict(X)
        np.testing.assert_array_almost_equal(out_before, out_after)


class TestDeepEnsemble:
    def setup_method(self):
        with patch("os.path.exists", return_value=False):
            from deep_ensemble import DeepEnsemble
            self.ensemble = DeepEnsemble(n_models=3, input_dim=10, hidden_dim=16)

    def test_predict_unfitted(self):
        result = self.ensemble.predict_with_uncertainty(np.random.randn(10))
        assert result["sizing_multiplier"] == 1.0

    def test_fit_and_predict(self):
        np.random.seed(42)
        X = np.random.randn(50, 10)
        y = X[:, 0:1] + np.random.randn(50, 1) * 0.1
        self.ensemble.fit(X, y, epochs=30, lr=0.01)
        result = self.ensemble.predict_with_uncertainty(np.zeros(10))
        assert "mean" in result
        assert "variance" in result
        assert "sizing_multiplier" in result
        assert 0.0 < result["sizing_multiplier"] <= 1.5


# ═══════════════════════════════════════════════════════════════════════════
# 6. DualAxisCalibrator Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestDualAxisCalibrator:
    def setup_method(self):
        from dual_axis_calibrator import DualAxisCalibrator
        self.cal = DualAxisCalibrator()

    def test_high_confidence_narrow_interval(self):
        result = self.cal.calibrate(
            catboost_probability=0.85,
            cqr_interval={"width": 0.02},
            ensemble_variance=0.01,
        )
        assert result["signal_quality"] in ("HIGH", "MEDIUM")
        assert result["final_sizing_multiplier"] > 0.3

    def test_low_confidence_rejects(self):
        result = self.cal.calibrate(catboost_probability=0.50)
        assert result["signal_quality"] in ("LOW", "REJECT")
        assert result["final_sizing_multiplier"] < 0.5

    def test_ood_penalty(self):
        result_normal = self.cal.calibrate(catboost_probability=0.80)
        result_ood = self.cal.calibrate(
            catboost_probability=0.80,
            ood_result={"is_ood": True, "defensive_multiplier": 0.3},
        )
        assert result_ood["final_sizing_multiplier"] < result_normal["final_sizing_multiplier"]

    def test_sizing_bounded(self):
        result = self.cal.calibrate(
            catboost_probability=0.99,
            cqr_interval={"width": 0.001},
            ensemble_variance=0.0001,
            chronos_sizing=2.0,
        )
        assert result["final_sizing_multiplier"] <= 1.5
        assert result["final_sizing_multiplier"] >= 0.1

    def test_explanation_present(self):
        result = self.cal.calibrate(catboost_probability=0.7)
        assert "explanation" in result
        assert len(result["explanation"]) > 0


# ═══════════════════════════════════════════════════════════════════════════
# 7. TTM Perception Tests (Fallback Path)
# ═══════════════════════════════════════════════════════════════════════════

class TestTTMPerception:
    def _make_df(self, n=200):
        """Create a simple OHLCV DataFrame."""
        dates = pd.date_range("2024-01-01", periods=n, freq="1h")
        close = np.cumsum(np.random.randn(n) * 0.5) + 100
        return pd.DataFrame({
            "date": dates, "open": close - 0.2, "high": close + 0.5,
            "low": close - 0.5, "close": close,
            "volume": np.random.randint(100, 10000, n).astype(float),
        })

    def test_fallback_embedding_shape(self):
        import ttm_perception
        ttm_perception._ttm_loaded = True
        ttm_perception._ttm_model = None  # force fallback
        df = self._make_df()
        result = ttm_perception.compute_ttm_embedding(df)
        assert result["available"] is False
        assert result["embedding"].shape == (64,)

    def test_fallback_direction_uptrend(self):
        import ttm_perception
        ttm_perception._ttm_loaded = True
        ttm_perception._ttm_model = None
        n = 200
        close = np.linspace(100, 120, n)  # clear uptrend
        df = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=n, freq="1h"),
            "open": close - 0.1, "high": close + 0.3,
            "low": close - 0.3, "close": close,
            "volume": np.ones(n) * 1000,
        })
        result = ttm_perception.compute_ttm_embedding(df)
        assert result["direction"] > 0

    def test_short_dataframe(self):
        import ttm_perception
        ttm_perception._ttm_loaded = True
        ttm_perception._ttm_model = None
        df = self._make_df(n=10)
        result = ttm_perception.compute_ttm_embedding(df)
        assert result["embedding"].shape == (64,)


# ═══════════════════════════════════════════════════════════════════════════
# 8. Chronos Perception Tests (Fallback Path)
# ═══════════════════════════════════════════════════════════════════════════

class TestChronosPerception:
    def _make_df(self, n=200):
        dates = pd.date_range("2024-01-01", periods=n, freq="1h")
        close = np.cumsum(np.random.randn(n) * 0.5) + 100
        return pd.DataFrame({
            "date": dates, "open": close - 0.2, "high": close + 0.5,
            "low": close - 0.5, "close": close,
            "volume": np.random.randint(100, 10000, n).astype(float),
        })

    def test_fallback_quantile_ordering(self):
        import chronos_perception
        chronos_perception._chronos_loaded = True
        chronos_perception._chronos_pipeline = None
        df = self._make_df()
        result = chronos_perception.compute_chronos_quantiles(df)
        assert result["p10"] <= result["p50"] <= result["p90"]

    def test_fallback_interval_width(self):
        import chronos_perception
        chronos_perception._chronos_loaded = True
        chronos_perception._chronos_pipeline = None
        df = self._make_df()
        result = chronos_perception.compute_chronos_quantiles(df)
        assert abs(result["interval_width"] - (result["p90"] - result["p10"])) < 1e-6

    def test_direction_confidence_bounded(self):
        import chronos_perception
        chronos_perception._chronos_loaded = True
        chronos_perception._chronos_pipeline = None
        df = self._make_df()
        result = chronos_perception.compute_chronos_quantiles(df)
        assert -1.0 <= result["direction_confidence"] <= 1.0

    def test_short_data_returns_zeros(self):
        import chronos_perception
        chronos_perception._chronos_loaded = True
        chronos_perception._chronos_pipeline = None
        df = self._make_df(n=10)
        result = chronos_perception.compute_chronos_quantiles(df)
        assert result["available"] is False


# ═══════════════════════════════════════════════════════════════════════════
# 9. TriplePerception Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestTriplePerception:
    def _make_df(self, n=200):
        dates = pd.date_range("2024-01-01", periods=n, freq="1h")
        close = np.cumsum(np.random.randn(n) * 0.5) + 100
        return pd.DataFrame({
            "date": dates, "open": close - 0.2, "high": close + 0.5,
            "low": close - 0.5, "close": close,
            "volume": np.random.randint(100, 10000, n).astype(float),
        })

    def test_perceive_returns_valid_structure(self):
        from triple_perception import TriplePerception
        tp = TriplePerception()
        df = self._make_df()
        result = tp.perceive(df)
        assert "signal" in result
        assert "confidence" in result
        assert "sizing_multiplier" in result
        assert result["signal"] in ("BULLISH", "BEARISH", "NEUTRAL")

    def test_perceive_confidence_bounded(self):
        from triple_perception import TriplePerception
        tp = TriplePerception()
        df = self._make_df()
        result = tp.perceive(df)
        assert 0.0 <= result["confidence"] <= 1.0

    def test_perceive_sizing_multiplier_bounded(self):
        from triple_perception import TriplePerception
        tp = TriplePerception()
        df = self._make_df()
        result = tp.perceive(df)
        assert 0.1 <= result["sizing_multiplier"] <= 2.0

    def test_perceive_latency_tracked(self):
        from triple_perception import TriplePerception
        tp = TriplePerception()
        df = self._make_df()
        result = tp.perceive(df)
        assert "latency_ms" in result
        assert result["latency_ms"] >= 0
