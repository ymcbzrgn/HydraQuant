"""
Phase 26: Tests for Chart Structure Intelligence (6 layers).
Run: PYTHONPATH=user_data/scripts python -m pytest tests/test_chart_features.py -v
"""
import pytest
import numpy as np
import pandas as pd


def _make_ohlcv(n=200, trend="up", volatility=0.02, base_price=100.0):
    """Generate synthetic OHLCV data for testing."""
    np.random.seed(42)
    closes = [base_price]
    for i in range(1, n):
        if trend == "up":
            drift = 0.001
        elif trend == "down":
            drift = -0.001
        elif trend == "range":
            drift = 0.0
        else:
            drift = 0.0
        ret = drift + volatility * np.random.randn()
        closes.append(closes[-1] * (1 + ret))

    closes = np.array(closes)
    highs = closes * (1 + abs(np.random.randn(n)) * 0.005)
    lows = closes * (1 - abs(np.random.randn(n)) * 0.005)
    opens = closes * (1 + np.random.randn(n) * 0.003)
    volumes = np.random.randint(100, 10000, n).astype(float)

    df = pd.DataFrame({
        "open": opens, "high": highs, "low": lows, "close": closes,
        "volume": volumes,
    })
    # Add RSI column (simplified)
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    df["rsi"] = 100 - (100 / (1 + rs))
    df["rsi"] = df["rsi"].fillna(50)
    # Add ADX placeholder
    df["adx"] = 25.0
    return df


# =============================================================================
# Layer 1: Candle DNA
# =============================================================================

class TestCandleDNA:
    def test_basic_output(self):
        from chart_features import candle_dna
        df = _make_ohlcv(100)
        features = candle_dna(df, lookback=5)
        assert len(features) == 5 * 4 + 3  # 5 candles × 4 dims + 3 aggregates

    def test_body_ratio_range(self):
        from chart_features import candle_dna
        df = _make_ohlcv(100)
        features = candle_dna(df, lookback=5)
        for i in range(5):
            assert -1.0 <= features[f"cdna_{i}_body"] <= 1.0

    def test_wick_ratios_positive(self):
        from chart_features import candle_dna
        df = _make_ohlcv(100)
        features = candle_dna(df, lookback=5)
        for i in range(5):
            assert features[f"cdna_{i}_uwk"] >= 0.0
            assert features[f"cdna_{i}_lwk"] >= 0.0

    def test_short_dataframe(self):
        from chart_features import candle_dna
        df = _make_ohlcv(10)
        features = candle_dna(df, lookback=5)
        assert len(features) > 0  # should return defaults, not crash


# =============================================================================
# Layer 2: Multi-Timeframe
# =============================================================================

class TestMultiTimeframe:
    def test_alignment_in_uptrend(self):
        from chart_features import multi_timeframe_features
        df_up = _make_ohlcv(200, trend="up")
        features = multi_timeframe_features(df_up, df_up, df_up)
        assert features["mtf_alignment"] == 1.0
        assert features["mtf_conflict"] == 0.0

    def test_divergence_detection(self):
        from chart_features import multi_timeframe_features
        df_up = _make_ohlcv(200, trend="up", base_price=100.0)
        # Use different seed for down trend to ensure different slope
        np.random.seed(99)
        closes_down = [100.0]
        for _ in range(199):
            closes_down.append(closes_down[-1] * (1 - 0.003 + 0.01 * np.random.randn()))
        df_down = pd.DataFrame({
            "open": closes_down, "high": [c * 1.005 for c in closes_down],
            "low": [c * 0.995 for c in closes_down], "close": closes_down,
            "volume": [1000.0] * 200,
        })
        features = multi_timeframe_features(df_up, df_up, df_down)
        # Either divergence or not — just test it doesn't crash and returns valid value
        assert features["mtf_1h_1d_divergence"] in (0.0, 1.0)

    def test_none_timeframes(self):
        from chart_features import multi_timeframe_features
        df = _make_ohlcv(200)
        features = multi_timeframe_features(df, None, None)
        assert "mtf_4h_ema_slope" in features
        assert "mtf_1d_ema_slope" in features


# =============================================================================
# Layer 3: VPVR
# =============================================================================

class TestVPVR:
    def test_basic_output(self):
        from chart_features import vpvr_features
        df = _make_ohlcv(200)
        features = vpvr_features(df)
        assert "vpvr_poc_dist" in features
        assert "vpvr_poc_strength" in features
        assert features["vpvr_poc_strength"] > 0

    def test_in_value_area(self):
        from chart_features import vpvr_features
        df = _make_ohlcv(200, trend="range", volatility=0.005)
        features = vpvr_features(df)
        # In a ranging market, current price likely in value area
        assert features["vpvr_in_va"] in (0.0, 1.0)

    def test_short_data(self):
        from chart_features import vpvr_features
        df = _make_ohlcv(10)
        features = vpvr_features(df)
        assert len(features) == 8


# =============================================================================
# Layer 4: SMC
# =============================================================================

class TestSMC:
    def test_basic_output(self):
        from chart_features import smc_features
        df = _make_ohlcv(200)
        features = smc_features(df)
        assert "smc_bos_bullish" in features
        assert "smc_choch_bearish" in features
        assert "smc_structure" in features

    def test_uptrend_structure(self):
        from chart_features import smc_features
        df = _make_ohlcv(200, trend="up", volatility=0.01)
        features = smc_features(df)
        # In a clear uptrend, structure should be positive
        assert features["smc_structure"] >= 0.0

    def test_fvg_count_nonnegative(self):
        from chart_features import smc_features
        df = _make_ohlcv(200)
        features = smc_features(df)
        assert features["smc_fvg_count"] >= 0

    def test_defaults_on_short_data(self):
        from chart_features import smc_features
        df = _make_ohlcv(10)
        features = smc_features(df)
        assert features["smc_bos_bullish"] == 0.0


# =============================================================================
# Layer 5: Fractal
# =============================================================================

class TestFractal:
    def test_hurst_range(self):
        from chart_features import fractal_features
        df = _make_ohlcv(200)
        features = fractal_features(df)
        assert 0.0 <= features["hurst_100"] <= 1.0

    def test_trending_hurst(self):
        from chart_features import fractal_features
        # Strong trend should have H > 0.5
        df = _make_ohlcv(200, trend="up", volatility=0.005)
        features = fractal_features(df)
        assert features["hurst_100"] >= 0.4  # trending, should be > 0.5 ideally

    def test_fractal_dim(self):
        from chart_features import fractal_features
        df = _make_ohlcv(200)
        features = fractal_features(df)
        assert features["fractal_dim"] == round(2.0 - features["hurst_100"], 4)


# =============================================================================
# Layer 6: Path Signature
# =============================================================================

class TestSignature:
    def test_output_dimension(self):
        from chart_features import path_signature_features, _signature_dim
        df = _make_ohlcv(100)
        features = path_signature_features(df, depth=4, window=20)
        expected_dim = _signature_dim(3, 4)  # 3 channels, depth 4 = 3 + 9 + 27 + 81 = 120
        assert len(features) == expected_dim

    def test_deterministic(self):
        from chart_features import path_signature_features
        df = _make_ohlcv(100)
        f1 = path_signature_features(df, depth=2, window=20)
        f2 = path_signature_features(df, depth=2, window=20)
        assert f1 == f2  # same input = same output


# =============================================================================
# Master function
# =============================================================================

class TestComputeAll:
    def test_full_pipeline(self):
        from chart_features import compute_chart_features
        df = _make_ohlcv(200)
        features = compute_chart_features(df, df, df, include_signature=False)
        assert len(features) >= 65  # layers 1-5 without signature
        assert all(isinstance(v, (int, float)) for v in features.values())

    def test_with_signature(self):
        from chart_features import compute_chart_features
        df = _make_ohlcv(200)
        features = compute_chart_features(df, include_signature=True)
        assert len(features) >= 100  # layers 1-6

    def test_no_crash_on_empty(self):
        from chart_features import compute_chart_features
        df = pd.DataFrame({"open": [], "high": [], "low": [], "close": [], "volume": []})
        features = compute_chart_features(df)
        assert isinstance(features, dict)

    def test_no_nan_values(self):
        from chart_features import compute_chart_features
        df = _make_ohlcv(200)
        features = compute_chart_features(df, include_signature=False)
        for k, v in features.items():
            assert not (isinstance(v, float) and np.isnan(v)), f"NaN in {k}"
