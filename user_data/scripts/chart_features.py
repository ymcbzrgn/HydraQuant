"""
Phase 26: Chart Structure Intelligence — 6-Layer Numerical Chart Analysis

Replaces VLM-based chart reading with mathematical feature extraction.
Captures what a human trader "sees" when looking at a chart:
  Layer 1: Candlestick DNA (mum dizisi silüeti)
  Layer 2: Multi-Timeframe Structure (çok ölçekli çelişki/uyum)
  Layer 3: Volume Profile / VPVR (destek/direnç zonları)
  Layer 4: Smart Money Concepts (BOS, CHoCH, FVG, Order Blocks, Liquidity)
  Layer 5: Fractal Features (Hurst exponent, piyasa pürüzlülüğü)
  Layer 6: Path Signature (geometrik yol özellikleri)

Total: ~130-230 features, <18ms compute, Tier-1 compatible.
All features feed into CatBoost + TTM + RL observation space.

References:
  - Kidger & Lyons "Deep Signature Transforms" NeurIPS 2019
  - Mandelbrot "Fractals and Scaling in Finance" 1997
  - ICT Smart Money Concepts methodology
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Phase 24: Neural Organism adaptive parameters
try:
    from neural_organism import _p
except ImportError:
    def _p(param_id, fallback=0.5, regime="_global"):
        return fallback


# =============================================================================
# LAYER 1: Candlestick DNA — Mum Dizisi Silüeti (~20-40 features)
# =============================================================================

def candle_dna(df: pd.DataFrame, lookback: int = 5) -> Dict[str, float]:
    """Encode last N candles as a DNA sequence.
    Each candle → (body_ratio, upper_wick_ratio, lower_wick_ratio, volume_relative).
    The SEQUENCE matters — doji→hammer→engulfing carries meaning that single patterns miss.
    """
    features = {}
    if len(df) < lookback + 20:
        return {f"cdna_{i}_{k}": 0.0 for i in range(lookback) for k in ["body", "uwk", "lwk", "vol"]}

    vol_avg = df["volume"].iloc[-(lookback + 20):-lookback].mean()
    if vol_avg == 0:
        vol_avg = 1.0

    for i in range(lookback):
        idx = -(lookback - i)
        row = df.iloc[idx]
        hl_range = row["high"] - row["low"]
        if hl_range < 1e-10:
            hl_range = 1e-10

        body = np.clip((row["close"] - row["open"]) / hl_range, -1.0, 1.0)
        upper_wick = max(0.0, (row["high"] - max(row["open"], row["close"])) / hl_range)
        lower_wick = max(0.0, (min(row["open"], row["close"]) - row["low"]) / hl_range)
        vol_rel = row["volume"] / vol_avg if vol_avg > 0 else 1.0

        features[f"cdna_{i}_body"] = round(body, 4)
        features[f"cdna_{i}_uwk"] = round(upper_wick, 4)
        features[f"cdna_{i}_lwk"] = round(lower_wick, 4)
        features[f"cdna_{i}_vol"] = round(min(vol_rel, 10.0), 4)  # cap at 10x

    # Aggregate sequence features
    bodies = [features[f"cdna_{i}_body"] for i in range(lookback)]
    features["cdna_body_trend"] = round(sum(bodies) / lookback, 4)  # avg direction
    features["cdna_body_reversal"] = 1.0 if (bodies[-1] * bodies[0] < 0) else 0.0  # direction flip
    features["cdna_vol_surge"] = round(max(features[f"cdna_{i}_vol"] for i in range(lookback)), 4)

    return features


# =============================================================================
# LAYER 2: Multi-Timeframe Structure (~30-50 features)
# =============================================================================

def multi_timeframe_features(
    df_1h: pd.DataFrame,
    df_4h: Optional[pd.DataFrame] = None,
    df_1d: Optional[pd.DataFrame] = None,
) -> Dict[str, float]:
    """Extract multi-timeframe alignment/conflict signals.
    Same price looks DIFFERENT at different scales — a trader sees all 3 simultaneously.
    """
    features = {}

    # 1h features (primary — already have RSI, ADX, etc. from populate_indicators)
    if len(df_1h) >= 50:
        features["mtf_1h_ema_slope"] = _ema_slope(df_1h, 20)
        features["mtf_1h_rsi"] = _safe_rsi(df_1h)
        features["mtf_1h_bb_width"] = _bb_width(df_1h)
        features["mtf_1h_vol_ratio"] = _volume_ratio(df_1h, 20)
        features["mtf_1h_close_vs_ema50"] = _close_vs_ema(df_1h, 50)

    # 4h features
    if df_4h is not None and len(df_4h) >= 50:
        features["mtf_4h_ema_slope"] = _ema_slope(df_4h, 20)
        features["mtf_4h_rsi"] = _safe_rsi(df_4h)
        features["mtf_4h_bb_width"] = _bb_width(df_4h)
        features["mtf_4h_adx"] = _safe_adx(df_4h)
        features["mtf_4h_vol_ratio"] = _volume_ratio(df_4h, 20)
        features["mtf_4h_close_vs_ema50"] = _close_vs_ema(df_4h, 50)
        features["mtf_4h_trend"] = 1.0 if features.get("mtf_4h_ema_slope", 0) > 0 else -1.0
    else:
        for k in ["ema_slope", "rsi", "bb_width", "adx", "vol_ratio", "close_vs_ema50", "trend"]:
            features[f"mtf_4h_{k}"] = 0.0

    # 1d features
    if df_1d is not None and len(df_1d) >= 50:
        features["mtf_1d_ema_slope"] = _ema_slope(df_1d, 20)
        features["mtf_1d_rsi"] = _safe_rsi(df_1d)
        features["mtf_1d_adx"] = _safe_adx(df_1d)
        features["mtf_1d_trend"] = 1.0 if features.get("mtf_1d_ema_slope", 0) > 0 else -1.0
        # Distance from weekly high/low
        if len(df_1d) >= 7:
            week_high = df_1d["high"].iloc[-7:].max()
            week_low = df_1d["low"].iloc[-7:].min()
            current = df_1d["close"].iloc[-1]
            week_range = week_high - week_low if week_high > week_low else 1e-8
            features["mtf_1d_weekly_position"] = round((current - week_low) / week_range, 4)
    else:
        for k in ["ema_slope", "rsi", "adx", "trend", "weekly_position"]:
            features[f"mtf_1d_{k}"] = 0.0

    # Cross-timeframe conflict/alignment
    slopes = [
        features.get("mtf_1h_ema_slope", 0),
        features.get("mtf_4h_ema_slope", 0),
        features.get("mtf_1d_ema_slope", 0),
    ]
    same_sign = all(s > 0 for s in slopes) or all(s < 0 for s in slopes)
    features["mtf_alignment"] = 1.0 if same_sign else 0.0
    features["mtf_conflict"] = 0.0 if same_sign else 1.0

    # Divergence: 1h bullish but 1d bearish (or vice versa)
    if slopes[0] * slopes[2] < 0:
        features["mtf_1h_1d_divergence"] = 1.0
    else:
        features["mtf_1h_1d_divergence"] = 0.0

    return features


# =============================================================================
# LAYER 3: Volume Profile / VPVR (~8-12 features)
# =============================================================================

def vpvr_features(df: pd.DataFrame, lookback: int = 200, n_bins: int = 50) -> Dict[str, float]:
    """Volume Profile Visible Range — price clustering = support/resistance zones."""
    features = {}
    if len(df) < lookback:
        lookback = len(df)
    if lookback < 20:
        return {k: 0.0 for k in [
            "vpvr_poc_dist", "vpvr_above_vah", "vpvr_below_val",
            "vpvr_va_width", "vpvr_poc_strength", "vpvr_in_va",
            "vpvr_skew", "vpvr_current_vol_zone"
        ]}

    data = df.iloc[-lookback:]
    price_min, price_max = data["low"].min(), data["high"].max()
    if price_max <= price_min:
        price_max = price_min + 1e-8

    bins = np.linspace(price_min, price_max, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    # Distribute volume across price bins
    vol_at_price = np.zeros(n_bins)
    for _, row in data.iterrows():
        candle_low, candle_high = row["low"], row["high"]
        for j in range(n_bins):
            if bins[j + 1] >= candle_low and bins[j] <= candle_high:
                vol_at_price[j] += row["volume"] / max(1, int((candle_high - candle_low) / (bins[1] - bins[0]) + 1))

    total_vol = vol_at_price.sum()
    if total_vol == 0:
        total_vol = 1.0

    # POC (Point of Control) — highest volume price level
    poc_idx = np.argmax(vol_at_price)
    poc = bin_centers[poc_idx]

    # Value Area (70% of volume)
    sorted_indices = np.argsort(vol_at_price)[::-1]
    cumulative = 0.0
    va_indices = []
    for idx in sorted_indices:
        cumulative += vol_at_price[idx]
        va_indices.append(idx)
        if cumulative >= total_vol * 0.70:
            break
    val = bin_centers[min(va_indices)]  # Value Area Low
    vah = bin_centers[max(va_indices)]  # Value Area High

    current = data["close"].iloc[-1]

    features["vpvr_poc_dist"] = round((current - poc) / poc, 6) if poc != 0 else 0.0
    features["vpvr_above_vah"] = 1.0 if current > vah else 0.0
    features["vpvr_below_val"] = 1.0 if current < val else 0.0
    features["vpvr_va_width"] = round((vah - val) / poc, 6) if poc != 0 else 0.0
    features["vpvr_poc_strength"] = round(vol_at_price[poc_idx] / (total_vol / n_bins), 4)
    features["vpvr_in_va"] = 1.0 if val <= current <= vah else 0.0
    features["vpvr_skew"] = round(float(np.mean(vol_at_price[n_bins // 2:]) - np.mean(vol_at_price[:n_bins // 2])), 4)

    # Current price's volume zone strength
    current_bin = min(int((current - price_min) / (price_max - price_min) * n_bins), n_bins - 1)
    current_bin = max(0, current_bin)
    features["vpvr_current_vol_zone"] = round(vol_at_price[current_bin] / (total_vol / n_bins), 4)

    return features


# =============================================================================
# LAYER 4: Smart Money Concepts (~15-20 features)
# =============================================================================

def smc_features(df: pd.DataFrame, swing_lookback: int = 5) -> Dict[str, float]:
    """BOS, CHoCH, Order Blocks, FVG, Liquidity Sweeps.
    Captures institutional order flow footprints numerically."""
    features = {}
    n = len(df)
    if n < swing_lookback * 4:
        return _smc_defaults()

    # Detect swing highs and lows (zigzag)
    swing_highs, swing_lows = _detect_swings(df, swing_lookback)

    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return _smc_defaults()

    current = df["close"].iloc[-1]
    last_sh = swing_highs[-1]
    last_sl = swing_lows[-1]
    prev_sh = swing_highs[-2] if len(swing_highs) >= 2 else last_sh
    prev_sl = swing_lows[-2] if len(swing_lows) >= 2 else last_sl

    # BOS (Break of Structure)
    features["smc_bos_bullish"] = 1.0 if current > last_sh["price"] else 0.0
    features["smc_bos_bearish"] = 1.0 if current < last_sl["price"] else 0.0

    # BOS count in last 24 candles
    bos_bull_count = sum(1 for sh in swing_highs if sh["idx"] > n - 24 and current > sh["price"])
    bos_bear_count = sum(1 for sl in swing_lows if sl["idx"] > n - 24 and current < sl["price"])
    features["smc_bos_count_bull"] = float(bos_bull_count)
    features["smc_bos_count_bear"] = float(bos_bear_count)

    # CHoCH (Change of Character)
    higher_highs = last_sh["price"] > prev_sh["price"]
    higher_lows = last_sl["price"] > prev_sl["price"]
    lower_highs = last_sh["price"] < prev_sh["price"]
    lower_lows = last_sl["price"] < prev_sl["price"]

    was_uptrend = higher_highs and higher_lows
    was_downtrend = lower_highs and lower_lows
    features["smc_choch_bearish"] = 1.0 if was_uptrend and current < last_sl["price"] else 0.0
    features["smc_choch_bullish"] = 1.0 if was_downtrend and current > last_sh["price"] else 0.0

    # Market structure
    if higher_highs and higher_lows:
        features["smc_structure"] = 1.0   # uptrend
    elif lower_highs and lower_lows:
        features["smc_structure"] = -1.0  # downtrend
    else:
        features["smc_structure"] = 0.0   # range/transition

    # Swing distances
    features["smc_sh_dist"] = round((current - last_sh["price"]) / current, 6) if current > 0 else 0.0
    features["smc_sl_dist"] = round((current - last_sl["price"]) / current, 6) if current > 0 else 0.0

    # FVG (Fair Value Gap) — gap between candle wicks
    fvg_count, nearest_fvg_dist = _detect_fvg(df, lookback=50, current_price=current)
    features["smc_fvg_count"] = float(fvg_count)
    features["smc_fvg_nearest_dist"] = round(nearest_fvg_dist, 6)

    # Order Block — last candle before strong impulse
    ob_dist, ob_type = _detect_order_block(df, swing_highs, swing_lows, current)
    features["smc_ob_dist"] = round(ob_dist, 6)
    features["smc_ob_bullish"] = 1.0 if ob_type == "bullish" else 0.0
    features["smc_ob_bearish"] = 1.0 if ob_type == "bearish" else 0.0

    # Liquidity sweep — price pokes above/below swing then reverses
    features["smc_liq_sweep_bull"] = float(_detect_liq_sweep(df, swing_lows, "bullish"))
    features["smc_liq_sweep_bear"] = float(_detect_liq_sweep(df, swing_highs, "bearish"))

    return features


def _smc_defaults() -> Dict[str, float]:
    return {
        "smc_bos_bullish": 0.0, "smc_bos_bearish": 0.0,
        "smc_bos_count_bull": 0.0, "smc_bos_count_bear": 0.0,
        "smc_choch_bearish": 0.0, "smc_choch_bullish": 0.0,
        "smc_structure": 0.0, "smc_sh_dist": 0.0, "smc_sl_dist": 0.0,
        "smc_fvg_count": 0.0, "smc_fvg_nearest_dist": 0.0,
        "smc_ob_dist": 0.0, "smc_ob_bullish": 0.0, "smc_ob_bearish": 0.0,
        "smc_liq_sweep_bull": 0.0, "smc_liq_sweep_bear": 0.0,
    }


# =============================================================================
# LAYER 5: Fractal Features — Hurst Exponent (~3-5 features)
# =============================================================================

def fractal_features(df: pd.DataFrame, windows: List[int] = None) -> Dict[str, float]:
    """Hurst exponent via R/S analysis + fractal dimension.
    H > 0.5 = trending (momentum works)
    H = 0.5 = random walk (nothing works)
    H < 0.5 = mean-reverting (reversion works)
    """
    if windows is None:
        windows = [50, 100, 200]
    features = {}
    prices = df["close"].values

    for w in windows:
        if len(prices) >= w:
            h = _hurst_rs(prices[-w:])
            features[f"hurst_{w}"] = round(h, 4)
        else:
            features[f"hurst_{w}"] = 0.5  # default: random walk assumption

    # Primary Hurst (100-period)
    h100 = features.get("hurst_100", 0.5)
    features["fractal_dim"] = round(2.0 - h100, 4)

    # Regime classification from Hurst
    if h100 > 0.55:
        features["hurst_regime"] = 1.0   # trending
    elif h100 < 0.45:
        features["hurst_regime"] = -1.0  # mean-reverting
    else:
        features["hurst_regime"] = 0.0   # random

    return features


def _hurst_rs(prices: np.ndarray) -> float:
    """Rescaled Range (R/S) analysis for Hurst exponent estimation."""
    n = len(prices)
    if n < 20:
        return 0.5

    returns = np.diff(np.log(prices + 1e-10))
    if len(returns) < 10:
        return 0.5

    # Divide into sub-series of different lengths
    rs_values = []
    lengths = []
    for div in [2, 4, 8, 16]:
        sub_len = len(returns) // div
        if sub_len < 4:
            continue
        rs_list = []
        for i in range(div):
            sub = returns[i * sub_len:(i + 1) * sub_len]
            mean_sub = sub.mean()
            deviate = np.cumsum(sub - mean_sub)
            r = deviate.max() - deviate.min()
            s = sub.std()
            if s > 1e-10:
                rs_list.append(r / s)
        if rs_list:
            rs_values.append(np.mean(rs_list))
            lengths.append(sub_len)

    if len(rs_values) < 2:
        return 0.5

    # Log-log regression: log(R/S) = H * log(n) + c
    log_rs = np.log(rs_values)
    log_n = np.log(lengths)
    try:
        coeffs = np.polyfit(log_n, log_rs, 1)
        h = float(np.clip(coeffs[0], 0.0, 1.0))
    except (np.linalg.LinAlgError, ValueError):
        h = 0.5

    return h


# =============================================================================
# LAYER 6: Path Signature (~50-100 features)
# =============================================================================

def path_signature_features(df: pd.DataFrame, depth: int = 4, window: int = 20) -> Dict[str, float]:
    """Path signature = universal geometric fingerprint of time series.
    Captures direction (depth 1), curvature (depth 2), oscillation (depth 3).
    Uses esig library (CPU, no PyTorch required).
    """
    features = {}
    if len(df) < window:
        # Return zero features with expected dimension
        n_features = _signature_dim(3, depth)  # 3 channels
        return {f"sig_{i}": 0.0 for i in range(n_features)}

    try:
        import esig
        # Build multi-dimensional path: (close_normalized, volume_normalized, rsi_normalized)
        close_vals = df["close"].iloc[-window:].values.astype(float)
        vol_vals = df["volume"].iloc[-window:].values.astype(float)

        # Normalize to [0,1] to avoid numerical issues
        close_norm = _normalize_minmax(close_vals)
        vol_norm = _normalize_minmax(vol_vals)

        # Third channel: RSI if available, else price returns
        if "rsi" in df.columns:
            rsi_vals = df["rsi"].iloc[-window:].fillna(50).values.astype(float) / 100.0
        else:
            rsi_vals = np.diff(close_norm, prepend=close_norm[0])

        path = np.column_stack([close_norm, vol_norm, rsi_vals])

        sig = esig.stream2sig(path, depth)
        for i, v in enumerate(sig):
            features[f"sig_{i}"] = round(float(v), 6)

    except ImportError:
        logger.debug("[ChartFeatures] esig not installed — signature features disabled")
        n_features = _signature_dim(3, depth)
        features = {f"sig_{i}": 0.0 for i in range(n_features)}
    except Exception as e:
        logger.warning(f"[ChartFeatures] Signature computation failed: {e}")
        n_features = _signature_dim(3, depth)
        features = {f"sig_{i}": 0.0 for i in range(n_features)}

    return features


def _signature_dim(channels: int, depth: int) -> int:
    """Calculate signature dimension: sum(channels^k for k=1..depth)."""
    return sum(channels ** k for k in range(1, depth + 1))


# =============================================================================
# MASTER FUNCTION: Compute all 6 layers
# =============================================================================

def compute_chart_features(
    df_1h: pd.DataFrame,
    df_4h: Optional[pd.DataFrame] = None,
    df_1d: Optional[pd.DataFrame] = None,
    include_signature: bool = True,
) -> Dict[str, float]:
    """Compute all 6 layers of chart structure intelligence.
    Returns flat dict of ~130-230 features for CatBoost/TTM/RL input.

    Args:
        df_1h: Primary timeframe OHLCV DataFrame (must have: open, high, low, close, volume)
        df_4h: Optional 4h timeframe DataFrame
        df_1d: Optional daily timeframe DataFrame
        include_signature: Whether to compute path signatures (requires esig)

    Returns:
        Dict[str, float] — all feature names prefixed by layer for clarity
    """
    all_features = {}

    try:
        all_features.update(candle_dna(df_1h))
    except Exception as e:
        logger.warning(f"[ChartFeatures] Layer 1 (Candle DNA) failed: {e}")

    try:
        all_features.update(multi_timeframe_features(df_1h, df_4h, df_1d))
    except Exception as e:
        logger.warning(f"[ChartFeatures] Layer 2 (Multi-TF) failed: {e}")

    try:
        all_features.update(vpvr_features(df_1h))
    except Exception as e:
        logger.warning(f"[ChartFeatures] Layer 3 (VPVR) failed: {e}")

    try:
        all_features.update(smc_features(df_1h))
    except Exception as e:
        logger.warning(f"[ChartFeatures] Layer 4 (SMC) failed: {e}")

    try:
        all_features.update(fractal_features(df_1h))
    except Exception as e:
        logger.warning(f"[ChartFeatures] Layer 5 (Fractal) failed: {e}")

    if include_signature:
        try:
            all_features.update(path_signature_features(df_1h))
        except Exception as e:
            logger.warning(f"[ChartFeatures] Layer 6 (Signature) failed: {e}")

    logger.debug(f"[ChartFeatures] Computed {len(all_features)} features")
    return all_features


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _normalize_minmax(arr: np.ndarray) -> np.ndarray:
    mn, mx = arr.min(), arr.max()
    if mx - mn < 1e-10:
        return np.full_like(arr, 0.5)
    return (arr - mn) / (mx - mn)


def _ema_slope(df: pd.DataFrame, period: int, lookback: int = 5) -> float:
    """EMA slope over last N candles — positive = uptrend."""
    close = df["close"].values
    if len(close) < period + lookback:
        return 0.0
    ema = pd.Series(close).ewm(span=period, adjust=False).mean().values
    slope = (ema[-1] - ema[-lookback]) / (ema[-lookback] if ema[-lookback] != 0 else 1.0)
    return round(slope, 6)


def _safe_rsi(df: pd.DataFrame, period: int = 14) -> float:
    if "rsi" in df.columns:
        val = df["rsi"].iloc[-1]
        return round(float(val) if not pd.isna(val) else 50.0, 2)
    close = df["close"].values
    if len(close) < period + 1:
        return 50.0
    deltas = np.diff(close[-(period + 1):])
    gains = np.where(deltas > 0, deltas, 0).mean()
    losses = np.where(deltas < 0, -deltas, 0).mean()
    if losses == 0:
        return 100.0
    rs = gains / losses
    return round(100.0 - (100.0 / (1 + rs)), 2)


def _safe_adx(df: pd.DataFrame) -> float:
    if "adx" in df.columns:
        val = df["adx"].iloc[-1]
        return round(float(val) if not pd.isna(val) else 20.0, 2)
    return 20.0


def _bb_width(df: pd.DataFrame, period: int = 20) -> float:
    close = df["close"]
    if len(close) < period:
        return 0.0
    sma = close.rolling(period).mean().iloc[-1]
    std = close.rolling(period).std().iloc[-1]
    if pd.isna(sma) or sma == 0:
        return 0.0
    return round((2 * std) / sma, 6)


def _volume_ratio(df: pd.DataFrame, period: int = 20) -> float:
    vol = df["volume"]
    if len(vol) < period + 1:
        return 1.0
    avg = vol.iloc[-(period + 1):-1].mean()
    if avg == 0:
        return 1.0
    return round(min(vol.iloc[-1] / avg, 10.0), 4)


def _close_vs_ema(df: pd.DataFrame, period: int = 50) -> float:
    close = df["close"].values
    if len(close) < period:
        return 0.0
    ema = pd.Series(close).ewm(span=period, adjust=False).mean().values[-1]
    if ema == 0:
        return 0.0
    return round((close[-1] - ema) / ema, 6)


def _detect_swings(df: pd.DataFrame, lookback: int = 5) -> Tuple[list, list]:
    """Detect swing highs and lows using simple lookback comparison."""
    highs = []
    lows = []
    n = len(df)
    for i in range(lookback, n - lookback):
        # Swing high: higher than all neighbors
        if all(df["high"].iloc[i] >= df["high"].iloc[i - j] for j in range(1, lookback + 1)) and \
           all(df["high"].iloc[i] >= df["high"].iloc[i + j] for j in range(1, min(lookback + 1, n - i))):
            highs.append({"idx": i, "price": df["high"].iloc[i]})
        # Swing low
        if all(df["low"].iloc[i] <= df["low"].iloc[i - j] for j in range(1, lookback + 1)) and \
           all(df["low"].iloc[i] <= df["low"].iloc[i + j] for j in range(1, min(lookback + 1, n - i))):
            lows.append({"idx": i, "price": df["low"].iloc[i]})

    return highs, lows


def _detect_fvg(df: pd.DataFrame, lookback: int = 50, current_price: float = 0.0) -> Tuple[int, float]:
    """Detect Fair Value Gaps — price gaps between candle wicks."""
    n = len(df)
    start = max(0, n - lookback)
    fvg_count = 0
    nearest_dist = float("inf")

    for i in range(start + 2, n):
        # Bullish FVG: candle[i-2].high < candle[i].low (gap up)
        if df["high"].iloc[i - 2] < df["low"].iloc[i]:
            fvg_mid = (df["high"].iloc[i - 2] + df["low"].iloc[i]) / 2
            fvg_count += 1
            dist = abs(current_price - fvg_mid) / current_price if current_price > 0 else 0
            nearest_dist = min(nearest_dist, dist)

        # Bearish FVG: candle[i-2].low > candle[i].high (gap down)
        if df["low"].iloc[i - 2] > df["high"].iloc[i]:
            fvg_mid = (df["low"].iloc[i - 2] + df["high"].iloc[i]) / 2
            fvg_count += 1
            dist = abs(current_price - fvg_mid) / current_price if current_price > 0 else 0
            nearest_dist = min(nearest_dist, dist)

    if nearest_dist == float("inf"):
        nearest_dist = 0.0

    return fvg_count, nearest_dist


def _detect_order_block(
    df: pd.DataFrame,
    swing_highs: list,
    swing_lows: list,
    current_price: float
) -> Tuple[float, str]:
    """Order block = last candle before strong impulsive move."""
    best_dist = float("inf")
    best_type = "none"

    # Bullish OB: candle before a strong move up from swing low
    for sl in swing_lows[-3:]:
        idx = sl["idx"]
        if idx > 0 and idx < len(df) - 1:
            ob_price = df["close"].iloc[idx - 1]
            dist = abs(current_price - ob_price) / current_price if current_price > 0 else 0
            if dist < best_dist:
                best_dist = dist
                best_type = "bullish"

    # Bearish OB: candle before a strong move down from swing high
    for sh in swing_highs[-3:]:
        idx = sh["idx"]
        if idx > 0 and idx < len(df) - 1:
            ob_price = df["close"].iloc[idx - 1]
            dist = abs(current_price - ob_price) / current_price if current_price > 0 else 0
            if dist < best_dist:
                best_dist = dist
                best_type = "bearish"

    if best_dist == float("inf"):
        best_dist = 0.0

    return best_dist, best_type


def _detect_liq_sweep(df: pd.DataFrame, swings: list, direction: str) -> bool:
    """Liquidity sweep: price pokes beyond swing point then reverses."""
    if len(swings) < 1 or len(df) < 3:
        return False

    last_swing = swings[-1]
    current = df["close"].iloc[-1]
    prev_close = df["close"].iloc[-2]

    if direction == "bullish":
        # Price went below swing low then closed back above
        return prev_close < last_swing["price"] and current > last_swing["price"]
    else:
        # Price went above swing high then closed back below
        return prev_close > last_swing["price"] and current < last_swing["price"]
