"""
hurst_estimator.py — 3-method Hurst exponent vote.

Sprint 2026-05-02 — adopted from jesse/jesse/indicators/hurst_exponent.py.
Three independent estimators reduce method bias; their median is the
canonical regime indicator.

Methods:
  • R/S (Hurst 1951)        — rescaled range over multiple lags
  • DMA (Alessio 2002)      — detrending moving average  Eur. Phys. J. B 27:197
  • DSOD (Istas/Lang 1994)  — discrete second-order differences

Output:
  H < 0.45 → mean-reverting market structure
  H ≈ 0.50 → random walk (no edge)
  H > 0.55 → persistent / trending market structure

The 3-method vote produces (median, agreement) — high agreement = high
confidence in the regime classification. Median dampens single-method
outliers (e.g., R/S unstable on small samples).

References:
  - Hurst, H. E. (1951). ASCE Transactions, 116(776), 770-808
  - Alessio, E. et al. (2002). Eur. Phys. J. B, 27:197
  - Istas, J., Lang, G. (1994). Ann. Inst. Poincaré, 33:407-436
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)


def _hurst_rs(prices: Sequence[float], min_chunksize: int = 8,
              max_chunksize: int = 200, num_chunksize: int = 5) -> float:
    """Rescaled-Range estimator. Returns NaN-safe Hurst H."""
    try:
        import numpy as np
    except ImportError:
        return 0.5
    arr = np.asarray(prices, dtype=float)
    if arr.size < max(20, min_chunksize * 2):
        return 0.5
    log_returns = np.diff(np.log(arr[arr > 0])) if (arr > 0).all() else np.diff(arr)
    if log_returns.size < min_chunksize:
        return 0.5
    chunk_sizes = np.unique(np.linspace(
        min_chunksize, min(max_chunksize, log_returns.size), num_chunksize
    ).astype(int))
    if chunk_sizes.size < 2:
        return 0.5
    rs_values = []
    valid_sizes = []
    for size in chunk_sizes:
        n_chunks = log_returns.size // size
        if n_chunks < 1:
            continue
        rs_chunk = []
        for i in range(n_chunks):
            chunk = log_returns[i * size:(i + 1) * size]
            mean = chunk.mean()
            cumdev = (chunk - mean).cumsum()
            r = cumdev.max() - cumdev.min()
            s = chunk.std()
            if s > 0:
                rs_chunk.append(r / s)
        if rs_chunk:
            rs_values.append(np.mean(rs_chunk))
            valid_sizes.append(size)
    if len(valid_sizes) < 2:
        return 0.5
    log_sizes = np.log(valid_sizes)
    log_rs = np.log(rs_values)
    slope = np.polyfit(log_sizes, log_rs, 1)[0]
    return float(max(0.0, min(1.0, slope)))


def _hurst_dma(prices: Sequence[float], min_window: int = 8,
               max_window: int = 200, num_windows: int = 5) -> float:
    """Detrending Moving Average estimator (Alessio 2002)."""
    try:
        import numpy as np
    except ImportError:
        return 0.5
    arr = np.asarray(prices, dtype=float)
    if arr.size < max(20, min_window * 2):
        return 0.5
    windows = np.unique(np.linspace(
        min_window, min(max_window, arr.size // 2), num_windows
    ).astype(int))
    if windows.size < 2:
        return 0.5
    sigmas = []
    valid_w = []
    for w in windows:
        if w < 2 or w >= arr.size:
            continue
        # Moving average
        ma = np.convolve(arr, np.ones(w) / w, mode="valid")
        # Aligned price subset (drop first w-1 prices to match ma length)
        price_sub = arr[w - 1:]
        if price_sub.size != ma.size or ma.size < 2:
            continue
        sigma_sq = np.mean((price_sub - ma) ** 2)
        if sigma_sq > 0:
            sigmas.append(np.sqrt(sigma_sq))
            valid_w.append(w)
    if len(valid_w) < 2:
        return 0.5
    log_w = np.log(valid_w)
    log_sigma = np.log(sigmas)
    slope = np.polyfit(log_w, log_sigma, 1)[0]
    return float(max(0.0, min(1.0, slope)))


def _hurst_dsod(prices: Sequence[float]) -> float:
    """Discrete Second-Order Difference (Istas/Lang 1994). Most robust
    to short-sample noise; uses the variance ratio of single vs double
    differencing to estimate H directly without lag fitting."""
    try:
        import numpy as np
    except ImportError:
        return 0.5
    arr = np.asarray(prices, dtype=float)
    if arr.size < 20:
        return 0.5
    # First differences
    d1 = np.diff(arr)
    # Second-order differences (lag 2)
    d2 = arr[2:] - arr[:-2]
    if d1.size < 2 or d2.size < 2:
        return 0.5
    var_d1 = np.var(d1)
    var_d2 = np.var(d2)
    if var_d1 <= 0:
        return 0.5
    ratio = var_d2 / var_d1
    if ratio <= 0:
        return 0.5
    # ratio = 2^(2H) → H = 0.5 * log2(ratio)
    h = 0.5 * np.log2(ratio)
    return float(max(0.0, min(1.0, h)))


def hurst_three_vote(prices: Sequence[float]) -> Dict[str, float]:
    """Run all three estimators and return (median, agreement).

    Returns dict:
      • rs / dma / dsod  — individual estimates
      • median           — robust H (used as the canonical signal)
      • agreement        — 1.0 - max_pairwise_distance, in [0, 1]
                           1.0 = all 3 methods within 0.05 of each other
                           0.0 = methods disagree wildly (low confidence)

    Empty / too-short input returns 0.5 (random walk default).
    """
    rs = _hurst_rs(prices)
    dma = _hurst_dma(prices)
    dsod = _hurst_dsod(prices)
    estimates = sorted([rs, dma, dsod])
    median = estimates[1]
    # Audit Finding #14 fix (2026-05-02): use MAX pairwise distance
    # (not just sorted range) — equivalent here for n=3 but more
    # principled and extensible if a 4th estimator is ever added.
    pairwise = [abs(rs - dma), abs(dma - dsod), abs(rs - dsod)]
    max_pair = max(pairwise) if pairwise else 0.0
    try:
        from neural_organism import _p
        decay = float(_p("envelope.hurst.agreement_decay", 0.5))
    except Exception:
        decay = 0.5
    # Agreement: full when max_pair=0, zero when max_pair=decay
    agreement = max(0.0, min(1.0, 1.0 - max_pair / max(1e-6, decay)))
    return {
        "rs": round(rs, 4),
        "dma": round(dma, 4),
        "dsod": round(dsod, 4),
        "median": round(median, 4),
        "agreement": round(agreement, 4),
    }


def hurst_regime_label(hurst: float) -> str:
    """Map Hurst value to qualitative regime label.

    Audit Finding #6 fix (2026-05-02): thresholds now sourced from
    PARAM_REGISTRY (envelope.hurst.mean_revert_thr / trending_thr) so
    the previously-dead registry entries become live tuning knobs.
    """
    try:
        from neural_organism import _p
        mr_thr = float(_p("envelope.hurst.mean_revert_thr", 0.45))
        tr_thr = float(_p("envelope.hurst.trending_thr", 0.55))
    except Exception:
        mr_thr, tr_thr = 0.45, 0.55
    if hurst < mr_thr:
        return "mean_reverting"
    if hurst > tr_thr:
        return "trending"
    return "random_walk"


def publish_hurst_to_pheromone(prices: Sequence[float],
                               pair: Optional[str] = None) -> Optional[Dict]:
    """Compute 3-vote Hurst and deposit to pheromone field.

    Audit Finding #1 (BLOCKER, 2026-05-02): the prior implementation
    deposited at a SHARED key "hurst_3vote" with no pair suffix. With
    multi-pair populate_indicators, every pair overwrote the previous
    pair's Hurst — BTC's regime ended up applied to ETH/SOL sizing.
    Now deposited at "hurst_3vote::<pair>" so each pair has its own
    isolated regime read.

    A backwards-compatible "hurst_3vote" deposit is also kept for any
    legacy consumer that expects the unsuffixed key — value is the most
    recently published pair (use pair-suffixed key for correctness).
    """
    if prices is None or len(prices) < 20:
        return None
    result = hurst_three_vote(prices)
    result["regime"] = hurst_regime_label(result["median"])
    result["hurst"] = result["median"]
    if pair:
        result["pair"] = pair
    try:
        from pheromone_field import get_pheromone_field
        pf = get_pheromone_field()
        # Per-pair deposit — canonical, isolated.
        if pair:
            pf.deposit(
                "hurst_estimator", f"hurst_3vote::{pair}",
                result, half_life=600.0,
            )
        # Backwards-compat unsuffixed deposit (last-wins semantics —
        # legacy callers should migrate to the pair-suffixed key).
        pf.deposit(
            "hurst_estimator", "hurst_3vote",
            result, half_life=600.0,
        )
    except Exception:
        pass
    return result
