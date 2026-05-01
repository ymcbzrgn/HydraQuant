"""
olmar_optimizer.py — On-Line Moving Average Reversion (Li & Hoi 2012).

Sprint 2026-05-02 — adopted from Lean's MeanReversionPortfolioConstruction
Model. Cross-pair mean-reversion alpha — when a basket of correlated
pairs all dip together, expect them to revert toward the basket median
faster than the median itself moves.

Mechanism (pure math, no fitting):
1. Compute price-relatives: x_i = price_i / SMA_i(window)
2. Mean of relatives: x̄ = mean(x_i)
3. Expected reversion = ε - x̄ where ε is the reversion threshold
4. Passive-Aggressive step: w_new = w_old - τ * (x_i - x̄)
   τ = max(0, (w·x - ε) / |x - x̄|²)
5. Project onto probability simplex (Duchi et al. ICML 2008)

Result: weight vector that overweights pairs trading BELOW their basket
average and underweights pairs trading ABOVE.

References:
  - Li, B., Hoi, S. C. (2012). On-line portfolio selection with moving
    average reversion. arXiv:1206.4626
  - Duchi, J., Shalev-Shwartz, S., Singer, Y., Chandra, T. (2008).
    Efficient Projections onto the L1-Ball. ICML 2008.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)


def _params():
    try:
        from neural_organism import _p
        window = int(_p("envelope.olmar.window", 5))
        eps = float(_p("envelope.olmar.reversion_threshold", 1.0))
        return window, eps
    except Exception:
        return 5, 1.0


def _simplex_projection(v):
    """Project vector v onto the probability simplex (Duchi 2008).
    Returns numpy array summing to 1 with non-negative entries."""
    import numpy as np
    n = len(v)
    if n == 0:
        return v
    u = np.sort(v)[::-1]  # descending
    cssv = np.cumsum(u) - 1.0
    rho_candidates = np.where(u - cssv / np.arange(1, n + 1) > 0)[0]
    if rho_candidates.size == 0:
        return np.full(n, 1.0 / n)
    rho = rho_candidates[-1]
    theta = cssv[rho] / (rho + 1)
    return np.maximum(v - theta, 0.0)


def olmar_weights(
    price_history: Dict[str, Sequence[float]],
    current_weights: Optional[Dict[str, float]] = None,
) -> Optional[Dict[str, float]]:
    """Compute OLMAR weights for the given basket.

    Args:
      price_history: {pair: [last N prices including current]}
      current_weights: starting weights (default: equal across pairs)

    Returns: {pair: weight} summing to 1.0, or None on insufficient data.
    """
    try:
        import numpy as np
    except ImportError:
        return None
    window, eps = _params()
    pairs = list(price_history.keys())
    if len(pairs) < 2:
        return None
    # SMA window must be ≤ len(prices)
    min_len = min(len(price_history[p]) for p in pairs)
    if min_len < window:
        return None

    price_relatives = []
    for p in pairs:
        prices = np.asarray(price_history[p][-window:], dtype=float)
        if prices[-1] <= 0:
            return None
        sma = prices.mean()
        if sma <= 0:
            return None
        # Relative = current / sma; >1 means above average
        price_relatives.append(prices[-1] / sma)
    pr = np.asarray(price_relatives)
    pr_mean = pr.mean()

    # Current weight vector (default equal)
    n = len(pairs)
    if current_weights:
        w = np.array([current_weights.get(p, 1.0 / n) for p in pairs])
        w = w / w.sum() if w.sum() > 0 else np.full(n, 1.0 / n)
    else:
        w = np.full(n, 1.0 / n)

    # PA step toward reversion: shrink overpriced, grow underpriced
    deviation = pr - pr_mean
    second_norm = float(np.dot(deviation, deviation))
    if second_norm <= 1e-12:
        # All pairs at the same relative level → no rebalance signal
        return {pairs[i]: float(w[i]) for i in range(n)}
    expected_value = float(np.dot(w, pr))
    step = max(0.0, (expected_value - eps) / second_norm)
    # Move OPPOSITE the deviation: overpriced (deviation>0) → reduce weight
    w_new = w - step * deviation
    # Project to simplex (non-negative, sum=1)
    w_proj = _simplex_projection(w_new)
    # Final normalization for numerical safety
    s = w_proj.sum()
    if s <= 0:
        return None
    w_final = w_proj / s
    return {pairs[i]: float(w_final[i]) for i in range(n)}


def publish_olmar_to_pheromone(weights: Dict[str, float]) -> None:
    """Deposit OLMAR weights for sizing layer consumption."""
    try:
        from pheromone_field import get_pheromone_field
        get_pheromone_field().deposit(
            "olmar", "mean_revert_weights",
            {"weights": weights, "n_pairs": len(weights)},
            half_life=1800.0,
        )
    except Exception:
        pass
