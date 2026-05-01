"""
portfolio_optimizer.py — Black-Litterman + Max-Sharpe joint allocation.

Sprint 2026-05-02 — adopted from Lean's BlackLittermanOptimizationPortfolio
ConstructionModel (cookbook formula) + MaximumSharpeRatioPortfolioOptimizer
(SLSQP solver).

Why we need this:
HydraQuant has per-pair Bayesian Kelly that sizes EACH pair independently.
But pairs are correlated — sizing 5 BTC-correlated trades like they're
independent overweights crypto-beta risk. Black-Litterman blends:

  • An equilibrium prior μ̂ (cap-weighted historical returns)
  • Agent VIEWS Q (e.g., MADAM debate: "BTC +0.6%, ETH +0.3%, SOL -0.2%")
  • Confidence Ω in those views
  → Posterior μ_BL, Σ_BL

Then Max-Sharpe SLSQP solves:
  max  (w·μ_BL - rf) / sqrt(w·Σ_BL·w)
  s.t. Σ w = 1, 0 ≤ w_i ≤ max_weight

Result: cross-pair joint weights that respect the agent debate AND the
covariance structure. Unlike per-pair Kelly which ignores correlation.

References:
  - Black, F., Litterman, R. (1990). Asset Allocation Combining Views
  - http://www.blacklitterman.org/cookbook.html
  - https://quant.stackexchange.com/q/18521 — Max-Sharpe under simplex
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)


def _params() -> Tuple[float, float, int]:
    """Pull (tau, delta, lookback_candles) from PARAM_REGISTRY."""
    try:
        from neural_organism import _p
        tau = float(_p("envelope.bl.tau", 0.025))
        delta = float(_p("envelope.bl.delta", 2.5))
        lookback = int(_p("envelope.bl.lookback_candles", 168))
        return tau, delta, lookback
    except Exception:
        return 0.025, 2.5, 168


def black_litterman_posterior(
    returns_matrix,                    # np.ndarray (T, N)
    market_caps_or_prior_weights,      # np.ndarray (N,) — sums to 1
    P,                                 # np.ndarray (K, N) — view-pick matrix
    Q,                                 # np.ndarray (K,) — view returns
    omega=None,                        # np.ndarray (K, K) optional view confidence
):
    """Compute posterior μ and Σ from prior + views.

    Returns (mu_bl, sigma_bl) numpy arrays. None on bad input.
    """
    try:
        import numpy as np
    except ImportError:
        return None, None
    if returns_matrix is None or returns_matrix.size == 0:
        return None, None
    R = np.asarray(returns_matrix, dtype=float)
    if R.ndim != 2 or R.shape[0] < 5 or R.shape[1] < 2:
        return None, None
    N = R.shape[1]
    Sigma = np.cov(R, rowvar=False)
    if Sigma.shape != (N, N):
        return None, None
    w_eq = np.asarray(market_caps_or_prior_weights, dtype=float)
    if w_eq.size != N:
        w_eq = np.full(N, 1.0 / N)
    w_eq = w_eq / w_eq.sum() if w_eq.sum() > 0 else np.full(N, 1.0 / N)

    tau, delta, _ = _params()
    # Implied equilibrium returns: π = δ Σ w_eq
    pi = delta * Sigma.dot(w_eq)
    # τΣ
    ts = tau * Sigma
    P_arr = np.asarray(P, dtype=float)
    Q_arr = np.asarray(Q, dtype=float).reshape(-1)
    K = P_arr.shape[0]
    # Default Ω = diag(P τΣ Pᵀ) — proportional to estimation error
    if omega is None:
        omega = np.diag(np.diag(P_arr @ ts @ P_arr.T))
    # Combined posterior precision
    try:
        ts_inv = np.linalg.pinv(ts)
        omega_inv = np.linalg.pinv(omega)
        post_prec = ts_inv + P_arr.T @ omega_inv @ P_arr
        post_cov = np.linalg.pinv(post_prec)
        mu_bl = post_cov @ (ts_inv @ pi + P_arr.T @ omega_inv @ Q_arr)
        sigma_bl = Sigma + post_cov
    except np.linalg.LinAlgError:
        return None, None
    return np.asarray(mu_bl).flatten(), sigma_bl


def max_sharpe_weights(
    mu, sigma, max_weight: float = 0.30, risk_free: float = 0.0
):
    """SLSQP solver for max-Sharpe weights on the simplex.

    s.t. Σ w_i = 1, 0 ≤ w_i ≤ max_weight.
    Returns weight array (N,) summing to 1, or None on solver failure.
    """
    try:
        import numpy as np
        from scipy.optimize import minimize
    except ImportError:
        return None
    mu = np.asarray(mu, dtype=float)
    N = mu.size
    sigma = np.asarray(sigma, dtype=float)
    if sigma.shape != (N, N) or N < 1:
        return None

    def neg_sharpe(w):
        port_ret = np.dot(w, mu) - risk_free
        port_vol = np.sqrt(max(1e-12, np.dot(w, sigma).dot(w)))
        return -port_ret / port_vol

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(0.0, max_weight) for _ in range(N)]
    x0 = np.full(N, 1.0 / N)
    try:
        res = minimize(
            neg_sharpe, x0, method="SLSQP",
            bounds=bounds, constraints=constraints,
            options={"maxiter": 200, "ftol": 1e-8},
        )
        if not res.success:
            return None
        w = np.clip(res.x, 0.0, max_weight)
        s = w.sum()
        if s <= 0:
            return None
        return w / s
    except Exception as e:
        logger.debug(f"[Portfolio:MaxSharpe] solver failed: {e}")
        return None


def joint_pair_weights(
    pairs: List[str],
    returns_history: Dict[str, Sequence[float]],
    agent_views: Optional[Dict[str, float]] = None,
    max_weight: float = 0.30,
) -> Optional[Dict[str, float]]:
    """End-to-end: returns a dict {pair: weight_fraction}.

    `returns_history`: {pair: [r1, r2, ...]} — last N period returns.
    `agent_views`: optional {pair: expected_return_view} from MADAM debate
                   or any directional model. Pairs not in views are
                   treated as equilibrium-only.

    Returns None on insufficient data.
    """
    try:
        import numpy as np
    except ImportError:
        return None
    if not pairs or len(pairs) < 2:
        return None
    # Build aligned returns matrix
    series = []
    for p in pairs:
        h = returns_history.get(p, [])
        if not h:
            return None
        series.append(list(h))
    min_len = min(len(s) for s in series)
    if min_len < 5:
        return None
    R = np.column_stack([np.asarray(s[-min_len:], dtype=float) for s in series])
    N = R.shape[1]
    # Equal-weight prior (could swap with market-cap if available)
    prior_w = np.full(N, 1.0 / N)
    # Build views: identity-style picks (one view per pair with non-zero view)
    view_pairs = [p for p in pairs if agent_views and p in agent_views and agent_views[p] != 0.0]
    if view_pairs:
        P_rows = []
        Q_vals = []
        for vp in view_pairs:
            row = np.zeros(N)
            row[pairs.index(vp)] = 1.0
            P_rows.append(row)
            Q_vals.append(float(agent_views[vp]))
        P = np.vstack(P_rows)
        Q = np.array(Q_vals)
        mu_bl, sigma_bl = black_litterman_posterior(R, prior_w, P, Q)
    else:
        # Audit Finding #11 fix (2026-05-02): when no views, blend sample
        # mean toward equilibrium prior μ̂ = δΣw_eq via shrinkage. Pure
        # sample mean has known estimation-error explosion (Jorion 1986;
        # Michaud 1989) — extreme weights driven by 30-day noise. The
        # shrinkage coefficient is `tau` (the BL parameter) so the user
        # gets a single tunable knob. Also log a warning so operators
        # see when MADAM views are absent.
        try:
            from neural_organism import _p
            shrink = float(_p("envelope.bl.tau", 0.025)) * 4.0  # tau×4 ≈ 0.10 default
        except Exception:
            shrink = 0.10
        sample_mean = R.mean(axis=0)
        equilibrium_mu = delta * np.cov(R, rowvar=False).dot(prior_w)
        # Linear blend: small shrink → mostly sample, large shrink → equilibrium
        shrink = max(0.0, min(0.9, shrink))
        mu_bl = (1.0 - shrink) * sample_mean + shrink * equilibrium_mu
        sigma_bl = np.cov(R, rowvar=False)
        logger.warning(
            "[Portfolio:BL] no agent views — shrinking sample mean toward "
            f"equilibrium (tau×4={shrink:.2f}). High estimation-error risk; "
            "wire MADAM debate views into joint_pair_weights(...agent_views=...) "
            "for proper Black-Litterman."
        )
    if mu_bl is None or sigma_bl is None:
        return None
    weights = max_sharpe_weights(mu_bl, sigma_bl, max_weight=max_weight)
    if weights is None:
        return None
    return {pairs[i]: float(weights[i]) for i in range(N)}


def publish_joint_weights(weights_dict: Dict[str, float]) -> None:
    """Deposit joint weights to pheromone field for sizing consumption."""
    try:
        from pheromone_field import get_pheromone_field
        get_pheromone_field().deposit(
            "portfolio_optimizer", "joint_weights",
            {"weights": weights_dict, "n_pairs": len(weights_dict)},
            half_life=1800.0,
        )
    except Exception:
        pass
