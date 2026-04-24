"""
order_flow.py — Phase 26 Sprint 2, Task 11C

Order Flow Intelligence — CVD + Liquidation Radar.

CVD (Cumulative Volume Delta): tracks aggressive buyer vs seller accumulation.
  CVD rising + price flat = hidden accumulation (bullish divergence)
  CVD falling + price flat = hidden distribution (bearish divergence)

VPIN-lite: Volume-synchronized informed flow estimate.
  High VPIN = toxic flow (informed traders), widen spread or reduce size.

Liquidation Radar: estimates squeeze probability from crowding metrics.
  squeeze_prob > 0.6 → clamp sizing
  squeeze_prob > 0.8 + same direction signal → VETO trade

Integration:
  - Reads from: derivatives_data (funding, OI, L/S ratio)
  - Writes to: pheromone_field ("order_flow_state")
  - Consumed by: market_maker_mode (11D), constitution checks, evidence_engine
"""

import os
import sys
import math
import time
import logging
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
logger = logging.getLogger("order_flow")

from ai_config import AI_DB_PATH
from db import get_db_connection, init_db

# Phase 27 Fix 9 (D5 Ajani): optional flowrisk dependency — when installed,
# switches from the Phase 26 Order-Imbalance-Ratio proxy to a real
# volume-bucketed RecursiveVPIN. When absent, we fall back to the legacy path.
try:
    import flowrisk  # type: ignore
    _FLOWRISK_AVAILABLE = True
except Exception:
    flowrisk = None
    _FLOWRISK_AVAILABLE = False

# Phase 27 Task 10 (D1 Ajani): optional `tick` dependency for weekly MLE refit
# of Hawkes (α, β). The O(1) intensity tracker below works without tick.
try:
    import tick  # type: ignore
    _TICK_AVAILABLE = True
except Exception:
    tick = None
    _TICK_AVAILABLE = False

# VPIN thresholds
VPIN_SAFE = 0.3
VPIN_CAUTION = 0.6
VPIN_DANGER = 0.7
VPIN_TOXIC = 0.8

# Squeeze thresholds
SQUEEZE_CLAMP_THRESHOLD = 0.6
SQUEEZE_VETO_THRESHOLD = 0.8

# Phase 27 Task 10 Hawkes thresholds (PHASE27_ALPHA.md §Task 10b)
HAWKES_VETO = 0.95
HAWKES_CLAMP_HIGH = 0.90
HAWKES_CLAMP_MED = 0.80
HAWKES_CLAMP_LOW = 0.70


class HawkesIntensityTracker:
    """Phase 27 Task 10: O(1) recursive Hawkes(α, β) intensity per pair.

    λ(t) = μ + Σᵢ α · exp(−β(t−tᵢ))   (self-exciting)
    Branching ratio n = α/β. n→1 ⇒ self-reinforcing cascade / criticality.

    Starts with a neutral baseline (α=0.8, β=1.0 → n=0.8) that lets the weekly
    MLE refit job (`_hawkes_mle_refit` in scheduler.py) replace these with
    values fit to observed event arrivals when the `tick` library is available.
    """

    DEFAULT_MU = 1.0
    DEFAULT_ALPHA = 0.8
    DEFAULT_BETA = 1.0

    def __init__(self, pair: str):
        self.pair = pair
        self.mu = self.DEFAULT_MU
        self.alpha = self.DEFAULT_ALPHA
        self.beta = self.DEFAULT_BETA
        self._intensity = self.mu
        self._last_t: Optional[float] = None
        self._last_refit: Optional[float] = None
        # Phase 27 Item 7: rolling event timestamps for the custom MLE refit.
        # 1024 events ≈ 17 hours of typical 1/min flow — enough to fit α,β,μ
        # without unbounded memory growth.
        self._event_times: deque = deque(maxlen=1024)

    def register_event(self, ts: Optional[float] = None) -> float:
        """Record an event and return the new intensity λ(t).
        O(1): decay the old intensity, add α, done."""
        now = ts if ts is not None else time.monotonic()
        if self._last_t is None:
            self._intensity = self.mu + self.alpha
        else:
            dt = max(0.0, now - self._last_t)
            decayed = (self._intensity - self.mu) * math.exp(-self.beta * dt)
            self._intensity = self.mu + decayed + self.alpha
        self._last_t = now
        self._event_times.append(now)
        return self._intensity

    def current_intensity(self, ts: Optional[float] = None) -> float:
        """Read-only intensity at `ts` (no event registered)."""
        now = ts if ts is not None else time.monotonic()
        if self._last_t is None:
            return self.mu
        dt = max(0.0, now - self._last_t)
        decayed = (self._intensity - self.mu) * math.exp(-self.beta * dt)
        return self.mu + decayed

    @property
    def branching_ratio(self) -> float:
        """n = α/β ∈ [0, 1)  — >0.8 danger, >0.95 cascade inevitable."""
        if self.beta <= 0:
            return 0.0
        return self.alpha / self.beta

    def set_params(self, alpha: float, beta: float, mu: Optional[float] = None):
        """MLE refit hook — guarded so n stays strictly below 1 (stable process)."""
        self.beta = max(1e-4, float(beta))
        self.alpha = max(0.0, min(float(alpha), 0.99 * self.beta))
        if mu is not None:
            self.mu = max(0.0, float(mu))
        self._last_refit = time.monotonic()

    def sizing_mult(self) -> float:
        """Task 10b intensity-based multiplier: min(1, baseline / current)."""
        current = self.current_intensity()
        if current <= self.mu or current <= 0:
            return 1.0
        return max(0.1, min(1.0, self.mu / current))


def _percentile_rank(value: float, history: List[float]) -> float:
    """Empirical CDF — fraction of history ≤ value. Used for composite toxicity."""
    if not history:
        return 0.5
    arr = np.asarray(history, dtype=np.float64)
    return float(np.mean(arr <= value))


class OrderFlowAnalyzer:
    """CVD + VPIN + Liquidation radar."""

    def __init__(self):
        self._cvd_history: Dict[str, deque] = {}
        self._volume_buckets: Dict[str, deque] = {}
        # Phase 27 Fix 9: VPIN + Kyle + Amihud histories for empirical CDF
        # percentile-ranking (absolute thresholds hide regime-relative state).
        self._vpin_history: Dict[str, deque] = {}
        self._kyle_history: Dict[str, deque] = {}
        self._amihud_history: Dict[str, deque] = {}
        # Phase 27 Fix 9: RecursiveVPIN estimator per pair when flowrisk present.
        self._vpin_estimators: Dict[str, object] = {}
        # Phase 27 Task 10: per-pair Hawkes intensity trackers.
        self._hawkes: Dict[str, HawkesIntensityTracker] = {}
        # Phase 27 Fix 8: orderbook cache. Set via publish_orderbook() once the
        # Bybit depth stream is wired in — LOB encoding is skipped until then.
        self._last_orderbook: Dict[str, Dict] = {}
        init_db()

    def publish_orderbook(self, pair: str, orderbook: Dict) -> None:
        """Caller-side hook: strategy pushes a Level-2 orderbook snapshot here,
        analyze() picks it up on the next invocation. Once the Bybit depth
        stream is wired in (see TODO in analyze), this becomes the LOB entry
        point. Kept as a thin setter so the integration surface is explicit.
        """
        if not isinstance(orderbook, dict):
            return
        if not hasattr(self, "_last_orderbook") or self._last_orderbook is None:
            self._last_orderbook = {}
        self._last_orderbook[pair] = orderbook

    def _get_hawkes(self, pair: str) -> HawkesIntensityTracker:
        tracker = self._hawkes.get(pair)
        if tracker is None:
            tracker = HawkesIntensityTracker(pair)
            self._hawkes[pair] = tracker
        return tracker

    def _get_vpin_estimator(self, pair: str):
        """Lazily build a RecursiveVPIN estimator when flowrisk is installed."""
        if not _FLOWRISK_AVAILABLE:
            return None
        est = self._vpin_estimators.get(pair)
        if est is None:
            try:
                from flowrisk import RecursiveVPIN  # type: ignore
                est = RecursiveVPIN(bucket_size=None, ewma_span=50)
                self._vpin_estimators[pair] = est
            except Exception as e:
                logger.debug(f"[OrderFlow:VPIN] RecursiveVPIN init failed for {pair}: {e}")
                est = None
        return est

    def analyze(self, pair: str, trades: List[Dict] = None) -> Dict:
        """Analyze order flow for a pair.

        Args:
            trades: Recent trades [{"price": float, "amount": float, "side": "buy"|"sell"}, ...]
                    If None, reads from derivatives_data.

        Returns: {"cvd_slope": float, "flow_toxicity": float, "squeeze_prob_long": float, ...}
        """
        result = {
            "cvd": 0.0,
            "cvd_slope": 0.0,
            "flow_toxicity": 0.0,
            "toxicity_composite": 0.0,
            "vpin": 0.0,
            "kyle_lambda": 0.0,
            "amihud": 0.0,
            "aggression_state": "neutral",
            "squeeze_probability_long": 0.0,
            "squeeze_probability_short": 0.0,
            "liq_cluster_distance": 1.0,
            "large_lot_detected": False,
            # Phase 27 Task 10 Hawkes additions
            "hawkes_intensity": 0.0,
            "hawkes_branching_ratio": 0.0,
            "hawkes_sizing_mult": 1.0,
        }

        # CVD from trades
        if trades:
            result.update(self._compute_cvd(pair, trades))
            result.update(self._compute_vpin(pair, trades))
            # Phase 27 Task 10: feed each trade as an event into the O(1) Hawkes tracker.
            tracker = self._get_hawkes(pair)
            for _ in trades:
                tracker.register_event()
            result["hawkes_intensity"] = round(tracker.current_intensity(), 4)
            result["hawkes_branching_ratio"] = round(tracker.branching_ratio, 4)
            result["hawkes_sizing_mult"] = round(tracker.sizing_mult(), 4)

        # Phase 27 Fix 8 / LOB integration — WIRE STATUS: NOT WIRED YET.
        # `lob_encoder.encode()` needs a Level-2 orderbook snapshot; the bot
        # only receives Level-1 tickers via the scheduler pipeline today. The
        # previous attempt to read from `trades` dict / `self._last_orderbook`
        # was unreachable (trades is a List[Dict], and _last_orderbook is
        # never populated). Keeping it would be worst-of-both: looks wired,
        # isn't. Wire path below when a real orderbook source exists.
        #
        # TODO(hydraquant): pipe Bybit depth@20 stream into publish_orderbook(pair, ob)
        #                    and then call `self.publish_orderbook(pair, orderbook)`
        #                    ahead of `analyze(pair, trades=[...])`. That method
        #                    populates `_last_orderbook` which the block below
        #                    would consume.
        orderbook = None
        last_ob = getattr(self, "_last_orderbook", None)
        if isinstance(last_ob, dict):
            orderbook = last_ob.get(pair)
        if orderbook:
            try:
                from lob_encoder import get_lob_encoder
                lob = get_lob_encoder()
                # Task 14: encode_and_publish deposits 'lob_state' to the
                # pheromone field so regime_classifier can read spread_regime
                # and HydraSizer's ILLIQUID detection works. The old
                # `encode()` returned the features but never broadcasted.
                lob_feat = lob.encode_and_publish(orderbook, pair=pair)
                result["lob_imbalance"] = lob_feat.get("imbalance_score", 0.0)
                result["lob_microprice_dev_bps"] = lob_feat.get("microprice_deviation_bps", 0.0)
                result["lob_spread_regime"] = lob_feat.get("spread_regime", "unknown")
            except Exception as e:
                logger.debug(f"[OrderFlow:LOB] encode failed ({pair}): {e}")

        # Liquidation radar from derivatives data
        liq_data = self._compute_liquidation_radar(pair)
        result.update(liq_data)

        # Aggression state
        if result["cvd_slope"] > 0.5:
            result["aggression_state"] = "aggressive_buying"
        elif result["cvd_slope"] < -0.5:
            result["aggression_state"] = "aggressive_selling"
        elif abs(result["cvd_slope"]) < 0.1:
            result["aggression_state"] = "balanced"

        return result

    def _compute_cvd(self, pair: str, trades: List[Dict]) -> Dict:
        """Compute Cumulative Volume Delta."""
        if pair not in self._cvd_history:
            self._cvd_history[pair] = deque(maxlen=500)

        buy_vol = sum(t["amount"] for t in trades if t.get("side") == "buy")
        sell_vol = sum(t["amount"] for t in trades if t.get("side") == "sell")
        delta = buy_vol - sell_vol

        self._cvd_history[pair].append(delta)
        cvd_values = list(self._cvd_history[pair])

        # CVD cumulative
        cvd = sum(cvd_values)

        # CVD slope (trend)
        if len(cvd_values) >= 10:
            recent = np.array(cvd_values[-10:])
            x = np.arange(len(recent))
            if recent.std() > 0:
                slope = float(np.polyfit(x, recent, 1)[0])
            else:
                slope = 0.0
        else:
            slope = 0.0

        # Large lot detection
        total_vol = buy_vol + sell_vol
        large_lot = False
        if trades:
            max_trade = max(t["amount"] for t in trades)
            avg_trade = total_vol / len(trades) if trades else 1
            large_lot = max_trade > avg_trade * 5

        return {
            "cvd": round(float(cvd), 4),
            "cvd_slope": round(float(slope), 4),
            "large_lot_detected": large_lot,
        }

    def _compute_vpin(self, pair: str, trades: List[Dict]) -> Dict:
        """Phase 27 Fix 9 (D5): compute volume-bucketed VPIN + Kyle λ + Amihud,
        then combine them into a CDF-percentile-based composite toxicity.

        The Phase 26 version returned `|buy - sell| / total` — that is Order
        Imbalance Ratio, NOT VPIN. Real VPIN requires volume bucketing; we use
        `flowrisk.RecursiveVPIN` when available and fall back to the legacy
        imbalance ratio (plus Kyle & Amihud regardless) otherwise.
        """
        if not trades:
            return {"vpin": 0.0, "flow_toxicity": 0.0,
                    "kyle_lambda": 0.0, "amihud": 0.0,
                    "toxicity_composite": 0.0}

        buy_vol = sum(t["amount"] for t in trades if t.get("side") == "buy")
        sell_vol = sum(t["amount"] for t in trades if t.get("side") == "sell")
        total_vol = buy_vol + sell_vol

        if total_vol == 0:
            return {"vpin": 0.0, "flow_toxicity": 0.0,
                    "kyle_lambda": 0.0, "amihud": 0.0,
                    "toxicity_composite": 0.0}

        # ── Real VPIN when flowrisk is installed, imbalance-ratio fallback otherwise ──
        vpin = abs(buy_vol - sell_vol) / total_vol  # safe fallback
        est = self._get_vpin_estimator(pair)
        if est is not None:
            try:
                for t in trades:
                    est.update(
                        price=float(t.get("price", 0.0)),
                        volume=float(t.get("amount", 0.0)),
                        side=t.get("side", "buy"),
                    )
                est_val = getattr(est, "vpin", None)
                if est_val is None and callable(getattr(est, "value", None)):
                    est_val = est.value()
                if est_val is not None:
                    vpin = float(est_val)
            except Exception as e:
                logger.debug(f"[OrderFlow:VPIN] RecursiveVPIN update failed ({pair}): {e}")

        # ── Kyle's λ: OLS regression of signed returns on signed volume ──
        kyle_lambda = 0.0
        try:
            prices = np.asarray([t.get("price", 0.0) for t in trades], dtype=np.float64)
            signed_vols = np.asarray(
                [(+1.0 if t.get("side") == "buy" else -1.0) * float(t.get("amount", 0.0))
                 for t in trades],
                dtype=np.float64,
            )
            if len(prices) >= 2 and np.all(prices > 0):
                rets = np.diff(np.log(prices))
                sv = signed_vols[1:]
                denom = float(np.var(sv))
                if denom > 1e-12:
                    kyle_lambda = float(abs(np.cov(rets, sv, bias=True)[0, 1] / denom))
        except Exception:
            kyle_lambda = 0.0

        # ── Amihud illiquidity: mean |return| / dollar_volume ──
        amihud = 0.0
        try:
            prices = np.asarray([t.get("price", 0.0) for t in trades], dtype=np.float64)
            amounts = np.asarray([float(t.get("amount", 0.0)) for t in trades], dtype=np.float64)
            if len(prices) >= 2 and np.all(prices > 0):
                rets = np.abs(np.diff(np.log(prices)))
                dollar_vol = (prices[1:] * amounts[1:])
                mask = dollar_vol > 0
                if mask.any():
                    amihud = float(np.mean(rets[mask] / dollar_vol[mask]))
        except Exception:
            amihud = 0.0

        # ── Composite toxicity: regime-relative CDF percentile rank ──
        # Histories are per-pair deques so each market learns its own baseline.
        for name, value, hist_map in (
            ("vpin", vpin, self._vpin_history),
            ("kyle", kyle_lambda, self._kyle_history),
            ("amihud", amihud, self._amihud_history),
        ):
            if pair not in hist_map:
                hist_map[pair] = deque(maxlen=500)
            hist_map[pair].append(float(value))

        vpin_pct = _percentile_rank(vpin, list(self._vpin_history[pair]))
        kyle_pct = _percentile_rank(kyle_lambda, list(self._kyle_history[pair]))
        amihud_pct = _percentile_rank(amihud, list(self._amihud_history[pair]))
        composite = 0.50 * vpin_pct + 0.30 * kyle_pct + 0.20 * amihud_pct

        # Legacy `flow_toxicity` kept so call sites (evidence_engine, MM mode)
        # don't regress; new `toxicity_composite` is the CDF-based successor.
        if vpin > VPIN_TOXIC:
            legacy_toxicity = 1.0
        elif vpin > VPIN_DANGER:
            legacy_toxicity = 0.8
        elif vpin > VPIN_CAUTION:
            legacy_toxicity = 0.5
        elif vpin > VPIN_SAFE:
            legacy_toxicity = 0.3
        else:
            legacy_toxicity = 0.1

        return {
            "vpin": round(float(vpin), 4),
            "kyle_lambda": round(float(kyle_lambda), 6),
            "amihud": round(float(amihud), 8),
            "flow_toxicity": round(float(legacy_toxicity), 2),
            "toxicity_composite": round(float(composite), 4),
        }

    def _compute_liquidation_radar(self, pair: str) -> Dict:
        """Estimate squeeze probability from derivatives data."""
        conn = get_db_connection(AI_DB_PATH)
        try:
            row = conn.execute("""
                SELECT funding_rate, long_short_ratio, open_interest_usd
                FROM derivatives_data
                WHERE pair = ?
                ORDER BY timestamp DESC LIMIT 1
            """, (pair,)).fetchone()

            if not row:
                return {
                    "squeeze_probability_long": 0.0,
                    "squeeze_probability_short": 0.0,
                    "liq_cluster_distance": 1.0,
                }

            funding = row["funding_rate"] or 0
            ls_ratio = row["long_short_ratio"] or 1.0
            oi = row["open_interest_usd"] or 0

            # Squeeze probability estimation
            # High funding + crowded longs → long squeeze risk
            squeeze_long = 0.0
            if funding > 0.01 and ls_ratio > 1.5:
                squeeze_long = min(0.3 + (funding - 0.01) * 50 + (ls_ratio - 1.5) * 0.2, 1.0)

            # Negative funding + crowded shorts → short squeeze risk
            squeeze_short = 0.0
            if funding < -0.01 and ls_ratio < 0.7:
                squeeze_short = min(0.3 + abs(funding + 0.01) * 50 + (0.7 - ls_ratio) * 0.3, 1.0)

            # Liquidation cluster distance (proxy: inverse of OI concentration)
            liq_distance = 1.0 / (1.0 + abs(funding) * 1000)

            return {
                "squeeze_probability_long": round(float(squeeze_long), 3),
                "squeeze_probability_short": round(float(squeeze_short), 3),
                "liq_cluster_distance": round(float(liq_distance), 3),
            }

        finally:
            conn.close()

    def should_veto_trade(self, pair: str, signal: str) -> tuple:
        """Check if order flow conditions warrant a trade veto.

        Returns (should_veto: bool, reason: str)

        Phase 27 Task 10: adds Hawkes branching-ratio veto at n ≥ 0.95 — a
        self-reinforcing cascade is imminent and entering is strictly dominated
        by waiting.
        """
        result = self.analyze(pair)

        # Phase 27 Task 10: Hawkes cascade veto (strictest gate, runs first)
        n = result.get("hawkes_branching_ratio", 0.0)
        if n >= HAWKES_VETO:
            return True, f"Hawkes cascade imminent (n={n:.2f} ≥ {HAWKES_VETO})"

        # Squeeze veto: high squeeze prob + signal in same direction as crowded side
        if signal == "BEARISH" and result["squeeze_probability_short"] > SQUEEZE_VETO_THRESHOLD:
            return True, f"short squeeze risk {result['squeeze_probability_short']:.0%}"
        if signal == "BULLISH" and result["squeeze_probability_long"] > SQUEEZE_VETO_THRESHOLD:
            return True, f"long squeeze risk {result['squeeze_probability_long']:.0%}"

        # Toxic flow veto — composite toxicity preferred, legacy flow_toxicity as fallback
        if result.get("toxicity_composite", 0.0) > 0.85 or result["flow_toxicity"] > 0.9:
            return True, (f"toxic flow (vpin={result['vpin']:.2f} "
                          f"composite={result.get('toxicity_composite', 0):.2f})")

        return False, ""

    def get_sizing_adjustment(self, pair: str) -> float:
        """Get sizing multiplier based on order flow. Returns 0.2-1.0.

        Phase 27 Task 10 layers Hawkes-based clamps on top of the Phase 26
        squeeze/toxicity reductions:
          0.70 ≤ n < 0.80 → ×0.6
          0.80 ≤ n < 0.90 → ×0.4
          0.90 ≤ n < 0.95 → ×0.2  (n ≥ 0.95 is a full veto in should_veto_trade)
        """
        result = self.analyze(pair)

        mult = 1.0
        if result["flow_toxicity"] > 0.75 or result.get("toxicity_composite", 0.0) > 0.75:
            mult *= 0.6
        if result["squeeze_probability_long"] > SQUEEZE_CLAMP_THRESHOLD:
            mult *= 0.7
        if result["squeeze_probability_short"] > SQUEEZE_CLAMP_THRESHOLD:
            mult *= 0.7

        # Phase 27 Task 10 Hawkes clamps
        n = result.get("hawkes_branching_ratio", 0.0)
        if n >= HAWKES_CLAMP_HIGH:
            mult *= 0.2
        elif n >= HAWKES_CLAMP_MED:
            mult *= 0.4
        elif n >= HAWKES_CLAMP_LOW:
            mult *= 0.6

        return max(mult, 0.2)

    def refit_hawkes_mle(self) -> int:
        """Phase 27 Task 10 / Item 7: custom Hawkes-Exp MLE refit per pair.

        Replaces the `tick` library dependency with a scipy.optimize.minimize
        negative-log-likelihood pass. For univariate Hawkes-Exp the NLL on
        an event sequence t_1 < ... < t_n in [0, T] is:

           NLL(μ,α,β) = ∫ λ(s) ds  −  Σ log λ(t_i)
                     = μT + α/β · Σ (1 − exp(−β(T−t_i)))
                       −  Σ log(μ + α · A_i)

        where A_i = Σ_{j<i} exp(−β(t_i − t_j)) is computed in O(n) recursively.

        Each tracker maintains a rolling deque of event timestamps in
        `_event_times` (added below); MLE only fires when ≥30 events accumulate.
        """
        from scipy import optimize as _optim
        import math as _math

        updated = 0
        for pair, tracker in self._hawkes.items():
            events = list(getattr(tracker, "_event_times", []) or [])
            if len(events) < 30:
                continue
            # Normalise so t_0 = 0 (numerical stability).
            t0 = events[0]
            ev = [t - t0 for t in events]
            T = ev[-1] + 1e-6

            def _nll(params):
                mu, alpha, beta = params
                if mu <= 0 or alpha < 0 or beta <= 0 or alpha >= 0.99 * beta:
                    return 1e9
                # Σ log λ(t_i) — O(n) recursion: A_i = exp(−β·dt)·(A_{i-1}+1)
                A_prev = 0.0
                log_sum = 0.0
                for i in range(len(ev)):
                    if i == 0:
                        A_i = 0.0
                    else:
                        dt = ev[i] - ev[i - 1]
                        A_i = _math.exp(-beta * dt) * (A_prev + 1.0)
                    lam = mu + alpha * A_i
                    if lam <= 0:
                        return 1e9
                    log_sum += _math.log(lam)
                    A_prev = A_i
                # Compensator integral
                integ = mu * T + (alpha / beta) * sum(
                    1.0 - _math.exp(-beta * (T - t)) for t in ev
                )
                return integ - log_sum

            try:
                x0 = [max(tracker.mu, 0.1), max(tracker.alpha, 0.1),
                      max(tracker.beta, 1.0)]
                bounds = [(1e-3, 100.0), (0.0, 10.0), (1e-2, 100.0)]
                res = _optim.minimize(_nll, x0=x0, method="L-BFGS-B",
                                       bounds=bounds, options={"maxiter": 50})
                if res.success:
                    mu_hat, alpha_hat, beta_hat = res.x
                    tracker.set_params(alpha=alpha_hat, beta=beta_hat,
                                        mu=mu_hat)
                    updated += 1
            except Exception as e:
                logger.debug(f"[Hawkes:MLE] {pair} optimise failed: {e}")
        return updated

    def publish_to_pheromone(self, result: Dict, pair: str):
        """Publish order flow state to pheromone field."""
        try:
            from pheromone_field import get_pheromone_field
            pfield = get_pheromone_field()
            pfield.deposit("order_flow", "order_flow_state", {
                "pair": pair,
                "cvd_slope": result["cvd_slope"],
                "flow_toxicity": result["flow_toxicity"],
                "squeeze_long": result["squeeze_probability_long"],
                "squeeze_short": result["squeeze_probability_short"],
                "aggression": result["aggression_state"],
            })
        except Exception:
            pass


# Singleton
_of_instance = None

def get_order_flow() -> OrderFlowAnalyzer:
    global _of_instance
    if _of_instance is None:
        _of_instance = OrderFlowAnalyzer()
    return _of_instance
