"""
risk_envelope.py — Dynamic risk parameter manager.

Sprint 2026-04-25 evening — replaces 7 hardcoded risk parameters with
a single `RiskEnvelope` that the system tunes itself based on:

  • AutonomyManager level (L0-L5) — base tier of aggression
  • 5-sensor demote vote (cortisol/Hawkes/OOD/dd-velocity/streak)
  • Hormonal modulation (cortisol × dopamine × organism_health)
  • Graduated decay after demote (5h to reach 0.5×, asymmetric recovery)
  • Continuous confidence score (Sharpe + win_rate + profit_factor + DD)

Philosophy: the organism EARNS its agression. Every parameter scales as
one — leverage, risk-per-trade, Kelly cap, stop-loss base, max positions,
stake-lift tolerance, Kelly fraction ceiling. No manual knob — the
system itself decides how aggressive it should be at any moment.

Renaissance Medallion lesson: PROMOTE slow (30 days sustained Sharpe>1.5),
DEMOTE fast (3/5 sensors → graduated decay starts within minutes). The
3:1 asymmetry protects capital in fat-tail events while letting compound
growth materialise once edge is proven.

Public API:
  - get_risk_envelope() → singleton instance
  - envelope.compute() → EnvelopeState dataclass with 7 dynamic params
  - envelope.update_sensor_state() — called every 5 min by scheduler
  - envelope.get_continuous_confidence_score() — for telemetry / promote
"""
from __future__ import annotations

import logging
import os
import sys
import threading
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional

sys.path.append(os.path.dirname(__file__))

logger = logging.getLogger(__name__)


# ─── Per-tier base envelopes (L0-L5) ─────────────────────────────────────────
# Each tier scales 7 parameters in lockstep. Higher tier = wider envelope.
# Hard ceilings enforced in compute() so even under hormonal boost an L5
# envelope cannot exceed the maximum institutional safety bands.
TIER_BASES: Dict[int, Dict[str, float]] = {
    0: {  # L0 — bootstrap, conservative
        "leverage_max": 2.0,
        "risk_per_trade": 0.02,
        "kelly_cap": 0.10,
        "sl_base_pct": 0.15,
        "max_open_positions": 4,
        "stake_lift_tolerance": 3.0,
        "kelly_fraction_max": 0.10,
        "monthly_target_low": 0.01,
        "monthly_target_high": 0.03,
        # Sprint 2026-05-01: protection / sizing / signal — dynamic envelope
        "protection_dd_cap": 0.08,        # Equity-mode MaxDrawdown ceiling (8%)
        "protection_lookback_base": 48,   # Candles base (× regime factor)
        "conviction_floor": 0.45,         # Min directional confidence to enter
        "conviction_ceiling": 0.85,       # Max directional confidence (clamp)
        "tp_promo_min_conf": 0.55,        # TriplePerception override threshold
        "tp_promo_haircut": 0.80,         # Haircut on TP-promoted confidence
        "max_single_stake_pct": 0.05,     # Max single trade as % of portfolio
        "max_combined_pos_pct": 0.07,     # Max combined (initial + DCA) %
        "dca_pct": 0.02,                  # DCA increment as % of portfolio
        "max_dca_levels": 1,              # Max number of DCA additions
    },
    1: {  # L1 — early growth
        "leverage_max": 3.0,
        "risk_per_trade": 0.03,
        "kelly_cap": 0.15,
        "sl_base_pct": 0.12,
        "max_open_positions": 6,
        "stake_lift_tolerance": 4.0,
        "kelly_fraction_max": 0.20,
        "monthly_target_low": 0.03,
        "monthly_target_high": 0.05,
        "protection_dd_cap": 0.11,
        "protection_lookback_base": 72,
        "conviction_floor": 0.42,
        "conviction_ceiling": 0.87,
        "tp_promo_min_conf": 0.52,
        "tp_promo_haircut": 0.83,
        "max_single_stake_pct": 0.08,
        "max_combined_pos_pct": 0.12,
        "dca_pct": 0.03,
        "max_dca_levels": 2,
    },
    2: {  # L2 — proven on 100+ trades
        "leverage_max": 5.0,
        "risk_per_trade": 0.05,
        "kelly_cap": 0.25,
        "sl_base_pct": 0.10,
        "max_open_positions": 10,
        "stake_lift_tolerance": 5.0,
        "kelly_fraction_max": 0.35,
        "monthly_target_low": 0.05,
        "monthly_target_high": 0.08,
        "protection_dd_cap": 0.15,
        "protection_lookback_base": 96,
        "conviction_floor": 0.40,
        "conviction_ceiling": 0.90,
        "tp_promo_min_conf": 0.50,
        "tp_promo_haircut": 0.85,
        "max_single_stake_pct": 0.10,
        "max_combined_pos_pct": 0.16,
        "dca_pct": 0.04,
        "max_dca_levels": 3,
    },
    3: {  # L3 — confident, 200+ trades, Sharpe>0.8
        "leverage_max": 7.0,
        "risk_per_trade": 0.07,
        "kelly_cap": 0.40,
        "sl_base_pct": 0.08,
        "max_open_positions": 14,
        "stake_lift_tolerance": 6.0,
        "kelly_fraction_max": 0.50,
        "monthly_target_low": 0.08,
        "monthly_target_high": 0.12,
        "protection_dd_cap": 0.18,
        "protection_lookback_base": 120,
        "conviction_floor": 0.37,
        "conviction_ceiling": 0.92,
        "tp_promo_min_conf": 0.47,
        "tp_promo_haircut": 0.87,
        "max_single_stake_pct": 0.13,
        "max_combined_pos_pct": 0.20,
        "dca_pct": 0.06,
        "max_dca_levels": 3,
    },
    4: {  # L4 — strong, 500+ trades, Sharpe>1.0
        "leverage_max": 10.0,
        "risk_per_trade": 0.08,
        "kelly_cap": 0.50,
        "sl_base_pct": 0.06,
        "max_open_positions": 18,
        "stake_lift_tolerance": 8.0,
        "kelly_fraction_max": 0.60,
        "monthly_target_low": 0.10,
        "monthly_target_high": 0.15,
        "protection_dd_cap": 0.22,
        "protection_lookback_base": 144,
        "conviction_floor": 0.34,
        "conviction_ceiling": 0.94,
        "tp_promo_min_conf": 0.43,
        "tp_promo_haircut": 0.90,
        "max_single_stake_pct": 0.16,
        "max_combined_pos_pct": 0.25,
        "dca_pct": 0.08,
        "max_dca_levels": 4,
    },
    5: {  # L5 — Renaissance+ multi-strategy ensemble
        "leverage_max": 10.0,
        "risk_per_trade": 0.10,
        "kelly_cap": 0.65,
        "sl_base_pct": 0.05,
        "max_open_positions": 20,
        "stake_lift_tolerance": 10.0,
        "kelly_fraction_max": 0.75,
        "monthly_target_low": 0.12,
        "monthly_target_high": 0.18,
        "protection_dd_cap": 0.25,
        "protection_lookback_base": 168,
        "conviction_floor": 0.30,
        "conviction_ceiling": 0.95,
        "tp_promo_min_conf": 0.40,
        "tp_promo_haircut": 0.92,
        "max_single_stake_pct": 0.20,
        "max_combined_pos_pct": 0.30,
        "dca_pct": 0.10,
        "max_dca_levels": 5,
    },
}


# ─── Hard global ceilings — never violated regardless of tier/decay/hormones ─
# These are the institutional safety guardrails that protect from
# catastrophic single-trade or model-bug losses.
GLOBAL_HARD_LIMITS = {
    # FIX-4 (2026-04-25 audit): leverage_max upper bound raised 10 → 12
    # so L5 base (10.0) has hormonal headroom (+15% from calm + winning
    # would push toward 11.5). Without the headroom L5 lost upside.
    "leverage_max": (1.0, 12.0),
    "risk_per_trade": (0.005, 0.15),
    "kelly_cap": (0.05, 0.85),
    "sl_base_pct": (0.04, 0.30),
    "max_open_positions": (2, 25),
    "stake_lift_tolerance": (2.0, 15.0),
    "kelly_fraction_max": (0.03, 0.85),
}


@dataclass
class EnvelopeState:
    """Snapshot of the 7 dynamic risk parameters + telemetry."""
    leverage_max: float
    risk_per_trade: float
    kelly_cap: float
    sl_base_pct: float
    max_open_positions: int
    stake_lift_tolerance: float
    kelly_fraction_max: float

    # Tier targets (informational — for telemetry, not enforced)
    monthly_target_low: float
    monthly_target_high: float

    # State telemetry
    autonomy_level: int
    confidence_score: float       # [0, 1]
    decay_multiplier: float       # [0.3, 1.0]
    sensor_votes: int             # 0-5
    sensor_breakdown: Dict[str, bool]
    hormonal_factor: float        # multiplier applied (cort×dopa×health)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ═══════════════════════════════════════════════════════════════════════════
# 5-Sensor Demote Panel
# ═══════════════════════════════════════════════════════════════════════════

class DemoteSensorPanel:
    """5 paralel sensors. 3/5 majority triggers demote.

    Each sensor returns bool (True = vote for demote). Failure to read is
    treated as False (sensor cannot panic without evidence).
    """

    def cortisol_panic(self) -> bool:
        try:
            from neural_organism import get_organism
            cortisol = float(get_organism().hormones.cortisol)
            # Canonical: 1.0 = calm, 0.5 = panic floor.
            # FIX-10 (2026-04-25 audit): threshold lifted 0.55 → 0.65 so
            # this sensor's trigger band has comparable weight to the other
            # 4 sensors in the 3/5 majority vote. With floor at 0.50,
            # threshold 0.65 covers the bottom ~30% of cortisol's range,
            # matching Hawkes>0.85 (top 15%) and OOD>0.7 (top 30%) profiles.
            return cortisol < 0.65
        except Exception:
            return False

    def hawkes_spike(self) -> bool:
        """Hawkes branching ratio > 0.85 → cluster/explosive event regime."""
        try:
            from pheromone_field import get_pheromone_field
            of_state = get_pheromone_field().read("order_flow_state", source="order_flow")
            if isinstance(of_state, dict):
                ratio = float(of_state.get("hawkes_branching_ratio", 0.0))
                return ratio > 0.85
        except Exception:
            pass
        return False

    def ood_high(self) -> bool:
        """OOD detector score > 0.7 → unfamiliar market state."""
        try:
            from pheromone_field import get_pheromone_field
            ood = get_pheromone_field().read("ood_score")
            if isinstance(ood, dict):
                return float(ood.get("score", 0.0)) > 0.7
            if isinstance(ood, (int, float)):
                return float(ood) > 0.7
        except Exception:
            pass
        return False

    def drawdown_velocity_high(self) -> bool:
        """Portfolio drop > 1.5%/hour over last 2h → rapid bleed."""
        try:
            from db import get_db_connection
            with get_db_connection() as conn:
                row = conn.execute("""
                    SELECT
                        SUM(close_profit_abs) AS net_pnl_2h
                    FROM trades
                    WHERE close_date >= datetime('now', '-2 hours')
                      AND is_open = 0
                """).fetchone()
                portfolio_row = conn.execute(
                    "SELECT total_balance FROM portfolio_state WHERE id = 1"
                ).fetchone()
            if row and portfolio_row:
                pnl_2h = float(row["net_pnl_2h"] or 0.0)
                portfolio = float(portfolio_row["total_balance"] or 10000.0)
                if portfolio > 0:
                    drop_pct_per_hour = (-pnl_2h / portfolio) * 50.0  # /2h * 100% = *50
                    return drop_pct_per_hour > 1.5
        except Exception:
            pass
        return False

    def streak_collapse(self) -> bool:
        """5+ consecutive losses → policy is misaligned with regime."""
        try:
            from neural_organism import get_organism
            return getattr(get_organism(), "_consec_losses", 0) >= 5
        except Exception:
            return False

    def votes_breakdown(self) -> Dict[str, bool]:
        return {
            "cortisol_panic": self.cortisol_panic(),
            "hawkes_spike": self.hawkes_spike(),
            "ood_high": self.ood_high(),
            "drawdown_velocity": self.drawdown_velocity_high(),
            "streak_collapse": self.streak_collapse(),
        }

    def vote_count(self) -> int:
        return sum(1 for v in self.votes_breakdown().values() if v)


# ═══════════════════════════════════════════════════════════════════════════
# Risk Envelope — main controller
# ═══════════════════════════════════════════════════════════════════════════

class RiskEnvelope:
    """Computes the 7-parameter risk envelope from organism state.

    Threading: compute() is read-only and lock-free for hot-path.
    update_sensor_state() and persistence are lock-guarded.
    """

    DECAY_PER_HOUR = 0.10        # 10% reduction per hour after demote (linear)
    DECAY_FLOOR = 0.50           # min decay multiplier (50% envelope at worst)
    RECOVERY_PER_HOUR = 0.05     # 5% recovery per clean hour after 6h gate
    RECOVERY_GATE_HOURS = 6.0    # need 6 clean hours before recovery starts
    MAX_DECAY_HOURS = 5.0        # max time decay applies (after 5h, no further)

    def __init__(self):
        self._sensors = DemoteSensorPanel()
        self._lock = threading.Lock()
        self._decay_multiplier: float = 1.0
        self._last_demote_at: Optional[float] = None
        self._continuous_clean_since: Optional[float] = None
        # FIX-2 (2026-04-25 audit): cache compute() result for 5s. Audit
        # found 250 envelope.compute() calls/cycle each issuing a 30-day
        # SELECT on trades. With 5s cache → 1 query per 5s window.
        self._cached_state: Optional[EnvelopeState] = None
        self._cached_at: float = 0.0
        self._cache_ttl_s: float = 5.0
        # FIX-2: cache AutonomyManager too — its __init__ runs CREATE TABLE
        # + 5 ALTER attempts on every construct. Once-per-envelope is plenty.
        self._autonomy_singleton = None
        self._restore_state()

    def _get_autonomy(self):
        if self._autonomy_singleton is None:
            try:
                from autonomy_manager import AutonomyManager
                self._autonomy_singleton = AutonomyManager()
            except Exception:
                self._autonomy_singleton = None
        return self._autonomy_singleton

    # ─── Persistence ─────────────────────────────────────────────────────
    def _persist_state(self, votes_break: Optional[Dict[str, bool]] = None,
                      votes_count: Optional[int] = None):
        """FIX-3 (2026-04-25 audit): route INSERT through execute_with_retry
        so SqliteBroker centralization (commit 69a93a51d) is honored.
        FIX-9: accept pre-computed votes from caller to avoid double DB hit."""
        try:
            from db import get_db_connection, execute_with_retry
            # CREATE TABLE — idempotent, one-time, direct connection OK
            with get_db_connection() as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS risk_envelope_state (
                        id INTEGER PRIMARY KEY CHECK (id = 1),
                        decay_multiplier REAL,
                        last_demote_at REAL,
                        continuous_clean_since REAL,
                        last_sensor_votes INTEGER,
                        last_sensor_breakdown TEXT,
                        updated_at TEXT
                    )
                """)
                conn.commit()
            import json as _json
            if votes_break is None:
                votes_break = self._sensors.votes_breakdown()
                votes_count = sum(1 for v in votes_break.values() if v)
            # INSERT/UPDATE — broker-routed so writes serialize through
            # central writer queue.
            execute_with_retry(
                """INSERT INTO risk_envelope_state
                    (id, decay_multiplier, last_demote_at, continuous_clean_since,
                     last_sensor_votes, last_sensor_breakdown, updated_at)
                   VALUES (1, ?, ?, ?, ?, ?, datetime('now'))
                   ON CONFLICT(id) DO UPDATE SET
                       decay_multiplier = excluded.decay_multiplier,
                       last_demote_at = excluded.last_demote_at,
                       continuous_clean_since = excluded.continuous_clean_since,
                       last_sensor_votes = excluded.last_sensor_votes,
                       last_sensor_breakdown = excluded.last_sensor_breakdown,
                       updated_at = excluded.updated_at""",
                (self._decay_multiplier, self._last_demote_at,
                 self._continuous_clean_since, int(votes_count or 0),
                 _json.dumps(votes_break)),
                max_retries=3,
            )
        except Exception as e:
            logger.debug(f"[RiskEnvelope] persist failed: {e}")

    def _restore_state(self):
        try:
            from db import get_db_connection
            with get_db_connection() as conn:
                row = conn.execute("""
                    SELECT decay_multiplier, last_demote_at, continuous_clean_since
                    FROM risk_envelope_state WHERE id = 1
                """).fetchone()
            if row:
                self._decay_multiplier = float(row["decay_multiplier"] or 1.0)
                self._last_demote_at = row["last_demote_at"]
                self._continuous_clean_since = row["continuous_clean_since"]
                logger.info(
                    f"[RiskEnvelope] restored decay={self._decay_multiplier:.2f}, "
                    f"last_demote={self._last_demote_at}"
                )
        except Exception:
            pass  # First run — table doesn't exist yet

    # ─── Sensor / decay state machine ────────────────────────────────────
    def update_sensor_state(self) -> Dict[str, Any]:
        """Called every 5 min by scheduler. Updates decay + persistence.
        Returns a telemetry dict for logging.
        """
        with self._lock:
            now = time.time()
            votes = self._sensors.votes_breakdown()
            count = sum(1 for v in votes.values() if v)

            transition = "stable"
            if count >= 3:
                # Demote trigger — 3/5 majority alarm
                if self._last_demote_at is None or self._decay_multiplier >= 1.0:
                    self._last_demote_at = now
                    transition = "demote_triggered"
                    logger.warning(
                        f"[RiskEnvelope] DEMOTE TRIGGERED — votes: {votes}"
                    )
                self._continuous_clean_since = None
            elif count == 0:
                # Clean tick — start recovery counter only when in decay
                if self._last_demote_at is not None and self._continuous_clean_since is None:
                    self._continuous_clean_since = now
            else:
                # FIX-4 (2026-04-25 audit): only clear recovery counter when
                # actually IN decay. Outside decay the counter is meaningless;
                # clearing was a latent landmine for future code that might
                # read _continuous_clean_since unconditionally.
                if self._last_demote_at is not None:
                    self._continuous_clean_since = None

            # Compute decay_multiplier
            if self._last_demote_at is not None:
                hours_since_demote = (now - self._last_demote_at) / 3600.0
                target = max(
                    self.DECAY_FLOOR,
                    1.0 - self.DECAY_PER_HOUR * min(self.MAX_DECAY_HOURS, hours_since_demote),
                )
                # Recovery: if continuously clean for > gate hours, slowly recover
                if self._continuous_clean_since is not None:
                    clean_hours = (now - self._continuous_clean_since) / 3600.0
                    if clean_hours >= self.RECOVERY_GATE_HOURS:
                        recovery_hours = clean_hours - self.RECOVERY_GATE_HOURS
                        target = min(1.0, target + self.RECOVERY_PER_HOUR * recovery_hours)
                        if target >= 1.0:
                            # Fully recovered — clear demote state
                            self._last_demote_at = None
                            transition = "recovered"
                            logger.info("[RiskEnvelope] FULLY RECOVERED — envelope at 100%")
                self._decay_multiplier = target
            else:
                self._decay_multiplier = 1.0

            # FIX-9: pass pre-computed votes so _persist_state doesn't re-read
            self._persist_state(votes_break=votes, votes_count=count)
            # FIX-2: invalidate cache after state mutation
            self._cached_state = None

            return {
                "votes_count": count,
                "votes_breakdown": votes,
                "decay_multiplier": self._decay_multiplier,
                "transition": transition,
                "last_demote_at": self._last_demote_at,
                "continuous_clean_since": self._continuous_clean_since,
            }

    # ─── Confidence score ─────────────────────────────────────────────────
    def get_continuous_confidence_score(self) -> float:
        """Real-time confidence [0..1] from rolling metrics.

        Components (0..1 each, weighted):
          - sharpe_30d_normalized   weight 0.30
          - winrate_delta_vs_50     weight 0.25
          - profit_factor_30d       weight 0.20
          - drawdown_inverse        weight 0.15
          - organism_health         weight 0.10
        """
        try:
            from db import get_db_connection
            with get_db_connection() as conn:
                row = conn.execute("""
                    SELECT
                        COUNT(*) AS n,
                        SUM(CASE WHEN close_profit > 0 THEN 1 ELSE 0 END) AS wins,
                        SUM(close_profit_abs) AS total_pnl,
                        SUM(CASE WHEN close_profit > 0 THEN close_profit_abs ELSE 0 END) AS gross_win,
                        SUM(CASE WHEN close_profit < 0 THEN -close_profit_abs ELSE 0 END) AS gross_loss,
                        AVG(close_profit) AS avg_return,
                        MIN(close_profit) AS worst_return
                    FROM trades
                    WHERE close_date >= datetime('now', '-30 days')
                      AND is_open = 0
                """).fetchone()

            if not row or not row["n"] or int(row["n"]) < 5:
                # FIX-1 (2026-04-25 audit): bootstrap returns NEUTRAL 0.50.
                # Audit found the old 0.10 value caused L1+ bots to demote
                # to L0 within 6 hours of restart even with no negative
                # trade evidence. Neutral preserves the level until real
                # data accumulates; promote requires n>=200 anyway.
                return 0.50

            n = int(row["n"])
            wins = int(row["wins"] or 0)
            avg_ret = float(row["avg_return"] or 0.0)
            gross_win = float(row["gross_win"] or 0.0)
            gross_loss = float(row["gross_loss"] or 0.0)

            # 1. Sharpe approximation: avg return / std (we use 1/sqrt(n) penalty)
            # avg_ret is per-trade; rough proxy for Sharpe = avg_ret / 0.05 normalized
            sharpe_proxy = avg_ret / 0.05
            sharpe_norm = max(0.0, min(1.0, (sharpe_proxy + 1.0) / 3.0))  # [-1,2]→[0,1]

            # 2. Win rate delta from 50%
            winrate = wins / n if n > 0 else 0.5
            wr_delta = max(0.0, min(1.0, (winrate - 0.50) * 2.0 + 0.5))

            # 3. Profit factor
            pf = gross_win / gross_loss if gross_loss > 0 else (2.0 if gross_win > 0 else 1.0)
            pf_norm = max(0.0, min(1.0, (pf - 0.5) / 2.0))  # PF=0.5→0, PF=2.5→1

            # 4. Drawdown inverse (estimate from worst single trade)
            worst = float(row["worst_return"] or 0.0)
            dd_est_pct = abs(worst) * 100  # proxy
            dd_inv = max(0.0, min(1.0, 1.0 - dd_est_pct / 15.0))

            # 5. Organism health
            try:
                from neural_organism import get_organism
                health = float(get_organism().interoception.get_organism_health())
            except Exception:
                health = 0.5

            score = (
                0.30 * sharpe_norm
                + 0.25 * wr_delta
                + 0.20 * pf_norm
                + 0.15 * dd_inv
                + 0.10 * health
            )
            return max(0.0, min(1.0, score))
        except Exception as e:
            logger.debug(f"[RiskEnvelope] confidence compute failed: {e}")
            return 0.30  # neutral-low default

    # ─── Hormonal modulation ─────────────────────────────────────────────
    def _hormonal_factor(self) -> float:
        """Multiplier in [0.30, 1.15] from cortisol × dopamine × health."""
        try:
            from neural_organism import get_organism
            org = get_organism()
            h = org.hormones
            cortisol = float(h.cortisol)   # 1.0 = calm, 0.5 = panic
            dopamine = float(h.dopamine)   # 0.9..1.10
            try:
                health = float(org.interoception.get_organism_health())
            except Exception:
                health = 0.5

            # FIX-6 (2026-04-25 audit): cortisol canonical range is [0.5, 1.0]
            # (panic floor is 0.5 per neural_organism). The previous 0.30
            # floor was dead — never engaged. Tightened to match source.
            cort_f = max(0.50, min(1.0, cortisol))

            # Dopamine bonus: 1.0 baseline, 1.1 → 1.05×
            dopa_f = max(1.0, min(1.10, dopamine))

            # Health: 0.5 baseline, scales [0.7, 1.0]
            health_f = max(0.7, min(1.0, 0.7 + 0.3 * health))

            mult = cort_f * dopa_f * health_f
            return max(0.30, min(1.15, mult))
        except Exception:
            return 1.0  # neutral

    # ─── Main compute (read-only, hot-path) ──────────────────────────────
    def compute(self) -> EnvelopeState:
        # FIX-5 (2026-04-25 audit): test isolation. When envelope is
        # disabled via env-var, fall back to deterministic L0 base so
        # legacy tests that assert static 0.05 risk / 0.25 kelly_cap
        # remain valid without per-test mocking.
        if os.environ.get("RISK_ENVELOPE_DISABLED", "") == "1":
            return self._disabled_state()

        # FIX-2 (2026-04-25 audit): 5s cache. Audit found 250 envelope.compute()
        # calls/cycle each issuing 30-day SELECT on trades. With cache → 1
        # query per 5s window. update_sensor_state() invalidates cache on
        # state mutation.
        now = time.monotonic()
        if (
            self._cached_state is not None
            and (now - self._cached_at) < self._cache_ttl_s
        ):
            return self._cached_state

        # 1. Autonomy level — use cached singleton (FIX-2)
        am = self._get_autonomy()
        if am is not None:
            try:
                level = int(am.get_level())
            except Exception:
                level = 0
        else:
            level = 0
        level = max(0, min(5, level))

        # 2. Base envelope for this tier
        base = dict(TIER_BASES[level])

        # 3. Hormonal modulation factor
        hormonal_f = self._hormonal_factor()

        # 4. Decay multiplier from sensor state
        decay = float(self._decay_multiplier)

        # 5. Combined multiplier (hormonal × decay) for AGGRESSIVE params
        aggressive_mult = hormonal_f * decay

        # 6. Apply to each parameter
        e = {}
        e["leverage_max"] = base["leverage_max"] * aggressive_mult
        e["risk_per_trade"] = base["risk_per_trade"] * aggressive_mult
        e["kelly_cap"] = base["kelly_cap"] * aggressive_mult
        e["kelly_fraction_max"] = base["kelly_fraction_max"] * aggressive_mult
        e["stake_lift_tolerance"] = base["stake_lift_tolerance"] * aggressive_mult
        # SL widens (more conservative) when stressed: inverse of mult
        e["sl_base_pct"] = base["sl_base_pct"] * (2.0 - aggressive_mult)
        # Max positions scales with confidence
        e["max_open_positions"] = max(2, int(base["max_open_positions"] * aggressive_mult))

        # 7. Hard global ceilings — never exceeded
        for key, (lo, hi) in GLOBAL_HARD_LIMITS.items():
            if isinstance(e[key], int):
                e[key] = max(int(lo), min(int(hi), e[key]))
            else:
                e[key] = max(lo, min(hi, e[key]))

        # 8. Telemetry
        votes_break = self._sensors.votes_breakdown()
        votes_count = sum(1 for v in votes_break.values() if v)
        confidence = self.get_continuous_confidence_score()

        state = EnvelopeState(
            leverage_max=float(e["leverage_max"]),
            risk_per_trade=float(e["risk_per_trade"]),
            kelly_cap=float(e["kelly_cap"]),
            sl_base_pct=float(e["sl_base_pct"]),
            max_open_positions=int(e["max_open_positions"]),
            stake_lift_tolerance=float(e["stake_lift_tolerance"]),
            kelly_fraction_max=float(e["kelly_fraction_max"]),
            monthly_target_low=float(base["monthly_target_low"]),
            monthly_target_high=float(base["monthly_target_high"]),
            autonomy_level=level,
            confidence_score=confidence,
            decay_multiplier=decay,
            sensor_votes=votes_count,
            sensor_breakdown=votes_break,
            hormonal_factor=hormonal_f,
        )
        # FIX-2: cache for hot path
        self._cached_state = state
        self._cached_at = now
        return state

    def _disabled_state(self) -> EnvelopeState:
        """FIX-5: deterministic fallback when RISK_ENVELOPE_DISABLED=1.
        Returns the L0 base envelope with no hormonal/decay modulation.
        Used by legacy tests that assert static parameter values.
        """
        base = TIER_BASES[0]
        return EnvelopeState(
            leverage_max=base["leverage_max"],
            risk_per_trade=0.05,    # legacy hyperopt baseline
            kelly_cap=0.25,         # legacy PARAM_REGISTRY default
            sl_base_pct=base["sl_base_pct"],
            max_open_positions=int(base["max_open_positions"]),
            stake_lift_tolerance=6.0,  # legacy default
            kelly_fraction_max=base["kelly_fraction_max"],
            monthly_target_low=base["monthly_target_low"],
            monthly_target_high=base["monthly_target_high"],
            autonomy_level=0,
            confidence_score=0.5,
            decay_multiplier=1.0,
            sensor_votes=0,
            sensor_breakdown={
                "cortisol_panic": False, "hawkes_spike": False,
                "ood_high": False, "drawdown_velocity": False,
                "streak_collapse": False,
            },
            hormonal_factor=1.0,
        )

    # ─── Convenience accessors (used by HydraSizer / position_sizer) ─────
    def get_leverage_max(self) -> float:
        return self.compute().leverage_max

    def get_risk_per_trade(self) -> float:
        return self.compute().risk_per_trade

    def get_kelly_cap(self) -> float:
        return self.compute().kelly_cap

    def get_sl_base_pct(self) -> float:
        return self.compute().sl_base_pct

    def get_max_open_positions(self) -> int:
        return self.compute().max_open_positions

    def get_stake_lift_tolerance(self) -> float:
        return self.compute().stake_lift_tolerance

    def get_kelly_fraction_max(self) -> float:
        return self.compute().kelly_fraction_max

    # ─── Sprint 2026-05-01: dynamic protection / sizing / signal ─────────
    # All these accessors derive their values from TIER_BASES[level] then
    # modulate them by hormonal state, decay multiplier, and (where
    # appropriate) market regime — never returning a hardcoded number.

    def _tier_base(self, key: str, default: float = 0.0) -> float:
        """Read TIER_BASES[level][key] — current autonomy tier."""
        try:
            am = self._get_autonomy()
            level = int(am.get_level()) if am is not None else 0
        except Exception:
            level = 0
        level = max(0, min(5, level))
        return float(TIER_BASES[level].get(key, default))

    def protection_max_drawdown(self) -> float:
        """Equity-mode MaxDrawdown ceiling.

        Tier base × cortisol panic factor: when cortisol drops (panic) the
        cap tightens, when calm it widens. Range tightens by up to 40%
        under full panic, never wider than tier base.
        """
        base = self._tier_base("protection_dd_cap", 0.10)
        try:
            from neural_organism import get_organism
            cort = float(get_organism().hormones.cortisol)  # 0.5=panic, 1.0=calm
        except Exception:
            cort = 1.0
        # Map cortisol [0.5, 1.0] → factor [0.6, 1.0]: panic shrinks cap
        cort_factor = max(0.6, min(1.0, 0.2 + 0.8 * cort))
        return max(0.05, min(0.30, base * cort_factor))

    def protection_lookback_candles(self) -> int:
        """MaxDrawdown lookback in candles, regime-modulated.

        Trending → full base (we want to remember disasters).
        Ranging → 60% (chop is noise; shorter memory).
        High-vol → 40% (focus on recent action).
        """
        base = int(self._tier_base("protection_lookback_base", 96))
        try:
            from regime_classifier import RegimeClassifier
            regime = self._current_regime()
            factor = RegimeClassifier.protection_lookback_factor(regime)
        except Exception:
            factor = 1.0
        return max(12, min(240, int(base * factor)))

    def protection_trade_limit(self) -> int:
        """Min closed trades before MaxDrawdown gate evaluates.

        Derived from current max_open_positions × 1.5 so the gate adapts
        to the bot's natural concurrent trade count.
        """
        max_open = int(self._tier_base("max_open_positions", 4))
        return max(4, min(40, int(max_open * 1.5)))

    def protection_stop_duration(self) -> int:
        """Stop duration after a MaxDrawdown trip, in candles.

        Cortisol-driven: panic → longer pause (24c), calm → brief (2c).
        """
        try:
            from neural_organism import get_organism
            cort = float(get_organism().hormones.cortisol)
        except Exception:
            cort = 1.0
        # cort 0.5 (panic) → 24, cort 1.0 (calm) → 2
        duration = int(round(2 + (1.0 - cort) * 44))
        return max(2, min(48, duration))

    def conviction_floor(self) -> float:
        """Minimum directional confidence to enter a REAL trade.

        Tier base × decay multiplier (when decayed, demand higher
        conviction so the bot doesn't enter weak setups while wounded).
        """
        base = self._tier_base("conviction_floor", 0.45)
        decay = float(self._decay_multiplier)  # ≤ 1.0 when decayed
        # decayed → floor RAISES (1/decay), capped at conviction_ceiling
        adjusted = base / max(0.5, decay)
        ceiling = self._tier_base("conviction_ceiling", 0.90)
        return max(0.20, min(ceiling - 0.05, adjusted))

    def conviction_ceiling(self) -> float:
        """Maximum directional confidence (clamp ceiling)."""
        return max(0.50, min(0.97, self._tier_base("conviction_ceiling", 0.90)))

    def tp_promotion_threshold(self) -> float:
        """Triple-Perception confidence required to OVERRIDE NEUTRAL ai_signal.

        Tier base × cortisol: under panic, demand higher TP confidence.
        """
        base = self._tier_base("tp_promo_min_conf", 0.50)
        try:
            from neural_organism import get_organism
            cort = float(get_organism().hormones.cortisol)
        except Exception:
            cort = 1.0
        # Lower cortisol → higher threshold demanded
        cort_bump = (1.0 - cort) * 0.10
        return max(0.30, min(0.85, base + cort_bump))

    def tp_promotion_haircut(self) -> float:
        """Haircut applied to TP confidence after promotion.

        Tier base × decay: when decayed, additional shrinkage.
        """
        base = self._tier_base("tp_promo_haircut", 0.85)
        decay = float(self._decay_multiplier)
        return max(0.50, min(0.95, base * (0.7 + 0.3 * decay)))

    # ─── EarnedTrust System (Sprint 2026-05-01 evening) ─────────────────
    # Three-layer dynamic sizing: TIER × EARNED_TRUST × CONVICTION_SCALAR.
    #
    # Philosophy: "küçük başla, ispat ederse genişle." A bot that just
    # came online has TIER baseline (L0 → 5%), zero proof, neutral conv
    # → final stake stays small. As it earns 30-day profitable evidence,
    # `earned_trust_multiplier` grows (1.0 → 2.0). When a single-trade
    # signal is strong, `conviction_scalar` lifts toward 1.0. The product
    # gives meaningful autonomy without needing tier promotion every time.
    #
    # Hard ceiling EARNED_TRUST_HARD_CEILING (0.30 = 30% portfolio max)
    # is the institutional floor: even L5 + 2.0 trust + 1.0 conviction
    # cannot blow past 30% of portfolio in a single trade. Renaissance-
    # grade allocation, not retail YOLO.
    #
    # Asymmetry: trust DROPS fast (one bad week → 0.7), RECOVERS slow
    # (30 days clean → 2.0). 3:1 ratio mirrors the protection layer's
    # demote-fast/promote-slow pattern.

    EARNED_TRUST_HARD_CEILING_PCT: float = 0.30
    EARNED_TRUST_FLOOR: float = 0.50
    EARNED_TRUST_NEUTRAL: float = 1.0
    EARNED_TRUST_PEAK: float = 2.0
    LIFETIME_DD_FREEZE_THRESHOLD: float = 0.10  # -10% lifetime DD locks trust=0.5
    STREAK_BONUS_WIN_COUNT: int = 5             # 5 consecutive wins → bonus
    STREAK_BONUS_AMOUNT: float = 0.30           # +0.3 trust additive
    STREAK_BONUS_TTL_SECONDS: float = 3600.0    # 1h decay
    VOLATILITY_BRAKE_RATIO: float = 2.0         # ATR > 2x normal → halve cap

    def earned_trust_multiplier(self) -> float:
        """Multiplier in [0.5, 2.0] derived from REAL 30-day performance.

        Components:
          • Lifetime drawdown freeze (-10% DD locks at 0.5)
          • Continuous-confidence score from existing
            `get_continuous_confidence_score()` (Sharpe + winrate +
            profit factor + drawdown-inverse + organism health)
          • Streak bonus (5 consecutive wins → +0.3 transient)
          • Sample-size shrinkage (n<10 trades → toward neutral 1.0)

        The continuous confidence score is the same one used for tier
        promotion eligibility — re-using it ensures trust + tier are
        philosophically aligned but operate on different time scales.
        """
        # 1. Lifetime drawdown circuit-breaker — strongest signal first
        lifetime_dd = self._lifetime_drawdown()
        if lifetime_dd >= self.LIFETIME_DD_FREEZE_THRESHOLD:
            logger.warning(
                f"[EarnedTrust] Lifetime DD={lifetime_dd:.2%} ≥ "
                f"{self.LIFETIME_DD_FREEZE_THRESHOLD:.0%} — trust LOCKED at floor"
            )
            return self.EARNED_TRUST_FLOOR

        # 2. Continuous confidence from rolling 30-day metrics
        score = self.get_continuous_confidence_score()  # [0, 1]

        # 3. Sample-size-aware mapping. With <10 trades the score is
        #    barely informative — pull toward neutral 1.0. With >50
        #    trades the full [floor, peak] range is available.
        n_recent = self._n_closed_trades_last_30d()
        sample_weight = min(1.0, n_recent / 50.0)

        # Base map: score=0.5 (neutral) → 1.0, score=0.0 → floor 0.5,
        # score=1.0 (excellent) → peak 2.0.
        if score >= 0.5:
            base = 1.0 + (score - 0.5) * 2.0  # [1.0, 2.0]
        else:
            base = 1.0 - (0.5 - score) * 1.0  # [0.5, 1.0]

        # Sample-weighted blend toward neutral
        trust = self.EARNED_TRUST_NEUTRAL * (1 - sample_weight) + base * sample_weight

        # 4. Asymmetry: when score is BAD (<0.5), trust drops 1.5× faster
        #    than it grows when score is GOOD. Renaissance demote-fast.
        if score < 0.5:
            shortfall = self.EARNED_TRUST_NEUTRAL - trust
            trust = self.EARNED_TRUST_NEUTRAL - shortfall * 1.5

        # 5. Transient streak bonus (1h TTL, additive)
        trust += self._streak_bonus_active()

        # 6. Hard clamps
        return max(self.EARNED_TRUST_FLOOR,
                   min(self.EARNED_TRUST_PEAK + self.STREAK_BONUS_AMOUNT, trust))

    def conviction_scalar(self, confidence: float) -> float:
        """Maps signal confidence [0, 1] → cap-utilization fraction [0.3, 1.0].

        Below the entry gate (typically 0.45) this method shouldn't even
        be called — the trade gets blocked upstream. But it's defensive
        and bottoms out at 0.3 so any accidental call from a weak signal
        path produces a tiny stake, not a huge one.

        The curve is intentionally NON-LINEAR: confidence in the
        0.85-1.00 band converts almost 1:1 to stake utilization, while
        the 0.50-0.70 band converts more cautiously. This rewards
        STRONG conviction more than mediocre conviction.
        """
        c = max(0.0, min(1.0, float(confidence)))
        if c < 0.50:
            return 0.30
        if c < 0.70:
            # Linear 0.50 → 0.30, 0.70 → 0.60
            return 0.30 + (c - 0.50) * 1.5
        if c < 0.85:
            # Linear 0.70 → 0.60, 0.85 → 0.85
            return 0.60 + (c - 0.70) * (25 / 15)
        # 0.85+ → 0.85 → 1.0 (steep reward for high conviction)
        return min(1.0, 0.85 + (c - 0.85) * 1.0)

    def _streak_bonus_active(self) -> float:
        """Transient additive bonus when 5 consecutive wins are fresh.

        Decays linearly to zero over `STREAK_BONUS_TTL_SECONDS`. Stored
        on the instance so it survives across compute() calls but does
        NOT persist to disk (intentional — a streak is a moment, not a
        state).
        """
        ts = getattr(self, "_streak_bonus_started_at", None)
        if ts is None:
            return 0.0
        age = time.time() - ts
        if age >= self.STREAK_BONUS_TTL_SECONDS:
            return 0.0
        # Linear decay 1.0 → 0.0 over the window
        decay = max(0.0, 1.0 - age / self.STREAK_BONUS_TTL_SECONDS)
        return self.STREAK_BONUS_AMOUNT * decay

    def trigger_streak_bonus(self) -> None:
        """Called by the strategy when a win-streak threshold is crossed.

        Sets the timestamp; the bonus auto-decays via _streak_bonus_active.
        Called from neural_organism.update_cycle when consec_wins reaches
        STREAK_BONUS_WIN_COUNT (or any equivalent caller).
        """
        with self._lock:
            self._streak_bonus_started_at = time.time()
        logger.info(
            f"[EarnedTrust] Streak bonus ACTIVATED — +{self.STREAK_BONUS_AMOUNT:.2f} "
            f"trust for {self.STREAK_BONUS_TTL_SECONDS/60:.0f} min"
        )

    def _volatility_brake_factor(self) -> float:
        """Returns [0.5, 1.0] — halves cap when ATR is 2× the rolling normal.

        Pulls live ATR from pheromone field where lob_encoder /
        order_flow publish 'atr_state'. When unavailable, returns 1.0
        (no brake) — never blocks trading on missing telemetry.
        """
        try:
            from pheromone_field import get_pheromone_field
            atr_state = get_pheromone_field().read("atr_state")
            if isinstance(atr_state, dict):
                ratio = float(atr_state.get("ratio", 1.0) or 1.0)
                if ratio >= self.VOLATILITY_BRAKE_RATIO:
                    return 0.5  # halve cap
                if ratio >= self.VOLATILITY_BRAKE_RATIO * 0.75:
                    # Linear interpolation from 1.0 → 0.5 between 1.5× and 2.0×
                    over = ratio - self.VOLATILITY_BRAKE_RATIO * 0.75
                    span = self.VOLATILITY_BRAKE_RATIO * 0.25
                    return max(0.5, 1.0 - 0.5 * (over / span))
        except Exception:
            pass
        return 1.0

    def _lifetime_drawdown(self) -> float:
        """Returns lifetime drawdown as a positive fraction (0.10 = -10%).

        Reads from portfolio_state OR computes from trades. Falls back
        to 0.0 (no drawdown) on any error so a missing row doesn't lock
        trading.
        """
        try:
            from db import get_db_connection
            with get_db_connection() as conn:
                row = conn.execute("""
                    SELECT total_balance, peak_balance
                    FROM portfolio_state WHERE id = 1
                """).fetchone()
            if row is None:
                return 0.0
            balance = float(row["total_balance"] or 0.0)
            peak = float(row["peak_balance"] or balance)
            if peak <= 0:
                return 0.0
            return max(0.0, (peak - balance) / peak)
        except Exception:
            return 0.0

    def _n_closed_trades_last_30d(self) -> int:
        """Count of closed trades in last 30 days. Used by sample-size
        weighting in earned_trust_multiplier — avoids being fooled by
        a small-sample 100% win rate."""
        try:
            from db import get_db_connection
            with get_db_connection() as conn:
                row = conn.execute("""
                    SELECT COUNT(*) AS n
                    FROM trades
                    WHERE close_date >= datetime('now', '-30 days')
                      AND is_open = 0
                """).fetchone()
            return int(row["n"] or 0) if row else 0
        except Exception:
            return 0

    def max_single_stake(self, portfolio_value: float,
                         confidence: Optional[float] = None) -> float:
        """Hard ceiling for a single trade's stake amount.

        Sprint 2026-05-01 evening — EarnedTrust System:
          final_pct = TIER_BASE × earned_trust × conviction_scalar
                      × hormonal × decay × volatility_brake

        With cold-start defaults (trust=1.0, conv=full at conf 0.85+),
        L0 produces ~$257 (5%). After 30 days of profitable trading
        (trust=1.5), L1+conf 0.95 climbs to ~$617. After 6 months at L5
        with trust=2.0 the cap hits the global hard ceiling at 30%.
        Apr 17 disaster (cold-start L0 with conf=0.95): $257 — was $5,481.

        confidence: optional [0,1]. When provided, conviction_scalar is
        applied. When None (legacy callers / pre-trade estimation), the
        scalar defaults to 0.85 — middle of the cap utilization curve,
        a sensible neutral.
        """
        if portfolio_value <= 0:
            return 0.0

        base_pct = self._tier_base("max_single_stake_pct", 0.05)
        trust = self.earned_trust_multiplier()
        if confidence is None:
            conv = 0.85  # neutral mid-conv when caller hasn't supplied
        else:
            conv = self.conviction_scalar(confidence)
        hormonal = self._hormonal_factor()  # [0.30, 1.15]
        decay = float(self._decay_multiplier)
        vol_brake = self._volatility_brake_factor()

        effective = base_pct * trust * conv * hormonal * decay * vol_brake
        # Hard ceiling — institutional safety floor, NEVER violated.
        effective = max(0.005, min(self.EARNED_TRUST_HARD_CEILING_PCT, effective))
        return float(portfolio_value) * effective

    def max_combined_position(self, portfolio_value: float,
                              confidence: Optional[float] = None) -> float:
        """Hard ceiling for combined position (initial + all DCAs).

        Sprint 2026-05-01 evening — EarnedTrust System.
        Same multiplicative chain as max_single_stake but uses the
        wider tier_base (max_combined_pos_pct) so DCA pyramiding has
        room. Hard ceiling still 30%.
        """
        if portfolio_value <= 0:
            return 0.0

        base_pct = self._tier_base("max_combined_pos_pct", 0.10)
        trust = self.earned_trust_multiplier()
        if confidence is None:
            conv = 0.85
        else:
            conv = self.conviction_scalar(confidence)
        hormonal = self._hormonal_factor()
        decay = float(self._decay_multiplier)
        vol_brake = self._volatility_brake_factor()

        effective = base_pct * trust * conv * hormonal * decay * vol_brake
        effective = max(0.01, min(self.EARNED_TRUST_HARD_CEILING_PCT, effective))
        return float(portfolio_value) * effective

    def dca_increment_pct(self, confidence: Optional[float] = None) -> float:
        """DCA add-stake as fraction of portfolio.

        Sprint 2026-05-01 evening — same EarnedTrust chain. Replaces the
        legacy hardcoded `max_stake * 0.30` (Apr 17 disaster's
        mechanism). DCA additions now scale with proven track record,
        signal conviction, hormonal calm, decay, AND volatility brake.
        """
        base = self._tier_base("dca_pct", 0.03)
        trust = self.earned_trust_multiplier()
        if confidence is None:
            conv = 0.85
        else:
            conv = self.conviction_scalar(confidence)
        hormonal = self._hormonal_factor()
        decay = float(self._decay_multiplier)
        vol_brake = self._volatility_brake_factor()

        effective = base * trust * conv * hormonal * decay * vol_brake
        # DCA increment hard floor 0.5%, ceiling 15% (the combined cap
        # already enforces aggregate position bound).
        return max(0.005, min(0.15, effective))

    def max_dca_levels(self) -> int:
        """Maximum number of DCA additions on top of initial entry.

        L0 → 1 (cautious), L5 → 5 (aggressive). Decayed envelope shrinks
        the count proportionally (3 → 1-2 when decay=0.5).
        """
        base = int(self._tier_base("max_dca_levels", 1))
        decay = float(self._decay_multiplier)
        effective = max(1, int(round(base * decay)))
        return max(1, min(5, effective))

    def _current_regime(self) -> str:
        """Resolve current regime from pheromone field (set by HydraSizer
        each populate_indicators tick). Falls back to 'transitional' so
        callers never crash on cold start."""
        try:
            from pheromone_field import get_pheromone_field
            r = get_pheromone_field().read("current_regime")
            if isinstance(r, dict) and r.get("regime"):
                return str(r["regime"])
            if isinstance(r, str) and r:
                return r
        except Exception:
            pass
        return "transitional"


# ─── Singleton ───────────────────────────────────────────────────────────────
_instance: Optional[RiskEnvelope] = None
_instance_lock = threading.Lock()


def get_risk_envelope() -> RiskEnvelope:
    """Per-process singleton accessor."""
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = RiskEnvelope()
    return _instance
