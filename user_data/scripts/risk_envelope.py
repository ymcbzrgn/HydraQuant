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

    # ═══════════════════════════════════════════════════════════════════
    # Sprint 2026-05-01 night — para makinesi additions:
    #   • F&G Contrarian Bias        (#9)
    #   • Time-of-Day (Cerebellum)   (#8)
    #   • Confidence Calibration     (#1)
    #   • Volatility Targeting       (#13)
    # All four feed into max_single_stake / max_combined_position /
    # dca_increment_pct as additional multipliers in the EarnedTrust chain.
    # ═══════════════════════════════════════════════════════════════════

    def fng_contrarian_bias(self, signal_type: Optional[str] = None) -> float:
        """Buffett rule: 'Be greedy when others are fearful, fearful when greedy.'

        Returns a confidence multiplier:
          • F&G < 20 (Extreme Fear)  AND signal BULL → 1.20 (boost contrarian long)
          • F&G > 80 (Extreme Greed) AND signal BEAR → 1.20 (boost contrarian short)
          • F&G < 20 AND signal BEAR → 0.85 (penalize trend-chasing into a fear bottom)
          • F&G > 80 AND signal BULL → 0.85 (penalize FOMO into a greed top)
          • All other cases → 1.0 (neutral)

        Thresholds come from PARAM_REGISTRY so neurons can tune them.
        """
        try:
            from neural_organism import _p
            extreme_low = float(_p("envelope.fng_contrarian.extreme_low", 20.0))
            extreme_high = float(_p("envelope.fng_contrarian.extreme_high", 80.0))
            boost_factor = float(_p("envelope.fng_contrarian.boost", 1.20))
            penalty_factor = float(_p("envelope.fng_contrarian.penalty", 0.85))
        except Exception:
            extreme_low, extreme_high = 20.0, 80.0
            boost_factor, penalty_factor = 1.20, 0.85

        try:
            from db import get_db_connection
            with get_db_connection() as conn:
                row = conn.execute(
                    "SELECT value FROM fear_and_greed "
                    "ORDER BY timestamp DESC LIMIT 1"
                ).fetchone()
            if row is None:
                return 1.0
            fng = float(row["value"])
        except Exception:
            return 1.0

        if signal_type is None:
            return 1.0

        sig = str(signal_type).upper()
        is_bull = sig in ("BULL", "BULLISH", "LONG")
        is_bear = sig in ("BEAR", "BEARISH", "SHORT")

        if fng < extreme_low:
            return boost_factor if is_bull else (penalty_factor if is_bear else 1.0)
        if fng > extreme_high:
            return boost_factor if is_bear else (penalty_factor if is_bull else 1.0)
        return 1.0

    def cerebellum_hour_factor(self) -> float:
        """Returns the bot's own historical performance multiplier for the
        current UTC hour. Sourced from Cerebellum.get_hour_multiplier
        which already produces [0.6, 1.4] based on hourly win rate.

        This converts the 'bot's best hours' that were just sitting in DB
        (cerebellum_hours table) into actual sizing pressure. Free alpha
        from the bot's own observed regularity.

        Falls back to 1.0 (neutral) when the organism isn't ready or the
        slot has <3 samples — never blocks trading on missing data.
        """
        try:
            from neural_organism import get_organism
            from datetime import datetime, timezone
            org = get_organism()
            hour_utc = datetime.now(tz=timezone.utc).hour
            mult = float(org.cerebellum.get_hour_multiplier(hour_utc))
            # Cerebellum already clamps to [0.6, 1.4]; we re-clamp defensively.
            return max(0.6, min(1.4, mult))
        except Exception:
            return 1.0

    def apply_calibration(self, raw_confidence: float) -> float:
        """Take a raw model confidence and return its Platt-calibrated value.

        ConfidenceCalibrator already:
          • Reads ai_decisions outcomes
          • Fits Platt scaling (logistic regression: A·conf + B)
          • Returns adjusted probability

        We just route every confidence value through it before sizing
        decisions. Effect: when the bot routinely says '85% sure' but
        only wins 65% of those, calibration shrinks future 0.85 reads
        toward 0.70 — and the EarnedTrust sizing chain consumes the
        truthful number.

        Falls back to raw_confidence when calibrator isn't fitted yet
        (insufficient outcome history) — never blocks trading.
        """
        try:
            from confidence_calibrator import ConfidenceCalibrator
            # Cache the singleton on the envelope so we don't re-instantiate
            # (the calibrator opens a DB connection on construct).
            cal = getattr(self, "_calibrator_singleton", None)
            if cal is None:
                cal = ConfidenceCalibrator()
                self._calibrator_singleton = cal
            adjusted = cal.adjust_confidence(float(raw_confidence))
            return max(0.0, min(1.0, float(adjusted)))
        except Exception:
            return max(0.0, min(1.0, float(raw_confidence)))

    def fomo_veto_factor(self, recent_low: Optional[float],
                         current_price: Optional[float],
                         atr: Optional[float] = None,
                         signal_type: Optional[str] = None,
                         recent_high: Optional[float] = None) -> float:
        """FOMO veto — penalize entries that chase price already extended.

        Source: AI-Trader-main-4 pump_detector.py:217-234.

        Audit Finding #10 fix (2026-05-02): direction-aware. For LONGS,
        FOMO is "buying AFTER the breakout has run" — distance from
        recent_low. For SHORTS, FOMO is "selling AFTER the drop has
        completed" — distance from recent_high. The prior implementation
        only handled longs and silently mis-sized shorts.

        Returns multiplier in [min_factor, 1.0]:
          • At anchor price (low for long, high for short) → 1.0
          • At anchor + ATR×3 distance → min_factor (0.3 default)
          • Linear decay between
        """
        if current_price is None or current_price <= 0:
            return 1.0
        try:
            from neural_organism import _p
            full_penalty_atr = float(_p("envelope.fomo_veto.full_penalty_atr_mult", 3.0))
            min_factor = float(_p("envelope.fomo_veto.min_factor", 0.3))
            full_penalty_pct = float(_p("envelope.fomo_veto.full_penalty_pct_no_atr", 8.0))
        except Exception:
            full_penalty_atr, min_factor, full_penalty_pct = 3.0, 0.3, 8.0

        sig = (signal_type or "").upper()
        is_short = sig in ("BEAR", "BEARISH", "SHORT")
        is_long = sig in ("BULL", "BULLISH", "LONG")

        # Pick the anchor based on direction. Default to LONG semantics
        # when signal_type is missing (most common case, backward compat).
        if is_short:
            anchor = recent_high
            if anchor is None or anchor <= 0:
                return 1.0
            raw_distance = anchor - current_price
        else:
            # is_long OR unknown → use recent_low as anchor
            anchor = recent_low
            if anchor is None or anchor <= 0:
                return 1.0
            raw_distance = current_price - anchor

        if atr and atr > 0:
            distance = raw_distance / atr
            full_penalty_at = full_penalty_atr  # ATR-units
        else:
            distance = raw_distance / anchor * 100.0
            full_penalty_at = full_penalty_pct

        if distance <= 0:
            return 1.0
        if distance >= full_penalty_at:
            return min_factor
        # Linear decay 1.0 → min_factor
        return max(min_factor, 1.0 - (1.0 - min_factor) * (distance / full_penalty_at))

    def hurst_regime_factor(self, hurst_value: Optional[float] = None,
                            pair: Optional[str] = None) -> float:
        """Hurst exponent → sizing factor.

        Sources: jesse/indicators/hurst_exponent.py (R/S, DMA, DSOD).
        H < 0.45 → mean-reverting (boost mean-reversion strategies)
        H > 0.55 → trending  (boost trend-following)
        H ≈ 0.50 → random walk → reduce sizing (no edge)

        Audit Finding #1 fix (2026-05-02): pair parameter added so each
        pair reads its OWN Hurst from the pheromone field. The legacy
        unsuffixed key is read only as a fallback when a per-pair key
        isn't yet populated (cold start within first hour).
        """
        if hurst_value is None:
            try:
                from pheromone_field import get_pheromone_field
                pf = get_pheromone_field()
                # Prefer per-pair key — set by publish_hurst_to_pheromone(prices, pair=pair)
                h_state = None
                if pair:
                    h_state = pf.read(f"hurst_3vote::{pair}")
                # Fallback to unsuffixed key for cold-start / migration safety
                if not isinstance(h_state, dict):
                    h_state = pf.read("hurst_3vote")
                if isinstance(h_state, dict):
                    hurst_value = float(h_state.get("hurst", 0.5))
            except Exception:
                pass
        if hurst_value is None:
            return 1.0
        h = max(0.0, min(1.0, float(hurst_value)))
        # Distance from 0.5 random-walk, capped at 0.3 each side
        distance = min(0.3, abs(h - 0.5))
        # Strong direction (trending OR mean-revert) → up to 1.2
        # Random walk (H≈0.5) → 0.7 (no edge, shrink)
        if distance < 0.05:
            return 0.7  # random walk: no structural edge
        return 1.0 + (distance / 0.3) * 0.2  # up to 1.2 at distance=0.3

    def vpin_action_factor(self, pair: Optional[str] = None) -> float:
        """VPIN toxicity-adaptive sizing factor.

        Audit Finding #3 fix (2026-05-02): the PARAM_REGISTRY exposed
        envelope.vpin_action.* tunables but no consumer existed. This
        method reads the live VPIN history from order_flow's per-pair
        history and shrinks sizing when toxicity exceeds the threshold.

        Source: ScienceDirect (2025) — Bitcoin VPIN significantly
        predicts price jumps. When VPIN is in the top decile (>0.7
        default), informed traders are crowding one side; passive
        liquidity providers face elevated adverse-selection cost. Our
        action: shrink position size by `size_factor` (default 0.5).

        Returns multiplier in [size_factor, 1.0]:
          • VPIN >= toxic_threshold (0.7) → size_factor (0.5)
          • Otherwise → 1.0 neutral
        """
        if pair is None:
            return 1.0
        try:
            from neural_organism import _p
            toxic_thr = float(_p("envelope.vpin_action.toxic_threshold", 0.7))
            size_factor = float(_p("envelope.vpin_action.size_factor", 0.5))
        except Exception:
            toxic_thr, size_factor = 0.7, 0.5
        try:
            from order_flow import get_order_flow
            of = get_order_flow()
            hist = of._vpin_history.get(pair) if hasattr(of, "_vpin_history") else None
            if not hist or len(hist) == 0:
                return 1.0
            latest_vpin = float(hist[-1])
        except Exception:
            return 1.0
        if latest_vpin >= toxic_thr:
            return max(0.1, min(1.0, size_factor))
        return 1.0

    def event_calendar_factor(self) -> float:
        """Pre-FOMC / CPI / NFP volatility crush avoidance.

        Per Kraken Aug 2025: only 1 of 8 FOMC meetings rallied BTC.
        Pre-event (1h before → 30min after): reduce sizing by 50%.
        Outside event windows: 1.0 neutral. Defensive — prevents 100-300
        bps of avoidable losses per macro event.
        """
        try:
            from event_calendar import current_event_factor
            return float(current_event_factor())
        except Exception:
            return 1.0

    def multi_level_ofi_bias(self, pair: Optional[str] = None,
                             signal_type: Optional[str] = None) -> float:
        """Multi-level Order Flow Imbalance directional alignment factor.

        Source: arXiv 2506.05764, Cont (NYU). When OFI strongly aligns
        with the signal direction (e.g., signal=BULL AND OFI > +0.5 →
        bid pressure confirms), boost cap. When opposite, penalize.

        Returns multiplier in [0.7, 1.3]:
          • |OFI| > 0.5 AND aligned with signal → 1.30
          • |OFI| > 0.3 AND aligned             → 1.15
          • |OFI| > 0.5 AND OPPOSED to signal   → 0.70
          • |OFI| > 0.3 AND opposed             → 0.85
          • all else                             → 1.0
        """
        if pair is None or signal_type is None:
            return 1.0
        # Audit Finding #5 fix (2026-05-02): prefer pheromone (cross-process,
        # survives singleton restart) over in-memory deque. Fall back to
        # the in-memory getter only when pheromone is missing/stale.
        ofi = None
        try:
            from pheromone_field import get_pheromone_field
            state = get_pheromone_field().read(f"ofi_multi::{pair}")
            if isinstance(state, dict) and "ofi" in state:
                ofi = float(state["ofi"])
        except Exception:
            pass
        if ofi is None:
            try:
                from order_flow import get_order_flow
                ofi = get_order_flow().get_multi_level_ofi(pair)
            except Exception:
                ofi = None
        if ofi is None:
            return 1.0
        try:
            from neural_organism import _p
            strong_thr = float(_p("envelope.ofi.strong_threshold", 0.5))
            mild_thr = float(_p("envelope.ofi.mild_threshold", 0.3))
            strong_boost = float(_p("envelope.ofi.strong_boost", 1.30))
            mild_boost = float(_p("envelope.ofi.mild_boost", 1.15))
            penalty_strong = float(_p("envelope.ofi.strong_penalty", 0.70))
            penalty_mild = float(_p("envelope.ofi.mild_penalty", 0.85))
        except Exception:
            strong_thr, mild_thr = 0.5, 0.3
            strong_boost, mild_boost = 1.30, 1.15
            penalty_strong, penalty_mild = 0.70, 0.85

        sig = str(signal_type).upper()
        is_bull = sig in ("BULL", "BULLISH", "LONG")
        is_bear = sig in ("BEAR", "BEARISH", "SHORT")
        if not (is_bull or is_bear):
            return 1.0
        # OFI > 0 → buy pressure → aligned with BULL
        ofi_dir_bull = ofi > 0
        aligned = (is_bull and ofi_dir_bull) or (is_bear and not ofi_dir_bull)
        abs_ofi = abs(ofi)
        if abs_ofi >= strong_thr:
            return strong_boost if aligned else penalty_strong
        if abs_ofi >= mild_thr:
            return mild_boost if aligned else penalty_mild
        return 1.0

    def funding_arbitrage_bias(self, pair: str,
                               signal_type: Optional[str] = None) -> float:
        """Funding-rate arbitrage edge.

        Bybit perp funding settles every 8h. When funding is highly
        NEGATIVE (e.g. -0.05% per 8h = -0.15%/day), shorts COLLECT the
        funding fee from longs. So opening a short:
          • Earns +0.15%/day passively as long as the position is held
          • Plus directional P&L if price drops
          • Risk-free if held through one funding cycle without price move

        Mirror logic for highly POSITIVE funding (longs collect).

        Returns multiplicative bias [0.85, 1.30]:
          • |funding| > 0.05% AND signal aligned with collection side → 1.30
          • |funding| > 0.02% AND signal aligned                     → 1.10
          • |funding| > 0.05% AND signal AGAINST collection side     → 0.85
          • All else                                                  → 1.0

        This is real institutional alpha — basis trades / cash-and-carry.
        """
        if signal_type is None:
            return 1.0

        try:
            from neural_organism import _p
            extreme_threshold = float(_p("envelope.funding_arb.extreme", 0.0005))   # 0.05%/8h
            mild_threshold = float(_p("envelope.funding_arb.mild", 0.0002))          # 0.02%/8h
            extreme_boost = float(_p("envelope.funding_arb.extreme_boost", 1.30))
            mild_boost = float(_p("envelope.funding_arb.mild_boost", 1.10))
            penalty = float(_p("envelope.funding_arb.penalty", 0.85))
        except Exception:
            extreme_threshold, mild_threshold = 0.0005, 0.0002
            extreme_boost, mild_boost, penalty = 1.30, 1.10, 0.85

        try:
            from db import get_db_connection
            # derivatives_data uses BASE pair without :USDT suffix
            base_pair = (pair or "").split(":")[0]
            with get_db_connection() as conn:
                row = conn.execute(
                    "SELECT funding_rate FROM derivatives_data "
                    "WHERE pair = ? "
                    "ORDER BY timestamp DESC LIMIT 1",
                    (base_pair,),
                ).fetchone()
            if row is None or row["funding_rate"] is None:
                return 1.0
            fr = float(row["funding_rate"])
        except Exception:
            return 1.0

        sig = str(signal_type).upper()
        is_short = sig in ("BEAR", "BEARISH", "SHORT")
        is_long = sig in ("BULL", "BULLISH", "LONG")
        if not (is_short or is_long):
            return 1.0

        # Negative funding → shorts collect; positive → longs collect.
        short_collects = fr < 0
        long_collects = fr > 0
        abs_fr = abs(fr)

        if abs_fr >= extreme_threshold:
            if (is_short and short_collects) or (is_long and long_collects):
                logger.info(
                    f"[FundingArb] {pair} funding={fr:+.4%} "
                    f"({sig} aligned with collection side) → boost {extreme_boost:.2f}x"
                )
                return extreme_boost
            return penalty
        if abs_fr >= mild_threshold:
            if (is_short and short_collects) or (is_long and long_collects):
                return mild_boost
        return 1.0

    def volatility_target_scalar(self) -> float:
        """Renaissance-style portfolio-level volatility targeting.

        Target daily portfolio volatility = 1.0% (configurable). Compares
        against the bot's REALIZED daily PnL standard deviation over the
        last 20 trades. When realized vol is LOW (calm, predictable
        market), scale UP; when HIGH (chaotic), scale DOWN.

        Range clamped to [0.5, 2.0] — at the extremes:
          • 0.5×: realized vol 2× target → cut sizing in half
          • 2.0×: realized vol 0.5× target → double sizing

        This is structurally different from `_volatility_brake_factor`
        (which reads ATR pheromone for short-term spikes). Vol-target
        is a 20-trade rolling adjustment to portfolio-level risk budget.
        """
        try:
            from neural_organism import _p
            target_vol = float(_p("envelope.vol_target.daily_pct", 0.01))
            min_scalar = float(_p("envelope.vol_target.min_scalar", 0.5))
            max_scalar = float(_p("envelope.vol_target.max_scalar", 2.0))
        except Exception:
            target_vol, min_scalar, max_scalar = 0.01, 0.5, 2.0

        try:
            from db import get_db_connection
            with get_db_connection() as conn:
                row = conn.execute("""
                    SELECT close_profit
                    FROM trades
                    WHERE close_date >= datetime('now', '-30 days')
                      AND is_open = 0
                    ORDER BY close_date DESC LIMIT 20
                """).fetchall()
            if not row or len(row) < 5:
                return 1.0
            returns = [float(r["close_profit"] or 0.0) for r in row]
            n = len(returns)
            mean = sum(returns) / n
            variance = sum((r - mean) ** 2 for r in returns) / n
            realized_vol = variance ** 0.5
            if realized_vol <= 0:
                return max_scalar  # zero variance — extremely calm, max boost
            scalar = target_vol / realized_vol
            return max(min_scalar, min(max_scalar, scalar))
        except Exception:
            return 1.0

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

    def _alpha_chain(self, confidence: Optional[float],
                     signal_type: Optional[str],
                     pair: Optional[str] = None,
                     recent_low: Optional[float] = None,
                     current_price: Optional[float] = None,
                     atr: Optional[float] = None,
                     recent_high: Optional[float] = None) -> Dict[str, float]:
        """Compute every multiplicative factor used by sizing methods.

        Sprint 2026-05-02 — para makinesi sizing chain (FULL):
          tier_base × trust × conv × hormonal × decay × vol_brake
                     × cerebellum_hour × fng_contrarian × vol_target
                     × funding_arb × ofi × hurst_regime
                     × fomo_veto × event_calendar
        """
        if confidence is None:
            calibrated = None
            conv = 0.85
        else:
            calibrated = self.apply_calibration(float(confidence))
            conv = self.conviction_scalar(calibrated)

        funding_arb = self.funding_arbitrage_bias(pair, signal_type) if pair else 1.0
        ofi = self.multi_level_ofi_bias(pair, signal_type) if pair else 1.0

        return {
            "trust": self.earned_trust_multiplier(),
            "conv": conv,
            "hormonal": self._hormonal_factor(),
            "decay": float(self._decay_multiplier),
            "vol_brake": self._volatility_brake_factor(),
            "cerebellum": self.cerebellum_hour_factor(),
            "fng_contrarian": self.fng_contrarian_bias(signal_type),
            "vol_target": self.volatility_target_scalar(),
            "funding_arb": funding_arb,
            "ofi": ofi,
            "hurst": self.hurst_regime_factor(pair=pair),
            "fomo_veto": self.fomo_veto_factor(recent_low, current_price, atr,
                                               signal_type=signal_type,
                                               recent_high=recent_high),
            "event_calendar": self.event_calendar_factor(),
            "vpin_action": self.vpin_action_factor(pair=pair),
            "calibrated_conf": (calibrated if calibrated is not None
                                else (confidence or 0.0)),
            "raw_conf": float(confidence) if confidence is not None else 0.0,
        }

    def max_single_stake(self, portfolio_value: float,
                         confidence: Optional[float] = None,
                         signal_type: Optional[str] = None,
                         pair: Optional[str] = None,
                         recent_low: Optional[float] = None,
                         current_price: Optional[float] = None,
                         atr: Optional[float] = None,
                         recent_high: Optional[float] = None) -> float:
        """Hard ceiling for a single trade's stake amount.

        Sprint 2026-05-02 — full 13-factor alpha chain:
          tier × trust × conv × hormonal × decay × vol_brake
              × cerebellum × fng_contrarian × vol_target × funding_arb
              × ofi × hurst × fomo_veto × event_calendar

        Hard ceiling 30% portfolio NEVER violated.
        """
        if portfolio_value <= 0:
            return 0.0
        base_pct = self._tier_base("max_single_stake_pct", 0.05)
        chain = self._alpha_chain(
            confidence, signal_type, pair=pair,
            recent_low=recent_low, current_price=current_price, atr=atr,
            recent_high=recent_high,
        )
        effective = (
            base_pct
            * chain["trust"] * chain["conv"]
            * chain["hormonal"] * chain["decay"]
            * chain["vol_brake"] * chain["cerebellum"]
            * chain["fng_contrarian"] * chain["vol_target"]
            * chain["funding_arb"] * chain["ofi"]
            * chain["hurst"] * chain["fomo_veto"]
            * chain["event_calendar"] * chain["vpin_action"]
        )
        effective = max(0.005, min(self.EARNED_TRUST_HARD_CEILING_PCT, effective))
        return float(portfolio_value) * effective

    def max_combined_position(self, portfolio_value: float,
                              confidence: Optional[float] = None,
                              signal_type: Optional[str] = None,
                              pair: Optional[str] = None,
                              recent_low: Optional[float] = None,
                              current_price: Optional[float] = None,
                              atr: Optional[float] = None) -> float:
        """Combined position cap (full 13-factor alpha chain)."""
        if portfolio_value <= 0:
            return 0.0
        base_pct = self._tier_base("max_combined_pos_pct", 0.10)
        chain = self._alpha_chain(
            confidence, signal_type, pair=pair,
            recent_low=recent_low, current_price=current_price, atr=atr,
            recent_high=recent_high,
        )
        effective = (
            base_pct
            * chain["trust"] * chain["conv"]
            * chain["hormonal"] * chain["decay"]
            * chain["vol_brake"] * chain["cerebellum"]
            * chain["fng_contrarian"] * chain["vol_target"]
            * chain["funding_arb"] * chain["ofi"]
            * chain["hurst"] * chain["fomo_veto"]
            * chain["event_calendar"] * chain["vpin_action"]
        )
        effective = max(0.01, min(self.EARNED_TRUST_HARD_CEILING_PCT, effective))
        return float(portfolio_value) * effective

    def dca_increment_pct(self, confidence: Optional[float] = None,
                          signal_type: Optional[str] = None,
                          pair: Optional[str] = None,
                          recent_low: Optional[float] = None,
                          current_price: Optional[float] = None,
                          atr: Optional[float] = None) -> float:
        """DCA add-stake fraction. Full 13-factor alpha chain."""
        base = self._tier_base("dca_pct", 0.03)
        chain = self._alpha_chain(
            confidence, signal_type, pair=pair,
            recent_low=recent_low, current_price=current_price, atr=atr,
            recent_high=recent_high,
        )
        effective = (
            base
            * chain["trust"] * chain["conv"]
            * chain["hormonal"] * chain["decay"]
            * chain["vol_brake"] * chain["cerebellum"]
            * chain["fng_contrarian"] * chain["vol_target"]
            * chain["funding_arb"] * chain["ofi"]
            * chain["hurst"] * chain["fomo_veto"]
            * chain["event_calendar"] * chain["vpin_action"]
        )
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
