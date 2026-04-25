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
