"""
Phase 27 Task 1 (E1 Ajani): Per-pair Bayesian Kelly — Prensip 0.

Replaces the Phase 3.5.2 single-row global Kelly (`CHECK(id = 1)`) that forced
BTC, ETH, DOGE etc. to share one Beta posterior. Every (pair, regime) now owns
its row in bayesian_kelly_per_pair (schema defined in db.py).

Why: Peters (2019, Nature Physics) — time-average growth rate g = μ − σ²/2.
Global Kelly 0.835 was a Simpson's Paradox of winning-pair aggregates; BTC's
σ≈0.70 demands Kelly ≤ 0.25, ETH's σ≈0.90 demands ≤ 0.18, SOL's σ≈1.20 ≤ 0.10.
Single global Kelly → 3–15x over-leverage per asset → geometric wealth decay.
ETH lost $700 at 78% WR because sizing was structurally wrong.

Public API (back-compat shape preserved where possible; pair is now REQUIRED):
    bk = BayesianKelly()
    bk.update(won, pnl_pct, pair, regime="_global")
    bk.kelly_fraction(pair, regime="_global") -> float
    bk.win_probability(pair, regime="_global") -> float
    bk.per_pair_size(pair, regime, portfolio_value, confidence, ...) -> dict  # 7-step pipeline

    sizer = PositionSizer()
    sizer.calculate_stake_fraction(confidence, pair, regime="_global") -> float

7-step pipeline (E1, matches PHASE27_ALPHA.md §PRENSİP 0):
    1. Beta posterior       p = α/(α+β)
    2. Raw Kelly            f_raw = (b·p − q)/b
    3. Vol drag (Peters)    f_drag = f_raw − σ²/(2·b)
    4. Vol-of-vol shrink    f_vov  = f_drag · σ²/(σ² + var(σ))
    5. Baker-McHale         f_bm   = f_vov · (1 − σ_p²/(p·q))
    6. Trade graduation     f_grad = f_bm · {0.125, 0.25, 0.50, 0.75} by N
    7. Portfolio ENB        f_enb  = f_grad · min(1, ENB/n_pairs)
    Final cap: 3% per position (Constitution).
"""

import math
import os
import sys
import logging
import threading
from datetime import datetime, timezone
from typing import Optional, Dict, Any

sys.path.append(os.path.dirname(__file__))

from autonomy_manager import AutonomyManager

logger = logging.getLogger(__name__)

from ai_config import AI_DB_PATH as DB_PATH
from db import get_db_connection

try:
    from neural_organism import _p
except ImportError:
    def _p(param_id, fallback=0.5, regime="_global"):
        return fallback


# Decay applied on each update so old evidence fades (effective window ~50 trades).
# Identical constant to Phase 25 BayesianKelly; keeps behaviour after migration.
_DECAY = 0.98
_PRIOR_ALPHA = 2.0  # Jeffreys-leaning informative prior (matches db.py default)
_PRIOR_BETA = 2.0


REAL_KELLY_TABLE = "bayesian_kelly_per_pair"
SHADOW_KELLY_TABLE = "bayesian_kelly_shadow_per_pair"
# Revize Tur-2 (H7): whitelist for the table_name kwarg. `BayesianKelly`
# builds SQL via f-strings so any caller-supplied table name would turn
# into a direct SQL injection if we ever accepted arbitrary input.
ALLOWED_KELLY_TABLES = frozenset({REAL_KELLY_TABLE, SHADOW_KELLY_TABLE})


class BayesianKelly:
    """Per-pair Beta posterior. DB is source of truth (no in-memory drift).

    EK Sprint 2026-04-23: the table name is now parameterised so the shadow
    ledger (populated by `_forgone_learn_feedback`) can live in its own
    table and never contaminate live sizing. Use `get_real_kelly()` for
    production sizing and `get_shadow_kelly()` for analytics / parallel
    evaluation.
    """

    def __init__(self, db_path: str = DB_PATH, table_name: str = REAL_KELLY_TABLE):
        if table_name not in ALLOWED_KELLY_TABLES:
            raise ValueError(
                f"BayesianKelly: table_name={table_name!r} is not whitelisted "
                f"(allowed={sorted(ALLOWED_KELLY_TABLES)})"
            )
        self.db_path = db_path
        self.table_name = table_name
        # Primary schema owner is db.py (init_db at line ~479). We used to also
        # CREATE TABLE here with a DIFFERENT schema (CHECK(id=1)) which is what
        # caused the global-Kelly bug. The block below intentionally mirrors
        # db.py's schema EXACTLY so it is a no-op in prod (table already exists)
        # but bootstraps tests that use tmp_path without calling init_db().
        self._table_ready: bool = False

    # ─── Helpers ───────────────────────────────────────────────────────────

    def _conn(self):
        return get_db_connection(self.db_path)

    def _ensure_schema(self, conn) -> None:
        """Idempotent with db.py — mirrors the per-pair Beta posterior schema.
        Used for both the real and shadow ledgers; the table name is chosen
        by the constructor."""
        if self._table_ready:
            return
        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self.table_name} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pair TEXT NOT NULL,
                regime TEXT NOT NULL DEFAULT '_global',
                alpha REAL DEFAULT 2.0,
                beta_param REAL DEFAULT 2.0,
                avg_win REAL DEFAULT 0.0,
                avg_loss REAL DEFAULT 0.0,
                n_trades INTEGER DEFAULT 0,
                annual_volatility REAL,
                vol_of_vol REAL,
                last_sharpe REAL,
                updated_at TEXT,
                UNIQUE(pair, regime)
            )
            """
        )
        idx_name = f"idx_bk_{self.table_name}"
        conn.execute(
            f"CREATE INDEX IF NOT EXISTS {idx_name} "
            f"ON {self.table_name}(pair, regime)"
        )
        conn.commit()
        self._table_ready = True

    def _default_row(self) -> Dict[str, Any]:
        return {
            "alpha": _PRIOR_ALPHA,
            "beta_param": _PRIOR_BETA,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "n_trades": 0,
            "annual_volatility": None,
            "vol_of_vol": None,
            "last_sharpe": None,
        }

    def _load_pair(self, pair: str, regime: str = "_global") -> Dict[str, Any]:
        """Load row for (pair, regime). Returns informative prior if no row exists."""
        if not pair:
            raise ValueError("BayesianKelly: pair is required (Prensip 0).")
        with self._conn() as conn:
            self._ensure_schema(conn)
            row = conn.execute(
                f"SELECT alpha, beta_param, avg_win, avg_loss, n_trades, "
                f"annual_volatility, vol_of_vol, last_sharpe "
                f"FROM {self.table_name} WHERE pair = ? AND regime = ?",
                (pair, regime),
            ).fetchone()
        if row is None:
            return self._default_row()
        return {k: row[k] for k in row.keys()}

    # ─── Mutation ──────────────────────────────────────────────────────────

    def update(self, won: bool, pnl_pct: float = 0.0,
               pair: Optional[str] = None, regime: str = "_global") -> None:
        """Update per-pair Beta posterior with decay after a trade completes.

        DECAY=0.98 shrinks old evidence so α+β does not grow unbounded.
        Without decay, after 1000 trades α≈850 → new trades are invisible.
        """
        if not pair:
            raise ValueError("BayesianKelly.update: pair is required (Prensip 0).")

        stats = self._load_pair(pair, regime)

        alpha = max(1.0, float(stats["alpha"]) * _DECAY)
        beta_p = max(1.0, float(stats["beta_param"]) * _DECAY)
        avg_win = float(stats["avg_win"] or 0.0)
        avg_loss = float(stats["avg_loss"] or 0.0)
        n_trades = int(stats["n_trades"] or 0)

        # Mega Sprint 2026-04-23 (B.1.3): if β has run away (β/α > 20 after
        # 100+ trades) the slot has mathematically converged to p_win < 5%
        # — the sampler cannot recover on its own because every new loss
        # inflates β further. Snapping back to the Jeffreys-leaning prior
        # (2/2) gives the slot a chance to re-learn.
        if beta_p > 20 * alpha and n_trades > 100:
            logger.warning(
                f"[BayesianKelly:{pair}/{regime}] β/α>20 after {n_trades} trades "
                f"(α={alpha:.2f}, β={beta_p:.2f}) — auto-reset to Jeffreys prior"
            )
            alpha, beta_p = _PRIOR_ALPHA, _PRIOR_BETA

        if won:
            alpha += 1.0
            if pnl_pct > 0:
                avg_win = (avg_win * n_trades + abs(pnl_pct)) / (n_trades + 1)
        else:
            beta_p += 1.0
            if pnl_pct < 0:
                avg_loss = (avg_loss * n_trades + abs(pnl_pct)) / (n_trades + 1)

        n_trades += 1
        now = datetime.now(tz=timezone.utc).isoformat()

        with self._conn() as conn:
            self._ensure_schema(conn)
            conn.execute(
                f"""
                INSERT INTO {self.table_name}
                    (pair, regime, alpha, beta_param, avg_win, avg_loss,
                     n_trades, annual_volatility, vol_of_vol, last_sharpe, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(pair, regime) DO UPDATE SET
                    alpha = excluded.alpha,
                    beta_param = excluded.beta_param,
                    avg_win = excluded.avg_win,
                    avg_loss = excluded.avg_loss,
                    n_trades = excluded.n_trades,
                    updated_at = excluded.updated_at
                """,
                (
                    pair, regime, alpha, beta_p, avg_win, avg_loss, n_trades,
                    stats.get("annual_volatility"), stats.get("vol_of_vol"),
                    stats.get("last_sharpe"), now,
                ),
            )
            conn.commit()

    def hard_reset_all(self, reason: str = "sprint_cleanup") -> int:
        """Reset every `bayesian_kelly_per_pair` row back to the Jeffreys
        prior (α=β=2, n=0). Used after structural bugs (e.g. the 2026-04-23
        shadow label corruption) to evict the poisoned posterior in one
        shot rather than waiting for the decay to bleed it out naturally.
        Returns the number of rows reset.
        """
        with self._conn() as conn:
            self._ensure_schema(conn)
            cur = conn.execute(
                f"UPDATE {self.table_name} "
                "SET alpha = ?, beta_param = ?, n_trades = 0, "
                "    avg_win = 0.0, avg_loss = 0.0, "
                "    updated_at = ?",
                (_PRIOR_ALPHA, _PRIOR_BETA,
                 datetime.now(tz=timezone.utc).isoformat()),
            )
            conn.commit()
            n = cur.rowcount
        logger.warning(
            f"[BayesianKelly:{self.table_name}:hard_reset] {n} rows reset reason={reason}"
        )
        return n

    def set_pair_stats(self, pair: str, regime: str = "_global",
                       alpha: float = _PRIOR_ALPHA, beta_param: float = _PRIOR_BETA,
                       avg_win: float = 0.0, avg_loss: float = 0.0,
                       n_trades: int = 0,
                       annual_volatility: Optional[float] = None,
                       vol_of_vol: Optional[float] = None,
                       last_sharpe: Optional[float] = None) -> None:
        """Seed / overwrite a per-pair row. Used for tests and initial seeding
        from historical OHLCV (e.g. annual_volatility from chart_features)."""
        if not pair:
            raise ValueError("BayesianKelly.set_pair_stats: pair is required.")
        now = datetime.now(tz=timezone.utc).isoformat()
        with self._conn() as conn:
            self._ensure_schema(conn)
            conn.execute(
                f"""
                INSERT INTO {self.table_name}
                    (pair, regime, alpha, beta_param, avg_win, avg_loss,
                     n_trades, annual_volatility, vol_of_vol, last_sharpe, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(pair, regime) DO UPDATE SET
                    alpha = excluded.alpha,
                    beta_param = excluded.beta_param,
                    avg_win = excluded.avg_win,
                    avg_loss = excluded.avg_loss,
                    n_trades = excluded.n_trades,
                    annual_volatility = excluded.annual_volatility,
                    vol_of_vol = excluded.vol_of_vol,
                    last_sharpe = excluded.last_sharpe,
                    updated_at = excluded.updated_at
                """,
                (pair, regime, alpha, beta_param, avg_win, avg_loss, n_trades,
                 annual_volatility, vol_of_vol, last_sharpe, now),
            )
            conn.commit()

    # ─── Read (Kelly math) ─────────────────────────────────────────────────

    def win_probability(self, pair: Optional[str] = None,
                        regime: str = "_global") -> float:
        """Beta posterior mean: p = α/(α+β)."""
        if not pair:
            raise ValueError("BayesianKelly.win_probability: pair is required.")
        stats = self._load_pair(pair, regime)
        a = float(stats["alpha"])
        b = float(stats["beta_param"])
        return a / (a + b)

    def kelly_fraction(self, pair: Optional[str] = None,
                       regime: str = "_global") -> float:
        """Raw Kelly with sanity cap (0.25). Used by calculate_stake_fraction.

        For the full E1 7-step pipeline with vol drag, Baker-McHale, graduation,
        and ENB, call per_pair_size() instead.
        """
        if not pair:
            raise ValueError("BayesianKelly.kelly_fraction: pair is required.")
        stats = self._load_pair(pair, regime)
        a = float(stats["alpha"])
        b = float(stats["beta_param"])
        p = a / (a + b)
        q = 1.0 - p
        avg_win = float(stats["avg_win"] or 0.0)
        avg_loss = float(stats["avg_loss"] or 0.0)
        if avg_loss > 0:
            b_ratio = avg_win / avg_loss
        else:
            b_ratio = 1.5  # pre-trade default, keeps fraction positive for p > 0.4
        b_ratio = max(b_ratio, 0.01)
        f = (b_ratio * p - q) / b_ratio
        # RE-4 (2026-04-25): Kelly cap is now envelope-driven. L0 → 0.10,
        # L5 → 0.65, modulated by hormonal × decay. Static fallback to the
        # PARAM_REGISTRY value preserves backward compat when envelope module
        # is unavailable (e.g. early init or test env).
        try:
            from risk_envelope import get_risk_envelope
            cap = float(get_risk_envelope().get_kelly_cap())
        except Exception:
            cap = _p("sizing.kelly_cap", 0.25)
        return max(0.0, min(f, cap))

    # ─── E1 7-Step Per-Pair Pipeline ───────────────────────────────────────

    def per_pair_size(
        self,
        pair: str,
        regime: str = "_global",
        portfolio_value: float = 0.0,
        confidence: float = 1.0,
        sigma_annual: Optional[float] = None,
        vol_of_vol: Optional[float] = None,
        enb: Optional[float] = None,
        n_pairs: Optional[int] = None,
        max_position_fraction: float = 0.03,
    ) -> Dict[str, Any]:
        """Full 7-step pipeline from PHASE27_ALPHA.md §PRENSİP 0 / E1 ajani.

        External inputs (sigma, vol_of_vol, enb) fall back to the row's cached
        values, then to neutral defaults if still missing. Task 11 (Group 4)
        will wire up live sigma from chart_features; this version is safe to
        deploy before that — defaults give conservative sizing.
        """
        if not pair:
            raise ValueError("BayesianKelly.per_pair_size: pair is required.")

        stats = self._load_pair(pair, regime)
        alpha_i = float(stats["alpha"])
        beta_i = float(stats["beta_param"])

        # Step 1 — Beta posterior
        p_i = alpha_i / (alpha_i + beta_i)
        q_i = 1.0 - p_i

        # Step 2 — Raw Kelly
        avg_win = float(stats["avg_win"] or 0.0)
        avg_loss = float(stats["avg_loss"] or 0.0)
        if avg_loss > 0:
            b_i = abs(avg_win / avg_loss)
        else:
            b_i = 1.5
        b_i = max(b_i, 0.01)
        f_raw = max(0.0, (b_i * p_i - q_i) / b_i)

        # Step 3 — Vol drag (Peters ergodicity)
        if sigma_annual is None:
            cached = stats.get("annual_volatility")
            sigma_annual = float(cached) if cached is not None else 0.5
        drag = (sigma_annual ** 2) / 2.0
        f_drag = max(0.0, f_raw - drag / (b_i + 1e-8))

        # Step 4 — Vol-of-vol shrinkage
        if vol_of_vol is None:
            cached_vov = stats.get("vol_of_vol")
            vol_of_vol = float(cached_vov) if cached_vov is not None else 0.0
        if vol_of_vol > 0 and sigma_annual > 0:
            vov_shrink = (sigma_annual ** 2) / (sigma_annual ** 2 + vol_of_vol)
        else:
            vov_shrink = 1.0
        f_vov = f_drag * vov_shrink

        # Step 5 — Baker-McHale parameter uncertainty
        N = alpha_i + beta_i - (_PRIOR_ALPHA + _PRIOR_BETA)  # trades beyond prior
        if N > 0:
            sigma_p_sq = (p_i * q_i) / N
            bm_shrink = max(0.1, 1.0 - sigma_p_sq / (p_i * q_i + 1e-8))
        else:
            bm_shrink = 0.1
        f_bm = f_vov * bm_shrink

        # Step 6 — Trade count graduation
        if N < 30:
            grad = 0.125
        elif N < 100:
            grad = 0.25
        elif N < 300:
            grad = 0.50
        else:
            grad = 0.75
        f_grad = f_bm * grad

        # Step 7 — Portfolio ENB (Meucci effective number of bets)
        if enb is not None and n_pairs is not None and n_pairs > 0:
            portfolio_scale = min(1.0, enb / n_pairs)
        else:
            portfolio_scale = 1.0
        f_enb = f_grad * portfolio_scale

        # Confidence modulation and hard cap
        confidence = max(0.0, min(1.0, float(confidence)))
        fraction = min(f_enb * confidence, max_position_fraction)
        dollar_size = fraction * portfolio_value if portfolio_value > 0 else 0.0

        return {
            "fraction": round(fraction, 6),
            "dollar_size": round(dollar_size, 2),
            "breakdown": {
                "p": round(p_i, 4),
                "b": round(b_i, 4),
                "f_raw": round(f_raw, 6),
                "f_drag": round(f_drag, 6),
                "f_vov": round(f_vov, 6),
                "f_bm": round(f_bm, 6),
                "f_grad": round(f_grad, 6),
                "f_enb": round(f_enb, 6),
                "N": int(max(0, N)),
                "sigma_annual": round(sigma_annual, 4),
                "vol_of_vol": round(vol_of_vol, 6),
                "grad": grad,
                "portfolio_scale": round(portfolio_scale, 4),
            },
        }


class PositionSizer:
    """Translates AI confidence metrics into actionable per-pair Freqtrade stakes.

    Per-pair Kelly (via BayesianKelly.kelly_fraction(pair)) replaces the Phase
    3.5.2 global Kelly. `calculate_stake_fraction` is the light-weight path
    used inside custom_stake_amount; the full E1 pipeline lives on
    BayesianKelly.per_pair_size for Task 11 (CAAT sizing formula).
    """

    def __init__(self, max_portfolio_risk_per_trade: float = 0.05,
                 confidence_exponent: float = 1.0):
        self.max_risk = _p("sizing.max_risk", max_portfolio_risk_per_trade)
        self.exponent = _p("sizing.confidence_exponent", confidence_exponent)
        self.autonomy = AutonomyManager()
        self.bayesian_kelly = BayesianKelly()

    def _effective_max_risk(self, pair: str, regime: str = "_global") -> float:
        # RE-4 (2026-04-25): max_risk is now envelope-driven. The static
        # field `self.max_risk` survives as a hyperopt knob and as a hard
        # ceiling — envelope can only LOWER it, never raise above it.
        try:
            from risk_envelope import get_risk_envelope
            envelope_risk = float(get_risk_envelope().get_risk_per_trade())
            effective_max_risk = min(self.max_risk * 2.0, envelope_risk)  # envelope can go up to 2× hyperopt baseline
        except Exception:
            effective_max_risk = self.max_risk

        kelly_autonomy = self.autonomy.get_kelly_fraction()
        kelly_bayesian = self.bayesian_kelly.kelly_fraction(pair, regime)
        # Mega Sprint 2026-04-23 (B.2): Kelly floor lifted from 0.5% → 1.5%.
        floor = float(_p("sizing.kelly_floor_fraction", 0.015))
        return min(effective_max_risk, kelly_autonomy, max(kelly_bayesian, floor))

    def calculate_stake_fraction(self, confidence: float,
                                 pair: Optional[str] = None,
                                 regime: str = "_global",
                                 current_regime_modifier: float = 1.0) -> float:
        """Fraction of proposed stake to use for this trade.

        Args:
            confidence: LLM / model confidence in [0, 1].
            pair: Freqtrade pair string (e.g. 'BTC/USDT:USDT'). REQUIRED.
            regime: Market regime; defaults to '_global'.
            current_regime_modifier: External multiplier (trend bull/bear, etc.)

        Returns:
            Fraction in [min_fraction, max_risk * 1.5]. Typically 0.0005–0.075.
        """
        if not pair:
            raise ValueError(
                "PositionSizer.calculate_stake_fraction: pair is required "
                "(Prensip 0 — global Kelly bugu tekrar etmesin)."
            )

        trust_curve_multiplier = math.pow(max(confidence, 0.01), self.exponent)
        effective_risk = self._effective_max_risk(pair, regime)
        fraction = effective_risk * trust_curve_multiplier * current_regime_modifier

        min_fraction = effective_risk * _p("sizing.min_fraction_mult", 0.01)
        fraction = max(fraction, min_fraction)
        final_fraction = min(fraction, effective_risk * _p("sizing.max_fraction_mult", 1.5))

        return round(final_fraction, 4)

    def print_sizing_table(self, pair: str = "BTC/USDT:USDT"):
        """Diagnostic helper — prints the confidence → stake% curve for one pair."""
        logger.info(
            f"Position Sizing Curve (pair={pair}, Max Risk: {self.max_risk*100}%, "
            f"Exponent: {self.exponent})"
        )
        logger.info("Confidence | Stake %")
        logger.info("--------------------")
        for i in range(1, 11):
            conf = i / 10.0
            stake = self.calculate_stake_fraction(conf, pair=pair)
            logger.info(f"   {conf:.1f}     |  {stake*100:.2f}%")


_real_kelly_instance: Optional[BayesianKelly] = None
_shadow_kelly_instance: Optional[BayesianKelly] = None
# Revize Tur-2 (H8): serialise singleton construction so the first caller
# in a multi-threaded scheduler job does not race with the second and
# build two separate BayesianKelly objects pointing at the same DB.
_kelly_factory_lock = threading.Lock()


def get_real_kelly(db_path: str = DB_PATH) -> BayesianKelly:
    """Module-level singleton for the production per-pair Kelly ledger.
    This is the posterior that drives live sizing via PositionSizer."""
    global _real_kelly_instance
    if _real_kelly_instance is None or _real_kelly_instance.db_path != db_path:
        with _kelly_factory_lock:
            if _real_kelly_instance is None or _real_kelly_instance.db_path != db_path:
                _real_kelly_instance = BayesianKelly(
                    db_path=db_path, table_name=REAL_KELLY_TABLE
                )
    return _real_kelly_instance


def get_shadow_kelly(db_path: str = DB_PATH) -> BayesianKelly:
    """Module-level singleton for the shadow-only Kelly ledger.

    EK Sprint 2026-04-23: the shadow ledger is fed by `_forgone_learn_feedback`
    so we can compare the organism's theoretical p_win against the real one
    without the shadow stream poisoning live sizing. Sizing consumers MUST
    use `get_real_kelly()`; analytics / what-if views use this one.
    """
    global _shadow_kelly_instance
    if _shadow_kelly_instance is None or _shadow_kelly_instance.db_path != db_path:
        with _kelly_factory_lock:
            if _shadow_kelly_instance is None or _shadow_kelly_instance.db_path != db_path:
                _shadow_kelly_instance = BayesianKelly(
                    db_path=db_path, table_name=SHADOW_KELLY_TABLE
                )
    return _shadow_kelly_instance


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sizer = PositionSizer()
    sizer.print_sizing_table()
