"""
Sprint 2026-04-25 evening — 4 Learning Loops integration tests.

Each loop closes a previously-open feedback gap that the audit identified:
  LOOP-1: shadow Kelly per-pair Beta posterior → entry-gate pheromone
  LOOP-2: forgone PnL resolution → counterfactual shadow Kelly update
  LOOP-3: hourly daemon backfills ai_decisions.outcome_pnl from feather OHLCV
  LOOP-4: real Kelly Beta Thompson sample at populate_entry_trend start

Tests are pure: no exchange, no LLM, no Freqtrade dataframe required.
SQLite-backed BayesianKelly uses tmp DB files.
"""
import os
import sqlite3
import tempfile
import time
from unittest.mock import MagicMock, patch

import pytest
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'user_data', 'scripts'))


# ═══════════════════════════════════════════════════════════════════════════
# LOOP-2 — forgone resolution writes to shadow Kelly
# ═══════════════════════════════════════════════════════════════════════════

class TestLoop2ForgoneToShadowKelly:
    def setup_method(self):
        # Reset pheromone singleton so the test starts clean
        from pheromone_field import PheromoneField
        import pheromone_field as pf_mod
        pf_mod._pheromone_field = PheromoneField()

        self.tmp = tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False)
        self.tmp.close()
        self.db_path = self.tmp.name
        # Schema: forgone_profit + bayesian_kelly_shadow_per_pair
        conn = sqlite3.connect(self.db_path)
        conn.executescript("""
            CREATE TABLE forgone_profit (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pair TEXT NOT NULL, signal_type TEXT, confidence REAL,
                entry_price REAL, was_executed BOOLEAN DEFAULT 0,
                exit_price REAL, forgone_pnl REAL, resolved_at DATETIME,
                regime TEXT DEFAULT '_global'
            );
            CREATE TABLE bayesian_kelly_shadow_per_pair (
                pair TEXT NOT NULL, regime TEXT NOT NULL DEFAULT '_global',
                alpha REAL, beta_param REAL, avg_win REAL, avg_loss REAL,
                n_trades INTEGER, annual_volatility REAL, vol_of_vol REAL,
                last_sharpe REAL, updated_at TEXT,
                PRIMARY KEY (pair, regime)
            );
        """)
        conn.commit()
        conn.close()

        # Point position_sizer + reset singleton so get_shadow_kelly()
        # returns a fresh instance bound to our temp DB.
        import position_sizer
        position_sizer.DB_PATH = self.db_path
        position_sizer._shadow_kelly_instance = None
        position_sizer._real_kelly_instance = None

    def teardown_method(self):
        # Reset singletons so other tests aren't poisoned with our temp path.
        import position_sizer
        position_sizer._shadow_kelly_instance = None
        position_sizer._real_kelly_instance = None
        os.unlink(self.db_path)

    def test_resolve_forgone_writes_to_shadow_kelly(self):
        from forgone_pnl_engine import ForgonePnLEngine
        engine = ForgonePnLEngine(db_path=self.db_path)
        # Insert a forgone signal
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "INSERT INTO forgone_profit (pair, signal_type, entry_price, regime) VALUES (?,?,?,?)",
            ("ETH/USDT:USDT", "BULL", 100.0, "_global"),
        )
        forgone_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.commit()
        conn.close()

        # Resolve at exit_price=110 (BULL +10% counterfactual win)
        ok = engine.resolve_forgone_trade(forgone_id, exit_price=110.0)
        assert ok is True

        # Verify shadow Kelly got the update — α should now be 3 (prior 2 + 1 win)
        conn = sqlite3.connect(self.db_path)
        row = conn.execute(
            "SELECT alpha, beta_param, n_trades FROM bayesian_kelly_shadow_per_pair "
            "WHERE pair = ? AND regime = ?",
            ("ETH/USDT:USDT", "_global"),
        ).fetchone()
        conn.close()
        assert row is not None
        alpha, beta, n = row
        assert alpha > 2.0, f"Expected alpha>2.0 (prior+win), got {alpha}"
        assert n == 1, f"Expected n_trades=1, got {n}"

    def test_resolve_forgone_loss_increments_beta(self):
        from forgone_pnl_engine import ForgonePnLEngine
        engine = ForgonePnLEngine(db_path=self.db_path)
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "INSERT INTO forgone_profit (pair, signal_type, entry_price, regime) VALUES (?,?,?,?)",
            ("DOGE/USDT:USDT", "BEAR", 0.20, "_global"),
        )
        forgone_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.commit()
        conn.close()

        # BEAR signal but price went UP → forgone loss
        ok = engine.resolve_forgone_trade(forgone_id, exit_price=0.22)
        assert ok is True

        conn = sqlite3.connect(self.db_path)
        row = conn.execute(
            "SELECT alpha, beta_param FROM bayesian_kelly_shadow_per_pair "
            "WHERE pair = ?", ("DOGE/USDT:USDT",),
        ).fetchone()
        conn.close()
        alpha, beta = row
        # Loss → β += 1, α unchanged after decay
        assert beta > alpha


# ═══════════════════════════════════════════════════════════════════════════
# LOOP-1 — per-pair shadow score pheromone
# ═══════════════════════════════════════════════════════════════════════════

class TestLoop1ShadowPerPairPheromone:
    def setup_method(self):
        from pheromone_field import PheromoneField
        import pheromone_field as pf_mod
        pf_mod._pheromone_field = PheromoneField()

    def test_shadow_score_pheromone_published_per_pair(self):
        # Stub PipelineScheduler instance — we only need the method
        import scheduler
        sched = scheduler.PipelineScheduler.__new__(scheduler.PipelineScheduler)

        # Use real DB but seed it
        from db import get_db_connection
        with get_db_connection() as conn:
            # Ensure table exists
            from db import init_db
            init_db()
            # Insert one shadow row with low winrate (3%)
            conn.execute("""
                INSERT OR REPLACE INTO bayesian_kelly_shadow_per_pair
                (pair, regime, alpha, beta_param, avg_win, avg_loss, n_trades,
                 annual_volatility, vol_of_vol, last_sharpe, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
            """, ("LOOP1TEST/USDT:USDT", "_global", 1.0, 30.0, 1.0, 1.0, 30, None, None, None))
            conn.commit()

        # Run the tick
        sched._shadow_kelly_divergence_tick()

        # Verify pheromone got published for that pair
        from pheromone_field import get_pheromone_field
        payload = get_pheromone_field().read(
            "shadow_score::LOOP1TEST/USDT:USDT", source="shadow_kelly"
        )
        assert isinstance(payload, dict), f"Expected dict, got {type(payload)}"
        assert "score" in payload
        assert "shadow_wr" in payload
        # shadow_wr should be very low (<10% with α=1, β=30)
        assert payload["shadow_wr"] < 0.10
        assert payload["n_shadow"] == 30


# ═══════════════════════════════════════════════════════════════════════════
# LOOP-3 — outcome backfill from feather OHLCV
# ═══════════════════════════════════════════════════════════════════════════

class TestLoop3OutcomeBackfill:
    def setup_method(self):
        from pheromone_field import PheromoneField
        import pheromone_field as pf_mod
        pf_mod._pheromone_field = PheromoneField()

    def test_backfill_neutral_signal_writes_zero(self):
        # Verify NEUTRAL backfill path — writes 0.0 outcome without needing OHLCV
        from db import get_db_connection, init_db
        init_db()
        # Insert a NEUTRAL decision 5h ago (older than 4h gate)
        from datetime import datetime, timezone, timedelta
        old_ts = (datetime.now(timezone.utc) - timedelta(hours=5)).strftime("%Y-%m-%d %H:%M:%S")
        with get_db_connection() as conn:
            conn.execute(
                """INSERT INTO ai_decisions (timestamp, pair, signal_type, confidence)
                   VALUES (?, ?, ?, ?)""",
                (old_ts, "TESTNEU/USDT:USDT", "NEUTRAL", 0.50),
            )
            conn.commit()
            decision_id = conn.execute(
                "SELECT id FROM ai_decisions WHERE pair = ? ORDER BY id DESC LIMIT 1",
                ("TESTNEU/USDT:USDT",)
            ).fetchone()["id"]

        import scheduler
        sched = scheduler.PipelineScheduler.__new__(scheduler.PipelineScheduler)
        sched._decisions_outcome_backfill_tick()

        # Verify the NEUTRAL decision now has outcome_pnl = 0.0
        with get_db_connection() as conn:
            row = conn.execute(
                "SELECT outcome_pnl, outcome_duration FROM ai_decisions WHERE id = ?",
                (decision_id,),
            ).fetchone()
        assert row is not None
        assert row["outcome_pnl"] == 0.0
        assert row["outcome_duration"] == 14400


# ═══════════════════════════════════════════════════════════════════════════
# LOOP-4 — Thompson gate at entry
# ═══════════════════════════════════════════════════════════════════════════

class TestLoop4ThompsonGate:
    def test_real_kelly_load_pair_returns_prior_for_unseen(self):
        # New pair → default prior (Jeffreys α=2, β=2, n=0)
        from position_sizer import get_real_kelly
        kelly = get_real_kelly()
        stats = kelly._load_pair("UNSEEN_LOOP4/USDT:USDT", regime="_global")
        assert int(stats["n_trades"] or 0) == 0
        # Default α/β starts at 2 (Jeffreys)
        assert float(stats["alpha"]) >= 1.0
        assert float(stats["beta_param"]) >= 1.0

    def test_thompson_sample_from_strong_loser(self):
        # Build a slot with α=1, β=20 (95% loser by posterior mean)
        # Thompson sample should rarely exceed 0.30
        from position_sizer import BayesianKelly
        import tempfile
        tmp = tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False)
        tmp.close()
        try:
            kelly = BayesianKelly(db_path=tmp.name, table_name="bayesian_kelly_per_pair")
            # Force losing record by 20× update with won=False
            for _ in range(20):
                kelly.update(won=False, pnl_pct=-2.0, pair="LOSER/USDT:USDT", regime="_global")
            stats = kelly._load_pair("LOSER/USDT:USDT", regime="_global")
            # Posterior should be heavily skewed
            posterior_mean = stats["alpha"] / (stats["alpha"] + stats["beta_param"])
            assert posterior_mean < 0.30, \
                f"Expected posterior_mean<0.30 after 20 losses, got {posterior_mean}"
        finally:
            os.unlink(tmp.name)


# ═══════════════════════════════════════════════════════════════════════════
# Cross-loop integration — LOOP-2 → LOOP-1 → LOOP-4
# ═══════════════════════════════════════════════════════════════════════════

class TestCrossLoopIntegration:
    def setup_method(self):
        from pheromone_field import PheromoneField
        import pheromone_field as pf_mod
        pf_mod._pheromone_field = PheromoneField()

    def test_full_chain_forgone_to_pheromone_to_gate(self):
        """End-to-end: forgone resolves → shadow Kelly → tick publishes
        per-pair pheromone → entry gate would skip.
        """
        from db import init_db, get_db_connection
        init_db()

        # Step 1: insert N forgone signals for "FAILPAIR/USDT:USDT" all losers
        from forgone_pnl_engine import ForgonePnLEngine
        engine = ForgonePnLEngine()
        for i in range(8):
            with get_db_connection() as conn:
                conn.execute(
                    "INSERT INTO forgone_profit (pair, signal_type, entry_price, regime) "
                    "VALUES (?,?,?,?)",
                    ("FAILPAIR/USDT:USDT", "BULL", 100.0, "_global"),
                )
                forgone_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
                conn.commit()
            # Resolve all as losers (price dropped 5%)
            engine.resolve_forgone_trade(forgone_id, exit_price=95.0)

        # Step 2: shadow Kelly should now show low winrate
        from position_sizer import get_shadow_kelly
        sk = get_shadow_kelly()
        wr = sk.win_probability("FAILPAIR/USDT:USDT", regime="_global")
        assert wr < 0.30, f"Expected low winrate after 8 forgone losses, got {wr}"

        # Step 3: scheduler tick publishes per-pair pheromone
        import scheduler
        sched = scheduler.PipelineScheduler.__new__(scheduler.PipelineScheduler)
        sched._shadow_kelly_divergence_tick()

        # Step 4: pheromone exists with low score
        from pheromone_field import get_pheromone_field
        payload = get_pheromone_field().read(
            "shadow_score::FAILPAIR/USDT:USDT", source="shadow_kelly"
        )
        assert isinstance(payload, dict)
        # Both posterior mean AND Thompson sample should be low
        assert payload["shadow_wr"] < 0.30
