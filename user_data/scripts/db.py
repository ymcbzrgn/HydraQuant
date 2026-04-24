"""
db.py v2 — Phase 28: Centralized Database Layer

Merkezi SQLite baglanti yonetimi. Tum AI modulleri bu dosyadaki
get_connection() veya get_db_connection() fonksiyonunu kullanir.
Connection pooling, tutarli timeout, retry logic ve tum tablo
tanimlari burada.
"""

import sqlite3
import os
import json
import logging
import threading
import time
from datetime import datetime
from contextlib import contextmanager

logger = logging.getLogger(__name__)

from ai_config import AI_DB_PATH as DB_PATH

# ---------------------------------------------------------------------------
# Connection Pool — Thread-safe Singleton
# ---------------------------------------------------------------------------

class _ConnectionPool:
    """Thread-safe SQLite connection pool.
    Her thread kendi connection'ini alir, isini bitirince pool'a geri verir.
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._pool: list[sqlite3.Connection] = []
        self._pool_lock = threading.Lock()
        # Memory-pressure revision (2026-04-21): the old max=8 × 8MB page cache
        # pinned >64 MB of SQLite private cache alone. Halved the pool and
        # page cache to shave both RSS and WAL-pinning pressure.
        self._max_size = 4
        self._busy_timeout_ms = 30000
        self._active_count = 0
        self._total_created = 0
        self._release_count = 0
        self._initialized = True
        logger.info(f"[DB] Connection pool initialized (max={self._max_size}, timeout={self._busy_timeout_ms}ms)")

    def _create_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(DB_PATH, timeout=self._busy_timeout_ms / 1000, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(f"PRAGMA busy_timeout={self._busy_timeout_ms}")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA cache_size=-2000")
        conn.execute("PRAGMA mmap_size=0")
        self._total_created += 1
        # Wrap so conn.close() returns to pool instead of destroying
        return _PooledConnection(conn, self)

    def acquire(self) -> sqlite3.Connection:
        with self._pool_lock:
            if self._pool:
                conn = self._pool.pop()
                self._active_count += 1
                return conn
        with self._pool_lock:
            self._active_count += 1
        return self._create_connection()

    def release(self, conn):
        with self._pool_lock:
            self._active_count -= 1
            self._release_count += 1
            should_truncate = self._release_count % 50 == 0
            # Checkpoint BEFORE returning to pool so no sibling thread grabs
            # this connection mid-pragma. WAL grew to 166 MB uncheckpointed in
            # production before this; TRUNCATE bounds it without hot-path lock
            # contention (one call every 50 releases).
            if should_truncate:
                try:
                    raw = conn._conn if isinstance(conn, _PooledConnection) else conn
                    raw.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                except Exception:
                    pass
            if len(self._pool) < self._max_size:
                self._pool.append(conn)
            else:
                try:
                    # Actually close the underlying connection
                    raw = conn._conn if isinstance(conn, _PooledConnection) else conn
                    raw.close()
                except Exception:
                    pass

    def close_all(self):
        with self._pool_lock:
            for conn in self._pool:
                try:
                    # Close raw connection directly to avoid deadlock
                    # (conn.close() would call release() which re-acquires _pool_lock)
                    raw = conn._conn if isinstance(conn, _PooledConnection) else conn
                    raw.close()
                except Exception:
                    pass
            self._pool.clear()
            logger.info(f"[DB] Pool closed. Total created: {self._total_created}")

    @property
    def stats(self) -> dict:
        with self._pool_lock:
            return {
                "pool_size": len(self._pool),
                "active": self._active_count,
                "total_created": self._total_created,
                "max_size": self._max_size,
            }


class _PooledConnection:
    """Wrapper: conn.close() returns connection to pool instead of destroying it.
    Includes automatic retry on 'database is locked' for commit/execute operations.
    """

    _RETRY_MAX = 5
    _RETRY_BASE_WAIT = 0.3  # seconds

    def __init__(self, conn: sqlite3.Connection, pool: '_ConnectionPool'):
        self._conn = conn
        self._pool = pool

    def _retry_on_locked(self, func, *args, **kwargs):
        """Retry a database operation on 'database is locked' errors."""
        for attempt in range(self._RETRY_MAX):
            try:
                return func(*args, **kwargs)
            except sqlite3.OperationalError as e:
                if "database is locked" in str(e) and attempt < self._RETRY_MAX - 1:
                    wait = self._RETRY_BASE_WAIT * (attempt + 1)
                    logger.warning(f"[DB] database is locked, retry {attempt+1}/{self._RETRY_MAX} in {wait:.1f}s")
                    time.sleep(wait)
                else:
                    raise

    def __getattr__(self, name):
        return getattr(self._conn, name)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            self._conn.rollback()
        else:
            self._retry_on_locked(self._conn.commit)

    def close(self):
        """Return to pool instead of destroying."""
        self._pool.release(self)

    def execute(self, *args, **kwargs):
        return self._retry_on_locked(self._conn.execute, *args, **kwargs)

    def executemany(self, *args, **kwargs):
        return self._retry_on_locked(self._conn.executemany, *args, **kwargs)

    def cursor(self):
        return self._conn.cursor()

    def commit(self):
        return self._retry_on_locked(self._conn.commit)

    def rollback(self):
        return self._conn.rollback()

    @property
    def row_factory(self):
        return self._conn.row_factory

    @row_factory.setter
    def row_factory(self, value):
        self._conn.row_factory = value


_pool = _ConnectionPool()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_db_connection(db_path: str = None) -> sqlite3.Connection:
    """Returns a pooled connection to AI DB, or a direct connection to a custom path.

    If db_path is None or matches AI_DB_PATH → pool'dan verir (PooledConnection).
    If db_path is different (e.g. test tmp_path) → direct connection (no pool).

    IMPORTANT: Pooled connections auto-return on close(). Direct connections truly close.
    """
    if db_path is None or db_path == DB_PATH:
        return _pool.acquire()
    # Custom path — direct connection (for tests with tmp_path)
    conn = sqlite3.connect(db_path, timeout=30, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
    return conn


def release_connection(conn: sqlite3.Connection):
    """Return connection to pool."""
    _pool.release(conn)


@contextmanager
def get_connection():
    """Context manager: auto-release connection back to pool.

    Usage:
        with get_connection() as conn:
            conn.execute("SELECT ...")
            conn.commit()
    """
    conn = _pool.acquire()
    try:
        yield conn
    except Exception:
        try:
            conn.rollback()
        except Exception:
            pass
        raise
    finally:
        _pool.release(conn)


def execute_with_retry(sql: str, params=None, max_retries: int = 3, commit: bool = True, db_path: str = None):
    """Execute SQL with retry on 'database is locked'.

    If db_path is provided, uses get_db_connection(db_path) (respects the caller's
    DB path — needed for tests that use tmp_path, and for any module that persists
    its own data to a scoped DB). Otherwise uses the default pooled connection.
    """
    import contextlib
    last_error = None
    for attempt in range(max_retries):
        try:
            conn_ctx = get_db_connection(db_path) if db_path else get_connection()
            with conn_ctx as conn:
                cursor = conn.execute(sql, params or ())
                if commit:
                    conn.commit()
                return cursor
        except sqlite3.OperationalError as e:
            if "database is locked" in str(e) and attempt < max_retries - 1:
                wait = 0.5 * (attempt + 1)
                logger.warning(f"[DB] database is locked, retry {attempt+1}/{max_retries} in {wait}s")
                time.sleep(wait)
                last_error = e
            else:
                raise
    raise last_error


def get_pool_stats() -> dict:
    return _pool.stats


# ---------------------------------------------------------------------------
# Schema — ALL tables defined here
# ---------------------------------------------------------------------------

def init_db():
    """Create all AI pipeline tables. Idempotent."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

    with get_connection() as conn:
        c = conn.cursor()

        # === PHASE 18-19: Core ===
        c.execute('''CREATE TABLE IF NOT EXISTS market_news (
            id INTEGER PRIMARY KEY AUTOINCREMENT, source TEXT NOT NULL,
            title TEXT NOT NULL, summary TEXT, url TEXT UNIQUE,
            published_at DATETIME, sentiment_score REAL, raw_data TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            title_hash TEXT, is_embedded BOOLEAN DEFAULT 0)''')

        c.execute('''CREATE TABLE IF NOT EXISTS fear_and_greed (
            id INTEGER PRIMARY KEY AUTOINCREMENT, value INTEGER NOT NULL,
            classification TEXT, timestamp DATETIME UNIQUE,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP)''')

        c.execute('''CREATE TABLE IF NOT EXISTS ai_decisions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP, pair TEXT NOT NULL,
            signal_type TEXT, confidence REAL, position_size REAL, entry_price REAL,
            model_used TEXT, rag_context_ids TEXT, reasoning_summary TEXT,
            regime TEXT, trust_score_at_decision REAL, outcome_pnl REAL,
            outcome_duration INTEGER,
            agent_votes_json TEXT)''')

        c.execute('''CREATE TABLE IF NOT EXISTS forgone_profit (
            id INTEGER PRIMARY KEY AUTOINCREMENT, pair TEXT NOT NULL,
            signal_type TEXT, signal_time DATETIME DEFAULT CURRENT_TIMESTAMP,
            confidence REAL, entry_price REAL, was_executed BOOLEAN DEFAULT 0,
            exit_price REAL, forgone_pnl REAL, resolved_at DATETIME)''')

        c.execute('''CREATE TABLE IF NOT EXISTS embedding_cache (
            text_hash TEXT PRIMARY KEY, text_content TEXT NOT NULL,
            gemini_embedding BLOB, bge_embedding BLOB,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP)''')

        c.execute('''CREATE VIRTUAL TABLE IF NOT EXISTS bm25_index USING fts5(
            doc_id UNINDEXED, content, tokenize = 'porter unicode61')''')

        c.execute('''CREATE TABLE IF NOT EXISTS portfolio_state (
            id INTEGER PRIMARY KEY CHECK (id = 1), stake_currency TEXT DEFAULT 'USDT',
            total_balance REAL DEFAULT 0.0, free_balance REAL DEFAULT 0.0,
            in_trades REAL DEFAULT 0.0, assets_json TEXT DEFAULT '{}', updated_at TEXT)''')

        c.execute('''CREATE TABLE IF NOT EXISTS hypothetical_portfolio (
            id INTEGER PRIMARY KEY AUTOINCREMENT, trade_pair TEXT NOT NULL,
            trade_pnl_pct REAL NOT NULL, balance_before REAL NOT NULL,
            balance_after REAL NOT NULL, trade_closed_at DATETIME DEFAULT CURRENT_TIMESTAMP)''')

        c.execute('''CREATE TABLE IF NOT EXISTS coin_sentiment_rolling (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP, coin TEXT NOT NULL,
            sentiment_1h REAL DEFAULT 0, sentiment_4h REAL DEFAULT 0,
            sentiment_24h REAL DEFAULT 0, news_count_24h INTEGER DEFAULT 0)''')

        c.execute('''CREATE TABLE IF NOT EXISTS signal_health (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP, pair TEXT NOT NULL,
            signal_source TEXT NOT NULL, signal_type TEXT, confidence REAL, latency_ms REAL)''')

        # === PHASE 19: Market Data ===
        c.execute('''CREATE TABLE IF NOT EXISTS derivatives_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT, pair TEXT NOT NULL,
            open_interest_usd REAL, funding_rate REAL, long_short_ratio REAL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')

        c.execute('''CREATE TABLE IF NOT EXISTS macro_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT, metric_name TEXT NOT NULL,
            value REAL NOT NULL, prev_value REAL, change_pct REAL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')

        c.execute('''CREATE TABLE IF NOT EXISTS defi_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT, metric_name TEXT NOT NULL,
            value REAL NOT NULL, change_pct REAL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')

        c.execute('''CREATE TABLE IF NOT EXISTS ohlcv_patterns (
            id INTEGER PRIMARY KEY AUTOINCREMENT, pair TEXT NOT NULL,
            timeframe TEXT DEFAULT '1h', timestamp TEXT, fingerprint TEXT NOT NULL,
            outcome_1h REAL, outcome_4h REAL, outcome_24h REAL, direction TEXT,
            indicators_json TEXT, created_at DATETIME DEFAULT CURRENT_TIMESTAMP)''')

        c.execute('''CREATE TABLE IF NOT EXISTS search_trends (
            id INTEGER PRIMARY KEY AUTOINCREMENT, keyword TEXT NOT NULL,
            interest_score INTEGER, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')

        # === PHASE 20: Agent Pool + Evidence Engine ===
        c.execute('''CREATE TABLE IF NOT EXISTS agent_memory (
            id INTEGER PRIMARY KEY AUTOINCREMENT, agent_type TEXT NOT NULL,
            pair TEXT NOT NULL, regime TEXT, signal TEXT NOT NULL, strength REAL,
            key_argument TEXT, evidence_engine_confidence REAL,
            final_outcome_pnl REAL, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')

        c.execute('''CREATE TABLE IF NOT EXISTS agent_performance (
            id INTEGER PRIMARY KEY AUTOINCREMENT, agent_type TEXT NOT NULL,
            pair TEXT NOT NULL, regime TEXT, signal TEXT NOT NULL,
            outcome_pnl REAL, was_correct BOOLEAN,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')

        c.execute('''CREATE TABLE IF NOT EXISTS opportunity_scores (
            id INTEGER PRIMARY KEY AUTOINCREMENT, pair TEXT NOT NULL,
            composite_score REAL NOT NULL, top_type TEXT, momentum_score REAL,
            reversion_score REAL, funding_score REAL, regime_shift_score REAL,
            volume_anomaly_score REAL, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')

        c.execute('''CREATE TABLE IF NOT EXISTS evidence_audit_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT, pair TEXT NOT NULL,
            signal TEXT NOT NULL, confidence REAL NOT NULL, sub_scores_json TEXT,
            contradictions_json TEXT, evidence_sources_json TEXT, regime TEXT,
            max_confidence_cap REAL,
            atr_ratio_at_entry REAL,
            volume_z REAL,
            funding_rate REAL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')

        c.execute('''CREATE TABLE IF NOT EXISTS cross_pair_cache (
            id INTEGER PRIMARY KEY CHECK (id = 1), data_json TEXT, timestamp TEXT)''')

        # === NEURAL ORGANISM (Phase 24-25) ===
        c.execute('''CREATE TABLE IF NOT EXISTS neuron_state (
            param_id TEXT NOT NULL, organ TEXT NOT NULL,
            regime TEXT NOT NULL DEFAULT '_global',
            current_val REAL NOT NULL, default_val REAL NOT NULL,
            min_bound REAL NOT NULL, max_bound REAL NOT NULL,
            alpha REAL DEFAULT 2.0, beta_param REAL DEFAULT 2.0,
            prior_strength REAL DEFAULT 5.0, frozen INTEGER DEFAULT 0,
            update_count INTEGER DEFAULT 0,
            activity_ema REAL DEFAULT 0.0, theta_m REAL DEFAULT 0.0,
            last_updated TEXT, PRIMARY KEY (param_id, regime))''')

        c.execute('''CREATE TABLE IF NOT EXISTS neuron_synapses (
            source TEXT NOT NULL, target TEXT NOT NULL,
            weight REAL DEFAULT 0.5, synapse_type TEXT DEFAULT 'excitatory',
            fire_count INTEGER DEFAULT 0, PRIMARY KEY (source, target))''')

        c.execute('''CREATE TABLE IF NOT EXISTS hormone_state (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            cortisol REAL DEFAULT 1.0, dopamine REAL DEFAULT 1.0,
            serotonin REAL DEFAULT 1.0, adrenaline REAL DEFAULT 1.0,
            market_stress REAL DEFAULT 0.0, portfolio_health REAL DEFAULT 0.5,
            info_quality REAL DEFAULT 0.5, updated_at TEXT,
            trough_cortisol REAL DEFAULT 1.0, trough_cortisol_time TEXT)''')
        # Phase 27 Fix 7: idempotent ALTER for existing deployments where hormone_state
        # predates the hysteresis columns. SQLite errors on duplicate ADD COLUMN; catch.
        for _sql in (
            "ALTER TABLE hormone_state ADD COLUMN trough_cortisol REAL DEFAULT 1.0",
            "ALTER TABLE hormone_state ADD COLUMN trough_cortisol_time TEXT",
        ):
            try:
                c.execute(_sql)
            except sqlite3.OperationalError:
                pass  # column already exists

        c.execute('''CREATE TABLE IF NOT EXISTS hippocampus_episodes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pair TEXT, fingerprint TEXT NOT NULL,
            outcome_pnl REAL NOT NULL, regime TEXT, timestamp TEXT)''')

        c.execute('''CREATE TABLE IF NOT EXISTS amygdala_state (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            fear_level REAL DEFAULT 0.0, peak_fear REAL DEFAULT 0.0,
            peak_time TEXT, tier TEXT DEFAULT 'normal', updated_at TEXT)''')

        c.execute('''CREATE TABLE IF NOT EXISTS immune_memory (
            pair TEXT NOT NULL, loss_pct REAL NOT NULL,
            ban_until TEXT, consecutive_losses INTEGER DEFAULT 1,
            regime TEXT, timestamp TEXT)''')

        c.execute('''CREATE TABLE IF NOT EXISTS organism_audit (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            trade_pair TEXT, trade_pnl REAL, hormones TEXT,
            fear_tier TEXT, overrides TEXT, phase TEXT, timestamp TEXT,
            event_type TEXT, details_json TEXT)''')

        c.execute('''CREATE TABLE IF NOT EXISTS cerebellum_hours (
            hour INTEGER PRIMARY KEY, wins INTEGER DEFAULT 1,
            losses INTEGER DEFAULT 1, avg_pnl REAL DEFAULT 0.0, updated_at TEXT)''')

        c.execute('''CREATE TABLE IF NOT EXISTS interoception_state (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            param_drift REAL DEFAULT 0.0, belief_width REAL DEFAULT 0.5,
            pred_error REAL DEFAULT 0.5, hormone_stability REAL DEFAULT 1.0,
            trade_freq REAL DEFAULT 0.0, win_rate REAL DEFAULT 0.5,
            data_completeness REAL DEFAULT 0.5, consec_dir INTEGER DEFAULT 0,
            updated_at TEXT)''')

        c.execute('''CREATE TABLE IF NOT EXISTS immune_bcells (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            threat_fingerprint TEXT NOT NULL UNIQUE,
            severity REAL DEFAULT 0.0, encounter_count INTEGER DEFAULT 1,
            last_encounter TEXT, antibody_strength REAL DEFAULT 1.0)''')

        c.execute('''CREATE TABLE IF NOT EXISTS evolution_population (
            genome_id INTEGER PRIMARY KEY AUTOINCREMENT,
            params_json TEXT NOT NULL, fitness REAL DEFAULT 0.0,
            novelty_score REAL DEFAULT 0.0, generation INTEGER DEFAULT 0,
            created_at TEXT, is_active INTEGER DEFAULT 0)''')

        c.execute('''CREATE TABLE IF NOT EXISTS sleep_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_date TEXT, episodes_replayed INTEGER,
            synapses_pruned INTEGER, habits_broken INTEGER,
            counterfactuals TEXT, duration_sec REAL, timestamp TEXT)''')

        c.execute('''CREATE TABLE IF NOT EXISTS dmn_discoveries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            discovery_type TEXT, description TEXT,
            param_ids TEXT, potential_improvement REAL, timestamp TEXT)''')

        # === OTHER MODULE TABLES ===
        c.execute('''CREATE TABLE IF NOT EXISTS hot_buffer (
            doc_id TEXT PRIMARY KEY, content TEXT, embedding BLOB,
            metadata TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')

        c.execute('''CREATE TABLE IF NOT EXISTS semantic_cache (
            id INTEGER PRIMARY KEY AUTOINCREMENT, query_text TEXT NOT NULL,
            query_embedding BLOB NOT NULL, response TEXT NOT NULL, pair TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            ttl_seconds INTEGER DEFAULT 300)''')

        c.execute('''CREATE TABLE IF NOT EXISTS magma_edges (
            id INTEGER PRIMARY KEY AUTOINCREMENT, graph_type TEXT NOT NULL,
            source TEXT NOT NULL, relation TEXT NOT NULL, target TEXT NOT NULL,
            weight REAL DEFAULT 1.0, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            metadata TEXT DEFAULT '{}',
            UNIQUE(graph_type, source, relation, target))''')

        c.execute('''CREATE TABLE IF NOT EXISTS bayesian_kelly (
            id INTEGER PRIMARY KEY AUTOINCREMENT, pair TEXT, regime TEXT,
            alpha REAL DEFAULT 1.0, beta_param REAL DEFAULT 1.0,
            kelly_fraction REAL DEFAULT 0.01, trade_count INTEGER DEFAULT 0,
            updated_at TEXT, UNIQUE(pair, regime))''')

        # Phase 27 Task 1 (E1 Ajani): Per-pair Bayesian Kelly — Prensip 0.
        # Replaces legacy single-row bayesian_kelly. Every (pair, regime) has
        # its own Beta posterior + vol drag + vol-of-vol inputs for the
        # 7-step sizing pipeline in position_sizer.py.
        c.execute('''CREATE TABLE IF NOT EXISTS bayesian_kelly_per_pair (
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
            UNIQUE(pair, regime))''')
        c.execute('''CREATE INDEX IF NOT EXISTS idx_bk_per_pair
            ON bayesian_kelly_per_pair(pair, regime)''')

        # EK Sprint 2026-04-23: shadow ledger. Same schema as the real one
        # but only updated by the forgone-feedback scheduler job. Live
        # sizing never reads this — it's a parallel analytics ledger.
        c.execute('''CREATE TABLE IF NOT EXISTS bayesian_kelly_shadow_per_pair (
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
            UNIQUE(pair, regime))''')
        c.execute('''CREATE INDEX IF NOT EXISTS idx_bk_shadow_per_pair
            ON bayesian_kelly_shadow_per_pair(pair, regime)''')

        # Phase 27 Fix 2C (J4 Ajani): Argument quality scoring — each agent's
        # argument patterns (regex-bucketed) get win-rate + avg PnL tracked so
        # R1 prompts can inject "your best argument in this regime was X (78% acc)".
        c.execute('''CREATE TABLE IF NOT EXISTS argument_quality (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            agent_type TEXT NOT NULL,
            argument_pattern TEXT NOT NULL,
            regime TEXT NOT NULL,
            times_used INTEGER DEFAULT 0,
            times_correct INTEGER DEFAULT 0,
            avg_pnl_when_used REAL DEFAULT 0.0,
            quality_score REAL DEFAULT 0.5,
            updated_at TEXT,
            UNIQUE(agent_type, argument_pattern, regime))''')
        c.execute('''CREATE INDEX IF NOT EXISTS idx_arg_quality
            ON argument_quality(agent_type, regime)''')

        # Phase 27 Fix 6 (H3 Ajani): per-pair / per-regime confidence thresholds.
        # Forgone alpha feedback drives these up/down so the bot stops missing
        # profitable regimes and stops entering unprofitable ones.
        c.execute('''CREATE TABLE IF NOT EXISTS pair_thresholds (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pair TEXT NOT NULL,
            regime TEXT NOT NULL DEFAULT '_global',
            confidence_threshold REAL DEFAULT 0.50,
            forgone_alpha_7d REAL DEFAULT 0.0,
            last_adjusted TEXT,
            adjustment_reason TEXT,
            UNIQUE(pair, regime))''')
        c.execute('''CREATE INDEX IF NOT EXISTS idx_pair_thr
            ON pair_thresholds(pair, regime)''')

        # Phase 27 Task 12 (B5 Ajani): 4-Layer regime detection snapshot.
        # ADX is demoted from primary signal to confirmation (Layer 3); the
        # earlier layers (VPIN → BOCPD → Causal Edge Instability) have longer
        # lead time and flag regime shifts BEFORE ADX even moves.
        c.execute('''CREATE TABLE IF NOT EXISTS regime_layers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pair TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            layer0_vpin REAL, layer0_alert INTEGER,
            layer1_bocpd_residual REAL, layer1_alert INTEGER,
            layer2_causal_instability REAL, layer2_alert INTEGER,
            layer3_adx_regime TEXT,
            regime_change_prob REAL,
            sizing_modifier REAL,
            status TEXT,
            UNIQUE(pair, timestamp))''')
        c.execute('''CREATE INDEX IF NOT EXISTS idx_regime_layers_pair
            ON regime_layers(pair, timestamp)''')

        # Phase 27 Task 15 (C4 Ajani): RLAIF reward history — 5-dim rubric
        # scores from 3 LLM judges (Gemini / Groq / Mistral) fused via WCO so
        # a single sycophant judge cannot dominate.
        c.execute('''CREATE TABLE IF NOT EXISTS rlaif_rewards (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            trade_id INTEGER,
            pair TEXT,
            timestamp TEXT,
            signal_quality REAL,
            sizing_quality REAL,
            timing_quality REAL,
            risk_management REAL,
            regime_alignment REAL,
            composite REAL,
            provider_scores TEXT,
            env_reward REAL,
            total_reward REAL,
            outcome_pnl REAL)''')
        c.execute('''CREATE INDEX IF NOT EXISTS idx_rlaif_trade
            ON rlaif_rewards(trade_id)''')

        # Phase 27 Task 16 (G2 Ajani): Weekly LLM hypothesis history —
        # research-loop candidates with 6-gate validation trail.
        c.execute('''CREATE TABLE IF NOT EXISTS hypothesis_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            hypothesis_id TEXT UNIQUE,
            parameter TEXT,
            current_value REAL,
            proposed_value REAL,
            mechanism TEXT,
            falsification TEXT,
            affected_pairs TEXT,
            is_sharpe REAL,
            oos_sharpe REAL,
            deflated_sharpe REAL,
            n_hypotheses_in_batch INTEGER,
            validation_result TEXT,
            deployed INTEGER DEFAULT 0,
            deployed_at TEXT,
            shadow_period_sharpe REAL,
            live_period_sharpe REAL,
            rolled_back INTEGER DEFAULT 0,
            created_at TEXT)''')
        c.execute('''CREATE INDEX IF NOT EXISTS idx_hyp_deployed
            ON hypothesis_history(deployed)''')

        # Phase 27 Task 23 (I2 Ajani): Adversarial exploit archive. Each
        # ExploiterAgent-proposed scenario is stored with its target weakness
        # and defensive counter so the nightly regression test can re-probe
        # the strategy against every historical exploit.
        c.execute('''CREATE TABLE IF NOT EXISTS exploit_archive (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pair TEXT, regime TEXT,
            exploit_scenario TEXT NOT NULL,
            target_weakness TEXT,
            predicted_loss REAL,
            was_defended INTEGER,
            defense_description TEXT,
            was_validated_by_outcome INTEGER,
            created_at TEXT,
            ttl_expiry TEXT)''')
        c.execute('''CREATE INDEX IF NOT EXISTS idx_exploit_pair
            ON exploit_archive(pair, regime)''')

        # Phase 27 Task 24 (F3 Ajani): Autopoietic Integrity Index history.
        # Four-layer identity drift monitor (Structural / Functional /
        # Behavioral / Representational) — architecture_evolver queries
        # the most recent composite AII before accepting any mutation.
        c.execute('''CREATE TABLE IF NOT EXISTS autopoietic_integrity (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            structural_score REAL,
            functional_score REAL,
            behavioral_score REAL,
            representational_score REAL,
            aii_composite REAL,
            status TEXT,
            action_taken TEXT)''')
        c.execute('''CREATE INDEX IF NOT EXISTS idx_aii_timestamp
            ON autopoietic_integrity(timestamp)''')

        # Phase 27 Task 25 (I5 Ajani): Trade-as-language sequence patterns.
        # N-gram, PrefixSpan, and conditional-grammar discoveries persisted so
        # the pattern library survives restarts and can feed back into
        # sizing / MADAM coordinator prompts. UNIQUE(pattern, regime) so the
        # weekly cycle upserts instead of flooding the table each Sunday.
        c.execute('''CREATE TABLE IF NOT EXISTS sequence_patterns (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pattern TEXT NOT NULL,
            n_gram_size INTEGER,
            occurrences INTEGER,
            expected_occurrences REAL,
            chi2_score REAL,
            p_value REAL,
            next_outcome_distribution TEXT,
            regime TEXT DEFAULT '_any',
            updated_at TEXT,
            UNIQUE(pattern, regime))''')
        c.execute('''CREATE INDEX IF NOT EXISTS idx_seq_pattern
            ON sequence_patterns(pattern)''')
        # Phase 27 Task 25 audit fix: CREATE TABLE IF NOT EXISTS won't add a
        # UNIQUE constraint to a pre-existing table. A separate UNIQUE INDEX
        # has the same semantics as the inline UNIQUE(pattern, regime) AND it
        # works on legacy installs where the table was created before the
        # constraint was added.
        try:
            c.execute('''CREATE UNIQUE INDEX IF NOT EXISTS ux_seq_pattern_regime
                ON sequence_patterns(pattern, regime)''')
        except sqlite3.IntegrityError:
            # Pre-existing duplicate rows — remove keeping the most recent.
            c.execute('''DELETE FROM sequence_patterns WHERE id NOT IN (
                SELECT MAX(id) FROM sequence_patterns GROUP BY pattern, regime
            )''')
            c.execute('''CREATE UNIQUE INDEX IF NOT EXISTS ux_seq_pattern_regime
                ON sequence_patterns(pattern, regime)''')

        # Phase 27 Fix 6: regime column on forgone_profit so the resolver /
        # adaptive-threshold jobs can group alpha-left-on-the-table by regime.
        # Idempotent ALTER — silently ignored if column already exists.
        try:
            c.execute("ALTER TABLE forgone_profit ADD COLUMN regime TEXT")
        except sqlite3.OperationalError:
            pass
        # Data Acceleration audit fix: capture the 6 evidence sub-scores +
        # trust_score at signal time so the shadow-trade CatBoost pipeline
        # gets REAL feature values instead of a constant 0.5 placeholder.
        # Each ALTER is idempotent — silently ignored if the column already
        # exists (legacy installs are auto-migrated on next init_db()).
        for _col, _typ in (
            ("trust_score", "REAL"),
            ("sub_trend", "REAL"),
            ("sub_momentum", "REAL"),
            ("sub_crowd", "REAL"),
            ("sub_evidence", "REAL"),
            ("sub_macro", "REAL"),
            ("sub_risk", "REAL"),
        ):
            try:
                c.execute(f"ALTER TABLE forgone_profit ADD COLUMN {_col} {_typ}")
            except sqlite3.OperationalError:
                pass

        c.execute('''CREATE TABLE IF NOT EXISTS binary_embeddings (
            text_hash TEXT PRIMARY KEY, binary_vec BLOB,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP)''')

        c.execute('''CREATE TABLE IF NOT EXISTS risk_budget (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            daily_var_limit REAL DEFAULT 0.05, current_usage REAL DEFAULT 0.0,
            max_portfolio_heat REAL DEFAULT 0.10, current_heat REAL DEFAULT 0.0,
            updated_at TEXT)''')

        c.execute('''CREATE TABLE IF NOT EXISTS ai_lessons (
            id INTEGER PRIMARY KEY AUTOINCREMENT, lesson_type TEXT,
            content TEXT, context TEXT, score REAL DEFAULT 0.0,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP)''')

        c.execute('''CREATE TABLE IF NOT EXISTS autonomy_state (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            current_level INTEGER DEFAULT 0, trust_alpha REAL DEFAULT 1.0,
            trust_beta REAL DEFAULT 1.0, total_trades INTEGER DEFAULT 0,
            successful_trades INTEGER DEFAULT 0, updated_at TEXT)''')

        c.execute('''CREATE TABLE IF NOT EXISTS pattern_trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT, pair TEXT, timeframe TEXT,
            pattern_type TEXT, entry_date TEXT, exit_date TEXT,
            pnl_pct REAL, duration_hours REAL, regime TEXT,
            features_json TEXT, created_at DATETIME DEFAULT CURRENT_TIMESTAMP)''')

        c.execute('''CREATE TABLE IF NOT EXISTS llm_calls (
            id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            model TEXT, provider TEXT, prompt_tokens INTEGER, completion_tokens INTEGER,
            cost_usd REAL, latency_ms REAL, purpose TEXT, success BOOLEAN DEFAULT 1)''')

        c.execute('''CREATE TABLE IF NOT EXISTS system_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT, metric_name TEXT NOT NULL,
            value REAL NOT NULL, metadata TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')

        c.execute('''CREATE TABLE IF NOT EXISTS model_slot_stats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            slot_id TEXT UNIQUE,
            alpha REAL DEFAULT 1.0,
            beta_param REAL DEFAULT 1.0,
            total_calls INTEGER DEFAULT 0,
            success_count INTEGER DEFAULT 0,
            quality_pass_count INTEGER DEFAULT 0,
            avg_latency_ms REAL DEFAULT 500.0,
            p95_latency_ms REAL DEFAULT 5000.0,
            last_updated TEXT)''')

        # EK Sprint 2026-04-23 (EK.2.10): LinUCB per-slot posterior. A and b
        # are stored as pickled numpy arrays so the router can restart and
        # resume learning without losing its contextual-bandit memory.
        c.execute('''CREATE TABLE IF NOT EXISTS linucb_state (
            slot_id TEXT PRIMARY KEY,
            a_blob BLOB,
            b_blob BLOB,
            n_updates INTEGER DEFAULT 0,
            last_updated TEXT)''')

        c.execute('''CREATE TABLE IF NOT EXISTS backtest_processed (
            id INTEGER PRIMARY KEY AUTOINCREMENT, file_hash TEXT UNIQUE,
            filename TEXT, processed_at DATETIME DEFAULT CURRENT_TIMESTAMP)''')

        c.execute('''CREATE TABLE IF NOT EXISTS memorag_global (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            draft TEXT, metadata TEXT, updated_at TEXT)''')

        c.execute('''CREATE TABLE IF NOT EXISTS graph_communities (
            id INTEGER PRIMARY KEY AUTOINCREMENT, community_id INTEGER,
            label TEXT, nodes_json TEXT, summary TEXT,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP)''')

        c.execute('''CREATE TABLE IF NOT EXISTS kg_entities (
            id INTEGER PRIMARY KEY AUTOINCREMENT, entity TEXT NOT NULL,
            entity_type TEXT, source_doc TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP)''')

        c.execute('''CREATE TABLE IF NOT EXISTS kg_relationships (
            id INTEGER PRIMARY KEY AUTOINCREMENT, subject TEXT NOT NULL,
            predicate TEXT NOT NULL, object TEXT NOT NULL, source_doc TEXT,
            confidence REAL DEFAULT 1.0,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP)''')

        c.execute('''CREATE TABLE IF NOT EXISTS chunk_quality_scores (
            chunk_id TEXT PRIMARY KEY, total_score REAL DEFAULT 0.0,
            retrieval_count INTEGER DEFAULT 0, positive_outcomes INTEGER DEFAULT 0,
            negative_outcomes INTEGER DEFAULT 0,
            last_updated DATETIME DEFAULT CURRENT_TIMESTAMP)''')

        c.execute('''CREATE TABLE IF NOT EXISTS trade_chunk_map (
            id INTEGER PRIMARY KEY AUTOINCREMENT, trade_id TEXT NOT NULL,
            chunk_id TEXT NOT NULL, pair TEXT, outcome_pnl REAL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')

        c.execute('''CREATE TABLE IF NOT EXISTS chunk_quality_flags (
            id INTEGER PRIMARY KEY AUTOINCREMENT, chunk_id TEXT NOT NULL,
            flag_type TEXT NOT NULL, reason TEXT,
            flagged_at DATETIME DEFAULT CURRENT_TIMESTAMP)''')

        c.execute('''CREATE TABLE IF NOT EXISTS rag_quality_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT, query TEXT,
            retrieval_method TEXT, relevance_score REAL, latency_ms REAL,
            chunk_count INTEGER, pair TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')

        # === PHASE 26 SPRINT 2: Causal + Counterfactual ===
        c.execute('''CREATE TABLE IF NOT EXISTS causal_discoveries (
            id INTEGER PRIMARY KEY AUTOINCREMENT, source_var TEXT NOT NULL,
            target_var TEXT NOT NULL, causal_strength REAL NOT NULL,
            time_lag INTEGER DEFAULT 0, p_value REAL, method TEXT DEFAULT 'PCMCI+',
            n_observations INTEGER, regime TEXT,
            discovered_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            is_active BOOLEAN DEFAULT 1,
            UNIQUE(source_var, target_var, time_lag, regime))''')
        # Audit fix (2026-04-19): purge legacy GNN_attention rows that were
        # mis-tagged as causal edges. gnn_organism now writes to its own
        # `gnn_attention_patterns` table; this DELETE is a one-shot cleanup
        # that is harmless to re-run (only removes rows that shouldn't exist).
        try:
            c.execute("DELETE FROM causal_discoveries WHERE method != 'PCMCI+'")
        except sqlite3.OperationalError:
            pass

        c.execute('''CREATE TABLE IF NOT EXISTS counterfactual_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT, original_trade_id INTEGER,
            intervention_var TEXT NOT NULL, original_value REAL,
            counterfactual_value REAL, original_outcome_pnl REAL,
            counterfactual_outcome_pnl REAL, ate REAL, confidence_lower REAL,
            confidence_upper REAL, method TEXT DEFAULT 'DoWhy', regime TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')

        # === PHASE 26 SPRINT 2: RL Agents ===
        c.execute('''CREATE TABLE IF NOT EXISTS rl_replay_buffer (
            id INTEGER PRIMARY KEY AUTOINCREMENT, episode_id INTEGER NOT NULL,
            step INTEGER NOT NULL, state_json TEXT NOT NULL,
            action_json TEXT NOT NULL, reward REAL NOT NULL,
            next_state_json TEXT, done BOOLEAN DEFAULT 0,
            source TEXT DEFAULT 'live', regime TEXT, pair TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')

        c.execute('''CREATE TABLE IF NOT EXISTS rl_checkpoints (
            id INTEGER PRIMARY KEY AUTOINCREMENT, model_name TEXT NOT NULL,
            organ TEXT, version INTEGER DEFAULT 1, path TEXT NOT NULL,
            training_episodes INTEGER, reward_mean REAL, reward_std REAL,
            sharpe REAL, is_active BOOLEAN DEFAULT 1,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP)''')

        # === PHASE 26 SPRINT 2: World Model + Dream Engine ===
        c.execute('''CREATE TABLE IF NOT EXISTS world_model_states (
            id INTEGER PRIMARY KEY AUTOINCREMENT, pair TEXT NOT NULL,
            timeframe TEXT DEFAULT '1h', timestamp DATETIME NOT NULL,
            ttm_embedding BLOB, chart_features_json TEXT, regime TEXT,
            fng INTEGER, hormones_json TEXT,
            UNIQUE(pair, timeframe, timestamp))''')

        c.execute('''CREATE TABLE IF NOT EXISTS world_model_rollouts (
            id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT NOT NULL,
            rollout_idx INTEGER, step INTEGER, state_embedding BLOB,
            predicted_reward REAL, predicted_regime TEXT, event_injected TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')

        c.execute('''CREATE TABLE IF NOT EXISTS dream_scenarios (
            id INTEGER PRIMARY KEY AUTOINCREMENT, dream_session_id TEXT NOT NULL,
            trajectory_idx INTEGER, step INTEGER, initial_state BLOB,
            event_type TEXT, state_after BLOB, reward REAL,
            passed_filter BOOLEAN DEFAULT 0, filter_reason TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')

        # === PHASE 26 SPRINT 2: Self-Model + Metacognition ===
        c.execute('''CREATE TABLE IF NOT EXISTS organ_performance_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT, organ TEXT NOT NULL,
            regime TEXT, metric TEXT NOT NULL, value REAL NOT NULL,
            sample_size INTEGER, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')

        c.execute('''CREATE TABLE IF NOT EXISTS hormone_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT, cortisol REAL, dopamine REAL,
            serotonin REAL, adrenaline REAL, market_stress REAL,
            organism_health REAL, trigger_event TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')

        c.execute('''CREATE TABLE IF NOT EXISTS self_model_profile (
            id INTEGER PRIMARY KEY AUTOINCREMENT, profile_type TEXT NOT NULL,
            key TEXT NOT NULL, value REAL NOT NULL, confidence REAL,
            sample_size INTEGER, updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(profile_type, key))''')

        # === PHASE 26 SPRINT 2: Backtest Training Pipeline (13B) ===
        c.execute('''CREATE TABLE IF NOT EXISTS backtest_training_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pair TEXT NOT NULL,
            open_date TEXT NOT NULL,
            close_date TEXT NOT NULL,
            direction TEXT NOT NULL DEFAULT 'long',
            profit_pct REAL NOT NULL,
            trade_duration_hours REAL,
            exit_reason TEXT,
            label INTEGER NOT NULL,
            label_name TEXT NOT NULL,
            features_json TEXT NOT NULL,
            n_features INTEGER,
            source TEXT DEFAULT 'backtest',
            strategy TEXT,
            regime TEXT,
            timeframe TEXT DEFAULT '1h',
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(pair, open_date, source, strategy))''')

        c.execute('''CREATE TABLE IF NOT EXISTS catboost_training_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_version TEXT NOT NULL,
            n_train INTEGER, n_test INTEGER,
            n_features INTEGER,
            train_accuracy REAL, test_accuracy REAL,
            train_f1 REAL, test_f1 REAL,
            feature_importance_json TEXT,
            label_distribution_json TEXT,
            model_path TEXT,
            hyperparams_json TEXT,
            data_sources_json TEXT,
            trained_at DATETIME DEFAULT CURRENT_TIMESTAMP)''')

        # === PHASE 26 SPRINT 2: Trinity RL→RAG Feedback (10B) ===
        c.execute('''CREATE TABLE IF NOT EXISTS rl_relevance_feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_id TEXT NOT NULL UNIQUE,
            trade_pnl REAL,
            relevance_delta REAL DEFAULT 0,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')

        # === INDICES ===
        for idx_sql in [
            'CREATE INDEX IF NOT EXISTS idx_market_news_published ON market_news(published_at)',
            'CREATE INDEX IF NOT EXISTS idx_ai_decisions_pair ON ai_decisions(pair)',
            'CREATE INDEX IF NOT EXISTS idx_sentiment_rolling_coin ON coin_sentiment_rolling(coin, timestamp)',
            'CREATE INDEX IF NOT EXISTS idx_signal_health_ts ON signal_health(timestamp)',
            'CREATE INDEX IF NOT EXISTS idx_deriv_pair_ts ON derivatives_data(pair, timestamp)',
            'CREATE INDEX IF NOT EXISTS idx_macro_name_ts ON macro_data(metric_name, timestamp)',
            'CREATE INDEX IF NOT EXISTS idx_defi_name_ts ON defi_data(metric_name, timestamp)',
            'CREATE INDEX IF NOT EXISTS idx_trends_kw_ts ON search_trends(keyword, timestamp)',
            'CREATE INDEX IF NOT EXISTS idx_ohlcv_pair ON ohlcv_patterns(pair)',
            'CREATE INDEX IF NOT EXISTS idx_agent_mem_type ON agent_memory(agent_type, regime)',
            'CREATE INDEX IF NOT EXISTS idx_agent_perf ON agent_performance(agent_type, regime)',
            'CREATE INDEX IF NOT EXISTS idx_opp_pair_ts ON opportunity_scores(pair, timestamp)',
            'CREATE INDEX IF NOT EXISTS idx_evidence_pair_ts ON evidence_audit_log(pair, timestamp)',
            # Neural Organism
            'CREATE INDEX IF NOT EXISTS idx_neuron_organ ON neuron_state(organ, regime)',
            'CREATE INDEX IF NOT EXISTS idx_hippo_pair ON hippocampus_episodes(pair, regime)',
            'CREATE INDEX IF NOT EXISTS idx_immune_pair ON immune_memory(pair)',
            'CREATE INDEX IF NOT EXISTS idx_organism_ts ON organism_audit(timestamp)',
            'CREATE INDEX IF NOT EXISTS idx_magma_graphs ON magma_edges(graph_type, source)',
            'CREATE INDEX IF NOT EXISTS idx_magma_pruning ON magma_edges(timestamp, weight)',
            # Phase 26 Sprint 2
            'CREATE INDEX IF NOT EXISTS idx_causal_src ON causal_discoveries(source_var, target_var)',
            'CREATE INDEX IF NOT EXISTS idx_causal_regime ON causal_discoveries(regime, is_active)',
            'CREATE INDEX IF NOT EXISTS idx_replay_episode ON rl_replay_buffer(episode_id, step)',
            'CREATE INDEX IF NOT EXISTS idx_replay_source ON rl_replay_buffer(source, regime)',
            'CREATE INDEX IF NOT EXISTS idx_wm_pair_ts ON world_model_states(pair, timestamp)',
            'CREATE INDEX IF NOT EXISTS idx_wm_rollout ON world_model_rollouts(run_id, rollout_idx)',
            'CREATE INDEX IF NOT EXISTS idx_dream_session ON dream_scenarios(dream_session_id)',
            'CREATE INDEX IF NOT EXISTS idx_organ_perf ON organ_performance_history(organ, regime, timestamp)',
            'CREATE INDEX IF NOT EXISTS idx_hormone_ts ON hormone_history(timestamp)',
            'CREATE INDEX IF NOT EXISTS idx_selfmodel ON self_model_profile(profile_type, key)',
            # Backtest Training Pipeline
            'CREATE INDEX IF NOT EXISTS idx_bt_train_pair ON backtest_training_data(pair, open_date)',
            'CREATE INDEX IF NOT EXISTS idx_bt_train_source ON backtest_training_data(source, strategy)',
            'CREATE INDEX IF NOT EXISTS idx_bt_train_label ON backtest_training_data(label)',
        ]:
            c.execute(idx_sql)

        # Ensure market_news has columns added after initial schema
        for col, typedef in [("title_hash", "TEXT"), ("is_embedded", "BOOLEAN DEFAULT 0")]:
            try:
                c.execute(f"ALTER TABLE market_news ADD COLUMN {col} {typedef}")
            except sqlite3.OperationalError:
                pass

        # Sprint 2: Ensure organism_audit has event_type + details_json columns
        for col, typedef in [("event_type", "TEXT"), ("details_json", "TEXT")]:
            try:
                c.execute(f"ALTER TABLE organism_audit ADD COLUMN {col} {typedef}")
            except sqlite3.OperationalError:
                pass

        conn.commit()

    logger.info(f"[DB] Database initialized at {DB_PATH}")


def _get_table_count() -> list:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        ).fetchall()
        return [r[0] for r in rows]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    init_db()
    tables = _get_table_count()
    print(f"Tables: {len(tables)}")
    for t in sorted(tables):
        print(f"  - {t}")
    print(f"Pool stats: {get_pool_stats()}")
    _pool.close_all()
