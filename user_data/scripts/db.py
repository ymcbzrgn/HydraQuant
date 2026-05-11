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

# Phase 30 A.32 — Canonical re-export for downstream imports
AI_DB_PATH = DB_PATH

# Phase 30 A.32 — Cleanup 0-byte legacy ai_data.sqlite at user_data/ root
try:
    from pathlib import Path as _Path
    _legacy = _Path(__file__).parent.parent / "ai_data.sqlite"
    if _legacy.exists() and _legacy.stat().st_size == 0:
        _legacy.unlink()
except Exception:
    pass

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
        # Task 13: automatic passive WAL checkpoint every 1000 pages
        # (~4 MB). Prior to this the WAL file had no upper bound between
        # the 50-release TRUNCATE triggers — a quiet process could carry
        # a multi-MB WAL indefinitely. PASSIVE lets other readers/writers
        # continue; TRUNCATE happens later via the release-count path +
        # the new size-based scheduler job.
        conn.execute("PRAGMA wal_autocheckpoint=1000")
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
            # Task 12: rollback BEFORE returning to pool. Borrowers that
            # crashed mid-transaction (neural_organism persist paths used
            # `conn = get_db_connection(...); ...; conn.close()` without
            # a finally/rollback) left an open txn sitting on the conn.
            # The next borrower's BEGIN was then effectively part of the
            # dead borrower's uncommitted write — the classic "pool
            # transaction leak".
            try:
                raw = conn._conn if isinstance(conn, _PooledConnection) else conn
                raw.rollback()
            except Exception:
                pass
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


class _RetryingCursor:
    """Proxy around `sqlite3.Cursor` that runs `execute` / `executemany`
    through the same `_retry_on_locked` wrapper as the parent connection.
    Prior to this, ~100 call sites in the codebase opened a raw cursor
    (`c = conn.cursor(); c.execute(...)`) and silently bypassed the
    retry logic — a single `database is locked` race on the raw cursor
    crashed the caller even though the pooled conn would have retried."""

    def __init__(self, cursor, retry_runner):
        self._cursor = cursor
        self._retry = retry_runner

    def execute(self, *args, **kwargs):
        return self._retry(self._cursor.execute, *args, **kwargs)

    def executemany(self, *args, **kwargs):
        return self._retry(self._cursor.executemany, *args, **kwargs)

    def executescript(self, *args, **kwargs):
        return self._retry(self._cursor.executescript, *args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._cursor, name)

    def __iter__(self):
        return iter(self._cursor)


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
        """Retry a database operation on 'database is locked' errors.

        Phase 30 A.7: jittered exponential backoff so concurrent freqtrade +
        scheduler + rag-service processes don't synchronize their retries on
        the same SQLITE_BUSY tick.
        """
        import random as _r
        for attempt in range(self._RETRY_MAX):
            try:
                return func(*args, **kwargs)
            except sqlite3.OperationalError as e:
                if "database is locked" in str(e) and attempt < self._RETRY_MAX - 1:
                    base = self._RETRY_BASE_WAIT * (attempt + 1)
                    wait = base * _r.uniform(0.5, 1.5)
                    logger.warning(f"[DB] database is locked, retry {attempt+1}/{self._RETRY_MAX} in {wait:.2f}s (jittered)")
                    time.sleep(wait)
                else:
                    raise

    def __getattr__(self, name):
        return getattr(self._conn, name)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            # Best-effort rollback; a failed commit from the caller's
            # finally-block inside __exit__ must not mask the original
            # exception.
            try:
                self._conn.rollback()
            except Exception:
                pass
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
        # Task 12: hand out a retrying cursor so raw `cursor.execute(...)`
        # sites inherit the lock-retry contract automatically. The
        # wrapper transparently forwards attribute access so the
        # existing .fetchone()/.fetchall()/.lastrowid usage is unchanged.
        return _RetryingCursor(self._conn.cursor(), self._retry_on_locked)

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


def _looks_like_write(sql: str) -> bool:
    """Heuristic: does this statement mutate? Used to decide whether to
    route through the SQLite write broker when it's available.
    """
    stripped = (sql or "").lstrip().upper()
    return stripped.startswith(("INSERT", "UPDATE", "DELETE", "REPLACE"))


def execute_with_retry(sql: str, params=None, max_retries: int = 3, commit: bool = True,
                       db_path: str = None, bypass_broker: bool = False):
    """Execute SQL with retry on 'database is locked'.

    If db_path is provided, uses get_db_connection(db_path) (respects the caller's
    DB path — needed for tests that use tmp_path, and for any module that persists
    its own data to a scoped DB). Otherwise uses the default pooled connection.

    Task 12: when the SQLite write broker is alive AND the caller targets
    the canonical AI_DB_PATH AND the statement mutates, route through the
    broker so writes from multiple processes serialise through a single
    WAL-friendly writer. Reads and test-path (db_path != DB_PATH) writes
    bypass the broker and use the local pool. Callers that MUST use the
    pool (e.g. migrations inside init_db) can pass `bypass_broker=True`.
    """
    # Fast-path: broker routing for writes to the canonical DB. Task 27:
    # the earlier predicate required `db_path is None`, so every caller
    # that explicitly passed AI_DB_PATH (llm_cost_tracker, semantic_cache,
    # slippage_forecaster — the hot-path writers) silently bypassed the
    # broker. Treating the explicit path as equivalent to the default
    # makes routing reach the writers Task 11 was built to protect.
    _canonical = (db_path is None) or (db_path == DB_PATH)
    if (not bypass_broker) and _canonical and _looks_like_write(sql):
        try:
            from sqlite_broker import get_write_client
            client = get_write_client()
            if client.is_alive(cache_seconds=30.0):
                resp = client.write(sql, params=list(params or []),
                                     want_lastrowid=True)
                if resp.get("ok"):
                    # Callers treat the returned object like a cursor and
                    # usually only read `lastrowid` / `rowcount`. Give the
                    # shim no-op fetch methods too so any future caller
                    # chaining `.fetchone()` / `.fetchall()` degrades
                    # quietly instead of raising AttributeError.
                    class _BrokerCursor:
                        def __init__(self, lastrowid, rowcount):
                            self.lastrowid = lastrowid
                            self.rowcount = rowcount
                            self.description = None
                        def fetchone(self): return None
                        def fetchall(self): return []
                        def fetchmany(self, size=1): return []
                        def close(self): pass
                    return _BrokerCursor(
                        resp.get("lastrowid"),
                        int(resp.get("rowcount", -1)),
                    )
        except Exception:
            # Any broker hiccup falls back to the in-process pool path.
            pass

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
                # Task 27: final fallback — spool the write for async
                # replay when the broker returns. Only for writes to the
                # canonical DB and only when the error was a stuck lock
                # (not a constraint violation). Non-write SELECTs fall
                # through and re-raise as before.
                if _canonical and _looks_like_write(sql) and "database is locked" in str(e):
                    try:
                        from sqlite_broker import spool_write
                        if spool_write(sql, params=list(params or []), priority=5):
                            logger.warning(
                                "[DB] pool exhausted + broker down — write spooled "
                                "for async replay"
                            )
                            class _SpoolCursor:
                                lastrowid = None
                                rowcount = -1
                                description = None
                                def fetchone(self): return None
                                def fetchall(self): return []
                                def fetchmany(self, size=1): return []
                                def close(self): pass
                            return _SpoolCursor()
                    except Exception:
                        pass
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
            outcome_duration REAL,
            agent_votes_json TEXT,
            _status_cache TEXT)''')

        c.execute('''CREATE TABLE IF NOT EXISTS forgone_profit (
            id INTEGER PRIMARY KEY AUTOINCREMENT, pair TEXT NOT NULL,
            signal_type TEXT, signal_time DATETIME DEFAULT CURRENT_TIMESTAMP,
            confidence REAL, entry_price REAL, was_executed BOOLEAN DEFAULT 0,
            exit_price REAL, forgone_pnl REAL, resolved_at DATETIME)''')

        c.execute('''CREATE TABLE IF NOT EXISTS embedding_cache (
            text_hash TEXT PRIMARY KEY, text_content TEXT NOT NULL,
            gemini_embedding BLOB, bge_embedding BLOB,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            last_used_at DATETIME DEFAULT CURRENT_TIMESTAMP)''')
        # AUDIT-9 (2026-04-25): true LRU needs last_used_at, refreshed on
        # every cache HIT (not just on miss INSERT). Idempotent migration
        # adds the column for DBs created before this change.
        try:
            c.execute("ALTER TABLE embedding_cache ADD COLUMN last_used_at DATETIME DEFAULT CURRENT_TIMESTAMP")
        except sqlite3.OperationalError:
            pass  # column already exists

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

        # ⚠️  DEPRECATED 2026-04-23 (Mega Sprint Phase 27 Task 1):
        # This single-row legacy table was retired when per-pair Beta
        # posteriors took over (bayesian_kelly_per_pair below). NO live
        # writers remain — `position_sizer.py:75` whitelists ONLY the new
        # tables and would raise on any attempt to write here. Kept as
        # CREATE TABLE IF NOT EXISTS for forward-compatibility (existing
        # production rows like the alpha=37.22/beta=14.88 fossil from
        # Apr 17 17:12 are harmless: no consumer reads them). Do NOT
        # rewire anything to this table.
        c.execute('''CREATE TABLE IF NOT EXISTS bayesian_kelly (
            id INTEGER PRIMARY KEY AUTOINCREMENT, pair TEXT, regime TEXT,
            alpha REAL DEFAULT 1.0, beta_param REAL DEFAULT 1.0,
            kelly_fraction REAL DEFAULT 0.01, trade_count INTEGER DEFAULT 0,
            updated_at TEXT, UNIQUE(pair, regime))''')

        # Phase 27 Task 1 (E1 Ajani): Per-pair Bayesian Kelly — Prensip 0.
        # Replaces legacy single-row bayesian_kelly. Every (pair, regime) has
        # its own Beta posterior + vol drag + vol-of-vol inputs for the
        # 7-step sizing pipeline in position_sizer.py.
        # Sprint 2026-05-05 (B-CONNECT C1): `side` column splits long vs short
        # Beta posteriors. Live migration handled by position_sizer._ensure_schema.
        c.execute('''CREATE TABLE IF NOT EXISTS bayesian_kelly_per_pair (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pair TEXT NOT NULL,
            regime TEXT NOT NULL DEFAULT '_global',
            side TEXT NOT NULL DEFAULT '_any',
            alpha REAL DEFAULT 2.0,
            beta_param REAL DEFAULT 2.0,
            avg_win REAL DEFAULT 0.0,
            avg_loss REAL DEFAULT 0.0,
            n_trades INTEGER DEFAULT 0,
            annual_volatility REAL,
            vol_of_vol REAL,
            last_sharpe REAL,
            updated_at TEXT,
            UNIQUE(pair, regime, side))''')
        c.execute('''CREATE INDEX IF NOT EXISTS idx_bk_per_pair
            ON bayesian_kelly_per_pair(pair, regime, side)''')

        # EK Sprint 2026-04-23: shadow ledger. Same schema as the real one
        # but only updated by the forgone-feedback scheduler job. Live
        # sizing never reads this — it's a parallel analytics ledger.
        c.execute('''CREATE TABLE IF NOT EXISTS bayesian_kelly_shadow_per_pair (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pair TEXT NOT NULL,
            regime TEXT NOT NULL DEFAULT '_global',
            side TEXT NOT NULL DEFAULT '_any',
            alpha REAL DEFAULT 2.0,
            beta_param REAL DEFAULT 2.0,
            avg_win REAL DEFAULT 0.0,
            avg_loss REAL DEFAULT 0.0,
            n_trades INTEGER DEFAULT 0,
            annual_volatility REAL,
            vol_of_vol REAL,
            last_sharpe REAL,
            updated_at TEXT,
            UNIQUE(pair, regime, side))''')
        c.execute('''CREATE INDEX IF NOT EXISTS idx_bk_shadow_per_pair
            ON bayesian_kelly_shadow_per_pair(pair, regime, side)''')

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

        # Canonical risk_budget schema (matches risk_budget.py RiskBudgetManager).
        # Per-day ledger; UNIQUE(date) prevents duplicate-row race between
        # scheduler._daily_reset and RiskBudgetManager._load_state firing in
        # the same second (Apr 11 2026 produced two rows for a single date).
        c.execute('''CREATE TABLE IF NOT EXISTS risk_budget (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            initial_budget REAL NOT NULL,
            consumed REAL DEFAULT 0.0,
            multiplier REAL DEFAULT 1.0,
            updated_at TEXT,
            UNIQUE(date))''')
        # Self-healing migration for any install where the old single-row
        # schema (daily_var_limit/current_usage/max_portfolio_heat/current_heat)
        # was created first — add the canonical columns in-place. Existing
        # pre-UNIQUE ledger rows are deduped before the UNIQUE INDEX is created.
        for _col, _typ in [
            ("date", "TEXT"),
            ("initial_budget", "REAL"),
            ("consumed", "REAL DEFAULT 0.0"),
            ("multiplier", "REAL DEFAULT 1.0"),
        ]:
            try:
                c.execute(f"ALTER TABLE risk_budget ADD COLUMN {_col} {_typ}")
            except sqlite3.OperationalError:
                pass
        try:
            c.execute("CREATE UNIQUE INDEX IF NOT EXISTS ux_risk_budget_date ON risk_budget(date)")
        except sqlite3.IntegrityError:
            c.execute("DELETE FROM risk_budget WHERE id NOT IN (SELECT MAX(id) FROM risk_budget WHERE date IS NOT NULL GROUP BY date)")
            try:
                c.execute("CREATE UNIQUE INDEX IF NOT EXISTS ux_risk_budget_date ON risk_budget(date)")
            except sqlite3.IntegrityError:
                logger.warning("[DB] risk_budget UNIQUE(date) index still blocked — duplicate NULL dates may exist")

        c.execute('''CREATE TABLE IF NOT EXISTS ai_lessons (
            id INTEGER PRIMARY KEY AUTOINCREMENT, decision_id INTEGER,
            pair TEXT, signal TEXT, outcome_pnl REAL, lesson_text TEXT, lesson_type TEXT,
            content TEXT, context TEXT, score REAL DEFAULT 0.0,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP)''')

        c.execute('''CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT, pair TEXT, stake_amount REAL,
            open_date DATETIME, close_date DATETIME)''')

        # Canonical autonomy_state schema (matches autonomy_manager.AutonomyManager).
        # api_ai.py reads level/promoted_at/sharpe_estimate/max_drawdown_pct/days_at_level;
        # the old (current_level/trust_alpha/trust_beta/successful_trades) schema was
        # never materialised and had no producers — kept here only as an ADD COLUMN
        # safety net for installs where init_db ran before AutonomyManager.
        c.execute('''CREATE TABLE IF NOT EXISTS autonomy_state (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            level INTEGER DEFAULT 0,
            promoted_at TEXT,
            total_trades INTEGER DEFAULT 0,
            sharpe_estimate REAL DEFAULT 0.0,
            max_drawdown_pct REAL DEFAULT 0.0,
            days_at_level INTEGER DEFAULT 0,
            updated_at TEXT)''')
        for _col, _typ in [
            ("level", "INTEGER DEFAULT 0"),
            ("promoted_at", "TEXT"),
            ("sharpe_estimate", "REAL DEFAULT 0.0"),
            ("max_drawdown_pct", "REAL DEFAULT 0.0"),
            ("days_at_level", "INTEGER DEFAULT 0"),
        ]:
            try:
                c.execute(f"ALTER TABLE autonomy_state ADD COLUMN {_col} {_typ}")
            except sqlite3.OperationalError:
                pass
        # If a legacy install has data in the retired `current_level` column,
        # copy it into `level` so the autonomy ladder survives a rename.
        try:
            c.execute("UPDATE autonomy_state SET level = current_level WHERE (level IS NULL OR level = 0) AND current_level IS NOT NULL")
        except sqlite3.OperationalError:
            pass

        c.execute('''CREATE TABLE IF NOT EXISTS pattern_trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT, pair TEXT, timeframe TEXT,
            pattern_type TEXT, entry_date TEXT, exit_date TEXT,
            pnl_pct REAL, duration_hours REAL, regime TEXT,
            features_json TEXT, created_at DATETIME DEFAULT CURRENT_TIMESTAMP)''')

        c.execute('''CREATE TABLE IF NOT EXISTS llm_calls (
            id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            model TEXT, provider TEXT, prompt_tokens INTEGER, completion_tokens INTEGER,
            cost_usd REAL, latency_ms REAL, purpose TEXT, success BOOLEAN DEFAULT 1)''')

        # Sprint 2026-05-01: schema aligned with production (system_monitor.py
        # already creates this table with metric_value/metadata_json column
        # names). Two writers exist in this codebase that diverged:
        #   • db.py legacy: (metric_name, value, metadata)
        #   • system_monitor.py + scheduler new writers: (metric_name, metric_value, metadata_json)
        # Production DBs were initialized by system_monitor.py first, so the
        # new column names are authoritative. This migration aligns db.py to
        # match. Existing fresh DBs created against the old shape pick up
        # the new shape on next init via additive ALTER TABLE.
        c.execute('''CREATE TABLE IF NOT EXISTS system_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
            metric_name TEXT NOT NULL,
            metric_value REAL NOT NULL,
            metadata_json TEXT)''')
        # Idempotent forward-migration for DBs created before this change.
        existing_cols = {r[1] for r in c.execute(
            "PRAGMA table_info(system_metrics)").fetchall()}
        if "metric_value" not in existing_cols and "value" in existing_cols:
            try:
                c.execute("ALTER TABLE system_metrics RENAME COLUMN value TO metric_value")
            except Exception:
                # SQLite < 3.25 doesn't support RENAME COLUMN — fall back to
                # ADD COLUMN + double-write convention. The new writers all
                # use metric_value, the legacy writer (scheduler.py:3975)
                # has been migrated as part of this sprint.
                try:
                    c.execute("ALTER TABLE system_metrics ADD COLUMN metric_value REAL")
                except Exception:
                    pass
        if "metadata_json" not in existing_cols and "metadata" in existing_cols:
            try:
                c.execute("ALTER TABLE system_metrics RENAME COLUMN metadata TO metadata_json")
            except Exception:
                try:
                    c.execute("ALTER TABLE system_metrics ADD COLUMN metadata_json TEXT")
                except Exception:
                    pass

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
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            sizing_mult REAL, danger_level REAL)''')
        # Retro-compat: older installs missed these columns. The lifecycle
        # tick needs its sizing_mult / danger_level in dedicated slots so
        # organism_health stops being overloaded with multiplier values.
        for _col, _typ in [
            ("sizing_mult", "REAL"),
            ("danger_level", "REAL"),
        ]:
            try:
                c.execute(f"ALTER TABLE hormone_history ADD COLUMN {_col} {_typ}")
            except sqlite3.OperationalError:
                pass

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

        # === PHASE 30 — Production Forensics + Hardening Migrations ===
        # A.27: Realtime price anomaly detector
        c.execute('''CREATE TABLE IF NOT EXISTS price_anomaly_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pair TEXT NOT NULL,
            kind TEXT NOT NULL,
            magnitude REAL NOT NULL,
            close REAL,
            prev_close REAL,
            ts DATETIME DEFAULT CURRENT_TIMESTAMP)''')

        # A.28: ai_lessons UNIQUE dedup index (existing duplicates removed by migrate helper)
        try:
            c.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_ai_lessons_uniq ON ai_lessons(decision_id, pair)")
        except sqlite3.OperationalError:
            pass

        # A.31: llm_calls error/status columns (idempotent)
        for col, typedef in [
            ("error", "TEXT DEFAULT NULL"),
            ("error_class", "TEXT DEFAULT NULL"),
            ("status", "TEXT DEFAULT 'success'"),
        ]:
            try:
                c.execute(f"ALTER TABLE llm_calls ADD COLUMN {col} {typedef}")
            except sqlite3.OperationalError:
                pass

        # A.33: RAG endpoint latency
        c.execute('''CREATE TABLE IF NOT EXISTS rag_endpoint_latency (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            endpoint TEXT NOT NULL,
            pair TEXT,
            regime TEXT,
            latency_ms INTEGER,
            status_code INTEGER,
            timeout_breach INTEGER DEFAULT 0,
            ts DATETIME DEFAULT CURRENT_TIMESTAMP)''')

        # A.35: Service restart events
        c.execute('''CREATE TABLE IF NOT EXISTS service_restart_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            service TEXT NOT NULL,
            n_restarts INTEGER,
            last_restart_ts TEXT,
            detection_ts DATETIME DEFAULT CURRENT_TIMESTAMP,
            delta_since_last INTEGER DEFAULT 0,
            suspected_cause TEXT)''')

        # A.4: Tool result disk persist
        c.execute('''CREATE TABLE IF NOT EXISTS tool_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tool_name TEXT NOT NULL,
            preview TEXT,
            full_path TEXT,
            byte_size INTEGER,
            kind TEXT,
            ts DATETIME DEFAULT CURRENT_TIMESTAMP)''')

        # A.6: Doom loop detector result hash
        c.execute('''CREATE TABLE IF NOT EXISTS doom_loop_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pair TEXT NOT NULL,
            decision_hash TEXT,
            consecutive_count INTEGER,
            window_start TEXT,
            window_end TEXT,
            severity TEXT,
            ts DATETIME DEFAULT CURRENT_TIMESTAMP)''')

        # A.8: Unsuccessful agent decisions
        c.execute('''CREATE TABLE IF NOT EXISTS agent_pool_unsuccessful_decisions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pair TEXT NOT NULL,
            agent_name TEXT,
            round_idx INTEGER,
            failure_class TEXT,
            failure_text TEXT,
            recovered INTEGER DEFAULT 0,
            ts DATETIME DEFAULT CURRENT_TIMESTAMP)''')

        # A.10: SHA-256 prompt integrity
        c.execute('''CREATE TABLE IF NOT EXISTS prompt_hashes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            agent_name TEXT NOT NULL UNIQUE,
            prompt_sha256 TEXT NOT NULL,
            recorded_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            last_verified_at DATETIME)''')
        c.execute('''CREATE TABLE IF NOT EXISTS prompt_integrity_violations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            agent_name TEXT NOT NULL,
            expected_hash TEXT,
            actual_hash TEXT,
            ts DATETIME DEFAULT CURRENT_TIMESTAMP)''')

        # A.14: JSON parse failures
        c.execute('''CREATE TABLE IF NOT EXISTS parse_failures (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source TEXT NOT NULL,
            raw_text TEXT,
            error_text TEXT,
            recovery_method TEXT,
            recovered INTEGER DEFAULT 0,
            ts DATETIME DEFAULT CURRENT_TIMESTAMP)''')

        # A.15: News cluster
        c.execute('''CREATE TABLE IF NOT EXISTS news_clusters (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            cluster_key TEXT NOT NULL,
            news_id INTEGER,
            jaccard_score REAL,
            cluster_size INTEGER,
            ts DATETIME DEFAULT CURRENT_TIMESTAMP)''')

        # A.16: Threat classification
        c.execute('''CREATE TABLE IF NOT EXISTS threat_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            news_id INTEGER,
            tier TEXT NOT NULL,
            score REAL,
            keywords TEXT,
            ts DATETIME DEFAULT CURRENT_TIMESTAMP)''')

        # A.17: LLM response cache
        c.execute('''CREATE TABLE IF NOT EXISTS llm_response_cache (
            cache_key TEXT PRIMARY KEY,
            model TEXT,
            response_text TEXT,
            tokens_in INTEGER,
            tokens_out INTEGER,
            hit_count INTEGER DEFAULT 0,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            last_hit_at DATETIME)''')

        # A.20: JSONL scratchpad index
        c.execute('''CREATE TABLE IF NOT EXISTS scratchpad_jobs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id TEXT NOT NULL,
            scratchpad_path TEXT,
            started_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            ended_at DATETIME,
            status TEXT,
            line_count INTEGER DEFAULT 0)''')

        # A.23: Plateau detection
        c.execute('''CREATE TABLE IF NOT EXISTS plateau_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            metric TEXT NOT NULL,
            window_days INTEGER,
            std_pct REAL,
            mean REAL,
            severity TEXT,
            ts DATETIME DEFAULT CURRENT_TIMESTAMP)''')

        # A.25: Workflow event bus
        c.execute('''CREATE TABLE IF NOT EXISTS workflow_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            kind TEXT NOT NULL,
            payload_json TEXT,
            consumer TEXT,
            ts DATETIME DEFAULT CURRENT_TIMESTAMP)''')

        # A.29: Autonomy diagnostic snapshots
        c.execute('''CREATE TABLE IF NOT EXISTS autonomy_diagnostics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            level INTEGER,
            days_stuck INTEGER,
            n_trades_30d INTEGER,
            winrate_30d REAL,
            sharpe_approx_30d REAL,
            worst_drawdown_30d REAL,
            eligible INTEGER DEFAULT 0,
            ts DATETIME DEFAULT CURRENT_TIMESTAMP)''')

        # B.6: Provider capabilities + B.5 adaptive concurrency
        c.execute('''CREATE TABLE IF NOT EXISTS provider_capabilities (
            model TEXT PRIMARY KEY,
            context_window INTEGER,
            supports_json INTEGER DEFAULT 0,
            supports_tools INTEGER DEFAULT 0,
            cost_per_1m_in REAL DEFAULT 0,
            cost_per_1m_out REAL DEFAULT 0,
            current_concurrency INTEGER DEFAULT 1,
            target_concurrency INTEGER DEFAULT 1,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP)''')

        # B.7: Cross-process rate guard state
        c.execute('''CREATE TABLE IF NOT EXISTS rate_guard_state (
            provider TEXT PRIMARY KEY,
            window_start TEXT,
            count INTEGER DEFAULT 0,
            limit_rpm INTEGER,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP)''')

        # B.8: Tool loop guardrails events
        c.execute('''CREATE TABLE IF NOT EXISTS tool_loop_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pattern TEXT NOT NULL,
            context_json TEXT,
            ts DATETIME DEFAULT CURRENT_TIMESTAMP)''')

        # B.9: Error taxonomy
        c.execute('''CREATE TABLE IF NOT EXISTS error_taxonomy_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            error_class TEXT,
            taxonomy_label TEXT,
            retryable INTEGER DEFAULT 0,
            permanent INTEGER DEFAULT 0,
            count INTEGER DEFAULT 1,
            last_seen DATETIME DEFAULT CURRENT_TIMESTAMP)''')

        # B.13: Iteration budget
        c.execute('''CREATE TABLE IF NOT EXISTS iteration_budget_state (
            run_id TEXT PRIMARY KEY,
            parent_iters INTEGER DEFAULT 0,
            child_iters INTEGER DEFAULT 0,
            tokens_in INTEGER DEFAULT 0,
            tokens_out INTEGER DEFAULT 0,
            budget_breached INTEGER DEFAULT 0,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP)''')

        # B.16: Hourly KPI rollup
        c.execute('''CREATE TABLE IF NOT EXISTS kpi_rollup_hourly (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            hour_bucket TEXT NOT NULL UNIQUE,
            n_trades INTEGER DEFAULT 0,
            wins INTEGER DEFAULT 0,
            losses INTEGER DEFAULT 0,
            pnl_sum REAL DEFAULT 0,
            n_llm_calls INTEGER DEFAULT 0,
            avg_latency_ms REAL DEFAULT 0,
            n_anomalies INTEGER DEFAULT 0,
            ts DATETIME DEFAULT CURRENT_TIMESTAMP)''')

        # B.18: Telemetry single
        c.execute('''CREATE TABLE IF NOT EXISTS telemetry_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            kind TEXT NOT NULL,
            severity TEXT DEFAULT 'info',
            source_module TEXT,
            payload_json TEXT,
            ts DATETIME DEFAULT CURRENT_TIMESTAMP)''')

        # C.1/C.2: Composite risk manager decisions
        c.execute('''CREATE TABLE IF NOT EXISTS composite_risk_decisions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pair TEXT NOT NULL,
            check_name TEXT NOT NULL,
            passed INTEGER DEFAULT 0,
            reason TEXT,
            modifier_applied REAL,
            ts DATETIME DEFAULT CURRENT_TIMESTAMP)''')

        # C.6/C.7: Redteam audit
        c.execute('''CREATE TABLE IF NOT EXISTS redteam_audit_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_kind TEXT NOT NULL,
            agent_name TEXT,
            attack_template TEXT,
            success INTEGER DEFAULT 0,
            iterations INTEGER DEFAULT 0,
            details_json TEXT,
            ts DATETIME DEFAULT CURRENT_TIMESTAMP)''')

        # C.10: Contradiction matrix
        c.execute('''CREATE TABLE IF NOT EXISTS contradiction_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pair TEXT NOT NULL,
            agent_a TEXT,
            agent_b TEXT,
            disagreement REAL,
            time_decay_weight REAL DEFAULT 1.0,
            ts DATETIME DEFAULT CURRENT_TIMESTAMP)''')

        # C.13: Deploy hash history
        c.execute('''CREATE TABLE IF NOT EXISTS deploy_hash_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            commit_hash TEXT,
            mismatches INTEGER,
            ts TEXT)''')

        # D.1: Shadow Kelly promotions
        c.execute('''CREATE TABLE IF NOT EXISTS shadow_kelly_promotions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pair TEXT NOT NULL,
            score REAL,
            streak INTEGER,
            promoted INTEGER DEFAULT 0,
            blocked_by TEXT,
            ts DATETIME DEFAULT CURRENT_TIMESTAMP)''')

        # D.2: Plan/verify SOP
        c.execute('''CREATE TABLE IF NOT EXISTS plan_verify_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pair TEXT NOT NULL,
            plan_json TEXT,
            verify_verdict TEXT,
            verify_reason TEXT,
            executed INTEGER DEFAULT 0,
            ts DATETIME DEFAULT CURRENT_TIMESTAMP)''')

        # D.9: Real-capital promotion gate history
        c.execute('''CREATE TABLE IF NOT EXISTS promotion_gate_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            eligibility_pct REAL,
            passed INTEGER,
            blocked_by TEXT,
            metrics_json TEXT,
            ts DATETIME DEFAULT CURRENT_TIMESTAMP)''')

        # === PHASE 30 INDICES ===
        for idx_sql in [
            'CREATE INDEX IF NOT EXISTS idx_anomaly_pair_ts ON price_anomaly_events(pair, ts)',
            'CREATE INDEX IF NOT EXISTS idx_rag_lat_ts ON rag_endpoint_latency(ts)',
            'CREATE INDEX IF NOT EXISTS idx_rag_lat_endpoint ON rag_endpoint_latency(endpoint)',
            'CREATE INDEX IF NOT EXISTS idx_restart_svc ON service_restart_events(service, detection_ts)',
            'CREATE INDEX IF NOT EXISTS idx_tool_results_ts ON tool_results(ts)',
            'CREATE INDEX IF NOT EXISTS idx_doom_pair ON doom_loop_events(pair, ts)',
            'CREATE INDEX IF NOT EXISTS idx_unsuccessful_pair ON agent_pool_unsuccessful_decisions(pair, ts)',
            'CREATE INDEX IF NOT EXISTS idx_parse_failures_src ON parse_failures(source, ts)',
            'CREATE INDEX IF NOT EXISTS idx_news_clusters_key ON news_clusters(cluster_key, ts)',
            'CREATE INDEX IF NOT EXISTS idx_threat_tier ON threat_events(tier, ts)',
            'CREATE INDEX IF NOT EXISTS idx_llm_cache_last_hit ON llm_response_cache(last_hit_at)',
            'CREATE INDEX IF NOT EXISTS idx_workflow_kind ON workflow_events(kind, ts)',
            'CREATE INDEX IF NOT EXISTS idx_autonomy_diag_ts ON autonomy_diagnostics(ts)',
            'CREATE INDEX IF NOT EXISTS idx_telemetry_kind ON telemetry_events(kind, severity, ts)',
            'CREATE INDEX IF NOT EXISTS idx_composite_risk_pair ON composite_risk_decisions(pair, ts)',
            'CREATE INDEX IF NOT EXISTS idx_redteam_kind ON redteam_audit_runs(run_kind, ts)',
            'CREATE INDEX IF NOT EXISTS idx_contradiction_pair ON contradiction_events(pair, ts)',
            'CREATE INDEX IF NOT EXISTS idx_promotion_ts ON promotion_gate_history(ts)',
            'CREATE INDEX IF NOT EXISTS idx_plan_verify_pair ON plan_verify_runs(pair, ts)',
        ]:
            c.execute(idx_sql)

        # === PHASE 30 — Dedup helper for ai_lessons (A.28) ===
        try:
            c.execute("""DELETE FROM ai_lessons
                         WHERE id NOT IN (SELECT MAX(id) FROM ai_lessons GROUP BY decision_id, pair)""")
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
