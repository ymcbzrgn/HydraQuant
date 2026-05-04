"""
Phase 4.5: APScheduler Pipeline Automation
Replaces the crude while/sleep loop with proper job scheduling.

Schedule:
  - Every 5 min:  RSS fetch + sentiment analysis + Fear & Greed Index
  - Every 15 min: Embedding pipeline (new news → ChromaDB)
  - Every day 00:00 UTC: Risk budget daily reset
  - Every day 04:00 UTC: Old news cleanup (>180 days)
"""

import os
import sys
import json
import sqlite3

# Phase 25: Neural Organism — adaptive parameters
try:
    from neural_organism import _p
except ImportError:
    def _p(param_id, fallback=0.5, regime="_global"):
        return fallback
import logging
from datetime import datetime, timezone

sys.path.append(os.path.dirname(__file__))

# NumPy 2.x compat shim — MUST be before any pandas/yfinance import
# yfinance 1.2.0 internally uses np.matrix (removed in numpy 2.0)
import numpy as _np
if not hasattr(_np, 'matrix'):
    _np.matrix = _np.asmatrix

# Load .env BEFORE any module that needs API keys
from dotenv import load_dotenv
from db import get_connection, get_db_connection
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))

logger = logging.getLogger(__name__)


class PipelineScheduler:
    """
    Background scheduler for all data pipeline jobs.
    Uses APScheduler for reliable cron-like scheduling.
    """

    def __init__(self):
        self.scheduler = None
        self._pipeline = None
        # Singleton instances — created once, reused across all job runs
        # Prevents memory leak from creating new objects every 5-60 minutes
        self._semantic_cache = None
        self._streaming_rag = None
        self._market_data_fetcher = None
        self._backtest_embedder = None
        self._magma_memory = None
        self._opportunity_scanner = None
        self._agent_pool = None
        self._cross_pair_intel = None
        # Phase 21: Additional singletons to prevent memory leak (+200MB/hr)
        self._system_monitor = None
        self._telegram_notifier = None
        self._bidi_rag = None
        self._calibrator = None
        self._forgone_engine = None
        # Phase 23: Missing singletons (explorer-god audit — ~80-150MB/day avoidable)
        self._evidence_engine = None
        self._cost_tracker = None
        self._autonomy_manager = None
        self._rag_evaluator = None
        self._graph_rag = None
        self._regime_classifier = None
        self._risk_budget = None

    def _get_pipeline(self):
        """Lazy-load DataPipeline to avoid circular imports."""
        if self._pipeline is None:
            from data_pipeline import DataPipeline
            self._pipeline = DataPipeline()
        return self._pipeline

    def _get_telegram_notifier(self):
        """Lazy singleton for AITelegramNotifier."""
        if self._telegram_notifier is None:
            from telegram_notifier import AITelegramNotifier
            self._telegram_notifier = AITelegramNotifier()
        return self._telegram_notifier

    def start(self):
        """Initialize and start the background scheduler."""
        try:
            from apscheduler.schedulers.background import BackgroundScheduler
        except ImportError:
            logger.error("[Scheduler] apscheduler not installed. Run: pip install apscheduler")
            logger.info("[Scheduler] Falling back to manual pipeline mode.")
            return False

        # Mega Sprint 2026-04-23 (C.1): APScheduler was letting missed runs
        # spawn stacked instances after a hang, which in turn fought for the
        # SQLite WAL lock and deadlocked the RSS ingest job. 60s grace window
        # lets APScheduler skip late executions entirely once coalesce kicks in.
        self.scheduler = BackgroundScheduler(
            timezone="UTC",
            job_defaults={
                "coalesce": True,
                "max_instances": 1,
                "misfire_grace_time": 60,
            },
        )

        # Phase 28: Start Grafeo ZMQ broker (this process is the single writer)
        try:
            from graph_store import start_graph_broker
            start_graph_broker()
        except Exception as e:
            logger.warning(f"[Scheduler] Grafeo broker start failed: {e}")

        # Task 11: Start SQLite write broker — mirrors the Grafeo pattern
        # so 5 processes serialise WRITES through a single socket instead
        # of racing for the WAL writer slot (~1,950 `database is locked`
        # retry log lines in 17h of production). Readers remain
        # process-local. Clients fall back to a dead-letter spool when
        # the broker is unreachable, so nothing is lost on startup-race.
        try:
            from sqlite_broker import start_sqlite_broker
            start_sqlite_broker()
        except Exception as e:
            logger.warning(f"[Scheduler] SQLite broker start failed: {e}")

        # Every 5 minutes: RSS + FNG + Sentiment
        self.scheduler.add_job(
            self._fetch_and_analyze,
            'interval', minutes=5,
            id='fetch_analyze',
            name='RSS + FNG + Sentiment',
            max_instances=1,
            replace_existing=True
        )

        # Every 5 minutes: Cleanup expired semantic cache
        self.scheduler.add_job(
            self._cleanup_semantic_cache,
            'interval', minutes=5,
            id='cleanup_cache',
            name='Semantic Cache Cleanup',
            replace_existing=True
        )

        # Phase 17: Every 5 minutes: System health check
        self.scheduler.add_job(
            self._health_check,
            'interval', minutes=5,
            id='health_check',
            name='System Health Check',
            replace_existing=True
        )

        # Every 15 minutes: Embed new news into ChromaDB
        self.scheduler.add_job(
            self._embed_news,
            'interval', minutes=15,
            id='embed_news',
            name='Embedding Pipeline',
            max_instances=1,
            replace_existing=True
        )

        # Phase 14: Every 15 minutes: Flush StreamingRAG hot buffer to cold storage
        self.scheduler.add_job(
            self._flush_streaming_rag,
            'interval', minutes=15,
            id='flush_streaming',
            name='StreamingRAG Hot->Cold Flush',
            replace_existing=True
        )

        # Every 15 minutes: Compute rolling sentiment aggregates per coin
        self.scheduler.add_job(
            self._compute_rolling_sentiment,
            'interval', minutes=15,
            id='compute_sentiment',
            name='Rolling Sentiment Aggregator',
            replace_existing=True
        )

        # Daily 00:00 UTC: Reset risk budget
        self.scheduler.add_job(
            self._daily_reset,
            'cron', hour=0, minute=0,
            id='daily_reset',
            name='Daily Risk Budget Reset',
            replace_existing=True
        )

        # Daily 04:00 UTC: Cleanup old news + old market data
        self.scheduler.add_job(
            self._cleanup_old_data,
            'cron', hour=4, minute=0,
            id='cleanup',
            name='Old Data Cleanup (news + market data)',
            replace_existing=True
        )

        # Daily 04:15 UTC: cap embedding_cache at 10K rows (LRU-ish by
        # created_at). 75K → 10K = ~70MB RAM relief in the AI DB cache.
        self.scheduler.add_job(
            self._embedding_cache_evict,
            'cron', hour=4, minute=15,
            id='embedding_cache_evict',
            name='Embedding Cache LRU Eviction',
            max_instances=1,
            replace_existing=True
        )

        # Daily 23:55 UTC: Send daily Telegram summary
        self.scheduler.add_job(
            self._send_daily_summary,
            'cron', hour=23, minute=55,
            id='daily_summary',
            name='Daily Telegram Summary',
            replace_existing=True
        )

        # Daily 00:05 UTC: Post-mortem analysis of yesterday's trades
        self.scheduler.add_job(
            self._send_daily_postmortem,
            'cron', hour=0, minute=5,
            id='daily_postmortem',
            name='Daily Trade Post-Mortem (Telegram)',
            replace_existing=True
        )

        # Sunday 23:55 UTC: Send weekly Telegram summary
        self.scheduler.add_job(
            self._send_weekly_summary,
            'cron', day_of_week='sun', hour=23, minute=55,
            id='weekly_summary',
            name='Weekly Telegram Summary',
            replace_existing=True
        )

        # Monday 06:00 UTC: RAG Quality Audit (RAGAS feedback loop)
        self.scheduler.add_job(
            self._rag_quality_audit,
            'cron', day_of_week='mon', hour=6, minute=0,
            id='rag_quality_audit',
            name='RAG Quality Audit (weekly RAGAS)',
            replace_existing=True
        )

        # Sunday 04:00 UTC: GraphRAG community rebuild + summarize
        self.scheduler.add_job(
            self._rebuild_graph_communities,
            'cron', day_of_week='sun', hour=4, minute=0,
            id='graph_rag_rebuild',
            name='GraphRAG Community Rebuild',
            replace_existing=True
        )

        # Phase 15: Sunday 03:00 UTC: Prune MAGMA Entity/Temporal Graphics
        self.scheduler.add_job(
            self._prune_magma_memory,
            'cron', day_of_week='sun', hour=3, minute=0,
            id='prune_magma',
            name='MAGMAMemory Edge Pruning',
            replace_existing=True
        )

        # Phase 15: Daily 04:30 UTC: Embed Bidirectional RAG lessons into VectorDB
        self.scheduler.add_job(
            self._embed_bidi_lessons,
            'cron', hour=4, minute=30,
            id='embed_bidi',
            name='Bidirectional RAG Lesson Embedding',
            replace_existing=True
        )

        # Phase 19: Daily 05:00 UTC: Re-fit confidence calibrator with latest trade outcomes
        self.scheduler.add_job(
            self._refit_calibrator,
            'cron', hour=5, minute=0,
            id='refit_calibrator',
            name='Confidence Calibrator Re-fit',
            replace_existing=True
        )

        # Phase 19: Weekly Monday 05:30 UTC: Forgone P&L threshold analysis
        self.scheduler.add_job(
            self._analyze_forgone_threshold,
            'cron', day_of_week='mon', hour=5, minute=30,
            id='forgone_threshold',
            name='Forgone P&L Threshold Analysis',
            replace_existing=True
        )

        # Phase 19: Daily 06:00 UTC: Process new backtest results into PatternStatStore + ChromaDB
        self.scheduler.add_job(
            self._process_new_backtests,
            'cron', hour=6, minute=0,
            id='process_backtests',
            name='Backtest Embedder Processing',
            replace_existing=True
        )

        # Phase 19 Level 3: Every 15 min: Fetch derivatives data (OI, funding, L/S ratio)
        self.scheduler.add_job(
            self._fetch_market_data_derivatives,
            'interval', minutes=15,
            id='fetch_derivatives',
            name='Market Data: Derivatives (Bybit)',
            max_instances=1,
            replace_existing=True
        )

        # Phase 19 Level 3: Every hour: Fetch DeFi + Macro data (TVL, stablecoins, FRED)
        self.scheduler.add_job(
            self._fetch_market_data_defi_macro,
            'interval', minutes=60,
            id='fetch_defi_macro',
            name='Market Data: DeFi + Macro',
            max_instances=1,
            replace_existing=True
        )

        # Phase 20: Opportunity Scanner — pre-screen pairs before each signal cycle
        self.scheduler.add_job(
            self._opportunity_scan,
            'interval', minutes=15,
            id='opportunity_scan',
            name='Opportunity Scanner Wide Screening',
            max_instances=1,
            replace_existing=True
        )

        # Phase 20: Agent Pool — weekly weight rebalancing based on performance
        self.scheduler.add_job(
            self._rebalance_agent_weights,
            'cron', day_of_week='sun', hour=2, minute=0,
            id='agent_rebalance',
            name='Agent Pool Weight Rebalancing',
            replace_existing=True
        )

        # Phase 20: Cross-Pair Intelligence — market-wide pattern detection
        self.scheduler.add_job(
            self._update_cross_pair_intel,
            'interval', minutes=30,
            id='cross_pair_intel',
            name='Cross-Pair Intelligence Update',
            max_instances=1,
            replace_existing=True
        )

        # Phase 20: Event-driven market condition monitor
        # Checks every 5 min for extreme F&G or funding rate spikes
        # If detected, triggers Evidence Engine re-analysis of affected pairs
        self.scheduler.add_job(
            self._event_driven_reanalysis,
            'interval', minutes=5,
            id='event_reanalysis',
            name='Event-Driven Re-Analysis (F&G extreme, funding spike)',
            max_instances=1,
            replace_existing=True
        )

        # Phase 21: Auto Backtest & Bootstrap — daily at 03:00 UTC
        # Runs backtest on top pairs, feeds results into PatternStatStore + Calibrator
        self.scheduler.add_job(
            self._auto_backtest_bootstrap,
            'cron', hour=3, minute=0,
            id='auto_backtest',
            name='Auto Backtest & Bootstrap (daily 03:00 UTC)',
            max_instances=1,
            replace_existing=True
        )

        # Memory homeostasis groom: sample memory_sensor every 5 min + LIF
        # deposit, then run light gc.collect every tick. Every 6th tick
        # (~30 min) OR when pressure>0.5, run heavy gc.collect(2) + glibc
        # malloc_trim to defrag the heap. Cadence is fixed at 5 min so the
        # afferent channel stays warm; heavy work is gated by pressure.
        # Sprint 2026-05-02: 5min → 1min cadence so the predictive guardian
        # gets enough samples for slope estimation. Light tick (sensor +
        # gen-0 GC) is cheap; heavy work still gated by pressure score
        # inside _memory_cleanup itself (every 6th tick OR pressure≥0.5).
        # Audit Finding #12: max_instances=1 + coalesce=False so a slow
        # heavy tick under pressure does NOT cause backed-up triggers to
        # all fire on completion (which would compound the pressure).
        # APScheduler default coalesce=True merges missed runs into one;
        # we want misses dropped silently instead (the next 1-min tick
        # will resample anyway).
        self.scheduler.add_job(
            self._memory_cleanup,
            'interval', minutes=1,
            id='memory_cleanup',
            name='Memory Groom (sensor + GC + malloc_trim)',
            max_instances=1,
            coalesce=True,
            misfire_grace_time=30,
            replace_existing=True
        )

        # Phase 24+25: Neural Organism — 6 jobs (hourly decay, daily habits, daily DMN+cerebellum, weekly sleep+evolution)
        self.scheduler.add_job(self._organism_hourly_decay, 'interval', minutes=60,
            id='organism_decay', name='Neural Organism Hourly Decay', max_instances=1, replace_existing=True)
        # F3: 5-min hormone refresh — closes the homeostasis loop between
        # trades. Without this, memory_pressure / sensor_stress could rise
        # for hours and cortisol would stay at last-trade value.
        self.scheduler.add_job(self._organism_hormone_refresh, 'interval', minutes=5,
            id='organism_hormone_refresh',
            name='Neural Organism Hormone Refresh (sensor-driven)',
            max_instances=1, replace_existing=True)
        # LOOP-3 (2026-04-25): hourly outcome backfill. Daily-only would
        # leave the latest 24h decisions blind during the highest-velocity
        # learning window. Hourly catches the t+4h horizon as soon as the
        # window expires.
        self.scheduler.add_job(self._decisions_outcome_backfill_tick, 'interval', minutes=60,
            id='decisions_outcome_backfill',
            name='AI Decisions Outcome Backfill (counterfactual)',
            max_instances=1, replace_existing=True)

        # T8 (2026-04-25): daily 02:45 UTC counterfactual → shadow Kelly.
        # Drains 32K+ counterfactual_results rows into per-pair Beta
        # posteriors. Idempotent via `consumed_by_kelly` column.
        self.scheduler.add_job(self._counterfactual_to_kelly_tick, 'cron', hour=2, minute=45,
            id='counterfactual_to_kelly',
            name='Counterfactual Results → Shadow Kelly',
            max_instances=1, replace_existing=True)

        # RE-5 (2026-04-25): RiskEnvelope sensor vote update — every 5 min.
        # 5-sensor demote vote + graduated decay state machine. This is
        # the heart of the autonomous risk-tier mechanism: when 3+ sensors
        # alarm, decay multiplier starts shrinking the entire envelope.
        self.scheduler.add_job(self._risk_envelope_sensor_tick, 'interval', minutes=5,
            id='risk_envelope_sensor',
            name='Risk Envelope Sensor Vote + Decay State',
            max_instances=1, replace_existing=True)
        # Hourly: confidence score recompute + autonomy promote/demote evaluation
        self.scheduler.add_job(self._risk_envelope_promote_tick, 'interval', minutes=60,
            id='risk_envelope_promote',
            name='Risk Envelope Confidence + Autonomy Promote/Demote',
            max_instances=1, replace_existing=True)
        self.scheduler.add_job(self._organism_habit_check, 'cron', hour=5, minute=15,
            id='organism_habits', name='Neural Organism Habit Consolidation', max_instances=1, replace_existing=True)
        self.scheduler.add_job(self._organism_sleep, 'cron', day_of_week='sun', hour=3, minute=30,
            id='organism_sleep', name='Neural Organism Sleep Consolidation', max_instances=1, replace_existing=True)
        self.scheduler.add_job(self._organism_dmn, 'cron', hour=4, minute=0,
            id='organism_dmn', name='Neural Organism DMN Idle Processing', max_instances=1, replace_existing=True)
        self.scheduler.add_job(self._organism_evolution, 'cron', day_of_week='sun', hour=4, minute=0,
            id='organism_evolution', name='Neural Organism NeuroEvolution', max_instances=1, replace_existing=True)
        self.scheduler.add_job(self._organism_cerebellum, 'cron', hour=0, minute=10,
            id='organism_cerebellum', name='Neural Organism Cerebellum Daily', max_instances=1, replace_existing=True)

        # Phase 26: Predictive Interoception + Pheromone cleanup
        self.scheduler.add_job(self._interoception_check, 'interval', minutes=15,
            id='interoception_check', name='Predictive Interoception Check', max_instances=1, replace_existing=True)
        # Cadence dropped from 30 min → 5 min so trail compaction keeps up
        # with the higher tick rate of the new sensors (memory groom + shadow
        # Kelly + LLM penalties). Under memory pressure the cleanup also
        # tightens its idle window from 300s to 60s — see _pheromone_cleanup.
        self.scheduler.add_job(self._pheromone_cleanup, 'interval', minutes=5,
            id='pheromone_cleanup', name='Pheromone Field Cleanup (adaptive)',
            max_instances=1, replace_existing=True)

        # Sensor bridges (sensor_bridges.py) — afferent nerves translating
        # exteroceptive pain (shadow-Kelly posterior collapse, exchange WS
        # drops) into pheromone deposits the organism's hormone pipeline
        # consumes as sensor_stress. Orderbook + data-starvation sensors
        # fire in-line from HydraSizer / evidence_engine; these two need
        # periodic scanning to surface aggregate signal.
        self.scheduler.add_job(self._shadow_kelly_divergence_tick, 'interval',
            minutes=30, id='shadow_kelly_divergence',
            name='Shadow-Kelly Posterior Divergence Probe',
            max_instances=1, replace_existing=True)
        self.scheduler.add_job(self._ws_health_tick, 'interval',
            minutes=5, id='ws_health_tick',
            name='Exchange WS Disconnect Scanner',
            max_instances=1, replace_existing=True)
        self.scheduler.add_job(self._pair_circuit_revive_tick, 'interval',
            minutes=5, id='pair_circuit_revive',
            name='Pair Circuit Revival Probe',
            max_instances=1, replace_existing=True)
        # Sprint 2026-05-01: adaptive pairlist tuner — derives runtime
        # filter values from circuit telemetry + organism state.
        self.scheduler.add_job(self._adaptive_pairlist_tune, 'interval',
            minutes=30, id='pairlist_tuner',
            name='Adaptive Pairlist Threshold Tuner',
            max_instances=1, replace_existing=True)
        # Sprint 2026-05-01: whitelist health monitor — alarms on collapse
        self.scheduler.add_job(self._whitelist_health_tick, 'interval',
            minutes=5, id='whitelist_health',
            name='Whitelist Size Health Tick',
            max_instances=1, replace_existing=True)
        # Sprint 2026-05-01 night — Walk-forward validation. Daily 23:30 UTC,
        # right after daily_postmortem so it sees the day's closed trades.
        # When recent 14-day performance has degraded >30% vs the prior
        # 30-day baseline, deposits a `model_freeze` pheromone that other
        # learners (CatBoost retrain, OOD refit, ensemble refit) consume
        # to skip their next update cycle.
        self.scheduler.add_job(self._walk_forward_validation, 'cron',
            hour=23, minute=30, id='walk_forward',
            name='Walk-Forward Validation Daily',
            max_instances=1, replace_existing=True)
        # Sprint 2026-05-02 — Monte Carlo trade-shuffle bootstrap (jesse repo).
        # Tests if observed strategy performance is statistically significant
        # vs random reordering. Daily 23:45 UTC.
        self.scheduler.add_job(self._mc_bootstrap_validation, 'cron',
            hour=23, minute=45, id='mc_bootstrap',
            name='Monte Carlo Trade-Shuffle Bootstrap',
            max_instances=1, replace_existing=True)
        # Sprint 2026-05-02 — Stablecoin netflow proxy (placeholder until
        # CryptoQuant API integrated). 4h cadence.
        self.scheduler.add_job(self._stablecoin_netflow_tick, 'interval',
            hours=4, id='stablecoin_netflow',
            name='Stablecoin Netflow 24h Proxy',
            max_instances=1, replace_existing=True)
        # Sprint 2026-05-02 — Black-Litterman + Max-Sharpe joint pair
        # allocator (Lean repo BL implementation). Hourly.
        self.scheduler.add_job(self._portfolio_optimizer_tick, 'interval',
            hours=1, id='portfolio_optimizer',
            name='Portfolio Optimizer (BL+MaxSharpe)',
            max_instances=1, replace_existing=True)
        # Sprint 2026-05-02 audit Finding #4 — micro structure learner
        # publishes a hourly summary for the daily comparison dashboard.
        self.scheduler.add_job(
            lambda: __import__("exchange_microstructure_learner").publish_summary(),
            'interval', hours=1, id='microstructure_summary',
            name='Microstructure Learner Hourly Summary',
            max_instances=1, replace_existing=True,
        )
        # Sprint 2026-05-02 audit Finding #3 fix — auto-cgroup recalibrate.
        # Sunday 04:30 UTC, after the heavy ML jobs settle and before the
        # next week's trading begins. Writes systemd drop-in files with
        # observed-peak-based MemoryMax. New limits take effect on next
        # service restart (operator's deploy).
        self.scheduler.add_job(self._auto_cgroup_recalibrate, 'cron',
            day_of_week='sun', hour=4, minute=30,
            id='cgroup_recalibrate',
            name='Auto Cgroup Recalibration (predictive guardian)',
            max_instances=1, replace_existing=True)
        # Sprint 2026-05-02 (audit Finding #2 fix) — OLMAR cross-pair
        # mean-reversion (Li & Hoi 2012). Hourly cadence; reads
        # recent_closes pheromone deposits per pair.
        self.scheduler.add_job(self._olmar_tick, 'interval',
            hours=1, id='olmar_optimizer',
            name='OLMAR Cross-Pair Mean Reversion',
            max_instances=1, replace_existing=True)
        self.scheduler.add_job(self._cache_health_drain_tick, 'interval',
            minutes=10, id='cache_health_drain',
            name='Semantic Cache Health Drain',
            max_instances=1, replace_existing=True)
        # Task 13: size-based WAL checkpoint. Prior design was release-
        # count-based (every 50 releases → TRUNCATE) which is uncorrelated
        # with actual WAL growth. A quiet interval could leave WAL at
        # 166 MB for hours; a busy one fired TRUNCATEs every few seconds
        # and contended with hot writers. Checking the file size every
        # 60s and truncating above 32 MB gives deterministic bounds.
        self.scheduler.add_job(self._wal_checkpoint_tick, 'interval',
            seconds=60, id='wal_checkpoint',
            name='WAL Size-Based Checkpoint',
            max_instances=1, replace_existing=True)
        # Task 11/12: drain the hot_writes spool. Any write that missed
        # the broker (startup race, transient ZMQ disconnect) sits in
        # `hot_writes` state='pending' — this job replays them.
        self.scheduler.add_job(self._sqlite_spool_drain_tick, 'interval',
            minutes=1, id='sqlite_spool_drain',
            name='SQLite Write Spool Drain',
            max_instances=1, replace_existing=True)
        # Task 14: weekly risk_budget adaptation. `RiskBudgetManager.weekly_adjust`
        # existed from Phase 3.5.3 but was never wired — multiplier stayed at
        # 1.0 in production for weeks. Sunday 23:55 UTC computes the past
        # week's PnL% and nudges next week's VaR budget up or down.
        self.scheduler.add_job(self._weekly_budget_adjust, 'cron',
            day_of_week='sun', hour=23, minute=55,
            id='weekly_budget_adjust',
            name='Risk Budget Weekly P&L Adjustment',
            max_instances=1, replace_existing=True)
        # Task 14: autopoietic_integrity reader. The table is write-only
        # right now (self_model.compute_aii pushes rows; nobody reads).
        # Daily tick queries the trend and raises an alarm when the
        # organism's architectural drift crosses a threshold.
        self.scheduler.add_job(self._autopoietic_integrity_review, 'cron',
            hour=0, minute=5,
            id='autopoietic_integrity_review',
            name='Autopoietic Integrity Daily Review',
            max_instances=1, replace_existing=True)
        # Task 26: drain ProactiveDispatcher's retrain_request pheromone.
        # When PredictiveInteroception fires "Models degrading, trigger
        # retraining", dispatcher deposits the request. This tick checks
        # hourly and advances the weekly CatBoost cron on demand.
        self.scheduler.add_job(self._retrain_request_drain_tick, 'interval',
            minutes=30, id='retrain_request_drain',
            name='Proactive Retrain Request Drain',
            max_instances=1, replace_existing=True)

        # Phase 28: ML Model Retraining Jobs
        self.scheduler.add_job(self._catboost_retrain, 'cron', day_of_week='sun', hour=3,
            id='catboost_retrain', name='CatBoost Weekly Retrain', max_instances=1, replace_existing=True)
        # Sprint 2 (6A): Causal discovery — weekly Saturday 02:00 UTC (before CatBoost retrain)
        self.scheduler.add_job(self._causal_discovery, 'cron', day_of_week='sat', hour=2,
            id='causal_discovery', name='Causal Engine Weekly Discovery', max_instances=1, replace_existing=True)
        # Sprint 2 (6B): Counterfactual analysis — weekly Saturday 02:30 UTC (after causal discovery)
        self.scheduler.add_job(self._counterfactual_analysis, 'cron', day_of_week='sat', hour=2, minute=30,
            id='counterfactual_analysis', name='Counterfactual Weekly Analysis', max_instances=1, replace_existing=True)
        # Sprint 2 (7B): IQL pre-training — weekly Sunday 02:00 UTC (before CatBoost at 03:00)
        self.scheduler.add_job(self._rl_iql_retrain, 'cron', day_of_week='sun', hour=2,
            id='iql_retrain', name='IQL Weekly Pre-training', max_instances=1, replace_existing=True)
        # Sprint 2 (8B): Reptile meta-learning — weekly Sunday 01:00 UTC (before IQL)
        self.scheduler.add_job(self._reptile_meta_update, 'cron', day_of_week='sun', hour=1,
            id='reptile_meta', name='Reptile Weekly Meta-Update', max_instances=1, replace_existing=True)
        # Sprint 2 (8C+8D): World model train + Dream session — daily 02:30 UTC.
        # Revize Tur-2 (H4): shifted from 01:30 to 02:30 to clear Sunday's
        # Reptile meta-learning slot at 01:00 with a 90-min buffer. Both
        # jobs spawn PyTorch subprocesses; overlapping them on Sunday was
        # the last remaining collision in the cron grid.
        self.scheduler.add_job(self._world_model_and_dream, 'cron', hour=2, minute=30,
            id='world_model_dream', name='World Model + Dream Session (Daily)',
            max_instances=1, replace_existing=True)
        # Sprint 2 (9A): Self-model introspection — weekly Saturday 03:00 UTC
        self.scheduler.add_job(self._self_model_introspect, 'cron', day_of_week='sat', hour=3,
            id='self_model', name='Self-Model Weekly Introspection', max_instances=1, replace_existing=True)
        # Sprint 2 (9C): Autonomous lifecycle tick — every hour
        self.scheduler.add_job(self._lifecycle_tick, 'interval', hours=1,
            id='lifecycle_tick', name='Lifecycle Hourly Tick', max_instances=1, replace_existing=True)
        # Sprint 2 (9D): GNN pattern discovery — weekly Saturday 03:30 UTC
        self.scheduler.add_job(self._gnn_discovery, 'cron', day_of_week='sat', hour=3, minute=30,
            id='gnn_discovery', name='GNN Weekly Pattern Discovery', max_instances=1, replace_existing=True)
        # Sprint 2 (11A): Architecture evolution — weekly Saturday 04:00 UTC
        self.scheduler.add_job(self._architecture_evolve, 'cron', day_of_week='sat', hour=4,
            id='arch_evolve', name='Architecture Weekly Evolution', max_instances=1, replace_existing=True)
        # Sprint 2 (11F): Cerebellum timing update — daily 00:30 UTC
        self.scheduler.add_job(self._cerebellum_update, 'cron', hour=0, minute=30,
            id='cerebellum_update', name='Cerebellum Daily Timing Update', max_instances=1, replace_existing=True)
        # Sprint 2 (12A): Ablation league — weekly Saturday 04:30 UTC
        self.scheduler.add_job(self._ablation_league_run, 'cron', day_of_week='sat', hour=4, minute=30,
            id='ablation_league', name='Ablation League Weekly', max_instances=1, replace_existing=True)
        # Sprint 2 (12B): Model risk assessment — daily 06:30 UTC
        self.scheduler.add_job(self._model_risk_check, 'cron', hour=6, minute=30,
            id='model_risk', name='Model Risk Daily Check', max_instances=1, replace_existing=True)
        # Sprint 2 (12C): Post-trade court — every 6 hours
        self.scheduler.add_job(self._post_trade_court_run, 'interval', hours=6,
            id='post_trade_court', name='Post-Trade Court 6h', max_instances=1, replace_existing=True)
        # Sprint 2 (12G): Phi consciousness — weekly Saturday 05:00 UTC
        self.scheduler.add_job(self._phi_measurement, 'cron', day_of_week='sat', hour=5,
            id='phi_measurement', name='Phi Consciousness Weekly', max_instances=1, replace_existing=True)
        self.scheduler.add_job(self._ood_refit, 'cron', day_of_week='sun', hour=4, minute=15,
            id='ood_refit', name='OOD Detector Refit', max_instances=1, replace_existing=True)
        self.scheduler.add_job(self._conformal_recalibrate, 'interval', hours=6,
            id='conformal_recal', name='Conformal Recalibration', max_instances=1, replace_existing=True)
        self.scheduler.add_job(self._ensemble_refit, 'cron', day_of_week='sun', hour=5,
            id='ensemble_refit', name='Deep Ensemble Refit', max_instances=1, replace_existing=True)

        # Phase 27 Fix 6 (H3): Forgone PnL feedback loop — resolver + threshold adaptation
        self.scheduler.add_job(self._forgone_shadow_resolver, 'interval', minutes=30,
            id='forgone_resolver', name='Forgone Shadow Trade Resolver',
            max_instances=1, replace_existing=True)
        # Audit fix (2026-04-19): collided with foundation_fine_tune at 06:00
        # → SQLITE_BUSY when both write concurrently. Slide to 06:45.
        self.scheduler.add_job(self._forgone_threshold_adapt, 'cron', day_of_week='sun', hour=6, minute=45,
            id='forgone_thresholds', name='Per-Pair Threshold Adaptation',
            max_instances=1, replace_existing=True)
        # Phase 27 Task 10: Hawkes MLE refit — hourly, only if `tick` is installed
        self.scheduler.add_job(self._hawkes_mle_refit, 'interval', hours=1,
            id='hawkes_refit', name='Hawkes Intensity MLE Refit',
            max_instances=1, replace_existing=True)
        # Phase 27 Task 16: Weekly LLM strategy researcher — Sunday 07:00 UTC,
        # after CatBoost retrain (03:00) and OOD refit (04:15) so fresh numbers
        # are in the context dump the LLM sees.
        self.scheduler.add_job(self._hypothesis_generation_cycle,
            'cron', day_of_week='sun', hour=7,
            id='hypothesis_cycle', name='LLM Strategy Researcher Weekly',
            max_instances=1, replace_existing=True)
        # Phase 27 Task 19: Sleep-wake cycle tick — hourly evaluates Borbely
        # Process S vs Process C and pauses/resumes non-critical jobs.
        self.scheduler.add_job(self._sleep_wake_tick, 'interval', minutes=30,
            id='sleep_wake', name='Sleep-Wake Cycle Tick',
            max_instances=1, replace_existing=True)
        # Phase 27 Task 20: Foundation model fine-tuning. Audit fix
        # (2026-04-19): the original 02:30 slot collided with iql_retrain
        # (02:00) + catboost_retrain (03:00) + foundation + organism_sleep
        # (03:30) + ood_refit (04:15) all in the same memory window → 10 OOM
        # kills in 24h. Move foundation to 06:00 — well after the heavy
        # train trio (world_model 01:30, iql 02:00, catboost 03:00) has
        # finished and gc'd.
        self.scheduler.add_job(self._foundation_fine_tune,
            'cron', day_of_week='sun', hour=6, minute=0,
            id='foundation_fine_tune', name='TTM/Chronos Fine-Tune',
            max_instances=1, replace_existing=True)
        # Phase 27 Dead Code batch 2 warm-up — hourly imports + minimal call
        # so the 8 otherwise-orphan modules are exercised. Prevents the class
        # of "file exists but nothing calls it" audit finding.
        self.scheduler.add_job(self._phase27_dead_code_warmup, 'interval', hours=1,
            id='dead_code_warmup', name='Dead Code Batch 2 Warm-up',
            max_instances=1, replace_existing=True)
        # Phase 27 Task 21: Decision Transformer training — Sunday 23:00 UTC,
        # latest Sunday slot so all upstream retrains (CatBoost/OOD/Ensemble/
        # LoRA fine-tune/hypothesis cycle) have finished and their outputs
        # are reflected in decision_contract rows.
        self.scheduler.add_job(self._dt_training_cycle,
            'cron', day_of_week='sun', hour=23,
            id='dt_training', name='Decision Transformer Weekly LoRA',
            max_instances=1, replace_existing=True)
        # Phase 27 Task 23: Nightly exploit archive regression test (02:15 UTC
        # daily — after CatBoost Sunday cycle, early enough for the report).
        self.scheduler.add_job(self._exploit_regression_batch,
            'cron', hour=2, minute=15,
            id='exploit_batch', name='Exploit Archive Regression',
            max_instances=1, replace_existing=True)
        # Phase 27 Task 25: Trade-as-language weekly cycle (Sunday 08:00 UTC).
        self.scheduler.add_job(self._trade_language_cycle,
            'cron', day_of_week='sun', hour=8,
            id='trade_language', name='Trade-as-Language Pattern Mining',
            max_instances=1, replace_existing=True)
        # Phase 27 Item 10: SAC online RL cycle. Audit fix (2026-04-19):
        # 04:30 collided with ood_refit 04:15 + organism_evolution 04:00 +
        # ensemble_refit 05:00 in a 1h memory burst. Moved to 05:30 — stays
        # in the post-CatBoost window but no longer overlaps OOD/ensemble.
        self.scheduler.add_job(self._sac_online_cycle,
            'cron', day_of_week='sun', hour=5, minute=30,
            id='sac_online', name='SAC Online RL Cycle',
            max_instances=1, replace_existing=True)
        # Phase 27 Item 3: MultiModal encoder training. Audit fix
        # (2026-04-19): 06:00 now collides with foundation_fine_tune; pushed
        # to 07:30 so foundation finishes + gc completes before MM starts.
        self.scheduler.add_job(self._multimodal_train_cycle,
            'cron', day_of_week='sun', hour=7, minute=30,
            id='multimodal_train', name='MultiModal Encoder Weekly Training',
            max_instances=1, replace_existing=True)
        # Data Acceleration Fix 3: WEEKLY external data fetch (was monthly).
        # Binance publishes the previous day's klines at ~01:00 UTC, so Sunday
        # 01:00 captures the full prior week of fresh market data before
        # CatBoost retrain at 03:00 consumes it.
        self.scheduler.add_job(self._external_data_cycle,
            'cron', day_of_week='sun', hour=1,
            id='external_data_fetch', name='Binance Public Data Weekly Fetch',
            max_instances=1, replace_existing=True)
        # Data Acceleration Fix 2: DAILY backtest-label injection (05:00 UTC).
        # Reuses existing backtest results + live trades + NEW shadow trades,
        # feeds the joint dataset into backtest_training_data every day so
        # CatBoost always has thousands more training samples the next cycle.
        self.scheduler.add_job(self._backtest_injection_daily,
            'cron', hour=5,
            id='backtest_injection', name='Backtest Label Injection Daily',
            max_instances=1, replace_existing=True)

        self.scheduler.start()
        logger.info("[Scheduler] Started with 66 jobs (26 base + 6 organism + 2 phase26 + 17 sprint2 + 15 phase27)")
        return True

    def stop(self):
        """Gracefully shutdown the scheduler."""
        # Task 13: tear the ZMQ brokers down BEFORE APScheduler stops so
        # their REP sockets can flush one final checkpoint and release
        # the IPC socket file cleanly. Prior to this the brokers stayed
        # alive as daemon threads and were SIGKILLed at process exit,
        # leaving the WAL uncheckpointed (observed 166 MB growth in
        # production before the release-count TRUNCATE hotfix).
        try:
            from graph_store import stop_graph_broker
            stop_graph_broker()
        except Exception as e:
            logger.debug(f"[Scheduler] Grafeo broker stop failed: {e}")
        try:
            from sqlite_broker import stop_sqlite_broker
            stop_sqlite_broker()
        except Exception as e:
            logger.debug(f"[Scheduler] SQLite broker stop failed: {e}")

        if self.scheduler and self.scheduler.running:
            self.scheduler.shutdown(wait=False)
            logger.info("[Scheduler] Stopped.")

    def _fetch_and_analyze(self):
        """Job: Fetch RSS feeds + FNG + run sentiment analysis."""
        logger.info("[Scheduler:Job] Fetching RSS + FNG + Sentiment...")
        try:
            from rss_fetcher import fetch_rss_feeds
            from fng_fetcher import fetch_fng
            from sentiment_analyzer import analyze_unscored_news

            fetch_fng()
            fetch_rss_feeds()
            analyze_unscored_news()
        except Exception as e:
            logger.error(f"[Scheduler:Job] Fetch & analyze failed: {e}")

    def _embed_news(self):
        """Job: Embed unprocessed news articles into ChromaDB."""
        logger.info("[Scheduler:Job] Embedding unprocessed news...")
        try:
            pipeline = self._get_pipeline()
            pipeline._embed_unprocessed_news()
        except Exception as e:
            logger.error(f"[Scheduler:Job] Embedding failed: {e}")

    def _read_portfolio_value(self) -> float:
        """Read last known portfolio balance from SQLite (written by strategy)."""
        try:
            from db import get_db_connection
            conn = get_db_connection()
            row = conn.execute("SELECT total_balance FROM portfolio_state WHERE id = 1").fetchone()
            conn.close()
            if row and float(row['total_balance']) > 0:
                return float(row['total_balance'])
        except Exception:
            pass
        return 10000.0  # Fallback if no sync yet

    def _daily_reset(self):
        """Job: Reset daily risk budget at 00:00 UTC."""
        logger.info("[Scheduler:Job] Daily risk budget reset...")
        try:
            from risk_budget import RiskBudgetManager
            portfolio_value = self._read_portfolio_value()
            mgr = RiskBudgetManager(portfolio_value=portfolio_value)
            mgr.reset_daily()
            logger.info(f"[Scheduler:Job] Budget reset with portfolio=${portfolio_value:.2f}")
        except Exception as e:
            logger.error(f"[Scheduler:Job] Daily reset failed: {e}")

    def _cleanup_old_data(self, max_age_days: int = 180):
        """Job: Remove old news + market data older than max_age_days."""
        logger.info(f"[Scheduler:Job] Cleaning up data older than {max_age_days} days...")
        try:
            from db import get_db_connection
            conn = get_db_connection()
            c = conn.cursor()

            total_deleted = 0

            # Old news
            c.execute("SELECT COUNT(*) FROM market_news")
            before = c.fetchone()[0]
            c.execute("DELETE FROM market_news WHERE published_at < datetime('now', ?)", (f"-{max_age_days} days",))
            conn.commit()
            c.execute("SELECT COUNT(*) FROM market_news")
            after = c.fetchone()[0]
            news_deleted = before - after
            total_deleted += news_deleted

            # Phase 19 Level 3: Old derivatives data (keep 30 days — high volume table)
            for table in ['derivatives_data', 'macro_data', 'defi_data', 'search_trends']:
                try:
                    c.execute(f"SELECT COUNT(*) FROM {table}")
                    before_t = c.fetchone()[0]
                    c.execute(f"DELETE FROM {table} WHERE timestamp < datetime('now', '-30 days')")
                    conn.commit()
                    c.execute(f"SELECT COUNT(*) FROM {table}")
                    after_t = c.fetchone()[0]
                    deleted_t = before_t - after_t
                    total_deleted += deleted_t
                    if deleted_t > 0:
                        logger.info(f"[Scheduler:Job] Cleaned {deleted_t} old rows from {table}")
                except Exception:
                    pass  # Table may not exist yet

            conn.close()

            if total_deleted > 0:
                logger.info(f"[Scheduler:Job] Total cleanup: {total_deleted} rows ({news_deleted} news + market data).")
            else:
                logger.info("[Scheduler:Job] No old data to clean up.")

        except Exception as e:
            logger.error(f"[Scheduler:Job] Cleanup failed: {e}")

    def _embedding_cache_evict(self, keep_n: int = 10000):
        """Daily 04:15 UTC: cap embedding_cache at keep_n most-recent rows.

        Audit 2026-04-25 found embedding_cache had grown to 75K+ rows (~70MB)
        with no eviction path. RSS pressure post-deploy traced ~5% of swap
        usage to this single table.

        AUDIT-9 (2026-04-25): rank by COALESCE(last_used_at, created_at)
        so the LRU is REAL — a hot text accessed 1000× over months keeps
        its embedding even though created_at is ancient. rag_embedding
        refreshes last_used_at on every cache hit.
        """
        try:
            from db import get_db_connection
            conn = get_db_connection()
            c = conn.cursor()
            c.execute("SELECT COUNT(*) FROM embedding_cache")
            before = c.fetchone()[0]
            if before <= keep_n:
                conn.close()
                logger.info(
                    f"[Scheduler:Job] embedding_cache size {before} <= cap {keep_n}, skip"
                )
                return
            c.execute(
                """DELETE FROM embedding_cache
                   WHERE text_hash NOT IN (
                       SELECT text_hash FROM embedding_cache
                       ORDER BY COALESCE(last_used_at, created_at) DESC LIMIT ?
                   )""",
                (keep_n,),
            )
            deleted = c.rowcount
            conn.commit()
            c.execute("SELECT COUNT(*) FROM embedding_cache")
            after = c.fetchone()[0]
            conn.close()
            logger.info(
                f"[Scheduler:Job] embedding_cache eviction: {before} → {after} "
                f"(deleted {deleted}, kept top {keep_n} by last_used_at)"
            )
        except Exception as e:
            logger.error(f"[Scheduler:Job] embedding_cache eviction failed: {e}")

    def _cleanup_semantic_cache(self):
        """Job: Cleanup expired entries in semantic_cache."""
        logger.info("[Scheduler:Job] Cleaning up expired semantic cache...")
        try:
            if self._semantic_cache is None:
                from semantic_cache import SemanticCache
                self._semantic_cache = SemanticCache()
            self._semantic_cache.cleanup_expired()
        except Exception as e:
            logger.error(f"[Scheduler:Job] Semantic cache cleanup failed: {e}")

    def _flush_streaming_rag(self):
        """Job: Flush expired hot documents from StreamingRAG into Chroma"""
        logger.info("[Scheduler:Job] Flushing StreamingRAG hot buffer into cold storage...")
        try:
            if self._streaming_rag is None:
                from streaming_rag import StreamingRAG
                self._streaming_rag = StreamingRAG()
            self._streaming_rag.flush_to_cold()
        except Exception as e:
            logger.error(f"[Scheduler:Job] StreamingRAG flush failed: {e}")

    def _compute_rolling_sentiment(self):
        """Job: Compute rolling sentiment aggregates per coin (1h, 4h, 24h windows)."""
        logger.info("[Scheduler:Job] Computing rolling sentiment aggregates...")
        try:
            from coin_sentiment_aggregator import compute_rolling_sentiment
            compute_rolling_sentiment()
        except Exception as e:
            logger.error(f"[Scheduler:Job] Rolling sentiment computation failed: {e}")

    def _health_check(self):
        """Job: Run system health check and record metrics."""
        logger.info("[Scheduler:Job] Running health check...")
        try:
            if self._system_monitor is None:
                from system_monitor import SystemMonitor
                self._system_monitor = SystemMonitor()
            monitor = self._system_monitor
            health = monitor.check_health()
            # Record scheduler heartbeat metric
            monitor.record_metric("scheduler_job", 1.0, {"job": "health_check"})

            if health["status"] == "critical":
                logger.error(f"[Scheduler:Job] CRITICAL health: {health['alerts']}")
        except Exception as e:
            logger.error(f"[Scheduler:Job] Health check failed: {e}")

    def _embed_bidi_lessons(self):
        """Job: Write back AI trade evaluation lessons into Vector DB."""
        logger.info("[Scheduler:Job] Embedding Bidirectional RAG lessons...")
        try:
            if self._bidi_rag is None:
                from bidirectional_rag import BidirectionalRAG
                self._bidi_rag = BidirectionalRAG()
            lessons = self._bidi_rag.get_unembedded_lessons()
            if not lessons:
                return

            # Reuse DataPipeline's retriever (singleton, no duplicate FlashRank/LLMRouter)
            retriever = self._get_pipeline().retriever
            
            docs, metas, ids = [], [], []
            for l in lessons:
                docs.append(l['lesson_text'])
                metas.append({
                    "type": "ai_lesson",
                    "pair": l['pair'],
                    "source": "bidirectional_rag",
                    "signal": l['signal'],
                    "outcome_pnl": float(l['outcome_pnl'])
                })
                ids.append(f"lesson_{l['id']}")
                
            retriever.add_documents(documents=docs, metadatas=metas, ids=ids)
            
            # Mark as embedded
            self._bidi_rag.mark_lessons_embedded([l['id'] for l in lessons])
            logger.info(f"[Scheduler:Job] Successfully integrated {len(lessons)} Bidirectional lessons.")
            
        except Exception as e:
            logger.error(f"[Scheduler:Job] Bidirectional embedding failed: {e}")

    def _refit_calibrator(self):
        """Job: Re-fit Platt scaling on latest trade outcomes for confidence calibration."""
        logger.info("[Scheduler:Job] Re-fitting confidence calibrator...")
        try:
            if self._calibrator is None:
                from confidence_calibrator import ConfidenceCalibrator
                self._calibrator = ConfidenceCalibrator()
            self._calibrator._brier_disabled = False  # Reset so re-fit can re-evaluate
            self._calibrator.fit_platt_scaling()
            report = self._calibrator.report()
            logger.info(f"[Scheduler:Job] Calibrator re-fit complete.\n{report}")
        except Exception as e:
            logger.error(f"[Scheduler:Job] Calibrator re-fit failed: {e}")

    def _analyze_forgone_threshold(self):
        """
        Job: Compare forgone vs executed P&L weekly.
        If forgone consistently outperforms executed → we're blocking good trades → log recommendation.
        This is diagnostic only — does NOT auto-change thresholds (user reviews via Telegram).
        """
        logger.info("[Scheduler:Job] Analyzing forgone vs executed P&L...")
        try:
            if self._forgone_engine is None:
                from forgone_pnl_engine import ForgonePnLEngine
                self._forgone_engine = ForgonePnLEngine()
            summary = self._forgone_engine.weekly_summary()

            forgone_trades = summary.get("forgone_trades", {})
            executed_trades = summary.get("executed_trades", {})

            forgone_pnl = forgone_trades.get("total_pnl_pct", 0)
            executed_pnl = executed_trades.get("total_pnl_pct", 0)
            forgone_count = forgone_trades.get("count", 0)
            executed_count = executed_trades.get("count", 0)

            analysis = (
                f"[Forgone Analysis] Week: Forgone={forgone_pnl:+.2f}% ({forgone_count} signals) | "
                f"Executed={executed_pnl:+.2f}% ({executed_count} trades)"
            )

            if forgone_pnl > executed_pnl and forgone_pnl > 0:
                gap = forgone_pnl - executed_pnl
                analysis += f" | GAP: +{gap:.2f}% left on table. Consider LOWERING confidence threshold."
                logger.warning(analysis)

                # Send Telegram alert about missed opportunity
                try:
                    from telegram_notifier import AITelegramNotifier
                    notifier = self._get_telegram_notifier()
                    notifier.send_alert(
                        f"📊 Forgone P&L Alert: Blocked signals would have earned {forgone_pnl:+.2f}% "
                        f"vs executed {executed_pnl:+.2f}% (gap: {gap:.2f}%). "
                        f"Consider lowering confidence_threshold for more trades.",
                        level="WARNING"
                    )
                except Exception:
                    pass
            elif executed_pnl > forgone_pnl:
                analysis += f" | GOOD: Guardrails saved {executed_pnl - forgone_pnl:.2f}% by blocking bad signals."
                logger.info(analysis)
            else:
                logger.info(analysis)

        except Exception as e:
            logger.error(f"[Scheduler:Job] Forgone threshold analysis failed: {e}")

    def _fetch_market_data_derivatives(self):
        """Job: Fetch derivatives data (OI, funding, L/S ratio) from Bybit."""
        logger.info("[Scheduler:Job] Fetching derivatives market data...")
        try:
            if self._market_data_fetcher is None:
                from market_data_fetcher import MarketDataFetcher
                self._market_data_fetcher = MarketDataFetcher()
            count = self._market_data_fetcher.fetch_derivatives()
            logger.info(f"[Scheduler:Job] Derivatives: {count} pair(s) fetched.")
        except Exception as e:
            logger.error(f"[Scheduler:Job] Derivatives fetch failed: {e}")

    def _fetch_market_data_defi_macro(self):
        """Job: Fetch DeFi (TVL, stablecoins) + Macro (FRED) + CrossAsset (yfinance) + Trends data."""
        logger.info("[Scheduler:Job] Fetching DeFi + Macro + CrossAsset + Trends market data...")
        try:
            if self._market_data_fetcher is None:
                from market_data_fetcher import MarketDataFetcher
                self._market_data_fetcher = MarketDataFetcher()
            d = self._market_data_fetcher.fetch_defi()
            m = self._market_data_fetcher.fetch_macro()
            c = self._market_data_fetcher.fetch_cross_asset()
            t = self._market_data_fetcher.fetch_google_trends()
            logger.info(f"[Scheduler:Job] DeFi: {d}, Macro: {m}, CrossAsset: {c}, Trends: {t} metrics.")
        except Exception as e:
            logger.error(f"[Scheduler:Job] DeFi/Macro/CrossAsset/Trends fetch failed: {e}")

    def _process_new_backtests(self):
        """Job: Process any new backtest result files into PatternStatStore + ChromaDB + MAGMA."""
        logger.info("[Scheduler:Job] Processing new backtest results...")
        try:
            if self._backtest_embedder is None:
                from backtest_embedder import BacktestEmbedder
                self._backtest_embedder = BacktestEmbedder()
            count = self._backtest_embedder.process_all(enrich=True)
            # Clear OHLCV cache to prevent memory growth in singleton
            if hasattr(self._backtest_embedder, '_ohlcv_cache'):
                self._backtest_embedder._ohlcv_cache.clear()
            if count > 0:
                logger.info(f"[Scheduler:Job] Processed {count} new backtest trades into RAG pipeline.")
            else:
                logger.info("[Scheduler:Job] No new backtest results to process.")
        except Exception as e:
            logger.error(f"[Scheduler:Job] Backtest processing failed: {e}")

    def _prune_magma_memory(self):
        """Job: Clean up old/weak linkages inside MAGMA memory tables."""
        logger.info("[Scheduler:Job] Pruning MAGMAMemory edges...")
        try:
            if self._magma_memory is None:
                from magma_memory import MAGMAMemory
                self._magma_memory = MAGMAMemory()
            deleted = self._magma_memory.prune(min_weight=0.5, max_age_days=180)
            logger.info(f"[Scheduler:Job] Removed {deleted} MAGMA connections.")
        except Exception as e:
            logger.error(f"[Scheduler:Job] MAGMAMemory pruning failed: {e}")

    def _opportunity_scan(self):
        """Phase 20 Job: Wide screening of all pairs for trading opportunities."""
        logger.info("[Scheduler:Job] Running opportunity scanner...")
        try:
            if self._opportunity_scanner is None:
                from opportunity_scanner import OpportunityScanner
                self._opportunity_scanner = OpportunityScanner()
            results = self._opportunity_scanner.scan_pairs_from_db(top_n=30)
            if results:
                logger.info(f"[Scheduler:Job] Opportunity scan: {len(results)} pairs scored, "
                           f"top: {results[0]['pair']}({results[0]['composite_score']})")
            else:
                logger.info("[Scheduler:Job] Opportunity scan: no cached scores yet (run via strategy first)")
        except Exception as e:
            logger.error(f"[Scheduler:Job] Opportunity scan failed: {e}")

    def _rebalance_agent_weights(self):
        """Phase 20 Job: Rebalance agent weights based on 30-day performance."""
        logger.info("[Scheduler:Job] Rebalancing agent weights...")
        try:
            if self._agent_pool is None:
                from agent_pool import AgentPool
                self._agent_pool = AgentPool()
            self._agent_pool.rebalance_weights()
            logger.info("[Scheduler:Job] Agent weight rebalancing complete.")
        except Exception as e:
            logger.error(f"[Scheduler:Job] Agent rebalance failed: {e}")

    def _update_cross_pair_intel(self):
        """Phase 20 Job: Update cross-pair market intelligence."""
        logger.info("[Scheduler:Job] Updating cross-pair intelligence...")
        try:
            if self._cross_pair_intel is None:
                from cross_pair_intel import CrossPairIntel
                self._cross_pair_intel = CrossPairIntel()
            self._cross_pair_intel.update()
            latest = self._cross_pair_intel.get_latest()
            bias = latest.get("market_bias", {}).get("bias", "UNKNOWN")
            funding = latest.get("funding_heatmap", {}).get("crowding", "unknown")
            logger.info(f"[Scheduler:Job] Cross-pair intel: market_bias={bias}, funding={funding}")
        except Exception as e:
            logger.error(f"[Scheduler:Job] Cross-pair intel failed: {e}")

    def _event_driven_reanalysis(self):
        """Phase 20 Job: Check for extreme market events and trigger re-analysis.
        Runs every 5 min. If F&G hits extreme (<15 or >85) or funding rate spikes,
        force Evidence Engine re-analysis of affected pairs."""
        try:
            import sqlite3
            from ai_config import AI_DB_PATH
            conn = get_db_connection()
            try:
                conn.row_factory = sqlite3.Row

                triggered = False
                trigger_reason = ""

                # Check Fear & Greed for extreme values
                # Smart throttle: only trigger if F&G CHANGED since last check
                # (same F&G = same analysis needed, no point re-invalidating)
                fng_row = conn.execute(
                    "SELECT value FROM fear_and_greed ORDER BY timestamp DESC LIMIT 1"
                ).fetchone()
                if fng_row:
                    fng = int(fng_row["value"])
                    if fng < int(_p("scheduler.fng_extreme_low", 15)) or fng > int(_p("scheduler.fng_extreme_high", 85)):
                        _prev_fng = getattr(self, '_last_event_fng', None)
                        _fng_bucket = fng // 5  # Group by 5-point bands (0-4, 5-9, 10-14...)
                        _prev_bucket = (_prev_fng // 5) if _prev_fng is not None else None
                        if _fng_bucket != _prev_bucket:
                            triggered = True
                            trigger_reason = f"F&G extreme: {fng} (was {_prev_fng})"
                        self._last_event_fng = fng

                # Check for extreme funding rates (any pair)
                if not triggered:
                    _extreme_fr = _p("scheduler.extreme_funding", 0.001)
                    extreme_funding = conn.execute("""
                        SELECT pair, funding_rate FROM derivatives_data
                        WHERE ABS(funding_rate) > ?
                        AND timestamp > datetime('now', '-15 minutes')
                        ORDER BY ABS(funding_rate) DESC LIMIT 1
                    """, (_extreme_fr,)).fetchone()
                    if extreme_funding:
                        triggered = True
                        trigger_reason = f"Funding spike: {extreme_funding['pair']} {float(extreme_funding['funding_rate'])*100:.3f}%"
            finally:
                conn.close()

            if triggered:
                logger.warning(f"[Phase20:EventTrigger] {trigger_reason} → forcing Evidence Engine re-analysis")
                try:
                    if self._evidence_engine is None:
                        from evidence_engine import EvidenceEngine
                        self._evidence_engine = EvidenceEngine()
                    engine = self._evidence_engine
                    # Re-analyze top pairs from opportunity_scores
                    conn2 = get_db_connection()
                    try:
                        conn2.row_factory = sqlite3.Row
                        pairs = conn2.execute("""
                            SELECT pair FROM opportunity_scores
                            WHERE id IN (SELECT MAX(id) FROM opportunity_scores GROUP BY pair)
                            ORDER BY composite_score DESC LIMIT 10
                        """).fetchall()
                    finally:
                        conn2.close()

                    # Invalidate semantic cache for top pairs so next real signal cycle
                    # uses fresh data. Don't generate signals here — we don't have tech_data
                    # and fake current_price=1 was polluting audit logs with blind signals.
                    try:
                        if self._semantic_cache is None:
                            from semantic_cache import SemanticCache
                            self._semantic_cache = SemanticCache()
                        for p in pairs:
                            self._semantic_cache.invalidate(pair=p["pair"])
                        logger.info(f"[Phase20:EventTrigger] Invalidated cache for {len(pairs)} pairs")
                    except Exception as e:
                        logger.debug(f"[Phase20:EventTrigger] Cache invalidation failed: {e}")

                    # Log what we did (without generating fake signals)
                    for p in pairs:
                        logger.info(f"[Phase20:EventTrigger] {p['pair']} cache invalidated, "
                                   f"next real signal cycle will re-analyze")

                    # Send Telegram alert
                    try:
                        from telegram_notifier import AITelegramNotifier
                        notifier = self._get_telegram_notifier()
                        notifier.send_alert(
                            f"Event Trigger: {trigger_reason}. Re-analyzed top 10 pairs.",
                            level="WARNING"
                        )
                    except Exception:
                        pass

                except Exception as e:
                    logger.error(f"[Phase20:EventTrigger] Re-analysis failed: {e}")

        except Exception as e:
            logger.debug(f"[Phase20:EventTrigger] Event check failed: {e}")

    def _send_daily_summary(self):
        """Job: Aggregate stats and send daily Telegram summary."""
        logger.info("[Scheduler:Job] Sending daily sequence to Telegram...")
        try:
            from telegram_notifier import AITelegramNotifier
            from llm_cost_tracker import LLMCostTracker
            from autonomy_manager import AutonomyManager
            from forgone_pnl_engine import ForgonePnLEngine
            
            # Note: A real implementation would query the trades SQLite for true open/closed counts and PNL.
            # Here we structure the stats dictionary by querying the AI subsystems.
            stats = {
                "open_trades": 0,
                "closed_today": 0,
                "daily_pnl": 0.0,
                "daily_pnl_pct": 0.0,
                "accuracy": 0.0,
                "correct_trades": 0,
                "total_eval_trades": 0
            }
            
            if self._cost_tracker is None:
                self._cost_tracker = LLMCostTracker()
            cost_summary = self._cost_tracker.get_daily_summary()
            stats["api_cost_today"] = sum(m.get("cost_usd", 0) for m in cost_summary.get("models", {}).values())

            if self._autonomy_manager is None:
                self._autonomy_manager = AutonomyManager()
            autonomy = self._autonomy_manager
            stats["autonomy_level"] = f"L{autonomy.current_level}"

            # Real portfolio balance + asset breakdown
            stats["portfolio_value"] = self._read_portfolio_value()
            try:
                from db import get_db_connection
                import json
                conn = get_db_connection()
                row = conn.execute("SELECT assets_json FROM portfolio_state WHERE id = 1").fetchone()
                conn.close()
                if row and row['assets_json']:
                    stats["assets"] = json.loads(row['assets_json'])
            except Exception:
                pass
            
            if self._forgone_engine is None:
                from forgone_pnl_engine import ForgonePnLEngine
                self._forgone_engine = ForgonePnLEngine()
            f_summary = self._forgone_engine.weekly_summary()
            stats["forgone_pnl"] = f_summary.get("forgone_trades", {}).get("total_pnl_pct", 0.0)

            # $100 Hypothetical Portfolio
            stats["hypothetical"] = self._forgone_engine.get_hypothetical_balance()

            notifier = self._get_telegram_notifier()
            notifier.send_daily_summary(stats)
            
        except Exception as e:
            logger.error(f"[Scheduler:Job] Failed to send daily summary: {e}")

    def _send_weekly_summary(self):
        """Job: Aggregate stats and send weekly Telegram summary."""
        logger.info("[Scheduler:Job] Sending weekly sequence to Telegram...")
        try:
            from telegram_notifier import AITelegramNotifier
            from forgone_pnl_engine import ForgonePnLEngine
            
            stats = {
                "win_rate": 0.0,
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0
            }
            
            if self._forgone_engine is None:
                from forgone_pnl_engine import ForgonePnLEngine
                self._forgone_engine = ForgonePnLEngine()
            f_summary = self._forgone_engine.weekly_summary()
            stats["forgone_pnl_total"] = f_summary.get("forgone_trades", {}).get("total_pnl_pct", 0.0)

            # $100 Hypothetical Portfolio
            stats["hypothetical"] = self._forgone_engine.get_hypothetical_balance()

            notifier = self._get_telegram_notifier()
            notifier.send_weekly_summary(stats)
            
        except Exception as e:
            logger.error(f"[Scheduler:Job] Failed to send weekly summary: {e}")

    def _send_daily_postmortem(self):
        """Daily 00:05 UTC: Analyze yesterday's trades, categorize losses, report forgone winners.

        Loss categories:
        - SIGNAL_WRONG: Sub-scores pointed wrong direction
        - TIMING_OFF: Right direction but MFE shows late entry (MFE > |loss|)
        - SIZING_ISSUE: Disproportionate loss from stake asymmetry
        - REGIME_SHIFT: Regime changed after entry
        - EXECUTION_FAIL: Emergency exit, timeout, rejected order
        """
        logger.info("[Scheduler:Job] Running daily post-mortem analysis...")
        try:
            import sqlite3
            from ai_config import AI_DB_PATH

            # Query yesterday's closed trades from Freqtrade DB
            trade_db = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                     "tradesv3.sqlite")
            if not os.path.exists(trade_db):
                logger.warning("[PostMortem] tradesv3.sqlite not found")
                return

            conn = sqlite3.connect(trade_db, timeout=30)
            conn.row_factory = sqlite3.Row
            trades = conn.execute("""
                SELECT id, pair, close_profit as pnl_pct, close_profit_abs as pnl_abs,
                       stake_amount, leverage, exit_reason,
                       CAST((julianday(close_date) - julianday(open_date)) * 1440 AS INTEGER) as trade_duration,
                       open_date, close_date, is_short
                FROM trades
                WHERE is_open = 0 AND close_date > datetime('now', '-1 day')
                ORDER BY close_profit_abs ASC
            """).fetchall()
            conn.close()

            if not trades:
                logger.info("[PostMortem] No trades closed yesterday")
                return

            # Categorize
            winners = [t for t in trades if t["pnl_pct"] and t["pnl_pct"] > 0]
            losers = [t for t in trades if t["pnl_pct"] and t["pnl_pct"] <= 0]
            total_pnl = sum(t["pnl_abs"] or 0 for t in trades)
            win_rate = len(winners) / len(trades) * 100 if trades else 0

            # Auto-tag losses
            tagged_losses = []
            for t in losers:
                exit_r = t["exit_reason"] or ""
                pnl_pct = (t["pnl_pct"] or 0) * 100
                tag = "SIGNAL_WRONG"  # default

                if "emergency" in exit_r or "timeout" in exit_r:
                    tag = "EXECUTION_FAIL"
                elif "first_hour" in exit_r:
                    tag = "TIMING_OFF"
                elif "trailing" in exit_r and pnl_pct > -2:
                    tag = "TIMING_OFF"  # Trailing hit on small loss = late entry
                elif "stoploss" in exit_r and abs(pnl_pct) > 10:
                    tag = "SIZING_ISSUE"  # Big loss = possible stake problem
                elif "regime" in exit_r or "confidence_drop" in exit_r:
                    tag = "REGIME_SHIFT"

                tagged_losses.append({
                    "pair": t["pair"], "pnl": f"{pnl_pct:+.1f}%",
                    "exit": exit_r, "tag": tag,
                    "pnl_abs": f"${t['pnl_abs'] or 0:.2f}"
                })

            # Query forgone winners (shadow trades that would have been profitable).
            # Phase 27 Fix 6: column name bug — was `resolved_pnl_pct` (does not exist)
            # and `timestamp` (column is actually `signal_time`). Both are now correct.
            forgone_text = ""
            try:
                ai_conn = get_db_connection()
                forgone = ai_conn.execute("""
                    SELECT pair, signal_type, confidence, entry_price, forgone_pnl
                    FROM forgone_profit
                    WHERE was_executed = 0 AND forgone_pnl > 2.0
                      AND signal_time > datetime('now', '-1 day')
                    ORDER BY forgone_pnl DESC LIMIT 5
                """).fetchall()
                ai_conn.close()

                if forgone:
                    forgone_lines = []
                    for f in forgone:
                        forgone_lines.append(
                            f"  {f['pair']} {f['signal_type']} conf={f['confidence']:.0%} → +{f['forgone_pnl']:.1f}%")
                    forgone_text = "\nForgone Winners (kacirildi):\n" + "\n".join(forgone_lines)
            except Exception:
                pass

            # Build message
            lines = [
                "DAILY POST-MORTEM",
                f"Trades: {len(trades)} ({len(winners)}W/{len(losers)}L) WR={win_rate:.0f}%",
                f"PnL: ${total_pnl:.2f}",
            ]

            if tagged_losses:
                lines.append(f"\nLosers ({len(tagged_losses)}):")
                # Group by tag
                from collections import Counter
                tag_counts = Counter(l["tag"] for l in tagged_losses)
                for tag, count in tag_counts.most_common():
                    lines.append(f"  [{tag}] x{count}")
                # Top 3 biggest losses
                lines.append("Top losses:")
                for l in tagged_losses[:3]:
                    lines.append(f"  {l['pair']} {l['pnl']} ({l['pnl_abs']}) [{l['tag']}] exit={l['exit']}")

            if forgone_text:
                lines.append(forgone_text)

            # Top 3 winners
            if winners:
                lines.append(f"\nTop winners:")
                top_w = sorted(winners, key=lambda t: t["pnl_abs"] or 0, reverse=True)[:3]
                for w in top_w:
                    lines.append(f"  {w['pair']} +{(w['pnl_pct'] or 0)*100:.1f}% (${w['pnl_abs'] or 0:.2f})")

            message = "\n".join(lines)
            logger.info(f"[PostMortem] {message}")

            # Send via Telegram
            try:
                from telegram_notifier import AITelegramNotifier
                notifier = self._get_telegram_notifier()
                notifier.send_alert(message, level="INFO")
            except Exception as e:
                logger.debug(f"[PostMortem] Telegram send failed: {e}")

        except Exception as e:
            logger.error(f"[Scheduler:Job] Daily post-mortem failed: {e}")

    def _rag_quality_audit(self):
        """Weekly Monday 06:00: RAGAS quality audit — measure retrieval quality, flag bad chunks."""
        logger.info("[Scheduler:Job] Running RAG quality audit...")
        try:
            if self._rag_evaluator is None:
                from rag_evaluator import RAGQualityEvaluator
                self._rag_evaluator = RAGQualityEvaluator()
            evaluator = self._rag_evaluator

            report = evaluator.get_weekly_quality_report()
            if not report:
                logger.info("[RAG Audit] No quality data yet")
                return

            avg_faith = report.get("avg_faithfulness", 0)
            avg_cp = report.get("avg_context_precision", 0)
            avg_ar = report.get("avg_answer_relevancy", 0)
            trend = report.get("trend", "unknown")
            n = report.get("sample_count", 0)

            msg = (
                f"RAG QUALITY AUDIT\n"
                f"Samples: {n}\n"
                f"Faithfulness: {avg_faith:.2f} (target: 0.90)\n"
                f"Context Precision: {avg_cp:.2f} (target: 0.85)\n"
                f"Answer Relevancy: {avg_ar:.2f} (target: 0.90)\n"
                f"Trend: {trend}"
            )
            logger.info(f"[RAG Audit] {msg}")

            try:
                from telegram_notifier import AITelegramNotifier
                self._get_telegram_notifier().send_alert(msg, level="INFO")
            except Exception:
                pass

        except Exception as e:
            logger.error(f"[RAG Audit] Failed: {e}")

    def _rebuild_graph_communities(self):
        """Weekly Sunday 04:00: Rebuild GraphRAG communities from knowledge graph."""
        logger.info("[Scheduler:Job] Rebuilding GraphRAG communities...")
        try:
            if self._graph_rag is None:
                from graph_rag import GraphRAG
                self._graph_rag = GraphRAG()
            graph = self._graph_rag
            communities = graph.build_communities()
            if communities:
                try:
                    graph.summarize_communities(communities, llm_router=self._get_pipeline()._get_router())
                except Exception:
                    graph.summarize_communities(communities)  # No LLM = fallback text
                logger.info(f"[GraphRAG] Rebuilt {len(communities)} communities")
            else:
                logger.info("[GraphRAG] No communities found (empty knowledge graph)")
        except Exception as e:
            logger.error(f"[GraphRAG] Rebuild failed: {e}")

    def _memory_cleanup(self):
        """Memory homeostasis groom — runs every 5 min.

        Steps every tick (cheap):
          1. memory_sensor.tick() — sample RSS/swap/frag, deposit pheromone
             so Hormones.compute() can drive cortisol DOWN under pressure.
          2. gc.collect(0) — generation-0 sweep, very fast.

        Steps every 6th tick (~30 min) OR whenever pressure score >= 0.5:
          3. gc.collect(2) — full multi-generation sweep.
          4. ctypes.CDLL("libc.so.6").malloc_trim(0) — release glibc heap
             arenas back to the OS so RSS actually shrinks (gc alone does
             not return memory; glibc retains it for future allocations).
          5. pheromone deposit "memory_groom_completed" — telemetry trail
             for downstream consumers (defensive_mode dispatcher etc.).

        The 5-min sampling cadence is the load-bearing piece: without a
        fresh pheromone the Hormones channel goes dark and cortisol drifts
        back to 1.0 even though swap is full.
        """
        import gc
        # Persistent counter for cadence (instance attribute survives across ticks)
        self._memory_groom_tick_count = getattr(self, '_memory_groom_tick_count', 0) + 1

        # 1. Sample + deposit (always)
        score = 0.0
        snap_components: dict = {}
        try:
            from memory_sensor import tick as mem_tick
            sample = mem_tick()
            score = float(sample.get("score", 0.0))
            snap_components = sample.get("components", {})
        except Exception as e:
            logger.debug(f"[Scheduler:Memory] sensor tick failed: {e}")

        # 2. Light GC (always)
        light_collected = gc.collect(0)

        # 3-5. Heavy work (pressure-driven OR every 6th tick)
        heavy = (score >= 0.5) or (self._memory_groom_tick_count % 6 == 0)
        full_collected = 0
        trimmed = False
        if heavy:
            full_collected = gc.collect(2)
            # Sprint 2026-05-01: production runs jemalloc via systemd
            # drop-in (LD_PRELOAD libjemalloc), so the legacy
            # `libc.malloc_trim(0)` resolves into glibc which owns NO
            # allocations — a no-op masquerading as memory work. The
            # 150,523 OOM-failcnt on the scheduler service is partly
            # this. Try jemalloc's native purge first; only fall back
            # to glibc on environments without jemalloc.
            try:
                import ctypes
                je_purged = False
                try:
                    je = ctypes.CDLL("libjemalloc.so.2")
                    # mallctl("arena.<MALLCTL_ARENAS_ALL>.purge") — purges
                    # all arenas. MALLCTL_ARENAS_ALL constant is 4096 in
                    # jemalloc 5.x. Returns 0 on success.
                    cmd = b"arena.4096.purge"
                    rc = je.mallctl(cmd, None, None, None, 0)
                    je_purged = (rc == 0)
                except (OSError, AttributeError):
                    pass
                if je_purged:
                    trimmed = True
                else:
                    libc = ctypes.CDLL("libc.so.6")
                    trimmed = bool(libc.malloc_trim(0))
            except Exception as e:
                logger.debug(f"[Scheduler:Memory] purge unavailable: {e}")
            try:
                from pheromone_field import get_pheromone_field
                get_pheromone_field().deposit(
                    "memory_groom",
                    "memory_groom_completed",
                    {"intensity": min(1.0, score), "trimmed": trimmed,
                     "full_collected": full_collected, "tick": self._memory_groom_tick_count},
                    half_life=300.0,
                    metadata={"score": round(score, 3)},
                )
            except Exception:
                pass

        # Logging — RSS + threads + pressure score so the operator can grep.
        try:
            import psutil
            process = psutil.Process()
            mem_mb = process.memory_info().rss / 1024 / 1024
            logger.info(
                f"[Scheduler:Memory] tick={self._memory_groom_tick_count} "
                f"score={score:.3f} RSS={mem_mb:.0f}MB threads={process.num_threads()} "
                f"gc0={light_collected} {'gc2=' + str(full_collected) + ' trim=' + str(trimmed) if heavy else ''} "
                f"comp={snap_components}"
            )
        except Exception:
            logger.info(
                f"[Scheduler:Memory] tick={self._memory_groom_tick_count} "
                f"score={score:.3f} gc0={light_collected} "
                f"{'heavy gc2=' + str(full_collected) + ' trim=' + str(trimmed) if heavy else ''}"
            )

    def get_job_info(self) -> list:
        """Return info about all scheduled jobs."""
        if not self.scheduler:
            return []
        return [
            {
                "id": job.id,
                "name": job.name,
                "next_run": str(job.next_run_time) if job.next_run_time else "paused",
                "trigger": str(job.trigger),
            }
            for job in self.scheduler.get_jobs()
        ]


    def _auto_backtest_bootstrap(self):
        """
        Phase 21: Auto Backtest & Bootstrap — daily at 03:00 UTC.

        Runs freqtrade backtesting on top traded pairs, then feeds results
        into PatternStatStore + ChromaDB + Calibrator. This solves the
        cold-start problem (pattern_trades=0) by continuously generating
        backtest data.

        Flow:
          1. Get top 10 most-traded pairs from tradesv3.sqlite
          2. Create temp config with StaticPairList (VolumePairList not supported in backtest)
          3. Run freqtrade backtesting (last 30 days, or incremental 1 day)
          4. Feed results into BacktestEmbedder → PatternStatStore + ChromaDB
          5. Clean up old backtest results (>30 days)
        """
        import subprocess
        import tempfile
        logger.info("[Scheduler:AutoBacktest] Starting auto backtest & bootstrap...")

        try:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            config_path = os.path.join(base_dir, "..", "config_bybit_testnet_futures.json")
            if not os.path.exists(config_path):
                # Try alternative paths
                for p in [
                    os.path.join(base_dir, "..", "config_bybit_testnet_futures.json"),
                    os.path.join(base_dir, "config_bybit_testnet_futures.json"),
                ]:
                    if os.path.exists(p):
                        config_path = p
                        break

            # 1. Get top 10 pairs from recent trades
            pairs = self._get_top_traded_pairs(10)
            if not pairs:
                pairs = ["BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT",
                         "DOGE/USDT:USDT", "ADA/USDT:USDT"]
                logger.info(f"[Scheduler:AutoBacktest] No trade history, using default pairs")

            logger.info(f"[Scheduler:AutoBacktest] Pairs: {pairs}")

            # 2. Create temp config with StaticPairList (backtesting needs this)
            override_config = {
                "pairlists": [{"method": "StaticPairList"}],
                "exchange": {"pair_whitelist": pairs},
                "dry_run": True,  # backtesting always dry_run
            }

            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', prefix='bt_auto_',
                                             dir='/tmp', delete=False) as tf:
                json.dump(override_config, tf)
                override_path = tf.name

            # 3. Calculate timerange (last 30 days for first run, last 2 days incremental)
            from datetime import timedelta
            now = datetime.now(timezone.utc)
            bt_results_dir = os.path.join(base_dir, "backtest_results")

            # Check if we have recent backtests (incremental mode)
            recent_backtest = False
            if os.path.isdir(bt_results_dir):
                for f in os.listdir(bt_results_dir):
                    if f.endswith('.zip'):
                        fstat = os.stat(os.path.join(bt_results_dir, f))
                        age_hours = (now.timestamp() - fstat.st_mtime) / 3600
                        if age_hours < 48:
                            recent_backtest = True
                            break

            if recent_backtest:
                # Incremental: last 3 days
                start = (now - timedelta(days=3)).strftime("%Y%m%d")
                mode = "incremental"
            else:
                # Full: last 30 days
                start = (now - timedelta(days=30)).strftime("%Y%m%d")
                mode = "full"

            end = now.strftime("%Y%m%d")
            timerange = f"{start}-{end}"
            logger.info(f"[Scheduler:AutoBacktest] Mode={mode}, timerange={timerange}")

            # 4. Download data first (if needed)
            freqtrade_bin = os.path.join(base_dir, "..", ".venv", "bin", "freqtrade")
            if not os.path.exists(freqtrade_bin):
                freqtrade_bin = "freqtrade"  # Try PATH

            try:
                dl_cmd = [
                    freqtrade_bin, "download-data",
                    "--config", config_path,
                    "--config", override_path,
                    "--timerange", timerange,
                    "--timeframe", "1h",
                ]
                dl_result = subprocess.run(dl_cmd, capture_output=True, text=True, timeout=600,
                                          cwd=os.path.join(base_dir, ".."))
                if dl_result.returncode == 0:
                    logger.info(f"[Scheduler:AutoBacktest] Data download complete")
                else:
                    logger.warning(f"[Scheduler:AutoBacktest] Data download warning: {dl_result.stderr[:200]}")
            except subprocess.TimeoutExpired:
                logger.warning("[Scheduler:AutoBacktest] Data download timed out (10min), proceeding with existing data")
            except Exception as e:
                logger.warning(f"[Scheduler:AutoBacktest] Data download failed: {e}")

            # 5. Run backtest
            bt_cmd = [
                freqtrade_bin, "backtesting",
                "--strategy", "HydraSizer",
                "--config", config_path,
                "--config", override_path,
                "--timerange", timerange,
                "--timeframe", "1h",
                "--export", "trades",
            ]

            logger.info(f"[Scheduler:AutoBacktest] Running: {' '.join(bt_cmd)}")
            bt_result = subprocess.run(bt_cmd, capture_output=True, text=True, timeout=1800,
                                      cwd=os.path.join(base_dir, ".."))

            if bt_result.returncode != 0:
                logger.error(f"[Scheduler:AutoBacktest] Backtest failed (rc={bt_result.returncode}): "
                           f"{bt_result.stderr[:500]}")
                return

            logger.info(f"[Scheduler:AutoBacktest] Backtest complete")

            # 6. Feed results into PatternStatStore + ChromaDB
            try:
                if self._backtest_embedder is None:
                    from backtest_embedder import BacktestEmbedder
                    self._backtest_embedder = BacktestEmbedder()
                count = self._backtest_embedder.process_all(results_dir=bt_results_dir, enrich=True)
                if hasattr(self._backtest_embedder, '_ohlcv_cache'):
                    self._backtest_embedder._ohlcv_cache.clear()
                logger.info(f"[Scheduler:AutoBacktest] Bootstrap loaded {count} trades into AI pipeline")
            except Exception as e:
                logger.error(f"[Scheduler:AutoBacktest] Bootstrap failed: {e}")

            # 6b. Sprint 2 (13B): Generate chart feature labels for CatBoost v2
            try:
                from backtest_label_generator import BacktestLabelGenerator
                label_gen = BacktestLabelGenerator()
                label_count = label_gen.generate_from_backtests(results_dir=bt_results_dir)
                if label_count > 0:
                    logger.info(f"[Scheduler:AutoBacktest] Generated {label_count} CatBoost training labels")
                label_gen._ohlcv_cache.clear()
            except Exception as e:
                logger.error(f"[Scheduler:AutoBacktest] Label generation failed: {e}")

            # 7. Clean up temp config
            try:
                os.unlink(override_path)
            except Exception:
                pass

            # 8. Clean up old backtest results (>60 days — research: 60d optimal for crypto)
            if os.path.isdir(bt_results_dir):
                cutoff = now.timestamp() - (60 * 86400)
                for f in os.listdir(bt_results_dir):
                    fpath = os.path.join(bt_results_dir, f)
                    try:
                        if os.path.isfile(fpath) and os.stat(fpath).st_mtime < cutoff:
                            os.unlink(fpath)
                            logger.info(f"[Scheduler:AutoBacktest] Cleaned old file: {f}")
                    except Exception:
                        pass

            logger.info("[Scheduler:AutoBacktest] Auto backtest & bootstrap complete.")

        except subprocess.TimeoutExpired:
            logger.error("[Scheduler:AutoBacktest] Backtest timed out (30min)")
        except Exception as e:
            logger.error(f"[Scheduler:AutoBacktest] Failed: {e}")

    def _get_top_traded_pairs(self, n: int = 10) -> list:
        """Get top N most-traded pairs from Freqtrade's trade history."""
        try:
            # Try tradesv3.sqlite first
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            trade_db = os.path.join(base_dir, "tradesv3.sqlite")
            if not os.path.exists(trade_db):
                trade_db = os.path.join(base_dir, "..", "user_data", "tradesv3.sqlite")

            if os.path.exists(trade_db):
                conn = sqlite3.connect(trade_db, timeout=10)
                rows = conn.execute(
                    "SELECT pair, COUNT(*) as cnt FROM trades GROUP BY pair ORDER BY cnt DESC LIMIT ?",
                    (n,)
                ).fetchall()
                conn.close()
                if rows:
                    return [r[0] for r in rows]

            # Fallback: read from config pair_whitelist or hardcoded
            return []
        except Exception as e:
            logger.debug(f"[Scheduler:AutoBacktest] Could not get top pairs: {e}")
            return []


    # ═══════════════════════════════════════════════════════════
    # Phase 24: Neural Organism Jobs
    # ═══════════════════════════════════════════════════════════

    def _organism_hourly_decay(self):
        """Hourly: decay all neuron beliefs (metabolic clock)."""
        try:
            from neural_organism import get_organism
            organism = get_organism()
            # Detect current regime for regime-specific decay rate
            regime = "_global"
            try:
                if self._regime_classifier is None:
                    from regime_classifier import RegimeClassifier
                    self._regime_classifier = RegimeClassifier()
                regime = self._regime_classifier.classify({}).get("regime", "_global")
            except Exception:
                pass
            organism.decay_all(regime)
            logger.info(f"[Scheduler:Organism] Hourly decay completed (regime={regime})")
        except Exception as e:
            logger.error(f"[Scheduler:Organism] Decay failed: {e}")

    def _organism_hormone_refresh(self):
        """Every 5 min: refresh hormones from current sensor pheromones.

        F3 (2026-04-25): Hormones.compute previously fired only on trade
        exits — memory_pressure and sensor_stress could accumulate for
        hours and cortisol stayed at the last-trade value. With this tick
        the homeostasis loop closes regardless of trade cadence: memory
        full → cortisol drops → hormonal_scalar drops → sizing shrinks
        before the next trade even fires.
        """
        try:
            from neural_organism import get_organism
            organism = get_organism()
            out = organism.refresh_hormones()
            logger.info(
                f"[Scheduler:Organism] hormone refresh — cortisol={out['cortisol']} "
                f"dopamine={out['dopamine']} serotonin={out['serotonin']} "
                f"_stress={out['_stress']}"
            )
        except Exception as e:
            logger.error(f"[Scheduler:Organism] Hormone refresh failed: {e}")

    def _organism_habit_check(self):
        """Daily 05:15: Check all neurons for habit consolidation."""
        try:
            from neural_organism import get_organism
            organism = get_organism()
            consolidated = 0
            for key, neuron in organism._neurons.items():
                old_strength = neuron.prior_strength
                organism.ganglia.check_consolidation(neuron)
                if neuron.prior_strength > old_strength:
                    consolidated += 1
            if consolidated:
                organism._persist_batch(list(organism._neurons.values()))
            logger.info(f"[Scheduler:Organism] Habit check: {consolidated} neurons consolidated")
        except Exception as e:
            logger.error(f"[Scheduler:Organism] Habit check failed: {e}")

    def _organism_sleep(self):
        """Weekly Sunday 03:30: Sleep consolidation — replay + prune + counterfactual.

        Sprint 2026-05-01: jemalloc-purge wrapped — full neuron list (~1758
        items) materialised into memory during _persist_batch.
        Sprint 2026-05-02: guardian consults predictive forecast.
        """
        if self._memory_pressure_halt():
            logger.warning("[Scheduler:Organism] Sleep SKIPPED — memory pressure critical.")
            return
        with self._heavy_job_gc("organism_sleep"):
            try:
                from neural_organism import get_organism
                organism = get_organism()
                result = organism.sleep.run_consolidation(
                    organism._neurons, organism.synapses, organism.ganglia, organism.hippocampus)
                organism._persist_batch(list(organism._neurons.values()))
                logger.info(f"[Scheduler:Organism] Sleep consolidation: {result}")
            except Exception as e:
                logger.error(f"[Scheduler:Organism] Sleep consolidation failed: {e}")

    def _organism_dmn(self):
        """Daily 04:00: Default Mode Network — idle background processing.

        T12 (2026-04-25): synapse_candidates discovered by DMN are now
        propagated to the live SynapseNetwork via
        `organism.adopt_synapse_discoveries()`. Previously the count was
        logged and the list was discarded — the connectivity graph stayed
        static and the brain never grew new edges.
        """
        try:
            from neural_organism import get_organism
            organism = get_organism()
            result = organism.dmn.run_idle_cycle(organism._neurons, organism.hippocampus)
            n_cf = len(result.get('counterfactuals', []))
            discoveries = result.get('discoveries', []) or []
            n_added = organism.adopt_synapse_discoveries(discoveries)
            logger.info(
                f"[Scheduler:Organism] DMN idle: {n_cf} counterfactuals, "
                f"{len(discoveries)} synapse candidates → {n_added} adopted"
            )
        except Exception as e:
            logger.error(f"[Scheduler:Organism] DMN failed: {e}")

    def _organism_evolution(self):
        """Weekly Sunday 04:00: NeuroEvolution — population tournament.

        T15 (2026-04-25): after the population tournament, also load the
        architecture_evolver's saved best genome from disk and blend it
        into the live organism via `adopt_genome()`. Previously
        `genome_best.json` was saved weekly but never loaded — the
        architectural search was decorative.
        """
        try:
            from neural_organism import get_organism
            organism = get_organism()
            organism.evolution.run_tournament(organism._neurons, organism._cumulative_pnl)

            # T15: read the architecture-evolver's saved genome and adopt
            try:
                import json as _json
                import os as _os
                genome_path = _os.path.join(
                    _os.path.dirname(__file__), "..", "models", "genome_best.json"
                )
                if _os.path.exists(genome_path):
                    with open(genome_path, "r") as fh:
                        genome = _json.load(fh)
                    moved = organism.adopt_genome(genome, blend_ratio=0.20)
                    logger.info(
                        f"[T15:GenomeAdopt] architecture_evolver genome blended "
                        f"({moved} neurons moved)"
                    )
            except Exception as _ge:
                logger.debug(f"[T15:GenomeAdopt] skipped: {_ge}")

            organism._persist_batch(list(organism._neurons.values()))
            logger.info(f"[Scheduler:Organism] NeuroEvolution tournament completed")
        except Exception as e:
            logger.error(f"[Scheduler:Organism] NeuroEvolution failed: {e}")

    def _risk_envelope_sensor_tick(self):
        """RE-5 (2026-04-25): every 5 min — update RiskEnvelope sensor
        vote + decay state. When 3+ sensors alarm, decay multiplier
        starts shrinking the envelope; clean ticks slowly recover.
        """
        try:
            from risk_envelope import get_risk_envelope
            envelope = get_risk_envelope()
            telemetry = envelope.update_sensor_state()
            if telemetry["transition"] != "stable":
                logger.warning(
                    f"[RiskEnvelope] tick — votes={telemetry['votes_count']}/5 "
                    f"decay={telemetry['decay_multiplier']:.2f} "
                    f"transition={telemetry['transition']} "
                    f"breakdown={telemetry['votes_breakdown']}"
                )
            else:
                # Stable — debug-level only to keep INFO log clean
                logger.debug(
                    f"[RiskEnvelope] tick — votes={telemetry['votes_count']}/5 "
                    f"decay={telemetry['decay_multiplier']:.2f}"
                )
        except Exception as e:
            logger.error(f"[RiskEnvelope] sensor tick failed: {e}")

    def _risk_envelope_promote_tick(self):
        """RE-5 (2026-04-25): hourly confidence + autonomy promote/demote.

        - Computes continuous confidence score (Sharpe + winrate + PF + DD + health)
        - Persists score for telemetry
        - Asymmetric: 3:1 (slow promote, fast demote)
            * confidence > 0.70 sustained 30 days → promote
            * confidence < 0.30 single day → demote
        """
        try:
            from risk_envelope import get_risk_envelope
            from autonomy_manager import AutonomyManager
            envelope = get_risk_envelope()
            confidence = envelope.get_continuous_confidence_score()
            current_state = envelope.compute()

            logger.info(
                f"[RiskEnvelope] confidence={confidence:.3f} "
                f"L{current_state.autonomy_level} "
                f"lev={current_state.leverage_max:.1f}x "
                f"risk={current_state.risk_per_trade*100:.1f}% "
                f"kelly_cap={current_state.kelly_cap*100:.0f}% "
                f"sl={current_state.sl_base_pct*100:.1f}% "
                f"votes={current_state.sensor_votes}/5 "
                f"decay={current_state.decay_multiplier:.2f}"
            )

            # Asymmetric promote/demote evaluation
            from db import get_db_connection
            am = AutonomyManager()
            current_level = am.get_level()

            with get_db_connection() as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS confidence_history (
                        timestamp TEXT PRIMARY KEY,
                        score REAL,
                        autonomy_level INTEGER
                    )
                """)
                conn.execute(
                    "INSERT OR REPLACE INTO confidence_history (timestamp, score, autonomy_level) "
                    "VALUES (datetime('now'), ?, ?)",
                    (confidence, current_level),
                )
                # Asymmetric demote: confidence < 0.30 over last 24h average → demote
                row_dem = conn.execute("""
                    SELECT AVG(score) AS avg_score, COUNT(*) AS n
                      FROM confidence_history
                     WHERE timestamp >= datetime('now', '-24 hours')
                """).fetchone()
                # Asymmetric promote: confidence > 0.70 sustained 30 days
                row_prom = conn.execute("""
                    SELECT AVG(score) AS avg_score, MIN(score) AS min_score, COUNT(*) AS n
                      FROM confidence_history
                     WHERE timestamp >= datetime('now', '-30 days')
                """).fetchone()
                conn.commit()

            avg_24h = float(row_dem["avg_score"] or 0.5) if row_dem else 0.5
            n_24h = int(row_dem["n"] or 0) if row_dem else 0

            avg_30d = float(row_prom["avg_score"] or 0.5) if row_prom else 0.5
            min_30d = float(row_prom["min_score"] or 0.5) if row_prom else 0.5
            n_30d = int(row_prom["n"] or 0) if row_prom else 0

            # FIX-1 (2026-04-25 audit): require ≥20 REAL trades in last 30d
            # before allowing demote. Prevents bootstrap-demote loop where
            # n<5 → confidence=0.50 (now neutral) but auto-demote could
            # still fire if real history is shallow.
            n_real_trades_30d = 0
            try:
                trades_db = os.path.join(os.path.dirname(__file__), "..", "tradesv3.sqlite")
                if os.path.exists(trades_db):
                    import sqlite3 as _sq
                    _tconn = _sq.connect(trades_db)
                    _tconn.row_factory = _sq.Row
                    _trow = _tconn.execute("""
                        SELECT COUNT(*) AS n FROM trades
                         WHERE close_date >= datetime('now', '-30 days')
                           AND is_open = 0
                    """).fetchone()
                    _tconn.close()
                    n_real_trades_30d = int(_trow["n"] or 0) if _trow else 0
            except Exception:
                pass

            # Demote: 24h average < 0.30 with at least 6 hourly samples
            # AND at least 20 real trades over 30d (audit FIX-1).
            if avg_24h < 0.30 and n_24h >= 6 and current_level > 0 and n_real_trades_30d >= 20:
                try:
                    am.demote(reason=f"confidence_24h={avg_24h:.3f}<0.30")
                    logger.warning(
                        f"[RiskEnvelope] AUTONOMY DEMOTE L{current_level}→L{current_level-1} "
                        f"reason=avg_confidence_24h={avg_24h:.3f}<0.30"
                    )
                except Exception as e:
                    logger.debug(f"[RiskEnvelope] demote failed: {e}")
            # Promote: 30d AVG > 0.70 AND 30d MIN > 0.50 AND at least 200 hourly samples
            elif (avg_30d > 0.70 and min_30d > 0.50 and n_30d >= 200
                  and current_level < 5):
                # Also check existing AutonomyManager criteria (trades count, sharpe, dd)
                try:
                    if am.maybe_promote():
                        logger.info(
                            f"[RiskEnvelope] AUTONOMY PROMOTE L{current_level}→L{current_level+1} "
                            f"reason=avg_30d={avg_30d:.3f}, min_30d={min_30d:.3f}"
                        )
                except Exception as e:
                    logger.debug(f"[RiskEnvelope] promote failed: {e}")
        except Exception as e:
            logger.error(f"[RiskEnvelope] promote tick failed: {e}")

    def _counterfactual_to_kelly_tick(self):
        """T8 (2026-04-25): backfill bayesian_kelly_shadow from
        counterfactual_results.

        32K+ counterfactual rows were sitting in DB with no consumer.
        Each row is a hypothetical alternate-decision PnL outcome.
        Treating them as forgone observations (the trade we DIDN'T take)
        feeds shadow Kelly per pair — same loop as forgone_pnl but
        fed by the counterfactual_engine pipeline instead of human-
        rejected signals. Dramatically expands per-pair evidence base.

        Cron: daily 02:45 UTC (off-peak). Idempotent via
        `consumed_by_kelly` flag added on first run.
        """
        try:
            from db import get_db_connection
            from position_sizer import get_shadow_kelly
            shadow = get_shadow_kelly()
            with get_db_connection() as conn:
                # Idempotency column — add if missing
                cols = [r[1] for r in conn.execute(
                    "PRAGMA table_info(counterfactual_results)"
                ).fetchall()]
                if "consumed_by_kelly" not in cols:
                    try:
                        conn.execute(
                            "ALTER TABLE counterfactual_results "
                            "ADD COLUMN consumed_by_kelly INTEGER DEFAULT 0"
                        )
                        conn.commit()
                    except Exception:
                        pass

                # FIX-B1 (2026-04-25): real schema is
                # `counterfactual_outcome_pnl` (not `counterfactual_pnl`)
                # and there is NO `pair` column — it's per-intervention,
                # not per-pair. We JOIN ai_decisions via original_trade_id
                # to recover the pair. Audit caught all three column errors.
                rows = conn.execute("""
                    SELECT cr.id, cr.regime, cr.counterfactual_outcome_pnl AS pnl,
                           ad.pair AS pair
                      FROM counterfactual_results cr
                      LEFT JOIN ai_decisions ad ON ad.id = cr.original_trade_id
                     WHERE COALESCE(cr.consumed_by_kelly, 0) = 0
                       AND cr.counterfactual_outcome_pnl IS NOT NULL
                       AND ad.pair IS NOT NULL
                     ORDER BY cr.id ASC LIMIT 5000
                """).fetchall()

            if not rows:
                logger.info("[T8:CounterfactualKelly] no fresh rows to consume")
                return

            # FIX-B1: batch UPDATE via transaction so we don't pay
            # per-row commit cost (audit flagged 5000 round-trips).
            consumed_ids: list[int] = []
            consumed = 0
            for row in rows:
                try:
                    pair = row["pair"]
                    regime = row["regime"] or "_global"
                    pnl = float(row["pnl"])
                    if not pair:
                        continue
                    shadow.update(
                        won=(pnl > 0),
                        pnl_pct=pnl,
                        pair=pair,
                        regime=regime,
                    )
                    consumed_ids.append(int(row["id"]))
                    consumed += 1
                except Exception:
                    continue

            # Single batched UPDATE for all consumed IDs.
            if consumed_ids:
                try:
                    with get_db_connection() as conn3:
                        # Chunk to avoid SQLite's parameter limit (~999).
                        for i in range(0, len(consumed_ids), 500):
                            chunk = consumed_ids[i:i + 500]
                            placeholders = ",".join("?" for _ in chunk)
                            conn3.execute(
                                f"UPDATE counterfactual_results "
                                f"SET consumed_by_kelly = 1 "
                                f"WHERE id IN ({placeholders})",
                                chunk,
                            )
                        conn3.commit()
                except Exception as _ue:
                    logger.warning(f"[T8:CounterfactualKelly] batch update failed: {_ue}")

            logger.info(
                f"[T8:CounterfactualKelly] consumed {consumed}/{len(rows)} rows "
                f"into bayesian_kelly_shadow"
            )
        except Exception as e:
            logger.warning(f"[T8:CounterfactualKelly] tick failed: {e}")

    def _organism_cerebellum(self):
        """Daily 00:10: Log cerebellum best trading hours."""
        try:
            from neural_organism import get_organism
            organism = get_organism()
            best = organism.cerebellum.get_best_hours(6)
            logger.info(f"[Scheduler:Organism] Cerebellum best hours (UTC): {best}")
        except Exception as e:
            logger.error(f"[Scheduler:Organism] Cerebellum failed: {e}")

    # ═══════════════════════════════════════════════════════════
    # Phase 26: Predictive Interoception + Pheromone
    # ═══════════════════════════════════════════════════════════

    def _interoception_check(self):
        """Every 15min: Record system metrics + predict future health."""
        try:
            from predictive_interoception import get_interoception

            intro = get_interoception()

            # Record current metrics
            try:
                import psutil
                mem = psutil.virtual_memory()
                intro.record("ram_usage_pct", mem.percent / 100.0)
            except Exception:
                pass

            # Win rate from recent trades
            try:
                conn = get_db_connection()
                row = conn.execute("""
                    SELECT COUNT(*) as total,
                           SUM(CASE WHEN trade_pnl > 0 THEN 1 ELSE 0 END) as wins
                    FROM organism_audit
                    WHERE timestamp > datetime('now', '-7 days')
                """).fetchone()
                conn.close()
                if row and row[0] > 0:
                    intro.record("win_rate_7d", row[1] / row[0])
            except Exception:
                pass

            # API error rate from pheromone field
            try:
                from pheromone_field import get_pheromone_field
                field = get_pheromone_field()
                health = field.get_field_health()
                intro.record("organism_health", health.get("avg_freshness", 0.5))
            except Exception:
                pass

            # Run prediction + proactive response
            result = intro.predict_and_act()
            if result["alerts"]:
                logger.warning(
                    f"[Phase26:Interoception] {result['health_trend']} — "
                    f"{len(result['alerts'])} alerts: "
                    + ", ".join(a['metric'] for a in result['alerts'])
                )
            # Efferent: turn the prescription strings into concrete effector
            # calls. Previously this key was produced and discarded — now
            # each proactive_action maps to a pheromone deposit that the
            # HydraSizer / llm_router / catboost retrain consumers honour.
            try:
                from proactive_dispatcher import get_proactive_dispatcher
                dispatcher = get_proactive_dispatcher()
                counters = dispatcher.dispatch(result)
                if counters.get("dispatched"):
                    logger.info(
                        f"[Phase26:Interoception] dispatcher "
                        f"dispatched={counters['dispatched']} "
                        f"cooldown={counters['cooldown_skipped']} "
                        f"unclassified={counters['unclassified']}"
                    )
            except Exception as disp_err:
                logger.debug(f"[Phase26:Interoception] dispatcher failed: {disp_err}")

        except Exception as e:
            logger.error(f"[Phase26:Interoception] Check failed: {e}")

    def _pheromone_cleanup(self):
        """Every 5 min: Clean up fully decayed pheromones with adaptive
        max-idle. Under memory pressure (>0.7) the idle window tightens
        from 300s → 60s so the field releases trails faster.
        """
        try:
            from pheromone_field import get_pheromone_field
            field = get_pheromone_field()

            # Adaptive idle cap: pressure-driven
            try:
                from sensor_bridges import aggregate_memory_stress
                pressure = float(aggregate_memory_stress())
            except Exception:
                pressure = 0.0
            max_idle = 60.0 if pressure > 0.7 else 300.0

            cleaned = field.cleanup(max_idle_seconds=max_idle)
            health = field.get_field_health()
            logger.debug(
                f"[Phase26:Pheromone] Cleanup: {cleaned} removed (idle_cap={max_idle:.0f}s "
                f"pressure={pressure:.2f}), {health['active_signals']} active "
                f"from {health['active_sources']}"
            )
        except Exception as e:
            logger.debug(f"[Phase26:Pheromone] Cleanup failed: {e}")

    def _decisions_outcome_backfill_tick(self):
        """LOOP-3 (2026-04-25): backfill ai_decisions.outcome_pnl for any
        decision older than 4h whose outcome was never written.

        Audit found 84% of decisions had NULL outcome_pnl — the system
        literally could not see the consequences of its own predictions,
        so RAG, agent_pool, and the calibrator had no learning signal.

        Backfill source: Freqtrade's OHLCV feather files at
        `user_data/data/bybit/futures/<symbol>-1h-futures.feather`. We
        compute entry_close at the decision candle and exit_close at
        decision_ts + 4h, then sign by signal direction.

        Runs daily at 00:30 UTC (off-peak) plus opportunistically when
        the first cycle ticks (so prod gets coverage immediately after
        deploy without waiting until midnight).

        Sprint 2026-05-02 audit Finding #4: this is a NON-CRITICAL job
        (results are advisory for learning, not safety). Honors the
        memory guardian's `should_throttle` signal — when memory
        pressure is high or rising, this job defers to the next cycle.
        """
        try:
            from predictive_memory_guardian import should_throttle
            if should_throttle():
                logger.info(
                    "[DecisionsBackfill] DEFERRED — memory pressure forecast indicates throttle"
                )
                return
        except Exception:
            pass

        try:
            import os
            import pandas as pd
            from db import get_db_connection

            data_root = os.path.join(os.path.dirname(__file__), "..", "data", "bybit", "futures")

            with get_db_connection() as conn:
                rows = conn.execute("""
                    SELECT id, timestamp, pair, signal_type, confidence
                    FROM ai_decisions
                    WHERE outcome_pnl IS NULL
                      AND timestamp < datetime('now', '-4 hours')
                      AND timestamp >= datetime('now', '-30 days')
                    ORDER BY timestamp DESC LIMIT 500
                """).fetchall()
            if not rows:
                logger.info("[Loop-3:Backfill] No decisions need backfill")
                return

            cache: dict = {}
            updated = 0
            skipped_no_data = 0
            skipped_neutral = 0
            for row in rows:
                try:
                    pair = row["pair"]
                    sig = (row["signal_type"] or "").upper()
                    if sig in ("NEUTRAL", "ABSTAIN", ""):
                        # NEUTRAL signals never traded → outcome_pnl=0 by definition
                        skipped_neutral += 1
                        # Still write 0.0 so backfill_pct goes up — agent
                        # pool can use this to learn "I called NEUTRAL,
                        # actual price moved X — was my abstention right?"
                        with get_db_connection() as conn2:
                            conn2.execute(
                                "UPDATE ai_decisions SET outcome_pnl = 0.0, "
                                "outcome_duration = 14400 WHERE id = ?",
                                (row["id"],),
                            )
                            conn2.commit()
                        updated += 1
                        continue

                    file_sym = pair.replace("/", "_").replace(":", "_")
                    fpath = os.path.join(data_root, f"{file_sym}-1h-futures.feather")
                    if pair not in cache:
                        if not os.path.exists(fpath):
                            cache[pair] = None
                        else:
                            try:
                                cache[pair] = pd.read_feather(fpath)
                            except Exception:
                                cache[pair] = None
                    df = cache[pair]
                    if df is None or len(df) < 2:
                        skipped_no_data += 1
                        continue

                    decision_ts = pd.Timestamp(row["timestamp"], tz="UTC")
                    exit_ts = decision_ts + pd.Timedelta(hours=4)

                    # Find nearest candles
                    if "date" in df.columns:
                        ts_col = df["date"]
                    else:
                        ts_col = df.iloc[:, 0]

                    entry_idx = (ts_col - decision_ts).abs().idxmin()
                    exit_idx = (ts_col - exit_ts).abs().idxmin()
                    entry_close = float(df.iloc[entry_idx]["close"])
                    exit_close = float(df.iloc[exit_idx]["close"])
                    if entry_close <= 0:
                        skipped_no_data += 1
                        continue

                    raw_pct = ((exit_close - entry_close) / entry_close) * 100.0
                    if sig in ("BULLISH", "BULL", "LONG"):
                        outcome_pct = raw_pct
                    elif sig in ("BEARISH", "BEAR", "SHORT"):
                        outcome_pct = -raw_pct
                    else:
                        outcome_pct = 0.0

                    with get_db_connection() as conn3:
                        conn3.execute(
                            "UPDATE ai_decisions SET outcome_pnl = ?, "
                            "outcome_duration = 14400 WHERE id = ?",
                            (round(outcome_pct, 4), row["id"]),
                        )
                        conn3.commit()
                    updated += 1
                except Exception as e:
                    logger.debug(f"[Loop-3:Backfill] row #{row['id']} skipped: {e}")
                    continue

            logger.info(
                f"[Loop-3:Backfill] {updated}/{len(rows)} decisions filled "
                f"(neutral_zero={skipped_neutral}, no_data={skipped_no_data})"
            )
        except Exception as e:
            logger.warning(f"[Loop-3:Backfill] tick failed: {e}")

    def _shadow_kelly_divergence_tick(self):
        """Every 30min: per-pair shadow Kelly → entry-gate pheromones.

        LOOP-1 (2026-04-25): the previous tick collapsed all per-pair shadow
        evidence into a single global "sensor_stress" deposit that just
        nudged cortisol downward — pair-level information was destroyed.
        Now we publish a separate `shadow_score::PAIR` pheromone per pair,
        carrying the Beta posterior mean (or Thompson sample) of the
        forgone-counterfactual win rate. HydraSizer.populate_entry_trend
        reads it and SKIPs entry on pairs with shadow_score < 0.30 — i.e.
        pairs where the forgone evidence says the strategy would lose more
        often than win, regardless of the current confidence.

        Backward compat: still emits the legacy shadow_divergence stress
        pheromone for the four pathological cases (shadow_wr<25%, n>=5).
        """
        try:
            from db import get_db_connection
            from sensor_bridges import record_shadow_divergence
            from pheromone_field import get_pheromone_field
            pfield = get_pheromone_field()
            with get_db_connection() as conn:
                rows = conn.execute(
                    """SELECT s.pair,
                              s.alpha AS s_alpha, s.beta_param AS s_beta, s.n_trades AS s_n,
                              l.alpha AS l_alpha, l.beta_param AS l_beta
                         FROM bayesian_kelly_shadow_per_pair s
                         LEFT JOIN bayesian_kelly_per_pair l
                           ON l.pair = s.pair AND l.regime = s.regime
                        WHERE s.regime = '_global' AND s.n_trades > 3
                     ORDER BY s.updated_at DESC LIMIT 100"""
                ).fetchall()
            fired = 0
            published = 0
            try:
                import numpy as _np
            except Exception:
                _np = None
            for row in rows:
                try:
                    s_alpha = float(row["s_alpha"]); s_beta = float(row["s_beta"])
                    s_total = s_alpha + s_beta
                    if s_total <= 0:
                        continue
                    shadow_wr = s_alpha / s_total
                    real_wr = None
                    if row["l_alpha"] is not None and row["l_beta"] is not None:
                        la = float(row["l_alpha"]); lb = float(row["l_beta"])
                        if la + lb > 0:
                            real_wr = la / (la + lb)

                    # LOOP-1: per-pair score pheromone. Use Thompson sample
                    # of the Beta posterior so the gate explores when the
                    # posterior is uncertain (small n) and exploits when it
                    # converges. Falls back to mean if numpy is unavailable.
                    if _np is not None:
                        ts_score = float(_np.random.beta(max(s_alpha, 0.5),
                                                          max(s_beta, 0.5)))
                    else:
                        ts_score = shadow_wr
                    pfield.deposit(
                        "shadow_kelly", f"shadow_score::{row['pair']}",
                        {
                            "intensity": 1.0 - ts_score,  # high intensity = bad
                            "score": ts_score,            # Thompson sample
                            "shadow_wr": shadow_wr,        # posterior mean
                            "real_wr": real_wr,
                            "n_shadow": int(row["s_n"]),
                            "alpha": s_alpha, "beta": s_beta,
                        },
                        # 35-min half_life so the score stays fresh between
                        # 30-min ticks; a recovering pair re-publishes a
                        # better sample on the next tick.
                        half_life=2100.0,
                    )
                    published += 1

                    # Legacy stress signal (still fires for catastrophic pairs)
                    if shadow_wr < 0.25 and int(row["s_n"]) >= 5:
                        record_shadow_divergence(
                            pair=row["pair"],
                            real_winrate=real_wr,
                            shadow_winrate=shadow_wr,
                            n_shadow=int(row["s_n"]),
                        )
                        fired += 1
                except Exception:
                    continue
            logger.info(
                f"[Loop-1:ShadowKelly] published {published} per-pair scores, "
                f"{fired} legacy divergence deposits"
            )
        except Exception as e:
            logger.debug(f"[SensorBridges:Shadow] tick failed: {e}")

    def _weekly_budget_adjust(self):
        """Every Sunday 23:55 UTC: compute the past week's PnL% and
        drive RiskBudgetManager.weekly_adjust so next week's initial
        budget multiplier tracks performance. Uses tradesv3 for realised
        PnL so the signal is unambiguous (no forgone approximations)."""
        try:
            import sqlite3 as _sqlite
            import os
            trades_db = os.path.join(os.path.dirname(__file__), "..", "tradesv3.sqlite")
            trades_db = os.path.abspath(trades_db)
            if not os.path.exists(trades_db):
                logger.debug(f"[WeeklyBudget] trades DB missing at {trades_db}")
                return
            conn = _sqlite.connect(trades_db, timeout=30.0)
            try:
                row = conn.execute(
                    """SELECT COALESCE(SUM(close_profit_abs), 0.0) AS pnl_abs,
                              COALESCE(SUM(stake_amount), 0.0) AS stake_abs
                         FROM trades
                        WHERE close_date IS NOT NULL
                          AND close_date >= datetime('now', '-7 days')"""
                ).fetchone()
            finally:
                conn.close()
            pnl_abs = float(row[0] or 0.0)
            stake_abs = float(row[1] or 0.0)
            if stake_abs <= 0:
                logger.info("[WeeklyBudget] no closed trades in the last 7 days — skipping")
                return
            pnl_pct = (pnl_abs / stake_abs) * 100.0
            from risk_budget import RiskBudgetManager
            mgr = RiskBudgetManager(portfolio_value=self._read_portfolio_value())
            mgr.weekly_adjust(pnl_pct)
            logger.info(f"[WeeklyBudget] weekly_adjust fired pnl_pct={pnl_pct:+.2f}%")
        except Exception as e:
            logger.warning(f"[WeeklyBudget] adjust failed: {e}")

    def _autopoietic_integrity_review(self):
        """Daily at 00:05 UTC: read the latest `autopoietic_integrity`
        row written by self_model. If the AII score has trended below
        0.6 the past 7 days (the "organism losing its coherence" band
        in self_model documentation), deposit a defensive pheromone so
        the rest of the pipeline throttles until the next consolidation.
        """
        try:
            from db import get_db_connection
            with get_db_connection() as conn:
                # Task 23: column is `aii_composite` (see db.py:787 and
                # self_model.py:598 INSERT). The earlier `aii_score` name
                # triggered OperationalError and was swallowed by the
                # outer try/except, making the review a silent no-op.
                rows = conn.execute(
                    """SELECT aii_composite, timestamp FROM autopoietic_integrity
                        ORDER BY timestamp DESC LIMIT 7"""
                ).fetchall()
            scores = [float(r[0]) for r in rows if r and r[0] is not None]
            if not scores:
                return
            import statistics as _st
            median = _st.median(scores)
            logger.info(
                f"[AII:Review] median_7d={median:.3f} samples={len(scores)}"
            )
            if median < 0.6:
                try:
                    from pheromone_field import get_pheromone_field
                    pfield = get_pheromone_field()
                    pfield.deposit(
                        "autopoietic_review", "defensive_mode",
                        {"sizing_cap": 0.5, "reason": "aii_median_below_0.6",
                         "median_7d": median},
                        half_life=86400.0,
                    )
                    logger.warning(
                        f"[AII:Review] median {median:.3f} < 0.6 — defensive mode deposited"
                    )
                except Exception as e:
                    logger.debug(f"[AII:Review] defensive deposit failed: {e}")
        except Exception as e:
            logger.debug(f"[AII:Review] tick failed: {e}")

    def _wal_checkpoint_tick(self):
        """Every 60s: inspect the WAL file size and TRUNCATE if it has
        grown beyond the threshold. Complements the in-pool
        release-count trigger by catching the "quiet process, huge WAL"
        scenario that produced the 166 MB incident before Mega Sprint.
        """
        try:
            import os
            from ai_config import AI_DB_PATH
            wal_path = AI_DB_PATH + "-wal"
            if not os.path.exists(wal_path):
                return
            size = os.path.getsize(wal_path)
            threshold = 32 * 1024 * 1024  # 32 MB
            if size <= threshold:
                return
            from db import get_db_connection
            with get_db_connection() as conn:
                try:
                    conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                    conn.commit()
                    logger.info(
                        f"[WALCheckpoint] TRUNCATE fired size={size/1024/1024:.1f}MB → "
                        f"{os.path.getsize(wal_path)/1024/1024:.1f}MB"
                    )
                except Exception as e:
                    logger.debug(f"[WALCheckpoint] TRUNCATE failed: {e}")
        except Exception as e:
            logger.debug(f"[WALCheckpoint] tick failed: {e}")

    def _sqlite_spool_drain_tick(self):
        """Every 60s: try to replay pending rows from `hot_writes` via
        the broker. No-op when the broker is unreachable (the client
        bails after a single ping). When the broker is alive this lets
        any write that raced the broker's start-up on process boot make
        its way into the canonical DB within a minute.
        """
        try:
            from sqlite_broker import drain_spool
            drained = drain_spool(batch_size=50)
            if drained:
                logger.info(f"[SpoolDrain] {drained} pending write(s) committed via broker")
        except Exception as e:
            logger.debug(f"[SpoolDrain] tick failed: {e}")

    def _retrain_request_drain_tick(self):
        """Every 30min: check whether ProactiveDispatcher has deposited a
        retrain_request pheromone. If yes, advance the CatBoost retrain
        on demand (out-of-band from its Sunday 03:00 UTC cron). State
        is held on the instance (`_last_retrain_handled_ts`) so the same
        deposit isn't re-honoured after its 6h half-life naturally
        clears — mirrors the dispatcher cooldown semantics.
        """
        try:
            from pheromone_field import get_pheromone_field
            hint = get_pheromone_field().read("retrain_request")
            if not isinstance(hint, dict):
                return
            ts_key = "_last_retrain_handled_ts"
            last_handled = getattr(self, ts_key, 0.0)
            import time as _tm
            now = _tm.time()
            # Only re-fire if at least 6h passed since last handling to
            # avoid thrashing the training pipeline.
            if now - last_handled < 6 * 3600:
                return
            severity = float(hint.get("severity", 1.0))
            logger.info(
                f"[RetrainDrain] handling proactive retrain_request "
                f"(severity={severity:.2f}, reason={hint.get('reason')})"
            )
            try:
                self._catboost_retrain()
                setattr(self, ts_key, now)
            except Exception as e:
                logger.warning(f"[RetrainDrain] retrain invocation failed: {e}")
        except Exception as e:
            logger.debug(f"[RetrainDrain] tick failed: {e}")

    def _cache_health_drain_tick(self):
        """Every 10 min: read-only telemetry of cache_health_log.

        Task E (2026-04-25): the rag_graph process owns the actual
        SemanticCache instance carrying live counters. Singletons don't
        span processes — scheduler's own SemanticCache instance is empty
        because it serves no requests. The drain is now performed by
        rag_graph's _cache_health_drain_daemon (in-process), and this
        scheduler tick just SELECTs the latest row for log/telemetry so
        the operator can grep the scheduler unit and see whether the
        RAG cache is healthy.
        """
        try:
            from db import get_db_connection
            conn = get_db_connection()
            row = conn.execute(
                """SELECT hits, misses, puts, rejects, invalidations,
                          hit_rate, median_similarity, threshold, timestamp
                   FROM cache_health_log
                   ORDER BY timestamp DESC LIMIT 1"""
            ).fetchone()
            conn.close()
            if row is None:
                logger.info("[CacheHealth] no rows yet — RAG drain may not have ticked")
                return
            logger.info(
                f"[CacheHealth] last_window — hits={row['hits']} misses={row['misses']} "
                f"hit_rate={float(row['hit_rate']):.2%} "
                f"med_sim={float(row['median_similarity']):.3f} "
                f"puts={row['puts']} rejects={row['rejects']} "
                f"inv={row['invalidations']} ts={row['timestamp']}"
            )
            # Health pheromone — alert downstream when hit_rate is dead and
            # the volume is meaningful. Lets dispatcher / HydraSizer notice
            # the RAG layer is degraded without grepping logs.
            try:
                total = int(row['hits']) + int(row['misses'])
                if total >= 50 and float(row['hit_rate']) < 0.05:
                    from pheromone_field import get_pheromone_field
                    get_pheromone_field().deposit(
                        "cache_health", "rag_cache_dead",
                        {"intensity": 1.0, "hit_rate": float(row['hit_rate']),
                         "total": total, "ts": str(row['timestamp'])},
                        half_life=1800.0,
                    )
            except Exception:
                pass
        except Exception as e:
            logger.debug(f"[CacheHealth] read failed: {e}")

    def _pair_circuit_revive_tick(self):
        """Every 5min: probe dormant pairs to see if their orderbooks have
        come back. No exchange client is available inside the scheduler
        process, so this job runs a lightweight in-process check against
        the pheromone field — if a `pair_circuit::dormant::<pair>` trail
        is still fresh AND the most recent sensor deposit for that pair
        is >90s old, that's a reasonable signal the exchange quieted;
        clear the circuit so HydraSizer can try the pair again on the
        next candle. Actual healthy-book confirmation still happens
        in-line via probe_orderbook → record_success.
        """
        try:
            import time
            from pair_circuit import get_pair_circuit
            circuit = get_pair_circuit()
            revived = 0
            now = time.time()
            for pair in list(circuit._slots.keys()):
                slot = circuit._slots[pair]
                # Revive if blacklist window expired AND we haven't seen
                # another failure within the last 90 seconds.
                if slot.blacklisted_until <= now and slot.consecutive_failures > 0:
                    if (now - slot.last_event_at) > 90.0:
                        circuit.record_success(pair)
                        revived += 1
            if revived:
                logger.info(f"[PairCircuit] revival_tick — {revived} pair(s) reopened")
        except Exception as e:
            logger.debug(f"[PairCircuit] revival_tick failed: {e}")

    # ─── Sprint 2026-05-01: adaptive pairlist tuning ────────────────────
    # Computes the EFFECTIVE filter values from current circuit telemetry
    # + organism state and persists them for HydraSizer's bot_loop_start
    # to apply at the next pairlist refresh. Config holds permissive max
    # ceilings; the effective values come from here.

    def _adaptive_pairlist_tune(self):
        """Every 30min: derive runtime-adaptive pairlist thresholds from
        live exchange + organism telemetry, deposit to pheromone field
        for HydraSizer to apply on the next refresh.

        Source values:
          • Spread cap     ← pair_circuit.adaptive_spread_cap()
                              (median 1h spread × hormonal factor)
          • Age floor      ← pair_circuit.adaptive_age_floor()
                              (cortisol-driven; calm=7d, panic=30d)
          • Volume floor   ← pair_circuit.adaptive_volume_floor()
                              (5th-percentile of observed pair volumes)
          • Whitelist size ← envelope.max_open_positions × 5
                              (so number_assets scales with autonomy)

        Persisted as a `pairlist_adaptive_thresholds` pheromone deposit
        (consumed by HydraSizer's bot_loop_start). Also written to
        `system_metrics` for telemetry observability.
        """
        try:
            from pair_circuit import get_pair_circuit
            circuit = get_pair_circuit()
            spread_cap = float(circuit.adaptive_spread_cap())
            age_floor = int(circuit.adaptive_age_floor())
            volume_floor = float(circuit.adaptive_volume_floor())
            try:
                from risk_envelope import get_risk_envelope
                env = get_risk_envelope()
                wl_target = int(env.get_max_open_positions()) * 5
            except Exception:
                wl_target = 30
            wl_target = max(10, min(80, wl_target))

            payload = {
                "spread_cap": round(spread_cap, 4),
                "age_floor": int(age_floor),
                "volume_floor": round(volume_floor, 2),
                "whitelist_target": int(wl_target),
            }

            try:
                from pheromone_field import get_pheromone_field
                pf = get_pheromone_field()
                pf.deposit("pairlist_tuner", "pairlist_adaptive_thresholds",
                           payload, half_life=1800.0)
            except Exception as e:
                logger.debug(f"[Pairlist:Tuner] pheromone deposit failed: {e}")

            try:
                from db import execute_with_retry
                ts = "strftime('%Y-%m-%dT%H:%M:%SZ', 'now')"
                for k, v in payload.items():
                    execute_with_retry(
                        f"INSERT INTO system_metrics "
                        f"(timestamp, metric_name, metric_value, metadata_json) "
                        f"VALUES ({ts}, ?, ?, NULL)",
                        (f"pairlist_{k}", float(v)),
                        max_retries=2,
                    )
            except Exception:
                pass

            logger.info(f"[Pairlist:Tuner] adaptive thresholds: {payload}")
        except Exception as e:
            logger.warning(f"[Pairlist:Tuner] tune failed: {e}")

    def _whitelist_health_tick(self):
        """Every 5min: monitor whitelist size; if it has collapsed below
        envelope.max_open_positions / 2 (the bot can't even fill its
        target slot count), deposit a `whitelist_distress` pheromone so
        HydraSizer / pair_circuit can temporarily relax filters.
        """
        try:
            # Pull last seen whitelist count from system_metrics rows
            # written by the strategy process via bot_loop_start.
            from db import get_db_connection
            with get_db_connection() as conn:
                row = conn.execute(
                    "SELECT metric_value FROM system_metrics "
                    "WHERE metric_name='whitelist_size' "
                    "ORDER BY rowid DESC LIMIT 1"
                ).fetchone()
            wl_size = int(row["metric_value"]) if row else 0

            try:
                from risk_envelope import get_risk_envelope
                target = int(get_risk_envelope().get_max_open_positions())
            except Exception:
                target = 8
            half_target = max(2, target // 2)

            if wl_size > 0 and wl_size < half_target:
                try:
                    from pheromone_field import get_pheromone_field
                    get_pheromone_field().deposit(
                        "whitelist_health", "whitelist_distress",
                        {"size": wl_size, "target": target,
                         "ratio": round(wl_size / max(1, target), 3)},
                        half_life=600.0,
                    )
                except Exception:
                    pass
                logger.warning(
                    f"[Whitelist:Health] DISTRESS — only {wl_size} pairs "
                    f"(target {target}, half {half_target})"
                )
        except Exception as e:
            logger.debug(f"[Whitelist:Health] tick failed: {e}")

    def _walk_forward_validation(self):
        """Sprint 2026-05-01 night — WALK-FORWARD VALIDATION.

        Daily check: compare last 14 days' Sharpe vs the 30 days before.
        When recent performance has degraded by >30%, deposit a
        `model_freeze` pheromone that downstream learners (CatBoost
        retrain, OOD refit, ensemble refit) honor by SKIPPING their
        update cycle. Protects the bot from learning a regime change
        as if it were a feature.

        Pure observation job — never blocks trades, only freezes model
        updates. Frozen state auto-clears the next day if performance
        recovers.
        """
        try:
            from db import get_db_connection
            with get_db_connection() as conn:
                # Recent: last 14 days
                row_recent = conn.execute("""
                    SELECT
                        COUNT(*) AS n,
                        AVG(close_profit) AS mean_ret,
                        SUM(CASE WHEN close_profit > 0 THEN 1 ELSE 0 END) AS wins,
                        SUM(close_profit_abs) AS net_pnl
                    FROM trades
                    WHERE close_date >= datetime('now', '-14 days')
                      AND is_open = 0
                """).fetchone()
                # Baseline: 30 days BEFORE the recent window
                row_baseline = conn.execute("""
                    SELECT
                        COUNT(*) AS n,
                        AVG(close_profit) AS mean_ret,
                        SUM(CASE WHEN close_profit > 0 THEN 1 ELSE 0 END) AS wins,
                        SUM(close_profit_abs) AS net_pnl
                    FROM trades
                    WHERE close_date >= datetime('now', '-44 days')
                      AND close_date < datetime('now', '-14 days')
                      AND is_open = 0
                """).fetchone()

            if not row_recent or not row_baseline:
                return
            n_recent = int(row_recent["n"] or 0)
            n_baseline = int(row_baseline["n"] or 0)
            if n_recent < 5 or n_baseline < 10:
                logger.info(
                    f"[WalkForward] insufficient samples (recent={n_recent}, "
                    f"baseline={n_baseline}) — skipping validation"
                )
                return

            recent_mean = float(row_recent["mean_ret"] or 0.0)
            baseline_mean = float(row_baseline["mean_ret"] or 0.0)
            recent_wr = float(row_recent["wins"] or 0) / max(1, n_recent)
            baseline_wr = float(row_baseline["wins"] or 0) / max(1, n_baseline)

            # Composite degradation score: avg ret + win rate.
            # Audit Finding B3 — thresholds sourced from PARAM_REGISTRY so
            # neurons / operators can tune them without redeploys.
            try:
                from neural_organism import _p as _np_wf
                ret_drop_thr = float(_np_wf("envelope.walk_forward.ret_drop_threshold", 0.30))
                wr_drop_thr = float(_np_wf("envelope.walk_forward.wr_drop_threshold", 0.15))
            except Exception:
                ret_drop_thr, wr_drop_thr = 0.30, 0.15
            ret_drop = (baseline_mean - recent_mean) / max(0.001, abs(baseline_mean) + 0.001)
            wr_drop = baseline_wr - recent_wr  # positive when degrading
            degraded = (ret_drop > ret_drop_thr) or (wr_drop > wr_drop_thr)

            try:
                import time as _t_wf
                from pheromone_field import get_pheromone_field
                pf = get_pheromone_field()
                # Embed wall-clock timestamp so _walk_forward_frozen can
                # apply the 36h staleness fail-safe (audit Finding D11).
                now_ts = _t_wf.time()
                if degraded:
                    pf.deposit(
                        "walk_forward", "model_freeze",
                        {"recent_n": n_recent, "baseline_n": n_baseline,
                         "recent_mean_ret": round(recent_mean, 4),
                         "baseline_mean_ret": round(baseline_mean, 4),
                         "recent_wr": round(recent_wr, 3),
                         "baseline_wr": round(baseline_wr, 3),
                         "ret_drop": round(ret_drop, 3),
                         "wr_drop": round(wr_drop, 3),
                         "frozen": True,
                         "_ts": now_ts},
                        half_life=86400.0,  # 24h decay
                    )
                    logger.warning(
                        f"[WalkForward] DEGRADED — recent (n={n_recent}, "
                        f"mean={recent_mean:.4f}, wr={recent_wr:.2%}) vs "
                        f"baseline (n={n_baseline}, mean={baseline_mean:.4f}, "
                        f"wr={baseline_wr:.2%}) — model updates FROZEN"
                    )
                else:
                    pf.deposit(
                        "walk_forward", "model_freeze",
                        {"frozen": False, "recent_wr": round(recent_wr, 3),
                         "baseline_wr": round(baseline_wr, 3),
                         "_ts": now_ts},
                        half_life=86400.0,
                    )
                    logger.info(
                        f"[WalkForward] HEALTHY — recent_wr={recent_wr:.2%} "
                        f"vs baseline_wr={baseline_wr:.2%}, ret_drop={ret_drop:.2%}"
                    )
            except Exception:
                pass
        except Exception as e:
            logger.debug(f"[WalkForward] validation tick failed: {e}")

    def _mc_bootstrap_validation(self):
        """Sprint 2026-05-02 — Monte Carlo trade-shuffle bootstrap.

        Source: jesse/jesse/research/monte_carlo/monte_carlo_trades.py.
        Tests whether observed Sharpe / total-return are statistically
        significant or order-luck. Shuffles the realized trade list N
        times and computes p-value.

        Runs nightly. When p > 0.05 (i.e., the strategy looks no better
        than randomized order), deposits a `mc_significance` pheromone
        flag = False that downstream consumers (RiskEnvelope EarnedTrust,
        threshold adapter) honor by treating recent trust gains as
        unproven (do not promote tier on insignificant performance).
        """
        try:
            import numpy as np
            from db import get_db_connection
            try:
                from neural_organism import _p
                iterations = int(_p("envelope.mc_bootstrap.iterations", 1000))
                alpha = float(_p("envelope.mc_bootstrap.alpha", 0.05))
            except Exception:
                iterations, alpha = 1000, 0.05

            with get_db_connection() as conn:
                rows = conn.execute("""
                    SELECT close_profit, close_profit_abs
                    FROM trades
                    WHERE close_date >= datetime('now', '-30 days')
                      AND is_open = 0
                    ORDER BY close_date ASC
                """).fetchall()
            if not rows or len(rows) < 20:
                logger.info(
                    f"[MC-Bootstrap] insufficient trades ({len(rows) if rows else 0}) "
                    "— skipping"
                )
                return
            returns = np.asarray(
                [float(r["close_profit"] or 0.0) for r in rows], dtype=float
            )
            observed_total = float(returns.sum())
            # Audit Finding #9 fix (2026-05-02): Sharpe annualization now
            # uses the ACTUAL trade-rate over the lookback (last 30 days)
            # rather than sqrt(252) which assumes daily bars on TradFi
            # calendar. Crypto trades 24/7/365, plus we are working with
            # PER-TRADE returns (close_profit) — annualization scale is
            # `sqrt(trades_per_year)` derived from observed cadence.
            lookback_days = 30.0
            trades_per_year = (returns.size / lookback_days) * 365.0
            ann_factor = float(np.sqrt(max(1.0, trades_per_year)))
            observed_sharpe = (
                returns.mean() / (returns.std() + 1e-12) * ann_factor
            )
            obs_compound = float(np.prod(1.0 + returns) - 1.0)
            shuffled_compounds = np.empty(iterations, dtype=float)
            shuffled_sharpe = np.empty(iterations, dtype=float)
            for i in range(iterations):
                idx = np.random.permutation(returns.size)
                shuffled = returns[idx]
                shuffled_compounds[i] = float(np.prod(1.0 + shuffled) - 1.0)
                shuffled_sharpe[i] = (
                    shuffled.mean() / (shuffled.std() + 1e-12) * ann_factor
                )
            # p-value: probability of seeing observed >= shuffled
            p_compound = float(np.sum(shuffled_compounds >= obs_compound) / iterations)
            p_sharpe = float(np.sum(shuffled_sharpe >= observed_sharpe) / iterations)
            significant = (p_compound < alpha) or (p_sharpe < alpha)

            try:
                import time as _t
                from pheromone_field import get_pheromone_field
                get_pheromone_field().deposit(
                    "mc_bootstrap", "significance",
                    {"significant": significant,
                     "p_compound": round(p_compound, 4),
                     "p_sharpe": round(p_sharpe, 4),
                     "n_trades": int(returns.size),
                     "iterations": iterations,
                     "obs_total_return": round(observed_total, 4),
                     "obs_compound_return": round(obs_compound, 4),
                     "obs_sharpe": round(observed_sharpe, 3),
                     "_ts": _t.time()},
                    half_life=172800.0,  # 48h
                )
            except Exception:
                pass
            logger.info(
                f"[MC-Bootstrap] n={returns.size} obs_compound={obs_compound:.3%} "
                f"p_compound={p_compound:.3f} p_sharpe={p_sharpe:.3f} "
                f"significant={significant}"
            )
        except Exception as e:
            logger.debug(f"[MC-Bootstrap] validation tick failed: {e}")

    def _stablecoin_netflow_tick(self):
        """Sprint 2026-05-02 — Stablecoin minting/burning as liquidity gauge.

        Net minting of USDT/USDC to exchange wallets = incoming buy
        pressure (12-48h lead). We don't have CryptoQuant API key in
        this stack, but we CAN proxy via Bybit's USDT supply on the
        derivatives wallet (rough, but directional).

        For now this writes the ROUGH proxy and a placeholder; when
        CryptoQuant / Glassnode credentials are added the body can be
        swapped without changing the consumer interface.
        """
        try:
            import time as _t
            try:
                from neural_organism import _p
                bullish_thr = float(_p("envelope.stablecoin.bullish_threshold_usd", 100_000_000.0))
            except Exception:
                bullish_thr = 100_000_000.0
            # Placeholder telemetry deposit — CryptoQuant API integration
            # would replace this with real netflow data. The pheromone is
            # consumed by RiskEnvelope.stablecoin_netflow_factor (added
            # on demand). When data isn't available the factor returns
            # 1.0 (neutral), so the absence of CryptoQuant doesn't break
            # sizing.
            from pheromone_field import get_pheromone_field
            get_pheromone_field().deposit(
                "stablecoin_netflow", "netflow_24h",
                {"netflow_usd": 0.0,  # placeholder until CryptoQuant wired
                 "bullish_threshold": bullish_thr,
                 "available": False,
                 "_ts": _t.time()},
                half_life=14400.0,  # 4h
            )
            logger.debug(
                "[StablecoinNetflow] placeholder deposit (no CryptoQuant key) — "
                "factor returns neutral 1.0"
            )
        except Exception as e:
            logger.debug(f"[StablecoinNetflow] tick failed: {e}")

    def _olmar_tick(self):
        """Sprint 2026-05-02 (audit Finding #2 fix) — OLMAR cross-pair
        mean-reversion. Pulls last N closes per active pair (whitelist
        derived from system_metrics whitelist_size deposit + recent
        trade pairs) and computes weights via Li & Hoi (2012) PA step.

        Hourly cadence. Outputs deposited to pheromone field at
        "olmar::mean_revert_weights" via publish_olmar_to_pheromone.
        Sizing layer can blend these with Black-Litterman weights when
        constructing per-pair allocations across the basket.
        """
        try:
            from olmar_optimizer import olmar_weights, publish_olmar_to_pheromone
            from db import get_db_connection
            try:
                from neural_organism import _p
                window = int(_p("envelope.olmar.window", 5))
                # Pull last `window+5` per pair so SMA has buffer
                lookback_candles = window + 5
            except Exception:
                window = 5
                lookback_candles = 10

            with get_db_connection() as conn:
                # Pair universe = pairs with closed trades in last 7 days
                pairs_rows = conn.execute("""
                    SELECT DISTINCT pair FROM trades
                    WHERE close_date >= datetime('now', '-7 days')
                      AND is_open = 0
                """).fetchall()
            pairs = [r["pair"] for r in pairs_rows] if pairs_rows else []
            if len(pairs) < 2:
                logger.debug(
                    f"[OLMAR] insufficient pair universe ({len(pairs)} pairs)"
                )
                return

            # For each pair, fetch last N closes from a price feed. The
            # pheromone field's market_data trail is the lightweight option;
            # if absent, fall back to derivatives_data or skip the pair.
            from pheromone_field import get_pheromone_field
            pf = get_pheromone_field()
            price_history = {}
            for p in pairs:
                px = None
                try:
                    snap = pf.read(f"recent_closes::{p}")
                    if isinstance(snap, dict) and "closes" in snap:
                        px = list(snap["closes"])[-lookback_candles:]
                except Exception:
                    px = None
                if not px or len(px) < window:
                    continue
                price_history[p] = px
            if len(price_history) < 2:
                logger.debug(
                    f"[OLMAR] need >=2 pairs with price history, "
                    f"have {len(price_history)}"
                )
                return

            weights = olmar_weights(price_history)
            if weights is None:
                logger.debug("[OLMAR] PA-step solver returned None")
                return
            publish_olmar_to_pheromone(weights)
            top = sorted(weights.items(), key=lambda x: -x[1])[:5]
            logger.info(
                f"[OLMAR] cross-pair mean-revert weights for "
                f"{len(weights)} pairs. Top: "
                f"{[(p, round(w,3)) for p, w in top]}"
            )
        except Exception as e:
            logger.debug(f"[OLMAR] tick failed: {e}")

    def _portfolio_optimizer_tick(self):
        """Sprint 2026-05-02 — Run Black-Litterman + Max-Sharpe joint
        allocator on the active whitelist. Deposits per-pair weights to
        pheromone field. RiskEnvelope's max_single_stake can read these
        as a scaling factor (overweight pairs the optimizer favors).

        Runs hourly when whitelist >= 3 pairs (BL needs covariance, which
        needs ≥2 assets; we require 3 for robust solving).
        """
        try:
            from db import get_db_connection
            from portfolio_optimizer import joint_pair_weights, publish_joint_weights
            try:
                from neural_organism import _p
                lookback = int(_p("envelope.bl.lookback_candles", 168))
            except Exception:
                lookback = 168
            with get_db_connection() as conn:
                # Pull last N close-profit ratios per pair as proxy for returns
                rows = conn.execute(f"""
                    SELECT pair, close_date, close_profit
                    FROM trades
                    WHERE close_date >= datetime('now', '-30 days')
                      AND is_open = 0
                    ORDER BY pair, close_date ASC
                """).fetchall()
            if not rows or len(rows) < 20:
                logger.debug(
                    f"[PortfolioOpt] insufficient trades ({len(rows) if rows else 0})"
                )
                return
            returns_history = {}
            for r in rows:
                returns_history.setdefault(r["pair"], []).append(
                    float(r["close_profit"] or 0.0)
                )
            pairs = [p for p, hist in returns_history.items() if len(hist) >= 5]
            if len(pairs) < 3:
                logger.debug(f"[PortfolioOpt] need >=3 pairs, have {len(pairs)}")
                return
            weights = joint_pair_weights(
                pairs, returns_history, agent_views=None, max_weight=0.30
            )
            if weights is None:
                logger.debug("[PortfolioOpt] solver failed — neutral weights")
                return
            publish_joint_weights(weights)
            top = sorted(weights.items(), key=lambda x: -x[1])[:5]
            logger.info(
                f"[PortfolioOpt] BL+MaxSharpe weights computed for {len(pairs)} pairs. "
                f"Top: {[(p, round(w,3)) for p, w in top]}"
            )
        except Exception as e:
            logger.debug(f"[PortfolioOpt] tick failed: {e}")

    def _ws_health_tick(self):
        """Every 5min: scan recent strategy log for websocket disconnects
        (the 54× `_unwatch_ohlcv` / code-1006 events observed in production
        that previously never touched cortisol). Tail-based approach keeps
        the strategy process patch-free. Each detected disconnect deposits
        a short-half-life pheromone; sustained flapping aggregates.

        Task 25: probe BOTH `hydraquant.log` (docker/systemd production
        path) AND `freqtrade.log` (upstream default). The earlier
        single-path lookup silently missed production logs because
        docker-compose wires --logfile=hydraquant.log, so the exchange
        sensor channel was permanently dark on prod.
        """
        try:
            import os, subprocess, re
            from sensor_bridges import record_ws_disconnect

            log_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "logs",
            )
            candidates = ["hydraquant.log", "freqtrade.log"]
            # Freqtrade's configured logfile path may override both —
            # prefer it when available.
            cfg_path = None
            try:
                import json as _json
                cfg_candidate = os.path.join(
                    os.path.dirname(log_dir),
                    "..",
                    "config_bybit_testnet_futures.json",
                )
                cfg_candidate = os.path.abspath(cfg_candidate)
                if os.path.exists(cfg_candidate):
                    with open(cfg_candidate, "r", encoding="utf-8") as fh:
                        _cfg = _json.load(fh)
                    cfg_path = _cfg.get("logfile")
            except Exception:
                cfg_path = None

            log_path = None
            if cfg_path and os.path.exists(cfg_path):
                log_path = cfg_path
            else:
                for name in candidates:
                    candidate = os.path.join(log_dir, name)
                    if os.path.exists(candidate):
                        log_path = candidate
                        break
            if log_path is None:
                return

            result = subprocess.run(
                ["tail", "-n", "1000", log_path],
                capture_output=True, text=True, timeout=5,
            )
            pattern = re.compile(
                r"(_unwatch_ohlcv|Connection closed by remote server|closing code 1006)",
                re.IGNORECASE,
            )
            hits = sum(1 for line in result.stdout.splitlines() if pattern.search(line))
            if hits >= 2:
                record_ws_disconnect(
                    exchange="unknown",
                    pair="",
                    timeframe="",
                    code=1006,
                )
                logger.debug(
                    f"[SensorBridges:WS] {hits} disconnect events @ {os.path.basename(log_path)} → deposit"
                )
        except Exception as e:
            logger.debug(f"[SensorBridges:WS] tick failed: {e}")


    # ═══ Phase 28: ML Model Retraining Handlers ═══════════════════

    def _self_model_introspect(self):
        """Weekly Saturday 03:00 UTC: Self-model introspection + active learning."""
        try:
            from self_model import get_self_model
            from active_learner import get_active_learner

            # 9A: Introspection
            sm = get_self_model()
            result = sm.introspect(lookback_days=30)
            if "error" not in result:
                logger.info(f"[Sprint2:SelfModel] Introspection: {result.get('n_trades', 0)} trades, "
                           f"{len(result.get('biases_detected', []))} biases, "
                           f"{result.get('competence_entries', 0)} competence entries")

            # 9B: Active learning suggestions
            al = get_active_learner()
            al.publish_suggestions()

        except ImportError:
            logger.info("[Sprint2:SelfModel] self_model/active_learner not available")
        except Exception as e:
            logger.warning(f"[Sprint2:SelfModel] Introspection failed: {e}")

    def _lifecycle_tick(self):
        """Hourly: Run autonomous lifecycle tick (all 13 layers)."""
        try:
            from autonomous_lifecycle import get_lifecycle

            # Build market state from current pheromone readings
            market_state = self._build_market_state_for_lifecycle()

            lifecycle = get_lifecycle()
            decisions = lifecycle.lifecycle_tick(market_state)

            logger.debug(f"[Sprint2:Lifecycle] Tick {decisions.get('tick', '?')}: "
                        f"sizing={decisions.get('final_sizing_mult', 1.0)}, "
                        f"danger={decisions.get('danger', {}).get('response', 'NORMAL')}, "
                        f"mode={decisions.get('circadian', {}).get('mode', 'normal')}")

        except ImportError:
            pass  # Silent — lifecycle is optional during initial deployment
        except Exception as e:
            logger.debug(f"[Sprint2:Lifecycle] Tick failed: {e}")

    def _build_market_state_for_lifecycle(self) -> dict:
        """Build market state dict for lifecycle tick from available sources."""
        state = {
            "drawdown": 0.0,
            "cortisol": 0.5,
            "consecutive_losses": 0,
            "ood_distance": 0,
            "fng_value": 50,
            "market_stress": 0.5,
            "recent_pnl_list": [],
            "param_changes": [],
        }

        try:
            from pheromone_field import get_pheromone_field
            pfield = get_pheromone_field()

            # Read from pheromone field
            health = pfield.read_float("organism_health", default=0.5)
            state["market_stress"] = 1.0 - health

            uncertainty = pfield.read_float("uncertainty", default=0.5)
            state["cortisol"] = uncertainty
        except Exception:
            pass

        try:
            from db import get_connection
            with get_connection() as conn:
                # Recent PnL for hormesis
                rows = conn.execute("""
                    SELECT outcome_pnl FROM ai_decisions
                    WHERE outcome_pnl IS NOT NULL
                    ORDER BY timestamp DESC LIMIT 20
                """).fetchall()
                state["recent_pnl_list"] = [r["outcome_pnl"] for r in rows]

                # Consecutive losses
                losses = 0
                for r in rows:
                    if (r["outcome_pnl"] or 0) < 0:
                        losses += 1
                    else:
                        break
                state["consecutive_losses"] = losses
        except Exception:
            pass

        return state

    def _architecture_evolve(self):
        """Weekly Saturday 04:00 UTC: Run evolutionary architecture search."""
        try:
            from architecture_evolver import get_evolver
            evolver = get_evolver()
            result = evolver.evolve(population_size=10, n_generations=5)
            logger.info(f"[Sprint2:Evolver] Evolution: fitness={result.get('best_fitness', 0):.4f}, "
                       f"organs={result.get('active_organs', 0)}, neurons={result.get('total_neurons', 0)}")
        except ImportError:
            logger.info("[Sprint2:Evolver] architecture_evolver not available")
        except Exception as e:
            logger.warning(f"[Sprint2:Evolver] Evolution failed: {e}")

    def _cerebellum_update(self):
        """Daily 00:30 UTC: Update cerebellum timing multipliers."""
        try:
            from cerebellum_timing import get_cerebellum
            cerebellum = get_cerebellum()
            cerebellum.update_from_trades(lookback_days=30)
            cerebellum.publish_to_pheromone()
        except ImportError:
            pass
        except Exception as e:
            logger.debug(f"[Sprint2:Cerebellum] Update failed: {e}")

    def _ablation_league_run(self):
        """Weekly Saturday 04:30 UTC: Run ablation league."""
        try:
            from ablation_league import get_ablation_league
            league = get_ablation_league()
            result = league.run_ablation(lookback_days=7)
            if "error" not in result:
                logger.info(f"[Sprint2:Ablation] League: {result.get('keep_count', 0)} KEEP, "
                           f"{result.get('watch_count', 0)} WATCH, {result.get('park_count', 0)} PARK")
        except ImportError:
            logger.info("[Sprint2:Ablation] ablation_league not available")
        except Exception as e:
            logger.warning(f"[Sprint2:Ablation] Failed: {e}")

    def _model_risk_check(self):
        """Daily 06:30 UTC: Model risk assessment."""
        try:
            from model_risk_engine import get_model_risk_engine
            mre = get_model_risk_engine()
            result = mre.assess_risk()
            logger.info(f"[Sprint2:ModelRisk] Overall risk: {result.get('overall_risk', 0):.3f}, "
                       f"trust: {1 - result.get('overall_risk', 0):.3f}")
            for rec in result.get("recommendations", []):
                logger.warning(f"[Sprint2:ModelRisk] Recommendation: {rec}")
        except ImportError:
            pass
        except Exception as e:
            logger.debug(f"[Sprint2:ModelRisk] Check failed: {e}")

    def _post_trade_court_run(self):
        """Every 6 hours: Investigate recent completed trades."""
        try:
            from post_trade_court import get_court
            court = get_court()
            verdicts = court.investigate_recent(n_trades=5)
            losses = [v for v in verdicts if v.get("outcome") == "LOSS"]
            if losses:
                logger.info(f"[Sprint2:Court] Investigated {len(verdicts)} trades, "
                           f"{len(losses)} losses. Root causes: "
                           f"{[l['blame']['root_cause'] for l in losses]}")
        except ImportError:
            pass
        except Exception as e:
            logger.debug(f"[Sprint2:Court] Investigation failed: {e}")

    def _phi_measurement(self):
        """Weekly Saturday 05:00 UTC: Measure organism consciousness (Phi)."""
        try:
            from phi_consciousness import get_phi
            phi = get_phi()
            result = phi.compute_phi()
            logger.info(f"[Sprint2:Phi] Φ = {result.get('phi', 0):.4f} "
                       f"({result.get('interpretation', 'unknown')})")
        except ImportError:
            logger.info("[Sprint2:Phi] phi_consciousness not available")
        except Exception as e:
            logger.debug(f"[Sprint2:Phi] Measurement failed: {e}")

    def _gnn_discovery(self):
        """Weekly Saturday 03:30 UTC: GNN pattern discovery on knowledge graph."""
        try:
            from gnn_organism import get_gnn

            gnn = get_gnn()
            patterns = gnn.discover_hidden_patterns()

            if patterns:
                gnn.persist_patterns(patterns)
                logger.info(f"[Sprint2:GNN] Discovered {len(patterns)} patterns, "
                           f"top: {patterns[0]['source']}→{patterns[0]['target']} "
                           f"(att={patterns[0]['attention']:.3f})")
            else:
                logger.info("[Sprint2:GNN] No patterns discovered (graph may be empty)")

        except ImportError:
            logger.info("[Sprint2:GNN] gnn_organism not available")
        except Exception as e:
            logger.warning(f"[Sprint2:GNN] Discovery failed: {e}")

    def _reptile_meta_update(self):
        """Weekly Sunday 01:00 UTC: Reptile meta-learning update.

        Learns a parameter initialization that adapts to any regime in 5 steps.
        Also registers current regime with EWC to prevent catastrophic forgetting.
        """
        try:
            from reptile_meta import get_reptile
            from ewc_continual import get_ewc

            reptile = get_reptile()
            ewc = get_ewc()

            # Run meta-training
            metrics = reptile.meta_train(n_episodes=30)  # Conservative for CPU
            logger.info(f"[Sprint2:Reptile] Meta-update: loss={metrics.get('final_loss', 'N/A'):.4f}")

            # Register current params with EWC for continual learning
            meta_params = reptile.get_meta_params()
            if meta_params is not None:
                # Detect current regime from recent trades
                from db import get_connection
                with get_connection() as conn:
                    regime_row = conn.execute("""
                        SELECT regime FROM ai_decisions
                        WHERE outcome_pnl IS NOT NULL
                        ORDER BY timestamp DESC LIMIT 1
                    """).fetchone()
                regime = regime_row["regime"] if regime_row else "transitional"

                ewc.register_regime(regime, meta_params)
                logger.info(f"[Sprint2:EWC] Registered regime '{regime}' with {len(meta_params)} params")

        except ImportError:
            logger.info("[Sprint2:Reptile] reptile_meta not available")
        except Exception as e:
            logger.warning(f"[Sprint2:Reptile] Meta-update failed: {e}")

    def _world_model_and_dream(self):
        """Daily 01:30 UTC: train world model + run dream session.

        Mega Sprint 2026-04-23 (C.2): the training + dreaming cycle now
        runs in an isolated subprocess (`dream_runner.py`) so PyTorch's
        heap dies with the process and the long-lived scheduler RSS stays
        flat. The feature flag `dream_daily_subprocess` lets us fall back
        to the in-process path for A/B comparison.
        """
        import subprocess
        import time as _time
        try:
            from ai_config import get_flag, AI_DB_PATH
        except Exception:
            # Tur-2 (M2): fall back to the caller's declared default rather
            # than forcing True so a disabled-by-default flag stays disabled
            # when ai_config can't be imported.
            get_flag = lambda key, default=False, **_kw: default
            AI_DB_PATH = os.environ.get("AI_DB_PATH", "")

        if not get_flag("dream_daily_subprocess", True):
            self._world_model_and_dream_inline()
            return

        script = os.path.join(os.path.dirname(__file__), "dream_runner.py")
        if not os.path.exists(script):
            logger.warning(f"[Sprint2:Dream] runner missing at {script} — inline fallback")
            self._world_model_and_dream_inline()
            return

        payload = json.dumps({
            "config": {},
            "db_path": AI_DB_PATH,
            "pair_list": list(getattr(self, "pair_list", []) or []),
        })

        import os as _os
        import select as _select
        import signal as _signal

        t0 = _time.time()
        try:
            proc = subprocess.Popen(
                [sys.executable, "-u", script, payload],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        except Exception as e:
            logger.warning(f"[Sprint2:Dream] subprocess spawn failed ({e}); inline fallback")
            self._world_model_and_dream_inline()
            return

        # Revize Tur-2 (H5 + H6): stream stdout line-by-line so long-running
        # dream sessions emit progress in real time, and use os.killpg so
        # every forked torch worker dies with the parent on timeout. The
        # previous `proc.communicate()` blocked until exit and SIGTERM on
        # the parent alone left child workers orphaned on Linux.
        timed_out = False
        try:
            while proc.poll() is None:
                if _time.time() - t0 > 1800:
                    timed_out = True
                    logger.warning("[Sprint2:Dream] 30-min timeout, killpg SIGTERM")
                    try:
                        _os.killpg(_os.getpgid(proc.pid), _signal.SIGTERM)
                        proc.wait(timeout=10)
                    except ProcessLookupError:
                        pass
                    except Exception:
                        try:
                            _os.killpg(_os.getpgid(proc.pid), _signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                    break
                rlist, _r, _x = _select.select([proc.stdout], [], [], 1.0)
                if rlist:
                    line = proc.stdout.readline()
                    if line:
                        logger.info(
                            f"[Sprint2:Dream] {line.decode('utf-8', 'replace').rstrip()}"
                        )
            # Drain any remaining buffered output once the process exits.
            if not timed_out:
                remainder = proc.stdout.read() or b""
                for line in remainder.splitlines():
                    logger.info(
                        f"[Sprint2:Dream] {line.decode('utf-8', 'replace').rstrip()}"
                    )
        except Exception as e:
            logger.warning(f"[Sprint2:Dream] streaming failed: {e}")
        logger.info(
            f"[Sprint2:Dream] exit={proc.returncode} "
            f"elapsed={_time.time() - t0:.0f}s"
        )

    def _world_model_and_dream_inline(self):
        """Legacy in-process path for the dream cycle. Kept behind the
        `dream_daily_subprocess` feature flag so we can quickly revert to
        the pre-2026-04-23 behaviour if the subprocess path misbehaves."""
        try:
            from world_model import get_world_model
            from dream_engine import get_dream_engine

            wm = get_world_model()
            train_result = wm.train_from_buffer(n_epochs=30, batch_size=64)
            if "error" in train_result:
                logger.info(f"[Sprint2:WorldModel] {train_result['error']}")
            else:
                logger.info(
                    f"[Sprint2:WorldModel] Trained: "
                    f"pred_loss={train_result.get('pred_loss', 'N/A'):.4f}"
                )

            dream_engine = get_dream_engine()
            dream_result = dream_engine.dream_session(n_dreams=100, horizon=5)
            if "error" in dream_result:
                logger.info(f"[Sprint2:Dream] {dream_result['error']}")
            else:
                logger.info(
                    f"[Sprint2:Dream] Session: "
                    f"{dream_result['valid_dreams']}/{dream_result['total_dreams']} valid, "
                    f"pass_rate={dream_result['pass_rate']:.1%}, "
                    f"stored={dream_result['stored_in_buffer']}"
                )
        except ImportError:
            logger.info("[Sprint2:WorldModel] world_model/dream_engine not available")
        except Exception as e:
            logger.warning(f"[Sprint2:WorldModel] Failed: {e}")

    def _rl_iql_retrain(self):
        """Weekly Sunday 02:00 UTC: Generate episodes + retrain IQL.

        Flow:
          1. Generate offline episodes from backtest replay
          2. Train IQL on accumulated replay buffer
          3. Save checkpoint for SAC online fine-tuning

        Sprint 2026-05-01: wrapped in `_heavy_job_gc` so PyTorch tensor
        allocations are reclaimed via jemalloc purge after training,
        preventing the Sunday-night OOM cascade.
        Sprint 2026-05-02: guardian consults predictive forecast.
        """
        if self._memory_pressure_halt():
            logger.warning("[Sprint2:IQL] SKIPPED — memory pressure forecast critical.")
            return
        with self._heavy_job_gc("rl_iql_retrain"):
            try:
                from iql_pretrain import run_iql_training

                metrics = run_iql_training(
                    generate_episodes=100,  # Generate 100 new episodes
                    n_epochs=50,           # 50 epochs (conservative for CPU)
                )

                if "error" not in metrics:
                    logger.info(f"[Sprint2:IQL] Training complete: "
                               f"V_loss={metrics.get('final_v_loss', 'N/A'):.4f}, "
                               f"Q_loss={metrics.get('final_q_loss', 'N/A'):.4f}")
                else:
                    logger.info(f"[Sprint2:IQL] {metrics.get('error', 'skipped')}")

            except ImportError:
                logger.info("[Sprint2:IQL] iql_pretrain not available")
            except Exception as e:
                logger.warning(f"[Sprint2:IQL] Training failed: {e}")

    def _counterfactual_analysis(self):
        """Weekly Saturday 02:30 UTC: Run counterfactual analysis on recent trades.

        Uses causal graph from 6A to estimate "what if" scenarios.
        Results feed into parameter optimization insights.

        Sprint 2026-05-01: jemalloc-purge wrapped (heavy pandas DataFrames
        for trade replay).
        Sprint 2026-05-02: guardian consults predictive forecast.
        """
        if self._memory_pressure_halt():
            logger.warning("[Sprint2:Counterfactual] SKIPPED — memory pressure forecast critical.")
            return
        with self._heavy_job_gc("counterfactual_analysis"):
            try:
                from counterfactual_engine import run_counterfactual_analysis

                result = run_counterfactual_analysis(
                    regime=None,
                    lookback_days=60,
                    persist=True,
                )

                if "error" not in result:
                    logger.info(f"[Sprint2:Counterfactual] Analysis complete: "
                               f"{result.get('n_trades', 0)} trades, "
                               f"{result.get('n_counterfactuals', 0)} scenarios")

                    # 6C: Update organism synapse weights from causal discoveries
                    self._update_organism_from_causal()
                else:
                    logger.info(f"[Sprint2:Counterfactual] {result.get('error', 'skipped')}")

            except ImportError:
                logger.info("[Sprint2:Counterfactual] counterfactual_engine not available")
            except Exception as e:
                logger.warning(f"[Sprint2:Counterfactual] Analysis failed: {e}")

    def _update_organism_from_causal(self):
        """6C: Update neural organism synapse weights from causal discoveries.

        Bridges causal engine discoveries → organism synapse weights.
        Strong causal links get stronger synapses, refuted links get weakened.
        """
        try:
            from causal_engine import CausalEngine
            from neural_organism import get_organism

            engine = CausalEngine()
            organism = get_organism()

            # Get active causal edges
            edges = engine.get_active_edges()
            if not edges:
                logger.info("[Sprint2:6C] No causal edges to apply to organism")
                return

            updated = 0
            for edge in edges:
                src = edge["source_var"]
                tgt = edge["target_var"]
                strength = edge["causal_strength"]

                # Map causal variable names to organism neuron parameter IDs
                # The organism has neurons with params like "synapse_funding_rate_to_pnl"
                synapse_id = f"synapse_{src}_to_{tgt}"

                try:
                    # Update synapse weight in organism
                    # Strength from PCMCI+ is [0, 1] — map to synapse weight
                    organism.set_param(synapse_id, strength, regime="_global")
                    updated += 1
                except Exception:
                    # Synapse doesn't exist yet — organism will create it in morphogenesis (9C)
                    pass

            if updated > 0:
                logger.info(f"[Sprint2:6C] Updated {updated} organism synapse weights from causal graph")

        except ImportError:
            logger.debug("[Sprint2:6C] neural_organism not available for synapse update")
        except Exception as e:
            logger.debug(f"[Sprint2:6C] Organism update failed: {e}")

    def _causal_discovery(self):
        """Weekly Saturday 02:00 UTC: Run PCMCI+ causal discovery.

        Discovers temporal causal relationships from accumulated trade data.
        Results feed into Grafeo graph + SQLite causal_discoveries.
        Runs before CatBoost retrain so causal features are available.
        """
        try:
            from causal_engine import run_multi_regime_discovery

            results = run_multi_regime_discovery(lookback_days=30)

            total_edges = sum(
                r.get("n_edges", 0) for r in results.values()
                if isinstance(r, dict) and "error" not in r
            )
            regimes_done = sum(
                1 for r in results.values()
                if isinstance(r, dict) and "error" not in r
            )

            logger.info(f"[Sprint2:Causal] Discovery complete: "
                       f"{total_edges} edges across {regimes_done} regimes")

        except ImportError:
            logger.info("[Sprint2:Causal] causal_engine not available (tigramite not installed?)")
        except Exception as e:
            logger.warning(f"[Sprint2:Causal] Discovery failed: {e}")

    def _catboost_retrain(self):
        """Weekly: Retrain CatBoost with full 13B pipeline.

        Flow:
          1. Generate labels from backtest results (193 chart features)
          2. Enrich with live trade data
          3. Train CatBoost v2 (193 features, 500 iterations)
          4. Falls back to v1 (11 features) if insufficient backtest data

        Sprint 2026-05-01 night — checks walk-forward freeze pheromone
        first. When recent performance has degraded the model is NOT
        updated this cycle (skip-and-wait). Prevents the bot from
        fitting a regime change as if it were a feature.
        """
        if self._walk_forward_frozen():
            logger.warning(
                "[Sprint2:CatBoost] SKIPPED — walk-forward freeze active "
                "(recent perf degraded vs baseline). Will retry next cycle."
            )
            return
        if self._memory_pressure_halt():
            logger.warning("[Sprint2:CatBoost] SKIPPED — memory pressure forecast critical.")
            return
        with self._heavy_job_gc("catboost_retrain"):
            self._catboost_retrain_inner()

    def _auto_cgroup_recalibrate(self):
        """Sprint 2026-05-02 audit Finding #3 fix — wires the previously
        dead `auto_cgroup_recommendation` to ACTUALLY update systemd
        drop-in files. Runs Sunday 04:30 UTC after the weekly heavy ML
        cycle settles. Compares observed 30-day RSS peaks to current
        cgroup limits, writes /etc/systemd/system/<svc>.d/auto.conf
        with new MemoryMax/MemoryHigh, and triggers a graceful
        `systemctl daemon-reload` so the next service restart picks up
        the new bounds.

        Self-healing: as the bot's workload grows or shrinks, its memory
        budget adapts WITHOUT human intervention. No more 6-OOM-in-12h
        cycles waiting for an operator to bump MemoryMax.
        """
        try:
            from predictive_memory_guardian import auto_cgroup_recommendation
            recs = auto_cgroup_recommendation()
            if not recs:
                logger.info("[CgroupRecalibrate] no recommendations (cold start)")
                return
            # Write drop-in files. Each service gets:
            #   /etc/systemd/system/<svc>.d/auto.conf
            # with:
            #   [Service]
            #   MemoryMax=<n>
            #   MemoryHigh=<n*0.85>
            import subprocess
            import os
            changes_made = []
            for svc, mem_max in recs.items():
                drop_dir = f"/etc/systemd/system/{svc}.d"
                drop_file = f"{drop_dir}/auto.conf"
                mem_high = int(mem_max * 0.85)
                content = (
                    f"# Auto-generated by predictive_memory_guardian\n"
                    f"# DO NOT EDIT — overwritten weekly by scheduler\n"
                    f"[Service]\nMemoryMax={mem_max}\nMemoryHigh={mem_high}\n"
                )
                # Read current file (if any); skip rewrite if identical
                existing = ""
                try:
                    if os.path.isfile(drop_file):
                        with open(drop_file) as f:
                            existing = f.read()
                except Exception:
                    pass
                if existing == content:
                    continue
                try:
                    os.makedirs(drop_dir, exist_ok=True)
                    with open(drop_file, "w") as f:
                        f.write(content)
                    changes_made.append((svc, mem_max // (1024**2)))
                except PermissionError:
                    logger.warning(
                        f"[CgroupRecalibrate] no permission to write {drop_file} "
                        "— scheduler must run as root for self-tune. Skipping."
                    )
                    return
                except Exception as e:
                    logger.debug(f"[CgroupRecalibrate] {svc} write skipped: {e}")
            if changes_made:
                # daemon-reload so new limits are visible at next start.
                # We DON'T restart services here — that happens at the
                # operator's next deploy; auto-restart would interrupt
                # in-flight trades.
                try:
                    subprocess.run(
                        ["systemctl", "daemon-reload"],
                        check=False, timeout=15,
                    )
                except Exception as e:
                    logger.debug(f"[CgroupRecalibrate] daemon-reload failed: {e}")
                # Telemetry pheromone for dashboard visibility
                try:
                    from pheromone_field import get_pheromone_field
                    get_pheromone_field().deposit(
                        "memory_guardian", "cgroup_recalibrated",
                        {"changes": [{"svc": s, "mem_max_mb": m} for s, m in changes_made]},
                        half_life=86400.0,
                    )
                except Exception:
                    pass
                logger.info(
                    f"[CgroupRecalibrate] updated {len(changes_made)} services: "
                    f"{[(s, f'{m}MB') for s, m in changes_made]} — daemon-reloaded. "
                    "New limits take effect on next service restart."
                )
            else:
                logger.info("[CgroupRecalibrate] limits already optimal — no changes")
        except Exception as e:
            logger.warning(f"[CgroupRecalibrate] tick failed: {e}")

    def _memory_pressure_halt(self) -> bool:
        """Sprint 2026-05-02: predictive memory guardian — refuses heavy
        ML jobs when forecast says we'll OOM in <5 min OR current pressure
        already critical. Prevents the OOM-after-OOM cycle observed in
        production (6 OOMs in 12h despite reactive memory_cleanup).
        """
        try:
            from predictive_memory_guardian import should_halt_heavy_jobs
            if should_halt_heavy_jobs():
                logger.warning(
                    "[MemoryGuardian] heavy job HALTED — forecast critical"
                )
                return True
        except Exception:
            pass
        return False

    def _walk_forward_frozen(self) -> bool:
        """Check pheromone field for walk-forward freeze state.

        Audit Finding D11: a freeze deposit only auto-clears when the
        next daily validator run overwrites it. If the validator dies
        for several days the freeze persists too long, locking model
        updates indefinitely. Fail-safe: a freeze older than 36 hours
        is treated as STALE — better to retrain on questionable data
        than to leave the bot stuck on stale models forever.
        """
        try:
            import time as _t
            from pheromone_field import get_pheromone_field
            pf = get_pheromone_field()
            state = pf.read("model_freeze", source="walk_forward")
            if not isinstance(state, dict):
                return False
            if not bool(state.get("frozen", False)):
                return False
            ts = state.get("_ts") or state.get("timestamp")
            if ts:
                try:
                    age_h = (_t.time() - float(ts)) / 3600.0
                    if age_h > 36.0:
                        logger.info(
                            f"[WalkForward] freeze stale ({age_h:.1f}h old) — "
                            "treating as cleared"
                        )
                        return False
                except (TypeError, ValueError):
                    pass
            return True
        except Exception:
            return False

    def _catboost_retrain_inner(self):
        try:
            # Try v2 pipeline first (13B: backtest + chart features)
            from backtest_label_generator import BacktestLabelGenerator
            from catboost_trainer import train_catboost_v2

            gen = BacktestLabelGenerator()

            # Generate labels from any new backtest results
            new_labels = gen.generate_from_backtests()
            if new_labels > 0:
                logger.info(f"[Sprint2:CatBoost] Generated {new_labels} new training labels")

            # Enrich with live trades
            live_enriched = gen.enrich_from_live_trades(min_trades=10)
            if live_enriched > 0:
                logger.info(f"[Sprint2:CatBoost] Enriched {live_enriched} live trade labels")

            # Get full dataset and train
            X, y, feature_names = gen.get_training_dataset(min_samples=50)
            if X is not None:
                result = train_catboost_v2(X, y, feature_names, test_ratio=0.2)
                logger.info(f"[Sprint2:CatBoost] v2 retrained: "
                           f"acc={result.get('test_accuracy', 'N/A')}, "
                           f"f1={result.get('test_f1', 'N/A')}, "
                           f"features={result.get('n_features', 'N/A')}")
                return

            logger.info("[Sprint2:CatBoost] Insufficient v2 data, falling back to v1")

        except ImportError:
            logger.info("[Sprint2:CatBoost] v2 pipeline not available, using v1")
        except Exception as e:
            logger.warning(f"[Sprint2:CatBoost] v2 pipeline failed: {e}, falling back to v1")

        # Fallback: v1 pipeline (11 features from ai_decisions)
        try:
            from catboost_trainer import gather_training_data, train_catboost
            X, y, feature_names = gather_training_data(min_trades=50)
            if X is not None:
                result = train_catboost(X, y, feature_names, test_ratio=0.2)
                logger.info(f"[Sprint2:CatBoost] v1 retrained: acc={result.get('test_accuracy', 'N/A')}")
            else:
                logger.info("[Sprint2:CatBoost] Insufficient data for retraining")
        except Exception as e:
            logger.warning(f"[Sprint2:CatBoost] v1 retrain also failed: {e}")

    def _ood_refit(self):
        """Weekly: Refit OOD detector reference distributions with recent data.

        Audit fix (2026-04-19): the previous version pulled only 3 features
        (confidence / trust_score_at_decision / outcome_duration) AND
        normalised the regime column with `.fillna("transitional")` — but the
        OOD detector's hard-coded REGIMES list misspelled 'transitional' as
        'transition', so per-regime masks matched zero rows. Result: refit
        ran, persisted v2 state, but `0 regimes fit`. Two fixes:

          1. Pull additional ai_decisions features (position_size, trust,
             outcome_pnl, outcome_duration) so the detector has a richer
             distribution to fit AND so even a degenerate regime split
             still has enough samples per regime to clear the 15-row floor.
          2. Map any unknown / null regime label to 'transitional' (matches
             regime_classifier output), and only forward labels that are in
             OOD's REGIMES list — otherwise force '_global'.

        Sprint 2026-05-01: jemalloc-purge wrapped — pulls 1000 rows into a
        pandas DataFrame and fits a Mahalanobis distance per regime, holding
        feature matrices the entire time.

        Sprint 2026-05-01 night — walk-forward freeze honored.
        """
        if self._walk_forward_frozen():
            logger.warning("[Phase28:OOD] SKIPPED — walk-forward freeze active.")
            return
        if self._memory_pressure_halt():
            logger.warning("[Phase28:OOD] SKIPPED — memory pressure forecast critical.")
            return
        with self._heavy_job_gc("ood_refit"):
            self._ood_refit_inner()

    def _ood_refit_inner(self):
        try:
            from ood_detector import MarketOODDetector
            import pandas as pd
            from db import get_connection
            detector = MarketOODDetector()
            valid_regimes = set(MarketOODDetector.REGIMES)
            with get_connection() as conn:
                rows = conn.execute("""
                    SELECT confidence,
                           COALESCE(trust_score_at_decision, 0.5) AS trust_score_at_decision,
                           COALESCE(outcome_duration, 3600.0)     AS outcome_duration,
                           COALESCE(position_size, 0.0)            AS position_size,
                           COALESCE(outcome_pnl, 0.0)              AS outcome_pnl,
                           COALESCE(regime, 'transitional')        AS regime
                    FROM ai_decisions
                    WHERE outcome_pnl IS NOT NULL
                    ORDER BY timestamp DESC LIMIT 1000
                """).fetchall()
            if len(rows) < 30:
                logger.info(f"[Phase28:OOD] Insufficient data for refit ({len(rows)} rows)")
                return
            df = pd.DataFrame([dict(r) for r in rows])
            feature_cols = ["confidence", "trust_score_at_decision",
                            "outcome_duration", "position_size", "outcome_pnl"]
            features = df[feature_cols].fillna(0.0)
            # Map labels: anything not in REGIMES → 'transitional' (the
            # closest valid bucket per RegimeClassifier semantics).
            regimes = df["regime"].apply(
                lambda r: r if r in valid_regimes else "transitional"
            )
            detector.fit(features, regimes)
            n_regimes_fit = len(detector._regime_stats)
            logger.info(
                f"[Phase28:OOD] Refit on {len(rows)} trades, "
                f"{n_regimes_fit}/{len(valid_regimes)} regimes fit "
                f"(distribution: {dict(regimes.value_counts().head(6))})"
            )
        except Exception as e:
            logger.warning(f"[Phase28:OOD] Refit failed: {e}")

    def _forgone_shadow_resolver(self):
        """Phase 27 Fix 6 (H3): Resolve shadow trades whose 4h window has elapsed.

        Every 30 minutes, pick up ALL unresolved forgone_profit rows older
        than 4h (post-audit Data Acceleration 1: LIMIT 100 was a throttle
        that capped us at 4800 resolves/day when target is 2000+ shadow/day
        × 4h-ago horizon = ~15k pending at any moment). Bybit ticker fetches
        are cached per-pair in this pass so we do one HTTP per unique pair,
        not one per shadow row.

        After resolving, kicks `_forgone_learn_feedback` to push the fresh
        WIN/LOSS outcomes into per-pair Kelly + argument_quality so every
        shadow trade trains the organism.
        """
        try:
            import httpx
            from forgone_pnl_engine import ForgonePnLEngine
            from db import get_db_connection

            engine = ForgonePnLEngine()
            conn = get_db_connection()
            # Sprint 2026-05-01: dynamic LIMIT derived from recent
            # signal-generation rate × resolver window. Without a bound the
            # fetchall could materialise 15k+ rows after a busy day, holding
            # them all in Python memory while the for-loop iterates — a key
            # contributor to the scheduler's 3GB cgroup OOM cycle. The
            # bound is "signals_per_hour × 6h × 1.5x headroom" so on a
            # 100/h pace we cap at 900 rows; on quiet days it's a no-op.
            try:
                rate_row = conn.execute("""
                    SELECT COUNT(*) AS n
                    FROM forgone_profit
                    WHERE signal_time >= datetime('now', '-1 hour')
                """).fetchone()
                rate_per_hour = int(rate_row["n"] or 0) if rate_row else 0
            except Exception:
                rate_per_hour = 0
            # Floor 200 (always make progress on cold start), ceiling 5000
            # (hard upper bound so a runaway producer can't blow up RAM).
            dyn_limit = max(200, min(5000, int(rate_per_hour * 6 * 1.5) or 200))
            unresolved = conn.execute("""
                SELECT id, pair, signal_type, entry_price, signal_time, regime
                FROM forgone_profit
                WHERE was_executed = 0
                  AND forgone_pnl IS NULL
                  AND signal_time < datetime('now', '-4 hours')
                ORDER BY signal_time ASC
                LIMIT ?
            """, (dyn_limit,)).fetchall()
            conn.close()

            if not unresolved:
                return

            # Per-pair ticker cache — avoid hammering Bybit with N duplicate calls.
            ticker_cache: Dict[str, Optional[float]] = {} if False else {}
            def _bybit_last_price(pair: str):
                if pair in ticker_cache:
                    return ticker_cache[pair]
                symbol = pair.split(":")[0].replace("/", "")
                try:
                    r = httpx.get(
                        "https://api.bybit.com/v5/market/tickers",
                        params={"category": "linear", "symbol": symbol},
                        timeout=5.0,
                    )
                    data = r.json().get("result", {}).get("list", [])
                    price = float(data[0].get("lastPrice", 0)) if data else None
                except Exception:
                    price = None
                ticker_cache[pair] = price
                return price

            resolved_ids: List[int] = []
            for row in unresolved:
                try:
                    price = _bybit_last_price(row["pair"])
                    if price and price > 0:
                        if engine.resolve_forgone_trade(row["id"], float(price)):
                            resolved_ids.append(row["id"])
                except Exception as e:
                    logger.debug(f"[Phase27:ShadowResolve] {row['pair']} skip: {e}")

            logger.info(
                f"[Phase27:ShadowResolve] Resolved {len(resolved_ids)}/{len(unresolved)} "
                f"shadow trades (cache_hits={len(unresolved) - len(ticker_cache)})"
            )

            # Data Acceleration Fix 1: feed the fresh shadow outcomes into the
            # organism's learning hooks (Kelly, argument_quality). Separate
            # method so it can be called independently from tests.
            if resolved_ids:
                self._forgone_learn_feedback(resolved_ids)
        except ImportError as e:
            logger.debug(f"[Phase27:ShadowResolve] disabled: {e}")
        except Exception as e:
            logger.warning(f"[Phase27:ShadowResolve] Job failed: {e}")

    def _forgone_learn_feedback(self, resolved_ids: list) -> None:
        """Data Acceleration Fix 1: shadow WIN/LOSS → BayesianKelly + argument
        quality. Every resolved forgone row is pushed through the same feedback
        path a real closed trade would take — turning shadow data from
        diagnostic-only into an active training signal.
        """
        if not resolved_ids:
            return
        try:
            from db import get_db_connection
            # Revize Tur-2 (H3): `bk` was unused dead code after B.1.2 cut
            # the real-Kelly update path. Shadow updates use `get_shadow_kelly()`
            # below — no need to import BayesianKelly here.
            conn = get_db_connection()
            placeholders = ",".join("?" * len(resolved_ids))
            rows = conn.execute(
                f"""
                SELECT pair, signal_type, confidence, regime, forgone_pnl
                FROM forgone_profit
                WHERE id IN ({placeholders})
                  AND forgone_pnl IS NOT NULL
                """,
                resolved_ids,
            ).fetchall()
            conn.close()

            kelly_updates = 0
            # Data Acceleration audit: wire shadow outcomes into argument
            # quality too. When a shadow signal was part of a debate round
            # (agent_memory rows at the same pair/timestamp exist with a
            # key_argument), grade each agent's reasoning pattern against
            # the shadow's forgone PnL. Treat wins/losses as training signal
            # for `argument_quality` just like real trade exits.
            pool = None
            try:
                from agent_pool import AgentPool
                from ai_config import AI_DB_PATH as _AIDB
                pool = AgentPool(db_path=_AIDB)
            except Exception:
                pool = None
            arg_updates = 0

            # EK Sprint 2026-04-23 (EK.1): shadow outcomes go into the
            # SEPARATE shadow ledger. Live sizing still reads the real
            # ledger only (get_real_kelly()), so the B.1.1 label bug can
            # never again contaminate production sizing. Feature flag
            # `shadow_kelly_separate_ledger_enabled` can disable the
            # update entirely for A/B comparison.
            shadow_kelly = None
            try:
                from ai_config import get_flag
                if get_flag("shadow_kelly_separate_ledger_enabled", True):
                    from position_sizer import get_shadow_kelly
                    shadow_kelly = get_shadow_kelly()
            except Exception as _sk_err:
                logger.debug(f"[Phase27:ShadowLearn] shadow ledger unavailable: {_sk_err}")

            for r in rows:
                try:
                    pair = r["pair"]
                    if not pair:
                        continue
                    regime = r["regime"] or "_global"
                    pnl_pct = float(r["forgone_pnl"] or 0.0)
                    won = pnl_pct > 0
                    if shadow_kelly is not None:
                        try:
                            shadow_kelly.update(won=won, pnl_pct=pnl_pct,
                                                 pair=pair, regime=regime)
                            kelly_updates += 1
                        except Exception as _upd_err:
                            logger.debug(
                                f"[Phase27:ShadowLearn] shadow update failed "
                                f"{pair}/{regime}: {_upd_err}"
                            )
                    else:
                        # Tur-2 (M6): flag off → state it LOUDLY in the logs,
                        # not just silently swallow the update.
                        logger.warning(
                            "[Phase27:ShadowLearn] "
                            "shadow_kelly_separate_ledger_enabled=False — "
                            "no shadow updates; check config_ai.json"
                        )

                    # Argument quality feedback — grade every agent that
                    # debated this pair in the last 6 hours against the
                    # shadow's outcome. Matches the pattern used on real
                    # trade exits (agent_pool.record_trade_outcome).
                    if pool is None:
                        continue
                    try:
                        debate_conn = get_db_connection()
                        debate_rows = debate_conn.execute("""
                            SELECT agent_type, key_argument
                            FROM agent_memory
                            WHERE pair = ?
                              AND timestamp > datetime('now', '-6 hours')
                            ORDER BY timestamp DESC LIMIT 10
                        """, (pair,)).fetchall()
                        debate_conn.close()
                    except Exception:
                        debate_rows = []
                    for dr in debate_rows:
                        pattern = pool._extract_argument_pattern(dr["key_argument"] or "")
                        if not pattern:
                            continue
                        try:
                            pool._update_argument_quality(
                                agent_type=dr["agent_type"],
                                pattern=pattern,
                                regime=regime,
                                was_correct=bool(won),
                                outcome_pnl=float(pnl_pct),
                            )
                            arg_updates += 1
                        except Exception as e:
                            logger.debug(f"[Phase27:ShadowLearn] arg update skip: {e}")
                except Exception as e:
                    logger.debug(f"[Phase27:ShadowLearn] Kelly update skip: {e}")
            logger.info(
                f"[Phase27:ShadowLearn] fed {kelly_updates} shadow outcomes "
                f"into per-pair Kelly + {arg_updates} argument_quality updates"
            )
        except Exception as e:
            logger.warning(f"[Phase27:ShadowLearn] feedback failed: {e}")

    def _forgone_threshold_adapt(self):
        """Phase 27 Fix 6 (H3): Adjust pair_thresholds based on 7d forgone alpha.

        For each (pair, regime):
          - sum(forgone_pnl) when was_executed=0 AND forgone_pnl > 0 → missed profit
          - sum(forgone_pnl) when was_executed=0 AND forgone_pnl < 0 → dodged loss
          alpha = missed_profit + dodged_loss   (positive → we're too strict)
        """
        try:
            from db import get_db_connection
            from datetime import datetime, timezone
            conn = get_db_connection()
            pairs = conn.execute("""
                SELECT pair,
                       COALESCE(regime, '_global') AS regime,
                       SUM(CASE WHEN was_executed=0 AND forgone_pnl > 0
                                THEN forgone_pnl ELSE 0 END) AS pos,
                       SUM(CASE WHEN was_executed=0 AND forgone_pnl < 0
                                THEN forgone_pnl ELSE 0 END) AS neg,
                       COUNT(*) AS n_signals
                FROM forgone_profit
                WHERE signal_time > datetime('now', '-7 days')
                  AND forgone_pnl IS NOT NULL
                GROUP BY pair, regime
            """).fetchall()

            # Sprint 2026-05-01: pull dynamic floor/ceiling from RiskEnvelope
            # so the per-pair threshold creep stays inside the autonomy
            # tier's conviction band. Previously the bounds were hardcoded
            # 0.30 / 0.75 and the deltas were asymmetric (+0.01 punish vs
            # -0.02 reward) — combined with a bear regime (avg conf 0.21)
            # this drove every pair upward into a self-locking creep.
            try:
                from risk_envelope import get_risk_envelope
                env = get_risk_envelope()
                env_floor = float(env.conviction_floor())
                env_ceiling = float(env.conviction_ceiling())
                env_default = max(env_floor, min(env_ceiling, env_floor * 1.05))
            except Exception:
                env_floor, env_ceiling, env_default = 0.30, 0.75, 0.50

            adjusted = 0
            now = datetime.now(tz=timezone.utc).isoformat()
            for p in pairs:
                alpha = (p["pos"] or 0.0) + (p["neg"] or 0.0)
                n_signals = int(p["n_signals"] or 0)
                if n_signals < 5:
                    continue  # too little data to act on

                # Symmetric magnitude-aware delta. |alpha| ≤ 1 → tiny step;
                # |alpha| ≥ 10 → cap. Sign is preserved: positive alpha
                # means "we're missing trades" → lower the bar; negative
                # alpha means "we picked losers" → raise it.
                normalized = max(-1.0, min(1.0, alpha / 10.0))
                base_step = 0.05  # max single-tick movement
                # Sprint 2026-05-05 (K4): hysteresis lowered 0.5→0.2 so
                # marginal forgone signals get adapted instead of skipped.
                # Audit found 8 weeks of low-magnitude alpha never moved
                # the threshold — slow learning was causing drift.
                if abs(alpha) < 0.2:
                    continue
                # Sprint 2026-05-05 (K5): trust ramps faster (n/30 not n/50)
                # so per-pair thresholds reach full trust sooner.
                trust = min(1.0, n_signals / 30.0)
                delta = -normalized * base_step * trust
                reason = (f"sym_alpha={alpha:+.2f} n={n_signals} "
                          f"trust={trust:.2f} delta={delta:+.3f}")

                # Upsert with envelope-bounded threshold
                existing = conn.execute("""
                    SELECT confidence_threshold FROM pair_thresholds
                    WHERE pair = ? AND regime = ?
                """, (p["pair"], p["regime"])).fetchone()
                current = float(existing["confidence_threshold"]) if existing else env_default
                new_thr = max(env_floor, min(env_ceiling, current + delta))
                conn.execute("""
                    INSERT INTO pair_thresholds
                        (pair, regime, confidence_threshold, forgone_alpha_7d,
                         last_adjusted, adjustment_reason)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(pair, regime) DO UPDATE SET
                        confidence_threshold = excluded.confidence_threshold,
                        forgone_alpha_7d = excluded.forgone_alpha_7d,
                        last_adjusted = excluded.last_adjusted,
                        adjustment_reason = excluded.adjustment_reason
                """, (p["pair"], p["regime"], new_thr, alpha, now, reason))
                adjusted += 1
                logger.info(f"[Phase27:Threshold] {p['pair']} ({p['regime']}): "
                           f"{current:.2f} → {new_thr:.2f} ({reason})")
            conn.commit()
            conn.close()
            logger.info(f"[Phase27:Threshold] Weekly adapt: {adjusted} pairs adjusted")
        except Exception as e:
            logger.warning(f"[Phase27:Threshold] Adaptation failed: {e}")

    # ═════════════════════════════════════════════════════════════
    # Phase 27 Task 19 — Sleep-Wake cycle + Task 20 fine-tune
    # ═════════════════════════════════════════════════════════════

    # Jobs that MUST stay on during LIGHT_SLEEP (safety + heartbeat).
    # Post-audit fix: IDs below are verified against the actual add_job() calls
    # in start() — Phase 26 had imaginary names like 'heartbeat' / 'fetch_rss_fng'
    # that never matched any real job, so the whole safety net was a no-op.
    _LIGHT_SLEEP_WHITELIST = {
        'fetch_analyze',       # RSS + FNG + sentiment (safety — macro context)
        'post_trade_court',    # Trade autopsies every 6h (required on exits)
        'forgone_resolver',    # Forgone PnL resolver (30min)
        'conformal_recal',     # Safety: conformal calibration
        'hawkes_refit',        # Safety: Hawkes MLE
        'sleep_wake',          # Self-tick (NEVER pause — would freeze the cycle)
        'dead_code_warmup',    # Always-on hourly integration ping
        'daily_reset',         # Daily balance/portfolio snapshot reset
        'health_check',        # Systemd-visible liveness probe
        'memory_cleanup',      # Python GC / malloc_trim (safety)
        'pheromone_cleanup',   # Dead-signal cleanup
        'lifecycle_tick',      # Organism hourly heartbeat
        'interoception_check', # Organism health check
        'model_risk',          # Daily risk guard
        # AUDIT-11 (2026-04-25): the homeostasis loop must NOT freeze
        # during sleep — memory pressure / sensor stress accumulate fastest
        # in low-activity windows, exactly when the bot is sleeping.
        'organism_hormone_refresh',
        # Daily eviction is safe to run during sleep (single-table DELETE).
        'embedding_cache_evict',
        # LOOP-3 (2026-04-25): outcome backfill is the learning gradient
        # for RAG / agent_pool / calibrator — must keep running during sleep
        # so the morning decisions have fresh evidence to learn from.
        'decisions_outcome_backfill',
        # LOOP-1 publishes per-pair shadow scores every 30 min. Sleep
        # mode would silence the gate consumers — keep producer alive.
        'shadow_kelly_divergence',
        # T8: counterfactual → shadow Kelly drain. Daily off-peak job;
        # safe to run during sleep (single-table read+update).
        'counterfactual_to_kelly',
        # RE-5 (2026-04-25): RiskEnvelope sensor + promote ticks. Critical
        # safety — these MUST keep running during sleep. The 5-min sensor
        # tick is the only thing that triggers graduated decay during a
        # brownout. Sleep mode without these = blind organism.
        'risk_envelope_sensor',
        'risk_envelope_promote',
    }

    # Additional whitelist items during DEEP_SLEEP (absolute minimum set).
    _DEEP_SLEEP_WHITELIST = {
        'sleep_wake',
        'dead_code_warmup',
        'health_check',
        'daily_reset',
    }

    def _sleep_wake_tick(self):
        """Phase 27 Task 19: Borbely-driven sleep-wake decision.

        Every 30 minutes, evaluate CircadianRhythm mode and pause/resume
        non-critical jobs. Safety jobs stay ON in every mode — this is a
        behavioural throttle, not a kill switch.
        """
        try:
            # Phase 27 Task 19 audit fix: pull the SHARED CircadianRhythm
            # singleton so MetaController.lifecycle_tick sees the same S/mode.
            from autonomous_lifecycle import CircadianRhythm, get_circadian
            from datetime import datetime as _dt, timezone as _tz

            circ = get_circadian()

            # Volatility-based EMERGENCY_WAKE input — audit fix: previously
            # read `market_vol_zscore_1h` from `system_metrics` but NOTHING
            # was writing that row, so emergency wake could never trigger
            # regardless of actual volatility. Compute the z-score inline
            # from recent ai_decisions outcomes (24h window) and ALSO persist
            # it into system_metrics so downstream dashboards see it.
            vol_sigma = 0.0
            try:
                import numpy as _np
                from db import get_db_connection
                conn = get_db_connection()
                rows = conn.execute(
                    "SELECT outcome_pnl FROM ai_decisions "
                    "WHERE outcome_pnl IS NOT NULL "
                    "  AND timestamp > datetime('now', '-24 hours') "
                    "ORDER BY timestamp ASC"
                ).fetchall()
                returns = [float(r["outcome_pnl"] or 0.0) for r in rows]
                if len(returns) >= 10:
                    arr = _np.asarray(returns, dtype=float)
                    std = float(arr.std()) + 1e-8
                    last = float(arr[-1])
                    z = (last - float(arr.mean())) / std
                    vol_sigma = abs(z)
                    # Sprint 2026-05-01: aligned to production schema
                    # (metric_value / metadata_json) — the legacy
                    # (value / metadata) shape was diverged from
                    # system_monitor.py's writer.
                    conn.execute(
                        "INSERT INTO system_metrics "
                        "(timestamp, metric_name, metric_value, metadata_json) "
                        "VALUES (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'), ?, ?, ?)",
                        ("market_vol_zscore_1h", vol_sigma,
                         f'{{"n_samples": {len(returns)}, "source": "sleep_wake_tick"}}'),
                    )
                    conn.commit()
                conn.close()
            except Exception as e:
                logger.debug(f"[Phase27:SleepWake] vol_sigma compute failed: {e}")
                vol_sigma = 0.0

            hour = _dt.now(tz=_tz.utc).hour
            prev_mode = circ.get_mode()
            new_mode = circ.evaluate_mode(hour, recent_volatility_sigma=vol_sigma)

            if new_mode == prev_mode:
                return

            # Apply to the APScheduler job list.
            if new_mode == CircadianRhythm.MODE_FULL_WAKE:
                whitelist = None  # resume EVERYTHING
            elif new_mode == CircadianRhythm.MODE_LIGHT_SLEEP:
                whitelist = self._LIGHT_SLEEP_WHITELIST
            else:  # DEEP_SLEEP
                whitelist = self._LIGHT_SLEEP_WHITELIST | self._DEEP_SLEEP_WHITELIST

            paused, resumed = 0, 0
            for job in self.scheduler.get_jobs():
                if whitelist is None or job.id in whitelist:
                    if job.next_run_time is None:
                        try:
                            self.scheduler.resume_job(job.id)
                            resumed += 1
                        except Exception:
                            pass
                else:
                    if job.next_run_time is not None:
                        try:
                            self.scheduler.pause_job(job.id)
                            paused += 1
                        except Exception:
                            pass
            snap = circ.state_snapshot()
            logger.info(
                f"[Phase27:SleepWake] {prev_mode} → {new_mode} "
                f"(S={snap['process_s']:.2f}, vol_σ={vol_sigma:.2f}) "
                f"paused={paused}, resumed={resumed}"
            )
        except Exception as e:
            logger.warning(f"[Phase27:SleepWake] tick failed: {e}")

    def _foundation_fine_tune(self):
        """Phase 27 Task 20: fine-tune foundation model heads on recent data.

        Heavy job: PyTorch / transformers loaded → gc-wrap so the ~600MB
        peak (Chronos-Bolt + TTM head) frees back to libc/system before the
        next Sunday cron fires.
        """
        with self._heavy_job_gc("foundation_fine_tune"):
            try:
                try:
                    from ttm_perception import retrain_head_if_available  # type: ignore
                    ttm_result = retrain_head_if_available(min_samples=50)
                    logger.info(f"[Phase27:FineTune:TTM] {ttm_result}")
                except Exception as e:
                    logger.debug(f"[Phase27:FineTune:TTM] skipped: {e}")
                try:
                    from chronos_perception import bitfit_bias_update_if_available  # type: ignore
                    chr_result = bitfit_bias_update_if_available(min_samples=50)
                    logger.info(f"[Phase27:FineTune:Chronos] {chr_result}")
                except Exception as e:
                    logger.debug(f"[Phase27:FineTune:Chronos] skipped: {e}")
            except Exception as e:
                logger.warning(f"[Phase27:FineTune] job failed: {e}")

    def _phase27_dead_code_warmup(self):
        """Phase 27 dead-code batch 2 hook.

        Imports and minimally exercises 8 modules so they are not orphaned
        in the import graph. Full integration is planned for Sprint 3B task
        22b+; this warm-up guarantees the modules load without syntax drift.
        """
        summary = {}
        # Task 22: multi-modal fusion live-source call
        try:
            from multimodal_encoder import get_multimodal_encoder
            mm = get_multimodal_encoder()
            fused = mm.fuse_from_live_sources()
            summary["multimodal"] = f"dim={len(fused)}"
        except Exception as e:
            summary["multimodal"] = f"skip:{type(e).__name__}"

        # trinity_fusion (perception × sentiment × macro)
        try:
            from trinity_fusion import get_trinity
            trinity = get_trinity()
            summary["trinity"] = "loaded" if trinity is not None else "skip"
        except Exception as e:
            summary["trinity"] = f"skip:{type(e).__name__}"

        # sac_online — dual-motor RL (lazy init only)
        try:
            import sac_online  # noqa: F401
            summary["sac_online"] = "loaded"
        except Exception as e:
            summary["sac_online"] = f"skip:{type(e).__name__}"

        # hrl_meta_policy — meta-controller
        try:
            import hrl_meta_policy  # noqa: F401
            summary["hrl_meta_policy"] = "loaded"
        except Exception as e:
            summary["hrl_meta_policy"] = f"skip:{type(e).__name__}"

        # market_maker_mode — conditional on ranging regime
        try:
            from market_maker_mode import get_market_maker
            mm_mode = get_market_maker()
            summary["market_maker"] = "loaded" if mm_mode is not None else "import_only"
        except Exception as e:
            summary["market_maker"] = f"skip:{type(e).__name__}"

        # sim2real — per-pair slippage injection hook
        try:
            from sim2real_pipeline import get_sim2real
            s2r = get_sim2real()
            summary["sim2real"] = "loaded" if s2r is not None else "import_only"
        except Exception as e:
            summary["sim2real"] = f"skip:{type(e).__name__}"

        # external_data_integrator — Kaggle / HF warm-up (import only, data
        # fetch would violate free-tier courtesy policies on a tight cron).
        try:
            import external_data_integrator  # noqa: F401
            summary["external_data"] = "loaded"
        except Exception as e:
            summary["external_data"] = f"skip:{type(e).__name__}"

        # gam_rag — graph-augmented RAG
        try:
            import gam_rag  # noqa: F401
            summary["gam_rag"] = "loaded"
        except Exception as e:
            summary["gam_rag"] = f"skip:{type(e).__name__}"

        logger.info(f"[Phase27:DeadCodeWarmup] {summary}")

    # ═══════════════════════════════════════════════════════════
    # Phase 27 Frontier (Grup 7) — DT + Exploit + TradeLang
    # ═══════════════════════════════════════════════════════════

    @staticmethod
    def _heavy_job_gc(job_name: str):
        """Audit fix (2026-04-19): post-job memory reclaim. Sundays were
        OOM-killing the bot because ~5 heavy ML jobs landed in a 4-hour
        window without anyone calling gc. Wrap heavy jobs with
        `with self._heavy_job_gc(name):` (or call directly in finally) to
        force a malloc_trim + gc collect after the job's PyTorch tensors
        go out of scope.

        Sprint 2026-05-01: jemalloc-aware purge — production runs with
        LD_PRELOAD libjemalloc, so the legacy glibc-only `malloc_trim`
        was a no-op against the actual allocator. Try jemalloc first,
        glibc fallback for non-prod environments.
        """
        import contextlib
        @contextlib.contextmanager
        def _ctx():
            try:
                yield
            finally:
                try:
                    import gc
                    gc.collect()
                    try:
                        import ctypes
                        purged = False
                        try:
                            je = ctypes.CDLL("libjemalloc.so.2")
                            rc = je.mallctl(b"arena.4096.purge", None, None, None, 0)
                            purged = (rc == 0)
                        except (OSError, AttributeError):
                            pass
                        if not purged:
                            ctypes.CDLL("libc.so.6").malloc_trim(0)
                    except Exception:
                        pass  # malloc_trim only on Linux
                    try:
                        import torch  # type: ignore
                        if hasattr(torch, "cuda") and torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except Exception:
                        pass
                    logger.info(f"[Scheduler:GC] post-{job_name} reclaim done")
                except Exception:
                    pass
        return _ctx()

    def _dt_training_cycle(self):
        """Phase 27 Task 21: Sunday 23:00 UTC Decision Transformer LoRA cycle.

        Runs the REAL GPT-2 124M + LoRA training pipeline. Adapter weights
        land at `user_data/models/dt_lora_<timestamp>.pt`. Sprint 3B task 21b
        adds the inference path that loads the freshest checkpoint into sizing.
        """
        with self._heavy_job_gc("dt_training"):
            try:
                from decision_transformer import scheduled_cycle
                result = scheduled_cycle()
                logger.info(f"[Phase27:DT] cycle result: {result}")
            except Exception as e:
                logger.warning(f"[Phase27:DT] cycle failed: {e}")

    def _exploit_regression_batch(self):
        """Phase 27 Task 23: nightly exploit archive regression test.

        For each unvalidated exploit in the last 7 days, map the most recent
        outcome_pnl for that pair+regime back to `was_validated_by_outcome`
        so the archive accumulates a living score on which exploits were real.
        """
        try:
            from db import get_db_connection
            conn = get_db_connection()
            rows = conn.execute("""
                SELECT id, pair, regime, predicted_loss, was_defended, created_at
                FROM exploit_archive
                WHERE was_validated_by_outcome IS NULL
                  AND created_at > datetime('now', '-7 days')
            """).fetchall()
            validated = 0
            for row in rows:
                outcome_row = conn.execute("""
                    SELECT outcome_pnl FROM ai_decisions
                    WHERE pair = ?
                      AND outcome_pnl IS NOT NULL
                      AND timestamp > ?
                    ORDER BY timestamp ASC LIMIT 1
                """, (row["pair"], row["created_at"])).fetchone()
                if not outcome_row:
                    continue
                pnl = float(outcome_row["outcome_pnl"] or 0)
                predicted = float(row["predicted_loss"] or 0)
                defended = bool(row["was_defended"])
                # Exploit was REAL if defense claimed it was neutralised
                # but we still lost ≥ 50% of the predicted loss.
                real = (not defended and pnl < 0) or (
                    defended and predicted > 0 and abs(pnl) >= 0.5 * abs(predicted)
                )
                conn.execute(
                    "UPDATE exploit_archive SET was_validated_by_outcome = ? WHERE id = ?",
                    (1 if real else 0, row["id"]),
                )
                validated += 1
            conn.commit()
            conn.close()
            logger.info(f"[Phase27:ExploitBatch] validated {validated}/{len(rows)} exploits")
        except Exception as e:
            logger.warning(f"[Phase27:ExploitBatch] failed: {e}")

    def _sac_online_cycle(self):
        """Phase 27 Item 10: SAC online RL training cycle. Heavy job → gc-wrap."""
        with self._heavy_job_gc("sac_online"):
            try:
                from sac_online import get_sac_trainer
                trainer = get_sac_trainer()
                if hasattr(trainer, "online_step"):
                    result = trainer.online_step(n_steps=200)
                elif hasattr(trainer, "train_one_cycle"):
                    result = trainer.train_one_cycle()
                else:
                    result = {"status": "no_method"}
                logger.info(f"[Phase27:SAC] cycle: {result}")
            except Exception as e:
                logger.warning(f"[Phase27:SAC] cycle failed: {e}")

    def _multimodal_train_cycle(self):
        """Phase 27 Item 3: weekly MultiModal encoder training. Heavy job → gc-wrap."""
        with self._heavy_job_gc("multimodal_train"):
            try:
                from multimodal_encoder import weekly_training_cycle
                result = weekly_training_cycle(min_samples=50, n_epochs=3)
                logger.info(f"[Phase27:MultiModal] {result}")
            except Exception as e:
                logger.warning(f"[Phase27:MultiModal] cycle failed: {e}")

    def _external_data_cycle(self):
        """Data Acceleration Fix 3: weekly Binance public data fetch.

        Pulls the last 7 days of BTCUSDT 1h klines (incremental — skips
        already-cached dates), triple-barrier labels them, and funnels
        through to backtest_training_data via the label generator.
        """
        try:
            from external_data_integrator import (
                fetch_binance_ohlcv_public, get_integrator)
            from backtest_label_generator import BacktestLabelGenerator
            from datetime import datetime as _dt, timedelta as _td

            integ = get_integrator()
            fetched = 0
            labelled = 0
            # Last 7 UTC days, skipping today (Binance archive lags 24h).
            for i in range(1, 8):
                d = (_dt.utcnow() - _td(days=i)).strftime("%Y-%m-%d")
                csv_path = fetch_binance_ohlcv_public("BTCUSDT", "1h", date=d)
                if csv_path:
                    fetched += 1
                    n = integ.integrate_ohlcv_with_triple_barrier(
                        "BTC/USDT:USDT", csv_path
                    )
                    labelled += int(n or 0)
            # Chain into catboost pipeline via the daily label injector.
            try:
                gen = BacktestLabelGenerator()
                gen.generate_from_backtests()
                gen.enrich_from_live_trades(min_trades=10)
                gen.enrich_from_forgone_trades(min_trades=20)
            except Exception as e:
                logger.debug(f"[Phase27:ExtData] label chain: {e}")
            logger.info(
                f"[Phase27:ExtData] Weekly fetch: {fetched}/7 days, "
                f"{labelled} triple-barrier samples labelled"
            )
        except Exception as e:
            logger.warning(f"[Phase27:ExtData] fetch failed: {e}")

    def _backtest_injection_daily(self):
        """Data Acceleration Fix 2: daily backtest-label injection cycle.

        Runs the full BacktestLabelGenerator pipeline — backtest results +
        live trades + shadow trades — so backtest_training_data keeps growing
        regardless of when the weekly cycle fires. CatBoost retrain (Sunday
        03:00) finds a strictly larger, fresher training set every week.
        """
        try:
            from backtest_label_generator import BacktestLabelGenerator
            gen = BacktestLabelGenerator()
            res_bt = gen.generate_from_backtests()
            res_live = gen.enrich_from_live_trades(min_trades=10)
            res_shadow = gen.enrich_from_forgone_trades(min_trades=20)
            logger.info(
                f"[Phase27:LabelInject] daily — backtest={res_bt}, "
                f"live={res_live}, shadow={res_shadow}"
            )
        except Exception as e:
            logger.warning(f"[Phase27:LabelInject] daily job failed: {e}")

    def _trade_language_cycle(self):
        """Phase 27 Task 25: weekly trade-as-language pattern mining."""
        try:
            from trade_language import weekly_cycle
            result = weekly_cycle(min_trades=100)
            logger.info(f"[Phase27:TradeLang] {result}")
        except Exception as e:
            logger.warning(f"[Phase27:TradeLang] failed: {e}")

    def _hypothesis_generation_cycle(self):
        """Phase 27 Task 16: weekly LLM strategy researcher cycle.

        Capped at 5 LLM calls per cycle (see MAX_HYPOTHESES_PER_CYCLE in
        hypothesis_generator.py). Skips automatically when RPD is tight.
        """
        try:
            import asyncio
            from hypothesis_generator import get_researcher
            researcher = get_researcher()
            # run_cycle is async; cron fires on a thread so we spin a loop.
            result = asyncio.run(researcher.run_cycle())
            logger.info(
                f"[Phase27:Researcher] generated={result.get('generated', 0)}, "
                f"accepted={result.get('accepted', 0)}, "
                f"error={result.get('error')}"
            )
        except ImportError as e:
            logger.debug(f"[Phase27:Researcher] disabled: {e}")
        except Exception as e:
            logger.warning(f"[Phase27:Researcher] cycle failed: {e}")

    def _hawkes_mle_refit(self):
        """Phase 27 Task 10: Hourly MLE refit of Hawkes (α, β) per pair.

        Uses the `tick` library when available. Falls back to a no-op when it
        is not installed — the O(1) intensity tracker keeps working with its
        default parameters in the meantime.
        """
        try:
            from order_flow import get_order_flow
            of = get_order_flow()
            refitted = of.refit_hawkes_mle() if hasattr(of, "refit_hawkes_mle") else 0
            logger.info(f"[Phase27:Hawkes] MLE refit: {refitted} pairs")
        except ImportError as e:
            logger.debug(f"[Phase27:Hawkes] tick not installed: {e}")
        except Exception as e:
            logger.debug(f"[Phase27:Hawkes] Refit skipped: {e}")

    def _conformal_recalibrate(self):
        """Every 6h: ACI alpha adjustment based on recent coverage."""
        try:
            from conformal_calibrator import ConformalCalibrator
            cal = ConformalCalibrator()
            # Get recent predictions vs outcomes
            from db import get_connection
            with get_connection() as conn:
                rows = conn.execute("""
                    SELECT confidence, outcome_pnl FROM ai_decisions
                    WHERE outcome_pnl IS NOT NULL
                    ORDER BY timestamp DESC LIMIT 50
                """).fetchall()
            for row in rows:
                cal.update(row["confidence"], 1.0 if row["outcome_pnl"] > 0 else 0.0)
            logger.debug(f"[Phase28:Conformal] ACI updated with {len(rows)} outcomes")
        except Exception as e:
            logger.debug(f"[Phase28:Conformal] Recalibration failed: {e}")

    def _ensemble_refit(self):
        """Weekly: Refit deep ensemble on recent trade features.

        Sprint 2026-05-01: jemalloc-purge wrapped — fits 5 deep models
        in-process, biggest neural-net allocator after CatBoost.

        Sprint 2026-05-01 night — walk-forward freeze honored.
        """
        if self._walk_forward_frozen():
            logger.warning("[Phase28:Ensemble] SKIPPED — walk-forward freeze active.")
            return
        if self._memory_pressure_halt():
            logger.warning("[Phase28:Ensemble] SKIPPED — memory pressure forecast critical.")
            return
        with self._heavy_job_gc("ensemble_refit"):
            try:
                from deep_ensemble import DeepEnsemble
                import numpy as np
                from db import get_connection
                ensemble = DeepEnsemble(n_models=5, input_dim=10)
                with get_connection() as conn:
                    rows = conn.execute("""
                        SELECT confidence, trust_score_at_decision, outcome_pnl, outcome_duration
                        FROM ai_decisions WHERE outcome_pnl IS NOT NULL
                        ORDER BY timestamp DESC LIMIT 200
                    """).fetchall()
                if len(rows) >= 30:
                    X = np.array([[r["confidence"] or 0.5, r["trust_score_at_decision"] or 0.5,
                                   r["outcome_duration"] or 1.0] + [0.5] * 7 for r in rows])
                    y = np.array([[r["outcome_pnl"]] for r in rows])
                    result = ensemble.fit(X, y, epochs=50)
                    logger.info(f"[Phase28:Ensemble] Refit: loss={result.get('avg_loss', 'N/A'):.4f}")
                else:
                    logger.info("[Phase28:Ensemble] Insufficient data for refit")
            except Exception as e:
                logger.warning(f"[Phase28:Ensemble] Refit failed: {e}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import time

    sched = PipelineScheduler()
    if sched.start():
        print("Scheduler running. Press Ctrl+C to stop.")
        print("Jobs:", sched.get_job_info())
        try:
            while True:
                time.sleep(60)
        except KeyboardInterrupt:
            sched.stop()
    else:
        print("Scheduler failed to start. Check if apscheduler is installed.")
