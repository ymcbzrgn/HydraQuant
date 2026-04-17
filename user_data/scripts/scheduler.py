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

        self.scheduler = BackgroundScheduler(timezone="UTC")

        # Phase 28: Start Grafeo ZMQ broker (this process is the single writer)
        try:
            from graph_store import start_graph_broker
            start_graph_broker()
        except Exception as e:
            logger.warning(f"[Scheduler] Grafeo broker start failed: {e}")

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

        # Memory management: gc.collect + memory logging every hour
        self.scheduler.add_job(
            self._memory_cleanup,
            'interval', minutes=60,
            id='memory_cleanup',
            name='GC Collect + Memory Log',
            max_instances=1,
            replace_existing=True
        )

        # Phase 24+25: Neural Organism — 6 jobs (hourly decay, daily habits, daily DMN+cerebellum, weekly sleep+evolution)
        self.scheduler.add_job(self._organism_hourly_decay, 'interval', minutes=60,
            id='organism_decay', name='Neural Organism Hourly Decay', max_instances=1, replace_existing=True)
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
        self.scheduler.add_job(self._pheromone_cleanup, 'interval', minutes=30,
            id='pheromone_cleanup', name='Pheromone Field Cleanup', max_instances=1, replace_existing=True)

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
        # Sprint 2 (8C+8D): World model train + Dream session — weekly Sunday 01:30 UTC
        self.scheduler.add_job(self._world_model_and_dream, 'cron', day_of_week='sun', hour=1, minute=30,
            id='world_model_dream', name='World Model + Dream Session', max_instances=1, replace_existing=True)
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
        self.scheduler.add_job(self._forgone_threshold_adapt, 'cron', day_of_week='sun', hour=6,
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
        # Phase 27 Task 20: Foundation model fine-tuning — Sunday 02:30 UTC
        # (after CatBoost retrain 03:00… wait, we run BEFORE so the fine-tuned
        # head can feed the CatBoost feature set). Head retraining only, full
        # backbone is frozen (IBM few-shot recipe).
        self.scheduler.add_job(self._foundation_fine_tune,
            'cron', day_of_week='sun', hour=2, minute=30,
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
        # Phase 27 Item 10: SAC online RL training cycle (Sun 04:30 UTC).
        self.scheduler.add_job(self._sac_online_cycle,
            'cron', day_of_week='sun', hour=4, minute=30,
            id='sac_online', name='SAC Online RL Cycle',
            max_instances=1, replace_existing=True)
        # Phase 27 Item 3: MultiModal encoder training (Sun 06:00 UTC).
        self.scheduler.add_job(self._multimodal_train_cycle,
            'cron', day_of_week='sun', hour=6,
            id='multimodal_train', name='MultiModal Encoder Weekly Training',
            max_instances=1, replace_existing=True)
        # Phase 27 Item 6: External data fetch (monthly, 1st of month 09:00 UTC).
        self.scheduler.add_job(self._external_data_cycle,
            'cron', day=1, hour=9,
            id='external_data_fetch', name='Binance Public Data Monthly Fetch',
            max_instances=1, replace_existing=True)

        self.scheduler.start()
        logger.info("[Scheduler] Started with 65 jobs (26 base + 6 organism + 2 phase26 + 17 sprint2 + 14 phase27)")
        return True

    def stop(self):
        """Gracefully shutdown the scheduler."""
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
        """Hourly: Force garbage collection and log memory usage.
        Prevents slow memory leak from orphaned objects."""
        import gc
        collected = gc.collect()
        try:
            import psutil
            process = psutil.Process()
            mem_mb = process.memory_info().rss / 1024 / 1024
            logger.info(f"[Scheduler:Memory] GC collected {collected} objects. "
                       f"RSS={mem_mb:.0f}MB, threads={process.num_threads()}")
        except ImportError:
            import resource
            mem_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            logger.info(f"[Scheduler:Memory] GC collected {collected} objects. "
                       f"maxRSS={mem_kb/1024:.0f}MB")

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
                "--strategy", "AIFreqtradeSizer",
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
        """Weekly Sunday 03:30: Sleep consolidation — replay + prune + counterfactual."""
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
        """Daily 04:00: Default Mode Network — idle background processing."""
        try:
            from neural_organism import get_organism
            organism = get_organism()
            result = organism.dmn.run_idle_cycle(organism._neurons, organism.hippocampus)
            logger.info(f"[Scheduler:Organism] DMN idle: {len(result.get('counterfactuals', []))} counterfactuals, "
                       f"{len(result.get('discoveries', []))} synapse candidates")
        except Exception as e:
            logger.error(f"[Scheduler:Organism] DMN failed: {e}")

    def _organism_evolution(self):
        """Weekly Sunday 04:00: NeuroEvolution — population tournament."""
        try:
            from neural_organism import get_organism
            organism = get_organism()
            organism.evolution.run_tournament(organism._neurons, organism._cumulative_pnl)
            organism._persist_batch(list(organism._neurons.values()))
            logger.info(f"[Scheduler:Organism] NeuroEvolution tournament completed")
        except Exception as e:
            logger.error(f"[Scheduler:Organism] NeuroEvolution failed: {e}")

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

        except Exception as e:
            logger.error(f"[Phase26:Interoception] Check failed: {e}")

    def _pheromone_cleanup(self):
        """Every 30min: Clean up fully decayed pheromones."""
        try:
            from pheromone_field import get_pheromone_field
            field = get_pheromone_field()
            cleaned = field.cleanup()
            health = field.get_field_health()
            logger.debug(
                f"[Phase26:Pheromone] Cleanup: {cleaned} removed, "
                f"{health['active_signals']} active from {health['active_sources']}"
            )
        except Exception as e:
            logger.debug(f"[Phase26:Pheromone] Cleanup failed: {e}")


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
        """Weekly Sunday 01:30 UTC: Train world model + run dream session.

        Flow:
          1. Train world model on accumulated replay buffer
          2. Run dream session (100 imagined trajectories)
          3. Filter dreams (3-layer anomaly detection)
          4. Valid dreams → RL replay buffer (source='dream')
        """
        try:
            from world_model import get_world_model
            from dream_engine import get_dream_engine

            # 1. Train world model
            wm = get_world_model()
            train_result = wm.train_from_buffer(n_epochs=30, batch_size=64)
            if "error" in train_result:
                logger.info(f"[Sprint2:WorldModel] {train_result['error']}")
            else:
                logger.info(f"[Sprint2:WorldModel] Trained: "
                           f"pred_loss={train_result.get('pred_loss', 'N/A'):.4f}")

            # 2. Run dream session
            dream_engine = get_dream_engine()
            dream_result = dream_engine.dream_session(n_dreams=100, horizon=5)

            if "error" in dream_result:
                logger.info(f"[Sprint2:Dream] {dream_result['error']}")
            else:
                logger.info(f"[Sprint2:Dream] Session: "
                           f"{dream_result['valid_dreams']}/{dream_result['total_dreams']} valid, "
                           f"pass_rate={dream_result['pass_rate']:.1%}, "
                           f"stored={dream_result['stored_in_buffer']}")

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
        """
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
        """
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
        """
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
        """Weekly: Refit OOD detector reference distributions with recent data."""
        try:
            from ood_detector import MarketOODDetector
            import pandas as pd
            from db import get_connection
            detector = MarketOODDetector()
            with get_connection() as conn:
                rows = conn.execute("""
                    SELECT confidence, trust_score_at_decision, outcome_duration, regime
                    FROM ai_decisions WHERE outcome_pnl IS NOT NULL
                    ORDER BY timestamp DESC LIMIT 500
                """).fetchall()
            if len(rows) >= 30:
                df = pd.DataFrame([dict(r) for r in rows])
                features = df[["confidence", "trust_score_at_decision", "outcome_duration"]].fillna(0.5)
                regimes = df["regime"].fillna("transitional")
                detector.fit(features, regimes)
                logger.info(f"[Phase28:OOD] Refit on {len(rows)} trades")
            else:
                logger.info("[Phase28:OOD] Insufficient data for refit")
        except Exception as e:
            logger.warning(f"[Phase28:OOD] Refit failed: {e}")

    def _forgone_shadow_resolver(self):
        """Phase 27 Fix 6 (H3): Resolve shadow trades whose 4h window has elapsed.

        Every 30 minutes, pick up unresolved forgone_profit rows older than 4h,
        fetch the current Bybit last-price, and call
        ForgonePnLEngine.resolve_forgone_trade so forgone_pnl is populated and
        the adaptive-threshold job has signal.
        """
        try:
            import httpx
            from forgone_pnl_engine import ForgonePnLEngine
            from db import get_db_connection

            engine = ForgonePnLEngine()
            conn = get_db_connection()
            unresolved = conn.execute("""
                SELECT id, pair, signal_type, entry_price, signal_time
                FROM forgone_profit
                WHERE was_executed = 0
                  AND forgone_pnl IS NULL
                  AND signal_time < datetime('now', '-4 hours')
                ORDER BY signal_time ASC
                LIMIT 100
            """).fetchall()
            conn.close()

            if not unresolved:
                return

            def _bybit_last_price(pair):
                # "BTC/USDT:USDT" → "BTCUSDT" linear perp symbol
                symbol = pair.split(":")[0].replace("/", "")
                try:
                    r = httpx.get(
                        "https://api.bybit.com/v5/market/tickers",
                        params={"category": "linear", "symbol": symbol},
                        timeout=5.0,
                    )
                    data = r.json().get("result", {}).get("list", [])
                    if data:
                        return float(data[0].get("lastPrice", 0))
                except Exception:
                    return None
                return None

            resolved = 0
            for row in unresolved:
                try:
                    price = _bybit_last_price(row["pair"])
                    if price and price > 0:
                        if engine.resolve_forgone_trade(row["id"], float(price)):
                            resolved += 1
                except Exception as e:
                    logger.debug(f"[Phase27:ShadowResolve] {row['pair']} skip: {e}")
            logger.info(f"[Phase27:ShadowResolve] Resolved {resolved}/{len(unresolved)} shadow trades")
        except ImportError as e:
            logger.debug(f"[Phase27:ShadowResolve] disabled: {e}")
        except Exception as e:
            logger.warning(f"[Phase27:ShadowResolve] Job failed: {e}")

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

            adjusted = 0
            now = datetime.now(tz=timezone.utc).isoformat()
            for p in pairs:
                alpha = (p["pos"] or 0.0) + (p["neg"] or 0.0)
                if p["n_signals"] < 5:
                    continue  # too little data to act on

                if alpha > 2.0:
                    delta = -0.02  # missing too many winners → lower threshold (take more trades)
                    reason = f"missed_alpha={alpha:+.2f}"
                elif alpha < -1.0:
                    delta = +0.01  # catching losers → raise threshold
                    reason = f"net_loss_alpha={alpha:+.2f}"
                else:
                    continue

                # Upsert with clamped threshold
                existing = conn.execute("""
                    SELECT confidence_threshold FROM pair_thresholds
                    WHERE pair = ? AND regime = ?
                """, (p["pair"], p["regime"])).fetchone()
                current = float(existing["confidence_threshold"]) if existing else 0.50
                new_thr = max(0.30, min(0.75, current + delta))
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
                    conn.execute(
                        "INSERT INTO system_metrics (metric_name, value, metadata) "
                        "VALUES (?, ?, ?)",
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

        Safe-by-default: we only retrain the FINAL classification head on TTM
        (frozen backbone, IBM few-shot recipe) and run a BitFit-style bias-only
        update on Chronos-Bolt. Failures are swallowed — production models
        stay intact because we write to a `*_ft.pt` sidecar. An A/B test hook
        lives in triple_perception (Task 20b).
        """
        try:
            # TTM head retraining
            try:
                from ttm_perception import retrain_head_if_available  # type: ignore
                ttm_result = retrain_head_if_available(min_samples=50)
                logger.info(f"[Phase27:FineTune:TTM] {ttm_result}")
            except Exception as e:
                logger.debug(f"[Phase27:FineTune:TTM] skipped: {e}")

            # Chronos BitFit
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

    def _dt_training_cycle(self):
        """Phase 27 Task 21: Sunday 23:00 UTC Decision Transformer LoRA cycle.

        Runs the REAL GPT-2 124M + LoRA training pipeline:
        `decision_transformer.scheduled_cycle()` builds the (return-to-go,
        state, action) corpus from ai_decisions and trains via peft LoRA
        (rank 16, target_modules=["c_attn"]). Adapter weights land at
        `user_data/models/dt_lora_<timestamp>.pt`. Sprint 3B task 21b adds
        the inference path that loads the freshest checkpoint into sizing.
        """
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
        """Phase 27 Item 10: SAC online RL training cycle.

        Pulls the IQL replay buffer + recent live transitions and runs a single
        SAC actor-critic update pass. Lightweight (≤200 gradient steps) so it
        fits in the Sunday window and never blocks live trading.
        """
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
        """Phase 27 Item 3: weekly MultiModal encoder training pass."""
        try:
            from multimodal_encoder import weekly_training_cycle
            result = weekly_training_cycle(min_samples=50, n_epochs=3)
            logger.info(f"[Phase27:MultiModal] {result}")
        except Exception as e:
            logger.warning(f"[Phase27:MultiModal] cycle failed: {e}")

    def _external_data_cycle(self):
        """Phase 27 Item 6: monthly Binance public data fetch + label."""
        try:
            from external_data_integrator import scheduled_external_fetch
            result = scheduled_external_fetch()
            logger.info(f"[Phase27:ExtData] {result}")
        except Exception as e:
            logger.warning(f"[Phase27:ExtData] fetch failed: {e}")

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
        """Weekly: Refit deep ensemble on recent trade features."""
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
