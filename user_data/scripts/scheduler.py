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

    def _get_pipeline(self):
        """Lazy-load DataPipeline to avoid circular imports."""
        if self._pipeline is None:
            from data_pipeline import DataPipeline
            self._pipeline = DataPipeline()
        return self._pipeline

    def start(self):
        """Initialize and start the background scheduler."""
        try:
            from apscheduler.schedulers.background import BackgroundScheduler
        except ImportError:
            logger.error("[Scheduler] apscheduler not installed. Run: pip install apscheduler")
            logger.info("[Scheduler] Falling back to manual pipeline mode.")
            return False

        self.scheduler = BackgroundScheduler(timezone="UTC")

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

        # Sunday 23:55 UTC: Send weekly Telegram summary
        self.scheduler.add_job(
            self._send_weekly_summary,
            'cron', day_of_week='sun', hour=23, minute=55,
            id='weekly_summary',
            name='Weekly Telegram Summary',
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

        self.scheduler.start()
        logger.info("[Scheduler] Started with 23 jobs")
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
            from system_monitor import SystemMonitor
            monitor = SystemMonitor()
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
            from bidirectional_rag import BidirectionalRAG
            bidi_rag = BidirectionalRAG()
            lessons = bidi_rag.get_unembedded_lessons()
            if not lessons:
                return

            # Push to VectorDB using same retriever
            from hybrid_retriever import HybridRetriever
            retriever = HybridRetriever(collection_name="crypto_news")
            
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
            bidi_rag.mark_lessons_embedded([l['id'] for l in lessons])
            logger.info(f"[Scheduler:Job] Successfully integrated {len(lessons)} Bidirectional lessons.")
            
        except Exception as e:
            logger.error(f"[Scheduler:Job] Bidirectional embedding failed: {e}")

    def _refit_calibrator(self):
        """Job: Re-fit Platt scaling on latest trade outcomes for confidence calibration."""
        logger.info("[Scheduler:Job] Re-fitting confidence calibrator...")
        try:
            from confidence_calibrator import ConfidenceCalibrator
            calibrator = ConfidenceCalibrator()
            calibrator.fit_platt_scaling()
            report = calibrator.report()
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
            from forgone_pnl_engine import ForgonePnLEngine
            engine = ForgonePnLEngine()
            summary = engine.weekly_summary()

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
                    notifier = AITelegramNotifier()
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
            conn = sqlite3.connect(AI_DB_PATH, timeout=10)
            conn.row_factory = sqlite3.Row

            triggered = False
            trigger_reason = ""

            # Check Fear & Greed for extreme values
            fng_row = conn.execute(
                "SELECT value FROM fear_and_greed ORDER BY timestamp DESC LIMIT 1"
            ).fetchone()
            if fng_row:
                fng = int(fng_row["value"])
                if fng < 15 or fng > 85:
                    triggered = True
                    trigger_reason = f"F&G extreme: {fng}"

            # Check for extreme funding rates (any pair)
            if not triggered:
                extreme_funding = conn.execute("""
                    SELECT pair, funding_rate FROM derivatives_data
                    WHERE ABS(funding_rate) > 0.001
                    AND timestamp > datetime('now', '-15 minutes')
                    ORDER BY ABS(funding_rate) DESC LIMIT 1
                """).fetchone()
                if extreme_funding:
                    triggered = True
                    trigger_reason = f"Funding spike: {extreme_funding['pair']} {float(extreme_funding['funding_rate'])*100:.3f}%"

            conn.close()

            if triggered:
                logger.warning(f"[Phase20:EventTrigger] {trigger_reason} → forcing Evidence Engine re-analysis")
                try:
                    from evidence_engine import EvidenceEngine
                    engine = EvidenceEngine()
                    # Re-analyze top pairs from opportunity_scores
                    conn2 = sqlite3.connect(AI_DB_PATH, timeout=10)
                    conn2.row_factory = sqlite3.Row
                    pairs = conn2.execute("""
                        SELECT pair FROM opportunity_scores
                        WHERE id IN (SELECT MAX(id) FROM opportunity_scores GROUP BY pair)
                        ORDER BY composite_score DESC LIMIT 10
                    """).fetchall()
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
                        notifier = AITelegramNotifier()
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
            
            cost_tracker = LLMCostTracker()
            cost_summary = cost_tracker.get_daily_summary()
            stats["api_cost_today"] = sum(m.get("cost_usd", 0) for m in cost_summary.get("models", {}).values())

            autonomy = AutonomyManager()
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
            
            forgone_engine = ForgonePnLEngine()
            f_summary = forgone_engine.weekly_summary()
            stats["forgone_pnl"] = f_summary.get("forgone_trades", {}).get("total_pnl_pct", 0.0)

            # $100 Hypothetical Portfolio
            stats["hypothetical"] = forgone_engine.get_hypothetical_balance()

            notifier = AITelegramNotifier()
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
            
            forgone_engine = ForgonePnLEngine()
            f_summary = forgone_engine.weekly_summary()
            stats["forgone_pnl_total"] = f_summary.get("forgone_trades", {}).get("total_pnl_pct", 0.0)

            # $100 Hypothetical Portfolio
            stats["hypothetical"] = forgone_engine.get_hypothetical_balance()

            notifier = AITelegramNotifier()
            notifier.send_weekly_summary(stats)
            
        except Exception as e:
            logger.error(f"[Scheduler:Job] Failed to send weekly summary: {e}")

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
                logger.info(f"[Scheduler:AutoBacktest] Bootstrap loaded {count} trades into AI pipeline")
            except Exception as e:
                logger.error(f"[Scheduler:AutoBacktest] Bootstrap failed: {e}")

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
