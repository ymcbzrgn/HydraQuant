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
import sqlite3
import logging
from datetime import datetime, timezone

sys.path.append(os.path.dirname(__file__))

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

        # Daily 00:00 UTC: Reset risk budget
        self.scheduler.add_job(
            self._daily_reset,
            'cron', hour=0, minute=0,
            id='daily_reset',
            name='Daily Risk Budget Reset',
            replace_existing=True
        )

        # Daily 04:00 UTC: Cleanup old news
        self.scheduler.add_job(
            self._cleanup_old_news,
            'cron', hour=4, minute=0,
            id='cleanup',
            name='Old News Cleanup',
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

        self.scheduler.start()
        logger.info("[Scheduler] Started with 11 jobs: fetch(5m), cache_cleanup(5m), health(5m), embed(15m), flush(15m), reset(day), cleanup(day), daily_summary(23:55), weekly_summary(sun), prune_magma(sun), embed_bidi(day)")
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

    def _cleanup_old_news(self, max_age_days: int = 180):
        """Job: Remove news older than max_age_days from SQLite + ChromaDB."""
        logger.info(f"[Scheduler:Job] Cleaning up news older than {max_age_days} days...")
        try:
            from db import get_db_connection
            conn = get_db_connection()
            c = conn.cursor()

            # Count before
            c.execute("SELECT COUNT(*) FROM market_news")
            before = c.fetchone()[0]

            # Delete old articles
            c.execute(
                "DELETE FROM market_news WHERE published_at < datetime('now', ?)",
                (f"-{max_age_days} days",)
            )
            conn.commit()

            # Count after
            c.execute("SELECT COUNT(*) FROM market_news")
            after = c.fetchone()[0]
            conn.close()

            deleted = before - after
            if deleted > 0:
                logger.info(f"[Scheduler:Job] Cleaned up {deleted} old news articles.")
            else:
                logger.info("[Scheduler:Job] No old articles to clean up.")

        except Exception as e:
            logger.error(f"[Scheduler:Job] Cleanup failed: {e}")

    def _cleanup_semantic_cache(self):
        """Job: Cleanup expired entries in semantic_cache."""
        logger.info("[Scheduler:Job] Cleaning up expired semantic cache...")
        try:
            from semantic_cache import SemanticCache
            cache = SemanticCache()
            cache.cleanup_expired()
        except Exception as e:
            logger.error(f"[Scheduler:Job] Semantic cache cleanup failed: {e}")

    def _flush_streaming_rag(self):
        """Job: Flush expired hot documents from StreamingRAG into Chroma"""
        logger.info("[Scheduler:Job] Flushing StreamingRAG hot buffer into cold storage...")
        try:
            from streaming_rag import StreamingRAG
            s_rag = StreamingRAG()
            s_rag.flush_to_cold()
        except Exception as e:
            logger.error(f"[Scheduler:Job] StreamingRAG flush failed: {e}")

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

    def _prune_magma_memory(self):
        """Job: Clean up old/weak linkages inside MAGMA memory tables."""
        logger.info("[Scheduler:Job] Pruning MAGMAMemory edges...")
        try:
            from magma_memory import MAGMAMemory
            magma = MAGMAMemory()
            deleted = magma.prune(min_weight=0.5, max_age_days=180)
            logger.info(f"[Scheduler:Job] Removed {deleted} MAGMA connections.")
        except Exception as e:
            logger.error(f"[Scheduler:Job] MAGMAMemory pruning failed: {e}")

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
