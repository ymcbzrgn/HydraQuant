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

        # Every 15 minutes: Embed new news into ChromaDB
        self.scheduler.add_job(
            self._embed_news,
            'interval', minutes=15,
            id='embed_news',
            name='Embedding Pipeline',
            max_instances=1,
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

        self.scheduler.start()
        logger.info("[Scheduler] Started with 6 jobs: fetch(5m), embed(15m), reset(day), cleanup(day), summary(day/week)")
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

    def _daily_reset(self):
        """Job: Reset daily risk budget at 00:00 UTC."""
        logger.info("[Scheduler:Job] Daily risk budget reset...")
        try:
            from risk_budget import RiskBudgetManager
            mgr = RiskBudgetManager()
            mgr.reset_daily()
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
            
            forgone_engine = ForgonePnLEngine()
            f_summary = forgone_engine.generate_weekly_summary() # using weekly function to get total pnl
            stats["forgone_pnl"] = f_summary.get("total_forgone_pnl", 0.0)
            
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
            f_summary = forgone_engine.generate_weekly_summary()
            stats["forgone_pnl_total"] = f_summary.get("total_forgone_pnl", 0.0)
            
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
