"""
Phase 17: Lightweight System Health Monitor (Prometheus Alternative)
SQLite-based metrics collection — no extra RAM overhead.
"""

import os
import sys
import json
import time
import sqlite3
import logging
import shutil
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional

sys.path.append(os.path.dirname(__file__))
from ai_config import AI_DB_PATH

logger = logging.getLogger(__name__)


class SystemMonitor:
    """
    Hafif monitoring: metrik kayıt, sağlık kontrolü, dashboard verisi.
    SQLite tablosu: system_metrics (timestamp, metric_name, metric_value, metadata_json)
    """

    def __init__(self, db_path: str = None):
        self.db_path = db_path or AI_DB_PATH
        self._init_table()

    def _init_table(self):
        try:
            conn = sqlite3.connect(self.db_path, timeout=10)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS system_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
                    metric_name TEXT NOT NULL,
                    metric_value REAL NOT NULL,
                    metadata_json TEXT
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_metrics_name_ts
                ON system_metrics(metric_name, timestamp)
            """)
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"[SystemMonitor] Table init failed: {e}")

    def record_metric(self, name: str, value: float, metadata: dict = None):
        """Record a single metric data point."""
        try:
            meta_json = json.dumps(metadata) if metadata else None
            conn = sqlite3.connect(self.db_path, timeout=10)
            conn.execute(
                "INSERT INTO system_metrics (metric_name, metric_value, metadata_json) VALUES (?, ?, ?)",
                (name, value, meta_json)
            )
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"[SystemMonitor] Failed to record metric '{name}': {e}")

    def get_dashboard_data(self, hours: int = 24) -> dict:
        """Summarize metrics from the last N hours for dashboard display."""
        cutoff = (datetime.now(tz=timezone.utc) - timedelta(hours=hours)).strftime("%Y-%m-%dT%H:%M:%SZ")

        try:
            conn = sqlite3.connect(self.db_path, timeout=10)
            conn.row_factory = sqlite3.Row
            c = conn.cursor()

            # RAG latency average
            row = c.execute(
                "SELECT AVG(metric_value) as avg_val FROM system_metrics WHERE metric_name = 'rag_latency_ms' AND timestamp >= ?",
                (cutoff,)
            ).fetchone()
            rag_latency_avg = float(row["avg_val"]) if row and row["avg_val"] else 0.0

            # LLM cost today (sum)
            today_start = datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT00:00:00Z")
            row = c.execute(
                "SELECT SUM(metric_value) as total FROM system_metrics WHERE metric_name = 'llm_cost' AND timestamp >= ?",
                (today_start,)
            ).fetchone()
            llm_cost_today = float(row["total"]) if row and row["total"] else 0.0

            # Cache hit rate (average of recent hits: 1.0 = hit, 0.0 = miss)
            row = c.execute(
                "SELECT AVG(metric_value) as avg_val FROM system_metrics WHERE metric_name = 'cache_hit' AND timestamp >= ?",
                (cutoff,)
            ).fetchone()
            cache_hit_rate = float(row["avg_val"]) if row and row["avg_val"] else 0.0

            # Total decisions
            row = c.execute(
                "SELECT COUNT(*) as cnt FROM system_metrics WHERE metric_name = 'decision_logged' AND timestamp >= ?",
                (cutoff,)
            ).fetchone()
            total_decisions = int(row["cnt"]) if row else 0

            # Error rate
            row_total = c.execute(
                "SELECT COUNT(*) as cnt FROM system_metrics WHERE metric_name IN ('llm_success', 'llm_error') AND timestamp >= ?",
                (cutoff,)
            ).fetchone()
            row_errors = c.execute(
                "SELECT COUNT(*) as cnt FROM system_metrics WHERE metric_name = 'llm_error' AND timestamp >= ?",
                (cutoff,)
            ).fetchone()
            total_calls = int(row_total["cnt"]) if row_total else 0
            error_count = int(row_errors["cnt"]) if row_errors else 0
            error_rate = (error_count / total_calls) if total_calls > 0 else 0.0

            # Retrieval count
            row = c.execute(
                "SELECT COUNT(*) as cnt FROM system_metrics WHERE metric_name = 'retrieval' AND timestamp >= ?",
                (cutoff,)
            ).fetchone()
            retrieval_count = int(row["cnt"]) if row else 0

            # Active pairs (from metadata)
            rows = c.execute(
                "SELECT DISTINCT metadata_json FROM system_metrics WHERE metric_name = 'decision_logged' AND timestamp >= ? AND metadata_json IS NOT NULL",
                (cutoff,)
            ).fetchall()
            active_pairs = set()
            for r in rows:
                try:
                    meta = json.loads(r["metadata_json"])
                    if "pair" in meta:
                        active_pairs.add(meta["pair"])
                except (json.JSONDecodeError, TypeError):
                    pass

            # Uptime: hours since first metric recorded
            row = c.execute(
                "SELECT MIN(timestamp) as first_ts FROM system_metrics"
            ).fetchone()
            uptime_hours = 0.0
            if row and row["first_ts"]:
                try:
                    first_dt = datetime.fromisoformat(row["first_ts"].replace("Z", "+00:00"))
                    uptime_hours = (datetime.now(tz=timezone.utc) - first_dt).total_seconds() / 3600
                except (ValueError, TypeError):
                    pass

            conn.close()

            return {
                "rag_latency_avg_ms": round(rag_latency_avg, 2),
                "llm_cost_today": round(llm_cost_today, 4),
                "cache_hit_rate": round(cache_hit_rate, 4),
                "total_decisions": total_decisions,
                "error_rate": round(error_rate, 4),
                "retrieval_count": retrieval_count,
                "active_pairs": sorted(active_pairs),
                "uptime_hours": round(uptime_hours, 2),
                "last_updated": datetime.now(tz=timezone.utc).isoformat()
            }

        except Exception as e:
            logger.error(f"[SystemMonitor] Dashboard data failed: {e}")
            return {
                "rag_latency_avg_ms": 0.0, "llm_cost_today": 0.0,
                "cache_hit_rate": 0.0, "total_decisions": 0,
                "error_rate": 0.0, "retrieval_count": 0,
                "active_pairs": [], "uptime_hours": 0.0,
                "last_updated": datetime.now(tz=timezone.utc).isoformat()
            }

    def check_health(self) -> dict:
        """System health check with component status."""
        checks = {}
        alerts = []

        # 1. Database connectivity
        try:
            conn = sqlite3.connect(self.db_path, timeout=5)
            conn.execute("SELECT 1")
            conn.close()
            checks["database"] = True
        except Exception:
            checks["database"] = False
            alerts.append("Database connection failed")

        # 2. LLM Router — any successful call in last 5 minutes?
        try:
            five_min_ago = (datetime.now(tz=timezone.utc) - timedelta(minutes=5)).strftime("%Y-%m-%dT%H:%M:%SZ")
            conn = sqlite3.connect(self.db_path, timeout=5)
            row = conn.execute(
                "SELECT COUNT(*) as cnt FROM llm_calls WHERE timestamp >= ?",
                (five_min_ago,)
            ).fetchone()
            checks["llm_router"] = (row[0] > 0) if row else False
            conn.close()
        except Exception:
            checks["llm_router"] = False

        # 3. ChromaDB — can we import and connect?
        try:
            import chromadb
            from ai_config import CHROMA_PERSIST_DIR
            client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
            client.heartbeat()
            checks["chromadb"] = True
        except Exception:
            checks["chromadb"] = False
            alerts.append("ChromaDB unavailable")

        # 4. Scheduler — last job run within 15 minutes?
        try:
            fifteen_min_ago = (datetime.now(tz=timezone.utc) - timedelta(minutes=15)).strftime("%Y-%m-%dT%H:%M:%SZ")
            conn = sqlite3.connect(self.db_path, timeout=5)
            row = conn.execute(
                "SELECT COUNT(*) as cnt FROM system_metrics WHERE metric_name = 'scheduler_job' AND timestamp >= ?",
                (fifteen_min_ago,)
            ).fetchone()
            checks["scheduler"] = (row[0] > 0) if row else False
            conn.close()
        except Exception:
            checks["scheduler"] = False

        # 5. Disk usage
        try:
            usage = shutil.disk_usage(os.path.dirname(self.db_path) or "/")
            checks["disk_usage_pct"] = round((usage.used / usage.total) * 100, 1)
            if checks["disk_usage_pct"] > 90:
                alerts.append(f"Disk usage critical: {checks['disk_usage_pct']}%")
        except Exception:
            checks["disk_usage_pct"] = 0.0

        # 6. Memory usage
        try:
            import psutil
            mem = psutil.virtual_memory()
            checks["memory_usage_pct"] = round(mem.percent, 1)
            if mem.percent > 90:
                alerts.append(f"Memory usage critical: {mem.percent}%")
        except ImportError:
            # psutil not available — estimate from /proc if on Linux
            checks["memory_usage_pct"] = 0.0
        except Exception:
            checks["memory_usage_pct"] = 0.0

        # Determine overall status
        critical_checks = [checks.get("database", False), checks.get("chromadb", False)]
        if not all(critical_checks):
            status = "critical"
        elif alerts:
            status = "degraded"
        else:
            status = "healthy"

        return {
            "status": status,
            "checks": checks,
            "alerts": alerts
        }

    def get_hourly_summary(self, hours: int = 24) -> List[dict]:
        """Hourly metric summaries for chart data."""
        try:
            conn = sqlite3.connect(self.db_path, timeout=10)
            conn.row_factory = sqlite3.Row
            cutoff = (datetime.now(tz=timezone.utc) - timedelta(hours=hours)).strftime("%Y-%m-%dT%H:%M:%SZ")

            rows = conn.execute("""
                SELECT
                    strftime('%Y-%m-%dT%H:00:00Z', timestamp) as hour,
                    metric_name,
                    AVG(metric_value) as avg_value,
                    COUNT(*) as count
                FROM system_metrics
                WHERE timestamp >= ?
                GROUP BY hour, metric_name
                ORDER BY hour ASC
            """, (cutoff,)).fetchall()

            conn.close()

            # Group by hour
            hourly = {}
            for r in rows:
                h = r["hour"]
                if h not in hourly:
                    hourly[h] = {"hour": h, "metrics": {}}
                hourly[h]["metrics"][r["metric_name"]] = {
                    "avg": round(float(r["avg_value"]), 4),
                    "count": int(r["count"])
                }

            return list(hourly.values())

        except Exception as e:
            logger.error(f"[SystemMonitor] Hourly summary failed: {e}")
            return []

    def cleanup_old_metrics(self, max_age_days: int = 30):
        """Remove metrics older than max_age_days."""
        try:
            conn = sqlite3.connect(self.db_path, timeout=10)
            conn.execute(
                "DELETE FROM system_metrics WHERE timestamp < datetime('now', ?)",
                (f"-{max_age_days} days",)
            )
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"[SystemMonitor] Cleanup failed: {e}")
