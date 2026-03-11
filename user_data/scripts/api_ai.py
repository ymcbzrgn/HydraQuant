"""
AI Dashboard API — Bağımsız FastAPI endpoint.
Freqtrade ile birlikte veya ayrı çalışabilir.
Çalıştırma: uvicorn api_ai:app --host 0.0.0.0 --port 8890
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import sqlite3
import os
import sys
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone, timedelta

# Ensure local imports work dynamically
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ai_config import AI_DB_PATH
from autonomy_manager import AutonomyManager, PROMOTION_CRITERIA
from risk_budget import RiskBudgetManager
from llm_cost_tracker import LLMCostTracker
from forgone_pnl_engine import ForgonePnLEngine
from memo_rag import MemoRAG
from bidirectional_rag import BidirectionalRAG
from system_monitor import SystemMonitor

app = FastAPI(title="Freqtrade AI API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_db_conn():
    conn = sqlite3.connect(AI_DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    return conn

@app.get("/api/ai/status")
def ai_status():
    """Genel AI sistem durumu."""
    autonomy = AutonomyManager(db_path=AI_DB_PATH)
    cost_tracker = LLMCostTracker(db_path=AI_DB_PATH)
    daily_cost = cost_tracker.get_daily_summary().get("total_cost", 0.0)

    return {
        "status": "online",
        "autonomy_level": autonomy.current_level,
        "active_model": "gemini-2.5-flash",
        "daily_cost": daily_cost,
        "cache_hit_rate": 0.0, # Placeholder until metrics are built out
        "uptime": "100%"
    }
    
@app.get("/api/ai/lessons")
def get_ai_lessons(limit: int = 50):
    """Returns Bidirectional RAG trade evaluation lessons."""
    try:
        with get_db_conn() as conn:
            rows = conn.execute(
                "SELECT id, pair, signal, outcome_pnl, lesson_text, is_embedded, timestamp "
                "FROM ai_lessons ORDER BY timestamp DESC LIMIT ?", (limit,)
            ).fetchall()
            return [dict(r) for r in rows]
    except sqlite3.OperationalError:
        return []

@app.get("/api/ai/memorag")
def get_memorag_global():
    """Returns the current globally compressed MemoRAG corpus."""
    try:
        memorag = MemoRAG(db_path=AI_DB_PATH)
        global_summary = memorag.get_global_memory()
        return {
            "global_memory": global_summary,
            "status": "active" if len(global_summary) > 50 else "building"
        }
    except Exception as e:
        return {"global_memory": f"Error loading memory: {e}", "status": "error"}

@app.get("/api/ai/sentiment/{pair:path}")
def get_sentiment(pair: str):
    """Belirli pair için sentiment verisi."""
    with get_db_conn() as conn:
        try:
            row = conn.execute(
                "SELECT sentiment_1h, sentiment_4h, sentiment_24h, news_count_24h, timestamp "
                "FROM coin_sentiment_rolling WHERE coin = ? ORDER BY timestamp DESC LIMIT 1",
                (pair.split("/")[0],)
            ).fetchone()

            if row:
                return {
                    "pair": pair,
                    "sentiment_1h": row["sentiment_1h"],
                    "sentiment_4h": row["sentiment_4h"],
                    "sentiment_24h": row["sentiment_24h"],
                    "fear_greed": 50,
                    "source_count": row["news_count_24h"],
                    "last_update": row["timestamp"]
                }
        except sqlite3.OperationalError:
            pass
            
    return {
        "pair": pair,
        "sentiment_1h": 0.0,
        "sentiment_4h": 0.0,
        "sentiment_24h": 0.0,
        "fear_greed": 50,
        "source_count": 0,
        "last_update": datetime.now(tz=timezone.utc).isoformat()
    }

@app.get("/api/ai/signals")
def get_signals(limit: int = 20):
    """Son AI sinyalleri listesi."""
    with get_db_conn() as conn:
        try:
            # First attempt with outcome_pnl if available
            rows = conn.execute(
                "SELECT pair, signal_type, confidence, reasoning_summary, timestamp, outcome_pnl FROM ai_decisions "
                "ORDER BY timestamp DESC LIMIT ?", (limit,)
            ).fetchall()
        except sqlite3.OperationalError:
            try:
                # Basic fallback
                rows = conn.execute(
                    "SELECT pair, signal_type, confidence, reasoning_summary, timestamp FROM ai_decisions "
                    "ORDER BY timestamp DESC LIMIT ?", (limit,)
                ).fetchall()
            except sqlite3.OperationalError:
                return []
            
    res = []
    for r in rows:
        d = dict(r)
        res.append({
            "pair": d.get("pair", ""),
            "signal": d.get("signal_type", "NEUTRAL"),
            "confidence": float(d.get("confidence", 0.0)),
            "reasoning": d.get("reasoning_summary", ""),
            "timestamp": d.get("timestamp", ""),
            "outcome": str(round(d.get("outcome_pnl", 0.0), 2)) + "%" if d.get("outcome_pnl") is not None else "Pending"
        })
    return res

@app.get("/api/ai/cost")
def get_cost_summary():
    """LLM maliyet özeti."""
    cost_tracker = LLMCostTracker(db_path=AI_DB_PATH)
    summary = cost_tracker.get_daily_summary()
    return {
        "today_cost": summary.get("total_cost", 0.0),
        "models": summary.get("calls_by_model", {}),
        "budget_remaining": max(0.0, 1.0 - summary.get("total_cost", 0.0))
    }

@app.get("/api/ai/autonomy")
def get_autonomy():
    """Autonomy level detayları."""
    autonomy = AutonomyManager(db_path=AI_DB_PATH)
    return {
        "current_level": autonomy.current_level,
        "kelly_fraction": autonomy.get_kelly_fraction(),
        "criteria": PROMOTION_CRITERIA.get(autonomy.current_level, {}),
        "history": [] # Historic events placeholder
    }

@app.get("/api/ai/risk")
def get_risk():
    """Risk budget durumu."""
    risk_manager = RiskBudgetManager(db_path=AI_DB_PATH)
    daily_budget = float(risk_manager.daily_budget)
    consumed = float(risk_manager._consumed)
    utilization_pct = min(100.0, (consumed / daily_budget) * 100) if daily_budget > 0 else 0.0
    
    return {
        "daily_budget": daily_budget,
        "consumed": consumed,
        "utilization_pct": utilization_pct,
        "active_positions": 0
    }

@app.get("/api/ai/forgone")
def get_forgone_pnl():
    """Forgone P&L tracker."""
    engine = ForgonePnLEngine(db_path=AI_DB_PATH)
    stats = engine.weekly_summary()
    return {
        "total_forgone": stats.get('total_forgone_pnl', 0.0),
        "weekly_summary": stats,
        "recent_signals": stats.get('opportunities_tracked', 0)
    }

@app.get("/api/ai/health")
def get_health():
    """System health check."""
    monitor = SystemMonitor(db_path=AI_DB_PATH)
    return monitor.check_health()

@app.get("/api/ai/metrics")
def get_metrics(hours: int = 24):
    """Dashboard metrics for the last N hours."""
    monitor = SystemMonitor(db_path=AI_DB_PATH)
    return monitor.get_dashboard_data(hours=hours)

@app.get("/api/ai/confidence-history")
def get_confidence_history(pair: str = None, days: int = 7):
    """Confidence calibration geçmişi."""
    cutoff = (datetime.now(tz=timezone.utc) - timedelta(days=days)).isoformat()
    query = "SELECT timestamp, confidence as predicted_confidence, outcome_pnl as actual_outcome FROM ai_decisions WHERE timestamp >= ?"
    params = [cutoff]
    
    if pair:
        query += " AND pair = ?"
        params.append(pair)
        
    query += " ORDER BY timestamp ASC"
    
    with get_db_conn() as conn:
        try:
            rows = conn.execute(query, params).fetchall()
            return [dict(r) for r in rows]
        except sqlite3.OperationalError:
            return []

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8890)
