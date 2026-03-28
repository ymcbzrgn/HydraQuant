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

    # Read last used model from llm_calls
    active_model = "awaiting first call"
    try:
        with get_db_conn() as conn:
            row = conn.execute("SELECT model FROM llm_calls ORDER BY id DESC LIMIT 1").fetchone()
            if row:
                active_model = row["model"]
    except Exception:
        pass

    return {
        "status": "online",
        "autonomy_level": autonomy.current_level,
        "active_model": active_model,
        "daily_cost": daily_cost,
        "cache_hit_rate": 0.0,
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

def _get_fear_greed() -> int:
    """Read latest Fear & Greed Index from DB."""
    try:
        with get_db_conn() as conn:
            row = conn.execute(
                "SELECT value FROM fear_and_greed ORDER BY timestamp DESC LIMIT 1"
            ).fetchone()
            if row:
                return int(row["value"])
    except Exception:
        pass
    return 50  # neutral fallback

@app.get("/api/ai/sentiment/{pair:path}")
def get_sentiment(pair: str):
    """Belirli pair için sentiment verisi."""
    fng = _get_fear_greed()
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
                    "fear_greed": fng,
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
        "fear_greed": fng,
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
    today_cost = summary.get("total_cost", 0.0)

    # Read real daily budget from RiskBudgetManager (1% of portfolio by default)
    try:
        risk_mgr = RiskBudgetManager(db_path=AI_DB_PATH)
        daily_budget = float(risk_mgr.daily_budget)
    except Exception:
        daily_budget = 10.0  # fallback

    return {
        "today_cost": today_cost,
        "models": summary.get("calls_by_model", {}),
        "budget_remaining": max(0.0, daily_budget - today_cost)
    }

@app.get("/api/ai/autonomy")
def get_autonomy():
    """Autonomy level detayları."""
    autonomy = AutonomyManager(db_path=AI_DB_PATH)

    # Fetch promotion criteria as dict for frontend
    raw_criteria = PROMOTION_CRITERIA.get(autonomy.current_level, ())
    criteria_dict = {}
    if len(raw_criteria) >= 4:
        criteria_dict = {
            "min_trades": raw_criteria[0],
            "min_sharpe": raw_criteria[1],
            "max_drawdown": raw_criteria[2] / 100.0,
            "min_days": raw_criteria[3],
        }

    # Read autonomy state for basic history
    history = []
    try:
        with get_db_conn() as conn:
            row = conn.execute(
                "SELECT level, promoted_at, total_trades, sharpe_estimate, "
                "max_drawdown_pct, days_at_level, updated_at FROM autonomy_state WHERE id = 1"
            ).fetchone()
            if row and row["promoted_at"]:
                history.append({
                    "old_level": max(0, row["level"] - 1),
                    "new_level": row["level"],
                    "timestamp": row["promoted_at"],
                    "reason": f"Promoted after {row['total_trades']} trades, Sharpe {row['sharpe_estimate']:.2f}"
                })
    except Exception:
        pass

    return {
        "current_level": autonomy.current_level,
        "kelly_fraction": autonomy.get_kelly_fraction(),
        "criteria": criteria_dict,
        "history": history
    }

@app.get("/api/ai/risk")
def get_risk():
    """Risk budget durumu (gerçek portfolio bakiyesiyle)."""
    # Read real portfolio from SQLite (synced by strategy)
    portfolio_value = 10000.0
    with get_db_conn() as conn:
        try:
            row = conn.execute("SELECT total_balance FROM portfolio_state WHERE id = 1").fetchone()
            if row and float(row['total_balance']) > 0:
                portfolio_value = float(row['total_balance'])
        except Exception:
            pass

    risk_manager = RiskBudgetManager(portfolio_value=portfolio_value, db_path=AI_DB_PATH)
    daily_budget = float(risk_manager.daily_budget)
    consumed = float(risk_manager._consumed)
    utilization_pct = min(100.0, (consumed / daily_budget) * 100) if daily_budget > 0 else 0.0

    # Count active positions (recent signals that are still pending outcome)
    active_positions = 0
    try:
        with get_db_conn() as conn:
            row = conn.execute(
                "SELECT COUNT(*) as cnt FROM ai_decisions WHERE outcome_pnl IS NULL "
                "AND timestamp >= datetime('now', '-24 hours')"
            ).fetchone()
            if row:
                active_positions = row["cnt"]
    except Exception:
        pass

    return {
        "portfolio_value": portfolio_value,
        "daily_budget": daily_budget,
        "consumed": consumed,
        "utilization_pct": utilization_pct,
        "active_positions": active_positions
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

@app.get("/api/ai/signal/{pair:path}")
def get_signal_for_pair(pair: str):
    """Proxy to RAG Signal Service — for dashboard display."""
    try:
        import requests
        resp = requests.get(f"http://127.0.0.1:8891/signal/{pair}", timeout=60)
        if resp.status_code == 200:
            return resp.json()
        return {"signal": "NEUTRAL", "confidence": 0.0, "reasoning": f"RAG service returned {resp.status_code}"}
    except Exception as e:
        return {"signal": "NEUTRAL", "confidence": 0.0, "reasoning": f"RAG service error: {e}"}

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

@app.get("/api/ai/portfolio")
def get_portfolio():
    """Gerçek exchange bakiyesi (strategy tarafından persist ediliyor)."""
    with get_db_conn() as conn:
        try:
            row = conn.execute("SELECT * FROM portfolio_state WHERE id = 1").fetchone()
            if row:
                import json
                assets = json.loads(row["assets_json"]) if row["assets_json"] else {}
                # Compute total USD from enriched assets
                total_usd = 0.0
                for info in assets.values():
                    if isinstance(info, dict) and "usd" in info:
                        total_usd += info["usd"]
                    elif isinstance(info, (int, float)):
                        total_usd += info  # Old format: assume stake currency
                return {
                    "stake_currency": row["stake_currency"],
                    "total_balance": row["total_balance"],
                    "free_balance": row["free_balance"],
                    "in_trades": row["in_trades"],
                    "assets": assets,
                    "total_portfolio_usd": round(total_usd, 2),
                    "updated_at": row["updated_at"],
                }
            return {"total_balance": 0, "note": "No portfolio data yet. Bot has not synced."}
        except sqlite3.OperationalError:
            return {"total_balance": 0, "note": "portfolio_state table not created yet."}

@app.get("/api/ai/market-sentiment")
def get_market_sentiment():
    """Genel piyasa sentiment özeti (Fear & Greed + top coins)."""
    fng = _get_fear_greed()
    sentiment_data = {"fear_greed": fng, "coins": {}}

    try:
        with get_db_conn() as conn:
            rows = conn.execute(
                "SELECT coin, sentiment_1h, sentiment_4h, sentiment_24h, news_count_24h, timestamp "
                "FROM coin_sentiment_rolling ORDER BY timestamp DESC LIMIT 20"
            ).fetchall()
            seen = set()
            for r in rows:
                coin = r["coin"]
                if coin not in seen:
                    seen.add(coin)
                    sentiment_data["coins"][coin] = {
                        "sentiment_1h": r["sentiment_1h"],
                        "sentiment_4h": r["sentiment_4h"],
                        "sentiment_24h": r["sentiment_24h"],
                        "news_count": r["news_count_24h"],
                    }
    except Exception:
        pass

    return sentiment_data

@app.get("/api/ai/settings")
def get_ai_settings():
    """AI config read-only view."""
    autonomy = AutonomyManager(db_path=AI_DB_PATH)
    try:
        risk_mgr = RiskBudgetManager(db_path=AI_DB_PATH)
        daily_var_pct = risk_mgr.daily_var_pct
        daily_budget = float(risk_mgr.daily_budget)
    except Exception:
        daily_var_pct = 0.01
        daily_budget = 100.0

    return {
        "autonomy_level": autonomy.current_level,
        "daily_var_pct": daily_var_pct,
        "daily_budget": daily_budget,
        "semantic_cache_ttl": 300,
        "confidence_exponent": 2.0,
        "rag_chunk_overlap": 100,
    }

@app.get("/api/ai/daily-stats")
def get_daily_stats():
    """Bugünkü trade istatistikleri (Daily P&L)."""
    today = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
    stats = {
        "daily_pnl": 0.0,
        "daily_pnl_pct": 0.0,
        "closed_today": 0,
        "wins": 0,
        "losses": 0,
        "best_trade": None,
    }

    try:
        with get_db_conn() as conn:
            rows = conn.execute(
                "SELECT pair, outcome_pnl FROM ai_decisions "
                "WHERE outcome_pnl IS NOT NULL AND date(timestamp) = ?",
                (today,)
            ).fetchall()

            if rows:
                total_pnl = 0.0
                best_pnl = -float('inf')
                best_pair = ""
                for r in rows:
                    pnl = float(r["outcome_pnl"])
                    total_pnl += pnl
                    if pnl > 0:
                        stats["wins"] += 1
                    elif pnl < 0:
                        stats["losses"] += 1
                    if pnl > best_pnl:
                        best_pnl = pnl
                        best_pair = r["pair"]
                stats["closed_today"] = len(rows)
                stats["daily_pnl"] = round(total_pnl, 2)

                # Compute pct from portfolio value
                portfolio_value = 10000.0
                try:
                    prow = conn.execute("SELECT total_balance FROM portfolio_state WHERE id = 1").fetchone()
                    if prow and float(prow['total_balance']) > 0:
                        portfolio_value = float(prow['total_balance'])
                except Exception:
                    pass
                stats["daily_pnl_pct"] = round((total_pnl / portfolio_value) * 100, 2)
                stats["best_trade"] = f"+${best_pnl:.2f} ({best_pair})" if best_pnl > 0 else f"${best_pnl:.2f} ({best_pair})"
    except Exception:
        pass

    return stats


@app.get("/api/ai/alerts")
def get_alerts(limit: int = 20):
    """Son sistem alertleri (health + autonomy + risk)."""
    alerts: list = []

    # Health alerts
    try:
        monitor = SystemMonitor(db_path=AI_DB_PATH)
        health = monitor.check_health()
        for a in health.get("alerts", []):
            alerts.append({"level": "WARNING", "message": a, "timestamp": datetime.now(tz=timezone.utc).isoformat()})
    except Exception:
        pass

    # Budget alert
    try:
        risk_mgr = RiskBudgetManager(db_path=AI_DB_PATH)
        util = risk_mgr.budget_utilization()
        if util >= 1.0:
            alerts.append({"level": "ERROR", "message": f"Risk budget exceeded ({util*100:.0f}%)", "timestamp": datetime.now(tz=timezone.utc).isoformat()})
        elif util >= 0.75:
            alerts.append({"level": "WARNING", "message": f"Risk budget {util*100:.0f}% consumed", "timestamp": datetime.now(tz=timezone.utc).isoformat()})
    except Exception:
        pass

    # Cost alert
    try:
        cost_tracker = LLMCostTracker(db_path=AI_DB_PATH)
        daily_cost = cost_tracker.get_daily_cost()
        if daily_cost > 5.0:
            alerts.append({"level": "ERROR", "message": f"Daily API cost ${daily_cost:.2f} exceeds $5 budget", "timestamp": datetime.now(tz=timezone.utc).isoformat()})
    except Exception:
        pass

    return alerts[:limit]


@app.get("/api/ai/hypothetical")
def get_hypothetical():
    """$100 simülasyon portföyü durumu."""
    result = {
        "current_balance": 100.0,
        "total_return_pct": 0.0,
        "total_trades": 0,
        "today_pnl_pct": 0.0,
    }

    try:
        engine = ForgonePnLEngine(db_path=AI_DB_PATH)
        hyp = engine.get_hypothetical_portfolio()
        if hyp:
            result.update(hyp)
    except Exception:
        pass

    # Fallback: compute from ai_decisions if forgone engine doesn't have it
    if result["total_trades"] == 0:
        try:
            with get_db_conn() as conn:
                rows = conn.execute(
                    "SELECT outcome_pnl FROM ai_decisions WHERE outcome_pnl IS NOT NULL ORDER BY timestamp ASC"
                ).fetchall()
                if rows:
                    balance = 100.0
                    today = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
                    today_pnl = 0.0
                    for r in rows:
                        pnl_pct = float(r["outcome_pnl"]) / 100.0
                        balance *= (1 + pnl_pct * 0.01)  # Weighted
                    result["current_balance"] = round(balance, 2)
                    result["total_return_pct"] = round(balance - 100.0, 2)
                    result["total_trades"] = len(rows)
        except Exception:
            pass

    return result


@app.get("/api/ai/market-data")
def get_market_data(pair: str = "BTC/USDT"):
    """Phase 19 Level 3: Get latest derivatives, DeFi, and macro data."""
    try:
        from market_data_fetcher import MarketDataFetcher
        fetcher = MarketDataFetcher()
        return {
            "derivatives": fetcher.get_latest_derivatives(pair),
            "defi": fetcher.get_latest_defi(),
            "macro": fetcher.get_latest_macro(),
        }
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/ai/pattern-stats")
def get_pattern_stats(pair: str = None, regime: str = None, rsi: str = None):
    """Phase 19: Query historical backtest pattern statistics."""
    try:
        from pattern_stat_store import PatternStatStore
        store = PatternStatStore()
        stats = store.query(pair=pair, regime=regime, rsi_bucket=rsi)
        return stats
    except Exception as e:
        return {"error": str(e), "matching_trades": 0}


@app.get("/api/ai/calibration")
def get_calibration():
    """Phase 19: Get confidence calibration report and Brier score."""
    try:
        from confidence_calibrator import ConfidenceCalibrator
        cal = ConfidenceCalibrator()
        return {
            "brier_score": cal.brier_score(),
            "calibration_curve": cal.calibration_curve(),
            "platt_a": cal._platt_a,
            "platt_b": cal._platt_b,
            "calibrated": cal._calibrated,
        }
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/ai/opportunities")
def get_opportunities(top_n: int = 20):
    """Phase 20: Latest opportunity scanner results."""
    try:
        with get_db_conn() as conn:
            rows = conn.execute("""
                SELECT pair, composite_score, top_type, momentum_score, reversion_score,
                       funding_score, regime_shift_score, volume_anomaly_score, timestamp
                FROM opportunity_scores
                WHERE id IN (SELECT MAX(id) FROM opportunity_scores GROUP BY pair)
                ORDER BY composite_score DESC
                LIMIT ?
            """, (top_n,)).fetchall()
            return [dict(r) for r in rows]
    except Exception:
        return []


@app.get("/api/ai/agents")
def get_agent_performance():
    """Phase 20: Agent pool performance statistics."""
    try:
        from agent_pool import AgentPool
        pool = AgentPool(db_path=AI_DB_PATH)
        return pool.get_performance_summary()
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/ai/evidence/{pair:path}")
def get_evidence_audit(pair: str, limit: int = 10):
    """Phase 20: Evidence Engine audit log for a pair."""
    try:
        with get_db_conn() as conn:
            rows = conn.execute("""
                SELECT pair, signal, confidence, sub_scores_json, contradictions_json,
                       evidence_sources_json, regime, max_confidence_cap, timestamp
                FROM evidence_audit_log
                WHERE pair = ? ORDER BY timestamp DESC LIMIT ?
            """, (pair, limit)).fetchall()
            results = []
            for r in rows:
                d = dict(r)
                # Parse JSON fields for cleaner API response
                for jf in ("sub_scores_json", "contradictions_json", "evidence_sources_json"):
                    if d.get(jf):
                        try:
                            d[jf] = __import__('json').loads(d[jf])
                        except Exception:
                            pass
                results.append(d)
            return results
    except Exception:
        return []


@app.get("/api/ai/cross-pair")
def get_cross_pair_intel():
    """Phase 20: Cross-pair market intelligence."""
    try:
        from cross_pair_intel import CrossPairIntel
        intel = CrossPairIntel(db_path=AI_DB_PATH)
        return intel.get_latest()
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8890)
