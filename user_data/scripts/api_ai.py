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
from db import get_connection, get_db_connection

app = FastAPI(title="Freqtrade AI API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_db_conn():
    conn = get_db_connection()
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
        resp = requests.get(f"http://127.0.0.1:8891/signal/{pair}", timeout=30)
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


@app.get("/api/ai/llm_routing_insights")
def llm_routing_insights(hours: int = 24):
    """EK.2.13: per (task, provider) routing stats for the last `hours`.

    Surfaces what LinUCB has settled on for each pipeline stage so a human
    can cross-check whether the bandit's picks look sensible (e.g. Gemini
    dominating coordinator_debate, Groq dominating agent_pool_r1) and
    whether any provider is stuck at 0% success after cold-start.
    """
    from db import get_db_connection

    conn = get_db_connection()
    try:
        rows = conn.execute(
            """
            SELECT COALESCE(NULLIF(agent_name, ''), 'unknown') AS task,
                   COALESCE(NULLIF(trading_pair, ''), 'unknown') AS pair,
                   provider,
                   COUNT(*)           AS n_calls,
                   AVG(latency_ms)    AS avg_latency_ms,
                   SUM(CASE WHEN status='success' THEN 1 ELSE 0 END) * 1.0 / COUNT(*)
                                      AS success_rate
            FROM llm_calls
            WHERE datetime(timestamp) >= datetime('now', ? )
            GROUP BY task, provider
            ORDER BY task, n_calls DESC
            """,
            (f"-{int(hours)} hours",),
        ).fetchall()
    finally:
        conn.close()

    return {
        "hours": hours,
        "routing": [
            {
                "task": r["task"],
                "pair": r["pair"],
                "provider": r["provider"],
                "n_calls": int(r["n_calls"] or 0),
                "avg_latency_ms": round(float(r["avg_latency_ms"] or 0.0), 1),
                "success_rate": round(float(r["success_rate"] or 0.0), 3),
            }
            for r in rows
        ],
    }

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

    try:
        from neural_organism import _p as _np
        conf_exp = float(_np("sizing.confidence_exponent", 1.0))
    except Exception:
        conf_exp = 1.0
    return {
        "autonomy_level": autonomy.current_level,
        "daily_var_pct": daily_var_pct,
        "daily_budget": daily_budget,
        "semantic_cache_ttl": 300,
        "confidence_exponent": conf_exp,
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


# ═══════════════════════════════════════════════════════════
# SPRINT 2: Phase 26 AI Organism Endpoints
# ═══════════════════════════════════════════════════════════


@app.get("/api/ai/organism")
def get_organism_status():
    """Sprint 2: Full organism status — lifecycle, hormones, phi."""
    result = {}
    try:
        from autonomous_lifecycle import get_lifecycle
        lc = get_lifecycle()
        result["lifecycle"] = lc.get_status()
    except Exception:
        result["lifecycle"] = {"error": "not available"}

    try:
        from phi_consciousness import get_phi
        result["phi"] = get_phi().compute_phi()
    except Exception:
        result["phi"] = {"error": "not available"}

    try:
        from self_model import get_self_model
        sm = get_self_model()
        result["self_model"] = sm.get_status()
    except Exception:
        result["self_model"] = {"error": "not available"}

    return result


@app.get("/api/ai/causal-graph")
def get_causal_graph():
    """Sprint 2: Active causal discoveries from PCMCI+."""
    try:
        from causal_engine import CausalEngine
        engine = CausalEngine()
        edges = engine.get_active_edges()
        return {"edges": edges, "count": len(edges)}
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/ai/causal-parents/{variable}")
def get_causal_parents(variable: str):
    """Sprint 2: Get causal parents of a variable."""
    try:
        from causal_engine import CausalEngine
        engine = CausalEngine()
        parents = engine.get_causal_parents(variable)
        return {"variable": variable, "parents": parents}
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/ai/pheromone")
def get_pheromone_state():
    """Sprint 2: Current pheromone field state."""
    try:
        from pheromone_field import get_pheromone_field
        pfield = get_pheromone_field()
        keys = ["prediction", "uncertainty", "organism_health",
                "HORMONE_STATE", "FEAR_LEVEL", "lifecycle_state",
                "multimodal_fusion", "cerebellum_timing",
                "exploration_suggestions", "lob_state",
                "order_flow_state", "mm_state"]
        state = {}
        for key in keys:
            val = pfield.read(key)
            if val is not None:
                state[key] = val
        return {"signals": state, "active_count": len(state)}
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/ai/constitution")
def get_constitution_status():
    """Sprint 2: Constitution enforcer status."""
    try:
        from constitution import get_constitution
        return get_constitution().get_status()
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/ai/cerebellum")
def get_cerebellum_schedule():
    """Sprint 2: 24-hour performance schedule."""
    try:
        from cerebellum_timing import get_cerebellum
        cerebellum = get_cerebellum()
        return {
            "schedule": cerebellum.get_full_schedule(),
            "best_hours": cerebellum.get_best_hours(5),
            "worst_hours": cerebellum.get_worst_hours(5),
        }
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/ai/ablation")
def get_ablation_league():
    """Sprint 2: Module ablation league table."""
    try:
        from ablation_league import get_ablation_league
        league = get_ablation_league()
        return {"league": league.get_league_table()}
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/ai/model-risk")
def get_model_risk():
    """Sprint 2: Model risk assessment."""
    try:
        from model_risk_engine import get_model_risk_engine
        return get_model_risk_engine().assess_risk()
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/ai/training-status")
def get_training_status():
    """Sprint 2: CatBoost + IQL training pipeline status."""
    result = {}
    try:
        from backtest_label_generator import BacktestLabelGenerator
        gen = BacktestLabelGenerator()
        result["training_data"] = gen.get_status()
    except Exception:
        result["training_data"] = {"error": "not available"}

    try:
        conn = __import__('db').get_db_connection(AI_DB_PATH)
        try:
            row = conn.execute("""
                SELECT model_version, n_train, n_features,
                       train_accuracy, test_accuracy, test_f1, trained_at
                FROM catboost_training_runs
                ORDER BY trained_at DESC LIMIT 1
            """).fetchone()
            result["catboost_latest"] = dict(row) if row else None
        finally:
            conn.close()
    except Exception:
        result["catboost_latest"] = None

    return result


@app.get("/api/ai/benchmark")
def get_atcb_benchmark():
    """Sprint 2: ATCB benchmark results."""
    try:
        import os, json
        report_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "models", "atcb_benchmark_latest.json"
        )
        if os.path.exists(report_path):
            with open(report_path) as f:
                return json.load(f)
        return {"error": "no benchmark results yet"}
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/ai/decision-contracts")
def get_recent_contracts():
    """Sprint 2: Recent decision contracts (provenance trail)."""
    try:
        from decision_contract import get_decision_contract
        dc = get_decision_contract()
        return {"contracts": dc.get_recent_contracts(10)}
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/ai/counterfactuals")
def get_counterfactual_insights():
    """Sprint 2: Counterfactual analysis insights."""
    try:
        from counterfactual_engine import CounterfactualEngine
        engine = CounterfactualEngine()
        return {"optimal_params": engine.get_optimal_params()}
    except Exception as e:
        return {"error": str(e)}


# ═══ PHASE 30 — Vue dashboard backends (C.9) ═══

@app.get("/api/v1/ai/regime")
def phase30_regime_watch():
    """Phase 30 C.9 — RegimeWatch.vue feed (regime_layers schema-aware)."""
    try:
        from db import AI_DB_PATH, get_db_connection
        with get_db_connection(AI_DB_PATH) as conn:
            rows = conn.execute(
                """SELECT pair,
                          COALESCE(layer3_adx_regime, 'unknown') AS regime,
                          COALESCE(regime_change_prob * 100, 0) AS adx,
                          300 AS ttl_seconds
                   FROM regime_layers
                   WHERE timestamp >= datetime('now', '-30 minutes')
                   ORDER BY timestamp DESC LIMIT 50"""
            ).fetchall()
            return [{"pair": r[0], "regime": r[1], "adx": float(r[2] or 0),
                     "ttl_seconds": int(r[3])} for r in rows]
    except Exception as e:
        return [{"error": str(e)}]


@app.get("/api/v1/ai/organism")
def phase30_organism_health():
    """Phase 30 C.9 — OrganismHealth.vue feed (streak_state schema-aware)."""
    try:
        from db import AI_DB_PATH, get_db_connection
        with get_db_connection(AI_DB_PATH) as conn:
            row = conn.execute(
                """SELECT cortisol, dopamine, serotonin, adrenaline,
                          market_stress, portfolio_health
                   FROM hormone_state WHERE id=1"""
            ).fetchone()
            streak_row = conn.execute(
                "SELECT consec_wins, consec_losses FROM streak_state WHERE id=1"
            ).fetchone()
            if not row:
                return {"cortisol": 1, "dopamine": 1, "serotonin": 1,
                        "adrenaline": 1, "market_stress": 0,
                        "portfolio_health": 0.5, "streak": 0}
            wins = int(streak_row[0] if streak_row else 0)
            losses = int(streak_row[1] if streak_row else 0)
            return {
                "cortisol": float(row[0] or 1.0),
                "dopamine": float(row[1] or 1.0),
                "serotonin": float(row[2] or 1.0),
                "adrenaline": float(row[3] or 1.0),
                "market_stress": float(row[4] or 0.0),
                "portfolio_health": float(row[5] or 0.5),
                "streak": wins - losses,
            }
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/v1/ai/agents/scorecard")
def phase30_agent_scorecard():
    """Phase 30 C.9 — AgentScorecard.vue (agent_performance schema-aware)."""
    try:
        from db import AI_DB_PATH, get_db_connection
        from audit_recovery_rate import weekly_summary

        rec = weekly_summary()
        rec_by_agent = {r["agent"]: r["recovery_rate"]
                        for r in rec.get("per_agent_class", [])}

        with get_db_connection(AI_DB_PATH) as conn:
            rows = conn.execute(
                """SELECT agent_type,
                          COUNT(*) AS n,
                          SUM(CASE WHEN was_correct=1 THEN 1 ELSE 0 END) AS wins,
                          AVG(COALESCE(outcome_pnl, 0)) AS avg_pnl
                   FROM agent_performance
                   WHERE timestamp >= datetime('now', '-30 days')
                   GROUP BY agent_type
                   ORDER BY n DESC"""
            ).fetchall()
            out = []
            for agent_type, n, wins, avg_pnl in rows:
                n = int(n or 0)
                wins = int(wins or 0)
                trust = (wins / n) if n else 0.5
                out.append({
                    "name": agent_type or "?",
                    "trust": float(trust),
                    "n_decisions": n,
                    "winrate": float(trust),
                    "recovery_rate": float(rec_by_agent.get(agent_type or "?", 0.0)),
                    "avg_pnl": float(avg_pnl or 0.0),
                })
            return out
    except Exception as e:
        return [{"error": str(e)}]


@app.get("/api/v1/ai/promotion_gate")
def phase30_promotion_gate():
    """Phase 30 C.9/D.9 — PromotionGate.vue feed."""
    try:
        from promotion_gate import evaluate_gate
        r = evaluate_gate(window_days=14)
        return {
            "passed": r.passed,
            "eligibility_pct": r.eligibility_pct,
            "blocked_by": r.blocked_by,
            "metrics": r.metrics,
        }
    except Exception as e:
        return {"passed": False, "eligibility_pct": 0.0,
                "blocked_by": ["error"], "metrics": {"error": str(e)}}


@app.get("/api/v1/ai/telemetry")
def phase30_telemetry(kind_prefix: str = "", since_hours: int = 24, limit: int = 100):
    """Phase 30 B.18 — Telemetry single feed."""
    try:
        from telemetry import query
        return query(kind_prefix=kind_prefix, since_hours=since_hours, limit=limit)
    except Exception as e:
        return [{"error": str(e)}]


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8890)
