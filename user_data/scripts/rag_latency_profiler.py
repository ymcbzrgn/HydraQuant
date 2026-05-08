"""Phase 30 A.33 — RAG endpoint latency profiler.

Two parts:
1. FastAPI middleware (used inside rag_graph.py --serve) timing each request.
2. Daily report (CLI / scheduler hook) over rag_endpoint_latency.

Tracks p50/p95/p99 + timeout_breach count per (endpoint, regime).
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_THRESHOLD_MS = 40_000


def fastapi_middleware_factory(timeout_threshold_ms: int = DEFAULT_TIMEOUT_THRESHOLD_MS):
    """Returns an async ASGI middleware coroutine factory.

    Usage:
        from rag_latency_profiler import fastapi_middleware_factory
        @app.middleware("http")
        async def latency_track(request, call_next):
            return await fastapi_middleware_factory()(request, call_next)
    """
    async def _middleware(request, call_next):
        start = time.time()
        response = None
        status = 0
        try:
            response = await call_next(request)
            status = getattr(response, "status_code", 0)
        finally:
            latency_ms = int((time.time() - start) * 1000)
            try:
                from db import AI_DB_PATH, get_db_connection

                with get_db_connection(AI_DB_PATH) as conn:
                    pair = ""
                    try:
                        pair = request.path_params.get("pair", "") or ""
                    except Exception:
                        pass
                    breach = 1 if latency_ms >= timeout_threshold_ms else 0
                    conn.execute(
                        """INSERT INTO rag_endpoint_latency
                           (endpoint, pair, latency_ms, status_code, timeout_breach)
                           VALUES (?, ?, ?, ?, ?)""",
                        (str(request.url.path), pair, latency_ms, int(status), breach),
                    )
                    conn.commit()
            except Exception:
                pass
        return response

    return _middleware


def _percentile(sorted_vals: List[int], pct: float) -> int:
    if not sorted_vals:
        return 0
    k = int(round((len(sorted_vals) - 1) * pct))
    return sorted_vals[max(0, min(len(sorted_vals) - 1, k))]


def daily_report(window_hours: int = 24) -> List[Dict[str, Any]]:
    try:
        from db import AI_DB_PATH, get_db_connection

        with get_db_connection(AI_DB_PATH) as conn:
            rows = conn.execute(
                f"""SELECT endpoint, latency_ms, timeout_breach
                    FROM rag_endpoint_latency
                    WHERE ts >= datetime('now', '-{int(window_hours)} hours')"""
            ).fetchall()
    except Exception as e:
        logger.error(f"[RAGProfiler] db: {e}")
        return []

    by_endpoint: Dict[str, List[int]] = {}
    breaches: Dict[str, int] = {}
    for endpoint, latency_ms, breach in rows:
        by_endpoint.setdefault(endpoint, []).append(int(latency_ms or 0))
        if int(breach or 0):
            breaches[endpoint] = breaches.get(endpoint, 0) + 1

    out = []
    for endpoint, lats in by_endpoint.items():
        s = sorted(lats)
        out.append({
            "endpoint": endpoint,
            "n": len(s),
            "p50_ms": _percentile(s, 0.5),
            "p95_ms": _percentile(s, 0.95),
            "p99_ms": _percentile(s, 0.99),
            "max_ms": s[-1] if s else 0,
            "timeout_breaches": breaches.get(endpoint, 0),
            "breach_rate": round(breaches.get(endpoint, 0) / len(s), 4) if s else 0.0,
        })
    return sorted(out, key=lambda r: r["n"], reverse=True)


def get_rag_signal_with_fallback(pair: str, port: int = 8891, timeout: tuple = (5, 40)):
    """Phase 30 A.33 — RAG client with timeout-aware fallback."""
    try:
        import requests

        return requests.post(
            f"http://127.0.0.1:{port}/signal/{pair}", timeout=timeout
        ).json()
    except Exception as e:
        logger.warning(f"[RAG:fallback] {pair} timeout/err: {e} — using EvidenceEngine direct")
        try:
            from evidence_engine import EvidenceEngine  # type: ignore

            return EvidenceEngine().score(pair, fallback=True)
        except Exception as e2:
            logger.error(f"[RAG:fallback] direct call failed: {e2}")
            return {"signal_action": "neutral", "confidence": 0.0, "fallback": True}
