"""
Freqtrade Local Model Server
Hosts ColBERT, BGE, FlashRank on a single FastAPI service.
Port: 8895

Phase 21 fixes:
- torch.no_grad() + explicit tensor cleanup in ColBERT
- Semaphore(1) for ColBERT to prevent concurrent OOM
- BGE input size limits (max 64 texts, 2048 chars each)
- RSS circuit breaker (503 at 2.5GB)
"""
import logging
import time
import threading
import gc
import os
from typing import List, Optional
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from flashrank import Ranker, RerankRequest as FlashRankRerankRequest
import uvicorn

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Freqtrade Model Server")

_colbert_model = None
_bge_model = None
_flashrank_model = None

# Per-model last-inference timestamps (seconds since epoch). Used by the
# idle reaper thread to unload models that haven't served in a while —
# RSS goes back down without restarting the whole service. Each endpoint
# refreshes its own timestamp on entry.
_colbert_last_ts: float = 0.0
_bge_last_ts: float = 0.0
_flashrank_last_ts: float = 0.0
# Serialize loads — two requests can race during cold-start
_load_lock = threading.Lock()

# Serialize ColBERT requests — only 1 at a time to prevent OOM from concurrent tensors
_colbert_lock = threading.Semaphore(1)

# Sprint 2026-05-06 (F6): static fallbacks for memory thresholds. Real
# values come from PARAM_REGISTRY via _param() so the cgroup_recalibrate
# cron + neural BCM/STDP weight updates can adapt them weekly without a
# code redeploy. Defaults here are tighter than the prior 3500/4500/3600/
# 1800/600/300 set so that under traffic the reaper actually fires and
# the 8 cgroup OOM-kills observed on 2026-05-05 do not recur.
_RSS_WARN_MB = 3300
_RSS_LIMIT_MB = 4200
_IDLE_DEFAULT_S = 3600.0
_IDLE_WARN_S = 900.0
_IDLE_LIMIT_S = 300.0
_REAPER_INTERVAL_S = 60.0


def _param(key: str, fallback: float) -> float:
    """Read a tunable from PARAM_REGISTRY when neural_organism is importable;
    otherwise fall back to the static value above. The lookup is best-effort
    and cheap — neural_organism caches per-key reads in its own LRU."""
    try:
        from neural_organism import _p as _neural_p  # type: ignore
        return float(_neural_p(key, fallback))
    except Exception:
        return float(fallback)


def _get_rss_mb() -> float:
    """Get current RSS in MB without psutil dependency."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
    except Exception as e:
        logger.warning(f"[MemoryGuard] Cannot read /proc/self/status: {e} — RSS guard disabled")
    return 0.0


def _ensure_colbert():
    """Idempotent ColBERT loader. Returns the model or None on failure."""
    global _colbert_model
    if _colbert_model is not None:
        return _colbert_model
    with _load_lock:
        if _colbert_model is not None:
            return _colbert_model
        t0 = time.time()
        try:
            import torch
            torch.set_num_threads(2)
            from sentence_transformers import SentenceTransformer
            _colbert_model = SentenceTransformer("jinaai/jina-colbert-v2", trust_remote_code=True)
            logger.info(f"ColBERT loaded in {time.time()-t0:.1f}s")
        except Exception as e:
            logger.error(f"ColBERT load failed: {e}")
    return _colbert_model


def _ensure_bge():
    """Idempotent BGE loader. Returns the model or None on failure."""
    global _bge_model
    if _bge_model is not None:
        return _bge_model
    with _load_lock:
        if _bge_model is not None:
            return _bge_model
        t0 = time.time()
        try:
            from sentence_transformers import SentenceTransformer
            _bge_model = SentenceTransformer("philschmid/bge-base-financial-matryoshka")
            logger.info(f"BGE loaded in {time.time()-t0:.1f}s")
        except Exception as e:
            logger.error(f"BGE load failed: {e}")
    return _bge_model


def _ensure_flashrank():
    """Idempotent FlashRank loader. Returns the model or None on failure."""
    global _flashrank_model
    if _flashrank_model is not None:
        return _flashrank_model
    with _load_lock:
        if _flashrank_model is not None:
            return _flashrank_model
        t0 = time.time()
        try:
            from flashrank import Ranker
            _flashrank_model = Ranker()
            logger.info(f"FlashRank loaded in {time.time()-t0:.1f}s")
        except Exception as e:
            logger.error(f"FlashRank load failed: {e}")
    return _flashrank_model


def _unload(name: str) -> bool:
    """Drop the named model + run gc.collect. Returns True if memory freed.

    AUDIT-10 (2026-04-25): the unload path now acquires `_load_lock` so it
    serialises with the `_ensure_*` loaders. Without this, the reaper
    could clobber a fresh load that completed an instant after its None
    check — the next request would pay a cold-load cost for nothing.
    """
    global _colbert_model, _bge_model, _flashrank_model
    rss_before = _get_rss_mb()
    with _load_lock:
        if name == "colbert":
            if _colbert_model is None:
                return False
            _colbert_model = None
        elif name == "bge":
            if _bge_model is None:
                return False
            _bge_model = None
        elif name == "flashrank":
            if _flashrank_model is None:
                return False
            _flashrank_model = None
        else:
            return False
    gc.collect()
    try:
        import torch
        if hasattr(torch, "cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    rss_after = _get_rss_mb()
    logger.info(
        f"[ModelReaper] Unloaded {name}: RSS {rss_before:.0f}MB → {rss_after:.0f}MB"
    )
    return True


def _idle_threshold() -> float:
    """Pick the tightest idle window the current RSS justifies.

    Sprint 2026-05-06 (F6): thresholds re-read from PARAM_REGISTRY each
    call so the auto cgroup recalibration job + neural BCM/STDP credit
    assignment can drift them based on memory-pressure outcomes without
    requiring a service restart.
    """
    rss = _get_rss_mb()
    rss_limit = _param("model_server.rss_limit_mb", _RSS_LIMIT_MB)
    rss_warn = _param("model_server.rss_warn_mb", _RSS_WARN_MB)
    if rss > rss_limit:
        return _param("model_server.idle_limit_s", _IDLE_LIMIT_S)
    if rss > rss_warn:
        return _param("model_server.idle_warn_s", _IDLE_WARN_S)
    return _param("model_server.idle_default_s", _IDLE_DEFAULT_S)


def _reaper_loop():
    """Daemon thread: every reaper-interval, evict any model idle past
    the threshold. Sprint 2026-05-06 (F6): interval is PARAM-driven —
    weekly auto-cgroup recalibrate can tighten it under memory pressure.
    """
    while True:
        try:
            interval = _param("model_server.reaper_interval_s", _REAPER_INTERVAL_S)
            time.sleep(max(15.0, interval))
            now = time.time()
            threshold = _idle_threshold()

            checks = (
                ("colbert", _colbert_model, _colbert_last_ts),
                ("bge", _bge_model, _bge_last_ts),
                ("flashrank", _flashrank_model, _flashrank_last_ts),
            )
            for name, model, last_ts in checks:
                if model is None:
                    continue
                # Models that have never served (last_ts==0) keep their
                # startup tenancy until the threshold elapses since boot.
                age = now - (last_ts or _process_start_ts)
                if age > threshold:
                    _unload(name)
        except Exception as e:
            logger.debug(f"[ModelReaper] tick error: {e}")


_process_start_ts = time.time()
_reaper_thread: Optional[threading.Thread] = None


def _start_reaper_once():
    """Idempotent reaper bootstrap. Called from startup hook."""
    global _reaper_thread
    if _reaper_thread is not None and _reaper_thread.is_alive():
        return
    _reaper_thread = threading.Thread(
        target=_reaper_loop, name="model-reaper", daemon=True
    )
    _reaper_thread.start()
    logger.info(f"[ModelReaper] started (default idle={_IDLE_DEFAULT_S:.0f}s)")


def _load_models():
    """Initial model load (cold-start). Endpoints will lazy-reload on demand."""
    _ensure_colbert()
    _ensure_bge()
    _ensure_flashrank()


@app.middleware("http")
async def memory_guard(request: Request, call_next):
    """RSS circuit breaker — Sprint 2026-05-06 (F6): PARAM-driven thresholds
    so the cgroup_recalibrate job can tighten/loosen them weekly without a
    service redeploy."""
    rss = _get_rss_mb()
    rss_limit = _param("model_server.rss_limit_mb", _RSS_LIMIT_MB)
    rss_warn = _param("model_server.rss_warn_mb", _RSS_WARN_MB)
    if rss > rss_limit:
        gc.collect()
        rss = _get_rss_mb()
        if rss > rss_limit:
            logger.warning(f"[MemoryGuard] RSS={rss:.0f}MB > {rss_limit:.0f}MB, returning 503")
            return JSONResponse(
                status_code=503,
                content={"error": "memory_pressure", "rss_mb": round(rss)},
                headers={"Retry-After": "5"},
            )
    elif rss > rss_warn:
        gc.collect()
    return await call_next(request)


class EmbedRequest(BaseModel):
    texts: List[str]

class EmbedResponse(BaseModel):
    embeddings: List[List[float]]

class RerankRequest(BaseModel):
    query: str
    documents: List[str]
    top_k: Optional[int] = 10

class RerankResult(BaseModel):
    index: int
    score: float
    text: str

class RerankResponse(BaseModel):
    results: List[RerankResult]


@app.get("/health")
def health():
    rss = _get_rss_mb()
    now = time.time()

    def _idle(last_ts: float) -> Optional[float]:
        if last_ts <= 0:
            return None
        return round(now - last_ts, 1)

    return {
        "status": "online",
        "colbert": "active" if _colbert_model else "evicted",
        "bge": "active" if _bge_model else "evicted",
        "flashrank": "active" if _flashrank_model else "evicted",
        "idle_s": {
            "colbert": _idle(_colbert_last_ts),
            "bge": _idle(_bge_last_ts),
            "flashrank": _idle(_flashrank_last_ts),
        },
        "rss_mb": round(rss),
        "memory_pressure": rss > _param("model_server.rss_warn_mb", _RSS_WARN_MB),
        "idle_threshold_s": _idle_threshold(),
    }


@app.post("/embed/bge", response_model=EmbedResponse)
def embed_bge(req: EmbedRequest):
    global _bge_last_ts
    bge = _ensure_bge()
    if bge is None:
        return EmbedResponse(embeddings=[])
    _bge_last_ts = time.time()
    try:
        # Input limits: max 64 texts, max 2048 chars each
        texts = [t[:2048] for t in req.texts[:64]]
        embeddings = bge.encode(texts, normalize_embeddings=True).tolist()
        return EmbedResponse(embeddings=embeddings)
    except Exception as e:
        logger.warning(f"[BGE] Encode failed, returning empty: {e}")
        return EmbedResponse(embeddings=[])


@app.post("/rerank/colbert", response_model=RerankResponse)
def rerank_colbert(req: RerankRequest):
    global _colbert_last_ts
    colbert = _ensure_colbert()
    if colbert is None:
        return RerankResponse(results=[])
    _colbert_last_ts = time.time()

    # Serialize ColBERT requests to prevent concurrent tensor OOM
    acquired = _colbert_lock.acquire(timeout=30)
    if not acquired:
        logger.warning("[ColBERT] Semaphore timeout (30s), returning original order")
        results = [RerankResult(index=i, score=1.0/(i+1), text=req.documents[i][:512])
                   for i in range(min(len(req.documents), req.top_k or 10))]
        return RerankResponse(results=results)

    try:
        import numpy as np
        import torch

        max_docs = 15
        max_doc_len = 4096
        docs_to_rank = req.documents[:max_docs]
        truncated_docs = [d[:max_doc_len] for d in docs_to_rank]

        with torch.no_grad():
            query_emb = colbert.encode(req.query)
            scores = []
            for i, doc in enumerate(truncated_docs):
                try:
                    doc_emb = colbert.encode(doc)
                    if hasattr(query_emb, "shape") and len(query_emb.shape) == 2:
                        if hasattr(doc_emb, "shape") and len(doc_emb.shape) == 2:
                            sim = float(np.mean(np.max(np.dot(query_emb, doc_emb.T), axis=1)))
                        else:
                            q_flat = query_emb.mean(axis=0) if len(query_emb.shape) == 2 else query_emb
                            sim = float(np.dot(q_flat, doc_emb) / (np.linalg.norm(q_flat) * np.linalg.norm(doc_emb) + 1e-8))
                    else:
                        sim = float(np.dot(query_emb, doc_emb) / (np.linalg.norm(query_emb) * np.linalg.norm(doc_emb) + 1e-8))
                    scores.append((i, sim, req.documents[i][:512]))
                    del doc_emb
                except RuntimeError as re:
                    logger.debug(f"[ColBERT] Doc {i} tensor error, assigning neutral: {re}")
                    scores.append((i, 0.0, req.documents[i][:512]))

            del query_emb

        scores.sort(key=lambda x: x[1], reverse=True)
        results = [RerankResult(index=s[0], score=s[1], text=s[2]) for s in scores[:req.top_k]]
        gc.collect()
        return RerankResponse(results=results)
    except RuntimeError as e:
        logger.warning(f"[ColBERT] Tensor error, returning original order: {e}")
        results = [RerankResult(index=i, score=1.0/(i+1), text=req.documents[i][:512])
                   for i in range(min(len(req.documents), req.top_k or 10))]
        return RerankResponse(results=results)
    except Exception as e:
        logger.warning(f"[ColBERT] Unexpected error, returning original order: {e}")
        results = [RerankResult(index=i, score=1.0/(i+1), text=req.documents[i][:512])
                   for i in range(min(len(req.documents), req.top_k or 10))]
        return RerankResponse(results=results)
    finally:
        _colbert_lock.release()


@app.post("/rerank/flashrank", response_model=RerankResponse)
def rerank_flashrank(req: RerankRequest):
    global _flashrank_last_ts
    flash = _ensure_flashrank()
    if flash is None:
        return RerankResponse(results=[])
    _flashrank_last_ts = time.time()
    try:
        passages = [{"id": i, "text": doc} for i, doc in enumerate(req.documents)]
        flash_req = FlashRankRerankRequest(query=req.query, passages=passages)
        reranked = flash.rerank(flash_req)
        results = []
        for r in reranked[:req.top_k]:
            results.append(RerankResult(
                index=r.get("id", 0),
                score=r.get("score", 0.0),
                text=r.get("text", "")
            ))
        return RerankResponse(results=results)
    except Exception as e:
        logger.warning(f"[FlashRank] Rerank failed, returning original order: {e}")
        results = [RerankResult(index=i, score=1.0/(i+1), text=req.documents[i])
                   for i in range(min(len(req.documents), req.top_k or 10))]
        return RerankResponse(results=results)


@app.on_event("startup")
def startup():
    _load_models()
    _start_reaper_once()


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8895, limit_concurrency=2)
