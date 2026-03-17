"""
Freqtrade Local Model Server
Hosts ColBERT, BGE, FlashRank on a single FastAPI service.
Port: 8895
"""
import logging
import time
from typing import List, Optional
from fastapi import FastAPI
from pydantic import BaseModel
from flashrank import Ranker, RerankRequest as FlashRankRerankRequest
import uvicorn

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Freqtrade Model Server")

_colbert_model = None
_bge_model = None
_flashrank_model = None


def _load_models():
    global _colbert_model, _bge_model, _flashrank_model

    t0 = time.time()
    try:
        from sentence_transformers import SentenceTransformer
        _colbert_model = SentenceTransformer("jinaai/jina-colbert-v2", trust_remote_code=True)
        logger.info(f"ColBERT loaded in {time.time()-t0:.1f}s")
    except Exception as e:
        logger.error(f"ColBERT load failed: {e}")

    t0 = time.time()
    try:
        from sentence_transformers import SentenceTransformer
        _bge_model = SentenceTransformer("philschmid/bge-base-financial-matryoshka")
        logger.info(f"BGE loaded in {time.time()-t0:.1f}s")
    except Exception as e:
        logger.error(f"BGE load failed: {e}")

    t0 = time.time()
    try:
        from flashrank import Ranker, RerankRequest as FlashRankRerankRequest
        _flashrank_model = Ranker()
        logger.info(f"FlashRank loaded in {time.time()-t0:.1f}s")
    except Exception as e:
        logger.error(f"FlashRank load failed: {e}")


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
    return {
        "status": "online",
        "colbert": "active" if _colbert_model else "unavailable",
        "bge": "active" if _bge_model else "unavailable",
        "flashrank": "active" if _flashrank_model else "unavailable",
    }


@app.post("/embed/bge", response_model=EmbedResponse)
def embed_bge(req: EmbedRequest):
    if _bge_model is None:
        return EmbedResponse(embeddings=[])
    embeddings = _bge_model.encode(req.texts, normalize_embeddings=True).tolist()
    return EmbedResponse(embeddings=embeddings)


@app.post("/rerank/colbert", response_model=RerankResponse)
def rerank_colbert(req: RerankRequest):
    if _colbert_model is None:
        return RerankResponse(results=[])
    import numpy as np
    query_emb = _colbert_model.encode(req.query)
    doc_embs = _colbert_model.encode(req.documents)
    scores = []
    for i, doc_emb in enumerate(doc_embs):
        if hasattr(query_emb, "shape") and len(query_emb.shape) == 2:
            sim = float(np.mean(np.max(np.dot(query_emb, doc_emb.T), axis=1)))
        else:
            sim = float(np.dot(query_emb, doc_emb) / (np.linalg.norm(query_emb) * np.linalg.norm(doc_emb) + 1e-8))
        scores.append((i, sim, req.documents[i]))
    scores.sort(key=lambda x: x[1], reverse=True)
    results = [RerankResult(index=s[0], score=s[1], text=s[2]) for s in scores[:req.top_k]]
    return RerankResponse(results=results)


@app.post("/rerank/flashrank", response_model=RerankResponse)
def rerank_flashrank(req: RerankRequest):
    if _flashrank_model is None:
        return RerankResponse(results=[])
    passages = [{"id": i, "text": doc} for i, doc in enumerate(req.documents)]
    flash_req = FlashRankRerankRequest(query=req.query, passages=passages)
    reranked = _flashrank_model.rerank(flash_req)
    results = []
    for r in reranked[:req.top_k]:
        results.append(RerankResult(
            index=r.get("id", 0),
            score=r.get("score", 0.0),
            text=r.get("text", "")
        ))
    return RerankResponse(results=results)


@app.on_event("startup")
def startup():
    _load_models()


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8895)
