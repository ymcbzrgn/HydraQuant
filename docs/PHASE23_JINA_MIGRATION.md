# Phase 23: Dual API Embedding + Jina Reranker Migration

## Amaç

ColBERT + BGE local modellerini API'lere taşı, FlashRank local olarak KORU.
3.3GB RAM kurtarma, OOM sorununu kökten bitirme, kaliteyi ARTIRMA.

**Kritik bilgi:** Jina free tier 10M token ONE-TIME (aylık yenilenmiyor). ~43 günde biter.
Sonrası pay-as-you-go: toplam ~$0.12/ay. Para önemli değil, KALİTE önemli.

---

## Mimari: C+D (Dual API + Local Fallback)

```
MEVCUT (3.5GB RAM, 62 OOM/48h):
  BGE (local 1.3GB) + Gemini (API) → RRF → ColBERT (local 2GB) + FlashRank (local 200MB)
  
YENİ (200MB RAM, 0 OOM, DAHA İYİ KALİTE):
  Jina Embed (API) + Gemini Embed (API) → RRF → Jina Rerank (API) + FlashRank (local)
                                                              │
                                                              ▼ (tüm API'ler çökerse)
                                                      ColBERT (arşivden ayağa kalk)
```

### Neden Bu Mimari En İyisi

| Özellik | Eski | Yeni |
|---------|------|------|
| Embedding | BGE (local 1.3GB) + Gemini (API) | **Jina (API) + Gemini (API)** |
| Reranking primary | ColBERT (local 2GB) | **Jina Reranker v2 (API)** |
| Reranking secondary | FlashRank (local 200MB) | **FlashRank (local 200MB) — AYNI** |
| Emergency fallback | Yok | **ColBERT (arşivden, otomatik start)** |
| RAM | 3.5GB | **200MB** (FlashRank only) |
| RAM kazancı | — | **3.3GB** |
| Maliyet | $0 (ama OOM) | **$0.12/ay (ama 0 OOM)** |
| Kalite | İyi | **Daha iyi (iki farklı API embedding = çeşitlilik)** |
| SPOF | Model server çökerse = kör | **Hiçbir SPOF yok** |

### 5 Katmanlı Redundancy Zinciri

```
Katman 1: Jina Embed + Gemini Embed → RRF → Jina Rerank       ← NORMAL
Katman 2: Gemini Embed only → Jina Rerank                      ← Jina embed DOWN
Katman 3: Jina Embed only → Jina Rerank                        ← Gemini DOWN
Katman 4: (Herhangi embed) → FlashRank local                   ← Jina Rerank DOWN
Katman 5: BM25 keyword → FlashRank local                       ← Tüm embedding API DOWN
Emergency: ColBERT local (arşivden) otomatik start              ← TÜM API'ler uzun süre DOWN
Absolute: Evidence Engine (embedding gerektirmez, LLM-free)     ← HER ŞEY DOWN
```

---

## Maliyet Detayı

```
Jina embedding:   ~40K token/gün × 30 = 1.2M token/ay × $0.018/1M = $0.02/ay
Jina reranking:   ~192K token/gün × 30 = 5.8M token/ay × $0.018/1M = $0.10/ay
Gemini embedding: $0.00 (free, 10 API key)
FlashRank:        $0.00 (local)
TOPLAM:           $0.12/ay

Free tier (10M one-time): ~43 gün (1.4 ay)
Yoğun kullanım (2x): ~22 gün, sonra $0.25/ay
```

---

## Yapılacaklar (Detaylı)

### 1. Jina Reranker API Entegrasyonu
**Dosya:** `colbert_reranker.py` → güncelle (Jina primary, ColBERT emergency)

```python
class JinaReranker:
    """Jina Reranker v2 API — ColBERT kalitesinde, 0 RAM."""
    API_URL = "https://api.jina.ai/v1/rerank"
    
    def __init__(self):
        self.api_key = os.environ.get("JINA_API_KEY")
        self._jina_available = True
        self._colbert_process = None  # Emergency: lazy start
    
    def rerank(self, query: str, documents: list, top_k: int = 5) -> list:
        # Try 1: Jina API
        if self._jina_available and self.api_key:
            try:
                response = httpx.post(self.API_URL, json={
                    "model": "jina-reranker-v2-base-multilingual",
                    "query": query,
                    "documents": documents,
                    "top_n": top_k
                }, headers={"Authorization": f"Bearer {self.api_key}"}, timeout=10)
                if response.status_code == 200:
                    return self._parse_jina_response(response.json())
                if response.status_code == 402:  # Payment required
                    logger.warning("[JinaReranker] Free tier exhausted, switching to FlashRank")
                    self._jina_available = False
            except Exception as e:
                logger.warning(f"[JinaReranker] API error: {e}")
        
        # Try 2: FlashRank local (always available, 200MB)
        return self._flashrank_rerank(query, documents, top_k)
    
    def _emergency_colbert_start(self):
        """TÜM API'ler uzun süre DOWN → ColBERT'i arşivden ayağa kaldır."""
        if self._colbert_process is None:
            logger.warning("[EMERGENCY] Starting archived ColBERT model server...")
            import subprocess
            self._colbert_process = subprocess.Popen(
                ["python3", "user_data/scripts/model_server_archived.py"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
```

**Model:** `jina-reranker-v2-base-multilingual`
- Dimension sorunu YOK (reranker raw text alır, skor döner)
- MTEB reranking leaderboard'da ColBERT v2 seviyesi
- Multilingual (TR haberler dahil)
- 8K token context window

**DUAL RERANKING PIPELINE (Jina + FlashRank AKTİF BİRLİKTE):**

FlashRank sadece fallback DEĞİL — aktif olarak Jina ile birlikte çalışır:

```
Candidate dokümanlar (30 adet, RRF sonrası)
    │
    ▼
┌──────────────────────────────────────────────┐
│ Adım 1: JINA RERANK (API, SOTA)             │
│ 30 doküman → Jina skorları → top 15 seç     │
└──────────────────┬───────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────┐
│ Adım 2: FLASHRANK RERANK (local, 200MB)      │
│ Aynı 30 dokümanı bağımsız olarak rerank et   │
│ → FlashRank skorları → top 15 seç            │
└──────────────────┬───────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────┐
│ Adım 3: ENSEMBLE FUSION                      │
│ final_score = 0.6 × jina + 0.4 × flashrank  │
│                                               │
│ İki model hemfikir → YÜKSEK güven            │
│ İki model uyuşmaz → DÜŞÜK güven              │
│   → Neural Organism uncertainty sinyali       │
│                                               │
│ Top 5 seç → RAG pipeline'a gönder            │
└──────────────────────────────────────────────┘
```

**Neden sadece fallback değil, aktif dual reranking:**
1. İki FARKLI reranker aynı dokümanları bağımsız değerlendirir → tek model'in kaçırdığını diğeri yakalar
2. Uyuşmazlık = belirsizlik sinyali → interoception.data_completeness ve serotonin'e feedback
3. FlashRank zaten 200MB ile RAM'de → çağırmamak israf
4. Jina DOWN olduğunda FlashRank soğuk başlangıç yapmaz — zaten ISINMIŞ ve aktif

```python
class DualReranker:
    """Jina + FlashRank parallel reranking with ensemble scoring."""
    
    def rerank(self, query: str, documents: list, top_k: int = 5) -> list:
        jina_scores = {}
        flash_scores = {}
        
        # Paralel: iki reranker aynı dokümanları değerlendirir
        # Jina (API)
        try:
            jina_results = self.jina.rerank(query, documents, top_k=15)
            for doc_id, score in jina_results:
                jina_scores[doc_id] = score
        except Exception:
            pass  # Jina DOWN → FlashRank tek başına
        
        # FlashRank (local, her zaman çalışır)
        flash_results = self.flashrank.rerank(query, documents, top_k=15)
        for doc_id, score in flash_results:
            flash_scores[doc_id] = score
        
        # Ensemble: iki modelin skorlarını birleştir
        all_docs = set(jina_scores.keys()) | set(flash_scores.keys())
        ensemble = []
        agreement_scores = []
        
        for doc_id in all_docs:
            j = jina_scores.get(doc_id, 0)
            f = flash_scores.get(doc_id, 0)
            
            if jina_scores:  # Jina available
                final = 0.6 * j + 0.4 * f
                # Uyuşmazlık ölçümü: iki model ne kadar farklı düşünüyor?
                disagreement = abs(j - f)
                agreement_scores.append(1.0 - disagreement)
            else:
                final = f  # Jina DOWN → FlashRank only
            
            ensemble.append((doc_id, final))
        
        # Neural Organism'a reranker agreement sinyali gönder
        if agreement_scores:
            avg_agreement = sum(agreement_scores) / len(agreement_scores)
            # Düşük agreement → düşük serotonin → organizma temkinli olur
            self._report_agreement_to_organism(avg_agreement)
        
        ensemble.sort(key=lambda x: x[1], reverse=True)
        return ensemble[:top_k]
    
    def _report_agreement_to_organism(self, agreement: float):
        """Reranker uyuşmazlığını Neural Organism'a bildir."""
        try:
            from neural_organism import get_organism
            org = get_organism()
            # Agreement düşükse → info_quality düşer → serotonin düşer
            # Bu otomatik olarak confidence'ı daraltır
            org.interoception.sensors["data_completeness"] *= (0.5 + 0.5 * agreement)
        except Exception:
            pass
```

### 2. Jina Embedding API Entegrasyonu
**Dosya:** `rag_embedding.py` → BGE yerine Jina embedding ekle

```python
class JinaEmbedder:
    """Jina Embeddings v3 API — BGE yerine, 0 RAM."""
    API_URL = "https://api.jina.ai/v1/embeddings"
    
    def embed(self, texts: list, task: str = "retrieval.passage") -> list:
        response = httpx.post(self.API_URL, json={
            "model": "jina-embeddings-v3",
            "input": texts,
            "task": task,  # retrieval.passage, retrieval.query, classification
            "dimensions": 768,  # Gemini ile aynı dim → RRF uyumlu
        }, headers={"Authorization": f"Bearer {self.api_key}"}, timeout=15)
        return [item["embedding"] for item in response.json()["data"]]
```

**Önemli:** `dimensions: 768` → Gemini embedding ile AYNI boyut. ChromaDB'deki mevcut vektörlerle uyumlu. Matryoshka truncation ile 768'e düşürülür.

**Dual embedding flow:**
```python
# hybrid_retriever.py güncellemesi
def _dense_search(self, query, collection):
    results = []
    
    # Embedding 1: Gemini (free, primary)
    try:
        gemini_vec = self.gemini_embedder.embed(query)
        gemini_results = collection.query(query_embeddings=[gemini_vec], n_results=30)
        results.append(("gemini", gemini_results))
    except Exception:
        pass
    
    # Embedding 2: Jina (API, secondary)
    try:
        jina_vec = self.jina_embedder.embed([query], task="retrieval.query")[0]
        jina_results = collection.query(query_embeddings=[jina_vec], n_results=30)
        results.append(("jina", jina_results))
    except Exception:
        pass
    
    # RRF fusion: iki farklı perspektif birleşir
    if len(results) == 2:
        return self.reciprocal_rank_fusion([results[0][1], results[1][1]])
    elif len(results) == 1:
        return results[0][1]  # Tek kaynak
    else:
        return []  # İkisi de çöktü → BM25'e düş
```

### 3. Model Server Arşivleme
**Dosya:** `model_server.py` → `model_server_archived.py` olarak rename

```bash
mv user_data/scripts/model_server.py user_data/scripts/model_server_archived.py
```

- `freqtrade-models.service` → varsayılan DISABLED
- ColBERT + BGE sadece emergency fallback olarak kalır
- FlashRank model_server'dan ÇIKARTILIR → hybrid_retriever.py içine taşınır (in-process)
- Emergency: tüm API'ler 5+ dakika DOWN → model_server_archived.py otomatik start

### 4. FlashRank In-Process Taşıma
**Dosya:** `hybrid_retriever.py` → FlashRank'ı direkt import et

```python
# Eski: HTTP ile model_server'a istek at
# Yeni: Direkt Python import (200MB, in-process)
class InProcessFlashRank:
    def __init__(self):
        from flashrank import Ranker
        self.ranker = Ranker(model_name="ms-marco-MiniLM-L-6-v2")
    
    def rerank(self, query, documents, top_k=5):
        results = self.ranker.rerank(query, documents)
        return sorted(results, key=lambda x: x["score"], reverse=True)[:top_k]
```

**Avantaj:** HTTP overhead yok, model_server process'i yok, direkt Python çağrısı.

### 5. ChromaDB Dual Collection Stratejisi

```python
# Mevcut: tek collection, tek embedding (768-dim Gemini)
# Yeni: tek collection, iki embedding kaynağı (ikisi de 768-dim)

# Jina embedding = 768-dim (Matryoshka truncation ile)
# Gemini embedding = 768-dim (native)
# AYNI ChromaDB collection'a yazılabilir → storage artmaz

# Metadata'da hangi embedder kullanıldığını sakla
collection.add(
    documents=[text],
    embeddings=[vector],
    metadatas=[{"embedder": "jina_v3", "dim": 768}]  # veya "gemini_001"
)
```

### 6. Environment Variables
```bash
# .env'ye ekle
JINA_API_KEY="jina_xxxxxxxxxxxxxxxx"

# ai_config.py'ye ekle
JINA_API_KEY = os.environ.get("JINA_API_KEY")
JINA_EMBED_MODEL = "jina-embeddings-v3"
JINA_RERANK_MODEL = "jina-reranker-v2-base-multilingual"
JINA_EMBED_DIM = 768  # Gemini ile uyumlu
```

### 7. rag_graph.py Model Server Health Check Güncelleme
```python
# Eski: model_server sağlık kontrolü (HTTP /health)
# Yeni: Jina + Gemini API sağlık kontrolü
def _check_embedding_health():
    jina_ok = _test_jina_api()
    gemini_ok = _test_gemini_api()
    flashrank_ok = True  # Always available (in-process)
    return {
        "jina": jina_ok,
        "gemini": gemini_ok, 
        "flashrank": flashrank_ok,
        "embedding_sources": sum([jina_ok, gemini_ok]),
        "reranking_available": jina_ok or flashrank_ok,
    }
```

---

## Etkilenen Dosyalar

| Dosya | Değişiklik | Öncelik |
|-------|-----------|---------|
| `colbert_reranker.py` | Jina Reranker API + FlashRank fallback + ColBERT emergency | P0 |
| `hybrid_retriever.py` | Dual API embedding (Jina+Gemini) + RRF + InProcessFlashRank | P0 |
| `rag_embedding.py` | JinaEmbedder class ekle, BGE kaldır | P0 |
| `model_server.py` | → `model_server_archived.py` rename | P0 |
| `ai_config.py` | JINA_API_KEY, JINA_EMBED_MODEL, JINA_RERANK_MODEL | P0 |
| `.env` | JINA_API_KEY ekle | P0 |
| `rag_graph.py` | Model server health → API health check | P1 |
| `scheduler.py` | model_server health job güncelle | P1 |
| `freqtrade-models.service` | disable (opsiyonel tutulur) | P1 |
| `smoke_test.py` | Jina API connectivity test ekle | P2 |
| `deployment_check.py` | Jina key validation | P2 |

---

## Doğrulama

1. `freqtrade-models` servisi OLMADAN tüm RAG pipeline çalışmalı
2. Dual embedding: Jina + Gemini → RRF → sonuçlar DAHA İYİ olmalı
3. Reranking: Jina Reranker → FlashRank fallback zinciri çalışmalı
4. RAM: model_server process'i yok → ~3.3GB freed
5. OOM: 0 kill (FlashRank 200MB tek local model)
6. Emergency: Jina + Gemini DOWN simüle et → FlashRank devreye girmeli
7. Extended emergency: tüm API 5 dk DOWN → ColBERT archived model start
8. ChromaDB: mevcut 768-dim vektörler Jina 768-dim ile uyumlu
9. Cost: 10M free token ~43 günde tükenmeli (monitoring ekle)

## Token Budget Monitoring

```python
# scheduler.py'ye ekle: günlük Jina token kullanım raporu
def _check_jina_budget():
    """Günlük Jina API kullanım kontrolü."""
    # Jina API usage endpoint
    response = httpx.get("https://api.jina.ai/v1/usage",
                         headers={"Authorization": f"Bearer {JINA_API_KEY}"})
    usage = response.json()
    remaining = usage.get("remaining_tokens", 0)
    
    if remaining < 1_000_000:  # 1M token kaldıysa uyar
        send_telegram(f"⚠️ Jina token düşük: {remaining:,} kaldı. Top-up gerekebilir.")
    
    logger.info(f"[Jina Budget] Remaining: {remaining:,} tokens")
```

---

## Timeline

| Gün | İş | Süre |
|-----|-----|------|
| 1 | Jina API key al, .env + ai_config | 10 dk |
| 1 | JinaEmbedder class yaz, rag_embedding.py'ye ekle | 30 dk |
| 1 | JinaReranker class yaz, colbert_reranker.py güncelle | 30 dk |
| 1 | hybrid_retriever.py dual embedding + RRF güncelle | 45 dk |
| 1 | InProcessFlashRank taşı (model_server'dan çıkart) | 20 dk |
| 2 | model_server.py → model_server_archived.py | 5 dk |
| 2 | rag_graph.py health check güncelle | 15 dk |
| 2 | Emergency ColBERT auto-start mekanizması | 20 dk |
| 2 | scheduler.py Jina budget monitoring | 10 dk |
| 2 | Smoke test + doğrulama | 30 dk |
| 2 | Deploy + freqtrade-models.service disable | 15 dk |
| | **TOPLAM** | **~4 saat** |

---

## Sonuç

- **RAM:** 3.5GB → 200MB (**3.3GB kurtarıldı**)
- **OOM:** 62/48h → **0** (FlashRank 200MB tek local model)
- **Kalite:** Daha iyi (dual embedding çeşitliliği + Jina reranker SOTA)
- **Maliyet:** $0.12/ay (free tier 43 gün, sonra pay-as-you-go)
- **Redundancy:** 5 katman + ColBERT emergency + Evidence Engine absolute fallback
- **Süre:** 2 gün (4 saat aktif çalışma)
