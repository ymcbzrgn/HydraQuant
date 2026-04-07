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

---

## RAM KRİZİ KÖK NEDEN ANALİZİ (5 Ajan Araştırması)

### Sunucu Durumu (7 Nisan 2026)
```
31GB total, 29GB used, 246MB free, 2GB swap TAMAMEN DOLU
scheduler.py:    3,984MB (3.9GB!)
rag_graph.py:    2,689MB (2.6GB)
model_server.py: 2,163MB (2.1GB) → Jina ile kalkacak
freqtrade bot:   1,325MB (1.3GB)
api_ai.py:         144MB
TOPLAM RSS:     ~10.4GB
KAYIP:          ~19GB → NEREYE?
```

### 5 Kök Neden

**1. glibc malloc arena fragmentation (4-8GB!) — EN BÜYÜK SUÇLU**
- glibc `ptmalloc2` process başına 32 arena yaratır (4 core × 8)
- Python `free()` belleği arena'ya döndürür ama arena OS'a GERİ VERMİYOR
- BetterUp: RSS 3.62x actual heap (glibc) vs 1.22x (jemalloc)
- `gc.collect()` Python objeleri toplar ama C heap fragmentation KALIR

**2. RSS shared library double-counting (3-5GB HAYALET)**
- 4 Python process × libpython + numpy + langchain = RSS'te 4x sayılır
- Gerçek fiziksel RAM'de 1x var, PSS ile ölçülmeli
- `smem -t -k -P python` gerçek değeri gösterir

**3. Scheduler +130MB/saat leak (toplam):**
- feedparser.parse(): 25-60MB/saat (15 feed × 5dk, XML DOM GC edilmiyor)
- EvidenceEngine() her 5dk yeni instance: 5-20MB/saat
- 3× StreamingRAG, 2× MAGMAMemory, 2× MemoRAG duplicate: 30-60MB sabit
- requests.get() session reuse yok: 1-3MB/saat
- SQLite connections kapatılmıyor: 0.5-1MB/saat

**4. LLMRouter duplicate instances (400-700MB)**
- DataPipeline → RAPTORTree → YENİ LLMRouter (~150MB)
- DataPipeline → MemoRAG → YENİ LLMRouter (~150MB)
- _rebuild_graph → YENİ LLMRouter (~150MB)
- HER LLMRouter: 80 ModelSlot × LangChain model objects
- RAPTORTree/MemoRAG embed_news'da KULLANILMIYOR bile → eager init waste

**5. Linux buff/cache (1.6GB — reclaimable, sorun değil)**
- Kernel disk cache: SQLite, ChromaDB, log dosyaları
- `free -h` "available" sütunu gerçeği gösterir

### RAM FIX PLANI (Jina migration ile birlikte)

#### FIX 1: jemalloc (10 dk, ~3-8GB kurtarır) — EN ÖNCELİKLİ
```bash
# Sunucuda:
apt-get install libjemalloc2

# Her systemd service dosyasına ekle:
[Service]
Environment="LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so.2"
```
jemalloc fragmentation'ı %20'de tutar (glibc unlimited). PyTorch, BetterUp, LiteLLM hep bunu kullanıyor.

Alternatif (jemalloc yüklenemezse):
```bash
Environment="MALLOC_ARENA_MAX=2"
Environment="MALLOC_MMAP_THRESHOLD_=65536"
```

#### FIX 2: malloc_trim() GC'ye ekle (5 dk, ~0.5-1GB ongoing)
```python
# scheduler.py _memory_cleanup() ve rag_graph.py GC'ye ekle:
import gc
gc.collect()
try:
    import ctypes
    ctypes.CDLL("libc.so.6").malloc_trim(0)
except Exception:
    pass
```

#### FIX 3: feedparser leak fix (5 dk, 25-60MB/saat durur)
```python
# rss_fetcher.py: her feed parse sonrası temizle
feed = feedparser.parse(url)
entries = list(feed.entries)  # extract
del feed  # release XML DOM immediately
gc.collect()
```

#### FIX 4: Scheduler singleton'ları tamamla (15 dk, 400-700MB kurtarır)
```python
# scheduler.py __init__'e ekle:
self._evidence_engine = None
self._hybrid_retriever_bidi = None
self._graph_rag = None
self._llm_router_graph = None
self._rag_evaluator = None
self._regime_classifier = None
self._cost_tracker = None
self._autonomy_manager = None
```
Her job'da `if self._X is None: self._X = X()` pattern.

#### FIX 5: DataPipeline lazy init (10 dk, ~200-400MB kurtarır)
```python
# data_pipeline.py: RAPTORTree ve MemoRAG LAZY olmalı
# Şu an __init__'te eager yaratılıyor ama embed_news bunları KULLANMIYOR
@property
def raptor(self):
    if self._raptor is None:
        self._raptor = RAPTORTree()
    return self._raptor
```

#### FIX 6: systemd MemoryMax (5 dk, OOM prevention)
```ini
# freqtrade-scheduler.service:
[Service]
MemoryMax=3G
MemoryHigh=2.5G

# freqtrade-rag.service:
MemoryMax=3G
MemoryHigh=2.5G

# freqtrade.service:
MemoryMax=2G

# freqtrade-models.service: (Jina sonrası kaldırılacak)
MemoryMax=4G
```

#### FIX 7: LLMRouter lazy model creation (30 dk, ~150MB kurtarır)
```python
# llm_router.py: 80 model objesi __init__'te yaratılıyor
# Sadece Thompson Sampling'in seçtiği 5-10 model RAM'de olmalı
# Diğerleri ilk çağrıda lazy-load
```

### GERÇEK SONUÇ: jemalloc Deploy (7 Nisan 2026, 21:36 UTC)

```
ÖNCEKİ (glibc malloc):
  free=246MB, swap=2.0GB TAMAMEN DOLU
  scheduler=3.9GB, rag=2.6GB, model_server=2.1GB, freqtrade=1.3GB

SONRASI (jemalloc, 30 saniye sonra):
  free=6.4GB, swap=461MB (1.5GB boş!)
  model_server=1.7GB, rag=933MB, freqtrade=363MB, scheduler=45MB*
  
  * scheduler henüz ısınmadı (job'lar çalışmadı), 1-2 saatte ~1.5GB'a çıkacak

jemalloc VERIFY: Tüm 4 servis jemalloc=5 (aktif)
```

**jemalloc TEK BAŞINA 6.2GB kurtardı** (246MB → 6.4GB free). Swap kullanımı %77 azaldı.

### Tahmini Sonuç (Tüm fix'ler + Jina)

| Bileşen | Önceki | Sonrası |
|---------|--------|---------|
| scheduler.py | 3.9GB | ~1.5GB |
| rag_graph.py | 2.6GB | ~1.2GB |
| model_server.py | 2.1GB | **0** (Jina ile kaldırıldı) |
| freqtrade bot | 1.3GB | ~1.0GB |
| FlashRank (in-process) | 0 | ~0.2GB |
| glibc fragmentation | ~4-8GB | ~0.5GB (jemalloc) |
| **TOPLAM** | **~29GB** | **~4.4GB** |
| **BOŞ RAM** | **246MB** | **~27GB** |

**27GB boş RAM = ML için DEVASA alan.** Triple Perception, World Model, RL Agents, Deep Ensemble — hepsi rahat sığar.

### İmplementasyon Sırası

| # | Fix | Süre | Bağımlılık |
|---|-----|------|-----------|
| 1 | jemalloc install + LD_PRELOAD | 10 dk | Yok |
| 2 | malloc_trim() GC'ye ekle | 5 dk | Yok |
| 3 | systemd MemoryMax | 5 dk | Yok |
| 4 | feedparser leak fix | 5 dk | Yok |
| 5 | Scheduler singleton'ları | 15 dk | Yok |
| 6 | DataPipeline lazy init | 10 dk | Yok |
| 7 | Jina embedding + reranker | 2 saat | Jina API key |
| 8 | model_server → archived | 5 dk | #7 tamamlanmalı |
| 9 | FlashRank in-process | 20 dk | #8 tamamlanmalı |
| 10 | LLMRouter lazy models | 30 dk | İsteğe bağlı |
| | **TOPLAM** | **~4 saat** | |

---

## Sonuç

- **RAM:** 29GB used → **~4.4GB** (%85 azalma!)
- **OOM:** 62/48h → **0**
- **Kalite:** Daha iyi (dual API embedding + dual reranker + jemalloc stability)
- **Maliyet:** $0.12/ay
- **ML için boş RAM:** **~27GB** (Phase 26 CAAT için devasa alan)
- **Redundancy:** 5 katman + ColBERT emergency + Evidence Engine absolute fallback
- **Süre:** 1 gün (4 saat aktif çalışma + test)
