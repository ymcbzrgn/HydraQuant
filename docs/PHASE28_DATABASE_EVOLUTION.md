# Phase 28: Database Evolution — From "Salla Patis" to Enterprise-Grade

## "Organizma artik dogru organlara sahip olacak — hafizasi, sinirleri ve refleksleri upgrade ediliyor."

> **Prereq:** Phase 26 Sprint 1 COMPLETE (Triple Perception deployed, canlida calisiyor)
> **Blocker:** Phase 26 Sprint 2 bu Phase'den SONRA baslar
> **Motivasyon:** 45 "database is locked" hatasi (9 dk firtina), ChromaDB RAM obezitesi, connection pooling SIFIR, 22 dosyada daginik DB erisimi

---

## Neden Bu Phase Gerekli?

### Mevcut Durum (Phase 26 Sprint 1 Sonrasi Analiz — 11 Nisan 2026)

| Metrik | Deger | Sorun |
|--------|-------|-------|
| SQLite tablolari | 54 | Calisiyor ama "database is locked" firtinalari |
| ChromaDB collection | 10+ | RAM obez (1-2GB HNSW in-memory), post-filter (kalite dusuyor) |
| sqlite3 import eden dosya | 22 | Her biri kendi baglantisini aciyor |
| Connection pooling | YOK | Her islem yeni connection, lock contention |
| busy_timeout tutarliligi | 5/22 dosya | Sadece 5 dosyada 30s timeout, geri kalan default |
| Baglanti pattern'i | 2 farkli | db.py vs direkt sqlite3.connect() — inconsistent |
| Duplicate table CREATE | 15 tablo | Ayni tablo hem db.py'de hem modulde tanimlaniyor |
| Graph DB | YOK | MAGMA, Causal, Agent network hep SQLite'ta duz tablo |
| Analytics | SQLite | OHLCV analiz, backtest — SQLite OLAP icin yavas |

### "database is locked" Firtinasi (10 Nisan 2026, 18:25-18:34)

9 dakikada 45 hata. Etkilenen moduller:
- `rag_embedding` (embedding cache yazimi)
- `semantic_cache` (cache yazimi + cleanup)
- `llm_cost_tracker` (LLM cagri loglama)
- `streaming_rag` (hot buffer flush)
- `hybrid_retriever` (FTS5 + Binary BGE yazimi)
- `data_pipeline` (embedding push)
- `system_monitor` (metrik kaydi)
- `scheduler` (sentiment hesaplama)

**Kok Neden:** 22 dosya ayni `ai_data.sqlite`'a connection pooling olmadan, farkli timeout'larla yaziyor.

---

## Arastirma Sureci

3 arastirma agenti, toplam **70 web arama**, 20+ aday incelendi:

### Elenen Adaylar ve Nedenler

| Aday | Neden Elendi |
|------|-------------|
| **SurrealDB** | BSL 1.1 lisans — GPLv3 ile embedded modda uyumsuz. Server modda <1ms localhost ama embedded istiyoruz |
| **KuzuDB** | OLDU — Apple Ekim 2025'te satin aldi, GitHub repo ARCHIVED |
| **Milvus Lite** | Kendi dokumantasyonu "production icin onerilmez" diyor |
| **LibSQL** | BEGIN CONCURRENT hala experimental, Python SDK experimental |
| **Qdrant embedded** | "Test senaryolari icin" diyor dokumantasyonu |
| **Weaviate embedded** | Subprocess spawn ediyor — gercek embedded degil |
| **EdgeDB** | Server gerektiriyor, embedded degil |
| **rqlite** | Distributed — tek sunucu icin overkill |
| **FalkorDB Lite** | SSPL lisans — GPL uyumsuz |
| **Memgraph** | BSL 1.1 lisans + server-only |
| **Neo4j** | AGPL + Commons Clause, Python embedded ABANDONED |
| **ArcadeDB** | JVM-based, 500MB+ RAM overhead |
| **CogDB** | Standart sorgu dili yok, 355 star |

### Kazanan Mimari

| Katman | Arac | Lisans | Rol | RAM |
|--------|------|--------|-----|-----|
| **Relational (OLTP)** | SQLite-WAL | Public Domain | 54 tablo, trade state, config, audit | ~200MB |
| **Vector Search** | LanceDB | Apache 2.0 | ChromaDB replacement, RAG embeddings, 10+ collection | ~500-800MB (disk-based, memory-mapped) |
| **Graph** | Grafeo | Apache 2.0 | Causal graph, MAGMA, agent networks, Cypher queries | ~136MB |
| **Analytics (OLAP)** | DuckDB | MIT | Backtest analiz, OHLCV, rolling Sharpe, Dream Engine data | ~200MB |
| **Experimental Vector** | Zvec (Alibaba) | Apache 2.0 | 1 collection'da deneme, gelecege yatirim, 2x LanceDB benchmark | ~100MB |

**Toplam tahmini RAM:** ~1.1-1.5GB (mevcut SQLite+ChromaDB ~2-3GB'dan DAHA AZ)

---

## Neden Bu Araclar?

### LanceDB — Vector Search Sahi
- **700M vektor production deployment** (AWS blog dogrulamis)
- **1B+ vektor** S3 uzerinde (AWS Architecture blog)
- Sub-millisecond latency NVMe'de
- **Pre-filter** metadata filtreleme (ChromaDB post-filter yapiyor — sonuc kalitesi duser)
- Disk-based Lance format: memory-mapped I/O, RAM'i sogurtmuyor
- Lance format v2.2: Parquet'ten **2000x hizli** random access, **%50 daha az** depolama
- Kurucu: AQR Capital + Barclays eski quant'i
- DuckDB entegrasyonu (SQL ile vector sorgulama)
- Versioned data (time travel — geri alin, karsilastirin)
- Kaynaklar:
  - https://sprytnyk.dev/posts/running-lancedb-in-production/
  - https://aws.amazon.com/blogs/architecture/a-scalable-elastic-database-and-search-solution-for-1b-vectors-built-on-lancedb-and-amazon-s3/
  - https://www.lancedb.com/blog/lance-format-v2-2-benchmarks-half-the-storage-none-of-the-slowdown

### Grafeo — Graph DB Devi
- **Rust-native**, PyO3 ile true in-process
- **4 sorgu dili:** Cypher + GQL (ISO standard) + SPARQL + Gremlin
- SNB Interactive benchmark: KuzuDB'den **1.8x hizli** (2,904ms vs 5,333ms)
- Graph analytics: **0.4ms**
- HNSW vector search DAHIL (secondary vector capability)
- **136MB RAM** (rakipler 200-500MB)
- Disk persistence + WAL checkpointing
- KuzuDB olunce ortaya cikan boslugu dolduruyor
- Kaynak: https://github.com/GrafeoDB/grafeo

### DuckDB — Analytics Motoru
- **25K+ GitHub star**, MIT lisans
- SQLite'tan **10-100x hizli** OLAP sorgularda
- Pandas/Polars/Arrow zero-copy entegrasyonu
- `ATTACH 'ai_data.sqlite' AS ai` — mevcut SQLite verilerini dogrudan sorgula
- Columnar storage: aggregation, window function, GROUP BY icin ideal
- VSS extension (experimental vector search — Zvec'e alternatif)
- Kaynak: https://duckdb.org/

### Zvec (Alibaba) — Gelecege Yatirim
- Alibaba'nin Proxima motoru uzerinde
- **8000+ QPS** Cohere 10M benchmark'inda (LanceDB'nin ~2 kati)
- **0.5ms** filtered query latency
- RabitQ quantization + SIMD auto-dispatch
- v0.3.0 — cok genc, 1 collection'da experimental olarak dene
- 6-12 ay icinde LanceDB'yi gecebilir
- Kaynak: https://github.com/alibaba/zvec

---

## Mimari Diyagram

```
  ┌────────────���──────────────���─────────────────────────────────────────┐
  │                    HydraQuant / CAAT Organism                       │
  │                                                                     │
  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐           │
  │  │ Evidence  │  │ Triple   │  │ Agent    │  │ Neural   │  ...      │
  │  │ Engine   │  │ Percep.  │  │ Pool     │  │ Organism │           │
  │  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘           │
  │       │              │              │              │                 │
  │       ▼              ▼              ▼              ▼                 │
  │  ╔═══════════════════════════���════════════════���═════════════════╗    │
  │  ║              UNIFIED DB LAYER (db.py v2)                    ║    │
  │  ║  Connection Pool + Retry + Consistent Timeout + Singleton   ║    │
  │  ╚══════╦═══════════╦═══════════╦═══════════╦═════════════════╝    │
  │         │           │           │           │                       │
  │    ┌────▼────┐ ┌────▼────┐ ┌────▼────┐ ┌────▼────┐  ┌────────┐   │
  │    │ SQLite  │ │ LanceDB │ │ Grafeo  │ │ DuckDB  │  │ Zvec   │   │
  │    │  -WAL   │ │         │ │         │ │         │  │ (exp.) │   │
  │    │         │ │ Vector  │ │ Graph   │ │Analytics│  │ 1 coll │   │
  │    │54 tablo │ │10+ coll │ │ Causal  │ │ OHLCV   │  │        │   │
  │    │ OLTP    │ │ RAG     │ │ MAGMA   │ │Backtest │  │        │   │
  │    │ State   │ │Embedding│ │ Agent   │ │ Sharpe  │  │        │   │
  │    └─────────┘ └─────────┘ └─────────┘ └─────────┘  └────────┘   │
  │     Public      Apache 2.0   Apache 2.0    MIT       Apache 2.0   │
  │     Domain      ~800MB       ~136MB        ~200MB     ~100MB      │
  └────────────────────────────────────────────────────────────���────────┘
```

---

---

## PHASE 26 EKSIK ANALIZI (3 Explorer-God, 157 Tool, Tam Tarama)

### KRITIK BULGULAR — Sprint 1 Borclari

| # | Gap | Ciddiyet | Detay |
|---|-----|----------|-------|
| **G1** | CatBoost training pipeline SIFIR | KRITIK | catboost_model.py yok, training script yok, model dosyasi yok. triple_perception.py her cagrildiginda "No pre-trained CatBoost model yet" logu basiyor, CatBoost branch dead code |
| **G2** | sizing_multiplier ASLA kullanilmiyor | KRITIK | OOD+CQR+DeepEnsemble+Chronos pipeline sizing_multiplier hesapliyor (AIFreqtradeSizer L836,L851) ama custom_stake_amount() (L1096-1200) HICBIR ZAMAN okumuyor. Tum uncertainty-based sizing isi COPE ATILIYOR |
| **G3** | neural_organism.py pheromone = SIFIR | YUKSEK | 2186 satirlik organizma pheromone_field'dan habersiz. Zero import, zero read, zero write |
| **G4** | evidence_engine.py pheromone = SIFIR | YUKSEK | Planlanan kopru: perception deposits → EE reads → rag_graph benefits. EE'de sifir pheromone kodu |
| **G5** | 9/10 Phase 26 modulu test = SIFIR | YUKSEK | 3170 satir kod, sadece chart_features icin 23 test. Geri kalan 9 modul SIFIR test |
| **G6** | Pheromone deposit'leri hic okunmuyor | ORTA | triple_perception SIGNAL_PREDICTION + SIGNAL_UNCERTAINTY deposit ediyor, KIMSE okumuyor. Decay edip temizleniyor |
| **G7** | Scheduler retraining job'lari yok | ORTA | CatBoost/OOD/Conformal/DeepEnsemble periodic refit icin scheduler job'i SIFIR |
| **G8** | requirements-phase26.txt install'a baglanmamis | DUSUK | Dosya var, icerik dogru, ama Dockerfile/setup'a referans SIFIR |

### Sprint 2 Altyapi Ihtiyaclari (Yeni Tablolar)

| Tablo | Amac | Kullanan Task | DB Hedefi |
|-------|------|---------------|-----------|
| `causal_discoveries` | Tigramite PCMCI+ cikti | 6A | Grafeo |
| `counterfactual_results` | DoWhy intervention sonuc | 6B | SQLite/DuckDB |
| `rl_replay_buffer` | State/action/reward | 7A-7D | DuckDB |
| `rl_checkpoints` | Model metadata | 7B-7D | SQLite |
| `world_model_states` | Sirali state snapshot'lari | 8C, 8D | DuckDB |
| `world_model_rollouts` | Hayal kurma sonuclari | 8C, 8D | DuckDB |
| `dream_scenarios` | Ruya senaryolari | 8D | DuckDB |
| `organ_performance_history` | Organ metrik zaman serisi | 9A | DuckDB |
| `hormone_history` | Hormon snapshot gecmisi | 9A | DuckDB |
| `self_model_profile` | Metacognition yetkinlik haritasi | 9A | SQLite |
| LanceDB `rl_relevance` fields | RL feedback on RAG docs | 10B | LanceDB |

### Sprint 2 Kapsam Disi (Sprint 3+ veya gelecek phase)

| Oge | Neden Deferred |
|-----|---------------|
| Process 12: Evolutionary Architecture Search | Priority #21, "Cok zor" |
| Process 14: Market Microstructure Intelligence | Priority #16, 6 subsection, buyuk is |
| Process 15: Cerebellum 24-slot | Priority #22, arsitektur section bile yok |
| Novel #8: Hormonal Market Making | Process 14'e bagimli |
| Constitution enforcement runtime | YAML tanimli ama runtime kodu yok |
| Latency Tiers with_latency_guard() | Tanimli ama sarma kodu yok |
| Ablation League Table | Haftalik ablation test protokolu |
| Model Risk Engine | ModelRiskEngine class tanimli ama sprint'e atanmamis |
| Post-Trade Court | PostTradeCourt class tanimli ama sprint'e atanmamis |
| Decision Contract | JSON schema tanimli ama sprint'e atanmamis |
| Canli Vizualizasyon (10 Vue comp) | Devasa UI isi |
| ATCB Benchmark Suite | 10-senaryo benchmark |
| Phi (IIT) Bilinc Metrigi | Teorik |
| Data Pipeline Sim2Real | Domain randomization |
| 13 Autonomous Lifecycle Layer detay | Kismen Sprint 2 ile overlap ama ayri dekompoze edilmemis |

---

## Sprint Plani (GUNCELLENMIS — Phase 26 Tamiri Dahil)

### Sprint 28.0: Altyapi + Phase 26 Temel Tamir (2-3 gun)

| # | Task | Dosya | Aciklama |
|---|------|-------|----------|
| 0A | `db.py` v2 — Connection Pool + Singleton | `db.py` | Thread-safe connection pool, tutarli 30s busy_timeout, retry logic |
| 0B | Tum 22 dosyada direkt `sqlite3.connect()` → `db.get_connection()` | 22 dosya | Merkezi baglanti yonetimi. "database is locked" KOKTEN cozulur |
| 0C | Duplicate CREATE TABLE temizligi | `db.py` + 15 modul | Tablo tanimlari SADECE db.py'de. Moduller create etmesin |
| 0D | `requirements-phase28.txt` | Yeni dosya | `lancedb`, `grafeo`, `duckdb`, `zvec` + versiyonlar |
| 0E | **requirements-phase26.txt → install akisina bagla** | Dockerfile + setup | Mevcut dosya var ama hicbir yere referans yok. Dockerfile.ai'ye `-r requirements-phase26.txt` ekle |
| 0F | **Sprint 2 yeni tablolari db.py'ye ekle** | `db.py` | causal_discoveries, rl_replay_buffer, rl_checkpoints, world_model_states, world_model_rollouts, dream_scenarios, organ_performance_history, hormone_history, self_model_profile, counterfactual_results |
| 0G | **sizing_multiplier → custom_stake_amount() BAGLANTISI** | `AIFreqtradeSizer.py` | KRITIK: L1096-1200'de sizing_multiplier okunmali. Dual-Axis calibration sonucu trade size'i etkilemeli |
| 0H | **Phase 26 modul testleri** | `tests/test_phase26_*.py` | 9 moduldaki SIFIR testi duzelt: triple_perception, ttm, chronos, ood, conformal, deep_ensemble, dual_axis, pheromone, interoception |

### Sprint 28.1: ChromaDB → LanceDB Gocu + RAG Entegrasyonu (2-3 gun)

| # | Task | Dosya | Aciklama |
|---|------|-------|----------|
| 1A | `lance_store.py` — LanceDB wrapper | Yeni dosya | `create_table()`, `add()`, `search()`, `delete()`. ChromaDB API benzeri ama LanceDB backend |
| 1B | Migration script: Chroma → Lance | Yeni script | 10+ collection'i veri kaybetmeden tasi. Embedding model: Gemini + Jina. ID parity test |
| 1C | `hybrid_retriever.py` → LanceDB | Mevcut dosya | `crypto_news`, `crypto_news_bge` collection'lari Lance'a |
| 1D | `gam_rag.py` → LanceDB | Mevcut dosya | GAM-RAG collection Lance'a |
| 1E | `rag_setup.py` → LanceDB | Mevcut dosya | 7 collection Lance'a |
| 1F | `ai_config.py` → LanceDB singleton | Mevcut dosya | `get_chroma_client()` → `get_lance_db()` |
| 1G | ChromaDB dependency kaldir | `requirements*.txt` | `chromadb` pip'ten cikar, `lancedb` ekle |
| 1H | Smoke test + shadow karsilastirma | Test | Ayni sorgu ChromaDB ve LanceDB'de calistir, sonuclari karsilastir |
| 1I | **LanceDB'ye rl_relevance alanlari ekle** | `lance_store.py` | Sprint 2 Trinity (10B) icin: rl_relevance, rl_update_count, avg_trade_pnl alanlari her collection'a |
| 1J | **rag_graph.py Phase 26 entegrasyonu** | `rag_graph.py` | SADECE comment olan satirlari (L1742-1744) gercek koda cevir. Evidence Engine pheromone okusin |
| 1K | **evidence_engine.py pheromone okuma** | `evidence_engine.py` | Pheromone field'dan SIGNAL_PREDICTION, SIGNAL_UNCERTAINTY, SIGNAL_HEALTH oku, sinyal kalitesine kat |

### Sprint 28.2: Grafeo Graph + Phase 26 Organism Entegrasyonu (2-3 gun)

| # | Task | Dosya | Aciklama |
|---|------|-------|----------|
| 2A | `graph_store.py` — Grafeo wrapper | Yeni dosya | Node/Edge CRUD, Cypher query, shortest_path, multi-hop traversal |
| 2B | MAGMA Graph → Grafeo | `magma_memory.py` | `magma_edges` SQLite tablosu → Grafeo graph nodes/edges (4 graph_type korunacak: semantic, temporal, causal, entity) |
| 2C | Agent Network → Grafeo | `agent_pool.py` | `agent_memory`, `agent_performance` → Grafeo ile agent iliskiler |
| 2D | Cross-pair Intel → Grafeo | `cross_pair_intel.py` | Pair korelasyonlari, funding rate iliskileri → graph edges |
| 2E | Sprint 2 hazirlik: Causal Graph sema | `graph_store.py` | Tigramite PCMCI+ icin: source_var, target_var, causal_strength, time_lag, p_value, regime |
| 2F | Sprint 2 hazirlik: RL Agent sema | `graph_store.py` | 5 Organ Agent (Sizing, Confidence, Defense, Timing, Memory) icin node/edge semasi |
| 2G | Sprint 2 hazirlik: GNN sema | `graph_store.py` | PyTorch Geometric uyumlu export: node feature matrix + COO edge index + edge attributes |
| 2H | **neural_organism.py ↔ pheromone_field entegrasyonu** | `neural_organism.py` | KRITIK: Organizma pheromone okusin (SIGNAL_PREDICTION → sizing kararlari, SIGNAL_UNCERTAINTY → defansif mod, SIGNAL_HEALTH → hormon modulasyonu) ve yazsin (HORMONE_STATE, FEAR_LEVEL) |
| 2I | **Pheromone consumer'lari baglama** | Birden fazla dosya | triple_perception deposit'lerini okuyan consumer zinciri: pheromone → evidence_engine → neural_organism → sizing |

### Sprint 28.3: DuckDB Analytics + Zvec + CatBoost Altyapi (2-3 gun)

| # | Task | Dosya | Aciklama |
|---|------|-------|----------|
| 3A | `analytics_engine.py` — DuckDB wrapper | Yeni dosya | SQLite ATTACH, OHLCV analiz, rolling Sharpe, window functions |
| 3B | Backtest analiz pipeline | Mevcut + yeni | `pattern_stat_store.py` agir sorgulari DuckDB'ye yonlendir |
| 3C | Zvec experimental setup | Yeni dosya | 1 collection (ornegin `crypto_news`), LanceDB ile paralel calistir, sonuclari karsilastir |
| 3D | Benchmark: Lance vs Zvec vs eski Chroma | Script | Ayni 1000 sorguyu 3 sistemde calistir, latency/recall karsilastir |
| 3E | **CatBoost training data pipeline** | Yeni dosya | OHLCV + chart_features + TTM embeddings + EE sub-scores → labeled training CSV. Walk-forward temporal split (no future leak) |
| 3F | **CatBoost training script** | Yeni dosya | CatBoostClassifier().fit(), feature importance log, AUC/calibration, model → `user_data/models/catboost_signal_v1.cbm` |
| 3G | **CatBoost scheduler job** | `scheduler.py` | Haftalik retraining job. Ayrica OOD refit, Conformal recalibration, Deep Ensemble refit job'lari |
| 3H | **Sprint 2 DuckDB tablolari hazirla** | `analytics_engine.py` | rl_replay_buffer, world_model_states, world_model_rollouts, dream_scenarios, organ_performance_history, hormone_history icin DuckDB semalari |
| 3I | **user_data/models/rl/ dizini olustur** | Filesystem | Sprint 2 model checkpoint'lari icin: iql_*.pt, sac_*.pt, hrl_meta_*.pt, world_model_*.pt |

### Sprint 28.4: Entegrasyon + Test + Deploy (2-3 gun)

| # | Task | Dosya | Aciklama |
|---|------|-------|----------|
| 4A | Full system smoke test | Tum moduller | 54 tablo + 10 Lance collection + Grafeo graph + DuckDB analytics birlikte calisiyor mu? |
| 4B | RAM benchmark | Script | Mevcut (SQLite+Chroma) vs yeni (SQLite+Lance+Grafeo+DuckDB) RSS karsilastirmasi |
| 4C | **CatBoost ilk training calistir** | Sunucu | Backtest verisinden training data olustur → model train → smoke test → triple_perception artik CatBoost branch'i kullansin |
| 4D | **sizing_multiplier end-to-end test** | Test | custom_stake_amount() → sizing_multiplier okunuyor mu? OOD/CQR/Ensemble sonucu gercekten trade size'i etkiliyor mu? |
| 4E | **Pheromone end-to-end test** | Test | triple_perception deposit → evidence_engine read → neural_organism read → sizing impact zinciri calisiyor mu? |
| 4F | Sunucu deploy | Server | `pip install` + data migration + systemd restart |
| 4G | 24 saat monitoring | Canli | "database is locked" SIFIR, RAM azalmis, CatBoost model calisiyor, sizing_multiplier aktif |
| 4H | Chroma tamamen kaldir | Cleanup | `CHROMA_PERSIST_DIR` sil, Chroma kodu temizle |
| 4I | **Phase 26 Sprint 1 DONE checkmark** | Dokuman | Tum Sprint 1 borclari kapanmis mi? Changelog guncelle |

---

## Embedding Modelleri (Degismiyor)

Mevcut embedding yapisi korunacak (Phase 23 sonrasi):
- **Gemini** (gemini-embedding-001 / gemini-embedding-2-preview, 768d, API) — primary
- **Jina Embeddings v3** (jina-embeddings-v3, 768d, API) — secondary (Phase 23'te BGE'yi replace etti)
- **Jina Reranker v3** (jina-reranker-v3, API) — reranking
- **BGE-Financial** (legacy model server) — SADECE emergency fallback (Jina down ise)
- **RRF fusion** — Gemini + Jina embedding sonuclarini birlestirme

LanceDB'ye geciste tum vektorler oldugu gibi tasinacak. Embedding model upgrade ayri bir Phase'de yapilabilir.

---

## Risk Analizi

| Risk | Olasilik | Etki | Onlem |
|------|----------|------|-------|
| Grafeo immaturity (525 star) | ORTA | YUKSEK | Repository pattern ile soyutla. Swap icin RyuGraph (MIT, KuzuDB drop-in) hazir tut |
| Zvec immaturity (v0.3.0) | YUKSEK | DUSUK | Sadece 1 experimental collection. Primary her zaman LanceDB |
| LanceDB concurrent writer limiti | DUSUK | ORTA | Singleton write pattern + retry. 100K olcekte sorun degil |
| Migration sirasinda veri kaybi | DUSUK | KRITIK | ChromaDB'yi silme! Shadow mode: 1 hafta iki sistem paralel calissin |
| DuckDB VSS experimental | DUSUK | DUSUK | Analytics icin kullan, vector search icin LanceDB/Zvec |

---

## Basari Kriterleri

### DB Evolution
- [ ] "database is locked" hatasi: **SIFIR** (24 saat monitoring)
- [ ] RAM kullanimi: mevcut'tan **EN AZ 500MB dusuk**
- [ ] Vector search latency: **<5ms** (ChromaDB'nin 10-50ms'inden dusuk)
- [ ] Graph query latency: **<1ms** (Grafeo benchmark: 0.4ms)
- [ ] ChromaDB: **tamamen kaldirilmis**

### Phase 26 Sprint 1 Borc Kapatma
- [ ] CatBoost model: **catboost_signal_v1.cbm MEVCUT ve triple_perception kullanıyor**
- [ ] sizing_multiplier: **custom_stake_amount() OKUYOR ve trade size'i etkiliyor**
- [ ] Pheromone zinciri: **deposit → evidence_engine read → neural_organism read CALISIYOR**
- [ ] neural_organism.py: **pheromone_field import ediyor, okuyor, yaziyor**
- [ ] evidence_engine.py: **pheromone'dan SIGNAL_PREDICTION okuyor**
- [ ] Scheduler retraining: **CatBoost (haftalik) + OOD refit + Conformal recal + Ensemble refit job'lari VAR**
- [ ] requirements-phase26.txt: **Dockerfile.ai'ye baglanmis**

### Test Coverage
- [ ] Phase 26 modul testleri: **10/10 modul test edilmis** (chart_features + 9 yeni)
- [ ] Tum 112+ pytest: **PASS**
- [ ] End-to-end: sizing_multiplier + pheromone zinciri test edilmis

### Sprint 2 Hazirlik
- [ ] Grafeo semalari: causal, RL agent, GNN export HAZIR
- [ ] DuckDB semalari: rl_replay_buffer, world_model_states, dream_scenarios HAZIR
- [ ] LanceDB rl_relevance alanlari: HAZIR
- [ ] user_data/models/rl/ dizini: HAZIR
- [ ] Sprint 2 yeni SQLite tablolari: db.py'de TANIMLI

### Canli Sistem
- [ ] Canli trade akisi: **kesintisiz**
- [ ] Tum servisler: **restart sonrasi stable**

---

## Phase 26 Sprint 2 Durumu (UNUTMA!)

**Sprint 1 TAMAMLANDI ve CANLIDA:**
- Triple Perception (TTM + Chronos-Bolt + CatBoost scaffold) DEPLOYED
- 193 chart feature CALISIYOR
- OOD Detector, CQR Calibrator, Deep Ensemble DEPLOYED
- Stigmergic Pheromone DEPLOYED
- Predictive Interoception CALISIYOR (2 alert: organism_health, prediction_error_avg)
- Hormonlar normal: cortisol=1.0, dopamine=1.05, serotonin=0.786, adrenaline=1.0

**Sprint 2 TODO (Phase 28'den SONRA baslar):**

| Gun | Task | Durum |
|-----|------|-------|
| 6A | causal_engine.py — Tigramite PCMCI+ | BEKLIYOR → Grafeo graph'a yazacak |
| 6B | counterfactual_engine.py — DoWhy | BEKLIYOR |
| 6C | Causal → Organism entegrasyonu | BEKLIYOR → Grafeo synapse gucellemesi |
| 7A | rl_environment.py — Custom RL env | BEKLIYOR |
| 7B | iql_pretrain.py — Offline pre-training | BEKLIYOR |
| 7C | sac_online.py — Online fine-tuning | BEKLIYOR |
| 7D | hrl_meta_policy.py — Hierarchical meta-agent | BEKLIYOR |
| 8A | ewc_continual.py — Catastrophic forgetting | BEKLIYOR |
| 8B | reptile_meta.py — Meta-learning | BEKLIYOR |
| 8C | world_model.py — JEPA-RSSM | BEKLIYOR → DuckDB analytics kullanacak |
| 8D | dream_engine.py — Dream-Augmented Learning | BEKLIYOR → DuckDB analytics kullanacak |
| 9A | self_model.py — Metacognition | BEKLIYOR |
| 9B | active_learner.py — Bilgi arayisi | BEKLIYOR |
| 9C | Autonomous Lifecycle (13 katman) | BEKLIYOR |
| 9D | gnn_organism.py — GNN on MAGMA | BEKLIYOR → Grafeo graph'tan okuyacak |
| 10A | Multi-Modal Fusion | BEKLIYOR |
| 10B | LLM×RL×RAG Trinity | BEKLIYOR → LanceDB RAG kullanacak |
| 10C | Constitution + Latency Tiers | BEKLIYOR |
| 10D-E | Entegrasyon test + Deploy | BEKLIYOR |

**CatBoost Training:** Sprint 1'den kalan borc. Sprint 2'de 2A-2D olarak planli.

---

## Tahmini Zaman

| Sprint | Sure | Aciklama |
|--------|------|----------|
| 28.0 Altyapi + Phase 26 temel tamir | 2-3 gun | db.py v2, pooling, sizing_multiplier fix, Sprint 2 tablolari, testler |
| 28.1 LanceDB + RAG entegrasyon | 2-3 gun | ChromaDB → LanceDB goc + rag_graph + evidence_engine pheromone |
| 28.2 Grafeo + Organism entegrasyon | 2-3 gun | Graph DB + neural_organism pheromone + Sprint 2 semalari |
| 28.3 DuckDB + Zvec + CatBoost | 2-3 gun | Analytics + CatBoost training pipeline + scheduler jobs |
| 28.4 Deploy + Test + Dogrulama | 2-3 gun | CatBoost ilk training + end-to-end test + canli gecis + 24h monitoring |
| **TOPLAM** | **10-15 gun** | Sprint 2'den once tamamlanmali |

**Not:** Eski tahmin 7-12 gundi. Phase 26 borc kapatma isleri eklendigi icin 10-15 gune cikti.

---

## Kaynaklar

### LanceDB
- [700M vektor production deployment](https://sprytnyk.dev/posts/running-lancedb-in-production/)
- [AWS 1B+ vektor mimarisi](https://aws.amazon.com/blogs/architecture/a-scalable-elastic-database-and-search-solution-for-1b-vectors-built-on-lancedb-and-amazon-s3/)
- [Lance format v2.2 benchmark](https://www.lancedb.com/blog/lance-format-v2-2-benchmarks-half-the-storage-none-of-the-slowdown)
- [GitHub](https://github.com/lancedb/lancedb)

### Grafeo
- [GitHub](https://github.com/GrafeoDB/grafeo)
- [HackerNews tartisma](https://news.ycombinator.com/item?id=47467567)

### DuckDB
- [DuckDB.org](https://duckdb.org/)
- [VSS Extension](https://duckdb.org/2024/05/03/vector-similarity-search-vss)
- [DuckPGQ Graph Extension](https://duckpgq.org/)

### Zvec (Alibaba)
- [GitHub](https://github.com/alibaba/zvec)
- [Benchmark](https://zvec.org/en/docs/benchmarks/)
- [MarkTechPost duyurusu](https://www.marktechpost.com/2026/02/10/alibaba-open-sources-zvec-an-embedded-vector-database-bringing-sqlite-like-simplicity-and-high-performance-on-device-rag-to-edge-applications/)

### Elenen Adaylar
- [KuzuDB Apple acquisition](https://www.theregister.com/2025/10/14/kuzudb_abandoned/)
- [SurrealDB BSL 1.1](https://surrealdb.com/license)
- [Milvus Lite "not for production"](https://milvus.io/blog/embedded-milvus.md)
- [FalkorDB SSPL](https://dbdb.io/db/falkordb)

### Karsilastirmalar
- [Encore: Best Vector DBs 2026](https://encore.dev/articles/best-vector-databases)
- [4xxi: Vector DB Comparison](https://4xxi.com/articles/vector-database-comparison/)
- [FAISS vs LanceDB (Zilliz)](https://zilliz.com/comparison/faiss-vs-lancedb)
