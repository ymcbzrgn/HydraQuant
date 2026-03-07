# FREQTRADE AI-ENHANCED TRADING SYSTEM - ROADMAP

## Server Specs (Dedicated)
- **RAM:** 32 GB ECC
- **CPU:** 4 Core Platinum
- **Disk:** 160 GB E-NVMe
- **OS:** Linux (no GPU)
- **Lokasyon:** Istanbul / Bursa
- **Purpose:** 100% dedicated to this trading system

## Architecture Decisions

### HABER BEDAVA CEKiLiR (Temel Kural)
> Haber ve sentiment verisi icin ASLA ucretli abonelik KULLANILMAZ.
> Tum news pipeline %100 ucretsiz kaynaklardan olusur: API + RSS + Scrapling (fallback).
> Toplam haber maliyeti: $0.00/ay

**News Pipeline Mimarisi (5 Katman, $0/ay):**

| Katman | Kaynak | Yontem | Frekans | Veri |
|--------|--------|--------|---------|------|
| 1. Primary | cryptocurrency.cv API | REST + SSE streaming | Real-time | 200+ kaynak, built-in sentiment, 662K+ arsiv |
| 2. RSS | CoinTelegraph (7 feed), CoinDesk, Decrypt, The Block, CryptoSlate, ChainGPT | feedparser | Her 5-10dk | 100-200 makale/gun, headline+summary |
| 3. Sentiment | alternative.me Fear & Greed Index | REST API (key gereksiz) | Her 5dk | 0-100 skor, piyasa rejimi |
| 4. Enrichment | CryptoPanic (free tier), Alpha Vantage News, CryptoCompare | REST API | 15dk-1saat | Community votes, pre-computed sentiment |
| 5. Fallback | Scrapling (D4Vinci/Scrapling) | Adaptive scraping | Sadece gerektiginde | Cloudflare bypass, anti-bot, adaptive parser |

**RSS Feed Listesi (Dogrulanmis, Calisan):**
```
TIER 1 (Ana kaynaklar):
  CoinDesk:              https://www.coindesk.com/arc/outboundfeeds/rss/?outputType=xml
  CoinTelegraph (all):   https://cointelegraph.com/rss
  CoinTelegraph (BTC):   https://cointelegraph.com/rss/tag/bitcoin
  CoinTelegraph (ETH):   https://cointelegraph.com/rss/tag/ethereum
  CoinTelegraph (Alt):   https://cointelegraph.com/rss/tag/altcoin
  CoinTelegraph (DeFi):  https://cointelegraph.com/rss/tag/defi
  CoinTelegraph (Analiz):https://cointelegraph.com/rss/tag/market-analysis
  CoinTelegraph (Reg):   https://cointelegraph.com/rss/tag/regulation
  Decrypt:               https://decrypt.co/feed
  The Block:             https://www.theblock.co/rss.xml

TIER 2 (Ek kaynaklar):
  CryptoSlate:           https://cryptoslate.com/feed/
  CryptoPotato:          https://cryptopotato.com/feed/
  CryptoNews:            https://cryptonews.com/news/feed/
  The Defiant:           https://thedefiant.io/feed/
  Bitcoin Magazine:       https://bitcoinmagazine.com/feed (TEK tam metin veren site)

TIER 3 (AI-curated + mainstream):
  ChainGPT (all):        https://app.chaingpt.org/rssfeeds.xml
  ChainGPT (BTC):        https://app.chaingpt.org/rssfeeds-bitcoin.xml
  ChainGPT (ETH):        https://app.chaingpt.org/rssfeeds-ethereum.xml
  Yahoo Finance:         https://finance.yahoo.com/news/rssindex
```

**Neden Scraping Degil RSS/API?**
- RSS headline + summary, sentiment analizi icin YETERLI (arastirma: headline'lar sentiment sinyalinin %60-80'ini tasir)
- RSS: yasal risk sifir, IP ban yok, bakim minimal
- Scraping: ToS ihlali riski, IP ban, site degisirse kirilir
- Scrapling SADECE fallback: RSS/API olmayan siteler icin

**Scrapling (Fallback Scraper):**
- Repo: github.com/D4Vinci/Scrapling (25.3K star, BSD-3-Clause)
- Python 3.10+, GPU gereksiz, 32GB RAM'de rahat calisir
- 3 katman: Fetcher (HTTP) -> StealthyFetcher (Cloudflare bypass) -> DynamicFetcher (full browser)
- BeautifulSoup'tan 784x hizli, adaptive parser (site degisse bile element'leri bulur)
- Spider API (Scrapy benzeri), ProxyRotator, pause/resume
- Kurulum: `pip install "scrapling[fetchers]"` + `scrapling install`
- RAM: Fetcher ~50MB, StealthyFetcher ~500MB (Chromium), DynamicFetcher ~500MB

**robots.txt Analizi (Dogrulanmis):**
- CoinDesk: AI botlari BLOKE, RSS kullan
- CoinTelegraph: News sayfalari acik, ama RSS tercih
- Decrypt: Hicbir kisitlama yok
- The Block: Cloudflare aktif koruma, RSS kullan
- Bitcoin Magazine: Crawl-delay 5s, AI botlari bloke DEGIL

### API-First LLM Strategy
> Yerel LLM (4GB RAM, 2 tok/s) YERINE API-tabanli LLM kullanimi
> Bu karar ~5 GB RAM ve ~43 GB disk tasarrufu saglar

**LLM Provider Hierarchy (failover chain):**
1. **Gemini Flash** (birincil) - Hizli, ucuz/ucretsiz, trading analizi icin yeterli
2. **Groq** (ikincil) - Ultra hizli inference, ucretsiz tier
3. **OpenRouter** (ucuncul) - 100+ model, fallback

**Neden API-First:**
- 0 GB RAM kullanimi (sunucunun 32 GB'ini RAG + sentiment + FreqAI + LoRA'ya adayabiliriz)
- Gemini Flash: ~1500 req/dk ucretsiz, trading icin fazlasiyla yeterli
- Groq: Llama/Mixtral ~6000 tok/s (yerel 2 tok/s'ye karsi 3000x hizli)
- Sunucunun tum gucunu RAG pipeline + sentiment + FreqAI'ya adayabiliriz
- Gemini'nin reasoning kalitesi yerel 4B modelden kat kat iyi

### DeepMind Embedding Ceiling (Eylul 2025)
> Google DeepMind matematiksel olarak kanitladi: tek-vektor embedding'lerin bir tavani var
> 512-dim embedding ~500K dokumanda bozulmaya baslar
> 1024-dim ~4M, 4096-dim ~250M dokumanda bozulur

**Bizim cozumumuz (tavani asmak icin 4 katman):**
1. Hybrid Search (BM25 + dense) - embedding'e bagimsiz keyword eslestirme
2. Dual Embedding (Gemini + BGE-Financial) - iki farkli bakis acisi
3. ColBERTv2 late interaction reranking - token-seviyesi eslestirme
4. Knowledge Graph (LazyGraphRAG) - iliskisel bilgi, embedding'e bagimli degil

---

## RAG MASTER ARCHITECTURE: 86+ Teknikten 42 Secilmis Teknik

### Teknik Secim Kriterleri:
1. Kripto trading'e uygulanabilirlik (>= HIGH)
2. 32GB RAM / 4 Core Platinum / no-GPU kisitlamalari
3. API-first LLM stratejisiyle uyumluluk
4. Implementasyon karmasikligi vs. katki orani

### Secilen 42 Teknik (Kategorize)

```
QUERY-SIDE (Sorgu Optimizasyonu):
  1. Multi-Query RAG           - Tek sorgudan 3-5 farkli perspektif
  2. RAG-Fusion + RRF          - Paralel retrieval + skor birlestirme
  3. HyDE                      - Hipotetik dokuman embedding
  4. Step-Back Prompting       - Soyut kavram seviyesinde retrieval
  5. Query Decomposition       - Karmasik sorguyu parcala
  6. Query Routing (Adaptive)  - Basit/orta/karmasik yonlendirme
  7. Query Rewriting           - LLM ile sorgu duzeltme

RETRIEVAL (Getirme):
  8. Dense Retrieval (HNSW)    - Semantik vektor arama
  9. Sparse Retrieval (BM25)   - Keyword eslestirme
  10. Hybrid Search + RRF      - Dense + Sparse birlestirme
  11. ColBERTv2 Late Interaction - Token-seviyesi eslestirme
  12. Parent-Child Retrieval   - Kucuk chunk bul, buyuk parent goster
  13. Sentence Window Retrieval - Cumle bul, etrafini genislet
  14. Tool-Augmented RAG       - Canli API'leri retrieval olarak kullan

CHUNKING (Belge Bolme):
  15. Recursive Chunking       - Hiyerarsik ayiricilarla bolme
  16. Semantic Chunking        - Embedding benzerligi ile bolme
  17. Contextual Chunking      - Her chunk'a LLM ile context ekle (Anthropic)
  18. Parent-Child Chunking    - Kucuk+buyuk chunk ciftleri
  19. RAPTOR Enhanced          - Hiyerarsik ozet agaci (%76 daha kompakt)
  20. Late Chunking            - Once embed et, sonra bol (Jina)
  21. Proposition Chunking     - Atomik bilgi unitelerine ayir

EMBEDDING:
  22. Dual Embedding           - Gemini + BGE-Financial (RRF fusion)
  23. Matryoshka Flexibility   - Boyut esnek (256-768 dim arasi)
  24. Binary Quantization      - 32x depolama azaltma (ilk tarama icin)
  25. Embedding Cache          - SQLite-vec ile tekrar embed onleme

RERANKING:
  26. FlashRank (CPU-native)   - Hafif cross-encoder (birincil)
  27. ColBERT Reranker         - Token-seviyesi reranking (ikincil)
  28. Multi-Reranker Ensemble  - Birden fazla reranker birlestirme

GENERATION-SIDE (Uretim):
  29. Self-RAG                 - Kosullu retrieval + self-critique
  30. CRAG (Corrective RAG)    - Retrieval kalite degerlendirme
  31. Faithful/Attributed RAG  - Citation + audit trail
  32. Speculative RAG          - Draft-verify pattern

GRAPH-BASED:
  33. LazyGraphRAG             - Entity-iliski grafi (%0.1 maliyet)
  34. Temporal Graph (TG-RAG)  - Zamansal iliski grafi
  35. CDF-RAG                  - Causal reasoning (sahte korelasyonlari ayir)

MEMORY:
  36. GAM-RAG                  - Kullandikca ogrenen bellek
  37. MAGMA                    - 4 grafik bellek (semantic/temporal/causal/entity)
  38. MemoRAG                  - Global corpus bellegi
  39. Bidirectional RAG        - Dogrulanmis bilgiyi geri yaz

TEMPORAL/REAL-TIME:
  40. TS-RAG                   - Zaman serisi pattern retrieval
  41. StreamingRAG             - Gercek zamanli veri entegrasyonu
  42. Temporal Decay           - Zaman agirlikli skor azaltma

OPTIMIZATION (Tumune uygulanir):
  A. Semantic Caching          - 65x latency azaltma
  B. Prompt Caching            - %90 API maliyet azaltma
  C. Offline/Online Ayirimi    - Indeksleme vs sorgu ayir
  D. Intelligent Query Routing - %30-45 maliyet azaltma

EVALUATION (Surekli izleme):
  E. RAGAS Framework           - Faithfulness, Context Precision, Answer Relevancy
  F. DeepEval                  - Hallucination detection + CI/CD entegrasyonu
  G. RAG Triad (TruLens)       - 3 metrik ile pipeline diag
  H. Lynx                     - Hallucination guardian
```

### Elenip KULLANILMAYACAK Teknikler (ve nedeni):
```
- REALM/RETRO: Pre-training gerektirir, biz hazir model kullaniyoruz
- Full GraphRAG (Microsoft): Cok pahali indeksleme, LazyGraphRAG yeterli
- Multimodal RAG: Chart analizi icin fayda/maliyet orani dusuk (su an)
- ColPali/ColQwen: Visual document indexing, oncelik degil
- Video-RAG/Audio-RAG: Trading'de gereksiz
- SignRAG/AffordanceRAG: Farkli domain (otonom surucu/robotik)
- RAGDiffusion: Goruntu uretimi, alakasiz
- PageIndex: Tek dokuman, cok LLM call, yavas
- Full Knowledge Graph (KG-RAG): Ontoloji insa maliyeti yuksek
- Microservice RAG: 4 core sunucu icin overhead fazla
- LLM-as-Reranker: 4-6 saniye latency, FlashRank yeterli
- REALM: Pre-training altyapisi lazim
```

---

## PHASE 0: Foundation & Infrastructure (Week 1)
> Binance testnet'te dry-run ile sistemi ayaga kaldir

### 0.1 Freqtrade Core Setup
- [ ] Binance testnet config olustur (spot + futures)
- [ ] Docker compose ile deploy
- [ ] FreqUI kurulumu ve API server aktif etme
- [ ] Telegram bot baglantisi
- [ ] Temel strateji secimi: InformativeSample.py (multi-tf, temiz yapi, AI base icin ideal)
      - Alternatif: CustomStoplossWithPSAR.py (custom_stoploss hook mevcut)
      - NOT: Repo'da 65 strateji var, HICBIRI FreqAI kullanmiyor - temiz zemin
- [ ] Dry-run ile 48 saat test

### 0.2 Development Environment
- [ ] Python 3.13 virtualenv
- [ ] ChromaDB + SQLite-vec kurulumu
- [ ] sentence-transformers kurulumu
- [ ] LangChain + LangGraph kurulumu
- [ ] Gemini API SDK (google-generativeai) kurulumu
- [ ] Groq SDK + OpenRouter SDK kurulumu
- [ ] feedparser kurulumu (RSS feed parsing)
- [ ] Scrapling kurulumu: pip install "scrapling[fetchers]" && scrapling install
- [ ] API key'leri environment variable olarak ayarla:
      GEMINI_API_KEY, GROQ_API_KEY, OPENROUTER_API_KEY, CRYPTOPANIC_API_KEY,
      CRYPTOCOMPARE_API_KEY, ALPHAVANTAGE_API_KEY

### 0.3 Data Pipeline Foundation (HABER BEDAVA CEKiLiR)
- [ ] cryptocurrency.cv API entegrasyonu (primary, key gereksiz, 200+ kaynak, built-in sentiment)
      - REST: /api/news, /api/ai/sentiment
      - SSE streaming: real-time haber akisi
- [ ] RSS feed aggregator kurulumu (feedparser ile):
      - Tier 1: CoinDesk, CoinTelegraph (7 kategori), Decrypt, The Block
      - Tier 2: CryptoSlate, CryptoPotato, CryptoNews, The Defiant, Bitcoin Magazine
      - Tier 3: ChainGPT (AI-curated), Yahoo Finance
      - Deduplikasyon: title similarity hashing (ayni haber 5-10 feed'de cikar)
- [ ] Fear & Greed Index API baglantisi (alternative.me, key gereksiz, her 5dk)
- [ ] CryptoPanic API key al (free tier, 100 req/gun, community sentiment votes)
- [ ] CryptoCompare API key al (free tier, 50 call/saat, 150+ kaynak aggregator)
- [ ] Alpha Vantage API key al (free tier, 25 req/gun, pre-computed sentiment skorlari)
- [ ] Scrapling kurulumu (fallback scraper): pip install "scrapling[fetchers]" && scrapling install
- [ ] Veri toplama cron job'lari olustur:
      - cryptocurrency.cv SSE: real-time (surekli dinle)
      - RSS feeds: her 5-10dk (feedparser poll)
      - Fear & Greed: her 5dk
      - CryptoPanic: her 15dk
      - CryptoCompare: her 5dk (50/saat limit dahilinde)
      - Alpha Vantage: 1-2x/saat
- [ ] SQLite veritabani semasiyla haber/sentiment storage
- [ ] AI Decision logging tablosu olustur (Phase 3.5 icin altyapi, Beta distribution + forgone P&L)
- [ ] Config schema'ya AI section'lar ekle: llm_config, rag_config, sentiment_config, autonomy_config
      - Freqtrade config_schema.py'ye yeni top-level property'ler
      - Env var pattern: FREQTRADE__LLM_CONFIG__API_KEYS__GEMINI
- [ ] Docker variant olustur: docker/Dockerfile.ai (freqai pattern takip et)
      - AI dependencies: langchain, chromadb, sentence-transformers, google-generativeai, groq
      - News dependencies: feedparser, scrapling[fetchers], sseclient-py (SSE streaming)

---

## PHASE 1: Sentiment Engine (Week 2)
> CryptoBERT + FinBERT ile gercek zamanli sentiment

### 1.1 Model Setup
- [ ] CryptoBERT (ElKulako/cryptobert, 100M param) yukle - CPU'da ~200ms/inference
- [ ] FinBERT (ProsusAI/finbert, 110M param) yukle - CPU'da ~200ms/inference
- [ ] ONNX Runtime ile CPU optimizasyonu (2-3x hizlanma)

### 1.2 Sentiment Pipeline (Multi-Source, $0/ay)
- [ ] cryptocurrency.cv + RSS feeds + CryptoCompare haberlerini CryptoBERT'e besle
      - feedparser -> title+summary extraction -> CryptoBERT -> sentiment score
      - Deduplikasyon: ayni haberi birden fazla kaynaktan almamak icin title hash
- [ ] Her coin icin rolling sentiment skoru hesapla (1h, 4h, 24h window)
      - Kaynak agirliklandirma: Tier 1 RSS x1.0, Tier 2 x0.8, Tier 3 x0.6
- [ ] FinBERT ile genel piyasa sentiment skoru
- [ ] Fear & Greed Index entegrasyonu (alternative.me, her 5dk, contrarian sinyal)
- [ ] CryptoPanic community votes entegrasyonu (bullish/bearish/neutral etiketleri bedava!)
- [ ] Alpha Vantage pre-computed sentiment skorlari (cross-validation icin)
- [ ] Sentiment skorlarini FreqAI feature olarak ekle

### 1.3 FreqAI Integration
- [ ] Custom FreqAI strategy olustur: teknik gostergeler + sentiment features
- [ ] XGBoost/LightGBM model ile sentiment-enhanced tahmin
- [ ] Backtesting: sentiment'li vs sentiment'siz karsilastirma

### RAM Budget: ~4.0 GB (toplam ~6.5 GB with Phase 0)
```
CryptoBERT (ONNX):       ~400 MB
FinBERT (ONNX):          ~400 MB
Inference buffer:        ~200 MB
News pipeline:           ~500 MB (feedparser + SSE + buyuk dedup cache + API clients)
  - Scrapling (standby):  ~50 MB (fallback, browser lazimsa +500MB, her zaman kullanilabilir)
Sentiment data store:    ~200 MB
Freqtrade core:          ~1.5 GB
```

---

## PHASE 2: Hybrid RAG Foundation (Week 3-6, 4 hafta)
> 14 temel RAG teknigi ile saglam altyapi
> NOT: Asil plan 2 haftaydi ama 40+ subtask icin gercekci timeline 4 hafta.
> Week 3-4: Dense + BM25 + basic chunking + ChromaDB
> Week 5-6: ColBERT + caching + query routing + tool-augmented RAG

### 2.1 Embedding Layer (DUAL EMBEDDING + Matryoshka + Quantization)
> Iki embedding modelini AYNI ANDA kullan, sonuclari RRF ile birlestir
> Matryoshka ile boyut esnekligi, binary quantization ile 32x tasarruf

- [ ] **Gemini text-embedding-004** (API, genel semantik)
      - Ucretsiz (1500 req/dk), 768 dim, cok dilli, Turkce destegi
      - Genel anlam, baglamsal benzerlik, soyut kavramlar
- [ ] **BGE-Base-Financial-Matryoshka** (yerel, finans-ozel)
      - 110M param, ~500 MB RAM, CPU'da hizli
      - Matryoshka: 768 -> 256 dim truncation destegi
      - Finans terminolojisi: PE ratio, yield curve, MACD crossover
- [ ] **Binary Quantization Layer:**
      ```
      Ilk Tarama: Binary quantized embeddings (32x kucuk, ultra hizli)
        |
        v
      Top-100 aday secilir
        |
        v
      Full-precision reranking (FlashRank + ColBERT)
        |
        v
      Top-10 final sonuc
      ```
- [ ] **Dual Embedding Fusion Pipeline:**
      ```
      Sorgu/Dokuman
        |
        +---> Gemini embed --> ChromaDB Collection A --> Top-30
        |
        +---> BGE-Financial embed --> ChromaDB Collection B --> Top-30
        |
        v
      Reciprocal Rank Fusion (RRF) --> Top-15
        |
        v
      FlashRank + ColBERT Reranker --> Top-5 (final)
      ```
- [ ] **all-MiniLM-L6-v2** (ultra-hafif offline fallback, 22M, ~80 MB)
- [ ] Embedding cache: SQLite-vec ile ayni metin tekrar embed edilmez
- [ ] Batch embedding: Yeni haberler toplu embed edilir (verimlilik)

### 2.2 Chunking Strategy (6 Katmanli)
> Her belge turune uygun chunking stratejisi

- [ ] **Recursive Chunking** (varsayilan): 512 token, %15 overlap
      - Paragraf -> cumle -> kelime hiyerarsisiyle bol
      - Benchmark: %69 dogruluk (en iyi genel performans)
- [ ] **Contextual Chunking** (Anthropic yontemi):
      - Her chunk'a LLM ile "Bu chunk, X raporunun Y bolumunden..." context ekle
      - %67 retrieval hata azaltma (en buyuk tek iyilestirme!)
      - Gemini Flash ile context uretimi (ucuz + hizli)
- [ ] **Parent-Child Chunking** (hassas retrieval + genis context):
      - Child: 128 token (retrieval icin, hassas eslestirme)
      - Parent: 512 token (LLM'e verilir, tam context)
      - Child eslestir -> Parent dondur
- [ ] **RAPTOR Enhanced** (hiyerarsik ozet agaci):
      - Leaf: Orijinal chunk'lar
      - Level 1: Cluster ozetleri (Gemini ile)
      - Level 2: Meta-ozetler
      - %76 daha kompakt (Enhanced versiyon, 2026)
      - "Genel piyasa trendi" sorgusu -> ust seviye
      - "BTC RSI degeri" sorgusu -> leaf seviye
- [ ] **Late Chunking** (Jina yontemi):
      - Once tum dokumani embed et (8192 token context)
      - Sonra chunk sinirlari uygula + mean pooling
      - Zamir/referans problemi cozulur ("onun fiyati" -> "BTC'nin fiyati")
- [ ] **Proposition Chunking** (atomik bilgi uniteleri):
      - Her chunk tek bir bilgi parcasi: "BTC ATH $108K, 17 Aralik 2024"
      - Ozellikle fiyat, tarih, metrik verileri icin
- [ ] Metadata zenginlestirme: tarih, kaynak, coin, sentiment skoru, belge turu

### 2.3 Hybrid Search (Dense + Sparse + Late Interaction)
```
Query
  |
  v
[Query Router] -- Basit/Orta/Karmasik siniflandir
  |
  v
[Query Transformer] -- Multi-query + HyDE + Step-Back (karmasikliga gore)
  |
  +--> [Dense Search] Dual embedding + ChromaDB (HNSW index)
  |
  +--> [Sparse Search] BM25 (rank_bm25, ticker/terim icin)
  |
  +--> [Late Interaction] ColBERTv2 (token-seviyesi, PLAID compression)
  |
  v
[Reciprocal Rank Fusion] -- Uc sonucu birlestir
  |
  v
[Multi-Reranker Ensemble]
  |-- FlashRank (hizli, CPU-native, birincil)
  |-- ColBERT reranker (hassas, ikincil)
  |
  v
[Top-K Results] (5-10 chunk)
```

- [ ] ChromaDB dual collection + HNSW index olustur
- [ ] BM25 index olustur (ticker sembolleri, ozel terimler icin)
- [ ] ColBERTv2 + PLAID compression kurulumu
- [ ] Reciprocal Rank Fusion implementasyonu
- [ ] Multi-reranker ensemble: FlashRank + ColBERT
- [ ] Binary quantization ile ilk tarama hizlandirma
- [ ] Hybrid search pipeline testi

### 2.4 Tool-Augmented RAG
> RAG sadece belge aramak degil, canli veriye de erismek demek

- [ ] **Canli Fiyat Retrieval**: Exchange API'den anlik fiyat cekme
- [ ] **On-Chain Retrieval**: Blockchain explorer API (whale hareketleri, TVL)
- [ ] **Order Book Retrieval**: Depth verisi (likidite analizi)
- [ ] **Indicator Calculator**: Teknik gosterge hesaplama (RSI, MACD, BB)
- [ ] **Portfolio State**: Mevcut pozisyon ve risk durumu sorgulama
- [ ] Tool registry: Her tool'un cagrilma kosullari + response formati

### 2.5 Vector DB Architecture
```
ChromaDB Collections:
  +-- crypto_news          (haberler, 180 gun rolling window)
  |     +-- child_chunks   (128 token, retrieval icin)
  |     +-- parent_chunks  (512 token, context icin)
  |
  +-- market_reports       (piyasa raporlari, analizler)
  |     +-- raptor_tree    (hiyerarsik ozet agaci)
  |
  +-- trade_history        (gecmis trade'ler ve sonuclari)
  +-- technical_signals    (teknik analiz sinyalleri)
  +-- on_chain_data        (on-chain metrikler ve anomaliler)
  +-- time_series_patterns (TS-RAG icin benzeri fiyat pattern'leri)
  +-- knowledge_graph      (LazyGraphRAG entity-iliski verileri)

SQLite-vec:
  +-- embedding_cache      (tekrar embed onleme)
  +-- semantic_cache       (sorgu-yanit onbellegi, 15dk TTL)
  +-- memory_store         (uzun sureli bellek, GAM-RAG)
  +-- decision_log         (Gemini karar kayitlari, distillation icin)
```

### 2.6 Optimization Layer (Tum Pipeline'a Uygulanir)
- [ ] **Semantic Caching**: Benzer sorgulari taniyip onceki yaniti dondur
      - 65x latency azaltma
      - Embedding cosine similarity > 0.92 -> cache hit
      - TTL: 5 dakika (volatil piyasa) / 1 saat (stabil piyasa)
      - Cache invalidation: Buyuk fiyat hareketi veya breaking news
- [ ] **Prompt Caching**: Gemini API prefix caching
      - Sistem prompt + strateji context'i cache'le
      - %90 maliyet azaltma tekrarlanan cagrilarda
- [ ] **Offline/Online Pipeline Ayirimi**:
      - Offline: Chunking, embedding, indeksleme (arka planda)
      - Online: Retrieval, reranking, generation (dusuk latency)
- [ ] **Intelligent Query Routing**:
      - "BTC fiyati ne?" -> Dogrudan API call (RAG bypass)
      - "BTC ne olacak?" -> Tam RAG pipeline
      - "Stop-loss nedir?" -> Statik bilgi cache'den

### RAM Budget: ~10.5 GB (toplam ~17 GB)
```
BGE-Financial Embedding:   ~500 MB
Gemini Embed cache:        ~500 MB  (+200 MB, yuksek hit rate)
all-MiniLM-L6-v2:          ~80 MB
ColBERTv2 (PLAID compressed): ~400 MB
BM25 Index:                ~300 MB  (+100 MB, buyuk inverted index)
ChromaDB (500K+ docs):     ~5 GB   (+2 GB, HNSW recall +5-8%)
FlashRank Reranker:        ~300 MB
Semantic Cache:            ~1.5 GB  (+1 GB, 3x entry, hit rate %80)
LangChain/Pipeline:        ~500 MB
RAPTOR tree structures:    ~300 MB
Tool-Augmented modules:    ~200 MB
Query Router/Transformer:  ~200 MB
```

---

## PHASE 3: Advanced RAG Techniques (Week 7-12, 6 hafta - 3 Tier)
> 15 gelismis RAG teknigi - Akilli retrieval + self-correction + graph
> NOT: 15 ileri teknik 2 haftada IMKANSIZ. 3 tier'a bolundu:
> Tier 1 (Week 7-8):  Self-RAG, CRAG, LazyGraphRAG (en kritik 3)
> Tier 2 (Week 9-10): MAGMA, TS-RAG, Temporal Decay, GAM-RAG (bellek + zaman)
> Tier 3 (Week 11-12): CDF-RAG, MADAM-RAG, FLARE, HyDE, Speculative, CoT-RAG, Adaptive, RAG-Fusion (MVP sonrasi defer edilebilir)

### 3.1 Self-RAG (Kosullu Retrieval + Self-Critique)
> Model kendi basina ne zaman retrieval gerektigine karar verir
> Her uretimi self-critique ile degerlendirir

- [ ] Retrieval karar modulu: "Bu sorgu icin retrieval gerekli mi?"
      - Basit hesaplama/bilgi -> retrieval atlama (latency azaltma)
      - Belirsiz/gunceli gereken bilgi -> retrieval tetikle
- [ ] Self-critique tokenlari: Uretilen yanitin kalitesini degerlendir
      - "Bu yanit retrieved evidence'a dayanıyor mu?" (faithfulness check)
      - "Bu yanit soruyu cevapliyor mu?" (relevance check)
- [ ] Confidence skoru: Her trade sinyaline guven puani ekle
      - <0.35 -> Toz pozisyon (confidence^2 egrisi, komisyon kendini filtreler)
      - 0.35-0.50 -> Mikro kesfedici pozisyon
      - 0.50-0.70 -> Standart pozisyon
      - 0.70-0.85 -> Yuksek inanc pozisyonu
      - >0.85 -> Maksimum inanc pozisyonu
      - NOT: HICBIR seviyede trade ENGELLENMEZ (canonical skala, bkz. Confidence-Bazli Pozisyon Boyutlandirma)

### 3.2 Corrective RAG (CRAG) - Enhanced
> Retrieval kalitesini degerlendir, kotu sonuclari duzelt

- [ ] Retrieval Evaluator modulu:
      - **Correct** (>0.8 relevance) -> Dogrudan kullan
      - **Ambiguous** (0.4-0.8) -> Decompose-then-recompose ile isle
      - **Incorrect** (<0.4) -> At + alternatif retrieval tetikle
- [ ] Corrective Actions:
      - Web search fallback (DuckDuckGo API)
      - Farkli collection'dan retrieval
      - Query rewriting + tekrar deneme
      - Gemini Grounding (Google Search) ile dogrulama
- [ ] Finansal haber dogrulama: manipulatif/yanlis haberleri filtrele
- [ ] CRAG + Adaptive RAG entegrasyonu: Karmasikliga gore farkli CRAG stratejisi

### 3.3 Adaptive RAG (Intelligent Query Routing)
> Sorgu karmasikligina gore pipeline sec

- [ ] Query complexity classifier:
      ```
      SIMPLE  -> Tek retrieval + dogrudan yanit
                 Ornek: "BTC son fiyat", "RSI nedir?"

      MEDIUM  -> Multi-query + hybrid search + reranking
                 Ornek: "BTC teknik gorunumu", "ETH sentiment"

      COMPLEX -> Multi-hop + chain-of-retrieval + multi-agent
                 Ornek: "Fed karari + BTC korelasyonu + altcoin etkisi"

      NO_RAG  -> LLM-only (statik bilgi, hesaplama)
                 Ornek: "Moving average nasil hesaplanir?"
      ```
- [ ] Router modulu: Gemini Flash ile hizli siniflandirma
- [ ] Maliyet takibi: Her pipeline'in ortalama cost/latency'si

### 3.4 RAG-Fusion (Multi-Query + RRF)
- [ ] Tek sorgudan 3-5 farkli perspektif sorgu uret
- [ ] Her sorgu icin paralel retrieval
- [ ] RRF ile sonuclari birlestir
- [ ] Ornek: "BTC ne olacak?" ->
      "BTC teknik analiz gorunumu nedir?",
      "BTC haberleri son 24 saat neler?",
      "BTC on-chain metrikleri nasil?",
      "Genel kripto piyasa sentiment nedir?",
      "BTC ile korele olan varliklar ne durumda?"

### 3.5 HyDE (Hypothetical Document Embeddings)
- [ ] Sorgudan once hipotetik cevap uret (Gemini Flash ile)
- [ ] Hipotetik cevabi embedding'e donustur
- [ ] Bu embedding ile retrieval yap
- [ ] **Secici kullanim**: Sadece standart retrieval dusuk guvenli oldugunda
      - Normal retrieval confidence > 0.7 -> HyDE atla
      - Normal retrieval confidence < 0.7 -> HyDE devreye
- [ ] Finansal jargon farklarini kapat

### 3.6 Speculative RAG (Draft-Verify Pattern)
> Kucuk model birden fazla taslak uretir, buyuk model dogrular

- [ ] Draft asamasi: Yerel model (LoRA) veya Gemini Flash
      - 3-5 farkli piyasa yorumu uret (paralel)
      - Her taslak farkli retrieved subset kullanir
- [ ] Verify asamasi: Gemini Flash veya Groq
      - Taslaklari evidence ile karsilastir
      - En iyi desteklenen taslagi sec
- [ ] Trading uygulamasi: "BTC icin 3 senaryo uret, en iyi destekleneni sec"

### 3.7 LazyGraphRAG (Iliskisel Bilgi Grafi)
> Entity-iliski grafigi ile cok-adimli (multi-hop) reasoning
> LazyGraphRAG: Full GraphRAG maliyetinin %0.1'i, ayni kalite

- [ ] Entity extraction: Gemini Flash ile belgelerdeki entity'leri cikar
      - Token'lar (BTC, ETH, SOL...)
      - Borsalar (Binance, Kraken, Bybit...)
      - Protokoller (Uniswap, Aave, Lido...)
      - Regulatorler (SEC, CFTC, MiCA...)
      - Olaylar (Halving, Merge, Hard Fork...)
- [ ] Iliski cikartma: Entity'ler arasindaki iliskiler
      - "BTC" --listed_on--> "Binance"
      - "ETH" --powers--> "Uniswap"
      - "SEC" --investigates--> "Coinbase"
      - "Fed" --affects--> "DXY" --inverse_corr--> "BTC"
- [ ] Lazy evaluation: Grafi sorgu zamaninda, on-demand olustur
      - Full GraphRAG: $20-500/corpus indeksleme
      - LazyGraphRAG: $0.02-0.50/corpus (100-1000x ucuz)
- [ ] Multi-hop query: "Fed faiz artirirsa BTC'ye nasil etki eder?"
      Fed -> faiz artisi -> DXY guclenmesi -> BTC baski -> altcoin dump

### 3.8 Temporal Graph RAG (TG-RAG)
> Zamansal iliski grafi: entity iliskilerinin zamanla degisimini takip et

- [ ] Timestamped relations: Her iliski bir zaman damgasi tasir
      - "SEC --sues--> Ripple" [2020-12] ... "SEC --settles--> Ripple" [2025-03]
- [ ] Multi-granularity temporal summaries:
      - Saat seviyesi: Anlik fiyat hareketleri
      - Gun seviyesi: Gunluk ozet
      - Hafta seviyesi: Haftalik trend
- [ ] Temporal reasoning: "Gecen sefer BTC 100K'yi kirdiginda ne olmustu?"

### 3.9 CDF-RAG (Causal Dynamic Feedback)
> Sahte korelasyonlari gercek nedensellikten ayir
> Finansal piyasalarda kritik: korelasyon != nedensellik

- [ ] Dual-path retrieval:
      - Semantik vektor arama (normal)
      - Causal graph traversal (nedensellik zinciri)
- [ ] Causal consistency check: Uretilen yanit nedensellik mantigi ile tutarli mi?
      - "BTC yukseldi CUNKU Elon tweet atti" -> Gercek nedensellik mi?
      - "BTC yukseldi VE Elon tweet atti" -> Sadece korelasyon mu?
- [ ] RL-based query refinement: Sorgulari nedensellik odakli hale getir
- [ ] Trading uygulamasi: Sahte sinyalleri filtrele, gercek katalizmanlari bul

### 3.10 MADAM-RAG (Multi-Agent Debate for Conflicts)
> Celisken sinyaller oldugunda ajan'lar tartisir

- [ ] Bull Agent: Yukselis argumanlarini toplar ve savunur
- [ ] Bear Agent: Dusus argumanlarini toplar ve savunur
- [ ] Debate rounds: 2-3 tur tartisma
- [ ] Aggregator: Tartisma sonucunu sentezle
- [ ] %15.8 iyilestirme (celisken evidence senaryolarinda)
- [ ] Ornek: BTC teknik bullish AMA haber bearish ->
      Bull: "RSI oversold, volume artis, destek korunuyor"
      Bear: "SEC sorusturma haberi, whale satisi, funding rate negatif"
      Aggregator: "Kisa vadeli risk yuksek, orta vadeli bullish"

### 3.11 FLARE (Forward-Looking Active Retrieval)
> Uretim sirasinda belirsizlik algila, gerektiginde retrieval tetikle

- [ ] Uretim baslat (Gemini Flash)
- [ ] Her cumle sonrasi confidence kontrol
      - Confidence yuksek -> devam et
      - Confidence dusuk -> o cumleyi sorgu olarak kullan, retrieval yap
- [ ] Dinamik retrieval: Sadece belirsiz noktalar icin veri cek
- [ ] Trading uygulamasi: Analiz yazarken belirsiz kalilan noktalar icin
      otomatik ek veri getirme

### 3.12 Chain-of-Thought RAG (CoT-RAG)
> Her reasoning adimini retrieved evidence ile destekle

- [ ] Adim 1: Piyasa durumu tespiti [evidence: son fiyat verileri]
- [ ] Adim 2: Teknik analiz [evidence: gosterge degerleri]
- [ ] Adim 3: Sentiment degerlendirme [evidence: haber skoru]
- [ ] Adim 4: Risk degerlendirme [evidence: portfolio durumu]
- [ ] Adim 5: Karar [evidence: tum adimlarin sentezi]
- [ ] Her adim aciklanabilir ve izlenebilir (audit trail)

### 3.13 Memory-Augmented RAG (GAM-RAG)
> Kullandikca ogrenen, Hebbian ogrenme inspirasyonlu bellek
> %61 inference maliyet azaltma, 0 ek egitim

- [ ] Gain-Adaptive Memory:
      - Yeni bilgi geldikce "gain" hesapla (yenilik + faydalilik)
      - Yuksek gain -> bellete ekle
      - Dusuk gain -> atla (bellek sismesini onle)
- [ ] Kalman-filter uncertainty tracking:
      - Her bellek ogesi bir guven skoru tasir
      - Zamanla dogrulanan bilgilerin guveni artar
      - Yanlis cikan bilgilerin guveni duser ve sonunda silinir
- [ ] Trading uygulamasi:
      - "BTC 100K'da guclu direnç" -> trade basarili -> guven artar
      - "SOL 200'e cikacak" -> yanlis cikti -> guven duser, silinir

### 3.14 MAGMA (Multi-Graph Agentic Memory)
> 4 ortogonal grafik ile bilgiyi 4 farkli perspektiften organize et
> %45.5 daha yuksek reasoning dogrulugu, %95 token azaltma

- [ ] **Semantic Graph**: Bilginin anlamsal iliskisi
      "RSI oversold" --indicates--> "potansiyel reversal"
- [ ] **Temporal Graph**: Olaylarin zamansal sirasi
      "Fed karari" --precedes--> "piyasa tepkisi" --precedes--> "trend degisimi"
- [ ] **Causal Graph**: Neden-sonuc iliskileri
      "faiz artisi" --causes--> "dolar guclenme" --causes--> "BTC baski"
- [ ] **Entity Graph**: Varliklar arasi iliskiler
      "BTC" --correlates--> "ETH" --powers--> "DeFi TVL"
- [ ] Policy-guided traversal: Sorguya gore hangi grafikleri gezmeli?

### 3.15 Temporal-Aware RAG + TS-RAG
> Zaman duyarli retrieval + zaman serisi pattern eslestirme

- [ ] Temporal decay: skor *= exp(-lambda * yas_gun)
      - 1 saatlik haber: x1.0 (tam agirlik)
      - 1 gunluk haber: x0.9
      - 1 haftalik haber: x0.5
      - 1 aylik haber: x0.1
- [ ] Time window filtering: "Son 4 saat", "son 1 hafta", "son 3 ay"
- [ ] **TS-RAG (Time Series Pattern Retrieval)**:
      - Mevcut fiyat pattern'ini al
      - Tarihsel olarak benzer pattern'leri bul
      - Bu pattern'lerin sonuclarini getir
      - "Son 5 sefer BTC bu formasyon yaptığında ne oldu?"
      - Adaptive Retrieval Mixer (ARM): Benzerlik skoruna gore agirlikla
- [ ] StreamingRAG: Surekli akan veriyi gercek zamanli indeksle
      - Yeni mum verisi -> aninda embed + indeksle
      - Yeni haber -> aninda chunk + embed + indeksle
      - Sub-minute data freshness

### RAM Budget: ~5.7 GB ek (toplam ~22.7 GB)
```
Self-RAG modules:         ~200 MB
CRAG evaluator:           ~300 MB
LazyGraphRAG:             ~700 MB  (+200 MB, zengin entity graph)
Temporal Graph:           ~300 MB
CDF-RAG causal engine:    ~200 MB
MAGMA 4-graph memory:     ~1.0 GB  (+500 MB, 2 hafta derinlik)
GAM-RAG memory store:     ~500 MB  (+200 MB, daha fazla pattern)
TS-RAG pattern index:     ~1.0 GB  (+500 MB, 2x temporal pattern)
MADAM debate agents:      ~200 MB
Pipeline overhead:        ~1 GB
```

---

## PHASE 3.5: Autonomy Scaffolding (Week 13-14, 2 hafta)
> Autonomy altyapisi: Decision logging, position sizing engine, risk budget
> Bu altyapi olmadan Phase 4-6'daki AI trade ozellikleri CALISMAZ.

### 3.5.1 Decision Logging Engine
- [ ] Her AI karari icin kayit sistemi (Beta distribution trust icin gerekli)
      ```sql
      ai_decisions (
        id, timestamp, pair, signal_type, confidence,
        position_size, entry_price, model_used, rag_context_ids,
        reasoning_summary, regime, trust_score_at_decision,
        outcome_pnl, outcome_duration -- trade kapaninca doldurulur
      )
      ```
- [ ] Forgone P&L paper trade engine: Reddedilen/kuculttulen sinyalleri simule et
- [ ] SQLite tablosu: ForgoneProfit (pair, signal_time, confidence, was_executed, forgone_pnl)

### 3.5.2 Position Sizing Engine
- [ ] Confidence → position size pipeline (confidence^alpha curve)
- [ ] Bayesian Kelly hesaplama modulu (Beta distribution mean → Kelly fraction)
- [ ] Freqtrade hook entegrasyonu: `custom_stake_amount()` callback'ine bagla
- [ ] Trust level fractional Kelly: L2=0.10, L3=0.25, L4=0.50, L5=0.75
- [ ] ATR-based stop-loss mesafe hesaplama

### 3.5.3 Risk Budget System (Pod Model)
- [ ] Gunluk VaR butcesi hesaplama: portfolio * %1
- [ ] Trade basina butce tuketim: position_size * volatility * (1/confidence)
- [ ] Haftalik butce ayarlama: karli hafta multiplier artir, kayipli hafta azalt
- [ ] Aylik reset mekanizmasi
- [ ] Mutlak taban: butce ASLA base'in %25'unun altina dusmez

### 3.5.4 Autonomy Level Management
- [ ] Seviye tracking: Mevcut seviye (0-5) config + DB'de
- [ ] Yukseltme/dusurme kriterlerini otomatik kontrol (haftalik)
- [ ] Telegram bildirimi: Seviye degisikligi
- [ ] Dashboard widget: Mevcut seviye + terfi ilerleme cubugu

### RAM Budget: ~200 MB ek (negligible, mevcut buffer icinde)

---

## PHASE 4: Gemini-Powered Trading Brain (Week 15-16)
> API-first LLM: Gemini Flash + Groq + OpenRouter failover chain
> + Bidirectional RAG: Dogrulanmis bilgiyi geri yaz
> + MemoRAG: Global corpus bellegi

### 4.1 LLM Provider Setup (Failover Chain)
- [ ] **Gemini Flash** (birincil) - google-generativeai SDK
      - Ucretsiz tier: 1500 req/dk, 1M token/dk
      - Structured output (JSON mode) destegi
      - Grounding with Google Search destegi
      - **Prompt Caching**: Sistem prompt'u cache'le (%90 maliyet azaltma)
- [ ] **Groq** (ikincil fallback) - groq SDK
      - Ucretsiz tier: 30 req/dk, 14K tok/dk
      - Llama 3.3 70B: ~6000 tok/s
- [ ] **OpenRouter** (ucuncul fallback) - openai-compat SDK
      - 100+ model, kredi bazli
- [ ] **Failover Logic:**
      ```python
      async def llm_call(prompt, **kwargs):
          for provider in [gemini, groq, openrouter]:
              try:
                  return await provider.generate(prompt, **kwargs)
              except (RateLimitError, APIError):
                  continue
          # Tum API'ler down -> yerel LoRA model devreye
          return await local_lora.generate(prompt, **kwargs)
      ```

### 4.2 Trading Brain Roles (Gemini-Powered)
```
Gemini Flash Gorevleri (0 RAM, aninda yanit):

  1. RAG Generator: Retrieved context'ten analiz uret
  2. Query Transformer: Multi-query + HyDE icin sorgu uret
  3. CRAG Evaluator: Retrieval kalitesini degerlendir
  4. Contextual Chunker: Her chunk'a context ekle
  5. Entity Extractor: LazyGraphRAG icin entity cikar
  6. Sentiment Summarizer: 4 saatlik sentiment ozeti
  7. Trade Narrator: "Neden bu trade acildi?" aciklamasi
  8. Risk Assessor: Acik pozisyonlarin risk degerlendirmesi
  9. Market Regime Detector: Bull/Bear/Sideways tespiti
  10. News Analyzer: Haber etkisi ve onemi degerlendirmesi
  11. Causal Reasoner: CDF-RAG nedensellik degerlendirmesi
  12. Debate Moderator: MADAM-RAG tartisma yonetimi

Groq Gorevleri (failback + agir isler):
  13. Deep Analysis: Karmasik multi-hop analiz
  14. Strategy Debate: Bull vs Bear argumanlari (tam uzunluk)
  15. Weekly Report: Haftalik derin piyasa analizi
```

### 4.3 Bidirectional RAG (Kendi Kendini Gelistiren Bilgi Tabani)
> Dogrulanmis trade sonuclarini bilgi tabanina geri yaz
> Sistem zamanla kendi bilgi tabaninı zenginlestirir

- [ ] Grounding check: Yeni bilgi mevcut evidence ile tutarli mi?
      - NLI-based entailment check
      - Source attribution verification
      - Novelty detection (tekrar bilgi ekleme)
- [ ] Write-back kosullari:
      - Trade tamamlandi + sonuc belli
      - Analiz dogrulandi (min 3 onaylanmis ornek)
      - Piyasa pattern'i tekrarlandi (en az 2 kez)
- [ ] Ornek:
      "BTC, halving sonrasi 6 ay icinde ortalama %300 yukselir"
      -> 2024 halving sonrasi dogrulandi -> bilgi tabanina yaz
      -> Gelecek sorularda retrieval ile gelir

### 4.4 MemoRAG (Global Corpus Bellegi)
> Tum bilgi tabaninin "zihinsel haritasi"
> Implicit bilgi ihtiyaclarini tespit eder

- [ ] Hafif uzun-menzil LLM: Tum corpus'un KV-compressed bellegi
- [ ] Draft generation: Sorgu icin ipucu uret, retrieval'i yonlendir
- [ ] Global understanding: "Piyasanin genel durumu" gibi sorular icin
      - Flat retrieval bunu cevaplayamaz (cok genel)
      - MemoRAG: Tum corpus'u "bilir", ozet cikarabilir

### 4.5 Scheduled Tasks
```
Her 5 dakika:  Sentiment feature guncelleme (CryptoBERT/FinBERT - YEREL)
Her 5 dakika:  StreamingRAG indeksleme (yeni mumlar, haberler)
Her 15 dakika: RAG-enhanced teknik analiz snapshot (Gemini Flash)
Her 15 dakika: Semantic cache invalidation check
Her 1 saat:    Piyasa ozeti raporu (Gemini Flash)
Her 1 saat:    LazyGraphRAG entity/iliski guncelleme
Her 4 saat:    Derin analiz + strateji parametresi onerisi (Gemini Flash)
Her 4 saat:    GAM-RAG memory consolidation
Her gun:       Performans degerlendirmesi + ogrenme (Gemini Flash)
Her gun:       Bidirectional RAG write-back (dogrulanmis bilgiler)
Her gun:       Temporal decay uygula (eski bilgileri onceliksizlestir)
Her hafta:     Haftalik strateji review (Groq - Llama 70B)
Her hafta:     RAPTOR tree yeniden olustur
```

### 4.6 Background: Yerel Model Distillation (Gemini -> Local LoRA + DPO)
> Gemini'nin gercek trade kararlari + sonuclari ile yerel modeli egit
> Sentetik veri DEGIL, gercek piyasa verisi ile ogrenme
> DPO ile tercih ogrenme: basarili trade > basarisiz trade

- [ ] **Veri Toplama (Otomatik, arka planda):**
      Her Gemini karari kaydedilir:
      ```json
      {
        "timestamp": "2026-03-15T14:30:00Z",
        "pair": "BTC/USDT",
        "gemini_analysis": "RSI oversold + bullish divergence + positive news",
        "gemini_decision": "LONG",
        "confidence": 0.85,
        "rag_context": ["haber1", "haber2", "teknik_sinyal"],
        "result": "WIN",
        "pnl_percent": 2.3,
        "market_regime": "bullish"
      }
      ```
- [ ] **Egitim Verisi Biriktirme:**
      - Minimum 500 trade karari birikene kadar bekle (~2-4 hafta)
      - Basarili (WIN) ve basarisiz (LOSS) kararlari dengeli tut
- [ ] **Yerel Model:** Qwen3-0.6B veya SmolLM2-1.7B
      - QLoRA: 4-bit quantized base + LoRA adapter (~1-2 GB RAM)
- [ ] **SFT + DPO Pipeline (2026 best practice):**
      ```
      Adim 1: SFT (Supervised Fine-Tuning)
        - Gemini'nin basarili analizleri ile egit
        - "Boyle bir piyasa durumunda su analizi yap"

      Adim 2: DPO (Direct Preference Optimization)
        - WIN/LOSS ciftleri ile tercih ogrenme
        - Basarili trade analizi > basarisiz trade analizi
        - RLHF'den daha basit, ayni kalite

      Adim 3: RAG entegrasyonu
        - SFT+DPO modeli RAG pipeline icinde calisir
        - API down olursa yerel model devreye
      ```
- [ ] **Egitim Dongusu (Haftalik, Pazar gecesi):**
      ```
      Pazar 03:00 -> Egitim baslar
        v
      Son haftanin Gemini kararlarini yukle (50-100 ornek)
        v
      SFT + DPO egitim (CPU, ~2-4 saat)
        v
      Validation: Gecmis haftanin %20'si ile test
        v
      Basarili -> LoRA adapter guncelle | Basarisiz -> eski adapter koru
        v
      Pazar 07:00 -> Tamamlandi
      ```

### 4.7 API Cost Estimation
```
Gemini Flash (ucretsiz tier):
  - 1500 req/dk = gunluk ~2M request mumkun
  - Bizim kullanim: ~500-1000 req/gun (cok rahat)
  - Prompt Caching ile: %90 maliyet azaltma

Groq (ucretsiz tier):
  - 30 req/dk = gunluk ~43K request mumkun
  - Bizim kullanim: ~50-100 req/gun

OpenRouter (ucretli fallback):
  - Sadece acil durumlarda, ~$0.001/req
  - Aylik maliyet tahmini: <$5
```

### RAM Budget: ~5.7 GB ek (toplam ~28.4 GB)
```
API SDK'lari:         ~100 MB
HTTP connection pool: ~200 MB
Response cache:       ~500 MB  (+300 MB, buyuk cache)
Karar log DB:         ~200 MB
MemoRAG global memory: ~700 MB  (+400 MB, zengin global memory)
Bidirectional RAG:    ~200 MB
LoRA Training:        ~2.5 GB  ALWAYS-RESIDENT (buffer'dan calmaz, batch 32)
Yerel Model Fallback: ~1.0 GB  ALWAYS-RESIDENT (aninda failover, her zaman sicak)
-- Her zaman: ~5.4 GB (LoRA+Model dahil, Pazar gecesi ek RAM yok) --
```

---

## PHASE 5: Multi-Agent Trading System (Week 17-18)
> TradingAgents-inspired + MADAM-RAG debate + Agentic RAG

### 5.1 Agent Architecture (LangGraph + Hierarchical RAG)
```
                    +-------------------+
                    |   ORCHESTRATOR    |
                    | (Ana Koordinator) |
                    +--------+----------+
                             |
        +--------------------+--------------------+
        |                    |                    |
   +----v----+         +----v----+         +----v----+
   |ANALYST  |         |RESEARCH |         |EXECUTION|
   | TEAM    |         |  TEAM   |         |  TEAM   |
   +----+----+         +----+----+         +----+----+
        |                    |                    |
   +----+----+          +---+---+          +----+----+
   |    |    |          |       |          |    |    |
   v    v    v          v       v          v    v    v
 Tech Senti News     Bull   Bear       Trader Risk  PM
 Anal  Anal  Anal    Rsrch  Rsrch             Mgr
```

### 5.2 Agent Roles (RAG-Enhanced)
- [ ] **Technical Analyst Agent**:
      - Retrieval: TS-RAG + tool-augmented (canli gostergeler)
      - Reasoning: CoT-RAG ile adim adim analiz
- [ ] **Sentiment Analyst Agent**:
      - Retrieval: CryptoBERT/FinBERT + haber RAG
      - Memory: GAM-RAG ile sentiment trend bellegi
- [ ] **News Analyst Agent**:
      - Retrieval: CRAG ile haberleri filtrele + dogrula
      - Graph: LazyGraphRAG ile entity iliskilerini coz
- [ ] **Bullish Researcher**: RAG-Fusion ile yukselis argumanlarini topla
- [ ] **Bearish Researcher**: RAG-Fusion ile dusus argumanlarini topla
- [ ] **Trader Agent**:
      - MADAM-RAG debate sonuclarini sentezle
      - Speculative RAG ile coklu senaryo degerlendir
      - CDF-RAG ile nedensellik kontrol et
- [ ] **Risk Manager Agent**:
      - Tool-augmented: Portfolio durumu, max drawdown kontrolu
      - Temporal-aware: Risk metrikleri zaman serisi
- [ ] **Portfolio Manager Agent**:
      - MemoRAG: Global portfolio state bellegi
      - Bidirectional RAG: Basarili stratejileri kaydet

### 5.3 Agent Communication (A-RAG Pattern)
- [ ] Hierarchical retrieval interfaces:
      - keyword_search: Kesin terim arama
      - semantic_search: Anlamsal benzerlik arama
      - chunk_read: Belirli bir chunk'i oku
      - tool_call: API/hesaplama cagir
      - graph_query: LazyGraphRAG traversal
- [ ] LangGraph state machine ile agent orkestrasyon
- [ ] Shared memory: MAGMA 4-graph + ChromaDB + SQLite
- [ ] Structured output: JSON schema ile agent ciktilari
- [ ] Debate protocol: MADAM-RAG ile Bull vs Bear

### RAM Budget: ~4.4 GB ek (toplam ~31.7 GB = tam kapasite)
```
LangGraph runtime:    ~1.0 GB  (+500 MB, 7 concurrent agent)
Agent state machines: ~700 MB  (+200 MB, daha buyuk working memory)
MADAM debate engine:  ~700 MB  (+400 MB, uzun debate context)
Shared memory:        ~700 MB
Communication buffer: ~1.3 GB  (+300 MB, parallel debate threads)
```

---

## PHASE 6: Feedback Loop & Self-Improvement (Week 19-22, 4 hafta)
> GAM-RAG + Bidirectional RAG + RAGAS + DeepEval = kendi kendini iyilestiren sistem
> FELSEFE: Hata yaparsa yapsin! Hatasini logla, kisa vadede para kaybetsin,
> uzun vadede akillansin. Engellemek = ogrenme firsatini OLDURUR.

### 6.1 Performance Tracking ("Her Hata Bir Ders")
- [ ] Her trade'in tam RAG context'i + LLM analizi + sonucu + confidence + rejim kaydet
- [ ] GAM-RAG ile basarili pattern'leri otomatik bellete ekle
- [ ] Basarisiz pattern'leri "lessons learned" olarak kaydet (SILME, ogrenme materyali!)
- [ ] Bidirectional RAG: Dogrulanmis insight'lari bilgi tabanina yaz
- [ ] **Hata Kategorilendirme** (her kayipli trade):
      ```
      HATA TURU           | AKSIYON                    | ORNEK
      --------------------|----------------------------|---------------------------
      Hallucination       | Confidence formulu ayarla  | LLM sahte haber uydurdu
      Rejim yanilgisi     | HMM parametrelerini guncelle | Bull'da bear sinyali
      Timing hatasi       | Alpha decay weight guncelle | Gec girdi, sinyal coktu
      Position sizing     | Kelly parametrelerini guncelle | Pozisyon cok buyuk/kucuk
      Korelasyon hatasi   | Correlation matrix guncelle | 3 korele pozisyon birden
      Dogru analiz, kotu sonuc | HICBIR SEY YAPMA (market noise) | Analiz dogru ama piyasa irrasyonel
      ```
- [ ] **Forgone P&L Raporu** (haftalik):
      - Reddedilen/kuculttulen sinyallerin gercek sonucu ne olurdu?
      - Guardrail'lar DEGER YARATIYOR MU yoksa YOKEDIYOR MU?
      - Bu TEK EN ONEMLI diagnostik rapor

### 6.2 RAG Quality Monitoring (4 Katmanli)
- [ ] **RAGAS Framework**:
      - Faithfulness: Yanit retrieved evidence'a dayaniyor mu? (hedef: >0.90)
      - Context Precision: Retrieval uygun mu? (hedef: >0.85)
      - Answer Relevancy: Yanit soruyu cevapliyor mu? (hedef: >0.90)
- [ ] **DeepEval** (CI/CD Entegrasyonlu):
      - FaithfulnessMetric: Hallucination olcumu
      - HallucinationMetric: Fabricated bilgi tespiti
      - ContextualRecallMetric: Retrieval coverage
      - pytest entegrasyonu: Her deployment oncesi otomatik test
- [ ] **RAG Triad (TruLens)**:
      - Context Relevance -> Retriever'i degerlendir
      - Groundedness -> Generator'u degerlendir
      - Answer Relevance -> End-to-end kalite
      - Dusuk skor -> otomatik alert + hangi bileseni fix'le
- [ ] **Lynx Hallucination Guardian**:
      - Trade sinyali oncesi son kontrol
      - Hallucinated fiyat/event tespiti
      - Hallucination tespit -> confidence DUSUR (pozisyon kucultme, BLOKLAMA degil)

### 6.3 Strategy Adaptation
- [ ] Haftalik otomatik hyperopt (en basarili parametre araliklari)
- [ ] Piyasa rejimi tespiti (bull/bear/sideways) -> strateji degisimi
- [ ] Adaptive RAG: Basarili retrieval pattern'lerini onceliklendir
- [ ] GAM-RAG memory konsolidasyonu: Haftalik bellek temizligi

### 6.4 Continuous Learning
- [ ] SFT + DPO ile LoRA adaptoru guncelle (haftalik)
- [ ] BGE-Financial fine-tune on crypto data (aylik, eger gerekiyorsa)
- [ ] Pairlist optimizasyonu (RAG-informed pair secimi)
- [ ] RAPTOR tree yeniden olusturma (haftalik)
- [ ] LazyGraphRAG entity/iliski guncelleme (gunluk)

---

## PHASE 7: Deep Analysis & Expansion (Week 23+)
> Groq ile derin analiz + sistem olgunlastirma

### 7.1 Weekly Deep Analysis (Groq - Llama 3.3 70B)
```
Pazar 02:00 -> Haftalik analiz baslar (Groq API, ~6000 tok/s)
  |
  v
Tum haftanin verilerini topla:
  - 7 gunluk haber ozeti (RAG'dan)
  - Teknik gosterge trendleri
  - On-chain metrikleri
  - Sentiment trend degisimi
  - LazyGraphRAG entity iliski degisimleri
  - CDF-RAG nedensellik analizi
  - GAM-RAG bellek ozeti
  - Gecmis haftanin trade performansi
  |
  v
Groq Llama-70B analiz eder (~2-3 dakika, API hizinda):
  - Haftalik piyasa raporu
  - MAGMA 4-graph analizi
  - Onumuzdeki hafta icin tahminler
  - Strateji degisikligi onerileri
  - Risk uyarilari
  |
  v
Sonuclari Bidirectional RAG ile bilgi tabanina yaz
Strateji parametrelerini guncelle
Telegram'a haftalik rapor gonder
Pazar 02:05 -> Tamamlandi
```

### 7.2 Opsiyonel: AirLLM Offline Analysis
- [ ] Sadece internet kesintisi senaryosu icin
- [ ] Qwen2.5-7B-Q4 (~5 GB disk, hafif) yeterli
- [ ] Gemini + Groq + OpenRouter + yerel LoRA hepsi down olursa devreye girer

---

## RAM ALLOCATION SUMMARY (32 GB ECC, 5-Yil Surdurulebilir)

```
Component                       RAM (GB)    Phase   24GB'den Fark
------------------------------------------------------------------------
OS + System                     2.0         -       -
Freqtrade Core                  1.5         0       -
FreqUI + API Server             0.5         0       -
CryptoBERT (ONNX)               0.4         1       -
FinBERT (ONNX)                  0.4         1       -
News Pipeline (RSS+SSE+API)     0.5         0       +0.2 (buyuk dedup cache)
Scrapling (standby, browser off) 0.05       0       -
Data Pipeline + SQLite          0.5         1       -
BGE-Financial Embedding         0.5         2       -
Gemini Embed cache              0.5         2       +0.2 (yuksek hit rate)
all-MiniLM-L6-v2 (fallback)    0.1         2       -
ColBERTv2 (PLAID compressed)   0.4         2       -
BM25 Index                      0.3         2       +0.1 (buyuk inverted index)
ChromaDB (500K+ docs)           5.0         2       +2.0 (HNSW recall +5-8%)
FlashRank Reranker              0.3         2       -
Semantic Cache                  1.5         2       +1.0 (3x entry, hit %80)
RAPTOR tree + optimization      0.5         2       -
LangChain/Pipeline              0.5         2       -
Self-RAG + CRAG modules         0.5         3       -
LazyGraphRAG                    0.7         3       +0.2 (zengin entity graph)
Temporal Graph + CDF-RAG        0.5         3       -
MAGMA 4-graph memory            1.0         3       +0.5 (2 hafta derinlik)
GAM-RAG memory store            0.5         3       +0.2 (daha fazla pattern)
TS-RAG pattern index            1.0         3       +0.5 (2x temporal pattern)
Advanced RAG pipeline           0.5         3       -
MemoRAG + Bidirectional RAG     0.7         4       +0.2 (zengin global memory)
API SDK + cache                 0.5         4       +0.1 (buyuk response cache)
LoRA Training (always-resident) 2.5         4       +2.5 YENI: buffer'dan calmaz
Yerel Model Fallback (standby)  1.0         4       +1.0 YENI: aninda failover
LangGraph + Agents              3.0         5       +1.0 (7 concurrent agent)
MADAM debate + comm             0.7         5       +0.2 (uzun debate context)
Monitoring + Evaluation         0.3         6       -
LiteLLM + Prometheus + Grafana  0.5         0+      -
Buffer / Headroom               3.15        -       +2.3 (OOM riski sifir)
------------------------------------------------------------------------
TOTAL                         ~31.7 / 32 GB

Pazar gecesi (egitim):
  LoRA egitim:              ZATEN RESIDENT (2.5 GB, ek RAM gerekmiyor!)
  Yerel model fallback:    ZATEN RESIDENT (1.0 GB, ek RAM gerekmiyor!)
  Scrapling browser:        +0.5 GB (buffer'dan, artik CALISABILIR)
  Pazar gecesi toplam:     ~32.2 GB (0.5 buffer kullanim, 2.65 GB bos)

UYARI @%88 (28.2 GB): ChromaDB -> memory-mapped mode
KRITIK @%94 (30.1 GB): Semantic cache eviction + agent sayisi 5'e dusur
swappiness=30 (rahat ama hala swap-averse)

KALDIRILAN KISITLAMALAR (24 GB'dan gelen):
  [X] "Scrapling browser + LoRA ASLA ayni anda calismaz" -> KALDIRILDI
  [X] "Pazar gecesi Scrapling devre disi" -> KALDIRILDI
  [X] "ChromaDB memory-mapped zorunlu" -> Ihtiyari (>700K doc'ta)
  [X] "swappiness=10" -> swappiness=30'a rahatlatildi
  [X] "Phase 5 sonrasi 32 GB upgrade" -> ZATEN 32 GB
```

---

## DISK ALLOCATION (160 GB E-NVMe, 5-Yil Surdurulebilir)

```
Component                    Disk (GB)   Rolling/Purge Politikasi
-----------------------------------------------------
OS + System                  10          -
Docker + Images              15          docker image prune --all (aylik)
Freqtrade + Dependencies     5           -
Scrapling Browser Binaries   0.5         Versiyon guncelleme: eskiyi sil
ML Models (ONNX + embeddings) 4
  - CryptoBERT: 0.5
  - FinBERT: 0.5
  - BGE-Financial: 0.5
  - ColBERTv2: 0.5
  - FlashRank: 0.3
  - all-MiniLM-L6-v2: 0.1
  - LoRA adapters: 0.5       Son 8 adapter tut, eskileri sil (auto-purge)
  - Qwen3-0.6B quantized: 0.6
ChromaDB Data (dual embedding) 25        AUTO-COMPACT: 1M doc cap, eskiyi temporal decay ile sil
  - Binary quantized index: 2
  - Full precision index: 23
  - COMPACT KURALI: >1M doc olursa en dusuk gain-score %20'yi sil
  - VACUUM: Haftalik (Pazar 03:00), fragmantasyonu temizle
SQLite Databases             8           VACUUM + WAL checkpoint (haftalik)
  - Embedding cache: 2        LRU eviction, max 2 GB hard cap
  - Semantic cache: 2         TTL-based eviction (30 gun max)
  - GAM-RAG memory: 1         Gain-score threshold: dusuk gain -> purge
  - MAGMA graphs: 1           Node pruning: 6 aydir dokunulmayan edge sil
  - Decision log: 1           1 yil rolling window, ozet tut detay sil
  - Trade history: 1          ASLA SILME (audit trail, sonsuz)
Historical Price Data        30          AUTO-PURGE: 1m data 1 yil, 5m 3 yil, 15m+ sonsuza kadar
                                          Eski 1m veri silinince ~4 GB/yil geri kazanilir
News Archive (180 days)      18          180 gun rolling window (otomatik, cron)
LazyGraphRAG graph data      5           SNAPSHOT PURGE: Son 6 snapshot tut, eskileri sil
TS-RAG pattern archive       5           RELEVANCE PURGE: Sharpe<0 pattern'lari 90 gun sonra sil
RAPTOR tree snapshots        3           Son 4 tree versiyonu tut
Backtest Results             8           1 yil rolling, sadece best-of tut (top %10)
Logs + Monitoring            5           90 gun rolling (logrotate)
LLM Observability Logs       4           90 gun rolling (logrotate)
FreqUI Build                 1           -
Prompt Templates + Versions  1           Git versioning, disk'te sadece aktif versiyon
Buffer / Growth Headroom     13
-----------------------------------------------------
TOTAL                        ~160 / 160 GB

5-YILLIK BUYUME PROJEKSIYONU:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Bilesen              Yil 1    Yil 2    Yil 3    Yil 5    Purge ile
──────────────────────────────────────────────────────────────────
ChromaDB             30 GB    35 GB    40 GB    50 GB    25 GB (1M cap)
Historical Price     36 GB    42 GB    48 GB    60 GB    35 GB (1m purge)
News Archive         18 GB    18 GB    18 GB    18 GB    18 GB (180d rolling)
SQLite DBs           9 GB     10 GB    11 GB    13 GB    10 GB (rolling+vacuum)
Logs                 5 GB     5 GB     5 GB     5 GB     5 GB (90d rolling)
LLM Logs             4 GB     4 GB     4 GB     4 GB     4 GB (90d rolling)
Backtest             10 GB    12 GB    14 GB    18 GB    8 GB (top %10 only)
Graph data           6 GB     7 GB     8 GB     10 GB    6 GB (snapshot purge)
TS-RAG               5.5 GB   6 GB     6.5 GB   7 GB     5 GB (relevance purge)
──────────────────────────────────────────────────────────────────
PURGE OLMADAN 5 yil: ~200 GB (DISK PATLAK!)
PURGE ILE 5 yil:     ~140 GB (DISK %88, guvenli)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

AUTO-MAINTENANCE CRON (dokunmadan calisir):
  Her gun 04:00:   News archive 180 gun otesindekileri sil
  Her gun 04:00:   Log rotation (90 gun)
  Her Pazar 03:00: SQLite VACUUM + WAL checkpoint
  Her Pazar 03:00: ChromaDB compact + HNSW optimize
  Her Pazar 03:00: Docker image prune
  Her ay 1.:       LoRA adapter purge (son 8 tut)
  Her ay 1.:       LazyGraphRAG eski snapshot sil (son 6 tut)
  Her ay 1.:       RAPTOR tree purge (son 4 tut)
  Her ay 1.:       Backtest results purge (top %10 tut)
  Her ay 1.:       TS-RAG irrelevant pattern purge (Sharpe<0, >90 gun)
  Her ay 1.:       MAGMA graph edge pruning (6 aydir dokunulmayan)
  Her ay 1.:       Historical 1m data purge (>1 yil)
  Her 3 ay:        Disk usage raporu -> Telegram alert
  ALARM:           Disk %85 -> Telegram uyari
  KRITIK:          Disk %90 -> Agresif purge (tum rolling window'lari %50 kisalt)

RAM AUTO-MANAGEMENT (dokunmadan calisir):
  ChromaDB:        >1M doc -> otomatik low-gain doc eviction
  MAGMA graphs:    >10K node -> en az kullanilan %10 edge prune
  GAM-RAG:         >50K pattern -> gain-score en dusuk %20 sil
  Semantic cache:  LRU eviction, max 1.5 GB hard cap (asla asma)
  Gemini cache:    LRU eviction, max 0.5 GB hard cap
  News dedup:      Bloom filter reset her 30 gun (false positive birikimi onle)
  UYARI @%88:      ChromaDB -> memory-mapped mode (RAM'den disk'e)
  KRITIK @%94:     Semantic cache %50 evict + agent sayisi 5'e dusur
```

---

## RAG QUALITY EVOLUTION (Updated: 42 Teknik ile)

```
Phase 0: No RAG                                    0/10
Phase 1: Sentiment-only (not really RAG)            1/10
Phase 2: Hybrid RAG Foundation (14 teknik)          6/10
  + Dual embedding (Gemini + BGE-Financial)
  + Binary quantization (32x depolama tasarrufu)
  + Hybrid search (dense + BM25 + ColBERT + RRF)
  + Multi-reranker ensemble (FlashRank + ColBERT)
  + 6 chunking stratejisi (recursive, contextual, parent-child,
    RAPTOR, late, proposition)
  + Tool-augmented RAG (canli API erisimi)
  + Semantic caching (65x latency azaltma)
  + Prompt caching (%90 API maliyet azaltma)
  + Intelligent query routing (%30-45 maliyet azaltma)
Phase 3: Advanced RAG Techniques (15 teknik ek)     8.5/10
  + Self-RAG (kosullu retrieval, self-critique)
  + CRAG Enhanced (kalite degerlendirme + duzeltme)
  + Adaptive RAG (akilli yonlendirme)
  + RAG-Fusion (multi-query perspektif)
  + HyDE (hipotetik dokuman)
  + Speculative RAG (draft-verify)
  + LazyGraphRAG (entity iliskileri, %0.1 maliyet)
  + TG-RAG (zamansal graf)
  + CDF-RAG (nedensellik reasoning)
  + MADAM-RAG (celisken sinyal debate)
  + FLARE (aktif retrieval)
  + CoT-RAG (aciklanabilir reasoning)
  + GAM-RAG (ogrenen bellek)
  + MAGMA (4-grafik bellek)
  + TS-RAG + Temporal Decay (zaman serisi)
Phase 4: LLM-Powered + Bidirectional + MemoRAG      9.2/10
  + Gemini Flash generation (API, cok zeki)
  + Bidirectional RAG (self-improving knowledge base)
  + MemoRAG (global corpus bellegi)
  + SFT + DPO local model (knowledge distillation)
Phase 5: Multi-Agent RAG                            9.6/10
  + 8 specialized agent (her biri RAG-enhanced)
  + MADAM-RAG debate (celisken sinyallerde)
  + Agentic RAG (hierarchical retrieval interfaces)
  + Shared MAGMA memory across agents
Phase 6: Self-Improving + Monitored                 9.8/10
  + RAGAS + DeepEval + RAG Triad (3 evaluation framework)
  + Lynx hallucination guardian
  + GAM-RAG continuous learning
  + Bidirectional RAG write-back
  + SFT + DPO weekly retraining
  + Adaptive strategy optimization
```

### 9.8/10 vs Basic 1/10: Tam Karsilastirma (42 Teknik)

| Boyut | Basic RAG (1/10) | Bizim Sistem (9.8/10) |
|-------|-----------------|---------------------|
| Embedding | Tek genel model | Dual (Gemini + BGE-Financial) + Matryoshka + Binary Quant |
| Chunking | Sabit 500 char | 6 strateji: Recursive + Contextual + Parent-Child + RAPTOR + Late + Proposition |
| Search | Sadece dense | Dense + BM25 + ColBERTv2 + RRF (3 yontemli hybrid) |
| Reranking | Yok | FlashRank + ColBERT multi-reranker ensemble |
| Query | Ham sorgu | Multi-query + HyDE + Step-back + Decomposition + Routing + Rewriting (7 teknik) |
| Evaluation | Yok | Self-RAG critique + CRAG evaluator (uretim oncesi) |
| Correction | Yok | CRAG corrective actions + web search fallback |
| Graph | Yok | LazyGraphRAG + Temporal Graph + CDF-RAG causal graph |
| Memory | Stateless | GAM-RAG + MAGMA 4-graph + MemoRAG global (3 bellek sistemi) |
| Temporal | Zaman-kor | Temporal decay + TS-RAG pattern + StreamingRAG real-time |
| Causal | Yok | CDF-RAG nedensellik + MADAM-RAG debate |
| Citation | Yok | Attributed RAG + tam audit trail |
| Self-Improve | Yok | Bidirectional RAG write-back + GAM-RAG gain-adaptive |
| Generator | Tek LLM | Gemini + Groq + OpenRouter failover + yerel LoRA fallback |
| Training | Yok | SFT + DPO ile haftalik distillation |
| Evaluation | Yok | RAGAS + DeepEval + RAG Triad + Lynx (4 framework) |
| Agents | Yok | 8 specialized agent + MADAM debate + Agentic RAG interfaces |
| Caching | Yok | Semantic cache (65x) + Prompt cache (%90) + Embedding cache |
| Routing | Hep ayni | Intelligent routing: no-RAG / simple / medium / complex |
| Hallucination | Kontrolsuz | Self-RAG + CRAG + Lynx + faithfulness check + citation (5 katman) |
| Forecasting | Yok | TS-RAG pattern retrieval + FinSeer-inspired similarity |
| Live Data | Yok | Tool-augmented RAG (canli API retrieval) |

---

## PROMPT ENGINEERING ARCHITECTURE
> Prompt = sistemin beyni. Kotu prompt = kotu trade. Iyi prompt = tutarli, aciklanabilir, guvenilir sinyaller.

### System Prompt Tasarimi (Contract Format)
```
Her LLM cagrisinin system prompt'u su yapida olmali:

ROLE: Senin rolu ve uzmanligin (spesifik, yil tecrubesi dahil)
SUCCESS CRITERIA: Basari olcutleri (Sharpe > 1.5, max %2 risk/trade)
CONSTRAINTS: Kati kurallar (oncelik sirasina gore)
  1. Sermayeyi koru (birincil)
  2. Sermayeyi buyut (ikincil)
  3. Sadece saglanan verileri kullan, ASLA tahmin yapma
  4. Belirsizlik = KUCUK pozisyon, trade acmayi ENGELLEME
UNCERTAINTY HANDLING: Belirsizlik durumunda ne yap
OUTPUT FORMAT: JSON schema (Pydantic ile enforce)
```

### Chain-of-Thought (CoT) Sablonlari
Her trade analizi su adimlari ZORLA takip etmeli:
```
Adim 1: Market Rejimi Tespiti
  -> {regime: trending_bullish|trending_bearish|ranging|volatile|compression}
  -> {confidence: 0-1, evidence: [cite specific data]}

Adim 2: Teknik Analiz
  -> {indicators: {rsi, macd, bb, volume}, confluence_score: 0-5}
  -> {key_levels: {support: [], resistance: []}}

Adim 3: Sentiment Degerlendirme
  -> {sentiment_score: -1 to +1, alignment_with_regime: bool}

Adim 4: Risk Degerlendirme
  -> {rr_ratio, atr_stop, position_size_pct, portfolio_impact}

Adim 5: Karar
  -> {action: LONG|SHORT|HOLD, entry, sl, tp, confidence, reasoning}
```

### FinCoT (Financial Chain-of-Thought)
- Mermaid diagram olarak expert reasoning pathway embed et
- %63.2 -> %80.5 dogruluk artisi (Qwen3-8B uzerinde)
- Cikti uzunlugu 8.9x azalma (maliyet tasarrufu!)
- Zero-shot (fine-tuning gerektirmez)

### Anti-Hallucination Prompting (5 Katman)
```
Katman 1: Strict Grounding
  "SADECE saglanan verileri kullan. ASLA fiyat, tarih, gosterge tahmini yapma.
   Veri eksikse 'DATA NOT AVAILABLE' yaz."

Katman 2: Source Attribution
  "Her iddia icin kaynak belirt:
   'RSI(14) = 72.3, kaynak: Binance 4H OHLCV, 2026-03-07 12:00 UTC'"

Katman 3: Chain-of-Verification (CoVe)
  Analiz uret -> Dogrulama sorulari uret -> Yanit kontrol -> Tutarsizlik tespit

Katman 4: Uncertainty Disclosure
  "Guvenin dusukse acikca belirt ve pozisyon boyutunu otomatik kucult.
   Dusuk guvenle kucuk pozisyon AC, hatasindan OGREN. Trade'i ENGELLEME."

Katman 5: Tool-Grounded Data
  "Tum piyasa verilerini TOOL CALL ile al. Hafizadan ASLA fiyat/event hatirlatma."
```

### Prompt Sablonlari (Gorev Bazli)

**Giris Analizi:**
```json
{
  "task": "entry_analysis",
  "inputs": ["regime", "ohlcv", "indicators", "sentiment"],
  "output_schema": {
    "entry_justified": "bool",
    "entry_price": "float",
    "stop_loss": "float",
    "take_profit": "float",
    "position_size_pct": "float",
    "rr_ratio": "float",
    "confluence_score": "int (0-5)",
    "confidence": "float (0-1)",
    "reasoning": "string",
    "invalidation": "string"
  },
  "temperature": 0.1,
  "max_tokens": 500
}
```

**Cikis Analizi:** entry + current_price + unrealized_pnl -> HOLD/TIGHTEN/PARTIAL/CLOSE
**Risk Degerlendirme:** portfolio_state + proposed_trade -> APPROVE/DOWNSIZE/MONITOR
**Rejim Tespiti:** ohlcv_30d -> regime + confidence + transition_signals
**Sentiment Skorlama:** news_items -> per_item_score + aggregate + trend

### Multi-Agent Role Prompting (TradingAgents Pattern)
```
Agent 1: Technical Analyst (T=0.2, top-p=0.80)
  "30 yillik algoritmik trader. Trend-following specialist."

Agent 2: Sentiment Analyst (T=0.3, top-p=0.85)
  "Kripto sosyal medya ve haber analisti. CryptoBERT/FinBERT veri yorumlayici."

Agent 3: Bullish Researcher (T=0.5, top-p=0.90)
  "Firsatlari ve buyume potansiyelini savunan iyimser arastirmaci."

Agent 4: Bearish Researcher (T=0.5, top-p=0.90)
  "Riskleri ve zayifliklari sorgulayan kusku arastirmaci."

Agent 5: Risk Manager (T=0.1, top-p=0.75)
  "Muhafazakar risk yoneticisi. Kurallar keskin, taviz yok."

Agent 6: Trader (T=0.2, top-p=0.80)
  "Tum analizleri sentezleyen karar verici. Self-consistency ile 3-5x calistir."

Agent 7: News Analyst (T=0.2, top-p=0.80)
  "Kripto haber analisti. CRAG ile haberleri filtrele, LazyGraphRAG ile entity iliskilerini coz."

Agent 8: Portfolio Manager (T=0.15, top-p=0.80)
  "FINCON modeli: TEK karar verici. MemoRAG global portfolio bellegi. Bidirectional RAG ile basarili stratejileri kaydet. Diger ajanlar DANISMANDIR, PM override edebilir."
```

### Reflective Few-Shot (CryptoTrade Pattern)
- Statik ornekler YERINE dinamik gecmis trade sonuclari kullan
- Her 10 trade'de reflection: basarili pattern'leri kural olarak kaydet
- Basarisiz pattern'leri anti-pattern olarak kaydet
- ChromaDB'de sakla, gelecek prompt'lara enjekte et

### Self-Consistency (Coklu Ornekleme)
- Ayni analizi 3-5 kez calistir (farkli temperature)
- Cogunluk oyu ile karar ver
- 4/5 HOLD + 1/5 LONG = HOLD (daha guvenilir)
- Maliyet: 3-5x ama kritik kararlar icin deger

---

## PARAMETRIC PLANNING (LLM + RAG + Trading Parametreleri)
> Her parametre belgeli, test edilmis, rejime gore ayarlanabilir olmali

### LLM Hyperparametreler (Gorev Bazli)

| Gorev | Temperature | Top-p | Max Tokens | Freq Penalty |
|-------|-------------|-------|------------|--------------|
| Trade sinyali (BUY/SELL/HOLD) | 0.05-0.15 | 0.75-0.80 | 50-200 | 0.1 |
| Giris/cikis analizi + reasoning | 0.2-0.3 | 0.80-0.85 | 500-1,000 | 0.2 |
| Piyasa ozeti / gunluk brifing | 0.4-0.5 | 0.85-0.90 | 1,500-3,000 | 0.3 |
| Strateji uretimi / beyin firtinasi | 0.7-0.9 | 0.90-0.95 | 1,000-2,000 | 0.1 |
| Risk degerlendirme | 0.1-0.2 | 0.75-0.85 | 500-1,000 | 0.2 |
| Rejim tespiti | 0.2-0.3 | 0.80-0.85 | 300-500 | 0.2 |
| Entity extraction (GraphRAG) | 0.0-0.1 | 0.75 | 200-500 | 0.1 |

**Kural**: Temperature VEYA top-p ayarla, IKISINI AYNI ANDA degistirme!
**Kural**: Output token'lar input'tan 4-8x daha pahali -> max_tokens siki tut!

### Market Rejim-Bazli Parametre Degisimi

| Rejim | LLM Temp | RAG Alpha | Retrieval K | Temporal Half-Life | Pozisyon Limiti |
|-------|----------|-----------|-------------|--------------------|----|
| Trending | 0.2-0.4 | 0.5-0.6 | 10-15 | 7-14 gun | 8-12 |
| Ranging | 0.4-0.6 | 0.6-0.7 | 15-20 | 14-21 gun | 4-6 |
| Volatile | 0.1-0.2 | 0.4-0.5 | 20-30 | 3-7 gun | 2-4 |
| Quiet | 0.3-0.5 | 0.7 | 10 | 14 gun | 6-10 |

### RAG Pipeline Parametreleri

| Parametre | Baslangic Degeri | Aralik | Aciklama |
|-----------|-----------------|--------|----------|
| Retrieval top-k | 15 | 10-30 | Sorgu karmasikligina gore |
| Reranked top-k | 5 | 3-10 | LLM'e verilen final chunk sayisi |
| Chunk size (haber) | 512 token | 256-1024 | Benchmark: %69 dogruluk @512 |
| Chunk overlap | %15 | %10-20 | Recursive chunking icin |
| Temporal alpha | 0.7 | 0.4-0.7 | Semantik vs recency agirligi |
| Temporal half-life (haber) | 7 gun | 1-14 | Kripto haberleri icin |
| Temporal half-life (rapor) | 60 gun | 30-90 | Ceyreklik raporlar icin |
| Similarity threshold | 0.5 | 0.3-0.7 | Embedding modeline gore kalibre et |
| Semantic cache threshold | 0.92 | 0.85-0.95 | Cache hit icin cosine benzerlik |
| Semantic cache TTL | 5 dk | 1-60 dk | Volatil piyasada kisa, stabil'de uzun |
| CRAG confidence threshold | 0.4 | 0.3-0.6 | Altinda -> incorrect siniflandir |
| HyDE activation threshold | 0.7 | 0.5-0.8 | Normal retrieval guveni bu altindaysa HyDE devrede |

### Multi-Agent Parametreleri

| Parametre | Baslangic | Aralik | Aciklama |
|-----------|-----------|--------|----------|
| Debate round sayisi | 2 | 2-5 | Real-time: 2, EOD analiz: 3-5 |
| Confidence -> position_size alpha | 2.0 | 1.5-3.0 | Konveks ceza ustu (dusuk guven = orantisiz kucuk pozisyon) |
| Min viable position | $5 | $1-20 | Bunun altinda komisyon > kar, dogal filtre |
| Self-consistency sample count | 3 | 3-5 | Kritik kararlar icin |
| Agent memory top-k | 5 | 3-10 | Her agent basina bellek ogeleri |

### Temporal Decay Formulu
```
score(query, doc, time) = alpha * cosine(query, doc) + (1-alpha) * 0.5^(age_days / half_life)

Parametreler:
  alpha = 0.7 (semantik agirligi)
  half_life = icerik turune gore:
    - Breaking crypto news: 1-3 gun
    - Piyasa analizi/sentiment: 7-14 gun
    - Ceyreklik rapor: 30-90 gun
    - Teknik dokumantasyon: 180+ gun
```

### Confidence-Bazli Pozisyon Boyutlandirma (Surekli Egri, Engelleme YOK)

| Confidence | Pozisyon Boyutu | Aciklama |
|------------|-----------------|----------|
| < 0.35 | confidence^2 * max | Toz pozisyon (komisyon > kar -> kendini filtreler) |
| 0.35-0.50 | confidence^2 * max | Mikro kesfedici pozisyon |
| 0.50-0.70 | confidence^2 * max | Standart pozisyon |
| 0.70-0.85 | confidence^2 * max | Yuksek inanc pozisyonu |
| > 0.85 | confidence^2 * max | Maksimum inanc pozisyonu |

> NOT: HICBIR seviyede trade ENGELLENMEZ. Dusuk guven = kucuk pozisyon.
> Pozisyon boyutu surekli egridir (confidence^alpha), binary degil.
> Komisyon doğal filtre gorevi gorur: cok kucuk pozisyonlar karli OLAMAZ.

### ATR-Bazli Dinamik Stop-Loss
```
Position Size = Dollar Risk / (ATR * Multiplier)
Max Position = min(Risk_Position, 10% * Portfolio)

Multiplier:
  - Scalping (5m): 1.5x ATR
  - Swing (4h): 2.0x ATR
  - Position (1d): 3.0x ATR
```

---

## TOKEN COST CONTROL & OPTIMIZATION
> Her token para. Gereksiz token = kayip. Akilli optimizasyon = x10 tasarruf.

### API Rate Limit Yonetimi

**Gemini Flash (Free Tier):**
| Model | RPM | TPM | RPD |
|-------|-----|-----|-----|
| 2.5 Flash | 10 | 250,000 | 250 |
| 2.5 Flash-Lite | 15 | 250,000 | 1,000 |
| 2.5 Pro | 5 | 250,000 | 100 |

**Gemini Flash (Tier 1 - billing account bağla, $0):**
| Model | RPM | TPM | RPD |
|-------|-----|-----|-----|
| 2.5 Flash | 300 | 2,000,000 | 1,500 |

**Groq (Free Tier):**
| Model | RPM | TPM | TPD |
|-------|-----|-----|-----|
| Llama 3.1-8B | 30 | 6,000 | 500,000 |
| Llama 3.3-70B | 30 | 12,000 | 100,000 |

**OpenRouter:** Free: 20 RPM, 50 RPD. $10 deposit: 1,000 RPD, 10+ RPS.

### Maliyet Karsilastirmasi (1M input + 1M output token)

| Provider/Model | Toplam Maliyet | Hiz |
|----------------|---------------|-----|
| OpenRouter free modeller | **$0.00** | Degisken |
| Groq Llama 3.1-8B | **$0.13** | Ultra-hizli |
| Gemini Flash-Lite | **$0.50** | Hizli |
| Gemini 2.5 Flash | **$2.80** | Hizli |
| Groq Llama 3.3-70B | **$1.38** | Ultra-hizli |
| Gemini 2.5 Pro | **$11.25** | Orta |

### Gorev-Bazli Model Routing (Maliyet Optimizasyonu)

| Gorev | Model | Neden |
|-------|-------|-------|
| Basit siniflandirma (BUY/SELL/HOLD) | Gemini Flash-Lite ($0.50/M) | Hizli, ucuz |
| Analiz + reasoning | Gemini 2.5 Flash ($2.80/M) | Dengeli |
| Karmasik multi-hop karar | Groq DeepSeek R1 veya Gemini Pro | Dogruluk kritik |
| Entity extraction | Gemini Flash-Lite | Basit, yapisal |
| Sentiment ozeti | Groq Llama 8B ($0.13/M) | Ultra ucuz + hizli |
| Fallback/overflow | OpenRouter free modeller ($0) | Sifir maliyet |

### 7 Katmanli Maliyet Optimizasyonu

```
Katman 1: Gorev-bazli model routing          -> %40-85 tasarruf
  Ucuz model yeterli olan goreve pahali model gonderme!

Katman 2: Output token kontrolu              -> %30-50 tasarruf
  max_tokens siki tut, structured output mode kullan
  Output tokenlari input'tan 4-8x daha pahali!

Katman 3: Gemini prefix/context caching      -> %90 tasarruf (input uzerinde)
  System prompt + RAG context'i cache'le
  Min 1,024 token (Flash), TTL 1 saat
  Cache reads: base fiyatin %10'u

Katman 4: Semantic caching                   -> %65-73 tasarruf
  Benzer sorgu -> onceki yanitiy dondur
  Cosine similarity > 0.92 = cache hit
  TTL: 5dk (volatil) / 60dk (stabil)

Katman 5: Prompt compression (LLMLingua-2)   -> %50 input azaltma
  20x compression, %1.5 performans kaybi
  RAG context'leri 2-3 chunk'a sinirla (4-8 yerine)

Katman 6: Request batching                   -> %30-50 tasarruf
  Mum kapanisinda tum pair analizlerini batch'le
  Multi-timeframe veriyi tek prompt'a birlestir
  Gemini Batch API: %50 indirim

Katman 7: Token sayma + butceleme           -> Kontrol
  Her request oncesi tiktoken ile token say
  Gunluk butce: $2 Gemini, $1 Groq, $0.50 OpenRouter
  LiteLLM ile per-agent butce enforce et
```

### Tahmini Gunluk Maliyet (Tum Optimizasyonlarla)
```
Normal gun (optimizasyonsuz):     ~$5-10/gun
7 katman optimizasyon sonrasi:    ~$0.20-0.50/gun
  - Model routing:               -%60
  - Output control:              -%40
  - Prefix caching:              -%80 (input)
  - Semantic caching:            -%65 (tekrarlar)
  - Compression:                 -%50 (RAG context)
  - Batching:                    -%30
```

---

## AI AUTONOMY & TRADE EXECUTION PHILOSOPHY
> "The safest possible trading bot -- one that never trades -- guarantees exactly one outcome:
> zero returns minus infrastructure costs." -- Anti-Over-Engineering Research, 2026
>
> KAS YAPARKEN GOZ CIKARMA! AI'a guc verip elini kolunu baglama!

### Core Principle: Trade-First, Not Block-First

**Renaissance Technologies Kurali**: "Never override the computer."
2007'de 3 gunde $1B kaybettiler, mudahale ETMEDILER, yil sonunda %85.9 getiri.
150,000-300,000 trade/gun, %50.75 isabet orani, TAM OTONOM.

**Bizim Felsefemiz:**
```
DEFAULT = TRADE YAP (blacklist yaklaşımı)
  Tehlikeli bir durum var mi? -> Evet: Pozisyonu KUCULT (engelleme!)
                              -> Hayir: Trade yap

DEFAULT ≠ TRADE YAPMA (whitelist yaklaşımı)
  Tum kosullar uygun mu? -> 5/5 filtre gecti mi? -> Consensus var mi? -> ENGELLE
```

**Neden Default-Allow:**
- Guvenlik ihlalerinin aksine, kotu trade'lerin maliyeti SINIRLI (stop-loss + pozisyon boyutu)
- Iyi trade'leri kacirmanin maliyeti SINIRSIZ ve GORUNMEZ (opportunity cost)
- Trading'de Type II hata (iyi trade'i kacirmak) > Type I hata (kotu trade yapmak)

### The Over-Guardrail Problem (Matematik)

```
Stacked Filter Felaketi:
  Her filtre %80 gecirme oranina sahip olsun (makul gorunur):

  1 filtre:  %80.0 gecis   (5 trade'den 4'u gecer)
  2 filtre:  %64.0 gecis
  3 filtre:  %51.2 gecis   (yarisini kaybettin!)
  5 filtre:  %32.8 gecis   (3 trade'den 2'si engellendi)
  7 filtre:  %20.9 gecis   (5 trade'den 4'u engellendi!)
  10 filtre: %10.7 gecis   (neredeyse HICBIR trade gecmiyor)

  -> Her "makul" filtre tek basina zararsiz gorunur
  -> Ama birlikte kullaninca sistemi FELC ederler
```

**Alpha Decay (Sinyal Bozulması):**
- Sinyal alpha'si 5-10-15-30-60 dakikada OLCULEBILIR sekilde azalir
- Birkaç saniyelik gecikme bile getiriyi %5.6-10 dusurur
- %70 guvenle SIMDI trade yapmak > %95 guvenle 30 dakika sonra trade yapmak
- Volatil piyasada (tam trade yapmak istedigin an) alpha EN HIZLI bozulur

**KDD 2026 Bulgusu (Kritik):**
> "Daha fazla guvenlik katmani eklemek, bull market'ta AI'yi DAHA konservatif yapiyor
> (tam tersi etki!) ve bear market'ta kayiplari ONLEMIYOR."
> LLM stratejileri bull'da asiri muhafazakar, bear'de asiri agresif (patolojik).
> Cozum: Daha fazla guvenlik katmani DEGIL, rejim farkindaligi.

### 10 Prensip: AI Trading Autonomy

```
1. DEFAULT TRADE YAPTIR
   Basarili tum sistemler (FinMem, TradingGroup, CryptoTrade) agresifi DEFAULT yapar.
   Konservatif mod SADECE para kaybedildiginde devreye girer.

2. GUVEN BOYUT BELIRLER, IZIN DEGIL
   CryptoTrade: %30 guven bile trade yapar -- sadece kucuk pozisyon.
   Asla binary trade/trade-yok karari verme. Surekli pozisyon boyutu egrisi kullan.

3. RISK YONETIMI AYARLAR, ENGELLEMEZ
   TradingAgents: Risk manager rapor sunar, PM override edebilir.
   FINCON: Manager TEK karar verici. Risk analisti VETO YETKISINE SAHIP DEGIL.

4. POZISYON BOYUTU = BIRINCIL RISK KONTROLU
   Trade engellemek DEGIL, pozisyon kucultmek ana savunma hatti.
   %1-2 risk/trade, ATR-bazli stop, Kelly fraksiyonu = yeterli koruma.

5. SELF-ADAPTIVE RISK (Sadece Kaybederken Konservatif)
   FinMem: Kumulatif getiri < 0 (3 gun) -> konservatif moda gec
   FinMem: Kumulatif getiri > 0 -> agresif moda geri don (DEFAULT)
   Beklenen risk degil, GERCEKLESEN kayip tetikler.

6. REJIM FARKINDALIĞI > KATMAN KARMASIKLIGI
   KDD 2026: Daha fazla analiz katmani = bull'da DAHA KOTU performans
   Basit rejim tespiti (trend yonu) > 7 katman guvenlik kontrolu

7. ONCE TRADE YAP, SONRA OGREN
   CryptoTrade reflective: Onceki trade sonuclarindan ogren, trade ONCESI analiz paralizi yapma.
   Optimistic execution: Trade baslat, risk paralel degerlendir, sadece felaket varsa iptal et.

8. ARSITEKTUR > MODEL
   "When Agents Trade" benchmark: Framework tasarimi > LLM modeli.
   Muhendislik zamanini daha iyi analiz katmanlarina degil, karar arsitekturine yatir.

9. YAPISAL ANLASMAZLIK, KONSENSUS DEGIL
   TradingAgents: Bull/Bear TARTISMA bilgi olarak kullanilir, consensus ARANMAZ.
   Anlasmazlik = karar vericiyi bilgilendirir, trade'i ENGELLEMEZ.
   Voting > Consensus: Reasoning gorevlerinde voting %13.2 daha iyi.

10. DEVELOPER PROJECTION TUZAGI
    Lee & See (2004): Guvensizlik = disuse (sistemi kullanmama).
    Gelistirici kendi risk aversiyon'unu AI'a yansitir.
    AI dogasi geregi risk-notr. Katman katman guvenlik eklemek = AI'in dogasina karsi savasmak.
```

### Graduated Autonomy Framework (6 Seviye)

> Prop trading firmalarinin ispat edilmis modeli: Kucuk basla, kazan, buyut.

| Seviye | Ad | Insan Rolu | Max Pozisyon | Max Gunluk Trade | Pair | Onay |
|--------|-----|-----------|-------------|-----------------|------|------|
| 0 | Backtest | N/A | N/A | Sinirsiz | Tumu (tarihsel) | N/A |
| 1 | Paper | Gozlemci | $0 (simule) | Sinirsiz | Tumu | Yok |
| 2 | Micro-Live | Onayli | $10-50/trade | 5 | 2-3 pair | Her trade bildirim |
| 3 | Small-Live | Danisman | $50-200/trade | 15 | 5-8 pair | Gunluk ozet |
| 4 | Standard | Izleyici | 0.5 Kelly | 30 | Tum uygun pair'ler | Haftalik review |
| 5 | Full Autonomy | Gozlemci | 0.75 Kelly | Sinirsiz | Tumu | Sadece istisna alert |

**Seviye Yukseltme Kriterleri:**

| Gecis | Gereksinimler |
|-------|--------------|
| 0→1 | Backtest Sharpe > 1.0, max DD < %15, min 2 rejimde test |
| 1→2 | 100+ paper trade, Sharpe > 0.8, max DD < %10, min 30 gun |
| 2→3 | 100+ live micro trade, fees sonrasi pozitif P&L, Sharpe > 0.5, min 60 gun |
| 3→4 | 200+ live trade, Sharpe > 0.7, max DD < %10, son 4 aydan 3'u pozitif, min 90 gun |
| 4→5 | 500+ live trade, Sharpe > 1.5, max DD < %10, son 6 aydan 5'i pozitif, min 1 yuksek-vol rejim, min 180 gun |

**Seviye Dusurme Tetikleyicileri:**

| Tetikleyici | Aksiyon | Toparlanma |
|-------------|---------|-----------|
| Gunluk kayip > %3 | 24 saat boyunca pozisyon %25'e kucult (trade DEVAM eder, durdurma yok) | Otomatik normal boyuta don |
| Haftalik kayip > %5 | Bir seviye dus | Standart yukseltme kriterleri, min 2 hafta |
| Aylik DD > %10 | Iki seviye dus | Standart kriterler, min 30 gun |
| Tek trade > 5x ortalama kayip | 48 saat boyunca dust-only pozisyon + otomatik analiz | Manuel review |
| 7+ gun trade yok (firsat varken) | Alert + review | Manuel onay |

**Trust Score (Beta Dagilimi):**
```
Trust = Beta(alpha, beta) dagilimi
  - Her basarili trade: alpha += 1
  - Her basarisiz trade: beta += 1
  - Beklenen trust = alpha / (alpha + beta)
  - Baslangic: Beta(1,1) = %50 (uniform prior, belirsiz)
  - 200 basarili + 1 basarisiz: %99.5 (ufak dusus)
  - 5 basarili + 1 basarisiz: %75.0 (anlamli dusus)
  -> Doğal asimetri: Uzun track record'da tek kayip az etkiler
  -> Kisa track record'da her kayip buyuk etkiler

Exponential decay (eski trade'ler daha az etkili):
  alpha = sum(weighted_successes), lambda=0.995/trade
  beta = sum(weighted_failures)

Trust Score = Beta_mean * Recency_weight * Regime_adjustment
  Recency: 1.0 (son 7 gunde trade) -> %5/hafta azalma (inaktif)
  Regime: 0.6 (farkli rejim) / 0.8 (yuksek vol) / 1.0 (normal) / 1.1 (en iyi rejim)
```

### Confidence-Based Self-Regulation (TRADE ENGELLEME YOK)

> AI kendi kendini yonetir. Dusuk guven = kucuk pozisyon. Yuksek guven = buyuk pozisyon.
> ASLA binary trade/trade-yok karari vermez. Pozisyon boyutu SUREKLI bir egridir.

**Surekli Pozisyon Boyutu Egrisi:**
```
position_size = max_position * kelly_fraction * calibrated_confidence^alpha

Parametreler:
  max_position = portfolio'nun %3'u (TEK hard limit)
  kelly_fraction = 0.25 (baslangic, Bayesian guncelleme ile adapte)
  alpha = 2.0 (konveks ceza: dusuk guven orantisiz cezalandirilir)

Ornek ($10,000 portfolio, base_kelly=0.25, alpha=2):
  %95 guven: $10,000 * 0.03 * 0.25 * 0.95^2 = $67.69  (anlamli pozisyon)
  %80 guven: $10,000 * 0.03 * 0.25 * 0.80^2 = $48.00  (standart pozisyon)
  %60 guven: $10,000 * 0.03 * 0.25 * 0.60^2 = $27.00  (kucuk pozisyon)
  %50 guven: $10,000 * 0.03 * 0.25 * 0.50^2 = $18.75  (mikro pozisyon)
  %40 guven: $10,000 * 0.03 * 0.25 * 0.40^2 = $12.00  (toz pozisyon)
  <%35 guven: Komisyon > potansiyel kar -> kendini filtreler (dis mudahale YOK)

-> HICBIR confidence seviyesinde trade ENGELLENMEZ
-> Dusuk guven otomatik olarak ekonomik olarak anlamsiz pozisyon uretir
-> Sistem kendini filtreler, harici guardrail GEREKSIZ
```

**Bayesian Kelly (Kendi Kendini Ogreten Pozisyon Boyutu):**
```
Kelly: f* = (b*p - q) / b
  b = win/loss orani (Bayesian tahmin)
  p = kazanma olasiligi (Beta dagilimi mean'i)
  q = 1 - p

-> Kayip serisi: p duser -> Kelly kucuk -> pozisyon otomatik kucuk
-> Kazanc serisi: p artar -> Kelly buyur -> pozisyon otomatik buyuk
-> HIC harici mudahale gerekmez, matematik kendini duzeltir

Fractional Kelly per Trust Level:
  Level 2 (Micro):    0.10 * full_kelly
  Level 3 (Small):    0.25 * full_kelly
  Level 4 (Standard): 0.50 * full_kelly
  Level 5 (Full):     0.75 * full_kelly
  (Full Kelly = ASLA, cunku %50 drawdown olasiligi var)
```

**Selective Retrieval (Self-RAG Style):**
```
Yuksek guven (>%85): Base model sinyaliyle trade. Ek veri CEKME. Hizli execution.
Orta guven (%60-85): Ek veri kaynaklarina danis (sentiment, on-chain, order book).
                      Guven guncelle. Pozisyonu yeniden boyutlandir.
Dusuk guven (<%60):  Tam retrieval. Guven hala dusukse -> toz pozisyon (engellemez!)

-> Pahali veri kaynaklarina SADECE gerektiginde eris (maliyet tasarrufu)
-> Dusuk guven = ENGELLEME degil, KUCUK pozisyon
```

### Dynamic Risk Budget (Millennium/Citadel Pod Modeli)

> AI'a gunluk bir risk butcesi ver. Istedigini gibi harcasin. Butce bitince dur.
> Bu KENDINDEN YONETIM, dis engelleme DEGIL.

```
Gunluk VaR Butcesi:
  - AI'nin gunluk kayip butcesi: portfolio'nun %1'i
  - Her trade butceden tuketir: position_size * asset_volatility * (1/confidence)
  - Dusuk guven trade'ler DAHA FAZLA butce tuketir (dogal ceza)
  - Butce bitince: Yeni pozisyon yok, mevcut pozisyonlar yonetilebilir
  - Bu KENDINDEN yonetim, DIS engelleme degil

Haftalik Butce Ayarlama:
  Karli hafta:  multiplier = min(1.0 + sharpe_bu_hafta * 0.1, 1.5)
  Kayipli hafta: multiplier = max(0.5, 1.0 - drawdown_pct * 2)

Aylik Reset:
  - Butce HER AY base'e donur
  - Kotu seriden sonra kalici aclik ONLENIR

Mutlak Taban:
  - Butce ASLA base'in %25'unun altina dusmez
  - Sistem ASLA tamamen durmaz (en kotude toz pozisyonlar)
```

### Minimalist Safety Checklist (SADECE BUNLAR)

> Daha az = daha iyi. Her ek kontrol opportunity cost'tur.
> Renaissance: %50.75 isabet orani + milyonlarca kucuk trade = devasa getiri.

**Tier 1: Vazgecilmez (bunlar olmadan CANLI OLMA)**
1. **Pozisyon boyutu**: Trade basina max %1-2 risk (BIRINCIL kontrol)
2. **Stop-loss**: Her trade'de otomatik cikis (hayatta kalma garantisi)
3. **Max drawdown kill switch**: Portfolio -%20 → tum trading dur (ACIL DURUM freni)
4. **Max gunluk kayip**: Gunde -%5 → gun icin dur
5. **Exchange baglanti kontrolu**: API canli mi? Order filled mi?

**Tier 2: Onemli (olceklendirmeden once ekle)**
6. Max acik pozisyon limiti (3-5 kucuk hesap)
7. Varlik basina max pozisyon (%10-20 portfolio)
8. Order execution dogrulama (partial fill handle)
9. Loglama (her sinyal, emir, fill, hata + timestamp)
10. Paper trade'den canli gecis (min 2 hafta paper)

**Tier 3: Iyi olur (yukaridakiler stabil olduktan sonra)**
11. Korelasyon/exposure izleme
12. Volatilite ayarlama (ATR-bazli pozisyon kucultme)
13. Performans dashboard (win rate, profit factor, DD curves)

**EKLEME bunlari:**
```
✗ Tum AI ajanlarinin consensus'unu bekleme
✗ 5+ gostergenin ayni anda onay vermesini isteme
✗ "Sermayem var mi? Riskim limitler icinde mi? Exchange canli mi?" disinda pre-trade check
✗ Sentiment analizini bloklayici gate olarak kullanma
✗ Kademeli validation katmanlari (her biri reject edebilen)
✗ %90+ kesinlik gerektiren confidence threshold
✗ Ayni risk kavramini birden fazla katmanda kontrol etme
```

### Signal-to-Trade Conversion Metrics

> Trade yapan bot = iyi bot. Trade yapmayan bot = pahalı paperweight.

```
Hedef Conversion Rate: %60-80
  <%30: Guardrail'lar KESINLIKLE asiri agresif -> acil revizyon
  %30-50: Muhtemelen fazla filtre var -> parametreleri gevset
  %50-70: Saglıklı bölge -> izle
  %70-90: Iyi dengelenmiş -> optimal
  >%90: Felaket tespiti zayif olabilir -> kontrol et

Forgone P&L Takibi (ZORUNLU):
  - Reddedilen her sinyali PAPER TRADE et
  - Reddedilen sinyaller tutarli olarak karli cikiyorsa -> guardrail'lar DEGER YOKEDIYOR
  - Bu TEK EN ONEMLI diagnostik metrik
  - Haftalik rapor: "Reddedilen sinyallerin toplam P&L'i neydi?"

Override Frequency:
  - Her red'di neden koduyla kaydet
  - Bir kontrol >%50 redleri olusturuyorsa -> muhtemelen KOTU KALIBRE
  - Bloklanan vs execute edilen trade'lerin getirilerini DUZENLI karsilastir
```

### Basarili Sistemlerin Karsilastirmasi

| Sistem | Default Durusu | Risk Yaklasimi | Anahtar Inovasyon |
|--------|---------------|----------------|-------------------|
| Renaissance | Tam otonom | Pozisyon boyutu | "Never override the computer" |
| TradingAgents | PM override edebilir | Danismanlık (veto yok) | Yapisal tartisma |
| CryptoTrade | Her gun trade | Reflective (trade sonrasi) | Zero-shot + yansitma |
| FinMem | Agresif default | Self-adaptive (sadece kaybederken konservatif) | Katmanli bellek |
| FINCON | Manager tek karar verici | Episodic self-critique | Hiyerarsi > demokrasi |
| TradingGroup | Agresif default | PnL'e gore stil adapte | Kendi kendine yansitma |
| Numerai | Tam otonom | Stake-weighted meta-model | 1200+ model ensemble |
| Taleb Barbell | %90 guvende + %10 risk | Felaket koruması + serbest risk | Ortada "belki" yok |

---

## LLM GUARDRAILS & SAFETY (Trade-First Uyumlu)
> Trade ENGELLEMEK icin degil, pozisyon BOYUTLANDIRMAK icin guvenlik katmanlari.
> Felsefe: Felaket onle, geri kalani pozisyon boyutuyla yonet.

### Output Validation (Pydantic + Instructor)
```python
# Her trade sinyali bu schemaya ZORLA uyumlu olmali
class TradeSignal(BaseModel):
    action: Literal["LONG", "SHORT", "HOLD"]
    pair: str
    confidence: float = Field(ge=0, le=1)
    entry_price: float = Field(gt=0)
    stop_loss: float = Field(gt=0)
    take_profit: float = Field(gt=0)
    position_size_pct: float = Field(gt=0, le=5.0)  # max %5
    rr_ratio: float = Field(gt=0)
    reasoning: str = Field(min_length=50)
    invalidation: str

# Instructor ile retry: validation fail -> hata mesajini LLM'e gonder -> tekrar dene
# max_retries=3, her retry'da validation hatasi feedback olarak eklenir
```

### Guardrails Framework Secimi
- **Guardrails AI**: Pydantic entegrasyonu, validator hub, 3M+ download
- **Instructor**: Structured output + automatic retry, 11K+ stars
- **NeMo Guardrails**: Input/output/retrieval/execution rails
- **Pydantic validation**: Ek kutuphane gerektirmeden schema enforce

### Override Hierarchy (Degistirilmez Oncelik Sirasi)
```
Level 1 (MUTLAK): Exchange limitleri, margin gereksinimleri, regulator kisitlamalar
  -> ASLA override edilemez, hard-coded

Level 2 (SISTEM): Freqtrade korumalari
  -> MaxDrawdown (%20), StoplossGuard (4 trade), CooldownPeriod (5 mum)
  -> Config'den gelir, LLM degistiremez

Level 3 (STRATEJI): Pozisyon boyutu kurallari
  -> Max %2 risk/trade, max %5 pozisyon, Quarter-Kelly
  -> Strateji parametresi, LLM onerebilir ama override edemez

Level 4 (AI): LLM-generated sinyaller
  -> Confidence-weighted oneriler
  -> SADECE Level 1-3 sinirlar icinde calisabilir

Level 5 (VARSAYILAN): LLM unavailable oldugunda
  -> FreqAI-only mode, rule-based fallback
```

### Finansal Guvenlik Kontrolleri (Boyutlama Odakli)
> NOT: Asagidaki 8 check "layered validation gate" DEGILDIR.
> Bunlar POZISYON BOYUTUNU AYARLAR, trade'i ENGELLEMEZ.
> Hepsi paralel calisir, sonuc = final pozisyon boyutu (0 degilse trade acilir).
> "EKLEME" listesindeki (satir 1717) "kademeli validation katmanlari" = her biri REJECT edebilen gateler.
> Bizimkiler = her biri boyutu AYARLAYAN sizing kontroller. Fark bu.

```
Pre-Trade Sizing (her trade oncesi - BOYUTLAMA, ENGELLEME DEGIL):
  1. Pozisyon boyutu <= %3 portfolio (hard cap) ✓
  2. Risk per trade <= %2 portfolio (ATR-bazli) ✓
  3. Portfolio correlation > 0.7 -> pozisyonu %50 kucult (engelleme!) ✓
  4. Ayni anda max 12 pozisyon (soft limit, trust level'a gore degisir) ✓
  5. Drawdown %5-10 -> pozisyonu %50 kucult | %10-15 -> %25 | >%20 -> KILL SWITCH ✓
  6. Confidence -> pozisyon boyutuna dogrudan map (engelleme yok, confidence^2 egrisi) ✓
  7. Fiyat sanity check: entry +-1% current market ✓
  8. Exchange API canli mi? ✓

  ENGELLEME SADECE: Exchange API down | Kill switch aktif | Pair delisted

Post-Trade Adaptive Response:
  - Consecutive stoploss > 3 -> pozisyon boyutunu %50 kucult (5 trade icin)
  - Gunluk drawdown > %3 -> pozisyon boyutunu %75'e kucult
  - Gunluk drawdown > %5 -> pozisyon boyutunu %50'ye kucult
  - Gunluk drawdown > %10 -> pozisyon boyutunu %25'e kucult
  - Gunluk drawdown > %20 -> KILL SWITCH (TEK gercek engelleme)
  NOT: %3-%15 arasi KUCULTME, DURDURMA degil. Sistem trade yapmaya DEVAM eder.
```

### Hallucination Prevention (Confidence Adjustment, Bloklama DEGIL)
```
1. Fiyat Dogrulama: Her LLM fiyat referansini canli exchange verisiyle karsilastir
   Tolerans: +-1% (ustundeyse confidence %50 dusur -> pozisyon otomatik kuculur)

2. Event Dogrulama: Gemini Grounding (Google Search) ile haber dogrula
   Hallucinated event tespiti -> confidence %75 dusur + LOG (ogrenme icin)

3. Gosterge Cross-Check: LLM'in soyledigi RSI/MACD degerleri
   vs gercek hesaplanan degerler karsilastir
   Sapma > %5 -> confidence %50 dusur + LOG (hangi gostergede hallucinate etti?)

4. Lynx Hallucination Guardian: Trade sinyali oncesi son kontrol
   Hallucinated fiyat/event -> confidence DUSUR (pozisyon kucultme)
   ENGELLEME YOK: Kucuk pozisyonla trade yap, sonucu LOGLA, hatasindan OGREN

NOT: Tum hallucination tespitleri LOGLENIR ve haftalik raporda analiz edilir.
     Sistem zamanla hangi hallucination pattern'lerinin gercek kayba yol actigini
     ogrenir ve Bayesian Kelly otomatik olarak adapte olur.
```

### Confidence Calibration
- **Verbalized confidence**: LLM'den 0-1 arasi guven skoru iste
- **Self-consistency**: 3-5 kez calistir, uyum orani = guven
- **Post-hoc calibration**: Brier score ile gercek sonuclarla kalibre et
- **Uyari**: LLM confidence tahminleri KOTU kalibre edilmis (ECE 0.108-0.427)
  -> Self-consistency ile DOGRULA, tek basina LLM skor'una GUVENME

### Graceful Degradation (7 Seviye)
```
Level 1: Tam API analizi (Gemini Flash + RAG + tum context)
Level 2: Basitlestirilmis analiz (daha az token, daha az context)
Level 3: Ikincil API (Groq, daha hizli ama daha az context)
Level 4: Ucuncul API (OpenRouter, en ucuz model)
Level 5: Yerel LoRA model (temel analiz)
Level 6: Cached yanit (son bilinen iyi analiz)
Level 7: FreqAI-only (saf ML, LLM yok)
Level 8: Rule-based gostergeler (AI yok)
```

### Failover Architecture (Circuit Breaker Pattern)
```
CLOSED (normal) -> Hatalar izlenir
  |
  5 ardisik hata / 30 saniye -> OPEN (blocking)
  |
  60 saniye cooldown -> HALF-OPEN (probe)
  |
  Tek test request -> Basarili: CLOSED | Basarisiz: OPEN

Exponential Backoff: 200ms -> 400ms -> 800ms -> 1600ms -> max 60s
Retry: 429, 500, 502, 503, 504 icin
No Retry: 400, 401, 403 icin
Provider-specific: Groq retry-after header'i honor et
```

---

## LLM OBSERVABILITY & MONITORING
> Olcemedigini yonetemezsin. Her token, her latency, her maliyet izlenmeli.

### Monitoring Stack (Server-Uyumlu, Hafif)
```
LiteLLM Proxy (gateway + maliyet takip + butce + failover)
  + SQLite logging (sifir overhead, custom)
  + Prometheus (metrik export)
  + Grafana (dashboard + Telegram alert)

Toplam RAM: ~500 MB
Toplam Disk: ~10 MB/gun log
```

### LiteLLM Entegrasyonu
- Tum LLM cagrilari LiteLLM proxy uzerinden gider
- Per-agent API key + butce: market_analyzer=$5/gun, signal_gen=$2/gun
- Otomatik failover: Gemini -> Groq -> OpenRouter
- Maliyet takip: per-request input/output token + USD
- Rate limit enforce: token bucket, %80 guvenlik margini
- Response header: x-litellm-response-cost

### Neleri Loglamali (SQLite Schema)
```sql
llm_calls (
  trace_id, timestamp, model, provider, agent_name,
  input_tokens, output_tokens, cost_usd,
  latency_ms, ttft_ms, temperature,
  status, prompt_hash, prompt_version,
  cache_hit, trading_pair, signal_type
)

llm_daily_summary (
  date, model, provider, agent,
  total_calls, total_input_tokens, total_output_tokens,
  total_cost, avg_latency, error_count, cache_hits
)

prompt_versions (
  version_id, template_name, template_hash,
  template_text, created_at, active
)
```

**LOGLAMAMA**: Raw prompt/response default olarak loglama (disk + privacy)
  -> Sadece: errors, anomalies, A/B test samples icin full log

### Grafana Dashboard Layout
```
Row 1 - Maliyet & Butce:
  - Gunluk harcama by provider (bar chart)
  - Aylik kumulatif vs butce (gauge)
  - Sinyal basina maliyet trendi (time series)

Row 2 - Performans:
  - p95 latency by provider (time series)
  - p95 TTFT (time series)
  - Hata orani by provider (time series)
  - Cache hit rate (time series)

Row 3 - Hacim:
  - Dakika basina istek by agent (time series)
  - Istek basina token dagilimi (histogram)
  - Provider failover events (event annotations)

Row 4 - RAG Sagligi:
  - Retrieval latency p50/p95
  - Embedding API latency
  - Context window kullanim dagilimi
```

### Telegram Alert Kurallari
| Alert | Kosul | Oncelik |
|-------|-------|---------|
| Butce uyari | Gunluk harcama > %80 limit | WARNING |
| Butce kritik | Gunluk harcama > %95 limit | CRITICAL + auto-fallback |
| Yuksek latency | p95 > 10s, 5dk boyunca | WARNING |
| Hata spike | Hata orani > %5, 5dk | WARNING |
| Provider down | %100 hata, 2dk | CRITICAL + auto-failover |
| Cache bozulma | Hit rate baseline'dan >%50 dusus | WARNING |
| Maliyet spike | Saatlik maliyet > 3x 7-gun ort | CRITICAL |

### A/B Testing (Prompt Versiyonlari)
- **Shadow testing**: Her request'i Control + Treatment'a gonder
  - Sadece Control yaniti kullanilir
  - Treatment loglara kaydedilir, asenkron degerlendirilir
  - Sifir risk
- **Metrikler**: Sinyal dogrulugu, token verimliligi, latency, edge case performansi
- **Istatistiksel anlamlilik**: Min 100-200 sinyal per variant
- **Prompt versiyonlama**: Semantic versioning (v1.0.0), hash + creation date, immutable

---

## TRADE DECISION FRAMEWORK
> Sinyalden execute'a kadar tam karar pipeline'i

### Trade Sinyal Skorlama Sistemi

**Weighted Linear Combination:**
```
S = Sum(weight_i * normalized_indicator_i)
T = S * RegimeFilter * VolatilityAdjustment

Normalizasyon: z-score = (value - rolling_mean_14) / rolling_std_14

Agirliklar (baslangic, rejime gore dinamik):
  - Teknik/Kantitatif:  %35-40  (en backtestable)
  - On-chain metrikler:  %15-20  (kriptoya ozel, leading)
  - Sentiment (NLP):     %15-20  (kisa vadeli alpha, gurultulu)
  - LLM Analizi:         %15-20  (kalitatif edge)
  - Rejim filtreleri:    %10-15  (carpmac/gate, toplamsal degil)
```

**Rejime Gore Dinamik Agirlik Ayarlama:**
- Bull market: Sentiment/subjektif agirlik arttir (%76 -> %66 performans farki)
- Bear market: Faktuel/on-chain agirlik arttir (%-15 -> %-20 fark)
- FinMem self-adaptive: Kumulatif getiri < 0 -> otomatik conservative moda gec

### Karar Pipeline'i (Size-Through, Block-Never*)
```
* ENGELLEME SADECE: Exchange down | Kill switch | Pair delisted

Stage 1: PRE-FILTER (Evren Tarama)
  -> Likidite filtresi (24h volume > $10M)
  -> Korelasyon kontrolu (mevcut pozisyonlarla)
  -> SONUC: Likidite dusuk -> pozisyon %50 kucult | Korelasyon > 0.7 -> %50 kucult

Stage 2: ANALYSIS (Cok Kaynakli Veri Toplama, PARALEL)
  -> Teknik gostergeler (FreqAI + tool-augmented RAG)
  -> On-chain metrikler (DeFiLlama/blockchain API)
  -> Sentiment (CryptoBERT + FinBERT + haber RAG)
  -> LLM sentezi (Gemini Flash, CoT reasoning)
  -> SONUC: Veri eksik -> selective retrieval tetikle, trade BEKLEME

Stage 3: SCORING (Sinyal Sentezi)
  -> Z-score normalizasyon + agirlikli kombinasyon
  -> Rejim carpani + volatilite ayari
  -> SONUC: Skor -> confidence -> pozisyon boyutu egrisi (confidence^2)
  -> Dusuk skor = KUCUK pozisyon, ENGELLEME degil

Stage 4: RISK SIZING (Pozisyon Boyutlandirma)
  -> Confidence-based Kelly fraction
  -> ATR-bazli stop-loss mesafesi
  -> Max exposure check (%3 portfolio hard cap)
  -> Drawdown-adjusted sizing (DD %5+ -> kucult)
  -> SONUC: Final pozisyon boyutu (ENGELLEME yok, BOYUTLAMA var)

Stage 5: EXECUTION (Hizli Emir Yonetimi)
  -> Emir tipi secimi (limit/market)
  -> Slippage tahmini
  -> Post-trade monitoring baslat
  -> SONUC: Trade execute. Alpha decay BEKLEME!

Post-Trade: REFLECTION (CryptoTrade Pattern)
  -> Trade sonucunu kaydet
  -> Basarili pattern'leri bellete yaz
  -> Basarisiz pattern'leri anti-pattern olarak kaydet
  -> Bayesian Kelly parametrelerini guncelle
```

### Multi-Agent Karar Mekanizmasi (Tartisma, Consensus DEGIL)
- **Voting > Consensus**: Reasoning gorevlerinde voting %13.2 daha iyi (arxiv 2502.19130)
- **FINCON modeli**: Manager TEK karar verici, analistler DANISMANDIR, VETO YETKISI YOK
- **Anlasmazlik = BILGI**: Bull/Bear tartismasi PM'i bilgilendirir, trade'i ENGELLEMEZ
- **All-Agents Drafting (AAD)**: Tartisma oncesi bagimsiz ilk cozumler -> +%3.3
- **Extreme sentiment**: Sentiment < -0.70 -> pozisyon boyutunu %25'e kucult (ENGELLEME DEGIL)
- **En guvenli ajan liderlik**: Belirli domain'de en yuksek calibrated confidence'a sahip ajan onder olur

### Multi-Timeframe Analiz
```
Top-Down Yaklasim (Zorunlu):

1. Haftalik/Gunluk (Makro Bias):
   -> Birincil trend yonu ve yapisal seviyeler
   -> Sadece buyuk resim yonunde trade ac

2. 4S/1S (Ara Onay):
   -> Swing yapisi, destek/direnc zonlari
   -> Rejim siniflandirmasi

3. 15dk/5dk (Giris Zamanlama):
   -> Hassas giris tetikleyicileri
   -> Momentum diverjans, order flow

Confluence Skoru: 0-5 (kac timeframe uyumlu)
  5 = tum timeframe'ler uyumlu -> maksimum pozisyon
  3-4 = cogunluk uyumlu -> standart pozisyon
  1-2 = dusuk confluence -> kucuk pozisyon (%25 of max)
  0 = hic uyum yok -> toz pozisyon (komisyon filtreler, ENGELLEME YOK)
```

### Risk-Reward Framework
```
Pozisyon Boyutlandirma (3 Kisitli):
  Size = min(
    Dollar_Risk / (ATR * Multiplier),     # ATR-bazli
    Kelly% * Portfolio,                     # Kelly-bazli (Quarter Kelly)
    10% * Portfolio / Current_Price         # Max pozisyon limiti
  )

Stop-Loss (ATR-bazli):
  - 5m scalping: 1.5x ATR
  - 4h swing: 2.0x ATR
  - 1d position: 3.0x ATR
  -> ATR x2 ile drawdown %32 azalir (1000 trade calismasi)

Take-Profit:
  - Birincil hedef: En yakin resistance (R:R min 1.5:1)
  - Ikincil hedef: Fibonacci 1.618 extension
  - Trailing stop: Trend devam ederse karı kor
```

### Market Rejim Tespiti
```
Tespit Yontemleri (Hibrit):
  1. HMM (Hidden Markov Model): 3-state (bull/bear/sideways)
  2. Rolling 20-gun ortalama getiri: Pozitif = bullish, negatif = bearish
  3. LLM-bazli nowcasting: Klasik modelleri %9.47 gecer

4 Rejim:
  TRENDING:    Sustained yonlu hareket, genis stoplar
  RANGING:     Tanimli destek/direnc arasi, siki stoplar
  VOLATILE:    Yuksek salgi, kisa timeframe, azaltilmis pozisyon
  COMPRESSION: Azalan volatilite, potansiyel kirilma

Rejim Gecisi Tespiti:
  - RLMF (RL from Market Feedback): %15 tahmin dogruluk artisi
  - UYARI: LLM'ler bull'da asiri muhafazakar, bear'de asiri agresif
    -> Rejim-aware risk override'lar ZORUNLU
```

### Performance Attribution (SHAP)
- Her feature'in bireysel trade kararina katkisini olc
- Hangi sinyal kaynaklari kar getiriyor? (teknik vs sentiment vs on-chain)
- Ablation testing: Her sinyal kaynagini tek tek cikar, marjinal katkiyi olc
- Signal decay monitoring: Information Coefficient zamanla dusuyor mu?
- LLM_trader pattern: Her 10 trade'de granular analiz (ADX level, confluence)

### Alpha Arena Dersleri (Arastirma Bulgulari)
```
DERS 1: Ayni prompt farkli modellerde CILDIZCA farkli sonuc verir
  -> Model-agnostik prompt OLAMAZ, her model icin AYRI optimize et
DERS 2: Prompt'u her model icin backtest et (paper trade)
  -> Canli'ya almadan once minimum 100 sinyal paper trade
DERS 3: Kucuk Cince modeller (Qwen, DeepSeek) trading'de bazi Western modelleri gecti
  -> Model buyuklugu != trading performansi
DERS 4: Multi-model ensemble (voting) tek model'den daha stabil
  -> Self-consistency (3-5x) + majority vote kullan
```

---

## TECH STACK SUMMARY

```
Core Trading:
  - Freqtrade (Python 3.13)
  - FreqUI (Vue 3 + TypeScript)
  - Binance Testnet (spot + futures)

LLM Providers (API-first, failover chain):
  - Gemini Flash (birincil - ucretsiz, hizli, prompt caching)
  - Groq (ikincil - Llama 70B, 6000 tok/s, derin analiz)
  - OpenRouter (ucuncul - 100+ model, son care)

AI/ML Models (yerel, CPU):
  - CryptoBERT (sentiment, 100M, ONNX)
  - FinBERT (sentiment, 110M, ONNX)
  - XGBoost/LightGBM (FreqAI prediction)
  - BGE-Base-Financial-Matryoshka (finans embedding, 110M)
  - ColBERTv2 + PLAID (late interaction retrieval/reranking)
  - all-MiniLM-L6-v2 (hafif fallback embedding, 22M)
  - FlashRank (CPU-native reranking)
  - Qwen3-0.6B / SmolLM2-1.7B (LoRA distilled fallback)

RAG Stack (42 Teknik, 7 Kategori):
  QUERY:     Multi-Query, RAG-Fusion, HyDE, Step-Back, Decomposition, Routing, Rewriting
  RETRIEVAL: Dense(HNSW), Sparse(BM25), ColBERTv2, Parent-Child, Tool-Augmented, Sentence Window
  CHUNKING:  Recursive, Contextual(Anthropic), Parent-Child, RAPTOR, Late(Jina), Proposition
  EMBEDDING: Dual(Gemini+BGE), Matryoshka, Binary Quantization, Embedding Cache
  RERANKING: FlashRank, ColBERT, Multi-Reranker Ensemble
  GRAPH:     LazyGraphRAG, Temporal Graph, CDF-RAG Causal
  MEMORY:    GAM-RAG, MAGMA, MemoRAG, Bidirectional RAG
  TEMPORAL:  TS-RAG, StreamingRAG, Temporal Decay
  GENERATION: Self-RAG, CRAG, Speculative RAG, FLARE, CoT-RAG, MADAM-RAG, Attributed RAG
  OPTIMIZE:  Semantic Cache, Prompt Cache, Query Routing, Offline/Online Split
  EVALUATE:  RAGAS, DeepEval, RAG Triad, Lynx

Vector DB + Storage:
  - ChromaDB (7 collection, HNSW, dual embedding)
  - SQLite-vec (embedding cache, semantic cache, memory, decisions)
  - rank_bm25 (sparse/keyword search)

Data Sources (HABER BEDAVA CEKiLiR - $0/ay):
  Primary:
  - cryptocurrency.cv API (200+ kaynak, built-in sentiment, SSE streaming, key gereksiz)
  RSS Feeds (20+ kaynak, feedparser ile):
  - CoinDesk, CoinTelegraph (7 kategori), Decrypt, The Block, CryptoSlate
  - CryptoPotato, CryptoNews, The Defiant, Bitcoin Magazine, ChainGPT
  Sentiment APIs:
  - alternative.me Fear & Greed Index (key gereksiz, her 5dk)
  - CryptoPanic (free tier, community sentiment votes)
  - Alpha Vantage News Sentiment (free tier, pre-computed skorlar)
  Market Data:
  - CryptoCompare API (free tier, 50 call/saat, haber aggregator)
  - CoinGecko API (prices + market cap)
  - Blockchain Explorer APIs (on-chain data)
  Dogrulama:
  - Gemini Grounding (Google Search entegrasyonu)
  Fallback Scraper:
  - Scrapling (RSS/API olmayan siteler icin, Cloudflare bypass, adaptive parser)

Orchestration:
  - LangChain + LangGraph (agent orchestration)
  - 8 specialized agents (multi-agent trading)

Prompt Engineering:
  - Contract Format system prompts (Role + Criteria + Constraints + Output)
  - FinCoT (Financial Chain-of-Thought, Mermaid diagrams)
  - 5-katman anti-hallucination (Grounding + Attribution + CoVe + Uncertainty Disclosure + Tools)
  - Reflective Few-Shot (CryptoTrade pattern, dinamik trade history)
  - Self-Consistency (3-5x sampling + majority vote)
  - Multi-agent role prompting (TradingAgents 6-role pattern)

Cost Control:
  - LiteLLM Proxy (unified gateway + budget enforcement + failover)
  - 7-katman maliyet optimizasyonu ($5-10/gun -> $0.20-0.50/gun)
  - Gorev-bazli model routing (%40-85 tasarruf)
  - Gemini prefix caching (%90 input tasarrufu)
  - Semantic caching (%65-73 tekrar sorgu tasarrufu)
  - LLMLingua-2 prompt compression (%50 input azaltma)
  - Token counting (tiktoken) + per-agent budgets

AI Autonomy & Trade Execution:
  - Trade-First philosophy (Default-Allow, blacklist not whitelist)
  - 6-level graduated autonomy (Backtest -> Paper -> Micro -> Small -> Standard -> Full)
  - Beta distribution trust tracking (Bayesian, natural asymmetry)
  - Confidence-based continuous position sizing (confidence^alpha curve, NEVER blocks)
  - Bayesian Kelly: self-learning position sizing (losses auto-shrink)
  - Dynamic VaR risk budget (Millennium/Citadel pod model)
  - ONE hard constraint: max_position_cap (3% portfolio)
  - Minimalist safety: 5 non-negotiable + 5 important + 3 nice-to-have
  - Signal-to-trade conversion target: 60-80%
  - Forgone P&L tracking (measure guardrail cost)

Guardrails & Safety (Trade-First Uyumlu):
  - Pydantic + Instructor (structured output enforcement + auto-retry)
  - 5-seviye override hierarchy (Exchange > System > Strategy > AI > Default)
  - Pre-trade SIZING (engelleme degil) + post-trade ADAPTIVE response
  - Circuit breaker pattern (5 hata/30s -> OPEN -> 60s cooldown -> HALF-OPEN)
  - Confidence calibration (self-consistency + Brier score post-hoc)
  - 8-seviye graceful degradation (Gemini -> Groq -> OR -> LoRA -> Cache -> FreqAI -> Rules)
  - ENGELLEME SADECE: Exchange down | Kill switch (DD >%20) | Pair delisted

Decision Framework (Size-Through, Block-Never):
  - Weighted scoring (z-score normalization + regime multiplier)
  - 5-stage size-through pipeline (pre-filter -> analysis -> scoring -> risk-sizing -> exec)
  - Post-trade reflection (CryptoTrade pattern, not pre-trade analysis paralysis)
  - Multi-agent DEBATE (not consensus) - FINCON hierarchy model
  - Confidence -> position size continuous curve (no binary gates)
  - ATR-based position sizing (Bayesian Kelly + trust-level fractional)
  - HMM + LLM hybrid regime detection
  - SHAP-based performance attribution

Infrastructure:
  - Docker + Docker Compose
  - SQLite (trade DB + memory + cache + graphs + LLM logs)
  - Nginx (reverse proxy)
  - Systemd (process management)

Monitoring & Observability:
  - LiteLLM Proxy (cost tracking + budget alerts)
  - Prometheus + Grafana (dashboards + Telegram alerts)
  - RAGAS + DeepEval + RAG Triad + Lynx (RAG quality)
  - Telegram bot (trade alerts + reports + monitoring alerts)
  - SQLite-based LLM call logging (prompt hash, tokens, cost, latency)
  - A/B testing framework (shadow testing for prompt versions)
```

---

## RISK MITIGATION (Comprehensive)

**API & Altyapi Riskleri:**
1. **API Down**: 8-seviye graceful degradation: Gemini -> Groq -> OpenRouter -> LoRA -> Cache -> FreqAI -> Rules
2. **Rate Limit**: Token bucket self-limiting (%80 margin) + semantic cache + prompt cache + exponential backoff + jitter
3. **Rate Limit Tespit**: Her request oncesi token say (tiktoken), RPM/TPM/RPD ayri takip, %80'de uyar
4. **API Maliyet Patlamasi**: 7 katmanli maliyet optimizasyonu + LiteLLM per-agent butce + Telegram alert
5. **Provider Sagligi**: Per-provider health score (success_rate*0.5 + 1/latency*0.3 + availability*0.2)
6. **Circuit Breaker**: 5 hata/30s = OPEN, 60s cooldown, single-probe HALF-OPEN

**Veri & RAG Riskleri:**
7. **Hallucination**: 5 katman prompt + Self-RAG + CRAG + Lynx + citation + fiyat cross-check (7 katman toplam)
8. **Sahte Korelasyon**: CDF-RAG causal reasoning + MADAM-RAG multi-agent debate
9. **Embedding Ceiling**: DeepMind limitini hybrid search + ColBERT + LazyGraphRAG ile as
10. **Eski Veri**: Temporal decay (half-life 1-14 gun) + StreamingRAG real-time + cache invalidation
11. **Bilgi Tabani Kirlenmesi**: Bidirectional RAG grounding check + NLI entailment + novelty detection
12. **Celisken Sinyal**: MADAM-RAG debate (Bull vs Bear) + voting mekanizmasi (veto yok, confidence ayarlama)

**Prompt & Output Riskleri:**
13. **Malformed Output**: Pydantic + Instructor ile structured output enforce + 3 retry with feedback
14. **Prompt Injection**: User input izole, LLM trade execute yetkisi yok, policy check zorunlu
15. **Model Degisikligi**: Model pinning (specific snapshot), prompt A/B test before deploy
16. **Confidence Overcalibration**: Self-consistency ile dogrula, tek LLM skoru'na guvenme

**Trading Riskleri (Trade-First Uyumlu):**
17. **Asiri Pozisyon**: %3 hard cap + Bayesian Kelly + trust level fractional Kelly
18. **Consecutive Loss**: Pozisyon %50 KUCULT (engelleme!), Bayesian Kelly otomatik adapte, hatalardan OGREN
19. **Rejim Yanilgisi**: HMM + LLM hibrit tespit + otomatik parametre degisim + hatayi LOGLA
20. **Signal Decay**: SHAP attribution + IC monitoring + haftalik retraining + forgone P&L tracking
21. **Over-Guardrailing**: Signal-to-trade conversion %60-80 hedefi, forgone P&L haftalik rapor, guardrail cost olcumu

**Sunucu Riskleri:**
22. **RAM Overflow**: 3.15 GB buffer + alarm @%88 + Pazar gecesi 2.65 GB bos
      - LoRA + Yerel Model: Always-resident (buffer'dan calmaz)
      - Scrapling browser: Her zaman calisabilir (Pazar gecesi dahil)
      - ChromaDB memory-mapped: Ihtiyari, sadece >700K doc'ta aktif et
      - swappiness=30 (rahat ama hala swap-averse)
      - UYARI @%88 (28.2 GB): ChromaDB memory-mapped mode
      - KRITIK @%94 (30.1 GB): Semantic cache eviction + agent azalt
23. **CPU Bottleneck**: Batch inference + binary quantization + caching + offline/online ayirimi
24. **Disk Full**: Auto-maintenance cron (DISK ALLOCATION bolumune bak)
      - 180 gun news rolling, 90 gun log rolling, 1 yil backtest rolling
      - ChromaDB 1M doc cap (low-gain eviction), Historical 1m data 1 yil purge
      - ALARM @%85, KRITIK @%90 (agresif purge: tum rolling window %50 kisalt)
      - 5-yillik projeksiyon: purge ile %88 disk kullanimi (guvenli)
25. **Internet Kesintisi**: all-MiniLM (embed) + FreqAI (ML) + LoRA (LLM) + cached yanit
26. **Downtime**: Docker restart policy + systemd watchdog + Telegram alert
27. **5-Yillik Surdurulebilirlik**: Tum buyuyen bilesen icin auto-purge politikasi tanimli
      - RAM: ChromaDB >1M doc eviction, MAGMA >10K node prune, GAM-RAG >50K pattern purge
      - Disk: Rolling window + relevance purge + gain-score eviction
      - Trade history: ASLA SILINMEZ (audit trail)
      - Cron: Her gun 04:00 + Her Pazar 03:00 + Her ay 1. otomatik bakim
      - Telegram: 3-aylik disk/RAM raporu otomatik gonderilir

---

## SUCCESS METRICS

| Metric | Phase 0 Target | Phase 6 Target |
|--------|---------------|----------------|
| Win Rate | >50% | >65% |
| Sharpe Ratio | >0.5 | >1.5 |
| Max Drawdown | <20% | <10% |
| RAG Faithfulness (RAGAS) | N/A | >0.90 |
| RAG Context Precision | N/A | >0.85 |
| RAG Answer Relevancy | N/A | >0.90 |
| Hallucination Rate (Lynx) | N/A | <5% |
| Semantic Cache Hit Rate | N/A | >40% |
| Avg Query Latency | N/A | <2s (simple), <5s (complex) |
| API Cost per Day | $0 | <$0.50 |
| System Uptime | >95% | >99.5% |
| Trade Signal Explainability | None | Full CoT + Citation |
| Self-Improvement Rate | None | Measurable weekly gain |
| Signal-to-Trade Conversion | N/A | 60-80% |
| Forgone P&L (blocked signals) | Not tracked | < executed P&L |
| Autonomy Level | L0 (backtest) | L4-L5 (standard/full) |
| Trust Score (Beta dist.) | 0.50 (uniform) | >0.70 |
| Daily VaR Budget Utilization | N/A | 50-80% |

---

## RESEARCH REFERENCES

Ana Kaynaklar (5+12 arastirma/explore ajani tarafindan derlendi, Mart 2026):

**RAG Surveys:**
- TuringPost: 12 + 16 + 11 = 39 RAG tipi cataloglari
- Bhavishya Pandit: 25 Types of RAG serisi (3 bolum)
- ArXiv: RAG Comprehensive Survey (2506.00054)
- ArXiv: Agentic RAG Survey (2501.09136)

**Kritik Makaleler:**
- Self-RAG (NeurIPS 2023): Asai et al.
- CRAG (ICLR 2024): Yan et al.
- GraphRAG (Microsoft 2024): Local to Global
- LazyGraphRAG (Microsoft 2025): %0.1 maliyet
- GAM-RAG (Mart 2026): Gain-adaptive memory
- MAGMA (Ocak 2026): Multi-graph agentic memory
- TS-RAG (NeurIPS 2025): Time series RAG
- CDF-RAG (2025): Causal dynamic feedback
- MADAM-RAG (2025): Multi-agent debate
- DeepMind LIMIT (Eylul 2025): Embedding tavani kaniti
- Anthropic Contextual Retrieval (2024): %67 hata azaltma
- A-RAG (Subat 2026): Hierarchical agentic RAG
- ComoRAG (AAAI 2026): Cognitive memory-organized RAG
- MemoRAG (TheWebConf 2025): Global corpus memory
- FinSeer (2025): Financial time series retriever

**RAG**: 86+ teknik arastirildi, 42 secildi, 44 elendi
**Prompt Engineering**: 19 teknik arastirildi (system prompt, CoT, FinCoT, ToT, GoT,
  Step-Back, ReAct, few-shot, self-consistency, meta-prompting, DSPy, anti-hallucination,
  CryptoTrade reflective, ATLAS Adaptive-OPRO, context engineering, role-based, prompt security)
**Token Cost**: 7 katmanli optimizasyon, 3 provider rate limit analizi
**Guardrails**: 4 framework (Guardrails AI, Instructor, NeMo, LLM Guard), 5-level override
**Observability**: 8 platform karsilastirildi, LiteLLM+Prometheus+Grafana secildi
**Decision Framework**: 8 gercek dunya sistemi incelendi (TradingAgents, ElliottAgents,
  FinGPT/FinRL, CryptoTrade, FinMem, LLM_trader, FS-ReasoningAgent, Increase Alpha)
**Parametric Planning**: Gorev-bazli temperature/top-p/max_tokens, rejim-bazli ayarlama,
  temporal decay formulu, confidence-bazli pozisyon boyutlandirma, ATR stop-loss

**AI Autonomy & Trade Execution**: 5 arastirma ajani + 12 explore ajani (codebase review), 80+ kaynak
  - Autonomy vs Safety dengesi (Renaissance, Two Sigma, DE Shaw, Citadel modelleri)
  - Graduated autonomy (SAE L0-L5, ALFUS, prop firm scaling, Beta distribution trust)
  - Anti-over-engineering (filter math, minimalist design, opportunity cost)
  - Confidence-based self-regulation (Bayesian Kelly, MC Dropout, conformal prediction, pod model)
  - Basarili sistemler (TradingAgents, CryptoTrade, FinMem, FINCON, TradingGroup)
  - KDD 2026: "Can LLM Strategies Outperform in Long Run?" (rejim farkindaligi > katman karmasikligi)
  - Lee & See 2004: Trust in Automation (developer projection trap)
  - Alpha decay research (Maven Securities, arxiv 2502.04284)
  - Circuit breaker magnet effect (MIT Sloan)
  - Taleb Barbell strategy (antifragile portfolio design)

**Codebase Exploration (12 Explore Agent):**
  - ROADMAP blocking language scan (4 bulgu, hepsi duzeltildi)
  - RAM/Disk budget verification (tasmak yok, 350MB unaccounted -> buffer icinde)
  - Phase timeline realism (orijinal 13 hafta -> gercekci 24 hafta)
  - FreqUI analysis (82 component, 10 store, 12 view, 0 AI component)
  - Strategies analysis (65 strateji, 0 FreqAI, best base: InformativeSample.py)
  - FreqAI deep dive (15 model, 3 abstract method, custom model extension points)
  - Trading flow (12 strategy callback, 3 kritik hook: custom_stake_amount, confirm_entry/exit)
  - RPC/API endpoints (90+ endpoint, WebSocket pub-sub, JWT auth)
  - Config/deployment (JSON schema extensible, env vars, Docker multi-stage)
  - Test infrastructure (1450+ test, 94 dosya, FreqAI test fixtures mevcut)
  - Data pipeline (CustomData per-trade, migration support, ChromaDB in-process mumkun)
  - Cross-section consistency (14 tutarsizlik bulundu, hepsi duzeltildi)

**Toplam: 17 arastirma ajani + 12 explore ajani, 230+ teknik/framework/pattern arastirildi**

---

## FREQTRADE INTEGRATION POINTS (Hook Mapping)
> AI sistemi freqtrade core'u DEGISTIRMEDEN, strategy callback'leri uzerinden entegre olur.
> Asagidaki 12 hook'tan 3'u KRITIK, geri kalani progressive enhancement.

### Kritik Hooklar (AI Entegrasyonu icin ZORUNLU)

| # | Hook | Dosya | Kullanim | Oncelik |
|---|------|-------|----------|---------|
| 1 | `custom_stake_amount()` | strategy/interface.py:620 | **Confidence → Position Size** (ana sizing hook) | KRITIK |
| 2 | `confirm_trade_entry()` | strategy/interface.py:353 | **AI final entry approval** (felaket kontrolu) | KRITIK |
| 3 | `confirm_trade_exit()` | strategy/interface.py:389 | **AI exit approval** (erken cikis onleme) | KRITIK |

### Ek Hooklar (Progressive Enhancement)

| # | Hook | Dosya | Kullanim | Oncelik |
|---|------|-------|----------|---------|
| 4 | `populate_entry_trend()` | strategy/interface.py:228 | Sinyal uretimi (teknik + AI features) | Phase 1 |
| 5 | `populate_exit_trend()` | strategy/interface.py:228 | Cikis sinyali | Phase 1 |
| 6 | `custom_stoploss()` | strategy/interface.py:441 | AI dinamik stoploss (ATR-based) | Phase 4 |
| 7 | `custom_exit()` | strategy/interface.py:471 | AI custom exit kararlari | Phase 4 |
| 8 | `adjust_trade_position()` | strategy/interface.py:649 | AI-driven DCA/pyramiding | Phase 5 |
| 9 | `bot_loop_start()` | strategy/interface.py:228 | Pre-loop AI data refresh | Phase 1 |
| 10 | `order_filled()` | strategy/interface.py:427 | Post-fill confidence tracking | Phase 3.5 |
| 11 | `custom_entry_price()` | strategy/interface.py:501 | AI fiyat optimizasyonu | Phase 5 |
| 12 | `custom_exit_price()` | strategy/interface.py:529 | AI cikis fiyat optimizasyonu | Phase 5 |

### Confidence-Based Sizing Implementation (custom_stake_amount)
```python
def custom_stake_amount(self, pair, current_time, current_rate,
                        proposed_stake, min_stake, max_stake,
                        leverage, entry_tag, side, **kwargs) -> float:
    # 1. AI confidence al
    confidence = self.ai_engine.get_confidence(pair, current_rate)

    # 2. Kelly fraction (Bayesian, trust level'a gore)
    kelly = self.ai_engine.get_kelly_fraction(pair)

    # 3. Confidence^alpha curve
    alpha = 2.0
    sizing_factor = confidence ** alpha
    sized_stake = proposed_stake * kelly * sizing_factor

    # 4. Hard cap: %3 portfolio
    portfolio_cap = self.wallets.get_total() * 0.03
    sized_stake = min(sized_stake, portfolio_cap)

    # 5. Forgone P&L: Eger stake cok kucukse, paper trade olarak logla
    if sized_stake < min_stake:
        self.ai_engine.log_forgone_signal(pair, confidence, proposed_stake)

    return max(min(sized_stake, max_stake), min_stake)
```

### API Endpoint Mapping (FreqUI icin)
```
Yeni AI Endpoint'leri (api_ai.py router):
  GET  /api/v1/ai/confidence/{pair}     - Pair icin confidence skoru
  GET  /api/v1/ai/sentiment/{pair}      - Sentiment analizi
  GET  /api/v1/ai/reasoning/{trade_id}  - Trade karari aciklamasi
  GET  /api/v1/ai/risk                  - Portfolio risk durumu
  GET  /api/v1/ai/forgone-pnl           - Forgone P&L ozeti
  GET  /api/v1/ai/autonomy              - Mevcut otonom seviye + trust score
  GET  /api/v1/ai/models                - Model durumu (Gemini/Groq/LoRA)
  POST /api/v1/ai/settings              - AI ayarlari (autonomy level, params)

Yeni WebSocket Event'leri:
  ai_signal      - Yeni AI sinyali + confidence
  ai_prediction  - Fiyat tahmini
  risk_alert     - Risk esik uyarisi
  autonomy_change - Seviye degisikligi
  model_switch   - Provider failover bildirimi
```

---

## FREQUI AI DASHBOARD PLAN
> Mevcut FreqUI: 82 component, 10 store, 12 view, 30+ endpoint, 0 AI component
> AI entegrasyonu icin asagidaki component'ler PARALEL gelistirilecek

### Yeni Component'ler (src/components/ai/)
```
SentimentDisplay.vue      - Pair bazli sentiment gostergesi (bullish/bearish/neutral)
ConfidenceScore.vue       - Sinyal confidence % + trend + renk kodlama
TradeReasoning.vue        - LLM karar aciklamasi (collapsible detay)
AutonomyLevel.vue         - Mevcut seviye (L0-L5) + terfi ilerleme + trust curve
RiskPanel.vue             - VaR butce kullanimi + drawdown + rejim
ForgnePnLTracker.vue      - Reddedilen sinyallerin P&L karsilastirmasi
ModelStatusCard.vue       - Provider durumu (Gemini/Groq/LoRA) + latency + cost
AISignalPanel.vue         - Son AI sinyalleri listesi + confidence
```

### Yeni Store (src/stores/aiStore.ts)
```typescript
// Pinia store for AI state
- currentSentiment: Map<pair, SentimentData>
- confidenceScores: Map<pair, ConfidenceScore>
- tradeReasonings: Map<tradeId, Reasoning>
- autonomyLevel: number (0-5)
- trustScore: number (0-1)
- riskBudget: { used: number, total: number }
- forgnePnl: ForgnePnLSummary
- modelStatus: Map<provider, ModelStatus>
```

### Yeni View'lar
```
AISettingsView.vue     - AI konfigurasyonu (model, autonomy, params)
AIAnalyticsView.vue    - Derin AI analitik (signal quality, cost, attribution)
RiskDashboardView.vue  - Risk odakli gorunum (VaR, drawdown, exposure)
```

### GelistirmeTakvimi (Ana phase'lerle PARALEL)
```
Phase 1 ile paralel:  SentimentDisplay + ModelStatusCard
Phase 3.5 ile paralel: ConfidenceScore + RiskPanel + AutonomyLevel
Phase 4 ile paralel:  TradeReasoning + AISignalPanel
Phase 5 ile paralel:  ForgnePnLTracker + aiStore tamamla
Phase 6 ile paralel:  AIAnalyticsView + RiskDashboardView + AISettingsView
```

---

## REALISTIC TIMELINE SUMMARY
> Orijinal plan: 13 hafta (3 ay). Gercekci plan: 24 hafta (6 ay).
> Solo developer, dedicated server, full-time varsayimi.

```
Phase 0:   Week 1       (0.5 hafta)  - Foundation & Infrastructure
Phase 1:   Week 2-3     (1.5 hafta)  - Sentiment Engine
Phase 2:   Week 3-6     (4 hafta)    - Hybrid RAG Foundation (14 teknik)
Phase 3:   Week 7-12    (6 hafta)    - Advanced RAG (15 teknik, 3 tier)
Phase 3.5: Week 13-14   (2 hafta)    - Autonomy Scaffolding (YENi)
Phase 4:   Week 15-16   (2 hafta)    - Gemini Trading Brain
Phase 5:   Week 17-18   (2 hafta)    - Multi-Agent System
Phase 6:   Week 19-22   (4 hafta)    - Feedback Loop & Self-Improvement
Phase 7:   Week 23+     (ongoing)    - Deep Analysis & Expansion
FreqUI:    Paralel      (surekli)    - AI Dashboard development

TOPLAM:    ~24 hafta (~6 ay)

MVP Boundary (Week 14 - 3.5 ay):
  Phase 0-2 + Phase 3 Tier 1 + Phase 3.5 tamamlandi
  = Calisir durumda: sentiment + hybrid RAG + Self-RAG + CRAG + LazyGraphRAG
    + confidence sizing + risk budget + graduated autonomy altyapisi
  = Binance testnet'te paper trade'e hazir
```

### RAG Teknik Dagilimi (Phase bazli)
```
Phase 2 (14 teknik):
  EMBEDDING: Dual, Matryoshka, Binary Quant, Cache (4)
  CHUNKING:  Recursive, Contextual, Parent-Child, RAPTOR, Late, Proposition (6)
  RETRIEVAL: Dense, BM25, ColBERTv2, Parent-Child, Sentence Window, Tool-Augmented (6 - retrieval tarafta 3 core + 3 pattern)
  SEARCH:    Hybrid + RRF (1)
  RERANKING: FlashRank, ColBERT, Multi-Ensemble (3)
  OPTIMIZE:  Semantic Cache, Prompt Cache, Query Routing, Offline/Online (4)
  -> Toplam unique teknik: 14 (bazi teknikler chunking+retrieval kategorisinde cift sayilir)

Phase 3 Tier 1 (3 teknik - KRITIK):
  Self-RAG, CRAG, LazyGraphRAG

Phase 3 Tier 2 (4 teknik - BELLEK+ZAMAN):
  MAGMA, TS-RAG, Temporal Decay, GAM-RAG

Phase 3 Tier 3 (8 teknik - MVP SONRASI DEFER EDILEBILIR):
  CDF-RAG, MADAM-RAG, FLARE, HyDE, Speculative RAG, CoT-RAG, Adaptive RAG, RAG-Fusion

Phase 4 (3 teknik):
  Bidirectional RAG, MemoRAG, StreamingRAG

Phase 5 (1 teknik):
  Attributed/Faithful RAG (audit trail)

Phase 6 (4 framework):
  RAGAS, DeepEval, RAG Triad, Lynx
```
