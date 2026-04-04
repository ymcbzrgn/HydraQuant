# Phase 23: Jina API Migration — Yapılacaklar

## Amaç
ColBERT + BGE + FlashRank local modellerini kaldırıp Jina API'ye geçiş.
3.5GB RAM kurtarma, OOM sorununu kökten bitirme.

## Yapılacaklar

### 1. Jina Reranker API Entegrasyonu
- `colbert_reranker.py` → Jina Reranker API çağrısına dönüştür
- `hybrid_retriever.py` → FlashRank + ColBERT yerine Jina Reranker kullan
- Model: `jina-reranker-v2-base-multilingual`
- Free tier: 1M token/ay, 500 RPM
- Dimension sorunu YOK (reranker raw text alır, skor döner)

### 2. BGE Embedding Kaldırma
- `rag_embedding.py` → BGE fallback'i kaldır, Gemini embedding-001 primary (768 dim)
- ChromaDB'deki mevcut vektörler zaten 768 dim (uyumlu)
- Dual embedding → single Gemini embedding

### 3. Model Server Kaldırma
- `model_server.py` → arşivle (local isteyenler için)
- `freqtrade-models.service` → disable et
- systemd MemoryMax=4GB limiti artık irrelevant

### 4. Kod Temizliği
- `model_server.py` → `model_server_local.py` olarak rename (arşiv)
- Config'e `RERANKER_MODE=jina|local` flag ekle
- README'ye "Local vs API" dökümanı yaz

### 5. Jina API Key
- `.env`'ye `JINA_API_KEY` ekle
- https://jina.ai adresinden free key al

### 6. Etkilenen Dosyalar
- `user_data/scripts/colbert_reranker.py` — Jina HTTP client'a dönüştür
- `user_data/scripts/hybrid_retriever.py` — FlashRank/ColBERT → Jina Reranker
- `user_data/scripts/rag_embedding.py` — BGE fallback kaldır
- `user_data/scripts/model_server.py` — arşivle
- `user_data/scripts/rag_graph.py` — model server health check kaldır

### 7. Doğrulama
- `freqtrade-models` servisi olmadan tüm RAG pipeline çalışmalı
- Reranking kalitesi düşmemeli (Jina MTEB'de ColBERT'ten iyi)
- RAM kullanımı ~3.5GB düşmeli
- 0 OOM kill

## Notlar
- systemd'de `MemoryMax=4294967296` (4GB) limiti var — model server kalkınca irrelevant
- Local model desteğini kodda tut ama optional yap
- Projeyi klonlayanlar için "API-only" default olsun
