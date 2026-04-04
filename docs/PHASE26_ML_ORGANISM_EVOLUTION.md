# Phase 26: ML-Powered Neural Organism — Kural Tabanlıdan Gerçek ML'e Geçiş

## Prereq: Phase 23 (Jina Migration) tamamlanmalı — 3.5GB RAM boşalacak

## Mevcut Durum (Phase 25)

Organizma yaşıyor ama "beyni" kural tabanlı:
- Thompson Sampling (Beta dağılımı) — istatistiksel ama ML DEĞİL
- BCM sliding threshold — biyolojik ilham ama hala formül
- Hormonlar — sabit formüller (stress = 0.3 * (1 - fng/20))
- Credit assignment — organ bazlı ağırlıklar, öğrenilmiş değil
- Hippocampus — exact fingerprint match, fuzzy pattern recognition YOK
- PredictiveModel — ortalama PnL tahmin, gerçek prediction DEĞİL

## Hedef

Her subsistemi gerçek ML modelleri ile değiştirmek. Jina migration sonrası 3.5GB RAM + API-based embedding = ML için yeterli kaynak.

---

## Evrim Planı: 8 ML Upgrade

### 1. Hippocampus → Embedding-based Similarity Search
**Şimdi:** Exact JSON fingerprint match (bucketed kategoriler)
**ML ile:** Situation fingerprint → Jina/Gemini embedding → ChromaDB cosine similarity
- Fuzzy matching: "F&G=11 + trending_bear" ile "F&G=14 + trending_bear" benzer olarak tanınır
- Continuous features: F&G=9 ve F&G=15 arasındaki FARK da öğrenilir
- **Model:** Jina embedding API (free tier, zaten entegre olacak)
- **RAM:** ~0 (API-based)
- **Satır:** Hippocampus.store_episode() ve recall() güncelle, ChromaDB collection ekle

### 2. PredictiveModel → Lightweight Neural Network
**Şimdi:** Ortalama geçmiş PnL (çok basit)
**ML ile:** 2-layer MLP: situation_features → predicted_optimal_params
- Input: 12 feature (F&G, ADX, funding, regime_onehot, volatility, hour, consec_losses, ...)
- Output: top 10 kritik parametre tahmin değerleri
- Train: her trade kapandığında online learning (SGD, 1 batch)
- **Model:** PyTorch veya scikit-learn MLPRegressor (~50KB model)
- **RAM:** ~20MB (tiny model)
- **Satır:** ~100 (train + predict + feature engineering)

### 3. Credit Assignment → Gradient-Based Attribution
**Şimdi:** Organ bazlı sabit ağırlıklar (signal=0.5, sizing=0.3, defense=0.4)
**ML ile:** Integrated Gradients veya SHAP — hangi parametre GERÇEKTEN sonucu etkiledi?
- Trade sonucu bir "forward pass" gibi düşünülür
- Gradient: ∂PnL/∂param_i = her parametrenin gerçek katkısı
- Attention-like weighting: parametreler arası ilişki öğrenilir
- **Model:** Basit linear regression + gradient (PyTorch autograd)
- **RAM:** ~10MB
- **Satır:** ~80

### 4. Hormones → Learned Stress Model
**Şimdi:** Sabit formül (cortisol = 1.0 - stress * 0.4)
**ML ile:** F&G + drawdown + consec_losses + ... → optimal hormone levels
- Train: geçmiş trade'lerde "hangi hormon seviyesi en iyi sonucu verdi?"
- Online learning: her trade'den sonra hormon-sonuç ilişkisini güncelle
- Non-linear: Random Forest veya XGBoost (freqtrade zaten kullanıyor)
- **Model:** XGBoost regressor, 4 output (cortisol, dopamine, serotonin, adrenaline)
- **RAM:** ~15MB
- **Satır:** ~80

### 5. Synapse Discovery → Correlation Learning
**Şimdi:** 12 sabit seed synapse (hardcoded kausal zincirler)
**ML ile:** Parametre değişimleri arasındaki korelasyonu otomatik keşfet
- Pearson/Spearman correlation matrix (293 x 293)
- Anlamlı korelasyonlar (|r| > 0.3, p < 0.05) → yeni synapse
- Granger causality: A'nın değişimi B'nin değişimini tahmin ediyor mu?
- **Model:** Scipy stats + rolling correlation window
- **RAM:** ~30MB (correlation matrix)
- **Satır:** ~100

### 6. Cerebellum → Time-Series Pattern Recognition
**Şimdi:** 24-slot win/loss counter
**ML ile:** Saat + gün + ay bazlı performans pattern'leri
- Feature: hour_of_day, day_of_week, month, is_weekend, session (asia/europe/us)
- Hidden patterns: "Pazartesi sabah Asya session'ında ETH short daha iyi"
- **Model:** Decision tree veya small GBM
- **RAM:** ~5MB
- **Satır:** ~60

### 7. Mirror Neurons → Order Flow Toxicity (VPIN)
**Şimdi:** Basit funding rate + L/S ratio hesabı
**ML ile:** VPIN (Volume-Synchronized Probability of Informed Trading)
- Trade bazlı volume buckets → informed vs uninformed trade classification
- Tick-level data (Bybit WebSocket) → real-time toxicity estimate
- Research: Easley-Lopez de Prado-O'Hara (2012), crypto adaptation (2025)
- **Model:** VPIN algorithm + EMA smoothing
- **RAM:** ~50MB (tick data buffer)
- **Satır:** ~150 (WebSocket + VPIN calculation)

### 8. NeuroEvolution → CMA-ES / Optuna Hyperparameter Optimization
**Şimdi:** 5 genome population, basit blend
**ML ile:** CMA-ES (Covariance Matrix Adaptation Evolution Strategy)
- Population-based optimization of parameter sets
- Novelty search: reward behavioral diversity
- Walk-forward validation: train on period A, validate on period B
- **Model:** Optuna veya CMA-ES library
- **RAM:** ~20MB
- **Satır:** ~120

---

## RAM Bütçesi (Jina Migration Sonrası)

| Bileşen | Önceki | Sonrası |
|---------|--------|---------|
| Model Server (ColBERT+BGE+FlashRank) | 3.5GB | **0** (kaldırıldı) |
| ML Organism (8 model) | 0 | **~150MB** |
| Jina API calls | 0 | **~0** (API-based) |
| **Net kazanç** | | **~3.35GB** |

## Öncelik Sırası

| # | Upgrade | Etki | Zorluk | Öncelik |
|---|---------|------|--------|---------|
| 1 | Hippocampus → embedding search | Çok yüksek | Kolay (Jina zaten var) | P0 |
| 2 | PredictiveModel → MLP | Yüksek | Orta | P0 |
| 3 | Hormones → XGBoost | Yüksek | Orta | P1 |
| 4 | Credit Assignment → gradients | Yüksek | Orta | P1 |
| 5 | Synapse Discovery → correlation | Orta | Kolay | P1 |
| 6 | Cerebellum → time-series | Orta | Kolay | P2 |
| 7 | NeuroEvolution → CMA-ES | Orta | Zor | P2 |
| 8 | Mirror → VPIN | Orta | Zor | P2 |

## Geçiş Stratejisi

1. Her ML upgrade AYNI `_p()` interface'ini kullanır — consumer dosyalar DEĞİŞMEZ
2. ML modeli çökerse → mevcut kural tabanlı subsistem fallback olarak kalır
3. Online learning: her trade bir training sample, batch training gereksiz
4. A/B test: ML vs rule-based çıktıları karşılaştırılır, ML daha iyiyse aktif olur
5. Tüm model ağırlıkları SQLite'da saklanır (PyTorch state_dict → blob)

## Timeline

- **Gün 1-2 (Phase 23):** Jina migration, model server kaldır, 3.5GB RAM boşalt
- **Gün 3-4:** Hippocampus → Jina embedding + ChromaDB similarity search
- **Gün 5-7:** PredictiveModel → MLP + online learning
- **Hafta 2:** Hormones → XGBoost + Credit → gradient attribution
- **Hafta 3:** Synapse discovery + Cerebellum upgrade
- **Hafta 4:** NeuroEvolution → CMA-ES + VPIN mirror neurons

## Notlar

- FreqAI zaten XGBoost/LightGBM/PyTorch destekliyor (freqtrade/freqai/)
- Mevcut FreqAI abstract methods: train(), fit(), predict() — aynı pattern kullanılabilir
- scikit-learn, XGBoost, PyTorch zaten requirements'da var
- Jina reranker API: 1M token/ay free, dimension sorunu yok (raw text → score)
- ChromaDB: zaten 7 collection var, 1 tane daha (hippocampus_ml) sorun olmaz
