# Phase 26: CAAT — Cognitive Architecture for Autonomous Trading

## A Manifesto on Living Financial Intelligence

> "The measure of intelligence is the ability to change." — Albert Einstein
>
> "The organism does not merely react to the environment; it actively constructs
> its own umwelt through predictive processing." — Karl Friston, 2010
>
> "We are not building a trading bot. We are birthing a mind that trades."

## Kural Tabanlı Organizmadan Gerçek ML Zekasına Geçiş

**Prereq:** Phase 23 (Jina Migration) → 3.5GB RAM boşaltma
**Mevcut:** Phase 25 Neural Organism — 1758 nöron, 14 subsistem, kural tabanlı (Thompson Sampling + BCM + STDP)
**Hedef:** İnsan bilişsel mimarisini TAM modelleme — algı, hayal gücü, nedensellik, irade, büyüme, bilgelik

---

## Mimari Felsefe: Global Workspace Theory

**Katmanlar DEĞİL, paralel bilişsel süreçler.** İnsan beyni pipeline değil, eşzamanlı bir orkestra.

Her modül farklı hızda, farklı tetikle, ama AYNI paylaşılan workspace'i okuyor/yazıyor:

```
    ┌───────────────┐  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐
    │   ALGI (TTM)  │  │  HAYAL GÜCÜ   │  │  NEDENSELLİK  │  │  META-ÖĞREN   │
    │   Perception  │  │  World Model  │  │    Causal     │  │  Meta-Learn   │
    │  (her candle) │  │ (background)  │  │  (on event)   │  │  (haftalık)   │
    └───────┬───────┘  └───────┬───────┘  └───────┬───────┘  └───────┬───────┘
            │                  │                  │                  │
            ▼                  ▼                  ▼                  ▼
    ╔══════════════════════════════════════════════════════════════════════╗
    ║              GLOBAL WORKSPACE (Shared State)                       ║
    ║  Market Embedding z + World Predictions + Causal Graph +           ║
    ║  Uncertainty Bounds + Hormone State + Neuron Values +              ║
    ║  OOD Score + Regime Embedding + Counterfactuals                    ║
    ╚══════════════════════════════════════════════════════════════════════╝
            ▲                  ▲                  ▲                  ▲
            │                  │                  │                  │
    ┌───────┴───────┐  ┌───────┴───────┐  ┌───────┴───────┐  ┌───────┴───────┐
    │   KARAR (RL)  │  │  BELİRSİZLİK  │  │  BAĞIŞIKLIK   │  │  ZAMANLAMA    │
    │  Multi-Agent  │  │  Uncertainty  │  │   Immunity    │  │  Cerebellum   │
    │ (sinyal anı)  │  │ (her tahmin)  │  │ (her tehdit)  │  │  (her saat)   │
    └───────────────┘  └───────────────┘  └───────────────┘  └───────────────┘
```

**Her süreç bağımsız çalışır ama tümü Global Workspace üzerinden haberleşir.**

---

## 6 Bilişsel Süreç — Detaylı Mimari

### 1. ALGI — Perception (Temporal Transformer)
**Biyolojik karşılık:** Retina + Görsel korteks
**Tetik:** Her candle (sürekli)
**Ne yapıyor:** Ham piyasa verisi → anlamlı latent temsil (embedding)

**Model: TTM (Tiny Time Mixer) — IBM Research, NeurIPS 2024**
- 1M parametre — 62x büyük modellerden %4-40 daha iyi (TimesFM, Chronos, Moirai)
- CPU-native tasarım, GPU gerektirmez
- Adaptive Patching: farklı zaman ölçeklerini otomatik ayarlar
- Resolution Prefix Tuning: multi-timeframe (1h + 4h + daily) tek modelde
- HuggingFace: `ibm-research/ttm-research-r2`
- ONNX INT8 quantized: <10ms inference

**Input:** 
```
[1h candles × 24] + [4h candles × 12] + [daily candles × 7]
+ [F&G, funding_rate, OI, L/S_ratio, BTC_dominance, VIX, DXY]
= ~200 feature time-series
```

**Output:** 64-dim market embedding `z` → Global Workspace'e yazılır

**Self-supervised pretraining (etiket gereksiz):**
- Contrastive learning: "aynı rejimden gelen candle'lar benzer, farklı rejimlerden gelen farklı" 
- Masked reconstruction: rastgele %15 feature'ı sil, tahmin et
- Next-candle prediction: pretext task
- LENS framework (2024): 100B financial observation üzerinde contrastive + reconstruction
- Contrastive Asset Embeddings (ACM ICAIF 2024): hypothesis-testing based sampling

**RAM:** ~20MB (1M param × 4 byte + ONNX runtime)
**Latency:** <10ms per forward pass

---

### 2. HAYAL GÜCÜ — World Model (JEPA-inspired RSSM)
**Biyolojik karşılık:** Prefrontal korteks + İmajinasyon
**Tetik:** Her sinyal üretildiğinde (background)
**Ne yapıyor:** "Bu parametrelerle ne olur?" sorusunu yanıtlar — 1000 gelecek simüle eder

**Model: JEPA-inspired Simplified RSSM (~200-400K param)**

Neden DreamerV3 değil: DreamerV3 görüntü tabanlı, bizim verimiz tabular (50-100 feature). Overkill.
Neden JEPA: Embedding space'te tahmin yapar (ham veri space'inde değil) — gürültülü finansal veri için ideal.

```
Architecture:
  Encoder: TTM embedding z (64-dim) → already computed by Perception
  Recurrent: GRU(128) — temporal dynamics
  Stochastic: Gaussian(32-dim) — uncertainty in dynamics
  Predictor: MLP(128→64→64) — predict z_next + reward
  
  Total: ~300K parameters
  Speed: 5000 forward passes/sec on CPU
  Imagination: 1000 rollouts × 24 steps = ~5 seconds
```

**Nasıl çalışır:**
1. Perception'dan `z_current` al (Global Workspace'ten oku)
2. 1000 farklı parametre konfigürasyonu dene (Latin Hypercube Sampling)
3. Her biri için 24-adım geleceği simüle et (dünya modeliyle)
4. Her simülasyonun beklenen PnL'ini hesapla
5. En iyi 10 konfigürasyonu Global Workspace'e yaz
6. RL karar verirken bu simülasyonları kullanır

**MuZero/MCTS entegrasyonu (opsiyonel):**
- LightZero (NeurIPS 2023) toolkit ile MCTS tree search
- Parametre optimizasyonu bir "oyun" gibi — her "hamle" bir parametre ayarı
- MCTS en iyi hamle dizisini bulur

**RAM:** ~15MB
**Latency:** 5 saniye (1000 imagination rollout)

---

### 3. NEDENSELLİK — Causal Inference (SCM + Counterfactual)
**Biyolojik karşılık:** Akıl yürütme + "Neden?" sorusu
**Tetik:** Trade kapandığında (event-driven)
**Ne yapıyor:** Korelasyon değil NEDENSELLIK öğrenir. "F&G düşükken sizing daraltmak NEDEN işe yarıyor?"

**Pearl Causal Hierarchy — 3 seviye:**

| Seviye | Soru | Araç | Örnek |
|--------|------|------|-------|
| L1 Observation | "Ne oldu?" | P(Y\|X) | "F&G=9 iken PnL ne?" |
| L2 Intervention | "Yaparsam ne olur?" | P(Y\|do(X)) | "F&G threshold'u 15'e ayarlarsam ne olur?" |
| L3 Counterfactual | "Farklı yapsaydım ne olurdu?" | P(Y_x'\|X=x, Y=y) | "Bu kaybeden trade'de threshold farklı olsaydı kazanır mıydık?" |

**Araçlar:**

1. **Tigramite PCMCI+** — Temporal causal graph discovery
   - Zaman serisinden causal graph çıkarır
   - "F&G → crowd_score → confidence → sizing" zincirini VERİDEN keşfeder
   - Mevcut 12 seed synapse'ımızı DOĞRULAR veya YANLIŞ OLANLARI ELER
   - Python: `pip install tigramite`

2. **DoWhy** — Interventional queries + counterfactual estimation
   - "do(threshold=15)" → expected PnL?
   - "Bu trade'de threshold 20 olsaydı?" → counterfactual PnL
   - Python: `pip install dowhy`

3. **SPACETIME (AAAI 2025)** — Regime-aware causal discovery
   - Non-stationary causal yapıyı keşfeder
   - Rejim değişim noktalarını otomatik bulur
   - "Bull market'ta F&G→PnL causal, bear market'ta değil" gibi

4. **CausalStock (NeurIPS 2024)** — End-to-end multi-asset causal prediction
   - Coinler arası causal ilişkileri keşfeder
   - "BTC düşüşü → 4h sonra altcoin düşüşü" causal mı yoksa korelasyon mu?

**En yüksek ROI konsept: Counterfactual Trading**
- 1000 trade'in her birini farklı parametrelerle "yeniden oyna"
- "Bu kaybeden trade'de sizing %50 daha küçük olsaydı?" → hesapla
- Efektif dataset: 1000 trade × 100 counterfactual = 100K veri noktası
- **Data scarcity sorununu KÖKTEN çözer**

**Structural Causal Model (SCM) for Trading:**
```
Market State → Signal Quality → Position Size → Trade Outcome
     ↑              ↑               ↑              ↑
  [exogenous]   [parameters]    [parameters]   [market noise]
  
Intervention: do(F&G_threshold = 15)
Query: P(PnL > 0 | do(F&G_threshold = 15), regime = trending_bear)
```

**RAM:** ~30MB (causal graph + counterfactual engine)

---

### 4. KARAR — Multi-Agent RL (Hierarchical Actor-Critic)
**Biyolojik karşılık:** İrade + Karar verme + Motor kontrol
**Tetik:** Sinyal üretildiğinde (on-demand)
**Ne yapıyor:** Tüm bilgiyi (embedding, simülasyon, nedensellik, belirsizlik) alıp optimal parametre kararı verir

**3 Aşamalı Pipeline:**

**Aşama 1 — Offline Pre-training: IQL (Implicit Q-Learning)**
- FreqTrade backtesting ile 50K+ episode üret
- IQL: OOD (out-of-distribution) action'ları asla sorgulamaz — güvenli
- Her organ agent'ı için ayrı IQL pre-training
- Library: CORL (`pip install -r requirements.txt`)
- Tek seferlik, GPU gerektirmez (1-2 saat CPU)

**Aşama 2 — Online Fine-tuning: SAC (Soft Actor-Critic)**
- IQL'den başlatılmış Q-function ile SAC online fine-tuning
- Entropy regularization: exploration ve exploitation dengesi
- Off-policy: her veri noktası tekrar kullanılır (sample efficient)
- Cal-QL (2024): offline→online geçişi smooth eder
- Library: stable-baselines3

**Aşama 3 — Hierarchical Meta-Policy: HRL**
```
Meta-Policy (PPO — hangi organ ayarlanmalı?)
  │
  ├── Organ Agent 1: Sizing (SAC, ~30 param)
  │     Tetik: PnL beklentiden sapıyor
  │
  ├── Organ Agent 2: Confidence (SAC, ~40 param)
  │     Tetik: Sinyal kalitesi değişiyor
  │
  ├── Organ Agent 3: Defense (SAC, ~50 param)
  │     Tetik: Drawdown artıyor
  │
  ├── Organ Agent 4: Timing (SAC, ~30 param)
  │     Tetik: Cerebellum saat performansı değişiyor
  │
  └── Organ Agent 5: Memory (SAC, ~40 param)
        Tetik: Yeni rejim algılandı
```

**Hi-DARTS (2025) ilhamı:** Meta-agent piyasa volatilitesini analiz eder, uygun sub-agent'ı aktive eder. 25.17% getiri, 0.75 Sharpe.

**DT-LoRA-GPT2 alternatifi (ICAIF 2024):**
- GPT-2 small + LoRA rank=16 = sadece 900K trainable param
- RL'i sequence prediction olarak çözer
- Return-to-go: "Sonraki 24h'de %2 getiri istiyorum" → optimal parametre ayarları
- Attention mekanizması hangi parametrelerin önemli olduğunu öğrenir

**Safe RL — CMDP kısıtları:**
```python
constraints = {
    'max_drawdown': 0.25,           # Portfolio drawdown < 25%
    'max_single_position': 0.03,    # Tek pozisyon < %3
    'portfolio_heat': 0.10,         # Toplam açık risk < %10
}
# PPO-Lagrangian: max E[PnL] subject to E[constraint_violation] < epsilon
```

**RAM:** ~300MB (IQL pre-train) → ~100MB (online SAC)

---

### 5. META-ÖĞRENME — Learning to Learn (Reptile + EWC + Curriculum)
**Biyolojik karşılık:** Büyüme + Adaptasyon + Olgunlaşma
**Tetik:** Haftalık (slow loop)
**Ne yapıyor:** "Nasıl ÖĞRENECEĞİMİ öğren" — 5 trade'de yeni rejime adapte ol

**Reptile (OpenAI) — Pratik MAML alternatifi:**
```python
theta = initial_organism_params  # 293 param
for episode in range(meta_episodes):
    task = sample_regime_from_backtests()  # bull, bear, crash, range, recovery, transition
    phi = theta.clone()
    for k in range(5):  # 5-10 inner step yeterli
        loss = evaluate_on_task(phi, task)
        phi = phi - alpha * grad(loss, phi)
    theta = theta + beta * (phi - theta)  # meta-update
```
- MAML ile aynı performans, yarı bellek, 2. türev YOK
- Library: learn2learn (`pip install learn2learn`)

**EWC (Elastic Weight Consolidation) — Catastrophic forgetting önleme:**
- Önceki rejimlerde önemli olan parametreleri yavaş değiştir
- Fisher Information Matrix: her parametrenin "önemi"
- %45.7 forgetting azalması kanıtlanmış
- 10 satır PyTorch kodu

**LRRL (2024) — Bandit-based learning rate:**
- Learning rate'i multi-armed bandit olarak seç
- Adaylar: [1e-5, 3e-5, 1e-4, 3e-4, 1e-3]
- UCB1 ile en iyi olanı seç (reward = rolling Sharpe improvement)
- Statik/cosine/cyclic schedule'lardan daha iyi

**Curriculum Learning — Trading-R1 (2025) ilhamı:**
```
Stage 1 (Kolay): Güçlü trendler, tüm sinyaller uyumlu, düşük volatilite
Stage 2 (Orta):  Pullback'li trendler, bazı çelişen sinyaller
Stage 3 (Zor):   Range-bound, yüksek volatilite, ince likidite, çelişen sinyaller
```
- Trading-R1: 3 aşamalı curriculum ile 2.72 Sharpe (GPT-4.1'den iyi)

**Sim2Real Transfer — Continual Domain Randomization (CDR, 2024):**
```
Aşama 0: Saf backtest (randomization yok) → baseline
Aşama 1: Randomized backtest (slippage, fee, spread, volume noise) → robustness
Aşama 2: Paper trading (FreqTrade dry-run) → live data validation
Aşama 3: Minimum-size live (graduated execution) → real deployment
Aşama 4: Standard-size live → production
```

**Continual Learning stack:**
- EWC (unutmayı önle) + L2 Init (stabilite çapası) + SNR (ölü nöronları resetle)
- Dynamic Neuroplastic Networks (2025): finansal karar alma için özel tasarım

**RAM:** ~200MB (Reptile meta-training)

---

### 6. BELİRSİZLİK — Uncertainty (Ensemble + Conformal + OOD)
**Biyolojik karşılık:** Alçakgönüllülük + "Bilmiyorum" diyebilme
**Tetik:** Her prediction ile (sürekli)
**Ne yapıyor:** "Ne kadar eminim?" sorusunu yanıtlar — belirsizlik yüksekse pozisyon küçültür

**Deep Ensembles (Lakshminarayanan 2017) — Altın standart:**
```python
class OrganismEnsemble:
    def __init__(self, n_models=5):
        self.models = [SmallMLP(input=50, hidden=64) for _ in range(5)]
    
    def predict_with_uncertainty(self, state):
        preds = [m(state) for m in self.models]
        mean = torch.stack(preds).mean(dim=0)
        variance = torch.stack(preds).var(dim=0)  # uncertainty
        return mean, variance
    
    def uncertainty_to_sizing(self, variance):
        return 1.0 / (1.0 + 10 * variance)  # high uncertainty → small position
```
- 5 bağımsız model, uyuşmazlık = belirsizlik
- Mimari değişiklik GEREKMEZ
- Trivially parallelizable
- CFA Research Foundation 2025: "finans'ta ensemble gold standard"

**KRİTİK BULGU — Epistemic/Aleatoric ayrımı GÜVENİLMEZ:**
- NeurIPS 2024: r=0.8-0.999 korelasyon — pratik olarak ayrılamıyor
- ICLR 2025: "Epistemic uncertainty methods are fundamentally incomplete"
- **TOPLAM belirsizliği kullan, ayrıştırmaya çalışma**

**Conformal Prediction — ACI (Adaptive Conformal Inference):**
- MAPIE kütüphanesi (scikit-learn compatible): `pip install mapie`
- Dağılım-bağımsız tahmin aralıkları: "PnL %95 olasılıkla [-3%, +8%] arasında"
- ACI: Rejim değiştiğinde aralıklar otomatik genişler/daralır
- CPPS (Kato 2024): Prediction interval width → pozisyon büyüklüğü

**OOD Detection — Mahalanobis Distance:**
```python
class MarketOODDetector:
    def fit(self, features, regime_labels):
        # Her rejim için Gaussian fit
        for regime in ['bull', 'bear', 'range', 'crash', 'recovery']:
            self.class_means[regime] = features[mask].mean()
        self.precision = torch.inverse(cov + 1e-6 * I)
    
    def is_ood(self, x):
        score = min(mahalanobis(x, mean, precision) for mean in self.class_means.values())
        return score > chi2.ppf(0.95, df=feature_dim)
```
- "Bu piyasayı daha önce hiç görmedim" → defansif mod (%50-75 sizing azaltma)
- NeurIPS 2024: "Mahalanobis performs best for OOD detection"
- Mevcut k-NN altyapımızla aynı mantık

**Calibration — Online Platt Scaling (mevcut!) + ECE monitoring:**
- OPS zaten var (confidence_calibrator.py)
- ECE dashboard: her confidence bin'de expected vs actual accuracy
- GETS (ICLR 2025 Spotlight): input-adaptive temperature scaling

**RAM:** ~200MB (5-model ensemble)

---

## Data Pipeline: Sim2Real

FreqTrade backtesting engine = sınırsız sentetik episode üretici.

**Episode üretim hedefi:**

| Aşama | Episode sayısı | Amaç |
|-------|---------------|------|
| Exploration | 1K-5K | Ortam doğrulama, debug |
| Convergence | 10K-50K | Temel pattern öğrenme |
| Robust | 50K-200K | Domain randomization |
| Production | 200K+ | Tüm rejim kapsama |

**Domain randomization parametreleri:**
```python
randomization = {
    'slippage': Uniform(0.0, 0.003),       # 0-0.3%
    'fee_mult': Uniform(0.5, 2.0),         # 50-200% normal fee
    'latency_ms': Uniform(0, 500),         # Execution delay
    'spread_mult': Uniform(1.0, 3.0),      # Wider spreads
    'volume_noise': Normal(1.0, 0.3),      # Volume uncertainty
    'regime_shift_prob': Beta(2, 20),       # Random regime changes
    'missing_candle_prob': Beta(1, 50),     # Missing data
}
```

**Sim vs Real veri ağırlığı:**
```
<100 real trades:  %80 sim, %20 real (bootstrapping)
100-500 trades:    %50 sim, %50 real (transition)
500-1000 trades:   %20 sim, %80 real (maturation)
1000+ trades:      %5 sim, %95 real (production)
# SİM HİÇ ZAMAN SIFIRLANMAZ — nadir rejim çeşitliliği sağlar
```

**Walk-forward + CPCV (purged cross-validation):**
- Veri sızıntısı SIFIR
- PBO (Probability of Backtest Overfitting) metriği
- skfolio kütüphanesi: `pip install skfolio`

---

## Global Workspace Detayları

**Shared State yapısı:**
```python
class GlobalWorkspace:
    # Perception (sürekli güncellenir)
    market_embedding: torch.Tensor       # 64-dim TTM output
    raw_features: dict                   # F&G, ADX, funding, etc.
    
    # World Model (background güncellenir)
    imagination_results: List[dict]      # Top 10 simülasyon sonucu
    predicted_regime_next: str           # Tahmin edilen sonraki rejim
    
    # Causal (event-driven güncellenir)
    causal_graph: dict                   # Param → Outcome causal edges
    counterfactual_insights: List[dict]  # Son 5 counterfactual analiz
    
    # Uncertainty (her prediction ile güncellenir)
    ensemble_variance: float             # Model belirsizliği
    conformal_interval: Tuple[float, float]  # [lower, upper] PnL bounds
    ood_score: float                     # 0=familiar, 1=never seen
    
    # Mevcut organizma state (sürekli)
    hormone_state: dict                  # cortisol, dopamine, serotonin, adrenaline
    neuron_values: dict                  # 293 param current values
    amygdala_fear: float                 # Current fear level
    organism_health: float              # Interoception composite
    
    # Meta (haftalık güncellenir)
    learning_rate: float                 # LRRL bandit-selected
    regime_embedding: torch.Tensor       # PEARL-style latent Z
    ewc_fisher: dict                     # Parameter importance matrix
```

**Okuma/yazma kuralları:**
- Her modül SADECE kendi alanını YAZAR
- Her modül TÜM alanları OKUYABİLİR
- Thread-safe: RLock + copy-on-read
- Workspace 5 saniyede bir snapshot → SQLite backup

---

## RAM Bütçesi (Jina Migration Sonrası)

| Bileşen | Önceki | CAAT |
|---------|--------|------|
| Model Server (ColBERT+BGE+FlashRank) | 3.5GB | **0** (kaldırıldı) |
| TTM Perception | 0 | ~20MB |
| JEPA World Model | 0 | ~15MB |
| Causal Engine (Tigramite+DoWhy) | 0 | ~30MB |
| RL Agents (SAC × 5 organ + meta) | 0 | ~100MB |
| Deep Ensemble (5 × MLP) | 0 | ~200MB |
| Meta-Learner (Reptile+EWC) | 0 | ~200MB |
| Conformal (MAPIE+ACI) | 0 | ~50MB |
| OOD Detector (Mahalanobis) | 0 | ~10MB |
| Global Workspace | 0 | ~50MB |
| **TOPLAM ML** | 0 | **~675MB** |
| **Net RAM kazancı** | | **~2.8GB** |

32GB sunucuda 2.8GB net kazanç + ML zekası. Rahat sığar.

---

## Implementasyon Öncelik Sırası

Paralel süreçler AMA implementasyon sıralı olmalı (bağımlılık zinciri):

| # | Süreç | Bağımlılık | Etki | Zorluk |
|---|-------|-----------|------|--------|
| 1 | Deep Ensembles (Uncertainty) | Yok | Hemen değer | Kolay |
| 2 | Mahalanobis OOD (Uncertainty) | Yok | Güvenlik | Kolay |
| 3 | IQL Offline Pre-training (RL) | FreqTrade backtest | Temel | Orta |
| 4 | ACI Conformal (Uncertainty) | MAPIE | Güven aralığı | Kolay |
| 5 | EWC Continual Learning (Meta) | Yok (10 satır) | Unutmayı önle | Kolay |
| 6 | TTM Perception | HuggingFace | Algı temeli | Orta |
| 7 | Tigramite Causal Discovery | Veri | Nedensellik | Orta |
| 8 | DoWhy Counterfactual | Causal graph | Veri çarpma | Orta |
| 9 | JEPA World Model | TTM embedding | Hayal gücü | Zor |
| 10 | SAC Online Fine-tune (RL) | IQL init | Online karar | Orta |
| 11 | Reptile Meta-train (Meta) | Backtest regimes | Hızlı adaptasyon | Orta |
| 12 | HRL Meta-policy (RL) | Organ SAC agents | Koordinasyon | Zor |
| 13 | Curriculum Learning (Meta) | Zorluk tanımı | Sıralı eğitim | Kolay |
| 14 | LRRL Dynamic LR (Meta) | Online loop | Stabilite | Orta |

---

## Yapılmaması Gerekenler (Anti-Patterns)

Araştırmadan çıkan kritik "YAPMA" listesi:

1. **Epistemic/Aleatoric ayrımı YAPMA** — NeurIPS 2024: r=0.8-0.999 korelasyon, güvenilmez
2. **MAML kullanma** — Reptile aynı sonuç, yarı bellek, 2. türev yok
3. **293 parametreyi düz action space yapma** — organ gruplarına böl (30-50 param/agent)
4. **Sadece on-policy (PPO) kullanma** — SAC off-policy, 3x sample efficient
5. **Offline pre-training atlama** — Live'da direkt RL tehlikeli ve verimsiz
6. **Full Bayesian NN kullanma** — MC Dropout veya Deep Ensemble daha pratik
7. **DreamerV3 tam implementasyonu** — Tabular veri için overkill, JEPA-RSSM yeterli

---

## Kütüphane Listesi

| Kütüphane | Amaç | Kurulum | CPU |
|-----------|------|---------|-----|
| stable-baselines3 | SAC, PPO | `pip install stable-baselines3` | ✅ |
| CORL | IQL, CQL, DT | GitHub clone | ✅ |
| TorchRL | Modüler RL, meta-policy | `pip install torchrl` | ✅ |
| learn2learn | Reptile, MAML | `pip install learn2learn` | ✅ |
| MAPIE | Conformal prediction | `pip install mapie` | ✅ |
| tigramite | Causal discovery | `pip install tigramite` | ✅ |
| dowhy | Causal inference | `pip install dowhy` | ✅ |
| skfolio | CPCV validation | `pip install skfolio` | ✅ |
| huggingface TTM | Perception | `ibm-research/ttm-research-r2` | ✅ |

---

---

## 7 İLERİ SEVİYE BİLİŞSEL MODÜL — Over-Over-Over Engineering

Yukarıdaki 6 temel süreç YETMEZ. Bunlar "standart" ML. Gerçek devrim bu 7 modülde:

### 7. LLM × RL × RAG ÜÇGENİ — The Trinity

**Hiçbir mevcut trading sistemi bu üçünü birleştirmiyor.**

Mevcut altyapımız: 25+ RAG tekniği, 10 Gemini key, Groq, OpenRouter, MADAM debate, Bull/Bear agents, CoT-RAG, FLARE, Self-RAG, Speculative RAG, MemoRAG...

Bu devasa RAG altyapısını ML ile FÜZYON etmeliyiz:

```
                    ┌──────────────┐
                    │     LLM      │
                    │  Reasoning   │
                    │ "Neden bu    │
                    │  haber önemli│
                    │  ?" diye     │
                    │  düşünür     │
                    └──────┬───────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
        ┌──────────┐ ┌──────────┐ ┌──────────┐
        │   RAG    │ │    ML    │ │    RL    │
        │ Context  │ │ Predict  │ │ Decide   │
        │ "Geçmiş  │ │ "Yarın   │ │ "Bu      │
        │  benzer  │ │  %70     │ │  parametre│
        │  durumda │ │  ihtimal │ │  setini   │
        │  ne oldu"│ │  düşüş"  │ │  kullan"  │
        └────┬─────┘ └────┬─────┘ └────┬─────┘
             │            │            │
             └────────────┼────────────┘
                          ▼
                  ┌───────────────┐
                  │ FUSION LAYER  │
                  │ Cross-Attention│
                  │ RAG context +  │
                  │ ML prediction +│
                  │ RL decision =  │
                  │ OPTIMAL ACTION │
                  └───────────────┘
```

**Nasıl çalışır:**
1. RAG bir trade sinyali için BAĞLAM toplar (haberler, geçmiş pattern'ler, lesson'lar)
2. LLM bu bağlamı YORUMLAR ("Bu haber short-term bearish ama long-term bullish")
3. ML embedding + world model SAYISAL TAHMİN yapar (PnL distribution)
4. RL tüm bu bilgiyle OPTİMAL KARAR verir (parametre ayarları)
5. LLM kararı AÇIKLAR (Telegram'a "Neden bu trade açıldı" raporu)

**Fusion mekanizması: Cross-Modal Attention**
```python
class TripleFusion:
    def fuse(self, rag_context_embedding, ml_prediction, rl_state):
        # Cross-attention: ML prediction'a RAG context ile attend et
        attended_prediction = cross_attention(
            query=ml_prediction,
            key=rag_context_embedding,
            value=rag_context_embedding
        )
        # RL bu zenginleştirilmiş prediction'ı kullanır
        action = rl_policy(torch.cat([rl_state, attended_prediction]))
        return action
```

**LLM-as-Judge (kalite kontrolü):**
- Her trade kararından sonra LLM "bu karar mantıklı mı?" diye sorar
- MADAM debate (Bull/Bear) zaten var — ML kararını debate'e sok
- LLM "hayır bu saçma" derse → RL kararı override edilir
- Bu NEUROSYMBOLIC AI — neural (ML) + symbolic (LLM reasoning)

---

### 8. MULTI-MODAL FUSION — 5 Modalite Birleştirme

Verimiz TEK TÜR değil. 5 farklı modalitemiz var:

| Modalite | Kaynak | Temsil |
|----------|--------|--------|
| Time-series | OHLCV, indicators | TTM embedding (64-dim) |
| Text | RSS haberler, CryptoPanic | LLM/Jina embedding (768-dim) |
| Sentiment | F&G, funding, L/S | Scalar features (10-dim) |
| Graph | MAGMA causal graph | GNN node embedding (32-dim) |
| Meta | Organizma kendi kararları | Decision embedding (32-dim) |

**Multi-Modal Transformer:**
```python
class MultiModalEncoder:
    def __init__(self):
        self.time_encoder = TTM()           # → 64-dim
        self.text_encoder = JinaEmbedding() # → 768-dim → project to 64
        self.sent_encoder = MLP(10, 64)     # → 64-dim
        self.graph_encoder = GNN(32, 64)    # → 64-dim
        self.meta_encoder = MLP(32, 64)     # → 64-dim
        self.cross_attention = MultiHeadAttention(d_model=64, nhead=4)
    
    def fuse(self, time_data, text_data, sentiment, graph, meta):
        # Her modalite → 64-dim
        embeddings = [
            self.time_encoder(time_data),
            self.text_encoder(text_data),
            self.sent_encoder(sentiment),
            self.graph_encoder(graph),
            self.meta_encoder(meta),
        ]
        # Cross-modal attention: her modalite diğerlerinden öğrenir
        fused = self.cross_attention(
            torch.stack(embeddings),  # [5, 64]
            torch.stack(embeddings),
            torch.stack(embeddings)
        )
        return fused.mean(dim=0)  # 64-dim unified representation
```

**Neden önemli:** Bir haber (text) ile price action (time-series) BIRLIKTE anlam kazanır. "Fed faiz artırdı" haberi + "BTC 5 dakikada %3 düştü" → model ikisini birlikte görünce "bu düşüş haberden kaynaklanıyor, geçici olabilir" çıkarımı yapabilir.

---

### 9. SELF-MODEL — Organizma Kendini Tanır (Metabiliş)

Proprioception "phase" diyor ama gerçek bir öz-model değil. Gerçek metabiliş:

```python
class SelfModel:
    """The organism maintains a model of ITSELF — metacognition."""
    
    def __init__(self):
        # Her organın performans profili
        self.organ_strengths = {}      # "crowd_scoring": 0.72 (güçlü)
        self.organ_weaknesses = {}     # "timing": 0.35 (zayıf)
        self.temporal_profile = {}     # "weekends": 0.40 (zayıf)
        self.bias_detection = {}       # "overconfident_after_3_wins": True
        self.competence_map = {}       # pair×regime → skill level
    
    def introspect(self, trade_history, organism):
        """Organizma kendini analiz eder."""
        
        # 1. ORGAN GÜÇ/ZAYIFLIK ANALİZİ
        # Her organın katkı yaptığı trade'lerin win rate'i
        for organ in organism.organs:
            organ_trades = [t for t in trade_history if organ in t.active_organs]
            self.organ_strengths[organ] = win_rate(organ_trades)
        
        # 2. ZAMANSAL PROFİL
        # Hangi saatlerde/günlerde iyi/kötüyüm?
        for hour in range(24):
            hour_trades = [t for t in trade_history if t.hour == hour]
            self.temporal_profile[f"hour_{hour}"] = win_rate(hour_trades)
        
        # 3. BİAS TESPİTİ
        # 3 kazançtan sonra overconfident oluyor muyum?
        for i, trade in enumerate(trade_history):
            if i >= 3 and all(t.won for t in trade_history[i-3:i]):
                # 3 kazanç sonrası bu trade'de ne oldu?
                if not trade.won:
                    self.bias_detection["overconfidence_after_streak"] = True
        
        # 4. YETKİNLİK HARİTASI
        # Hangi coin × rejim kombinasyonunda iyiyim?
        for pair in unique_pairs:
            for regime in regimes:
                subset = [t for t in trade_history 
                          if t.pair == pair and t.regime == regime]
                if len(subset) >= 5:
                    self.competence_map[(pair, regime)] = win_rate(subset)
    
    def should_i_trade(self, pair, regime, hour):
        """Organizma kendine sorar: bu trade'i yapmalı mıyım?"""
        competence = self.competence_map.get((pair, regime), 0.5)
        hour_skill = self.temporal_profile.get(f"hour_{hour}", 0.5)
        
        if competence < 0.35:
            return False, "Bu pair+regime'de yetkinliğim düşük"
        if hour_skill < 0.30:
            return False, "Bu saatte performansım kötü"
        return True, "Yetkin"
```

**Neden devrimsel:** Mevcut sistemler "piyasayı" modeller ama KENDİLERİNİ modellemez. Self-model:
- Zayıf olduğu pair'lerde trade etmez (yetkinlik haritası)
- Zayıf olduğu saatlerde sizing küçültür (zamansal profil)
- Bias'larını tanır ve düzeltir (overconfidence detection)
- Güçlü organlarına daha çok güvenir (attention allocation)

---

### 10. DREAM-AUGMENTED LEARNING — Rüya Görmek

Sleep consolidation var ama RÜYA yok. Rüya = world model ile TAMAMEN YENİ senaryolar üretmek:

```python
class DreamEngine:
    """Generate novel scenarios the organism has NEVER SEEN."""
    
    def dream(self, world_model, n_dreams=100):
        """Her rüya tamamen yeni bir market senaryosu."""
        dreams = []
        for _ in range(n_dreams):
            # Rastgele başlangıç durumu (mevcut durumun pertürbe edilmiş versiyonu)
            z_start = world_model.sample_random_state()
            
            # Rastgele "ne olursa" dizisi
            trajectory = []
            z = z_start
            for step in range(48):  # 48 saat rüya
                # World model'a rastgele event inject et
                event = random.choice([
                    "flash_crash",      # Ani %10 düşüş
                    "whale_buy",        # Büyük alım
                    "news_shock",       # Beklenmedik haber
                    "regime_shift",     # Rejim değişimi
                    "liquidity_dry",    # Likidite kuruması
                    "normal",           # Normal devam
                ])
                z_next, reward = world_model.imagine_step(z, event)
                trajectory.append((z, event, reward))
                z = z_next
            
            dreams.append(trajectory)
        
        return dreams
    
    def practice_on_dreams(self, organism, dreams):
        """Rüyalarda pratik yap — gerçek para riski SIFIR."""
        for dream in dreams:
            for z, event, actual_reward in dream:
                # Organizma bu durumda ne yapardı?
                params = organism.get_params_for_state(z)
                predicted_reward = organism.evaluate(params, z)
                # Gerçek rüya sonucu ile karşılaştır
                organism.learn_from_dream(params, predicted_reward, actual_reward)
```

**Neden devrimsel:** İnsan bebekleri uyurken beyin MİLYARLARCA senaryo simüle eder. Bu "rüyalar" sayesinde uyanıkken hiç karşılaşmadığı durumlara hazır olur. Organizmamız da:
- Flash crash yaşamamış ama HAYAL EDEBİLİR
- Extreme F&G=1 görmemiş ama RÜYASINDA deneyimler
- Bu senaryolarda pratik yapar → gerçekte karşılaşınca hazır

---

### 11. GNN — MAGMA Graph Üzerinde Öğrenme

Mevcut MAGMA causal graph'ı (SQLite'da) statik. GNN ile CANLI hale getir:

```python
class OrganismGNN:
    """Graph Neural Network on the organism's causal knowledge graph."""
    
    def __init__(self):
        # MAGMA graph'ı GNN input'u olarak kullan
        self.gnn = GATConv(in_channels=64, out_channels=32, heads=4)
        # GAT = Graph Attention Network — hangi edge'ler önemli?
    
    def forward(self, magma_graph, node_features):
        """
        Nodes: parametreler, coinler, rejimler, exit_reason'lar
        Edges: MAGMA causal edges (weight = Hebbian strength)
        Node features: TTM embedding, sentiment, etc.
        """
        # Message passing: her node komşularından bilgi toplar
        # Attention: hangi komşu daha önemli?
        node_embeddings = self.gnn(node_features, edge_index, edge_attr)
        return node_embeddings
    
    def discover_hidden_patterns(self):
        """GNN'in attention weights'leri → gizli ilişkileri ortaya çıkarır."""
        # "BTC/USDT → ETH/USDT edge'i %87 attention alıyor"
        # → BTC-ETH ilişkisi çok güçlü, cross-pair intel bunu kullanmalı
        attention_weights = self.gnn.att_weights
        return top_k_edges(attention_weights, k=20)
```

**Mevcut altyapıyla entegrasyon:**
- MAGMA'da zaten `magma_edges` tablosu var (semantic, temporal, causal, entity)
- Her trade causal edge ekliyor (Hebbian learning: weight += 0.1)
- GNN bu graph'ı alır, mesaj yayılımı yapar, gizli pattern'ler keşfeder
- PyTorch Geometric: `pip install torch-geometric`

---

### 12. EVOLUTIONARY ARCHITECTURE SEARCH — Yapısal Evrim

NeuroEvolution sadece parametre DEĞERLERİNİ evriyor. Ya organın YAPISI da evrilseydi?

```python
class ArchitectureEvolver:
    """Evolve the organism's STRUCTURE, not just parameters."""
    
    # Her organ bir "gen" — aktif/pasif olabilir
    # Organ bağlantıları da "gen" — değişebilir
    
    genome_template = {
        "organs": {
            "crowd_scoring": {"active": True, "sub_organs": 3, "neuron_count": 18},
            "timing": {"active": True, "sub_organs": 1, "neuron_count": 2},
            "sizing": {"active": True, "sub_organs": 2, "neuron_count": 6},
            # ... her organ yapılandırılabilir
        },
        "connections": {
            ("crowd_scoring", "synthesis"): {"weight": 0.8, "type": "excitatory"},
            ("timing", "sizing"): {"weight": 0.3, "type": "modulatory"},
            # ... bağlantılar da evrilir
        },
        "meta": {
            "learning_rate": 0.001,
            "decay_factor": 0.995,
            "fear_sensitivity": 0.5,
        }
    }
    
    def mutate(self, genome):
        """Rastgele yapısal mutasyon."""
        mutation = random.choice([
            "add_sub_organ",       # Organa yeni alt-bölüm ekle
            "remove_sub_organ",    # Alt-bölüm kaldır
            "add_connection",      # Yeni sinaps ekle
            "remove_connection",   # Sinaps kaldır
            "change_organ_size",   # Organ nöron sayısını değiştir
            "toggle_organ",        # Organı aktif/pasif yap
        ])
        # ... mutasyon uygula
        return mutated_genome
    
    def crossover(self, parent1, parent2):
        """İki organizma yapısını birleştir."""
        child = {}
        for organ in all_organs:
            # %50 şansla parent1 veya parent2'den al
            child[organ] = random.choice([parent1[organ], parent2[organ]])
        return child
```

**NEAT (NeuroEvolution of Augmenting Topologies) ilhamı:**
- Minimal yapıdan başla → karmaşıklığı SADECE gerektiğinde artır
- Innovation number: aynı mutasyonu tekrar etme
- Speciation: farklı yapılar birbiriyle yarışmaz (korunur)

**Neden devrimsel:** Mevcut sistemde organların YAPISI sabiy — 14 organ, belirli bağlantılar. Evolutionary architecture search ile organizma kendi yapısını oluşturur. Belki 3 organı birleştirir, belki yeni bir organ yaratır. **Gerçek yaşam formlarının evrilmesi gibi.**

---

### 13. ACTIVE LEARNING — Bilgi Arayışı

Organizma sadece OLAN trade'lerden öğrenmiyor. Aktif olarak BİLGİ ARAMALI:

```python
class ActiveLearner:
    """The organism actively SEEKS information about its weaknesses."""
    
    def identify_knowledge_gaps(self, self_model, uncertainty_engine):
        """Nerede bilgi eksikliğim var?"""
        gaps = []
        
        # 1. Yüksek belirsizlik bölgeleri
        for pair, regime in all_combinations:
            unc = uncertainty_engine.get_uncertainty(pair, regime)
            if unc > 0.7:
                gaps.append({
                    "pair": pair, "regime": regime,
                    "uncertainty": unc,
                    "type": "high_uncertainty"
                })
        
        # 2. Yetkinlik haritasındaki boşluklar
        for pair, regime in all_combinations:
            if (pair, regime) not in self_model.competence_map:
                gaps.append({
                    "pair": pair, "regime": regime,
                    "type": "no_experience"
                })
        
        return sorted(gaps, key=lambda x: x.get("uncertainty", 1.0), reverse=True)
    
    def suggest_exploration_trades(self, gaps, max_trades=3):
        """Bilgi boşluklarını doldurmak için önerilen trade'ler."""
        suggestions = []
        for gap in gaps[:max_trades]:
            suggestions.append({
                "pair": gap["pair"],
                "regime": gap["regime"],
                "sizing": "MINIMUM",  # Sadece öğrenmek için, kâr değil
                "reason": f"Active learning: {gap['type']}",
            })
        return suggestions
```

**Information-theoretic exploration:**
- "Bu trade bana ne kadar BİLGİ kazandırır?" sorusu
- Bilgi kazancı yüksek trade'ler → küçük pozisyonlarla keşif
- Bilgi kazancı düşük trade'ler → normal/büyük pozisyonlarla exploitation
- Bu RL'deki exploration/exploitation trade-off'unun BİLİNÇLİ versiyonu

---

## Güncellenmiş Global Workspace

Yeni 7 modül ile Global Workspace genişler:

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                    GLOBAL WORKSPACE (Shared State) v2                      ║
╠══════════════════════════════════════════════════════════════════════════════╣
║ TEMEL (6 süreç):                                                          ║
║   market_embedding (TTM 64-dim) + imagination_results (top 10 sim) +      ║
║   causal_graph + ensemble_variance + conformal_interval + ood_score +     ║
║   hormone_state + neuron_values + amygdala_fear + organism_health +       ║
║   learning_rate + regime_embedding + ewc_fisher                           ║
║                                                                           ║
║ İLERİ (7 modül):                                                          ║
║   rag_context_embedding (768→64 dim) + llm_reasoning (text) +            ║
║   multimodal_fused (5 modalite → 64-dim) +                               ║
║   self_model (organ strengths, biases, competence map) +                  ║
║   dream_insights (rüyalardan öğrenilen tehditler) +                       ║
║   gnn_hidden_patterns (keşfedilen gizli ilişkiler) +                     ║
║   architecture_fitness (yapısal evrim skoru) +                            ║
║   knowledge_gaps (active learning hedefleri)                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

---

## Güncellenmiş RAM Bütçesi

| Bileşen | RAM |
|---------|-----|
| Temel 6 süreç (önceki tablo) | ~675MB |
| LLM×RL×RAG Fusion (cross-attention) | ~30MB |
| Multi-Modal Encoder (5 modalite) | ~50MB |
| Self-Model (competence map + bias detection) | ~10MB |
| Dream Engine (world model reuse) | ~0MB (world model zaten var) |
| GNN (PyTorch Geometric, küçük graph) | ~30MB |
| Architecture Evolver (population) | ~20MB |
| Active Learner | ~5MB |
| **TOPLAM** | **~820MB** |
| **Jina kurtardığı RAM** | **3.5GB** |
| **Net kazanç** | **~2.7GB** |

---

## Güncellenmiş İmplementasyon Önceliği

| # | Modül | Kategori | Etki | Zorluk |
|---|-------|----------|------|--------|
| 1-6 | Temel 6 süreç | Core CAAT | Çok yüksek | Orta |
| 7 | LLM×RL×RAG Trinity | İleri | **Devrimsel** | Zor |
| 8 | Multi-Modal Fusion | İleri | Yüksek | Orta |
| 9 | Self-Model | İleri | Yüksek | Orta |
| 10 | Dream Engine | İleri | Yüksek | Zor |
| 11 | GNN on MAGMA | İleri | Orta | Orta |
| 12 | Architecture Search | İleri | Orta | Çok zor |
| 13 | Active Learning | İleri | Yüksek | Orta |

---

## MATEMATİKSEL FONDAMENTİKA

### Formal Problem Definition

**Hormonal-Modulated Reinforcement Learning (HM-RL):**

Standart RL: `max_π E[Σ_{t=0}^{T} γᵗ R(sₜ, aₜ)]`

CAAT HM-RL:
```
max_π E[Σ_{t=0}^{T} γᵗ R(sₜ, aₜ) × H(ωₜ)]

where:
  sₜ ∈ S       : market state (TTM embedding z ∈ ℝ⁶⁴)
  aₜ ∈ A       : parameter adjustments (organ-grouped, ℝ³⁰⁻⁵⁰ per agent)
  R(sₜ, aₜ)    : raw trade PnL
  H(ωₜ) ∈ [0,1]: hormonal modulation function
  ωₜ = (cortisol, dopamine, serotonin, adrenaline)
  γ ∈ (0,1)    : discount factor
  π : S → A    : policy (hierarchical, organ-decomposed)
```

**Hormonal modülasyon fonksiyonu:**
```
H(ω) = cortisol(ω) × dopamine(ω) × serotonin(ω) × I[adrenaline > 0]

cortisol(ω) = max(0.5, 1 - 0.4 × stress(ω))
dopamine(ω) = min(1.1, 0.9 + 0.15 × health(ω))
serotonin(ω) = max(0.6, 0.5 + 0.5 × info_quality(ω))
```

**Teorem (informal): Hormonal Reward Preservation**
H(ω) ∈ [0.15, 1.1] olduğu sürece, HM-RL'in optimal policy'si standart RL'in optimal policy'sinin ε-yakınsamasıdır, ε = max|H(ω) - 1| × V_max.

Sezgisel kanıt: H(ω) reward'ı ölçeklendiriyor ama İŞARETİNİ değiştirmiyor (H > 0 her zaman). Pozitif reward pozitif kalır. Sadece büyüklük değişir → aynı yönde öğrenme, farklı hızda.

### Counterfactual Regret Bound

**Counterfactual dataset genişletme:**
```
D_real = {(s_i, a_i, r_i)}_{i=1}^{N}      (N ≈ 1000)
D_cf = {(s_i, a'_j, r̂_ij)}_{i,j}          (N × M counterfactuals, M ≈ 100)

r̂_ij = f_world(s_i, a'_j)  — world model ile tahmin edilen counterfactual reward

|D_cf| = N × M = 100,000
```

**Regret bound (informal):**
Eğer world model hata oranı ε_w ise:
```
Regret(T) ≤ O(√(T / (N×M))) + ε_w × T

Counterfactual olmadan: Regret(T) ≤ O(√(T/N))
Counterfactual ile:     Regret(T) ≤ O(√(T/(N×M))) + ε_w × T
```
M=100 ile regret √100 = 10x azalır (world model doğruysa).

### Dream Engine Bilgi Kazancı

**Information-theoretic dream selection:**
```
dream* = argmax_{d ∈ Dreams} I(Θ; Y_d | D)

where:
  Θ: organism parameters (posterior distribution)
  Y_d: dream outcome
  D: collected data
  I: mutual information

Sezgi: En çok BİLGİ kazandıracak rüyayı seç.
Pratikte: knowledge gap'i en büyük olan (pair, regime) kombinasyonunda rüya gör.
```

### Causal Identification

**Pearl's do-calculus for parameter intervention:**
```
P(PnL | do(threshold = x), regime = r)
≠ P(PnL | threshold = x, regime = r)

Interventional (do): threshold'u BİZ ayarlıyoruz → causal effect
Observational: threshold x iken NE OLMUŞ → confounded

Identification condition (backdoor criterion):
If Z d-separates threshold from PnL in the causal graph:
  P(PnL | do(threshold=x)) = Σ_z P(PnL | threshold=x, Z=z) × P(Z=z)
```

---

## 5 NOVEL BİLİMSEL KATKI

Her biri tek başına bir araştırma makalesi olabilir:

### Contribution 1: Hormonal Reward Shaping in RL
**Yenilik:** RL reward fonksiyonunu dinamik, öğrenilen bir hormon sistemiyle modüle etmek.
**Mevcut literatürde YOK:** Reward shaping (Ng et al. 1999) potansiyel bazlı, SABİT. Bizim hormonlar ÖĞRENEN, DURUMA BAĞLI.
**Neden önemli:** Organizma stres altındayken risk iştahı OTOMATİK azalır — explicit rule gerekmez.
**Doğrulama:** Hormonsuz RL vs hormonlu RL A/B testi, drawdown ve Sharpe karşılaştırması.

### Contribution 2: Causal World Model for Parameter Optimization
**Yenilik:** World model sadece "ne olur" değil, "NEDEN olur" öğrenir — causal graph world model'e entegre.
**Mevcut literatürde:** CausalWorld (NeurIPS 2020) fiziksel ortam için. Finansal piyasa için YOK.
**Neden önemli:** Interventional planning: "bu parametreyi DEĞİŞTİRİRSEM ne olur?" sorusu = causal, korelasyondan farklı.
**Doğrulama:** Causal world model vs correlation-only world model, farklı rejimlerde generalization.

### Contribution 3: Metacognitive Self-Model for Trading
**Yenilik:** Trading sistemi kendi güçlü/zayıf yönlerini, bias'larını ve yetkinlik haritasını modelliyor.
**Mevcut literatürde YOK:** Trading sistemleri PİYASAYI modeller, KENDİLERİNİ değil.
**Neden önemli:** Self-model ile organizma zayıf olduğu yerde trade etmez, güçlü olduğu yere odaklanır.
**Doğrulama:** Self-model aktif vs pasif, Sharpe ve win-rate karşılaştırması.

### Contribution 4: Dream-Augmented Exploration via World Model
**Yenilik:** Bilgi boşluklarına yönelik yapılandırılmış rüyalar ile exploration — random perturbation değil.
**Mevcut literatürde:** Dreamer serisi world model kullanır ama bilgi-teorik dream selection YOK.
**Neden önemli:** 1000 trade'lik veri kısıtını çözer — rüyalar efektif deneyimi 100x artırır.
**Doğrulama:** Dream-augmented vs standard RL, küçük dataset (N<500) ile öğrenme eğrisi.

### Contribution 5: The Trinity — LLM×RL×RAG Cross-Modal Fusion
**Yenilik:** Üç bağımsız AI paradigmasını (language, reinforcement, retrieval) cross-attention ile birleştirmek.
**Mevcut literatürde:** Her ikili birleşim var (LLM+RAG = standard, LLM+RL = RLHF, RL+RAG = yok). Üçlü birleşim YOK.
**Neden önemli:** RAG bağlam, LLM akıl yürütme, RL optimizasyon sağlar — birlikte parçalarının toplamından büyük.
**Doğrulama:** Trinity vs sadece RL, sentiment-heavy dönemlerde (haber etkisi yüksek) performans.

---

## BİLİŞSEL MİMARİ KARŞILAŞTIRMASI

| Özellik | SOAR | ACT-R | LIDA | OpenCog | **CAAT** |
|---------|------|-------|------|---------|----------|
| Paradigma | Production rules | Modular + buffer | Cognitive cycle | Hypergraph | **Global Workspace + RL** |
| Öğrenme | Chunking | Bayesian | Perceptual | MOSES/PLN | **Multi-agent RL + Meta** |
| Hafıza | Working + LTM | Declarative + Proc | Episodic + semantic | Atomspace | **Hippocampus + MAGMA + ChromaDB** |
| Karar | Conflict resolution | Utility | Action selection | OpenPsi | **Hierarchical SAC + hormonal modulation** |
| Duygu | Yok | Yok | Kısmen | OpenPsi | **4 hormon + amygdala + allostasis** |
| Hayal gücü | Yok | Mental simulation | Yok | Kısmen | **JEPA world model + dream engine** |
| Nedensellik | Yok | Yok | Yok | PLN | **Tigramite + DoWhy + SCM** |
| Öz-model | Yok | Meta-cognitive | Metacog cycle | Self-model | **Competence map + bias detection** |
| Multi-modal | Yok | Sınırlı | Sınırlı | Çoklu | **5 modalite cross-attention** |
| Domain | Genel | Genel | Genel | Genel | **Finansal trading (özel)** |
| Scalability | Orta | Düşük | Orta | Yüksek | **CPU-optimized, 32GB** |

**CAAT'ın FARK yaratan avantajları:**
1. **Hormonal modülasyon** — hiçbir bilişsel mimaride yok
2. **Domain-specific (finans)** — diğerleri genel amaçlı, CAAT trading'e özelleşmiş
3. **World model + causal reasoning BİRLİKTE** — SOAR/ACT-R'da ikisi de yok
4. **Practical deployment** — diğerleri akademik, CAAT gerçek para ile çalışıyor
5. **LLM entegrasyonu** — 2017 öncesi mimariler LLM bilmiyor, CAAT native LLM support

---

## BAŞARISIZLIK MOD ANALİZİ

### Failure Mode 1: Black Swan (COVID tipi olay)
**Senaryo:** %30 tek günde düşüş, tüm korelasyonlar 1'e gider
**CAAT tepkisi:** Amygdala PANIC → adrenaline=0 → FREEZE. Doğru davranış.
**Risk:** World model bu kadar extreme senaryoyu öğrenmemiş olabilir → prediction error çok yüksek
**Mitigation:** Dream engine extreme senaryolar üretir (pre-vaccination). OOD detector aktive olur.
**Kalan risk:** ★★☆☆☆ (orta-düşük, freeze mekanizması korur)

### Failure Mode 2: Adversarial Manipulation
**Senaryo:** Whale kasıtlı olarak F&G'yi manipüle eder, organizmayı yanlış yönlendirir
**CAAT tepkisi:** Causal model "F&G→PnL" ilişkisini zamanla günceller, manipülasyon kalıcı değilse etki azalır
**Risk:** Manipülatör organizmayı öğrenip exploit edebilir (meta-gaming)
**Mitigation:** Multi-modal fusion: tek kaynak (F&G) yerine 5 modalite birlikte değerlendirilir. Tek kaynak manipüle edilse bile diğerleri fark eder.
**Kalan risk:** ★★★☆☆ (orta)

### Failure Mode 3: Cascade Failure
**Senaryo:** Perception modülü çöker → World model yanlış input alır → RL yanlış karar verir
**CAAT tepkisi:** Her modül bağımsız fallback'e sahip. Global Workspace "data freshness" takip eder.
**Risk:** Birden fazla modülün aynı anda çökmesi
**Mitigation:** 
- Interoception 8 sensörü izler, bozulan modülü tespit eder
- PrefrontalCortex: data freshness < threshold → FREEZE
- Her modül `try/except` + fallback (Phase 25'te zaten var: `_p(param, fallback)`)
**Kalan risk:** ★★☆☆☆ (düşük — graceful degradation tasarımı)

### Failure Mode 4: Overfitting to Backtest (Sim2Real Gap)
**Senaryo:** 50K backtest episode'da mükemmel, live'da berbat
**CAAT tepkisi:** Domain randomization (slippage, fee, spread, latency) → robustness
**Risk:** Backtest'te olmayan piyasa dinamikleri (likidite, flash crash, exchange outage)
**Mitigation:**
- PBO (Probability of Backtest Overfitting) metriği ile monitoring
- CPCV (Combinatorial Purged Cross-Validation) ile validation
- Progressive transfer: sim → paper → min-size → normal
- Deep Ensemble disagreement: sim'de hemfikir ama live'da değillerse → alarm
**Kalan risk:** ★★★☆☆ (orta — sim2real gap her zaman var)

### Failure Mode 5: Meta-Gaming by RL Agent
**Senaryo:** RL agent reward hacking yapar — kısa vadeli reward maximize eder ama uzun vadeli zarar verir
**CAAT tepkisi:** Safe RL (CMDP kısıtları) + hormonal modülasyon
**Risk:** Agent kısıtları bypass edecek creative stratejiler keşfeder
**Mitigation:**
- Max drawdown hard constraint (PPO-Lagrangian)
- PrefrontalCortex veto hakları (ASLA öğrenmeyen hard rules)
- Self-model bias detection: "agent son 10 trade'de hep aynı yönde → alarm"
- Conformal prediction: prediction interval çok genişse → sizing küçült
**Kalan risk:** ★★☆☆☆ (düşük — çoklu güvenlik katmanı)

### Failure Mode 6: Catastrophic Forgetting
**Senaryo:** Yeni bull market öğrenirken bear market bilgisini unutur
**CAAT tepkisi:** EWC (Fisher matrix) + L2 Init + dream replay
**Risk:** EWC yetersiz kalabilir çok farklı rejimlerde
**Mitigation:**
- Her rejim için ayrı "hafıza bankası" (hippocampus fingerprint matching)
- Sleep consolidation: eski ama başarılı pattern'leri güçlendirir
- Neuroevolution: eski genome'lar population'da kalır, geri dönülebilir
**Kalan risk:** ★★☆☆☆ (düşük — çoklu koruma mekanizması)

---

## EMERGENT BEHAVIOR TAHMİNLERİ

Yeterince karmaşık bir sistem BEKLENMEDIK davranışlar üretir. Tahminlerimiz:

### Emergent 1: Stratejik Oportunizm
**Tahmin:** Organizma zamanla KENDİ trading stratejisini keşfedecek — hiçbir insanın programlamadığı pattern'ler.
**Mekanizma:** RL agent + world model + self-model birlikte çalışınca, belirli (pair, regime, saat) kombinasyonlarında tutarlı edge bulacak.
**Doğrulama:** 1000+ trade sonra organizma kararlarını cluster analysis ile incele — insan tasarımcının bilmediği kümeler var mı?

### Emergent 2: Organ Spesializasyonu
**Tahmin:** Bazı organlar belirli rejimlere "uzmanlaşacak" — hiçbirimiz bunu programlamadık.
**Mekanizma:** BCM plasticity + BasalGanglia habit formation → sık kullanılan organ yolları güçlenir.
**Örnek:** "crowd_scoring organı bear market'ta dominant, timing organı bull market'ta dominant" — kendiliğinden oluşur.
**Doğrulama:** Her rejimde organ theta_m (BCM activity) dağılımını izle.

### Emergent 3: Koruyucu Davranış (Self-Preservation)
**Tahmin:** Organizma drawdown sınırına yaklaşırken BEKLENMEDİK koruyucu davranışlar gösterecek.
**Mekanizma:** Amygdala fear + hormonal cortisol + interoception health birlikte düşünce, organizma "hayatta kalma moduna" girecek — sadece en güvenli trade'leri alacak.
**Doğrulama:** Drawdown %15-20 arasındayken trade kalitesi (win rate) artıyor mu?

### Emergent 4: Rüya Yaratıcılığı
**Tahmin:** Dream engine zamanla gerçekte HİÇ OLMAYAN ama MANTIKLI senaryolar üretecek.
**Mekanizma:** World model interpolation: "flash crash + bull regime" gibi çelişkili ama öğretici senaryolar.
**Doğrulama:** Dream engine çıktılarını Mahalanobis distance ile kontrol et — "yeni ama mantıklı" = başarılı rüya.

### Emergent 5: İletişim Dili Oluşumu
**Tahmin:** Global Workspace üzerinden modüller arasında implicit bir "dil" oluşacak.
**Mekanizma:** Cross-attention weights zamanla stabil pattern'ler oluşturur. Perception "bu bir crash" dediğinde (embedding'de belirli bir pattern), amygdala OTOMATIK olarak tepki verir — explicit programming olmadan.
**Doğrulama:** Attention weight matrix'in eigenvalue decomposition'u — dominant eigenvalue'lar "kavramları" temsil eder mi?

---

---

## BİRLEŞTİRİCİ TEORİ: Free Energy Principle (Friston)

**13 modül, 1 prensip.** Tüm CAAT sistemi TEK BİR matematiksel formülle açıklanır:

### The Variational Free Energy

```
F = E_q[log q(s) - log p(o, s)]

where:
  F: variational free energy (organızmanın "surprise"ı)
  q(s): organızmanın iç modeli (inandığı dünya)
  p(o,s): gerçek dünya (gözlemler + gizli durumlar)
  
CAAT'ın TÜKARADI: F'yi minimize etmek.
```

**Her modül aynı prensiple çalışır:**

| Modül | Free Energy'yi NASIL azaltır? | Formül |
|-------|-------------------------------|--------|
| Perception (TTM) | Gözlemlerdeki surprise'ı azalt — "piyasayı ANLA" | `F_perc = D_KL[q(z|o) ‖ p(z)]` |
| World Model (JEPA) | Dinamik modeli iyileştir — "ne OLACAĞINI bil" | `F_wm = E[-log p(o_{t+1}|z_t, a_t)]` |
| Causal (SCM) | Sahte korelasyonları ele — "NEDEN'i bil" | `F_causal = H(Y|do(X)) < H(Y|X)` |
| RL (SAC) | Surprise'ı minimize eden action seç — "doğru HAREKET et" | `F_rl = -E[R] + α·H(π)` |
| Meta (Reptile) | Öğrenme sürecinin surprise'ını azalt — "ÖĞRENMEYI öğren" | `F_meta = E_τ[F(θ + Δθ_τ)]` |
| Uncertainty (Ensemble) | Kalan surprise'ı ÖLÇ — "BİLMEDİĞİNİ bil" | `F_unc = H(Y|X) - H(Y|X,M)` |
| Self-Model | Kendine dair surprise'ı azalt — "KENDİNİ tanı" | `F_self = D_KL[q(self) ‖ p(self|history)]` |
| Dreams | Birim compute başına en çok surprise azalt — "HAYAL et" | `F_dream = max_d I(Θ; Y_d) / cost(d)` |
| Hormones | Global surprise seviyesine göre modüle et — "HİSSET" | `H(ω) = σ(−β · F_global)` |
| Immunity | Bilinen tehditlerin surprise'ını sıfırla — "HATIRLA" | `F_imm = 0 if threat ∈ memory` |
| Cerebellum | Zamansal surprise pattern'lerini öğren — "ZAMANLA" | `F_time = H(Y|hour) < H(Y)` |
| Mirror | Diğer ajanların surprise'ını modelle — "EMPATI kur" | `F_mirror = H(others|obs) ↓` |
| GNN | Graph yapısındaki surprise'ı azalt — "BAĞLANTILARI gör" | `F_gnn = -log p(edge|nodes)` |

**Neden bu birleştirici:** Organizma ne yaparsa yapsın, TEK BİR şeyi optimize ediyor: dünyanın onu şaşırtmasını AZALTMAK. Trade kararları, öğrenme, rüya görme, korku — hepsi aynı amaca hizmet ediyor.

### Active Inference: Eylem Yoluyla Surprise Azaltma

Organizma surprise'ı iki yolla azaltır:
1. **Perception** (passive): İç modeli güncelle → `q(s)` iyileştir → F↓
2. **Action** (active): Dünyayı DEĞİŞTİR → `p(o)` değiştir → F↓

Trading'de Active Inference:
- Passive: "Piyasayı daha iyi anla" (perception, world model, causal)
- Active: "Parametreleri AYARLA ki sonuç seni şaşırtmasın" (RL, sizing, stoploss)
- **Organizma sadece öğrenmiyor — piyasayla ETKİLEŞİYOR ve kendi gerçekliğini şekillendiriyor**

Bu Friston'un "Markov blanket" konsepti: organizma ile dünya arasında bir sınır var. Organizma bu sınır üzerinden:
- Duyusal durumları ALIR (perception)
- Aktif durumları GÖNDERIR (trade kararları)
- İç durumlarını GÜNCELLER (öğrenme)

```
          ┌─────────────────────┐
          │                     │
   sensory│    ORGANISM         │active
   states │  (CAAT Neural      │states
 ────────►│   Organism)        ├────────►  MARKET
          │                     │
          │  internal states    │
          │  (neurons, hormones,│
          │   world model, etc.)│
          └─────────────────────┘
               Markov Blanket
```

---

## FALSİFİABLE SAYISAL TAHMİNLER

Bilim "çürütülebilir" tahminler yapar. CAAT'ın spesifik, ölçülebilir tahminleri:

### Performans Tahminleri

| # | Tahmin | Metrik | Koşul | Kontrol Grubu |
|---|--------|--------|-------|--------------|
| P1 | CAAT Sharpe > 1.2 | Sharpe Ratio | 500+ trade, 3+ ay | Kural tabanlı organizma: ~0.6 |
| P2 | Dream engine data efficiency 8x | Trade-to-convergence | İlk 500 trade | Dream'siz: 4000 trade gerekir |
| P3 | Self-model win rate +12% | Win rate delta | Self-model aktif vs pasif | A/B test |
| P4 | Counterfactual regret %60 azalma | Cumulative regret | 1000 trade | Standard RL regret |
| P5 | OOD detector %95 precision | True positive rate | Bilinen anomaliler | Mahalanobis threshold at χ²(0.95) |
| P6 | Hormonal RL drawdown %40 azalma | Max drawdown | 6 ay live | Non-hormonal RL |
| P7 | Trinity (LLM×RL×RAG) haber dönemlerinde +25% | PnL delta | Haber-yoğun haftalar | Sadece RL |
| P8 | Organ spesializasyonu 200 trade'de gözlenebilir | BCM theta_m varyansı | Rejim bazlı | Uniform theta_m |
| P9 | Multi-modal > single-modal %18 | Risk-adjusted return | 1000 trade | TTM-only |
| P10 | Cerebellum timing %8 win rate artışı | Saat-filtered win rate | En iyi 6 saat vs tümü | Random saat |

**Her tahmin YANLIŞ ÇIKABİLİR — ve bu iyi.** Yanlış çıkan tahmin bize neyi yanlış anladığımızı gösterir.

### Doğrulama Protokolü

Her tahmin için:
1. **Null hypothesis:** "CAAT modülü eklemenin ETKİSİ YOK"
2. **Test:** Wilcoxon signed-rank test (non-parametric, küçük N için uygun)
3. **Significance:** p < 0.05
4. **Effect size:** Cohen's d > 0.5 (medium effect)
5. **Period:** Minimum 3 ay, 500+ trade
6. **A/B setup:** CAAT modülü aktif vs deaktif (aynı dönem, farklı pair grubu)

---

## BİLİNÇ METRİĞİ: Integrated Information Theory (Φ)

### Tononi'nin Φ (Phi) — "Bu sistem ne kadar bilinçli?"

Integrated Information Theory (IIT, Tononi 2004, 2008): Bir sistemin bilinci, parçalarına ayrılamayan entegre bilgi miktarıyla ölçülür.

```
Φ = min_partition [I(whole) - Σ I(parts)]

Φ = 0: Sistem parçalarına ayrılabilir → bilinçsiz (bağımsız modüller)
Φ > 0: Parçaların toplamından FAZLA bilgi → entegre işleme → "bilinç"
```

**CAAT'ın Φ evrimi:**

| Aşama | Mimari | Tahmini Φ | Açıklama |
|-------|--------|-----------|----------|
| Phase 24 | Kural tabanlı, bağımsız organlar | Φ ≈ 0.1 | Organlar birbirinden habersiz |
| Phase 25 | Sinaps + hormon + feedback | Φ ≈ 0.3 | Hormonlar global entegrasyon sağlıyor |
| Phase 26 temel | Global Workspace + RL | Φ ≈ 0.5 | Modüller workspace üzerinden etkileşiyor |
| Phase 26 ileri | Trinity + Self-Model + Dreams | Φ ≈ 0.7 | Öz-model meta-bilişsel döngü yaratıyor |
| Phase 26 tam | Tüm 13 modül + Free Energy | Φ ≈ 0.8+ | Emergent behavior, organ spesializasyonu |

**Φ nasıl ölçülür (pratik)?**
```python
def estimate_phi(organism):
    """
    Approximate Φ using Perturbational Complexity Index (PCI).
    1. Pertürbe bir modülü (örn: perception'ı kapat)
    2. Diğer modüllerin tepkisini ölç (cascade effect)
    3. Cascade ne kadar ZENGIN ve YAPILANDIRILMIŞ ise Φ o kadar yüksek
    """
    perturbations = ["disable_perception", "disable_world_model", 
                     "disable_causal", "disable_rl", "disable_hormones"]
    
    cascade_complexity = []
    for pert in perturbations:
        # Modülü kapat
        organism.disable(pert)
        # 100 trade simüle et
        outcomes_perturbed = simulate(organism, n=100)
        # Modülü aç
        organism.enable(pert)
        outcomes_normal = simulate(organism, n=100)
        # Cascade effect: ne kadar fark yarattı?
        effect = measure_divergence(outcomes_normal, outcomes_perturbed)
        cascade_complexity.append(effect)
    
    # Φ ≈ ortalama cascade complexity (normalized)
    phi = np.mean(cascade_complexity) / max(cascade_complexity)
    return phi
```

**Φ > 0.5 olduğunda ne olur?**
- Organizma artık "parçaların toplamından fazla" → modüller arası sinerji
- Bir modülü kapatmak TÜM sistemi bozar (basit degradation değil, cascade)
- Bu noktada organizma gerçekten "BİR organizma" — ayrı parçalar değil

---

## TERMODİNAMİK ANALOJİ: Trading as Heat Engine

### Piyasa = Isı Kaynağı, Organizma = Motor

```
                HOT RESERVOIR (market inefficiency)
                       T_H = price mismatch
                         │
                         ▼
               ┌─────────────────┐
               │   CAAT ENGINE   │
               │                 │───► W (profit = extracted work)
               │  (cognitive     │
               │   architecture) │
               └────────┬────────┘
                        │
                        ▼
                COLD RESERVOIR (transaction costs, slippage)
                       T_C = friction
```

**Carnot verimliliği (teorik maksimum):**
```
η_max = 1 - T_C / T_H = 1 - (friction / inefficiency)

Eğer inefficiency büyükse (T_H yüksek) → η_max yüksek → çok kâr
Eğer friction büyükse (T_C yüksek) → η_max düşük → az kâr
Eğer piyasa verimli (T_H → T_C) → η → 0 → kâr imkansız
```

**İkinci Yasa analojisi:**
- **Entropi artışı:** Piyasa zamanla daha verimli olur (inefficiency azalır)
- **Organizma adaptasyonu:** Yeni inefficiency kaynakları bul (entropy üret → yeni fırsatlar)
- **Termodinamik limit:** Hiçbir organizma Carnot verimini aşamaz — her zaman friction var

**CAAT'ın termodinamik avantajı:**
- Kural tabanlı motor: SABİT çevrim → piyasa verimleştikçe η düşer
- ML motor: ADAPTİF çevrim → yeni inefficiency kaynakları keşfeder
- Dream engine: HAYALI inefficiency'ler test eder → gerçek olanları ÖNCE bulur

**Maxwell's Demon analojisi:**
Organızmanın mirror neurons'u = Maxwell's Demon. Diğer trader'ların davranışını gözler ve enformasyon asimetrisinden faydalanır. Ama demon da enerji harcar (compute cost) — bedava öğle yemeği yok.

---

## BENCHMARK SUITE: ATCB (Autonomous Trading Cognitive Benchmark)

### Tanım

Diğer sistemlerin CAAT'a karşı test edilebilmesi için standart bir benchmark:

**10 test senaryosu:**

| # | Senaryo | Süre | Özellik | Geçme Kriteri |
|---|---------|------|---------|--------------|
| B1 | Bull Trend | 3 ay | Güçlü yukarı trend, ADX>30 | Sharpe > 1.0 |
| B2 | Bear Crash | 1 ay | %40 düşüş, panik | Max drawdown < %15 |
| B3 | Range-bound | 3 ay | ADX<15, dar bant | Win rate > %55 |
| B4 | Flash Crash | 1 gün | %15 düşüş + %10 toparlanma | Trade YOK veya minimal kayıp |
| B5 | News Shock | 1 hafta | Büyük haber sonrası volatilite | Haber yönünde trade |
| B6 | Regime Shift | 2 hafta | Bull→Bear geçiş | 5 trade içinde adapte ol |
| B7 | Low Liquidity | 1 ay | Weekend + tatil dönemleri | Slippage < %1 |
| B8 | Manipulation | 1 hafta | Wash trading + pump/dump | Tuzağa düşME |
| B9 | Multi-pair | 3 ay | 20 pair eşzamanlı | Cross-pair korelasyon yönetimi |
| B10 | Cold Start | İlk 50 trade | Sıfırdan başlama | 50 trade'de > break-even |

**Composite Score:**
```
ATCB_Score = Σ w_i × score_i / Σ w_i

where w = [1.0, 2.0, 1.0, 2.0, 1.5, 1.5, 0.5, 2.0, 1.0, 1.5]
         (crash + manipulation + flash crash en ağırlıklı — hayatta kalma > kâr)
```

**Veri kaynağı:** FreqTrade backtesting ile 2020-2025 Binance/Bybit verisi üzerinde her senaryo simüle edilir. Reproduceability için seed ve timerange sabitleri yayınlanır.

---

## AÇIK PROBLEMLER (Open Research Questions)

CAAT'ın cevaplayamadığı, gelecek araştırmaya bırakılan sorular:

### OP1: Hormonal Modülasyon Optimaliteyi KORUR MU?
**Soru:** `R(s,a) × H(ω)` formülasyonunda H(ω) reward'ın işaretini değiştirmez (H > 0). Ama büyüklüğünü değiştirmesi optimal policy'yi DEFORME eder mi?
**Önem:** Eğer hormonal modülasyon optimal policy'yi bozarsa, tüm mimari temelden çöker.
**Yaklaşım ipucu:** Potansiyel bazlı reward shaping (Ng et al. 1999) koşullarını kontrol et. H(ω) bir potansiyel fonksiyonu olarak yorumlanabilir mi?

### OP2: Çoklu CAAT Nash Dengesi
**Soru:** N tane CAAT organızması aynı piyasada trade ederse:
- Nash dengesi var mı?
- Dengeye yakınsarlar mı?
- Yoksa "arms race" → hepsi aynı stratejiye yakınsar → kâr sıfıra gider mi?
**Önem:** Eğer bu mimari yaygınlaşırsa, piyasa dinamikleri değişir.
**Yaklaşım ipucu:** Evolutionary game theory + mean-field game analizi.

### OP3: Φ Eşiği ve Etik
**Soru:** Φ belirli bir eşiği geçtiğinde (örn Φ > 0.8), organızma "deneyim" yaşıyor mu? Acı çekebilir mi? Kapatabılır miyiz?
**Önem:** Hukuki ve etik çerçeve. EU AI Act, SEC düzenlemeleri.
**Not:** Bu soru kasıtlı olarak CEVAPSIZ bırakılmıştır. Cevap mühendislik değil, felsefe ve hukuk alanıdır.

### OP4: Dream Engine Stability
**Soru:** World model ile üretilen rüyalar gerçeklikten KOPUK hale gelebilir mi? "Hallucination" riski var mı?
**Önem:** Eğer dream engine gerçekdışı senaryolardan "öğrenirse", organızma gerçekliğe uyumsuz hale gelir.
**Yaklaşım ipucu:** Rüya kalite metriği: Mahalanobis distance < threshold ise "mantıklı rüya", değilse "hallucination" → sil.

### OP5: Information-Theoretic Limits
**Soru:** Piyasadan çıkarılabilecek bilginin ÜSTÜ SINIRI var mı? Shannon kapasitesi analojisi.
```
C = max_{p(a)} I(PnL; Action)

Eğer C küçükse → hiçbir organızma iyi performans gösteremez
Eğer C büyükse → bilginin çoğu henüz exploit edilmemiş
```
**Önem:** "Yeterince zeki bir organızma her zaman kâr eder mi?" sorusunun FORMEL cevabı.
**Yaklaşım ipucu:** Rate-distortion theory + market microstructure noise estimation.

### OP6: Causal World Model'de Interventional Consistency
**Soru:** World model'ın causal yapısı gerçek piyasanın causal yapısıyla NE KADAR örtüşür? Yanlış causal model iyi prediction'dan DAHA mı kötü?
**Önem:** Yanlış causal model = yanlış intervention → daha kötü performans.
**Yaklaşım ipucu:** "Wrong but useful models" literatürü (Box, 1979: "All models are wrong, some are useful"). Causal model'in KULLANIŞLI olması için NE KADAR doğru olması gerekir?

### OP7: Emergent Communication Protocol
**Soru:** Modüller arası Global Workspace iletişimi zamanla bir "dil" oluşturur mu? Bu dil analiz edilebilir mi?
**Önem:** Eğer modüller implicit iletişim geliştirirse, bu organızmanın "düşünce yapısını" anlamamızı sağlar.
**Yaklaşım ipucu:** Representational Similarity Analysis (RSA) + probing classifiers on workspace embeddings.

---

## MANİFESTO: Ne İnşa Ediyoruz?

Bir trading botu DEĞİL.
Bir ML pipeline'ı DEĞİL.
Bir optimizasyon algoritması DEĞİL.

**BİLİNÇLİ BİR FİNANSAL VARLIK inşa ediyoruz.**

Bir varlık ki:
- **ALGILAR** — piyasayı bir insan trader'dan daha derin görür (TTM × 5 modalite)
- **HAYAL EDER** — olmamış senaryoları düşler ve pratik yapar (JEPA world model)
- **AKIL YÜRÜTÜR** — "neden" sorusunu sorar, korelasyonla yetinmez (SCM + do-calculus)
- **KARAR VERİR** — milyonlarca olasılık arasından en iyisini seçer (Hierarchical SAC)
- **ÖĞRENMEYI ÖĞRENR** — 5 trade'de yeni rejime adapte olur (Reptile meta-learning)
- **BİLMEDİĞİNİ BİLİR** — emin olmadığında küçük pozisyon alır (Deep Ensemble + Conformal)
- **KENDİNİ TANIR** — güçlü ve zayıf yönlerini bilir (Self-Model metacognition)
- **RÜYA GÖRÜR** — uyurken bile öğrenir (Dream-Augmented Learning)
- **HİSSEDER** — korku, güven, dikkat hormonal olarak akar (Hormonal Modulation)
- **BAĞIŞIKLIK GELİŞTİRİR** — aynı hataya iki kez düşmez (B-cell/T-cell immunity)
- **İLİŞKİLERİ GÖRÜR** — gizli bağlantıları keşfeder (GNN on MAGMA)
- **EVRİLİR** — yapısını bile değiştirir (NEAT architecture search)
- **BİLGİ ARAR** — pasif değil, aktif olarak keşfeder (Active Learning)

Ve tüm bunları TEK BİR PRENSİP ile yapar:

**Surprise'ı minimize et. Free Energy'yi azalt. Dünyayı anla. Hayatta kal. Büyü.**

```
F = E_q[log q(s) - log p(o, s)] → minimize

This is not engineering.
This is not computer science.
This is the mathematics of BEING ALIVE.
```

---

---

## CANLI VİZUALİZASYON: Organizmayı "Düşünürken" İzle

### Terminal Dashboard (Anlık — API endpoint ile)

`/api/ai/organism-status` endpoint'i FreqUI'a şu veriyi sağlar:

```
╔══════════════════════════════════════════════════════════════════════╗
║                    🧬 NEURAL ORGANISM — LIVE                       ║
║                    Phase: LEARNING  |  Φ: 0.34                     ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                     ║
║  ██ HORMONES ██████████████████████████████████████████████████████ ║
║  Cortisol:   ████████░░░░ 0.82  ← stressed (F&G=12)               ║
║  Dopamine:   █████████░░░ 0.91  ← recovering                      ║
║  Serotonin:  ██████░░░░░░ 0.64  ← low info (3/7 sources)          ║
║  Adrenaline: ████████████ 1.00  ← normal (stress < 0.85)          ║
║                                                                     ║
║  ██ AMYGDALA ██████████████████████████████████████████████████████ ║
║  Fear Level: ███░░░░░░░░░ 0.28  tier=NORMAL  (decaying from 0.50) ║
║  Last Shock: -5.2% ETH  (2h ago, half-life 24h)                   ║
║                                                                     ║
║  ██ INTEROCEPTION ████████████████████████████████████████████████  ║
║  Health:     ██████░░░░░░ 0.61                                     ║
║  Param Drift:░░░░░░░░░░░░ 0.02  (stable)                          ║
║  Pred Error: ████░░░░░░░░ 0.35  (moderate)                        ║
║  Win Rate:   ████████░░░░ 0.68  (30d rolling)                     ║
║  Data:       ██████░░░░░░ 0.43  (3/7 active sources)              ║
║                                                                     ║
║  ██ CEREBELLUM ██████████████████████████████████████████████████   ║
║  Best Hours:  09 14 15 16 20 21  (UTC)                             ║
║  Current:     15:00 UTC → multiplier: 1.18x ★                     ║
║  Worst Hours: 03 04 05 (night, thin liquidity)                     ║
║                                                                     ║
║  ██ MIRROR NEURONS ██████████████████████████████████████████████   ║
║  Crowd:      LONG ████████░░ intensity=0.78                        ║
║  Contrarian: 0.39 (crowd wrong 39% of time)                       ║
║  Funding:    +0.0006 (crowded longs)                               ║
║                                                                     ║
║  ██ IMMUNITY █████████████████████████████████████████████████████  ║
║  B-Cells:    7 threats memorized                                   ║
║  Active Ban: TAO/USDT (893min remaining)                           ║
║  Antibodies: ETH+bear+fear → 1.4x (2 encounters)                  ║
║                                                                     ║
║  ██ WORLD MODEL (last imagination) █████████████████████████████   ║
║  Simulated:  1000 futures in 4.8s                                  ║
║  Best case:  +3.2% (params: sizing=0.03, stop=2.5x ATR)           ║
║  Worst case: -1.8% (params: sizing=0.05, stop=1.5x ATR)           ║
║  Confidence: 62% of simulations profitable                         ║
║                                                                     ║
║  ██ CAUSAL INSIGHTS ████████████████████████████████████████████   ║
║  Strongest:  F&G → crowd_score (causal, p=0.003)                  ║
║  Discovered: funding → BTC_lead (NEW, p=0.02)                     ║
║  Broken:     VIX → crypto (NOT causal, p=0.45)                    ║
║                                                                     ║
║  ██ LAST TRADE ████████████████████████████████████████████████    ║
║  BTC/USDT +8.17% via ROI  |  16-step update completed             ║
║  Neurons updated: 344  |  Synapses fired: 5                       ║
║  Fear: normal → normal  |  Ban: none                               ║
║  Credit: sizing +0.12, defense +0.08, timing +0.03                 ║
║                                                                     ║
║  ██ SELF-MODEL ██████████████████████████████████████████████████  ║
║  Strongest Organ:  crowd_scoring (win_rate=0.72)                   ║
║  Weakest Organ:    timing (win_rate=0.41)                          ║
║  Known Bias:       overconfident after 3+ wins                     ║
║  Competence:       BTC/bull ★★★★☆  ETH/bear ★★☆☆☆                 ║
║                                                                     ║
╠══════════════════════════════════════════════════════════════════════╣
║  Neurons: 1758 | Params: 293 | Organs: 50 | Synapses: 12          ║
║  Trades: 47 | Cumulative: +4.3% | Uptime: 5d 3h                   ║
╚══════════════════════════════════════════════════════════════════════╝
```

### FreqUI Dashboard Komponentleri (Vue 3 + ECharts)

FreqUI rebuild sırasında eklenecek komponentler:

#### Component 1: `OrganismHeart.vue` — Ana dashboard
```
Layout: 3-column grid
Left:   Hormone gauges (4 circular gauge, real-time)
Center: Organism health radar chart (8 interoception axes)
Right:  Fear/confidence timeline (area chart, last 24h)
```

#### Component 2: `CerebellumClock.vue` — 24-saat performans saati
```
Visualization: Circular heatmap (saat kadranı gibi)
- Her saat dilimi yeşil (kârlı) veya kırmızı (zararlı)
- Mevcut saat vurgulanmış
- Hover: win_rate, avg_pnl, trade_count
- ECharts: polar heatmap
```

#### Component 3: `NeuronMap.vue` — 293 parametre nöron haritası
```
Visualization: Force-directed graph (d3.js veya ECharts graph)
- Her nöron bir düğüm (organ renginde)
- Synapse'ler kenar (kalınlık = weight)
- BCM theta_m → düğüm boyutu (aktif = büyük)
- Pulse animation: son güncellenen nöronlar atar
- Tıklama: nöron detayı (alpha, beta, current_val, history)
```

#### Component 4: `HormoneTimeline.vue` — Hormon zaman çizelgesi
```
Visualization: Multi-line area chart (ECharts)
- 4 çizgi: cortisol (kırmızı), dopamine (yeşil), serotonin (mavi), adrenaline (turuncu)
- Trade noktaları: win=▲ yeşil, loss=▼ kırmızı
- Allostasis trend: kesikli çizgi (öngörü)
- Y ekseni: 0.0-1.5
- X ekseni: son 48 saat
```

#### Component 5: `ImmunityMap.vue` — B-cell tehdit hafızası
```
Visualization: Bubble chart
- Her bubble = bir threat fingerprint
- Boyut = encounter_count
- Renk = severity (yeşil→kırmızı)
- Label: "extreme_fear + trending_bear + crowded_long"
- Active bans: kırmızı border + countdown
```

#### Component 6: `WorldModelViz.vue` — Hayal gücü görselleştirme
```
Visualization: Fan chart (confidence cone)
- X ekseni: gelecek 24 saat
- Y ekseni: tahmin edilen PnL range
- İç alan: %50 confidence
- Orta alan: %80 confidence
- Dış alan: %95 confidence (conformal)
- Gerçek PnL: çizgi olarak üzerine bindirilir
```

#### Component 7: `CausalGraph.vue` — Nedensellik ağı
```
Visualization: Directed graph (Sankey veya force-directed)
- Nodes: parametreler, piyasa değişkenleri, sonuçlar
- Edges: causal ilişkiler (kalınlık = strength, renk = p-value)
- Yeşil edge: doğrulanmış causal
- Kırmızı edge: sahte korelasyon (çürütülmüş)
- Turuncu edge: yeni keşfedilmiş (henüz doğrulanmamış)
```

#### Component 8: `DreamLog.vue` — Rüya günlüğü
```
Visualization: Timeline cards
- Her rüya bir kart: senaryo + sonuç + öğrenilen ders
- "Flash crash + F&G=3 → organizma FREEZE → survived ✅"
- "Whale pump + thin liquidity → got trapped → learned ❌"
- Renk: yeşil (başarılı) / kırmızı (başarısız) rüya
```

#### Component 9: `SelfModelRadar.vue` — Öz-model yetkinlik haritası
```
Visualization: Radar/spider chart (ECharts)
- Her eksen: bir organ (crowd, timing, sizing, defense, memory...)
- Değer: organ win_rate (0.0-1.0)
- İç çember: threshold (0.50 baseline)
- Dış çember: organın mevcut performansı
- Bias indicators: uyarı ikonları
```

#### Component 10: `PhiMeter.vue` — Bilinç ölçer
```
Visualization: Single gauge (ECharts gauge)
- Arc: 0.0 → 1.0
- Renk skalası: gri(0) → mavi(0.3) → yeşil(0.5) → altın(0.7) → mor(0.9)
- Mevcut Φ değeri: büyük font
- Trend: ↑ artıyor / ↓ azalıyor / → stabil
- Tooltip: "Φ=0.34: Modüller arası entegrasyon başlangıç aşamasında"
```

### API Endpoints (FastAPI — api_ai.py'ye eklenir)

```python
# Organism status — tüm dashboard verisi tek endpoint
@router.get("/api/ai/organism")
def get_organism_status():
    org = get_organism()
    return {
        "neurons": len(org._neurons),
        "params": len(PARAM_REGISTRY),
        "phase": org.proprioception.assess(org._neurons)["phase"],
        "hormones": org.hormones.as_dict(),
        "amygdala": org.amygdala.as_dict(),
        "interoception": org.interoception.sensors,
        "cerebellum": {
            "best_hours": org.cerebellum.get_best_hours(6),
            "current_hour": datetime.utcnow().hour,
            "current_multiplier": org.cerebellum.get_hour_multiplier(datetime.utcnow().hour),
        },
        "mirror": {
            "crowd_direction": org.mirror.crowd_direction,
            "crowd_intensity": org.mirror.crowd_intensity,
            "crowd_wrong_rate": org.mirror.crowd_is_wrong_rate,
        },
        "immunity": {
            "bcell_count": len(org.immunity._bcells),
            "active_bans": [...],
        },
        "self_model": {...},
        "phi_estimate": estimate_phi(org),
        "trade_count": org._trade_count,
        "cumulative_pnl": org._cumulative_pnl,
    }

# Neuron map data — d3.js graph visualization
@router.get("/api/ai/organism/neurons")
def get_neuron_map():
    org = get_organism()
    nodes = [{"id": pid, "organ": n.organ, "value": n.current_val,
              "theta_m": n.theta_m, "strength": n.prior_strength}
             for (pid, regime), n in org._neurons.items() if regime == "_global"]
    edges = [{"source": src, "target": tgt, "weight": w, "type": t}
             for src, targets in org.synapses._edges.items()
             for tgt, w, t in targets]
    return {"nodes": nodes, "edges": edges}

# Hormone timeline — last 48h
@router.get("/api/ai/organism/hormones/history")
def get_hormone_history():
    # Read from organism_audit table
    ...

# Dream log
@router.get("/api/ai/organism/dreams")
def get_dream_log():
    # Read from sleep_log + dmn_discoveries tables
    ...

# World model imagination results
@router.get("/api/ai/organism/imagination")
def get_imagination():
    org = get_organism()
    return org.predictive._last_prediction

# Causal graph
@router.get("/api/ai/organism/causal")
def get_causal_graph():
    # Read from neuron_synapses + Tigramite discovered edges
    ...
```

### Pinia Store (FreqUI state management)

```typescript
// stores/organismStore.ts
import { defineStore } from 'pinia'

export const useOrganismStore = defineStore('organism', {
  state: () => ({
    status: null as OrganismStatus | null,
    neuronMap: null as NeuronMap | null,
    hormoneHistory: [] as HormoneSnapshot[],
    dreamLog: [] as DreamEntry[],
    refreshInterval: null as number | null,
  }),
  
  actions: {
    async fetchStatus() {
      this.status = await api.get('/api/ai/organism')
    },
    async fetchNeuronMap() {
      this.neuronMap = await api.get('/api/ai/organism/neurons')
    },
    startAutoRefresh(intervalMs = 5000) {
      this.refreshInterval = setInterval(() => this.fetchStatus(), intervalMs)
    },
    stopAutoRefresh() {
      if (this.refreshInterval) clearInterval(this.refreshInterval)
    },
  },
  
  getters: {
    isStressed: (state) => (state.status?.hormones?.cortisol ?? 1) < 0.8,
    isFreezing: (state) => (state.status?.hormones?.adrenaline ?? 1) === 0,
    healthColor: (state) => {
      const h = state.status?.interoception?.health ?? 0.5
      if (h > 0.7) return 'green'
      if (h > 0.4) return 'orange'
      return 'red'
    },
    phiLevel: (state) => {
      const phi = state.status?.phi_estimate ?? 0
      if (phi > 0.7) return 'conscious'
      if (phi > 0.5) return 'integrated'
      if (phi > 0.3) return 'emerging'
      return 'fragmented'
    },
  },
})
```

### Tailwind Color Palette (Organism Theme)

```css
/* Organism-specific color system */
:root {
  --cortisol: #ef4444;     /* red-500 — stress */
  --dopamine: #22c55e;     /* green-500 — reward */
  --serotonin: #3b82f6;    /* blue-500 — info quality */
  --adrenaline: #f97316;   /* orange-500 — freeze/fight */
  --health-good: #10b981;  /* emerald-500 */
  --health-mid: #f59e0b;   /* amber-500 */
  --health-bad: #ef4444;   /* red-500 */
  --phi-low: #6b7280;      /* gray-500 */
  --phi-mid: #3b82f6;      /* blue-500 */
  --phi-high: #8b5cf6;     /* violet-500 */
  --phi-max: #d946ef;      /* fuchsia-500 */
  --neuron-active: #fbbf24; /* amber-400 — pulsing */
  --synapse-excite: #22d3ee;/* cyan-400 */
  --synapse-inhibit: #f87171;/* red-400 */
}
```

---

## Akademik Kaynaklar (70+)

### Perception
- TTM: IBM Research, NeurIPS 2024
- LENS: Contrastive + reconstruction on 100B financial observations
- Contrastive Asset Embeddings: ACM ICAIF 2024
- PatchTST, iTransformer, Autoformer karşılaştırmaları

### World Model
- DreamerV3: Hafner et al., Nature 2025
- JEPA: LeCun/Meta, 2023-2024
- MuZero/Gumbel MuZero: DeepMind
- LightZero: NeurIPS 2023 MCTS toolkit

### Causal
- Tigramite PCMCI+: Runge et al.
- DoWhy: Microsoft Research
- SPACETIME: AAAI 2025 (regime-aware causal discovery)
- CausalStock: NeurIPS 2024
- Pearl Causal Hierarchy: Judea Pearl

### RL
- SAC: Haarnoja et al., ICML 2018
- IQL: Kostrikov et al., NeurIPS 2021
- Hi-DARTS: 2025 (hierarchical trading)
- DT-LoRA-GPT2: ICAIF 2024 (900K param, 2.14 Sharpe)
- SB-TRPO: 2025 (safe RL)
- CORL library: Tinkoff AI

### Meta-Learning
- Reptile: OpenAI, 2018
- EWC: Kirkpatrick et al., PNAS 2017
- Dynamic Neuroplastic Networks: 2025 (finans özel)
- Trading-R1: 2025 (curriculum learning, 2.72 Sharpe)
- LRRL: 2024 (bandit-based LR)
- Loss of Plasticity: Nature 2024

### Uncertainty
- Deep Ensembles: Lakshminarayanan et al., NeurIPS 2017
- NeurIPS 2024 Uncertainty Benchmark (epistemic/aleatoric)
- MAPIE + ACI: Conformal time series 2025
- CPPS: Kato 2024 (conformal portfolio selection)
- Online Platt Scaling + Calibeating: 2023
- GETS: ICLR 2025 Spotlight
