# Phase 26: CAAT — Cognitive Architecture for Autonomous Trading

---

## CHANGELOG & DURUM TAKIBI (Son guncelleme: 11 Nisan 2026)

### SPRINT 1: ALGI + BELIRSIZLIK + ALTYAPI — TAMAMLANDI ve CANLIDA

| Task | Dosya | Satir | Durum | Notlar |
|------|-------|-------|-------|--------|
| 1A | chart_features.py — Candle DNA | 41-77 | DONE (23 feature) | Plan: ~20 feature |
| 1A | chart_features.py — Multi-TF | 84-148 | DONE (~20 feature) | Plan: 30-50. Eksik: MACD, Ichimoku |
| 1B | chart_features.py — VPVR | 155-218 | DONE (8 feature) | POC/VAH/VAL tamam |
| 1B | chart_features.py — SMC | 225-305 | DONE (16 feature) | BOS/CHoCH/FVG/OB/Liq hepsi var |
| 1C | chart_features.py — Hurst | 312-387 | DONE (5 feature) | R/S analysis |
| 1C | chart_features.py — Path Sig | 394-441 | DONE (120 feature) | depth=4 (plandan fazla!) |
| 1D | test_chart_features.py | 252 satir | DONE (23 test) | Plandan fazla (150→252) |
| 2A | catboost_model.py | — | **YAPILMADI** | Standalone dosya yok. Sprint 2'ye borc |
| 2B | CatBoost feature pipeline | — | KISMI | triple_perception.py icinde inference stub var (L275-340) |
| 2C | CatBoost training script | — | **YAPILMADI** | Sprint 2'ye borc |
| 2D | CatBoost deploy | — | **YAPILMADI** | Model dosyasi yok (catboost_signal_v1.cbm) |
| 3A | ttm_perception.py | 298 satir | DONE | granite-timeseries-ttm-r2, 64-dim embedding |
| 3B | chronos_perception.py | 225 satir | DONE | chronos-bolt-small, P10/P50/P90 |
| 3C | triple_perception.py | 414 satir | DONE | TTM+Chronos+CatBoost fusion hub |
| 3D | Triple Perception entegrasyon | — | KISMI | AIFreqtradeSizer DONE (L811-839), rag_graph.py YAPILMADI |
| 4A | ood_detector.py | 262 satir | DONE | MarketOODDetector, Mahalanobis |
| 4B | conformal_calibrator.py | 252 satir | DONE | CQR+ACI, target_coverage=0.95 |
| 4C | deep_ensemble.py | 284 satir | DONE | 5x SmallMLP(64→32→1) |
| 4D | dual_axis_calibrator.py | 168 satir | DONE | CatBoost prob x CQR interval |
| 5A | pheromone_field.py | 294 satir | DONE | deposit/read/cleanup, thread-safe |
| 5B | Pheromone entegrasyon | — | KISMI | triple_perception+scheduler DONE, neural_organism+rag_graph YAPILMADI |
| 5C | predictive_interoception.py | 299 satir | DONE | Proaktif alertler calisiyor. SmallMLP YOK (basit prediktor) |
| 5D | Entegrasyon testi | — | **YAPILMADI** | Zero test coverage for Phase 26 modules (chart_features haric) |
| 5E | requirements-phase26.txt | — | **YAPILMADI** | Dosya yok |

**Sprint 1 Ozet:**
- 11/24 task DONE, 3/24 KISMI, 5/24 YAPILMADI
- Toplam yeni kod: ~3046 satir (plan: ~1800 — %69 fazla)
- Toplam test: 252 satir (sadece chart_features icin, geri kalan SIFIR)
- 193 chart feature CALISIYOR (plan: 130-230)
- TTM + Chronos CALISIYOR, CatBoost BEKLENIYOR (model yok)
- Pheromone + Interoception CALISIYOR (scheduler'da 15dk/30dk job)
- Hormonlar: cortisol=1.0, dopamine=1.05, serotonin=0.786, adrenaline=1.0

**Sprint 1 Borclari (Sprint 2'de veya Phase 28 sonrasi yapilacak):**
1. CatBoost training pipeline (2A-2D) — model dosyasi olusturulmali
2. rag_graph.py entegrasyonu — Triple Perception ve Pheromone sadece comment var
3. neural_organism.py pheromone entegrasyonu — sifir kod
4. Entegrasyon testleri — chart_features haric hicbir Phase 26 modulu test edilmedi
5. requirements-phase26.txt — dependency dosyasi olusturulmali
6. Interoception SmallMLP — basit prediktor kullanildi, plan MLP idi

### PHASE 28: DATABASE EVOLUTION — SPRINT 2'DEN ONCE YAPILACAK

Phase 28 araya giriyor (docs/PHASE28_DATABASE_EVOLUTION.md):
- ChromaDB → LanceDB (vector search)
- Grafeo (graph DB — Causal, MAGMA, Agent networks)
- DuckDB (analytics — backtest, OHLCV)
- Zvec (Alibaba, experimental 1 collection)
- SQLite connection pooling + db.py v2 merkezilestirme
- Tahmini sure: 7-12 gun
- **Sprint 2 bu bitmeden BASLAMAZ**

### SPRINT 2: KARAR + NEDENSELLIK + OTONOM YASAM — BEKLIYOR

Tum 20 task (6A-10E) BEKLIYOR durumda. Hicbiri baslamadi. Dosyalar olusturulmadi.

**Sprint 2'de EK olarak yapilacaklar (Sprint 1 borclari):**
- CatBoost training pipeline (2A-2D detaylari TANIMLANMALI)
- rag_graph.py Phase 26 entegrasyonu
- neural_organism.py pheromone entegrasyonu
- Phase 26 modul testleri

**Sprint 2'de KAPSAMDA OLMAYAN isler (Sprint 3+ veya gelecek phase):**
- Process 12: Evolutionary Architecture Search (priority #21)
- Process 14: Market Microstructure Intelligence (priority #16)
- Process 15: Cerebellum 24-slot timing (priority #22)
- Novel Contribution #8: Hormonal Market Making (Process 14 gerektirir)
- Model Risk Engine, Post-Trade Court, Ablation League Table (dokumanlanmis ama sprint'e atanmamis)

---

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
  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
  │1.ALGI    │ │2.HAYAL   │ │3.NEDEN   │ │4.KARAR   │ │5.META    │
  │Triple    │ │World     │ │Causal    │ │Multi-    │ │Reptile   │
  │Perception│ │Model     │ │Inference │ │Agent RL  │ │+EWC      │
  │(candle)  │ │(backgrnd)│ │(event)   │ │(sinyal)  │ │(hafta)   │
  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘
       │            │            │            │            │
       ▼            ▼            ▼            ▼            ▼
  ╔══════════════════════════════════════════════════════════════════╗
  ║            GLOBAL WORKSPACE (Shared State)                      ║
  ║  Market Embedding + World Predictions + Causal Graph +          ║
  ║  Uncertainty + Hormones + Neurons + OOD + Microstructure +      ║
  ║  Self-Model + Dream Insights + GNN Patterns + MM State          ║
  ╚══════════════════════════════════════════════════════════════════╝
       ▲            ▲            ▲            ▲            ▲
       │            │            │            │            │
  ┌────┴─────┐ ┌────┴─────┐ ┌────┴─────┐ ┌────┴─────┐ ┌────┴─────┐
  │6.UNCERT  │ │7.TRINITY │ │8.MULTI   │ │9.SELF    │ │10.DREAM  │
  │CatBoost  │ │LLM×RL    │ │MODAL     │ │MODEL     │ │ENGINE    │
  │×CQR      │ │×RAG      │ │5 modality│ │Metacog   │ │Rüya      │
  │(tahmin)  │ │(sinyal)  │ │(fusion)  │ │(günlük)  │ │(hafta)   │
  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘
       ▲            ▲            ▲            ▲            ▲
       │            │            │            │            │
  ┌────┴─────┐ ┌────┴─────┐ ┌────┴─────┐ ┌────┴─────┐ ┌────┴─────┐
  │11.GNN    │ │12.EVOLVE │ │13.ACTIVE │ │14.MICRO  │ │15.TIMING │
  │MAGMA     │ │Arch.     │ │LEARN     │ │STRUCTURE │ │Cerebellum│
  │Graph     │ │Search    │ │Explore   │ │+MM Mode  │ │24h clock │
  │(event)   │ │(aylık)   │ │(boşluk)  │ │(1-5dk)   │ │(her saat)│
  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘
```

**15 süreç, PARALEL çalışır, Global Workspace üzerinden haberleşir.**

---

## 15 Paralel Bilişsel Süreç — Detaylı Mimari

**Not:** Tüm süreçler PARALEL çalışır (pipeline DEĞİL). "6 temel + 7 ileri + 2 microstructure" ayrımı YOK — hepsi eşit vatandaş.

### 1. ALGI — Triple Perception (TTM + Chronos-Bolt + CatBoost)
**Biyolojik karşılık:** Retina + Görsel korteks + Somatosensory cortex
**Tetik:** Her candle (sürekli)
**Ne yapıyor:** Ham piyasa verisi → 3 farklı perspektiften analiz → fusion

**KRİTİK BULGU (Kasım 2025, arXiv 2511.18578):**
- Zero-shot TSFM'ler finansta İŞE YARAMAZ (R²=-2.80%, yönsel doğruluk < %50)
- CatBoost Sharpe 6.79 vs en iyi neural TSFM 3.66 (NeurIPS 2022: 3 yapısal neden)
- TimesFM 2.5 CPU'da VİABLE DEĞİL (arXiv 2602.10848: "GPU acceleration required")
- Finance-native pre-training ŞART — generic pre-train transfer etmiyor

**TRIPLE PERCEPTION MİMARİSİ:**

```
Market Data (OHLCV + indicators + macro + sentiment)
    │
    ├──→ TTM (1M param, <10ms, ~20MB) ──→ 64-dim embedding z
    │         Directional signal: up/down/flat + magnitude
    │         MLP-Mixer, multivariate, CPU-native, en iyi directional
    │
    ├──→ Chronos-Bolt-Small (48M param, ~100ms, ~191MB) ──→ Quantile P10-P90
    │         Uncertainty distribution: "PnL -%3 ile +%7 arası"
    │         Native quantile output, CPU-viable, TimesFM alternatifi
    │         HuggingFace: amazon/chronos-bolt-small
    │
    └──→ CatBoost (gradient boosting, <1ms inference, ~20MB model)
              Input: TTM embedding (64-dim) + raw indicators + EE sub-scores
                     + Chart Structure Features (~130-230 dim, aşağıda detay)
              CatBoost native embedding_features desteği (LDA + k-NN)
              Output: final prediction + calibrated probability + SHAP explanation
              NEDEN CatBoost: Sharpe 6.79 > tüm neural (düşük SNR, uninformative features, non-smooth targets)
              SHAP: "Bu trade'de order_block_proximity %28, hurst %19 etkili" → açıklanabilir
```

**Fusion: 3 perspektif birleşir**
```python
# TTM: directional signal (hızlı, hafif)
ttm_direction = ttm_model(ohlcv_data)  # "BULLISH", confidence 0.68

# Chronos-Bolt: uncertainty bounds (quantile)
quantiles = chronos_bolt(ohlcv_data)  # P10=-3%, P50=+1.2%, P90=+7%
interval_width = quantiles["P90"] - quantiles["P10"]  # dar = kesin

# CatBoost: final karar (TTM embedding + raw features + chart structure → prediction)
catboost_pred = catboost_model(
    embedding_features=[ttm_embedding_64dim],
    numeric_features=[rsi, atr, funding, fng, adx, ...],
    ee_subscores=[q1, q2, q3, q4, q5, q6],
    chart_structure=[                          # ← YENİ: 6 katmanlı chart intelligence
        *candle_dna_20dim,                     # son 5 mumun silüeti
        *multi_tf_features,                    # 1h + 4h + 1d göstergeler
        *vpvr_features,                        # POC, VAH, VAL mesafeleri
        *smc_features,                         # BOS, CHoCH, FVG, OB, liq sweep
        hurst_exponent, fractal_dimension,     # piyasa pürüzlülüğü
        *path_signature_truncated,             # geometrik path özellikleri
    ]
)  # prediction + calibrated_probability + shap_values

# Triple Fusion
final_confidence = catboost_pred.probability  # CatBoost iyi kalibre
sizing_multiplier = 1.0 / (1.0 + 5.0 * interval_width)  # Chronos uncertainty
signal_direction = ttm_direction  # TTM en iyi directional
```

**Neden triple > single:**
- TTM: Hızlı, directional sinyalde en iyi (MLP-Mixer, multivariate)
- Chronos-Bolt: Uncertainty ölçümünde en iyi (quantile, distribution-free)
- CatBoost: Final kararında en iyi (low SNR, tabular data king, SHAP)
- 3 farklı mimari = çeşitlilik = tek modelin kaçırdığını diğeri yakalar
- Uyuşmazlık = ek uncertainty sinyali (reranker agreement gibi)

**CHART STRUCTURE INTELLIGENCE — "Grafiğe Bakmak" Sayısal Olarak**

Bir insan trader grafiğe baktığında RSI/MACD görmez — **geometrik yapı, hacim kümelenmesi,
yapısal kırılmalar ve çok ölçekli hiyerarşi** görür. Bu bilgiyi VLM'e (vision model) gerek
kalmadan 6 katman halinde sayısallaştırıp CatBoost + TTM + RL + World Model'e veriyoruz.

**Katman 1: Candlestick DNA (Mum Dizisi Silüeti) — ~20-40 feature**
Tek mum anlamsız. Son 5-10 mumun SİLÜETİ bir "DNA dizisi" oluşturur:
```python
def candle_dna(ohlcv, lookback=5):
    """Her mumu 4 boyutlu vektöre encode et, son N mumun dizisini döndür."""
    features = []
    for candle in ohlcv[-lookback:]:
        body_ratio = (candle.close - candle.open) / max(candle.high - candle.low, 1e-8)
        upper_wick = (candle.high - max(candle.open, candle.close)) / max(candle.high - candle.low, 1e-8)
        lower_wick = (min(candle.open, candle.close) - candle.low) / max(candle.high - candle.low, 1e-8)
        vol_relative = candle.volume / rolling_avg_volume
        features.extend([body_ratio, upper_wick, lower_wick, vol_relative])
    return features  # 5 mum × 4 = 20 feature
```
CatBoost bu diziyi gördüğünde "doji → hammer → engulfing silsilesi = %68 dönüş" öğrenir.
Tek tek TA-Lib pattern'leri DEĞİL — **sekansın kendisi** bilgi taşır.

**Katman 2: Multi-Timeframe Structure — ~30-50 feature**
Aynı fiyat farklı ölçeklerde FARKLI görünür. Trader 3 ekrana bakar: günlük trend, 4h yapı, 1h giriş.
```
CatBoost feature'ları:
  1h: RSI, ADX, BB_width, EMA_slope, volume_ratio    (mevcut)
  4h: RSI_4h, ADX_4h, BB_width_4h, EMA_slope_4h      (YENİ)
  1d: RSI_1d, ADX_1d, trend_direction_1d, weekly_high/low distance (YENİ)
  
  Çapraz: tf_conflict = (1h bullish + 1d bearish → 1, uyumlu → 0)
          tf_alignment_score = 3 TF'nin yön uyumu (0.0 - 1.0)
```
"Günlükte düşüş ama saatlikte toparlanma" → CatBoost bunu **çelişki** olarak görür → uncertainty artar.
Mevcut informative_pairs (BTC+ETH 4h) bununla birleşir.

**Katman 3: Volume Profile / VPVR — ~8-12 feature**
Destek/direnç = fiyatın geçmişte KÜMELENDİĞİ seviyeler. Yatay çizgi değil, **hacim dağılımı**:
```python
def vpvr_features(ohlcv, lookback=200):
    """Volume Profile Visible Range — hacim bazlı destek/direnç."""
    price_bins = np.linspace(ohlcv.low.min(), ohlcv.high.max(), 50)
    volume_at_price = np.histogram(ohlcv.close, bins=price_bins, weights=ohlcv.volume)[0]
    
    poc = price_bins[np.argmax(volume_at_price)]  # Point of Control
    cumvol = np.cumsum(volume_at_price) / volume_at_price.sum()
    val = price_bins[np.searchsorted(cumvol, 0.15)]  # Value Area Low (%70 bandın altı)
    vah = price_bins[np.searchsorted(cumvol, 0.85)]  # Value Area High (%70 bandın üstü)
    
    current = ohlcv.close.iloc[-1]
    return {
        "poc_distance_pct": (current - poc) / poc,        # POC'a mesafe
        "above_vah": int(current > vah),                   # Değer alanı üstünde mi
        "below_val": int(current < val),                   # Değer alanı altında mı
        "value_area_width_pct": (vah - val) / poc,        # Değer alanı genişliği
        "poc_strength": volume_at_price.max() / volume_at_price.mean(),  # POC gücü
        "price_in_value_area": int(val <= current <= vah), # İçeride mi
        "nearest_high_volume_dist": ...,                   # En yakın yüksek hacim bölgesi
        "volume_profile_skew": ...,                        # Hacim dağılımı çarpıklığı
    }
```
"Fiyat POC'un altına düştü + POC güçlü → geri dönüş olasılığı yüksek" — CatBoost bunu öğrenir.

**Katman 4: Smart Money Concepts (SMC) — ~15-20 feature**
Kurumsal trader'ların ayak izlerini sayısallaştır:
```python
def smc_features(ohlcv, swing_lookback=5):
    """BOS, CHoCH, Order Blocks, FVG, Liquidity Sweeps."""
    swings = detect_swing_points(ohlcv, lookback=swing_lookback)  # zigzag
    
    # BOS (Break of Structure): önceki swing high/low kırıldı mı?
    last_swing_high = swings.last_high
    last_swing_low = swings.last_low
    bos_bullish = int(ohlcv.close.iloc[-1] > last_swing_high)  # Yükseliş yapısı kırıldı
    bos_bearish = int(ohlcv.close.iloc[-1] < last_swing_low)   # Düşüş yapısı kırıldı
    
    # CHoCH (Change of Character): HH/HL paterni bozuldu mu?
    was_uptrend = swings.higher_highs and swings.higher_lows
    now_broken = ohlcv.close.iloc[-1] < swings.last_higher_low
    choch = int(was_uptrend and now_broken)  # Trend dönüş sinyali
    
    # Order Block: güçlü impulsive hareketten önceki son mum
    ob_distance = distance_to_nearest_order_block(ohlcv, swings)
    ob_type = "bullish" if nearest_ob_is_bullish else "bearish"  # categorical
    
    # FVG (Fair Value Gap): ardışık mum fitilleri arasındaki boşluk
    fvg_count = count_unfilled_fvgs(ohlcv, lookback=50)
    nearest_fvg_distance = distance_to_nearest_fvg(ohlcv)
    
    # Liquidity Sweep: swing noktasının hemen üzerine çıkıp geri dönme (stop avı)
    liq_sweep_detected = detect_liquidity_sweep(ohlcv, swings, threshold_pct=0.002)
    
    return {
        "bos_bullish": bos_bullish,
        "bos_bearish": bos_bearish,
        "bos_count_24h": count_bos_last_n_candles(swings, 24),
        "choch_detected": choch,
        "ob_distance_pct": ob_distance,
        "ob_type": ob_type,                    # CatBoost categorical feature
        "fvg_unfilled_count": fvg_count,
        "fvg_nearest_distance_pct": nearest_fvg_distance,
        "liq_sweep_bullish": liq_sweep_detected["bullish"],
        "liq_sweep_bearish": liq_sweep_detected["bearish"],
        "market_structure": swings.structure,  # "uptrend" / "downtrend" / "range" (categorical)
        "swing_high_distance_pct": ...,
        "swing_low_distance_pct": ...,
        "impulse_strength": ...,               # Son impulsive hareketin gücü
    }
```
**Neden SMC önemli:** Geleneksel göstergeler (RSI, MACD) **lagging** — fiyat zaten hareket ettikten sonra sinyal verir. SMC yapısal kırılmaları ANINDA tespit eder. "Trend kırıldı" bilgisi RSI'dan 3-5 mum ÖNCE gelir.

**Katman 5: Piyasa Pürüzlülüğü (Fractal Features) — ~3-5 feature**
```python
def fractal_features(prices, windows=[50, 100, 200]):
    """Hurst exponent + fractal dimension → piyasa yapısı."""
    features = {}
    for w in windows:
        H = compute_hurst_exponent(prices[-w:])  # R/S analysis veya DFA
        features[f"hurst_{w}"] = H
        # H > 0.5 → trending (momentum çalışır)
        # H = 0.5 → random walk (hiçbir strateji çalışmaz)
        # H < 0.5 → mean-reverting (reversion çalışır)
    
    features["fractal_dim"] = 2.0 - features["hurst_100"]  # Hausdorff boyutu
    features["hurst_regime"] = "trending" if H > 0.55 else "reverting" if H < 0.45 else "random"
    return features
```
**Neden Hurst > ADX:** ADX sadece trend gücünü ölçer. Hurst piyasanın **istatistiksel doğasını** söyler — momentum mu yoksa mean-reversion mu oynayacağını BELİRLER. Organizma bu bilgiyle strateji SEÇİMİ yapar.

**Katman 6: Path Signature (Geometrik Yol Özellikleri) — ~50-100 feature**
**En derin katman — PhD seviyesi.** Rough Path Theory (Kidger & Lyons, NeurIPS 2019).

Path Signature bir zaman serisinin TÜM geometrik özelliklerini tek vektörde yakalar:
```python
import signatory  # veya esig

def path_signature_features(ohlcv, depth=4, window=20):
    """Path signature = zaman serisinin evrensel geometrik parmak izi."""
    # Path: (time, price, volume, rsi, ...) çok boyutlu yol
    path = torch.tensor([
        ohlcv.close[-window:].values,
        ohlcv.volume[-window:].values,
        ohlcv.rsi[-window:].values,
    ]).T.unsqueeze(0)  # [1, window, channels]
    
    # Signature: her derecede farklı bilgi
    sig = signatory.signature(path, depth=depth)
    # depth=1: yön (toplam değişim)
    # depth=2: eğrilik (fiyat-hacim lead-lag)
    # depth=3: salınım paterni (oscillation yapısı)
    # depth=4: karmaşık çapraz etkileşimler
    
    return sig.squeeze().numpy()  # ~50-100 dim (channel sayısına göre)
```

**Neden Path Signature devrimsel:**
- İnsan trader'ın "hissettiği" şey budur — grafiğin genel ŞEKLİ
- Zaman ölçeğinden BAĞIMSIZ (reparametrization invariant) → hızlı/yavaş hareket fark etmez
- Matematiksel olarak EVRENSEL: herhangi bir continuous fonksiyonu yakınsayabilir (Stone-Weierstrass)
- Fiyat-hacim arasındaki **lead-lag** ilişkisini depth=2'de otomatik yakalar
- CatBoost embedding_features olarak direkt kullanılabilir
- CPU'da <5ms (signatory kütüphanesi C++ backend)

**Akademik destek:**
- Kidger & Lyons "Deep Signature Transforms" NeurIPS 2019
- Morrill et al. "A Generalised Signature Method for Multivariate Time Series" 2020
- Liao et al. "Signature features for financial time series" 2024
- Perez Arribas et al. "Sig-Wasserstein GANs for TS generation" ICAIF 2020

**6 Katman Özet Tablosu:**

| Katman | Ne Yakalıyor | Feature | Hesaplama | Etki |
|--------|-------------|---------|-----------|------|
| 1. Candle DNA | Mum silüet dizisi | ~20-40 | <1ms | Orta |
| 2. Multi-TF | Çok ölçekli çelişki/uyum | ~30-50 | <5ms | Yüksek |
| 3. VPVR | Destek/direnç zonları | ~8-12 | <3ms | Yüksek |
| 4. SMC | Yapısal kırılmalar (BOS/CHoCH/FVG) | ~15-20 | <2ms | Çok yüksek |
| 5. Fractal | Piyasa pürüzlülüğü (Hurst) | ~3-5 | <2ms | Yüksek |
| 6. Signature | Geometrik path parmak izi | ~50-100 | <5ms | Çok yüksek |
| **TOPLAM** | | **~130-230** | **<18ms** | |

**Tüm katmanlar Tier-1 (<100ms) bütçesine rahat sığar. VLM API çağrısı YOK, tamamen local hesaplama.**

**Entegrasyon noktaları (6 modüle bağlanır):**
1. **CatBoost:** Tüm 130-230 feature direkt input → SHAP ile hangisi önemli otomatik çıkar
2. **TTM:** Chart features TTM'e ek input olarak verilebilir → embedding daha zengin
3. **RL:** Observation space'e chart structure eklenir → ajan "CHoCH sonrası ne yapmalıyım" öğrenir
4. **World Model:** JEPA zenginleştirilmiş embedding'den tahmin yapar → daha isabetli simülasyon
5. **Causal:** Tigramite "BOS → PnL causal mi?" sorusunu cevaplayabilir → sahte pattern'leri eler
6. **Organism:** Hurst rejim tespiti → hormonal yanıt (trending + Hurst>0.6 → dopamine↑ = agresif)

**CatBoost özel avantajları (Grinsztajn et al. NeurIPS 2022):**
1. Uninformative features'a dirençli (sadece bilgilendirici olanlarla split)
2. Non-smooth targets öğrenebilir (piecewise-constant, rejim kırılmaları)
3. Ordered boosting ile target leakage önleme
4. Symmetric/oblivious trees ile built-in regularization
5. Native embedding_features: TTM çıktısını direkt feature olarak alır

**KRİTİK: Training-Serving Skew Önlemi (Feature Noise Injection)**

CatBoost karar ağaçları belirli feature eşiklerine (thresholds) hassas overfit olur.
Backtest'te ADX=25.01 iken canlıda ADX=24.99 → tamamen farklı ağaç dalına gider.
Backtest motoru veriyi toplu (array) hesaplar, canlıda tick-by-tick → milimetrik farklar oluşur.

**Çözüm: Eğitimde feature noise injection (data augmentation):**
```python
def inject_feature_noise(X_train, noise_pct=0.01):
    """Her feature'a ±%1 Gaussian gürültü ekle → threshold'lara overfit engelle."""
    noise = np.random.normal(1.0, noise_pct, X_train.shape)
    return X_train * noise

# Eğitim sırasında 3-5 augmented kopya ile train et
X_augmented = pd.concat([X_train] + [inject_feature_noise(X_train) for _ in range(4)])
y_augmented = pd.concat([y_train] * 5)
model.fit(X_augmented, y_augmented)
```

**Ek önlem: Fuzzy threshold monitoring**
- Canlıda feature değerleri backtest dağılımından sapıyorsa → OOD detector alarm
- Feature distribution shift > 2σ → model güvenilirliği düşür (uncertainty artar)

**FreqAI entegrasyonu:**
- FreqAI CatBoost'u 2025.12'de kaldırdı ama BaseRegressionModel pipeline model-agnostic
- Custom CatBoost model ~50 satır: `fit()` override yeterli
- Walk-forward, data kitchen, feature pipeline, retraining hepsi hazır
- **Feature noise injection walk-forward pipeline'a eklenir → her retraining'de otomatik**

**CPU performansı:**
- TTM: <10ms inference, ~20MB RAM
- Chronos-Bolt: ~100ms inference, ~191MB RAM (mmap load)
- CatBoost: <1ms inference, ~20MB model, 5 saniyede 50K sample train
- **TOPLAM: ~230MB RAM, <120ms latency** (1h timeframe'de rahat)

**Self-supervised pretraining (TTM):**
- Contrastive learning: aynı rejimden benzer, farklı rejimden farklı
- Masked reconstruction: %15 feature sil, tahmin et
- LENS framework (2024): 100B financial observation üzerinde
- Contrastive Asset Embeddings (ACM ICAIF 2024)

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
  Encoder: TTM embedding z (64-dim) + chart structure features (~130-230 dim)
           → enriched state representation from Perception
  Recurrent: GRU(128) — temporal dynamics
  Stochastic: Gaussian(32-dim) — uncertainty in dynamics
  Predictor: MLP(128→64→64) — predict z_next + reward
  
  Total: ~300K parameters
  Speed: 5000 forward passes/sec on CPU
  Imagination: 1000 rollouts × 3-5 steps = ~1 second
```

**KRİTİK: Short-Horizon Rollout (Compounding Error Önlemi)**
Finansal zaman serilerinde her adımda tahmin hatası birikir (compounding error).
24 adım → hata katlanarak büyür → çöp trajectory. Deterministik ortamlarda (satranç, Go)
uzun rollout çalışır çünkü transition deterministik. Kripto'da stochastic + dışsal şoklar.
**Çözüm:** Rollout ufku 3-5 adım ile sınırlı. Kısa ama güvenilir > uzun ama çöp.

**Nasıl çalışır:**
1. Perception'dan `z_current` al (Global Workspace'ten oku)
2. 1000 farklı parametre konfigürasyonu dene (Latin Hypercube Sampling)
3. Her biri için **3-5 adım** geleceği simüle et (dünya modeliyle, short-horizon)
4. Her simülasyonun beklenen PnL'ini hesapla
5. En iyi 10 konfigürasyonu Global Workspace'e yaz
6. RL karar verirken bu simülasyonları kullanır

**MuZero/MCTS entegrasyonu (opsiyonel):**
- LightZero (NeurIPS 2023) toolkit ile MCTS tree search
- Parametre optimizasyonu bir "oyun" gibi — her "hamle" bir parametre ayarı
- MCTS en iyi hamle dizisini bulur
- MCTS depth da 3-5 ile sınırlı (aynı compounding error nedeniyle)

**RAM:** ~15MB
**Latency:** ~1 saniye (1000 × 5-step rollout)

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

**Calibration — Hybrid CatBoost + CQR (Novel Contribution #6):**

**KRİTİK BULGU (2025):** Platt scaling CatBoost'un kalibrasyonunu BOZUYOR!
Mevcut Platt calibrator'ımız (Brier=0.26, kalıcı devre dışı) bu yüzden sorunlu.

**Yeni Hybrid Calibrator: CatBoost Probability × CQR Interval**

İkisi FARKLI şeyleri ölçüyor — birleşince DAHA GÜÇLÜ:
```
CatBoost native probability: "Bu trade %72 kazanır" (nokta tahmini, iyi kalibre)
CQR interval: "PnL -%2 ile +%5 arası" (matematiksel %95 coverage garantisi)

Birleşim (NOVEL — kimse yapmadı):
  sizing = catboost_confidence × (1 / cqr_interval_width)
  
  Yüksek olasılık + dar aralık = BÜYÜK POZİSYON (kesin kazanç)
  Yüksek olasılık + geniş aralık = KÜÇÜK POZİSYON (belirsiz kazanç)
  Düşük olasılık + dar aralık = PAS GEÇ (kesin kayıp)
```

**CQR (Conformalized Quantile Regression):**
- Standard conformal'dan iyi (adaptive interval width)
- Raw quantile'dan iyi (%83.2 → %95 coverage düzeltme)
- Chronos-Bolt quantile head + CQR sarması = tek modelden calibrated uncertainty
- MAPIE kütüphanesi: `pip install mapie`
- Dağılım-bağımsız GARANTİ: rejim değişse bile coverage korunur

**Platt scaling KALDIRILIR.** CatBoost + CQR daha güçlü:
- CatBoost: zaten iyi kalibre (ordered boosting + symmetric trees)
- CQR: matematiksel garanti (Platt'ta yok)
- Birlikte: 2 boyutlu güven sistemi (probability × interval)

**RAM:** ~200MB (5-model ensemble) → ~50MB (CatBoost + CQR, daha hafif)

**Novel Contribution #6: Dual-Axis Calibration**
- Kimse CatBoost native probability + CQR interval'i BİRLEŞTİRMEDİ
- Bu iki-boyutlu güven sistemi hem "kazanır mıyım?" hem "ne kadar?" sorusunu aynı anda yanıtlar
- Position sizing ikisi birlikte modüle eder — tek boyutlu Platt'tan ÇOK daha bilgilendirici

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

## Global Workspace Detayları (v1 — bakınız v3 aşağıda birleşik versiyon)

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

## RAM Bütçesi → Birleşik tablo aşağıda (Güncellenmiş bölüme bakınız)

---

## Implementasyon Öncelik Sırası → Birleşik tablo aşağıda (Güncellenmiş bölüme bakınız)

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

### (Süreçler 7-13 devam ediyor — tümü eşit vatandaş, "ileri" değil)

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

**KRİTİK: Timestamp Alignment Guard (Time Dilation Önlemi)**

LOB/Microstructure verisi milisaniyeler içinde değişir (Tier-1, <100ms).
LLM/RAG çıkarımı 3-10 saniye sürer (Tier-3, <60s).
Cross-attention'a giren veriler FARKLI ZAMANLARA ait olabilir.

**Problem:** LLM "Bullish" derken LOB çoktan çökmüş olabilir. Stale veri + fresh yorum = çöp fusion.

**Çözüm: Her workspace alanına timestamp + staleness guard:**
```python
class TimestampedField:
    value: Any
    updated_at: float  # time.monotonic()
    max_age_ms: float  # tier'a göre: Tier-1=200ms, Tier-2=10s, Tier-3=60s

def fuse_with_alignment(fields: list[TimestampedField]) -> bool:
    """Tüm input'lar yeterince taze mi?"""
    now = time.monotonic()
    for f in fields:
        age_ms = (now - f.updated_at) * 1000
        if age_ms > f.max_age_ms:
            logger.warning(f"[TimeAlign] {f.name} stale: {age_ms:.0f}ms > {f.max_age_ms}ms")
            return False  # Fusion yapma, stale veri var
    return True
```

**Kurallar:**
- Tier-1 (LOB, RL) verisi 200ms'den eski → fusion'a SOKMA
- Tier-3 (LLM, RAG) verisi 60s'den eski → yeniden sorgula veya cache kullan
- Cross-tier fusion'da EN ESKİ verinin yaşı raporlanır → organism uncertainty artar
- Stale veri + taze veri birleştirilMEZ — ya hepsi taze, ya fusion atlanır

**LLM-as-Judge (kalite kontrolü):**
- Her trade kararından sonra LLM "bu karar mantıklı mı?" diye sorar
- MADAM debate (Bull/Bear) zaten var — ML kararını debate'e sok
- LLM "hayır bu saçma" derse → RL kararı override edilir
- Bu NEUROSYMBOLIC AI — neural (ML) + symbolic (LLM reasoning)
- **LLM judge asenkron çalışır — canlı hattı BLOKLAMAZ (Tier-3)**

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
            for step in range(5):  # 5 adım rüya (short-horizon, compounding error önlemi)
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

**KRİTİK: Dream Anomaly Filter (Model Exploit Önlemi)**

Model-based RL'in bilinen en büyük riski: RL ajanı world model'ın HATASINI exploit etmeyi öğrenir.
JEPA'da ufak bir istatistiksel sapma varsa, ajan piyasayı değil JEPA'nın halüsinasyonunu sömürür.
Rüyalarda %1000 kâr, canlıda anında batış.

**Çözüm: Rüya verisi RL'e girmeden önce 3 katmanlı filtre:**
```python
class DreamFilter:
    def is_valid_dream(self, trajectory, real_data_stats) -> bool:
        for z, event, reward in trajectory:
            # 1. Mahalanobis: rüya gerçek veriden çok mu uzak?
            maha_dist = mahalanobis(z, real_data_stats.mean, real_data_stats.precision)
            if maha_dist > chi2.ppf(0.99, df=z.shape[0]):
                return False  # Hallucination — gerçeklikten kopuk rüya

            # 2. Reward magnitude: gerçekçi mi?
            if abs(reward) > real_data_stats.max_abs_reward * 3.0:
                return False  # Unrealistic reward — model hatası

            # 3. Transition smoothness: ani sıçrama var mı?
            # (embedding space'te ardışık z'ler arası mesafe kontrolü)

        return True  # Geçerli rüya — RL'e gönderilebilir
```

**Kurallar:**
- Her rüya trajectory'si bu filtreden geçer → geçemezse SİLİNİR
- Filtrelenen rüya oranı > %50 → world model UNHEALTHY → ModelRiskEngine devreye girer
- RL ajanı ASLA filtresiz rüya verisi görmez

**Neden devrimsel:** İnsan bebekleri uyurken beyin MİLYARLARCA senaryo simüle eder. Bu "rüyalar" sayesinde uyanıkken hiç karşılaşmadığı durumlara hazır olur. Organizmamız da:
- Flash crash yaşamamış ama HAYAL EDEBİLİR
- Extreme F&G=1 görmemiş ama RÜYASINDA deneyimler
- Bu senaryolarda pratik yapar → gerçekte karşılaşınca hazır
- **Her rüya anomaly filter'dan geçer → hallucination'ları öğrenMEZ**

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

## Global Workspace v3 — BİRLEŞİK (Tüm 15 sürecin alanları):

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                    GLOBAL WORKSPACE (Shared State) v2                      ║
╠══════════════════════════════════════════════════════════════════════════════╣
║ TEMEL (6 süreç):                                                          ║
║   market_embedding (TTM 64-dim) + chart_structure (~130-230 dim) +        ║
║   imagination_results (top 10 sim) +                                      ║
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

## BİRLEŞİK RAM BÜTÇESİ (Tüm 15 Süreç)

| # | Bileşen | RAM |
|---|---------|-----|
| - | Model Server (Jina migration ile kaldırılıyor) | -3.5GB |
| 1 | Triple Perception (TTM 20MB + Chronos-Bolt 191MB + CatBoost 20MB) | ~230MB |
| 2 | JEPA World Model (GRU128 + Gaussian32) | ~15MB |
| 3 | Causal Engine (Tigramite + DoWhy) | ~30MB |
| 4 | RL Agents (SAC × 5 organ + HRL meta) | ~100MB |
| 5 | Meta-Learner (Reptile + EWC + LRRL) | ~200MB |
| 6 | Dual-Axis Calibration (CatBoost native + CQR/MAPIE + OOD Mahalanobis) | ~60MB |
| 7 | LLM×RL×RAG Fusion (cross-attention) | ~30MB |
| 8 | Multi-Modal Encoder (5 modalite × 64-dim) | ~50MB |
| 9 | Self-Model (competence map + bias detection) | ~10MB |
| 10 | Dream Engine (world model reuse) | ~0MB |
| 11 | GNN (PyTorch Geometric, küçük graph) | ~30MB |
| 12 | Architecture Evolver (population × 5) | ~20MB |
| 13 | Active Learner | ~5MB |
| 14 | Microstructure + MM (LOB encoder + VPIN + GLFT) | ~40MB |
| 15 | Cerebellum (24-slot) | ~1MB |
| - | Global Workspace (shared state) | ~50MB |
| | **TOPLAM ML (15 süreç)** | **~871MB** |
| | **Jina kurtardığı RAM** | **3.5GB** |
| | **Net RAM kazancı** | **~2.6GB** |

32GB sunucuda 2.6GB net kazanç + 15 paralel bilişsel süreç. Rahat sığar.

---

## BİRLEŞİK İMPLEMENTASYON ÖNCELİĞİ (Tüm 15 Süreç)

| # | Süreç | Bağımlılık | Etki | Zorluk | Latency Tier |
|---|-------|-----------|------|--------|-------------|
| 1 | CatBoost (Perception #1) | Yok | Çok yüksek | Kolay | Tier-1 (<100ms) |
| 2 | Mahalanobis OOD (#6) | Yok | Güvenlik | Kolay | Tier-1 |
| 3 | Chronos-Bolt (#1 quantile) | HuggingFace | Yüksek | Orta | Tier-2 (<5s) |
| 4 | CQR Calibration (#6) | MAPIE | Yüksek | Kolay | Tier-1 |
| 5 | TTM Perception (#1) | HuggingFace | Yüksek | Orta | Tier-2 |
| 6 | EWC Continual (#5) | Yok (10 satır) | Orta | Kolay | Tier-4 |
| 7 | IQL Offline Pre-train (#4) | FreqTrade backtest | Temel | Orta | Tier-4 |
| 8 | Tigramite Causal (#3) | Trade verisi | Yüksek | Orta | Tier-3 |
| 9 | DoWhy Counterfactual (#3) | Causal graph | Yüksek | Orta | Tier-3 |
| 10 | Self-Model (#9) | Trade history | Yüksek | Orta | Tier-3 |
| 11 | JEPA World Model (#2) | TTM embedding | Yüksek | Zor | Tier-3 |
| 12 | SAC Online Fine-tune (#4) | IQL init | Orta | Orta | Tier-1 |
| 13 | LLM×RL×RAG Trinity (#7) | RAG + RL | **Devrimsel** | Zor | Tier-3 |
| 14 | Multi-Modal Fusion (#8) | Jina + TTM | Yüksek | Orta | Tier-2 |
| 15 | Dream Engine (#10) | World model | Yüksek | Zor | Tier-4 |
| 16 | Market Making (#14) | Bybit WS | Yüksek | Zor | Tier-1 (1-5dk) |
| 17 | GNN on MAGMA (#11) | PyG | Orta | Orta | Tier-3 |
| 18 | Active Learning (#13) | Self-model | Yüksek | Orta | Tier-3 |
| 19 | HRL Meta-policy (#4) | Organ SAC | Zor | Tier-1 |
| 20 | Reptile Meta-train (#5) | Backtest regimes | Orta | Orta | Tier-4 |
| 21 | Architecture Search (#12) | Tüm sistem | Orta | Çok zor | Tier-4 |
| 22 | Cerebellum (#15) | Trade history | Kolay | Kolay | Tier-2 |

---

## MATEMATİKSEL FONDAMENTİKA

### Formal Problem Definition

**Hormone-Augmented Reinforcement Learning (HA-RL):**

Standart RL: `max_π E[Σ_{t=0}^{T} γᵗ R(sₜ, aₜ)]`

**KRİTİK TASARIM KARARI: Hormonlar Reward Çarpanı DEĞİL, State'in Parçası**

Eski (v8) yaklaşım: `R(s,a) × H(ω)` — reward'ı hormonlarla çarp.
**Problem:** Non-stationary reward → SAC/PPO yakınsaması (convergence) bozulur.
Ajan aynı state+action için farklı reward alır (hormon durumuna göre).
Off-policy replay buffer'daki eski deneyimler geçersizleşir.

**Yeni (v9) yaklaşım:** Hormonları observation space'e ekle, reward SAF PnL kalsın.

CAAT HA-RL:
```
max_π E[Σ_{t=0}^{T} γᵗ R(sₜ, aₜ)]

where:
  sₜ ∈ S       : AUGMENTED market state = (z, c, ω)
                  z ∈ ℝ⁶⁴  (TTM embedding)
                  c ∈ ℝ¹³⁰⁻²³⁰ (chart structure: candle DNA + SMC + VPVR + Hurst + signature)
                  ω ∈ ℝ⁴  (cortisol, dopamine, serotonin, adrenaline)
  aₜ ∈ A       : parameter adjustments (organ-grouped, ℝ³⁰⁻⁵⁰ per agent)
  R(sₜ, aₜ)    : raw trade PnL (SAF, modüle EDİLMEMİŞ)
  γ ∈ (0,1)    : discount factor
  π : S → A    : policy (hierarchical, organ-decomposed)
```

**Hormonlar observation space'te:**
```
s_augmented = concat(z_market, ω_hormones)
            = concat(ℝ⁶⁴, ℝ⁴)
            = ℝ⁶⁸

ω = (cortisol, dopamine, serotonin, adrenaline)
cortisol = max(0.5, 1 - 0.4 × stress)      ∈ [0.5, 1.0]
dopamine = min(1.1, 0.9 + 0.15 × health)    ∈ [0.9, 1.1]
serotonin = max(0.6, 0.5 + 0.5 × info_q)   ∈ [0.6, 1.0]
adrenaline = I[stress < 0.85]               ∈ {0, 1}
```

**Neden bu daha iyi:**
1. Reward SABIT → off-policy replay buffer geçerli kalır → SAC hızlı yakınsar
2. Ajan "cortisol yüksekken agresif trade = kötü sonuç" ilişkisini KENDİSİ öğrenir
3. Hormonlar hala etkili — ama IMPLICIT olarak (policy'nin içinde), explicit reward çarpanı olarak DEĞİL
4. Markov özelliği korunur: aynı (market_state, hormone_state) → aynı reward dağılımı

**Teorem (informal): Augmented State Sufficiency**
Eğer ω hormon vektörü, reward R(s,a) üzerindeki tüm non-market etkiyi capture ediyorsa,
augmented state s' = (z, ω) Markov özelliğini korur ve standart RL yakınsaması geçerlidir.

**Not:** Hormonlar hala Phase 25 organizmada get_param() modülasyonu için KULLANILIYOR.
Bu değişiklik sadece Phase 26 RL ajanlarını etkiler. Mevcut kural tabanlı hormon sistemi aynen çalışır.

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

## 8 NOVEL BİLİMSEL KATKI

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

### Contribution 6: Dual-Axis Calibration (CatBoost × CQR)
**Yenilik:** CatBoost native probability (nokta güven) + CQR interval (aralık güven) birleştirerek 2 boyutlu güven sistemi.
**Mevcut literatürde YOK:** Platt scaling veya CQR TEK BAŞINA kullanılıyor. İkisinin birleşimi yeni.
**Neden önemli:** `sizing = confidence × (1/interval_width)` — hem "kazanır mıyım?" hem "ne kadar belirsizim?" sorusunu aynı anda yanıtlar. Tek boyutlu Platt'tan çok daha bilgilendirici.
**Doğrulama:** CatBoost+CQR vs Platt-only, Brier score + interval coverage + realized PnL.
**Ek bulgu:** Platt scaling CatBoost'u BOZUYOR (Classifier Calibration at Scale, 2025) — bizim mevcut calibrator sorununun KÖK NEDENİ bu olabilir.

### Contribution 7: Triple Perception Ensemble (TTM + Chronos-Bolt + CatBoost)
**Yenilik:** 3 farklı mimari (MLP-Mixer + Transformer + Gradient Boosting) birlikte çalışır.
**Mevcut literatürde:** Dual model ensembles var ama TTM+quantile+GBM triple YOK.
**Neden önemli:** Her model farklı açıdan bakar. TTM directional, Chronos uncertainty, CatBoost final karar. Çeşitlilik = tek modelden DAHA İYİ.
**Doğrulama:** Triple vs single-model, Sharpe + drawdown + win rate karşılaştırması.

### Contribution 8: Hormonal Market Making (Cortisol→Gamma Modulation)
**Yenilik:** Stoikov/GLFT market-making parametrelerini biyolojik hormon sistemiyle modüle etmek.
**Mevcut literatürde:** Stoikov gamma sabit veya RL-learned. Hormonal modülasyon YOK.
**Neden önemli:** Organizma stresli iken spread otomatik genişler (daha az risk), sağlıklı iken daralır (daha agresif). Explicit rule gerekmez.
**Doğrulama:** Hormonal MM vs fixed-gamma MM, Sharpe + adverse selection rate.

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

---

## 14. MARKET MICROSTRUCTURE INTELLIGENCE — Duyu Sinir Sistemi

**"Tahmin yetmez, icra da zeki olmalı." — Signal quality × Execution quality = Real Alpha**

CAAT beyin, MMI beden. Orderbook, trade flow, likidasyon — piyasanın sinir uçları.

### 14.1 LOB Encoder (Order Book Perception)
**Tetik:** 100ms–1s (Bybit WebSocket)
**Input:** Top-N bid/ask levels, size, spread, update velocity
**Output:** `lob_embedding` (32-dim), `imbalance_score`, `spread_regime`

Features:
- Bid/Ask depth ratio (top 5 / 10 / 20)
- Microprice deviation: `(bid_size × ask + ask_size × bid) / (bid_size + ask_size)` vs mid
- Spread percentile (rolling 1h)
- Book pressure acceleration: `d/dt(imbalance)`

### 14.2 Order Flow Intelligence (CVD + Toxicity)
**Tetik:** Her trade tick / 1s aggregation
**Output:** `flow_toxicity`, `cvd_slope`, `aggression_state`

- **CVD (Cumulative Volume Delta):** Aggressor taraf birikimi. CVD divergence = hidden accumulation/distribution
- **Large lot burst detection:** Institutional flow proxy
- **VPIN-lite:** Volume-synchronized informed flow estimate

### 14.3 Liquidation Radar
**Tetik:** 1s–5s
**Input:** Funding, OI delta, liquidation prints, crowding metrics
**Output:** `squeeze_probability_long`, `squeeze_probability_short`, `liq_cluster_distance`

Entegrasyon:
- Squeeze olasılığı > 0.6 → pozisyon sizing clamp
- Liq cluster yakın → stop placement adaptif genişlet
- PrefrontalCortex: squeeze_prob > 0.8 + sinyal aynı yönde → VETO

### 14.4 Execution Policy Agent (Yeni RL Sub-Agent)
**Aksiyon:** limit/market oranı, order slicing, child order interval, aggression level
**Reward:** `realized_pnl - slippage_penalty - impact_penalty - non_fill_penalty`

### 14.5 Slippage Forecaster
**Model:** Lightweight GBM (5 feature → bps tahmini)
**Input:** spread, depth, volatility, urgency, order_size_percentile
**Entegrasyon:** Beklenen slippage > threshold → notional otomatik düşer veya trade pas geçilir

### Global Workspace v3 Ek Alanlar
```python
# Microstructure Intelligence
lob_embedding: torch.Tensor          # 32-dim
orderbook_imbalance: float           # [-1, +1]
flow_toxicity: float                 # [0, 1]
cvd_slope: float
squeeze_probability_long: float      # [0, 1]
squeeze_probability_short: float     # [0, 1]
expected_slippage_bps: float
execution_quality_score: float       # post-trade update
```

### Karar Füzyonu
```python
final_confidence = base_confidence * microstructure_confidence
if flow_toxicity > 0.75: position_size *= 0.6
if expected_slippage_bps > threshold: execution_mode = "passive_limit"
if squeeze_prob_short > 0.8 and signal == "short": veto = True
```

### 14.6 MARKET MAKING MODE — Regime-Based Dual Trading (Stoikov Ocak 2025)

**KRİTİK BULGU:** Sasha Stoikov (Avellaneda-Stoikov'un yaratıcısı) Ocak 2025'te crypto-specific MM makalesi yayınladı (SSRN 5066176). "Bar Portion" sinyali ile SOL/DOGE/GALA perpetual futures'ta **%45.84 cumulative return, Sharpe 0.78, max DD %3.94.**

**Bar Portion (BP) Alpha Sinyali:**
```
BP = (Close - Open) / (High - Low)
```
Body-to-range ratio. BP→+1.0 = güçlü bullish, BP→-1.0 = güçlü bearish, BP≈0 = doji.
BP smoothed (EMA) → reservation price'ı momentum yönünde skew eder.

**Regime-Based Dual Mode:**
```
Regime Detector (ADX + BOCPD)
    │
    ├── ADX < 20 (RANGING) → MARKET MAKING MODU
    │     • GLFT formülleri (terminal time YOK — 24/7 crypto'ya uygun)
    │     • Bid/ask around reservation price
    │     • Bar Portion ile directional skew
    │     • Cortisol → gamma (risk aversion) modülasyonu
    │     • Execution loop: 1-5 dakika (1h değil!)
    │
    ├── ADX 20-25 (TRANSITIONAL) → HAZIRLIK
    │     • MM inventory azalt
    │     • Spread genişlet
    │     • BOCPD changepoint bekle
    │
    └── ADX > 25 (TRENDING) → SİNYAL TAKİBİ MODU (mevcut sistem)
          • Directional trades (Evidence Engine + RL)
          • Standard Kelly sizing
```

**GLFT Formülleri (Gueant-Lehalle-Fernandez-Tapia, terminal time YOK):**
```python
c1 = (1.0 / (gamma * delta)) * log(1 + gamma * delta / k)
c2 = sqrt((gamma / (2 * A * delta * k)) * ((1 + gamma * delta / k) ** (k / (gamma * delta) + 1)))

half_spread = c1 + (delta / 2.0) * c2 * volatility
skew = c2 * volatility * bar_portion_ema  # BP ile directional skew

reservation_price = mid_price - skew * position
bid = reservation_price - half_spread
ask = reservation_price + half_spread
```

**Hormonal Gamma Modülasyonu (Akademik olarak doğrulanmış — Lalor & Swishchuk 2025):**
```python
gamma_effective = gamma_base * (2.0 - cortisol)  # cortisol=0.5 → gamma 2x (daha geniş spread)
alpha_inventory = 0.001 * (2.0 - cortisol)       # stress → güçlü inventory mean-reversion
```
Cortisol yükselince spread genişler (daha az risk), dopamine yükselince spread daralır (daha agresif).

**BOCPD (Bayesian Online Changepoint Detection) > ADX:**
- ADX lagging (14-period). BOCPD real-time.
- Tsaknaki et al. (Quantitative Finance 2024): NASDAQ order flow'da regime detection
- Score-Driven BOCPD (MBOC): time-varying correlation capture
- Dual-layer: ADX trend strength + BOCPD changepoint detection

**Bybit VIP 0 Gerçekleri:**
- Maker fee: 0.020% — MM spread bundan BÜYÜK olmalı
- BTC/USDT spread: 0.01-0.02% → major pair'lerde KÂRSIZ
- SOL/DOGE/GALA spread: 0.05-0.10% → mid-cap'lerde KÂRLI
- **Odak: mid-cap perpetual futures** (Stoikov'un test ettiği pairler)

**VPIN Adverse Selection Koruması:**
```
VPIN < 0.3:  Güvenli — normal spread
VPIN 0.3-0.6: Dikkat — spread genişlet
VPIN 0.6-0.7: Tehlike — sizing küçült
VPIN > 0.7:  Toksik akış — spread ÇOKK genişlet
VPIN > 0.8:  Likidasyon cascade riski — QUOTING DURDUR
```

**Implementasyon:** Hummingbot DEĞİL, doğrudan organism içinde. Freqtrade callback'leri ile:
- `custom_entry_price()` → bid
- `custom_exit_price()` → ask
- Ayrı 1-5dk execution loop (Bybit WebSocket)

### Anti-Fragile Guardrails
1. **Data dropout:** MMI feed giderse → Phase 26 core'a temiz degrade
2. **Latency kill-switch:** Micro modüller budget aşarsa → advisory-only mode
3. **Spoofing resistance:** Tek snapshot'a güvenme; persistence + fill confirmation
4. **Exchange heterogeneity:** Her borsa için ayrı normalization
5. **MM → Directional geçiş:** BOCPD changepoint → MM inventory unwind → 1 candle bekle → directional aktif

---

## CONSTITUTION — Değiştirilemez Kurallar

RL, LLM, hormonlar, dream engine — HİÇBİRİ bu kuralları aşamaz. DNA gibi.

```yaml
# CONSTITUTION — The Organism's Unbreakable Laws
version: 1.0
last_updated: 2026-04-07

safety_limits:
  max_drawdown_pct: 25.0           # Portfolio drawdown ASLA > %25
  max_single_position_pct: 3.0     # Tek pozisyon ASLA > %3
  max_leverage: 5.0                # Leverage ASLA > 5x
  max_portfolio_heat_pct: 10.0     # Toplam açık risk ASLA > %10
  atr_leverage_product_max: 8.0    # ATR% × leverage ASLA > %8

kill_switches:
  adrenaline_freeze_stress: 0.85   # stress > 0.85 → TÜM non-essential freeze
  oom_threshold_mb: 500            # Free RAM < 500MB → yeni trade alma
  consecutive_loss_freeze: 5       # 5 ardışık kayıp → 24h tam freeze

audit_requirements:
  every_trade_must_log:
    - feature_snapshot
    - uncertainty_bounds
    - hormone_state
    - regime_classification
    - active_overrides
    - module_contributions

override_hierarchy:
  # Üst kural alt kuralı ASLA ezmez
  1_constitution: "unchangeable — hardcoded"
  2_prefrontal_cortex: "learns from constitution, can add rules"
  3_hormonal_modulation: "dynamic but bounded by constitution"
  4_rl_policy: "optimizes within hormonal bounds"
  5_individual_neurons: "fine-tune within RL policy"
```

---

## LATENCY TIERS — Zeka Canlı Hattı Kilitlemez

Her modül bir latency tier'a atanır. Tier bütçesini aşan modül otomatik degrade olur.

| Tier | Latency Budget | Modüller | Aşarsa |
|------|---------------|----------|--------|
| **Tier-0 (Hard RT)** | < 1ms | Constitution checks, sizing clamp, adrenaline freeze | ASLA aşamaz — inline code |
| **Tier-1 (Soft RT)** | < 100ms | RL inference, hormone compute, get_param(), OOD check | Timeout → fallback value |
| **Tier-2 (Near RT)** | < 5s | TTM perception, ensemble prediction, conformal interval | Timeout → stale cache kullan |
| **Tier-3 (Background)** | < 60s | World model imagination, causal query, MADAM debate | Async thread, karar bekleMEZ |
| **Tier-4 (Offline)** | Unbounded | Dream engine, sleep consolidation, neuroevolution, ablation | Scheduler job, canlı hattı ETKİLEMEZ |

```python
# Her modül çağrısında latency guard
import time

def with_latency_guard(func, tier_budget_ms, fallback):
    start = time.monotonic()
    try:
        result = func()
        elapsed_ms = (time.monotonic() - start) * 1000
        if elapsed_ms > tier_budget_ms:
            logger.warning(f"[LatencyGuard] {func.__name__} took {elapsed_ms:.0f}ms > {tier_budget_ms}ms budget")
        return result
    except Exception:
        return fallback
```

---

## ABLATION LEAGUE TABLE — Modül Katkı Ölçümü

**Not:** Bu bölüm temel ablation konseptini tanımlar. Tam otonom versiyonu için
aşağıdaki "AUTONOMOUS ORGANISM LIFECYCLE" bölümüne bakınız — orada ablation
online Thompson Sampling + BraiNCA morphogenesis + Meta-Controller ile OTOMATİK çalışır.

Her modül haftalık KAPATILIP açılarak katkısı ölçülür. Katkısı yoksa park edilir.

```
HAFTALIK ABLATION RAPORU (otomatik, scheduler job)
╔══════════════════╦════════════╦══════════╦══════════╦═════════╗
║ Modül            ║ Sharpe Δ   ║ DD Δ     ║ WinRate Δ║ Karar   ║
╠══════════════════╬════════════╬══════════╬══════════╬═════════╣
║ Hormones         ║ +0.15      ║ -2.3%    ║ +4%      ║ KEEP ✅  ║
║ Cerebellum       ║ +0.03      ║ -0.5%    ║ +1%      ║ KEEP ✅  ║
║ Mirror Neurons   ║ +0.01      ║ -0.1%    ║ +0%      ║ WATCH ⚠ ║
║ Dream Engine     ║ -0.02      ║ +0.3%    ║ -1%      ║ PARK 🔴 ║
║ GNN              ║ +0.00      ║ +0.0%    ║ +0%      ║ PARK 🔴 ║
╚══════════════════╩════════════╩══════════╩══════════╩═════════╝

Karar kriterleri:
  KEEP:  Sharpe Δ > +0.05 VEYA DD Δ < -1%
  WATCH: -0.05 < Sharpe Δ < +0.05 (2 hafta izle)
  PARK:  Sharpe Δ < -0.05 VEYA DD Δ > +1% (deaktif et)
```

**Nasıl çalışır:**
1. Hafta 1: Modül A kapalı → performans ölç
2. Hafta 2: Modül A açık → performans ölç
3. Δ = Hafta2 - Hafta1 (aynı pair grubu, aynı piyasa koşulları)
4. Cross-validated: farklı pair gruplarında tekrar et
5. Sonuç: Ablation League Table → Telegram haftalık raporu

---

## DECISION CONTRACT — Zenginleştirilmiş Audit Trail

Her trade kararı zorunlu olarak şu bilgileri içerir (organism_audit tablosu genişletilir):

```json
{
  "trade_id": "ETH/USDT:USDT_2026-04-07T12:30:00Z",
  "timestamp": "2026-04-07T12:30:00Z",
  
  "feature_snapshot": {
    "fng": 13, "adx": 28.5, "rsi": 55, "regime": "trending_bull",
    "funding_rate": 0.0003, "btc_dominance": 56.7,
    "market_embedding_hash": "a3f2b1..."
  },
  
  "uncertainty": {
    "ensemble_variance": 0.12,
    "conformal_interval": [-2.1, 4.8],
    "ood_score": 0.15,
    "prediction_error": 0.3
  },
  
  "hormone_state": {
    "cortisol": 0.85, "dopamine": 1.02,
    "serotonin": 0.71, "adrenaline": 1.0
  },
  
  "module_contributions": {
    "evidence_engine": {"signal": "BULLISH", "confidence": 0.62},
    "world_model": {"expected_pnl": 1.8, "rollouts_profitable": "68%"},
    "causal": {"intervention_effect": 0.05},
    "mirror": {"crowd": "LONG", "contrarian_signal": 0.3},
    "cerebellum": {"hour_multiplier": 1.15}
  },
  
  "decision": {
    "action": "LONG", "confidence": 0.62,
    "sizing": 0.032, "leverage": 1.0,
    "execution_mode": "REAL"
  },
  
  "overrides": [
    "PrefrontalCortex: ATR*leverage check passed",
    "AdaptiveImmunity: no known threat"
  ],
  
  "outcome": {
    "pnl_pct": null,  // trade kapandığında doldurulur
    "exit_reason": null,
    "duration_hours": null,
    "module_blame": null  // Post-Trade Court doldurur
  }
}
```

---

## MODEL RISK ENGINE — World Model Freni

World model hata yapıyorsa, hayallere güvenme.

```python
class ModelRiskEngine:
    """Monitors model health and applies brakes when confidence is unwarranted."""
    
    def check_world_model_health(self, prediction_errors: list) -> dict:
        recent = prediction_errors[-20:]  # Son 20 trade
        avg_error = sum(recent) / len(recent) if recent else 0.5
        
        if avg_error > 0.7:
            return {"status": "UNHEALTHY", "imagination_weight": 0.2,
                    "dream_learning": False,
                    "reason": "World model error > 0.7 — imagination güvenilmez"}
        elif avg_error > 0.5:
            return {"status": "DEGRADED", "imagination_weight": 0.5,
                    "dream_learning": True,
                    "reason": "World model error elevated — imagination yarı ağırlıkta"}
        else:
            return {"status": "HEALTHY", "imagination_weight": 1.0,
                    "dream_learning": True, "reason": "Model healthy"}
    
    def check_counterfactual_bias(self, cf_errors: list) -> float:
        """Counterfactual tahminleri sistematik olarak yanlı mı?"""
        if not cf_errors: return 0.0
        mean_error = sum(cf_errors) / len(cf_errors)
        # Pozitif bias = model sürekli iyimser tahmin → tehlikeli
        if mean_error > 0.5:
            return 0.3  # Counterfactual ağırlığını %30'a düşür
        return 1.0
```

---

## POST-TRADE COURT — Kayıp Trade Dava Dosyası

Her kayıp trade için otomatik "dava dosyası" oluşturulur:

```python
class PostTradeCourt:
    """Automated blame assignment for losing trades."""
    
    def investigate(self, trade_contract: dict, organism) -> dict:
        """Hangi modül hatalı yönlendirdi?"""
        verdict = {
            "trade_id": trade_contract["trade_id"],
            "pnl_pct": trade_contract["outcome"]["pnl_pct"],
            "suspects": [],
            "primary_blame": None,
            "lessons": [],
        }
        
        modules = trade_contract["module_contributions"]
        
        # 1. Signal yanlış mıydı?
        if modules["evidence_engine"]["signal"] == "BULLISH" and verdict["pnl_pct"] < -3:
            verdict["suspects"].append({
                "module": "evidence_engine",
                "charge": "False bullish signal",
                "confidence_at_entry": modules["evidence_engine"]["confidence"],
            })
        
        # 2. World model yanıltıcı mıydı?
        if modules.get("world_model", {}).get("expected_pnl", 0) > 0 and verdict["pnl_pct"] < -3:
            verdict["suspects"].append({
                "module": "world_model",
                "charge": "Overly optimistic imagination",
                "expected_vs_actual": f"+{modules['world_model']['expected_pnl']}% vs {verdict['pnl_pct']}%",
            })
        
        # 3. Sizing çok büyük müydü?
        sizing = trade_contract["decision"]["sizing"]
        if sizing > 0.03 and verdict["pnl_pct"] < -5:
            verdict["suspects"].append({
                "module": "sizing",
                "charge": "Oversized position for the risk",
                "sizing_used": sizing,
            })
        
        # 4. Hormonlar gevşek miydi?
        cortisol = trade_contract["hormone_state"]["cortisol"]
        if cortisol > 0.95 and verdict["pnl_pct"] < -5:
            verdict["suspects"].append({
                "module": "hormones",
                "charge": "Insufficiently stressed — cortisol too relaxed",
                "cortisol": cortisol,
            })
        
        # En çok suçlu modülü belirle
        if verdict["suspects"]:
            verdict["primary_blame"] = verdict["suspects"][0]["module"]
            # Self-model'e feedback: bu modülün güvenilirliğini düşür
            organism.self_model.record_blame(verdict["primary_blame"])
        
        return verdict
```

**Entegrasyon:** Her trade kapandığında → Decision Contract doldurulur → kayıpsa Post-Trade Court çalışır → verdict organism_audit'e yazılır → Self-model güncellenir → Ablation League Table'da katkı skoru etkilenir.

---

## AUTONOMOUS ORGANISM LIFECYCLE — İnsan Müdahalesi SIFIR

**Manifestonun en kritik bölümü.** 15 bilişsel süreç, 8 novel katkı, Constitution — hepsi güzel.
Ama "kim bu sistemi çalıştırıyor?" sorusuna cevap yoksa organizma YAŞAMIYOR, sen YAŞATIYORSUN.

**Hedef:** Deploy'dan sonra insan müdahalesi SIFIR. Organizma kendi kendine öğrenir, kendi
hatalarını bulur, kendi mimarisini büyütür, kendi sağlığını korur. Sen sadece Telegram'da
bilgilendirme mesajlarını okursun. Aksiyon gerektiren mesaj = yılda 2-3 altyapı krizi, o kadar.

### 12 Katmanlı Otonom Yaşam Döngüsü

---

### Katman 1: ACTIVE INFERENCE CORE — Tek Birleştirici Karar Prensibi
**Kaynak:** Agentic Finance, MDPI Entropy 28(3), Mart 2026 — Active Inference'ın ilk finans uygulaması
**İmplementasyon:** `pymdp` (Python) veya `ActiveInference.jl`

Mevcut RL yaklaşımı (SAC/PPO): "reward'ı maximize et." Bu YETERSİZ.

Active Inference (Karl Friston): organizma reward'ı maximize etmez,
**SURPRİSE'I minimize eder — HER ŞEYLE İLGİLİ.**

**Expected Free Energy (EFE) = tek formül, TÜM kararları yönetir:**
```
G(π) = E_q[log q(s) - log p(o,s | π)]
      = pragmatic_value + epistemic_value

pragmatic = "bu aksiyon kâr getirir mi?" (exploitation)
epistemic = "bu aksiyon bana BİLGİ kazandırır mı?" (exploration)
```

**Neden RL'den ÜSTÜN:**
- RL'de exploration/exploitation trade-off'u elle ayarlanır (epsilon, entropy coeff)
- Active Inference'da exploration/exploitation **TEK FORMÜLDEN** otomatik çıkar
- Belirsizlik yüksekse → epistemic term baskın → organizma BİLGİ ARAR
- Belirsizlik düşükse → pragmatic term baskın → organizma KÂR ARAR
- **Passivity Paradox (2026 bulgusu):** bazen HİÇBİR ŞEY YAPMAMAK en akıllı aksiyon.
  Frozen belief transfer Sharpe 0.39 > naive adaptive Sharpe -0.28.
  EFE bunu otomatik keşfeder, standart RL ASLA keşfedemez.

**Tüm 15 bilişsel süreç EFE'nin farklı yüzleri:**
- Perception: piyasa surprise'ını azalt = pragmatic EFE
- World Model: gelecek surprise'ını azalt = predictive EFE
- Active Learning: bilgi boşluğunu doldur = epistemic EFE
- Dreams: hiç görmediğim senaryo hakkında surprise azalt = counterfactual EFE
- Self-Model: kendim hakkında surprise azalt = interoceptive EFE
- Hormones: surprise seviyesine göre modüle et = homeostatic EFE

**Phase 26 RL ajanları (Süreç 4) EFE-based policy'lerle enhance edilir.**
SAC hala kullanılır (continuous action space için) ama reward = -EFE.

---

### Katman 2: GELİŞİMSEL MORFOGENESİS — Mimari Kendini Büyütür
**Kaynak:** BraiNCA, arXiv 2604.01932, Nisan 2026
**GitHub:** github.com/LPioL/BraiNCA

Mevcut NEAT (Süreç 12): rastgele mutasyon, turnuva, en iyiyi seç. Bu evrim, gelişim DEĞİL.

**Fark:** Evrim = nesiller arası değişim (yavaş, CPU yoğun, 50 popülasyon).
Gelişim = tek organizmanın BÜYÜMESİ (hızlı, local kurallar, sıfır overhead).

BraiNCA: Attention + long-range connections eklenmiş Neural Cellular Automata.
Ağ bir tohumdan BÜYÜYOR — her hücre local kurallarla kendi bağlantılarını oluşturur.

**Organizma için:**
```python
class DevelopmentalGrowth:
    """BraiNCA-inspired: subsistemler arası bağlantılar BÜYÜR."""

    def growth_step(self, co_activation_matrix, synapses):
        """Her trade sonrası çağrılır — local kural, global etki."""
        for (A, B), co_act in co_activation_matrix.items():
            if co_act > self.growth_threshold:
                # Co-active subsistemler arasında bağlantı BÜYÜR
                if (A, B) not in synapses:
                    synapses.add_synapse(A, B, weight=co_act * 0.1)
                else:
                    synapses.strengthen(A, B, delta=co_act * 0.05)

            # Kullanılmayan bağlantılar ATROPHY (zayıflayıp ölür)
            for synapse in synapses.all():
                if synapse.last_used_trades_ago > 100:
                    synapses.weaken(synapse, delta=0.01)
                    if synapse.weight < 0.01:
                        synapses.prune(synapse)  # Ölü bağlantı kesilir
```

**İlk deploy:** 14 subsistem, 12 seed synapse (tohum).
**100 trade sonra:** Cerebellum↔MirrorNeurons güçlü bağ BÜYÜMÜŞ (hep birlikte aktif).
**500 trade sonra:** GNN hiçbir şeyle co-active değilmiş → bağları atrophy → otomatik park.
**1000 trade sonra:** Organizma kendi mimarisini OLUŞTURMUŞ — kimse tasarlamadı.

**NEAT'i REPLACE eder** (Süreç 12). NEAT popülasyon tabanlı (CPU yoğun).
BraiNCA local rule tabanlı (sıfır overhead, her trade'de çalışır).

---

### Katman 3: SELF-ORGANIZED CRITICALITY — Kaosun Kıyısında Optimal Hesaplama
**Kaynak:** ICLR 2025 (Yale) + HAG, Nature Communications 2025
**GitHub:** github.com/Finebouche/hag

Optimal hesaplama ne düzende ne kaosta olur — **ikisinin SINIRINDA** olur.

**Spectral radius ≈ 1.0:**
- < 1.0: sinyaller sönüyor → sistem DONMUŞ, öğrenmiyor
- > 1.0: sinyaller patlıyor → sistem ÇILDIRMIŞ, kaotik
- = 1.0: sinyaller ne sönüyor ne patlıyor → **BİLGİ İŞLEME MAKSİMUM**

**HAG (Hebbian Architecture Generation):** Unsupervised kural:
```
Δw_ij = η × (x_i × x_j − λ × w_ij)

Co-active nöronlar → bağ güçlenir
Inactive nöronlar → bağ zayıflar
λ regularization → spectral radius ≈ 1.0'a OTOMATİK yakınsar
```

**Organizma için:** Mevcut BCM + STDP kurallarına HAG EKLENİR (replace değil, enhance).
BCM: bireysel nöron plastisitesi. STDP: zamanlama. HAG: **global criticality tuning**.
Üçü birlikte: nöronlar bireysel olarak öğrenir (BCM), zamanlama öğrenir (STDP),
ve tüm sistem otomatik olarak kaosun kıyısında kalır (HAG).

**Mevcut homeostasis (Katman 2, Constrained Disorder) yerine geçer.**
Homeostasis "sabit aralıkta tut" der — kaba. SOC "optimal hesaplama noktasında tut" der — zarif.

**ICLR 2025 kanıtı:** Bu noktada eğitilen modeller, diğerlerini her benchmark'ta geçti.
Nature Comms 2025: HAG tüm geleneksel Echo State Network plastisitelerinden üstün.

---

### Katman 4: STİGMERGİK KOORDİNASYON — Feromon ile Lock-Free Haberleşme
**Kaynak:** S-MADRL, Nature Communications Engineering 2024

Mevcut Global Workspace: `RLock + copy-on-read`.
**Sorun:** GIL altında lock contention, race condition riski, stale data.

**Stigmergi:** Karıncalar birbirleriyle KONUŞMAZ. Yere feromon bırakır. Diğer karıncalar
feromonu okur. Feromon zamanla uçar (decay). Binlerce karınca mükemmel koordine olur.

```python
class PheromoneField:
    """Global Workspace'in stigmergic evrimi."""

    def __init__(self):
        self._field: dict[str, dict] = {}

    def deposit(self, source: str, signal: str, value: float, half_life_s: float = 30.0):
        """Modül feromon bırakır."""
        self._field[signal] = {
            "value": value, "source": source,
            "time": time.monotonic(), "half_life": half_life_s
        }

    def read(self, signal: str) -> float:
        """Feromonu oku — zamanla zayıflamış olabilir."""
        p = self._field.get(signal)
        if not p:
            return 0.0
        age = time.monotonic() - p["time"]
        return p["value"] * (0.5 ** (age / p["half_life"]))

    def read_all(self) -> dict[str, float]:
        """Tüm aktif feromonları oku (decay uygulanmış)."""
        return {sig: self.read(sig) for sig in self._field}
```

**Neden Global Workspace'ten ÜSTÜN:**
1. Lock YOK → race condition YOK → deadlock YOK
2. Doğal decay → stale data problemi KENDİLİĞİNDEN çözülür
3. Timestamp alignment BEDAVA — eski feromon zayıf, yeni güçlü
4. Modüller bağımsız → GIL sorunsuz (async I/O)
5. Ölçeklenebilir → 15 veya 150 modül, fark etmez
6. Emergent coordination: komut yok, feromonları takip ederek KENDİLİĞİNDEN koordine

**Örnek akış:**
```
CatBoost: deposit("catboost", "prediction_bearish", 0.78, half_life=60s)
Hormones: read("prediction_bearish") → 0.78 → cortisol↑
Amygdala: read("prediction_bearish") → 0.78 → fear↑
RL Agent: read("prediction_bearish") + read("cortisol") + read("fear") → sizing↓

30 saniye sonra:
RL Agent: read("prediction_bearish") → 0.39 (decay) → sinyal eskidi, güvenme
```

**Gemini'nin timestamp alignment eleştirisini KÖKTEN çözer.** Feromon decay = doğal timestamp.

**Global Workspace v3 ile birlikte çalışır:** v3 yapısı semantic olarak korunur,
ama mekanik olarak RLock yerine PheromoneField kullanılır.

---

### Katman 5: İNTEROSEPTİF PREDİKTİF KODLAMA — Arızayı ÖNCEDEN Gör
**Kaynak:** arXiv 2511.13668, 2025

Mevcut Interoception (Süreç 6 alt-bileşen): 8 sensör, sağlık raporu. REAKTİF.

**Prediktif kodlama:** Organizma kendi sağlığını ÖNCEDEN TAHMİN eder:
```python
class PredictiveInteroception:
    """Arızayı OLMADAN ÖNCE tahmin et."""

    def __init__(self):
        self.models = {
            "latency": SmallMLP(input=10, output=1),      # sonraki saat latency
            "ram_usage": SmallMLP(input=10, output=1),     # sonraki saat RAM
            "module_health": SmallMLP(input=20, output=15), # her modülün sağlığı
            "win_rate_trend": SmallMLP(input=30, output=1), # 7 günlük win rate trendi
        }

    def predict_and_act(self, current_state):
        for name, model in self.models.items():
            predicted = model(current_state)
            actual = measure(name)
            error = actual - predicted

            if error > 2 * std:  # Beklenenden KÖTÜ
                self.proactive_response(name, error)
                # "Latency artacak" → ŞİMDİDEN cache ısıt
                # "RAM dolacak" → ŞİMDİDEN gc.collect()
                # "Win rate düşecek" → ŞİMDİDEN defansif mod

            # Modeli güncelle (online learning)
            model.update(current_state, actual)
```

**Hiyerarşik tahmin:**
- Alt seviye: "RAM kullanımım 30 dk sonra ne olacak?" → proaktif temizlik
- Orta seviye: "Bu modülün performansı düşecek mi?" → proaktif devre dışı bırakma
- Üst seviye: "Genel sağlığım bu hafta nereye gidiyor?" → stratejik karar

**Kritik bulgu (2025):** Interoceptive precision dengesi:
- Çok yüksek precision → organizma katı, kendi modeline aşırı güvenir
- Çok düşük precision → organizma kendi bozulmasını fark etmez
- **Precision'ın kendisi de öğrenilmeli** → meta-interoception

**Mevcut Interoception sensörlerini REPLACE etmez, GÜÇLENDIRIR:**
8 sensör hala var ama artık sadece "ne" değil "ne OLACAK" sorusuna cevap veriyor.

---

### Katman 6: DANGER THEORY BAĞIŞIKLIĞI — Zeki Tehdit Algılama
**Kaynak:** Hosseini 2025 (Wiley), AIS Bio-Inspired (JAIBD 2025), CLONALG (GitHub)

Mevcut ImmuneMemory + AdaptiveImmunity: basit pair ban + B-cell/T-cell.

**3 mekanizmalı gerçek bağışıklık:**

**Negative Selection:** "Sağlıklı" profil öğren → uymayan = patojen:
```python
class NegativeSelector:
    def fit_healthy_profile(self, healthy_metrics, n_detectors=100):
        """Sağlıklı davranış profilinden random detektörler üret.
        Sağlıklıya uyan detektörler SİLİNİR. Kalanlar = anomali dedektörü."""
        self.detectors = []
        for _ in range(n_detectors * 10):
            detector = random_detector()
            if not matches_self(detector, healthy_metrics):
                self.detectors.append(detector)  # Self'e BENZEMEYEN = anomali bulur
            if len(self.detectors) >= n_detectors:
                break
```

**CLONALG:** Başarılı dedektör klonlanıp mutasyona uğrar → antikor ailesi:
```python
def clonal_expansion(self, successful_detector):
    """Tehdit bulan dedektörü klonla + hafif mutasyon → benzer tehditleri de yakala."""
    clones = [mutate(successful_detector, rate=0.05) for _ in range(10)]
    self.detectors.extend(clones)  # Bağışıklık hafızası ZENGİNLEŞİR
```

**Danger Theory:** Tepki tehdidin CİDDİYETİYLE orantılı:
- Drawdown spike = ÇOKK TEHLİKELİ → güçlü yanıt (sizing clamp + cortisol spike)
- Latency spike = ORTA → ılımlı yanıt (cache kullan, yeni trade bekleme)
- Tek hatalı log = HAFİF → izle, kaydet, tepki verme

**Katman 5 (Predictive Coding) ile BÜTÜNLEŞİK:** Artık bağışıklık sistemi
TEPKİSEL değil, PREDİKTİF. "Bu threat pattern'ini gördüm, gelişiyor" → ŞİMDİDEN antikor üret.

---

### Katman 7: ANTİFRAJİL HORMESİS — Stresle Güçlenmek
**Kaynak:** Applied Antifragility in Technical Systems, Springer 2025

Netflix Chaos Monkey'nin trading versiyonu. Organizma **kasıtlı olarak kendine stres uygular.**

**Shadow mode'da (gerçek para YOK) haftalık stress testleri:**
```
1. Rastgele bir modülü kapat → graceful degradation çalışıyor mu?
2. API latency'yi 5x artır → timeout handler doğru mu?
3. Sahte flash crash inject et → amygdala + adrenaline freeze doğru mu?
4. Sahte extreme F&G=1 inject et → cortisol + allostasis tepkisi doğru mu?
5. Feromon alanını temizle → modüller stale feromon durumunda çalışabiliyor mu?
```

**Hormesis prensibi:** Küçük zehir dozları bağışıklığı güçlendirir.
Strese dayanan yollar GÜÇLENİR (HAG Hebbian rule ile bağ güçlenir).
Strese dayanamayan yollar TESPİT EDİLİR → onarılır veya pruned.

**Criticality (Katman 3) ile BÜTÜNLEŞİK:** Stress testleri sistemi kaosun kıyısına
itip geri çeker → her iterasyonda critical point daha kesin bulunur.

**3 Taleb prensibi:**
- **Hormesis:** küçük stres = güçlenme
- **Optionality:** birden fazla yedek strateji, kayıp sınırlı kazanç sınırsız
- **Via Negativa:** başarısız bileşenleri KALDIR, yeni ekleme (less is more)

---

### Katman 8: DUAL-BRAIN COMPUTE — Kalite Kaybı Matematiksel Olarak İMKANSIZ
**Kaynak:** Speculative Decoding (Leviathan et al. ICML 2023), Anytime Algorithms (Zilberstein),
Conformal Quality Gates (Angelopoulos & Bates 2023), MoE Gating (DeepSeek-V2/V3 2024)

Eski yaklaşım: "CPU/RAM dolunca modül kapat" → **KALİTE KAYBI.**
Orta yaklaşım: "Hepsini çalıştır ama sırayla" → **GECİKME → stale feromon → dolaylı kalite kaybı.**
**Yeni yaklaşım:** Her modülün iki beyni var. Kalite ASLA düşmez, sadece YÜKSELEBİLİR.

**Mekanizma A: Dual-Brain (Hızlı Beyin + Derin Beyin)**

Her modülün İKİ versiyonu:
```
HIZLI BEYİN (her zaman, <1ms):          DERİN BEYİN (compute müsaitken):
  CatBoost → zaten <1ms                   CatBoost → full SHAP explanation
  Hormones → basit formül                  Hormones → allostasis trend analysis
  Causal → cached causal graph             Causal → Tigramite yeni keşif
  World Model → cached son simülasyon      World Model → 1000 taze rollout
  RL Agent → cached policy action          RL Agent → full EFE hesaplama
  Mirror → cached crowd direction          Mirror → taze funding+OI analiz
```

```python
class DualBrainModule:
    """Her modül hem hızlı hem derin düşünebilir."""

    def __init__(self, fast_fn, deep_fn, name):
        self.fast_fn = fast_fn    # <1ms, cache/lightweight, HER ZAMAN çalışır
        self.deep_fn = deep_fn    # ms→saniyeler, tam hesaplama
        self.name = name
        self._last_deep_result = None
        self._deep_freshness = 0

    def fast_brain(self, state) -> float:
        """Anında cevap — cache veya lightweight model."""
        if self._last_deep_result and self._deep_freshness > 0.5:
            return self._last_deep_result  # Son derin cevap hala taze
        return self.fast_fn(state)  # Lightweight hesaplama

    def deep_brain(self, state) -> float:
        """Tam hesaplama — compute müsaitken çağrılır."""
        result = self.deep_fn(state)
        self._last_deep_result = result
        self._deep_freshness = 1.0  # Taze
        return result
```

**GARANTİ:** Hızlı beyin HER ZAMAN çalışır → cevap HER ZAMAN var.
Derin beyin çalışırsa → cevap daha iyi olur. Çalışmazsa → hızlı beyin cevabı kullanılır.
Kalite hızlı beyinin ALTINA ASLA düşemez. Sadece YUKARI gidebilir.

**Mekanizma B: Computation Market (Dikkat Ekonomisi)**

Derin beyin compute'u sınırlı (CPU time). Hangi modül derin düşünsün?
**Modüller teklif veriyor — en çok bilgi kazandıracak olan kazanır:**

```python
class ComputationMarket:
    """Modüller EIG (Expected Information Gain) ile compute'a teklif verir."""

    def allocate(self, modules: list[DualBrainModule], budget_ms: float = 500):
        # Her modül "derin düşünürsem ne kadar fark yaratırım?" hesaplar
        bids = []
        for m in modules:
            # EIG: hızlı beyin cevabının belirsizliği (yüksek = derin beyin lazım)
            uncertainty = m.get_fast_brain_uncertainty()
            # Impact: bu modülün trade kararını ne kadar etkilediği
            impact = m.get_average_decision_impact()
            eig = uncertainty * impact  # Belirsiz VE etkili → derin düşün!
            bids.append((m, eig, m.deep_brain_cost_ms))

        # En yüksek EIG'den başla, budget bitene kadar
        bids.sort(key=lambda x: x[1], reverse=True)
        remaining = budget_ms
        for module, eig, cost in bids:
            if cost <= remaining:
                module.deep_brain(current_state)
                remaining -= cost
            # Budget bitse → kalan modüller hızlı beyinle devam
            # Ama cevapları VAR — sadece derin değil
```

**Örnek senaryolar:**
```
Temiz bull trend (ADX=35, sinyaller uyumlu):
  → CatBoost EIG=0.02, Causal EIG=0.01 → kimse derin düşünmez
  → Hızlı beyinler yeterli, compute BOŞTA (organizma rahat)

Rejim geçişi (CHoCH + belirsizlik yüksek):
  → CatBoost EIG=0.4, Causal EIG=0.6, WorldModel EIG=0.5
  → Market: Causal derin (200ms) + WorldModel derin (250ms) + CatBoost derin (50ms)
  → Compute tam kullanılıyor, EN ÖNEMLİ modüller derin düşünüyor

Flash crash:
  → Amygdala EIG=0.9 → amygdala derin düşünsün (freeze kararı kritik!)
  → Diğerleri hızlı beyinle idare → gecikme YOK, karar HIZLI
```

**İnsan beyni TAM BÖYLE çalışır:** Yürürken düşünmezsin (hızlı beyin).
Araba yaklaşınca TÜM dikkatini oraya verirsin (derin beyin, o modüle compute).
Diğer modüller otopilotta devam eder. Hiçbir modül KAPANMAZ.

**Mekanizma C: Conformal Quality Gate (Matematiksel Garanti)**

Hızlı beyin cevap verdi. Gerçekten yeterli mi? Sezgisel değil, **matematiksel kontrol:**

```python
class ConformalQualityGate:
    """Hızlı beyin cevabının yeterliliğini matematiksel olarak doğrula."""

    def check(self, fast_answer, conformal_model) -> tuple[bool, float]:
        interval = conformal_model.predict_interval(fast_answer, alpha=0.05)
        width = interval.upper - interval.lower

        if width < self.quality_threshold:
            return True, width   # Hızlı beyin YETERLİ — %95 garanti
        else:
            return False, width  # Aralık geniş → derin beyin ÇALIŞSIN
```

**Conformal prediction:** Dağılım-bağımsız %95 coverage garantisi.
"Hızlı beyin yeterli" diyorsa → %95 olasılıkla GERÇEKTEN yeterli. Sezgi değil, MATEMATİK.

**Üç mekanizma birlikte çalışır:**
```
Candle gelir
  → 15 modülün hızlı beyni ANINDA çalışır (<5ms toplam)
  → Conformal gate: 10 modülün hızlı cevabı yeterli (%95 garanti)
  → Kalan 5 modül belirsiz → Computation Market
    → EIG sıralaması: Causal(0.6) > WorldModel(0.5) > CatBoost(0.4) > Mirror(0.2) > Cerebellum(0.1)
    → Budget 500ms: Causal derin(200ms) + WorldModel derin(250ms) + CatBoost derin(50ms) = 500ms
    → Mirror+Cerebellum hızlı beyinle (cevapları VAR, sadece derin değil)
  → Tüm feromonlar güncellenir
  → Trade kararı verilir

Sonuç:
  - 15 modülden HİÇBİRİ kapatılmadı — hepsi cevap verdi
  - Sadece 3'ü derin düşündü (en çok lazım olan 3'ü)
  - Kalite: hızlı beyin zemininin ALTINA ASLA düşmedi
  - Compute: %100 verimli, ne israf ne eksik
  - 32GB sunucu: abanalım gitsin, ama AKILLI abanal
```

---

### Katman 9: AUTOPOİETİK KİMLİK — "Ben Hala BEN Miyim?"
**Kaynak:** Computational Autopoiesis, Academia.edu Ağustos 2025

Organizma zamanla DRİFT eder. 1000 trade sonra başlangıçtakinden tamamen farklı olabilir.

**ICAC (Introspective Clustering for Autonomous Correction):**
```python
class AutopoieticIdentity:
    def check_identity(self, current_params, baseline_params):
        """Organizma periyodik olarak kendini baseline ile karşılaştırır."""
        kl_div = kl_divergence(current_distribution, baseline_distribution)

        if kl_div < 0.1:
            return "HEALTHY_EVOLUTION"  # Sağlıklı evrim, kimlik korunuyor

        if kl_div > 0.3:
            # Performansa bak: drift iyi mi kötü mü?
            if current_sharpe > baseline_sharpe * 1.1:
                self.update_baseline(current_params)  # Pozitif evrim → YENİ baseline
                return "POSITIVE_DRIFT"
            else:
                self.soft_pullback(current_params, baseline_params, strength=0.3)
                return "NEGATIVE_DRIFT"  # Kötü drift → baseline'a doğru yumuşak geri çek
```

**Active Inference (Katman 1) ile BÜTÜNLEŞİK:** Kimlik koruması = "kendim hakkındaki
surprise'ı minimize et." EFE'nin interoceptive versiyonu.

**Catastrophic forgetting'den korur + reward hacking'den korur.**
Organizma hem evrilir hem de KENDİ KALIR.

---

### Katman 10: OTONOM CANARY DEPLOYMENT — Sıfır İnsan Değişiklik Yönetimi

Hiçbir değişiklik aniden canlıya çıkmaz:
```
NeuroEvolution / BraiNCA yeni yapı üretir
  → Shadow mode: 50 trade paralel (gerçek para YOK)
  → Shadow > live + %3?
     EVET → Canary: trade'lerin %10'unda yeni yapı
       → Canary 20 trade sonra hala iyi?
          EVET → Gradual: %10 → %30 → %60 → %100 (her aşama 10 trade)
          HAYIR → Instant rollback, yapı arşivlenir
     HAYIR → Arşivlenir (piyasa değişince tekrar denenebilir)
```

**Telegram:** "Genome v47 canlıya geçti (shadow'da +4.2%)" → bilgi, aksiyon DEĞİL.

---

### Katman 11: SİRKADYEN RİTİM — Organizmanın Biyolojik Saati

Sabit scheduler DEĞİL — Cerebellum verisiyle uyumlu dinamik yaşam döngüsü:

```
AKTİF SAATLER (Cerebellum best hours):
  → Tam algı, tüm modüller, agresif trade

SESSİZ SAATLER (thin liquidity, kötü performans):
  → Defansif: spread genişlet, sizing küçült
  → Arka plan: causal analysis, GNN keşif

GECE (03:00-06:00 UTC):
  → Sleep consolidation + Dream engine + DMN

HAFTA SONU:
  → Deep housekeeping: evolution tournament, BraiNCA growth review
  → Antifragil stress test (hormesis)
  → Identity check (autopoiesis)
  → Full Ablation raporu

AYLIK (Her ayın 1'i):
  → Baseline güncelleme kararı
  → Büyük genome turnuvası
  → Telegram: kapsamlı aylık rapor
```

---

### Katman 12: KOLEKTİF ZEKA — Organizmalar Arası Feromon Paylaşımı

Birden fazla pair organızması çalışıyorsa (BTC, ETH, SOL ayrı instance):

**Stigmergik cross-pollination:**
- Her organizma kendi PheromoneField'ına yazıyor
- Ek: "shared pheromone layer" — tüm organizmalar okuyabilir
- ETH organızması deposit("shared", "fvg_importance_bear", 0.85)
- BTC organızması read("shared", "fvg_importance_bear") → "ETH'de FVG önemli, ben de deneyeyim"
- Federated: ham veri paylaşılmaz, sadece öğrenilmiş bilgi (feromon)

---

### Katman 13: META-CONTROLLER — Kaç Beyin Kullanayım?
**Kaynak:** AAMC (Adaptive Agentic Meta-Controller), Neurocomputing 2026

İnsan beyni yürürken TÜM beynini kullanmaz. Basit iş = az kaynak.

**Task Complexity Estimator (TCE) + EFE-based routing:**
```
Piyasa durumu → TCE → karmaşıklık skoru

SIMPLE (temiz trend, ADX>30, sinyaller uyumlu):
  → CatBoost + Hormones + Constitution = yeterli
  → 3 modül, <20ms, minimal feromon trafiği

MEDIUM (karışık sinyaller, ADX 15-25):
  → + World Model + Causal + Cerebellum
  → 7 modül, <200ms

COMPLEX (rejim geçişi, CHoCH, yüksek uncertainty):
  → Tüm 15 modül aktif
  → <5s (background modüller async)

EMERGENCY (OOD çok yüksek, black swan):
  → Constitution + FREEZE
  → Sadece koruma, trade YOK
```

**Bu bir SAC policy ile ÖĞRENİLİR:** "hangi piyasa durumunda hangi modül kombinasyonu
en iyi sonuç verdi?" Meta-controller trade'lerden ÖĞRENİR, kural tabanlı DEĞİL.

**Neden kritik:** Basit piyasada 15 modül çalıştırmak = gereksiz compute,
gereksiz feromon gürültüsü, gereksiz karmaşıklık. **Zeka = ne zaman AZ düşüneceğini bilmektir.**

---

### 13 Katman Birlikte: Bir Trade'in Otonom Yaşam Yolculuğu

```
Candle gelir
  → K13 (Meta-Controller): "Piyasa MEDIUM, 7 modül derin düşünsün yeter"
  → K8 (Dual-Brain): 15 modülün hızlı beyni ANINDA çalışır (<5ms)
  → K8 (Conformal Gate): 10 modül yeterli, 5'i belirsiz
  → K8 (Compute Market): Causal + WorldModel + CatBoost derin düşünsün (EIG sıralı)
  → K4 (Stigmergi): Tüm modüller feromonlarını bırakır (bazıları derin, bazıları hızlı)
  → K1 (Active Inference): EFE hesapla → epistemic mi pragmatic mi?
  → K3 (Criticality): Spectral radius kontrol → 1.02, sağlıklı
  → Trade kararı verilir — 15 modülden HİÇBİRİ kapatılmadı, hepsi cevap verdi

Trade kapanır (kazanç)
  → K2 (Morphogenesis): co-active subsistemler arası bağ güçlenir
  → K6 (Bağışıklık): sonuç normal, tehdit yok
  → K5 (Predictive): "sonraki saatte latency tahminim 45ms" → OK
  → K9 (Autopoiesis): KL divergence 0.08, kimlik korunuyor
  → K1 (Active Inference): global surprise düştü → organizma DAHA İYİ anlıyor

Trade kapanır (kayıp)
  → Post-Trade Court: primary blame = evidence_engine
  → K6 (Bağışıklık): bu pattern'i kaydet, CLONALG ile antikor ailesi oluştur
  → K2 (Morphogenesis): evidence_engine↔sizing bağı zayıflasın mı? co-act düştü
  → K5 (Predictive): "win rate trend: hafif düşüş, henüz alarm değil"
  → K1 (Active Inference): global surprise arttı → daha çok epistemic action gerekli

Hafta sonu gelir
  → K7 (Hormesis): 5 stress test çalıştır → zayıf nokta: causal timeout handler
  → K11 (Sirkadyen): sleep + dream + evolution
  → K10 (Canary): genome v47 shadow test sonucu: +2.1% → canary başlat
  → K9 (Autopoiesis): haftalık identity check → sağlıklı evrim
  → K12 (Kolektif): ETH'den feromon: "hurst feature bu ay 2x önemli"

Sen ne yapıyorsun?
  → Telegram'da okuyorsun: "Haftalık: 12 trade, +3.4%, Sağlık: 0.81,
     Modüller: 15/15 aktif (hepsi cevap verdi, 8'i derin düşündü),
     GNN bağlantıları atrophy (morphogenesis pruning),
     Stress test: 5/5 geçti,
     Genome v47 canary'de 8/20 trade, şimdilik +1.2%"
  → Aksiyon: YOK. Organizma kendi kendine yaşıyor.
```

---

### Eski vs Yeni Karşılaştırma

| Bileşen | Eski Tasarım | PhD+++ Versiyonu |
|---------|-------------|-----------------|
| Karar prensibi | SAC/PPO (RL) | **Active Inference (EFE)** — exploration/exploitation otomatik |
| Mimari evrim | NEAT (popülasyon turnuva) | **BraiNCA morphogenesis** — mimari BÜYÜR, local kurallar |
| Denge noktası | Homeostasis (sabit aralık) | **Self-Organized Criticality** — kaosun kıyısı, HAG |
| Modül haberleşmesi | Global Workspace (RLock) | **Stigmergic pheromone** — lock-free, decay-based |
| Sağlık izleme | Interoception (reaktif) | **Predictive Coding** — arızayı ÖNCEDEN görür |
| Bağışıklık | Basit pair ban | **Negative Selection + CLONALG + Danger Theory** |
| Stres testi | Yok | **Hormesis** — Netflix Chaos Monkey for trading |
| Kimlik koruması | Yok | **Autopoiesis + ICAC** — drift detection |
| Deployment | Manuel | **Canary + Shadow + Gradual rollout** — otomatik |
| Modül seçimi | Hep hepsi açık | **Meta-Controller** — karmaşıklığa göre dinamik |
| Öğrenme kuralı | BCM + STDP | **BCM + STDP + HAG** — auto-criticality eklendi |
| Kaynak yönetimi | Degradation (modül kapat) | **Dual-Brain + Computation Market + Conformal Gate** — kalite kaybı matematiksel olarak İMKANSIZ |

### Novel Contribution #9: Stigmergic Cognitive Architecture
**Yenilik:** Bilişsel modüller arası koordinasyonu explicit message passing yerine
feromon bazlı stigmergi ile yapmak. Lock-free, doğal temporal decay, emergent coordination.
**Mevcut literatürde:** S-MADRL robotik/lojistik için. Finansal bilişsel mimaride YOK.
**Doğrulama:** Stigmergic vs RLock workspace, latency + coordination quality karşılaştırması.

### Novel Contribution #10: Developmental Morphogenesis for Trading Architecture
**Yenilik:** Trading sistemi mimarisinin BraiNCA-inspired local growth rules ile
KENDİ KENDİNE gelişmesi — NEAT-style evolution yerine developmental growth.
**Mevcut literatürde:** BraiNCA motor kontrol için. Finansal mimari growth YOK.
**Doğrulama:** BraiNCA-grown architecture vs fixed architecture, 1000 trade sonra Sharpe.

---

## Akademik Kaynaklar (100+)

### v10 Autonomous Lifecycle Kaynakları (Yeni)
- "Agentic Finance" MDPI Entropy 28(3), Mart 2026 — Active Inference'ın ilk finans uygulaması, Passivity Paradox
- BraiNCA: arXiv 2604.01932, Nisan 2026 — Attention+NCA developmental morphogenesis
- Edge of Chaos + LLM pretraining: ICLR 2025 (Yale) arXiv 2410.02536 — SOC ve optimal hesaplama
- HAG (Hebbian Architecture Generation): Nature Communications 2025 — auto-criticality, github.com/Finebouche/hag
- S-MADRL Stigmergy: Nature Communications Engineering 2024 — feromon bazlı multi-agent koordinasyon
- Interoceptive Predictive Coding: arXiv 2511.13668, 2025 — hierarchical self-prediction
- AAMC (Adaptive Agentic Meta-Controller): Rjoub et al. Neurocomputing 2026 — TCE + RL routing
- Prefrontal Meta-Control: Frontiers Computational Neuroscience 2025 — model-based vs model-free seçimi
- AIS Industrial Intrusion Detection: Hosseini 2025, Wiley — Negative Selection + Danger Theory
- CLONALG: De Castro & Von Zuben — clonal selection algorithm, github.com/christianrfg/clonalg
- Applied Antifragility in Technical Systems: Springer 2025 — hormesis, optionality, via negativa
- Computational Autopoiesis: Academia.edu Ağustos 2025 — ICAC, CDN self-reorganization
- Constrained Disorder Principle: MDPI Biology, Mart 2025 — controlled variability > fixed setpoint
- Meta-RL Homeostatic Regulation: CCN 2025, Yoshida — allostatic setpoint learning
- pymdp: Python Active Inference toolkit — EFE minimization implementation
- Speculative Decoding: Leviathan et al. ICML 2023 — dual-brain fast/slow inference
- PABEE (Patience-Based Early Exit): Zhou et al. 2020, extended 2024-25 — anytime inference
- DeepSeek-V2/V3 MoE Gating 2024 — computation market / expert routing
- Conformal Quality Gates: Angelopoulos & Bates tutorial 2023, adopted 2024-26
- Google Cascaded LLM Inference 2024 — easy→small model, hard→large model routing

### v9 Yeni Araştırma Bulguları
- Rahimikia et al. "Re(Visiting) TSFMs in Finance" arXiv 2511.18578 (Nov 2025) — CatBoost Sharpe 6.79
- Grinsztajn et al. "Why do tree-based models still outperform deep learning on tabular data?" NeurIPS 2022
- Chronos-Bolt: Amazon, late 2025 — 48M params, native quantile, CPU-viable
- TimesFM CPU infeasibility: arXiv 2602.10848 — "requires GPU acceleration"
- TimesFM_fin: pfnet arXiv 2412.09880 — finance-specific fine-tune
- Stoikov "Market Making in Crypto" SSRN 5066176 (Jan 2025) — Bar Portion alpha
- GLFT: Gueant-Lehalle-Fernandez-Tapia arXiv 1105.3115 — infinite horizon MM
- Lalor & Swishchuk "Deep RL in Non-Markov MM" MDPI Risks 2025
- Tsaknaki et al. "Online Learning of Order Flow" Quantitative Finance 2024 — BOCPD
- CQR: Romano et al. NeurIPS 2019 — Conformalized Quantile Regression
- TCP: arXiv 2507.05470 — Temporal Conformal Prediction
- CatBoost embedding features: catboost.ai/docs — native LDA + k-NN
- Classifier Calibration at Scale 2025 — Platt scaling degradation finding
- TabPFN v2.5: Nature 2024, Prior Labs — emerging neural tabular competitor
- Distributional CP: PNAS — regime-robust conformal prediction
- CPPS: Kato arXiv 2410.16333 — conformal portfolio selection
- hftbacktest: GitHub nkaz001 — GLFT backtesting toolkit
- Sadighian DRLMM: arXiv 1911.08647 — deep RL cryptocurrency MM
- Kidger & Lyons "Deep Signature Transforms" NeurIPS 2019 — path signature for time series
- Morrill et al. "Generalised Signature Method for Multivariate TS Feature Extraction" 2020
- Liao et al. "Signature features for financial time series" 2024
- Perez Arribas et al. "Sig-Wasserstein GANs for TS generation" ICAIF 2020
- Hurst exponent: Mandelbrot "Fractals and Scaling in Finance" 1997 — R/S analysis
- Smart Money Concepts: ICT (Inner Circle Trader) methodology — BOS, CHoCH, FVG, Order Blocks

### Önceki Kaynaklar (70+)

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
