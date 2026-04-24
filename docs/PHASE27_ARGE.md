# Phase 27 ARGE: CAAT Asymmetric Alpha — Araştırma & Geliştirme

## "Kaybettiğinde MİNİMAL kaybet, kazandığında BÜYÜK kazan"

> **NOT:** Yamaç sonra hatırlatacak. Sprint 2 bittikten ve yeterli trade verisi toplandıktan sonra implement edilecek.
> Bu doküman Sprint 3 vizyonudur. Sprint 2'nin CatBoost eğitimi, Causal Engine, RL agents, World Model, 
> Dream Engine, Self-Model VERİSİNE bağımlıdır. Veri olmadan formül çalışmaz.

---

## Motivasyon

Son 67 saatin analizi (9 Nisan 2026):
```
40 trade | 12W / 28L | Win Rate: 30%
Toplam PnL: -$87.09

Kazanç ortalaması: +$0.40/trade (ÇOK KÜÇÜK)
Kayıp ortalaması:  -$3.28/trade (ÇOK BÜYÜK)
Risk/Reward oranı: 0.12 (her $1 kazanç için $8.28 kayıp)

ETH katil trade: -$88.08 (tek trade, toplam kaybın %100'ü)
ETH hariç 35 trade: +$0.99 (neredeyse break-even)
```

**Sorun net:** Kazanırken erken çıkıyoruz (ROI target), kaybederken geç çıkıyoruz (büyük pozisyon + geniş stop).

**Sprint 2 hedefi:** Kayıp tarafını küçült (Sharpe iyileşir ama kazanç aynı kalır)
**Sprint 3 hedefi:** Kazanç tarafını BÜYÜT (asimetrik alpha — kayıp küçük, kazanç büyük)

---

## Araştırma Temeli (5 Cutting-Edge 2024-2026 Paper)

### 1. Friction-Adjusted Fractional Kelly (Sharpe 2.88!)
**Kaynak:** Singha, Aguilera-Toste & Lahiri, arXiv 2511.08571 (Nov 2025)

```
signal = EMA(price, slow) - EMA(price, fast)
conf = sigmoid(signal / sigma_signal)
raw_kelly = (mu_hat - cost) / sigma^2
position = min(raw_kelly * lambda_kelly * conf, W_max) * (vol_target / realized_vol)

lambda_kelly = 0.40  (fractional — yarım Kelly, güvenli)
W_max = 2.0          (leverage cap)
cost = 0.7bp linear + sqrt-impact (gamma=0.02)
```

**Sonuç:** Sharpe 2.88, max drawdown %0.52, annualized alpha %37.
**CAAT avantajı:** `mu_hat` basit momentum DEĞİL — CatBoost Sharpe 6.79 tahmin gücü.

### 2. Regime-Adaptive Trailing Stop (Sharpe 2.41, Calmar 3.18)
**Kaynak:** Bui & Nguyen, arXiv 2602.11708 (Feb 2026)

```
trailing_stop = highest_high(N) - alpha * ATR(14)
alpha_optimal = 2.5  (ATR multiplier sweet spot [2.0, 3.5])
long_allocation = 0.70  (asymmetric 70/30 long/short — crypto pozitif drift)
optimal_timeframe = 6H  (6 saatlik intraday volatilite rejimleri)
asset_selection = rolling_sharpe(6mo) > 0
```

**Keşif:** Crypto'da 70/30 long/short asymmetric allocation, equal weight'ten üstün.
**CAAT avantajı:** Hurst exponent ile regime detection → trailing genişliği otomatik.

### 3. Expected Free Energy (EFE) Sizing — Active Inference for Finance
**Kaynak:** Agentic Finance, MDPI Entropy 28(3):321 (Mart 2026)

```
G(pi) = E_Q[ln Q(s') - ln P(s'|o')] + E_Q[ln P(o'|C)]
         |_________________________|   |_______________|
              epistemic value           pragmatic value
              (bilgi kazancı)           (kâr tercihi)

position_weight ~ argmin_pi G(pi)
```

**Devrimsel bulgu:** Mean-Variance, Black-Scholes, Stochastic DCF hepsi EFE'nin ÖZEL DURUMLARI.
Epistemic value = 0 olduğunda klasik finans çıkıyor.

**CAAT avantajı:** CAAT zaten Active Inference Core (Katman 1) kullanıyor — bu formül doğal uyum.

**Entropic Sharpe Ratio (ESR) = excess return / informational work**
- Bilgi zenginken (düşük entropi, yüksek mutual information) → SIZE UP
- Belirsizlik yüksekken → SIZE DOWN

### 4. Volume-Weighted Time-Series Momentum (0.94%/gün, Sharpe 2.17)
**Kaynak:** Huang, Sangiorgi & Urquhart, SSRN 4825389 (Dec 2024)

```
TSMOM_signal(t) = sign(r_{t-K, t})                     # klasik
VWTSMOM_signal(t) = sign(sum(r_i * V_i) / sum(V_i))    # volume-weighted
position = VWTSMOM_signal * vol_target / realized_vol

K = 2 hafta  (crypto için — equities'deki 12 ay DEĞİL!)
```

**Keşifler:**
- Volume weighting: düşük hacimli fiyat hareketlerini filtreler
- **Hafta sonu momentum > hafta içi momentum** (takvim anomalisi!)
- Time-series momentum güçlü, cross-sectional momentum zayıf (fee sonrası)

**CAAT avantajı:** Chart Structure Intelligence Layer 3 (VPVR) zaten volume profile hesaplıyor.

### 5. Shannon's Demon + DreamerV3-XP (Volatility Harvesting + Dream Confidence)
**Kaynaklar:** 
- Shannon's Demon: Witte, arXiv 1508.05241
- DreamerV3-XP: arXiv 2510.21418 (Oct 2025)

**Volatility harvesting (opsiyon GEREKMİYOR):**
```
rebalancing_premium = 0.5 * w * (1-w) * sigma^2
optimal_w = 0.5  (maksimum prim 50/50 ağırlıkta)
rebalance_frequency = haftalık (crypto için)

Crypto sigma = %60-80 yıllık → %4.5-8.0 yıllık EK getiri
Yön tahmini gerekmez — piyasa SALLANIRKEN bile kâr
```

**Dream familiarity bonus:**
```
intrinsic_reward = Var(ensemble_reward_predictions)   # model uyuşmazlığı
dream_confidence = 1 / (1 + intrinsic_reward)         # yüksek uyuşmazlık = düşük güven

Ensemble hemfikir → "Bunu rüyamda gördüm" → SIZE UP
Ensemble uyuşmazlık → "Bu yeni" → SIZE DOWN
```

---

## CAAT Asymmetric Alpha Formula — Birleşik Formül

### 6 Parça, TEK Formül

```python
def caat_asymmetric_size(perception, organism, dream_engine, market_data, portfolio):
    """
    CAAT Sprint 3: Asymmetric Alpha Sizing Formula
    
    Combines:
      1. Friction-Adjusted Fractional Kelly (arXiv 2511.08571)
      2. Dual-Axis Continuous Confidence (CAAT Novel #6)
      3. Hormonal Position Sizing (CAAT Novel #11 — YENİ)
      4. Dream Familiarity Bonus (DreamerV3-XP + CAAT Novel #12 — YENİ)
      5. VWTSMOM + Fractal Regime Filter (CAAT Novel #13 — YENİ)
      6. Shannon Volatility Harvest (ranging markets)
    """
    
    # ─── PARÇA 1: Kelly Base (CatBoost + Chronos) ───
    mu = perception.catboost_predicted_return        # CatBoost Sharpe 6.79
    sigma2 = perception.chronos_interval_width ** 2  # Chronos-Bolt uncertainty
    cost = 0.0002 + estimate_slippage(market_data)   # Bybit fee + slippage
    
    if sigma2 < 1e-8:
        sigma2 = 1e-8  # division by zero guard
    
    base_kelly = max(0, (mu - cost) / sigma2) * 0.40  # 40% fractional Kelly
    
    # ─── PARÇA 2: Continuous Confidence (Dual-Axis) ───
    # Binary threshold YOK — %55 conf → %55 sizing, %90 conf → %90 sizing
    catboost_prob = perception.catboost_probability     # Axis 1: P(win)
    uncertainty = perception.uncertainty_composite       # Axis 2: CQR + ensemble + OOD
    
    if uncertainty < 0.01:
        uncertainty = 0.01
    
    confidence = 1.0 / (1.0 + math.exp(-10 * (catboost_prob / uncertainty - 0.5)))  # sigmoid
    
    # ─── PARÇA 3: Hormonal Sizing (CAAT UNIQUE — Novel #11) ───
    h = organism.hormones
    hormonal_scalar = (
        h.dopamine *                    # sağlıklı → büyük oyna
        (1.0 / max(h.cortisol, 0.3)) * # stresli → küçük oyna
        h.serotonin                     # bilgi kaliteli → büyük oyna
    )
    # Normalize to [0.5, 2.0] range
    hormonal_scalar = max(0.5, min(2.0, hormonal_scalar))
    
    # ─── PARÇA 4: Dream Familiarity Bonus (Novel #12) ───
    # Organizma bu senaryoyu rüyasında pratik etti mi?
    dream_variance = dream_engine.ensemble_variance_for_current_state()
    dream_familiarity = 1.0 / (1.0 + 10.0 * dream_variance)
    dream_bonus = 1.0 + dream_familiarity * 0.5  # max 1.5x boost
    # Tanıdık senaryo → büyük oyna. Yeni senaryo → dikkatli oyna.
    
    # ─── PARÇA 5: VWTSMOM + Fractal Regime Filter (Novel #13) ───
    vwtsmom = volume_weighted_momentum(market_data, lookback_days=14)
    hurst = perception.chart_features["hurst_100"]
    
    # Fractal-momentum hybrid
    if hurst > 0.55:
        regime_mult = 1.2   # trending → momentum güçlü, güven
    elif hurst < 0.45:
        regime_mult = 0.5   # mean-reverting → dikkatli
    else:
        regime_mult = 0.8   # random walk → çok dikkatli
    
    regime_filter = abs(vwtsmom) * regime_mult
    
    # Asymmetric allocation: crypto'da long bias
    if vwtsmom > 0:
        regime_filter *= 1.0   # long: tam güç
    else:
        regime_filter *= 0.43  # short: %43 (70/30 asymmetry)
    
    # Weekend momentum bonus
    if is_weekend():
        regime_filter *= 1.15  # hafta sonu momentum anomalisi
    
    # ─── PARÇA 6: Shannon Volatility Harvest (Ranging) ───
    adx = perception.chart_features.get("mtf_1h_adx", 25)
    if adx < 20:
        # Ranging market → Market Making modu
        # Rebalancing premium: 0.5 * w * (1-w) * sigma^2
        # Bu ayrı bir execution loop (Stoikov GLFT + Bar Portion)
        # Burada sizing'e etkisi: ranging'de yön trade'i KÜÇÜLT
        regime_filter *= 0.3  # yön trade küçük, MM kârı ayrı
    
    # ═══ FINAL SIZE ═══
    raw_size = base_kelly * confidence * hormonal_scalar * dream_bonus * regime_filter
    
    # Portfolio constraints (Constitution)
    max_position = portfolio.total * 0.03  # max %3
    min_position = 0.10  # minimum $0.10
    
    final_size = max(min_position, min(raw_size * portfolio.total, max_position))
    
    return final_size
```

---

## Adaptive Trailing Stop (Sprint 3)

```python
def caat_adaptive_trailing(trade, perception, organism):
    """
    Regime-aware trailing stop — trende göre genişlik.
    Trending: geniş trail → trende BİN, erken çıkma
    Ranging: sıkı trail → hızlı kâr al
    """
    atr = perception.chart_features.get("atr_14", trade.current_price * 0.02)
    hurst = perception.chart_features.get("hurst_100", 0.5)
    
    # Base ATR multiplier: 2.5 optimal (arXiv 2602.11708)
    if hurst > 0.55:
        # TRENDING: geniş trail — trende bin
        alpha = 3.5  # ATR × 3.5
        # ROI target KALDIR — sadece trailing stop
        remove_roi = True
    elif hurst < 0.45:
        # MEAN-REVERTING: sıkı trail — hızlı kâr al
        alpha = 1.5  # ATR × 1.5
        remove_roi = False
    else:
        # RANDOM: orta
        alpha = 2.5
        remove_roi = False
    
    # Cortisol modülasyonu: stresli → daha sıkı
    alpha *= (2.0 - organism.hormones.cortisol)  # cortisol=0.5 → alpha 2x
    
    trailing_stop = trade.highest_price - alpha * atr
    
    return trailing_stop, remove_roi
```

---

## Winner Pyramiding (Sprint 3)

```python
def caat_pyramid_decision(trade, perception, organism, dream_engine):
    """
    Kazanan trade'e ekleme kararı.
    Kaybedene DCA DEĞİL — kazanana PYRAMID.
    """
    current_pnl_pct = trade.current_profit_pct
    
    if current_pnl_pct < 0.02:
        return None  # %2'den az kârda pyramid yapma
    
    # Momentum devam ediyor mu?
    ttm_direction = perception.ttm_direction
    chronos_p50 = perception.chronos_p50
    hurst = perception.chart_features.get("hurst_100", 0.5)
    
    momentum_continues = (
        (trade.is_long and ttm_direction > 0.1 and chronos_p50 > 0) or
        (trade.is_short and ttm_direction < -0.1 and chronos_p50 < 0)
    )
    
    if not momentum_continues:
        return None  # momentum durmuş, ekleme
    
    if hurst < 0.50:
        return None  # trend yok, ekleme
    
    # Dream familiarity: bu senaryoda pratik yaptık mı?
    dream_var = dream_engine.ensemble_variance_for_current_state()
    if dream_var > 0.3:
        return None  # tanımadık senaryo, ekleme
    
    # Pyramid miktarı: mevcut pozisyonun %50'si
    # Her pyramid kademesinde küçülür: %50, %25, %12.5...
    pyramid_count = trade.get_custom_data("pyramid_count", 0)
    pyramid_fraction = 0.5 ** (pyramid_count + 1)  # diminishing
    
    pyramid_amount = trade.stake_amount * pyramid_fraction
    
    # Max 3 pyramid
    if pyramid_count >= 3:
        return None
    
    trade.set_custom_data("pyramid_count", pyramid_count + 1)
    
    return pyramid_amount
```

---

## Shannon Volatility Harvest (Ranging Markets)

```python
def caat_shannon_harvest(pairs, portfolio, rebalance_interval="weekly"):
    """
    Shannon's Demon: volatiliteden kâr — yön tahmini GEREKMİYOR.
    ADX < 20 olan pair'lerde aktif.
    
    Crypto sigma = %60-80 yıllık → %4.5-8.0 yıllık EK getiri
    Haftalık rebalance optimal.
    """
    premium = 0.0
    for pair in pairs:
        if pair.adx > 20:
            continue  # trending — Shannon geçersiz
        
        w = 0.5  # optimal weight (maximum premium)
        sigma = pair.annualized_volatility
        
        # Rebalancing premium
        pair_premium = 0.5 * w * (1 - w) * sigma ** 2
        premium += pair_premium
    
    # Haftalık rebalance: portföyü 50/50 cash/asset'e geri getir
    # Fiyat yükseldiyse → biraz sat (kâr al)
    # Fiyat düştüyse → biraz al (ucuzdan al)
    # Net: volatiliteden sistematik kâr
    
    return premium
```

---

## 3 Yeni Novel Contribution (Sprint 3)

### Novel Contribution #11: Hormonal Position Sizing
**Yenilik:** Trading pozisyon boyutunu biyolojik hormon sistemiyle modüle etmek.
**Mevcut literatürde YOK:** Kelly sizing + reward shaping var, ama hormon-bazlı sizing yok.
**Neden önemli:** Organizma stresli iken OTOMATİK küçük pozisyon — explicit rule gerekmez.
**Doğrulama:** Hormonal sizing vs fixed sizing, Sharpe + drawdown karşılaştırması.

### Novel Contribution #12: Dream-Familiar Conviction Sizing
**Yenilik:** Rüyada pratik yapılan senaryolarda pozisyon büyütme — ensemble variance bazlı.
**Mevcut literatürde:** DreamerV3-XP exploration bonus var, ama sizing'e bağlayan YOK.
**Neden önemli:** Organizma "bildiği" senaryolarda cesur, "bilmediği" senaryolarda temkinli.
**Doğrulama:** Dream-familiar sizing vs uniform sizing, risk-adjusted return.

### Novel Contribution #13: Fractal-Momentum Hybrid Filter
**Yenilik:** Hurst exponent × Volume-weighted momentum × Asymmetric allocation (70/30).
**Mevcut literatürde:** VWTSMOM (2024) ve Hurst ayrı ayrı var, birleşim YOK.
**Neden önemli:** Trend varken momentum takibi, trend yokken Shannon harvest — tek formül.
**Doğrulama:** Hybrid filter vs TSMOM-only vs buy-and-hold, Sharpe + Calmar.

---

## Beklenen Performans Etkisi

```
                    Şu an       Sprint 2      Sprint 3
Kayıp/trade:        -$3.28      -$0.50        -$0.30
Kazanç/trade:       +$0.40      +$0.40        +$2.50
Win rate:            30%         45%           50%
Risk/Reward:         0.12        0.80          8.33
Aylık (100t):       -$87        +$0.50        +$110
Sharpe (est):        -0.5        0.8           2.5+
```

**Kritik gereksinim:** Sprint 2'nin CatBoost eğitimi + Causal Engine + Dream Engine VERİSİ.
Veri olmadan formül çalışmaz. En az 500+ trade sonucu gerekli.

---

## Akademik Kaynaklar

- Singha et al. "Forecast-to-Fill" arXiv 2511.08571 (Nov 2025) — Friction-adjusted Kelly, Sharpe 2.88
- Bui & Nguyen "AdaptiveTrend" arXiv 2602.11708 (Feb 2026) — Regime trailing, Sharpe 2.41
- "Agentic Finance" MDPI Entropy 28(3):321 (Mar 2026) — EFE sizing, Entropic Sharpe Ratio
- Huang et al. "VWTSMOM" SSRN 4825389 (Dec 2024) — Volume-weighted momentum, 0.94%/day
- Han et al. "Crypto Momentum Risk-Managed" SSRN 4675565 — Weekend anomaly
- Witte "Shannon's Demon" arXiv 1508.05241 — Volatility harvesting formula
- DreamerV3-XP arXiv 2510.21418 (Oct 2025) — Ensemble variance exploration bonus
- Karassavidis et al. SSRN 5821842 — Sizing the Risk
- "Financial Information Theory" arXiv 2511.16339 — Entropy-based sizing

---

---

## PHASE 27 BOOST — PhD++ Araştırma Yönleri

> Bu bölüm Phase 27'yi 416 satırlık basic bir fazdan 3000+ satırlık PhD++ seviyesine
> çıkarmak için araştırma yol haritasıdır. Her madde bir araştırma sorusu + uygulama planıdır.

### I. ASYMMETRIC ALPHA'NIN GERÇEK KAYNAĞI — "Herkes Yanıldığında"

Asıl alpha sinyalin doğruluğunda değil, **kalabalığın yanlış olduğu anı yakalamakta.**

**Araştırma sorusu:** Forgone PnL engine veri topluyor — AI "açmadım ama açsaydım kazanırdım" diyor. Bu veriden:
- AI'ın en çok YANILDIĞI anların pattern'ı ne? (regime, saat, pair, sentiment)
- AI açmadı ama piyasa tersine gitti → **contrarian alpha sinyali**
- Crowd sentiment extreme (F&G <10 veya >90) + AI neutral = en güçlü fırsat

**Uygulama:** `forgone_alpha_detector.py` — forgone PnL engine verilerinden contrarian sinyal üret.
Referanslar: De Bondt & Thaler "Overreaction Hypothesis" (1985), Jegadeesh & Titman "Momentum" (1993)

### II. SIZING > SIGNAL — "Ne Kadar?" Sorusu Her Şeyden Önemli

**Devrimci iddia:** %50 accuracy ile PARA KAZANABİLİRSİN — sizing doğruysa.
%80 accuracy ile PARA KAYBEDEBİLİRSİN — sizing yanlışsa.

**Araştırma sorusu:** Optimal sizing formülü ne? Kelly statik, piyasa dinamik.

**Dinamik Kelly formülü:**
```
f*(t) = Kelly_base × causal_confidence × regime_modifier × cerebellum_hour × self_model_competence
```

Sprint 2 modülleri her çarpanı zaten üretiyor. Phase 27'de bunları tek formülde birleştir.

**Cutting-edge referanslar:**
- Karassavidis "Sizing the Risk" SSRN 5821842 (2026) — information-theoretic sizing
- "Financial Information Theory" arXiv 2511.16339 — entropy-based optimal fraction
- Thorp "Fortune's Formula" — orijinal Kelly uygulaması
- Cover "Universal Portfolios" (1991) — regret-free portfolio allocation

### III. CAUSAL REINFORCEMENT LEARNING (CRL) — 2025-2026'nın En Hot Konusu

RL + Causal Graph birleşimi. Sprint 2'de ikisi ayrı ayrı var (7A-7D + 6A). Phase 27'de birleştir.

**Neden devrimci:** Normal RL korelasyondan öğrenir — "funding yüksekken short açmak kârlı."
CRL nedensellikten öğrenir — "funding yüksekliği shortçuların ezilmesine NEDEN OLUYOR, o yüzden short."

**Araştırma sorusu:** Causal graph'ı RL'in reward shaping'ine nasıl entegre ederiz?
- Causal edge strength → reward bonus/penalty
- Counterfactual regret → exploration guidance
- "Bu aksiyon sahte korelasyona mı dayanıyor?" sorusunu RL her adımda sorsun

**Referanslar:**
- Lu et al. "Causal Reinforcement Learning: A Survey" (2024)
- Bareinboim "Causal Decision Making" (NeurIPS 2023 tutorial)
- Zhang et al. "Near-Optimal RL with Causal Graphs" (ICML 2025)

### IV. FOUNDATION MODEL FOR TRADING — GPT-2 LoRA Sequence Prediction

**Devrimci iddia:** Trade kararı bir "cümle" gibi. Geçmiş trade'ler = kelimeler, sonraki trade = tahmin.

Sprint 2'de decision_contract her trade'i JSON olarak logluyor. Bu bir CORPUS.
GPT-2 small (124M param) + LoRA rank=16 = sadece 900K trainable param.

**Araştırma sorusu:** Decision contract JSON'larından trade sequence öğrenip "sonraki optimal karar nedir?" tahmin edebilir miyiz?

**Return-conditioned generation:**
```
Prompt: "Son 5 trade: [WIN +2%, LOSS -1%, WIN +3%, WIN +1%, LOSS -0.5%]. 
         Sonraki 24h'de %2 getiri istiyorum."
Output: {sizing: 0.8, confidence_thresh: 0.55, stoploss: 2.5, leverage: 2.0}
```

**Referanslar:**
- Chen et al. "Decision Transformer" (NeurIPS 2021)
- ICAIF 2024 "DT-LoRA" — trading-specific Decision Transformer
- Janner et al. "Sequence Modeling as RL" (2021)

### V. ACTIVE INFERENCE SIZING — Friston Meets Kelly

Sprint 2'de Active Inference Core (9C) var ama sadece EFE hesaplıyor, sizing'e bağlamadık.

**Devrimci birleşim:** Kelly = information-theoretic optimal fraction. Active Inference = free energy minimization. İKİSİ AYNI ŞEY.

```
Kelly: f* = argmax E[log(W)]           — expected log wealth maximize et
EFE:   π* = argmin G(π)                — expected free energy minimize et
       G(π) = pragmatic + epistemic

Birleşim: f*(π) = Kelly_fraction × (1 - epistemic_uncertainty)
```

**Araştırma sorusu:** EFE'nin epistemic term'i = "bu trade bana ne kadar BİLGİ kazandırır?"
Bilgi kazancı yüksek trade → küçük pozisyon, öğrenme amaçlı (Active Learning, 9B).
Bilgi kazancı düşük, pragmatic value yüksek → büyük pozisyon, kâr amaçlı.

**BU İKİ DÜNYAYI BİRLEŞTİREN BAŞKA KİMSE YOK.** Genuinely novel contribution.

**Referanslar:**
- Friston "Active Inference" (2017)
- "Agentic Finance" MDPI Entropy 28(3) (Mar 2026) — ilk finans uygulaması
- Parr & Friston "Active Inference: The Free Energy Principle" (MIT Press 2022)

### VI. REGIME DETECTION 2.0 — Değişimi 4-8 Saat ÖNCE Yakala

Mevcut: ADX (14-period, lagging). Herkes aynı anda fark ediyor, alpha yok.

**Araştırma sorusu:** Regime değişimini ADX'ten ÖNCE nasıl yakalarız?

**BOCPD (Bayesian Online Changepoint Detection):**
- Tsaknaki et al. (Quantitative Finance 2024) — NASDAQ order flow regime detection
- Score-Driven BOCPD: time-varying correlation capture
- Causal engine verisiyle: "hangi causal edge'ler kırılıyor?" → regime kırılması işareti
- Order flow VPIN spike → regime transition öncü göstergesi

**Regime transition = EN YÜKSEK ALPHA FIRSATI** çünkü:
- Çoğu sistem lagging indicator kullanıyor (ADX, MA cross)
- 4-8 saat erken yakalarsan herkesin önünde pozisyon alırsın
- Sprint 2 cerebellum + self_model "bu regime'de yetkin miyim?" sorusunu cevaplıyor

### VII. LLM STRATEJİ KEŞİFÇİSİ — AI Kendi Stratejisini İcat Ediyor

**Devrimci konsept:** LLM'e haftanın trade'lerini ver, **yeni strateji HİPOTEZLERİ ürettir.**

```
Prompt: "Bu hafta 72 trade yapıldı. 46 win (%64). ZK/USDT 3 trade'den +$648.
         En kötü gün Pazar (-$40). En iyi gün Cumartesi (+$387).
         OOD detector %95 trade'de tetiklendi.
         Yeni bir strateji hipotezi öner."

LLM: "Hipotez: Cumartesi yüksek volume spike anlarında ZK/USDT long açılmalı,
      Pazar erken saatlerde tüm pozisyonlar kapatılmalı. OOD threshold 2.8 → 50'ye
      yükseltilmeli, mevcut threshold çok agresif."
```

Bu hipotezi backtest pipeline'dan geçir (13B). Çalışıyorsa CatBoost'a feature olarak ekle.
**AI kendi stratejisini evrimleştiriyor — gerçek otonom zeka.**

### VIII. INFORMATION-THEORETIC EDGE DETECTION — Shannon Meets Trading

**Araştırma sorusu:** Her trade'in "bilgi içeriği" ne kadar? Shannon entropy ile ölç.

**Düşük entropy sinyal** = herkes aynı şeyi düşünüyor = kalabalık doğru = alpha YOK
**Yüksek entropy sinyal** = modüller çelişiyor = belirsizlik = potansiyel alpha VAR

```
H(signal) = -Σ p_i × log(p_i)   — modüller arası uyuşmazlık

Düşük H → büyük pozisyon (kesinlik yüksek, herkes hemfikir)
Yüksek H → ya küçük pozisyon ya contrarian trade (belirsizlik = fırsat)
```

Sprint 2'de `disagreement` metriği var (triple_perception) ama Shannon entropy daha güçlü.

### IX. MULTI-TIMEFRAME CAUSAL GRAPH — "1 Saatte Bear, 1 Günde Bull"

Mevcut causal engine tek timeframe'de çalışıyor. Phase 27'de:

- 1h causal graph + 4h causal graph + 1d causal graph
- FARKLI timeframe'lerde FARKLI causal yapılar olabilir
- "1h'de funding→pnl causal, 1d'de DEĞİL" → timeframe-aware sizing
- Bu noktada hiçbir rakip yok — multi-timeframe causal discovery genuinely novel

### X. DREAM-GUIDED EXPLORATION — Rüyalardan Strateji Öğren

Sprint 2'de dream engine 100 trajectory üretiyor. Phase 27'de:

- En KÂRLI rüyaların ortak özelliklerini çıkar (cluster analysis)
- Bu özellikleri gerçek piyasada ARA — "rüyamda gördüğüm pattern şu an oluşuyor mu?"
- Dream-reality alignment score → confidence boost
- **İnsan beyni bunu yapıyor** — rüyada pratik yapar, uyanıkken tanır

### XI. ADVERSARIAL ROBUSTNESS — "Beni Kandırmaya Çalış"

AI'ın en büyük zayıflığı: adversarial manipulation. Phase 27'de:

- Kendi AI'ına SALDIRAN bir adversarial agent yaz
- "Hangi input benim AI'ımı en çok yanıltır?" sorusunu sor
- Bu input'lara karşı aşılama yap (adversarial training)
- Sonuç: AI'ın kırılma noktaları biliniyor ve güçlendirilmiş

### XII. CROSS-PROVIDER LLM ENSEMBLE — Bedava Süper Zeka

10 Gemini + Groq + Mistral + Cerebras + DeepSeek + 6 OpenRouter free model.
Toplam: ~25 ücretsiz LLM endpoint.

**Araştırma sorusu:** Aynı soruyu 3-5 farklı LLM'e sor. Cevapları nasıl birleştir?

- Majority voting: 3/5 bullish → bullish (basit)
- Confidence-weighted: her LLM'in geçmiş accuracy'sine göre ağırlıkla
- Adversarial: biri bullish biri bearish → en güçlü argümanı seç (MADAM gibi ama multi-provider)
- Cost: $0 (hepsi free tier)

**Hata kodlarından adaptif routing:**
- 429 (rate limit) → o provider'ı gün sonuna kadar atla
- 503 (overloaded) → 5dk cooldown sonra tekrar dene
- Başarılı call → Thompson Sampling ile o provider'ın score'unu artır
- Sonuç: free limitleri otomatik, adaptif, akıllıca dibine kadar vur

### XIII. EMERGENT STRATEGY DETECTION — "AI Ne Öğrendi?"

Sprint 2 modülleri haftalar geçtikçe öğrenecek. Ama NE öğrendiğini nasıl anlarız?

- Self-model competence map zamanla evrimleşiyor → görselleştir
- Causal graph zamanla değişiyor → "yeni causal edge keşfedildi" alerti
- CatBoost feature importance değişiyor → "AI artık RSI'ya değil FVG'ye bakıyor"
- Cerebellum saatlik multiplier değişiyor → "AI gece trade'lerden kaçınmayı öğrendi"

**Phase 27'nin en güzel çıktısı:** "AI şunları kendi başına öğrendi" raporu. İnsan müdahalesi sıfır.

---

## FELSEFE: KENDİMİZE HAS OLACAĞIZ

**Feyz al, kopyalama. Öğren, taklit etme.**

Kelly'den, Friston'dan, Shannon'dan, Thorp'tan ÖĞREN — ama sonra kendi formülünü yarat.
Bu sistem "Kelly + Active Inference kullanan bir bot" değil — **HydraQuant**.
Kendi isimlendirmemiz, kendi formüllerimiz, kendi terminolojimiz olacak.

Örnek:
- Kelly Criterion → **HydraQuant Adaptive Fraction (HAF)**
- Active Inference EFE → **HydraQuant Decision Energy (HDE)**
- Sharpe Ratio → **HydraQuant Risk-Adjusted Alpha (HRA)**

Başkaları paper'larında "HydraQuant formülünü kullandık" diyecek, tersi değil.

## FELSEFE: SİSTEM KENDİ KARARLARINI VERİR

"1 haftada gözlemle ve hardcode karar ver" YANLIŞ. Sprint 2'de şu modüller
bu kararları ZATEN OTOMATİK veriyor:

- `ablation_league.py` → hangi modül katkı yapıyor? KEEP/WATCH/PARK **otomatik**
- `self_model.py` → hangi pair×regime'de yetkinim? **otomatik güncellenir**
- `architecture_evolver.py` → hangi organ yapısı optimal? **otomatik evrilir**
- `autonomous_lifecycle.py` → danger level ne, sizing ne olmalı? **saatlik otomatik**
- `cerebellum_timing.py` → hangi saat iyi? **günlük otomatik güncellenir**
- `model_risk_engine.py` → modellere güvenebilir miyim? **günlük otomatik**

İnsan sadece CONSTITUTION'ı yazar (değişmez kurallar). Geri kalan HER ŞEY dinamik,
veri-driven, otomatik. Biz gözlemciyiz, müdahaleci değil.

**Phase 27'de insan müdahalesi = SIFIR olmalı.** Sistem kendi öğrenir, kendi adapte olur,
kendi evrilir. Renaissance'ın "never override the computer" prensibi — ama daha ileri:
computer kendini de override edebilir.

---

## KİSKANDIRACK SEVİYE — Gerçekten Novel Contribution'lar

Phase 27'de şu 4 tanesi DÜNYANIN İLKİ olabilir:

1. **Active Inference × Kelly Criterion birleşimi** — Friston'un EFE'si × Thorp'un Kelly'si = information-theoretic autonomous sizing. Kimse yapmadı.

2. **Multi-timeframe Causal Graph** — 1h/4h/1d'de farklı causal yapılar keşfetmek. Tigramite tek timeframe için tasarlandı, biz genişletiyoruz.

3. **Dream-Reality Alignment** — World model rüyalarının gerçek piyasa ile pattern eşleştirmesi. DreamerV3 bunu yapmıyor.

4. **LLM-Driven Strategy Evolution** — AI kendi trade verilerinden yeni strateji hipotezleri üretiyor, backtest ediyor, çalışanları deploy ediyor. Tam otonom evrim.

Bu 4'ü tek başına bir PhD tezi olur. HEPSİ BİR SİSTEMDE = dünyanın en gelişmiş autonomous trading organism.

### XIV. OOD DETECTOR 2.0 — Paranoyadan Zekaya

Mevcut OOD: Mahalanobis distance, 382 trade ile eğitildi, HER ŞEYİ şüpheli buluyor (distance=500).
Pazar refit ile 9,926 sample'a çıkacak ama yapısal sorun kalıyor:

**Araştırma sorusu:** "Tanımıyorum" demek yetmez — "tanımıyorum AMA bu tehlikeli mi yoksa fırsat mı?"

- OOD + regime = yeni regime keşfi (fırsat!)
- OOD + extreme sentiment = black swan (tehlike!)
- OOD + normal volume = sadece veri eksikliği (ignore)

Mevcut: tek threshold, binary karar. Olması gereken: **OOD tipi sınıflandırma**.

### XV. ANTI-OVERSOLD SHORT PROBLEMİ — "Dipte Short Açma"

14 Nisan dersi: RSI=5 + ADX=99'da short açıldı → zaten dip, reversal geldi, kayıp.
Trend doğru tespit edildi ama **timing yanlış** — trendin SONUNDA girdik.

**Araştırma sorusu:** "Trend doğru AMA geç mi kaldım?" sinyali nasıl üretilir?

- RSI <10 + BEARISH sinyal = "trend exhausted, short açma"
- RSI >90 + BULLISH sinyal = "trend exhausted, long açma"  
- Exhaustion detection: ADX yüksek AMA düşüyor = trend bitmek üzere
- Volume climax: aşırı volume spike = son nefes (selling/buying climax)

### XVI. TOPLU GİRİŞ KONTROLÜ — "Aynı Anda 4 Trade Açma"

14 Nisan: 17:02-17:03 arasında 4 short açıldı, hepsi kaybetti.
Aynı anda aynı yönde 4 trade = korelasyon riski, diversifikasyon yok.

**Araştırma sorusu:** Concurrent entry limiti nasıl olmalı?

- Aynı dakikada max 1 yeni trade
- Aynı yönde (hep short veya hep long) max 2 concurrent
- Cross-pair korelasyon kontrolü: "BTC düştü, hepsini short açma — hepsi aynı hareket"

### XVII. VERİ ZENGİNLEŞTİRME STRATEJİSİ

CatBoost %62'ye çıktı ama daha fazla ve daha çeşitli veri ile %70+'a çıkabilir:

- Triple Barrier labeling çalışıyor (9,647 sample üretildi) — daha fazla timerange
- Live trade verisi her gün birikiyor — haftalık otomatik retrain
- Farklı stratejilerle backtest çalıştır (SampleStrategy, InformativeSample) — çeşitlilik
- Sim2Real domain randomization ile augmented training data

---

## UFUK AÇICI ARAŞTIRMA KONULARI — "Herkes Kıskansın" Seviyesi

> Her biri tek başına bir PhD tezi. Hepsini bir sisteme koyan DÜNYADA YOK.
> Her konu için fintech-strategy-researcher ajanı yollanacak, derinlemesine araştırılacak.

### XVIII. ERGODİSİTE EKONOMİSİ — "Finans Sektörünün En Büyük Hatası"

**Ole Peters (2019, Nature Physics):** Tüm finans sektörü EXPECTED VALUE kullanıyor.
AMA expected value sadece SONSUZ paralel evrende geçerli. Biz TEK evrende yaşıyoruz.

```
Expected value:  E[wealth] = average across parallel universes  (YANLIŞ)
Time average:    <wealth>  = average across TIME for ONE person  (DOĞRU)

Kelly criterion aslında ERGODİK çözüm — ama kimse NEDEN Kelly'nin çalıştığını bilmiyor.
Cevap: Kelly time-average growth'u maximize ediyor, expected value'yu DEĞİL.
```

**Neden devrimci:** Mevcut tüm risk metrikleri (Sharpe, VaR, Expected Shortfall) yanlış
varsayıma dayanıyor. Ergodicity-corrected Sharpe = gerçek performans ölçümü.

**Araştırma:** Ole Peters'ın LML (London Mathematical Laboratory) çalışmaları,
ergodicity economics, time-average vs ensemble-average sizing formülü.

### XIX. TOPOLOJIK VERİ ANALİZİ (TDA) — "Piyasanın ŞEKLİNİ Gör"

**Persistent Homology:** Veriyi nokta bulutu olarak al, topolojik yapısını çıkar.
Delikler, döngüler, bağlantılı bileşenler — bunlar fiyat grafiğinde GİZLİ YAPILAR.

```
Price time series → Takens embedding → Point cloud → Persistent diagram
                                                       ↓
                                          "Bu yapı 2008 krizine benziyor"
                                          "Bu yapı 2021 bull run'a benziyor"
```

**Neden devrimci:** Teknik analiz 2D (fiyat vs zaman). TDA N-boyutlu yapıyı görür.
Gizli regime'ler, gizli korelasyonlar, gizli döngüler — insan gözünün göremediği.

**Referans:** Gidea & Katz "Topological Data Analysis of Financial Time Series" (2018)
Kısa paper ama finans TDA'nın temel taşı. Kripto'ya uygulanmamış.

### XX. ROUGH VOLATİLİTE — "Volatilite Pürüzsüz Değil, KABA"

**Gatheral et al. (2018):** Volatilite Brownian motion DEĞİL, fractional Brownian motion
izliyor. Hurst exponent H ≈ 0.1 (çok kaba, çok pürüzlü).

**Neden önemli:** Mevcut tüm opsiyon fiyatlaması, VaR hesabı, risk modeli
H=0.5 (smooth) varsayıyor. H=0.1 ile HER ŞEY değişiyor:
- Volatility clustering daha agresif
- Fat tails daha kalın
- Mean reversion daha hızlı

Sprint 2'de `chart_features.py` Hurst exponent hesaplıyor (hurst_50, hurst_100).
Phase 27'de bunu rough volatility modeline bağla → daha doğru uncertainty tahminleri.

### XXI. HAWKES PROCESSES — "Trade'ler Birbirini TETİKLER"

Piyasa Poisson DEĞİL — bir trade olunca DAHA FAZLA trade olur (self-exciting process).
Flash crash: 1 büyük satış → 10 stop tetiklenir → 100 likidasyon → cascade.

**Araştırma sorusu:** Hawkes process ile "cascade riski ne?" tahmin edebilir miyiz?

```
λ(t) = μ + Σ α × exp(-β × (t - t_i))

λ(t) = anlık trade yoğunluğu
μ = baseline rate
α × exp(...) = geçmiş trade'lerin tetikleme etkisi
```

Order flow (11C) verisine Hawkes fit → "şu an cascade başlangıcında mıyız?" sinyali.
Constitution'a hard block: Hawkes intensity > threshold → yeni trade açma.

### XXII. WASSERSTEIN DISTANCE — "Dağılımlar Arası GERÇEK Uzaklık"

Mevcut OOD: Mahalanobis (Gaussian varsayımı). Piyasa Gaussian DEĞİL.

**Wasserstein (Earth Mover's) distance:** Dağılım varsayımı YOK.
"Bu günün return dağılımını dünkü dağılıma dönüştürmek için minimum enerji ne kadar?"

```
W(P,Q) = inf E[|X-Y|]  — optimal transport problemi

W küçük → regime aynı
W büyük → regime değişiyor
W'nin TÜREVİ büyük → regime HIZLA değişiyor (changepoint!)
```

BOCPD'den daha güçlü: distribution-free, non-parametric, fat-tail friendly.

### XXIII. NEURAL ODE/SDE — "Piyasa Sürekli Zamanda Yaşıyor"

Mevcut modeller discrete-time (1h bar). Piyasa continuous-time'da hareket ediyor.

**Neural ODE (Chen et al. NeurIPS 2018):**
```
dz/dt = f_θ(z, t)  — neural network ODE'yi tanımlar
z(T) = z(0) + ∫₀ᵀ f_θ(z, t) dt  — ODE solver ile çöz
```

**Neural SDE:** Stochastic term ekle → uncertainty TABİATEN modelde.

World model (8C) discrete GRU kullanıyor. Neural SDE ile:
- Continuous-time dynamics
- Doğal uncertainty quantification
- Daha doğru rollout'lar (compounding error azalır)

### XXIV. KAUSAL TRANSPORTABILITY — "BTC'den Öğrendiğimi ETH'ye Taşı"

**Pearl & Bareinboim (2011):** Causal knowledge bir domain'den başka domain'e
NE ZAMAN transfer edilebilir?

Sprint 2'de causal_engine BTC için "funding→pnl" causal edge buluyor.
Bu ETH için de geçerli mi? SOL için? DOGE için?

**Araştırma sorusu:** Cross-pair causal transportability conditions nedir?
- Hangi causal edge'ler universal (tüm pair'lerde geçerli)?
- Hangi causal edge'ler pair-specific?
- Yeni listelenen bir coin'de SIFIR veri ile causal bilgiyi TRANSFER edebilir miyiz?

Bu çözülürse: yeni pair listelenir → sıfır trade geçmişi ile bile AKILLI karar.

### XXV. MULTI-AGENT GAME THEORY — "Piyasada Yalnız Değiliz"

Mevcut sistem piyasayı "doğa" olarak modelliyor — sabit, pasif.
Gerçek: piyasada BİNLERCE AI bot birbiriyle rekabet ediyor.

**Araştırma sorusu:** Diğer botların varlığını modele nasıl katarız?

- Nash equilibrium: herkes optimal oynarsa benim optimal stratejim ne?
- Adversarial awareness: "karşımdaki benim sinyallerimi exploit mi ediyor?"
- Meta-game: diğer botlar momentum kullanıyorsa → contrarian ol
- Fingerprinting: order flow'dan "bu bir bot mu insan mı?" tespiti

### XXVI. INFORMATION GEOMETRY — "Piyasa Bir MANİFOLD"

**Amari (1998):** Olasılık dağılımları bir manifold oluşturur.
Fisher Information Matrix = bu manifold'un metriği.

```
Piyasa state'i = bir noktada (dağılım parametreleri)
Regime değişimi = manifold üzerinde hareket
Geodesic distance = iki regime arasındaki EN KISA yol
Natural gradient = manifold'un eğriliğini hesaba katan gradient
```

Sprint 2'de EWC Fisher Information kullanıyor. Information geometry ile:
- Natural gradient descent (daha hızlı, daha stabil öğrenme)
- Regime arasındaki geodesic distance = "ne kadar farklı bir piyasadayım?"
- Manifold curvature = "piyasa ne kadar hızlı değişiyor?"

### XXVII. LÉVY PROCESSES — "Normal Dağılım YALAN"

Piyasa Gaussian DEĞİL. Fat tails, jumps, infinite variance.
Lévy processes: jumps + continuous part + drift.

```
X(t) = μt + σW(t) + J(t)
                      ↑
              Compound Poisson jumps
              (flash crash, whale buy)
```

World model (8C) Gaussian noise kullanıyor. Lévy noise ile:
- Flash crash'leri doğal olarak modelle
- Fat tail risk'i doğru hesapla
- Daha gerçekçi dream engine trajectory'leri

### XXVIII. OPTIMAL EXECUTION — "Almgren-Chriss for Crypto"

Büyük order'lar piyasayı hareket ettirir (market impact). 

**Araştırma sorusu:** Sizing artarsa kendi trade'imiz fiyatı ne kadar etkiler?

```
Market impact = γ × σ × √(V/ADV)

γ = permanent impact coefficient
σ = volatility  
V = order size
ADV = average daily volume
```

Mid-cap coin'lerde (ALCH, GIGGLE) impact büyük → sizing limiti gerekir.
Major'larda (BTC, ETH) impact küçük → agresif olunabilir.
Slippage forecaster (11E) bunu basitçe yapıyor ama Almgren-Chriss daha derin.

### XXIX. CONFORMAL PREDICTION 2.0 — "Garantili Prediction Intervals"

Mevcut conformal calibrator CQR kullanıyor. Phase 27'de:

- **Adaptive Conformal Inference (ACI):** non-stationary veri için (piyasa HIZLA değişiyor)
- **Conformal quantile regression:** sadece interval değil, full distribution tahmin
- **Multi-output conformal:** aynı anda fiyat + volume + volatility prediction intervals
- **Conformal risk control:** "bu trade'in %95 ihtimalle kaybı en fazla X$" garantisi

### XXX. REINFORCEMENT LEARNING FROM AI FEEDBACK (RLAIF)

RLHF = human feedback. RLAIF = AI feedback. 

**Araştırma sorusu:** Trade sonrası LLM'e sor "bu karar iyi miydi?" → reward signal olarak kullan.

```
Trade kapandı → LLM analiz eder → "kötü karar, timing yanlıştı" → 
→ negatif reward → RL bu pattern'den kaçınmayı öğrenir
```

Post-trade court (12C) zaten analiz yapıyor. Bunu RL reward'ına bağla = RLAIF.
Maliyet: trade başına 1 LLM call (free tier).

---

---

## HYDRAQUANT ORİJİNAL İNOVASYONLAR — "2026'nın Rönesansı"

> Bu bölüm KOPYALANMIŞ DEĞİL. Başkalarından feyz alındı ama formüller,
> konseptler, birleşimler TAMAMEN HydraQuant'a ait. Dünyada ilk.
> Her biri bir paper olabilir. Hepsi bir sistemde = devrim.

### HQ-1: CONFIDENCE INTEGRAL — "Anlık Güven Değil, Güvenin TARİHÇESİ"

Herkes anlık confidence kullanıyor: "şu an %62 bullish." HydraQuant soruyor:
"KAÇ SAATTİR %62 bullish?"

```
HQ_Confidence(t) = (1/T) × ∫₀ᵀ C(t-τ) × w(τ) dτ

C(t) = anlık confidence
w(τ) = time-decay kernel (recent = daha ağır)
T = integration window (örn. 4 saat)
```

4 saattir BULLISH diyen sinyal > 5 dakika önce BULLISH'e dönen sinyal.
Bu KIMSENIN kullanmadığı bir boyut — temporal persistence of conviction.

**HydraQuant farkı:** Mevcut pheromone field deposit'leri zaman damgalı.
Confidence integral = pheromone decay curve'ünün integrali. Altyapı HAZIR.

### HQ-2: METABOLIC COMPUTE ALLOCATION — "Organizma Yorulur"

Biyolojik organizmalar metabolik hıza göre enerji ayırır.
Koşarken kalp hızlanır, uyurken yavaşlar. HydraQuant:

```
Metabolic_Rate(t) = f(trade_frequency, volatility, danger_level, dream_count)

Yüksek MR → daha fazla LLM call, daha sık retraining, daha derin RAG
Düşük MR → minimal compute, sadece constitution check, enerji biriktir
```

**Neden devrimci:** Hiçbir trading sistemi compute allocation'ı BİYOLOJİK
metabolizma gibi modellemez. Bu gerçek bir "living system" tasarımı.
Sunucu maliyeti doğal olarak optimize olur — yoğun piyasada çok çalış,
sakin piyasada uyu.

### HQ-3: PHEROMONE GRADIENT NAVIGATION — "Karıncalar Gradient Takip Eder"

Karıncalar tek bir feromon izlemez — feromon GRADYANINI takip eder.
Yoğunluğun ARTIŞ YÖNÜ = doğru yol.

```
∇P(signal, t) = dP/dt = pheromone değişim hızı

∇P > 0 ve büyüyor → conviction artıyor → sizing artır
∇P > 0 ama yavaşlıyor → conviction zirve yapıyor → dikkat
∇P < 0 → conviction düşüyor → sizing azalt
∇P ikinci türev < 0 → conviction HIZLA düşüyor → çık
```

Sprint 2 pheromone field zaman damgalı deposit yapıyor.
Gradient = ardışık deposit'ler arası fark. Altyapı HAZIR.

**Kimse bunu yapmadı:** Stigmergic gradient-based trading signals. Genuinely novel.

### HQ-4: DREAM COHERENCE SCORE — "Rüyalarım Tutarlı mı?"

World model 1000 rüya üretiyor. Eğer AYNI başlangıçtan FARKLI rüyalar
birbirine benziyorsa → model güvenilir. Benzemiyor → model halüsinasyon.

```
DCS = 1 - Var(outcomes | same_initial_state) / Var(outcomes | all_states)

DCS ≈ 1 → rüyalar tutarlı → world model'a GÜVEN
DCS ≈ 0 → rüyalar kaotik → world model'a GÜVENME
DCS → model_risk_engine'e feedback
```

**Neden devrimci:** DreamerV3 rüya kalitesini ölçmez. Biz ölçüyoruz.
Model kendi güvenilirliğini BİLİYOR. Meta-meta-cognition.

### HQ-5: CAUSAL ENTROPY — "Nedensellik Haritasının Karmaşıklığı"

Causal graph'ın Shannon entropy'si:

```
H_causal = -Σ (s_i / S_total) × log(s_i / S_total)

s_i = i'inci causal edge'in strength'i
S_total = tüm edge'lerin toplam strength'i
```

**H düşük** → az sayıda edge domine ediyor → piyasa "basit" → kolay tahmin → büyük pozisyon
**H yüksek** → her yere eşit nedensellik dağılmış → piyasa "karmaşık" → zor tahmin → küçük pozisyon

**HydraQuant farkı:** Causal graph + information theory birleşimi.
Başkaları korelasyon entropy'si hesaplar, biz NEDENSELLIK entropy'si hesaplıyoruz.
Korelasyon yanıltır, nedensellik yanıltmaz.

### HQ-6: HORMONAL HYSTERESIS — "Stres HATIRLENIR"

Biyolojide kortizol düşse bile vücut stres HATILAR. Post-traumatic cautiousness.
Trading'de: dün büyük kayıp yaşadıysan bugün sakin olsa bile DİKKATLİ ol.

```
Effective_Cortisol(t) = max(current_cortisol, peak_cortisol × decay(t - t_peak))

decay(Δt) = exp(-Δt / τ_memory)  
τ_memory = 24 saat (stres hafızası)
```

Kortizol şu an 0.3 ama dün 0.9'du → effective = 0.9 × exp(-24/24) = 0.33
Kortizol şu an 0.3 ve dün de 0.3'tü → effective = 0.3

**Fark:** Aynı anlık okuma, FARKLI geçmiş → FARKLI sizing. Path-dependent risk.
Hiçbir trading sistemi hormonal hysteresis modellemez.

### HQ-7: INFORMATION ASYMMETRY INDEX — "Piyasadan NE KADAR Fazla Biliyoruz?"

Alpha = bilgi asimetrisi. Causal graph'ımız piyasa konsensüsünden FARKLI edge'ler
keşfettiyse, biz daha fazla biliyoruz.

```
IAI = |causal_edges_ours ∩ ¬consensus| / |causal_edges_ours|

IAI yüksek → bizim bildiğimiz ama piyasanın bilmediği çok → ALPHA VAR
IAI düşük → herkes aynı şeyi biliyor → alpha yok → küçük pozisyon
```

**Nasıl ölçülür:** 
- Bizim causal graph: PCMCI+ keşifleri
- Piyasa konsensüsü: public sentiment, funding rate yönü, analyst consensus
- Fark = alpha potansiyeli

### HQ-8: TEMPORAL ATTENTION SEQUENCES — "Saat Dizileri Önemli"

Cerebellum tek tek saatlere bakıyor. AMA saat DİZİLERİ daha önemli:
"3 iyi saat sonrası 1 kötü saat" = belirli bir pattern.

```
Sequence: [good, good, good, ???]
P(good | good, good, good) = ?
P(bad | good, good, good) = ?  ← streak reversal riski

Attention(seq) = Transformer(hour_embeddings) → next_hour_prediction
```

**HydraQuant farkı:** Cerebellum + Transformer attention = temporal sequence awareness.
Tek tek saat win rate'i herkes hesaplar. Saat DİZİSİ kimse hesaplamaz.

### HQ-9: ORGANISM COHERENCE TENSOR — "Modüller N-Boyutlu UYUM"

Şu an: "modül A ile B uyuşuyor mu?" (pairwise). Bu 2D düşünce.

```
Coherence Tensor C_ijk = correlation(module_i, module_j, module_k)

3-way coherence: CatBoost + Causal + SelfModel HEP BİRDEN uyuşuyor mu?
Tensor decomposition → gizli coordination pattern'ları ortaya çıkar
```

Örnek: CatBoost bullish + Causal bullish + SelfModel yetkin = **üçlü uyum → güçlü sinyal**
CatBoost bullish + Causal bearish + SelfModel yetkin = **çelişki → küçük pozisyon**

**HydraQuant farkı:** 30+ modül var. Pairwise coherence yetersiz.
Tensor coherence = N modülün birlikte uyumunu ölçen TEK SAYI.

### HQ-10: EVOLUTIONARY MEMORY — "Evrim HATIRLAR"

Architecture evolver (11A) her hafta genome'ları evriyor. Ama GEÇMİŞ evrimi hatırlamıyor.

```
Evolutionary_Memory = {
  generation_0: genome + fitness + regime + date,
  generation_1: genome + fitness + regime + date,
  ...
}

Pattern: "Bull market'ta organ X aktifti, bear'da pasifti"
→ Regime değişince GEÇMİŞ evrimi hatırla, sıfırdan evrimleştirme
→ "Bu rejimi daha önce gördüm, o zaman şu genome iyiydi" 
```

Biyolojide: epigenetik hafıza. Organizma ataların deneyimlerini taşır.
Trading'de: geçmiş regime'lerde hangi mimari iyi çalıştı → tekrar dene.

### HQ-11: FORGONE ALPHA HARVESTER — "Kaçırdıklarımızdan Öğren"

Forgone PnL engine "açmadığımız trade'ler ne kazandı?" veri topluyor.
Bu veri ALTIN MADENİ — ama kimse bunu ALPHA SINYALI olarak kullanmıyor.

```
Forgone_Alpha(pair, regime) = Σ forgone_pnl / n_forgone_trades

Eğer forgone_alpha sürekli pozitif → AI çok ihtiyatlı → threshold düşür
Eğer forgone_alpha sürekli negatif → AI doğru yapıyor → threshold koru
```

**HydraQuant farkı:** Forgone PnL'i sadece DİAGNOSTİK olarak değil,
AKTIF SİNYAL olarak kullan. "Kaçırdıklarım bana ne öğretiyor?"
Self-correcting system: çok kaçırıyorsa otomatik threshold düşürür.

### HQ-12: QUANTUM-INSPIRED SUPERPOSITION SIGNALS — "Sinyal Hem Bull Hem Bear"

Klasik: sinyal BULLISH veya BEARISH. Binary.
Quantum-inspired: sinyal SUPERPOSITION'da — çökene kadar IKISI DE.

```
|signal⟩ = α|BULL⟩ + β|BEAR⟩

|α|² = P(bull) = 0.62
|β|² = P(bear) = 0.38

"Gözlem" (trade açma) = wave function collapse
AMA açmadan ÖNCE: iki olasılığı DA sizing'e kat

Expected_sizing = |α|² × bull_sizing + |β|² × bear_sizing
```

**Neden devrimci:** Mevcut sistemler "bullish veya bearish" der.
HydraQuant "ikisi de, ağırlıklı" der. Bu daha dürüst ve daha doğru.
Quantum computing kullanmıyoruz ama quantum DÜŞÜNCE TARZI kullanıyoruz.

---

## HYDRAQUANT MANIFESTO — BİZ KİMİZ?

```
Biz Kelly DEĞİLİZ — Kelly'den öğrendik, HAF'ı yarattık.
Biz Friston DEĞİLİZ — Active Inference'tan öğrendik, HDE'yi yarattık.
Biz Renaissance DEĞİLİZ — "never override the computer"dan öğrendik,
                          ama bilgisayar KENDİNİ override edebilir dedik.
Biz DreamerV3 DEĞİLİZ — rüya görmekten öğrendik,
                         ama rüyanın KALİTESİNİ ölçtük (DCS).
Biz Pearl DEĞİLİZ — nedensellikten öğrendik,
                     ama nedenselliğin ENTROPİSİNİ ölçtük.

Biz HydraQuant'ız.
30+ modül, 52 scheduler job, 15 bilişsel süreç, 13 yaşam katmanı.
Hormonları olan, rüya gören, kendini tanıyan, evrimleşen bir organizma.
2026'nın Rönesansı.
```

---

## ARAŞTIRMA AJANI GÖREV LİSTESİ — DEVASA

> Her görev `fintech-strategy-researcher` ajanına yollanacak.
> Her sonuç Phase 27 ALPHA dökümanına beslenecek.
> Araştırma → Sentez → Kendi formülümüzü yarat → Implement → Test

### GRUP A: TEMELLİK DEĞİŞTİREN FİZİK + MATEMATİK

```
A1. "Ergodicity economics Ole Peters — time-average growth vs ensemble average.
     How does this change optimal position sizing? Compare Kelly criterion under
     ergodic vs non-ergodic assumptions. What is the CORRECT sizing formula for
     a single trader (not a fund with infinite parallel bets)? Include Peters'
     2019 Nature Physics paper and LML research."

A2. "Rough volatility Gatheral 2018 — Hurst exponent H≈0.1 in crypto markets.
     How does rough volatility change VaR calculations, option pricing, and
     uncertainty estimation? Compare with standard H=0.5 Brownian assumption.
     What trading strategies exploit rough volatility? Include fractional
     Brownian motion, RFSV model, and practical implementation."

A3. "Lévy processes for crypto — jump-diffusion models, alpha-stable distributions,
     tempered stable processes. How to model flash crashes and whale buys as jumps?
     Compare with Gaussian assumption. What is the correct tail risk measure when
     returns follow Lévy process? Include Merton jump-diffusion and Kou double-
     exponential model."

A4. "Information geometry in finance — Amari's natural gradient, Fisher information
     manifold, geodesic distance between market regimes. How does natural gradient
     descent improve RL training speed? How to measure 'distance' between two
     market regimes using information geometry? Include Riemannian optimization
     for portfolio management."

A5. "Topological Data Analysis for financial time series — persistent homology,
     Betti numbers, persistence diagrams. How to detect hidden market structures
     invisible to standard technical analysis? Applications to crypto regime
     detection and crash prediction. Include Takens embedding, Vietoris-Rips
     complex, and TDA-based trading signals."
```

### GRUP B: NEDENSELLIK + KARAR TEORİSİ

```
B1. "Causal Reinforcement Learning survey 2024-2026 — combining causal graphs
     with RL for trading. How to use causal knowledge for reward shaping,
     counterfactual exploration, and safe policy learning? Include interventional
     queries in RL, do-calculus for action selection, and causal confusion problem."

B2. "Causal transportability Pearl Bareinboim — when can causal knowledge transfer
     across domains? Application: learn causal structure from BTC, transfer to ETH.
     What are the conditions for valid transfer? How to handle distribution shift
     between crypto assets? Include selection diagrams and transportability formula."

B3. "Counterfactual trading strategies — Lopez de Prado, DoWhy applications.
     How to use counterfactual analysis for strategy improvement beyond simple
     backtesting? Include structural equation models for trading, interventional
     pricing, and counterfactual regret minimization."

B4. "Active Inference for financial decision making — Friston's Free Energy Principle
     applied to trading. Expected Free Energy for position sizing. Epistemic vs
     pragmatic value in trade selection. Include the MDPI Entropy 2026 paper
     'Agentic Finance' and pymdp implementation details."

B5. "Bayesian Online Changepoint Detection (BOCPD) for crypto regime switching.
     Compare with ADX, HMM, and Markov switching models. How to detect regime
     changes 4-8 hours BEFORE lagging indicators? Include Score-Driven BOCPD,
     Tsaknaki et al. 2024, and practical implementation for real-time detection."
```

### GRUP C: DEEP LEARNING + RL İNOVASYONLARI

```
C1. "Neural ODE and Neural SDE for financial time series — continuous-time market
     dynamics modeling. Compare with discrete GRU/LSTM. How to get natural uncertainty
     quantification from Neural SDE? Include Chen et al. 2018, latent SDE models,
     and applications to volatility forecasting."

C2. "Decision Transformer for trading — sequence modeling approach to RL.
     Return-conditioned generation: 'I want 2% return in next 24h, what params?'
     Include DT-LoRA (ICAIF 2024), GPT-2 for trading sequences, and comparison
     with standard RL approaches (PPO, SAC)."

C3. "Dream-augmented reinforcement learning — world model imagination for data
     augmentation. DreamerV3, MuZero, MCTS for trading parameter optimization.
     How to filter hallucinated dreams? How to measure dream quality?
     Include model-based RL pitfalls and our DreamFilter innovation."

C4. "RLAIF — Reinforcement Learning from AI Feedback for trading. Use LLM as
     reward model: trade outcome → LLM analysis → reward signal → RL learning.
     Compare with RLHF. Include Constitutional AI approach adapted for trading
     and multi-LLM ensemble feedback."

C5. "Foundation models for quantitative finance — large pretrained models for
     financial prediction. FinGPT, BloombergGPT, time-series foundation models
     (TimesFM, Chronos, Moirai). How to fine-tune for crypto trading?
     Include LoRA efficiency, few-shot learning, and domain adaptation."
```

### GRUP D: MARKET MICROSTRUCTURE + EXECUTION

```
D1. "Hawkes processes for crypto market microstructure — self-exciting trade arrival,
     cascade prediction, flash crash early warning. How to estimate Hawkes intensity
     from order flow? How to use Hawkes for position sizing (high intensity = danger)?
     Include multivariate Hawkes for cross-asset contagion."

D2. "Optimal execution Almgren-Chriss for crypto — market impact models, VWAP/TWAP
     strategies, order slicing. How much does our own trade move the price in mid-cap
     crypto perpetuals? Include Kyle's lambda, permanent vs temporary impact,
     and Bybit-specific considerations (VIP fees, maker/taker)."

D3. "Market making in crypto perpetual futures — Stoikov Jan 2025 paper, GLFT
     formulas without terminal time, Bar Portion alpha signal. Hormonal gamma
     modulation (Lalor & Swishchuk 2025). When to switch between market making
     and signal following? Include VPIN, adverse selection, and inventory risk."

D4. "Wasserstein distance and optimal transport for regime detection — distribution-free
     alternative to Mahalanobis. How to detect regime changes without Gaussian
     assumption? Include Earth Mover's Distance, Sinkhorn divergence, and
     applications to OOD detection in non-stationary markets."

D5. "Order flow toxicity indicators — VPIN (Volume-synchronized PIN), Kyle's lambda,
     adverse selection measures. How to detect informed trading in crypto? How to
     adjust sizing based on flow toxicity? Include PIN estimation methods and
     real-time implementation."
```

### GRUP E: PORTFÖY + SIZING OPTİMİZASYONU

```
E1. "Position sizing beyond Kelly — information-theoretic optimal fraction,
     entropy-based sizing, fractional Kelly variations. How does sizing change
     under non-Gaussian returns, regime uncertainty, and model risk?
     Include 'Sizing the Risk' (Karassavidis SSRN 2026) and 'Financial
     Information Theory' (arXiv 2511.16339)."

E2. "Adaptive conformal prediction for non-stationary financial data — ACI,
     conformal quantile regression, multi-output conformal prediction.
     How to provide GUARANTEED prediction intervals that adapt to market
     regime changes? Include distribution-free coverage guarantees and
     practical calibration for crypto."

E3. "Volatility harvesting and Shannon's Demon — extracting returns from
     volatility itself through rebalancing. How does this apply to crypto
     (most volatile asset class)? Include Cover's universal portfolios,
     constant-mix strategies, and the role of mean reversion."

E4. "Risk parity and equal risk contribution for crypto — how to allocate
     across multiple crypto positions so each contributes EQUAL risk?
     Include hierarchical risk parity (Lopez de Prado), nested clustering,
     and dynamic rebalancing with transaction costs."

E5. "Momentum and mean reversion combination strategies for crypto —
     time-series momentum (TSMOM), cross-sectional momentum, volume-weighted
     momentum (VWTSMOM SSRN 4825389). When to use momentum vs mean reversion?
     Include regime-conditional strategy selection."
```

### GRUP F: OTONOM YAŞAM + BİYOLOJİK SİSTEMLER

```
F1. "Self-organizing criticality in financial markets — power laws, scale-free
     dynamics, sand pile models. How to detect if a market is near criticality
     (about to crash)? How to MAINTAIN the organism at the edge of chaos for
     optimal computation? Include Bak et al. and Per Bak's sand pile."

F2. "Stigmergic coordination in multi-agent systems — pheromone-based
     communication without direct messaging. How do ant colonies optimize
     foraging? How to apply this to multi-module trading system coordination?
     Include digital pheromone models, evaporation dynamics, and our
     PheromoneField innovation."

F3. "Autopoiesis and self-maintaining systems — Maturana and Varela's theory
     of living systems applied to trading AI. How does the organism maintain
     its identity while adapting? When has it 'drifted too far' from its
     original design? Include identity verification and self-repair mechanisms."

F4. "Neuroevolution of augmenting topologies (NEAT) for trading architecture —
     evolving both network weights AND structure. How to evolve the organism's
     organ layout? Include innovation protection, speciation, and minimal
     complexity principle."

F5. "Integrated Information Theory (IIT) Phi metric for AI consciousness —
     Tononi's Φ as measure of system integration. How to increase Φ in our
     trading organism? Does higher Φ correlate with better trading performance?
     Include practical Φ approximation and consciousness as diagnostic metric."
```

### GRUP G: LLM + MULTI-MODAL + FUSION

```
G1. "Multi-provider LLM ensemble for trading — combining Gemini, Groq, Mistral,
     Cerebras, DeepSeek, OpenRouter free models. Majority voting vs confidence-
     weighted fusion vs adversarial debate. How to maximize FREE API usage?
     Include Thompson Sampling for model selection and adaptive rate limiting."

G2. "LLM-as-strategy-researcher — using LLMs to generate trading strategy
     hypotheses from trade data, backtest them automatically, and deploy
     successful ones. Self-improving trading strategies via LLM + backtest loop.
     Include prompt engineering for hypothesis generation."

G3. "Neurosymbolic AI for trading — combining neural networks (pattern recognition)
     with symbolic reasoning (LLM logic). How to get the best of both worlds?
     When should the neural system decide vs the symbolic system?
     Include neural-symbolic integration architectures."

G4. "Multi-modal fusion for financial prediction — combining price data (time-series),
     news (text), sentiment (scalar), order book (spatial), and social media (graph).
     Cross-attention mechanisms, modality dropout, missing modality handling.
     Include recent multi-modal finance papers 2024-2026."

G5. "Adaptive rate limiting and error-code-driven API routing — how to maximize
     throughput across multiple LLM providers with different rate limits, pricing,
     and availability patterns. Build a self-learning router that adapts to
     provider behavior in real-time. Include multi-armed bandit approaches."
```

### GRUP H: HYDRAQUANT ORİJİNAL KONSEPTLER İÇİN DOĞRULAMA

```
H1. "Temporal persistence of trading signals — does signal DURATION matter?
     Research if signals that persist for N hours have higher win rates than
     flash signals. Validate our HQ-1 Confidence Integral concept.
     Search for any existing work on signal duration vs profitability."

H2. "Path-dependent risk management — does HISTORY of risk (not just current level)
     affect optimal sizing? Research hysteresis in risk management, post-traumatic
     cautiousness in behavioral finance. Validate our HQ-6 Hormonal Hysteresis."

H3. "Contrarian alpha from forgone trades — can the trades NOT taken by an algorithm
     provide a profitable signal? Research inverse signals, contrarian indicators
     from model disagreement. Validate our HQ-11 Forgone Alpha Harvester."

H4. "Quantum-inspired optimization for portfolio allocation — QAOA, quantum annealing,
     superposition-based search. NOT quantum computing, but quantum THINKING for
     classical algorithms. Validate our HQ-12 Superposition Signals concept."

H5. "Dream coherence as model reliability metric — in model-based RL, how to measure
     if a world model is reliable? Consistency of imagined trajectories as diagnostic.
     Validate our HQ-4 Dream Coherence Score. Search for model reliability metrics
     in MBRL literature."
```

**TOPLAM: 40 ARAŞTIRMA GÖREVİ — 8 GRUP × 5 GÖREV**

**KURAL: Minimum 40 ajan, her ajanın TEK BİR görevi olacak.**
- 1 ajan = 1 görev = 1 derin araştırma = ajanın TÜM enerjisi o konuya
- Ajanlar paralel çalışabilir (5'li gruplar halinde yollanabilir)
- Grup A-G: `fintech-strategy-researcher` — internet araştırması (paper, strateji, referans)
- Grup H: `explorer-god` — kendi kodumuzu inceleyip HQ konseptlerini doğrulama + altyapı kontrolü
- Her ajan prompt'u yukarıdaki görev açıklamasını AYNEN içerecek

**Önerilen çalışma sırası:**
1. Önce GRUP H (HydraQuant doğrulama) — kendi fikirlerimizi doğrula
2. Sonra GRUP A + B (temeller) — fizik + nedensellik altyapısı
3. Sonra GRUP C + D (deep learning + microstructure) — teknik derinlik
4. Sonra GRUP E + F (sizing + biyoloji) — uygulama katmanı
5. Son GRUP G (LLM + fusion) — entegrasyon katmanı

### GRUP I: ÇILGIN FİKİRLER — "Ya Böyle Yaparsak?"

```
I1. "Market as a living organism — viewing the entire crypto market as a single
     organism with its own metabolism, circadian rhythms, immune responses.
     Our organism INSIDE another organism. How to model market-as-organism?
     Research: Sornette dragon kings, market ecology, predator-prey dynamics
     between momentum and mean-reversion traders."

I2. "Adversarial self-play for strategy improvement — train TWO copies of our
     organism against each other. One tries to BREAK the other's strategy.
     The winner's defense becomes the production strategy. Like AlphaGo
     self-play but for trading. Research: competitive co-evolution,
     adversarial training for robustness, red-team AI trading."

I3. "Emotional contagion in crypto markets — fear and greed SPREAD like viruses
     through social networks. Model this as an epidemiological SIR model.
     Susceptible→Infected→Recovered for market sentiment. When infection rate
     peaks = contrarian opportunity. Research: social contagion models,
     information cascades, herding behavior in crypto."

I4. "Sleep-wake cycle optimization — our organism has circadian rhythms but
     what if it could CHOOSE when to sleep? Reinforcement learning for
     optimal sleep scheduling: 'sleep now because nothing good happens in
     next 3 hours, wake up for London open.' Research: optimal stopping
     theory, attention allocation, energy-aware computing."

I5. "Trading as language — every trade is a 'word', every day is a 'sentence',
     every week is a 'paragraph'. Apply NLP techniques (n-grams, transformers,
     topic modeling) to trade SEQUENCES. Discover 'grammar' of profitable
     trading patterns. Research: sequence modeling for trading,
     trade pattern language, grammar induction from financial data."
```

### GRUP J: AGENT MEMORY KRİZİ — "Ajanlarımız Amnezi Hastası"

> Explorer-god audit sonucu (14 Nisan 2026): Agent'ların track record'u çalışıyor
> ama GERÇEK HAFIZASI YOK. ReflectionAgent kör. MAGMA bağlantısı sıfır.
> MiroFish'in gücü agent'ların HATIRLAMASI — bizimkiler her debate'te sıfırdan başlıyor.

```
J1. "ReflectionAgent'a GERÇEK hafıza ver — Round 3'te agent_memory ve 
     agent_performance tablolarından son 50 trade'in verilerini çek ve prompt'a
     enjekte et. 'TrendFollower son 20 trade'de trending_bull'da %78 win rate,
     ama ranging'de %31' gibi GERÇEK VERİ görsün. Şu an prompt 'silahların:
     tablolar' diyor ama tablolara hiç bakılmıyor. PROMPT YALAN SÖYLÜYOR.
     Araştır: episodic memory in LLM agents, retrieval-augmented agent memory."

J2. "Agent'lara MAGMA graph memory entegre et — her agent'ın her debate'teki
     argümanı, outcome'u, ikna ettiği/edildiği agent = MAGMA causal+temporal
     edge. 'TrendFollower dedi BULL, oldu BULL, strength +0.1' = Hebbian.
     Zaman içinde hangi agent kimi ne zaman ikna ediyor = social learning graph.
     Araştır: multi-agent memory architectures, social learning in MAS."

J3. "Agent-bazlı RAG — her agent KENDİ spesiyalizasyonuna göre RAG yapsın.
     TrendFollower → trend-specific patterns, momentum indicators
     FundingContrarian → funding rate history, squeeze events
     MacroCorrelator → macro news, cross-asset correlations
     Şu an hepsi AYNI RAG context alıyor = spesiyalizasyon YOK.
     Araştır: specialized retrieval for multi-agent systems, expert routing."

J4. "Agent key_argument geri okuma — agent_memory tablosuna key_argument
     yazılıyor ama HİÇBİR YERDE okunmuyor. Bu ALTIN MADENİ:
     'TrendFollower ETH için ne demişti? → BULL, çünkü ADX>30 + EMA cross.
     Sonuç: +%5.2 kazanç. Bu argüman GÜÇLÜ.'
     key_argument + outcome = argüman kalitesi skoru. Bunu debate'e feedback et.
     Araştır: argument quality scoring, debate outcome learning."

J5. "Agent pheromone + Grafeo tam entegrasyon — agent'lar pheromone field'a
     kendi sinyallerini yazsın, diğer agent'ların sinyallerini okusun.
     add_agent_interaction() DEAD CODE — debate'te kim kimi ikna etti bilgisi
     Grafeo'ya yazılsın. Agent sosyal ağı = kolektif zeka altyapısı.
     Araştır: stigmergic multi-agent coordination, digital pheromone for
     agent communication, swarm intelligence trading systems."
```

**TOPLAM: 50 ARAŞTIRMA GÖREVİ — 10 GRUP (A-J)**

Her görevin çıktısı:
1. En güncel akademik referanslar (2024-2026 öncelikli)
2. Pratik implementation stratejisi
3. HydraQuant'a nasıl entegre edilir?
4. Tahmini alpha katkısı
5. RAM/compute bütçesi
6. Bağımlılıklar (hangi Sprint 2 modülü gerekli?)

---

---

## Timeline

**Prereq:** Sprint 2 tamamlanmalı + en az 500 trade sonucu toplanmalı.
**Tahmini başlangıç:** Sprint 2 bitişinden 2-4 hafta sonra (veri toplama süresi).
**Tahmini süre:** 1 sprint (10 gün aktif çalışma).

> **YAMAÇ NOTU:** Bu doküman Sprint 3 vizyonudur. Sprint 2 bittikten sonra Yamaç hatırlatacak.
> O zaman güncel trade verileri ile formülün parametreleri kalibre edilecek.
> Şimdilik sadece PLAN — implementasyon Sprint 2 verisine bağlı.

---

## REAL MONEY TRANSITION: "Babaannem Bile Güvensin" Prensibi

**Testnet:** Trade-First — her sinyalde trade aç, veri topla, öğren. Kayıp = bedava ders.

**Gerçek para:** Data-First — veriler sağlam olmadan trade açma.

### Confidence-Based Execution Tiers (Gerçek Para İçin)

| Tier | Confidence | Arkadaki Veri | Aksiyon | Sizing |
|------|-----------|---------------|---------|--------|
| **Shadow** | < 0.50 | Yetersiz veri, model emin değil | Shadow trade (logla, para koyma) | 0% |
| **Micro** | 0.50 - 0.65 | Bazı veriler uyumlu, bazıları eksik | Minimum pozisyon (öğrenme amaçlı) | 10-25% |
| **Normal** | 0.65 - 0.80 | Çoğu modül uyumlu, CatBoost + RL hemfikir | Normal pozisyon | 50-80% |
| **Full** | > 0.80 | TÜM modüller uyumlu, causal graph destekliyor, self-model yetkin | Tam pozisyon | 100% |

### "AI'a Sırtını Yasla" Prensibi

AI'a güvenmek lazım. Çok katman = korku = trade açmama = en büyük kayıp.

**Sadece 2 HARD BLOCK var (constitution — değişmez):**
1. Constitution violation (drawdown >%25, leverage >5x, 5 consecutive loss)
2. Order Flow VETO (squeeze_prob > 0.8 + aynı yönde sinyal)

**Geri kalan her şey SIZING'i modüle eder, BLOKLAMAZ:**
- CatBoost düşük confidence → küçük pozisyon, ama trade AÇ
- Self-Model düşük competence → %70 sizing, ama trade AÇ  
- Cerebellum kötü saat → %80 sizing, ama trade AÇ
- Lifecycle cautious → %60 sizing, ama trade AÇ

**Neden:** Renaissance 2007'de 3 günde $1B kaybetti, intervene etmedi, yıl sonu +%85.9.
AI kendi verisiyle kendi kararını versin. Biz sadece constitution ile çerçeveyi çizelim.

---

## LLM DEEP INTEGRATION: Free Limitleri Dibine Kadar Vur

**Mevcut durum:** LLM sadece MADAM debate'te kullanılıyor (3 call/sinyal, sadece düşük confidence'ta).
10 Gemini key + Groq + Mistral + Cerebras + DeepSeek var ama çoğu boşta duruyor.

**Phase 27 araştırma konuları:**

### 1. LLM-as-Judge (her trade sonrası)
- Trade kapandığında LLM'e sor: "Bu kararı analiz et. Ne iyi gitti, ne kötü?"
- Post-trade court'un (12C) LLM destekli versiyonu
- Maliyet: 1 call/trade, Gemini Flash ucuz

### 2. LLM-Powered Causal Reasoning
- Causal engine (6A) istatistiksel edge buluyor ama NEDEN'i bilmiyor
- LLM'e sor: "funding_rate → pnl causal edge buldum. Bu neden mantıklı?"
- LLM saçma causal edge'leri filtreleyebilir (domain knowledge)

### 3. News Deep Analysis (şu an yüzeysel)
- Şu an: haber → embedding → similarity search
- Olması gereken: haber → LLM "bu haberin 1h/4h/1d etkisi ne olur?" → structured output
- Her haber için 3 zaman diliminde etki tahmini

### 4. Dream Scenario Narration
- World model (8C) sayısal trajectory üretiyor
- LLM bu trajectory'yi yorumlasın: "Bu senaryoda flash crash sonrası recovery var"
- İnsan-okunabilir dream raporları

### 5. Strategy Self-Reflection (haftalık)
- Haftanın tüm trade'lerini LLM'e ver
- "Bu hafta neyi iyi yaptın, neyi kötü? Gelecek hafta neye dikkat etmelisin?"
- Telegram'a haftalık AI self-reflection raporu

### 6. Cross-Provider Ensemble
- Aynı soruyu Gemini + Groq + Mistral'e sor
- 3 LLM hemfikirise → confidence boost
- Hemfikir değilse → uncertainty artır
- Free tier'ları parallel kullanarak maliyet = $0

### 7. LLM Feature Engineering
- LLM'e OHLCV pattern göster: "Bu mumları yorumla"
- LLM çıktısını feature olarak CatBoost'a ver
- Symbolic (LLM yorumu) + Numeric (chart features) fusion

**Araştırılacak:** Her provider'ın free tier limitleri, rate limit'leri, hangi model hangi iş için optimal, toplam günlük ücretsiz kapasite hesabı.

### Shadow Trading Altyapısı (Mevcut)

Bu altyapı ZATEN VAR:
- `forgone_pnl_engine.py` — "açsaydım ne olurdu?" takibi
- `decision_contract.py` — her karar JSON provenance ile loglanıyor
- `post_trade_court.py` — her trade otopsi ediliyor
- `self_model.py` — pair×regime yetkinlik haritası
- `constitution.py` — unbreakable safety limits

Gerçek paraya geçişte tek yapılacak: confidence threshold'u yükseltmek ve shadow tier'ı aktif etmek.
Kod değişikliği minimal — `HydraSizer.custom_stake_amount()`'da confidence < 0.50 → return 0 (shadow log).
