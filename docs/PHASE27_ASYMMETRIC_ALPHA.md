# Phase 27: CAAT Asymmetric Alpha Formula

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

## Timeline

**Prereq:** Sprint 2 tamamlanmalı + en az 500 trade sonucu toplanmalı.
**Tahmini başlangıç:** Sprint 2 bitişinden 2-4 hafta sonra (veri toplama süresi).
**Tahmini süre:** 1 sprint (10 gün aktif çalışma).

> **YAMAÇ NOTU:** Bu doküman Sprint 3 vizyonudur. Sprint 2 bittikten sonra Yamaç hatırlatacak.
> O zaman güncel trade verileri ile formülün parametreleri kalibre edilecek.
> Şimdilik sadece PLAN — implementasyon Sprint 2 verisine bağlı.
