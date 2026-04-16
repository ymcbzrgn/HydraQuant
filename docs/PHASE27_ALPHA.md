# Phase 27: CAAT Asymmetric Alpha — Implementation Blueprint

> **"Kaybettiginde MINIMAL kaybet, kazandiginda BUYUK kazan"**
> 
> Bu dokuman Phase 27 ARGE arastirmasinin SONUCUDUR.
> 50 arastirma ajani (10 grup x 5, toplam ~800K token arastirma ciktisi) +
> 1 codebase explorer + canli sunucu auditi + BTC/ETH forensics +
> 9 mimari karar sentezlenerek yazilmistir.
> 
> **ARGE != ALPHA.** ARGE arastirir, ALPHA uygular.
> **Phase 26 = 3592 satir blueprint → 33 modul, 15K satir kod, deploy.**
> **Phase 27 = bu dokuman → Sprint 3A + Sprint 3B → kod.**

---

## CHANGELOG & DURUM TAKIBI

### CANLI SISTEM DURUMU (15 Nisan 2026, 11:40 GMT+3)

```
Bot: RUNNING (PID=727306, v2026.3-dev-6e284921b)
Sunucu: vm433344.ovadns.com, 32GB ECC RAM, 4 Core Platinum
Uptime: Surekli (systemd)

ALL-TIME PERFORMANS:
  1,289 kapanmis trade | 815 win, 474 loss | %63.2 WR
  Toplam PnL: +$424.24 | Ort PnL/trade: +$0.33
  En iyi trade: +$331.75 (XRP) | En kotu: -$252.28 (ETH)

SON 7 GUN PERFORMANS:
  123 trade | 74 win, 49 loss | %60.2 WR | +$874.92 PnL
  ROI exits: 53 trade, +$1,468.63 (ort +$27.71) — ISIN GUCU
  Trailing stops: 55 trade, -$568.66 (ort -$10.34) — KAYIP KAYNAGI
  ai_flip_bullish: 1 trade, +$44.72 — AI-driven exit CALISIYOR

ACIK POZISYONLAR (4):
  XRP/USDT:USDT  | stake=$5,126.26 | acik
  BNB/USDT:USDT  | stake=$6.30     | acik
  UNI/USDT:USDT  | stake=$0.31     | STUCK (orderbook neredeyse bos)
  WLFI/USDT:USDT | stake=$0.08     | STUCK (orderbook TAMAMEN bos)

CODEBASE:
  111 Python dosya | 42,094 toplam satir
  93 modul BAGLI (%91.1) | 18 modul DEAD CODE (%8.9)
  22 AI modul strategy'den dogrudan cagriliyor
  52 scheduler job aktif

VERITABANI:
  67 tablo (ai_data.sqlite)
  6,950 ai_decisions | 12,074 forgone_profit
  180 agent_memory | 1 catboost_training_run
  0 causal_discoveries | 0 dream_scenarios | 0 world_model_rollouts
  11 magma_edges (hepsi backtest/news, 0 debate edge)

CATBOOST v2 (14 Nisan 2026):
  7,940 sample | %62 test accuracy
  Top features: smc_fvg_count(%13), cdna_4_vol(%10.5), cdna_3_vol(%8.4)
  Class balance: 5157 BULLISH, 2571 BEARISH, 332 NEUTRAL (ciddi imbalance)
```

### EXIT REASON ANALiZi (Son 7 Gun)

| Exit Reason | Sayi | Toplam PnL | Ort PnL | Yorum |
|-------------|------|-----------|---------|-------|
| roi | 53 | +$1,468.63 | +$27.71 | Ana kar kaynagi |
| trailing_stop_loss | 55 | -$568.66 | -$10.34 | Ana kayip kaynagi |
| stale_flat | 6 | +$3.30 | +$0.55 | Zaman asimi, notr |
| exit_signal | 3 | -$25.70 | -$8.57 | AI sinyal cikisi |
| ai_flip_bullish | 1 | +$44.72 | +$44.72 | AI yon degisimi — cok basarili |
| stop_loss | 1 | -$47.36 | -$47.36 | Hard stop |

**ASIMETRI SORUNU CANLIDA DOGRULANDI:** Trailing stop'lar $568 kaybettirirken ROI 
exit'ler $1468 kazandiriyor. Kazanc/kayip orani 2.58x — iyilesiyor ama hedefe (8x+) uzak.

### PAIR BAZLI PnL (Son 7 Gun, Top/Bottom)

| Pair | Trade | Win | PnL | Ort Stake | Yorum |
|------|-------|-----|-----|-----------|-------|
| ZK/USDT | 4 | 4 | +$653.99 | $308 | EN IYI — %100 WR |
| XRP/USDT | 2 | 2 | +$336.13 | $239 | Devasa pozisyon, devasa kar |
| 1000PEPE | 11 | 8 | +$248.33 | $180 | Tutarli performer |
| BNB/USDT | 2 | 2 | +$149.58 | $113 | Yuksek accuracy |
| ... | ... | ... | ... | ... | ... |
| DOGE/USDT | - | - | -$25.90 | - | Kayip |
| ADA/USDT | - | - | -$36.69 | - | Kayip |

### BTC/ETH KAYIP FORENSICS

```
BTC/USDT:USDT — TOPYEKUN:
  30 trade | 17 win (%57) | Toplam: -$329.46
  Ort stake: $303.86 | Max stake: $1,000
  En kotu: -$214.57 (14 Nis, trailing_stop_loss, 11 saat acik)
  Global Kelly: 0.835 | OLMASI GEREKEN: max 0.25 (vol drag = 0.245)

ETH/USDT:USDT — TOPYEKUN:
  46 trade | 36 win (%78!) | Toplam: -$699.86
  Ort stake: $138.44 | Max stake: $1,539.55
  En kotu: -$252.28 (3 Nis, trailing_stop_loss, 4 saat acik)
  Global Kelly: 0.835 | OLMASI GEREKEN: max 0.18 (vol drag = 0.405)

PARADOKS: ETH %78 win rate ile -$700 kaybetti.
Kazandiginda: max +$14.92 (kucuk pozisyon, erken ROI cikis)
Kaybettiginde: -$252.28 (buyuk pozisyon, gec trailing stop)

NEDEN SISTEM OGRENEMEDI:
  self_model_profile: 0 satir (TAMAMEN BOS — modul hic calismamis)
  agent_memory BTC: 5 kayit (30 trade'den sadece 5!)
  agent_memory ETH: 0 kayit (46 trade'den SIFIR!)
  bayesian_kelly: TEK SATIR (id=1, CHECK id=1) — per-pair YOK
  causal_discoveries: 0 satir — nedensellik motoru calismamis
```

### FORGONE ALPHA ANALiZi

```
Toplam forgone kayit: 12,074
Execute edilen sinyaller: 2,868 → +$1,726.83 toplam PnL
Execute EDILMEYEN sinyaller: 9,206 → +$1,247.66 toplam forgone PnL

Bot $1,247 kazandiracak trade'i ACMADI.

Top forgone pairs (kacirilmis kar):
  ZK/USDT:    $720.55 (sadece 5 sinyal!)
  VIRTUAL:    $358.90
  AERO:       $263.49
  LDO:        $204.33
  SUI:        $127.84
  TWT:        $123.07

Root cause: OOD detector distance=500 → defensive_mult=0.17 → tum sizing %17'de
+ dusuk confidence threshold → cogu sinyal execute edilmedi
+ forgone feedback loop YOK → sistem kendi hatasindan ders almiyor
```

### OOD DETECTOR DURUMU

```
Her pair, her saniye, ayni cikti:
  [OOD] DETECTED! distance=500.01 > threshold=2.80 (p=0.0000) → defensive_mult=0.17

ROOT CAUSE (D4 ajani):
  1. Mahalanobis distance Gaussian varsayimi → crypto NON-GAUSSIAN (kurtosis 9-18)
  2. High-dimensional distance concentration → d=20 feature'da tum noktalar uzak
  3. 1e-6 covariance regularization yetersiz → ill-conditioned precision matrix
  4. Chi-squared threshold Gaussian data icin kalibre → her sey OOD

COZUM: Wasserstein W1 drop-in replacement (Task 2'de detayli)
```

### HORMON SISTEM DURUMU

```
Saatlik lifecycle tick, son 20 kayit:
  cortisol: 0.0 (statik, HICBIR ZAMAN degismiyor)
  dopamine: 1.0
  serotonin: 1.1
  adrenaline: 0.3

hormone_history: sadece 20 kayit (20 saat = 20 tick)
hormone_state: tek satir (id=1), son guncelleme 08:18

SORUN: Cortisol hep 0.0 — stres HISSEDILMIYOR.
Allostasis EWMA calisiyor ama etki minimumal.
Amygdala fear hysteresis VAR ama cortisol'e BAGLI DEGIL.
fear sizing_mult hesaplaniyor ama KULLANILMIYOR.
```

---

## PRENSiP 0: HER PAIR BIR BIREY — DEGISTIRILEMEZ MIMARI PRENSIBI

> **"Global hafiza REFERANS, per-pair hafiza KARAR verir."**

Bu Phase 27'nin tek MUTLAK prensibidir. 50 arastirma ajaninin en tutarli
bulgusi: her pair'in farkli volatilitesi, farkli likiditesi, farkli causal yapisi,
farkli momentum karakteristigi var. Tek global parametre ile yonetmek
Simpson's Paradox'a dusmektir — ve BTC/ETH kaybimiz bunun KANITI.

### Ergodicity Economics Kaniti (A1 Ajani)

Peters (2019, Nature Physics): Tum finans sektoru EXPECTED VALUE kullaniyor.
AMA expected value sadece SONSUZ PARALEL EVRENDE gecerli. 
Biz TEK EVRENDE yasiyoruz. Dogru metrik: TIME-AVERAGE GROWTH RATE.

```
Expected value:  E[wealth] = average across parallel universes  (YANLIS)
Time average:    <wealth>  = average across TIME for ONE person  (DOGRU)

Time-average growth rate: g = mu - sigma^2/2

BTC:  mu=0.35, sigma=0.70 → drag=0.245 → g_max at Kelly ~0.25
ETH:  mu=0.30, sigma=0.90 → drag=0.405 → g_max at Kelly ~0.18
SOL:  mu=0.40, sigma=1.20 → drag=0.720 → g_max at Kelly ~0.10
DOGE: mu=0.25, sigma=1.50 → drag=1.125 → g_max at Kelly ~0.05

Global Kelly = 0.835 → HER ASSET ICIN 3-15x ASIM
                      → GEOMETRIK SERVET YOKEDILIYOR
```

**Kaynak:** Peters (2019) "The ergodicity problem in economics" Nature Physics 15:1216.
Cover & Thomas (1991) "Elements of Information Theory" Chapter 6.
Stiffelman (2026) "Investing is Compression" arXiv 2604.10758.

### Per-Pair Zorunluluk Tablosu

| Modul | Simdi | Phase 27 | Neden |
|-------|-------|----------|-------|
| Bayesian Kelly | Tek satir `CHECK(id=1)` | `UNIQUE(pair, regime)` | Her pair'in sigma'si farkli → optimal f farkli |
| self_model | 0 satir | Per-pair x regime competence | "ETH'de kotu oldugunumu bilmem lazim" |
| Agent Memory | ETH: 0 kayit | Her trade icin her agent | Agent'lar pair-spesifik ogrensin |
| OOD Detector | Tek global Mahalanobis | Per-cluster Wasserstein | Her pair'in dagilimi farkli |
| Confidence Threshold | Tek global 0.50 | Per-pair adaptive | Forgone feedback ile otomatik ayar |
| Hormone/Stress | Global tek cortisol | Per-pair stress | BTC stresliyken ZK sakin olabilir |
| Cerebellum | Global saat multiplier | Per-pair x saat | ETH gece kotu, BTC gece iyi olabilir |
| Trailing Stop | Sabit ATR mult | Per-pair regime-adaptive | Trending pair'de genis, ranging'de siki |
| Market Impact | Yok | Per-pair tier + depth cap | BTC impact 0, WLFI impact 52bps |

### Matematiksel Formul: Per-Pair 7-Step Sizing Pipeline (E1 Ajani)

Her trade icin, her pair `i` icin:

```python
def per_pair_kelly_size(pair: str, regime: str, portfolio_value: float) -> float:
    """
    E1 Ajani: 7-step per-pair sizing pipeline.
    Peters (2019) ergodicity + Baker-McHale (2013) uncertainty +
    Meucci ENB correlation correction.
    """
    # ─── STEP 1: Per-pair Beta posterior ───
    # Her pair x regime icin ayri alpha/beta
    alpha_i, beta_i = load_pair_kelly(pair, regime)  # DB: UNIQUE(pair, regime)
    p_i = alpha_i / (alpha_i + beta_i)  # win probability
    q_i = 1.0 - p_i
    
    # ─── STEP 2: Raw Kelly ───
    avg_win_i, avg_loss_i = load_pair_win_loss(pair, regime)
    b_i = abs(avg_win_i / avg_loss_i) if avg_loss_i != 0 else 1.0
    f_raw = (b_i * p_i - q_i) / b_i  # standard Kelly
    f_raw = max(0.0, f_raw)  # negatif Kelly = trade etme
    
    # ─── STEP 3: Volatility drag correction (Peters) ───
    # g = mu - sigma^2/2 → optimal leverage icin sigma^2 cikartilmali
    sigma_i = get_pair_annual_volatility(pair)  # chart_features'dan
    drag = (sigma_i ** 2) / 2.0
    f_drag = f_raw - drag / (b_i + 1e-8)
    f_drag = max(0.0, f_drag)
    
    # ─── STEP 4: Vol-of-vol shrinkage ───
    # Volatilite kendisi de belirsiz → effective variance buyur
    var_sigma = get_pair_vol_of_vol(pair)  # rolling std of rolling vol
    if var_sigma > 0:
        vov_shrink = sigma_i**2 / (sigma_i**2 + var_sigma)
    else:
        vov_shrink = 1.0
    f_vov = f_drag * vov_shrink
    
    # ─── STEP 5: Baker-McHale parameter uncertainty ───
    # N trade az → sigma_p buyuk → Kelly asiri tahmin
    N = alpha_i + beta_i - 2  # total trades (prior cikarilmis)
    if N > 0:
        sigma_p = math.sqrt(p_i * q_i / N)  # std error of win rate
        bm_shrink = max(0.1, 1.0 - (sigma_p**2) / (p_i * q_i + 1e-8))
    else:
        bm_shrink = 0.1  # sifir trade → minimum sizing
    f_bm = f_vov * bm_shrink
    
    # ─── STEP 6: Trade count graduation ───
    # "Hakkini kazan" — az trade ile buyuk sizing yok
    if N < 30:
        grad = 0.125  # 1/8 Kelly
    elif N < 100:
        grad = 0.25   # 1/4 Kelly
    elif N < 300:
        grad = 0.50   # 1/2 Kelly
    else:
        grad = 0.75   # 3/4 Kelly (ASLA full Kelly)
    f_grad = f_bm * grad
    
    # ─── STEP 7: Portfolio constraints ───
    # Meucci ENB: korelasyonlu pair'ler icin toplam riski sinirla
    enb = compute_effective_number_of_bets()  # eigenvalue-based
    n_pairs = count_active_pairs()
    portfolio_scale = min(1.0, enb / n_pairs)
    f_final = f_grad * portfolio_scale
    
    # Hard caps
    f_final = min(f_final, 0.03)  # Constitution: max %3 per position
    
    # Dollar amount
    dollar_size = f_final * portfolio_value
    
    # Impact constraint (D2 ajani)
    dollar_size = apply_impact_constraint(pair, dollar_size)
    
    return dollar_size
```

**Tablo: Pipeline Her Adimda Ne Yapar (BTC Ornegi)**

| Step | Islem | BTC Ornegi | ETH Ornegi |
|------|-------|-----------|-----------|
| 1 | Beta posterior | p=0.57 (17W/13L) | p=0.78 (36W/10L) |
| 2 | Raw Kelly | f=0.071 | f=0.142 |
| 3 | Vol drag | f=0.071-0.035=0.036 | f=0.142-0.058=0.084 |
| 4 | Vol-of-vol | f=0.036×0.85=0.031 | f=0.084×0.75=0.063 |
| 5 | Baker-McHale | f=0.031×0.82=0.025 | f=0.063×0.78=0.049 |
| 6 | Graduation (N<100) | f=0.025×0.25=0.006 | f=0.049×0.25=0.012 |
| 7 | Portfolio ENB | f=0.006×0.8=0.005 | f=0.012×0.8=0.010 |
| **Sonuc** | **$10K portfolio** | **$50** | **$100** |
| Mevcut | **Global 0.835** | **$303** (6x asim!) | **$138** (14x asim!) |

### DB Schema Degisikligi

```sql
-- position_sizer.py'deki mevcut (YANLIS):
CREATE TABLE IF NOT EXISTS bayesian_kelly (
    id INTEGER PRIMARY KEY CHECK (id = 1),  -- TEK SATIR!
    alpha REAL DEFAULT 1.0,
    beta_param REAL DEFAULT 1.0,
    avg_win REAL DEFAULT 0.0,
    avg_loss REAL DEFAULT 0.0,
    total_trades INTEGER DEFAULT 0,
    updated_at TEXT
);

-- Phase 27 (DOGRU — db.py'de zaten VAR ama kullanilmiyor):
CREATE TABLE IF NOT EXISTS bayesian_kelly_per_pair (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    pair TEXT NOT NULL,
    regime TEXT NOT NULL DEFAULT '_global',
    alpha REAL DEFAULT 2.0,        -- informative prior (> 1.0)
    beta_param REAL DEFAULT 2.0,   -- informative prior
    avg_win REAL DEFAULT 0.0,
    avg_loss REAL DEFAULT 0.0,
    n_trades INTEGER DEFAULT 0,
    annual_volatility REAL,         -- sigma for vol drag
    vol_of_vol REAL,               -- var(sigma) for step 4
    last_sharpe REAL,              -- pair-specific Sharpe
    updated_at TEXT,
    UNIQUE(pair, regime)
);
CREATE INDEX idx_bk_pair ON bayesian_kelly_per_pair(pair, regime);
```

---

## TEORIK TEMELLER: 50 AJAN SENTEZI — FIZIK + MATEMATIK + BIYOLOJI

> Bu bolum ARGE arastirmasinin akademik temellerini icerir.
> Her alt-bolum bir ajan grubunun anahtar bulgularini ozetler.
> Implementasyon detaylari SPRINT bolumlerindedir.

### A2: Rough Volatility — "Volatilite Purussuz Degil, KABA"

**Gatheral et al. (2018):** Volatilite Brownian motion DEGIL, fractional Brownian motion
izliyor. Hurst exponent H ≈ 0.1 (cok kaba, cok puruzlu). Mevcut tum risk modelleri H=0.5 varsayar.

```
Standart varsayim:  H = 0.5 (smooth Brownian) → VaR, opsion fiyatlama, risk modeli
Gercek:             H ≈ 0.1 (rough) → Her sey degisiyor:
  - VaR %40 UNDERESTIMATE ediyor (fat tails daha kalin)
  - Volatility clustering daha agresif
  - Mean reversion daha hizli
  - Opsion fiyatlama tamamen yanlis

Rough Fractional Stochastic Volatility (rFSV):
  dV(t) = κ(θ - V(t))dt + ξ V(t)^γ dW^H(t)
  W^H = fractional Brownian motion with H ≈ 0.1
```

**Sprint 2 baglantisi:** `chart_features.py` zaten hurst_50, hurst_100 hesapliyor. Phase 27'de
bunu rough volatility modeline bagla → daha dogru uncertainty tahminleri.

**Python:** `rough_bergomi` (pip), `rbergomi` GitHub. ~50MB RAM, CPU feasible.
**Kaynak:** Gatheral, Jaisson & Rosenbaum (2018) "Volatility is rough" Quantitative Finance.

### A3: Levy Processes — "Normal Dagilim YALAN"

**7 Jump-Diffusion Model (en pratigindan en teorik olana):**

| Model | Formul | Kullanim | Complexity |
|-------|--------|----------|-----------|
| Merton JD | `dS = μdt + σdW + JdN` (Poisson jumps) | Flash crash modelleme | LOW |
| Kou DEJD | Double-exponential jump sizes | Asimetrik jump (dusus > yukselis) | LOW |
| Merton-Hawkes | Merton + self-exciting jumps | Cascade prediction | MEDIUM |
| Variance Gamma | Brownian motion with gamma time change | Fat tail + skew | MEDIUM |
| NIG | Normal Inverse Gaussian | Flexible tail behavior | MEDIUM |
| CTS/CGMY | Tempered stable | En genis tail kontrol | HIGH |
| GTS | Generalized tempered stable | Teorik en genel | HIGH |

```
Gaussian VaR (95%): BTC icin ~$3,200 kayip tahmini
Levy VaR (95%):     BTC icin ~$4,500 kayip tahmini (%40 DAHA BUYUK)

Neden: Gaussian 6-sigma event'i "imkansiz" der.
       Levy "nadir ama beklenen" der.
       Crypto'da 6-sigma event'ler AYDA BIR olur.
```

**Sprint 3A baglantisi:** World model (8C) Gaussian noise kullaniyor. Levy noise ile:
- Flash crash'leri dogal olarak modelle
- Fat tail risk'i dogru hesapla → daha gercekci dream trajectories

**Python:** `levy_stable` (scipy.stats), `tempered_stable` (custom). CPU feasible.

### A4: Information Geometry — "Piyasa Bir MANIFOLD"

**Amari (1998):** Olasilik dagilimlari bir manifold olusturur.
Fisher Information Matrix = bu manifold'un metrigi.

```
Piyasa state'i = manifold uzerinde bir nokta (dagilim parametreleri)
Regime degisimi = manifold uzerinde hareket
Geodesic distance = iki regime arasindaki EN KISA yol
Natural gradient = manifold'un egriligini hesaba katan gradient

Pratik uygulama:
  1. Natural gradient descent → RL training 2-5x hizlanir
  2. Geodesic distance → "ne kadar farkli bir piyasadayim?" METRIK
  3. Manifold curvature → "piyasa ne kadar hizli degisiyor?"
```

**Sprint 2 baglantisi:** EWC (ewc_continual.py) zaten Fisher Information kullaniyor.
Phase 27'de natural gradient eklenir → daha hizli, daha stabil ogrenme.

**Python:** `geomstats` (MIT, GPU+CPU), `jax` backend. ~100MB RAM.

### A5: Topological Data Analysis — "Piyasanin SEKLINI Gor"

**Persistent Homology:** Veriyi nokta bulutu olarak al, topolojik yapisi cikar.
Delikler, donguler, baglantili bilesenler — fiyat grafiklerinde GIZLI YAPILAR.

```
Price time series → Takens embedding (delay=tau, dim=d)
  → Point cloud in R^d
    → Vietoris-Rips complex (epsilon filtration)
      → Persistent diagram (birth-death pairs)
        → Betti numbers: b0 (components), b1 (loops), b2 (voids)

Regime signature:
  Bull market: b1 yuksek (trendy loops), b0 dusuk (connected)
  Bear market: b0 yuksek (fragmented), b1 dusuk
  Crash: topology COKUYOR — Betti numbers ani degisim
```

**3-Tier Entegrasyon:**
1. **Tier 1 (Feature):** Persistence landscape → CatBoost feature (5 ek feature)
2. **Tier 2 (Regime):** Topological regime classifier (TDA + SVM)
3. **Tier 3 (Crash):** Betti number ani degisim → early warning signal

**Python:** `giotto-tda` (L2F Open Source), `ripser` (MIT). CPU feasible, ~200MB RAM.
**Kaynak:** Gidea & Katz (2018) "Topological Data Analysis of Financial Time Series"

### B4 Detay: EFE = Kelly Matematik Ispati

**6 paper'in sentezi (bu sentez YAYINLANMAMIS — genuinely novel):**

```
TEOREM (informal): EFE'yi log-wealth preference ile minimize etmek
Kelly growth rate + epistemic bonus'u maximize etmeye ESDEGER.

Ispat ozeti:
1. G(a) = -Pragmatic(a) - Epistemic(a)              [EFE decomposition]
2. p*(o') = ln(wealth after outcome o')              [log-wealth preference]
3. Pragmatic(a) = E[ln W|a] = Kelly objective        [tanim geregi]
4. Epistemic(a) = I(s'; o'|a) = mutual information   [EFE formulu]
5. Cover & Thomas: I(X;Y) = doubling rate artisi     [Ch.6 teoremi]
6. Dolayisiyla: -G(a) = E[ln W|a] + I(s';o'|a)
             = Kelly growth + information gain
             = "ne kadar para" + "ne kadar ogrenirim"

SONUC: Optimal Kelly fraction EFE altinda:
  f*_EFE = f*_Kelly × (1 + ε_epistemic)
  ε > 0 belirsizlik yuksekken (explore)
  ε → 0 inanclar yakinsadikca (exploit)
```

**Passivity Paradox (Agentic Finance, MDPI Entropy 2026):**
- Professional Agent (frozen beliefs): Sharpe 0.39
- Adaptive Agent (updating beliefs): Sharpe -0.28
- **YAVAS guncelleme > HIZLI guncelleme** non-stationary piyasalarda
- Mevcut Beta(alpha,beta) yavas update DOGRU davranis

**3-Tier Implementation:**
- **Tier 1 (10 satir):** EFE epistemic ratio → sizing_mult'a carpan olarak ekle
- **Tier 2 (80 satir):** EFEKellyBridge sinifi, precision mapping, surprise EMA
- **Tier 3 (300 satir):** Full POMDP pymdp v1.0 (JAX-first, 4 state x 8 obs x 4 action)

### C1: Neural ODE/SDE — "Piyasa Surekli Zamanda Yasiyor"

```
Mevcut world model: Discrete GRU (1h bar'lar arasinda atlama)
Neural ODE:         dz/dt = f_θ(z, t) — surekli zaman dinamikleri
Neural SDE:         dz = f_θ(z,t)dt + g_θ(z,t)dW — stochastic + uncertainty DOGAL

Avantajlar:
  1. Compounding error azalir (discrete step hatalari birikmez)
  2. Uncertainty quantification TABIATINDA var (SDE'nin diffusion terimi)
  3. Irregular time series handle eder (24/7 crypto, bazi pair'ler yuksek bazi dusuk frekans)

Pratik: torchdiffeq (Chen et al. NeurIPS 2018)
  import torchdiffeq
  z_T = torchdiffeq.odeint(func, z_0, t_span)  # ODE
  z_T = torchsde.sdeint(f, g, z_0, t_span)     # SDE

RAM: ~100-200MB (world model boyutunda). CPU: 5-10x slower than GRU.
Tavsiye: World model v2 olarak Sprint 3B'de, GRU → Neural SDE gecisi.
```

### E2: Adaptive Conformal Prediction 2.0

**Mevcut:** `conformal_calibrator.py` CQR + ACI, target_coverage=0.95, 6h recalibration.

**Yeni (2025-2026 literaturu):**

| Yontem | Avantaj | Mevcut CQR'a Gore |
|--------|---------|-------------------|
| **ACI (Gibbs 2021)** | Non-stationary'ye adapte | Zaten VAR ama daha agresif alpha update gerekli |
| **EnbPI (Xu & Xie 2021)** | Ensemble + bootstrap PI | Deep ensemble (5 model) ile DOGAL uyum |
| **CF-GNN (Zargarbashi 2023)** | Graph-based conformal | MAGMA graph ile entegre edilebilir |
| **Conformal Risk Control** | "P(kayip > X) < alpha" GARANTI | Risk budget'a dogrudan baglanir |
| **Multi-Output CP** | Fiyat + volume + vol ayni anda | Triple/Quad perception icin ideal |

**En yuksek etkili degisiklik:**
```python
# Conformal Risk Control — "bu trade'in %95 ihtimalle kaybi en fazla X$"
def conformal_risk_bound(predictions, residuals, alpha=0.05):
    """
    Distribution-free risk bound.
    Calibration set residuals'dan quantile hesapla.
    """
    n = len(residuals)
    sorted_residuals = np.sort(np.abs(residuals))
    q_level = math.ceil((1 - alpha) * (n + 1)) / n
    risk_bound = np.quantile(sorted_residuals, min(q_level, 1.0))
    return risk_bound  # "$X'den fazla kayip olmayacak" %95 garanti
```

### E4: Risk Parity — HRP (Lopez de Prado)

**Sorun:** Mevcut sizing per-signal (Kelly × confidence). Portfolio-level risk allocation YOK.
BTC ve ETH toplam riskin %80'ini olusturuyor cunku stake'leri buyuk.

**Hierarchical Risk Parity (HRP):**
```
1. Korelasyon matrisi hesapla (30+ pair)
2. Hierarchical clustering (single-linkage)
3. Quasi-diagonalization (seriation)
4. Recursive bisection ile weight allocation
5. Her cluster icinde equal risk contribution

Sonuc: Hicbir pair veya cluster toplam riskin X%'inden fazlasini tasimaz.
```

**ENB (Effective Number of Bets):**
```python
def effective_number_of_bets(correlation_matrix):
    """Meucci: gercek diversifikasyon olcumu."""
    eigenvalues = np.linalg.eigvalsh(correlation_matrix)
    eigenvalues = eigenvalues[eigenvalues > 0]
    p = eigenvalues / eigenvalues.sum()
    enb = np.exp(-np.sum(p * np.log(p + 1e-10)))
    return enb
    # 30 pair ama 0.85 korelasyon → ENB ≈ 3-5 (30 degil!)
    # Toplam Kelly allocation ENB/30 ile olceklenmeli
```

**Python:** `riskfolio-lib` (BSD, Lopez de Prado'nun HRP implementasyonu), `PyPortfolioOpt`.

### F1: Self-Organized Criticality — "Kaosun Kiyisinda"

**Per Bak (1996):** Piyasalar kum tepesi gibi — taneler (trade'ler) birikerek
kritik noktaya ulasir, sonra cig (crash) duser. Cig boyutlari power-law dagilir.

**10 Criticality Detection Metric (F1 ajani, oncelik sirasinda):**

| # | Metrik | Hesaplama | Lead Time | Complexity |
|---|--------|-----------|-----------|-----------|
| 1 | **Rolling variance** | `var(returns, window=100)` | 1-4h | LOW |
| 2 | Autocorrelation at lag-1 | `np.corrcoef(r[:-1], r[1:])` | 2-8h | LOW |
| 3 | Hawkes branching ratio | `n = alpha/beta` (D1) | 1-3h | MEDIUM |
| 4 | Return distribution kurtosis | `scipy.stats.kurtosis(returns)` | 2-6h | LOW |
| 5 | Power-law tail exponent | `powerlaw.Fit(returns).alpha` | 4-12h | MEDIUM |
| 6 | Entropy rate (Shannon) | `-sum(p * log(p))` on return bins | 2-6h | LOW |
| 7 | Detrended Fluctuation Analysis | `nolds.dfa(returns)` | 4-12h | MEDIUM |
| 8 | Multifractal spectrum width | `MFDFA` package | 6-24h | HIGH |
| 9 | Network topology flickering | Correlation network instability | 6-24h | HIGH |
| 10 | LPPLS (Sornette dragon king) | Log-periodic power law | 1-7 days | HIGH |

**6 Homeostatic Mechanism (organizma edge-of-chaos'ta kalmak icin):**
1. Criticality Thermometer: rolling variance + kurtosis → single score
2. Hormonal Gain Control: cortisol criticality'de artar → sizing otomatik kuculsun
3. Pheromone Evaporation Rate: high criticality → faster evaporation → faster adaptation
4. Module Diversity Index: Shannon entropy of organ weights → too low = fragile
5. Avalanche Size Monitoring: trade cluster size distribution → power-law mi?
6. Scheduler Correlation Matrix: eger tum modüller ayni sinyal veriyorsa → herding TEHLIKESI

### F3: Autopoiesis — Kimlik Koruma Cercevesi

**8 Teorik Framework Sentezi:**
1. Maturana-Varela: Organization (pattern) vs Structure (components) ayrimi
2. Luhmann: Operational closure — binary code = trade/no-trade ayrimi yetisi
3. Rosen (M,R): Closure to efficient causation — repair mechanism kendini repair edebilmeli
4. Pattee: Semantic closure — genotype (params) builds phenotype (behavior) reads genotype
5. Beer VSM: S5 (identity) = self_model → MISSING
6. Friston FEP: Self-evidencing — organizma kendi varligini maximize eder
7. Minary (2026): Computational autopoiesis convergence proof
8. Piaget: Assimilation (parametreleri guncelle) vs Accommodation (yapiyi degistir) dengesi

**Autopoietic Integrity Index (AII):**
```python
def compute_aii(organism, birth_snapshot):
    """4 katmanli kimlik olcumu."""
    # Layer A: Structural (essential organ'lar ve connection'lar korundu mu?)
    structural = 1 - graph_edit_distance(organism.genome, birth_snapshot.genome) / max_distance
    
    # Layer B: Functional (ATCB benchmark pass rate — organizma ISINI yapabiliyor mu?)
    functional = atcb_benchmark.run_quick_test(organism)
    
    # Layer C: Behavioral (trade pattern'lari tutarli mi? KL divergence)
    behavioral = 1 - kl_divergence(recent_behavior, baseline_behavior)
    
    # Layer D: Representational (parametre dagilimi benzer mi? CKA + Fisher-weighted)
    representational = centered_kernel_alignment(current_activations, birth_activations)
    
    aii = 0.30 * structural + 0.35 * functional + 0.20 * behavioral + 0.15 * representational
    
    # Threshold'lar:
    # AII > 0.8: GREEN — evrim normal devam
    # 0.6-0.8: YELLOW — mutation_rate yarilat, introspection artir
    # 0.4-0.6: RED — evrim durdur, self-repair baslat
    # < 0.4: CRITICAL — son GREEN snapshot'a rollback
    
    return aii
```

**Constitution Eki (identity_limits):**
```python
"identity_limits": {
    "essential_organs": ["crowd_scoring", "risk", "sizing"],  # ASLA kapatilamaz
    "essential_connections": [
        {"source": "risk", "target": "sizing", "type": "inhibitory"},  # Risk DAIMA sizing'i frenler
    ],
    "max_structural_drift": 0.4,
    "min_functional_score": 0.6,
    "min_aii": 0.5,
    "max_mutations_per_generation": 3,
}
```

### F4: NEAT Detay — Evrimsel Mimari Arama

**Mevcut architecture_evolver.py sorunlari (F4 ajani):**
1. Speciation YOK — tum genome'lar ayni havuzda, innovation eziliyor
2. Innovation protection YOK — yeni organ eklenmesi aninda fitnesssiz, elenecek
3. Bloat control YOK — organ sayisi sureksiz artabilir
4. Regime memory YOK — gecmis regime'lerin en iyi genome'u hatirlanmiyor
5. Fitness fonksiyonu gercek trade PnL'e BAGLI DEGIL — hardcoded heuristic

**4 NEAT Mekanizmasi Eklenecek:**

```python
# 1. SPECIATION (Stanley 2002):
delta(i,j) = c1*E/N + c2*D/N + c3*W_bar  # compatibility distance
# delta < threshold → same species → compete WITHIN species
# Target: 4-6 species (≈ regime count)

# 2. INNOVATION PROTECTION (Youth Bonus):
if genome.age < 5:  # ilk 5 generation
    fitness *= 1.3   # %30 bonus — yenilige sans ver

# 3. PHASED SEARCH (Colin Green/SharpNEAT):
if avg_complexity > threshold_high:
    mode = "SIMPLIFY"  # sadece organ cikarma mutation'lari
elif avg_complexity < threshold_low:
    mode = "COMPLEXIFY"  # organ ekleme mutation'lari
# ablation_league.py = functional bloat detector (zaten mevcut!)

# 4. REGIME GENOME ARCHIVE (MAP-Elites + Yang 2008):
archive = {
    ("trending_bull", "aggressive"): best_genome_for_this_combo,
    ("ranging", "conservative"): best_genome_for_this_combo,
    ...
}
# Regime degistiginde: archive'dan recall + %30-50 seed + mutation
```

### G2 Detay: LLM Strategy Researcher — Hypothesis Loop

**FINSABER Uyarisi (KDD 2026):** LLM sinyal uretmede 20 yillik testte buy-and-hold'u
yenemedi (p > 0.34). LLM'in rolu: SINYAL degil PARAMETRE optimizasyonu.

**6-Gate Validation Pipeline (her hipotez icin):**
```
Hipotez uretildi
  |→ [Gate 1] Complexity check: max 1 parametre degisikligi
  |→ [Gate 2] Novelty check: daha once test edilmis mi?
  |→ [Gate 3] Walk-forward backtest: 5+ rolling fold
  |→ [Gate 4] Deflated Sharpe Ratio: n_hypotheses correction
  |→ [Gate 5] Out-of-sample holdout: son %20 data
  |→ [Gate 6] Shadow trade: 1 hafta paper trade
  |→ Deploy + monitoring + auto-rollback
```

**Prompt Template (G2 ajani):**
```
SYSTEM: You are a quantitative strategy researcher.
Generate SPECIFIC, TESTABLE parameter adjustment hypotheses.
Do NOT suggest new signals — only adjust existing parameters.

CONSTRAINTS:
- Each hypothesis changes EXACTLY ONE parameter
- Must state MECHANISM (why this helps)
- Must state FALSIFICATION CONDITION (what disproves this)

INPUT:
{post_trade_court_verdicts}
{ablation_league_table}
{forgone_pnl_summary}
{current_parameters}

OUTPUT: JSON with id, parameter, current_value, proposed_value,
        mechanism, expected_impact, falsification, confidence
```

**hypothesis_history tablosu:**
```sql
CREATE TABLE hypothesis_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    hypothesis_id TEXT UNIQUE,
    parameter TEXT, current_value REAL, proposed_value REAL,
    mechanism TEXT, falsification TEXT,
    is_sharpe REAL, oos_sharpe REAL,
    deflated_sharpe REAL, n_hypotheses_in_batch INTEGER,
    deployed BOOLEAN DEFAULT FALSE,
    rolled_back BOOLEAN DEFAULT FALSE,
    created_at TEXT
);
```

### G3 Detay: Neurosymbolic — 7 Entegrasyon Mimarisi

**Mevcut durum: Kautz Type 3 (en gevsek)** — neural output → symbolic input, tek yon.

| # | Mimari | Complexity | Oncelik | Aciklama |
|---|--------|-----------|---------|----------|
| 1 | SHAP Narrative Injection | LOW | **P1** | Fix 4'te YAPILDI |
| 2 | Concept Bottleneck Bridge | MEDIUM | P2 | 8-12 symbolic concept (trend, vol, momentum...) |
| 3 | LLM-Guided Feature Selection | MEDIUM | P3 | LLM haftalik feature weight onerisi → CatBoost |
| 4 | Symbolic Rule Extraction | MED-HIGH | P4 | CatBoost decision path → IF-THEN rules |
| 5 | System 1/2 Arbitration | MEDIUM | P5 | Fast (neural) vs Slow (symbolic) ne zaman? |
| 6 | Bidirectional Verification | HIGH | P6 | Neural output → symbolic verify → neural learns |
| 7 | ILP Pattern Discovery | HIGH | P7 | Popper/Metagol: otomatik rule kesfetme |

**Concept Bottleneck Bridge (P2):**
```python
CONCEPTS = {
    "trend_direction": ["strong_up", "up", "neutral", "down", "strong_down"],
    "volatility_regime": ["compression", "normal", "expansion", "extreme"],
    "momentum_quality": ["clean", "choppy", "divergent"],
    "crowd_sentiment": ["euphoria", "greed", "neutral", "fear", "panic"],
    "support_proximity": ["at_support", "near", "far", "at_resistance"],
    "volume_conviction": ["confirming", "neutral", "diverging"],
    "timeframe_alignment": ["aligned", "mixed", "conflicting"],
    "pattern_maturity": ["forming", "mature", "exhausted"],
}

# CatBoost → concept mapping
# Evidence Engine → ayni concept'lere mapping
# CROSS-SYSTEM karsilastirma:
#   CatBoost: vol_regime=compression(0.85)
#   EE: q6_risk=low_vol(0.72)
#   → ALIGNED (guven artir)
#   vs
#   CatBoost: trend=strong_up(0.91)
#   EE: q1_trend=neutral(0.48)
#   → CONFLICT (guven azalt)
```

### G4 Detay: Multi-Modal Fusion — 7 Bug Fix + Mimari

**multimodal_encoder.py 7 Kritik Bug:**
1. UNTRAINED weights (random init, torch.no_grad) → training pipeline ekle
2. FAKE text embedding (hash-based random vector) → gercek Jina embedding
3. ZERO vector missing modality → learnable `nn.Parameter(torch.randn(dim))`
4. Mean pooling (esit agirlik) → MSGCA price-guided asymmetric attention
5. LOB not connected → lob_encoder output'u 6. modality olarak ekle
6. Never imported → trinity_fusion → strategy bagla
7. No staleness guard → timestamp check (trinity_fusion pattern)

**En pratik fusion mimarileri:**
- **GMU (Cold start):** 2 modality → gate → output. ~49K param, <0.2ms. Hemen kullanilabilir.
- **MSGCA (Mature):** Price GUIDES fusion. 8+ paper'da en iyi sonuc.
- **AECF (Missing modality):** Entropy-gated contrastive. +18pp mAP at %50 modality drop.

**CPU Fizibility:**
```
5 modality token × 64 dim cross-attention: ~410K FLOP = <1ms
Darboğaz fusion DEĞİL, upstream encoder'lar:
  TTM: ~100ms, CryptoBERT: ~50-200ms (API), GNN: ~10-50ms
```

### G5 Detay: Adaptive LLM Router — 12 Enhancement

**Combined free ceiling: 4,000-5,000+ RPD/gun**

| Provider | Model | RPM | RPD | Optimal Kullanim |
|----------|-------|-----|-----|-----------------|
| Gemini Flash | 2.5 Flash | 10 | 250 | Complex reasoning, trade decisions |
| Gemini Flash-Lite | 2.5 Lite | 15 | 1,000 | Bulk/background tasks |
| Groq Llama 70B | 3.3-70b | 30 | 14,400 | Medium analysis |
| Groq Llama 8B | 3.1-8b | 30 | 14,400 | Simple classification |
| Cerebras Qwen 235B | qwen-3 | ~30 | ~1,700 | Large context |
| Mistral Large | large | 60 | ~86K | Generous, backup |

**Top 3 Enhancement (P0-P2):**
```python
# P0: RPD Tracking (HEMEN — en yuksek etki)
class DailyQuota:
    def is_within_budget(self, reserve_pct=0.2):
        return self.calls_today < self.rpd_limit * (1 - reserve_pct)

# P1: Discounted TS (5 SATIR degisiklik)
# Mevcut: alpha *= 0.99 (hourly batch)
# Yeni: alpha = gamma * alpha + reward (per-call, gamma=0.997)

# P2: Query-type routing
def classify_query(messages):
    if length < 500: return "simple"      → Groq 8B
    elif has_code: return "code"           → Cerebras Qwen
    elif has_analysis: return "complex"    → Gemini Flash
    else: return "medium"                  → Groq 70B
```

### I1: Market as Living Organism — Vital Signs

**Piyasa da bir organizma. Bizim organizma ONUN ICINDE yasıyor.**

**Market Vital Signs (I1 ajani):**

| Vital Sign | Olcum | Saglikli | Stresli | Kriz |
|-----------|-------|---------|---------|------|
| Liquidity depth | Orderbook depth ($) | >$100M | $10-100M | <$10M |
| Spread | Bid-ask bps | <1 bps | 1-10 bps | >10 bps |
| Correlation | Cross-asset rho | 0.3-0.6 | >0.8 | >0.95 (herding) |
| Volume rhythm | Volume/ADV ratio | 0.8-1.2 | <0.5 or >2.0 | <0.2 or >5.0 |
| Funding rate | Perp funding | ±0.01% | >0.05% | >0.10% |
| OI change | 24h OI delta | ±5% | >15% | >30% |
| F&G velocity | dFNG/dt | slow | fast | crash (>20pts/day) |

**Sornette Dragon Kings:** Power-law'dan SAPAN buyuk event'ler. 
LPPLS (Log-Periodic Power Law Singularity) ile 1-7 gun ONCE tahmin edilebilir.
**Python:** `lppls` (pip), `DS-LPPLS` (GitHub).

### I2 Detay: Adversarial Self-Play — ExploiterAgent

**Mevcut R1-R3 debate → Onerilen R1-R4 adversarial debate:**
```
R1: Tum agent'lar pozisyon belirtir (DEGISMEZ)
R2a: DevilsAdvocate majority'yi challenge eder (DEGISMEZ)
R2b: ExploiterAgent SPESIFIK attack senaryosu sunar (YENI)
R3: Agent'lar exploit'a karsi SAVUNMA yapar veya KABUL eder (DEGISMIS)
R4: ReflectionAgent meta-analysis + exploit assessment (GENISLETILMIS)
```

**exploit_archive tablosu:**
```sql
CREATE TABLE exploit_archive (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    pair TEXT, regime TEXT,
    exploit_scenario TEXT NOT NULL,
    target_weakness TEXT,
    predicted_loss REAL,
    was_defended BOOLEAN,
    defense_description TEXT,
    was_validated_by_outcome BOOLEAN DEFAULT NULL,
    created_at TEXT, ttl_expiry TEXT
);
```

**7 Co-Evolution Failure Mode (F.Robotics 2024):**
1. Red Queen Cycling → Archive-based evaluation (tum gecmis exploit'lara test)
2. Arms Race Plateau → Periodic external scenario injection
3. Loss of Diversity → Regime-specific subspecialties
4. Exploit Hallucination → EvidenceValidator fact-check
5. Defensive Collapse → Forgone PnL tracking
6. Collusion → Zero-sum prompt design
7. Computational Spiraling → Offline nightly batch (inline DEGIL)

### I4 Detay: Sleep-Wake — Borbely + SMDP + Whittle

**Borbely Two-Process Model (1982, 43 yildir gecerli):**
```
Process S (uyku baskisi): Hesaplama yorgunlugu — RAM, error rate, cache staleness
  S(t) = S_max × (1 - e^(-t_wake / τ_rise))     — uyanikken artar

Process C (sirkadyen): Piyasa firsati ritmi — cerebellum'dan ogreniilmis
  C(t) = A × sin(2π×t/24 + φ) + B × sin(2π×t/12 + ψ) + baseline

Uyku tetiklenir: S(t) > C(t) + δ_upper
Uyanis tetiklenir: S(t) < C(t) - δ_lower
```

**3 Mod:**
| Mod | Job Sayisi | Sizing | Tetik |
|-----|-----------|--------|-------|
| FULL_WAKE | 52 | 1.0-1.3x | Varsayilan / wake trigger |
| LIGHT_SLEEP | 15-20 | 0.3x | Dusuk deger + pozisyon yok |
| DEEP_SLEEP | 5-7 | 0.0x | Uzun dusuk deger + pozisyon yok |

**Whittle Index Job Prioritization:**
- Her job'un "passivity subsidy"'si hesaplanir
- LIGHT_SLEEP'te en yuksek Whittle index'li K job calisir
- Always-on (5): stoploss monitor, emergency alert, heartbeat, price feed, log rotation
- High-index (15): evidence engine, funding monitor, news scan, cerebellum, organism decay

**Tahmini tasarruf: %42 compute reduction** (1248 → 724 job-hours/gun)

### I5 Detay: Trading as Language — 10 Strateji

**Graduated Pipeline (veri miktarina gore):**

| Asama | Min Trade | Yontem | Complexity |
|-------|-----------|--------|-----------|
| 1 | 100+ | N-gram frequency + chi-squared | LOW (~50 satir) |
| 2 | 500+ | PrefixSpan sequential pattern mining | LOW (~60 satir) |
| 3 | 500+ | HMM performance regime detection | MEDIUM (~100 satir) |
| 4 | 1000+ | LDA topic modeling (trade archetypes) | MEDIUM (~80 satir) |
| 5 | 1000+ | SAX-ABBA symbolic representation | MEDIUM |
| 6 | 1289+ | Conditional probability grammar rules | LOW-MED (~100 satir) |
| 7 | 2000+ | BPE motif discovery | MEDIUM |
| 8 | 2000+ | Decision Transformer (C2) | HIGH |
| 9 | 5000+ | Mini-GPT autoregressive generator | HIGH |
| 10 | 5000+ | Multimodal sequence fusion (SALMON) | VERY HIGH |

**Hemen yapilabilir (1289 trade yeterli):**
```python
# N-gram frequency analysis
from collections import Counter

def tokenize_trade(contract: dict) -> str:
    """decision_contract JSON → discrete token."""
    direction = contract["decision"]["signal"][:4]  # BULL/BEAR/NEUT
    conf_bucket = "HIGH" if contract["confidence"] > 0.65 else "MED" if contract["confidence"] > 0.45 else "LOW"
    regime = contract.get("regime", "UNK")[:4]
    outcome = "WIN" if contract.get("outcome_pnl", 0) > 0 else "LOSS"
    return f"{direction}_{conf_bucket}_{regime}_{outcome}"

def find_patterns(trades: List[dict], n: int = 3) -> List[tuple]:
    """N-gram pattern'lari bul, chi-squared ile test et."""
    tokens = [tokenize_trade(t) for t in trades]
    ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens)-n)]
    
    counts = Counter(ngrams)
    total = len(ngrams)
    
    # Baseline: beklenen frekans (uniform dagilim varsayimi)
    vocab_size = len(set(tokens))
    expected = total / (vocab_size ** n)
    
    significant = []
    for pattern, observed in counts.most_common(50):
        chi2 = (observed - expected) ** 2 / expected
        if chi2 > 10.83:  # p < 0.001
            significant.append((pattern, observed, chi2))
    
    return significant
```

---

## FIX FIRST: BOZUK TEMELI DUZELT

> Her fix icin: Sorun + Root Cause + Cozum + Pseudocode + DB + Kabul Kriteri

### Fix 1: OOD Detector — Mahalanobis → Wasserstein (D4 Ajani)

**Sorun:** `ood_detector.py` Mahalanobis distance = 500.01 HER pair icin HER saniye.
`defensive_mult = 0.17` → tum trade'ler normal sizing'in %17'si.

**Root Cause (D4 detayli analiz):**
1. `_fit_gaussian()` (L204-225): Gaussian varsayimi. Crypto kurtosis 9.78-18.58 → Gaussian DEGIL.
2. Curse of dimensionality: d=20 feature'da Mahalanobis distance KONSANTRE — ayrim gucu yok.
3. `1e-6 * I` regularization yetersiz → precision matrix'te devasa eigenvalue'lar.
4. `chi2.ppf(0.95, df=20)` = 5.6 threshold → distance 500 ile threshold 5.6 karsilastiriliyor.

**Cozum — 3 Fazli Gecis:**

**Faz 1: Per-Feature W1 Drop-In (~50 satir, HEMEN)**
```python
# ood_detector.py degisikligi
import scipy.stats

class MarketOODDetector:
    def _fit_reference(self, features: np.ndarray):
        """Gaussian FIT yerine empirical dagilimlari sakla."""
        self._reference_distributions = {}
        for col_idx in range(features.shape[1]):
            sorted_vals = np.sort(features[:, col_idx])
            self._reference_distributions[col_idx] = sorted_vals
        
        # Empirical threshold: training set'in kendi W1 dagilimi
        self._w1_scores_train = []
        for i in range(len(features)):
            score = self._compute_w1(features[i])
            self._w1_scores_train.append(score)
        self._threshold = np.percentile(self._w1_scores_train, 95)
    
    def _compute_w1(self, sample: np.ndarray) -> float:
        """Per-feature Wasserstein-1 distance, ortalamasi."""
        distances = []
        for col_idx, ref in self._reference_distributions.items():
            d = scipy.stats.wasserstein_distance(ref, [sample[col_idx]])
            distances.append(d)
        return np.mean(distances)
    
    def detect(self, features: np.ndarray) -> dict:
        w1_score = self._compute_w1(features)
        is_ood = w1_score > self._threshold
        
        # Sigmoid-based defensive multiplier (smooth, binary degil)
        # w1 = threshold → mult = 0.5
        # w1 = 2*threshold → mult = 0.17
        # w1 = 0.5*threshold → mult = 0.83
        ratio = w1_score / (self._threshold + 1e-8)
        defensive_mult = 1.0 / (1.0 + math.exp(3.0 * (ratio - 1.0)))
        
        return {
            "is_ood": is_ood,
            "distance": round(w1_score, 4),
            "threshold": round(self._threshold, 4),
            "p_value": 1.0 - (np.searchsorted(
                sorted(self._w1_scores_train), w1_score) / len(self._w1_scores_train)),
            "defensive_mult": round(max(0.10, min(1.0, defensive_mult)), 4),
            "closest_regime": self._find_closest_regime(features),
        }
```

**Faz 2: Sliced Wasserstein Regime Detection (POT kutuphanesi)**
```python
import ot  # pip install POT

def detect_regime_shift(window_current: np.ndarray, window_reference: np.ndarray,
                        n_projections: int = 100) -> float:
    """Sliced Wasserstein distance between two multivariate windows."""
    sw = ot.sliced_wasserstein_distance(
        window_current, window_reference,
        n_projections=n_projections, seed=42
    )
    return sw  # buyuk → regime farkli, kucuk → regime ayni
```

**Faz 3: Sinkhorn Changepoint Monitoring**
```python
def sinkhorn_changepoint(window_t: np.ndarray, window_t_minus_1: np.ndarray,
                         epsilon: float = 0.1) -> float:
    """Online changepoint detection via Sinkhorn divergence."""
    M = ot.dist(window_t, window_t_minus_1)  # cost matrix
    s = ot.sinkhorn2(
        np.ones(len(window_t)) / len(window_t),
        np.ones(len(window_t_minus_1)) / len(window_t_minus_1),
        M, reg=epsilon
    )
    return float(s)  # spike → changepoint
```

**Performans:**
```
Per-feature W1 (20 feature, 200 sample): ~0.5ms
Sliced W2 (20 feature, 100 projection): ~5-10ms
Sinkhorn (200x200 cost matrix): ~10-20ms
Toplam: <30ms (vs 300s candle interval — ihmal edilebilir)
```

**Yeni dependency:** `pip install POT` (MIT, ~2MB)

**Kabul Kriteri:**
- distance artik 500 DEGIL — pair'ler arasi farklilik var
- BTC distance ~ 1.5, WLFI distance ~ 8.2 gibi mantikli degerler
- defensive_mult 0.17'den sigmoid-based 0.4-1.0 arasina cikiyor
- Log'da `[OOD] BTC/USDT w1=1.47 thr=3.21 → mult=0.82` gorulecek

---

### Fix 2: Agent Memory Revolution (J1-J5 Ajanlari)

> 5 alt-fix, her biri bir J ajaninin EXACT bulgusu.

**FIX 2A: ReflectionAgent'a Gercek Hafiza (J1)**

**Sorun:** `agent_pool.py` L158-159 system prompt diyor ki:
`"Your weapons: agent_performance table, agent_memory table"`
AMA R3 prompt'u (L396-413) SIFIR DB sorgusu yapiyor. LLM halucinasyon yapmak ZORUNDA.

Ayrica `confidence_modifier` (L404) LLM'den isteniyor ama `_weighted_synthesis()` (L502-582) 
HICBIR YERDE okumuyor. R3 ciktisi tamamen IGNOR ediliyor.

**Cozum — Yeni metod + prompt degisikligi + synthesis fix:**

```python
# agent_pool.py — YENI METOD (L278 civarinda ekle)
def _get_reflection_context(self, pair: str, regime: str, 
                             agent_names: List[str]) -> str:
    """J1: ReflectionAgent icin DB'den gercek hafiza cek."""
    try:
        conn = self._get_conn()
        lines = []
        
        # 1. Her agent'in bu regime'deki performansi
        lines.append("=== AGENT PERFORMANCE (son 30 gun, bu regime) ===")
        for name in agent_names:
            if name == "ReflectionAgent":
                continue
            row = conn.execute("""
                SELECT COUNT(*) as total,
                       SUM(CASE WHEN was_correct THEN 1 ELSE 0 END) as wins,
                       ROUND(AVG(outcome_pnl), 2) as avg_pnl
                FROM agent_performance
                WHERE agent_type = ? AND regime = ?
                  AND timestamp > datetime('now', '-30 days')
            """, (name, regime)).fetchone()
            total = row["total"] or 0
            wins = row["wins"] or 0
            avg = row["avg_pnl"] or 0.0
            wr = (wins / total * 100) if total > 0 else 0
            lines.append(f"  {name}: {total} sinyal, {wr:.0f}% WR, "
                        f"ort PnL {avg:+.2f}%")
        
        # 2. Bu pair icin son 7 gunun agent hafizasi
        rows = conn.execute("""
            SELECT agent_type, signal, strength, key_argument, 
                   final_outcome_pnl
            FROM agent_memory
            WHERE pair = ? AND timestamp > datetime('now', '-7 days')
            ORDER BY timestamp DESC LIMIT 20
        """, (pair,)).fetchall()
        
        if rows:
            lines.append(f"\n=== HAFIZA: {pair} (son 7 gun) ===")
            for r in rows:
                outcome = (f"→ {r['final_outcome_pnl']:+.2f}%" 
                          if r['final_outcome_pnl'] is not None 
                          else "→ BEKLIYOR")
                arg = (r['key_argument'] or '')[:80]
                lines.append(
                    f"  {r['agent_type']}: {r['signal']}"
                    f"(guc={r['strength']:.2f}) {outcome} | {arg}"
                )
        
        conn.close()
        return "\n".join(lines) if lines else "Henuz yeterli veri yok."
    except Exception as e:
        logger.debug(f"[AgentPool:R3] Reflection context hatasi: {e}")
        return "Veri alinamadi."

# R3 PROMPT DEGISIKLIGI (L396-413 arasini degistir):
reflection_history = self._get_reflection_context(pair, regime, list(agents))

prompt_r3 = (
    f"Round 3 — META-ANALYSIS for {pair}:\n"
    f"Round 1 positions: {r1_summary}\n"
    f"Round 2 revisions: {', '.join(r2_revisions) if r2_revisions else 'None'}\n\n"
    f"HISTORICAL DATA (from your tables — REAL, not hallucinated):\n"
    f"{reflection_history}\n\n"
    f"Based on ACTUAL performance data above, provide meta-analysis in JSON:\n"
    f'{{"trust_most": "agent with best data", '
    f'"trust_least": "agent with worst data", '
    f'"meta_insight": "key lesson from data", '
    f'"confidence_modifier": -0.10 to +0.10}}'
)

# SYNTHESIS FIX (L567 civarinda, clamp'ten ONCE ekle):
r3 = positions.get("ReflectionAgent", {}).get("round3", {})
r3_modifier = 0.0
try:
    raw = r3.get("confidence_modifier", 0)
    r3_modifier = max(-0.10, min(0.10, float(raw or 0)))
except (TypeError, ValueError):
    pass
if r3_modifier != 0.0:
    confidence += r3_modifier
    logger.info(f"[AgentPool:R3] modifier: {r3_modifier:+.2f} → "
               f"conf={confidence:.4f}")
```

**FIX 2B: Agent-Specific RAG (J3)**

**Sorun:** `_process_retrieval_requests()` L441: `r.search(f"{pair} latest analysis", top_k=3)`
— BUTUN agent'lar AYNI generic query. TrendFollower trend verisi degil generic haber aliyor.

**Cozum:**
```python
# AGENT_REGISTRY'ye yeni alan ekle (L47-170 arasina):
"TrendFollower": {
    "best_regimes": ["trending_bull", "trending_bear"],
    "system_prompt": "...",
    "rag_keywords": "trend momentum EMA ADX continuation breakout",
    "rag_event_types": ["trend_reversal", "breakout"],
},
"FundingContrarian": {
    "rag_keywords": "funding rate squeeze long short ratio liquidation",
    "rag_event_types": ["funding_extreme", "liquidation_cascade"],
},
"MacroCorrelator": {
    "rag_keywords": "macro DXY VIX treasury yields correlation risk-on",
    "rag_event_types": ["fomc", "cpi_release", "fed_decision"],
},
# ... diger agent'lar icin de

# _process_retrieval_requests() imza degisikligi (L427):
def _process_retrieval_requests(self, response_text: str, pair: str,
                                 agent_name: str = "") -> str:
    # L441 degisikligi:
    agent_kw = AGENT_REGISTRY.get(agent_name, {}).get("rag_keywords", "")
    results = r.search(f"{pair} {agent_kw}", top_k=3)
```

**FIX 2C: key_argument Feedback (J4)**

**Sorun:** `key_argument` her debate'te yaziliyor (L677), HIC okunmuyor (0 SELECT).

**Cozum — Argument Quality Scoring:**
```python
# Yeni tablo
CREATE TABLE IF NOT EXISTS argument_quality (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_type TEXT NOT NULL,
    argument_pattern TEXT NOT NULL,
    regime TEXT NOT NULL,
    times_used INTEGER DEFAULT 0,
    times_correct INTEGER DEFAULT 0,
    avg_pnl_when_used REAL DEFAULT 0.0,
    quality_score REAL DEFAULT 0.5,
    updated_at TEXT,
    UNIQUE(agent_type, argument_pattern, regime)
);

# Argument pattern extraction (regex-based):
ARGUMENT_PATTERNS = {
    "adx_trend":       r"ADX\s*[><=]\s*\d+",
    "rsi_oversold":    r"RSI.*(oversold|below\s*30)",
    "rsi_overbought":  r"RSI.*(overbought|above\s*70)",
    "funding_extreme": r"funding.*(extreme|>0\.0[5-9])",
    "macd_signal":     r"MACD.*(cross|histogram|divergence)",
    "volume_anomaly":  r"volume.*(spike|anomal|surge|confirm)",
    "support_level":   r"(support|resistance).*(level|zone)",
    "ema_alignment":   r"EMA.*(align|cross|above|below)",
    "fng_extreme":     r"(fear|greed).*(extreme|panic|euphoria)",
    "momentum_strong": r"momentum.*(strong|accel|zone)",
}

# R1 prompt'a enjekte (L308 sonrasi):
arg_feedback = self._get_argument_quality(agent_name, regime)
# Cikti: "BEST: 'ADX>30 + volume confirm' (%78 acc, 47 kullanim)
#         WORST: 'RSI divergence' (%31 acc — DIKKAT, trende karsi)"
```

**FIX 2D: MAGMA Graph Memory (J2)**

**Sorun:** `add_agent_interaction()` ve `add_agent_node()` graph_store.py'de TANIMLI ama 
HICBIR YERDEN CAGRILMIYOR. %100 dead code. magma_edges'te 11 satir, 0 debate edge.

**Cozum — Debate Lifecycle Edge'leri:**
```python
# agent_pool.py run_debate() sonuna ekle (L420 civarinda):

# R1 sonrasi: her agent'in pozisyonunu kaydet
for name, pos in positions.items():
    if name == "ReflectionAgent":
        continue
    if self._graph_store:
        self._graph_store.add_edge(
            "entity", name.lower(),
            f"argued_{pos.get('direction', 'neutral').lower()}",
            pair.lower().replace("/", "_"),
            weight=pos.get("strength", 0.5),
            metadata={
                "key_argument": pos.get("key_argument", "")[:200],
                "regime": regime,
                "debate_id": f"{pair}_{datetime.now().strftime('%Y%m%d_%H%M')}",
            }
        )

# R2 sonrasi: persuasion event'lerini kaydet
for name, pos in positions.items():
    r2 = pos.get("round2", {})
    if r2.get("revised_direction") and r2["revised_direction"] != pos.get("direction"):
        # Agent IKNA EDILDI
        if self._graph_store:
            self._graph_store.add_agent_interaction(
                "devilsadvocate", name.lower(),
                interaction_type="persuaded",
                weight=1.0
            )
    elif r2.get("revised_direction"):
        # Agent DIRENDI
        if self._graph_store:
            self._graph_store.add_agent_interaction(
                name.lower(), "devilsadvocate",
                interaction_type="resisted",
                weight=1.0
            )
```

**FIX 2E: Agent Pheromone Deposit (J5)**

**Sorun:** Agent'lar pheromone field'a HICBIR SEY yazmiyor. 14 SIGNAL_* sabiti var, 
HICBIRI agent ile ilgili degil.

**Cozum:**
```python
# pheromone_field.py'ye ekle (L278 sonrasi):
SIGNAL_AGENT_CONSENSUS = "agent_consensus"
SIGNAL_AGENT_DISSENT = "agent_dissent"

# agent_pool.py run_debate() sonuna ekle:
from pheromone_field import get_pheromone_field, SIGNAL_AGENT_CONSENSUS, SIGNAL_AGENT_DISSENT

pfield = get_pheromone_field()
pfield.deposit("agent_pool", SIGNAL_AGENT_CONSENSUS, {
    "signal": result["signal"],
    "confidence": result["confidence"],
    "n_agents": len(positions),
    "pair": pair,
}, half_life=120)  # 2 dakika — sonraki perception cycle'da okunabilir

# Eger guclu dissent varsa ayrica kaydet
if bear_norm > 0.3 and bull_norm > 0.3:  # her iki taraf da guclu
    pfield.deposit("agent_pool", SIGNAL_AGENT_DISSENT, {
        "bull_strength": bull_norm,
        "bear_strength": bear_norm,
        "pair": pair,
    }, half_life=120)
```

**Kabul Kriteri (Tum 2A-2E):**
- ReflectionAgent loglarinda `HISTORICAL DATA:` blogu gorulecek (2A)
- Agent debate loglarinda agent-specific RAG keyword'leri gorulecek (2B)
- argument_quality tablosu populate oluyor (2C)
- magma_edges'te argued/persuaded/resisted edge'ler birikiyor (2D)
- Pheromone field'da SIGNAL_AGENT_CONSENSUS gorulecek (2E)

---

### Fix 3: Pheromone Trail Ring Buffer (F2 Ajani)

**Sorun:** `pheromone_field.py` L94: `self._field[compound_key] = pheromone` — her deposit 
oncekini SILIYOR. Gecmis yok, gradient yok, integral yok. HQ-1 (Confidence Integral) 
ve HQ-3 (Pheromone Gradient) implement edilemez.

**Akademik Temel (F2):**
- Dorigo & Stutzle (2004) ACO: `tau(t+1) = (1-rho)*tau(t) + SUM(delta_tau)` — evaporation + ACCUMULATION
- Usher & McClelland (2001) LIF neuron: `dx/dt = -k*x(t) + I(t)` — leak + integrate
- Stutzle & Hoos (2000) MMAS: `[tau_min, tau_max]` bounds — saturation korunmasi

**Cozum:**
```python
from collections import deque

@dataclass
class PheromoneTrail:
    """Ring buffer of deposits for a single signal key."""
    deposits: deque           # deque(maxlen=K) of Pheromone objects
    accumulated_value: float  # LIF membrane potential (leaky integrated)

class PheromoneField:
    def __init__(self, trail_length: int = 32):
        self._field: Dict[str, PheromoneTrail] = {}
        self._trail_length = trail_length
        self._TAU_MIN = 0.01   # MMAS alt sinir
        self._TAU_MAX = 10.0   # MMAS ust sinir
    
    def deposit(self, source: str, signal_type: str, value: Any,
                half_life: float = 30.0, metadata: dict = None):
        compound_key = f"{source}::{signal_type}"
        
        pheromone = Pheromone(
            value=value, source=source,
            deposited_at=time.monotonic(),
            half_life=half_life,
            metadata=metadata or {}
        )
        
        # Trail yoksa olustur
        if compound_key not in self._field:
            self._field[compound_key] = PheromoneTrail(
                deposits=deque(maxlen=self._trail_length),
                accumulated_value=0.0
            )
        
        trail = self._field[compound_key]
        
        # LIF-style leaky accumulation
        if len(trail.deposits) > 0:
            last = trail.deposits[-1]
            dt = pheromone.deposited_at - last.deposited_at
            decay = 0.5 ** (dt / half_life)
            trail.accumulated_value = trail.accumulated_value * decay + self._extract_float(value)
        else:
            trail.accumulated_value = self._extract_float(value)
        
        # MMAS bounds
        trail.accumulated_value = max(self._TAU_MIN, 
                                      min(self._TAU_MAX, trail.accumulated_value))
        
        # Ring buffer'a ekle (eski otomatik duser)
        trail.deposits.append(pheromone)
    
    def read_integral(self, signal_type: str, window_seconds: float = 3600.0,
                      source: str = None) -> float:
        """HQ-1: Leaky integral of signal over time window.
        4 saattir BULLISH diyen sinyal > 5 dakika once BULLISH'e donen sinyal.
        """
        matching = self._find_trails(signal_type, source)
        if not matching:
            return 0.0
        
        now = time.monotonic()
        total = 0.0
        weight_sum = 0.0
        
        for trail in matching:
            for pheromone in trail.deposits:
                age = now - pheromone.deposited_at
                if age > window_seconds:
                    continue
                decay = 0.5 ** (age / pheromone.half_life)
                val = self._extract_float(pheromone.value)
                total += val * decay
                weight_sum += decay
        
        return total / (weight_sum + 1e-8)
    
    def read_gradient(self, signal_type: str, source: str = None) -> float:
        """HQ-3: Temporal rate of change of pheromone concentration.
        gradient > 0 ve buyuyor → conviction artiyor → sizing artir
        gradient > 0 ama yavashyor → conviction zirve → dikkat
        gradient < 0 → conviction dusuyor → sizing azalt
        """
        matching = self._find_trails(signal_type, source)
        if not matching:
            return 0.0
        
        for trail in matching:
            if len(trail.deposits) < 2:
                return 0.0
            
            latest = trail.deposits[-1]
            previous = trail.deposits[-2]
            
            dt = latest.deposited_at - previous.deposited_at
            if dt < 0.001:
                return 0.0
            
            v_latest = self._extract_float(latest.value)
            v_previous = self._extract_float(previous.value)
            
            return (v_latest - v_previous) / dt
        
        return 0.0
    
    def read_acceleration(self, signal_type: str, source: str = None) -> float:
        """HQ-3: Second derivative — conviction hizlaniyor mu yavasliyor mu?"""
        matching = self._find_trails(signal_type, source)
        if not matching:
            return 0.0
        
        for trail in matching:
            if len(trail.deposits) < 3:
                return 0.0
            
            p1, p2, p3 = trail.deposits[-3], trail.deposits[-2], trail.deposits[-1]
            v1 = self._extract_float(p1.value)
            v2 = self._extract_float(p2.value)
            v3 = self._extract_float(p3.value)
            dt1 = p2.deposited_at - p1.deposited_at
            dt2 = p3.deposited_at - p2.deposited_at
            
            if dt1 < 0.001 or dt2 < 0.001:
                return 0.0
            
            g1 = (v2 - v1) / dt1
            g2 = (v3 - v2) / dt2
            
            return (g2 - g1) / ((dt1 + dt2) / 2)
        
        return 0.0
```

**Memory Impact:**
```
Mevcut: 1 Pheromone/key × ~100 key × 200 byte = ~20 KB
Yeni:   32 Pheromone/key × ~100 key × 200 byte = ~640 KB
Fark: +620 KB — ihmal edilebilir
```

**Kabul Kriteri:**
- `read_integral("prediction", 3600)` son 1 saatin confidence integralini donduruyor
- `read_gradient("prediction")` pozitif/negatif gradient veriyor
- `read_acceleration("prediction")` ivmelenme/yavaslama gosteriyor

---

### Fix 4: SHAP Narrative Injection (G3 Ajani)

**Sorun:** `triple_perception.py` L338-354: CatBoost SHAP top-5 hesaplaniyor.
`rag_graph.py` coordinator: MADAM debate sifir SHAP bilgisi aliyor.
Neural → Symbolic bilgi transferinde %90 kayip.

**Cozum:**
```python
# triple_perception.py — _catboost_predict() sonuna ekle (L354 civarinda):
def _format_shap_narrative(self, shap_values: dict, prediction: str, 
                            probability: float) -> str:
    """G3: SHAP sonuclarini MADAM-okunabilir dogal dile cevir."""
    sorted_features = sorted(shap_values.items(), 
                            key=lambda x: abs(x[1]), reverse=True)[:5]
    
    direction_word = "LONG" if prediction == "BULLISH" else "SHORT" if prediction == "BEARISH" else "NEUTRAL"
    
    lines = [
        f"CatBoost predicts {direction_word} ({probability:.0%} probability).",
        f"Top contributing features:"
    ]
    
    for feat_name, shap_val in sorted_features:
        direction = "supporting" if shap_val > 0 else "opposing"
        lines.append(f"  • {feat_name}: {shap_val:+.3f} ({direction})")
    
    # Uyumluluk analizi
    supporting = [f for f, v in sorted_features if v > 0]
    opposing = [f for f, v in sorted_features if v < 0]
    
    if len(opposing) == 0:
        lines.append("All top features AGREE on direction — high conviction.")
    elif len(opposing) >= 3:
        lines.append(f"WARNING: {len(opposing)}/5 features OPPOSE the prediction — "
                    f"mixed signal, reduce sizing.")
    
    return "\n".join(lines)

# perceive() return dict'e ekle:
result["shap_narrative"] = self._format_shap_narrative(
    shap_top_features, signal, confidence
)

# rag_graph.py — state dict'e SHAP narrative ekle:
# _get_trading_signal_inner() icinde perception sonucunu al:
shap_text = perception_result.get("shap_narrative", "")
# MADAM prompt'a ekle:
state["shap_context"] = shap_text

# coordinator_debate() prompt'unda (L1350 civarinda):
f"NEURAL MODEL ASSESSMENT:\n{state.get('shap_context', 'Not available')}\n\n"
```

**Kabul Kriteri:** MADAM debate loglarinda su gorulecek:
```
CatBoost predicts LONG (72% probability).
Top contributing features:
  • rsi_14: +0.230 (supporting)
  • bb_width: +0.180 (supporting)
  • volume_ratio: -0.120 (opposing)
All top features AGREE on direction — high conviction.
```

---

### Fix 5: Scheduler Job Debug (C3 Ajani)

**Sorun:** `causal_discoveries=0, dream_scenarios=0, world_model_rollouts=0`
Sprint 2 ogrenme motorlari SIFIR veri uretmis.

**Dream Engine 4 Kod Hatasi (C3 ajani bulgusu):**
1. `dream_engine.py` L199: `np.random.randn(TOTAL_STATE_DIM) * 0.1` — RANDOM initial state. 
   Olmasi gereken: MBPO-style branching (gercek state'ten baslat).
2. DreamFilter L57-129: Mahalanobis threshold 3.0 std → 2D veri icin tasarlanmis, 
   64D latent space'te HER dream filtreleniyor.
3. `_persist_session()` L319-334: `initial_state` kolonu INSERT'te YAZILMIYOR (6/10 kolon).
4. `event_injection` L303-311: %50 "normal" event — %50'si noise injection.

**Cozum:**
```python
# dream_engine.py degisiklikleri:

# 1. MBPO-style initial state (random degil, gercek state'ten branch):
def dream_session(self, initial_state: np.ndarray = None, ...):
    if initial_state is None:
        # Gercek piyasa state'ini al (random DEGIL)
        initial_state = self._get_latest_real_state()
        if initial_state is None:
            # Fallback: training set'ten rastgele bir state sec
            initial_state = self._sample_from_training_set()

# 2. Filter threshold'u latent dim'e gore ayarla:
def _is_plausible(self, state: np.ndarray) -> bool:
    # 2D icin 3.0 std dogru, 64D icin ~8.0 std gerekli
    # chi2.ppf(0.95, df=64) ≈ 83.7, sqrt = 9.15
    threshold = math.sqrt(scipy.stats.chi2.ppf(0.95, df=len(state)))
    return self._mahalanobis(state) < threshold

# 3. initial_state'i persist et:
conn.execute("""
    INSERT INTO dream_scenarios
    (dream_session_id, trajectory_idx, step,
     initial_state, event_type, state_after, reward, 
     passed_filter, filter_reason)
    VALUES (?, ?, ?, ?, ?, ?, ?, 1, NULL)
""", (session_id, i, step_idx, 
      initial_state.tobytes(),  # BLOB olarak kaydet
      step.get("event_type", "normal"),
      step.get("state_after", b""),
      step["reward"]))

# 4. Event injection oranini azalt:
EVENT_PROBS = {
    "normal": 0.70,      # %50 → %70 (daha az noise)
    "flash_crash": 0.05,
    "whale_buy": 0.05,
    "news_shock": 0.05,
    "regime_shift": 0.05,
    "squeeze": 0.05,
    "low_liquidity": 0.03,
    "black_swan": 0.02,
}
```

**Kabul Kriteri:** Pazar gunu job'lardan sonra:
- `dream_scenarios` > 0 satir
- `causal_discoveries` > 0 satir
- `world_model_rollouts` > 0 satir
- Scheduler logunda hata GORULMEYECEK

---

### Fix 6: Forgone PnL Feedback Loop (H3 Ajani)

**Sorun:** 12,074 kayit, $1,247 forgone PnL. Scheduler L631: 
`"diagnostic only -- does NOT auto-change thresholds"`.
Ayrica kolon adi BUGU: postmortem `resolved_pnl_pct` sorguluyor ama kolon `forgone_pnl`.

**Cozum:**

```sql
-- 1. regime kolonu ekle:
ALTER TABLE forgone_profit ADD COLUMN regime TEXT DEFAULT NULL;

-- 2. Kolon adi bugunu fix et (scheduler.py L1040, L1043):
-- YANLIS: SELECT ... resolved_pnl_pct ... timestamp
-- DOGRU:  SELECT ... forgone_pnl ... signal_time
```

```python
# 3. Shadow trade resolver job (yeni scheduler job):
def _resolve_shadow_trades(self):
    """4 saatlik pencerede shadow trade'leri cozumle."""
    conn = get_db_connection()
    unresolved = conn.execute("""
        SELECT id, pair, signal_type, entry_price, signal_time
        FROM forgone_profit
        WHERE was_executed = 0 AND forgone_pnl IS NULL
          AND signal_time < datetime('now', '-4 hours')
        LIMIT 100
    """).fetchall()
    
    for row in unresolved:
        current_price = self._get_current_price(row["pair"])
        if current_price:
            forgone_engine.resolve_forgone_trade(
                row["id"], current_price
            )

# 4. Per-pair adaptive threshold:
def _adjust_pair_thresholds(self):
    """Haftalik: per-pair forgone alpha'ya gore threshold ayarla."""
    conn = get_db_connection()
    pairs = conn.execute("""
        SELECT pair, regime,
               AVG(CASE WHEN was_executed=0 AND forgone_pnl>0 
                   THEN forgone_pnl ELSE 0 END) as forgone_positive,
               AVG(CASE WHEN was_executed=0 AND forgone_pnl<0 
                   THEN forgone_pnl ELSE 0 END) as forgone_negative
        FROM forgone_profit
        WHERE signal_time > datetime('now', '-7 days')
          AND forgone_pnl IS NOT NULL
        GROUP BY pair, regime
    """).fetchall()
    
    for p in pairs:
        forgone_alpha = p["forgone_positive"] + p["forgone_negative"]
        if forgone_alpha > 2.0:  # surekli kacirilmis kar
            # Threshold dusur (daha fazla trade ac)
            adjust_pair_threshold(p["pair"], p["regime"], delta=-0.02)
            logger.info(f"[Forgone] {p['pair']} threshold DUSURULDU "
                       f"(forgone_alpha={forgone_alpha:+.2f})")
        elif forgone_alpha < -1.0:  # kacirilmis zararlar
            # Threshold yukselt (daha secici ol)
            adjust_pair_threshold(p["pair"], p["regime"], delta=+0.01)
```

**Kabul Kriteri:**
- Kolon adi bug'i fixlenmis, postmortem gercek veri gosteriyor
- Shadow trade resolver calisiyor (forgone_pnl NULL satir sayisi azaliyor)
- Telegram raporunda pair-bazli forgone alpha gorulecek
- Per-pair threshold otomatik adjust ediliyor

---

### Fix 7: Cortisol Hysteresis — 8 Satir (H2 Ajani)

**Sorun:** `Hormones.compute()` (neural_organism.py L612-615) STATELESS.
Stres dusunce cortisol ANINDA dusuyor. Dunku buyuk kaybin hafizasi yok.
`Amygdala.get_current_fear()` (L689-697) ZATEN hysteresis kullaniyor — pattern mevcut.

**Cozum (Amygdala pattern'ini cortisol'e kopyala):**
```python
# neural_organism.py Hormones.__init__ (L583 sonrasi):
self._peak_cortisol: float = 1.0
self._peak_cortisol_time: Optional[datetime] = None

# neural_organism.py Hormones.compute() (L619 sonrasi):
raw_cortisol = self.cortisol  # mevcut hesaplamadan gelen deger

# Hysteresis: peak cortisol decay memory
if self._peak_cortisol_time is not None and raw_cortisol < self._peak_cortisol:
    hours = (datetime.now(tz=timezone.utc) - self._peak_cortisol_time).total_seconds() / 3600
    decayed_peak = self._peak_cortisol * (0.5 ** (hours / 24.0))  # 24h half-life
    self.cortisol = max(raw_cortisol, decayed_peak)  # HYSTERESIS LINE

# Peak tracking
if self.cortisol >= self._peak_cortisol:
    self._peak_cortisol = self.cortisol
    self._peak_cortisol_time = datetime.now(tz=timezone.utc)
```

**Ornek:**
```
Dun: cortisol = 0.9 (buyuk kayip)
Bugun: stres yok, raw_cortisol = 0.3
Hysteresis: decayed_peak = 0.9 * 0.5^(24/24) = 0.45
Effective cortisol = max(0.3, 0.45) = 0.45 (HAFIZA!)
Hysteresis OLMADAN: cortisol = 0.3 (dunku stres UNUTULMUS)
```

**Ayrica: Fear sizing_mult'u baglantisi (H2):**
```python
# Amygdala.process_loss() fear_response["sizing_mult"] HESAPLANIYOR ama KULLANILMIYOR.
# neural_organism.py L1677-1685 neuron modulation'da fear_response oku:
fear = self.amygdala.get_current_fear()
if fear > 0.3:
    # Fear-based sizing reduction (MEVCUT sizing pipeline'a ekle)
    fear_sizing = max(0.5, 1.0 - fear)  # fear=0.8 → sizing x0.2
```

**Kabul Kriteri:** 
- Buyuk kayiptan sonra cortisol 24 saat boyunca yavas yavas duser
- Log'da `cortisol=0.45 (peak_decay from 0.90)` gorulecek

---

### Fix 8: Dead Code Entegrasyon (Codebase Explorer)

**10 modul, 2,711 satir yazilmis ama baglanmamis:**

| # | Modul | Entegrasyon Plani | Sprint |
|---|-------|-------------------|--------|
| 1 | `lob_encoder.py` (195) | order_flow pipeline + multimodal input | 3A |
| 2 | `slippage_forecaster.py` (180) | sqrt-law update (D2), Almgren-Chriss | 3A |
| 3 | `trinity_fusion.py` (350) | multimodal_encoder → trinity → strategy | 3B |
| 4 | `market_maker_mode.py` (233) | ADX<20 regime, Stoikov GLFT, 14 prereq | 3B |
| 5 | `sac_online.py` (397) | IQL ile dual-motor RL | 3B |
| 6 | `hrl_meta_policy.py` (420) | Meta-controller: "hangi RL ne zaman?" | 3B |
| 7 | `multimodal_encoder.py` (289) | 7 bug fix (G4) + training pipeline | 3B |
| 8 | `sim2real_pipeline.py` (148) | Per-pair slippage injection (D2) | 3B |
| 9 | `external_data_integrator.py` (320) | Kaggle + HuggingFace veri | 3B |
| 10 | `gam_rag.py` (179) | Graph-augmented RAG | 3B |

**Oncelik: lob_encoder + slippage_forecaster Sprint 3A'da** (Hawkes + VPIN icin gerekli).
Geri kalan Sprint 3B'de.

### Fix 9: Real VPIN (D5 Ajani)

**Sorun:** `order_flow.py` L149: `vpin = abs(buy_vol - sell_vol) / total_vol` — 
Bu VPIN DEGIL, sadece Order Imbalance Ratio. Gercek VPIN volume bucketing gerektirir.

**Cozum:**
```python
# pip install flowrisk
from flowrisk import RecursiveVPIN

# order_flow.py'ye ekle:
class OrderFlowAnalyzer:
    def __init__(self):
        self._vpin_estimator = RecursiveVPIN(
            bucket_size=None,  # otomatik: daily_vol / 50
            ewma_span=50       # 50 bucket rolling window
        )
    
    def update_vpin(self, trades: List[dict]):
        for trade in trades:
            self._vpin_estimator.update(
                price=trade["price"],
                volume=trade["amount"],
                side=trade["side"]  # "buy" veya "sell"
            )
        return self._vpin_estimator.vpin  # gercek VPIN [0, 1]
    
    def get_toxicity_composite(self) -> float:
        """D5: Composite toxicity index."""
        vpin = self._vpin_estimator.vpin
        lambda_kyle = self._compute_kyle_lambda()  # OLS regression
        amihud = self._compute_amihud()  # |return| / dollar_volume
        
        # CDF-percentile based (absolute threshold DEGIL)
        vpin_pct = self._percentile_rank(vpin, self._vpin_history)
        kyle_pct = self._percentile_rank(lambda_kyle, self._kyle_history)
        amihud_pct = self._percentile_rank(amihud, self._amihud_history)
        
        composite = 0.50 * vpin_pct + 0.30 * kyle_pct + 0.20 * amihud_pct
        return composite  # 0.0 = clean, 1.0 = toxic
```

---

### Fix 10: CircadianRhythm → Cerebellum Baglantisi (I4)

**Sorun:** `autonomous_lifecycle.py` CircadianRhythm: hardcoded saatler.
`AGGRESSIVE_HOURS=[8,9,10,11,14,15,16,17]`, `SLEEP_HOURS=[22,23]`.
cerebellum_timing.py ogrenilmis saat multiplier'lari VAR ama CircadianRhythm OKUMUYOR.

**Cozum:**
```python
# CircadianRhythm.get_time_modulation() degisikligi:
def get_time_modulation(self) -> dict:
    hour = datetime.now(tz=timezone.utc).hour
    
    # HARDCODED YERINE cerebellum'dan oku:
    try:
        from cerebellum_timing import get_cerebellum
        cerebellum = get_cerebellum()
        timing = cerebellum.get_timing_multiplier(hour)
        
        if timing > 1.0:
            mode = "aggressive"
        elif timing < 0.5:
            mode = "sleep"
        else:
            mode = "conservative"
        
        return {"mode": mode, "sizing_modifier": timing}
    except Exception:
        # Fallback: eski hardcoded mantik
        return self._legacy_time_modulation(hour)
```

---

## SPRINT 3A: CORE SIZING + REGIME + QUAD PERCEPTION

### Task 11: CAAT Asymmetric Alpha Sizing Formula

**ARGE'deki 6-parca formul + 50 ajan bulgulariyla GUNCELLENMIS:**

```python
def caat_asymmetric_size(perception, organism, dream_engine, 
                          market_data, portfolio, pair, regime):
    """
    CAAT Sprint 3A: Asymmetric Alpha Sizing Formula — UPDATED
    
    50 ajan sentezi:
      1. Per-Pair 7-Step Kelly (E1 — Prensip 0)
      2. Dual-Axis Continuous Confidence (Phase 26 Novel #6)
      3. Hormonal Sizing + Hysteresis (Fix 7, HQ-6)
      4. Dream Familiarity Bonus (HQ-4 — dream data birikince)
      5. VWTSMOM + Fractal Regime + 2-Week Rule (E5)
      6. Shannon Volatility Harvest (E3 — ranging markets)
      7. Impact Constraint (D2 — NEW)
      8. Hawkes Cascade Guard (D1 — NEW)
      9. Confidence Integral (HQ-1 — NEW)
      10. Forgone Alpha Adjustment (HQ-11 — NEW)
    """
    
    # ─── PARCA 1: Per-Pair Kelly (Prensip 0) ───
    base_size = per_pair_kelly_size(pair, regime, portfolio.total)
    
    # ─── PARCA 2: Continuous Confidence ───
    catboost_prob = perception.catboost_probability
    uncertainty = perception.uncertainty_composite
    confidence = 1.0 / (1.0 + math.exp(-10 * (catboost_prob / max(uncertainty, 0.01) - 0.5)))
    
    # ─── PARCA 3: Hormonal Sizing + Hysteresis ───
    h = organism.hormones
    hormonal_scalar = (
        h.dopamine *
        (1.0 / max(h.cortisol, 0.3)) *  # cortisol artik HYSTERESIS ile
        h.serotonin
    )
    hormonal_scalar = max(0.5, min(2.0, hormonal_scalar))
    
    # ─── PARCA 4: Dream Familiarity (veri birikince aktif) ───
    dream_bonus = 1.0
    if dream_engine.has_data():
        dream_var = dream_engine.ensemble_variance_for_current_state()
        dream_familiarity = 1.0 / (1.0 + 10.0 * dream_var)
        dream_bonus = 1.0 + dream_familiarity * 0.5
    
    # ─── PARCA 5: VWTSMOM + Fractal + 2-Week Rule ───
    vwtsmom = volume_weighted_momentum(market_data, lookback_days=14)
    hurst = perception.chart_features["hurst_100"]
    
    if hurst > 0.55:
        regime_mult = 1.2
    elif hurst < 0.45:
        regime_mult = 0.5
    else:
        regime_mult = 0.8
    
    regime_filter = abs(vwtsmom) * regime_mult
    
    # Asymmetric allocation: crypto long bias
    if vwtsmom > 0:
        regime_filter *= 1.0
    else:
        regime_filter *= 0.43  # 70/30
    
    # ─── PARCA 6: Shannon Harvest (ranging) ───
    adx = perception.chart_features.get("mtf_1h_adx", 25)
    if adx < 20:
        regime_filter *= 0.3  # yon trade kucuk, MM kari ayri
    
    # ─── PARCA 7: Impact Constraint (D2 — YENI) ───
    adv = market_data.get("average_daily_volume", 1e9)
    depth = market_data.get("orderbook_depth_2pct", 1e6)
    participation = base_size / (adv * 0.042 + 1e-8)
    impact_bps = 0.34 * market_data.get("sigma_daily_bps", 200) * math.sqrt(participation)
    fee_bps = 4.0  # maker round-trip
    total_cost = impact_bps + fee_bps
    expected_alpha = abs(vwtsmom) * 100  # rough alpha estimate in bps
    if total_cost >= expected_alpha and expected_alpha > 0:
        return 0.0  # cost > alpha → SKIP
    cost_ratio = total_cost / (expected_alpha + 1e-8)
    impact_mult = max(0.1, 1.0 - cost_ratio)
    
    # Depth cap
    depth_cap = depth * 0.05  # max %5 of visible depth
    
    # ─── PARCA 8: Hawkes Cascade Guard (D1 — YENI) ───
    branching_ratio = market_data.get("hawkes_branching_ratio", 0.5)
    if branching_ratio >= 0.95:
        return 0.0  # VETO — cascade imminent
    elif branching_ratio >= 0.9:
        hawkes_mult = 0.2
    elif branching_ratio >= 0.8:
        hawkes_mult = 0.4
    elif branching_ratio >= 0.7:
        hawkes_mult = 0.6
    else:
        hawkes_mult = 1.0
    
    # ─── PARCA 9: Confidence Integral (HQ-1 — YENI) ───
    pfield = get_pheromone_field()
    conf_integral = pfield.read_integral("prediction", window_seconds=14400)  # 4h
    conf_gradient = pfield.read_gradient("prediction")
    
    # Sustained confidence bonus: 4 saattir ayni yonde → +20%
    integral_mult = 1.0 + 0.2 * min(1.0, abs(conf_integral))
    # Gradient: conviction artiyorsa bonus, dusuyorsa penalty
    gradient_mult = 1.0 + 0.1 * math.tanh(conf_gradient * 100)
    
    # ─── PARCA 10: Forgone Alpha Adjustment (HQ-11 — YENI) ───
    forgone_alpha = get_forgone_alpha(pair, regime)
    if forgone_alpha > 2.0:
        # Cok kacirilmis kar → cesur ol
        forgone_mult = 1.15
    elif forgone_alpha < -1.0:
        # Kacirilmis zararlar → ihtiyatli
        forgone_mult = 0.90
    else:
        forgone_mult = 1.0
    
    # ═══ FINAL SIZE ═══
    raw_size = (base_size * confidence * hormonal_scalar * dream_bonus * 
                regime_filter * impact_mult * hawkes_mult * 
                integral_mult * gradient_mult * forgone_mult)
    
    # Portfolio constraints (Constitution)
    max_position = portfolio.total * 0.03
    min_position = 0.10
    
    final_size = max(min_position, min(raw_size, max_position, depth_cap))
    
    return final_size
```

### Task 10b: Hawkes — 7 Strateji Detayi (D1 Ajani)

**Dokumanda branching ratio + intensity sizing var. Diger 5 strateji:**

**Strateji 3: Multivariate Cross-Asset Contagion**
```python
# 4-boyutlu Hawkes: BTC trades × altcoin trades × liquidation × large-lot
# Cross-excitation matrix alpha_ij: "BTC likidasyon → altcoin panik" olcer
# Yang et al. 2025: crypto cross-contagion %20 daha guclu (geleneksel piyasadan)

class MultivariateHawkes:
    def __init__(self, n_dims=4):
        self.mu = np.zeros(n_dims)           # baseline intensity per dim
        self.alpha = np.zeros((n_dims, n_dims))  # cross-excitation matrix
        self.beta = np.zeros((n_dims, n_dims))   # decay matrix
        self.R = np.zeros((n_dims, n_dims))      # recursive state
    
    def update(self, dim: int, timestamp: float):
        """O(D^2) per event — D=4 icin 16 islem."""
        for i in range(self.n_dims):
            for j in range(self.n_dims):
                dt = timestamp - self.last_event_time[j]
                self.R[i][j] = (math.exp(-self.beta[i][j] * dt) * 
                               (self.R[i][j] + (1 if j == dim else 0)))
    
    def get_contagion_risk(self) -> dict:
        """Cross-asset contagion matrix snapshot."""
        return {
            "btc_to_alt": self.alpha[0][1] * self.R[0][1],
            "liq_to_trades": self.alpha[2][0] * self.R[2][0],
            "liq_self_excite": self.alpha[2][2] * self.R[2][2],  # cascade risk
        }
```

**Strateji 4: Hawkes-LOB Return Forecasting (Capponi et al. 2023)**
- LOB data + Hawkes intensity → return sign prediction
- "Base Imbalance" regressor: Hawkes-modeled LOB state'ten turetilmis
- Crypto-specific: Bybit L2 orderbook WebSocket gerektirir
- **Prereq:** lob_encoder.py entegre edilmis olmali (Fix 8)

**Strateji 5: Order Flow Imbalance Forecasting (Bouri et al. 2024)**
```python
# Bivariate Hawkes: BUY arrivals × SELL arrivals ayri
# Cross-excitation: buy→buy (momentum), sell→buy (mean reversion)
# OFI forecast = near-term buy-sell imbalance prediction

class BivariateOFIHawkes:
    def __init__(self):
        self.buy_intensity = HawkesKernel()   # dim 0
        self.sell_intensity = HawkesKernel()  # dim 1
    
    def predict_ofi(self, horizon_seconds=60) -> float:
        """Predict Order Flow Imbalance in next horizon."""
        expected_buys = self.buy_intensity.expected_count(horizon_seconds)
        expected_sells = self.sell_intensity.expected_count(horizon_seconds)
        return (expected_buys - expected_sells) / (expected_buys + expected_sells + 1e-8)
```

**Strateji 6: Hawkes(p,q) Regime Classification (Wehrli 2021)**
- Flexible exogenous arrival + time-dependent feedback
- EM algorithm ile fit
- Endogenous/exogenous ratio SHIFT → regime transition 10-30dk ONCE
- **Complexity:** HIGH — EM iterative, CPU costly. Offline analysis icin uygun.

**Strateji 7: Markov-Modulated Hawkes — Manipulation Detection (Eur.J.Finance 2026)**
- Hidden Markov chain + Hawkes: normal vs manipulation state
- Wash trading, spoofing, layering detection
- Kucuk-cap crypto'da ozellikle degerli (WLFI, ALCH tipi pair'ler)
- **Complexity:** VERY HIGH. Arastirma asamasinda, Sprint 3B+ hedefi.

**Hawkes Danger Thresholds (konsolide):**

| Branching Ratio n | Piyasa Durumu | Sizing Aksiyonu |
|-------------------|--------------|-----------------|
| n < 0.5 | Dusuk reflexivite, exogenous-driven | Normal (1.0x) |
| 0.5 ≤ n < 0.7 | Orta endogenite | Hafif azalt (0.8x) |
| 0.7 ≤ n < 0.8 | Yuksek reflexivite | Dikkat (0.6x) |
| 0.8 ≤ n < 0.9 | TEHLIKE — criticality yakin | Clamp (0.4x) |
| 0.9 ≤ n < 0.95 | ON-KRIZ — cascade basliyor | Acil (0.2x) |
| n ≥ 0.95 | KRITIK — cascade kacinilmaz | **VETO** (yeni trade YOK) |

**Intensity-based sizing (smooth, threshold-free):**
```python
size_mult = min(1.0, lambda_baseline / lambda_current)
# lambda 3x baseline → sizing %33
# lambda 5x baseline → sizing %20
# Threshold-based'den DAHA ILAÇ — surekli, yavasca adapte olur
```

**Library Karsilastirmasi:**

| Library | Pros | Cons | Tavsiye |
|---------|------|------|---------|
| `tick` | En kapsamli, C++ backend, multivariate, MLE+EM | Son PyPI 2020, Python 3.11? | PRIMARY — import test yap |
| `HawkesPyLib` | Numba JIT, basit API | Sadece univariate | Fallback (univariate) |
| Custom numpy | Tam kontrol, 0 dependency | MLE kendin yaz | O(1) recursive icin (trivial) |
| `hawkeslib` | Iyi dokumantasyon | Az bakimli | Referans |

**RAM:** ~51MB (MLE refit peak). **CPU:** ~2 sn/saat (10K event refit).

### Task 12: 4-Layer Regime Detection (B5 Ajani)

**ADX primary'den CONFIRMATION'a indirgeniyor.**

```python
class FourLayerRegimeDetector:
    """
    B5: 4-katmanli ensemble regime detection.
    Her katman farkli lead time: en hizli → en yavas.
    ADX artik Layer 3 (confirmation), Layer 0 degil.
    """
    
    def __init__(self):
        self.vpin_analyzer = VPINAnalyzer()        # Layer 0: 1-3h lead
        self.bocpd = ScoreDrivenBOCPD()             # Layer 1: 2-6h lead
        self.causal_monitor = CausalEdgeMonitor()   # Layer 2: 4-12h lead
        self.adx_classifier = RegimeClassifier()    # Layer 3: confirmation
    
    def detect(self, market_data: dict) -> dict:
        # Layer 0: VPIN + Order Flow (FASTEST)
        vpin = self.vpin_analyzer.get_toxicity_composite()
        vpin_trend = self.vpin_analyzer.get_trend()  # accelerating?
        layer0_alert = vpin > 0.55 and vpin_trend > 0  # 2+ saat yuksek
        
        # Layer 1: Score-Driven BOCPD (parameter drift)
        bocpd_result = self.bocpd.update(market_data["returns"])
        residual_time = bocpd_result["median_residual_time"]
        score_magnitude = bocpd_result["score_magnitude"]
        layer1_alert = residual_time < 8 or score_magnitude > 2.0
        
        # Layer 2: Causal Edge Instability
        edge_stability = self.causal_monitor.compute_instability_index()
        layer2_alert = edge_stability > 0.25  # >25% edges breaking
        
        # Layer 3: ADX + EMA (SLOWEST — confirmation only)
        adx_regime = self.adx_classifier.classify(market_data)
        
        # Fusion: log-odds weighted
        alert_count = sum([layer0_alert, layer1_alert, layer2_alert])
        
        if alert_count == 0:
            regime_change_prob = 0.10
            status = "STABLE"
        elif alert_count == 1:
            regime_change_prob = 0.50
            status = "MICROSTRUCTURE_ANOMALY"
        elif alert_count == 2:
            regime_change_prob = 0.75
            status = "REGIME_CHANGE_LIKELY"
        else:
            regime_change_prob = 0.95
            status = "REGIME_CHANGE_IMMINENT"
        
        return {
            "current_regime": adx_regime,
            "regime_change_prob": regime_change_prob,
            "status": status,
            "layers": {
                "L0_vpin": {"alert": layer0_alert, "value": vpin},
                "L1_bocpd": {"alert": layer1_alert, "residual_time": residual_time},
                "L2_causal": {"alert": layer2_alert, "instability": edge_stability},
                "L3_adx": {"regime": adx_regime},
            },
            "sizing_modifier": 1.0 - regime_change_prob * 0.5,  # belirsizlikte kucul
        }
```

### Task 13: Kronos-mini + Foundation Model Landscape (C5 Ajani)

**7 Foundation Model Karsilastirmasi (C5 ajani tam tarama):**

| # | Model | Param | Pre-train Data | Quantile? | Crypto? | CPU? | Lisans |
|---|-------|-------|---------------|-----------|---------|------|--------|
| 1 | **Kronos-mini** | 4.1M | 12B+ K-line, 45 exchange | Hayir | **EVET (native)** | Olasi | MIT |
| 2 | **TTM R2** | <1M | IBM generic TS | Hayir | Hayir | **EVET** | Apache |
| 3 | **Chronos-Bolt** | 48M | Generic TS + synthetic | **EVET (P10-P90)** | Hayir | EVET | Apache |
| 4 | **Moirai 2.0** | ~11M (Small) | 36M series | **EVET** | Hayir | Olasi | Apache |
| 5 | **MOMENT** | ~40M | Multi-task TS | Hayir | Hayir | Olasi | MIT |
| 6 | **Lag-Llama** | ~10M | Probabilistic TS | **EVET (full dist)** | Hayir | EVET | Apache |
| 7 | **TimesFM 2.5** | 200M | Google internal | Hayir | Hayir | **HAYIR** | Proprietary |

**Neden Kronos-mini secildi:**
- **TEK finance-native model**: OHLCV candlestick tokenizer (2^20 vocabulary, coarse+fine 10-bit)
- Crypto exchange verileriyle pre-trained (Binance, Coinbase, Bybit dahil)
- RankIC: generic TSFM'lerden **%93 daha iyi** (arXiv 2508.02739, AAAI 2026)
- 4.1M param = TTM'den sadece 4x buyuk, Chronos'tan 12x kucuk
- MIT lisans, HuggingFace'te acik

**Generic TSFM'ler neden yetersiz (arXiv 2511.18578, Rahimikia et al.):**
```
Zero-shot TSFM'ler finansal veride:
  R-squared: -2.80% (NEGATIF — random'dan KOTU)
  Directional accuracy: <%50

CatBoost (window=252): Sharpe 6.79
Chronos + financial factors + augmentation: Sharpe 6.78

SONUC: Generic pre-training finance'a TRANSFER ETMIYOR.
       Domain-specific pre-training (Kronos) veya fine-tuning SART.
```

**Fine-Tuning Stratejileri (arXiv 2409.11302, NeurIPS 2024):**

| Teknik | Trainable Params | Performans | Tavsiye |
|--------|-----------------|------------|---------|
| **BitFit** | 256-768 | Chronos-Base'de EN IYI MSE | TTM icin ideal |
| **FourierFT** | ~1K | SOTA'yi gecti | Deneysel |
| **VeRA** | ~2K | LoRA'ya yakin | Memory-efficient |
| **LoRA** | ~100K+ | Standard baseline | Chronos icin AutoGluon API |
| **Full fine-tune** | Hepsi | En iyi absolute | TTM (<1M) icin OK |
| **Head retraining** | ~10K | Hizli, stabil | En guvenli baslangic |

**FiTEM Selective Forecasting (AALTD 2025):**
```python
# Model "bilmiyorum" diyebilmeli — her tahmin ZORUNLU degil
class SelectiveForecaster:
    def predict_with_abstention(self, x):
        pred, confidence = self.model(x)
        if confidence < self.abstention_threshold:
            return None, "ABSTAIN"  # tahmin yapma, sizing = 0
        return pred, confidence
    
    # Abstention ile hata azalir:
    # %10 coverage (en emin %10): %56.4 hata azalmasi
    # %50 coverage: %35.4 hata azalmasi
    # %100 coverage (hic abstain yok): %0 (baseline)
```

**Triple → Quad Perception:**

```python
class QuadPerception:
    """
    C5: TTM + Chronos + Kronos + CatBoost = 4 model ensemble.
    Her model farkli acidan bakar:
      TTM: directional signal (MLP-Mixer, 3ms, <1M param)
      Chronos: uncertainty quantification (P10/P50/P90)
      Kronos: financial grammar (OHLCV-native, 4.1M param)
      CatBoost: final decision (gradient boosting, 30 feature)
    
    Meta-learner: Ridge regression stacking (Super Learner optimality)
    """
    
    def __init__(self):
        self.ttm = TTMPerception()           # mevcut
        self.chronos = ChronosPerception()   # mevcut
        self.kronos = KronosPerception()     # YENI
        self.catboost = CatBoostModel()      # mevcut
        self.meta_learner = None             # Ridge regression
    
    def perceive(self, pair: str, ohlcv: pd.DataFrame) -> dict:
        # 4 model paralel calistir
        ttm_result = self.ttm.predict(ohlcv)
        chronos_result = self.chronos.predict(ohlcv)
        kronos_result = self.kronos.predict(ohlcv)  # YENI
        catboost_result = self.catboost.predict(ohlcv)
        
        # Meta-learner stacking
        if self.meta_learner:
            features = np.array([
                ttm_result["direction"],
                chronos_result["p50"],
                kronos_result["predicted_return"],
                catboost_result["probability"],
            ])
            final_confidence = self.meta_learner.predict(features.reshape(1, -1))[0]
        else:
            # Cold start: simple average
            final_confidence = np.mean([
                ttm_result.get("confidence", 0.5),
                chronos_result.get("confidence", 0.5),
                kronos_result.get("confidence", 0.5),
                catboost_result.get("probability", 0.5),
            ])
        
        return {
            "signal": self._determine_signal(final_confidence),
            "confidence": final_confidence,
            "ttm": ttm_result,
            "chronos": chronos_result,
            "kronos": kronos_result,
            "catboost": catboost_result,
            "disagreement": self._compute_disagreement(
                ttm_result, chronos_result, kronos_result, catboost_result
            ),
            "shap_narrative": catboost_result.get("shap_narrative", ""),
        }
```

**Kronos-mini spesifikasyonlari:**
- HuggingFace: `NeoQuasar/Kronos-mini`
- 4.1M parametre, MIT lisans
- OHLCV-native tokenizer (2^20 vocabulary)
- 12B+ K-line kaydinda pre-trained (45+ global exchange)
- Context: 2048 (TTM'in 512'sinden 4x buyuk)
- **CPU benchmark GEREKLI** — 32GB server'da inference suresi olculecek

**RAM Butcesi:**
```
TTM:         ~50MB (mevcut)
Chronos:     ~200MB (mevcut)
Kronos-mini: ~20MB (4.1M × 4 byte = 16.4MB + overhead)
CatBoost:    ~5MB (mevcut)
Toplam:      ~275MB (mevcut 250MB + 25MB artis)
```

---

## SPRINT 3B: LLM DEEP + ADVANCED + RESEARCH

> Sprint 3A TAMAMLANDIKTAN ve 500+ trade sonucu biriktikten SONRA baslanacak.
> Detayli arastirma bulgulari TEORIK TEMELLER bolumunde.
> Burada implementasyon detaylari.

### Task 15: RLAIF d-RLAIF — LLM-as-Judge (C4 Ajani)

**Mimari:** post_trade_court verdicts → 3 LLM rubric scoring → RL reward eklentisi

**Rubric-Based 5-Dimension Scoring:**
```python
class RLAIFRewardGenerator:
    RUBRIC = {
        "signal_quality":  {"weight": 0.25, "desc": "Confidence calibrated mi?"},
        "sizing_quality":  {"weight": 0.25, "desc": "Pozisyon edge'e orantili mi?"},
        "timing_quality":  {"weight": 0.20, "desc": "Entry/exit zamanlama"},
        "risk_management": {"weight": 0.15, "desc": "Stoploss + leverage sinirlar"},
        "regime_alignment":{"weight": 0.15, "desc": "Trade regime'e uygun mu?"},
    }
    
    def score_trade(self, verdict: dict, context: dict) -> dict:
        """3 LLM'den rubric-based scoring + WCO deconflicting."""
        scores = {}
        for provider in [self.gemini, self.groq, self.mistral]:
            scores[provider.name] = provider.invoke(
                self._build_rubric_prompt(verdict, context)
            )
        
        # Worst-Case Optimization: en pesimist judge kazanir
        wco_scores = {}
        for dim in self.RUBRIC:
            wco_scores[dim] = min(s[dim] for s in scores.values())
        
        composite = sum(wco_scores[d] * self.RUBRIC[d]["weight"] 
                       for d in self.RUBRIC) / 5.0
        return {"composite": composite * 2 - 1, "per_dim": wco_scores}  # [-1, +1]

# RL entegrasyonu:
total_reward = 0.80 * env_reward + 0.20 * llm_reward  # LLM ASLA >%50
```

**6 Reward Hacking Senaryosu + Mitigation:**
| # | Hack | Mitigation |
|---|------|-----------|
| 1 | PnL Avoidance (hic trade etme → 0 kayip) | Forgone PnL tracking |
| 2 | Confidence Anchoring (hep 0.50) | Entropy penalty on confidence distribution |
| 3 | Regime Exploitation (loss → "regime mismatch") | Verify regime with independent detector |
| 4 | LLM Sycophancy | Multi-LLM ensemble + WCO |
| 5 | Verbosity Bias | Fixed-format rubric output |
| 6 | Goodhart Overoptimization | Temporal holdout + correlation monitoring |

**Maliyet:** 15 trade/gun × 3 LLM = 45 call/gun. Tum free tier'lar icinde.

### Task 16: LLM Strategy Researcher (G2 Ajani)

(Detaylar TEORIK TEMELLER G2 bolumunde — prompt templates, 6-gate pipeline, hypothesis_history tablosu)

**Haftalik Cycle:**
```
Pazar gecesi CatBoost retrain SONRASI:
1. GATHER: PostTradeCourt + Ablation + Forgone + DecisionContracts
2. GENERATE: LLM → 3-5 structured JSON hipotez
3. GATE 1-2: Complexity + Novelty check
4. BACKTEST: Her hipotez icin walk-forward (5+ fold)
5. GATE 3-4: Deflated Sharpe + OOS holdout
6. REFLECT: LLM review of past hypothesis outcomes
7. DEPLOY: Shadow → Paper → Live (graduated)
```

### Task 17: Cross-Provider LLM Ensemble (G1 Ajani)

```python
async def ensemble_invoke(self, messages, n_judges=3):
    """3 farkli PROVIDER'dan (ayni model degil!) parallel query."""
    # Thompson Sampling ile 3 farkli provider sec
    slots = self._select_diverse_providers(n=n_judges)
    
    # Parallel query
    results = await asyncio.gather(*[
        slot.ainvoke(messages) for slot in slots
    ])
    
    # Confidence-weighted voting
    weights = [s.alpha / (s.alpha + s.beta) for s in slots]
    # ... weighted fusion
    
    # CDR (Conflict Detection Rate) = regime confusion detector
    signals = [parse_signal(r) for r in results]
    cdr = len(set(signals)) / len(signals)  # 1.0 = full disagreement
    if cdr > 0.66:
        logger.warning("[Ensemble] HIGH CDR — models confused about regime")
        # Sizing reduction signal
```

**KRITIK KONTROL:** 10 Gemini key ayni projeden mi farkli projelerden mi?
- Ayni proje: 10 key × 250 RPD = 250 RPD (key sayisi onemli DEGIL)
- Farkli proje: 10 key × 250 RPD = 2,500 RPD (10x FARK!)

### Task 18: Adaptive LLM Router (G5 Ajani)

(Detaylar TEORIK TEMELLER G5 bolumunde — 12 enhancement, free tier stacking)

**P0-P2 Quick Wins:**
1. RPD tracking + proactive slowdown at %80 (~60 satir)
2. Discounted TS: `alpha = 0.997 * alpha + reward` (~5 satir)
3. Query-type classifier + task-aware routing (~80 satir)

### Task 19: Sleep-Wake Cycle (I4 Ajani)

(Detaylar TEORIK TEMELLER I4 bolumunde — Borbely, SMDP, Whittle, %42 tasarruf)

**Implementation sirasi:**
1. P0: cerebellum → CircadianRhythm baglantisi (hardcoded kaldir)
2. P1: Process S + Process C implement et
3. P2: 3 mod (FULL/LIGHT/DEEP) scheduler'a ekle
4. P3: SMDP controller (once rule-based, sonra R-learning)

### Task 20: Foundation Model Fine-Tuning (C5 Ajani)

- **TTM:** IBM few-shot recipe, %5 training data. Head retraining.
- **Chronos-Bolt:** AutoGluon LoRA (`fine_tune_lora_config={"r": 8}`) VEYA BitFit (768 param!)
- **Kronos-mini:** Zaten crypto pre-trained. Fine-tune gerekli olmayabilir.
- **FiTEM:** Selective forecasting gate — model "abstain" → sizing kucult

### Task 21: Decision Transformer (C2 Ajani)

- GPT-2 124M + LoRA rank=16 = 900K trainable param
- decision_contract.py corpus → (return-to-go, state, action) tuples
- **Min 500 trade** with outcome data (1289 mevcut — YETERLI)
- Return-conditioned: "next 24h'de %2 istiyorum" → optimal params
- CPU: ~600MB RAM, 2-10h training. Pazar gecesi LoRA penceresi.

### Task 22: Multi-Modal Fusion Fix (G4 Ajani)

(Detaylar TEORIK TEMELLER G4 bolumunde — 7 bug, AECF, MSGCA, GMU)

**Oncelik:** P0 (learnable tokens + attention mask) → P1 (modality dropout + Jina) → P2 (training pipeline)

### Task 23: Adversarial Self-Play (I2 Ajani)

(Detaylar TEORIK TEMELLER I2 bolumunde — ExploiterAgent, exploit_archive, 7 failure mode)

**~200-300 satir ekleme, 0 yeni servis:**
- ExploiterAgent + DefenderAgent AGENT_REGISTRY'ye
- R2b adversarial round
- exploit_archive tablosu
- Nightly batch: stratejiyi exploit archive'a karsi test

### Task 24: Autopoietic Integrity (F3 Ajani)

(Detaylar TEORIK TEMELLER F3 bolumunde — 8 teori, AII 4-layer, constitution eki)

**Oncelik:**
- P0: _evaluate_fitness'i gercek trade PnL'e bagla
- P1: Constitution'a identity_limits ekle
- P2: Birth snapshot save/load
- P3: AII compute + graduated response

### Task 25: Trading as Language (I5 Ajani)

(Detaylar TEORIK TEMELLER I5 bolumunde — 10 strateji, graduated pipeline)

**Hemen yapilabilir:**
- N-gram frequency analysis (~50 satir)
- Conditional probability grammar rules (~100 satir)
- PrefixSpan sequential pattern mining (~60 satir)
- 1289 trade YETERLI

---

---

## EK ARASTIRMA BULGULARI: B1, B2, B3, D3, E3, E5, F2, F5, I3

### B1: Causal Reward Shaping — En Dusuk Efor, En Yuksek Etki

**Bareinboim (ICML 2025):** Causal graph edge'lerinden reward shaping fonksiyonu turet.
PCMCI+ causal edge strength → value upper bound → potential-based reward.

```python
# causal_engine.py'den edge'leri al:
edges = causal_engine.get_significant_edges(pair, min_strength=0.3)
# Ornek: funding_rate → pnl (strength=0.72, lag=2)

# RL reward'a causal bonus ekle:
def causal_reward_shaping(state, action, next_state, causal_edges):
    """Bareinboim: potential-based shaping from causal graph."""
    phi_s = sum(edge.strength * state[edge.source_idx] 
                for edge in causal_edges)
    phi_s_next = sum(edge.strength * next_state[edge.source_idx] 
                     for edge in causal_edges)
    return GAMMA * phi_s_next - phi_s  # potential difference
    # Garanti: optimal policy DEGISMEZ (Ng et al. 1999 theorem)
```

**Neden en dusuk efor:** Mevcut PCMCI+ edge'leri + mevcut RL reward → TEK bir fonksiyon ekleme.
**Neden en yuksek etki:** RL sparse reward problemini cozer (trade 1-4 saat surer, reward gecikmeli).

### B2: Causal Transportability — Cross-Pair Transfer

**Pearl & Bareinboim (2011):** Causal bilgi bir domain'den baskasina NE ZAMAN transfer edilebilir?

```
BTC'de kesfedilen: funding_rate → pnl (causal, strong)
ETH'ye transfer edilebilir mi?

S-Admissibility kosulu:
  1. Ayni causal mechanism mevcut mu? (funding rate ikisinde de var → EVET)
  2. Distribution shift nerede? (volatility farkli → adjustment gerekli)
  3. Selection bias var mi? (liquidity farkli → pair-specific katsayi)

Transportability formula:
  P_ETH(pnl | do(action)) = Σ_S P_BTC(pnl | do(action), S=s) × P_ETH(S=s)
  S = farklilik yaratan degiskenler (selection variables)
```

**Pratik:** Yeni listelenen coin'de SIFIR trade → BTC causal bilgisini TRANSFER et, pair-spesifik
katsayi adjustment ile. 30+ pair icin causal knowledge PAYLAS.

### B3: Counterfactual Engine — 3 Failure Point

**counterfactual_engine.py (623 satir) sorunlari:**
1. `_build_causal_model()` statik SCM kullaniyor — gercek PCMCI+ graph'i OKUMUYOR
2. Intervention degiskenleri hardcoded: `confidence=[0.3,0.4,...,0.8]` — gercek dagilimdan ORNEKLEMIYOR
3. DoWhy `identify_effect()` cagrisinda `graph` parametresi GML string — PCMCI+ ciktisi farkli format

**Fix:** causal_engine → counterfactual_engine zincirini kur. PCMCI+ graph'i DoWhy'a cevirecek adapter yaz.

### D3: Market Making — 14 Prerequisite (Simdi Entegre Edilemez)

**market_maker_mode.py (233 satir) akademik olarak solid (Stoikov GLFT + hormonal gamma)**
AMA 14 prerequisite eksik. En kritik 3:

1. **Strategy hook YOK:** `custom_entry_price` / `custom_exit_price` callback'leri baglı degil
2. **VPIN circuit breaker YOK:** Gercek VPIN (Task 10) ile baglanmali — toxic flow'da MM DURUR
3. **RegimeClassifier baglanmamis:** Standalone ADX kullanıyor, mevcut regime_classifier'ı okumuyor

**Sprint 3B'de SARTLI:** Sadece ranging regime (ADX<20) + liquid pair'lerde + VPIN<0.55 iken aktif.

### E3: Shannon Volatility Harvest — Revize Edilmis

**ARGE'deki premium beklentisi ASIRI IYIMSER:**
```
ARGE tahmini:  %4.5-8.0 yillik (sigma=%60-80 ile)
Gercek (E3):   %1.5-3.1 yillik (BTC declining vol + transaction costs)

Neden fark:
  1. BTC volatilitesi dusuyor (2021: %100+ → 2026: %60-70)
  2. Transaction costs: Bybit VIP0 round-trip 4-11 bps
  3. Autocorrelation pozitif olunca rebalancing ZARAR (momentum keser)
  4. Korelasyonlar yuksekken diversifikasyon premium duser
```

**Duzeltilmis strateji:**
- Sabit haftalik DEGiL → threshold-based rebalancing (%5-10 drift)
- ADX<20 VE autocorrelation < 0.1 VE correlation < 0.7 iken AKTIF
- Tahmini gercek premium: %1.5-2.0/yil — kucuk ama BEDAVA (yon tahmini gereksiz)

### E5: Momentum Detay — Dobrynskaya + Barroso

**Dobrynskaya Timing Map (2021, SSRN 3913263):**
```
Trade yas (gun):  Strateji:
  0-10             MOMENTUM (peak alpha, agresif sizing)
  10-14            MOMENTUM ama trail SIKILAS (reversal riski artiyor)
  14-21            EXIT ZONE (momentum bitmek uzere)
  21-42            REVERSAL (contrarian alpha basliyor)
  42+              Weak reversal, azalan getiri
```

**Risk-Managed Momentum (Barroso & Santa-Clara, Springer 2025):**
```python
# Volatility-scaled momentum:
position = signal * sigma_target / sigma_realized_6mo

# Crypto-specific bulgu: risk management augments RETURNS (crash mitigation degil)
# Sharpe: 1.12 → 1.42 (+%30)
# Neden: crypto'da uzun momentum crash'leri YOK, vol scaling sinyal netligi artirir
```

### F2: Stigmergy Detay — Adaptive Half-Life + MMAS

**Adaptive half-life (Mavrovouniotis 2013-2014):**
```python
# Sabit half-life SUBOPTIMAL dynamic ortamlarda
# Regime'a gore adapte et:

def adaptive_half_life(base_half_life: float, regime: str) -> float:
    REGIME_MULTIPLIERS = {
        "calm": 2.0,       # uzun hafiza — sakin piyasada gecmis onemli
        "normal": 1.0,     # varsayilan
        "volatile": 0.5,   # kisa hafiza — hizli adaptasyon
        "crisis": 0.25,    # cok kisa — eski bilgi hizla eskir
    }
    return base_half_life * REGIME_MULTIPLIERS.get(regime, 1.0)
```

**MMAS Bounds (Stutzle & Hoos 2000):**
```
TAU_MIN = 0.01  — hicbir sinyal TAMAMEN silinmez (minimum exploration garanti)
TAU_MAX = 10.0  — hicbir sinyal DOMINE edemez (saturation korunmasi)

Hunt et al. 2019: N>300 agent'ta pheromone koordinasyon RANDOM'dan KOTU
Bizim: ~30 modul × ~14 sinyal tipi — esik altinda ama buyumeye dikkat
```

**Genuinely Novel Bulgu (F2):** Stigmergic pheromone-based coordination for trading modules
literaturde BULUNAMADI. En yakin: ColCOS-Phi (robotik, IEEE 2021), S-MADRL (RL, 2025).

### F5: IIT Phi — PhiC Approximation

**PhiC (Virmani & Nagaraj 2019):** Exact Phi intractable (super-exponential).
PhiC lineer olceklenir, Python implementasyonu var.

```python
# PhiC hesaplama:
# 1. Modul aktivasyon zaman serilerini al
# 2. TUM modullerin birlesik serisinin ETC'sini hesapla
# 3. Her modulun AYRI ETC'sini hesapla
# 4. PhiC = birlesik_ETC - toplam(ayri_ETC'ler)

from etcpy import ETC  # pip install etcpy

def compute_phic(module_activations: Dict[str, np.ndarray]) -> float:
    """PhiC: irreducible integration measure."""
    # Birlesik seri
    combined = np.concatenate(list(module_activations.values()))
    etc_whole = ETC(combined)
    
    # Parcali ETC'ler
    etc_parts = sum(ETC(act) for act in module_activations.values())
    
    # Normalize
    phic = (etc_whole - etc_parts) / (etc_whole + 1e-8)
    return phic  # 0 = bagimsiz moduller, 1 = tam entegrasyon
```

**Falk et al. (2018, PLOS ONE):** Phi ↔ collective intelligence korelasyonu r=0.370 (p=0.003).
Moderately significant — diagnostic olarak kullan, trading signal olarak DEGIL.

### I3: SIR Sentiment Contagion — Tam Model

**Epidemiyolojik SIR piyasa sentiment'ine uygulanmis:**
```
S(t) = Susceptible: henuz sentiment dalgasindan etkilenmemis trader'lar
I(t) = Infected: aktif olarak fear/greed yayan trader'lar  
R(t) = Recovered: sentiment normallesme, pozisyon kapamis

dS/dt = -beta * S * I / N          (yeni enfeksiyon)
dI/dt = beta * S * I / N - gamma * I  (enfeksiyon - iyilesme)
dR/dt = gamma * I                    (iyilesme)

R_t = beta * S / (gamma * N)        (reproduction number)
R_t > 1 → panik YAYILIYOR → contrarian opportunity
R_t < 1 → panik SOGUYOR → trend following safe
R_t = 1 → tipping point
```

**Pratik F&G Velocity (HEMEN yapilabilir, 0 yeni data source):**
```python
# fear_and_greed tablosundan:
def fng_velocity_and_acceleration():
    fng_values = get_fng_history(days=7)
    
    velocity = np.diff(fng_values)       # dFNG/dt
    acceleration = np.diff(velocity)     # d²FNG/dt²
    
    # Peak panic detection:
    if velocity[-1] < -10 and acceleration[-1] > 0:
        # Panic YAVAŞLIYOR → dip olabilir → contrarian BUY signal
        return "PANIC_DECELERATING"
    elif velocity[-1] < -10 and acceleration[-1] < 0:
        # Panic HIZLANIYOR → henuz erken → bekle
        return "PANIC_ACCELERATING"
    elif velocity[-1] > 10 and acceleration[-1] < 0:
        # Greed YAVASLIYOR → peak olabilir → contrarian SELL signal
        return "GREED_DECELERATING"
```

---

## KONSOLIDE DB SCHEMA — TUM YENI TABLOLAR

```sql
-- ═══ SPRINT 3A YENI TABLOLAR ═══

-- Per-pair Bayesian Kelly (Prensip 0, Task 1)
CREATE TABLE IF NOT EXISTS bayesian_kelly_per_pair (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    pair TEXT NOT NULL, regime TEXT NOT NULL DEFAULT '_global',
    alpha REAL DEFAULT 2.0, beta_param REAL DEFAULT 2.0,
    avg_win REAL DEFAULT 0.0, avg_loss REAL DEFAULT 0.0,
    n_trades INTEGER DEFAULT 0, annual_volatility REAL,
    vol_of_vol REAL, last_sharpe REAL, updated_at TEXT,
    UNIQUE(pair, regime)
);

-- Argument quality scoring (Fix 2C, J4)
CREATE TABLE IF NOT EXISTS argument_quality (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_type TEXT NOT NULL, argument_pattern TEXT NOT NULL,
    regime TEXT NOT NULL, times_used INTEGER DEFAULT 0,
    times_correct INTEGER DEFAULT 0, avg_pnl_when_used REAL DEFAULT 0.0,
    quality_score REAL DEFAULT 0.5, updated_at TEXT,
    UNIQUE(agent_type, argument_pattern, regime)
);

-- Per-pair confidence threshold (Fix 6, HQ-11)
CREATE TABLE IF NOT EXISTS pair_thresholds (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    pair TEXT NOT NULL, regime TEXT NOT NULL DEFAULT '_global',
    confidence_threshold REAL DEFAULT 0.50,
    forgone_alpha_7d REAL DEFAULT 0.0,
    last_adjusted TEXT, adjustment_reason TEXT,
    UNIQUE(pair, regime)
);

-- 4-Layer regime state (Task 12)
CREATE TABLE IF NOT EXISTS regime_layers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    pair TEXT NOT NULL, timestamp TEXT NOT NULL,
    layer0_vpin REAL, layer0_alert BOOLEAN,
    layer1_bocpd_residual REAL, layer1_alert BOOLEAN,
    layer2_causal_instability REAL, layer2_alert BOOLEAN,
    layer3_adx_regime TEXT,
    regime_change_prob REAL, sizing_modifier REAL,
    UNIQUE(pair, timestamp)
);

-- Hawkes process state (Task 10, HQ-13)
CREATE TABLE IF NOT EXISTS hawkes_state (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    pair TEXT NOT NULL, timestamp TEXT NOT NULL,
    branching_ratio REAL, intensity_current REAL,
    intensity_baseline REAL, alpha REAL, beta REAL,
    last_refit TEXT,
    UNIQUE(pair)
);

-- ═══ SPRINT 3B YENI TABLOLAR ═══

-- RLAIF reward history (Task 15)
CREATE TABLE IF NOT EXISTS rlaif_rewards (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    trade_id INTEGER, timestamp TEXT,
    signal_quality REAL, sizing_quality REAL,
    timing_quality REAL, risk_management REAL,
    regime_alignment REAL, composite REAL,
    provider_scores TEXT,  -- JSON: {gemini: X, groq: Y, mistral: Z}
    env_reward REAL, total_reward REAL
);

-- Hypothesis history (Task 16, G2)
CREATE TABLE IF NOT EXISTS hypothesis_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    hypothesis_id TEXT UNIQUE,
    parameter TEXT, current_value REAL, proposed_value REAL,
    mechanism TEXT, falsification TEXT, affected_pairs TEXT,
    is_sharpe REAL, oos_sharpe REAL,
    deflated_sharpe REAL, n_hypotheses_in_batch INTEGER,
    validation_result TEXT,
    deployed BOOLEAN DEFAULT FALSE, deployed_at TEXT,
    shadow_period_sharpe REAL, live_period_sharpe REAL,
    rolled_back BOOLEAN DEFAULT FALSE, created_at TEXT
);

-- Exploit archive (Task 23, I2)
CREATE TABLE IF NOT EXISTS exploit_archive (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    pair TEXT, regime TEXT,
    exploit_scenario TEXT NOT NULL,
    target_weakness TEXT, predicted_loss REAL,
    was_defended BOOLEAN, defense_description TEXT,
    was_validated_by_outcome BOOLEAN DEFAULT NULL,
    created_at TEXT, ttl_expiry TEXT
);

-- Autopoietic integrity history (Task 24, F3)
CREATE TABLE IF NOT EXISTS autopoietic_integrity (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    structural_score REAL, functional_score REAL,
    behavioral_score REAL, representational_score REAL,
    aii_composite REAL, status TEXT,
    action_taken TEXT
);

-- Trade sequence patterns (Task 25, I5)
CREATE TABLE IF NOT EXISTS sequence_patterns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    pattern TEXT NOT NULL,  -- e.g., "BULL_HIGH_WIN,BULL_HIGH_WIN,BEAR_LOW"
    n_gram_size INTEGER,
    occurrences INTEGER, expected_occurrences REAL,
    chi2_score REAL, p_value REAL,
    next_outcome_distribution TEXT,  -- JSON: {WIN: 0.6, LOSS: 0.4}
    regime TEXT, updated_at TEXT
);
```

---

## RAM BUTCESI — PHASE 27 TOPLAM

```
═══ MEVCUT (Sprint 2 sonrasi, ~31.7GB kullaniliyor) ═══
Bot + Python + FreqAI:          ~2.0 GB
TTM model:                      ~0.05 GB
Chronos-Bolt model:             ~0.20 GB
CatBoost model:                 ~0.005 GB
Deep Ensemble (5 model):        ~0.05 GB
Neural Organism (1758 neuron):  ~0.10 GB
SQLite + LanceDB + Grafeo:      ~0.30 GB
Pheromone Field:                ~0.02 KB → 0.0006 GB (ring buffer sonrasi)
LLM API (no local model):      ~0.00 GB
Scheduler (52 job state):       ~0.05 GB
Diger moduller:                 ~0.50 GB
OS + System:                    ~1.50 GB
TOPLAM MEVCUT:                  ~4.75 GB (bufferli)
(Geri kalan: disk cache, Python memory fragmentation, vb.)

═══ PHASE 27 EKLEMELER ═══
Kronos-mini model:              +0.025 GB (4.1M × 4 byte + overhead)
POT (Wasserstein):              +0.002 GB
flowrisk (VPIN):                +0.001 GB
tick (Hawkes):                  +0.050 GB (MLE refit sirasinda peak)
etcpy (PhiC):                   +0.001 GB
giotto-tda (TDA):               +0.200 GB (persistent homology sirasinda peak)
Pheromone ring buffer:          +0.001 GB (640KB)
Hawkes state (per-pair):        +0.005 GB
BOCPD state:                    +0.015 GB
4-Layer regime state:           +0.010 GB
Per-pair Kelly tablosu:         +0.001 GB
DT-LoRA (GPT-2 inference):     +0.550 GB (Sprint 3B, conditional)
DT-LoRA (training peak):       +0.800 GB (Pazar gecesi, temporary)
TOPLAM EKLEME:                  ~0.86 GB (3B olmadan: ~0.31 GB)

═══ PHASE 27 SONRASI TOPLAM ═══
Sprint 3A sonrasi:              ~5.06 GB (buffer: ~27 GB — BOL)
Sprint 3B sonrasi:              ~5.55 GB (buffer: ~26.5 GB — BOL)
DT-LoRA training peak:         ~6.35 GB (buffer: ~25.7 GB — GUVENLI)
```

**SONUC:** RAM problemi YOK. 32GB'nin ~5-6GB'sini kullaniyoruz. Buyuk bir model
(7B LLM gibi) yuklemeden tum Phase 27 rahatca sigar.

---

## FAILURE MODE ANALIZI

### Failure Mode 1: Per-Pair Kelly Overfitting (N < 30)

**Senaryo:** Yeni pair 5 trade ile %100 win rate → Kelly f=0.50 → buyuk pozisyon → kayip
**Onlem:** Baker-McHale shrinkage (Step 5) + Trade count graduation (Step 6)
**Etki:** N<30 → f *= 0.125 (1/8 Kelly). 5 trade ile max sizing ihmal edilebilir.
**Geri donus:** 30+ trade birikince graduasyon otomatik yukseltir.

### Failure Mode 2: Wasserstein OOD False Negatives

**Senaryo:** Yeni regime Wasserstein'a "normal" gorunuyor ama gercekte OOD.
**Onlem:** Sliced Wasserstein (Faz 2) + Sinkhorn changepoint (Faz 3) katmanli koruma.
**Etki:** Tek katman fail edebilir, 3 katman birden fail etmesi cok dusuk olasilik.

### Failure Mode 3: Agent Memory Hallucination

**Senaryo:** ReflectionAgent DB verisi gorur ama yanlis yorumlar (LLM hallucination).
**Onlem:** confidence_modifier [-0.10, +0.10] arasinda CLAMP. Max etki %10.
**Etki:** Yanlis R3 analizi bile sizing'i max %10 degistirir — katastrofik degil.

### Failure Mode 4: Hawkes False Alarm (n > 0.9 ama cascade gelmedi)

**Senaryo:** Branching ratio gecici spike → tum trade'ler VETO → forgone alpha artar.
**Onlem:** VETO sadece n >= 0.95. n=0.9 → sizing x0.2, VETO degil.
**Etki:** Forgone PnL engine yakalayacak. Eger false alarm siksa threshold yukarir.

### Failure Mode 5: LLM Hypothesis Overfitting (G2 Strategy Researcher)

**Senaryo:** LLM 100 hipotez uretir, en iyi backtest secilir → overfit.
**Onlem:** 6-gate pipeline: Deflated Sharpe Ratio + OOS holdout + shadow trading.
**Etki:** Gate 4 (DSR) tum hipotez sayisini account eder. p<0.05 after correction.

### Failure Mode 6: Dream Engine Hallucination (C3)

**Senaryo:** World model gercek olmayan trajektoriler uretir, RL bunlardan ogrenir.
**Onlem:** 4 bug fix (C3) + Dream Coherence Score (HQ-4) + filter threshold fix.
**Etki:** DCS < 0.3 → dream data RL'e BASILMAZ. Model kendi guvenilirligini BILECEK.

### Failure Mode 7: Autopoietic Identity Loss (F3)

**Senaryo:** architecture_evolver risk organ'ini kapatir → organism safety'si bozulur.
**Onlem:** Constitution identity_limits: `essential_organs = ["risk", "sizing", "crowd_scoring"]`
**Etki:** Essential organ ASLA kapatilamaz. AII < 0.4 → otomatik rollback.

### Failure Mode 8: Sleep-Wake Missed Opportunity

**Senaryo:** Organizma DEEP_SLEEP'te, flash crash olur, pozisyon riski artar.
**Onlem:** EMERGENCY_WAKE: vol > 3-sigma VEYA liquidation risk → aninda tam uyanis.
**Etki:** 5 always-on job (stoploss monitor, emergency alert, heartbeat, price feed, log) ASLA uyumaz.

### Failure Mode 9: Pheromone Saturation (F2)

**Senaryo:** 30+ modul ayni anda deposit → field noise'a donusur.
**Onlem:** MMAS bounds [TAU_MIN, TAU_MAX] + per-signal normalization.
**Etki:** Hunt et al. 2019: N>300'de sorun. 30 modul × 14 sinyal = GUVENLI.

### Failure Mode 10: Cross-Provider LLM Inconsistency

**Senaryo:** Gemini BULLISH, Groq BEARISH, Mistral NEUTRAL → ne yapacagiz?
**Onlem:** CDR (Conflict Detection Rate) metriği. CDR > 0.66 → sizing kucult.
**Etki:** Yuksek CDR = regime confusion SiNYALi. Bu kendi basina degerli bilgi.

---

## EMERGENT BEHAVIOR TAHMINLERI

> Phase 26'da 5 emergent behavior tahmini yapilmisti. Phase 27 icin 7 yeni:

### Emergent 1: Per-Pair Personality Olusumu
Per-pair Kelly + per-pair agent memory + per-pair threshold ile
her pair "kisilik" gelistirecek. ZK: cesur (yuksek Kelly, yuksek confidence).
BTC: temkinli (dusuk Kelly, yuksek vol drag). Bu TASARLANMADI — emerge edecek.

### Emergent 2: Agent Alliance Kaliplari
MAGMA graph memory ile agent'lar arasi "ittifak" ve "rekabet" oruntuleri olusacak.
TrendFollower + MomentumRider = dogal muttefik (ikisi de momentum sever).
FundingContrarian + RiskMinimizer = dogal muttefik (ikisi de contrarian).
Bu allians'lar zamanla guclenecek (Hebbian reinforcement).

### Emergent 3: Circadian Expertise Spesiyalizasyonu
Sleep-wake optimization + per-pair cerebellum ile organizma belirli saatlerde
belirli pair'lerde uzmanlasmaya baslayacak. "Gece 2-5 UTC: sadece BTC, minimal sizing."
"US open 14-17 UTC: agresif, tum pair'ler." Bu TASARLANMADI — veri-driven emerge edecek.

### Emergent 4: Hypothesis Evolution Chain
LLM strategy researcher haftalik hipotezler uretecek. Zamanla hipotezler
birbirinin uzerine insa olacak: "ATR mult 3.0→2.5 ise SONRASI neden 2.5→2.2 denemeyelim?"
Bu zincirleme kesfetme davranisi TASARLANMADI — LLM reflection mechanism'den emerge edecek.

### Emergent 5: Exploit-Hardened Strategy
Adversarial self-play ile strateji zamanla "exploit-proof" olacak. 
Her exploit kesfedildiginde savunma gelistirilecek. 6 ay sonra exploit archive
OZEL bir stress-test suite olacak — hicbir public backtest bunu saglayamaz.

### Emergent 6: Information Asymmetry Oscillation
HQ-7 (Info Asymmetry Index) + forgone alpha ile organizma "ne kadar biliyoruz?"
sorusunu surekli soracak. Yuksek IAI → agresif. Dusuk IAI → muhafazakar.
Bu oscillation piyasa efficiency'sine ADAPTASYON — manual ayar gerekmez.

### Emergent 7: Dream-Reality Feedback Loop
Dream Coherence Score (HQ-4) ile organizma kendi ruya kalitesini olcecek.
Yuksek DCS → ruyalara GUVEN → dream data'dan daha fazla ogren.
Dusuk DCS → ruyalara SUPHE → gercek veriye agirlik ver.
Bu meta-bilis dongisu KENDi KENDINE kalibre olacak.

---

## YENI DEPENDENCY LISTESI (pip install)

```
# Sprint 3A
POT>=0.9.0              # Wasserstein distance, Sinkhorn, Sliced W2
flowrisk>=0.2            # Real VPIN (RecursiveVPIN + BVC)
tick>=0.7                # Hawkes process MLE (C++ backend)
etcpy>=0.1               # ETC compression complexity (PhiC)

# Sprint 3B (conditional)
giotto-tda>=0.6          # TDA persistent homology
lppls>=0.6               # Sornette dragon king detection
nolds>=0.5               # DFA, Hurst, Lyapunov
powerlaw>=1.5            # Power-law fitting
hmmlearn>=0.3            # Hidden Markov Models
prefixspan>=0.5          # Sequential pattern mining
gensim>=4.3              # LDA topic modeling (optional)
pymdp>=1.0               # Active Inference (Friston, optional)
riskfolio-lib>=6.0       # HRP, risk parity (optional)

# Kronos-mini (Sprint 3A)
# HuggingFace: NeoQuasar/Kronos-mini
# transformers + torch zaten mevcut
```

**Toplam disk eki:** ~200MB (cogu zaten mevcut dependency'ler uzerine)

---

## HYDRAQUANT ORIJINAL INOVASYONLAR — 15 KONSEPT

(HQ-1 ile HQ-15 detaylari yukarida FIX ve TASK bolumlerinde entegre edilmistir.
Burada sadece ozet tablo:)

| # | Konsept | Novel mi? | Altyapi | Sprint |
|---|---------|----------|---------|--------|
| HQ-1 | Confidence Integral | EVET | Ring buffer gerekli | 3A |
| HQ-2 | Metabolic Compute | EVET | Borbely model | 3B |
| HQ-3 | Pheromone Gradient | EVET | Ring buffer gerekli | 3A |
| HQ-4 | Dream Coherence | EVET | Dream engine fix gerekli | 3A |
| HQ-5 | Causal Entropy | EVET | Causal engine fix gerekli | 3A |
| HQ-6 | Hormonal Hysteresis | EVET | 8 satir fix | 3A |
| HQ-7 | Info Asymmetry Index | EVET | Causal engine gerekli | 3B |
| HQ-8 | Temporal Attention | EVET | N-gram + cerebellum | 3B |
| HQ-9 | Coherence Tensor | EVET | PhiC approximation | 3B |
| HQ-10 | Evolutionary Memory | EVET | MAP-Elites archive | 3B |
| HQ-11 | Forgone Alpha | EVET | Feedback loop fix | 3A |
| HQ-12 | Full Prob Propagation | KISMI | Output contract degismeli | 3A |
| HQ-13 | Hawkes Branching | EVET | order_flow + tick lib | 3A |
| HQ-14 | SIR Contagion | EVET | F&G velocity | 3B |
| HQ-15 | Exploit Archive | EVET | ExploiterAgent | 3B |

---

## BEKLENEN PERFORMANS ETKISI

```
                    Mevcut       Sprint 3A      Sprint 3B      Hedef
Kayip/trade:        -$13.11      -$3.00         -$1.50         -$1.00
Kazanc/trade:       +$20.50      +$25.00        +$35.00        +$50.00
Win rate:            60.2%        62%            65%            55-65%
Risk/Reward:         1.56         8.33           23.33          50+
Aylik (100t):       +$874        +$1,500        +$2,500        +$3,000+
Sharpe (est):        ~1.0         ~1.8           ~2.5+          ~3.0
OOD defensive:      x0.17        x0.5-1.0       adaptive       per-pair
Forgone alpha:      $1,247       $300           $100           <$50
BTC/ETH PnL:        -$1,029      -$100          +$100          +$200
Agent memory:       180 kayit    10K+           50K+           100K+
```

**UYARI:** Bu tahminler OPTIMISTIK. Gercek performans veri biriktikce netlesecek.
FINSABER (KDD 2026): LLM-based stratejiler 20 yillik testte buy-and-hold'u yenemedi.
Bizim avantajimiz: LLM sinyal URETMIYOR, parametre OPTIMIZE ediyor.

---

## TIMELINE — 7 GRUP, 7 COMMIT, 7 DEPLOY+VERIFY

> Her grup sonrasi: Claude commit TALEP EDER → kullanici commit atar →
> deploy (ssh hydra) → SSH verify (DB satirlari, log tag'leri) →
> min 2-4 saat gozlem → sorun varsa FIX → sonraki grup.

### SPRINT 3A: 4 GRUP (Fix Foundation + Core)

**GRUP 1: Temel Fix'ler — Sizing + OOD + Hormon**
```
Fix 1:  OOD Detector Mahalanobis → Wasserstein W1 (~50 satir degisiklik)
Task 1: Per-pair Bayesian Kelly (global tek satir → UNIQUE(pair, regime), 7-step pipeline)
Fix 7:  Cortisol hysteresis (8 satir, Amygdala pattern kopyala)

Verify: ssh hydra →
  - SELECT COUNT(*) FROM bayesian_kelly_per_pair; → > 0
  - grep "OOD.*distance" freqtrade.log → distance artik 500 DEGIL
  - grep "cortisol.*peak_decay" freqtrade.log → hysteresis calisiyor
```

**GRUP 2: Agent Intelligence — Hafiza + Pheromone + SHAP**
```
Fix 2A: ReflectionAgent'a gercek DB hafizasi (R3 prompt + confidence_modifier)
Fix 2B: Agent-specific RAG (rag_keywords per agent type)
Fix 2C: key_argument feedback (argument_quality tablosu + scoring)
Fix 2D: MAGMA graph memory (debate edge'leri: argued, persuaded, resisted)
Fix 2E: Agent pheromone deposit (SIGNAL_AGENT_CONSENSUS + SIGNAL_AGENT_DISSENT)
Fix 3:  Pheromone trail ring buffer (overwrite → accumulation, read_integral, read_gradient)
Fix 4:  SHAP narrative injection (CatBoost SHAP → MADAM prompt'a)

Verify: ssh hydra →
  - grep "HISTORICAL DATA" freqtrade.log → R3'te DB verisi gorunuyor
  - SELECT COUNT(*) FROM argument_quality; → > 0
  - SELECT COUNT(*) FROM magma_edges WHERE relation LIKE 'argued%'; → > 0
  - grep "agent_consensus" freqtrade.log → pheromone deposit calisiyor
  - grep "CatBoost predicts" freqtrade.log → SHAP narrative MADAM'da
```

**GRUP 3: Veri Motorlari — Scheduler + Forgone + VPIN + Hawkes**
```
Fix 5:  Scheduler job debug (causal/dream/world_model neden 0 veri?)
Fix 6:  Forgone PnL feedback loop (regime kolonu + resolver job + bug fix + adaptive threshold)
Fix 9:  Real VPIN (flowrisk kutuphanesi, volume bucketing)
Fix 10: CircadianRhythm → cerebellum baglantisi (hardcoded saatler kaldir)
Task 10: Hawkes branching ratio (O(1) recursive + tick MLE)

Verify: ssh hydra →
  - SELECT COUNT(*) FROM dream_scenarios; → > 0 (KRITIK — Sprint 2'de 0'di)
  - SELECT COUNT(*) FROM causal_discoveries; → > 0
  - grep "VPIN.*bucket" freqtrade.log → gercek VPIN calisiyor
  - grep "Hawkes.*branching" freqtrade.log → branching ratio loglanıyor
  - grep "forgone.*threshold.*adjusted" freqtrade.log → adaptive threshold
```

**GRUP 4: Core Formula + Regime + Perception + Dead Code**
```
Task 11: CAAT Asymmetric Alpha sizing formula (10 parca, per-pair)
Task 12: 4-Layer Regime Detection (VPIN→BOCPD→Causal Edge→ADX)
Task 13: Kronos-mini entegrasyonu (Quad Perception)
Task 14: Adaptive Trailing Stop + 2 Hafta Kurali
Fix 8:   Dead Code batch 1 entegrasyonu (lob_encoder, slippage_forecaster)
Task 11b: Impact-adjusted position sizing (D2 Almgren-Chriss)

Verify: ssh hydra →
  - grep "per_pair_kelly" freqtrade.log → sizing per-pair calisiyor
  - grep "Regime.*Layer0.*Layer1.*Layer2.*Layer3" freqtrade.log → 4-layer regime
  - grep "Kronos" freqtrade.log → Kronos-mini inference basarili
  - grep "trailing.*hurst" freqtrade.log → adaptive trailing
  - grep "impact_bps" freqtrade.log → impact-adjusted sizing
  - python -c "import lob_encoder" → import basarili (dead code degil artik)
```

### SPRINT 3B: 3 GRUP (LLM + Advanced + Research)

> Prereq: Sprint 3A TAMAM + 500+ trade sonucu

**GRUP 5: LLM Devrimi — Router + Judge + Researcher + Ensemble**
```
Task 18: Adaptive LLM Router (RPD tracking + Discounted TS + query-type routing)
Task 15: RLAIF d-RLAIF (rubric 5-dim + 3 LLM ensemble + WCO)
Task 16: LLM Strategy Researcher (hypothesis loop + 6-gate validation)
Task 17: Cross-Provider LLM Ensemble (ensemble_invoke + CDR)

Verify: ssh hydra →
  - SELECT COUNT(*) FROM rlaif_rewards; → > 0
  - SELECT COUNT(*) FROM hypothesis_history; → > 0
  - grep "RPD.*remaining" freqtrade.log → daily quota tracking
  - grep "CDR" freqtrade.log → conflict detection rate
```

**GRUP 6: Otonom Yasam — Uyku + Fine-tune + Fusion + Dead Code**
```
Task 19: Sleep-Wake Cycle (Borbely Process S+C + 3 mod + scheduler tier)
Task 20: Foundation Model Fine-tuning (TTM head retrain + Chronos LoRA/BitFit)
Task 22: Multi-Modal Fusion Fix (7 bug: learnable tokens, real Jina, training pipeline)
Dead Code batch 2: trinity_fusion, sac_online, hrl_meta_policy, market_maker_mode,
                   multimodal_encoder, sim2real_pipeline, external_data_integrator, gam_rag

Verify: ssh hydra →
  - grep "LIGHT_SLEEP\|DEEP_SLEEP\|FULL_WAKE" freqtrade.log → sleep-wake aktif
  - grep "fine_tune.*completed" freqtrade.log → fine-tuning basarili
  - python -c "import trinity_fusion" → dead code artik DEGIL
  - SELECT COUNT(*) FROM rlaif_rewards WHERE provider_scores IS NOT NULL; → multi-modal aktif
```

**GRUP 7: Frontier — DT + Adversarial + Identity + Language**
```
Task 21: Decision Transformer (GPT-2 + LoRA, decision_contract corpus)
Task 23: Adversarial Self-Play (ExploiterAgent + exploit_archive)
Task 24: Autopoietic Integrity (AII 4-layer + constitution identity_limits)
Task 25: Trading as Language (n-gram + PrefixSpan + conditional grammar)

Verify: ssh hydra →
  - ls user_data/models/dt_lora_*.pt → DT model mevcut
  - SELECT COUNT(*) FROM exploit_archive; → > 0
  - SELECT COUNT(*) FROM autopoietic_integrity; → > 0
  - SELECT COUNT(*) FROM sequence_patterns; → > 0
```

### TOPLU KONTROL — SPRINT 3A TAMAM MI?

Sprint 3A'nin 4 grubu bittikten sonra toplu dogrulama:

```sql
-- Bu sorgularin HEPSI > 0 donmeli:
SELECT 'bayesian_kelly_per_pair' as tbl, COUNT(*) FROM bayesian_kelly_per_pair
UNION ALL SELECT 'argument_quality', COUNT(*) FROM argument_quality
UNION ALL SELECT 'pair_thresholds', COUNT(*) FROM pair_thresholds
UNION ALL SELECT 'regime_layers', COUNT(*) FROM regime_layers
UNION ALL SELECT 'hawkes_state', COUNT(*) FROM hawkes_state
UNION ALL SELECT 'dream_scenarios', COUNT(*) FROM dream_scenarios
UNION ALL SELECT 'causal_discoveries', COUNT(*) FROM causal_discoveries
UNION ALL SELECT 'magma_edges (debate)', COUNT(*) FROM magma_edges WHERE relation LIKE 'argued%'
UNION ALL SELECT 'agent_memory', COUNT(*) FROM agent_memory WHERE timestamp > datetime('now','-7 days');

-- OOD artik 500 degil:
SELECT pair, distance, defensive_mult FROM (
  SELECT pair, MAX(timestamp) as t, distance, defensive_mult 
  FROM ... -- son OOD loglari
) WHERE distance < 100;  -- 500 yerine mantikli degerler

-- Per-pair Kelly farkli degerler:
SELECT pair, regime, ROUND(alpha/(alpha+beta_param), 4) as kelly_fraction
FROM bayesian_kelly_per_pair ORDER BY kelly_fraction DESC LIMIT 10;
```

---

## AKADEMIK KAYNAKLAR (50 Ajan, 200+ Paper)

### Temel (Prensip 0 + Sizing)
- Peters (2019) "Ergodicity problem in economics" Nature Physics 15:1216
- Stiffelman (2026) "Investing is Compression" arXiv 2604.10758
- Baker & McHale (2013) "Optimal Betting Under Parameter Uncertainty" Decision Analysis 10(3)
- Kelly (1956) "A New Interpretation of Information Rate" Bell System Tech J
- Cover & Thomas (1991) "Elements of Information Theory" Ch.6
- Meucci "Risk Budgeting Based on Optimized Uncorrelated Factors" SSRN 2276632
- Singha et al. (2025) "Forecast-to-Fill" arXiv 2511.08571 — Friction-adjusted Kelly

### OOD + Regime Detection
- Horvath et al. (2024) "Wasserstein Regime Clustering" J.Comp.Finance 28(1)
- Luan & Hamp (2025) "Sliced Wasserstein k-means" DSFE 5(3)
- Wang et al. (2023) "WOOD: Wasserstein OOD Detection" IEEE TNNLS
- Tsaknaki et al. (2025) "Score-Driven BOCPD" Quantitative Finance 25:307
- Adams & MacKay (2007) "Bayesian Online Changepoint Detection" arXiv 0710.3742

### Agent Memory + Debate
- Usher & McClelland (2001) "Leaky Competing Accumulator" Psychological Review 108:550
- Dorigo & Stutzle (2004) "Ant Colony Optimization" MIT Press
- Stanley & Miikkulainen (2002) "NEAT" Evolutionary Computation 10(2)
- Falk et al. (2018) "Integrated Information as Group Interaction Metric" PLOS ONE

### Deep Learning + RL
- Yun (2024) "DT-LoRA for Quantitative Trading" arXiv 2411.17900
- Google DeepMind (2024) "RLAIF vs RLHF" ICML
- Kronos (2025) "Foundation Model for Financial Markets" AAAI 2026
- Friston (2017) "Active Inference" — EFE = Kelly esdegerligi
- "Agentic Finance" MDPI Entropy 28(3):321 (2026)

### Market Microstructure
- Bacry et al. (2015) "Hawkes Processes in Finance" arXiv 1502.04592
- Almgren & Chriss (2000) "Optimal Execution of Portfolio Transactions"
- Donier & Bonart (2015) "Million Metaorder Analysis Bitcoin" arXiv 1412.4503
- Easley & O'Hara (2024) "VPIN in Bitcoin" SSRN 4814346

### LLM + Neurosymbolic
- "FINSABER" (KDD 2026) arXiv 2505.07078 — LLM 20-yil testte basarisiz
- "R&D-Agent-Quant" (NeurIPS 2025) arXiv 2505.15155 — Microsoft
- "AlphaAgent" (KDD 2025) arXiv 2502.16789
- "SHAPLLM" arXiv 2409.00079
- MSGCA arXiv 2406.06594

### Biyolojik + Otonom
- Borbely (1982) "Two-Process Model of Sleep" Human Neurobiology
- Maturana & Varela (1980) "Autopoiesis and Cognition"
- Bak (1996) "How Nature Works: SOC"
- Tononi (2004) "Integrated Information Theory"
- Beer (1985) "Viable System Model"

### Momentum + Strateji
- Huang et al. (2024) "VWTSMOM" SSRN 4825389
- Dobrynskaya (2021) "Crypto Momentum and Reversal" SSRN 3913263
- Wood et al. (2022) "Slow Momentum Fast Reversion" J.Financial Data Science
- Barroso & Santa-Clara (2025) "Crypto Momentum Has Its Moments" Springer FMPM
