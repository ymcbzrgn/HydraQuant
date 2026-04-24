# Phase 29 — Sensory Expansion & Self-Falsification

**ARGE Konsolide Raporu — 24 Nisan 2026**

> "Organizma yeni duyu organları kazanıyor ve kendini sınayabiliyor."

## 0. Amaç + Çıkış Noktası

Mega+EK+Tur2+Tur3 (20-24 Nisan) sprint'i **kayıp kaynaklarını kapattı** — Kelly corruption, shadow label bug, MinStakeGuard bottleneck, LLM routing sessizliği, Tier 3.5 MADAM bypass, Chandelier asimetrisi, argument_quality öğrenme döngüsü. 36 audit bulgusu çözüldü. Kod sağlam, test pass, deploy canlı.

**AMA**: Bu fix'ler **kazanç garantisi vermez** — sadece kendi ayağına sıkan bot'u durdurdu. Gerçek alpha gelmek için iki eksik var:

1. **Duyu kaynakları dar** — bot sadece OHLCV + RSS + sentiment görür. Derivatives positioning, on-chain flow, options dealer exposure, microstructure toxicity... hepsi erişilebilir ama bağlı değil.
2. **Self-falsification yok** — her fix "umarım çalışır" gambit. Backtest framework strategy'yi devre dışı bırakıyor (runmode guard'lar). Bir ay önce deploy edilen fix'lerin pnl etkisi hâlâ ölçülmedi — gerçek rakam yok.

Phase 29 bu iki eksikliği çözmek için: **organizma yeni duyular ekliyor + kendi kararlarını sınayabiliyor**.

## 1. Phase 29 vs Önceki Phase'ler

| Phase | Tema | Status |
|---|---|---|
| Phase 19 | Evidence-Based RAG | COMPLETE |
| Phase 20 | Hybrid Engine (MADAM + AgentPool) | COMPLETE |
| Phase 21-22 | System hardening + systemd | COMPLETE |
| Phase 23 | Jina Migration | COMPLETE |
| Phase 24-25 | Neural Organism (BCM+STDP, hormonal) | COMPLETE |
| Phase 26 | CAAT (cognitive architecture) | COMPLETE |
| Phase 27 | Asymmetric Alpha (Sprint 3A/3B) | COMPLETE |
| Phase 28 | DB Evolution (LanceDB + Grafeo + DuckDB) | COMPLETE |
| **Phase 29** | **Sensory Expansion + Self-Falsification** | **BU DOKÜMAN** |

## 2. Felsefe

HydraQuant'ın "neural organism" metaforu:
- **Duyu** (sensory) → dış dünyadan algı
- **Hafıza** (memory) → geçmiş deneyim
- **Rüya** (dream) → içsel simülasyon
- **Karar** (decision) → ajans
- **Refleks** (stop-loss) → acı eşiği
- **Öz-bilinç** (post-trade court) → içe bakış

Phase 29 şu iki boyutu genişletir:

### 2.1 Sensory Expansion
Bot şu anda **fiyat + haber** üzerinden algılar. Crypto alpha'nın önemli bir kısmı **derivatives positioning** (funding, open interest, options gamma) ve **on-chain flow** (exchange netflow, stablecoin issuance) içinde gizli. Bu kaynaklar:
- Ücretsiz (free API tier)
- Düşük latency (1-15dk cron ile aktif tutulabilir)
- Mevcut EvidenceEngine'e sub-score olarak bağlanabilir

### 2.2 Self-Falsification
Popper'ın falsification ilkesi: bir hipotez test edilemiyorsa bilimsel değildir. HydraQuant'ın her fix'i hipotez. Ama backtest framework kırık → fix'ler test edilemiyor → bilimsel değil, "umarım".

Self-falsification = bot kendi kararlarını geriye dönük test edebilmeli:
- Mock LLM ile backtest runnable olsun
- LLM response cache ile historical replay deterministic olsun
- CPCV walk-forward ile OOS performance statistically valid
- DoWhy-GCM ile her kayıp trade için Shapley attribution

## 3. 7 Gece ARGE Ajanı Bulguları

24 Nisan 2026, 01:00-03:00 TR arası 7 paralel deep research ajanı çalıştırıldı. Her ajan farklı konuda 2024-2025 literature + production patterns + HydraQuant somut uyarlama.

### 3.1 ARGE-1: Alternatif Alpha Kaynakları

**Sorun**: HydraQuant sadece OHLCV + RSS + sentiment alpha'sı çekiyor. Crypto alfa'nın %50+ u derivatives positioning + on-chain flow + microstructure'da.

**Top 5 kaynak**:

| # | Kaynak | Effort | IR lift (literatür) | Free tier? |
|---|--------|--------|---------------------|------------|
| 1 | Funding/OI Divergence | 1g | +0.5 | ✅ Binance/Bybit |
| 2 | On-chain Exchange Netflow | 3g | +0.3 | ✅ BitQuery 10K/ay |
| 3 | Order Book Imbalance (OBI) | 4g | +0.4 (timing) | ✅ WS unlimited |
| 4 | Deribit Options GEX | 3g | +0.3 (regime filter) | ✅ unlimited |
| 5 | Lag-Llama + Moirai Ensemble | 2g | +0.1 (uncertainty) | ✅ HuggingFace |

**Kritik uyarı**: IR rakamları **literature-derived**, HydraQuant'a özel değil. Gerçek IR bizim veride **UNKNOWN**. Backtest framework olmadan (Phase 29 Sprint 3C) doğrulama imkansız.

**Referanslar**:
- [CoinGlass Funding/OI API](https://coinglass.com/api-docs) (free tier rate-limited)
- [Deribit Insights GEX](https://insights.deribit.com/market-research/) + [SqueezeMetrics white paper](https://squeezemetrics.com/monitor/download/pdf/white_paper.pdf)
- [Lag-Llama arXiv 2310.08278](https://arxiv.org/abs/2310.08278)
- [Moirai-MoE arXiv 2410.10469](https://arxiv.org/abs/2410.10469)
- [BitQuery GraphQL](https://github.com/bitquery/graphql-over-http)

### 3.2 ARGE-2: 2024-2025 Trading LLM Literature

**Sorun**: HydraQuant MADAM'ı (4 agent + 1 Coordinator) Kim et al. 2025 "Debate or Vote?" paper'ına göre **martingale** — debate alone does not improve expected correctness. Sampling+voting eksik.

**Top 5 framework**:

1. **FinCon** (NeurIPS 2024, [arXiv 2407.06567](https://arxiv.org/abs/2407.06567)) — Conceptual Verbal Reinforcement (CVRF). Post-trade reflection → "conceptualized beliefs" tablosu → debate prompts'a prefix inject.

2. **Kim et al. 2025 "Debate or Vote?"** ([arXiv 2508.17536](https://arxiv.org/abs/2508.17536)) — Martingale proof. Fix: sampling+voting wrapper (each agent N=3 with temperature=0.7, majority vote action).

3. **TradingAgents** (Xiao et al., [arXiv 2412.20138](https://arxiv.org/abs/2412.20138)) — Bull/Bear dialectical pairs (thesis vs antithesis), 2-round rebuttal. HydraQuant'ın 4 parallel-critic pattern'i tek-yönlü.

4. **Mixture of Agents / MoA** ([arXiv 2406.04692](https://arxiv.org/abs/2406.04692)) — L1 proposers (cheap OSS) + L2 aggregator (premium). Beat GPT-4o on AlpacaEval. HydraQuant'ın 79-slot LLM router'ı zaten MoA şeklinde ama quality tier exploit edilmiyor.

5. **FinMem** (ICLR 2024, [arXiv 2311.13743](https://arxiv.org/abs/2311.13743)) — Layered memory (shallow/intermediate/deep). HydraQuant `magma_memory.py` monolithic, tier'lar yok.

**Kritik eksiklikler (literature perspective)**:
- Coordinator tek-nokta synthesis (martingale anti-pattern)
- Process reward model (PRM) yok — step-level credit assignment
- Speculative cascade yok (Google Research 2025) — latency optimization
- Token-level entropy signals yok (2025 UQ survey)

### 3.3 ARGE-3: Meta-learning for Trading

**Mevcut durum (verified)**:
- `reptile_meta.py` (319 LOC) — Sunday 01:00 UTC weekly
- `ewc_continual.py` (264 LOC) — Fisher per regime
- `hrl_meta_policy.py` (454 LOC) — 5 organ meta-state
- MAML, ProtoNet, ANIL, OML, PCGrad, domain adaptation: YOK

**Top 3 pragmatic**:

1. **Online Meta-Learning (OML)** ([Finn 2019](https://arxiv.org/abs/1902.08438) + MOML [Acar 2021](https://arxiv.org/abs/2106.02946))
   - Reptile weekly → BOCPD-triggered online update
   - Regret bound in non-stationary (crypto regime drift)
   - FTML buffer'ı `oml_tasks` tablosunda persist
   - Effort: 150 LOC, zero new deps

2. **ProtoNet + Mahalanobis** ([Snell 2017](https://arxiv.org/abs/1703.05175))
   - Yeni pair eklendiğinde cold-start fix
   - 128-d latent embedding → 3-NN prototype lookup
   - 72h shadow only → 2w blend → full Kelly
   - LanceDB (Phase 28) içinde prototype bank

3. **PCGrad Gradient Surgery** ([Yu 2020](https://arxiv.org/abs/2001.06782))
   - 25 independent Kelly → correlated pair cluster transfer
   - Agglomerative clustering on return-correlation
   - PCGrad within cluster, plain sum across
   - HRL meta-policy ile komplementer

### 3.4 ARGE-4: Causal Inference Advanced

**Mevcut**: `causal_engine.py` tigramite PCMCI+, MIN_OBS=30 (Tur-2'de 20'den revize), `causal_discoveries` hala 0 row. PCMCI+ **structure discovery** yapar ama **effect magnitude** vermez.

**Top 3 framework**:

1. **DoWhy + EconML** (Microsoft/py-why)
   - PCMCI+ graph → DoWhy `CausalModel(graph=...)` wrap
   - EconML `DML` / `CausalForestDML` for heterogeneous effects
   - Trade close → `explain_trade(trade_id)` → {cause_attribution: {funding: -0.4R, sentiment: +0.2R, noise: -0.8R}}
   - `counterfactual_results` tablosu dolar (şu an empty)
   - Effort: ~400 LOC, pip install

2. **DoWhy-GCM** — **en önemlisi**
   ```
   gcm.attribute_anomalies(causal_model, target='outcome_pnl', anomaly_samples=losing_trade)
   → {funding_spike: 0.42R, regime_flip: 0.31R, confidence_overweight: 0.19R, noise: 0.08R}
   ```
   Post-Trade Court'un "market_noise" fallback'ını **Shapley attribution**'la değiştir. ~200ms/trade, `trade_postmortems` tablosuna direkt yazım.

3. **Convergent Cross Mapping (CCM)** ([PNAS Nexus 2024](https://academic.oup.com/pnasnexus/article/3/1/pgad422/7460388), [ScienceDirect BTC 2025](https://www.sciencedirect.com/science/article/pii/S095741742500747X))
   - Nonlinear dynamical coupling (PCMCI+ zayıf)
   - `funding → price` veya `price → funding` per regime?
   - `lead_lag_matrix` tablosu → sizing: "leading indicator decayed" ignore

**Shadow ≈ Counterfactual**: HydraQuant `shadow_trades` tablosu IS counterfactual dataset. IPW + DML → unbiased sizing treatment effect. **Bedava goldmine**, şu an kullanılmıyor.

### 3.5 ARGE-5: Risk Management Advanced

**Payoff 0.66 problemi**: 13 günlük trade 190 trailing loss -$1052 vs 129 ROI win +$1910. Yapısal asimetri:
- Chandelier 2.5x ATR çok geniş (revize Tur-2'de 1.5x'e indirildi)
- ROI table 60m→5% erken alım vs loser -5%+ nefes
- Trailing tier 4/8/15% → 1/3/6% (revize Tur-2)

**Top 5 technique**:

1. **CVaR-Conditioned Kelly** ([MDPI 2026 Crypto](https://www.mdpi.com/2227-7072/14/3/53))
   - `f* = (μ - rf) / (λ · CVaR_95)` yerine `σ²`
   - Coherent risk measure (Artzner axioms)
   - `riskfolio-lib` free
   - Expected DD reduction: 20-35% (Man Group backtests on commodities, crypto amplifies)

2. **HAR-RV + Lee-Mykland Jump Detection**
   - `RV_t+1 = β0 + β_d·RV_t + β_w·RV_w + β_m·RV_m + γ·J_t`
   - Jump test: `|L| = |(log P_t - log P_{t-1}) / σ_t·√K| > 4.6`
   - Dynamic ATR: `k = 1.5 + 0.5·tanh((RV_forecast - RV_median)/RV_std)` range 1.0-2.0
   - HAR beats LSTM consistently ([Asia-Pacific 2024](https://link.springer.com/article/10.1007/s10690-024-09510-6))

3. **UI/CDaR Circuit Breaker**
   - `UI = sqrt(mean(DD²))` rolling 14-day
   - `CDaR_95 > 8%` → halve sizing + max_open_trades -= 2
   - **Şu anki 3% portfolio cap sadece instantaneous — death by thousand cuts yakalanamaz**
   - [Riskfolio-Lib](https://riskfolio-lib.readthedocs.io/) free, Chekhlov-Uryasev-Zabarankin showed 25-40% MaxDD reduction

4. **Hierarchical Risk Parity (HRP)** ([López de Prado 2016](https://pyportfolioopt.readthedocs.io/))
   - Recursive bisection on correlation-clustered tree
   - No matrix inversion (robust to singular cov — crypto cascade)
   - BTC-contagion cluster auto-downweight
   - NVIDIA RAPIDS: 15-20% Sharpe uplift vs 1/N

5. **Funding Rate + OI Cascade Detector** (crypto-specific)
   - Funding >+0.05% sustained 8h + OI +15% /24h + price within 2% prior high + liquidation asymmetry >3:1
   - Nov 2025 $2B liquidation + $19B flash crash had 6h+ divergence warning ([Coinchange](https://www.coinchange.io/blog/bitcoins-2-billion-reckoning-how-novembers-liquidations-cascade-exposed-cryptos-structural-fragilities), [insights4vc](https://insights4vc.substack.com/p/inside-the-19b-flash-crash))
   - Force exit all longs <1R profit → defensive
   - Taleb black swan insurance (implicit, crypto options illiquid)

**Payoff impact tahmini** (stacked):
- CVaR-Kelly: DD -20-35%, payoff +0.3-0.5
- HAR-RV: DD -15-25%, payoff +0.2-0.4
- UI/CDaR: DD -25-40% MaxDD, payoff neutral
- HRP: DD -30%, payoff +0.15
- Cascade detector: tail events only

**Toplam hedef**: payoff 0.66 → **1.3-1.5** (realistic), 1.8+ (optimistic literature). MaxDD -35%.

### 3.6 ARGE-6: LLM Routing SOTA 2025

**Mevcut**: LinUCB 5-dim feature, Thompson cold-start blend, circuit breaker, EMA latency, retroactive feedback (Tur-3). Gemini 9.5s mean latency dominant. Groq 0.3s (30x faster) under-utilized.

**Top 5 technique**:

1. **Speculative Cascade** ([Google Research 2025](https://research.google/blog/speculative-cascades-a-hybrid-approach-for-smarter-faster-llm-inference/), [Cascade Speculative Drafting NeurIPS 2024](https://arxiv.org/abs/2312.11462))
   - Groq (0.3s) draft → Gemini verifier (Grade 1-5 prompt) → accept if ≥4
   - Else fall through to full Gemini
   - Expected: 70-85% resolved by Groq alone
   - **Weighted mean latency**: 0.3s × 0.8 + 9.5s × 0.2 = **2.14s** (vs current 7.5s)
   - Hits <2s target, hits 5s budget with plenty margin

2. **LLMLingua-2 Prompt Compression** ([arXiv 2403.12968](https://arxiv.org/abs/2403.12968))
   - xlm-roberta token-importance classifier (Microsoft)
   - 3-6x compression at <5% quality loss on extractive
   - Break-even at input >1500 tokens
   - 80-120ms CPU overhead on 32GB no-GPU
   - HydraQuant `rag_synthesis`, `news_digest`, `weekly_report` qualify

3. **Semantic Cache** (GPTCache + Jina — zaten var)
   - Cosine threshold 0.93, TTL per task
   - Expected 15-30% hit rate on factsheet-style
   - 0ms latency + $0 cost on hit

4. **Pareto LinUCB** ([Drugan & Nowé 2014](https://arxiv.org/abs/2303.03789), [Bubeck 2024](https://arxiv.org/abs/2303.03789))
   - 3-head reward: quality × 1/latency × 1/cost
   - Regime-weighted scalarization:
     - High vol: latency 0.6 / quality 0.3 / cost 0.1
     - Low vol: quality 0.6 / cost 0.3 / latency 0.1
     - Idle: cost 0.7 / quality 0.2 / latency 0.1
   - ~60 LOC on existing ModelSlot

5. **MoA for Complex Tasks Only** ([Together AI](https://www.together.ai/blog/together-moa))
   - `madam_debate`, `cross_correlation`, `weekly_report`
   - 3 cheap providers (Groq, Cerebras, Mistral) parallel L1
   - Gemini aggregates L2
   - Cost ~2x single-call, quality +8-15% on synthesis benchmarks

### 3.7 ARGE-7: Backtest Framework (CRITICAL)

**Acı gerçek**: HydraQuant strategy **backtest'te TAMAMEN ÖLÜ**. Verified via `HydraSizer.py` grep:

```
:149   scanner tick          → runmode guard (skip backtest)
:613   _get_ai_signal        → HTTP 127.0.0.1:8891 120s timeout (fails per-candle)
:815   populate_entry_trend  → runmode guard
:950   populate_exit_trend   → runmode guard
:1500  custom_stake_amount   → runmode guard (confidence modulation)
:2341  custom_exit           → runmode guard (AI exit logic)
:2564  DCA logic             → runmode guard
```

**Sonuç**: Her fix (shadow label bug, Kelly reset, MinStakeGuard, Chandelier, MADAM ensemble, LinUCB, ...) backtest'te **ÇALIŞTIRILAMAZ** → OOS performance **UNKNOWN** → "umarım çalışır" literal.

**Scheduler backtest'te çalışmaz**: APScheduler BackgroundScheduler RAG service'te başlar, `freqtrade backtesting` komutu onu invoke etmez. Dream/causal/weekly jobs ASLA fire olmaz.

**Existing machinery** (eksik):
- `backtest_training_data` (db.py:934) — labelling only, not replay
- `backtest_embedder.py` — Lance ingestion
- `backtest_comparison.py` — zip compare, 148 lines
- **No `llm_response_cache` or deterministic prompt→response replay**
- **No CPCV, no purged CV, no deflated Sharpe, no PBO**

**3-tier plan**:

**Tier 1 (1-2 gün)**: `MockAISignal` deterministic RSI/ADX-based
- Backtest end-to-end runnable
- `user_data/scripts/mock_ai_signal.py` (60 satır)
- 7 runmode guard replace with backtest branch
- Benefit: downstream sizing/exit/DCA exercised

**Tier 2 (1 hafta)**: Historical LLM response replay
- New table `llm_response_cache`:
  ```sql
  (id, pair, ts_utc, prompt_hash, model_version, agent_role, response_json, retrieval_snapshot_id)
  ```
- `llm_router.py` branch: live → cache+call, backtest → SELECT by (pair, ts≤current, hash)
- `warm_llm_cache.py --timerange --pairs` CLI
- **Critical invariant**: Lance snapshot weekly → no look-ahead from future embeddings

**Tier 3 (2 hafta)**: CPCV walk-forward + PBO + Deflated Sharpe
- `walk_forward_runner.py` — 6 groups, C(6,2)=15 paths, embargo 24 bars
- López de Prado AFML ch.7
- Bailey 2014 PBO + Deflated Sharpe
- Regime-stratified Sharpe (bull/bear/chop)
- Drift detector: live distribution vs backtest replay KS-test

**Referanslar**:
- [Pardo 2008 WF](https://www.wiley.com/en-us/The+Evaluation+and+Optimization+of+Trading+Strategies-p-9780470128015)
- [López de Prado AFML](https://www.wiley.com/en-us/Advances+in+Financial+Machine+Learning-p-9781119482086)
- [Bailey & López de Prado 2014 PBO/DSR](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551)
- [Welch 2020 Signal Stability](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3622304)

## 4. Phase 29 Sprint Yapısı

| Sprint | Tema | Süre | Dependency |
|--------|------|------|-----------|
| **Sprint 3C** | **Backtest Revival** (Tier 1 mock) | 1-2 gün | — (MUST GO FIRST) |
| **Sprint 3D** | **Sensory Expansion Wave 1** (funding/OI, UI/CDaR, Dynamic ATR, Semantic cache) | 3-5 gün | 3C |
| **Sprint 3E** | **MADAM Upgrade** (Sampling+voting, MoA tier, CVRF beliefs) | 3-4 gün | 3C |
| **Sprint 3F** | **Causal Introspection** (DoWhy-GCM, CCM) | 4-5 gün | 3C + 3E |
| **Sprint 3G** | **Sensory Expansion Wave 2** (On-chain, GEX, Speculative Cascade) | 5-7 gün | 3D |
| **Sprint 3H** | **Risk Upgrade** (CVaR-Kelly, HRP, HAR-RV) | 5-7 gün | 3C |
| **Sprint 3I** | **Meta-learn** (OML, ProtoNet, PCGrad) | 7-10 gün | 3E + 3F |
| **Sprint 3J** | **Backtest Tier 2-3** (Replay cache + CPCV) | 10-14 gün | 3C |

**Total Phase 29**: ~8 sprint, 6-10 hafta, iteratif.

## 5. Rasyonel Impact Matrix (dürüst tahminler)

**Confidence legend**:
- 🟢 HIGH — matematiksel ya da kendi verimizde kanıtlı
- 🟡 MEDIUM — literature'da tutarlı, kendi verimizde validasyon YOK
- 🔴 LOW — literature çeşitli, bizim context'te tahmin

| Fix | Effort | Impact | Confidence |
|-----|--------|--------|------------|
| Backtest Tier 1 mock | 1-2g | **Her fix'i OOS test edebilmek** | 🟢 HIGH (matematik: runnable/not runnable) |
| Funding/OI divergence | 1g | IR +0.2-0.6 | 🟡 MEDIUM (literature broad) |
| UI/CDaR circuit | 2g | MaxDD -15-30% | 🟡 MEDIUM (Chekhlov-Uryasev equity, crypto extrap) |
| Dynamic ATR Chandelier | 1g | Payoff +0.1-0.3 | 🟡 MEDIUM (HAR-RV crypto papers) |
| Semantic cache | 2g | %15-30 LLM latency 0, cost save | 🟢 HIGH (deterministic on hit) |
| Sampling+voting MADAM | 4h | Signal noise -%20 | 🟡 MEDIUM (Kim 2025 proof) |
| MoA 2-layer | 4h | Synthesis quality +8-15% | 🟡 MEDIUM (AlpacaEval, trading extrap) |
| DoWhy-GCM postmortem | 3g | "market_noise" → Shapley specific | 🟢 HIGH (deterministic replacement) |
| HRP allocation | 3g | Sharpe +15-20%, MaxDD -30% | 🟡 MEDIUM (NVIDIA equity, crypto extrap) |
| Speculative cascade | 5g | Latency 7.5s → 2.1s | 🟢 HIGH (weighted-mean math) |
| On-chain netflow | 3g | Regime filter +10-15% hit rate | 🔴 LOW (CryptoQuant case studies) |
| Deribit GEX | 3g | Regime detection +20% hit | 🔴 LOW (SqueezeMetrics equity) |
| OBI / Kyle's lambda | 4g | Entry timing +8-12 bps/trade | 🔴 LOW (Cartea academic) |
| Lag-Llama ensemble | 2g | Uncertainty calibration +3% | 🔴 LOW (benchmark extrap) |
| OML online meta | 1 sprint | Regime drift adaptation (qualitative) | 🔴 LOW (Finn 2019 NLP/CV, trading extrap) |
| ProtoNet new pair | 1 sprint | Cold-start 72h → 24h | 🟡 MEDIUM (few-shot lit) |
| PCGrad transfer | 2 sprint | Cross-pair quality +5-10% | 🔴 LOW (multi-task learning generic) |
| CPCV walk-forward | 2 hafta | Statistical validity (qualitative) | 🟢 HIGH (deterministic replacement for single split) |
| Cascade detector | 2 hafta | Tail event insurance (event-based) | 🟡 MEDIUM (post-hoc case studies) |
| CVaR-Kelly | 1 hafta | Sizing robustness | 🟡 MEDIUM (MDPI crypto) |

**TOP 3 DÜRÜST**:
1. **Backtest Tier 1** — **MATEMATIKSEL KESİN**. Bu olmadan diğerleri tahmin.
2. **Speculative Cascade** — **MATEMATIKSEL KESİN**. Weighted-mean latency hesabı.
3. **Semantic Cache** — **MATEMATIKSEL KESİN**. Hit = 0ms. Sadece hit oranı belirsiz.

**EN BELIRSIZ (dikkatli ol)**:
- OML — academic results mostly NLP/CV, trading transfer marjinal
- Meta-learning çoğu — "adaptasyon faydası" qualitative, measurable değil
- Alpha sources (funding/OI, on-chain, GEX) — literature rakamları karışık, HydraQuant'a özel IR bilinmez

## 6. Yamac için Karar Noktaları

**Phase 29 başlasın mı?**
- EVET eğer: self-falsification + sensory expansion felsefi öncelikli
- HAYIR eğer: mevcut sprint sonuçlarını 1-2 hafta gözlemlemek istiyorsan (Kelly converge, ENSEMBLE ratio stabilize) → veriden sonra karar

**Sprint sıralama**:
- 3C MUST first (backtest revival) — yoksa her ölçüm tahmin kalır
- 3D/3E/3F paralel gidebilir — independent
- 3G/3H/3I son aşama — daha ileri literature

**Kaynak tahsisi** (kullanıcı + Claude + ajanlar):
- Her sprint: 1 prompt ön çalışma (kodcu) + 1-2 audit-god + 1 deploy + verify
- Toplam sprint: ~1-2 hafta her biri
- 8 sprint → ~3 ay Phase 29 tamamı

## 7. Kritik Uyarılar (Self-Falsification Principle)

1. **Her impact tahmini YANLIŞ olabilir** — backtest Tier 1'e kadar ölçüm yok.
2. **Literature ≠ HydraQuant** — paper'lar çoğu equity/NLP/CV, crypto transfer tahmin.
3. **Market non-stationary** — bir ay önce çalışan bir fix bir ay sonra çalışmayabilir. Drift detector (3J) bu nedenle kritik.
4. **Claude limit realities** — 8 sprint × iterasyon büyük iş. Her sprint iki ayrı Claude session gerektirebilir.
5. **Yamac insan** — bu kadar scope bazen demotivating olur. Sprint 3C + 3D küçük win → sonra uzun vadeli.

## 8. Sonuç

Phase 29 önceki phase'lerden **kalite farkı**:
- Phase 24-26 **yapı inşa etti** (organism, cognition, agents)
- Phase 27 **adaptasyon ekledi** (RL, Kelly, dream)
- Phase 28 **veri altyapısı genişletti** (LanceDB, Grafeo)
- **Phase 29 duyu + içe bakış** — organizma artık kendini sınar ve yeni duyular kazanır

Sprint 3C (backtest revival) olmadan hiçbir Phase 29 hedefi measurable değil. O yüzden 3C = **hayatta kalma koşulu**, diğer sprint'lerin fizibilite garantisi.

---

**Döküman durumu**: Phase 29 ARGE COMPLETE, ALPHA (implementation blueprint) → `PHASE29_ALPHA.md` sonraki adım.

**Yazım tarihi**: 24 Nisan 2026 03:00 TR  
**ARGE ajanları**: 7 paralel deep research (alpha / LLM lit / meta-learn / causal / risk / routing / backtest)  
**Yazar**: Claude Opus 4.7 max (kontrolcü rolü)  
**Onay bekliyor**: Yamac (Phase 29 başlat kararı + Sprint 3C ilk commit zaman planı)
