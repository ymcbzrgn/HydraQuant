# Evidence Engine

The LLM-free signal generator. Answers the trade-decision question from 30+ data feeds in ~50 ms at zero API cost. Runs **first** on every candle — LLMs only enter the loop when the Evidence Engine's uncertainty justifies the cost.

File: `user_data/scripts/evidence_engine.py` (1,262 lines).

## The six sub-questions

```python
# evidence_engine.py:101
DEFAULT_WEIGHTS = {
    "q1_trend":    0.22,
    "q2_momentum": 0.20,
    "q3_crowd":    0.22,
    "q4_evidence": 0.15,
    "q5_macro":    0.10,
    "q6_risk":     0.11,
}
```

| Sub-score | Inputs | Logic |
|---|---|---|
| **Q1 Trend** | EMA 20/50/200, ADX | Cascade alignment × ADX strength |
| **Q2 Momentum** | RSI, MACD | RSI > 50 zone — research-backed 2.8× alpha vs oversold reversal, 1.6× vs MACD crossover |
| **Q3 Crowd** | Fear & Greed, funding rate, long/short ratio | **Contrarian** — F&G < 20 or > 80 → fade the crowd |
| **Q4 Evidence** | k-NN historical matching, backtest pattern stats, OHLCV ensemble | Retrieval from `successful_trade_patterns.lance` + `pattern_trades` table |
| **Q5 Macro** | DXY, VIX, BTC dominance, Treasury yields | Low-correlation anchors — crash hedge |
| **Q6 Risk** | ATR, volume, volatility z-score | Higher risk → lower confidence (dampens sizing) |

## Regime-adaptive weight overrides

The six weights are **not fixed**. They redistribute by regime (`evidence_engine.py:113–131`):

| Regime | Shift |
|---|---|
| `ranging` | +weight to Q3 (crowd) and Q4 (evidence); trend-following agents matter less |
| `high_volatility` | Q6 (risk) jumps to 0.24; momentum gets dampened |
| `transitional` | Uniform — no regime has high confidence yet |
| `trending_bull` | Default |
| `trending_bear` | Default with Q3 polarity flipped for contrarian |
| `_global` | Default |

Weights are themselves neurons. The regime detector (`regime_classifier.py`) outputs a regime label every candle, and the weights for that regime are pulled via `_p("evidence.weights.qX_name", fallback, regime=regime)`.

## The pipeline

```python
# evidence_engine.py:218
def generate_signal(pair, regime, candles, macro, sentiment, ...):
    raw = self._gather(pair, regime, candles, macro, sentiment)
    patterns = self._analyze_patterns(raw)
    sub_scores = self._score_sub_questions(raw, patterns, regime)
    contradictions = self._detect_contradictions(sub_scores)
    signal, confidence = self._synthesize(sub_scores, contradictions, regime)
    return {
        "signal": signal,
        "confidence": confidence,
        "sub_scores": sub_scores,
        "contradictions": contradictions,
        "regime": regime,
    }
```

Latency: ~50 ms end-to-end.

## Dynamic-k sigmoid synthesis

Naive weighted averages have a failure mode in trading: when sub-scores agree strongly (+1, +1, +1, +1, +1, +1), the agent should be **very** confident; when they disagree, the agent should be **cautious**, not neutral.

HydraQuant's `_synthesize` uses a **dynamic-k sigmoid** where `k` scales with sub-score alignment:

- All six agree → high k → sigmoid saturates → confidence ≈ 0.9
- Three agree, three disagree → low k → sigmoid near linear → confidence ≈ 0.5
- All six disagree → k near zero → confidence explicitly pulled toward abstention

## Abstention — the uncommon default

Most ML pipelines emit a "neutral signal with 0.01 confidence" when they have no data. That value poisons every downstream statistic (signal_health, RLAIF reward, calibrator).

HydraQuant raises `EvidenceEngineDataError` (`:43`, `:122`) on insufficient data. Explicit comment in code:

> "we no longer emit NEUTRAL/0.01 rows that poison downstream statistics (signal_health, RLAIF, calibrator)."

The strategy handles the exception — it becomes a shadow trade, a forgone-P&L entry, or nothing, depending on context. The downstream pipes stay clean.

## The "blind sub-score" handling

If Q5 (macro) has no data (e.g. weekend, macro fetcher failed), its weight doesn't get ignored or defaulted — the remaining five weights **renormalize**. Q1 + Q2 + Q3 + Q4 + Q6 sum to 0.90 → divide by 0.90 → new weights.

This is done in `_synthesize` (`:1047`). The alternative — pretending Q5 is 0.5 — would introduce a persistent bias the organism would spend weeks un-learning.

## Organism-adaptive threshold

The downstream RAG pipeline gate is:

```python
# rag_graph.py:2121
_ee_threshold = _np("rag.evidence_first_threshold", 0.70)
if ee_confidence >= _ee_threshold:
    return signal   # fast path — no LLM
```

`rag.evidence_first_threshold` is itself a **neuron**. It starts at 0.70 and drifts up (requires stronger evidence to skip LLMs) when LLM debates consistently disagree with Evidence-only signals, and drifts down (trusts Evidence more) when debates consistently agree.

When no organism is attached (tests, isolated runs), fallback is 0.40 (safer / uses LLMs more).

## Background MADAM enrichment

Even on the fast path, a background thread fires MADAM debate to enrich the semantic cache for the next cycle:

```python
# rag_graph.py:2115
if ee_conf >= threshold:
    threading.Thread(
        target=_background_madam,
        args=(pair, regime, evidence_factsheet),
        daemon=True,
    ).start()
    return {"signal": signal, "confidence": ee_conf, "source": "EVIDENCE_ENGINE"}
```

The cache gets a higher-quality entry for the next call, even though the current call returned fast. This is amortization — LLM cost spread across many calls.

## Signal source telemetry

Every signal is tagged with its source so we can measure the fast-path / slow-path ratio:

- `EVIDENCE_ENGINE` — fast path
- `COORDINATOR` — slow path, coordinator synthesized R2/R3 output
- `AGENT_POOL_R1` / `AGENT_POOL_R2` — AgentPool came to consensus
- `ENSEMBLE` — multi-source voting used

The deploy health check (`deploy_health_check.py` section 3) requires at least 25% of signals to come from COORDINATOR + AGENT_POOL combined — otherwise LLMs aren't earning their keep, and investigation is triggered.

## Contradiction detection

`_detect_contradictions` (`:278`) flags:

- Q1 bullish + Q2 bearish (price up but momentum down = momentum trap)
- Q3 extreme greed + Q1 extreme bull (crowd euphoria at top)
- Q5 macro crash + any bullish = hedge warning
- Q6 volatility > 2σ + any strong signal = execution-risk warning

Contradictions do **not** cancel the signal. They:
- Reduce the confidence multiplier
- Feed into `evidence_audit_log` for later analysis
- Increase Chandelier ATR tighter (cortisol rises)

## See also

- [Architecture](ARCHITECTURE.md) — where Evidence Engine sits in the pipeline
- [Neural Organism](NEURAL_ORGANISM.md) — why the threshold is a neuron
- [LLM Routing](LLM_ROUTING.md) — what happens on the slow path
