# GOING LIVE PLAYBOOK
## An Engineer's Guide to Transitioning from Testnet to Real Money

*Compiled March 2026 from current research, institutional best practices, and hard-won lessons from algo trading failures.*

---

## TABLE OF CONTENTS

1. [Testnet-to-Live Checklist](#1-testnet-to-live-checklist)
2. [Start SMALL: The $100 Rule](#2-start-small-the-100-rule)
3. [Trading Psychology for Bot Operators](#3-trading-psychology-for-bot-operators)
4. [Monitoring Practices](#4-monitoring-practices)
5. [Kill Switch Protocols](#5-kill-switch-protocols)
6. [Renaissance/Jim Simons Lessons](#6-renaissancejim-simons-lessons)
7. [AI Trading Pitfalls](#7-ai-trading-pitfalls)
8. [Realistic Performance Expectations](#8-realistic-performance-expectations)
9. [Tax Implications (Turkey)](#9-tax-implications-turkey)
10. [The "Edge" Concept and Alpha Decay](#10-the-edge-concept-and-alpha-decay)
11. [The Bitcoin Correlation Problem](#11-the-bitcoin-correlation-problem)
12. [Realistic Expectations and Survivorship Bias](#12-realistic-expectations-and-survivorship-bias)

---

## 1. TESTNET-TO-LIVE CHECKLIST

### Minimum Testnet Duration

| Strategy Type | Minimum Duration | Minimum Trades | Confidence Level |
|---|---|---|---|
| Day/Scalp (many trades/day) | 2-3 months | 200-500 | High |
| Swing (trades lasting days) | 4-6 months | 100-200 | Medium-High |
| Position (trades lasting weeks) | 6-12 months | 50-100 | Medium |

**Statistical significance requires at minimum 100 trades.** Institutional standards demand 200-500 trades across multiple market regimes (trending up, trending down, sideways, high volatility, low volatility). Thirty trades is the absolute mathematical floor for any inference at all, but it is nowhere near enough for confidence.

### Pre-Live Verification Checklist

**Infrastructure:**
- [ ] Bot has run continuously for 30+ days without crashes
- [ ] API connectivity has been stable (measure uptime %)
- [ ] Reconnection logic works after network drops
- [ ] Order execution matches expected behavior (fills, partial fills, cancellations)
- [ ] Logging captures every decision, order, fill, and error
- [ ] All 112 unit tests pass
- [ ] Smoke test passes on production config

**Strategy Validation:**
- [ ] Backtest results match testnet results within 15% margin
- [ ] Strategy has been profitable across at least 2 different market conditions
- [ ] Drawdown on testnet never exceeded your pre-defined maximum (e.g., 25%)
- [ ] Win rate and profit factor are consistent (not degrading over time)
- [ ] Sharpe ratio > 1.0 on testnet (> 1.5 preferred)
- [ ] Testnet includes realistic fees (Binance: 0.1% maker/taker or your actual fee tier)

**Risk Controls:**
- [ ] Maximum position size is hard-coded (your 3% cap)
- [ ] Daily loss limit is configured and tested
- [ ] Kill switch has been tested (manually triggered at least once)
- [ ] Stoploss logic works correctly for both long and short
- [ ] No single trade can lose more than X% of portfolio
- [ ] Protections (CooldownPeriod, StoplossGuard, MaxDrawdown) are active

**Operational:**
- [ ] Telegram alerts are working for all critical events
- [ ] You have documented your strategy's expected behavior
- [ ] You have a written plan for what to do during drawdowns (read it BEFORE going live)
- [ ] You have set a "walk away" loss amount you can afford to lose entirely
- [ ] Exchange API keys have minimal permissions (trade only, no withdrawal)
- [ ] 2FA is enabled on the exchange account

**Paper-to-Live Gap Awareness:**
- [ ] You understand slippage will be worse in live (paper trading assumes perfect fills)
- [ ] You understand liquidity may be thin for some pairs at some hours
- [ ] You understand your testnet results represent the CEILING, not the floor

---

## 2. START SMALL: THE $100 RULE

### Why $100-500 Even If Your Bot Is "Ready"

**The cold truth:** 73% of automated crypto trading accounts fail within six months. Small configuration errors cost traders an average of 35% of capital before the issue is even identified. Your bot is sophisticated, but "works on testnet" is not the same as "works with real money."

### The Staging Ladder

| Stage | Capital | Duration | Purpose | Graduation Criteria |
|---|---|---|---|---|
| Stage 0 | $0 (testnet) | 3-6 months | Validate logic | 200+ trades, Sharpe > 1.0 |
| Stage 1 | $100 | 1-2 months | Discover live-vs-paper gaps | Profitable, no critical bugs |
| Stage 2 | $250 | 1-2 months | Validate with slightly larger positions | Consistent with testnet within 20% |
| Stage 3 | $500 | 2-3 months | Confirm edge persists | Still profitable after fees + slippage |
| Stage 4 | $1,000 | 3+ months | Meaningful but still affordable | Drawdown within expected bounds |
| Stage 5 | Scale up gradually | Ongoing | Never more than you can lose | Never risk rent/food/emergency money |

### What $100 Teaches You That Testnet Cannot

1. **Slippage is real.** Paper trading gives perfect fills. Live trading does not. On a $100 account, even $0.50 slippage per trade is 0.5%. Over hundreds of trades, this compounds.
2. **Fees matter more at small scale.** Binance 0.1% fee on a $10 trade = $0.01. On 50 trades/day = $0.50/day = $15/month. On a $100 account, that is 15% monthly overhead just from fees.
3. **Your emotional response is different.** "Even the best trading bot will fail if your psychology isn't prepared for what real trading feels like." Losing $5 of real money feels different from losing $5,000 of testnet money.
4. **Edge-case bugs appear.** Minimum order sizes, rounding errors, rate limits under real load, exchange maintenance windows -- these only surface with real money.

### The Fee Math You Must Do

```
Monthly trades estimate:        ___
Average trade size:             ___
Fee per trade (maker + taker):  ___
Monthly fee cost:               ___
Monthly fee as % of capital:    ___

If this number is > 5%, your account is too small for your strategy's trade frequency.
```

---

## 3. TRADING PSYCHOLOGY FOR BOT OPERATORS

### The Paradox: Automation Does NOT Eliminate Emotion

"Automation doesn't mean freedom from emotion -- it means you're forced to trust something you didn't create... Traders jump into automated systems expecting peace of mind. But after just a few trades, the panic kicks in. They pull the bot."

Even though you DID create this bot, you will still experience:

### The Five Emotional Traps

**Trap 1: The Refresh Loop**
You will check your phone every 5 minutes. You will watch every trade. You will lose sleep. This is normal. Set a rule: check at most 3x per day (morning, afternoon, end of day). The daily Telegram report exists for a reason.

**Trap 2: "Just This Once" Override**
The market is crashing. Your bot just opened a long. Every fiber of your being screams "close it manually!" DO NOT. If you override the computer, you are no longer running a systematic strategy -- you are an emotional trader with a fancy tool. (See Section 6: Renaissance never overrides.)

**Trap 3: Drawdown Panic**
Your bot will have losing streaks. A 10-15% drawdown is NORMAL for most strategies. A 20% drawdown is uncomfortable but not necessarily broken. You must decide BEFORE going live what drawdown level triggers a review vs. a shutdown. Write it down. Tape it to your monitor.

**Trap 4: Recency Bias After Wins**
After a winning week, you will want to increase position sizes or add more capital. Do NOT scale up during a winning streak. Scale up only after calm, scheduled reviews (monthly at earliest).

**Trap 5: Tinkering**
You will be tempted to "improve" the strategy while it is running live. Every parameter change resets your statistical sample. If you want to test changes, run them on a SEPARATE testnet instance. Never live-edit a running production strategy.

### Pre-Commitment Contract (Write This Down and Sign It)

```
I, [name], commit to the following rules before going live on [date]:

1. I will not manually override the bot unless the kill switch criteria (Section 5) are met.
2. I will not increase capital for at least 3 months.
3. I will check the bot at most 3 times per day.
4. I will not change strategy parameters on the live instance.
5. My maximum acceptable loss is $___. If I hit this, I stop.
6. I understand I may lose ALL of my initial capital.
7. I have read this document in full and understand the risks.

Signed: _______________  Date: _______________
```

---

## 4. MONITORING PRACTICES

### Daily Checks (5 minutes, ideally via Telegram report)

- [ ] Bot is running (process alive, no crash logs)
- [ ] Trades executed in last 24h (or expected quiet period)
- [ ] Current open positions and unrealized P&L
- [ ] Daily realized P&L
- [ ] No error messages or API failures in logs
- [ ] Exchange balance matches expected value

### Weekly Checks (30 minutes)

- [ ] Win rate vs. expected (is it degrading?)
- [ ] Average win size vs. average loss size
- [ ] Profit factor (gross profit / gross loss) -- should be > 1.2
- [ ] Maximum drawdown this week
- [ ] Slippage analysis: expected fill price vs. actual fill price
- [ ] Number of rejected/failed orders
- [ ] Fear & Greed Index trend vs. bot behavior
- [ ] Forgone P&L: what would rejected signals have earned?

### Monthly Checks (1-2 hours)

- [ ] Sharpe ratio (rolling 30-day)
- [ ] Compare live performance to backtest expectations
- [ ] Compare live performance to simple buy-and-hold BTC
- [ ] Review all losing trades: are losses within expected parameters?
- [ ] Check for regime change: has the market shifted (trending vs. ranging)?
- [ ] Review AI confidence scores: are they calibrated? (high confidence = high win rate?)
- [ ] Infrastructure: disk space, memory usage, API rate limit headroom
- [ ] Cost analysis: fees paid, slippage cost, API costs (Gemini/Groq)
- [ ] Tax P&L update (see Section 9)

### Key Metrics Dashboard

| Metric | Green | Yellow | Red |
|---|---|---|---|
| Daily P&L | > 0 | -1% to 0 | < -1% |
| Weekly P&L | > 0 | -3% to 0 | < -3% |
| Max Drawdown (rolling) | < 10% | 10-20% | > 20% |
| Win Rate | > 52% | 48-52% | < 48% |
| Profit Factor | > 1.5 | 1.0-1.5 | < 1.0 |
| Sharpe Ratio | > 1.5 | 0.5-1.5 | < 0.5 |
| Bot Uptime | > 99% | 95-99% | < 95% |
| Order Fill Rate | > 95% | 85-95% | < 85% |
| AI Confidence Calibration | Accurate | Slight drift | Systematically wrong |
| Forgone P&L | Low | Rising | Consistently high |

---

## 5. KILL SWITCH PROTOCOLS

### Automatic Shutdown Triggers (Hard Rules -- No Exceptions)

| Trigger | Threshold | Action |
|---|---|---|
| Daily loss | > 5% of portfolio | Halt all new trades, close positions |
| Weekly loss | > 10% of portfolio | Halt all trading for 48 hours |
| Max drawdown | > 25% from peak | Full shutdown, manual review required |
| Consecutive losses | > 10 in a row | Pause trading, investigate |
| API errors | > 5 in 1 hour | Pause trading, alert via Telegram |
| Exchange outage | Any duration | Halt, wait for confirmed stability |
| Abnormal position size | > 2x expected | Emergency close, investigate |
| Strategy not trading | > 48h with no trades | Alert (may indicate bug or no signals) |

### Manual Shutdown Decision Tree

```
Is the bot losing money?
  |
  +-- YES
  |     |
  |     Is the loss within expected drawdown parameters?
  |       |
  |       +-- YES --> Do NOT stop. Trust the system. Review at next scheduled check.
  |       |
  |       +-- NO --> Has the market regime changed fundamentally?
  |             |
  |             +-- YES --> Pause. Analyze. May need strategy update.
  |             |
  |             +-- NO --> Is there a bug? Check logs.
  |                   |
  |                   +-- YES --> Fix bug. Restart on testnet first.
  |                   |
  |                   +-- NO --> You may be in normal drawdown.
  |                              Wait for the weekly review.
  |
  +-- NO (bot is profitable)
        |
        Is the bot behaving as expected?
          |
          +-- YES --> Do nothing. Let it run.
          |
          +-- NO --> Investigate unexpected behavior even if profitable.
                     A profitable bug is still a bug.
```

### After a Shutdown

1. Do NOT restart immediately
2. Analyze what went wrong (logs, trade history, market data)
3. Fix the issue
4. Test the fix on testnet for at least 1 week
5. Restart with REDUCED capital (50% of previous)
6. Gradually scale back up over 2-4 weeks

---

## 6. RENAISSANCE / JIM SIMONS LESSONS

### "We Never Override the Computer"

Jim Simons' most important rule: **"The only rule is that we never override the computer. No one ever comes in any day and says the computer wants to do this and that's crazy and we shouldn't do it."**

Peter Brown (CEO of Renaissance Technologies): **"There's a danger that comes with success, and to avoid this we try to remember that we know how to build large mathematical models and that is all we know. We don't know any economics, don't have any insights into the markets, and just don't interfere with our trading systems."**

### The 2007 Quant Quake: The Defining Moment

In August 2007, the Medallion Fund lost $1 billion -- 20% -- in three days. The partners gathered in Simons' office, ready to override the computers and sell everything.

**They didn't.**

The models kept trading. By December 2007, Medallion was up 85.9% for the year.

This is the single most important lesson in systematic trading: **the time you most want to override the system is exactly the time you should not.**

### What You Can Learn from Renaissance (And What You Cannot)

**Applicable to your bot:**
- Never override a systematic decision based on emotion or "gut feel"
- Data quality matters more than model complexity
- Small edge + high frequency + discipline = compounding returns
- The system will have losing days, weeks, even months. That is expected.
- Track everything. Log everything. Analyze everything.

**NOT applicable to your bot:**
- Medallion averaged 66% annual returns before fees. You will not achieve this. They employed 300+ PhDs, had proprietary data feeds, co-located servers, and $10B+ in capital.
- Medallion's returns are partly from leverage (net of leverage, returns were 3-7% from pure alpha). Your risk tolerance should not assume leveraged returns.
- Renaissance made money on just over 50% of trades, but on thousands of trades per day. If your bot trades 5x/day, you need a HIGHER win rate to survive.

### Simons' Principles Applied to Your Situation

| Simons' Principle | Your Application |
|---|---|
| "We start with data, not models" | Let your RAG pipeline surface patterns, don't force your AI to confirm your biases |
| "Never override the computer" | Write it on your wall. Your pre-commitment contract (Section 3) is this principle |
| "Be comfortable with signals you can't explain" | If the AI + RAG sees something statistical, trade it even if you don't understand why |
| "Small edge, many bets" | Don't concentrate. Keep position sizes small (your 3% cap is good) |
| "Hire smart people, not finance people" | You're an engineer, not a trader. That's actually an advantage -- less intuition to override the system |

---

## 7. AI TRADING PITFALLS

### LLM Hallucination Risk

**The core problem:** General-purpose LLMs trained on historical data are unaware of current market conditions. When asked about prices or trends, they will "produce a confident response built on outdated information." This is not a bug -- it is how LLMs work.

**Specific risks in your system:**
1. **Gemini generates a confident but wrong market analysis.** Your multi-layer anti-hallucination pipeline helps, but is not foolproof.
2. **Sentiment analysis on stale news.** If your news pipeline has even a 30-minute delay, the market may have already priced in the information.
3. **Self-reinforcing loops.** If the AI generates analysis, stores it in ChromaDB, then retrieves it later as "evidence," it is citing its own hallucinations.

**Mitigations:**
- Always cross-validate LLM outputs with on-chain or order-book data
- Never let a single LLM call be the sole reason for a trade
- Monitor hallucination rate: track how often AI confidence is high but the trade loses
- Your CRAG (Corrective RAG) and Self-RAG implementations are good defenses, but audit them monthly
- Treat social sentiment as "noise-heavy" -- powerful for regime detection but dangerous as a sole signal

### Sentiment Lag: News Is Already Priced In

By the time your bot:
1. Scrapes a news article (minutes to hours after publication)
2. Processes it through the RAG pipeline (seconds to minutes)
3. Generates a trading signal (seconds)
4. Executes the trade (seconds)

...institutional algorithms with direct news feeds have already traded on that information. Your edge from news is NOT in speed -- it is in synthesis (combining multiple signals that others miss).

### Overfitting to Recent Data

Your AI system learns from recent data. This creates recency bias:
- If the last 3 months were a bull market, your system learns "buy is usually right"
- When the regime changes, the system keeps buying into a crash
- **Solution:** Monitor performance across regimes. If your system only makes money in one direction, it does not have an edge -- it has a directional bet.

### The Complexity Trap

You have 10+ RAG techniques, MADAM debate, multi-agent systems, graduated autonomy, etc. Each layer adds:
- More potential failure points
- More parameters to overfit
- More complexity to debug
- More ways for errors to cascade

**The uncomfortable question:** Would a simple moving average crossover with proper position sizing outperform your system after accounting for all the infrastructure cost and complexity? Test this. If you cannot beat a simple benchmark by a meaningful margin, the complexity is not earning its keep.

---

## 8. REALISTIC PERFORMANCE EXPECTATIONS

### What Returns Are Actually Realistic?

| Performance Level | Annual Return | Context |
|---|---|---|
| Elite (top 1%) | 30-50% | Medallion Fund territory, 300+ PhDs, billions in infra |
| Excellent (top 5%) | 15-30% | Successful quant funds, years of refinement |
| Good (top 20%) | 5-15% | Solid systematic strategy, beats most retail traders |
| Average | 0-5% | May not beat buy-and-hold after fees |
| Below Average | Negative | Where most algo traders end up |

**Critical reality checks:**
- Only 10-30% of traders using AI bots achieve consistent profitability
- Most retail AI bots achieve 10-40% annually for well-tested strategies (before fees and slippage eat into this)
- "Anyone expecting passive income should look elsewhere -- trading bots require active management and carry substantial risk"
- Bitcoin buy-and-hold returned ~150% in 2023 and ~120% in 2024. If your bot returned 30%, you UNDERPERFORMED simply holding BTC.

### The Benchmark You Must Beat

Your bot's performance is meaningless in isolation. You must compare against:

1. **Buy and hold BTC:** If you cannot beat this, your bot is not adding value
2. **Buy and hold ETH:** Same logic
3. **Risk-free rate:** Turkish government bonds (~30-40% in TRY, ~4-5% in USD)
4. **Simple momentum strategy:** 200-day moving average, long above / cash below

If your complex AI system cannot beat strategy #4 above by at least 5% annually, the AI infrastructure is not paying for itself.

### Drawdown Expectations

Even excellent strategies experience:
- 10-15% drawdowns: Normal, happens multiple times per year
- 15-25% drawdowns: Uncomfortable but expected 1-2x per year
- 25-40% drawdowns: Rare but possible, may indicate strategy issues
- 40%+ drawdowns: Almost certainly broken, shut down and investigate

---

## 9. TAX IMPLICATIONS (TURKEY)

### Current Status (March 2026)

**Turkey has recently formalized crypto tax obligations.** The situation is evolving rapidly:

1. **Income Tax:** Profits from cryptocurrency trading are taxable as personal income under existing Income Tax Law. Progressive rates range from 15% to 40% depending on annual income bracket.

2. **2025 Calendar Year Reporting:** Crypto buy-sell profits, mining/staking gains, and airdrop income from the 2025 calendar year must be declared in the annual tax return between 1-31 March 2026.

3. **Transaction Tax:** A 0.03% (3/1000) transaction tax on sales and transfers applies on SPK-licensed platforms. This is a withholding (stopaj) tax -- it is final and does not require a separate return if you use a licensed platform.

4. **Licensed vs. Unlicensed Platforms:**
   - Licensed platforms (SPK-registered): Withholding tax is automatically deducted. This is a final tax.
   - Unlicensed platforms / foreign exchanges (e.g., Binance without Turkish license): You must file under the progressive income tax schedule via annual tax return.

5. **OECD CARF:** Starting in 2026, the Crypto Asset Reporting Framework (CARF) enables international automatic exchange of crypto transaction data. Foreign exchanges will report your transactions to Turkish tax authorities.

### Record Keeping Requirements

**You MUST track (this is non-negotiable):**

- [ ] Every trade: date, time, pair, direction, entry price, exit price, quantity, fees
- [ ] Running P&L in both crypto and TRY terms
- [ ] Cost basis for every asset (FIFO method is standard)
- [ ] All fees paid (trading fees, withdrawal fees, network fees)
- [ ] Exchange statements / export CSVs regularly (exchanges can and do shut down)
- [ ] API costs (Gemini, Groq, etc.) -- these may be deductible as business expenses

**Freqtrade already logs all trades in its SQLite database.** Export monthly. Keep backups. Your trade history is your tax evidence.

### Action Items

1. Consult a Turkish tax advisor (mali musavir) who understands crypto before going live
2. Determine if Binance is an SPK-licensed platform in Turkey as of your go-live date
3. Set up monthly P&L export routine from Freqtrade
4. Track all infrastructure costs (server, API fees) as potential business deductions
5. Consider registering as a sole proprietorship (serbest meslek) if trading income is significant

---

## 10. THE "EDGE" CONCEPT AND ALPHA DECAY

### What Is an Edge?

An edge is any repeatable, statistically significant advantage that generates positive expected value per trade after all costs. It is not:
- A winning streak (that is luck)
- A profitable backtest (that may be overfitting)
- An AI that sounds confident (that is a language model doing its job)

### Alpha Decay: Every Edge Dies

**Research finding:** Alpha on new trading strategies decays in approximately 12 months on average. Different strategy types have different lifespans:

| Strategy Type | Typical Edge Lifespan |
|---|---|
| High-frequency arbitrage | Days to weeks |
| Momentum / trend following | 3-6 months |
| Sentiment-based | 1-3 months (news gets priced in faster over time) |
| Statistical arbitrage | 6-12 months |
| Fundamental / value | 1-3 years |

### Why Edges Die

1. **Other traders discover the same pattern** and arbitrage it away
2. **Market structure changes** (new regulations, new participants, new instruments)
3. **Your data sources degrade** (free news APIs change, sentiment sources become gamed)
4. **Execution costs rise** (exchange fee changes, increased slippage from crowded trades)

### How to Detect Edge Decay

Monitor these metrics on a rolling basis:

```
Monthly:
- Rolling 30-day Sharpe ratio: trending down?
- Rolling 30-day win rate: declining?
- Average profit per trade: shrinking?
- Profit factor: approaching 1.0?

Quarterly:
- Compare last 3 months to previous 3 months
- If Sharpe dropped > 50%, investigate
- If profit factor < 1.1, edge may be gone

Decision:
- If 3 consecutive months of declining Sharpe AND profit factor < 1.0:
  --> Edge is likely dead. Stop live trading. Return to research.
```

### The Uncomfortable Truth About AI Trading Edges

Your AI/RAG system's "edge" likely comes from:
1. **Sentiment synthesis** -- combining multiple news/data sources. This edge decays as other AI systems do the same.
2. **Fear & Greed contrarian signals** -- buying extreme fear, selling extreme greed. This is one of the oldest and most studied edges. It works... until too many bots trade it.
3. **LLM reasoning about market conditions.** This is novel but unproven at scale. It may be an edge, or it may be sophisticated randomness.

**Test for a real edge:** Run your system against a random signal generator with the same position sizing and risk management. If your system cannot significantly outperform random, you do not have an edge -- you have risk management (which is valuable, but different).

---

## 11. THE BITCOIN CORRELATION PROBLEM

### The Uncomfortable Statistics

- **77% of top cryptocurrencies** correlate at more than 60% with Bitcoin
- Major altcoins (ETH, XRP, SOL, etc.) correlate with Bitcoin at **87%+**
- Monero holds the record at **92.3%** correlation
- During market stress, correlations approach **1.0** (everything dumps together)

### What This Means for Your Bot

If your bot trades altcoins (anything other than BTC/USDT), most of your P&L is likely explained by Bitcoin's direction. Your sophisticated AI system may be generating signals that amount to "Bitcoin is going up, therefore buy this altcoin" with extra steps and extra fees.

**Test this:** Calculate the correlation between your bot's daily returns and BTC daily returns.
- If correlation > 0.8: You are effectively a leveraged Bitcoin position
- If correlation 0.5-0.8: Your bot adds some value beyond BTC exposure
- If correlation < 0.5: Your bot may genuinely have independent alpha

### Implications

1. **In bull markets,** your bot will look like a genius. So will everyone else. So will buy-and-hold.
2. **In bear markets,** everything drops together. Your bot will lose money. So will everyone else.
3. **The real test** is whether your bot makes money in sideways / choppy markets where BTC goes nowhere. That is where true edge lives.
4. **Diversification is limited** in crypto. Unlike stocks (where different sectors move independently), crypto is one big correlated mass dominated by Bitcoin.

---

## 12. REALISTIC EXPECTATIONS AND SURVIVORSHIP BIAS

### The Brutal Statistics

| Metric | Value | Source |
|---|---|---|
| Day traders who make money | 1-20% | FINRA, academic studies |
| Day traders who lose money | 80-99% | Multiple studies |
| Automated crypto accounts that fail in 6 months | 73% | Industry data |
| Median day trader annual profit (2020) | ~$13,000 | FINRA (likely inflated by survivorship bias) |
| Algo traders achieving consistent profitability with AI bots | 10-30% | Industry surveys |

### Survivorship Bias: What You're Not Seeing

Every success story you've read about trading bots is a **survivor.** For every person posting "my bot made 50% this year," there are 4-9 people who lost money and said nothing. The statistics brokers publish are based on existing clients and **do not include people who lost money and closed their accounts.**

The YouTube videos, Medium articles, and Reddit posts about profitable bots are the lottery winners showing off their tickets. The losers are silent.

### What This Means for You

You have built an impressive system: 112 tests, 10+ RAG techniques, graduated autonomy, multi-agent debate, risk budgeting, Bayesian Kelly sizing. This puts you ahead of the 73% who deploy bots without backtesting.

**But it does not guarantee profitability.** The market does not care about your architecture.

### A Realistic Mental Model

Think of your trading bot as a startup:
- Most startups fail (same as most trading bots)
- Being well-engineered improves your odds but does not guarantee success
- The first version almost never works as planned
- You need to iterate based on real-world feedback (live trading data)
- Capital preservation is more important than growth in the early stages
- It takes 1-2 years to know if you have a real edge

### The Final Checklist Before You Go Live

```
[ ] I understand I will probably lose money in the first 1-3 months
[ ] I am starting with $100-500 that I can afford to lose entirely
[ ] I have a written pre-commitment contract (Section 3)
[ ] I have hard kill switch thresholds programmed (Section 5)
[ ] I will NOT override the computer based on emotion (Section 6)
[ ] I know my strategy's edge may decay and I know how to detect it (Section 10)
[ ] I understand that most of my crypto P&L is correlated to BTC (Section 11)
[ ] I have consulted (or will consult) a tax advisor before meaningful gains
[ ] I have a monitoring schedule (daily/weekly/monthly) written down
[ ] I am prepared for this to take 1-2 years before I know if it works
[ ] I am doing this to learn, not to get rich quick
```

---

## APPENDIX A: RECOMMENDED READING

1. **"The Man Who Solved the Market"** by Gregory Zuckerman -- The Jim Simons story. Required reading before going live.
2. **"Trading in the Zone"** by Mark Douglas -- Trading psychology fundamentals
3. **"Algorithmic Trading"** by Ernest Chan -- Practical quant trading
4. **"Advances in Financial Machine Learning"** by Marcos Lopez de Prado -- ML pitfalls in finance

## APPENDIX B: QUICK REFERENCE CARD

```
+------------------------------------------+
|         DAILY QUICK CHECK (2 min)        |
+------------------------------------------+
| 1. Bot running?              [ ] Y  [ ] N |
| 2. Trades today?             [ ] Y  [ ] N |
| 3. Daily P&L > -1%?         [ ] Y  [ ] N |
| 4. Any error alerts?        [ ] Y  [ ] N |
| 5. Balance matches expected? [ ] Y  [ ] N |
|                                          |
| If ANY = NO --> Investigate immediately  |
+------------------------------------------+

+------------------------------------------+
|      EMERGENCY DECISION (30 seconds)     |
+------------------------------------------+
| Is loss > 5% today?                      |
|   YES --> Kill switch NOW                |
|   NO  --> Is it a bug or a drawdown?     |
|          BUG    --> Fix, test, restart    |
|          DRAWDOWN --> Let it run          |
+------------------------------------------+
```

---

*This document was compiled from web research conducted in March 2026. Markets change. Regulations change. Review and update this playbook quarterly.*

*Remember: The goal is not to get rich. The goal is to learn, survive, and compound small edges over time. If you do that for 2 years, you will know more about trading than 95% of retail traders.*
