# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# isort: skip_file
import logging
import sqlite3
import sys
import pandas as pd
import numpy as np
import os
from datetime import datetime
from freqtrade.strategy import IStrategy
from freqtrade.persistence import Trade
from freqtrade.strategy import CategoricalParameter, DecimalParameter, IntParameter
import talib.abstract as ta

# Add scripts dir to path for AI module imports
_scripts_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'scripts')
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)

from forgone_pnl_engine import ForgonePnLEngine
from confidence_calibrator import ConfidenceCalibrator

logger = logging.getLogger(__name__)

class AIFreqtradeSizer(IStrategy):
    """
    AI-powered strategy focusing on the "Sizing not Blocking" motto.
    Uses our own LLM Router + RAG pipeline (not FreqAI) for trade decisions.
    Injects real-time SQLite sentiment metrics into the feature set.
    """
    
    INTERFACE_VERSION = 3

    process_only_new_candles = True
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False
    can_short = True  # Futures: enable both LONG and SHORT
    startup_candle_count = 60  # Enough for SMA 50 + ATR 14

    # Minimal ROI (handled mostly by AI and custom stoploss)
    minimal_roi = {
        "0": 0.15,
        "60": 0.05,
        "120": 0.02,
        "240": 0
    }

    # Stoploss (Wide, rely on dynamic trailing/custom stoploss)
    stoploss = -0.20
    use_custom_stoploss = True

    # Trailing stop
    trailing_stop = False
    
    timeframe = '1h'

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.db_path = os.path.join(self.config['user_data_dir'], "db", "ai_data.sqlite")
        self.rag_script_path = os.path.join(self.config['user_data_dir'], "scripts", "rag_graph.py")
        self.ai_signal_cache = {} # Memory cache: { "BTC/USDT": {"signal": "BULLISH", "confidence": 0.8, "timestamp": datetime} }
        self.cache_ttl_hours = 4 # The LLM decision is valid for 4 hours unless trend sharply breaks
        
        # Phase 3.5: Forgone P&L Engine — tracks every missed signal
        self.forgone_engine = ForgonePnLEngine(db_path=self.db_path)
        # Map pair -> forgone_id for resolving on trade exit
        self._forgone_ids: dict = {}
        
        # Risk/Position Management Modules
        from risk_budget import RiskBudgetManager
        from position_sizer import BayesianKelly, PositionSizer
        from telegram_notifier import AITelegramNotifier
        from autonomy_manager import AutonomyManager
        
        self.risk_budget = RiskBudgetManager(db_path=self.db_path)
        self._bayesian_kelly = BayesianKelly(db_path=self.db_path)
        self.autonomy_manager = AutonomyManager(db_path=self.db_path)
        
        self._position_sizer = PositionSizer()
        # Share instances with the PositionSizer to ensure state synchronization
        self._position_sizer.bayesian_kelly = self._bayesian_kelly
        self._position_sizer.autonomy = self.autonomy_manager
        
        self._telegram = AITelegramNotifier()
        self._last_portfolio_sync = None  # Track last sync time

        logger.info("AIFreqtradeSizer initialized with MADAM-RAG, Forgone PNL, Risk Budget & Telegram.")

    def _get_sqlite_connection(self):
        try:
            conn = sqlite3.connect(self.db_path)
            return conn
        except sqlite3.Error as e:
            logger.error(f"Error connecting to AI SQLite DB: {e}")
            return None

    def _get_ai_signal(self, pair: str, current_time: datetime) -> dict:
        """
        The Bridge (Phase 5.1): Asks the RAG Signal Service for a decision.
        HTTP-first with subprocess fallback. Models stay loaded in the service.
        """
        # 1. Check Memory Cache
        cached = self.ai_signal_cache.get(pair)
        if cached:
            time_diff = (current_time - cached['timestamp']).total_seconds() / 3600
            if time_diff < self.cache_ttl_hours:
                return cached

        # 2. Cache Miss → HTTP call to RAG Signal Service
        logger.info(f"AI Signal Cache Miss for {pair}. Asking RAG Signal Service...")
        signal_data = {"signal": "NEUTRAL", "confidence": 0.0, "timestamp": current_time}

        try:
            import requests
            rag_service_url = self.config.get('ai_config', {}).get(
                'rag_service_url', 'http://127.0.0.1:8891')

            response = requests.get(
                f"{rag_service_url}/signal/{pair}",
                timeout=330  # slightly more than the 300s internal pipeline timeout
            )
            if response.status_code == 200:
                parsed = response.json()
                signal_data["signal"] = parsed.get("signal", "NEUTRAL")
                signal_data["confidence"] = parsed.get("confidence", 0.0)
                signal_data["reasoning"] = parsed.get("reasoning", "")
                logger.info(f"RAG Signal: {signal_data['signal']} ({signal_data['confidence']}) for {pair}")
            else:
                logger.warning(f"RAG service returned {response.status_code} for {pair}")
        except Exception as e:
            is_connection_error = False
            try:
                import requests as _req
                is_connection_error = isinstance(e, _req.exceptions.ConnectionError)
            except Exception:
                pass

            if is_connection_error:
                logger.warning(f"RAG service not running. Falling back to subprocess for {pair}")
                self._get_ai_signal_subprocess(pair, signal_data)
            else:
                logger.error(f"Error calling RAG Signal Service for {pair}: {e}")

        # 3. Save to Cache (but NEVER cache NEUTRAL — retry next candle)
        if signal_data.get("signal") != "NEUTRAL":
            self.ai_signal_cache[pair] = signal_data
        return signal_data

    def _get_ai_signal_subprocess(self, pair: str, signal_data: dict):
        """Legacy subprocess fallback — only used if HTTP service is down."""
        try:
            import subprocess
            import json
            result = subprocess.run(
                [sys.executable, self.rag_script_path, f"--pair={pair}"],
                capture_output=True, text=True, check=True, timeout=120
            )
            output = result.stdout
            if "--- JSON OUTPUT ---" in output:
                json_str = output.split("--- JSON OUTPUT ---")[1].strip()
                parsed = json.loads(json_str)
                signal_data["signal"] = parsed.get("signal", "NEUTRAL")
                signal_data["confidence"] = parsed.get("confidence", 0.0)
                logger.info(f"[Subprocess Fallback] {signal_data['signal']} ({signal_data['confidence']}) for {pair}")
        except subprocess.TimeoutExpired:
            logger.warning(f"Subprocess timed out for {pair} (120s)")
        except Exception as e:
            logger.error(f"Subprocess fallback failed for {pair}: {e}")

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """Compute technical indicators and sentiment features for sizing/stoploss."""
        # Technical indicators
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['adx'] = ta.ADX(dataframe, timeperiod=14)
        dataframe['ema_20'] = ta.EMA(dataframe, timeperiod=20)
        dataframe['sma_50'] = ta.SMA(dataframe, timeperiod=50)

        macd = ta.MACD(dataframe)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['macdhist'] = macd['macdhist']

        bollinger = ta.BBANDS(dataframe, timeperiod=20, nbdevup=2.0, nbdevdn=2.0)
        dataframe['bb_lower'] = bollinger['lowerband']
        dataframe['bb_upper'] = bollinger['upperband']

        # Sentiment features from SQLite (used by custom_stake_amount)
        conn = self._get_sqlite_connection()
        if conn:
            pair = metadata['pair']
            base_coin = pair.split('/')[0]
            try:
                fng_df = pd.read_sql_query(
                    "SELECT value as fng_value FROM fear_and_greed ORDER BY timestamp DESC LIMIT 1", conn)
                dataframe['%-fng_index'] = fng_df['fng_value'].iloc[0] if not fng_df.empty else 50
            except Exception:
                dataframe['%-fng_index'] = 50
            try:
                sent_df = pd.read_sql_query(
                    "SELECT sentiment_1h, sentiment_4h, sentiment_24h FROM coin_sentiment_rolling "
                    "WHERE coin = ? ORDER BY timestamp DESC LIMIT 1", conn, params=(base_coin,))
                if not sent_df.empty:
                    dataframe['%-sentiment_1h'] = sent_df['sentiment_1h'].iloc[0]
                    dataframe['%-sentiment_4h'] = sent_df['sentiment_4h'].iloc[0]
                    dataframe['%-sentiment_24h'] = sent_df['sentiment_24h'].iloc[0]
                else:
                    dataframe['%-sentiment_1h'] = 0.0
                    dataframe['%-sentiment_4h'] = 0.0
                    dataframe['%-sentiment_24h'] = 0.0
            except Exception:
                dataframe['%-sentiment_1h'] = 0.0
                dataframe['%-sentiment_4h'] = 0.0
                dataframe['%-sentiment_24h'] = 0.0
            conn.close()
        else:
            dataframe['%-fng_index'] = 50
            dataframe['%-sentiment_1h'] = 0.0
            dataframe['%-sentiment_4h'] = 0.0
            dataframe['%-sentiment_24h'] = 0.0

        return dataframe

    def populate_entry_trend(self, df: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        pair = metadata['pair']
        df['enter_long'] = 0
        df['enter_short'] = 0

        if self.dp.runmode.value in ('dry_run', 'live'):
            last_time = df['date'].iloc[-1]
            current_rate = df['close'].iloc[-1]

            # Phase 10: Invalidate semantic cache if sudden market movement is >3%
            if len(df) > 1:
                prev_close = df['close'].iloc[-2]
                if prev_close > 0 and abs(current_rate - prev_close) / prev_close > 0.03:
                    logger.info(f"Significant price movement >3% detected for {pair}. Invalidating semantic cache.")
                    if not hasattr(self, '_semantic_cache'):
                        from semantic_cache import SemanticCache
                        self._semantic_cache = SemanticCache(db_path=self.db_path)
                    self._semantic_cache.invalidate(pair=pair)

            ai_decision = self._get_ai_signal(pair, last_time)
            signal_type = ai_decision.get('signal', 'NEUTRAL')
            confidence = ai_decision.get('confidence', 0.0)
            is_bullish = signal_type == 'BULLISH'
            is_bearish = signal_type == 'BEARISH'

            # Forgone P&L: Log signal as NOT executed here.
            # Actual execution is confirmed in confirm_trade_entry().
            if signal_type != 'NEUTRAL':
                fid = self.forgone_engine.log_forgone_signal(
                    pair=pair,
                    signal_type="BULL" if is_bullish else "BEAR",
                    confidence=confidence,
                    entry_price=float(current_rate),
                    was_executed=False  # Will be updated in confirm_trade_entry
                )
                if fid:
                    self._forgone_ids[pair] = fid

            # Set entry signals based on AI decision (only last candle)
            if is_bullish:
                df.iloc[-1, df.columns.get_loc('enter_long')] = 1
            elif is_bearish:
                df.iloc[-1, df.columns.get_loc('enter_short')] = 1
        else:
            # Backtesting: Simple technical signals
            if 'rsi' in df.columns and 'macd' in df.columns:
                df.loc[(df['rsi'] < 35) & (df['macd'] > df['macdsignal']), 'enter_long'] = 1
                df.loc[(df['rsi'] > 65) & (df['macd'] < df['macdsignal']), 'enter_short'] = 1

        return df

    def populate_exit_trend(self, df: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        df['exit_long'] = 0
        df['exit_short'] = 0

        if self.dp.runmode.value in ('dry_run', 'live'):
            # In live mode, exits are handled by custom_stoploss + ROI + AI reversal signals
            pair = metadata['pair']
            cached = self.ai_signal_cache.get(pair)
            if cached:
                if cached['signal'] == 'BEARISH':
                    df.iloc[-1, df.columns.get_loc('exit_long')] = 1
                elif cached['signal'] == 'BULLISH':
                    df.iloc[-1, df.columns.get_loc('exit_short')] = 1
        else:
            # Backtesting: Technical exit signals
            if 'rsi' in df.columns and 'macd' in df.columns:
                df.loc[(df['rsi'] > 70) & (df['macd'] < df['macdsignal']), 'exit_long'] = 1
                df.loc[(df['rsi'] < 30) & (df['macd'] > df['macdsignal']), 'exit_short'] = 1

        return df

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:
        """
        Dynamic ATR-based stoploss. Sizing manages risk, so we allow wide breathing room
        but cut if trend drastically reverses (e.g. 3x ATR).
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()
        
        if 'atr' in last_candle:
            atr = last_candle['atr']
            # Allow 3 ATRs of breathing room relative to current rate
            new_stop = (current_rate - (3 * atr)) / current_rate
            result = new_stop - 1
            # Never exceed hard stop of -0.20
            return max(result, self.stoploss)
            
        return self.stoploss

    def _sync_portfolio_to_ai(self):
        """Bridge: Sync real exchange balance → AI modules (RiskBudget, Autonomy)."""
        try:
            stake = self.config.get('stake_currency', 'USDT')
            total = self.wallets.get_total(stake)
            free = self.wallets.get_free(stake)

            if total <= 0:
                return

            # Update RiskBudget with real portfolio value
            self.risk_budget.update_portfolio_value(total)

            # Persist to SQLite so scheduler/API can read it
            import json
            all_balances = {}
            total_portfolio_usd = total  # Start with stake currency

            for currency, wallet in self.wallets._wallets.items():
                if wallet.total > 0:
                    amount = round(wallet.total, 8)
                    if currency == stake:
                        all_balances[currency] = {"amount": amount, "usd": round(amount, 2)}
                    else:
                        usd = 0.0
                        try:
                            pair = f"{currency}/{stake}"
                            ticker = self.wallets._exchange.fetch_ticker(pair)
                            price = ticker.get('last', 0) or 0
                            usd = round(amount * price, 2)
                            total_portfolio_usd += usd
                        except Exception:
                            pass
                        all_balances[currency] = {"amount": amount, "usd": usd}

            conn = self._get_sqlite_connection()
            if conn:
                try:
                    conn.execute('''
                        CREATE TABLE IF NOT EXISTS portfolio_state (
                            id INTEGER PRIMARY KEY CHECK (id = 1),
                            stake_currency TEXT, total_balance REAL,
                            free_balance REAL, in_trades REAL,
                            assets_json TEXT, updated_at TEXT
                        )
                    ''')
                    in_trades = total - free
                    conn.execute('''
                        INSERT OR REPLACE INTO portfolio_state
                        (id, stake_currency, total_balance, free_balance, in_trades, assets_json, updated_at)
                        VALUES (1, ?, ?, ?, ?, ?, ?)
                    ''', (stake, total, free, in_trades, json.dumps(all_balances),
                          datetime.utcnow().isoformat()))
                    conn.commit()
                finally:
                    conn.close()

            self._last_portfolio_sync = current_time if hasattr(self, '_last_portfolio_sync') else datetime.utcnow()
            logger.debug(f"[Portfolio Sync] {stake}: stake=${total:.2f} total_usd=${total_portfolio_usd:.2f} assets={len(all_balances)}")
        except Exception as e:
            logger.debug(f"[Portfolio Sync] Skipped: {e}")

    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                            proposed_stake: float, min_stake: float, max_stake: float,
                            leverage: float, entry_tag: str, side: str,
                            **kwargs) -> float:
        """
        CORE PRINCIPLE: TRADE-FIRST AUTONOMY (Sizing not blocking).
        Instead of blocking a trade, we scale the size based on FreqAI confidence/market regime.
        """
        # Sync real exchange balance to AI modules (every trade entry)
        self._sync_portfolio_to_ai()

        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()

        # Base multiplier
        multiplier = 1.0
        
        # Scale based on sentiment & F&G
        if '%-fng_index' in last_candle:
            fng = last_candle['%-fng_index']
            # If Extreme Greed (> 80) or Extreme Fear (< 20), we reduce stake (contrarian caution)
            if fng > 80 or fng < 20:
                multiplier *= 0.5
                
        if '%-sentiment_24h' in last_candle:
            sent_24h = last_candle['%-sentiment_24h']
            # Positive sentiment gives slight sizing boost
            if sent_24h > 0.5:
                multiplier *= 1.2
            elif sent_24h < -0.5:
                # Still trade, but 70% smaller
                multiplier *= 0.3
                
        # Final Position Sizing math
        # We start with Kelly/Base Stake
        final_stake = proposed_stake * multiplier
        
        if self.dp.runmode.value in ('dry_run', 'live'):
            # Modulate stake heavily based on RAG Brain's LLM Confidence (Phase 3.5.2 logic)
            ai_decision = self._get_ai_signal(pair, current_time)
            confidence = ai_decision.get('confidence', 0.5)
            
            # Phase 6.3: Calibrate confidence using historical accuracy
            try:
                if not hasattr(self, '_calibrator'):
                    self._calibrator = ConfidenceCalibrator(db_path=self.db_path)
                confidence = self._calibrator.adjust_confidence(confidence)
            except Exception as e:
                logger.debug(f"Confidence calibration skipped: {e}")
            
            # Görev 1 Fix: Use PositionSizer to calculate fraction, which respects BayesianKelly and Autonomy logic
            fraction = self._position_sizer.calculate_stake_fraction(confidence)
            
            # Let it scale down to "dust" sizes if confidence is terribly low
            final_stake = final_stake * fraction
            
            # Phase 3.5.3: Risk Budget scaling — shrink if budget running low
            final_stake = self.risk_budget.scale_position(final_stake)

            # Autonomy max_stake cap (scales with real portfolio)
            portfolio_val = self.risk_budget.portfolio_value
            autonomy_cap = self.autonomy_manager.get_max_stake(portfolio_value=portfolio_val)
            if autonomy_cap is not None:
                final_stake = min(final_stake, autonomy_cap)

            # Consume budget for this trade
            atr_volatility = last_candle.get('atr', 0.02) / current_rate if current_rate > 0 else 0.02
            self.risk_budget.consume_budget(final_stake, atr_volatility, confidence)

        # Trade-First: ALWAYS trade at least min_stake. Confidence modulates SIZE, never PERMISSION.
        if final_stake < min_stake:
            final_stake = min_stake

        return min(final_stake, max_stake)

    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float,
                            time_in_force: str, current_time: datetime, entry_tag: str,
                            side: str, **kwargs) -> bool:
        """Mark the forgone P&L entry as ACTUALLY executed (trade opened)."""
        ai_decision = self.ai_signal_cache.get(pair, {})
        confidence = ai_decision.get('confidence', 0.5)
        signal_type = "BULL" if side == "long" else "BEAR"
        reasoning = ai_decision.get('reasoning', "Technical entry with AI confirmation")

        # Update the existing forgone entry (logged in populate_entry_trend) to was_executed=True
        fid = self._forgone_ids.pop(pair, None)
        if fid:
            self.forgone_engine.mark_executed(fid)

        logger.info(f"[Trade Entry] {pair} {signal_type} conf={confidence:.2f} stake=${amount*rate:.2f} — {reasoning}")
        return True

    def confirm_trade_exit(self, pair: str, trade: 'Trade', order_type: str, amount: float,
                           rate: float, time_in_force: str, exit_reason: str,
                           current_time: datetime, **kwargs) -> bool:
        """Resolve forgone trades and update Bayesian Kelly with trade outcome."""
        # Forgone P&L resolution
        fid = self._forgone_ids.pop(pair, None)
        if fid:
            self.forgone_engine.resolve_forgone_trade(fid, exit_price=rate)
        
        logger.info(f"[Trade Exit] {pair} reason={exit_reason}")

        # Phase 3.5.2: Bayesian Kelly update — learn from this trade
        try:
            pnl_pct = trade.calc_profit_ratio(rate) if hasattr(trade, 'calc_profit_ratio') else 0.0
            won = pnl_pct > 0
            # Eagerly initialized in __init__
            self._bayesian_kelly.update(won=won, pnl_pct=pnl_pct)
            logger.info(f"[BayesianKelly] Updated: {'WIN' if won else 'LOSS'} pnl={pnl_pct:.4f} → win_p={self._bayesian_kelly.win_probability():.3f} kelly_f={self._bayesian_kelly.kelly_fraction():.4f}")
        except Exception as e:
            logger.warning(f"[BayesianKelly] Update failed: {e}")

        # Hypothetical $100 Portfolio: compound every closed trade (position-size weighted)
        try:
            trade_pnl_pct = (trade.calc_profit_ratio(rate) * 100) if hasattr(trade, 'calc_profit_ratio') else 0.0
            # Weight by position size fraction so $100 sim mirrors real sizing
            portfolio_value = self.risk_budget.portfolio_value
            stake_fraction = (trade.stake_amount / portfolio_value) if portfolio_value > 0 else 0.01
            portfolio_pnl_pct = trade_pnl_pct * stake_fraction
            self.forgone_engine.record_trade_for_portfolio(pair, portfolio_pnl_pct)
        except Exception as e:
            logger.warning(f"[Portfolio] Update failed: {e}")

        return True
