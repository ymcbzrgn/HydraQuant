# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# isort: skip_file
import logging
import sqlite3
import pandas as pd
import numpy as np
import os
from datetime import datetime
from freqtrade.strategy import IStrategy
from freqtrade.persistence import Trade
from freqtrade.strategy import CategoricalParameter, DecimalParameter, IntParameter
import talib.abstract as ta

logger = logging.getLogger(__name__)

class AIFreqtradeSizer(IStrategy):
    """
    FreqAI-powered strategy focusing on the "Sizing not Blocking" motto.
    Uses LightGBM under the hood and injects real-time SQLite sentiment
    metrics into the feature set.
    """
    
    INTERFACE_VERSION = 3

    # Enable FreqAI
    process_only_new_candles = True
    use_sell_signal = True
    sell_profit_only = False
    ignore_roi_if_buy_signal = False
    startup_candle_count = 200

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

    def _get_sqlite_connection(self):
        try:
            conn = sqlite3.connect(self.db_path)
            return conn
        except sqlite3.Error as e:
            logger.error(f"Error connecting to AI SQLite DB: {e}")
            return None

    def feature_engineering_expand_all(self, dataframe: pd.DataFrame, period: int,
                                       metadata: dict, **kwargs) -> pd.DataFrame:
        """
        Populate features for the AI model. Includes technicals AND our custom Sentiment DB metrics.
        """
        # --- 1. Technical Features ---
        dataframe['%-rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['%-mfi'] = ta.MFI(dataframe, timeperiod=14)
        dataframe['%-adx'] = ta.ADX(dataframe, timeperiod=14)
        dataframe['%-sma'] = ta.SMA(dataframe, timeperiod=20)
        dataframe['%-ema'] = ta.EMA(dataframe, timeperiod=20)
        
        # Bollinger Bands
        bollinger = ta.BBANDS(dataframe, timeperiod=20, nbdevup=2.0, nbdevdn=2.0)
        dataframe['%-bb_lowerband'] = bollinger['lowerband']
        dataframe['%-bb_middleband'] = bollinger['middleband']
        dataframe['%-bb_upperband'] = bollinger['upperband']

        macd = ta.MACD(dataframe)
        dataframe['%-macd'] = macd['macd']
        dataframe['%-macdsignal'] = macd['macdsignal']
        dataframe['%-macdhist'] = macd['macdhist']

        # --- 2. Sentiment Features (DB Integration) ---
        conn = self._get_sqlite_connection()
        if conn:
            pair = metadata['pair']
            base_coin = pair.split('/')[0] # e.g. BTC from BTC/USDT
            
            # Fetch Fear and Greed Index
            try:
                fng_df = pd.read_sql_query("SELECT value as fng_value FROM fear_and_greed ORDER BY timestamp DESC LIMIT 1", conn)
                current_fng = fng_df['fng_value'].iloc[0] if not fng_df.empty else 50
                dataframe['%-fng_index'] = current_fng
            except Exception as e:
                logger.error(f"F&G fetch failed: {e}")
                dataframe['%-fng_index'] = 50

            # Fetch Coin Rolling Sentiment
            try:
                sent_df = pd.read_sql_query(f"SELECT sentiment_1h, sentiment_4h, sentiment_24h FROM coin_sentiment_rolling WHERE coin = '{base_coin}' ORDER BY timestamp DESC LIMIT 1", conn)
                if not sent_df.empty:
                    dataframe['%-sentiment_1h'] = sent_df['sentiment_1h'].iloc[0]
                    dataframe['%-sentiment_4h'] = sent_df['sentiment_4h'].iloc[0]
                    dataframe['%-sentiment_24h'] = sent_df['sentiment_24h'].iloc[0]
                else:
                    dataframe['%-sentiment_1h'] = 0.0
                    dataframe['%-sentiment_4h'] = 0.0
                    dataframe['%-sentiment_24h'] = 0.0
            except Exception as e:
                logger.error(f"Sentiment fetch failed: {e}")
                dataframe['%-sentiment_1h'] = 0.0
                dataframe['%-sentiment_4h'] = 0.0
                dataframe['%-sentiment_24h'] = 0.0

            conn.close()
        else:
            # Fallback if DB is locked/missing
            dataframe['%-fng_index'] = 50
            dataframe['%-sentiment_1h'] = 0.0
            dataframe['%-sentiment_4h'] = 0.0
            dataframe['%-sentiment_24h'] = 0.0

        return dataframe

    def feature_engineering_expand_basic(self, dataframe: pd.DataFrame, metadata: dict, **kwargs) -> pd.DataFrame:
        """Add non-shiftable features for FreqAI"""
        dataframe['%-day_of_week'] = dataframe['date'].dt.dayofweek
        dataframe['%-hour_of_day'] = dataframe['date'].dt.hour
        return dataframe

    def feature_engineering_standard(self, dataframe: pd.DataFrame, metadata: dict, **kwargs) -> pd.DataFrame:
        """Standardizing features (already handled largely by FreqAI)"""
        return dataframe

    def set_freqai_targets(self, dataframe: pd.DataFrame, metadata: dict, **kwargs) -> pd.DataFrame:
        """
        Define what the model should predict.
        Here we predict the max high and min low in the next 10 candles.
        """
        dataframe['&s-up_or_down'] = np.where(dataframe['close'].shift(-10) > dataframe['close'], "up", "down")
        return dataframe

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe = self.freqai.start(dataframe, metadata, self)
        
        # Basic indicators for stoploss/sizing logic outside AI
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)
        return dataframe

    def populate_entry_trend(self, df: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        enter_long = np.zeros(len(df), dtype=int)
        
        # Check if FreqAI predictions are available
        if "do_predict" in df.columns:
            # Sizing not blocking: If AI predicts "up", we enter.
            enter_long = (df['do_predict'] == 1) & (df['&s-up_or_down'] == "up")
            
        df['enter_long'] = enter_long
        return df

    def populate_exit_trend(self, df: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        exit_long = np.zeros(len(df), dtype=int)
        
        if "do_predict" in df.columns:
            # Exit if AI changes its mind to down
            exit_long = (df['do_predict'] == 1) & (df['&s-up_or_down'] == "down")
            
        df['exit_long'] = exit_long
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

    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                            proposed_stake: float, min_stake: float, max_stake: float,
                            leverage: float, entry_tag: str, side: str,
                            **kwargs) -> float:
        """
        CORE PRINCIPLE: TRADE-FIRST AUTONOMY (Sizing not blocking).
        Instead of blocking a trade, we scale the size based on FreqAI confidence/market regime.
        """
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
                
        final_stake = proposed_stake * multiplier
        
        # Enforce bounds
        if final_stake < min_stake:
            return 0.0 # Here we actually block if it's too risky to meet min exchange limits
        return min(final_stake, max_stake)
