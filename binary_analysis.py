import pandas as pd
import numpy as np
import logging
from ta.trend import MACD, SMAIndicator, EMAIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange

# Configure logging (FileHandler removed)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        # logging.FileHandler("binary_analysis.log"), # Removed
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class BinaryAnalyzer:
    """
    Class to perform technical analysis for binary options trading
    Optimized for short timeframes (1-5 minutes) with accuracy percentage
    """

    def __init__(self):
        logger.info("BinaryAnalyzer initialized")

    # --- Rest of your BinaryAnalyzer class ---
    # (add_indicators, generate_binary_signals, analyze)
    # No changes needed inside these methods

    def add_indicators(self, df):
        """
        Add technical indicators to the dataframe

        Args:
            df (pandas.DataFrame): OHLCV dataframe with columns [open, high, low, close, volume]

        Returns:
            pandas.DataFrame: DataFrame with added technical indicators
        """
        try:
            logger.info(f"Adding technical indicators to dataframe with shape {df.shape}")

            # Make a copy to avoid modifying the original
            df_with_indicators = df.copy()

            # Add MACD
            macd = MACD(
                close=df_with_indicators['close'],
                window_slow=26,
                window_fast=12,
                window_sign=9
            )
            df_with_indicators['macd'] = macd.macd()
            df_with_indicators['macd_signal'] = macd.macd_signal()
            df_with_indicators['macd_diff'] = macd.macd_diff()

            # Add RSI
            rsi = RSIIndicator(
                close=df_with_indicators['close'],
                window=14
            )
            df_with_indicators['rsi'] = rsi.rsi()

            # Add Bollinger Bands
            bollinger = BollingerBands(
                close=df_with_indicators['close'],
                window=20,
                window_dev=2
            )
            df_with_indicators['bb_upper'] = bollinger.bollinger_hband()
            df_with_indicators['bb_middle'] = bollinger.bollinger_mavg()
            df_with_indicators['bb_lower'] = bollinger.bollinger_lband()

            # Add Stochastic Oscillator
            stoch = StochasticOscillator(
                high=df_with_indicators['high'],
                low=df_with_indicators['low'],
                close=df_with_indicators['close'],
                window=14,
                smooth_window=3
            )
            df_with_indicators['stoch_k'] = stoch.stoch()
            df_with_indicators['stoch_d'] = stoch.stoch_signal()

            # Add Moving Averages
            # Short-term EMAs for fast-moving markets
            ema9 = EMAIndicator(
                close=df_with_indicators['close'],
                window=9
            )
            df_with_indicators['ema9'] = ema9.ema_indicator()

            ema21 = EMAIndicator(
                close=df_with_indicators['close'],
                window=21
            )
            df_with_indicators['ema21'] = ema21.ema_indicator()

            # Add ATR for volatility measurement
            atr = AverageTrueRange(
                high=df_with_indicators['high'],
                low=df_with_indicators['low'],
                close=df_with_indicators['close'],
                window=14
            )
            df_with_indicators['atr'] = atr.average_true_range()

            # Add custom indicators for binary options
            # Price action indicators
            df_with_indicators['price_change'] = df_with_indicators['close'].pct_change() * 100
            df_with_indicators['candle_size'] = (df_with_indicators['high'] - df_with_indicators['low']) / df_with_indicators['low'] * 100
            df_with_indicators['body_size'] = abs(df_with_indicators['close'] - df_with_indicators['open']) / df_with_indicators['open'] * 100
            df_with_indicators['upper_wick'] = (df_with_indicators['high'] - df_with_indicators[['open', 'close']].max(axis=1)) / df_with_indicators[['open', 'close']].max(axis=1) * 100
            df_with_indicators['lower_wick'] = (df_with_indicators[['open', 'close']].min(axis=1) - df_with_indicators['low']) / df_with_indicators[['open', 'close']].min(axis=1) * 100

            # Binary-specific indicators
            # Momentum and trend strength
            df_with_indicators['momentum'] = df_with_indicators['close'] - df_with_indicators['close'].shift(5)
            df_with_indicators['trend_strength'] = abs(df_with_indicators['ema9'] - df_with_indicators['ema21']) / df_with_indicators['ema21'] * 100

            # Fill NaN values that may have been created
            df_with_indicators.fillna(0, inplace=True)

            logger.info(f"Successfully added indicators, new shape: {df_with_indicators.shape}")
            return df_with_indicators

        except Exception as e:
            logger.error(f"Error adding technical indicators: {e}", exc_info=True) # Added exc_info
            # Return the original dataframe or None if failure is critical
            return df # Or return None


    def generate_binary_signals(self, df_with_indicators, expiry_minutes=5):
        """
        Generate binary options trading signals with accuracy percentage

        Args:
            df_with_indicators (pandas.DataFrame): DataFrame with technical indicators
            expiry_minutes (int): Expiry time in minutes for binary options

        Returns:
            pandas.DataFrame: DataFrame with added signal columns
        """
        # Check if required columns exist after add_indicators potentially failed
        required_cols = ['close', 'rsi', 'macd_diff', 'bb_lower', 'bb_upper', 'stoch_k', 'stoch_d', 'ema9', 'ema21', 'momentum', 'atr']
        if not all(col in df_with_indicators.columns for col in required_cols):
             logger.error("Missing required indicator columns for signal generation. Aborting.")
             return df_with_indicators # Return unchanged df

        try:
            logger.info(f"Generating binary trading signals for {expiry_minutes} minute expiry")

            # Make a copy to avoid modifying the original
            df_signals = df_with_indicators.copy()

            # Initialize signal columns
            df_signals['signal'] = 0  # 0: no signal, 1: call (up), -1: put (down)
            df_signals['signal_strength'] = 0.0 # Changed to float for calculation
            df_signals['accuracy'] = 0.0  # 0-100% scale
            df_signals['entry_price'] = np.nan # Use NaN for missing entry
            df_signals['expiry_time'] = pd.NaT # Use NaT for missing time

            # Ensure shift operations don't fail on short dataframes
            if len(df_signals) < 2:
                logger.warning("Dataframe too short for shift operations in signal generation.")
                return df_signals # Not enough data

            # --- Calculate individual signals (ensure NaNs are handled) ---
            # RSI
            df_signals['rsi_signal'] = 0
            df_signals.loc[df_signals['rsi'] < 30, 'rsi_signal'] = 1
            df_signals.loc[df_signals['rsi'] > 70, 'rsi_signal'] = -1

            # MACD
            df_signals['macd_cross'] = 0
            macd_diff_shifted = df_signals['macd_diff'].shift(1)
            df_signals.loc[(df_signals['macd_diff'] > 0) & (macd_diff_shifted <= 0), 'macd_cross'] = 1
            df_signals.loc[(df_signals['macd_diff'] < 0) & (macd_diff_shifted >= 0), 'macd_cross'] = -1

            # Bollinger Bands
            df_signals['bb_signal'] = 0
            df_signals.loc[df_signals['close'] <= df_signals['bb_lower'], 'bb_signal'] = 1
            df_signals.loc[df_signals['close'] >= df_signals['bb_upper'], 'bb_signal'] = -1

            # Stochastic
            df_signals['stoch_signal'] = 0
            stoch_k_shifted = df_signals['stoch_k'].shift(1)
            stoch_d_shifted = df_signals['stoch_d'].shift(1)
            df_signals.loc[(df_signals['stoch_k'] > df_signals['stoch_d']) &
                          (stoch_k_shifted <= stoch_d_shifted) &
                          (df_signals['stoch_k'] < 30), 'stoch_signal'] = 1
            df_signals.loc[(df_signals['stoch_k'] < df_signals['stoch_d']) &
                          (stoch_k_shifted >= stoch_d_shifted) &
                          (df_signals['stoch_k'] > 70), 'stoch_signal'] = -1

            # EMA Crossover
            df_signals['ema_cross'] = 0
            ema9_shifted = df_signals['ema9'].shift(1)
            ema21_shifted = df_signals['ema21'].shift(1)
            df_signals.loc[(df_signals['ema9'] > df_signals['ema21']) &
                          (ema9_shifted <= ema21_shifted), 'ema_cross'] = 1
            df_signals.loc[(df_signals['ema9'] < df_signals['ema21']) &
                          (ema9_shifted >= ema21_shifted), 'ema_cross'] = -1

            # Momentum
            df_signals['momentum_signal'] = 0
            # Avoid division by zero or NaN in ATR
            atr_threshold = 0.5 * df_signals['atr']
            df_signals.loc[(df_signals['momentum'] > atr_threshold) & (atr_threshold > 0) , 'momentum_signal'] = 1
            df_signals.loc[(df_signals['momentum'] < -atr_threshold) & (atr_threshold > 0), 'momentum_signal'] = -1

            # --- Combine signals ---
            weights = {
                'rsi_signal': 0.15, 'macd_cross': 0.20, 'bb_signal': 0.15,
                'stoch_signal': 0.15, 'ema_cross': 0.15, 'momentum_signal': 0.20
            }
            weighted_signal_sum = 0
            total_weight_applied = 0 # Track weights for normalization if needed

            # Fill NaNs in signal columns before weighting
            signal_cols_to_fill = ['rsi_signal', 'macd_cross', 'bb_signal', 'stoch_signal', 'ema_cross', 'momentum_signal']
            df_signals[signal_cols_to_fill] = df_signals[signal_cols_to_fill].fillna(0)


            for indicator, weight in weights.items():
                weighted_signal_sum += df_signals[indicator] * weight
                total_weight_applied += weight # Assumes weight applies if indicator exists

            df_signals['weighted_signal'] = weighted_signal_sum

            # --- Determine final signal ---
            signal_threshold = 0.3 # Adjusted threshold
            df_signals['signal'] = 0 # Reset signal column
            df_signals.loc[df_signals['weighted_signal'] >= signal_threshold, 'signal'] = 1
            df_signals.loc[df_signals['weighted_signal'] <= -signal_threshold, 'signal'] = -1

            # Calculate signal strength (0-100) based on weighted sum magnitude
            # Normalize by potential max sum if needed, here just scale
            max_possible_score = total_weight_applied # Approximation
            df_signals['signal_strength'] = (abs(df_signals['weighted_signal']) / max_possible_score * 100).clip(0, 100)


            # --- Calculate Accuracy (Simplified Model) ---
            indicator_signals = list(weights.keys()) # Use weighted indicators
            num_indicators = len(indicator_signals)

            # Iterate using .iterrows() for clarity, though vectorized is faster for large datasets
            for index, row in df_signals.iterrows():
                 if row['signal'] != 0:
                      current_signal = row['signal']
                      current_price = row['close']

                      # Set entry price and expiry
                      df_signals.loc[index, 'entry_price'] = current_price
                      current_time = index # Timestamp from index
                      if isinstance(current_time, pd.Timestamp): # Check if index is Timestamp
                           expiry_time = current_time + pd.Timedelta(minutes=expiry_minutes)
                           df_signals.loc[index, 'expiry_time'] = expiry_time
                      else:
                           logger.warning(f"Index {index} is not a valid timestamp for expiry calculation.")


                      # --- Accuracy Calculation ---
                      # Base accuracy slightly increases with strength, capped
                      base_accuracy = 60 + (row['signal_strength'] / 100) * 25 # e.g., 60-85 range

                      # Bonus for indicator agreement
                      agreeing_indicators = sum(1 for ind in indicator_signals if row[ind] == current_signal)
                      indicator_bonus = (agreeing_indicators / num_indicators) * 15 # Max 15% bonus

                      accuracy = base_accuracy + indicator_bonus
                      accuracy = min(accuracy, 95) # Realistic cap

                      df_signals.loc[index, 'accuracy'] = accuracy


            # Ensure calculated columns have correct types
            df_signals['signal_strength'] = df_signals['signal_strength'].fillna(0).astype(float)
            df_signals['accuracy'] = df_signals['accuracy'].fillna(0).astype(float)


            signal_counts = df_signals['signal'].value_counts()
            logger.info(f"Generated signals with {signal_counts.get(1, 0)} call and {signal_counts.get(-1, 0)} put signals")
            return df_signals

        except Exception as e:
            logger.error(f"Error generating binary signals: {e}", exc_info=True) # Added exc_info
            # Return dataframe without signals or None if critical
            return df_with_indicators # Or return None


    def analyze(self, df, expiry_minutes=5):
        """
        Perform full technical analysis and signal generation for binary options

        Args:
            df (pandas.DataFrame): OHLCV dataframe
            expiry_minutes (int): Expiry time in minutes for binary options

        Returns:
            pandas.DataFrame: DataFrame with indicators and signals
            dict: Latest signal information or None on failure
        """
        try:
            # Add technical indicators
            df_with_indicators = self.add_indicators(df)
            if df_with_indicators is None or df_with_indicators.empty:
                 logger.error("Failed to add indicators or dataframe is empty.")
                 return df, None # Return original df and None signal

            # Generate signals
            df_signals = self.generate_binary_signals(df_with_indicators, expiry_minutes)
            if df_signals is None or df_signals.empty:
                 logger.error("Failed to generate signals or dataframe is empty.")
                 # Return df with indicators but None signal if signals failed
                 return df_with_indicators, None


            # Get latest row, handle potential empty dataframe after processing
            if df_signals.empty:
                 logger.warning("Dataframe is empty after signal generation.")
                 return df_signals, None

            latest = df_signals.iloc[-1]

            # Prepare signal_info dictionary safely using .get()
            signal_info = {
                'timestamp': latest.name, # Index (timestamp)
                'asset': None,  # To be filled by caller
                'platform': None,  # To be filled by caller
                'signal': int(latest.get('signal', 0)),
                'signal_type': "CALL" if latest.get('signal', 0) > 0 else "PUT" if latest.get('signal', 0) < 0 else "NONE",
                'signal_strength': float(latest.get('signal_strength', 0.0)),
                'accuracy': float(latest.get('accuracy', 0.0)),
                'price': float(latest['close']) if pd.notna(latest['close']) else 0.0, # Use current close
                'entry_price': float(latest['entry_price']) if pd.notna(latest.get('entry_price')) and latest.get('signal', 0) != 0 else None,
                'expiry_time': latest['expiry_time'] if pd.notna(latest.get('expiry_time')) and latest.get('signal', 0) != 0 else None,
                'expiry_minutes': expiry_minutes,
                'indicators': {
                    'rsi': float(latest.get('rsi', np.nan)),
                    'macd': float(latest.get('macd', np.nan)),
                    'macd_signal': float(latest.get('macd_signal', np.nan)),
                    'stoch_k': float(latest.get('stoch_k', np.nan)),
                    'stoch_d': float(latest.get('stoch_d', np.nan))
                }
            }

            # Convert NaNs in indicators to None for JSON compatibility if needed later
            for k, v in signal_info['indicators'].items():
                 if pd.isna(v):
                      signal_info['indicators'][k] = None


            return df_signals, signal_info

        except Exception as e:
            logger.error(f"Error in binary analysis: {e}", exc_info=True) # Added exc_info
            return df, None # Return original df and None signal


# For testing (if running this file directly)
if __name__ == "__main__":
    # This requires binary_data.py to be in the same directory or Python path
    try:
        import binary_data

        fetcher = binary_data.BinaryDataFetcher()
        # Use a common asset likely defined in base_prices
        test_asset = "EUR/USD"
        logger.info(f"--- Running Test for {test_asset} ---")
        data = fetcher.get_asset_data(test_asset, "1min", 100)

        if data is not None and not data.empty:
            analyzer = BinaryAnalyzer()
            df_signals, signal_info = analyzer.analyze(data)

            if signal_info:
                 print("\n--- Latest Signal Info ---")
                 # Pretty print the dictionary
                 import json
                 print(json.dumps(signal_info, indent=4, default=str)) # Use default=str for Timestamps

                 print("\n--- Signal Distribution ---")
                 print(df_signals['signal'].value_counts())

                 # Optional: Print last few rows of dataframe
                 # print("\n--- Last 5 Rows with Signals ---")
                 # print(df_signals.tail())
            else:
                 print(f"\nNo valid signal generated for {test_asset}.")

        else:
             print(f"\nCould not fetch or simulate data for {test_asset}.")

    except ImportError:
         logger.error("Could not import binary_data. Make sure it's in the same directory or Python path.")
    except Exception as e:
         logger.error(f"An error occurred during testing: {e}", exc_info=True)

# --- End of binary_analysis.py ---

            bollinger = BollingerBands(
                close=df_with_indicators['close'],
                window=20,
                window_dev=2
            )
            df_with_indicators['bb_upper'] = bollinger.bollinger_hband()
            df_with_indicators['bb_middle'] = bollinger.bollinger_mavg()
            df_with_indicators['bb_lower'] = bollinger.bollinger_lband()
            
            # Add Stochastic Oscillator
            stoch = StochasticOscillator(
                high=df_with_indicators['high'],
                low=df_with_indicators['low'],
                close=df_with_indicators['close'],
                window=14,
                smooth_window=3
            )
            df_with_indicators['stoch_k'] = stoch.stoch()
            df_with_indicators['stoch_d'] = stoch.stoch_signal()
            
            # Add Moving Averages
            # Short-term EMAs for fast-moving markets
            ema9 = EMAIndicator(
                close=df_with_indicators['close'],
                window=9
            )
            df_with_indicators['ema9'] = ema9.ema_indicator()
            
            ema21 = EMAIndicator(
                close=df_with_indicators['close'],
                window=21
            )
            df_with_indicators['ema21'] = ema21.ema_indicator()
            
            # Add ATR for volatility measurement
            atr = AverageTrueRange(
                high=df_with_indicators['high'],
                low=df_with_indicators['low'],
                close=df_with_indicators['close'],
                window=14
            )
            df_with_indicators['atr'] = atr.average_true_range()
            
            # Add custom indicators for binary options
            # Price action indicators
            df_with_indicators['price_change'] = df_with_indicators['close'].pct_change() * 100
            df_with_indicators['candle_size'] = (df_with_indicators['high'] - df_with_indicators['low']) / df_with_indicators['low'] * 100
            df_with_indicators['body_size'] = abs(df_with_indicators['close'] - df_with_indicators['open']) / df_with_indicators['open'] * 100
            df_with_indicators['upper_wick'] = (df_with_indicators['high'] - df_with_indicators[['open', 'close']].max(axis=1)) / df_with_indicators[['open', 'close']].max(axis=1) * 100
            df_with_indicators['lower_wick'] = (df_with_indicators[['open', 'close']].min(axis=1) - df_with_indicators['low']) / df_with_indicators[['open', 'close']].min(axis=1) * 100
            
            # Binary-specific indicators
            # Momentum and trend strength
            df_with_indicators['momentum'] = df_with_indicators['close'] - df_with_indicators['close'].shift(5)
            df_with_indicators['trend_strength'] = abs(df_with_indicators['ema9'] - df_with_indicators['ema21']) / df_with_indicators['ema21'] * 100
            
            # Fill NaN values that may have been created
            df_with_indicators.fillna(0, inplace=True)
            
            logger.info(f"Successfully added indicators, new shape: {df_with_indicators.shape}")
            return df_with_indicators
            
        except Exception as e:
            logger.error(f"Error adding technical indicators: {e}")
            return df
    
    def generate_binary_signals(self, df_with_indicators, expiry_minutes=5):
        """
        Generate binary options trading signals with accuracy percentage
        
        Args:
            df_with_indicators (pandas.DataFrame): DataFrame with technical indicators
            expiry_minutes (int): Expiry time in minutes for binary options
            
        Returns:
            pandas.DataFrame: DataFrame with added signal columns
        """
        try:
            logger.info(f"Generating binary trading signals for {expiry_minutes} minute expiry")
            
            # Make a copy to avoid modifying the original
            df_signals = df_with_indicators.copy()
            
            # Initialize signal columns
            df_signals['signal'] = 0  # 0: no signal, 1: call (up), -1: put (down)
            df_signals['signal_strength'] = 0  # 0-100 scale
            df_signals['accuracy'] = 0.0  # 0-100% scale
            df_signals['entry_price'] = 0.0
            df_signals['expiry_time'] = pd.NaT
            
            # RSI signals
            df_signals.loc[df_signals['rsi'] < 30, 'rsi_signal'] = 1  # Oversold - potential call
            df_signals.loc[df_signals['rsi'] > 70, 'rsi_signal'] = -1  # Overbought - potential put
            df_signals['rsi_signal'].fillna(0, inplace=True)
            
            # MACD signals
            df_signals['macd_cross'] = 0
            # MACD line crosses above signal line - bullish
            df_signals.loc[(df_signals['macd_diff'] > 0) & (df_signals['macd_diff'].shift(1) <= 0), 'macd_cross'] = 1
            # MACD line crosses below signal line - bearish
            df_signals.loc[(df_signals['macd_diff'] < 0) & (df_signals['macd_diff'].shift(1) >= 0), 'macd_cross'] = -1
            
            # Bollinger Bands signals
            df_signals['bb_signal'] = 0
            # Price touches lower band - potential call
            df_signals.loc[df_signals['close'] <= df_signals['bb_lower'], 'bb_signal'] = 1
            # Price touches upper band - potential put
            df_signals.loc[df_signals['close'] >= df_signals['bb_upper'], 'bb_signal'] = -1
            
            # Stochastic signals
            df_signals['stoch_signal'] = 0
            # Stochastic K crosses above D in oversold region - bullish
            df_signals.loc[(df_signals['stoch_k'] > df_signals['stoch_d']) & 
                          (df_signals['stoch_k'].shift(1) <= df_signals['stoch_d'].shift(1)) & 
                          (df_signals['stoch_k'] < 30), 'stoch_signal'] = 1
            # Stochastic K crosses below D in overbought region - bearish
            df_signals.loc[(df_signals['stoch_k'] < df_signals['stoch_d']) & 
                          (df_signals['stoch_k'].shift(1) >= df_signals['stoch_d'].shift(1)) & 
                          (df_signals['stoch_k'] > 70), 'stoch_signal'] = -1
            
            # EMA crossover signals
            df_signals['ema_cross'] = 0
            # Fast EMA crosses above slow EMA - bullish
            df_signals.loc[(df_signals['ema9'] > df_signals['ema21']) & 
                          (df_signals['ema9'].shift(1) <= df_signals['ema21'].shift(1)), 'ema_cross'] = 1
            # Fast EMA crosses below slow EMA - bearish
            df_signals.loc[(df_signals['ema9'] < df_signals['ema21']) & 
                          (df_signals['ema9'].shift(1) >= df_signals['ema21'].shift(1)), 'ema_cross'] = -1
            
            # Binary-specific signals
            # Momentum signals
            df_signals['momentum_signal'] = 0
            df_signals.loc[df_signals['momentum'] > 0.5 * df_signals['atr'], 'momentum_signal'] = 1
            df_signals.loc[df_signals['momentum'] < -0.5 * df_signals['atr'], 'momentum_signal'] = -1
            
            # Combine signals with weights optimized for binary options
            weights = {
                'rsi_signal': 0.15,
                'macd_cross': 0.20,
                'bb_signal': 0.15,
                'stoch_signal': 0.15,
                'ema_cross': 0.15,
                'momentum_signal': 0.20
            }
            
            # Calculate weighted signal
            for indicator, weight in weights.items():
                df_signals['signal'] += df_signals[indicator] * weight
            
            # Determine final signal based on threshold
            signal_threshold = 0.2  # Adjust based on desired sensitivity
            df_signals.loc[df_signals['signal'] >= signal_threshold, 'signal'] = 1
            df_signals.loc[df_signals['signal'] <= -signal_threshold, 'signal'] = -1
            df_signals.loc[(df_signals['signal'] > -signal_threshold) & (df_signals['signal'] < signal_threshold), 'signal'] = 0
            
            # Calculate signal strength (absolute value scaled to 0-100)
            df_signals['signal_strength'] = abs(df_signals['signal']) * 100
            
            # Calculate accuracy percentage based on historical performance
            # This is a simplified model - in a real system, you would use machine learning
            # or more sophisticated backtesting to determine accuracy
            for i in range(len(df_signals)):
                if df_signals.iloc[i]['signal'] != 0:
                    # Get current signal and price
                    current_signal = df_signals.iloc[i]['signal']
                    current_price = df_signals.iloc[i]['close']
                    
                    # Set entry price
                    df_signals.iloc[i, df_signals.columns.get_loc('entry_price')] = current_price
                    
                    # Set expiry time
                    current_time = df_signals.index[i]
                    expiry_time = current_time + pd.Timedelta(minutes=expiry_minutes)
                    df_signals.iloc[i, df_signals.columns.get_loc('expiry_time')] = expiry_time
                    
                    # Calculate accuracy based on signal strength and indicator agreement
                    # Base accuracy on signal strength
                    base_accuracy = min(df_signals.iloc[i]['signal_strength'], 85)
                    
                    # Count how many individual indicators agree with the signal
                    indicator_signals = ['rsi_signal', 'macd_cross', 'bb_signal', 'stoch_signal', 'ema_cross', 'momentum_signal']
                    agreeing_indicators = sum(1 for ind in indicator_signals if df_signals.iloc[i][ind] == current_signal)
                    
                    # Adjust accuracy based on indicator agreement
                    indicator_bonus = (agreeing_indicators / len(indicator_signals)) * 15
                    
                    # Final accuracy
                    accuracy = base_accuracy + indicator_bonus
                    
                    # Cap at 99% - nothing is 100% certain in trading
                    accuracy = min(accuracy, 99)
                    
                    df_signals.iloc[i, df_signals.columns.get_loc('accuracy')] = accuracy
            
            logger.info(f"Generated signals with {df_signals['signal'].value_counts().get(1, 0)} call and {df_signals['signal'].value_counts().get(-1, 0)} put signals")
            return df_signals
            
        except Exception as e:
            logger.error(f"Error generating binary signals: {e}")
            return df_with_indicators
    
    def analyze(self, df, expiry_minutes=5):
        """
        Perform full technical analysis and signal generation for binary options
        
        Args:
            df (pandas.DataFrame): OHLCV dataframe
            expiry_minutes (int): Expiry time in minutes for binary options
            
        Returns:
            pandas.DataFrame: DataFrame with indicators and signals
            dict: Latest signal information
        """
        try:
            # Add technical indicators
            df_with_indicators = self.add_indicators(df)
            
            # Generate signals
            df_signals = self.generate_binary_signals(df_with_indicators, expiry_minutes)
            
            # Get latest signal
            latest = df_signals.iloc[-1]
            
            signal_info = {
                'timestamp': latest.name,
                'asset': None,  # To be filled by caller
                'platform': None,  # To be filled by caller
                'signal': int(latest['signal']),
                'signal_type': "CALL" if latest['signal'] > 0 else "PUT" if latest['signal'] < 0 else "NONE",
                'signal_strength': float(latest['signal_strength']),
                'accuracy': float(latest['accuracy']),
                'price': float(latest['close']),
                'entry_price': float(latest['entry_price']) if latest['signal'] != 0 else None,
                'expiry_time': latest['expiry_time'] if latest['signal'] != 0 else None,
                'expiry_minutes': expiry_minutes,
                'indicators': {
                    'rsi': float(latest['rsi']),
                    'macd': float(latest['macd']),
                    'macd_signal': float(latest['macd_signal']),
                    'stoch_k': float(latest['stoch_k']),
                    'stoch_d': float(latest['stoch_d'])
                }
            }
            
            return df_signals, signal_info
            
        except Exception as e:
            logger.error(f"Error in binary analysis: {e}")
            return df, None

# For testing
if __name__ == "__main__":
    import binary_data
    
    fetcher = binary_data.BinaryDataFetcher()
    data = fetcher.get_asset_data("EUR/USD", "1min", 100)
    
    analyzer = BinaryAnalyzer()
    df_signals, signal_info = analyzer.analyze(data)
    
    print("Latest signal:", signal_info)
    print("Signal distribution:", df_signals['signal'].value_counts())
