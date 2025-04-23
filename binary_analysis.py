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

    def add_indicators(self, df):
        """
        Add technical indicators to the dataframe

        Args:
            df (pandas.DataFrame): OHLCV dataframe with columns [open, high, low, close, volume]

        Returns:
            pandas.DataFrame: DataFrame with added technical indicators or original df on error
        """
        try:
            # Check if input is a DataFrame and not empty
            if not isinstance(df, pd.DataFrame) or df.empty:
                logger.warning("Input is not a valid DataFrame or is empty. Cannot add indicators.")
                return df # Return original/empty df

            # Check for required columns
            required_cols = ['open', 'high', 'low', 'close']
            if not all(col in df.columns for col in required_cols):
                 logger.error(f"Missing one or more required columns: {required_cols}")
                 return df # Return original df

            logger.info(f"Adding technical indicators to dataframe with shape {df.shape}")

            # Make a copy to avoid modifying the original
            df_with_indicators = df.copy()

            # --- Calculate Indicators (add checks for sufficient data length) ---
            min_data_length = 26 # Longest period used by MACD

            if len(df_with_indicators) < min_data_length:
                 logger.warning(f"Data length ({len(df_with_indicators)}) is less than minimum required ({min_data_length}). Some indicators might be NaN.")
                 # Decide whether to proceed or return

            # MACD
            macd = MACD(
                close=df_with_indicators['close'], window_slow=26, window_fast=12, window_sign=9, fillna=True # fillna=True in ta lib handles short data
            )
            df_with_indicators['macd'] = macd.macd()
            df_with_indicators['macd_signal'] = macd.macd_signal()
            df_with_indicators['macd_diff'] = macd.macd_diff()

            # RSI
            rsi = RSIIndicator(close=df_with_indicators['close'], window=14, fillna=True)
            df_with_indicators['rsi'] = rsi.rsi()

            # Bollinger Bands
            bollinger = BollingerBands(close=df_with_indicators['close'], window=20, window_dev=2, fillna=True)
            df_with_indicators['bb_upper'] = bollinger.bollinger_hband()
            df_with_indicators['bb_middle'] = bollinger.bollinger_mavg()
            df_with_indicators['bb_lower'] = bollinger.bollinger_lband()

            # Stochastic Oscillator
            stoch = StochasticOscillator(
                high=df_with_indicators['high'], low=df_with_indicators['low'], close=df_with_indicators['close'],
                window=14, smooth_window=3, fillna=True
            )
            df_with_indicators['stoch_k'] = stoch.stoch()
            df_with_indicators['stoch_d'] = stoch.stoch_signal()

            # EMAs
            ema9 = EMAIndicator(close=df_with_indicators['close'], window=9, fillna=True)
            df_with_indicators['ema9'] = ema9.ema_indicator()

            ema21 = EMAIndicator(close=df_with_indicators['close'], window=21, fillna=True)
            df_with_indicators['ema21'] = ema21.ema_indicator()

            # ATR
            atr = AverageTrueRange(
                high=df_with_indicators['high'], low=df_with_indicators['low'], close=df_with_indicators['close'],
                window=14, fillna=True
            )
            df_with_indicators['atr'] = atr.average_true_range()

            # --- Custom Price Action Indicators ---
            # Ensure calculations handle potential division by zero if open/close is zero
            df_with_indicators['price_change'] = df_with_indicators['close'].pct_change().fillna(0) * 100

            # Calculate candle/body/wick sizes safely
            low_gt_zero = df_with_indicators['low'] > 0
            df_with_indicators['candle_size'] = np.where(low_gt_zero, (df_with_indicators['high'] - df_with_indicators['low']) / df_with_indicators['low'] * 100, 0)
            df_with_indicators['candle_size'].fillna(0, inplace=True)


            open_gt_zero = df_with_indicators['open'] > 0
            df_with_indicators['body_size'] = np.where(open_gt_zero, abs(df_with_indicators['close'] - df_with_indicators['open']) / df_with_indicators['open'] * 100, 0)
            df_with_indicators['body_size'].fillna(0, inplace=True)


            max_oc = df_with_indicators[['open', 'close']].max(axis=1)
            max_oc_gt_zero = max_oc > 0
            df_with_indicators['upper_wick'] = np.where(max_oc_gt_zero, (df_with_indicators['high'] - max_oc) / max_oc * 100, 0)
            df_with_indicators['upper_wick'].fillna(0, inplace=True)


            min_oc = df_with_indicators[['open', 'close']].min(axis=1)
            min_oc_gt_zero = min_oc > 0 # Denominator needs to be > 0
            df_with_indicators['lower_wick'] = np.where(min_oc_gt_zero, (min_oc - df_with_indicators['low']) / min_oc * 100, 0)
            df_with_indicators['lower_wick'].fillna(0, inplace=True)


            # --- Binary Specific ---
            # Ensure shift doesn't go out of bounds
            shift_period = 5
            if len(df_with_indicators) > shift_period:
                 df_with_indicators['momentum'] = df_with_indicators['close'] - df_with_indicators['close'].shift(shift_period)
            else:
                 df_with_indicators['momentum'] = 0 # Or NaN
            df_with_indicators['momentum'].fillna(0, inplace=True)


            ema21_gt_zero = df_with_indicators['ema21'] > 0
            df_with_indicators['trend_strength'] = np.where(ema21_gt_zero, abs(df_with_indicators['ema9'] - df_with_indicators['ema21']) / df_with_indicators['ema21'] * 100, 0)
            df_with_indicators['trend_strength'].fillna(0, inplace=True)


            # Final fillna just in case, although ta lib with fillna=True helps
            # df_with_indicators.fillna(0, inplace=True) # Reconsider blanket fillna(0)

            logger.info(f"Successfully added indicators, new shape: {df_with_indicators.shape}")
            return df_with_indicators

        except Exception as e:
            logger.error(f"Error adding technical indicators: {e}", exc_info=True)
            return df # Return original dataframe on error


    def generate_binary_signals(self, df_with_indicators, expiry_minutes=5):
        """
        Generate binary options trading signals with accuracy percentage

        Args:
            df_with_indicators (pandas.DataFrame): DataFrame with technical indicators
            expiry_minutes (int): Expiry time in minutes for binary options

        Returns:
            pandas.DataFrame: DataFrame with added signal columns or original on error
        """
        try:
            # Check if input is valid and has required columns from add_indicators
            required_cols = ['close', 'rsi', 'macd_diff', 'bb_lower', 'bb_upper', 'stoch_k', 'stoch_d', 'ema9', 'ema21', 'momentum', 'atr']
            if not isinstance(df_with_indicators, pd.DataFrame) or not all(col in df_with_indicators.columns for col in required_cols):
                 logger.error("Missing required indicator columns or invalid input for signal generation.")
                 return df_with_indicators # Return input df

            logger.info(f"Generating binary trading signals for {expiry_minutes} minute expiry")

            # Make a copy
            df_signals = df_with_indicators.copy()

            # Initialize signal columns
            df_signals['signal'] = 0
            df_signals['weighted_signal'] = 0.0 # Store the raw weighted score
            df_signals['signal_strength'] = 0.0
            df_signals['accuracy'] = 0.0
            df_signals['entry_price'] = np.nan
            df_signals['expiry_time'] = pd.NaT

            # Need at least 2 rows for shift(1) operations
            if len(df_signals) < 2:
                logger.warning("Dataframe too short for signal generation (need >= 2 rows).")
                return df_signals

            # --- Calculate individual signals ---
            # (Using .loc for assignment is generally safer)
            df_signals['rsi_signal'] = 0
            df_signals.loc[df_signals['rsi'] < 30, 'rsi_signal'] = 1
            df_signals.loc[df_signals['rsi'] > 70, 'rsi_signal'] = -1

            df_signals['macd_cross'] = 0
            macd_diff_shifted = df_signals['macd_diff'].shift(1)
            df_signals.loc[(df_signals['macd_diff'] > 0) & (macd_diff_shifted <= 0), 'macd_cross'] = 1
            df_signals.loc[(df_signals['macd_diff'] < 0) & (macd_diff_shifted >= 0), 'macd_cross'] = -1

            df_signals['bb_signal'] = 0
            df_signals.loc[df_signals['close'] <= df_signals['bb_lower'], 'bb_signal'] = 1
            df_signals.loc[df_signals['close'] >= df_signals['bb_upper'], 'bb_signal'] = -1

            df_signals['stoch_signal'] = 0
            stoch_k_shifted = df_signals['stoch_k'].shift(1)
            stoch_d_shifted = df_signals['stoch_d'].shift(1)
            # Ensure shifted values are not NaN before comparison
            valid_stoch_shift = pd.notna(stoch_k_shifted) & pd.notna(stoch_d_shifted)
            df_signals.loc[valid_stoch_shift & (df_signals['stoch_k'] > df_signals['stoch_d']) & (stoch_k_shifted <= stoch_d_shifted) & (df_signals['stoch_k'] < 30), 'stoch_signal'] = 1
            df_signals.loc[valid_stoch_shift & (df_signals['stoch_k'] < df_signals['stoch_d']) & (stoch_k_shifted >= stoch_d_shifted) & (df_signals['stoch_k'] > 70), 'stoch_signal'] = -1

            df_signals['ema_cross'] = 0
            ema9_shifted = df_signals['ema9'].shift(1)
            ema21_shifted = df_signals['ema21'].shift(1)
            valid_ema_shift = pd.notna(ema9_shifted) & pd.notna(ema21_shifted)
            df_signals.loc[valid_ema_shift & (df_signals['ema9'] > df_signals['ema21']) & (ema9_shifted <= ema21_shifted), 'ema_cross'] = 1
            df_signals.loc[valid_ema_shift & (df_signals['ema9'] < df_signals['ema21']) & (ema9_shifted >= ema21_shifted), 'ema_cross'] = -1

            df_signals['momentum_signal'] = 0
            atr_threshold = 0.5 * df_signals['atr']
            # Ensure atr_threshold is not NaN and positive before using it
            valid_atr = pd.notna(atr_threshold) & (atr_threshold > 0)
            df_signals.loc[valid_atr & (df_signals['momentum'] > atr_threshold), 'momentum_signal'] = 1
            df_signals.loc[valid_atr & (df_signals['momentum'] < -atr_threshold), 'momentum_signal'] = -1

            # --- Combine signals ---
            weights = {
                'rsi_signal': 0.15, 'macd_cross': 0.20, 'bb_signal': 0.15,
                'stoch_signal': 0.15, 'ema_cross': 0.15, 'momentum_signal': 0.20
            }
            indicator_signals = list(weights.keys())

            # Fill NaNs in individual signal columns before weighting
            df_signals[indicator_signals] = df_signals[indicator_signals].fillna(0)

            # Calculate weighted sum
            df_signals['weighted_signal'] = sum(df_signals[indicator] * weight for indicator, weight in weights.items())

            # --- Determine final signal ---
            signal_threshold = 0.3 # Threshold for weighted signal to trigger
            df_signals['signal'] = 0 # Reset final signal
            df_signals.loc[df_signals['weighted_signal'] >= signal_threshold, 'signal'] = 1
            df_signals.loc[df_signals['weighted_signal'] <= -signal_threshold, 'signal'] = -1

            # Calculate signal strength (0-100)
            # Normalize absolute weighted signal by the sum of weights
            total_weight = sum(weights.values())
            df_signals['signal_strength'] = (abs(df_signals['weighted_signal']) / total_weight * 100).clip(0, 100)
            df_signals.loc[df_signals['signal'] == 0, 'signal_strength'] = 0 # Strength is 0 if no signal

            # --- Calculate Accuracy & Details (Simplified Model) ---
            num_indicators = len(indicator_signals)

            # Vectorized approach for calculation where possible
            signal_active = df_signals['signal'] != 0
            df_signals.loc[signal_active, 'entry_price'] = df_signals.loc[signal_active, 'close']

            # Calculate expiry time only for active signals and valid timestamps
            if isinstance(df_signals.index, pd.DatetimeIndex):
                 df_signals.loc[signal_active, 'expiry_time'] = df_signals.index[signal_active] + pd.Timedelta(minutes=expiry_minutes)
            else:
                 logger.warning("DataFrame index is not DatetimeIndex, cannot calculate expiry time.")
                 df_signals['expiry_time'] = pd.NaT # Ensure it's NaT

            # Calculate accuracy components
            base_accuracy = 60 + (df_signals['signal_strength'] / 100) * 25 # Base: 60-85

            agreeing_indicators = df_signals[indicator_signals].apply(lambda row: sum(1 for ind in indicator_signals if row[ind] == row['signal']), axis=1)
            indicator_bonus = (agreeing_indicators / num_indicators) * 15 # Max 15% bonus

            calculated_accuracy = (base_accuracy + indicator_bonus).clip(0, 95) # Cap at 95%

            # Apply accuracy only where signal is active
            df_signals.loc[signal_active, 'accuracy'] = calculated_accuracy
            df_signals['accuracy'] = df_signals['accuracy'].fillna(0).astype(float) # Fill NaNs and set type


            signal_counts = df_signals['signal'].value_counts()
            logger.info(f"Generated signals with {signal_counts.get(1, 0)} CALL and {signal_counts.get(-1, 0)} PUT signals")
            return df_signals

        except Exception as e:
            logger.error(f"Error generating binary signals: {e}", exc_info=True)
            return df_with_indicators # Return input df on error


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
            # Ensure df is valid before proceeding
            if not isinstance(df, pd.DataFrame) or df.empty:
                 logger.error("Input DataFrame is invalid or empty for analysis.")
                 return df, None

            # Add technical indicators
            df_with_indicators = self.add_indicators(df)
            # Check if indicators were added successfully
            if df_with_indicators is None or df_with_indicators.empty:
                 logger.error("Failed to add indicators.")
                 return df, None

            # Generate signals
            df_signals = self.generate_binary_signals(df_with_indicators, expiry_minutes)
            # Check if signals were generated successfully
            if df_signals is None or df_signals.empty:
                 logger.error("Failed to generate signals.")
                 # Return df with indicators if signals failed
                 return df_with_indicators, None

            # Get latest row safely
            latest = df_signals.iloc[-1]

            # Prepare signal_info dictionary using .get() for safety
            signal_info = {
                'timestamp': latest.name,
                'asset': None, # Filled by caller
                'platform': None, # Filled by caller
                'signal': int(latest.get('signal', 0)),
                'signal_type': "CALL" if latest.get('signal', 0) > 0 else "PUT" if latest.get('signal', 0) < 0 else "NONE",
                'signal_strength': float(latest.get('signal_strength', 0.0)),
                'accuracy': float(latest.get('accuracy', 0.0)),
                'price': float(latest['close']) if pd.notna(latest['close']) else np.nan,
                'entry_price': float(latest['entry_price']) if pd.notna(latest.get('entry_price')) else None,
                'expiry_time': latest['expiry_time'] if pd.notna(latest.get('expiry_time')) else None,
                'expiry_minutes': expiry_minutes,
                'indicators': {
                    # Use .get() with default np.nan for indicator values
                    'rsi': float(latest.get('rsi', np.nan)),
                    'macd': float(latest.get('macd', np.nan)),
                    'macd_signal': float(latest.get('macd_signal', np.nan)),
                    'stoch_k': float(latest.get('stoch_k', np.nan)),
                    'stoch_d': float(latest.get('stoch_d', np.nan))
                }
            }

            # Convert NaNs in indicators dict to None for cleaner output/JSON
            for k, v in signal_info['indicators'].items():
                 if pd.isna(v):
                      signal_info['indicators'][k] = None

            # Only return signal info if a signal was actually generated
            if signal_info['signal'] == 0:
                 logger.info("No trading signal generated for the latest data point.")
                 # Optionally return None instead of info for "NONE" signal
                 # return df_signals, None
                 # Or return the info with signal_type="NONE" as currently implemented

            return df_signals, signal_info

        except Exception as e:
            logger.error(f"Error in binary analysis: {e}", exc_info=True)
            # Return original df if analysis fails early, None signal
            return df, None


# For testing (if running this file directly)
if __name__ == "__main__":
    try:
        import binary_data # Assumes binary_data.py is available

        fetcher = binary_data.BinaryDataFetcher()
        test_asset = "EUR/USD"
        logger.info(f"--- Running Test for {test_asset} ---")
        data = fetcher.get_asset_data(test_asset, "1min", 100) # Fetch 100 candles

        if data is not None and not data.empty:
            analyzer = BinaryAnalyzer()
            df_signals, signal_info = analyzer.analyze(data, expiry_minutes=1) # Test with 1 min expiry

            if signal_info:
                 print("\n--- Latest Signal Info ---")
                 import json
                 # Handle Timestamps and NaT for JSON serialization
                 print(json.dumps(signal_info, indent=4, default=lambda x: str(x) if pd.notna(x) else None))

                 print("\n--- Signal Distribution ---")
                 print(df_signals['signal'].value_counts())

            else:
                 print(f"\nNo valid signal generated for {test_asset}.")
                 # print("\n--- Last 5 Rows of Processed Data ---")
                 # print(df_signals.tail()) # Print tail even if no signal
        else:
             print(f"\nCould not fetch or simulate data for {test_asset}.")

    except ImportError:
         logger.error("Could not import binary_data. Make sure it's in the same directory or Python path.")
    except Exception as e:
         logger.error(f"An error occurred during testing: {e}", exc_info=True)

# --- End of binary_analysis.py ---
