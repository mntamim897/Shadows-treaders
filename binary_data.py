import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

# Configure logging (FileHandler removed)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        # logging.FileHandler("binary_data.log"), # Removed
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class BinaryDataFetcher:
    """
    Class to fetch and process data for binary trading signal generation
    Focused on short timeframes (1-5 minutes) for binary options trading
    """

    def __init__(self, api_key=None):
        self.api_key = api_key
        # Free alternative API endpoints that don't require keys
        self.alternative_api_url = "https://www.alphavantage.co/query" # Example, not currently used

        # Supported assets for binary trading
        self.forex_pairs = ["EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD", "USD/CAD", "NZD/USD", "USD/CHF"]
        self.crypto_pairs = ["BTC/USD", "ETH/USD", "LTC/USD", "XRP/USD"]
        self.commodities = ["GOLD", "SILVER", "OIL", "NATURAL_GAS"]
        self.indices = ["US500", "NASDAQ", "DOW_JONES", "DAX"]

        # All supported assets
        self.all_assets = self.forex_pairs + self.crypto_pairs + self.commodities + self.indices

        # Supported platforms
        self.platforms = ["Quotex", "IQ Option", "Pocket Option", "Binomo", "Olymp Trade"]

        logger.info("BinaryDataFetcher initialized")

    def get_asset_data(self, asset, interval='1min', limit=100, platform="Quotex"):
        """
        Fetch asset data from API (currently simulates data)

        Args:
            asset (str): Asset symbol (e.g., 'EUR/USD', 'GOLD', 'BTC/USD')
            interval (str): Timeframe interval (1min, 5min)
            limit (int): Number of candles to fetch
            platform (str): Trading platform

        Returns:
            pandas.DataFrame: OHLCV data or None on error
        """
        try:
            logger.info(f"Fetching/Simulating {asset} data for {interval} timeframe on {platform}")

            # --- SIMULATED DATA ---
            # Remove or comment out this line to use a real API call below
            return self._simulate_asset_data(asset, interval, limit, platform)

            # --- Example Real API Call (Placeholder) ---
            # if not self.api_key:
            #     logger.warning("API Key not provided. Cannot fetch real data.")
            #     return None
            # params = { ... params for your chosen API ... }
            # response = requests.get(YOUR_API_ENDPOINT, params=params)
            # response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
            # data = response.json()
            # df = self._parse_api_response(data, asset) # Need a parsing function
            # return df

        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            return None
        except Exception as e:
            logger.error(f"Error fetching asset data for {asset}: {e}", exc_info=True) # Added exc_info
            return None


    def _simulate_asset_data(self, asset, interval, limit, platform):
        """
        Simulate asset data for development and testing

        Args:
            asset (str): Asset symbol
            interval (str): Timeframe interval (e.g., '1min', '5min')
            limit (int): Number of candles to generate
            platform (str): Trading platform

        Returns:
            pandas.DataFrame: Simulated OHLCV data with DatetimeIndex
        """
        try:
             logger.info(f"Simulating {limit} candles of {asset} data for {interval} on {platform}")

             # Base prices (ensure keys match asset names exactly)
             base_prices = {
                  "EUR/USD": 1.10, "GBP/USD": 1.30, "USD/JPY": 110.0, "AUD/USD": 0.75, "USD/CAD": 1.25, "NZD/USD": 0.70, "USD/CHF": 0.90,
                  "BTC/USD": 65000.0, "ETH/USD": 3500.0, "LTC/USD": 80.0, "XRP/USD": 0.50,
                  "GOLD": 2000.0, "SILVER": 25.0, "OIL": 75.0, "NATURAL_GAS": 2.5,
                  "US500": 5000.0, "NASDAQ": 16000.0, "DOW_JONES": 38000.0, "DAX": 18000.0
             }
             base_price = base_prices.get(asset, 100.0) # Default if asset not in dict

             # Create timestamps ending now
             end_time = datetime.utcnow() # Use UTC for consistency
             try:
                  # Extract minutes from interval string
                  minutes = int(interval.replace('min', ''))
                  delta = timedelta(minutes=minutes)
             except ValueError:
                  logger.warning(f"Invalid interval format '{interval}'. Defaulting to 1 minute.")
                  delta = timedelta(minutes=1)

             # Generate timestamps ending at 'end_time'
             timestamps = [end_time - timedelta(minutes=i*minutes) for i in range(limit)]
             timestamps.reverse() # Oldest first

             # Volatility factors (adjust as needed)
             volatility_factors = {
                  "EUR/USD": 0.0002, "GBP/USD": 0.0003, "USD/JPY": 0.0003, "AUD/USD": 0.0003, "USD/CAD": 0.0002, "NZD/USD": 0.0003, "USD/CHF": 0.0002,
                  "BTC/USD": 0.005, "ETH/USD": 0.006, "LTC/USD": 0.007, "XRP/USD": 0.008,
                  "GOLD": 0.002, "SILVER": 0.003, "OIL": 0.004, "NATURAL_GAS": 0.005,
                  "US500": 0.001, "NASDAQ": 0.0015, "DOW_JONES": 0.001, "DAX": 0.0015
             }
             volatility = volatility_factors.get(asset, 0.003) # Default volatility

             # Platform volatility multiplier
             platform_volatility_multipliers = {
                  "Quotex": 1.0, "IQ Option": 1.1, "Pocket Option": 1.2, "Binomo": 1.15, "Olymp Trade": 1.05
             }
             volatility_multiplier = platform_volatility_multipliers.get(platform, 1.0)
             adjusted_volatility = volatility * volatility_multiplier

             # --- Generate Price Data ---
             np.random.seed(int(end_time.timestamp())) # Seed with time for different results each run
             price_changes = np.random.normal(0, adjusted_volatility, limit) * base_price # Scale changes by base price
             # Start from base_price and apply cumulative changes
             # Make the first price the base_price adjusted by the first random change
             sim_prices = base_price + np.cumsum(price_changes)


             # --- Generate OHLCV Data ---
             data = []
             last_close = base_price # Start with base price for first candle's open

             for i, timestamp in enumerate(timestamps):
                  current_base_price = sim_prices[i] # Use simulated price as base

                  # Generate realistic offsets based on volatility
                  high_offset = abs(np.random.normal(0, adjusted_volatility * 0.6)) * current_base_price
                  low_offset = abs(np.random.normal(0, adjusted_volatility * 0.6)) * current_base_price
                  open_offset = np.random.normal(0, adjusted_volatility * 0.3) * current_base_price

                  open_price = last_close + open_offset # Open relates to previous close
                  close_price = current_base_price # Close is the target simulated price

                  high = max(open_price, close_price) + high_offset
                  low = min(open_price, close_price) - low_offset

                  # Ensure OHLC validity (e.g., High >= Open, High >= Close, etc.)
                  high = max(high, open_price, close_price)
                  low = min(low, open_price, close_price)

                  # Simulate volume
                  volume = abs(np.random.normal(1000, 500))

                  data.append({
                       'timestamp': timestamp,
                       'open': round(open_price, 5),
                       'high': round(high, 5),
                       'low': round(low, 5),
                       'close': round(close_price, 5),
                       'volume': int(volume)
                  })
                  last_close = close_price # Update for next iteration

             # --- Create DataFrame ---
             if not data:
                  logger.warning("No simulated data generated.")
                  return pd.DataFrame() # Return empty DataFrame

             df = pd.DataFrame(data)
             df['timestamp'] = pd.to_datetime(df['timestamp'])
             df.set_index('timestamp', inplace=True)

             logger.info(f"Successfully simulated {asset} data with shape {df.shape}")
             return df

        except Exception as e:
            logger.error(f"Error during data simulation for {asset}: {e}", exc_info=True) # Added exc_info
            return None


    # --- Other methods (get_all_assets_data, get_supported_assets, get_supported_platforms) ---
    # No changes needed in these methods

    def get_all_assets_data(self, asset_type=None, interval='1min', limit=100, platform="Quotex"):
        """
        Fetch data for all supported assets or assets of a specific type

        Args:
            asset_type (str): Type of assets to fetch (forex, crypto, commodities, indices, or None for all)
            interval (str): Timeframe interval
            limit (int): Number of candles to fetch
            platform (str): Trading platform

        Returns:
            dict: Dictionary of DataFrames with asset symbols as keys
        """
        result = {}

        assets_to_fetch = self.get_supported_assets(asset_type) # Use helper method

        if not assets_to_fetch:
             logger.warning(f"No assets found for type: {asset_type}")
             return result

        logger.info(f"Attempting to fetch/simulate data for {len(assets_to_fetch)} assets...")
        fetched_count = 0
        for asset in assets_to_fetch:
            data = self.get_asset_data(asset, interval, limit, platform)
            if data is not None and not data.empty: # Check if data is valid
                result[asset] = data
                fetched_count += 1
            else:
                logger.warning(f"No data returned for asset: {asset}")

        logger.info(f"Successfully fetched/simulated data for {fetched_count} out of {len(assets_to_fetch)} assets.")
        return result


    def get_supported_assets(self, asset_type=None):
        """
        Get list of supported assets

        Args:
            asset_type (str): Type of assets to return (forex, crypto, commodities, indices, or None for all)

        Returns:
            list: List of supported asset symbols
        """
        if asset_type == "forex":
            return self.forex_pairs[:] # Return copy
        elif asset_type == "crypto":
            return self.crypto_pairs[:]
        elif asset_type == "commodities":
            return self.commodities[:]
        elif asset_type == "indices":
            return self.indices[:]
        elif asset_type is None: # Check for None explicitly
             return self.all_assets[:]
        else:
             logger.warning(f"Unsupported asset type requested: {asset_type}")
             return [] # Return empty list for unknown type


    def get_supported_platforms(self):
        """
        Get list of supported trading platforms

        Returns:
            list: List of supported platforms
        """
        return self.platforms[:] # Return copy


# For testing (if running this file directly)
if __name__ == "__main__":
    try:
        fetcher = BinaryDataFetcher()
        test_asset = "EUR/USD"
        logger.info(f"--- Running Test for {test_asset} ---")
        data = fetcher.get_asset_data(test_asset, "1min", 10)

        if data is not None and not data.empty:
            print(f"\n--- Simulated Data for {test_asset} (first 5 rows) ---")
            print(data.head())
        else:
            print(f"\nCould not fetch or simulate data for {test_asset}.")

        # Test getting supported assets
        # print("\n--- Supported Forex Assets ---")
        # print(fetcher.get_supported_assets("forex"))
        # print("\n--- All Supported Assets ---")
        # print(fetcher.get_supported_assets())

    except Exception as e:
         logger.error(f"An error occurred during testing: {e}", exc_info=True)

# --- End of binary_data.py ---
