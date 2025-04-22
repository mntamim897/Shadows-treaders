import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("binary_data.log"),
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
        self.alternative_api_url = "https://www.alphavantage.co/query"
        
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
        Fetch asset data from API
        
        Args:
            asset (str): Asset symbol (e.g., 'EUR/USD', 'GOLD', 'BTC/USD')
            interval (str): Timeframe interval (1min, 5min)
            limit (int): Number of candles to fetch
            platform (str): Trading platform
            
        Returns:
            pandas.DataFrame: OHLCV data
        """
        try:
            logger.info(f"Fetching {asset} data for {interval} timeframe on {platform}")
            
            # For demo purposes, we'll simulate data instead of making actual API calls
            # This is because free APIs have rate limits and may not provide 1-min data
            return self._simulate_asset_data(asset, interval, limit, platform)
            
        except Exception as e:
            logger.error(f"Error fetching asset data: {e}")
            return None
    
    def _simulate_asset_data(self, asset, interval, limit, platform):
        """
        Simulate asset data for development and testing
        
        Args:
            asset (str): Asset symbol
            interval (str): Timeframe interval
            limit (int): Number of candles to generate
            platform (str): Trading platform
            
        Returns:
            pandas.DataFrame: Simulated OHLCV data
        """
        logger.info(f"Simulating {limit} candles of {asset} data for {interval} on {platform}")
        
        # Base prices for different assets (approximate real values)
        base_prices = {
            # Forex
            "EUR/USD": 1.10,
            "GBP/USD": 1.30,
            "USD/JPY": 110.0,
            "AUD/USD": 0.75,
            "USD/CAD": 1.25,
            "NZD/USD": 0.70,
            "USD/CHF": 0.90,
            # Crypto
            "BTC/USD": 65000.0,
            "ETH/USD": 3500.0,
            "LTC/USD": 80.0,
            "XRP/USD": 0.50,
            # Commodities
            "GOLD": 2000.0,
            "SILVER": 25.0,
            "OIL": 75.0,
            "NATURAL_GAS": 2.5,
            # Indices
            "US500": 5000.0,
            "NASDAQ": 16000.0,
            "DOW_JONES": 38000.0,
            "DAX": 18000.0
        }
        
        # Get base price for the asset or use default
        base_price = base_prices.get(asset, 100.0)
        
        # Create timestamps
        end_time = datetime.now()
        if interval == '1min':
            delta = timedelta(minutes=1)
        elif interval == '5min':
            delta = timedelta(minutes=5)
        else:
            delta = timedelta(minutes=1)
        
        timestamps = [(end_time - i * delta).strftime('%Y-%m-%d %H:%M:%S') for i in range(limit)]
        timestamps.reverse()  # Oldest first
        
        # Generate price data with realistic volatility
        # Different assets have different volatility levels
        volatility_factors = {
            # Forex pairs typically have lower volatility
            "EUR/USD": 0.0002,
            "GBP/USD": 0.0003,
            "USD/JPY": 0.0003,
            "AUD/USD": 0.0003,
            "USD/CAD": 0.0002,
            "NZD/USD": 0.0003,
            "USD/CHF": 0.0002,
            # Crypto has higher volatility
            "BTC/USD": 0.005,
            "ETH/USD": 0.006,
            "LTC/USD": 0.007,
            "XRP/USD": 0.008,
            # Commodities
            "GOLD": 0.002,
            "SILVER": 0.003,
            "OIL": 0.004,
            "NATURAL_GAS": 0.005,
            # Indices
            "US500": 0.001,
            "NASDAQ": 0.0015,
            "DOW_JONES": 0.001,
            "DAX": 0.0015
        }
        
        # Get volatility factor for the asset or use default
        volatility = volatility_factors.get(asset, 0.003)
        
        # Adjust volatility based on platform (some platforms have more volatile prices)
        platform_volatility_multipliers = {
            "Quotex": 1.0,
            "IQ Option": 1.1,
            "Pocket Option": 1.2,
            "Binomo": 1.15,
            "Olymp Trade": 1.05
        }
        
        volatility_multiplier = platform_volatility_multipliers.get(platform, 1.0)
        adjusted_volatility = volatility * volatility_multiplier
        
        np.random.seed(42)  # For reproducibility
        
        # Simulate price movement
        price_changes = np.random.normal(0, adjusted_volatility, limit)
        prices = np.cumsum(price_changes) + base_price
        
        # Generate OHLC data
        data = []
        for i, timestamp in enumerate(timestamps):
            # Base price for this candle
            base = prices[i]
            
            # Generate realistic OHLC
            high_offset = abs(np.random.normal(0, adjusted_volatility * 1.5))
            low_offset = abs(np.random.normal(0, adjusted_volatility * 1.5))
            
            # Ensure high is highest and low is lowest
            high = base + high_offset
            low = base - low_offset
            
            # Open and close somewhere between high and low
            if i == 0:
                open_price = base
            else:
                open_price = prices[i-1]
            
            close = prices[i]
            
            # Ensure OHLC relationship is maintained
            high = max(high, open_price, close)
            low = min(low, open_price, close)
            
            # Volume (not directly used for signals but included for completeness)
            volume = abs(np.random.normal(1000, 300))
            
            data.append({
                'timestamp': timestamp,
                'open': round(open_price, 5),
                'high': round(high, 5),
                'low': round(low, 5),
                'close': round(close, 5),
                'volume': int(volume)
            })
        
        # Create DataFrame
        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        
        logger.info(f"Successfully simulated {asset} data with shape {df.shape}")
        return df
    
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
        
        assets_to_fetch = []
        if asset_type == "forex":
            assets_to_fetch = self.forex_pairs
        elif asset_type == "crypto":
            assets_to_fetch = self.crypto_pairs
        elif asset_type == "commodities":
            assets_to_fetch = self.commodities
        elif asset_type == "indices":
            assets_to_fetch = self.indices
        else:
            assets_to_fetch = self.all_assets
        
        for asset in assets_to_fetch:
            data = self.get_asset_data(asset, interval, limit, platform)
            if data is not None:
                result[asset] = data
        
        logger.info(f"Fetched data for {len(result)} assets")
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
            return self.forex_pairs
        elif asset_type == "crypto":
            return self.crypto_pairs
        elif asset_type == "commodities":
            return self.commodities
        elif asset_type == "indices":
            return self.indices
        else:
            return self.all_assets
    
    def get_supported_platforms(self):
        """
        Get list of supported trading platforms
        
        Returns:
            list: List of supported platforms
        """
        return self.platforms

# For testing
if __name__ == "__main__":
    fetcher = BinaryDataFetcher()
    data = fetcher.get_asset_data("EUR/USD", "1min", 10)
    print(data.head())
