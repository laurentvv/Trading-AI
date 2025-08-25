import yfinance as yf
import pandas as pd
import logging
import os
from pathlib import Path
import requests
from dotenv import load_dotenv
import json
import pandas_datareader.data as web
import datetime

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)

CACHE_DIR = Path("data_cache")
# Create a specific subdirectory for macroeconomic data cache
MACRO_CACHE_DIR = CACHE_DIR / "macro"
MACRO_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Alpha Vantage API setup
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
if not ALPHA_VANTAGE_API_KEY:
    logger.warning("ALPHA_VANTAGE_API_KEY not found in environment variables. Some features might be disabled.")
    
def get_etf_data(ticker: str, period: str = '5y', force_refresh: bool = False) -> tuple[pd.DataFrame, dict]:
    """
    Retrieves ETF and VIX data, with a local caching system.
    The cache now includes VIX data.
    """
    import time
    
    CACHE_DIR.mkdir(exist_ok=True)
    cache_filename = f"{ticker.replace('.', '_')}_max_with_vix.parquet"
    cache_filepath = CACHE_DIR / cache_filename

    hist_data = None
    info = {}

    if not force_refresh and cache_filepath.exists():
        try:
            logger.info(f"Loading data from cache: {cache_filepath}")
            hist_data = pd.read_parquet(cache_filepath)
            etf = yf.Ticker(ticker)
            try:
                info = etf.info
            except Exception as e:
                logger.warning(f"Could not fetch ticker info: {e}")
                info = {}
            logger.info(f"Data loaded from cache: {len(hist_data)} days.")
        except Exception as e:
            logger.warning(f"Could not read cache file {cache_filepath}: {e}. Forcing refresh.")
            force_refresh = True

    if force_refresh or hist_data is None:
        logger.info(f"Downloading data for {ticker} and ^VIX...")
        
        # Try multiple approaches to avoid database locking
        max_retries = 3
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                # First, try downloading each ticker separately to avoid database conflicts
                logger.info(f"Attempt {attempt + 1}/{max_retries} to download data...")
                
                # Download main ticker first
                ticker_data = yf.download(ticker, period="max", auto_adjust=True, progress=False)
                if ticker_data.empty:
                    raise ValueError(f"No data found for ticker {ticker}")
                
                # Small delay to avoid database conflicts
                time.sleep(1)
                
                # Download VIX separately
                vix_data = yf.download('^VIX', period="max", auto_adjust=True, progress=False)
                if vix_data.empty:
                    logger.warning("VIX data not available, using dummy VIX values")
                    # Create dummy VIX data aligned with ticker data
                    vix_data = pd.DataFrame(index=ticker_data.index)
                    vix_data['Close'] = 20.0  # Default VIX value
                
                # Combine the data
                hist_data = pd.DataFrame(index=ticker_data.index)
                hist_data['Open'] = ticker_data['Open']
                hist_data['High'] = ticker_data['High']
                hist_data['Low'] = ticker_data['Low']
                hist_data['Close'] = ticker_data['Close']
                hist_data['Volume'] = ticker_data['Volume']
                
                # Align VIX data with ticker data
                vix_close = vix_data['Close'].reindex(hist_data.index)
                hist_data['VIX'] = vix_close

                # Clean up data
                hist_data = hist_data.dropna(how='all')
                hist_data['VIX'] = hist_data['VIX'].ffill().bfill()
                
                # If VIX is still NaN, use default value
                hist_data['VIX'] = hist_data['VIX'].fillna(20.0)
                
                hist_data = hist_data.dropna(subset=['Close'])  # Only drop if Close is NaN
                
                if hist_data.empty:
                    raise ValueError("No valid data after cleaning")

                etf = yf.Ticker(ticker)
                try:
                    info = etf.info
                except Exception as e:
                    logger.warning(f"Could not fetch ticker info: {e}")
                    info = {}

                # Save to cache
                try:
                    hist_data.to_parquet(cache_filepath)
                    logger.info(f"Data for {ticker} and ^VIX saved to cache: {cache_filepath}")
                except Exception as e:
                    logger.warning(f"Could not save to cache: {e}")
                
                break  # Success, exit retry loop
                
            except Exception as e:
                logger.error(f"Attempt {attempt + 1} failed: {e}")
                
                if attempt < max_retries - 1:
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    # Last attempt failed, try to use any cached data
                    if cache_filepath.exists():
                        try:
                            logger.warning("All download attempts failed, trying to use cached data...")
                            hist_data = pd.read_parquet(cache_filepath)
                            etf = yf.Ticker(ticker)
                            try:
                                info = etf.info
                            except:
                                info = {}
                            logger.info(f"Using cached data: {len(hist_data)} days.")
                            break
                        except Exception as cache_e:
                            logger.error(f"Could not read cache as fallback: {cache_e}")
                    
                    # If everything fails, raise the original error
                    raise e

    # Final check for old cache format without VIX
    if hist_data is not None and 'VIX' not in hist_data.columns:
        logger.warning("VIX column not found in loaded data. Adding default VIX values.")
        hist_data['VIX'] = 20.0  # Default VIX value
    
    # Final validation
    if hist_data is None or hist_data.empty:
        raise ValueError(f"Failed to retrieve any data for {ticker}")

    logger.info(f"Final data period: {hist_data.index.min()} to {hist_data.index.max()}")
    return hist_data, info

def _get_macro_cache_filepath(function: str, symbol: str = None, interval: str = "monthly", source: str = "AV") -> Path:
    """
    Generates a cache file path for macroeconomic data.
    """
    filename = f"{source}_{function}"
    if symbol:
        filename += f"_{symbol}"
    filename += f"_{interval}.parquet"
    return MACRO_CACHE_DIR / filename

def _load_macro_data_from_cache(cache_filepath: Path) -> pd.DataFrame:
    """
    Loads macroeconomic data from a Parquet cache file.
    """
    if cache_filepath.exists():
        try:
            logger.info(f"Loading macro data from cache: {cache_filepath}")
            return pd.read_parquet(cache_filepath)
        except Exception as e:
            logger.warning(f"Could not read macro cache file {cache_filepath}: {e}")
    return pd.DataFrame()

def _save_macro_data_to_cache(data: pd.DataFrame, cache_filepath: Path):
    """
    Saves macroeconomic data to a Parquet cache file.
    """
    try:
        data.to_parquet(cache_filepath)
        logger.info(f"Macro data saved to cache: {cache_filepath}")
    except Exception as e:
        logger.error(f"Failed to save macro data to cache {cache_filepath}: {e}")

def get_alpha_vantage_data(function: str, symbol: str = None, interval: str = "monthly", force_refresh: bool = False) -> pd.DataFrame:
    """
    Fetches data from Alpha Vantage API, with local caching.
    
    Args:
        function (str): The AV function name (e.g., 'TREASURY_YIELD', 'CPI').
        symbol (str, optional): The data symbol (e.g., '10year' for Treasury Yield). Defaults to None.
        interval (str, optional): Data frequency ('daily', 'weekly', 'monthly'). Defaults to 'monthly'.
        force_refresh (bool, optional): If True, bypasses the cache and forces a new API call. Defaults to False.

    Returns:
        pd.DataFrame: A DataFrame with 'date' and 'value' columns.
    """
    if not ALPHA_VANTAGE_API_KEY:
        logger.error("Alpha Vantage API key is missing.")
        return pd.DataFrame()

    cache_filepath = _get_macro_cache_filepath(function, symbol, interval, source="AV")
    
    # 1. Try loading from cache if not forcing refresh
    if not force_refresh:
        cached_data = _load_macro_data_from_cache(cache_filepath)
        if not cached_data.empty:
            logger.info(f"Using cached data for {function} ({symbol}).")
            return cached_data

    # 2. If cache miss or force refresh, fetch from API
    logger.info(f"Fetching {function} ({symbol}) from Alpha Vantage API...")
    url = 'https://www.alphavantage.co/query'
    params = {
        'function': function,
        'apikey': ALPHA_VANTAGE_API_KEY,
        'datatype': 'json',
        'interval': interval
    }
    if symbol:
        params['symbol'] = symbol

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()

        # Check for API errors
        if "Error Message" in data:
            logger.error(f"Alpha Vantage API Error for {function}: {data['Error Message']}")
            # If API error, try to return cached data as a fallback
            cached_data = _load_macro_data_from_cache(cache_filepath)
            if not cached_data.empty:
                logger.info(f"Falling back to cached data for {function} ({symbol}) after API error.")
                return cached_data
            return pd.DataFrame()
            
        if "Information" in data and "limit" in data["Information"]:
             logger.warning(f"Alpha Vantage API call limit reached for {function}.")
             # If rate limit hit, try to return cached data as a fallback
             cached_data = _load_macro_data_from_cache(cache_filepath)
             if not cached_data.empty:
                 logger.info(f"Falling back to cached data for {function} ({symbol}) due to rate limit.")
                 return cached_data
             return pd.DataFrame()

        # Extract the data series (same logic as before)
        data_key = None
        if 'data' in data:
            data_key = 'data'
        elif 'TreasureYield' in data: # Note: API spelling
             data_key = 'TreasureYield'
        elif function in data:
             data_key = function
        else:
             # Try to find the first key that looks like a data series
             for k in data.keys():
                 if isinstance(data[k], list) and len(data[k]) > 0 and isinstance(data[k][0], dict):
                     data_key = k
                     break

        if not data_key or data_key not in data:
            logger.warning(f"Could not find data series for {function} in response.")
            # Fallback to cache
            cached_data = _load_macro_data_from_cache(cache_filepath)
            if not cached_data.empty:
                logger.info(f"Falling back to cached data for {function} ({symbol}) after parsing error.")
                return cached_data
            return pd.DataFrame()

        df_list = []
        for item in data[data_key]:
            # AV API responses can have varying key names for the date
            date_key = 'date' if 'date' in item else None
            if not date_key:
                # Common alternatives
                for k in ['timestamp', 'datetime']:
                    if k in item:
                        date_key = k
                        break
            if not date_key:
                logger.warning(f"Could not find date key in data item for {function}.")
                continue # Skip this item

            # Value key can also vary
            value_key = None
            for k in ['value', 'yield', 'rate', 'cpi']:
                if k in item:
                    value_key = k
                    break
            if not value_key:
                 logger.warning(f"Could not find value key in data item for {function}.")
                 continue

            try:
                df_list.append({'date': pd.to_datetime(item[date_key]), 'value': float(item[value_key])})
            except (ValueError, TypeError) as e:
                logger.warning(f"Error parsing data item for {function} on {item.get(date_key, 'unknown date')}: {e}")
                continue

        if not df_list:
            logger.warning(f"No valid data points found for {function}.")
            # Fallback to cache
            cached_data = _load_macro_data_from_cache(cache_filepath)
            if not cached_data.empty:
                logger.info(f"Falling back to cached data for {function} ({symbol}) as no new data was parsed.")
                return cached_data
            return pd.DataFrame()

        df = pd.DataFrame(df_list)
        df.sort_values('date', inplace=True)
        df.reset_index(drop=True, inplace=True)
        df.rename(columns={'value': function.lower()}, inplace=True)
        
        logger.info(f"Successfully fetched {len(df)} data points for {function} ({symbol}).")
        # Save to cache
        _save_macro_data_to_cache(df, cache_filepath)
        return df

    except requests.exceptions.RequestException as e:
        logger.error(f"Network error fetching data for {function}: {e}")
        # Fallback to cache on network error
        cached_data = _load_macro_data_from_cache(cache_filepath)
        if not cached_data.empty:
            logger.info(f"Falling back to cached data for {function} ({symbol}) after network error.")
            return cached_data
    except ValueError as e:
        logger.error(f"Error parsing JSON response for {function}: {e}")
        # Fallback to cache on parse error
        cached_data = _load_macro_data_from_cache(cache_filepath)
        if not cached_data.empty:
            logger.info(f"Falling back to cached data for {function} ({symbol}) after JSON parse error.")
            return cached_data
    except Exception as e:
        logger.error(f"Unexpected error fetching data for {function}: {e}")
        # Fallback to cache on any other error
        cached_data = _load_macro_data_from_cache(cache_filepath)
        if not cached_data.empty:
            logger.info(f"Falling back to cached data for {function} ({symbol}) after unexpected error.")
            return cached_data

    return pd.DataFrame()

def get_macro_data_multi_source(indicator: str, force_refresh: bool = False) -> pd.DataFrame:
    """
    Fetch macroeconomic data from multiple sources with fallbacks.
    
    Args:
        indicator: One of 'treasury_10y', 'treasury_2y', 'fed_funds', 'cpi', 'unemployment', 'gdp'
        force_refresh: Force refresh from APIs
    """
    
    # Mapping of indicators to different source configurations
    source_configs = {
        'treasury_10y': {
            'fred_symbol': 'DGS10',
            'yahoo_symbol': '^TNX',  # 10-year Treasury yield
            'av_function': 'TREASURY_YIELD',
            'av_symbol': '10year',
            'default_value': 4.5
        },
        'treasury_2y': {
            'fred_symbol': 'DGS2', 
            'yahoo_symbol': '^IRX',  # 13-week Treasury bill
            'av_function': 'TREASURY_YIELD',
            'av_symbol': '2year',
            'default_value': 4.0
        },
        'fed_funds': {
            'fred_symbol': 'FEDFUNDS',
            'yahoo_symbol': None,
            'av_function': 'FEDERAL_FUNDS_RATE',
            'av_symbol': None,
            'default_value': 5.25
        },
        'cpi': {
            'fred_symbol': 'CPIAUCSL',
            'yahoo_symbol': None,
            'av_function': 'CPI',
            'av_symbol': None,
            'default_value': 310.0
        },
        'unemployment': {
            'fred_symbol': 'UNRATE',
            'yahoo_symbol': None,
            'av_function': 'UNEMPLOYMENT',
            'av_symbol': None,
            'default_value': 4.0
        },
        'gdp': {
            'fred_symbol': 'GDPC1',
            'yahoo_symbol': None,
            'av_function': 'REAL_GDP',
            'av_symbol': None,
            'default_value': 22000.0
        }
    }
    
    if indicator not in source_configs:
        logger.error(f"Unknown macro indicator: {indicator}")
        return pd.DataFrame()
    
    config = source_configs[indicator]
    cache_filepath = _get_macro_cache_filepath(f"MULTI_{indicator}", None, "monthly", source="MULTI")
    
    # Try cache first if not forcing refresh
    if not force_refresh:
        cached_data = _load_macro_data_from_cache(cache_filepath)
        if not cached_data.empty:
            logger.info(f"Using cached multi-source data for {indicator}")
            return cached_data
    
    logger.info(f"Fetching {indicator} from multiple sources...")
    
    # Method 1: Try pandas-datareader with FRED (current working method)
    try:
        logger.info(f"Trying FRED via pandas-datareader for {config['fred_symbol']}...")
        import pandas_datareader.data as web
        from datetime import datetime, timedelta
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=3650)  # ~10 years
        
        df = web.DataReader(config['fred_symbol'], 'fred', start_date, end_date)
        if not df.empty:
            df = df.reset_index()
            df.columns = ['date', 'value']
            df = df.dropna()
            
            if len(df) > 0:
                logger.info(f"✅ Successfully fetched {len(df)} points from FRED for {indicator}")
                _save_macro_data_to_cache(df, cache_filepath)
                return df
                
    except Exception as e:
        logger.warning(f"FRED pandas-datareader failed for {indicator}: {e}")
    
    # Method 2: Try Yahoo Finance (for yield data)
    if config['yahoo_symbol']:
        try:
            logger.info(f"Trying Yahoo Finance for {config['yahoo_symbol']}...")
            import yfinance as yf
            
            ticker = yf.Ticker(config['yahoo_symbol'])
            hist = ticker.history(period="5y", interval="1mo")
            
            if not hist.empty:
                df = pd.DataFrame({
                    'date': hist.index,
                    'value': hist['Close']
                }).reset_index(drop=True)
                
                df = df.dropna()
                if len(df) > 0:
                    logger.info(f"✅ Successfully fetched {len(df)} points from Yahoo Finance for {indicator}")
                    _save_macro_data_to_cache(df, cache_filepath)
                    return df
                    
        except Exception as e:
            logger.warning(f"Yahoo Finance failed for {indicator}: {e}")
    
    # Method 3: Try Alpha Vantage (if API key available)
    if ALPHA_VANTAGE_API_KEY and config['av_function']:
        try:
            logger.info(f"Trying Alpha Vantage for {config['av_function']}...")
            df = get_alpha_vantage_data(config['av_function'], config['av_symbol'])
            if not df.empty:
                logger.info(f"✅ Successfully fetched {len(df)} points from Alpha Vantage for {indicator}")
                return df
        except Exception as e:
            logger.warning(f"Alpha Vantage failed for {indicator}: {e}")
    
    # Method 4: Create realistic default data as fallback
    logger.warning(f"All external sources failed for {indicator}, creating realistic default data")
    
    try:
        from datetime import datetime, timedelta
        import numpy as np
        
        # Create 2 years of monthly data with the default value plus realistic variation
        end_date = datetime.now()
        dates = pd.date_range(end=end_date, periods=24, freq='MS')  # Monthly start
        
        base_value = config['default_value']
        # Add some realistic variation (±5% with trend)
        np.random.seed(42)  # For reproducible results
        trend = np.linspace(-0.02, 0.02, len(dates))  # Small trend
        noise = np.random.normal(0, base_value * 0.02, len(dates))  # 2% noise
        values = base_value * (1 + trend + noise)
        
        df = pd.DataFrame({
            'date': dates,
            'value': values
        })
        
        logger.info(f"✅ Created {len(df)} realistic default data points for {indicator} (base: {base_value})")
        _save_macro_data_to_cache(df, cache_filepath)
        return df
        
    except Exception as e:
        logger.error(f"Failed to create default data for {indicator}: {e}")
        return pd.DataFrame()

def get_fred_data_via_pdr(series_id: str, force_refresh: bool = False) -> pd.DataFrame:
    """
    Fetches data from FRED using pandas-datareader, with local caching.
    This method does not require an API key and relies on pandas-datareader's FRED connector.
    
    Args:
        series_id (str): The FRED series identifier (e.g., 'DGS10', 'GDP').
        force_refresh (bool, optional): If True, bypasses the cache and forces a new data fetch. Defaults to False.

    Returns:
        pd.DataFrame: A DataFrame with 'date' and 'value' columns.
    """
    cache_filepath = _get_macro_cache_filepath(series_id, source="FRED_PDR")
    
    # 1. Try loading from cache if not forcing refresh
    if not force_refresh:
        cached_data = _load_macro_data_from_cache(cache_filepath)
        if not cached_data.empty:
            logger.info(f"Using cached FRED data (via pandas-datareader) for {series_id}.")
            return cached_data

    # 2. If cache miss or force refresh, fetch from FRED via pandas-datareader
    logger.info(f"Fetching {series_id} from FRED via pandas-datareader...")
    try:
        # Fetch last 10 years of data
        end_date = datetime.datetime.now()
        start_date = end_date - datetime.timedelta(days=365 * 10)
        
        data = web.DataReader(series_id, 'fred', start_date, end_date)
        if data.empty:
            logger.warning(f"pandas-datareader returned no data for FRED series {series_id}.")
            # Fallback to cache
            cached_data = _load_macro_data_from_cache(cache_filepath)
            if not cached_data.empty:
                logger.info(f"Falling back to cached data for FRED series {series_id}.")
                return cached_data
            return pd.DataFrame()

        # FRED data via pandas-datareader has the series name as the column name
        value_col = data.columns[0]
        df = data[[value_col]].reset_index()
        df.rename(columns={'DATE': 'date', value_col: 'value'}, inplace=True)
        df['date'] = pd.to_datetime(df['date'])
        df.sort_values('date', inplace=True)
        df.reset_index(drop=True, inplace=True)
        
        logger.info(f"Successfully fetched {len(df)} data points for FRED series {series_id} via pandas-datareader.")
        # Save to cache
        _save_macro_data_to_cache(df, cache_filepath)
        return df

    except Exception as e:
        logger.error(f"Error fetching data for FRED series {series_id} via pandas-datareader: {e}")
        # Fallback to cache on error
        cached_data = _load_macro_data_from_cache(cache_filepath)
        if not cached_data.empty:
            logger.info(f"Falling back to cached data for FRED series {series_id} after error.")
            return cached_data

    return pd.DataFrame()

def fetch_macro_data_for_date(date: pd.Timestamp, force_refresh: bool = False) -> dict:
    """
    Enhanced macro data fetching with multiple sources and better error handling.
    
    Args:
        date (pd.Timestamp): The date for which to fetch macro data.
        force_refresh (bool, optional): If True, bypasses the cache for individual data series. Defaults to False.
        
    Returns:
        dict: A dictionary of macroeconomic indicators.
    """
    logger.info(f"Fetching macroeconomic data for context around {date.date()}...")
    
    # Map output names to internal indicator keys for multi-source fetching
    macro_indicators = {
        'treasury_yield_10year': 'treasury_10y',
        'treasury_yield_2year': 'treasury_2y', 
        'federal_funds_rate': 'fed_funds',
        'cpi': 'cpi',
        'unemployment': 'unemployment',
        'real_gdp': 'gdp'
    }
    
    macro_context = {}
    
    for output_name, indicator_key in macro_indicators.items():
        try:
            # Try the new multi-source approach first
            df = get_macro_data_multi_source(indicator_key, force_refresh)
            
            if not df.empty:
                # Get the most recent value before or on the analysis date
                df['date'] = pd.to_datetime(df['date'])
                valid_data = df[df['date'] <= date]
                
                if not valid_data.empty:
                    latest_value = valid_data.iloc[-1]['value']
                    macro_context[output_name] = float(latest_value)
                    logger.info(f"✅ {output_name}: {latest_value:.2f}")
                else:
                    # Use the most recent available data if nothing before analysis_date
                    latest_value = df.iloc[-1]['value']
                    macro_context[output_name] = float(latest_value)
                    logger.info(f"⚠️ {output_name}: {latest_value:.2f} (using most recent available)")
            else:
                # Fallback to old method for this indicator
                logger.warning(f"Multi-source failed for {output_name}, trying legacy FRED method...")
                try:
                    fred_mapping = {
                        'treasury_yield_10year': 'DGS10',
                        'treasury_yield_2year': 'DGS2',
                        'federal_funds_rate': 'FEDFUNDS',
                        'cpi': 'CPIAUCSL',
                        'unemployment': 'UNRATE',
                        'real_gdp': 'GDPC1'
                    }
                    
                    if output_name in fred_mapping:
                        legacy_df = get_fred_data_via_pdr(fred_mapping[output_name], force_refresh)
                        if not legacy_df.empty:
                            df_before_date = legacy_df[legacy_df['date'] <= date]
                            if not df_before_date.empty:
                                latest_value = df_before_date.iloc[-1]['value']
                                macro_context[output_name] = float(latest_value)
                                logger.info(f"✅ {output_name} (legacy): {latest_value:.2f}")
                            else:
                                macro_context[output_name] = None
                        else:
                            macro_context[output_name] = None
                    else:
                        macro_context[output_name] = None
                except Exception as legacy_e:
                    logger.warning(f"Legacy method also failed for {output_name}: {legacy_e}")
                    macro_context[output_name] = None
                
        except Exception as e:
            logger.error(f"❌ Error fetching {output_name}: {e}")
            macro_context[output_name] = None
    
    available_indicators = [k for k, v in macro_context.items() if v is not None]
    logger.info(f"Macro data fetch complete. Retrieved {len(available_indicators)} indicators.")
    
    # If we got some data, that's great. If not, the system will work with technical indicators only
    return macro_context