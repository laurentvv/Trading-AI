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

logger = logging.getLogger(__name__)

CACHE_DIR = Path("data_cache")
# Create a specific subdirectory for macroeconomic data cache
MACRO_CACHE_DIR = CACHE_DIR / "macro"
MACRO_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Alpha Vantage API setup
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
if not ALPHA_VANTAGE_API_KEY:
    logger.warning(
        "ALPHA_VANTAGE_API_KEY not found in environment variables. Some features might be disabled."
    )


def get_etf_data(
    ticker: str, period: str = "5y", force_refresh: bool = False
) -> tuple[pd.DataFrame, dict]:
    """
    Retrieves ETF and VIX data, with a local caching system.
    The cache now includes VIX data.
    """
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
            info = etf.info
            logger.info(f"Data loaded from cache: {len(hist_data)} days.")
        except Exception as e:
            logger.warning(
                f"Could not read cache file {cache_filepath}: {e}. Forcing refresh."
            )
            force_refresh = True

    if force_refresh or hist_data is None:
        logger.info(f"Downloading data for {ticker} and ^VIX...")
        try:
            all_data = yf.download([ticker, "^VIX"], period="max", auto_adjust=True)
            if all_data.empty:
                raise ValueError(f"No data found for tickers {ticker}, ^VIX")

            # Restructure the multi-level column dataframe
            close_prices = all_data["Close"]
            hist_data = pd.DataFrame(index=close_prices.index)
            hist_data["Open"] = all_data["Open"][ticker]
            hist_data["High"] = all_data["High"][ticker]
            hist_data["Low"] = all_data["Low"][ticker]
            hist_data["Close"] = close_prices[ticker]
            hist_data["Volume"] = all_data["Volume"][ticker]
            hist_data["VIX"] = close_prices["^VIX"]

            # Clean up data
            hist_data = hist_data.dropna(how="all")
            hist_data["VIX"] = hist_data["VIX"].ffill().bfill()
            hist_data = hist_data.dropna()

            etf = yf.Ticker(ticker)
            info = etf.info

            hist_data.to_parquet(cache_filepath)
            logger.info(f"Data for {ticker} and ^VIX saved to cache: {cache_filepath}")

        except Exception as e:
            logger.error(f"Error retrieving data: {e}")
            raise

    # Final check for old cache format without VIX
    if "VIX" not in hist_data.columns:
        logger.warning("VIX column not found in loaded data. Forcing a refresh.")
        return get_etf_data(ticker, period, force_refresh=True)

    logger.info(
        f"Final data period: {hist_data.index.min().date()} to {hist_data.index.max().date()}"
    )
    return hist_data, info


def _get_macro_cache_filepath(
    function: str, symbol: str = None, interval: str = "monthly", source: str = "AV"
) -> Path:
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


def get_alpha_vantage_data(
    function: str,
    symbol: str = None,
    interval: str = "monthly",
    force_refresh: bool = False,
) -> pd.DataFrame:
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
    url = "https://www.alphavantage.co/query"
    params = {
        "function": function,
        "apikey": ALPHA_VANTAGE_API_KEY,
        "datatype": "json",
        "interval": interval,
    }
    if symbol:
        params["symbol"] = symbol

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()

        # Check for API errors
        if "Error Message" in data:
            logger.error(
                f"Alpha Vantage API Error for {function}: {data['Error Message']}"
            )
            # If API error, try to return cached data as a fallback
            cached_data = _load_macro_data_from_cache(cache_filepath)
            if not cached_data.empty:
                logger.info(
                    f"Falling back to cached data for {function} ({symbol}) after API error."
                )
                return cached_data
            return pd.DataFrame()

        if "Information" in data and "limit" in data["Information"]:
            logger.warning(f"Alpha Vantage API call limit reached for {function}.")
            # If rate limit hit, try to return cached data as a fallback
            cached_data = _load_macro_data_from_cache(cache_filepath)
            if not cached_data.empty:
                logger.info(
                    f"Falling back to cached data for {function} ({symbol}) due to rate limit."
                )
                return cached_data
            return pd.DataFrame()

        # Extract the data series (same logic as before)
        data_key = None
        if "data" in data:
            data_key = "data"
        elif "TreasureYield" in data:  # Note: API spelling
            data_key = "TreasureYield"
        elif function in data:
            data_key = function
        else:
            # Try to find the first key that looks like a data series
            for k in data.keys():
                if (
                    isinstance(data[k], list)
                    and len(data[k]) > 0
                    and isinstance(data[k][0], dict)
                ):
                    data_key = k
                    break

        if not data_key or data_key not in data:
            logger.warning(f"Could not find data series for {function} in response.")
            # Fallback to cache
            cached_data = _load_macro_data_from_cache(cache_filepath)
            if not cached_data.empty:
                logger.info(
                    f"Falling back to cached data for {function} ({symbol}) after parsing error."
                )
                return cached_data
            return pd.DataFrame()

        df_list = []
        for item in data[data_key]:
            # AV API responses can have varying key names for the date
            date_key = "date" if "date" in item else None
            if not date_key:
                # Common alternatives
                for k in ["timestamp", "datetime"]:
                    if k in item:
                        date_key = k
                        break
            if not date_key:
                logger.warning(f"Could not find date key in data item for {function}.")
                continue  # Skip this item

            # Value key can also vary
            value_key = None
            for k in ["value", "yield", "rate", "cpi"]:
                if k in item:
                    value_key = k
                    break
            if not value_key:
                logger.warning(f"Could not find value key in data item for {function}.")
                continue

            try:
                df_list.append(
                    {
                        "date": pd.to_datetime(item[date_key]),
                        "value": float(item[value_key]),
                    }
                )
            except (ValueError, TypeError) as e:
                logger.warning(
                    f"Error parsing data item for {function} on {item.get(date_key, 'unknown date')}: {e}"
                )
                continue

        if not df_list:
            logger.warning(f"No valid data points found for {function}.")
            # Fallback to cache
            cached_data = _load_macro_data_from_cache(cache_filepath)
            if not cached_data.empty:
                logger.info(
                    f"Falling back to cached data for {function} ({symbol}) as no new data was parsed."
                )
                return cached_data
            return pd.DataFrame()

        df = pd.DataFrame(df_list)
        df.sort_values("date", inplace=True)
        df.reset_index(drop=True, inplace=True)
        df.rename(columns={"value": function.lower()}, inplace=True)

        logger.info(
            f"Successfully fetched {len(df)} data points for {function} ({symbol})."
        )
        # Save to cache
        _save_macro_data_to_cache(df, cache_filepath)
        return df

    except requests.exceptions.RequestException as e:
        logger.error(f"Network error fetching data for {function}: {e}")
        # Fallback to cache on network error
        cached_data = _load_macro_data_from_cache(cache_filepath)
        if not cached_data.empty:
            logger.info(
                f"Falling back to cached data for {function} ({symbol}) after network error."
            )
            return cached_data
    except ValueError as e:
        logger.error(f"Error parsing JSON response for {function}: {e}")
        # Fallback to cache on parse error
        cached_data = _load_macro_data_from_cache(cache_filepath)
        if not cached_data.empty:
            logger.info(
                f"Falling back to cached data for {function} ({symbol}) after JSON parse error."
            )
            return cached_data
    except Exception as e:
        logger.error(f"Unexpected error fetching data for {function}: {e}")
        # Fallback to cache on any other error
        cached_data = _load_macro_data_from_cache(cache_filepath)
        if not cached_data.empty:
            logger.info(
                f"Falling back to cached data for {function} ({symbol}) after unexpected error."
            )
            return cached_data

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
            logger.info(
                f"Using cached FRED data (via pandas-datareader) for {series_id}."
            )
            return cached_data

    # 2. If cache miss or force refresh, fetch from FRED via pandas-datareader
    logger.info(f"Fetching {series_id} from FRED via pandas-datareader...")
    try:
        # Fetch last 10 years of data
        end_date = datetime.datetime.now()
        start_date = end_date - datetime.timedelta(days=365 * 10)

        data = web.DataReader(series_id, "fred", start_date, end_date)
        if data.empty:
            logger.warning(
                f"pandas-datareader returned no data for FRED series {series_id}."
            )
            # Fallback to cache
            cached_data = _load_macro_data_from_cache(cache_filepath)
            if not cached_data.empty:
                logger.info(f"Falling back to cached data for FRED series {series_id}.")
                return cached_data
            return pd.DataFrame()

        # FRED data via pandas-datareader has the series name as the column name
        value_col = data.columns[0]
        df = data[[value_col]].reset_index()
        df.rename(columns={"DATE": "date", value_col: "value"}, inplace=True)
        df["date"] = pd.to_datetime(df["date"])
        df.sort_values("date", inplace=True)
        df.reset_index(drop=True, inplace=True)

        logger.info(
            f"Successfully fetched {len(df)} data points for FRED series {series_id} via pandas-datareader."
        )
        # Save to cache
        _save_macro_data_to_cache(df, cache_filepath)
        return df

    except Exception as e:
        logger.error(
            f"Error fetching data for FRED series {series_id} via pandas-datareader: {e}"
        )
        # Fallback to cache on error
        cached_data = _load_macro_data_from_cache(cache_filepath)
        if not cached_data.empty:
            logger.info(
                f"Falling back to cached data for FRED series {series_id} after error."
            )
            return cached_data

    return pd.DataFrame()


def fetch_macro_data_for_date(date: pd.Timestamp, force_refresh: bool = False) -> dict:
    """
    Fetches key macroeconomic data points relevant for a given date.
    This function now uses caching and tries multiple sources.

    Args:
        date (pd.Timestamp): The date for which to fetch macro data.
        force_refresh (bool, optional): If True, bypasses the cache for individual data series. Defaults to False.

    Returns:
        dict: A dictionary of macroeconomic indicators.
    """
    logger.info(f"Fetching macroeconomic data for context around {date.date()}...")

    # Define the mapping of our internal names to FRED series IDs
    fred_series_mapping = {
        "treasury_yield_10year": "DGS10",
        "treasury_yield_2year": "DGS2",
        "federal_funds_rate": "FEDFUNDS",
        "cpi": "CPIAUCSL",
        "unemployment": "UNRATE",
        "real_gdp": "GDPC1",
    }

    macro_data = {}

    for internal_name, fred_id in fred_series_mapping.items():
        data_df = get_fred_data_via_pdr(fred_id, force_refresh)

        if not data_df.empty:
            # For simplicity, take the most recent data point before or on the given date
            df_before_or_on_date = data_df[data_df["date"] <= date]
            if not df_before_or_on_date.empty:
                latest_value = df_before_or_on_date.iloc[-1]["value"]
                macro_data[internal_name] = latest_value
                logger.debug(
                    f"Fetched {internal_name} ({fred_id}): {latest_value} for date <= {date.date()}"
                )
            else:
                logger.warning(
                    f"No historical data found for {internal_name} ({fred_id}) before or on {date.date()}."
                )
        else:
            logger.warning(
                f"Failed to fetch or load data for {internal_name} ({fred_id}) from FRED via pandas-datareader. Skipping."
            )

    logger.info(f"Macro data fetch complete. Retrieved {len(macro_data)} indicators.")
    return macro_data
