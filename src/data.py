import yfinance as yf
import pandas as pd
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

CACHE_DIR = Path("data_cache")

def get_etf_data(ticker: str, period: str = '5y', force_refresh: bool = False) -> tuple[pd.DataFrame, dict]:
    """
    Retrieves ETF data, with a local caching system.

    Args:
        ticker (str): The ETF ticker.
        period (str): The data period to retrieve (e.g., '5y').
        force_refresh (bool): If True, forces download even if a cache exists.

    Returns:
        tuple[pd.DataFrame, dict]: A tuple containing the DataFrame of historical data
                                   and a dictionary of ETF information.
    """
    # Create cache filename
    cache_filename = f"{ticker.replace('.', '_')}_{period}.parquet"
    cache_filepath = CACHE_DIR / cache_filename

    # Check if cache exists and is not forced to be refreshed
    if not force_refresh and cache_filepath.exists():
        try:
            logger.info(f"Loading data from cache: {cache_filepath}")
            hist_data = pd.read_parquet(cache_filepath)

            # Retrieve ETF information (not cached)
            etf = yf.Ticker(ticker)
            try:
                info = etf.info
            except Exception:
                info = {"longName": "ETF NASDAQ France (from cache)", "currency": "EUR"}

            logger.info(f"Data loaded from cache: {len(hist_data)} days.")
            return hist_data, info
        except Exception as e:
            logger.warning(f"Could not read cache file {cache_filepath}: {e}. Downloading data.")

    # If cache does not exist or refresh is forced
    logger.info(f"Downloading data for {ticker}...")
    try:
        etf = yf.Ticker(ticker)
        hist_data = etf.history(period=period, auto_adjust=True, prepost=True)

        if hist_data.empty:
            raise ValueError(f"No data found for ticker {ticker}")

        # Data cleaning
        hist_data = hist_data.dropna()

        # ETF information
        try:
            info = etf.info
        except Exception:
            info = {"longName": "ETF NASDAQ France", "currency": "EUR"}

        # Save data to cache
        logger.info(f"Saving data to cache: {cache_filepath}")
        os.makedirs(CACHE_DIR, exist_ok=True)
        hist_data.to_parquet(cache_filepath)

        logger.info(f"Data retrieved: {len(hist_data)} trading days")
        logger.info(f"Period: {hist_data.index[0].date()} to {hist_data.index[-1].date()}")

        return hist_data, info

    except Exception as e:
        logger.error(f"Error retrieving data: {e}")
        raise
