import yfinance as yf
import pandas as pd
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

CACHE_DIR = Path("data_cache")

def get_etf_data(ticker: str, period: str = '5y', force_refresh: bool = False) -> tuple[pd.DataFrame, dict]:
    """
    Retrieves ETF and VIX data, with a local caching system.
    The cache now includes VIX data.
    """
    CACHE_DIR.mkdir(exist_ok=True)
    cache_filename = f"{ticker.replace('.', '_')}_{period}_with_vix.parquet"
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
            logger.warning(f"Could not read cache file {cache_filepath}: {e}. Forcing refresh.")
            force_refresh = True

    if force_refresh or hist_data is None:
        logger.info(f"Downloading data for {ticker} and ^VIX...")
        try:
            all_data = yf.download([ticker, '^VIX'], period=period, auto_adjust=True)
            if all_data.empty:
                raise ValueError(f"No data found for tickers {ticker}, ^VIX")

            # Restructure the multi-level column dataframe
            close_prices = all_data['Close']
            hist_data = pd.DataFrame(index=close_prices.index)
            hist_data['Open'] = all_data['Open'][ticker]
            hist_data['High'] = all_data['High'][ticker]
            hist_data['Low'] = all_data['Low'][ticker]
            hist_data['Close'] = close_prices[ticker]
            hist_data['Volume'] = all_data['Volume'][ticker]
            hist_data['VIX'] = close_prices['^VIX']

            # Clean up data
            hist_data = hist_data.dropna(how='all')
            hist_data['VIX'] = hist_data['VIX'].ffill().bfill()
            hist_data = hist_data.dropna()

            etf = yf.Ticker(ticker)
            info = etf.info

            hist_data.to_parquet(cache_filepath)
            logger.info(f"Data for {ticker} and ^VIX saved to cache: {cache_filepath}")

        except Exception as e:
            logger.error(f"Error retrieving data: {e}")
            raise

    # Final check for old cache format without VIX
    if 'VIX' not in hist_data.columns:
        logger.warning("VIX column not found in loaded data. Forcing a refresh.")
        return get_etf_data(ticker, period, force_refresh=True)

    logger.info(f"Final data period: {hist_data.index.min().date()} to {hist_data.index.max().date()}")
    return hist_data, info
