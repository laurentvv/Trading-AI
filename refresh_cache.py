
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from data import get_etf_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RefreshCache")

tickers = ["^NDX", "CL=F", "SXRV.DE", "CRUDP.PA"]

for ticker in tickers:
    logger.info(f"Force refreshing cache for {ticker}...")
    try:
        get_etf_data(ticker, force_refresh=True)
        logger.info(f"Successfully refreshed {ticker}")
    except Exception as e:
        logger.error(f"Failed to refresh {ticker}: {e}")
