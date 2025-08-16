import yfinance as yf
import pandas as pd
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

CACHE_DIR = Path("data_cache")

def get_etf_data(ticker: str, period: str = '5y', force_refresh: bool = False) -> tuple[pd.DataFrame, dict]:
    """
    Récupération des données de l'ETF, avec un système de cache local.

    Args:
        ticker (str): Le ticker de l'ETF.
        period (str): La période de données à récupérer (ex: '5y').
        force_refresh (bool): Si True, force le téléchargement même si un cache existe.

    Returns:
        tuple[pd.DataFrame, dict]: Un tuple contenant le DataFrame des données historiques
                                   et un dictionnaire d'informations sur l'ETF.
    """
    # Créer le nom du fichier de cache
    cache_filename = f"{ticker.replace('.', '_')}_{period}.parquet"
    cache_filepath = CACHE_DIR / cache_filename

    # Vérifier si le cache existe et n'est pas forcé à être rafraîchi
    if not force_refresh and cache_filepath.exists():
        try:
            logger.info(f"Chargement des données depuis le cache: {cache_filepath}")
            hist_data = pd.read_parquet(cache_filepath)

            # Récupérer les informations de l'ETF (ce n'est pas mis en cache)
            etf = yf.Ticker(ticker)
            try:
                info = etf.info
            except Exception:
                info = {"longName": "ETF NASDAQ France (from cache)", "currency": "EUR"}

            logger.info(f"Données chargées depuis le cache: {len(hist_data)} jours.")
            return hist_data, info
        except Exception as e:
            logger.warning(f"Impossible de lire le fichier de cache {cache_filepath}: {e}. Téléchargement des données.")

    # Si le cache n'existe pas ou si le rafraîchissement est forcé
    logger.info(f"Téléchargement des données pour {ticker}...")
    try:
        etf = yf.Ticker(ticker)
        hist_data = etf.history(period=period, auto_adjust=True, prepost=True)

        if hist_data.empty:
            raise ValueError(f"Aucune donnée trouvée pour le ticker {ticker}")

        # Nettoyage des données
        hist_data = hist_data.dropna()

        # Informations sur l'ETF
        try:
            info = etf.info
        except Exception:
            info = {"longName": "ETF NASDAQ France", "currency": "EUR"}

        # Sauvegarder les données dans le cache
        logger.info(f"Sauvegarde des données dans le cache: {cache_filepath}")
        os.makedirs(CACHE_DIR, exist_ok=True)
        hist_data.to_parquet(cache_filepath)

        logger.info(f"Données récupérées: {len(hist_data)} jours de cotation")
        logger.info(f"Période: {hist_data.index[0].date()} à {hist_data.index[-1].date()}")

        return hist_data, info

    except Exception as e:
        logger.error(f"Erreur lors de la récupération des données: {e}")
        raise
