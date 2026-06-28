"""
Exemple d'Intégration des Améliorations
Script de démonstration montrant comment intégrer tous les nouveaux modules
dans le système de trading AI existant.
"""

import logging
import numpy as np
import pandas as pd
import sqlite3
from typing import Dict
from datetime import datetime
from pathlib import Path
import subprocess
import json
import sys
import os
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv

# Imports des modules existants
from data import get_etf_data, fetch_macro_data_for_date, get_vincent_ganne_indicators

try:
    from t212_executor import get_t212_price
except ImportError:
    get_t212_price = None

from features import create_technical_indicators, create_features, select_features
from classic_model import train_ensemble_model, get_classic_prediction
from llm_client import get_llm_decision, get_visual_llm_decision, get_council_ticker_stance
from sentiment_analysis import get_sentiment_decision_from_score
from web_researcher import generate_search_query, get_web_context_sync, get_fallback_search_query

from chart_generator import generate_chart_image

from database import (
    init_db,
    insert_transactions_batch,
    insert_portfolio_state,
    get_latest_portfolio_state,
)
from timesfm_model import get_timesfm_prediction
from tensortrade_model import get_tensortrade_prediction
from eia_client import EIAClient
from oil_bench_model import OilBenchModel
from grebenkov_model import GrebenkovTrendModel
from hmm_model import HMMDecisionModel

# Imports des nouveaux modules d'amélioration
from enhanced_decision_engine import EnhancedDecisionEngine, ModelResult
from advanced_risk_manager import AdvancedRiskManager
from adaptive_weight_manager import AdaptiveWeightManager
from performance_monitor import PerformanceMonitor

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)

# --- Constants for the Alpha Vantage API ---
# IMPORTANT: It is strongly recommended to use an environment variable for your API key.
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
if not ALPHA_VANTAGE_API_KEY:
    logger.critical("CRITICAL: The ALPHA_VANTAGE_API_KEY environment variable is not set.")
    logger.critical("Please set it to your Alpha Vantage API key.")
    sys.exit(1)


class EnhancedTradingSystem:
    """
    Système de trading AI amélioré intégrant tous les nouveaux composants.
    Utilise une approche dual-ticker : analyse l'indice de référence pour les modèles,
    mais exécute les transactions sur l'ETF spécifié.
    """

    # Mapping des tickers de trading vers les tickers d'analyse (indices)
    ANALYSIS_MAPPING = {
        "SXRV.DE": "^NDX",  # iShares Nasdaq 100 -> Nasdaq 100 Index
        "SXRV.FRK": "^NDX",
        "CRUDP.PA": "CL=F",  # Lyxor WTI Oil -> Crude Oil Futures
        "QQQ": "^NDX",
        "SPY": "^GSPC",
    }

    def __init__(self, ticker: str = "QQQ", initial_portfolio_value: float = 100000):
        """
        Initialise le système de trading amélioré.

        Args:
            ticker: Symbole de l'ETF à trader
            initial_portfolio_value: Valeur initiale du portefeuille
        """
        self.ticker = ticker  # Ticker de TRADING (ex: SXRV.DE)
        self.analysis_ticker = self.ANALYSIS_MAPPING.get(ticker, ticker)  # Ticker d'ANALYSE (ex: ^NDX)
        self.initial_portfolio_value = initial_portfolio_value

        # Load centralized configuration
        self.config = self._load_config()

        # Initialisation des composants améliorés avec injection de config
        self.grebenkov_model = GrebenkovTrendModel()
        self.hmm_model = HMMDecisionModel()
        self.decision_engine = EnhancedDecisionEngine(config=self.config)
        self.risk_manager = AdvancedRiskManager(config=self.config)
        self.weight_manager = AdaptiveWeightManager(config=self.config)
        self.performance_monitor = PerformanceMonitor(ticker=self.ticker)

        # Configuration du système
        self.chart_output_path = Path("enhanced_trading_chart.png")

        # Initialize database and portfolio
        init_db()
        # Pour l'initialisation du portefeuille, on utilise le ticker de trading
        self._initialize_portfolio(hist_data=get_etf_data(ticker=self.ticker)[0])

        logger.info(f"Système de trading amélioré initialisé. Trading: {self.ticker} | Analyse: {self.analysis_ticker}")

    def _load_config(self) -> Dict:
        """Loads the scheduler_config.json file."""
        config_path = Path("scheduler_config.json")
        if config_path.exists():
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load scheduler_config.json: {e}")
        return {}

    def _initialize_portfolio(self, hist_data):
        """Initializes the portfolio if it doesn't exist."""
        if get_latest_portfolio_state(self.ticker) is None:
            logger.info(f"No existing portfolio found for {self.ticker}. Initializing a new one.")
            insert_portfolio_state(
                date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                ticker=self.ticker,
                position=0,
                cash=self.initial_portfolio_value,
                total_value=self.initial_portfolio_value,
                benchmark_value=self.initial_portfolio_value,
            )

    def prepare_data_and_features(self):
        """Prépare les données et indicateurs techniques basés sur l'indice d'analyse."""
        logger.info(f"Récupération et préparation des données pour l'indice {self.analysis_ticker}...")

        try:
            # 1. Récupération des données de l'indice pour l'analyse IA
            hist_data, info = get_etf_data(ticker=self.analysis_ticker)

            # 2. Récupération du prix actuel de l'ETF pour le trading
            current_etf_price = None
            if get_t212_price:
                try:
                    current_etf_price = get_t212_price(self.ticker)
                except Exception:
                    pass
            if not current_etf_price:
                etf_data, _ = get_etf_data(ticker=self.ticker)
                current_etf_price = etf_data["Close"].iloc[-1]
            logger.info(f"Prix actuel de l'ETF ({self.ticker}): {current_etf_price:.2f}")

            # Validate data
            if hist_data is None or hist_data.empty:
                raise ValueError(f"No data retrieved for analysis ticker {self.analysis_ticker}")

            logger.info(
                f"Analysis Data retrieved: {len(hist_data)} rows from {hist_data.index.min()} to {hist_data.index.max()}"
            )

            # Check minimum data requirements
            if len(hist_data) < 50:
                raise ValueError(f"Insufficient data for analysis: only {len(hist_data)} rows available")

            data_with_indicators = create_technical_indicators(hist_data)

            # Validate indicators
            if data_with_indicators.empty:
                raise ValueError("Technical indicators generation failed - empty result")

            # Contexte macroéconomique
            analysis_date = data_with_indicators.index[-1]
            macro_context = fetch_macro_data_for_date(analysis_date)

            # Indicateurs Vincent Ganne (Nouveau)
            vg_indicators = get_vincent_ganne_indicators()

            # Création des features avec contexte macro
            data_with_features = create_features(data_with_indicators, macro_context)

            # Final validation
            if data_with_features.empty:
                raise ValueError("Feature generation failed - empty result")

            logger.info(
                f"Final data with features: {len(data_with_features)} rows, {len(data_with_features.columns)} columns"
            )

            # On retourne aussi le prix de l'ETF pour les calculs de trading
            return (
                data_with_features,
                hist_data,
                macro_context,
                vg_indicators,
                current_etf_price,
            )

        except Exception as e:
            logger.error(f"Error in data preparation: {e}")
            raise

    def train_classic_model(self, data_with_features):
        """Entraîne le modèle classique."""
        logger.info("Entraînement du modèle quantitatif...")

        X, y, _ = select_features(data_with_features)

        # Log data quality information
        logger.info(f"Features shape: {X.shape}")
        logger.info(f"Target shape: {y.shape}")
        logger.info(f"NaN in features: {X.isnull().sum().sum()}")
        logger.info(f"NaN in target: {y.isnull().sum()}")
        logger.info(f"Unique target values: {y.unique()}")

        if len(y.unique()) < 2:
            logger.warning("Données insuffisantes pour l'entraînement - moins de 2 classes")
            return None

        # Check if we have enough valid data
        valid_data_count = (~y.isnull()).sum()
        if valid_data_count < 50:
            logger.warning(
                f"Données insuffisantes pour l'entraînement - seulement {valid_data_count} échantillons valides"
            )
            return None

        try:
            # Entraînement du modèle
            classic_pipeline, metrics, _ = train_ensemble_model(X, y)
            logger.info("Modèle classique entraîné avec succès")
            return classic_pipeline
        except Exception as e:
            logger.error(f"Erreur lors de l'entraînement du modèle: {e}")
            return None

    def get_model_predictions(
        self, data_with_features, classic_pipeline, vg_indicators=None, wti_data=None, nasdaq_data=None
    ):
        """Obtient les prédictions de tous les modèles.

        Architecture parallèle :
        - Phase A (parallèle) : news + search_query + visual_llm + timesfm/tensortrade/grebenkov
        - Phase B (dès que search_query done) : web_context
        - Phase C (dès que news + web_context done) : text_llm_decision

        Les appels LLM indépendants tournent dans un ThreadPoolExecutor.
        """
        logger.info("Génération des prédictions des modèles (parallèle)...")

        latest_data = data_with_features.tail(1)

        # 1. Prédiction du modèle classique
        if classic_pipeline is not None:
            try:
                if hasattr(classic_pipeline, "named_steps"):
                    feature_cols = [col for col in classic_pipeline.named_steps["scaler"].feature_names_in_ if col in latest_data.columns]
                else:
                    feature_cols = list(latest_data.columns)
                latest_features = latest_data[feature_cols]

                # Log feature information
                logger.info(f"Features disponibles pour prédiction: {len(feature_cols)}")
                logger.info(f"NaN dans les features: {latest_features.isnull().sum().sum()}")

                classic_pred, classic_conf = get_classic_prediction(classic_pipeline, latest_features)
                logger.info(f"Prédiction classique: {classic_pred}, confiance: {classic_conf:.3f}")
            except Exception as e:
                logger.error(f"Erreur lors de la prédiction classique: {e}")
                classic_pred, classic_conf = 0, 0.5
        else:
            logger.warning("Modèle classique non disponible, utilisation de valeurs par défaut")
            classic_pred, classic_conf = 0, 0.5

        # 2. Génération du graphique pour l'analyse visuelle
        chart_generated = generate_chart_image(
            data_with_features,
            self.chart_output_path,
            title=f"{self.ticker} - Enhanced Analysis Chart",
        )

        # ============================================================
        # PHASE A : lancement en parallèle des tâches indépendantes
        # ============================================================
        # Sequential critical path: 240 (search) + 30 (web) + 90 (news wait)
        # + 240 (text_llm) = 600s. Then visual_llm (300s) and 3 cpu_models
        # (180s each, parallel) are awaited in finalisation, adding at most
        # 300s. Total worst case: 600 + 300 = 900s = CYCLE_TIMEOUT_SECONDS.
        # We use 6 workers: 4 LLM-capable tasks + 3 CPU-model tasks = 7 in flight,
        # but LLM calls serialize inside Ollama anyway so 6 workers is fine.
        executor = ThreadPoolExecutor(max_workers=6, thread_name_prefix="model_pred")

        # Tâche A1 : fetch news (subprocess ~30s)
        def _fetch_news_task():
            headlines = []
            sentiment_score = 0
            try:
                script_path = Path(__file__).parent / "news_fetcher.py"
                python_executable = sys.executable
                process = subprocess.run(
                    [
                        python_executable,
                        str(script_path),
                        self.ticker,
                        ALPHA_VANTAGE_API_KEY,
                    ],
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
                if process.returncode != 0:
                    logger.error(f"News fetcher failed (exit {process.returncode}): {process.stderr[:500]}")
                else:
                    news_data = json.loads(process.stdout)
                    headlines = news_data.get("headlines", [])
                    sentiment_score = news_data.get("sentiment", 0)
                    if not headlines:
                        logger.warning(f"News fetcher returned 0 headlines. stderr: {process.stderr[:300]}")
                    logger.info(
                        f"Successfully fetched {len(headlines)} news headlines. Sentiment score: {sentiment_score:.2f}"
                    )
            except Exception as e:
                logger.error(f"Failed to fetch news: {e}")
            return headlines, sentiment_score

        # Tâche A2 : generate_search_query (LLM ~2min)
        def _search_query_task():
            try:
                return generate_search_query(self.analysis_ticker, latest_data=data_with_features)
            except ValueError as e:
                logger.error(f"Ticker validation failed: {e}")
                return get_fallback_search_query(self.analysis_ticker)
            except Exception as e:
                logger.error(f"Search query generation failed: {e}")
                return get_fallback_search_query(self.analysis_ticker)

        # Tâche A3 : visual LLM (LLM ~3min)
        def _visual_llm_task():
            if not chart_generated:
                return {"signal": "HOLD", "confidence": 0.0, "analysis": "Chart generation failed."}
            try:
                return get_visual_llm_decision(self.chart_output_path)
            except Exception as e:
                logger.error(f"Visual LLM failed: {e}")
                return {"signal": "HOLD", "confidence": 0.0, "analysis": f"Visual LLM error: {e}"}

        # Tâches A4/A5/A6 : CPU models indépendants (1 future chacun)
        def _timesfm_task():
            try:
                return get_timesfm_prediction(data_with_features)
            except Exception as e:
                logger.error(f"TimesFM failed: {e}")
                return {"signal": "HOLD", "confidence": 0.0, "analysis": f"TimesFM error: {e}"}

        def _tensortrade_task():
            try:
                return get_tensortrade_prediction(data_with_features)
            except Exception as e:
                logger.error(f"TensorTrade failed: {e}")
                return {"signal": "HOLD", "confidence": 0.0, "analysis": f"TensorTrade error: {e}"}

        def _grebenkov_task():
            try:
                grebenkov_data = {
                    "hist_data": data_with_features,
                    "wti_data": wti_data,
                    "nasdaq_data": nasdaq_data,
                    "ticker": self.ticker,
                }
                grebenkov_result = self.grebenkov_model.predict(grebenkov_data)
                return {
                    "signal": grebenkov_result.signal,
                    "confidence": grebenkov_result.confidence,
                    "reasoning": grebenkov_result.reasoning,
                }
            except Exception as e:
                logger.error(f"Grebenkov failed: {e}")
                return {"signal": "HOLD", "confidence": 0.0, "reasoning": f"Grebenkov error: {e}"}

        def _hmm_task():
            try:
                hmm_data = {
                    "hist_data": data_with_features,
                    "ticker": self.ticker,
                }
                hmm_result = self.hmm_model.predict(hmm_data)
                return {
                    "signal": hmm_result.signal,
                    "confidence": hmm_result.confidence,
                    "reasoning": hmm_result.reasoning,
                }
            except Exception as e:
                logger.error(f"HMM Model failed: {e}")
                return {"signal": "HOLD", "confidence": 0.0, "reasoning": f"HMM error: {e}"}

        logger.info("Lancement des tâches parallèles : news, search_query, visual_llm, 3 cpu_models")
        news_future = executor.submit(_fetch_news_task)
        search_query_future = executor.submit(_search_query_task)
        visual_llm_future = executor.submit(_visual_llm_task)
        timesfm_future = executor.submit(_timesfm_task)
        tensortrade_future = executor.submit(_tensortrade_task)
        grebenkov_future = executor.submit(_grebenkov_task)
        hmm_future = executor.submit(_hmm_task)

        # ============================================================
        # PHASE B : web_context dès que search_query est prêt
        # ============================================================
        # Timeout search_query : 240s (4 min). Cache hit returns in ~1ms.
        logger.info("Attente du search_query pour lancer le web_research...")
        try:
            search_query = search_query_future.result(timeout=240)
        except TimeoutError:
            logger.error("Search query LLM timeout (240s) — using canonical fallback")
            search_query = get_fallback_search_query(self.analysis_ticker)
        except Exception as e:
            logger.error(f"Search query future failed: {e}")
            search_query = get_fallback_search_query(self.analysis_ticker)

        logger.info("Début de la recherche Web Macro (timeout global 30s)...")
        # IMPORTANT: pas de `with` block — __exit__ appelle shutdown(wait=True)
        # qui bloquerait après le TimeoutError, défaisant le timeout lui-même.
        web_ex = ThreadPoolExecutor(max_workers=1, thread_name_prefix="web_research")
        web_future = web_ex.submit(get_web_context_sync, search_query)
        try:
            web_context = web_future.result(timeout=30)
            web_ex.shutdown(wait=True)
        except TimeoutError:
            logger.error("Web research outer timeout (30s) — using empty context")
            web_context = ""
            web_ex.shutdown(wait=False)  # orphan thread continues but doesn't block
        except Exception as e:
            logger.error(f"Web research failed: {e}")
            web_context = ""
            web_ex.shutdown(wait=False)
        logger.info("Recherche Web Macro terminée.")

        # ----------------------------------------------------------------
        # Web context summarization (optional Gemini summary tier).
        # The crawler returns raw markdown; if Gemini is enabled we replace
        # it with a tight macro summary, reducing prompt noise. On any
        # failure or if Gemini is disabled, web_context is left untouched
        # (the historical behaviour — raw markdown fed to the decision LLM).
        # ----------------------------------------------------------------
        if web_context:
            try:
                from src.gemini_gateway import GeminiGateway

                gw = GeminiGateway()
                if gw.enabled:
                    summarized = gw.summarize_web_context(web_context)
                    if summarized:
                        web_context = summarized
                        logger.info("Web context summarized via Gemini summary tier.")
            except Exception as e:
                logger.warning(f"Web summarization failed ({e}). Using raw context.")

        # ============================================================
        # PHASE C : text_llm_decision dépend de headlines + web_context
        # ============================================================
        logger.info("Attente des news pour lancer le text LLM...")
        try:
            headlines, sentiment_score = news_future.result(timeout=90)
        except TimeoutError:
            logger.error("News fetch timeout (90s) — using empty headlines")
            headlines, sentiment_score = [], 0
        except Exception as e:
            logger.error(f"News future failed: {e}")
            headlines, sentiment_score = [], 0

        sentiment_decision = get_sentiment_decision_from_score(sentiment_score)

        # Text LLM with outer timeout — _query_ollama has 600s internal timeout,
        # we cap at 240s here so the total sequential budget stays under 900s.
        # IMPORTANT: pas de `with` block — voir web_ex ci-dessus pour la raison.
        logger.info("Querying textual LLM for decision (timeout 240s, autres modèles en parallèle)...")
        text_ex = ThreadPoolExecutor(max_workers=1, thread_name_prefix="text_llm")
        text_future = text_ex.submit(
            get_llm_decision,
            latest_data,
            headlines=headlines,
            web_context=web_context,
            vg_indicators=vg_indicators,
            ticker=self.ticker,
        )
        try:
            text_llm_decision = text_future.result(timeout=240)
            text_ex.shutdown(wait=True)
        except TimeoutError:
            logger.error("Text LLM outer timeout (240s) — HOLD fallback")
            text_llm_decision = ModelResult("HOLD", 0.0, "Text LLM timeout", {"failed": True})
            text_ex.shutdown(wait=False)
        except Exception as e:
            logger.error(f"Text LLM failed: {e}")
            text_llm_decision = ModelResult("HOLD", 0.0, f"Text LLM error: {e}", {"failed": True})
            text_ex.shutdown(wait=False)

        # ============================================================
        # FINALISATION : récupération des résultats parallèles
        # Sequential critical path so far: 240 (search) + 30 (web) + 90 (news)
        # + 240 (text) = 600s. Finalisation adds at most 300s for visual_llm
        # (cpu_models are parallel and bounded at 180s each, well under 300s).
        # Total worst case: 600 + 300 = 900s ≤ CYCLE_TIMEOUT_SECONDS.
        # ============================================================
        logger.info("Attente des résultats parallèles restants (visual_llm + 3 cpu_models)...")
        try:
            visual_llm_decision = visual_llm_future.result(timeout=300)
        except TimeoutError:
            logger.error("Visual LLM timeout (300s) — HOLD fallback")
            visual_llm_decision = ModelResult("HOLD", 0.0, "Visual LLM timeout")
        except Exception as e:
            logger.error(f"Visual LLM future failed: {e}")
            visual_llm_decision = ModelResult("HOLD", 0.0, f"Visual LLM error: {e}")

        try:
            timesfm_decision = timesfm_future.result(timeout=180)
        except TimeoutError:
            logger.error("TimesFM timeout (180s) — HOLD fallback")
            timesfm_decision = ModelResult("HOLD", 0.0, "TimesFM timeout")
        except Exception as e:
            logger.error(f"TimesFM future failed: {e}")
            timesfm_decision = ModelResult("HOLD", 0.0, f"TimesFM error: {e}")

        try:
            tensortrade_decision = tensortrade_future.result(timeout=180)
        except TimeoutError:
            logger.error("TensorTrade timeout (180s) — HOLD fallback")
            tensortrade_decision = ModelResult("HOLD", 0.0, "TensorTrade timeout")
        except Exception as e:
            logger.error(f"TensorTrade future failed: {e}")
            tensortrade_decision = ModelResult("HOLD", 0.0, f"TensorTrade error: {e}")

        try:
            grebenkov_decision = grebenkov_future.result(timeout=180)
        except TimeoutError:
            logger.error("Grebenkov timeout (180s) — HOLD fallback")
            grebenkov_decision = ModelResult("HOLD", 0.0, "Grebenkov timeout")
        except Exception as e:
            logger.error(f"Grebenkov future failed: {e}")
            grebenkov_decision = ModelResult("HOLD", 0.0, f"Grebenkov error: {e}")

        try:
            hmm_decision = hmm_future.result(timeout=180)
        except TimeoutError:
            logger.error("HMM Model timeout (180s) — HOLD fallback")
            hmm_decision = ModelResult("HOLD", 0.0, "HMM timeout")
        except Exception as e:
            logger.error(f"HMM future failed: {e}")
            hmm_decision = ModelResult("HOLD", 0.0, f"HMM error: {e}")

        executor.shutdown(wait=False)

        return {
            "classic": {"prediction": classic_pred, "confidence": classic_conf},
            "text_llm": text_llm_decision,
            "visual_llm": visual_llm_decision,
            "sentiment": sentiment_decision,
            "timesfm": timesfm_decision,
            "tensortrade": tensortrade_decision,
            "grebenkov": grebenkov_decision,
            "hmm_model": hmm_decision,
        }

    def perform_enhanced_analysis(
        self,
        data_with_features,
        model_predictions,
        current_etf_price,
        vg_indicators=None,
    ):
        """Effectue l'analyse améliorée avec tous les nouveaux composants."""
        logger.info("Analyse améliorée en cours...")

        # Validate input data
        if data_with_features.empty:
            raise ValueError("Cannot perform analysis on empty data")

        # hist_data contient maintenant l'INDICE (NDX ou CL=F)
        hist_data = data_with_features[["Close", "Volume"]].dropna()

        # Check if we have sufficient data for analysis
        if hist_data.empty or len(hist_data) < 2:
            raise ValueError(f"Insufficient price data for analysis: {len(hist_data)} rows")

        # 1. Évaluation des risques (basée sur l'INDICE)
        risk_metrics = self.risk_manager.calculate_comprehensive_risk(
            price_data=hist_data["Close"], volume_data=hist_data["Volume"]
        )

        logger.info(f"Niveau de risque détecté: {risk_metrics.risk_level.name}")
        logger.info(f"Score de risque: {risk_metrics.overall_risk_score:.3f}")

        # 2. Resolve previous unresolved predictions for adaptive weights
        dates_prices = {}
        hist_close = hist_data["Close"]
        for i in range(max(0, len(hist_close) - 30), len(hist_close) - 1):
            date_str = hist_close.index[i]
            next_price = hist_close.iloc[i + 1]
            dates_prices[date_str] = (hist_close.iloc[i], next_price)
        self.weight_manager.resolve_previous_predictions(dates_prices)

        # 3. Calcul des poids adaptatifs
        returns = hist_data["Close"].pct_change().dropna()
        if len(returns) < 2:
            current_volatility = 0.15  # Default volatility
            logger.warning("Insufficient data for volatility calculation, using default")
        else:
            current_volatility = returns.std() * np.sqrt(252)

        weight_adjustment = self.weight_manager.calculate_adaptive_weights(
            market_data=hist_data["Close"], volatility=current_volatility
        )

        logger.info("Poids adaptatifs calculés:")
        for model, weight in weight_adjustment.model_weights.items():
            logger.info(f"  {model}: {weight:.3f}")

        # 4. Préparation des données de marché pour la décision
        latest_data = data_with_features.tail(1).iloc[0]
        market_data = {
            "volatility": current_volatility,
            "rsi": latest_data.get("RSI", 50),
            "macd": latest_data.get("MACD", 0),
            "bb_position": latest_data.get("BB_Position", 0.5),
        }

        # 5. Décision hybride améliorée
        # On désactive le modèle Vincent Ganne pour le Pétrole (absurde de l'utiliser sur lui-même)
        is_oil = EIAClient.is_oil_ticker(self.analysis_ticker)
        effective_vg_indicators = None if is_oil else vg_indicators

        # Inject is_oil flag into market_data for the decision engine
        market_data["is_oil"] = is_oil

        oil_bench_decision = None
        if EIAClient.is_oil_ticker(self.analysis_ticker):
            try:
                oil_model = OilBenchModel()
                oil_bench_decision = oil_model.analyze(ticker=self.analysis_ticker, headlines=None)
                logger.info(
                    f"OilBench signal: {oil_bench_decision['signal']} (conf={oil_bench_decision['confidence']:.2f})"
                )
            except Exception as e:
                logger.error(f"OilBench model failed (isolated): {e}")
                oil_bench_decision = None

        # Weekend council verdict (Niveau 3): retrieve the age-decayed stance
        # for the TRADING ticker (self.ticker, e.g. SXRV.DE) — NOT the analysis
        # ticker (^NDX). The council's VERDICT_TICKER block uses trading tickers
        # (same as the journal/DB it reads its context from).
        council_stance = get_council_ticker_stance(self.ticker)

        enhanced_decision = self.decision_engine.make_enhanced_decision(
            classic_pred=model_predictions["classic"]["prediction"],
            classic_conf=model_predictions["classic"]["confidence"],
            text_llm_decision=model_predictions["text_llm"],
            visual_llm_decision=model_predictions["visual_llm"],
            sentiment_decision=model_predictions["sentiment"],
            timesfm_decision=model_predictions["timesfm"],
            tensortrade_decision=model_predictions["tensortrade"],
            vincent_ganne_indicators=effective_vg_indicators,
            oil_bench_decision=oil_bench_decision,
            grebenkov_decision=model_predictions["grebenkov"],
            hmm_decision=model_predictions["hmm_model"],
            market_data=market_data,
            adaptive_weights=weight_adjustment.model_weights,
            council_stance=council_stance,
        )

        logger.info(f"Décision hybride: {enhanced_decision.final_signal}")
        logger.info(f"Consensus: {enhanced_decision.consensus_score:.2f}")
        logger.info(f"Confiance: {enhanced_decision.final_confidence:.2f}")

        # 6. Calcul de la taille de position optimale
        position_sizing = self.risk_manager.calculate_position_sizing(
            signal_strength=enhanced_decision.final_confidence,
            confidence=enhanced_decision.final_confidence,
            risk_metrics=risk_metrics,
            portfolio_value=self.initial_portfolio_value,
            current_price=current_etf_price,
        )

        logger.info(
            f"Taille de position recommandée: ${position_sizing.recommended_size:,.2f} (ETF: {current_etf_price:.2f})"
        )

        # 7. Vérification des overrides de risque
        risk_adjusted_signal, adjustment_reason = self.risk_manager.get_risk_adjusted_signal(
            enhanced_decision.final_signal,
            enhanced_decision.final_confidence,
            risk_metrics,
            price_data=hist_data["Close"],
            ticker=self.ticker,
        )

        if risk_adjusted_signal != enhanced_decision.final_signal:
            logger.warning(
                f"Signal ajuste par la gestion des risques: {enhanced_decision.final_signal} -> {risk_adjusted_signal}"
            )
            logger.warning(f"Raison: {adjustment_reason}")

        # 8. Enregistrement des prédictions pour l'apprentissage du poids adaptatif
        current_date = hist_data.index[-1].strftime("%Y-%m-%d")
        # The council is exempt from outcome-based performance tracking: it's a
        # weekly strategic verdict whose "correctness" can't be measured against
        # a per-cycle market direction. Its weight is fixed (see adaptive_weight_manager).
        outcome_tracked_models = {"council"}
        for dec in enhanced_decision.individual_decisions:
            if dec.model_name in outcome_tracked_models:
                continue
            self.weight_manager.record_model_prediction(
                date=current_date,
                model_name=dec.model_name,
                signal=dec.signal,
                confidence=dec.confidence,
                market_regime=risk_metrics.risk_level.name,
            )

        return {
            "enhanced_decision": enhanced_decision,
            "risk_metrics": risk_metrics,
            "weight_adjustment": weight_adjustment,
            "position_sizing": position_sizing,
            "risk_adjusted_signal": risk_adjusted_signal,
            "adjustment_reason": adjustment_reason,
            "market_data": market_data,
        }

    def run_enhanced_analysis(self, is_simulation=False):
        """
        Lance une analyse complète avec tous les composants améliorés.
        """
        logger.info(f"=== DÉMARRAGE DE L'ANALYSE {'SIMULÉE' if is_simulation else 'AMÉLIORÉE'} ===")

        try:
            # 1. Préparation des données (avec current_etf_price et vg_indicators)
            (
                data_with_features,
                hist_data,
                macro_context,
                vg_indicators,
                current_etf_price,
            ) = self.prepare_data_and_features()

            # 2. Entraînement du modèle classique
            classic_model = self.train_classic_model(data_with_features)

            # 3. Prédictions de tous les modèles

            # Récupération des données WTI et NASDAQ pour le modèle de Grebenkov
            wti_data, _ = get_etf_data(ticker="CRUDP.PA")
            nasdaq_data, _ = get_etf_data(ticker="SXRV.DE")

            model_predictions = self.get_model_predictions(
                data_with_features,
                classic_model,
                vg_indicators=vg_indicators,
                wti_data=wti_data,
                nasdaq_data=nasdaq_data,
            )

            # 4. Analyse améliorée
            analysis_results = self.perform_enhanced_analysis(
                data_with_features,
                model_predictions,
                current_etf_price,
                vg_indicators=vg_indicators,
            )

            # 5. Execute trade (Simulation or Hypothetical)
            trades = self._execute_hypothetical_trade(
                analysis_results,
                current_etf_price,
                hist_data.index[-1],
                is_simulation=is_simulation,
            )

            # 6. Mise à jour du monitoring (basée sur le rendement de l'ETF)
            performance_report = self.update_performance_monitoring(analysis_results, current_etf_price, trades)

            # 7. Affichage des résultats
            self.display_enhanced_results(analysis_results, performance_report)

            logger.info("=== ANALYSE TERMINÉE ===")

            return analysis_results, performance_report

        except Exception as e:
            logger.error(f"Erreur lors de l'analyse: {e}")
            raise

    def _execute_hypothetical_trade(self, analysis_results, current_price, analysis_date, is_simulation=False):
        """Executes a hypothetical trade based on the signal with strict simulation logic."""
        pending_transactions = []
        signal = analysis_results["risk_adjusted_signal"]
        transaction_cost_pct = 0.001  # 0.1%

        # Get current state from DB
        latest_portfolio_state = get_latest_portfolio_state(self.ticker)

        if latest_portfolio_state is None:
            # First time initialization
            current_position = 0.0
            current_cash = 1000.0 if is_simulation else self.initial_portfolio_value
            benchmark_value = current_cash
        else:
            current_position, current_cash, _, benchmark_value = latest_portfolio_state

        # Get latest transaction to enforce BUY/SELL alternation
        from database import get_latest_transaction

        last_tx = get_latest_transaction(self.ticker)
        last_type = last_tx[1] if last_tx else "SELL"  # Assume we start ready to BUY

        trades = []
        new_position = current_position
        new_cash = current_cash

        # Simulation Logic: Strict Alternation
        if "BUY" in signal and last_type == "SELL" and current_cash > 10:
            # We can BUY only if the last action was SELL
            cost_val = current_cash * transaction_cost_pct
            quantity = (current_cash - cost_val) / current_price

            pending_transactions.append((analysis_date.strftime("%Y-%m-%d %H:%M:%S"), self.ticker, "BUY", quantity, current_price, current_cash, analysis_results["enhanced_decision"].final_signal, f"Consensus Score: {analysis_results['enhanced_decision'].consensus_score:.2f}"))
            new_position = quantity
            new_cash = 0
            trades.append(
                {
                    "type": "BUY",
                    "quantity": quantity,
                    "price": current_price,
                    "cost": current_cash,
                }
            )
            logger.info(f"TRADE: Executed BUY of {quantity:.4f} shares at ${current_price:.2f}")

        elif "SELL" in signal and last_type == "BUY" and current_position > 0:
            # We can SELL only if we currently hold a position
            sell_val = current_position * current_price
            cost_val = sell_val * transaction_cost_pct
            new_cash = sell_val - cost_val

            pending_transactions.append((analysis_date.strftime("%Y-%m-%d %H:%M:%S"), self.ticker, "SELL", current_position, current_price, new_cash, analysis_results["enhanced_decision"].final_signal, f"P&L Trade: {((current_price / last_tx[3]) - 1) * 100:+.2f}%"))
            new_position = 0
            trades.append(
                {
                    "type": "SELL",
                    "quantity": current_position,
                    "price": current_price,
                    "cost": new_cash,
                }
            )
            logger.info(f"TRADE: Executed SELL at ${current_price:.2f}, New Balance: ${new_cash:.2f}")

        # Update portfolio state
        total_value = new_position * current_price + new_cash
        insert_portfolio_state(
            date=analysis_date.strftime("%Y-%m-%d %H:%M:%S"),
            ticker=self.ticker,
            position=new_position,
            cash=new_cash,
            total_value=total_value,
            benchmark_value=benchmark_value,
        )
        if pending_transactions:
            insert_transactions_batch(pending_transactions)
        return trades

    def update_performance_monitoring(self, analysis_results, current_etf_price, trades):
        """Met à jour le monitoring de performance."""
        logger.info("Mise à jour du monitoring de performance...")

        latest_portfolio_state = get_latest_portfolio_state(self.ticker)
        _, _, total_value, _ = latest_portfolio_state

        # Calculate daily return from monitoring history
        daily_return = 0.0
        try:
            conn = sqlite3.connect(self.performance_monitor.db_path)
            last_val_df = pd.read_sql_query(
                "SELECT portfolio_value FROM realtime_metrics WHERE ticker = ? ORDER BY timestamp DESC LIMIT 1",
                conn,
                params=(self.ticker,),
            )
            conn.close()
            if not last_val_df.empty:
                last_value = last_val_df.iloc[0]["portfolio_value"]
                if last_value > 0:
                    daily_return = (total_value / last_value) - 1
        except Exception as e:
            logger.warning(f"Could not calculate daily return from history: {e}")

        # Model accuracy (can't be calculated in real time without knowing the future)
        model_accuracy = {
            "classic": {
                "total_predictions": 0,
                "correct_predictions": 0,
            },  # Placeholder
            "llm_text": {"total_predictions": 0, "correct_predictions": 0},
            "llm_visual": {"total_predictions": 0, "correct_predictions": 0},
            "sentiment": {"total_predictions": 0, "correct_predictions": 0},
        }

        # Create RealTimeMetrics object from actual data
        self.performance_monitor.update_monitoring(
            portfolio_value=total_value,
            daily_return=daily_return,
            trades_data=trades,
            model_predictions=model_accuracy,
        )

        # Génération du dashboard
        self.performance_monitor.create_performance_dashboard(f"enhanced_performance_dashboard_{self.ticker}.png")

        # Génération du rapport
        performance_report = self.performance_monitor.generate_performance_report(days_back=7)

        return performance_report

    def display_enhanced_results(self, analysis_results, performance_report):
        """Affiche les résultats de l'analyse améliorée."""
        from rich.console import Console
        from rich.table import Table
        from rich.panel import Panel
        from rich.text import Text

        console = Console()

        # Table des résultats principaux
        main_table = Table(show_header=True, header_style="bold magenta")
        main_table.add_column("Métrique", style="dim", width=25)
        main_table.add_column("Valeur", justify="center")
        main_table.add_column("Détails", justify="left")

        decision = analysis_results["enhanced_decision"]
        risk = analysis_results["risk_metrics"]
        position = analysis_results["position_sizing"]

        # Décision finale
        signal_style = (
            "bold green"
            if "BUY" in decision.final_signal
            else "bold red"
            if "SELL" in decision.final_signal
            else "bold yellow"
        )

        main_table.add_row(
            "Signal Final",
            Text(decision.final_signal, style=signal_style),
            f"Confiance: {decision.final_confidence:.2%}",
        )

        # Consensus des modèles
        consensus_style = (
            "green" if decision.consensus_score > 0.7 else "yellow" if decision.consensus_score > 0.4 else "red"
        )

        main_table.add_row(
            "Consensus Modèles",
            Text(f"{decision.consensus_score:.2%}", style=consensus_style),
            f"Désaccord: {decision.disagreement_factor:.2%}",
        )

        # Niveau de risque
        risk_style = (
            "green"
            if risk.risk_level.name == "VERY_LOW"
            else "yellow"
            if risk.risk_level.name in ["LOW", "MODERATE"]
            else "red"
        )

        main_table.add_row(
            "Niveau de Risque",
            Text(risk.risk_level.name, style=risk_style),
            f"Score: {risk.overall_risk_score:.3f}",
        )

        # Position recommandée
        main_table.add_row(
            "Position Recommandée",
            f"${position.recommended_size:,.0f}",
            f"Kelly: ${position.kelly_criterion_size:,.0f}",
        )

        # Ajustement de risque
        if analysis_results["risk_adjusted_signal"] != decision.final_signal:
            main_table.add_row(
                "Ajustement Risque",
                Text(analysis_results["risk_adjusted_signal"], style="bold orange"),
                analysis_results["adjustment_reason"][:50] + "...",
            )

        # Affichage
        console.print("")
        console.print(
            Panel(
                main_table,
                title=f"[bold]Analyse de Trading Améliorée - {self.ticker}[/bold]",
                border_style="blue",
            )
        )

        # Détails des modèles individuels
        models_table = Table(show_header=True, header_style="bold cyan")
        models_table.add_column("Modèle", style="dim")
        models_table.add_column("Signal", justify="center")
        models_table.add_column("Confiance", justify="center")
        models_table.add_column("Poids Adaptatif", justify="center")

        weights = analysis_results["weight_adjustment"].model_weights

        for model_decision in decision.individual_decisions:
            model_name = model_decision.model_name
            signal_style = (
                "green" if "BUY" in model_decision.signal else "red" if "SELL" in model_decision.signal else "yellow"
            )

            models_table.add_row(
                model_name.replace("_", " ").title(),
                Text(model_decision.signal, style=signal_style),
                f"{model_decision.confidence:.2%}",
                f"{weights.get(model_name, 0):.3f}",
            )

        console.print(
            Panel(
                models_table,
                title="[bold]Détail des Modèles[/bold]",
                border_style="green",
            )
        )

        # Informations de performance si disponibles
        if not performance_report.get("error"):
            perf_summary = performance_report.get("performance_summary", {})
            console.print(
                Panel(
                    f"Retour sur 7 jours: {perf_summary.get('period_return', 0):.2%}"
                    f"Sharpe Ratio: {perf_summary.get('sharpe_ratio', 0):.2f}"
                    f"Taux de réussite: {perf_summary.get('win_rate', 0):.2%}"
                    f"Volatilité: {perf_summary.get('volatility', 0):.2%}",
                    title="[bold]Performance Récente[/bold]",
                    border_style="cyan",
                )
            )

        console.print("")


def main():
    """Fonction principale de démonstration."""
    # Configuration du logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    # Initialisation et exécution du système amélioré
    enhanced_system = EnhancedTradingSystem(ticker="QQQ")

    # Exécution de l'analyse
    results, report = enhanced_system.run_enhanced_analysis()

    print("\n" + "=" * 80)
    print("SYSTÈME DE TRADING AI AMÉLIORÉ - ANALYSE TERMINÉE")
    print("=" * 80)
    print("Graphiques générés: enhanced_trading_chart.png, enhanced_performance_dashboard.png")
    print(f"Décision finale: {results['enhanced_decision'].final_signal}")
    print(f"Niveau de risque: {results['risk_metrics'].risk_level.name}")
    print(f"Position recommandée: ${results['position_sizing'].recommended_size:,.2f}")


if __name__ == "__main__":
    main()
