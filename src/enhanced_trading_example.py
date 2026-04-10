"""
Exemple d'Intégration des Améliorations
Script de démonstration montrant comment intégrer tous les nouveaux modules
dans le système de trading AI existant.
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import subprocess
import json
import sys
import os
from dotenv import load_dotenv

# Imports des modules existants
from data import get_etf_data, fetch_macro_data_for_date
from features import create_technical_indicators, create_features, select_features
from classic_model import train_ensemble_model, get_classic_prediction
from llm_client import get_llm_decision, get_visual_llm_decision
from sentiment_analysis import get_sentiment_decision_from_score
from chart_generator import generate_chart_image

from database import init_db, insert_transaction, insert_portfolio_state, get_latest_portfolio_state, get_transactions_history
from timesfm_model import get_timesfm_prediction

# Imports des nouveaux modules d'amélioration
from enhanced_decision_engine import EnhancedDecisionEngine
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
        'SXRV.DE': '^NDX',   # iShares Nasdaq 100 -> Nasdaq 100 Index
        'SXRV.FRK': '^NDX',
        'CRUDP.PA': 'CL=F',  # Lyxor WTI Oil -> Crude Oil Futures
        'QQQ': '^NDX',
        'SPY': '^GSPC',
    }
    
    def __init__(self, 
                 ticker: str = 'QQQ',
                 initial_portfolio_value: float = 100000):
        """
        Initialise le système de trading amélioré.
        
        Args:
            ticker: Symbole de l'ETF à trader
            initial_portfolio_value: Valeur initiale du portefeuille
        """
        self.ticker = ticker  # Ticker de TRADING (ex: SXRV.DE)
        self.analysis_ticker = self.ANALYSIS_MAPPING.get(ticker, ticker) # Ticker d'ANALYSE (ex: ^NDX)
        self.initial_portfolio_value = initial_portfolio_value
        
        # Initialisation des composants améliorés
        self.decision_engine = EnhancedDecisionEngine()
        self.risk_manager = AdvancedRiskManager()
        self.weight_manager = AdaptiveWeightManager()
        self.performance_monitor = PerformanceMonitor()
        
        # Configuration du système
        self.chart_output_path = Path("enhanced_trading_chart.png")

        # Initialize database and portfolio
        init_db()
        # Pour l'initialisation du portefeuille, on utilise le ticker de trading
        self._initialize_portfolio(hist_data=get_etf_data(ticker=self.ticker)[0])
        
        logger.info(f"Système de trading amélioré initialisé. Trading: {self.ticker} | Analyse: {self.analysis_ticker}")
    
    def _initialize_portfolio(self, hist_data):
        """Initializes the portfolio if it doesn't exist."""
        if get_latest_portfolio_state(self.ticker) is None:
            logger.info(f"No existing portfolio found for {self.ticker}. Initializing a new one.")
            insert_portfolio_state(
                date=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                ticker=self.ticker,
                position=0,
                cash=self.initial_portfolio_value,
                total_value=self.initial_portfolio_value,
                benchmark_value=self.initial_portfolio_value
            )
    
    def prepare_data_and_features(self):
        """Prépare les données et indicateurs techniques basés sur l'indice d'analyse."""
        logger.info(f"Récupération et préparation des données pour l'indice {self.analysis_ticker}...")
        
        try:
            # 1. Récupération des données de l'indice pour l'analyse IA
            hist_data, info = get_etf_data(ticker=self.analysis_ticker)
            
            # 2. Récupération du prix actuel de l'ETF pour le trading
            etf_data, _ = get_etf_data(ticker=self.ticker)
            current_etf_price = etf_data['Close'].iloc[-1]
            logger.info(f"Prix actuel de l'ETF ({self.ticker}): {current_etf_price:.2f}")
            
            # Validate data
            if hist_data is None or hist_data.empty:
                raise ValueError(f"No data retrieved for analysis ticker {self.analysis_ticker}")
            
            logger.info(f"Analysis Data retrieved: {len(hist_data)} rows from {hist_data.index.min()} to {hist_data.index.max()}")
            
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
            
            # Création des features avec contexte macro
            data_with_features = create_features(data_with_indicators, macro_context)
            
            # Final validation
            if data_with_features.empty:
                raise ValueError("Feature generation failed - empty result")
            
            logger.info(f"Final data with features: {len(data_with_features)} rows, {len(data_with_features.columns)} columns")
            
            # On retourne aussi le prix de l'ETF pour les calculs de trading
            return data_with_features, hist_data, macro_context, current_etf_price
            
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
            return None, None
        
        # Check if we have enough valid data
        valid_data_count = (~y.isnull()).sum()
        if valid_data_count < 50:
            logger.warning(f"Données insuffisantes pour l'entraînement - seulement {valid_data_count} échantillons valides")
            return None, None
        
        try:
            # Entraînement du modèle
            classic_model, scaler, metrics, _ = train_ensemble_model(X, y)
            logger.info("Modèle classique entraîné avec succès")
            return classic_model, scaler
        except Exception as e:
            logger.error(f"Erreur lors de l'entraînement du modèle: {e}")
            return None, None
    
    def get_model_predictions(self, data_with_features, classic_model, scaler):
        """Obtient les prédictions de tous les modèles."""
        logger.info("Génération des prédictions des modèles...")
        
        latest_data = data_with_features.tail(1)
        
        # 1. Prédiction du modèle classique
        if classic_model is not None and scaler is not None:
            try:
                feature_cols = [col for col in scaler.feature_names_in_ 
                              if col in latest_data.columns]
                latest_features = latest_data[feature_cols]
                
                # Log feature information
                logger.info(f"Features disponibles pour prédiction: {len(feature_cols)}")
                logger.info(f"NaN dans les features: {latest_features.isnull().sum().sum()}")
                
                classic_pred, classic_conf = get_classic_prediction(
                    classic_model, scaler, latest_features)
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
            title=f"{self.ticker} - Enhanced Analysis Chart"
        )
        
        # 3. Analyse de sentiment et News (Déplacé avant le LLM pour servir de contexte)
        logger.info("Fetching live news and sentiment...")
        headlines = []
        sentiment_score = 0
        try:
            script_path = Path(__file__).parent / "news_fetcher.py"
            python_executable = sys.executable
            process = subprocess.run(
                [python_executable, str(script_path), self.ticker, ALPHA_VANTAGE_API_KEY],
                capture_output=True,
                text=True,
                check=True
            )
            news_data = json.loads(process.stdout)
            headlines = news_data.get("headlines", [])
            sentiment_score = news_data.get("sentiment", 0)
            logger.info(f"Successfully fetched {len(headlines)} news headlines. Sentiment score: {sentiment_score:.2f}")
        except Exception as e:
            logger.error(f"Failed to fetch news: {e}")

        sentiment_decision = get_sentiment_decision_from_score(sentiment_score)
        
        # 4. Prédictions LLM (Maintenant avec le contexte des news)
        text_llm_decision = get_llm_decision(latest_data, headlines=headlines)
        visual_llm_decision = get_visual_llm_decision(
            self.chart_output_path) if chart_generated else {
                "signal": "HOLD", "confidence": 0.0, 
                "analysis": "Chart generation failed."
            }
        
        # 5. Prédictions TimesFM
        logger.info("Génération de la prédiction TimesFM...")
        timesfm_decision = get_timesfm_prediction(data_with_features)

        return {
            'classic': {'prediction': classic_pred, 'confidence': classic_conf},
            'text_llm': text_llm_decision,
            'visual_llm': visual_llm_decision,
            'sentiment': sentiment_decision,
            'timesfm': timesfm_decision
        }
    
    def perform_enhanced_analysis(self, data_with_features, model_predictions, current_etf_price):
        """Effectue l'analyse améliorée avec tous les nouveaux composants."""
        logger.info("Analyse améliorée en cours...")
        
        # Validate input data
        if data_with_features.empty:
            raise ValueError("Cannot perform analysis on empty data")
        
        # hist_data contient maintenant l'INDICE (NDX ou CL=F)
        hist_data = data_with_features[['Close', 'Volume']].dropna()
        
        # Check if we have sufficient data for analysis
        if hist_data.empty or len(hist_data) < 2:
            raise ValueError(f"Insufficient price data for analysis: {len(hist_data)} rows")
        
        # 1. Évaluation des risques (basée sur l'INDICE)
        risk_metrics = self.risk_manager.calculate_comprehensive_risk(
            price_data=hist_data['Close'],
            volume_data=hist_data['Volume']
        )
        
        logger.info(f"Niveau de risque détecté: {risk_metrics.risk_level.name}")
        logger.info(f"Score de risque: {risk_metrics.overall_risk_score:.3f}")
        
        # 2. Calcul des poids adaptatifs
        returns = hist_data['Close'].pct_change().dropna()
        if len(returns) < 2:
            current_volatility = 0.15  # Default volatility
            logger.warning("Insufficient data for volatility calculation, using default")
        else:
            current_volatility = returns.std() * np.sqrt(252)
        
        weight_adjustment = self.weight_manager.calculate_adaptive_weights(
            market_data=hist_data['Close'],
            volatility=current_volatility
        )
        
        logger.info("Poids adaptatifs calculés:")
        for model, weight in weight_adjustment.model_weights.items():
            logger.info(f"  {model}: {weight:.3f}")
        
        # 3. Préparation des données de marché pour la décision
        latest_data = data_with_features.tail(1).iloc[0]
        market_data = {
            'volatility': current_volatility,
            'rsi': latest_data.get('RSI', 50),
            'macd': latest_data.get('MACD', 0),
            'bb_position': latest_data.get('BB_Position', 0.5)
        }
        
        # 4. Décision hybride améliorée
        enhanced_decision = self.decision_engine.make_enhanced_decision(
            classic_pred=model_predictions['classic']['prediction'],
            classic_conf=model_predictions['classic']['confidence'],
            text_llm_decision=model_predictions['text_llm'],
            visual_llm_decision=model_predictions['visual_llm'],
            sentiment_decision=model_predictions['sentiment'],
            timesfm_decision=model_predictions['timesfm'],
            market_data=market_data,
            adaptive_weights=weight_adjustment.model_weights
        )
        
        logger.info(f"Décision hybride: {enhanced_decision.final_signal}")
        logger.info(f"Consensus: {enhanced_decision.consensus_score:.2f}")
        logger.info(f"Confiance: {enhanced_decision.final_confidence:.2f}")
        
        # 5. Calcul de la taille de position optimale (basée sur le prix de l'ETF)
        position_sizing = self.risk_manager.calculate_position_sizing(
            signal_strength=enhanced_decision.final_confidence,
            confidence=enhanced_decision.final_confidence,
            risk_metrics=risk_metrics,
            portfolio_value=self.initial_portfolio_value,
            current_price=current_etf_price
        )
        
        logger.info(f"Taille de position recommandée: ${position_sizing.recommended_size:,.2f} (ETF: {current_etf_price:.2f})")
        
        # 6. Vérification des overrides de risque
        risk_adjusted_signal, adjustment_reason = self.risk_manager.get_risk_adjusted_signal(
            enhanced_decision.final_signal,
            enhanced_decision.final_confidence,
            risk_metrics
        )
        
        if risk_adjusted_signal != enhanced_decision.final_signal:
            logger.warning(f"Signal ajuste par la gestion des risques: "
                         f"{enhanced_decision.final_signal} -> {risk_adjusted_signal}")
            logger.warning(f"Raison: {adjustment_reason}")

        
        return {
            'enhanced_decision': enhanced_decision,
            'risk_metrics': risk_metrics,
            'weight_adjustment': weight_adjustment,
            'position_sizing': position_sizing,
            'risk_adjusted_signal': risk_adjusted_signal,
            'adjustment_reason': adjustment_reason,
            'market_data': market_data
        }
    
    def run_enhanced_analysis(self, is_simulation=False):
        """
        Lance une analyse complète avec tous les composants améliorés.
        """
        logger.info(f"=== DÉMARRAGE DE L'ANALYSE {'SIMULÉE' if is_simulation else 'AMÉLIORÉE'} ===")
        
        try:
            # 1. Préparation des données (avec current_etf_price)
            data_with_features, hist_data, macro_context, current_etf_price = self.prepare_data_and_features()
            
            # 2. Entraînement du modèle classique
            classic_model, scaler = self.train_classic_model(data_with_features)
            
            # 3. Prédictions de tous les modèles
            model_predictions = self.get_model_predictions(
                data_with_features, classic_model, scaler)
            
            # 4. Analyse améliorée
            analysis_results = self.perform_enhanced_analysis(
                data_with_features, model_predictions, current_etf_price)

            # 5. Execute trade (Simulation or Hypothetical)
            trades = self._execute_hypothetical_trade(analysis_results, current_etf_price, hist_data.index[-1], is_simulation=is_simulation)
            
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
        signal = analysis_results['risk_adjusted_signal']
        transaction_cost_pct = 0.001 # 0.1%

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
        last_type = last_tx[1] if last_tx else 'SELL' # Assume we start ready to BUY

        trades = []
        new_position = current_position
        new_cash = current_cash

        # Simulation Logic: Strict Alternation
        if "BUY" in signal and last_type == 'SELL' and current_cash > 10:
            # We can BUY only if the last action was SELL
            cost_val = current_cash * transaction_cost_pct
            quantity = (current_cash - cost_val) / current_price
            
            insert_transaction(
                date=analysis_date.strftime('%Y-%m-%d %H:%M:%S'),
                ticker=self.ticker,
                type='BUY',
                quantity=quantity,
                price=current_price,
                cost=current_cash,
                signal_source=analysis_results['enhanced_decision'].final_signal,
                reason=f"Consensus Score: {analysis_results['enhanced_decision'].consensus_score:.2f}"
            )
            new_position = quantity
            new_cash = 0
            trades.append({'type': 'BUY', 'quantity': quantity, 'price': current_price, 'cost': current_cash})
            logger.info(f"TRADE: Executed BUY of {quantity:.4f} shares at ${current_price:.2f}")

        elif "SELL" in signal and last_type == 'BUY' and current_position > 0:
            # We can SELL only if we currently hold a position
            sell_val = current_position * current_price
            cost_val = sell_val * transaction_cost_pct
            new_cash = sell_val - cost_val
            
            insert_transaction(
                date=analysis_date.strftime('%Y-%m-%d %H:%M:%S'),
                ticker=self.ticker,
                type='SELL',
                quantity=current_position,
                price=current_price,
                cost=new_cash,
                signal_source=analysis_results['enhanced_decision'].final_signal,
                reason=f"P&L Trade: {((current_price/last_tx[3])-1)*100:+.2f}%"
            )
            new_position = 0
            trades.append({'type': 'SELL', 'quantity': current_position, 'price': current_price, 'cost': new_cash})
            logger.info(f"TRADE: Executed SELL at ${current_price:.2f}, New Balance: ${new_cash:.2f}")

        # Update portfolio state
        total_value = new_position * current_price + new_cash
        insert_portfolio_state(
            date=analysis_date.strftime('%Y-%m-%d %H:%M:%S'),
            ticker=self.ticker,
            position=new_position,
            cash=new_cash,
            total_value=total_value,
            benchmark_value=benchmark_value
        )
        return trades

    def update_performance_monitoring(self, analysis_results, current_etf_price, trades):
        """Met à jour le monitoring de performance."""
        logger.info("Mise à jour du monitoring de performance...")
        
        latest_portfolio_state = get_latest_portfolio_state(self.ticker)
        _, _, total_value, _ = latest_portfolio_state

        # Note: accurate daily return for monitoring would need the ETF history
        # For simplicity, we use 0 or a placeholder if only current price is available
        daily_return = 0.0 

        # Model accuracy (can't be calculated in real time without knowing the future)
        model_accuracy = {
            'classic': {'total_predictions': 0, 'correct_predictions': 0}, # Placeholder
            'llm_text': {'total_predictions': 0, 'correct_predictions': 0},
            'llm_visual': {'total_predictions': 0, 'correct_predictions': 0},
            'sentiment': {'total_predictions': 0, 'correct_predictions': 0}
        }

        # Create RealTimeMetrics object from actual data
        self.performance_monitor.update_monitoring(
            portfolio_value=total_value,
            daily_return=daily_return,
            trades_data=trades,
            model_predictions=model_accuracy
        )
        
        # Génération du dashboard
        self.performance_monitor.create_performance_dashboard(
            "enhanced_performance_dashboard.png"
        )
        
        # Génération du rapport
        performance_report = self.performance_monitor.generate_performance_report(
            days_back=7
        )
        
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
        
        decision = analysis_results['enhanced_decision']
        risk = analysis_results['risk_metrics']
        position = analysis_results['position_sizing']
        
        # Décision finale
        signal_style = "bold green" if "BUY" in decision.final_signal else \
                      "bold red" if "SELL" in decision.final_signal else "bold yellow"
        
        main_table.add_row(
            "Signal Final",
            Text(decision.final_signal, style=signal_style),
            f"Confiance: {decision.final_confidence:.2%}"
        )
        
        # Consensus des modèles
        consensus_style = "green" if decision.consensus_score > 0.7 else \
                         "yellow" if decision.consensus_score > 0.4 else "red"
        
        main_table.add_row(
            "Consensus Modèles",
            Text(f"{decision.consensus_score:.2%}", style=consensus_style),
            f"Désaccord: {decision.disagreement_factor:.2%}"
        )
        
        # Niveau de risque
        risk_style = "green" if risk.risk_level.name == "VERY_LOW" else \
                    "yellow" if risk.risk_level.name in ["LOW", "MODERATE"] else "red"
        
        main_table.add_row(
            "Niveau de Risque",
            Text(risk.risk_level.name, style=risk_style),
            f"Score: {risk.overall_risk_score:.3f}"
        )
        
        # Position recommandée
        main_table.add_row(
            "Position Recommandée",
            f"${position.recommended_size:,.0f}",
            f"Kelly: ${position.kelly_criterion_size:,.0f}"
        )
        
        # Ajustement de risque
        if analysis_results['risk_adjusted_signal'] != decision.final_signal:
            main_table.add_row(
                "Ajustement Risque",
                Text(analysis_results['risk_adjusted_signal'], style="bold orange"),
                analysis_results['adjustment_reason'][:50] + "..."
            )
        
        # Affichage
        console.print("")
        console.print(Panel(
            main_table,
            title=f"[bold]Analyse de Trading Améliorée - {self.ticker}[/bold]",
            border_style="blue"
        ))
        
        # Détails des modèles individuels
        models_table = Table(show_header=True, header_style="bold cyan")
        models_table.add_column("Modèle", style="dim")
        models_table.add_column("Signal", justify="center")
        models_table.add_column("Confiance", justify="center")
        models_table.add_column("Poids Adaptatif", justify="center")
        
        weights = analysis_results['weight_adjustment'].model_weights
        
        for model_decision in decision.individual_decisions:
            model_name = model_decision.model_name
            signal_style = "green" if "BUY" in model_decision.signal else \
                          "red" if "SELL" in model_decision.signal else "yellow"
            
            models_table.add_row(
                model_name.replace('_', ' ').title(),
                Text(model_decision.signal, style=signal_style),
                f"{model_decision.confidence:.2%}",
                f"{weights.get(model_name, 0):.3f}"
            )
        
        console.print(Panel(
            models_table,
            title="[bold]Détail des Modèles[/bold]",
            border_style="green"
        ))
        
        # Informations de performance si disponibles
        if not performance_report.get('error'):
            perf_summary = performance_report.get('performance_summary', {})
            console.print(Panel(
                f"Retour sur 7 jours: {perf_summary.get('period_return', 0):.2%}"
                f"Sharpe Ratio: {perf_summary.get('sharpe_ratio', 0):.2f}"
                f"Taux de réussite: {perf_summary.get('win_rate', 0):.2%}"
                f"Volatilité: {perf_summary.get('volatility', 0):.2%}",
                title="[bold]Performance Récente[/bold]",
                border_style="cyan"
            ))
        
        console.print("")

def main():
    """Fonction principale de démonstration."""
    # Configuration du logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Initialisation et exécution du système amélioré
    enhanced_system = EnhancedTradingSystem(ticker='QQQ')
    
    # Exécution de l'analyse
    results, report = enhanced_system.run_enhanced_analysis()
    
    print("\n" + "="*80)
    print("SYSTÈME DE TRADING AI AMÉLIORÉ - ANALYSE TERMINÉE")
    print("="*80)
    print(f"Graphiques générés: enhanced_trading_chart.png, enhanced_performance_dashboard.png")
    print(f"Décision finale: {results['enhanced_decision'].final_signal}")
    print(f"Niveau de risque: {results['risk_metrics'].risk_level.name}")
    print(f"Position recommandée: ${results['position_sizing'].recommended_size:,.2f}")

if __name__ == "__main__":
    main()
