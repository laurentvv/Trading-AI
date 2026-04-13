import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import argparse
import logging
import sys
from pathlib import Path
from tqdm import tqdm

# Add src to path exactly like main.py
sys.path.append(str(Path(__file__).parent / 'src'))

from data import get_etf_data
from features import create_technical_indicators, create_features, select_features
from classic_model import train_ensemble_model, get_classic_prediction
from timesfm_model import TimesFMModel
from llm_client import get_llm_decision, get_visual_llm_decision
from enhanced_decision_engine import EnhancedDecisionEngine
from advanced_risk_manager import AdvancedRiskManager
from chart_generator import generate_chart_image

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def check_setup() -> bool:
    """Vérifie si TimesFM 2.5 est correctement installé (logique identique à main.py)"""
    vendor_path = Path(__file__).parent / "vendor" / "timesfm"
    if not vendor_path.exists():
        logger.error("❌ TimesFM 2.5 n'est pas installé dans /vendor. Lancez 'uv run setup'.")
        return False
    return True

class ClassicModelWrapper:
    """Wrapper pour le modèle classique afin d'avoir une interface cohérente."""
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_cols = []

    def train_and_predict(self, df: pd.DataFrame) -> tuple[int, float, str]:
        try:
            # 1. Préparation des indicateurs et features
            df_indicators = create_technical_indicators(df)
            df_features = create_features(df_indicators)
            X, y, feature_cols = select_features(df_features)
            
            # 2. Entraînement (on ré-entraîne régulièrement dans le backtest pour simuler la réalité)
            # Pour la performance, on pourrait ne ré-entraîner que tous les N jours, 
            # mais ici on fait simple et robuste.
            self.model, self.scaler, metrics, _ = train_ensemble_model(X, y)
            self.feature_cols = feature_cols
            
            # 3. Prédiction sur la dernière ligne
            latest_data = df_features.tail(1)
            # S'assurer que les colonnes correspondent exactement à celles de l'entraînement
            X_latest = latest_data[self.feature_cols]
            
            pred, conf = get_classic_prediction(self.model, self.scaler, X_latest)
            signal = "BUY" if pred == 1 else "SELL"
            
            return signal, conf, f"Classic model prediction based on {len(X)} samples."
        except Exception as e:
            logger.warning(f"Error in ClassicModel training/prediction: {e}")
            return "HOLD", 0.5, f"Error: {e}"

class Backtester:
    def __init__(self, ticker_etf="SXRV.DE", ticker_index="^NDX", initial_capital=1000.0, fast_mode=False, use_visual=False):
        self.ticker_etf = ticker_etf
        self.ticker_index = ticker_index
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.position = 0.0
        self.history = []
        self.fast_mode = fast_mode
        self.use_visual = use_visual
        
        # Initialisation des modèles
        self.classic_model = ClassicModelWrapper()
        self.timesfm = TimesFMModel.get_instance()  # Use singleton to avoid loading model twice
        self.decision_engine = EnhancedDecisionEngine()
        self.risk_manager = AdvancedRiskManager()
        
        # Temp path for charts if needed
        self.temp_chart_path = Path("temp_backtest_chart.png")

    def run(self, start_date, end_date):
        logger.info(f"🚀 Démarrage du Backtest: {start_date} -> {end_date}")
        if not self.fast_mode:
            logger.info("⚠️ MODE COMPLET: Appels LLM activés. Cela sera LENT (environ 10-20s par jour).")
            if self.use_visual:
                logger.info("📸 Analyse VISUELLE activée (+ temps de génération graphique).")
        
        # 1. Récupération des données
        df_index, _ = get_etf_data(self.ticker_index)
        df_etf, _ = get_etf_data(self.ticker_etf)
        
        # S'assurer que les index sont bien des datetime
        df_index.index = pd.to_datetime(df_index.index)
        df_etf.index = pd.to_datetime(df_etf.index)
        
        # Filtrage sur la période demandée
        start_ts = pd.to_datetime(start_date)
        end_ts = pd.to_datetime(end_date)
        test_dates = df_index.loc[start_ts:end_ts].index
        
        if len(test_dates) == 0:
            logger.error("❌ Aucune donnée trouvée pour cette période.")
            return

        logger.info(f"📅 Nombre de jours à tester: {len(test_dates)}")
        
        for current_date in tqdm(test_dates, desc="Backtesting"):
            # Slicing "Point-in-time" (on ne voit pas le futur)
            hist_index = df_index.loc[:current_date]
            hist_etf = df_etf.loc[:current_date]
            
            if len(hist_etf) == 0:
                continue
                
            current_price_etf = hist_etf['Close'].iloc[-1]
            
            # --- 2. Génération des Signaux ---
            # Modèle Classique
            classic_signal, classic_conf, _ = self.classic_model.train_and_predict(hist_index)
            classic_pred = 1 if classic_signal == "BUY" else 0
            
            # TimesFM
            tfm_decision = self.timesfm.predict(hist_index)
            
            # LLM
            if self.fast_mode:
                text_llm_decision = {"signal": "HOLD", "confidence": 0.0, "analysis": "Fast mode enabled."}
                visual_llm_decision = {"signal": "HOLD", "confidence": 0.0, "analysis": "Fast mode enabled."}
            else:
                # Appels réels à Ollama
                try:
                    # On doit préparer les features complètes (y compris Trend_Short/Long) pour le prompt
                    df_ind = create_technical_indicators(hist_index)
                    df_feat = create_features(df_ind)
                    latest_row = df_feat.tail(1)
                    text_llm_decision = get_llm_decision(latest_row)
                except Exception as e:
                    logger.warning(f"Erreur LLM Texte au {current_date}: {e}")
                    text_llm_decision = {"signal": "HOLD", "confidence": 0.0, "analysis": f"Error: {e}"}
                
                if self.use_visual:
                    try:
                        # On réutilise df_ind pour le graphique
                        generate_chart_image(df_ind, self.temp_chart_path, title=f"Backtest {current_date}")
                        visual_llm_decision = get_visual_llm_decision(self.temp_chart_path)
                    except Exception as e:
                        logger.warning(f"Erreur LLM Visuel au {current_date}: {e}")
                        visual_llm_decision = {"signal": "HOLD", "confidence": 0.0, "analysis": f"Error: {e}"}
                else:
                    visual_llm_decision = {"signal": "HOLD", "confidence": 0.0, "analysis": "Visual disabled."}

            # --- 3. Fusion de la décision ---
            # Préparation des données de marché pour l'engine
            returns = hist_index['Close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252) if len(returns) > 2 else 0.2
            
            # Correction: on calcule les indicateurs techniques requis par l'engine
            df_ind = create_technical_indicators(hist_index)
            latest_ind = df_ind.iloc[-1]
            
            market_data = {
                'volatility': volatility,
                'rsi': latest_ind.get('RSI', 50),
                'macd': latest_ind.get('MACD', 0),
                'bb_position': latest_ind.get('BB_Position', 0.5)
            }
            
            hybrid_decision = self.decision_engine.make_enhanced_decision(
                classic_pred=classic_pred,
                classic_conf=classic_conf,
                text_llm_decision=text_llm_decision,
                visual_llm_decision=visual_llm_decision,
                sentiment_decision={"signal": "HOLD", "confidence": 0.0}, # On simule sentiment neutre en backtest
                timesfm_decision=tfm_decision,
                market_data=market_data
            )
            
            # --- 4. Gestion des Risques & Sizing ---
            risk_metrics = self.risk_manager.calculate_comprehensive_risk(
                price_data=hist_index['Close'],
                volume_data=hist_index['Volume']
            )
            
            final_signal, adjustment_reason = self.risk_manager.get_risk_adjusted_signal(
                hybrid_decision.final_signal,
                hybrid_decision.final_confidence,
                risk_metrics
            )
            
            # Logging des décisions (demandé par l'utilisateur)
            logger.info(f"\n📅 {current_date.date()} | Price: {current_price_etf:.2f}€")
            for d in hybrid_decision.individual_decisions:
                logger.info(f"  ├─ {d.model_name:10}: {d.signal:10} (Conf: {d.confidence:.2f})")
            logger.info(f"  └─ FINAL: {final_signal:10} (Conf: {hybrid_decision.final_confidence:.2f})")
            if adjustment_reason:
                logger.info(f"     Risk Adj: {adjustment_reason}")
            
            # --- 5. Exécution Virtuelle ---
            self.execute_trade(current_date, final_signal, current_price_etf, hybrid_decision)

        self.generate_report(df_etf.loc[start_ts:end_ts])
        
        # Cleanup
        if self.temp_chart_path.exists():
            self.temp_chart_path.unlink()

    def execute_trade(self, date, signal, price, hybrid_decision):
        old_cash = self.cash
        old_pos = self.position
        
        # Stratégie simple : Tout ou rien pour le backtest (ou sizing basé sur confidence)
        if "BUY" in signal and self.cash > 1:
            qty = self.cash / price
            self.position += qty
            self.cash = 0
            trade_type = "BUY"
        elif "SELL" in signal and self.position > 0:
            self.cash = self.position * price
            self.position = 0
            trade_type = "SELL"
        else:
            trade_type = "HOLD"

        portfolio_value = self.cash + (self.position * price)
        
        self.history.append({
            "Date": date,
            "Type": trade_type,
            "Price": price,
            "Qty": self.position if trade_type == "BUY" else (old_pos if trade_type == "SELL" else 0),
            "Cash": self.cash,
            "Portfolio_Value": portfolio_value,
            "Decision": signal,
            "Conf": hybrid_decision.final_confidence,
            "Consensus": hybrid_decision.consensus_score
        })

    def generate_report(self, df_etf_period):
        if not self.history:
            logger.error("❌ Aucun historique généré.")
            return

        res = pd.DataFrame(self.history).set_index("Date")
        
        # Comparaison Buy & Hold
        start_price = df_etf_period['Close'].iloc[0]
        end_price = df_etf_period['Close'].iloc[-1]
        bh_final = (self.initial_capital / start_price) * end_price
        
        final_value = res['Portfolio_Value'].iloc[-1]
        total_return = (final_value / self.initial_capital - 1) * 100
        bh_return = (bh_final / self.initial_capital - 1) * 100
        
        # Métriques
        days = (res.index[-1] - res.index[0]).days
        ann_return = ((final_value / self.initial_capital) ** (365.25 / (days if days > 0 else 1)) - 1) * 100
        
        # Drawdown
        res['Peak'] = res['Portfolio_Value'].cummax()
        res['Drawdown'] = (res['Portfolio_Value'] - res['Peak']) / res['Peak']
        max_dd = res['Drawdown'].min() * 100
        
        logger.info("\n" + "="*40)
        logger.info("📊 RAPPORT DE PERFORMANCE (BACKTEST)")
        logger.info("="*40)
        logger.info(f"Période:          {res.index[0].date()} -> {res.index[-1].date()}")
        logger.info(f"Capital Initial:  {self.initial_capital:.2f}€")
        logger.info(f"Valeur Finale:    {final_value:.2f}€")
        logger.info(f"Rendement Total:  {total_return:.2f}%")
        logger.info(f"Rendement Annuel: {ann_return:.2f}%")
        logger.info(f"Buy & Hold:       {bh_return:.2f}%")
        logger.info(f"Max Drawdown:     {max_dd:.2f}%")
        logger.info("="*40)
        
        # Sauvegarde CSV
        res.to_csv("backtest_results.csv")
        logger.info("💾 Journal sauvegardé: backtest_results.csv")
        
        # Graphiques
        try:
            plt.figure(figsize=(12, 6))
            plt.plot(res.index, res['Portfolio_Value'], label='Stratégie IA Hybride', color='blue', linewidth=2)
            
            # Re-aligner Buy & Hold sur les dates du backtest
            bh_line = (self.initial_capital / start_price) * df_etf_period['Close']
            plt.plot(bh_line.index, bh_line, label='Buy & Hold (ETF)', color='gray', linestyle='--', alpha=0.7)
            
            plt.title(f"Performance Backtest: {self.ticker_etf} vs Buy & Hold", fontsize=14)
            plt.xlabel("Date")
            plt.ylabel("Valeur du Portefeuille (€)")
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig("backtest_performance.png")
            logger.info("📈 Graphique sauvegardé: backtest_performance.png")
        except Exception as e:
            logger.error(f"❌ Erreur lors de la génération du graphique: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", type=str, default="SXRV.DE", help="Ticker de l'ETF à tester")
    parser.add_argument("--index", type=str, default="^NDX", help="Ticker de l'indice pour l'analyse")
    parser.add_argument("--years", type=int, default=5, help="Nombre d'années de backtest")
    parser.add_argument("--fast", action="store_true", help="Mode rapide (simule le LLM)")
    parser.add_argument("--visual", action="store_true", help="Activer l'analyse visuelle (très lent)")
    args = parser.parse_args()
    
    # Période de backtest
    end = datetime.now().strftime('%Y-%m-%d')
    start = (datetime.now() - timedelta(days=365*args.years)).strftime('%Y-%m-%d')
    
    bt = Backtester(
        ticker_etf=args.ticker, 
        ticker_index=args.index, 
        fast_mode=args.fast,
        use_visual=args.visual
    )
    bt.run(start, end)
