import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import warnings
from datetime import datetime, timedelta
import logging

warnings.filterwarnings('ignore')

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ETFTradingSystem:
    """
    Système de trading IA amélioré pour ETF NASDAQ France
    """
    
    def __init__(self, ticker='FR0011871110.PA', period='5y'):
        self.ticker = ticker
        self.period = period
        self.model = None
        self.scaler = None
        self.feature_importance = None
        self.backtest_results = None
        
    def get_etf_data(self):
        """
        Récupération améliorée des données de l'ETF avec gestion d'erreurs
        """
        try:
            etf = yf.Ticker(self.ticker)
            hist_data = etf.history(period=self.period, auto_adjust=True, prepost=True)
            
            if hist_data.empty:
                raise ValueError(f"Aucune donnée trouvée pour le ticker {self.ticker}")
            
            # Nettoyage des données
            hist_data = hist_data.dropna()
            
            # Informations sur l'ETF
            try:
                info = etf.info
            except:
                info = {"longName": "ETF NASDAQ France", "currency": "EUR"}
            
            logger.info(f"Données récupérées: {len(hist_data)} jours de cotation")
            logger.info(f"Période: {hist_data.index[0].date()} à {hist_data.index[-1].date()}")
            
            return hist_data, info
            
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des données: {e}")
            raise
    
    def create_technical_indicators(self, data):
        """
        Création d'indicateurs techniques avancés
        """
        df = data.copy()
        
        # Rendements
        df['Returns'] = df['Close'].pct_change()
        df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # Moyennes mobiles multiples
        for window in [5, 10, 20, 50, 100, 200]:
            df[f'MA_{window}'] = df['Close'].rolling(window=window).mean()
            df[f'MA_{window}_slope'] = df[f'MA_{window}'].diff()
        
        # Moyennes mobiles exponentielles
        for span in [12, 26, 50]:
            df[f'EMA_{span}'] = df['Close'].ewm(span=span).mean()
        
        # RSI amélioré
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        df['RSI_MA'] = df['RSI'].rolling(window=5).mean()
        
        # MACD complet
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        
        # Bandes de Bollinger
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (2 * bb_std)
        df['BB_Lower'] = df['BB_Middle'] - (2 * bb_std)
        df['BB_Width'] = df['BB_Upper'] - df['BB_Lower']
        df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
        
        # Stochastique
        low_14 = df['Low'].rolling(window=14).min()
        high_14 = df['High'].rolling(window=14).max()
        df['Stoch_K'] = 100 * ((df['Close'] - low_14) / (high_14 - low_14))
        df['Stoch_D'] = df['Stoch_K'].rolling(window=3).mean()
        
        # Volume indicators
        df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
        
        # Volatilité
        df['Volatility'] = df['Returns'].rolling(window=20).std()
        df['ATR'] = self._calculate_atr(df, window=14)
        
        # Support et résistance
        df['Support'] = df['Low'].rolling(window=20).min()
        df['Resistance'] = df['High'].rolling(window=20).max()
        
        return df.dropna()
    
    def _calculate_atr(self, df, window=14):
        """Calcul de l'Average True Range"""
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        return true_range.rolling(window).mean()
    
    def create_features(self, data):
        """
        Création de features avancées pour le ML
        """
        df = data.copy()
        
        # Signaux de croisement
        df['MA_Cross_5_20'] = (df['MA_5'] > df['MA_20']).astype(int)
        df['MA_Cross_20_50'] = (df['MA_20'] > df['MA_50']).astype(int)
        df['EMA_Cross_12_26'] = (df['EMA_12'] > df['EMA_26']).astype(int)
        
        # Signaux RSI
        df['RSI_Oversold'] = (df['RSI'] < 30).astype(int)
        df['RSI_Overbought'] = (df['RSI'] > 70).astype(int)
        df['RSI_Neutral'] = ((df['RSI'] >= 30) & (df['RSI'] <= 70)).astype(int)
        
        # Signaux MACD
        df['MACD_Bull'] = (df['MACD'] > df['MACD_Signal']).astype(int)
        df['MACD_Cross'] = ((df['MACD'] > df['MACD_Signal']) & 
                           (df['MACD'].shift(1) <= df['MACD_Signal'].shift(1))).astype(int)
        
        # Signaux Bollinger Bands
        df['BB_Squeeze'] = (df['BB_Width'] < df['BB_Width'].rolling(20).mean()).astype(int)
        df['BB_Lower_Touch'] = (df['Close'] <= df['BB_Lower']).astype(int)
        df['BB_Upper_Touch'] = (df['Close'] >= df['BB_Upper']).astype(int)
        
        # Signaux de momentum
        df['Price_Above_MA20'] = (df['Close'] > df['MA_20']).astype(int)
        df['Price_Above_MA50'] = (df['Close'] > df['MA_50']).astype(int)
        df['Volume_Spike'] = (df['Volume_Ratio'] > 1.5).astype(int)
        
        # Features de tendance
        df['Trend_Short'] = np.where(df['MA_5'] > df['MA_20'], 1, 
                                   np.where(df['MA_5'] < df['MA_20'], -1, 0))
        df['Trend_Long'] = np.where(df['MA_20'] > df['MA_50'], 1,
                                  np.where(df['MA_20'] < df['MA_50'], -1, 0))
        
        # Target variable améliorée (rendement futur sur plusieurs horizons)
        df['Target_1d'] = np.where(df['Returns'].shift(-1) > 0, 1, 0)
        df['Target_3d'] = np.where(df['Close'].shift(-3) > df['Close'], 1, 0)
        df['Target_5d'] = np.where(df['Close'].shift(-5) > df['Close'], 1, 0)
        
        # Target principale basée sur un seuil de rendement
        threshold = df['Returns'].std() * 0.5  # 50% de la volatilité
        df['Target'] = np.where(df['Returns'].shift(-1) > threshold, 1, 0)
        
        return df.dropna()
    
    def select_features(self, data):
        """
        Sélection intelligente des features pour le modèle
        """
        feature_columns = [
            'Returns', 'Log_Returns',
            'MA_5', 'MA_20', 'MA_50', 'MA_5_slope', 'MA_20_slope',
            'EMA_12', 'EMA_26',
            'RSI', 'RSI_MA',
            'MACD', 'MACD_Signal', 'MACD_Histogram',
            'BB_Position', 'BB_Width',
            'Stoch_K', 'Stoch_D',
            'Volume_Ratio', 'Volatility', 'ATR',
            'MA_Cross_5_20', 'MA_Cross_20_50', 'EMA_Cross_12_26',
            'RSI_Oversold', 'RSI_Overbought', 'RSI_Neutral',
            'MACD_Bull', 'MACD_Cross',
            'BB_Squeeze', 'BB_Lower_Touch', 'BB_Upper_Touch',
            'Price_Above_MA20', 'Price_Above_MA50', 'Volume_Spike',
            'Trend_Short', 'Trend_Long'
        ]
        
        available_features = [col for col in feature_columns if col in data.columns]
        
        X = data[available_features]
        y = data['Target']
        
        logger.info(f"Features sélectionnées: {len(available_features)}")
        
        return X, y, available_features
    
    def train_ensemble_model(self, X, y):
        """
        Entraînement d'un modèle ensemble avec validation croisée
        """
        # Division des données
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=False
        )
        
        # Normalisation
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Test de plusieurs modèles
        models = {
            'RandomForest': RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                class_weight='balanced'
            ),
            'GradientBoosting': GradientBoostingClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            ),
            'LogisticRegression': LogisticRegression(
                random_state=42,
                class_weight='balanced',
                max_iter=1000
            )
        }
        
        # Validation croisée et sélection du meilleur modèle
        best_score = 0
        best_model = None
        best_name = ""
        
        for name, model in models.items():
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='f1')
            mean_score = cv_scores.mean()
            
            logger.info(f"{name} - Score CV: {mean_score:.4f} (+/- {cv_scores.std() * 2:.4f})")
            
            if mean_score > best_score:
                best_score = mean_score
                best_model = model
                best_name = name
        
        # Entraînement du meilleur modèle
        self.model = best_model
        self.model.fit(X_train_scaled, y_train)
        
        # Évaluation
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        
        # Métriques détaillées
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred)
        }
        
        logger.info(f"\n=== RÉSULTATS DU MODÈLE {best_name.upper()} ===")
        for metric, value in metrics.items():
            logger.info(f"{metric.capitalize()}: {value:.4f}")
        
        # Feature importance
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
        
        return self.model, self.scaler, X_test, y_test, metrics
    
    def generate_trading_decision(self, data):
        """
        Génération de décision de trading avec analyse multi-critères
        """
        if self.model is None or self.scaler is None:
            raise ValueError("Le modèle doit être entraîné avant de générer des décisions")
        
        # Préparation des dernières données
        latest_data = data.tail(1)
        
        # Features pour la prédiction
        feature_cols = [col for col in data.columns if col in self.scaler.feature_names_in_]
        latest_features = latest_data[feature_cols].values
        latest_features_scaled = self.scaler.transform(latest_features)
        
        # Prédiction IA
        prediction = self.model.predict(latest_features_scaled)[0]
        probabilities = self.model.predict_proba(latest_features_scaled)[0]
        confidence = max(probabilities)
        
        # Analyse technique complémentaire
        current = latest_data.iloc[0]
        
        technical_signals = {
            'ma_bullish': current['MA_5'] > current['MA_20'] > current['MA_50'],
            'rsi_favorable': 30 < current['RSI'] < 70,
            'macd_bullish': current['MACD'] > current['MACD_Signal'],
            'bb_position': current['BB_Position'],
            'volume_confirmation': current['Volume_Ratio'] > 1.0,
            'trend_alignment': current['Trend_Short'] == current['Trend_Long'] == 1
        }
        
        # Score technique combiné
        technical_score = sum([
            technical_signals['ma_bullish'] * 0.25,
            technical_signals['rsi_favorable'] * 0.15,
            technical_signals['macd_bullish'] * 0.20,
            (0.3 < technical_signals['bb_position'] < 0.7) * 0.15,
            technical_signals['volume_confirmation'] * 0.15,
            technical_signals['trend_alignment'] * 0.10
        ])
        
        # Décision finale
        if prediction == 1 and confidence > 0.6 and technical_score > 0.5:
            action = "ACHAT FORT"
        elif prediction == 1 and (confidence > 0.55 or technical_score > 0.4):
            action = "ACHAT"
        elif prediction == 0 and confidence > 0.6 and technical_score < 0.3:
            action = "VENTE"
        else:
            action = "ATTENDRE"
        
        decision = {
            'action': action,
            'ia_prediction': prediction,
            'confidence': confidence,
            'technical_score': technical_score,
            'current_price': current['Close'],
            'technical_signals': technical_signals,
            'risk_level': self._assess_risk(current)
        }
        
        return decision
    
    def _assess_risk(self, current_data):
        """Évaluation du niveau de risque"""
        volatility = current_data['Volatility']
        rsi = current_data['RSI']
        bb_position = current_data['BB_Position']
        
        risk_factors = [
            volatility > current_data.get('Volatility', 0) * 1.5,  # Volatilité élevée
            rsi > 75 or rsi < 25,  # RSI extrême
            bb_position > 0.9 or bb_position < 0.1,  # Prix aux bandes
        ]
        
        risk_score = sum(risk_factors) / len(risk_factors)
        
        if risk_score > 0.6:
            return "ÉLEVÉ"
        elif risk_score > 0.3:
            return "MOYEN"
        else:
            return "FAIBLE"
    
    def backtest_strategy(self, data):
        """
        Backtest complet de la stratégie avec métriques avancées
        """
        df = data.copy()
        
        # Génération des signaux de trading
        positions = []
        current_position = 0
        entry_price = 0
        
        for i in range(len(df)):
            if i < 50:  # Skip initial period
                positions.append(0)
                continue
            
            # Simulation de la décision en temps réel
            historical_data = df.iloc[:i+1]
            
            try:
                decision = self.generate_trading_decision(historical_data)
                
                # Logique de position
                if decision['action'] in ['ACHAT', 'ACHAT FORT'] and current_position == 0:
                    current_position = 1
                    entry_price = df.iloc[i]['Close']
                elif decision['action'] == 'VENTE' and current_position == 1:
                    current_position = 0
                # Stop loss et take profit
                elif current_position == 1:
                    current_price = df.iloc[i]['Close']
                    pct_change = (current_price - entry_price) / entry_price
                    
                    if pct_change < -0.03:  # Stop loss 3%
                        current_position = 0
                    elif pct_change > 0.06:  # Take profit 6%
                        current_position = 0
                
                positions.append(current_position)
                
            except:
                positions.append(current_position)
        
        df['Position'] = positions
        
        # Calcul des rendements de la stratégie
        df['Strategy_Returns'] = df['Returns'] * df['Position'].shift(1)
        df['Cumulative_Returns'] = (1 + df['Returns']).cumprod()
        df['Cumulative_Strategy'] = (1 + df['Strategy_Returns']).cumprod()
        
        # Métriques de performance
        self.backtest_results = self._calculate_performance_metrics(df)
        
        return df, self.backtest_results
    
    def _calculate_performance_metrics(self, df):
        """Calcul des métriques de performance avancées"""
        strategy_returns = df['Strategy_Returns'].dropna()
        benchmark_returns = df['Returns']
        
        # Rendements annualisés
        strategy_annual = strategy_returns.mean() * 252
        benchmark_annual = benchmark_returns.mean() * 252
        
        # Volatilités annualisées
        strategy_vol = strategy_returns.std() * np.sqrt(252)
        benchmark_vol = benchmark_returns.std() * np.sqrt(252)
        
        # Ratios de Sharpe
        risk_free_rate = 0.02  # 2% taux sans risque
        strategy_sharpe = (strategy_annual - risk_free_rate) / strategy_vol if strategy_vol > 0 else 0
        benchmark_sharpe = (benchmark_annual - risk_free_rate) / benchmark_vol if benchmark_vol > 0 else 0
        
        # Maximum Drawdown
        cumulative = (1 + strategy_returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Win rate
        winning_trades = strategy_returns[strategy_returns > 0]
        win_rate = len(winning_trades) / len(strategy_returns[strategy_returns != 0]) if len(strategy_returns[strategy_returns != 0]) > 0 else 0
        
        # Calmar ratio
        calmar_ratio = strategy_annual / abs(max_drawdown) if max_drawdown != 0 else 0
        
        return {
            'strategy_annual_return': strategy_annual,
            'benchmark_annual_return': benchmark_annual,
            'strategy_volatility': strategy_vol,
            'strategy_sharpe': strategy_sharpe,
            'benchmark_sharpe': benchmark_sharpe,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'calmar_ratio': calmar_ratio,
            'total_trades': len(strategy_returns[strategy_returns != 0]),
            'final_portfolio_value': df['Cumulative_Strategy'].iloc[-1]
        }
    
    def plot_analysis(self, data, backtest_data):
        """Visualisations avancées"""
        fig, axes = plt.subplots(4, 2, figsize=(20, 16))
        fig.suptitle('Analyse Complète du Système de Trading IA', fontsize=16, fontweight='bold')
        
        # 1. Prix et signaux de trading
        ax1 = axes[0, 0]
        ax1.plot(data.index, data['Close'], label='Prix de clôture', alpha=0.7)
        ax1.plot(data.index, data['MA_20'], label='MA 20', alpha=0.7)
        ax1.plot(data.index, data['MA_50'], label='MA 50', alpha=0.7)
        
        # Signaux d'achat/vente
        buy_signals = backtest_data['Position'].diff() == 1
        sell_signals = backtest_data['Position'].diff() == -1
        
        ax1.scatter(backtest_data.index[buy_signals], backtest_data['Close'][buy_signals],
                   color='green', marker='^', s=100, label='Achat', zorder=5)
        ax1.scatter(backtest_data.index[sell_signals], backtest_data['Close'][sell_signals],
                   color='red', marker='v', s=100, label='Vente', zorder=5)
        
        ax1.set_title('Signaux de Trading')
        ax1.legend()
    
