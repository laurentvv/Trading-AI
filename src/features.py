import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def _calculate_atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """Calcul de l'Average True Range"""
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())

    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    return true_range.rolling(window).mean()

def create_technical_indicators(data: pd.DataFrame) -> pd.DataFrame:
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
    df['ATR'] = _calculate_atr(df, window=14)

    # Support et résistance
    df['Support'] = df['Low'].rolling(window=20).min()
    df['Resistance'] = df['High'].rolling(window=20).max()

    return df.dropna()

def create_features(data: pd.DataFrame) -> pd.DataFrame:
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

def select_features(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list]:
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
