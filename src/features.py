import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def _calculate_atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """Calculates the Average True Range."""
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())

    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    return true_range.rolling(window).mean()

def create_technical_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """
    Creates advanced technical indicators.
    """
    df = data.copy()

    # Returns
    df['Returns'] = df['Close'].pct_change()
    df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))

    # Multiple moving averages
    for window in [5, 10, 20, 50, 100, 200]:
        df[f'MA_{window}'] = df['Close'].rolling(window=window).mean()
        df[f'MA_{window}_slope'] = df[f'MA_{window}'].diff()

    # Exponential moving averages
    for span in [12, 26, 50]:
        df[f'EMA_{span}'] = df['Close'].ewm(span=span).mean()

    # Improved RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    df['RSI_MA'] = df['RSI'].rolling(window=5).mean()

    # Full MACD
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
    df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']

    # Bollinger Bands
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + (2 * bb_std)
    df['BB_Lower'] = df['BB_Middle'] - (2 * bb_std)
    df['BB_Width'] = df['BB_Upper'] - df['BB_Lower']
    df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])

    # Stochastic
    low_14 = df['Low'].rolling(window=14).min()
    high_14 = df['High'].rolling(window=14).max()
    df['Stoch_K'] = 100 * ((df['Close'] - low_14) / (high_14 - low_14))
    df['Stoch_D'] = df['Stoch_K'].rolling(window=3).mean()

    # Volume indicators
    df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']

    # Volatility
    df['Volatility'] = df['Returns'].rolling(window=20).std()
    df['ATR'] = _calculate_atr(df, window=14)
    if 'VIX' in df.columns:
        df['VIX_MA_20'] = df['VIX'].rolling(window=20).mean()

    # Support and resistance
    df['Support'] = df['Low'].rolling(window=20).min()
    df['Resistance'] = df['High'].rolling(window=20).max()

    return df.dropna()

def _align_macro_data(market_data: pd.DataFrame, macro_data_dict: dict) -> pd.DataFrame:
    """
    Aligns macroeconomic data with market data based on date.
    This is a simple version that uses forward-fill for monthly data.
    """
    # Create a DataFrame for macro data
    # For simplicity, we assume the keys in macro_data_dict are the column names
    # and we have a single date context (from main.py). 
    # In a full implementation, macro_data_dict would be a time series.
    # Here, we just add the latest values as static features.
    # A more robust approach would involve resampling and merging time series.
    
    # For this step, we'll add the macro data as static features to the last row
    # and then forward-fill or interpolate if needed. However, for a single prediction,
    # static features are sufficient.
    
    # Let's create a DataFrame with a single row for the last date of market data
    last_date = market_data.index[-1]
    macro_df = pd.DataFrame([macro_data_dict], index=[last_date])
    
    # Join with market data
    # This will add NaN for all other dates, which is fine for feature creation
    # as select_features will only use the last row for the final prediction.
    # For walk-forward backtest, this approach needs refinement.
    combined_df = market_data.join(macro_df, how='left')
    
    # Forward-fill macro data to propagate the latest known values
    for col in macro_data_dict.keys():
        if col in combined_df.columns:
            combined_df[col] = combined_df[col].ffill() # Updated from deprecated fillna(method='ffill')
            
    return combined_df

def create_features(data: pd.DataFrame, macro_context: dict = None) -> pd.DataFrame:
    """
    Creates advanced features for ML.
    """
    df = data.copy()
    
    # If macro context is provided, align and add it
    if macro_context and isinstance(macro_context, dict) and macro_context:
        try:
            df = _align_macro_data(df, macro_context)
            available_macro = [k for k, v in macro_context.items() if v is not None]
            logger.info(f"Added {len(available_macro)} macroeconomic features: {list(available_macro)}")
        except Exception as e:
            logger.warning(f"Failed to add macroeconomic context: {e}")
            logger.info("Continuing with technical indicators only")
    else:
        logger.info("No macroeconomic context provided - using technical indicators only")

    # Crossover signals
    df['MA_Cross_5_20'] = (df['MA_5'] > df['MA_20']).astype(int)
    df['MA_Cross_20_50'] = (df['MA_20'] > df['MA_50']).astype(int)
    df['EMA_Cross_12_26'] = (df['EMA_12'] > df['EMA_26']).astype(int)

    # RSI signals
    df['RSI_Oversold'] = (df['RSI'] < 30).astype(int)
    df['RSI_Overbought'] = (df['RSI'] > 70).astype(int)
    df['RSI_Neutral'] = ((df['RSI'] >= 30) & (df['RSI'] <= 70)).astype(int)

    # MACD signals
    df['MACD_Bull'] = (df['MACD'] > df['MACD_Signal']).astype(int)
    df['MACD_Cross'] = ((df['MACD'] > df['MACD_Signal']) &
                       (df['MACD'].shift(1) <= df['MACD_Signal'].shift(1))).astype(int)

    # Bollinger Bands signals
    df['BB_Squeeze'] = (df['BB_Width'] < df['BB_Width'].rolling(20).mean()).astype(int)
    df['BB_Lower_Touch'] = (df['Close'] <= df['BB_Lower']).astype(int)
    df['BB_Upper_Touch'] = (df['Close'] >= df['BB_Upper']).astype(int)

    # Momentum signals
    df['Price_Above_MA20'] = (df['Close'] > df['MA_20']).astype(int)
    df['Price_Above_MA50'] = (df['Close'] > df['MA_50']).astype(int)
    df['Volume_Spike'] = (df['Volume_Ratio'] > 1.5).astype(int)

    # Trend features
    df['Trend_Short'] = np.where(df['MA_5'] > df['MA_20'], 1,
                               np.where(df['MA_5'] < df['MA_20'], -1, 0))
    df['Trend_Long'] = np.where(df['MA_20'] > df['MA_50'], 1,
                              np.where(df['MA_20'] < df['MA_50'], -1, 0))

    # Improved target variable (future return over multiple horizons)
    df['Target_1d'] = np.where(df['Returns'].shift(-1) > 0, 1, 0)
    df['Target_3d'] = np.where(df['Close'].shift(-3) > df['Close'], 1, 0)
    df['Target_5d'] = np.where(df['Close'].shift(-5) > df['Close'], 1, 0)

    # Main target based on a return threshold
    returns_std = df['Returns'].std()
    if pd.isna(returns_std) or returns_std == 0:
        threshold = 0.001  # Default 0.1% threshold
        logger.warning("Using default threshold for target variable")
    else:
        threshold = returns_std * 0.5  # 50% of volatility
    
    df['Target'] = np.where(df['Returns'].shift(-1) > threshold, 1, 0)
    
    # The target for the very last row will be NaN due to shift(-1).
    # This is expected as we don't know the future return for the last data point.
    # For training the model on historical data, we should exclude this last row 
    # to avoid issues with NaN targets. The select_features function and the 
    # training logic in classic_model.py already handle dropping NaNs in 'y'.
    # For making a prediction on the last row, we will use the features from this row,
    # but its 'Target' value is irrelevant and will be ignored.
    
    # Ensure the dataframe is sorted by index (date) to maintain order
    df.sort_index(inplace=True)
    return df

def select_features(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list]:
    """
    Intelligently selects features for the model.
    """
    # Core technical indicators (always required)
    core_features = [
        'Returns', 'Log_Returns',
        'MA_5', 'MA_20', 'MA_50', 'MA_5_slope', 'MA_20_slope',
        'EMA_12', 'EMA_26',
        'RSI', 'RSI_MA',
        'MACD', 'MACD_Signal', 'MACD_Histogram',
        'BB_Position', 'BB_Width',
        'Stoch_K', 'Stoch_D',
        'Volume_Ratio', 'Volatility', 'ATR', 'VIX', 'VIX_MA_20',
        'MA_Cross_5_20', 'MA_Cross_20_50', 'EMA_Cross_12_26',
        'RSI_Oversold', 'RSI_Overbought', 'RSI_Neutral',
        'MACD_Bull', 'MACD_Cross',
        'BB_Squeeze', 'BB_Lower_Touch', 'BB_Upper_Touch',
        'Price_Above_MA20', 'Price_Above_MA50', 'Volume_Spike',
        'Trend_Short', 'Trend_Long'
    ]
    
    # Optional macroeconomic features (nice to have but not required)
    macro_features = [
        'treasury_yield_10year', 'treasury_yield_2year',
        'federal_funds_rate', 'cpi', 'unemployment', 'real_gdp'
    ]
    
    # Combine all potential features
    all_potential_features = core_features + macro_features

    # Filter to only available features
    available_features = [col for col in all_potential_features if col in data.columns]
    missing_core = [col for col in core_features if col not in data.columns]
    missing_macro = [col for col in macro_features if col not in data.columns]
    
    # Log feature availability
    logger.info(f"Core features available: {len(core_features) - len(missing_core)}/{len(core_features)}")
    if missing_core:
        logger.warning(f"Missing CORE features: {missing_core[:3]}{'...' if len(missing_core) > 3 else ''}")
    
    if missing_macro:
        logger.info(f"Missing macro features (optional): {len(missing_macro)}/{len(macro_features)}")
        logger.debug(f"Macro features not available: {missing_macro}")
    else:
        logger.info("All macroeconomic features available")

    # Check if we have minimum required features
    min_required_features = ['Returns', 'MA_20', 'RSI', 'MACD', 'BB_Position', 'Volume_Ratio']
    missing_required = [col for col in min_required_features if col not in available_features]
    
    if missing_required:
        raise ValueError(f"Critical features missing: {missing_required}. Cannot proceed with analysis.")

    # Select features and target
    X = data[available_features].copy()
    y = data['Target'].copy()
    
    # Log initial data quality
    logger.info(f"Selected {len(available_features)} features for model training")
    logger.info(f"Initial data shape: X={X.shape}, y={y.shape}")
    
    # Remove rows where target is NaN (these are typically the last few rows due to forward-looking targets)
    valid_target_mask = ~y.isnull()
    X_clean = X[valid_target_mask].copy()
    y_clean = y[valid_target_mask].copy()
    
    logger.info(f"After removing NaN targets: X={X_clean.shape}, y={y_clean.shape}")
    
    # Check for infinite values and replace them
    inf_mask = np.isinf(X_clean).any(axis=1)
    if inf_mask.sum() > 0:
        logger.warning(f"Found {inf_mask.sum()} rows with infinite values, removing them")
        X_clean = X_clean[~inf_mask]
        y_clean = y_clean[~inf_mask]
    
    # Handle remaining NaN values in features
    nan_counts = X_clean.isnull().sum()
    features_with_nans = nan_counts[nan_counts > 0]
    if len(features_with_nans) > 0:
        logger.info(f"Filling NaN values in {len(features_with_nans)} features")
        # Forward fill, then backward fill, then fill with 0
        X_clean = X_clean.ffill().bfill().fillna(0)
    
    # Final validation
    if X_clean.empty or y_clean.empty:
        raise ValueError("No valid data remaining after cleaning")
    
    logger.info(f"Final clean data: X={X_clean.shape}, y={y_clean.shape}")
    logger.info(f"Target distribution: {y_clean.value_counts().to_dict()}")
    
    return X_clean, y_clean, available_features