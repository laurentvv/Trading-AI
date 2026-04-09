import pandas as pd
import numpy as np
import logging
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)

def train_ensemble_model(X: pd.DataFrame, y: pd.Series) -> tuple:
    """
    Trains an ensemble model with time-series safe cross-validation.
    """
    # Clean data - handle NaN values safely for time series
    logger.info(f"Original data shape: {X.shape}")
    logger.info(f"NaN values per column: {X.isnull().sum().sum()} total")
    
    # Remove rows where target is NaN
    valid_indices = ~y.isnull()
    X_clean = X[valid_indices].copy()
    y_clean = y[valid_indices].copy()
    
    logger.info(f"After removing NaN targets: {X_clean.shape}")
    
    # Handle NaN values in features using time-series safe forward fill, then backward fill, then 0
    X_imputed = X_clean.ffill().bfill().fillna(0)
    
    # Remove any remaining infinite values
    X_imputed = X_imputed.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    logger.info(f"Final clean data shape: {X_imputed.shape}")
    
    # Time-Series Data splitting (preserve order)
    X_train, X_test, y_train, y_test = train_test_split(
        X_imputed, y_clean, test_size=0.2, shuffle=False
    )

    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Model testing
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

    # Time-series safe cross-validation
    tscv = TimeSeriesSplit(n_splits=5)
    best_score = 0
    best_model = None
    best_name = ""

    for name, model in models.items():
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=tscv, scoring='f1')
        mean_score = cv_scores.mean()

        logger.info(f"{name} - TimeSeries CV Score: {mean_score:.4f} (+/- {cv_scores.std() * 2:.4f})")

        if mean_score > best_score:
            best_score = mean_score
            best_model = model
            best_name = name

    # Fallback if all models score 0
    if best_model is None:
        best_name = "LogisticRegression"
        best_model = models[best_name]

    # Best model training
    logger.info(f"Best model selected: {best_name}")
    best_model.fit(X_train_scaled, y_train)

    # Evaluation
    y_pred = best_model.predict(X_test_scaled)

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0)
    }

    logger.info(f"\n=== {best_name.upper()} MODEL RESULTS ===")
    for metric, value in metrics.items():
        logger.info(f"{metric.capitalize()}: {value:.4f}")

    # Feature importance
    feature_importance = None
    if hasattr(best_model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': best_model.feature_importances_
        }).sort_values('importance', ascending=False)

    return best_model, scaler, metrics, feature_importance

def get_classic_prediction(model, scaler, latest_features: pd.DataFrame) -> tuple[int, float]:
    """
    Generates a prediction from the trained classic model.
    """
    # Safe imputation
    latest_features_imputed = latest_features.ffill().bfill().fillna(0)
    latest_features_imputed = latest_features_imputed.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Scale the features
    latest_features_scaled = scaler.transform(latest_features_imputed)

    prediction = model.predict(latest_features_scaled)[0]
    probabilities = model.predict_proba(latest_features_scaled)[0]
    confidence = max(probabilities)

    return prediction, confidence

