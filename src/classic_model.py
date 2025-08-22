import pandas as pd
import numpy as np
import logging
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer

logger = logging.getLogger(__name__)

def train_ensemble_model(X: pd.DataFrame, y: pd.Series) -> tuple:
    """
    Trains an ensemble model with cross-validation.
    """
    # Clean data - handle NaN values
    logger.info(f"Original data shape: {X.shape}")
    logger.info(f"NaN values per column: {X.isnull().sum().sum()} total")
    
    # Remove rows where target is NaN
    valid_indices = ~y.isnull()
    X_clean = X[valid_indices].copy()
    y_clean = y[valid_indices].copy()
    
    logger.info(f"After removing NaN targets: {X_clean.shape}")
    
    # Handle NaN values in features using imputation
    imputer = SimpleImputer(strategy='median')  # Use median for robustness
    X_imputed = pd.DataFrame(
        imputer.fit_transform(X_clean),
        columns=X_clean.columns,
        index=X_clean.index
    )
    
    # Remove any remaining infinite values
    X_imputed = X_imputed.replace([np.inf, -np.inf], np.nan)
    X_imputed = X_imputed.fillna(X_imputed.median())
    
    logger.info(f"Final clean data shape: {X_imputed.shape}")
    logger.info(f"Remaining NaN values: {X_imputed.isnull().sum().sum()}")
    
    # Data splitting
    X_train, X_test, y_train, y_test = train_test_split(
        X_imputed, y_clean, test_size=0.2, random_state=42, shuffle=False
    )

    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Store imputer in scaler object for later use
    scaler.imputer = imputer

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

    # Cross-validation and best model selection
    best_score = 0
    best_model = None
    best_name = ""

    for name, model in models.items():
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='f1')
        mean_score = cv_scores.mean()

        logger.info(f"{name} - CV Score: {mean_score:.4f} (+/- {cv_scores.std() * 2:.4f})")

        if mean_score > best_score:
            best_score = mean_score
            best_model = model
            best_name = name

    # Best model training
    logger.info(f"Best model selected: {best_name}")
    best_model.fit(X_train_scaled, y_train)

    # Evaluation
    y_pred = best_model.predict(X_test_scaled)

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred)
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
    # Handle NaN values in the latest features using the same imputer
    if hasattr(scaler, 'imputer'):
        latest_features_imputed = pd.DataFrame(
            scaler.imputer.transform(latest_features),
            columns=latest_features.columns,
            index=latest_features.index
        )
    else:
        # Fallback if no imputer available
        latest_features_imputed = latest_features.fillna(latest_features.median())
    
    # Remove any infinite values
    latest_features_imputed = latest_features_imputed.replace([np.inf, -np.inf], np.nan)
    latest_features_imputed = latest_features_imputed.fillna(0)  # Final fallback
    
    # Scale the features
    latest_features_scaled = scaler.transform(latest_features_imputed)

    prediction = model.predict(latest_features_scaled)[0]
    probabilities = model.predict_proba(latest_features_scaled)[0]
    confidence = max(probabilities)

    return prediction, confidence

