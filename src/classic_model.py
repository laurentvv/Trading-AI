import pandas as pd
import logging
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)

def train_ensemble_model(X: pd.DataFrame, y: pd.Series) -> tuple:
    """
    Entraînement d'un modèle ensemble avec validation croisée
    """
    # Division des données
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False
    )

    # Normalisation
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

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
    logger.info(f"Meilleur modèle sélectionné: {best_name}")
    best_model.fit(X_train_scaled, y_train)

    # Évaluation
    y_pred = best_model.predict(X_test_scaled)

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
    feature_importance = None
    if hasattr(best_model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': best_model.feature_importances_
        }).sort_values('importance', ascending=False)

    return best_model, scaler, metrics, feature_importance

def get_classic_prediction(model, scaler, latest_features: pd.DataFrame) -> tuple[int, float]:
    """
    Génère une prédiction à partir du modèle classique entraîné.
    """
    latest_features_scaled = scaler.transform(latest_features)

    prediction = model.predict(latest_features_scaled)[0]
    probabilities = model.predict_proba(latest_features_scaled)[0]
    confidence = max(probabilities)

    return prediction, confidence
