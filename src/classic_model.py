import pandas as pd
import numpy as np
import logging
import hashlib
import pickle
from datetime import datetime
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)

_MODEL_CACHE_DIR = Path("data_cache") / "models"
_MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _data_hash(X: pd.DataFrame, y: pd.Series) -> str:
    h = hashlib.md5()
    h.update(X.values.tobytes())
    h.update(y.values.tobytes())
    return h.hexdigest()


def _cache_path(data_hash: str) -> Path:
    return _MODEL_CACHE_DIR / f"classic_model_{data_hash}.pkl"


def _load_cached_model(cache_hash: str):
    path = _cache_path(cache_hash)
    if path.exists():
        try:
            with open(path, "rb") as f:
                cached = pickle.load(f)
            logger.info(f"Modele classique charge depuis le cache (hash={cache_hash[:12]}...)")
            return (
                cached["model"],
                cached["scaler"],
                cached["metrics"],
                cached.get("feature_importance"),
            )
        except Exception as e:
            logger.warning(f"Erreur lecture cache modele: {e}")
    return None


def _save_model_cache(cache_hash: str, model, scaler, metrics, feature_importance, train_date: str = None):
    path = _cache_path(cache_hash)
    try:
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "model": model,
                    "scaler": scaler,
                    "metrics": metrics,
                    "feature_importance": feature_importance,
                    "train_date": train_date or datetime.now().isoformat(),
                },
                f,
            )
        logger.info(f"Modele classique sauvegarde dans le cache (hash={cache_hash[:12]}...)")
    except Exception as e:
        logger.warning(f"Erreur sauvegarde cache modele: {e}")


def _build_model_candidates() -> dict:
    """Return a dict of candidate classifiers for ensemble selection."""
    return {
        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            class_weight="balanced",
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42
        ),
        "LogisticRegression": LogisticRegression(random_state=42, class_weight="balanced", max_iter=1000),
    }


def train_ensemble_model(X: pd.DataFrame, y: pd.Series, walk_forward: bool = False, skip_cache: bool = False) -> tuple:
    """Train an ensemble model by selecting the best classifier via time-series CV.

    Args:
        X: Feature DataFrame.
        y: Target Series.
        walk_forward: If True, use walk-forward validation with a held-out test set.
        skip_cache: If True, bypass the model cache and force retraining.

    Returns:
        Tuple of (best_model, scaler, metrics_dict, feature_importance_df_or_None).
    """
    cache_hash = _data_hash(X, y)
    if not skip_cache:
        cached = _load_cached_model(cache_hash)
        if cached is not None:
            return cached

    logger.info(f"Original data shape: {X.shape}")
    logger.info(f"NaN values per column: {X.isnull().sum().sum()} total")

    valid_indices = ~y.isnull()
    X_clean = X[valid_indices].copy()
    y_clean = y[valid_indices].copy()

    logger.info(f"After removing NaN targets: {X_clean.shape}")

    X_imputed = X_clean.ffill().bfill().fillna(0)
    X_imputed = X_imputed.replace([np.inf, -np.inf], np.nan).fillna(0)

    logger.info(f"Final clean data shape: {X_imputed.shape}")

    models = _build_model_candidates()

    if walk_forward:
        best_score = 0
        best_model = None
        best_name = ""

        tscv = TimeSeriesSplit(n_splits=5)
        for name, model in models.items():
            pipe = Pipeline([("scaler", StandardScaler()), ("model", model)])
            cv_scores = cross_val_score(pipe, X_imputed, y_clean.values, cv=tscv, scoring="f1")
            mean_score = cv_scores.mean()
            logger.info(f"{name} - WalkForward CV Score: {mean_score:.4f} (+/- {cv_scores.std() * 2:.4f})")
            if mean_score > best_score:
                best_score = mean_score
                best_model = model
                best_name = name

        if best_model is None:
            best_name = "LogisticRegression"
            best_model = models[best_name]

        logger.info(f"Best model selected (walk-forward): {best_name}")

        holdout = max(1, int(len(X_imputed) * 0.2))
        X_train_wf = X_imputed.iloc[:-holdout]
        y_train_wf = y_clean.iloc[:-holdout]
        X_test_wf = X_imputed.iloc[-holdout:]
        y_test_wf = y_clean.iloc[-holdout:]

        scaler_wf = StandardScaler()
        X_train_scaled_wf = scaler_wf.fit_transform(X_train_wf)
        X_test_scaled_wf = scaler_wf.transform(X_test_wf)

        best_model.fit(X_train_scaled_wf, y_train_wf.values)
        y_pred = best_model.predict(X_test_scaled_wf)
        best_metrics = {
            "accuracy": accuracy_score(y_test_wf, y_pred),
            "precision": precision_score(y_test_wf, y_pred, zero_division=0),
            "recall": recall_score(y_test_wf, y_pred, zero_division=0),
            "f1": f1_score(y_test_wf, y_pred, zero_division=0),
        }

        logger.info(f"\n=== {best_name.upper()} MODEL RESULTS (Walk-Forward, held-out {holdout} rows) ===")
        for metric, value in best_metrics.items():
            logger.info(f"{metric.capitalize()}: {value:.4f}")

        feature_importance = None
        if hasattr(best_model, "feature_importances_"):
            feature_importance = pd.DataFrame(
                {"feature": X.columns, "importance": best_model.feature_importances_}
            ).sort_values("importance", ascending=False)

        _save_model_cache(cache_hash, best_model, scaler_wf, best_metrics, feature_importance)
        return best_model, scaler_wf, best_metrics, feature_importance

    X_train, X_test, y_train, y_test = train_test_split(X_imputed, y_clean, test_size=0.2, shuffle=False)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    tscv = TimeSeriesSplit(n_splits=5)
    best_score = 0
    best_model = None
    best_name = ""

    for name, model in models.items():
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=tscv, scoring="f1")
        mean_score = cv_scores.mean()
        logger.info(f"{name} - TimeSeries CV Score: {mean_score:.4f} (+/- {cv_scores.std() * 2:.4f})")
        if mean_score > best_score:
            best_score = mean_score
            best_model = model
            best_name = name

    if best_model is None:
        best_name = "LogisticRegression"
        best_model = models[best_name]

    logger.info(f"Best model selected: {best_name}")
    best_model.fit(X_train_scaled, y_train)

    y_pred = best_model.predict(X_test_scaled)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
    }

    logger.info(f"\n=== {best_name.upper()} MODEL RESULTS ===")
    for metric, value in metrics.items():
        logger.info(f"{metric.capitalize()}: {value:.4f}")

    feature_importance = None
    if hasattr(best_model, "feature_importances_"):
        feature_importance = pd.DataFrame(
            {"feature": X.columns, "importance": best_model.feature_importances_}
        ).sort_values("importance", ascending=False)

    _save_model_cache(cache_hash, best_model, scaler, metrics, feature_importance)
    return best_model, scaler, metrics, feature_importance


def retrain_if_stale(
    model, scaler, X_recent: pd.DataFrame, y_recent: pd.Series, last_train_date, max_age_days: int = 60
) -> tuple:
    """Retrain the ensemble model if it is older than *max_age_days*.

    Args:
        model: The currently loaded model.
        scaler: The currently fitted scaler.
        X_recent: Recent feature data for retraining.
        y_recent: Recent target data for retraining.
        last_train_date: Date string or Timestamp of the last training.
        max_age_days: Maximum age in days before triggering a retrain.

    Returns:
        Tuple of (model, scaler, train_date). May be the original values if not stale.
    """
    if isinstance(last_train_date, str):
        last_train_date = pd.Timestamp(last_train_date)
    elif not isinstance(last_train_date, pd.Timestamp):
        last_train_date = pd.Timestamp.now()

    age_days = (pd.Timestamp.now() - last_train_date).days
    if age_days < max_age_days:
        return model, scaler, last_train_date

    logger.info(f"Classic ML model stale ({age_days} days old). Retraining with {len(X_recent)} rows...")
    new_model, new_scaler, metrics, _ = train_ensemble_model(X_recent, y_recent, walk_forward=True, skip_cache=True)
    logger.info(f"Retrained. F1={metrics.get('f1', 0):.3f}")
    return new_model, new_scaler, pd.Timestamp.now()


def get_classic_prediction(model, scaler, latest_features: pd.DataFrame) -> tuple[int, float]:
    latest_features_imputed = latest_features.ffill().bfill().fillna(0)
    latest_features_imputed = latest_features_imputed.replace([np.inf, -np.inf], np.nan).fillna(0)

    latest_features_scaled = scaler.transform(latest_features_imputed)

    prediction = model.predict(latest_features_scaled)[0]
    probabilities = model.predict_proba(latest_features_scaled)[0]
    confidence = max(probabilities)

    return prediction, confidence
