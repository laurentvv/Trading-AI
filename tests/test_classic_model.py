import unittest
import shutil

import numpy as np
import pandas as pd

from src.classic_model import (
    train_ensemble_model,
    retrain_if_stale,
    _MODEL_CACHE_DIR,
)


def _make_features(n=300, seed=42):
    np.random.seed(seed)
    X = pd.DataFrame({
        "feat1": np.random.randn(n),
        "feat2": np.random.randn(n),
        "feat3": np.random.randn(n),
        "feat4": np.random.randn(n),
    })
    y = pd.Series(np.where(X["feat1"] + X["feat2"] > 0, 1, 0))
    return X, y


class TestClassicModel(unittest.TestCase):
    def setUp(self):
        if _MODEL_CACHE_DIR.exists():
            shutil.rmtree(_MODEL_CACHE_DIR, ignore_errors=True)

    def tearDown(self):
        if _MODEL_CACHE_DIR.exists():
            shutil.rmtree(_MODEL_CACHE_DIR, ignore_errors=True)

    def test_walk_forward_returns_valid_tuple(self):
        X, y = _make_features(n=300)
        model, scaler, metrics, feat_imp = train_ensemble_model(X, y, walk_forward=True)
        self.assertIsNotNone(model)
        self.assertIsNotNone(scaler)
        self.assertIsInstance(metrics, dict)
        self.assertIn("f1", metrics)
        self.assertIn("accuracy", metrics)

    def test_retrain_if_stale_skips_young_model(self):
        X, y = _make_features(n=300)
        model, scaler, metrics, _ = train_ensemble_model(X, y)
        today = pd.Timestamp.now()
        out_model, out_scaler, out_date = retrain_if_stale(
            model, scaler, X, y, today, max_age_days=60
        )
        self.assertIs(out_model, model)
        self.assertIs(out_scaler, scaler)

    def test_retrain_if_stale_retrains_old_model(self):
        X, y = _make_features(n=300)
        model, scaler, metrics, _ = train_ensemble_model(X, y)
        old_date = pd.Timestamp("2020-01-01")
        out_model, out_scaler, out_date = retrain_if_stale(
            model, scaler, X, y, old_date, max_age_days=60
        )
        self.assertIsNot(out_model, model)

    def test_retrain_bypasses_cache(self):
        X, y = _make_features(n=300, seed=1)
        model1, scaler1, _, _ = train_ensemble_model(X, y, walk_forward=True)
        X2, y2 = _make_features(n=300, seed=2)
        model2, scaler2, _, _ = train_ensemble_model(X2, y2, walk_forward=True, skip_cache=True)
        self.assertIsNot(model1, model2)


if __name__ == "__main__":
    unittest.main()
