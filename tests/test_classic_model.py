import unittest
import shutil

import numpy as np
import pandas as pd

from src.classic_model import (
    train_ensemble_model,
    retrain_if_stale,
    get_classic_prediction,
    CLASSIC_HOLD_MARGIN,
    _MODEL_CACHE_DIR,
)


def _make_features(n=300, seed=42):
    np.random.seed(seed)
    X = pd.DataFrame(
        {
            "feat1": np.random.randn(n),
            "feat2": np.random.randn(n),
            "feat3": np.random.randn(n),
            "feat4": np.random.randn(n),
        }
    )
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
        pipeline, metrics, feat_imp = train_ensemble_model(X, y, walk_forward=True)
        self.assertIsNotNone(pipeline)
        self.assertTrue(hasattr(pipeline, "named_steps"))
        self.assertIsInstance(metrics, dict)
        self.assertIn("f1", metrics)
        self.assertIn("accuracy", metrics)

    def test_retrain_if_stale_skips_young_model(self):
        X, y = _make_features(n=300)
        pipeline, metrics, _ = train_ensemble_model(X, y)
        today = pd.Timestamp.now()
        out_pipeline, out_date = retrain_if_stale(pipeline, X, y, today, max_age_days=60)
        self.assertIs(out_pipeline, pipeline)

    def test_retrain_if_stale_retrains_old_model(self):
        X, y = _make_features(n=300)
        pipeline, metrics, _ = train_ensemble_model(X, y)
        old_date = pd.Timestamp("2020-01-01")
        out_pipeline, out_date = retrain_if_stale(pipeline, X, y, old_date, max_age_days=60)
        self.assertIsNot(out_pipeline, pipeline)

    def test_retrain_bypasses_cache(self):
        X, y = _make_features(n=300, seed=1)
        pipeline1, _, _ = train_ensemble_model(X, y, walk_forward=True)
        X2, y2 = _make_features(n=300, seed=2)
        pipeline2, _, _ = train_ensemble_model(X2, y2, walk_forward=True, skip_cache=True)
        self.assertIsNot(pipeline1, pipeline2)

    def test_buy_signal_reachable_on_bullish_features(self):
        """Regression guard (July 2026, ADR-002 over-correction): classic emitted
        0 BUY over 30 PROD cycles because (a) isotonic calibration flattened
        proba toward 0.5 and (b) the HOLD_MARGIN dead-band demanded proba >= 0.58.
        With sigmoid calibration + margin 0.04, a strongly bullish feature set
        MUST yield prediction_int == 1 (BUY). This test would have caught the bug.
        """
        # Strong, learnable signal: y = 1 iff feat1 >> 0 (clear bullish proxy).
        n = 600
        rng = np.random.RandomState(7)
        X = pd.DataFrame({"feat1": rng.randn(n), "feat2": rng.randn(n)})
        y = pd.Series(np.where(X["feat1"] > 0.8, 1, 0))  # decisive threshold

        pipeline, _metrics, _ = train_ensemble_model(X, y, walk_forward=True, skip_cache=True)

        # Bullish feature row (feat1 deep in the positive class region).
        bull = pd.DataFrame({"feat1": [3.0], "feat2": [0.0]})
        pred_int, conf = get_classic_prediction(pipeline, bull)
        self.assertEqual(
            pred_int, 1,
            f"Bullish features must yield BUY (1); got {pred_int} (conf={conf:.3f}). "
            f"The dead-band (margin={CLASSIC_HOLD_MARGIN}) or calibration is again "
            f"suppressing BUY — re-check ADR-002 over-correction fix.",
        )

    def test_sell_signal_reachable_on_bearish_features(self):
        """Symmetric counterpart: strongly bearish features must yield SELL (0)."""
        n = 600
        rng = np.random.RandomState(7)
        X = pd.DataFrame({"feat1": rng.randn(n), "feat2": rng.randn(n)})
        y = pd.Series(np.where(X["feat1"] > 0.8, 1, 0))

        pipeline, _metrics, _ = train_ensemble_model(X, y, walk_forward=True, skip_cache=True)

        bear = pd.DataFrame({"feat1": [-3.0], "feat2": [0.0]})
        pred_int, conf = get_classic_prediction(pipeline, bear)
        self.assertEqual(
            pred_int, 0,
            f"Bearish features must yield SELL (0); got {pred_int} (conf={conf:.3f}).",
        )


if __name__ == "__main__":
    unittest.main()
