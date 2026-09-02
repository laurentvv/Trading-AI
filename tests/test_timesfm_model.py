import unittest
from dataclasses import dataclass
from unittest.mock import patch, MagicMock
import numpy as np
import pandas as pd

from src.timesfm_model import get_timesfm_prediction, TimesFMModel, TIMESFM_CONTEXT
from src.enhanced_decision_engine import ModelResult


@dataclass
class FakeForecastOutput:
    """Imite timesfm3.ForecastOutput (API 3.0 : attributs .forecast/.quantiles)."""
    forecast: np.ndarray
    quantiles: np.ndarray = None


class TestTimesFMModelWrapper(unittest.TestCase):
    def setUp(self):
        self.dummy_df = pd.DataFrame({"Close": [100, 105, 110]})

    @patch("src.timesfm_model.TimesFMModel")
    def test_get_timesfm_prediction_success(self, mock_timesfm_model_class):
        mock_instance = MagicMock()
        mock_prediction = ModelResult(
            signal="BUY",
            confidence=0.8,
            reasoning="Mock analysis",
            metadata={"predictions": [115.0]}
        )
        mock_instance.predict.return_value = mock_prediction
        mock_timesfm_model_class.get_instance.return_value = mock_instance

        result = get_timesfm_prediction(self.dummy_df)

        mock_timesfm_model_class.get_instance.assert_called_once()
        mock_instance.predict.assert_called_once()
        args, kwargs = mock_instance.predict.call_args
        self.assertEqual(args[0]["ticker"], "default")
        self.assertEqual(result, mock_prediction)

    @patch("src.timesfm_model.TimesFMModel")
    def test_get_timesfm_prediction_exception(self, mock_timesfm_model_class):
        mock_timesfm_model_class.get_instance.side_effect = Exception("Mocked initialization error")

        result = get_timesfm_prediction(self.dummy_df)

        self.assertEqual(result.signal, "HOLD")
        self.assertEqual(result.confidence, 0.0)
        self.assertTrue("Model error" in result.reasoning)


class TestPositionTracking(unittest.TestCase):
    def _make_model(self, fake_model):
        model = TimesFMModel.__new__(TimesFMModel)
        model.initialized = True
        model.model = fake_model
        model.vol_multiplier = 0.5
        model._positions = {}
        model._get_position = lambda t: model._positions.get(t, "FLAT")
        model._adaptive_threshold = lambda p: 0.005
        return model

    def test_position_tracking_no_double_buy(self):
        class FakeModel:
            def predict(self, context, horizon, return_quantiles=False, make_positive=False):
                return FakeForecastOutput(
                    forecast=np.array([110, 115]),
                    quantiles=np.full((horizon, 9), 112.0) if return_quantiles else None,
                )

        model = self._make_model(FakeModel())
        model._positions["TEST"] = "LONG"
        df = pd.DataFrame({"Close": np.linspace(100, 105, 30)})
        result = model.predict({"df": df, "ticker": "TEST"})
        self.assertEqual(result.signal, "HOLD")

    def test_position_tracking_sell_emitted_when_flat(self):
        # ADR-002: previously a SELL was forced to HOLD whenever the position
        # was FLAT, which suppressed every bearish vote in prod (0 SELL over
        # 610 predictions). A SELL must now be emitted as a directional vote
        # even from the default FLAT state; the risk manager decides whether
        # to act on it.
        class FakeModel:
            def predict(self, context, horizon, return_quantiles=False, make_positive=False):
                return FakeForecastOutput(
                    forecast=np.array([95, 90]),
                    quantiles=np.full((horizon, 9), 92.0) if return_quantiles else None,
                )

        model = self._make_model(FakeModel())
        df = pd.DataFrame({"Close": np.linspace(105, 100, 30)})
        result = model.predict({"df": df, "ticker": "TEST"})
        self.assertEqual(result.signal, "SELL")

    def test_quantiles_exposed_in_metadata(self):
        # Choix validé : médiane seule pour le signal, quantiles en métadonnées.
        class FakeModel:
            def predict(self, context, horizon, return_quantiles=False, make_positive=False):
                assert return_quantiles is True and make_positive is True
                return FakeForecastOutput(
                    forecast=np.array([110, 115]),
                    quantiles=np.linspace(100, 120, horizon * 9).reshape(horizon, 9),
                )

        model = self._make_model(FakeModel())
        df = pd.DataFrame({"Close": np.linspace(100, 105, 30)})
        result = model.predict({"df": df, "ticker": "TEST", "horizon": 5})
        self.assertIn("predictions", result.metadata)
        self.assertIn("quantiles", result.metadata)
        self.assertEqual(len(result.metadata["quantiles"]), 5)  # horizon
        self.assertEqual(len(result.metadata["quantiles"][0]), 9)  # P10..P90


class TestContextTruncation(unittest.TestCase):
    def test_context_truncated_to_timesfm_context(self):
        captured = {}

        class FakeModel:
            def predict(self, context, horizon, return_quantiles=False, make_positive=False):
                captured["len"] = len(context)
                return FakeForecastOutput(forecast=np.linspace(100, 110, horizon))

        model = TimesFMModel.__new__(TimesFMModel)
        model.initialized = True
        model.model = FakeModel()
        model.vol_multiplier = 0.5
        model._positions = {}
        model._get_position = lambda t: model._positions.get(t, "FLAT")
        model._adaptive_threshold = lambda p: 0.005

        df = pd.DataFrame({"Close": np.linspace(100, 200, TIMESFM_CONTEXT + 500)})
        model.predict({"df": df, "ticker": "TEST"})
        self.assertEqual(captured["len"], TIMESFM_CONTEXT)


class TestInitRetry(unittest.TestCase):
    def test_predict_retries_init_when_not_initialized(self):
        # L'ancien comportement : init échouée → HOLD 0.0 pour toujours.
        # Nouveau : predict() retente _try_init() (le 1er téléchargement du
        # checkpoint ~1,3 Go peut échouer en réseau).
        class FakeModel:
            def predict(self, context, horizon, return_quantiles=False, make_positive=False):
                return FakeForecastOutput(forecast=np.array([110, 115]))

        model = TimesFMModel.__new__(TimesFMModel)
        model.initialized = False
        model.model = None
        model.vol_multiplier = 0.5
        model._positions = {}

        def fake_try_init():
            model.initialized = True
            model.model = FakeModel()

        model._try_init = fake_try_init

        with patch("src.timesfm_model.TIMESFM3_AVAILABLE", True):
            df = pd.DataFrame({"Close": np.linspace(100, 105, 30)})
            result = model.predict({"df": df, "ticker": "TEST"})
        # Retry réussi : le modèle a prédit (115 > 105 ⇒ BUY) au lieu de HOLD 0.0.
        self.assertEqual(result.signal, "BUY")
        self.assertTrue(model.initialized)

    def test_predict_holds_when_api_unavailable(self):
        model = TimesFMModel.__new__(TimesFMModel)
        model.initialized = False
        model.model = None
        model.vol_multiplier = 0.5
        model._positions = {}

        with patch("src.timesfm_model.TIMESFM3_AVAILABLE", False):
            df = pd.DataFrame({"Close": np.linspace(100, 105, 30)})
            result = model.predict({"df": df, "ticker": "TEST"})
        self.assertEqual(result.signal, "HOLD")
        self.assertEqual(result.confidence, 0.0)


class TestAdaptiveThreshold(unittest.TestCase):
    def test_adaptive_threshold_high_vol(self):
        model = TimesFMModel.__new__(TimesFMModel)
        model.vol_multiplier = 0.5
        high_vol_prices = np.cumsum(np.random.randn(50) * 3) + 100
        threshold = model._adaptive_threshold(high_vol_prices)
        self.assertGreater(threshold, 0.005)

    def test_adaptive_threshold_low_vol(self):
        model = TimesFMModel.__new__(TimesFMModel)
        model.vol_multiplier = 0.5
        low_vol_prices = np.linspace(100, 101, 50)
        threshold = model._adaptive_threshold(low_vol_prices)
        self.assertGreaterEqual(threshold, 0.005)


class TestResetClearsPositions(unittest.TestCase):
    def test_reset_clears_positions(self):
        model = TimesFMModel.__new__(TimesFMModel)
        model._positions = {"A": "LONG", "B": "FLAT"}
        model.reset()
        self.assertEqual(model._positions, {})

    def test_reset_single_ticker(self):
        model = TimesFMModel.__new__(TimesFMModel)
        model._positions = {"A": "LONG", "B": "FLAT"}
        model.reset(ticker="A")
        self.assertNotIn("A", model._positions)
        self.assertIn("B", model._positions)


class TestGetTimesFmPredictionPassesTicker(unittest.TestCase):
    @patch("src.timesfm_model.TimesFMModel")
    def test_get_timesfm_prediction_passes_ticker(self, mock_cls):
        mock_instance = MagicMock()
        mock_instance.predict.return_value = ModelResult("HOLD", 0.0, "")
        mock_cls.get_instance.return_value = mock_instance

        df = pd.DataFrame({"Close": [100, 105, 110]})
        get_timesfm_prediction(df, ticker="MY_TICKER")

        mock_instance.predict.assert_called_once()
        args, kwargs = mock_instance.predict.call_args
        self.assertEqual(args[0]["ticker"], "MY_TICKER")


if __name__ == "__main__":
    unittest.main()
