import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import pandas as pd

from src.timesfm_model import get_timesfm_prediction, TimesFMModel
from src.enhanced_decision_engine import ModelResult


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
    def test_position_tracking_no_double_buy(self):
        model = TimesFMModel.__new__(TimesFMModel)
        model.initialized = False
        model.model = None
        model.vol_multiplier = 0.5
        model._positions = {}
        model._positions["TEST"] = "LONG"
        model._get_position = lambda t: model._positions.get(t, "FLAT")
        model._adaptive_threshold = lambda p: 0.005
        model.initialized = True

        class FakeModel:
            def forecast(self, horizon, inputs):
                return [np.array([110, 115])], None

        model.model = FakeModel()
        df = pd.DataFrame({"Close": np.linspace(100, 105, 30)})
        result = model.predict({"df": df, "ticker": "TEST"})
        self.assertEqual(result.signal, "HOLD")

    def test_position_tracking_no_sell_when_flat(self):
        model = TimesFMModel.__new__(TimesFMModel)
        model.initialized = False
        model.model = None
        model.vol_multiplier = 0.5
        model._positions = {}
        model._get_position = lambda t: model._positions.get(t, "FLAT")
        model._adaptive_threshold = lambda p: 0.005
        model.initialized = True

        class FakeModel:
            def forecast(self, horizon, inputs):
                return [np.array([95, 90])], None

        model.model = FakeModel()
        df = pd.DataFrame({"Close": np.linspace(105, 100, 30)})
        result = model.predict({"df": df, "ticker": "TEST"})
        self.assertEqual(result.signal, "HOLD")


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
