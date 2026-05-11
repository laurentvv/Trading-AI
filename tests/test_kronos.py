import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.kronos_model import KronosModel, get_kronos_prediction


class TestKronosModel(unittest.TestCase):
    @patch("src.kronos_model.KRONOS_AVAILABLE", True)
    @patch("src.kronos_model.KronosTokenizer", create=True)
    @patch("src.kronos_model.Kronos", create=True)
    @patch("src.kronos_model.KronosPredictor", create=True)
    def test_kronos_prediction(self, mock_predictor, mock_kronos, mock_tokenizer):
        # Create dummy instance and clear it just in case
        KronosModel._instance = None
        KronosModel.get_instance()

        # Setup mocks
        mock_predictor_instance = MagicMock()
        mock_predictor.return_value = mock_predictor_instance

        # Create a dummy DataFrame with prediction
        dates = pd.date_range("2023-01-01", periods=10)
        pred_df = pd.DataFrame(
            {
                "open": np.linspace(100, 110, 10),
                "high": np.linspace(105, 115, 10),
                "low": np.linspace(95, 105, 10),
                "close": np.linspace(102, 112, 10),  # Ends at 112
            },
            index=dates,
        )

        mock_predictor_instance.predict.return_value = pred_df

        # Input DataFrame
        input_dates = pd.date_range("2022-12-20", periods=12)
        input_df = pd.DataFrame(
            {
                "Open": np.linspace(90, 100, 12),
                "High": np.linspace(95, 105, 12),
                "Low": np.linspace(85, 95, 12),
                "Close": np.linspace(92, 102, 12),  # Last close is 102
            },
            index=input_dates,
        )

        # Run prediction
        result = get_kronos_prediction(input_df, pred_len=10)

        # Verify result
        self.assertIn("signal", result)
        self.assertIn("confidence", result)
        self.assertIn("forecast_df", result)

        # 102 -> 112 is ~9.8% increase, should be BUY
        self.assertEqual(result["signal"], "BUY")
        self.assertGreater(result["confidence"], 0)


if __name__ == "__main__":
    unittest.main()
