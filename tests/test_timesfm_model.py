import unittest
from unittest.mock import patch, MagicMock
import pandas as pd

from src.timesfm_model import get_timesfm_prediction

class TestTimesFMModelWrapper(unittest.TestCase):
    def setUp(self):
        # Create a dummy DataFrame for testing
        self.dummy_df = pd.DataFrame({'Close': [100, 105, 110]})

    @patch('src.timesfm_model.TimesFMModel')
    def test_get_timesfm_prediction_success(self, mock_timesfm_model_class):
        # Setup mock behavior
        mock_instance = MagicMock()
        mock_prediction = {
            "signal": "BUY",
            "confidence": 0.8,
            "analysis": "Mock analysis",
            "predictions": [115.0]
        }
        mock_instance.predict.return_value = mock_prediction
        mock_timesfm_model_class.get_instance.return_value = mock_instance

        # Call function
        result = get_timesfm_prediction(self.dummy_df)

        # Assertions
        mock_timesfm_model_class.get_instance.assert_called_once()
        mock_instance.predict.assert_called_once_with(self.dummy_df)
        self.assertEqual(result, mock_prediction)

    @patch('src.timesfm_model.TimesFMModel')
    def test_get_timesfm_prediction_exception(self, mock_timesfm_model_class):
        # Setup mock behavior to raise exception
        mock_timesfm_model_class.get_instance.side_effect = Exception("Mocked initialization error")

        # Call function
        result = get_timesfm_prediction(self.dummy_df)

        # Assertions
        self.assertEqual(result["signal"], "HOLD")
        self.assertEqual(result["confidence"], 0.0)
        self.assertTrue("Model error" in result["analysis"])

if __name__ == '__main__':
    unittest.main()
