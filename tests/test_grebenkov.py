import unittest
import pandas as pd
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.grebenkov_model import GrebenkovTrendModel

class TestGrebenkovTrendModel(unittest.TestCase):
    def setUp(self):
        self.model = GrebenkovTrendModel(eta=1/112, rho=1/20)

    def test_predict_missing_data(self):
        result = self.model.predict({})
        self.assertEqual(result.signal, "HOLD")
        self.assertEqual(result.confidence, 0.0)

    def test_predict_valid_data(self):
        # Generate dummy data
        dates = pd.date_range(start='1/1/2020', periods=800)
        np.random.seed(42)

        hist_data = pd.DataFrame({"Close": np.random.lognormal(0, 0.01, 800).cumprod() * 100}, index=dates)
        wti_data = pd.DataFrame({"Close": np.random.lognormal(0, 0.02, 800).cumprod() * 50}, index=dates)
        nasdaq_data = pd.DataFrame({"Close": np.random.lognormal(0, 0.015, 800).cumprod() * 10000}, index=dates)

        data = {
            "hist_data": hist_data,
            "wti_data": wti_data,
            "nasdaq_data": nasdaq_data,
            "ticker": "SXRV.DE"
        }

        result = self.model.predict(data)
        self.assertIn(result.signal, ["BUY", "SELL", "HOLD"])
        self.assertGreaterEqual(result.confidence, 0.0)
        self.assertLessEqual(result.confidence, 1.0)

if __name__ == '__main__':
    unittest.main()
