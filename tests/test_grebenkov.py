import unittest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.grebenkov_model import GrebenkovTrendModel


def _make_data(ticker="SXRV.DE", n=800):
    dates = pd.date_range(start="1/1/2020", periods=n)
    np.random.seed(42)
    hist_data = pd.DataFrame({"Close": np.random.lognormal(0, 0.01, n).cumprod() * 100}, index=dates)
    wti_data = pd.DataFrame({"Close": np.random.lognormal(0, 0.02, n).cumprod() * 50}, index=dates)
    nasdaq_data = pd.DataFrame({"Close": np.random.lognormal(0, 0.015, n).cumprod() * 10000}, index=dates)
    return {
        "hist_data": hist_data,
        "wti_data": wti_data,
        "nasdaq_data": nasdaq_data,
        "ticker": ticker,
    }


class TestGrebenkovTrendModel(unittest.TestCase):
    def setUp(self):
        self.model = GrebenkovTrendModel(eta=1 / 112, rho=1 / 20)

    def test_predict_missing_data(self):
        result = self.model.predict({})
        self.assertEqual(result.signal, "HOLD")
        self.assertEqual(result.confidence, 0.0)

    def test_predict_valid_data(self):
        result = self.model.predict(_make_data())
        self.assertIn(result.signal, ["BUY", "SELL", "HOLD"])
        self.assertGreaterEqual(result.confidence, 0.0)
        self.assertLessEqual(result.confidence, 1.0)


class TestResetClearsState(unittest.TestCase):
    def test_reset_clears_state(self):
        model = GrebenkovTrendModel()
        model._position_type = "LONG"
        model._peak_price = 150.0
        model._last_ticker = "AAPL"
        model.reset()
        self.assertIsNone(model._peak_price)
        self.assertEqual(model._position_type, "FLAT")
        self.assertIsNone(model._last_ticker)


class TestAutoResetOnTickerChange(unittest.TestCase):
    def test_auto_reset_on_ticker_change(self):
        model = GrebenkovTrendModel()
        data1 = _make_data(ticker="SXRV.DE")
        model.predict(data1)
        self.assertEqual(model._last_ticker, "SXRV.DE")
        model._peak_price = 999.0
        model._position_type = "LONG"

        data2 = _make_data(ticker="AAPL")
        model.predict(data2)
        self.assertEqual(model._last_ticker, "AAPL")
        self.assertNotEqual(model._peak_price, 999.0)

    def test_no_reset_same_ticker(self):
        model = GrebenkovTrendModel()
        data = _make_data(ticker="SXRV.DE")
        model.predict(data)
        ticker_before = model._last_ticker
        model.predict(data)
        self.assertEqual(model._last_ticker, ticker_before)
        self.assertNotEqual(model._last_ticker, None)


class TestStopLossTriggersSell(unittest.TestCase):
    def test_stop_loss_triggers_sell(self):
        model = GrebenkovTrendModel(stop_loss_pct=0.10)
        model._position_type = "LONG"
        model._peak_price = 100.0
        model._last_ticker = "SXRV.DE"
        data = _make_data()
        data["hist_data"] = pd.DataFrame({"Close": [85.0] * 800}, index=data["hist_data"].index)
        result = model.predict(data)
        self.assertEqual(result.signal, "SELL")


class TestTrailingStopTriggersSell(unittest.TestCase):
    def test_trailing_stop_triggers_sell(self):
        model = GrebenkovTrendModel(trailing_stop_pct=0.05)
        model._position_type = "LONG"
        model._peak_price = 100.0
        model._last_ticker = "SXRV.DE"
        data = _make_data()
        data["hist_data"] = pd.DataFrame({"Close": [93.0] * 800}, index=data["hist_data"].index)
        result = model.predict(data)
        self.assertEqual(result.signal, "SELL")


class TestATRAdaptiveThreshold(unittest.TestCase):
    def test_atr_adaptive_threshold(self):
        np.random.seed(42)
        dates = pd.date_range(start="1/1/2020", periods=800)

        low_vol_prices = np.linspace(100, 102, 800)
        hist_low = pd.DataFrame({"Close": low_vol_prices}, index=dates)
        pd.DataFrame({"Close": np.random.lognormal(0, 0.02, 800).cumprod() * 50}, index=dates)
        pd.DataFrame({"Close": np.random.lognormal(0, 0.015, 800).cumprod() * 10000}, index=dates)

        model = GrebenkovTrendModel()
        model._weight_to_signal(0.1, 0.1, hist_low)
        threshold_low = model._compute_atr(hist_low) / max(model._compute_atr_median(hist_low), 1e-8)

        high_vol_prices = np.cumsum(np.random.randn(800) * 2) + 100
        hist_high = pd.DataFrame({"Close": high_vol_prices}, index=dates)
        model._weight_to_signal(0.1, 0.1, hist_high)
        threshold_high = model._compute_atr(hist_high) / max(model._compute_atr_median(hist_high), 1e-8)

        self.assertGreater(threshold_high, threshold_low)


if __name__ == "__main__":
    unittest.main()
