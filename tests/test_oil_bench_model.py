import unittest
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from oil_bench_model import OilBenchModel, OilBenchConfig


class TestOilBenchSignalTranslation(unittest.TestCase):
    def setUp(self):
        self.model = OilBenchModel.__new__(OilBenchModel)
        self.model.config = OilBenchConfig()

    def test_strong_buy(self):
        result = self.model._translate_signal({"allocation": 80, "reasoning": "very bullish"})
        self.assertEqual(result["signal"], "STRONG_BUY")
        self.assertGreater(result["confidence"], 0.5)

    def test_buy(self):
        result = self.model._translate_signal({"allocation": 60, "reasoning": "bullish"})
        self.assertEqual(result["signal"], "BUY")
        self.assertAlmostEqual(result["confidence"], 0.2, places=1)

    def test_hold(self):
        result = self.model._translate_signal({"allocation": 50, "reasoning": "neutral"})
        self.assertEqual(result["signal"], "HOLD")
        self.assertTrue(0.3 <= result["confidence"] <= 0.35)

    def test_strong_sell(self):
        result = self.model._translate_signal({"allocation": 20, "reasoning": "very bearish"})
        self.assertEqual(result["signal"], "STRONG_SELL")
        self.assertGreater(result["confidence"], 0.5)


class TestOilBenchPrompt(unittest.TestCase):
    def setUp(self):
        self.model = OilBenchModel.__new__(OilBenchModel)
        self.model.config = OilBenchConfig()

    def test_construct_prompt_with_spread(self):
        price_data = {
            "wti": {"price": 80.0, "change_pct": 2.0},
            "brent": {"price": 85.0, "change_pct": 1.5},
            "dxy": {"price": 105.0, "change_pct": 0.5},
            "brent_spot": 95.0,
        }
        eia_text = "EIA Data Test"
        headlines = ["News 1", "News 2"]
        
        prompt = self.model._construct_prompt(price_data, eia_text, headlines)
        
        self.assertIn("Dated Brent Spread (Spot vs Futures): $10.00", prompt)
        self.assertIn("EIA Data Test", prompt)
        self.assertIn("WTI Spot: $80.00", prompt)
        self.assertIn("Brent Futures: $85.00", prompt)
        self.assertIn("Dated Brent Spread", prompt)
        self.assertIn("Historical norm is $1-$2", prompt)


if __name__ == "__main__":
    unittest.main()
