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


if __name__ == "__main__":
    unittest.main()
