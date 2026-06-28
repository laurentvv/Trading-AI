import unittest
from unittest.mock import patch

import pandas as pd

from src.core.tools import lookup_ohlc, NumericalReasoningEngine, AnswerConsolidationGate


# Fixture : un historique OHLCV suffisant pour calculer SMA200 et RSI.
_FAKE_HISTORY = pd.DataFrame(
    {
        "Open": [10.0] * 210,
        "High": [11.0] * 210,
        "Low": [9.5] * 210,
        "Close": [10.0 + (i % 5) * 0.1 for i in range(210)],
        "Volume": [1000.0] * 210,
    },
    index=pd.date_range("2025-01-01", periods=210, freq="D"),
)


class TestFinAcumen(unittest.TestCase):
    def test_consolidation_gate_valid(self):
        trajectory = [
            "Action: execute_python({'python_code': 'lookup_ohlc(\"WTI\", \"latest\", \"close\")'})"
        ]
        answer = {
            "action": "BUY",
            "confidence": 0.8,
            "reasoning": "Price is above MA and lookup_ohlc returned valid data.",
        }
        result = AnswerConsolidationGate.verify(trajectory, answer)
        self.assertTrue(result["valid"])

    def test_consolidation_gate_invalid_action(self):
        trajectory = []
        answer = {"action": "MAYBE", "confidence": 0.5, "reasoning": "Not sure."}
        result = AnswerConsolidationGate.verify(trajectory, answer)
        self.assertFalse(result["valid"])

    # --- Regression tests for the 4 FinAcumen prod bugs ---

    @patch("src.core.tools.yf.Ticker")
    def test_lookup_ohlc_list_request_returns_dict(self, mock_ticker_cls):
        """Bug A : un appel avec une LISTE d'indicateurs doit retourner un dict,
        pas lever AttributeError ('list' has no attribute 'lower')."""
        mock_ticker_cls.return_value.history.return_value = _FAKE_HISTORY.copy()
        result = lookup_ohlc("CRUDP.PA", "latest", ["close", "vwap", "rsi", "sma_50", "sma_200", "volume"])
        self.assertIsInstance(result, dict)
        self.assertIn("close", result)
        self.assertIn("rsi", result)
        self.assertIsNotNone(result["close"])
        self.assertIsNotNone(result["rsi"])

    @patch("src.core.tools.yf.Ticker")
    def test_lookup_ohlc_single_string_still_returns_float(self, mock_ticker_cls):
        """Compat descendante : une chaîne unique retourne toujours un float."""
        mock_ticker_cls.return_value.history.return_value = _FAKE_HISTORY.copy()
        result = lookup_ohlc("CRUDP.PA", "latest", "close")
        self.assertIsInstance(result, float)

    @patch("src.core.tools.yf.Ticker")
    def test_lookup_ohlc_derived_indicators_computed(self, mock_ticker_cls):
        """Bug B : rsi / sma_50 / sma_200 doivent etre calcules (pas None)."""
        mock_ticker_cls.return_value.history.return_value = _FAKE_HISTORY.copy()
        result = lookup_ohlc("SXRV.DE", "latest", ["rsi", "sma_50", "sma_200", "macd"])
        self.assertIsInstance(result, dict)
        self.assertIsNotNone(result["rsi"])
        self.assertIsNotNone(result["sma_50"])
        self.assertIsNotNone(result["sma_200"])
        self.assertIsNotNone(result["macd"])

    def test_sandbox_pd_np_without_import(self):
        """Bug C : pd/np sont pre-importes ; aucun 'import' necessaire."""
        eng = NumericalReasoningEngine()
        r = eng.execute("print(np.mean(np.array([2, 4, 6])))")
        self.assertTrue(r["success"], r.get("error"))
        self.assertIn("4", r["output"])

        r2 = eng.execute("print(pd.Series([1, 2, 3]).sum())")
        self.assertTrue(r2["success"], r2.get("error"))

    def test_sandbox_import_still_blocked(self):
        """Securite preservee : __import__ reste inaccessible."""
        eng = NumericalReasoningEngine()
        r = eng.execute("import os\nprint('leaked')")
        self.assertFalse(r["success"])
        self.assertIn("ImportError", r["error"])

    @patch("src.core.tools.yf.Ticker")
    def test_sandbox_full_finacumen_style_code(self, mock_ticker_cls):
        """Integration : le code typique genere par le LLM (liste d'indicateurs,
        pas d'import, calcul conditionnel) s'execute sans erreur."""
        mock_ticker_cls.return_value.history.return_value = _FAKE_HISTORY.copy()
        eng = NumericalReasoningEngine()
        code = (
            "data = lookup_ohlc('CRUDP.PA', 'latest', ['close', 'rsi', 'sma_50', 'sma_200'])\n"
            "price = data['close']\n"
            "rsi = data['rsi']\n"
            "if price > data['sma_50'] and rsi < 70:\n"
            "    print('BUY')\n"
            "else:\n"
            "    print('HOLD')\n"
        )
        r = eng.execute(code)
        self.assertTrue(r["success"], r.get("error"))
        self.assertIn(r["output"], ("BUY", "HOLD"))


if __name__ == "__main__":
    unittest.main()
