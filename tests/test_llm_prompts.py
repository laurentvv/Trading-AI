import unittest
import pandas as pd
import json
import sys
from pathlib import Path

# Ajouter le chemin racine pour importer src
sys.path.append(str(Path(__file__).parent.parent))

from src.llm_client import (
    construct_llm_prompt,
    get_llm_decision,
    get_visual_llm_decision,
)
from unittest.mock import patch, MagicMock


class TestLLMPrompts(unittest.TestCase):
    def setUp(self):
        self.sample_data = pd.DataFrame(
            [
                {
                    "Close": 15230.50,
                    "RSI": 42.15,
                    "MACD": 12.5,
                    "MACD_Signal": 15.2,
                    "BB_Position": 0.35,
                    "Trend_Short": 1,
                    "Trend_Long": -1,
                }
            ]
        )
        self.headlines = [
            "US Inflation data exceeds expectations",
            "Tech stocks rally on AI optimism",
        ]
        self.web_context = "OPEC+ signal possible supply cuts in June meeting."
        self.vg_indicators = {"HL_OIL_funding": -0.015, "HL_OIL_oi": 1250000.0}

    def test_construct_llm_prompt_content(self):
        """Affiche et vérifie le prompt texte généré."""
        prompt = construct_llm_prompt(self.sample_data, self.headlines, self.web_context, self.vg_indicators)
        self.assertIn("Close Price: 15230.50", prompt)
        self.assertIn("RSI (14): 42.15", prompt)
        self.assertIn("Speculative Sentiment (Hyperliquid", prompt)

    @patch("src.llm_client._query_nexus_vision")
    def test_visual_llm_prompt(self, mock_query_vision):
        """Affiche et vérifie l'appel visuel."""
        mock_query_vision.return_value = {
            "signal": "HOLD",
            "confidence": 0.65,
            "analysis": "Visual double bottom detected on H1 chart.",
            "_provider": "gemini_free",
            "_model": "gemini-2.5-flash",
        }

        dummy_path = Path("dummy_chart.png")
        dummy_path.write_bytes(b"fake_binary_data_for_test")

        try:
            result = get_visual_llm_decision(dummy_path)
            self.assertEqual(result.signal, "HOLD")
            self.assertEqual(result.confidence, 0.65)
            self.assertIn("nexus_vision", result.metadata.get("backend", ""))
        finally:
            if dummy_path.exists():
                dummy_path.unlink()

    @patch("src.llm_client._query_nexus")
    def test_llm_search_query_logic(self, mock_query):
        mock_query.return_value = {
            "signal": "BUY",
            "confidence": 0.88,
            "analysis": "Web context confirms bullish bias.",
            "_provider": "groq",
            "_model": "llama-3.3-70b-versatile",
        }

        result = get_llm_decision(self.sample_data, web_context=self.web_context)

        self.assertEqual(result.signal, "BUY")
        self.assertEqual(result.confidence, 0.88)
        self.assertIn("nexus", result.metadata.get("backend", ""))


if __name__ == "__main__":
    unittest.main()
