import unittest
from unittest.mock import patch, MagicMock, AsyncMock
import json
from src.llm_client import (
    _query_nexus,
    _query_ollama,
    _extract_council_verdict,
    strip_thinking_debris,
    get_council_ticker_stance,
    check_ai_health,
)


class TestLLMClient(unittest.TestCase):
    def setUp(self):
        self.prompt = "Analyze market data for SXRV.DE"

    @patch("src.llm_client._run_sync")
    def test_query_nexus_success(self, mock_run_sync):
        mock_run_sync.return_value = {
            "signal": "BUY",
            "confidence": 0.8,
            "analysis": "Test analysis",
            "_provider": "groq",
            "_model": "llama-3.3-70b-versatile",
        }

        result = _query_nexus(self.prompt)

        self.assertEqual(result["signal"], "BUY")
        self.assertEqual(result["confidence"], 0.8)
        self.assertEqual(result["analysis"], "Test analysis")

    @patch("src.llm_client._run_sync")
    def test_query_ollama_alias_success(self, mock_run_sync):
        mock_run_sync.return_value = {
            "signal": "BUY",
            "confidence": 0.85,
            "analysis": "Test analysis via alias",
        }

        result = _query_ollama({"prompt": self.prompt})

        self.assertEqual(result["signal"], "BUY")
        self.assertEqual(result["confidence"], 0.85)

    @patch("src.llm_client.AIGateway.get_configured_providers")
    def test_check_ai_health(self, mock_get_providers):
        mock_get_providers.return_value = ["gemini_free", "groq"]
        self.assertTrue(check_ai_health())

        mock_get_providers.return_value = []
        self.assertFalse(check_ai_health())


class TestStripThinkingDebris(unittest.TestCase):
    """Prose-mode output must be scrubbed of channel markers."""

    def test_removes_leading_channel_marker(self):
        text = "<|channel>thought\n<channel|>Le verdict est clair."
        cleaned = strip_thinking_debris(text)
        self.assertNotIn("<|channel", cleaned)
        self.assertNotIn("<channel|>", cleaned)
        self.assertIn("Le verdict est clair", cleaned)

    def test_no_debris_unchanged(self):
        text = "Un texte parfaitement propre."
        self.assertEqual(strip_thinking_debris(text), text)

    def test_collapses_excess_blank_lines(self):
        text = "Para 1.\n\n\n\n\nPara 2."
        cleaned = strip_thinking_debris(text)
        self.assertNotIn("\n\n\n", cleaned)
        self.assertIn("Para 1", cleaned)
        self.assertIn("Para 2", cleaned)

    def test_all_known_tokens_stripped(self):
        from src.llm_client import _THINKING_TOKENS
        text = " ".join(_THINKING_TOKENS) + " contenu"
        cleaned = strip_thinking_debris(text)
        for tok in _THINKING_TOKENS:
            self.assertNotIn(tok, cleaned)
        self.assertIn("contenu", cleaned)


class TestCouncilVerdictExtraction(unittest.TestCase):
    """Extracts only the Judge's verdict (not the debate transcript)."""

    def test_extracts_verdict_between_marker_and_annexe(self):
        report = (
            "# Rapport du Conseil\n\n"
            "*Date: 2026-06-27*\n\n"
            "## Verdict du Juge\n\nIntro narrative.\n\n---\n\n"
            "### VERDICT : Prudence.\nRecommandation: retirer CRUDP.\n\n"
            "---\n## Annexe : Transcription des Débats\n\nBeaucoup de texte...\n"
        )
        verdict = _extract_council_verdict(report)
        self.assertIn("Intro narrative", verdict)
        self.assertIn("Prudence", verdict)
        self.assertIn("retirer CRUDP", verdict)
        self.assertNotIn("Transcription", verdict)
        self.assertNotIn("Annexe", verdict)

    def test_no_marker_returns_empty(self):
        self.assertEqual(_extract_council_verdict("no verdict here"), "")

    def test_no_annexe_takes_rest_of_text(self):
        report = "## Verdict du Juge\n\nVerdict sans annexe."
        verdict = _extract_council_verdict(report)
        self.assertIn("Verdict sans annexe", verdict)

    def test_internal_separators_preserved(self):
        report = (
            "## Verdict du Juge\n\nIntro.\n\n---\n\n### Recommandation\nAction.\n\n"
            "## Annexe\n(transcript)"
        )
        verdict = _extract_council_verdict(report)
        self.assertIn("Intro", verdict)
        self.assertIn("Recommandation", verdict)
        self.assertIn("Action", verdict)

    def test_strips_thinking_channel_debris(self):
        report = (
            "## Verdict du Juge\n\n"
            "<|channel>thought\n<channel|>Le verdict est clair.\n\n"
            "## Annexe\ntranscript"
        )
        verdict = _extract_council_verdict(report)
        self.assertNotIn("<|channel", verdict)
        self.assertNotIn("<channel|>", verdict)
        self.assertIn("Le verdict est clair", verdict)


class TestCouncilVerdictContext(unittest.TestCase):
    def _make_report(self, verdict_text="Le conseil recommande la prudence."):
        return (
            f"# Rapport\n*Date: ...\n\n## Verdict du Juge\n\n{verdict_text}\n\n"
            "---\n## Annexe\n(transcript omitted)"
        )

    @patch("src.llm_client._find_latest_council_report", return_value=None)
    def test_no_report_returns_empty(self, _):
        from src.llm_client import get_council_verdict_context
        self.assertEqual(get_council_verdict_context(), "")

    @patch("src.llm_client.Path")
    @patch("src.llm_client._find_latest_council_report")
    def test_fresh_report_injects_verdict(self, mock_find, mock_path_cls):
        from src.llm_client import get_council_verdict_context
        from datetime import datetime, timedelta

        mock_report = MagicMock()
        mock_report.stem = "council_report_" + (datetime.now() - timedelta(days=2)).strftime("%Y-%m-%d")
        mock_find.return_value = mock_report

        mock_file = MagicMock()
        mock_file.__enter__.return_value.read.return_value = self._make_report()
        with patch("builtins.open", return_value=mock_file):
            ctx = get_council_verdict_context()
        self.assertIn("Weekend AI Council Verdict", ctx)
        self.assertIn("Le conseil recommande la prudence", ctx)

    @patch("src.llm_client._find_latest_council_report")
    def test_stale_report_returns_empty(self, mock_find):
        from src.llm_client import get_council_verdict_context
        from datetime import datetime, timedelta

        mock_report = MagicMock()
        mock_report.stem = "council_report_" + (datetime.now() - timedelta(days=10)).strftime("%Y-%m-%d")
        mock_find.return_value = mock_report
        self.assertEqual(get_council_verdict_context(), "")


_TICKER_REPORT = """# Rapport
## Verdict du Juge
Synthèse qualitative. Notons que Le Stratège a dit BUY (0.75) en prose.

VERDICT_TICKER:
SXRV.DE: BUY (0.65)
CRUDP.PA: SELL (0.90)
"""


class TestCouncilTickerStance(unittest.TestCase):
    @patch("src.llm_client._load_fresh_council_report")
    def test_parses_stance_per_ticker(self, mock_load):
        mock_load.return_value = (0.0, _TICKER_REPORT)
        sig, conf = get_council_ticker_stance("SXRV.DE")
        self.assertEqual(sig, "BUY")
        self.assertAlmostEqual(conf, 0.65)
        sig, conf = get_council_ticker_stance("CRUDP.PA")
        self.assertEqual(sig, "SELL")
        self.assertAlmostEqual(conf, 0.90)

    @patch("src.llm_client._load_fresh_council_report")
    def test_unknown_ticker_returns_none(self, mock_load):
        mock_load.return_value = (0.0, _TICKER_REPORT)
        sig, conf = get_council_ticker_stance("UNKNOWN.X")
        self.assertIsNone(sig)
        self.assertEqual(conf, 0.0)

    @patch("src.llm_client._load_fresh_council_report", return_value=None)
    def test_no_fresh_report_returns_none(self, _load):
        sig, conf = get_council_ticker_stance("SXRV.DE")
        self.assertIsNone(sig)
        self.assertEqual(conf, 0.0)

    @patch("src.llm_client._load_fresh_council_report")
    def test_decay_reduces_confidence_linearly(self, mock_load):
        mock_load.return_value = (3.0, _TICKER_REPORT)
        sig, conf = get_council_ticker_stance("CRUDP.PA")
        self.assertEqual(sig, "SELL")
        self.assertAlmostEqual(conf, 0.90 * (1 - 3 / 7), places=3)

    @patch("src.llm_client._load_fresh_council_report")
    def test_percent_value_rescaled(self, mock_load):
        report = "VERDICT_TICKER:\nCRUDP.PA: SELL (85)\n"
        mock_load.return_value = (0.0, report)
        sig, conf = get_council_ticker_stance("CRUDP.PA")
        self.assertEqual(sig, "SELL")
        self.assertAlmostEqual(conf, 0.85)


if __name__ == "__main__":
    unittest.main()
