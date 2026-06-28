import unittest
from unittest.mock import patch, MagicMock
import json
import requests
from src.llm_client import _query_ollama, _extract_council_verdict, strip_thinking_debris


class TestLLMClient(unittest.TestCase):
    def setUp(self):
        self.payload = {"model": "test-model", "prompt": "test prompt"}

    @patch("src.llm_client.requests.post")
    def test_query_ollama_success(self, mock_post):
        # Mock successful response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "response": json.dumps({"signal": "BUY", "confidence": 0.8, "analysis": "Test analysis"})
        }
        mock_post.return_value = mock_response

        result = _query_ollama(self.payload)

        self.assertEqual(result["signal"], "BUY")
        self.assertEqual(result["confidence"], 0.8)
        self.assertEqual(result["analysis"], "Test analysis")
        mock_post.assert_called_once()

    @patch("src.llm_client.time.sleep")
    @patch("src.llm_client.requests.post")
    def test_query_ollama_request_exception_fallback(self, mock_post, mock_sleep):
        # Mock requests.post to raise RequestException
        mock_post.side_effect = requests.exceptions.RequestException("API error")

        result = _query_ollama(self.payload)

        self.assertEqual(result["signal"], "HOLD")
        self.assertEqual(result["confidence"], 0.0)
        self.assertEqual(result["analysis"], "")
        self.assertEqual(mock_post.call_count, 3)
        # self.assertEqual(mock_sleep.call_count, 2)

    @patch("src.llm_client.time.sleep")
    @patch("src.llm_client.requests.post")
    def test_query_ollama_json_decode_error_fallback(self, mock_post, mock_sleep):
        # Mock successful request but invalid JSON in response
        mock_response = MagicMock()
        mock_response.json.return_value = {"response": "Invalid JSON string"}
        mock_post.return_value = mock_response

        result = _query_ollama(self.payload)

        self.assertEqual(result["signal"], "HOLD")
        self.assertEqual(result["confidence"], 0.0)
        self.assertEqual(result["analysis"], "")
        self.assertEqual(mock_post.call_count, 3)
        # self.assertEqual(mock_sleep.call_count, 2)

    @patch("src.llm_client.time.sleep")
    @patch("src.llm_client.requests.post")
    def test_query_ollama_value_error_fallback(self, mock_post, mock_sleep):
        # Mock successful request but missing required keys
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "response": json.dumps(
                {
                    "signal": "BUY"
                    # Missing confidence and analysis
                }
            )
        }
        mock_post.return_value = mock_response

        result = _query_ollama(self.payload)

        self.assertEqual(result["signal"], "HOLD")
        self.assertEqual(result["confidence"], 0.0)
        self.assertEqual(result["analysis"], "")
        self.assertEqual(mock_post.call_count, 3)
        # self.assertEqual(mock_sleep.call_count, 2)


class TestStripThinkingDebris(unittest.TestCase):
    """Prose-mode Gemma output must be scrubbed of <|think|> channel markers."""

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
        # 3+ consecutive newlines collapse to a single blank line
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
        # The verdict uses --- internally; the real boundary is ## Annexe.
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
        # Internal separators are preserved, but the transcript must NOT leak.
        self.assertNotIn("Transcription", verdict)
        self.assertNotIn("Annexe", verdict)

    def test_no_marker_returns_empty(self):
        self.assertEqual(_extract_council_verdict("no verdict here"), "")

    def test_no_annexe_takes_rest_of_text(self):
        report = "## Verdict du Juge\n\nVerdict sans annexe."
        verdict = _extract_council_verdict(report)
        self.assertIn("Verdict sans annexe", verdict)

    def test_internal_separators_preserved(self):
        """Regression: cutting at the first --- truncated the real verdict.

        The Judge's verdict contains internal --- separators (between intro
        and recommendations). Extraction must keep everything up to ## Annexe.
        """
        report = (
            "## Verdict du Juge\n\nIntro.\n\n---\n\n### Recommandation\nAction.\n\n"
            "## Annexe\n(transcript)"
        )
        verdict = _extract_council_verdict(report)
        self.assertIn("Intro", verdict)
        self.assertIn("Recommandation", verdict)
        self.assertIn("Action", verdict)

    def test_strips_thinking_channel_debris(self):
        """Gemma think-mode leaks <|channel>thought debris in prose mode."""
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
    """get_council_verdict_context: freshness + filename-date selection."""

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
        # Path(...) used in open() — but we use open() builtin, not Path. Patch builtin open.
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

    @patch("src.llm_client._find_latest_council_report")
    def test_future_dated_report_returns_empty(self, mock_find):
        """A report dated in the future (clock skew / tz) is ignored."""
        from src.llm_client import get_council_verdict_context
        from datetime import datetime, timedelta

        mock_report = MagicMock()
        mock_report.stem = "council_report_" + (datetime.now() + timedelta(days=2)).strftime("%Y-%m-%d")
        mock_find.return_value = mock_report
        self.assertEqual(get_council_verdict_context(), "")


if __name__ == "__main__":
    unittest.main()
