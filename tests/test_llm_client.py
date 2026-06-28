import unittest
from unittest.mock import patch, MagicMock
import json
import requests
from src.llm_client import (
    _query_ollama,
    _extract_council_verdict,
    strip_thinking_debris,
    get_council_ticker_stance,
)


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


# A report body containing the parseable VERDICT_TICKER block the Judge is now
# instructed to emit (see council_prompts.JUDGE_PROMPT).
_TICKER_REPORT = """# Rapport
## Verdict du Juge
Synthèse qualitative. Notons que Le Stratège a dit BUY (0.75) en prose.

VERDICT_TICKER:
SXRV.DE: BUY (0.65)
CRUDP.PA: SELL (0.90)
"""


class TestCouncilTickerStance(unittest.TestCase):
    """Parses the Judge's VERDICT_TICKER block per ticker, with age decay."""

    def _patch_loaded(self, age_days, text):
        """Helper: patch _load_fresh_council_report to return (age, text)."""
        ret = None if age_days is None else (age_days, text)
        return patch("src.llm_client._load_fresh_council_report", return_value=ret)

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
        """Stale or missing report → graceful skip (no council vote)."""
        sig, conf = get_council_ticker_stance("SXRV.DE")
        self.assertIsNone(sig)
        self.assertEqual(conf, 0.0)

    @patch("src.llm_client._load_fresh_council_report")
    def test_decay_reduces_confidence_linearly(self, mock_load):
        """Day 3 → confidence × (1 - 3/7) = confidence × 0.571."""
        mock_load.return_value = (3.0, _TICKER_REPORT)
        sig, conf = get_council_ticker_stance("CRUDP.PA")
        self.assertEqual(sig, "SELL")
        self.assertAlmostEqual(conf, 0.90 * (1 - 3 / 7), places=3)

    @patch("src.llm_client._load_fresh_council_report")
    def test_decay_reaches_zero_at_day_7(self, mock_load):
        mock_load.return_value = (7.0, _TICKER_REPORT)
        sig, conf = get_council_ticker_stance("CRUDP.PA")
        self.assertEqual(sig, "SELL")
        self.assertEqual(conf, 0.0)

    @patch("src.llm_client._load_fresh_council_report")
    def test_does_not_parse_prose_mentions(self, mock_load):
        """A stance mentioned in prose (before the block) is ignored.

        The report body says 'BUY (0.75)' in narrative but the block says
        SXRV.DE: BUY (0.65). We must pick up 0.65, not 0.75.
        """
        mock_load.return_value = (0.0, _TICKER_REPORT)
        sig, conf = get_council_ticker_stance("SXRV.DE")
        self.assertAlmostEqual(conf, 0.65)

    @patch("src.llm_client._load_fresh_council_report")
    def test_report_without_verdict_block_returns_none(self, mock_load):
        """A fresh report that lacks the VERDICT_TICKER block → graceful skip."""
        mock_load.return_value = (0.0, "# Rapport\n## Verdict du Juge\nProse only, no block.\n")
        sig, conf = get_council_ticker_stance("SXRV.DE")
        self.assertIsNone(sig)
        self.assertEqual(conf, 0.0)

    @patch("src.llm_client._load_fresh_council_report")
    def test_percent_value_rescaled(self, mock_load):
        """Judge may emit 85 (percent) instead of 0.85 — rescale with warning."""
        report = "VERDICT_TICKER:\nCRUDP.PA: SELL (85)\n"
        mock_load.return_value = (0.0, report)
        sig, conf = get_council_ticker_stance("CRUDP.PA")
        self.assertEqual(sig, "SELL")
        self.assertAlmostEqual(conf, 0.85)

    @patch("src.llm_client._load_fresh_council_report")
    def test_french_decimal_comma_handled(self, mock_load):
        """French LLM may emit 0,65 (comma) instead of 0.65."""
        report = "VERDICT_TICKER:\nSXRV.DE: BUY (0,65)\n"
        mock_load.return_value = (0.0, report)
        sig, conf = get_council_ticker_stance("SXRV.DE")
        self.assertEqual(sig, "BUY")
        self.assertAlmostEqual(conf, 0.65)


if __name__ == "__main__":
    unittest.main()
