"""API keys must never reach the log files (audit 2026-10-12).

httpx logs every request URL at INFO; the Gemini REST API carries its key as `?key=...`. From
August to October 2026 about 1 100 lines of trading.log* held the gemini_free key in clear, and an
Alpha Vantage `apikey=` slipped through the news stderr (the old pattern only matched free text).
"""

import logging
import unittest

from src.bootstrap import SecretRedactingFilter, redact_secrets

FAKE_KEY = "AQ.Ab8FAKEFAKEFAKEFAKEFAKE0123456789"


class TestRedactSecrets(unittest.TestCase):
    def test_gemini_key_in_url_is_masked(self):
        line = (
            "HTTP Request: POST https://generativelanguage.googleapis.com/v1beta/models/"
            f'gemini-3.5-flash-lite:generateContent?key={FAKE_KEY} "HTTP/1.1 401 Unauthorized"'
        )
        out = redact_secrets(line)
        self.assertNotIn(FAKE_KEY, out)
        self.assertIn("?key=***REDACTED***", out)
        self.assertIn('"HTTP/1.1 401 Unauthorized"', out)

    def test_alpha_vantage_apikey_among_other_parameters(self):
        line = "Max retries exceeded with url: /query?function=NEWS_SENTIMENT&apikey=FAKEAVKEY0123456&limit=50"
        out = redact_secrets(line)
        self.assertNotIn("FAKEAVKEY0123456", out)
        self.assertIn("function=NEWS_SENTIMENT", out)
        self.assertIn("&limit=50", out)

    def test_free_text_key_still_masked(self):
        out = redact_secrets("We have detected your API key as FAKEAVKEY0123456 and will throttle")
        self.assertNotIn("FAKEAVKEY0123456", out)

    def test_ordinary_text_untouched(self):
        line = "Cycle terminé avec succès : 2 tickers, monkey=3, keyboard=ok"
        self.assertEqual(redact_secrets(line), line)


class TestSecretRedactingFilter(unittest.TestCase):
    def _format(self, record: logging.LogRecord) -> str:
        self.assertTrue(SecretRedactingFilter().filter(record))
        return logging.Formatter("%(message)s").format(record)

    def test_filter_masks_lazy_format_arguments(self):
        # httpx logs with %-style arguments: the key is in record.args, not in record.msg.
        record = logging.LogRecord(
            "httpx", logging.INFO, __file__, 1, "HTTP Request: %s %s", ("POST", f"https://x/y?key={FAKE_KEY}"), None
        )
        self.assertNotIn(FAKE_KEY, self._format(record))

    def test_filter_masks_traceback(self):
        try:
            raise ConnectionError(f"failed for https://x/query?apikey={FAKE_KEY}")
        except ConnectionError:
            import sys

            record = logging.LogRecord("news", logging.ERROR, __file__, 1, "fetch failed", None, sys.exc_info())
        self.assertNotIn(FAKE_KEY, self._format(record))


if __name__ == "__main__":
    unittest.main()
