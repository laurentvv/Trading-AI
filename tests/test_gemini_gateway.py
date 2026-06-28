"""Unit tests for the Gemini gateway and quota tracker.

Pattern mirrors ``tests/test_llm_client.py``: ``unittest.TestCase`` +
``unittest.mock.patch``/``MagicMock``. No real network calls — the SDK and
SQLite layer are both mocked at the module boundary.

Test coverage:
* Gateway enabling/disabling (no key, no SDK, key present).
* Quota pre-flight short-circuit (does not call the SDK when over budget).
* The three use cases (decide / analyze_chart / summarize_web_context).
* **Multi-model cascade failover**: when the first model fails (429/503/over-
  quota), the gateway transparently tries the next one in the cascade.
* Error handling (ResourceExhausted, generic Exception, empty response).
* ``_parse_decision`` robustness (markdown fences, prose prefix, etc.).
* ``QuotaTracker`` RPM/RPD windows and reset semantics.
"""

from __future__ import annotations

import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src import gemini_gateway as gw_mod
from src.gemini_gateway import (
    GeminiGateway,
    REASONING_CASCADE,
    VISION_CASCADE,
    WEB_SUMMARY_CASCADE,
)
from src.gemini_quota import LIMITS, QuotaTracker


def _make_response(text: str) -> MagicMock:
    """Build a fake ``generate_content`` response carrying ``.text``."""
    r = MagicMock()
    r.text = text
    return r


class _EnabledGatewayMixin:
    """Common setUp: gateway enabled with a fake genai Client mounted.

    The new ``google-genai`` SDK is client-based (``genai.Client(api_key=...)``
    then ``client.models.generate_content(...)``). We mount a fake ``genai``
    module so tests run regardless of whether the wheel is installed, and we
    expose ``self.genai`` / ``self.mock_generate`` to drive call assertions.
    """

    def setUp(self):  # noqa: D401 - unittest hook
        self.mock_generate = MagicMock()
        mock_models = MagicMock()
        mock_models.generate_content = self.mock_generate
        self.mock_client = MagicMock()
        self.mock_client.models = mock_models

        self._fake_genai = MagicMock()
        self._fake_genai.Client = MagicMock(return_value=self.mock_client)
        # genai_types.GenerateContentConfig is a real constructor in tests;
        # a MagicMock returning its kwargs works for our assertions.
        self._fake_types = MagicMock()

        self._ctx = patch.dict("os.environ", {"GEMINI_API_KEY": "fake-key"}, clear=False)
        self._ctx.__enter__()
        self._sdk_patch = patch.object(gw_mod, "_SDK_AVAILABLE", True)
        self._sdk_patch.__enter__()
        self._genai_patch = patch.object(gw_mod, "genai", self._fake_genai)
        self._genai_patch.__enter__()
        self._types_patch = patch.object(gw_mod, "genai_types", self._fake_types)
        self._types_patch.__enter__()

    def tearDown(self):  # noqa: D401 - unittest hook
        self._types_patch.__exit__(None, None, None)
        self._genai_patch.__exit__(None, None, None)
        self._sdk_patch.__exit__(None, None, None)
        self._ctx.__exit__(None, None, None)

    def _make_gateway(self):
        # Each gateway gets a fresh quota mock so tests are independent.
        g = GeminiGateway()
        g._quota = MagicMock()
        g._quota.check.return_value = True  # default: quota available
        return g


# --------------------------------------------------------------------------- #
# Enablement
# --------------------------------------------------------------------------- #
class TestGatewayEnablement(unittest.TestCase):
    """An absent key or SDK must disable the gateway, never raise."""

    def test_disabled_when_no_api_key(self):
        with patch.dict("os.environ", {}, clear=True):
            with patch("src.gemini_gateway.os.getenv", return_value=""):
                g = GeminiGateway()
        self.assertFalse(g.enabled)

    def test_enabled_when_key_and_sdk_present(self):
        fake_client = MagicMock()
        fake_genai = MagicMock()
        fake_genai.Client = MagicMock(return_value=fake_client)
        with patch.object(gw_mod, "_SDK_AVAILABLE", True):
            with patch.object(gw_mod, "genai", fake_genai):
                with patch.object(gw_mod, "genai_types", MagicMock()):
                    with patch("src.gemini_gateway.os.getenv", return_value="fake-key"):
                        g = GeminiGateway()
        self.assertTrue(g.enabled)
        self.assertIs(g._client, fake_client)
        fake_genai.Client.assert_called_once_with(api_key="fake-key")

    def test_bad_key_disables_gateway_without_raising(self):
        """A Client() that raises must not crash the pipeline."""
        fake_genai = MagicMock()
        fake_genai.Client = MagicMock(side_effect=ValueError("bad key"))
        with patch.object(gw_mod, "_SDK_AVAILABLE", True):
            with patch.object(gw_mod, "genai", fake_genai):
                with patch.object(gw_mod, "genai_types", MagicMock()):
                    with patch("src.gemini_gateway.os.getenv", return_value="x"):
                        g = GeminiGateway()
        self.assertFalse(g.enabled)


# --------------------------------------------------------------------------- #
# decide()
# --------------------------------------------------------------------------- #
class TestGatewayDecide(_EnabledGatewayMixin, unittest.TestCase):
    def test_decide_success_on_first_model(self):
        g = self._make_gateway()
        self.mock_generate.return_value = _make_response(
            json.dumps({"signal": "BUY", "confidence": 0.82, "analysis": "Bullish."})
        )
        result = g.decide("market context")
        self.assertEqual(result["signal"], "BUY")
        # Only the first model in REASONING_CASCADE should have been called.
        self.assertEqual(self.mock_generate.call_count, 1)
        used_model = self.mock_generate.call_args.kwargs["model"]
        self.assertEqual(used_model, REASONING_CASCADE[0])
        g._quota.record.assert_called_once_with(REASONING_CASCADE[0])

    def test_decide_failover_when_first_model_quota_exhausted(self):
        """The headline cascade behaviour: model[0] over quota → model[1] wins."""
        g = self._make_gateway()
        # model[0] over quota (check=False), model[1] allowed → succeeds.
        g._quota.check.side_effect = [False, True]
        self.mock_generate.return_value = _make_response(
            json.dumps({"signal": "HOLD", "confidence": 0.3, "analysis": "Flat."})
        )
        result = g.decide("ctx")
        self.assertEqual(result["signal"], "HOLD")
        # Only model[1] actually hit the API (model[0] was pre-flighted out).
        self.assertEqual(self.mock_generate.call_count, 1)
        used_model = self.mock_generate.call_args.kwargs["model"]
        self.assertEqual(used_model, REASONING_CASCADE[1])
        g._quota.record.assert_called_once_with(REASONING_CASCADE[1])

    def test_decide_failover_on_429(self):
        """model[0] returns 429 → model[1] succeeds."""
        g = self._make_gateway()
        self.mock_generate.side_effect = [
            gw_mod.ResourceExhausted("429"),  # model[0]
            _make_response(json.dumps({"signal": "SELL", "confidence": 0.6, "analysis": "Breakdown."})),  # model[1]
        ]
        result = g.decide("ctx")
        self.assertEqual(result["signal"], "SELL")
        self.assertEqual(self.mock_generate.call_count, 2)
        g._quota.record.assert_called_once_with(REASONING_CASCADE[1])

    def test_decide_failover_on_503_overload(self):
        """A transient 503 (model overloaded) also triggers the cascade."""
        g = self._make_gateway()
        self.mock_generate.side_effect = [
            RuntimeError("503 UNAVAILABLE"),  # model[0]
            _make_response(json.dumps({"signal": "BUY", "confidence": 0.7, "analysis": "ok"})),  # model[1]
        ]
        result = g.decide("ctx")
        self.assertEqual(result["signal"], "BUY")
        self.assertEqual(self.mock_generate.call_count, 2)

    def test_decide_returns_none_when_whole_cascade_exhausted(self):
        """Every model over quota → None (caller falls back to free-llm)."""
        g = self._make_gateway()
        g._quota.check.return_value = False  # all models blocked
        result = g.decide("ctx")
        self.assertIsNone(result)
        self.mock_generate.assert_not_called()
        g._quota.record.assert_not_called()

    def test_decide_returns_none_when_all_models_error(self):
        g = self._make_gateway()
        # Every model raises → exhausts the cascade.
        self.mock_generate.side_effect = RuntimeError("boom")
        result = g.decide("ctx")
        self.assertIsNone(result)
        self.assertEqual(self.mock_generate.call_count, len(REASONING_CASCADE))
        g._quota.record.assert_not_called()

    def test_decide_normalizes_signal_case_and_types(self):
        g = self._make_gateway()
        self.mock_generate.return_value = _make_response(
            json.dumps({"signal": "hold", "confidence": 0, "analysis": "Flat."})
        )
        result = g.decide("ctx")
        self.assertEqual(result["signal"], "HOLD")
        self.assertEqual(result["confidence"], 0.0)


# --------------------------------------------------------------------------- #
# analyze_chart()
# --------------------------------------------------------------------------- #
class TestGatewayAnalyzeChart(_EnabledGatewayMixin, unittest.TestCase):
    def test_analyze_chart_success(self):
        g = self._make_gateway()
        self.mock_generate.return_value = _make_response(
            json.dumps({"signal": "SELL", "confidence": 0.6, "analysis": "H&S."})
        )
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            img_path = Path(tmp.name)
        try:
            result = g.analyze_chart(img_path)
            self.assertEqual(result["signal"], "SELL")
            # Image was passed as a Part in contents.
            args = self.mock_generate.call_args.kwargs
            self.assertEqual(args["model"], VISION_CASCADE[0])
            self.assertTrue(len(args["contents"]) >= 2)  # prompt + image part
        finally:
            img_path.unlink(missing_ok=True)

    def test_analyze_chart_missing_file_returns_none(self):
        g = self._make_gateway()
        result = g.analyze_chart(Path("does_not_exist.png"))
        self.assertIsNone(result)
        self.mock_generate.assert_not_called()

    def test_analyze_chart_failover_on_429(self):
        """Vision cascade: model[0] 429 → model[1] succeeds."""
        g = self._make_gateway()
        self.mock_generate.side_effect = [
            gw_mod.ResourceExhausted("429"),
            _make_response(json.dumps({"signal": "HOLD", "confidence": 0.4, "analysis": "Sideways."})),
        ]
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            img_path = Path(tmp.name)
        try:
            result = g.analyze_chart(img_path)
            self.assertEqual(result["signal"], "HOLD")
            self.assertEqual(self.mock_generate.call_count, 2)
            self.assertEqual(
                self.mock_generate.call_args.kwargs["model"], VISION_CASCADE[1]
            )
        finally:
            img_path.unlink(missing_ok=True)


# --------------------------------------------------------------------------- #
# summarize_web_context()
# --------------------------------------------------------------------------- #
class TestGatewaySummarize(_EnabledGatewayMixin, unittest.TestCase):
    def test_summarize_success(self):
        g = self._make_gateway()
        self.mock_generate.return_value = _make_response("Summary: Fed hawkish.")
        result = g.summarize_web_context("long markdown...")
        self.assertEqual(result, "Summary: Fed hawkish.")
        self.assertEqual(self.mock_generate.call_count, 1)
        self.assertEqual(
            self.mock_generate.call_args.kwargs["model"], WEB_SUMMARY_CASCADE[0]
        )

    def test_summarize_empty_input_returns_none(self):
        g = self._make_gateway()
        self.assertIsNone(g.summarize_web_context(""))
        self.assertIsNone(g.summarize_web_context("   "))
        self.mock_generate.assert_not_called()


# --------------------------------------------------------------------------- #
# _parse_decision — regression tests for the live bug where Gemini wrapped
# valid JSON in ```json fences or prefixed it with prose, defeating a naive
# json.loads. These cases must NEVER silently fall back to the next backend.
# --------------------------------------------------------------------------- #
class TestParseDecision(unittest.TestCase):
    """Exercises Gemini output shapes observed in production."""

    _VALID = (
        '{\n  "signal": "SELL",\n  "confidence": 0.65,\n'
        '  "analysis": "The price action shows a breakdown."\n}'
    )

    def test_plain_indented_json(self):
        result = GeminiGateway._parse_decision(self._VALID)
        self.assertIsNotNone(result)
        self.assertEqual(result["signal"], "SELL")
        self.assertEqual(result["confidence"], 0.65)

    def test_markdown_fenced_json(self):
        text = "```json\n" + self._VALID + "\n```"
        result = GeminiGateway._parse_decision(text)
        self.assertIsNotNone(result)
        self.assertEqual(result["signal"], "SELL")

    def test_prose_prefixed_json(self):
        text = "Here is the analysis:\n" + self._VALID
        result = GeminiGateway._parse_decision(text)
        self.assertIsNotNone(result)
        self.assertEqual(result["signal"], "SELL")

    def test_plain_fences_without_json_tag(self):
        text = "```\n" + self._VALID + "\n```"
        result = GeminiGateway._parse_decision(text)
        self.assertIsNotNone(result)

    def test_empty_text_returns_none(self):
        self.assertIsNone(GeminiGateway._parse_decision(""))
        self.assertIsNone(GeminiGateway._parse_decision(None))

    def test_non_json_returns_none(self):
        self.assertIsNone(GeminiGateway._parse_decision("not json at all"))

    def test_signal_normalized_to_uppercase(self):
        text = '{"signal": "buy", "confidence": 0.9, "analysis": "x"}'
        result = GeminiGateway._parse_decision(text)
        self.assertEqual(result["signal"], "BUY")


# --------------------------------------------------------------------------- #
# QuotaTracker
# --------------------------------------------------------------------------- #
class TestQuotaTracker(unittest.TestCase):
    """Exercises the local RPM/RPD ledger with a temp DB and frozen time."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = Path(self.tmpdir) / "quota.db"
        self.tracker = QuotaTracker(db_path=self.db_path)

    def _set_limits(self, model: str, rpm: int, rpd: int):
        self._orig = LIMITS.get(model)
        LIMITS[model] = (rpm, rpd)

    def tearDown(self):
        for model in ("gemini-3.5-flash",):
            if hasattr(self, "_orig"):
                if self._orig is None:
                    LIMITS.pop(model, None)
                else:
                    LIMITS[model] = self._orig

    def test_unknown_model_is_always_allowed(self):
        self.assertTrue(self.tracker.check("some-future-model"))

    def test_record_then_check_blocks_at_rpm_limit(self):
        self._set_limits("gemini-3.5-flash", rpm=2, rpd=50)
        self.tracker.record("gemini-3.5-flash")
        self.tracker.record("gemini-3.5-flash")
        self.assertFalse(self.tracker.check("gemini-3.5-flash"))

    def test_rpm_window_releases_after_60s(self):
        self._set_limits("gemini-3.5-flash", rpm=1, rpd=50)
        self.tracker.record("gemini-3.5-flash")
        self.assertFalse(self.tracker.check("gemini-3.5-flash"))
        with patch("src.gemini_quota.time.time", return_value=time.time() + 61):
            self.assertTrue(self.tracker.check("gemini-3.5-flash"))

    def test_rpd_limit_blocks_even_with_rpm_headroom(self):
        self._set_limits("gemini-3.5-flash", rpm=1000, rpd=1)
        self.tracker.record("gemini-3.5-flash")
        self.assertFalse(self.tracker.check("gemini-3.5-flash"))

    def test_status_reports_counts_and_limits(self):
        self.tracker.record("gemini-3.5-flash")
        status = self.tracker.status()
        self.assertIn("gemini-3.5-flash", status)
        pro = status["gemini-3.5-flash"]
        self.assertEqual(pro["rpm_used"], 1)
        self.assertEqual(pro["rpd_used"], 1)
        self.assertEqual(pro["rpm_limit"], LIMITS["gemini-3.5-flash"][0])

    def test_reset_clears_ledger(self):
        self.tracker.record("gemini-3.5-flash")
        self.tracker.reset()
        status = self.tracker.status()
        self.assertEqual(status["gemini-3.5-flash"]["rpd_used"], 0)


if __name__ == "__main__":
    unittest.main()
