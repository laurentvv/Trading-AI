"""Unit tests for the two-tier Gemini gateway and quota tracker.

Pattern mirrors ``tests/test_llm_client.py``: ``unittest.TestCase`` +
``unittest.mock.patch``/``MagicMock``. No real network calls — the SDK and
SQLite layer are both mocked at the module boundary.

Test coverage:
* Two-tier routing: paid key (decision/vision) vs free key (summary, fallback).
* Multi-model cascade failover within a tier.
* Paid → free fallback when the paid cap is reached.
* Quota pre-flight short-circuit + local daily cap on the paid tier.
* ``_parse_decision`` robustness (markdown fences, prose prefix, malformed JSON
  salvage, confidence clamping).
* ``QuotaTracker`` RPM/RPD windows and migration from the legacy schema.
"""

from __future__ import annotations

import json
import sqlite3
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src import gemini_gateway as gw_mod
from src.gemini_gateway import (
    GeminiGateway,
    REASONING_CASCADE_PAID,
    REASONING_CASCADE_FREE,
    VISION_CASCADE_PAID,
    WEB_SUMMARY_CASCADE_FREE,
)
from src.gemini_quota import (
    LIMITS_FREE,
    LIMITS_PAID,
    QuotaTracker,
    TIER_FREE,
    TIER_PAID,
)


def _make_response(text: str) -> MagicMock:
    r = MagicMock()
    r.text = text
    return r


def _fake_client_with(generate_side_effect=None, generate_return=None):
    """Build a fake genai.Client whose models.generate_content behaves as set."""
    mock_generate = MagicMock()
    if generate_side_effect is not None:
        mock_generate.side_effect = generate_side_effect
    elif generate_return is not None:
        mock_generate.return_value = generate_return
    client = MagicMock()
    client.models.generate_content = mock_generate
    return client, mock_generate


def _mount_fake_sdk(paid_client, free_client):
    """Context-manager-free patcher: mounts a fake genai module that returns
    the given clients for the paid/free keys. Returns the patch objects to exit."""
    def fake_Client(api_key=None, **_):
        if api_key == "paid-key":
            return paid_client
        if api_key == "free-key":
            return free_client
        return MagicMock()  # unknown key

    fake_genai = MagicMock()
    fake_genai.Client = MagicMock(side_effect=fake_Client)
    fake_types = MagicMock()

    ctx = patch.dict("os.environ", {
        "GEMINI_API_KEY_PAY": "paid-key",
        "GEMINI_API_KEY": "free-key",
    }, clear=False)
    sdk = patch.object(gw_mod, "_SDK_AVAILABLE", True)
    genai_p = patch.object(gw_mod, "genai", fake_genai)
    types_p = patch.object(gw_mod, "genai_types", fake_types)
    return ctx, sdk, genai_p, types_p


class _TwoTierGatewayMixin:
    """Mixin: build a gateway with both paid+free clients mounted, each with a
    fresh quota mock. Subclasses set ``self.paid_generate`` / ``self.free_generate``
    in setUp to drive call behaviour."""

    def setUp(self):  # noqa: D401
        self.paid_client, self.paid_generate = _fake_client_with(generate_return=_make_response("{}"))
        self.free_client, self.free_generate = _fake_client_with(generate_return=_make_response("{}"))
        patches = _mount_fake_sdk(self.paid_client, self.free_client)
        self._patches = patches
        for p in patches:
            p.__enter__()
        g = GeminiGateway()
        # Fresh quota mock so tests are independent; default: quota available.
        g._quota = MagicMock()
        g._quota.check.return_value = True
        self.gw = g

    def tearDown(self):  # noqa: D401
        for p in reversed(self._patches):
            p.__exit__(None, None, None)


# --------------------------------------------------------------------------- #
# Enablement
# --------------------------------------------------------------------------- #
class TestGatewayEnablement(unittest.TestCase):
    def test_disabled_when_no_keys(self):
        with patch.dict("os.environ", {}, clear=True):
            with patch("src.gemini_gateway.os.getenv", return_value=""):
                g = GeminiGateway()
        self.assertFalse(g.enabled)

    def test_paid_only(self):
        fake_genai = MagicMock()
        fake_genai.Client = MagicMock(return_value=MagicMock())
        with patch.object(gw_mod, "_SDK_AVAILABLE", True), \
             patch.object(gw_mod, "genai", fake_genai), \
             patch.object(gw_mod, "genai_types", MagicMock()), \
             patch("src.gemini_gateway.os.getenv",
                   side_effect=lambda k, d="": "paid-key" if k == "GEMINI_API_KEY_PAY" else ""):
            g = GeminiGateway()
        self.assertTrue(g.enabled_paid)
        self.assertFalse(g.enabled_free)
        self.assertTrue(g.enabled)  # enabled if EITHER key present

    def test_free_only(self):
        fake_genai = MagicMock()
        fake_genai.Client = MagicMock(return_value=MagicMock())
        with patch.object(gw_mod, "_SDK_AVAILABLE", True), \
             patch.object(gw_mod, "genai", fake_genai), \
             patch.object(gw_mod, "genai_types", MagicMock()), \
             patch("src.gemini_gateway.os.getenv",
                   side_effect=lambda k, d="": "free-key" if k == "GEMINI_API_KEY" else ""):
            g = GeminiGateway()
        self.assertFalse(g.enabled_paid)
        self.assertTrue(g.enabled_free)
        self.assertTrue(g.enabled)

    def test_bad_paid_key_disables_paid_only(self):
        """A paid Client() that raises disables paid, not the whole gateway."""
        def client_factory(api_key=None, **_):
            if api_key == "paid-key":
                raise ValueError("bad paid key")
            return MagicMock()
        fake_genai = MagicMock()
        fake_genai.Client = MagicMock(side_effect=client_factory)
        with patch.object(gw_mod, "_SDK_AVAILABLE", True), \
             patch.object(gw_mod, "genai", fake_genai), \
             patch.object(gw_mod, "genai_types", MagicMock()), \
             patch("src.gemini_gateway.os.getenv",
                   side_effect=lambda k, d="": {"GEMINI_API_KEY_PAY": "paid-key",
                                                "GEMINI_API_KEY": "free-key"}.get(k, "")):
            g = GeminiGateway()
        self.assertFalse(g.enabled_paid)
        self.assertTrue(g.enabled_free)


# --------------------------------------------------------------------------- #
# decide() — paid tier routing + cascade + paid→free fallback
# --------------------------------------------------------------------------- #
class TestGatewayDecide(_TwoTierGatewayMixin, unittest.TestCase):
    def test_decide_uses_paid_pro_first(self):
        """Decision goes to the PAID tier's head model (gemini-2.5-pro)."""
        self.paid_generate.return_value = _make_response(
            json.dumps({"signal": "BUY", "confidence": 0.8, "analysis": "Bullish."})
        )
        result = self.gw.decide("ctx")
        self.assertEqual(result["signal"], "BUY")
        self.assertEqual(self.paid_generate.call_count, 1)
        self.assertEqual(self.paid_generate.call_args.kwargs["model"], REASONING_CASCADE_PAID[0])
        self.free_generate.assert_not_called()  # paid succeeded, free untouched
        self.gw._quota.record.assert_called_once_with(REASONING_CASCADE_PAID[0], TIER_PAID)

    def test_decide_paid_cascade_failover_on_429(self):
        """Within the PAID tier: Pro 429s → next paid model succeeds."""
        self.paid_generate.side_effect = [
            gw_mod.ResourceExhausted("429"),  # Pro
            _make_response(json.dumps({"signal": "SELL", "confidence": 0.6, "analysis": "Down."})),  # next
        ]
        result = self.gw.decide("ctx")
        self.assertEqual(result["signal"], "SELL")
        self.assertEqual(self.paid_generate.call_count, 2)
        self.free_generate.assert_not_called()

    def test_decide_falls_back_to_free_when_paid_exhausted(self):
        """Whole PAID cascade fails → FREE cascade succeeds."""
        self.paid_generate.side_effect = RuntimeError("boom")  # all paid models fail
        self.free_generate.return_value = _make_response(
            json.dumps({"signal": "HOLD", "confidence": 0.3, "analysis": "Flat."})
        )
        result = self.gw.decide("ctx")
        self.assertEqual(result["signal"], "HOLD")
        self.assertEqual(self.paid_generate.call_count, len(REASONING_CASCADE_PAID))
        self.assertEqual(self.free_generate.call_count, 1)
        self.assertEqual(self.free_generate.call_args.kwargs["model"], REASONING_CASCADE_FREE[0])
        self.gw._quota.record.assert_called_once_with(REASONING_CASCADE_FREE[0], TIER_FREE)

    def test_decide_paid_cap_reached_routes_to_free(self):
        """Local paid daily cap reached → paid skipped → free used."""
        # Quota check returns False for paid, True for free.
        self.gw._quota.check.side_effect = lambda model, tier=TIER_FREE: tier != TIER_PAID
        self.free_generate.return_value = _make_response(
            json.dumps({"signal": "BUY", "confidence": 0.7, "analysis": "ok"})
        )
        result = self.gw.decide("ctx")
        self.assertEqual(result["signal"], "BUY")
        self.paid_generate.assert_not_called()  # paid pre-flighted out
        self.assertEqual(self.free_generate.call_count, 1)

    def test_decide_returns_none_when_both_tiers_exhausted(self):
        self.paid_generate.side_effect = RuntimeError("boom")
        self.free_generate.side_effect = RuntimeError("boom")
        result = self.gw.decide("ctx")
        self.assertIsNone(result)


# --------------------------------------------------------------------------- #
# analyze_chart() — paid tier, cascade, paid→free fallback
# --------------------------------------------------------------------------- #
class TestGatewayAnalyzeChart(_TwoTierGatewayMixin, unittest.TestCase):
    def test_vision_uses_paid_first(self):
        self.paid_generate.return_value = _make_response(
            json.dumps({"signal": "SELL", "confidence": 0.6, "analysis": "H&S."})
        )
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            img_path = Path(tmp.name)
        try:
            result = self.gw.analyze_chart(img_path)
            self.assertEqual(result["signal"], "SELL")
            self.assertEqual(self.paid_generate.call_args.kwargs["model"], VISION_CASCADE_PAID[0])
            # Image was passed as a Part in contents.
            contents = self.paid_generate.call_args.kwargs["contents"]
            self.assertTrue(len(contents) >= 2)
        finally:
            img_path.unlink(missing_ok=True)

    def test_vision_missing_file_returns_none(self):
        result = self.gw.analyze_chart(Path("nope.png"))
        self.assertIsNone(result)
        self.paid_generate.assert_not_called()

    def test_vision_falls_back_to_free_when_paid_fails(self):
        self.paid_generate.side_effect = gw_mod.ResourceExhausted("429")
        self.free_generate.return_value = _make_response(
            json.dumps({"signal": "HOLD", "confidence": 0.4, "analysis": "Sideways."})
        )
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            img_path = Path(tmp.name)
        try:
            result = self.gw.analyze_chart(img_path)
            self.assertEqual(result["signal"], "HOLD")
            self.free_generate.assert_called()
        finally:
            img_path.unlink(missing_ok=True)


# --------------------------------------------------------------------------- #
# summarize_web_context() — FREE tier only (preserves paid budget)
# --------------------------------------------------------------------------- #
class TestGatewaySummarize(_TwoTierGatewayMixin, unittest.TestCase):
    def test_summary_uses_free_not_paid(self):
        """Web summary must NEVER touch the paid tier."""
        self.free_generate.return_value = _make_response("Fed hawkish summary.")
        result = self.gw.summarize_web_context("long markdown...")
        self.assertEqual(result, "Fed hawkish summary.")
        self.assertEqual(self.free_generate.call_count, 1)
        self.assertEqual(self.free_generate.call_args.kwargs["model"], WEB_SUMMARY_CASCADE_FREE[0])
        self.paid_generate.assert_not_called()  # budget-preserving invariant

    def test_summary_empty_input_returns_none(self):
        self.assertIsNone(self.gw.summarize_web_context(""))
        self.assertIsNone(self.gw.summarize_web_context("   "))
        self.paid_generate.assert_not_called()
        self.free_generate.assert_not_called()


# --------------------------------------------------------------------------- #
# _parse_decision + _normalize_decision robustness
# --------------------------------------------------------------------------- #
class TestParseDecision(unittest.TestCase):
    _VALID = (
        '{\n  "signal": "SELL",\n  "confidence": 0.65,\n'
        '  "analysis": "The price action shows a breakdown."\n}'
    )

    def test_plain_indented_json(self):
        r = GeminiGateway._parse_decision(self._VALID)
        self.assertEqual(r["signal"], "SELL")
        self.assertEqual(r["confidence"], 0.65)

    def test_markdown_fenced_json(self):
        r = GeminiGateway._parse_decision("```json\n" + self._VALID + "\n```")
        self.assertEqual(r["signal"], "SELL")

    def test_prose_prefixed_json(self):
        r = GeminiGateway._parse_decision("Here is the analysis:\n" + self._VALID)
        self.assertEqual(r["signal"], "SELL")

    def test_empty_text_returns_none(self):
        self.assertIsNone(GeminiGateway._parse_decision(""))
        self.assertIsNone(GeminiGateway._parse_decision(None))

    def test_non_json_returns_none(self):
        self.assertIsNone(GeminiGateway._parse_decision("not json at all"))

    def test_signal_normalized_to_uppercase(self):
        r = GeminiGateway._parse_decision('{"signal": "buy", "confidence": 0.9, "analysis": "x"}')
        self.assertEqual(r["signal"], "BUY")

    def test_malformed_json_salvages_signal_and_confidence(self):
        """Broken JSON (unescaped quotes in analysis) → regex salvage."""
        malformed = '{"signal": "HOLD", "confidence": 0.45, "analysis": "Shows a "head" pattern."}'
        r = GeminiGateway._parse_decision(malformed)
        self.assertIsNotNone(r)
        self.assertEqual(r["signal"], "HOLD")
        self.assertEqual(r["confidence"], 0.45)

    def test_confidence_clamped_from_0_10_scale(self):
        """Gemini 2.5 Pro sometimes emits confidence on 0-10; must rescale."""
        r = GeminiGateway._normalize_decision({"signal": "BUY", "confidence": 8.5, "analysis": "x"})
        self.assertAlmostEqual(r["confidence"], 0.85)

    def test_confidence_preserved_when_already_0_1(self):
        r = GeminiGateway._normalize_decision({"signal": "BUY", "confidence": 0.85, "analysis": "x"})
        self.assertAlmostEqual(r["confidence"], 0.85)

    def test_confidence_clamped_at_1(self):
        r = GeminiGateway._normalize_decision({"signal": "BUY", "confidence": 99, "analysis": "x"})
        self.assertEqual(r["confidence"], 1.0)


# --------------------------------------------------------------------------- #
# QuotaTracker — two tiers + paid daily cap + migration
# --------------------------------------------------------------------------- #
class TestQuotaTracker(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = Path(self.tmpdir) / "quota.db"
        self.tracker = QuotaTracker(db_path=self.db_path)

    def tearDown(self):
        # Restore any LIMITS mutations.
        for d in (LIMITS_PAID, LIMITS_FREE):
            pass  # LIMITS not mutated in most tests; _set_limits restores below.

    def _set_limits(self, table, model, rpm, rpd):
        orig = table.get(model)
        table[model] = (rpm, rpd)
        self.addCleanup(lambda: table.pop(model) if orig is None else table.__setitem__(model, orig))

    def test_unknown_model_allowed_on_both_tiers(self):
        self.assertTrue(self.tracker.check("future-model", TIER_PAID))
        self.assertTrue(self.tracker.check("future-model", TIER_FREE))

    def test_paid_tier_blocked_at_rpm_limit(self):
        self._set_limits(LIMITS_PAID, "gemini-2.5-pro", rpm=2, rpd=1000)
        self.tracker.record("gemini-2.5-pro", TIER_PAID)
        self.tracker.record("gemini-2.5-pro", TIER_PAID)
        self.assertFalse(self.tracker.check("gemini-2.5-pro", TIER_PAID))

    def test_paid_and_free_ledgers_are_independent(self):
        """A paid call must NOT count against the free tier, and vice versa."""
        self._set_limits(LIMITS_PAID, "gemini-2.5-pro", rpm=1, rpd=1000)
        self._set_limits(LIMITS_FREE, "gemini-2.5-pro", rpm=1, rpd=1000)
        self.tracker.record("gemini-2.5-pro", TIER_PAID)
        # Paid now full (rpm=1); free still has its own budget.
        self.assertFalse(self.tracker.check("gemini-2.5-pro", TIER_PAID))
        self.assertTrue(self.tracker.check("gemini-2.5-pro", TIER_FREE))

    def test_paid_daily_cap_blocks_all_paid_models(self):
        """The paid daily cap is global across paid models."""
        with patch("src.gemini_quota._paid_daily_cap", return_value=1):
            self.tracker.record("gemini-2.5-pro", TIER_PAID)
            # Any paid model now blocked by the cap, even one with quota left.
            self.assertFalse(self.tracker.check("gemini-3.5-flash", TIER_PAID))
            # Free tier unaffected.
            self.assertTrue(self.tracker.check("gemini-3.5-flash", TIER_FREE))

    def test_rpm_window_releases_after_60s(self):
        self._set_limits(LIMITS_FREE, "gemini-3.5-flash", rpm=1, rpd=500)
        self.tracker.record("gemini-3.5-flash", TIER_FREE)
        self.assertFalse(self.tracker.check("gemini-3.5-flash", TIER_FREE))
        with patch("src.gemini_quota.time.time", return_value=time.time() + 61):
            self.assertTrue(self.tracker.check("gemini-3.5-flash", TIER_FREE))

    def test_status_reports_both_tiers_and_paid_cap(self):
        self.tracker.record("gemini-2.5-pro", TIER_PAID)
        self.tracker.record("gemini-3.1-flash-lite", TIER_FREE)
        status = self.tracker.status()
        self.assertIn("paid", status)
        self.assertIn("free", status)
        self.assertEqual(status["paid"]["gemini-2.5-pro"]["rpd_used"], 1)
        self.assertEqual(status["free"]["gemini-3.1-flash-lite"]["rpd_used"], 1)
        self.assertEqual(status["_paid_cap"]["used"], 1)

    def test_migrates_legacy_schema_without_tier_column(self):
        """A pre-2-tier DB (no 'tier' column) must migrate cleanly."""
        today_ts = int(time.time())  # within today, so RPD counts it
        # Build a legacy table by hand.
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute("DROP TABLE calls")
            conn.execute(
                "CREATE TABLE calls (id INTEGER PRIMARY KEY, model TEXT, ts INTEGER)"
            )
            conn.execute(
                "INSERT INTO calls (model, ts) VALUES ('gemini-2.5-flash', ?)",
                (today_ts,),
            )
            conn.commit()
        # Re-instantiate; migration should add 'tier' and backfill as free.
        tracker = QuotaTracker(db_path=self.db_path)
        status = tracker.status()
        # Legacy row backfilled to free tier, and counted in today's RPD.
        self.assertEqual(status["free"]["gemini-2.5-flash"]["rpd_used"], 1)


if __name__ == "__main__":
    unittest.main()
