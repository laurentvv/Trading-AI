"""Unit tests for Section 2.2 Stability & Resilience fixes (Audit PLAN.md).

Tests cover:
1. schedule.py subprocess timeout (2700s on trading cycle, 1800s on morning brief).
2. requests.get timeout parameter in src/data.py (Alpha Vantage).
3. Monday morning freshness: Friday's close is considered fresh on Monday morning (1 business day),
   while Friday's close is correctly refused on Tuesday (2 business days / stale at source).
"""

import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path
import subprocess
import pandas as pd
import sys

# Ensure src is on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import schedule
import src.data as data_mod


class TestSchedulerTimeouts(unittest.TestCase):
    """Verify subprocess timeouts are enforced in schedule.py."""

    @patch("src.llm_client.check_ai_health", return_value=True)
    @patch("schedule.subprocess.run")
    def test_run_trading_cycle_carries_timeout_and_handles_expiry(self, mock_run, mock_ai):
        """run_trading_cycle must pass timeout=2700 and catch TimeoutExpired without crashing."""
        mock_run.side_effect = subprocess.TimeoutExpired(cmd=["uv", "run", "main.py"], timeout=2700)

        # Must not raise
        schedule.run_trading_cycle()

        mock_run.assert_called_once()
        _, kwargs = mock_run.call_args
        self.assertEqual(kwargs.get("timeout"), 2700)

    @patch("schedule.subprocess.run")
    def test_run_morning_brief_carries_timeout_and_handles_expiry(self, mock_run):
        """run_morning_brief must pass timeout=1800 and catch TimeoutExpired without crashing."""
        mock_run.side_effect = subprocess.TimeoutExpired(cmd=["uv", "run", "morning_brief"], timeout=1800)

        # Must not raise
        schedule.run_morning_brief()

        mock_run.assert_called()
        # Find the call with morning_brief.py
        brief_call = [c for c in mock_run.call_args_list if any("morning_brief.py" in str(arg) for arg in c[0][0])]
        self.assertTrue(len(brief_call) > 0)
        self.assertEqual(brief_call[0][1].get("timeout"), 1800)


class TestNetworkTimeout(unittest.TestCase):
    """Verify raw requests in data.py carry timeouts."""

    @patch("src.data.ALPHA_VANTAGE_API_KEY", "test_key")
    @patch("src.data.requests.get")
    def test_alpha_vantage_requests_carries_timeout(self, mock_get, mock_key=None):
        """get_alpha_vantage_data must pass timeout to requests.get."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"Error Message": "Test mock"}
        mock_get.return_value = mock_resp

        data_mod.get_alpha_vantage_data("TREASURY_YIELD", "10year", force_refresh=True)

        mock_get.assert_called_once()
        _, kwargs = mock_get.call_args
        self.assertIn("timeout", kwargs)
        self.assertEqual(kwargs["timeout"], 20)


class TestPriceDataStalenessCalendarVsBusinessDays(unittest.TestCase):
    """Verify business-day aware staleness logic: Friday close is fresh on Monday, stale on Tuesday."""

    def test_friday_close_is_fresh_on_monday_morning(self):
        """Friday close (2026-09-04 00:00:00) evaluated on Monday morning (2026-09-07 08:30:00) must be FRESH."""
        friday_close = pd.Timestamp("2026-09-04 00:00:00")
        monday_morning = pd.Timestamp("2026-09-07 08:30:00")

        is_stale = data_mod._is_price_stale(last_date=friday_close, now=monday_morning)
        self.assertFalse(is_stale, "Friday close must be accepted on Monday morning (1 business day old)")

    def test_friday_close_is_stale_on_tuesday(self):
        """Friday close (2026-09-04 00:00:00) evaluated on Tuesday (2026-09-08 09:00:00) must be STALE."""
        friday_close = pd.Timestamp("2026-09-04 00:00:00")
        tuesday_now = pd.Timestamp("2026-09-08 09:00:00")

        is_stale = data_mod._is_price_stale(last_date=friday_close, now=tuesday_now)
        self.assertTrue(is_stale, "Friday close must be refused on Tuesday (2 business days old / stale at source)")

    def test_monday_close_is_fresh_on_tuesday(self):
        """Monday close evaluated on Tuesday morning must be FRESH."""
        monday_close = pd.Timestamp("2026-09-07 00:00:00")
        tuesday_now = pd.Timestamp("2026-09-08 09:00:00")

        is_stale = data_mod._is_price_stale(last_date=monday_close, now=tuesday_now)
        self.assertFalse(is_stale, "Monday close must be accepted on Tuesday")

    def test_old_data_always_stale(self):
        """Data older than 5 calendar days is unconditionally STALE."""
        old_bar = pd.Timestamp.now() - pd.Timedelta(days=8)
        self.assertTrue(data_mod._is_price_stale(old_bar))


if __name__ == "__main__":
    unittest.main()
