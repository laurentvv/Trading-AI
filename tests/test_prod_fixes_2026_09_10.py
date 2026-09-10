"""Tests for the 2026-09-10 PROD audit fixes (logs_prod J+8 analysis).

Covers four remediations:
  A — EIA brent_spot content-freshness gate + circuit breaker. The RBRTE
      series was stale AT SOURCE (latest period 2026-06-01, ~100d old) and
      re-cached every 6h with a fresh mtime; the stale spot leg made the
      Dated-vs-Futures spread read as an ever-deepening contango (-$5.71)
      while live Brent rallied — a fake bearish signal for OilBench.
  B — get_etf_data refuses stale-at-source downloads: on 2026-09-08/09 Yahoo
      kept serving CRUDP.PA bars ending 2026-09-04; the "successful" refresh
      was traded on for two days.
  C — Ratchet self-heal ADOPTS a standing broker stop instead of POSTing a
      duplicate (the standing stop reserves the shares -> 400
      selling-equity-not-owned, 6 identical ERRORs on 2026-09-03).
  D — A 400 selling-equity-not-owned on a SELL triggers an immediate position
      re-check; if the position is gone (stale READ endpoints — T212 demo
      served the pre-sale snapshot 30 min after the 2026-09-08 08:31 fill),
      the local state is reconciled at once and no stop is re-placed.
"""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src import data as data_mod
from src import eia_client as eia_mod
from src import t212_executor as t212


def _spot_rows(latest_days_ago: int, n: int = 5) -> list[dict]:
    """RBRTE-style rows whose NEWEST period is `latest_days_ago` days old."""
    today = pd.Timestamp.now().normalize()
    return [
        {
            "period": (today - pd.Timedelta(days=latest_days_ago + (n - 1 - i))).strftime("%Y-%m-%d"),
            "value": 90.0 + i,
        }
        for i in range(n)
    ]


class TestBrentSpotFreshnessGate(unittest.TestCase):
    """Fix A — stale brent_spot payloads are refused on every path."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        cache_dir = Path(self._tmp.name) / "eia"
        cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache_dir_patcher = patch.object(eia_mod, "EIA_CACHE_DIR", cache_dir)
        self._cache_dir_patcher.start()
        self.client = eia_mod.EIAClient()
        self.client.api_key = "test-key"

    def tearDown(self):
        self._cache_dir_patcher.stop()
        self._tmp.cleanup()

    def test_stale_source_payload_is_refused_and_not_cached(self):
        # ~100d-old latest period (the 2026-09-10 PROD condition: 2026-06-01).
        with patch.object(self.client, "_make_request", return_value=_spot_rows(100)), \
             patch.object(eia_mod, "MAX_BRENT_SPOT_AGE_DAYS", 21):
            df = self.client.get_brent_spot_price(days=30)

        self.assertTrue(df.empty, "A stale-at-source brent_spot must be refused (empty df)")
        self.assertFalse(
            (eia_mod.EIA_CACHE_DIR / "eia_brent_spot.parquet").exists(),
            "A refused payload must NOT be cached — a fresh mtime would hide the staleness",
        )
        self.assertIsNotNone(getattr(self.client, "_brent_refused_at", None), "Refusal must arm the breaker")

    def test_circuit_breaker_skips_http_after_refusal(self):
        http = MagicMock(return_value=_spot_rows(100))
        with patch.object(self.client, "_make_request", http), \
             patch.object(eia_mod, "MAX_BRENT_SPOT_AGE_DAYS", 21):
            self.client.get_brent_spot_price(days=30)   # refusal -> breaker armed
            self.client.get_brent_spot_price(days=30)   # breaker active -> no HTTP

        self.assertEqual(http.call_count, 1, "The breaker must skip the HTTP call after a content refusal")

    def test_stale_cached_payload_is_refused_too(self):
        # Simulate a cache entry written when the source was fresh, served
        # later while the source is stale: the gate must refuse cache hits.
        stale_df = pd.DataFrame(_spot_rows(100))
        with patch.object(self.client, "_get_from_cache", return_value=stale_df), \
             patch.object(eia_mod, "MAX_BRENT_SPOT_AGE_DAYS", 21):
            df = self.client.get_brent_spot_price(days=30)

        self.assertTrue(df.empty, "A stale brent_spot cache hit must be refused as well")

    def test_fresh_payload_is_returned_and_cached(self):
        http = MagicMock(return_value=_spot_rows(2))
        with patch.object(self.client, "_make_request", http):
            df = self.client.get_brent_spot_price(days=30)

        self.assertFalse(df.empty)
        self.assertEqual(len(df), 5)
        self.assertTrue(
            (eia_mod.EIA_CACHE_DIR / "eia_brent_spot.parquet").exists(),
            "A fresh payload must be cached as before",
        )


class TestStaleAtSourceDownloadRefusal(unittest.TestCase):
    """Fix B — a SUCCESSFUL download whose last bar is too old is refused."""

    @staticmethod
    def _frame(last_bar_days_ago: int) -> pd.DataFrame:
        end = pd.Timestamp.now().normalize() - pd.Timedelta(days=last_bar_days_ago)
        idx = pd.date_range(end=end, periods=10, freq="D")
        close = pd.Series([100.0 + i for i in range(10)], index=idx)
        return pd.DataFrame(
            {"Open": close, "High": close, "Low": close, "Close": close, "Volume": [1000.0] * 10},
            index=idx,
        )

    def test_successful_download_with_stale_content_raises(self):
        # 2026-09-08 condition: Yahoo "successfully" served CRUDP.PA ending
        # 4+ days back while the market had moved on.
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(data_mod, "CACHE_DIR", Path(tmp)), \
                 patch.object(data_mod, "_yf_download", side_effect=lambda *a, **k: self._frame(5)), \
                 patch.object(data_mod, "_inject_t212_live_price", side_effect=lambda h, t: h):
                with self.assertRaises(ValueError) as ctx:
                    data_mod.get_etf_data(ticker="TEST.PA", force_refresh=True)
        self.assertIn("Stale-at-source", str(ctx.exception))

    def test_fresh_download_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(data_mod, "CACHE_DIR", Path(tmp)), \
                 patch.object(data_mod, "_yf_download", side_effect=lambda *a, **k: self._frame(0)), \
                 patch.object(data_mod, "_inject_t212_live_price", side_effect=lambda h, t: h):
                hist, _ = data_mod.get_etf_data(ticker="TEST.PA", force_refresh=True)
        self.assertFalse(hist.empty)


def _ratchet_state() -> dict:
    return {
        "initial_budget": 1000.0,
        "current_capital": 1000.0,
        "active_position": {
            "ticker": "SXRVd_EQ",
            "quantity": 0.1982,
            "buy_budget": 289.0,
            "entry_price_etf": 1458.78,
            "entry_time": "2026-08-20T10:00:00",
            # peak ~1456.8 -> desired ratchet ~1311.1 < adopted stop 1312.90,
            # so adoption must NOT trigger a cancel-and-replace afterwards.
            "highest_value": 288.74,
        },
    }


def _ratchet_pos() -> dict:
    return {
        "quantity": 0.1982,
        "quantityAvailableForTrading": 0.0,  # reserved by the standing stop
        "averagePricePaid": 1458.779,
        "walletImpact": {"currentValue": 288.74},
    }


class TestRatchetSelfHealAdoptsStandingStop(unittest.TestCase):
    """Fix C — the self-heal adopts an existing stop instead of duplicating it."""

    def test_standing_stop_is_adopted_without_duplicate_post(self):
        state = _ratchet_state()
        standing = {"id": 54250294524, "stopPrice": 1312.90, "type": "STOP", "side": "SELL", "status": "WORKING"}
        saved = []
        with patch.object(t212, "_get_active_stop_order", return_value=standing), \
             patch.object(t212, "_place_stop_order") as place, \
             patch.object(t212, "_cancel_order") as cancel, \
             patch.object(t212, "save_portfolio_state", side_effect=lambda s, t: saved.append((s, t))):
            t212._ratchet_stop_order(state, _ratchet_pos(), "SXRVd_EQ", headers={})

        pos = state["active_position"]
        self.assertEqual(pos["stop_order_id"], 54250294524, "The standing broker stop must be adopted")
        self.assertEqual(pos["stop_price"], 1312.90)
        self.assertEqual(place.call_count, 0, "A duplicate stop POST would be refused selling-equity-not-owned")
        self.assertEqual(cancel.call_count, 0)
        self.assertEqual(len(saved), 1, "The adopted stop must be persisted to the state")

    def test_no_standing_stop_still_places_one(self):
        state = _ratchet_state()
        with patch.object(t212, "_get_active_stop_order", return_value=None), \
             patch.object(t212, "_place_stop_order", return_value=(111, 1312.90)) as place, \
             patch.object(t212, "save_portfolio_state") as save:
            t212._ratchet_stop_order(state, _ratchet_pos(), "SXRVd_EQ", headers={})

        place.assert_called_once()
        save.assert_called_once()
        self.assertEqual(state["active_position"]["stop_order_id"], 111)


def _sell_400_response() -> MagicMock:
    resp = MagicMock()
    resp.status_code = 400
    resp.text = (
        '{"type":"/api-errors/selling-equity-not-owned","title":"Error while placing the order",'
        '"status":400,"detail":"Selling more equities than owned, owned: 0.0"}'
    )
    return resp


class TestFailedSell400Reconciliation(unittest.TestCase):
    """Fix D — 400 selling-equity-not-owned means the READ endpoints may be stale."""

    def test_position_gone_reconciles_state_and_skips_reprotection(self):
        # 2026-09-08 09:01 condition: shares already sold, reads served the
        # pre-sale snapshot, order endpoint refused with owned: 0.0.
        state = _ratchet_state()
        state["active_position"]["stop_order_id"] = None
        state["active_position"]["stop_price"] = 1312.90
        with patch.object(t212, "_position_exists", return_value=False), \
             patch.object(t212, "_place_stop_order") as place, \
             patch.object(t212, "save_portfolio_state") as save:
            t212._handle_failed_sell(
                _sell_400_response(), False, "SXRVd_EQ", 0.1982,
                stop_released=True, prev_stop_price=1312.90, headers={}, state=state,
            )

        self.assertIsNone(state["active_position"], "State must be reconciled immediately")
        self.assertEqual(place.call_count, 0, "No stop to re-place — the position is gone")
        save.assert_called_once()

    def test_position_still_there_falls_through_to_reprotection(self):
        state = _ratchet_state()
        with patch.object(t212, "_position_exists", return_value=True), \
             patch.object(t212, "_place_stop_order", return_value=(999, 1312.90)) as place, \
             patch.object(t212, "save_portfolio_state"):
            t212._handle_failed_sell(
                _sell_400_response(), False, "SXRVd_EQ", 0.1982,
                stop_released=True, prev_stop_price=1312.90, headers={}, state=state,
            )

        self.assertIsNotNone(state["active_position"], "A live position must NOT be reset on a 400")
        place.assert_called_once()
        self.assertEqual(state["active_position"]["stop_order_id"], 999)

    def test_non_400_error_does_not_trigger_reconciliation(self):
        state = _ratchet_state()
        resp = MagicMock()
        resp.status_code = 500
        resp.text = "boom"
        with patch.object(t212, "_position_exists") as exists, \
             patch.object(t212, "_place_stop_order", return_value=(999, 1312.90)), \
             patch.object(t212, "save_portfolio_state"):
            t212._handle_failed_sell(
                resp, False, "SXRVd_EQ", 0.1982,
                stop_released=True, prev_stop_price=1312.90, headers={}, state=state,
            )

        self.assertEqual(exists.call_count, 0, "Only selling-equity-not-owned triggers the stale-read check")
        self.assertIsNotNone(state["active_position"])


if __name__ == "__main__":
    unittest.main()
