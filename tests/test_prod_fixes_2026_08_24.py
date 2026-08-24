"""Fixes from the 2026-08-24 PROD audit (logs_prod + Cockpit MAXIX).

Incidents fixed and tested here:
  C1 — sell POSTed with quantity 0 while a GTC stop reserved the shares
       (2026-08-20 09:37: 400 "Quantity is missing"). The stop is now
       released BEFORE the market sell and re-placed if the sale fails.
  C2 — _confirm_fill(SELL) could return the BUY fill (same |quantity|),
       recording the entry price as the sale price and zeroing the P&L;
       the fill price is now cross-checked against the cash actually moved.
  C3 — get_t212_positions/get_t212_order_history swallowed errors as empty,
       producing phantom "no position" states and FIFO realized=0 resets.
  C4 — the soft win-rate penalty suppressed models on statistically
       meaningless samples (one mis-recorded round-trip zeroed 6 models).
  C5 — council context SQL targeted tables that never existed
       (model_predictions / daily_metrics / system_alerts).
  C6 — Alpha Vantage stderr echoed a real API key into the logs.
  C7 — EIA crude_imports retried a stale payload every cycle (~226x/5 days).

All network access is mocked at `src.t212_executor._t212_session.request`
(same interception point as tests/test_t212_orders.py).
"""

import json
import sqlite3
import sys
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))


def _resp(status, payload):
    r = MagicMock()
    r.status_code = status
    r.text = json.dumps(payload) if payload is not None else ""
    r.json.return_value = payload
    return r


def _pos(ticker="X_EQ", qty=5.0, avg=100.0, value=None, available=None):
    p = {
        "instrument": {"ticker": ticker},
        "quantity": qty,
        "quantityAvailableForTrading": available if available is not None else qty,
        "averagePricePaid": avg,
        "currentPrice": avg,
        "walletImpact": {"currentValue": value if value is not None else qty * avg},
    }
    return p


class _Router:
    """Dispatch mocked session.request calls to per-endpoint queues."""

    def __init__(self, positions=None, market=None, stop=None, history=None, delete=None):
        self.positions = list(positions or [])
        self.market = list(market or [])
        self.stop = list(stop or [])
        self.history = list(history or [])
        self.delete = list(delete or [])
        self.calls = []

    def __call__(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        if method == "GET" and url.endswith("/equity/positions"):
            return self.positions.pop(0) if self.positions else _resp(200, [])
        if method == "POST" and url.endswith("/equity/orders/market"):
            item = self.market.pop(0)
            if isinstance(item, Exception):
                raise item
            return item
        if method == "POST" and url.endswith("/equity/orders/stop"):
            item = self.stop.pop(0)
            if isinstance(item, Exception):
                raise item
            return item
        if method == "GET" and "/equity/history/orders" in url:
            return self.history.pop(0) if self.history else _resp(200, {"items": []})
        if method == "DELETE" and "/equity/orders/" in url:
            return self.delete.pop(0) if self.delete else _resp(204, None)
        return _resp(200, {})

    def count(self, method, suffix=""):
        return sum(1 for m, u, _ in self.calls if m == method and u.endswith(suffix))


@patch("time.sleep", MagicMock())
class TestSellReleasesStopReservedShares(unittest.TestCase):
    """C1 — the 2026-08-20 09:37 'Quantity is missing' incident."""

    def _state(self):
        return {
            "current_capital": 286.02,
            "total_realized_pl": 0.0,
            "active_position": {
                "ticker": "X_EQ",
                "quantity": 0.197,
                "buy_budget": 284.31,
                "entry_price_etf": 1443.198,
                "entry_time": "2026-08-19T16:25:39",
                "stop_order_id": 53600390467,
                "stop_price": 1298.88,
            },
        }

    @patch("src.t212_executor._t212_session.request")
    def test_sell_cancels_stop_first_and_uses_total_quantity(self, mock_request):
        from src.t212_executor import _execute_sell_order

        # The standing GTC stop reserves the shares: available=0, total=0.197.
        current_pos = _pos(qty=0.197, avg=1443.198, value=286.02, available=0.0)
        state = self._state()
        router = _Router(
            positions=[_resp(200, [current_pos])],  # existed_before check
            market=[_resp(201, {"id": 77})],
            history=[
                _resp(200, {"items": [
                    {"order": {"status": "FILLED", "side": "SELL"}, "fill": {"quantity": -0.197, "price": 1451.2}},
                ]}),
            ],
            delete=[_resp(204, None)],
        )
        mock_request.side_effect = router

        with patch("src.t212_executor.save_portfolio_state"), \
             patch("src.t212_executor.insert_transaction"), \
             patch("src.t212_executor._update_feedback_loop"):
            _execute_sell_order(
                state, current_pos, "SXRV.DE", "X_EQ",
                "https://x", {}, "2026-08-20 09:52:49", "TEST", force_stop_loss=True,
            )

        # The market order carried the TOTAL quantity, not the reserved 0.
        market_calls = [(m, u, kw) for m, u, kw in router.calls if m == "POST" and u.endswith("/market")]
        self.assertEqual(len(market_calls), 1)
        self.assertAlmostEqual(market_calls[0][2]["json"]["quantity"], -0.197, places=6)

        # The stop was cancelled BEFORE the market POST (call ordering).
        delete_idx = [i for i, (m, u, kw) in enumerate(router.calls) if m == "DELETE"]
        market_idx = [i for i, (m, u, kw) in enumerate(router.calls) if m == "POST" and u.endswith("/market")]
        self.assertTrue(delete_idx, "stop pre-cancellation missing")
        self.assertLess(delete_idx[0], market_idx[0])

        # Realized P&L recorded from the true fill (1451.2 - 1443.198) * 0.197.
        self.assertAlmostEqual(state["total_realized_pl"], (1451.2 - 1443.198) * 0.197, places=4)
        self.assertIsNone(state["active_position"])

    @patch("src.t212_executor._t212_session.request")
    def test_failed_sell_after_release_replaces_the_stop(self, mock_request):
        from src.t212_executor import _execute_sell_order

        current_pos = _pos(qty=0.197, avg=1443.198, value=286.02, available=0.0)
        state = self._state()
        router = _Router(
            positions=[_resp(200, [current_pos])],
            market=[_resp(400, {"detail": "Quantity is missing"})],  # sale rejected
            delete=[_resp(204, None)],   # stop released
            stop=[_resp(201, {"id": 53600400000})],  # emergency re-placement
        )
        mock_request.side_effect = router

        with patch("src.t212_executor.save_portfolio_state"), \
             patch("src.t212_executor.insert_transaction"):
            _execute_sell_order(
                state, current_pos, "SXRV.DE", "X_EQ",
                "https://x", {}, "2026-08-20 09:37:06", "TEST", force_stop_loss=True,
            )

        # Position still open, stop re-placed at the previous level.
        self.assertIsNotNone(state["active_position"])
        self.assertEqual(state["active_position"]["stop_order_id"], 53600400000)
        self.assertAlmostEqual(state["active_position"]["stop_price"], 1298.88, places=2)
        stop_calls = [kw for m, u, kw in router.calls if m == "POST" and u.endswith("/orders/stop")]
        self.assertEqual(len(stop_calls), 1)
        self.assertAlmostEqual(stop_calls[0]["json"]["stopPrice"], 1298.88, places=2)


@patch("time.sleep", MagicMock())
class TestConfirmFillSellSide(unittest.TestCase):
    """C2 — never confirm a SELL from a BUY-side fill."""

    @patch("src.t212_executor._t212_session.request")
    def test_buy_fill_cannot_confirm_a_sell(self, mock_request):
        from src.t212_executor import _confirm_fill

        # Exact 2026-08-20 shape: only the BUY (same |qty|) is FILLED in
        # history while the SELL has not landed yet.
        router = _Router(history=[
            _resp(200, {"items": [
                {"order": {"status": "FILLED", "side": "BUY"}, "fill": {"quantity": 0.197, "price": 1443.198}},
            ]}),
        ])
        mock_request.side_effect = router
        result = _confirm_fill("X_EQ", {}, side="SELL", expected_qty=0.197)
        self.assertIsNone(result)

    @patch("src.t212_executor._t212_session.request")
    def test_cash_delta_overrides_incoherent_history_price(self, mock_request):
        from src.t212_executor import _reconcile_sell_fill_price

        # History says 1443.20 (entry echo) but cash moved at ~1451.2.
        summary = _resp(200, {"cash": {"availableToTrade": 2001.58}})
        router = _Router()
        router.calls  # not used; summary served by get_t212_account_summary
        with patch("src.t212_executor.get_t212_account_summary", return_value=summary.json()):
            price = _reconcile_sell_fill_price(cash_before=1715.69, fill_qty=0.197, fill_price=1443.20)
        self.assertAlmostEqual(price, (2001.58 - 1715.69) / 0.197, places=4)
        self.assertGreater(price, 1450.0)

    @patch("src.t212_executor._t212_session.request")
    def test_coherent_history_price_is_kept(self, mock_request):
        from src.t212_executor import _reconcile_sell_fill_price

        summary = _resp(200, {"cash": {"availableToTrade": 1220.0}})
        with patch("src.t212_executor.get_t212_account_summary", return_value=summary.json()):
            price = _reconcile_sell_fill_price(cash_before=1000.0, fill_qty=5.0, fill_price=44.0)
        self.assertEqual(price, 44.0)  # cash delta 220 == 5*44 -> coherent


class TestUnknownBrokerState(unittest.TestCase):
    """C3 — a failed fetch must mean UNKNOWN, never 'no position'."""

    @patch("src.t212_executor._t212_session.request")
    def test_positions_error_returns_none(self, mock_request):
        from src.t212_executor import get_t212_positions

        mock_request.return_value = _resp(429, {"errorMessage": "too many requests"})
        self.assertIsNone(get_t212_positions())

    @patch("src.t212_executor._t212_session.request")
    def test_positions_network_error_returns_none(self, mock_request):
        import requests as _rq

        from src.t212_executor import get_t212_positions

        mock_request.side_effect = _rq.exceptions.ConnectionError("boom")
        self.assertIsNone(get_t212_positions())

    @patch("src.t212_executor.get_t212_positions", return_value=None)
    def test_sync_returns_none_when_positions_unavailable(self, _mock):
        from src.t212_executor import sync_state_from_t212

        self.assertIsNone(sync_state_from_t212("X_EQ"))

    @patch("src.t212_executor.get_t212_order_history", return_value=None)
    @patch("src.t212_executor.get_t212_positions", return_value=[])
    def test_sync_carries_realized_when_history_unavailable(self, _p, _h):
        from src.t212_executor import sync_state_from_t212

        with tempfile.TemporaryDirectory() as tmp:
            state_file = Path(tmp) / "state.json"
            state_file.write_text(json.dumps({"tickers": {"X_EQ": {"total_realized_pl": 3.5}}}), encoding="utf-8")
            with patch("src.t212_executor.STATE_FILE", str(state_file)):
                state = sync_state_from_t212("X_EQ")
        self.assertIsNotNone(state)
        self.assertAlmostEqual(state["total_realized_pl"], 3.5, places=4)

    def test_get_portfolio_info_flags_unknown_positions(self):
        from src.t212_executor import _get_portfolio_info

        with patch("src.t212_executor.safe_request") as mock_safe:
            mock_safe.side_effect = [_resp(200, {"cash": {"availableToTrade": 100.0}}), None]
            info = _get_portfolio_info("https://x", {})
        self.assertTrue(info["cash_ok"])
        self.assertFalse(info["positions_ok"])
        self.assertEqual(info["positions"], [])

    def test_execute_aborts_when_positions_unknown(self):
        from src.t212_executor import execute_t212_trade

        bad_portfolio = {"cash": 0.0, "positions": [], "cash_ok": False, "positions_ok": False}
        with patch("src.t212_executor.load_portfolio_state", return_value={}), \
             patch("src.t212_executor._get_portfolio_info", return_value=bad_portfolio), \
             patch("src.t212_executor.get_auth_header", return_value={}), \
             patch("src.t212_executor._execute_buy_order") as mock_buy:
            execute_t212_trade("BUY", 0.5, ticker="CRUDP.PA")
        mock_buy.assert_not_called()


class TestWinRateMinSamples(unittest.TestCase):
    """C4 — no weight suppression below WIN_RATE_MIN_SAMPLES."""

    def _perf(self, n_obs, win_rate=0.0):
        from src.adaptive_weight_manager import ModelPerformance

        return ModelPerformance(
            model_name="llm_text", accuracy=0.4, precision=0.4, recall=0.4,
            f1_score=0.4, sharpe_ratio=0.0, win_rate=win_rate, avg_return=0.0,
            volatility=0.01, max_drawdown=0.1, last_updated=datetime.now(),
            n_observations=n_obs,
        )

    def test_penalty_skipped_on_small_sample(self):
        from src.adaptive_weight_manager import AdaptiveWeightManager

        wm = object.__new__(AdaptiveWeightManager)
        wm.base_weights = {"llm_text": 0.5, "classic": 0.5}
        weights = wm._apply_soft_win_rate_penalties(
            {"llm_text": 0.5, "classic": 0.5}, {"llm_text": self._perf(n_obs=8)}
        )
        self.assertAlmostEqual(weights["llm_text"], 0.5, places=6)

    def test_penalty_applies_on_sufficient_sample(self):
        from src.adaptive_weight_manager import AdaptiveWeightManager

        wm = object.__new__(AdaptiveWeightManager)
        wm.base_weights = {"llm_text": 0.5, "classic": 0.5}
        weights = wm._apply_soft_win_rate_penalties(
            {"llm_text": 0.5, "classic": 0.5}, {"llm_text": self._perf(n_obs=50)}
        )
        self.assertAlmostEqual(weights["llm_text"], 0.0, places=6)


class TestCouncilContextRealSchemas(unittest.TestCase):
    """C5 — council SQL must target the tables that actually exist."""

    def _build_dbs(self, tmp: Path):
        perf = sqlite3.connect(tmp / "model_performance.db")
        perf.execute(
            "CREATE TABLE model_performance_history (id INTEGER PRIMARY KEY, date TEXT, "
            "model_name TEXT, signal_predicted TEXT, actual_outcome INTEGER, return_1d REAL, "
            "return_5d REAL, confidence REAL, market_regime TEXT, created_at TEXT)"
        )
        today = datetime.now().strftime("%Y-%m-%d")
        perf.executemany(
            "INSERT INTO model_performance_history (date, model_name, signal_predicted, actual_outcome, return_1d) VALUES (?,?,?,?,?)",
            [
                (today, "classic", "SELL", -1, -0.01),      # correct
                (today, "llm_text", "HOLD", 0, 0.001),      # correct
                (today, "llm_visual", "BUY", -1, -0.01),    # wrong
            ],
        )
        perf.commit()
        perf.close()

        mon = sqlite3.connect(tmp / "performance_monitor.db")
        mon.execute(
            "CREATE TABLE daily_performance (date TEXT, ticker TEXT, starting_value REAL, "
            "ending_value REAL, daily_return REAL, benchmark_return REAL, trades_count INTEGER, "
            "wins INTEGER, losses INTEGER, max_intraday_drawdown REAL, volatility REAL)"
        )
        mon.execute(
            "INSERT INTO daily_performance VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (today, "CRUDP.PA", 1000.0, 1002.0, 0.002, 0.001, 1, 1, 0, 0.005, 0.01),
        )
        mon.execute(
            "CREATE TABLE performance_alerts (id INTEGER PRIMARY KEY, timestamp TEXT, ticker TEXT, "
            "alert_type TEXT, severity TEXT, message TEXT, model_name TEXT, metric_value REAL, "
            "threshold REAL, acknowledged INTEGER)"
        )
        mon.execute(
            "INSERT INTO performance_alerts (timestamp, ticker, alert_type, severity, message) VALUES (?,?,?,?,?)",
            (today + " 10:00:00", "CRUDP.PA", "DRAWDOWN", "WARNING", "drawdown 2%"),
        )
        mon.commit()
        mon.close()

    def test_fetch_model_performance_reads_real_table(self):
        from src.council import weekend_council as wc

        with tempfile.TemporaryDirectory() as tmp:
            self._build_dbs(Path(tmp))
            with patch.object(wc, "PERF_DB_PATH", Path(tmp) / "model_performance.db"), \
                 patch.object(wc, "MONITOR_DB_PATH", Path(tmp) / "performance_monitor.db"):
                perf_txt = wc.fetch_model_performance(days=7)
                mon_txt = wc.fetch_portfolio_monitoring()
        self.assertIn("Précision globale", perf_txt)
        self.assertIn("2/3", perf_txt)
        self.assertIn("Valeur finale=1002.00", mon_txt)
        self.assertIn("drawdown 2%", mon_txt)


class TestSecretRedaction(unittest.TestCase):
    """C6 — API keys echoed in stderr must be masked before logging."""

    def test_redacts_alpha_vantage_style_key(self):
        from src.enhanced_trading_example import _redact_secrets

        raw = (
            "Alpha Vantage rate-limit/notice for query 'oil': We have detected "
            "your API key as QBU3W4XGZXUO2MII and will throttle"
        )
        out = _redact_secrets(raw)
        self.assertNotIn("QBU3W4XGZXUO2MII", out)
        self.assertIn("***REDACTED***", out)
        self.assertIn("rate-limit/notice", out)

    def test_plain_text_untouched(self):
        from src.enhanced_trading_example import _redact_secrets

        raw = "Successfully fetched 10 news headlines for oil market"
        self.assertEqual(_redact_secrets(raw), raw)


class TestEiaCircuitBreaker(unittest.TestCase):
    """C7 — a stale crude_imports payload is not re-fetched every cycle."""

    def test_breaker_skips_http_after_refusal(self):
        import pandas as pd

        from src.eia_client import EIAClient

        client = EIAClient()
        stale = pd.DataFrame({
            "period": [pd.Timestamp("2026-05-01"), pd.Timestamp("2026-04-01"), pd.Timestamp("2026-03-01")],
            "quantity": [1.0, 2.0, 3.0],
        })
        with patch.object(client, "_make_request", wraps=client._make_request) as mock_req, \
             patch.object(client, "_get_from_cache", return_value=None), \
             patch.object(client, "_load_disk_cache_fallback", return_value=stale) as mock_fallback:
            mock_req.side_effect = None
            # Simulate the API answering with the stale payload.
            def _stale_fetch(path, params):
                return [
                    {"period": "2026-05-01", "quantity": 1.0},
                    {"period": "2026-04-01", "quantity": 2.0},
                    {"period": "2026-03-01", "quantity": 3.0},
                ]
            mock_req.side_effect = _stale_fetch
            first = client.get_crude_imports()
            self.assertEqual(mock_req.call_count, 1)

            second = client.get_crude_imports()  # breaker must skip the HTTP call
            self.assertEqual(mock_req.call_count, 1)
            self.assertEqual(mock_fallback.call_count, 1)
            self.assertEqual(len(first), 3)
            self.assertEqual(len(second), 3)


if __name__ == "__main__":
    unittest.main()
