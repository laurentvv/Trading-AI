"""GO-gates 1-3 tests — order safety, broker stops, fill confirmation.

Audit 2026-08-19 (docs/AUDIT_PROD_INDEPENDANT_2026-08-19.md):
  C1 — POST d'ordre sans timeout + retry aveugle -> double achat possible.
  C2 — aucune protection stop/TP côté broker.
  C3 — fill jamais vérifié, prix signal enregistré au lieu du prix de fill.

All network access is mocked at `src.t212_executor._t212_session.request`
(the same interception point as tests/test_t212.py — Session.get/post funnel
into Session.request).
"""

import json
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import requests

sys.path.insert(0, str(Path(__file__).parent.parent))


def _resp(status, payload):
    r = MagicMock()
    r.status_code = status
    r.text = json.dumps(payload) if payload is not None else ""
    r.json.return_value = payload
    return r


def _pos(ticker="X_EQ", qty=5.0, avg=100.0, value=None):
    return {
        "instrument": {"ticker": ticker},
        "quantity": qty,
        "quantityAvailableForTrading": qty,
        "averagePricePaid": avg,
        "currentPrice": avg,
        "walletImpact": {"currentValue": value if value is not None else qty * avg},
    }


class _Router:
    """Dispatch mocked session.request calls to per-endpoint queues.

    Unknown/empty queues get safe defaults (empty positions list, 200 {}).
    Queue items are responses, or exceptions to raise.
    """

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
class TestPostOrderMarket(unittest.TestCase):
    """GO-gate 1 (C1): timeout on the order POST + idempotence-by-reconciliation."""

    def setUp(self):
        from src.t212_executor import post_order_market, ORDER_POST_TIMEOUT

        self.post_order_market = post_order_market
        self.order_timeout = ORDER_POST_TIMEOUT

    @patch("src.t212_executor._t212_session.request")
    def test_order_post_carries_timeout(self, mock_request):
        router = _Router(market=[_resp(201, {"id": 1})])
        mock_request.side_effect = router

        resp, reconciled = self.post_order_market({"ticker": "X_EQ", "quantity": 5}, {}, "X_EQ")

        self.assertFalse(reconciled)
        self.assertEqual(resp.status_code, 201)
        post_calls = [kw for m, u, kw in router.calls if m == "POST" and u.endswith("/equity/orders/market")]
        self.assertEqual(len(post_calls), 1)
        self.assertEqual(post_calls[0]["timeout"], self.order_timeout)

    @patch("src.t212_executor._t212_session.request")
    def test_no_repost_after_network_error_when_position_appeared(self, mock_request):
        # The response was lost BUT the broker shows the new position: the
        # order executed. A re-POST would double-buy (official docs: the
        # endpoint is not idempotent).
        router = _Router(
            positions=[
                _resp(200, []),                       # existed_before: no position
                _resp(200, [_pos(qty=5)]),            # exists_now: position!
            ],
            market=[requests.exceptions.Timeout("response lost")],
        )
        mock_request.side_effect = router

        resp, reconciled = self.post_order_market({"ticker": "X_EQ", "quantity": 5}, {}, "X_EQ")

        self.assertTrue(reconciled)
        self.assertIsNone(resp)
        self.assertEqual(router.count("POST", "/equity/orders/market"), 1)

    @patch("src.t212_executor._t212_session.request")
    def test_no_repost_after_network_error_when_sell_position_disappeared(self, mock_request):
        router = _Router(
            positions=[
                _resp(200, [_pos(qty=5)]),   # existed_before
                _resp(200, []),              # exists_now: gone -> sell executed
            ],
            market=[requests.exceptions.ConnectionError("reset")],
        )
        mock_request.side_effect = router

        resp, reconciled = self.post_order_market({"ticker": "X_EQ", "quantity": -5}, {}, "X_EQ")

        self.assertTrue(reconciled)
        self.assertIsNone(resp)
        self.assertEqual(router.count("POST", "/equity/orders/market"), 1)

    @patch("src.t212_executor._t212_session.request")
    def test_retry_when_reconciliation_shows_no_execution(self, mock_request):
        # Network error, no position appeared: the order did NOT land — a
        # retry (with a fresh reconciliation) is safe.
        router = _Router(
            positions=[_resp(200, []), _resp(200, [])],  # both checks: no position
            market=[requests.exceptions.Timeout("lost"), _resp(201, {"id": 2})],
        )
        mock_request.side_effect = router

        resp, reconciled = self.post_order_market({"ticker": "X_EQ", "quantity": 5}, {}, "X_EQ")

        self.assertFalse(reconciled)
        self.assertEqual(resp.status_code, 201)
        self.assertEqual(router.count("POST", "/equity/orders/market"), 2)

    @patch("src.t212_executor._t212_session.request")
    def test_rate_limit_retry_is_safe(self, mock_request):
        # 429 means the order was rejected (not executed): plain retry.
        router = _Router(market=[_resp(429, {}), _resp(201, {"id": 3})])
        mock_request.side_effect = router

        resp, reconciled = self.post_order_market({"ticker": "X_EQ", "quantity": 5}, {}, "X_EQ")

        self.assertFalse(reconciled)
        self.assertEqual(resp.status_code, 201)
        self.assertEqual(router.count("POST", "/equity/orders/market"), 2)


@patch("time.sleep", MagicMock())
class TestConfirmFill(unittest.TestCase):
    """GO-gate 3 (C3): a 2xx is 'accepted', not 'executed'."""

    def setUp(self):
        from src.t212_executor import _confirm_fill, FILL_CONFIRM_ATTEMPTS

        self.confirm_fill = _confirm_fill
        self.max_attempts = FILL_CONFIRM_ATTEMPTS

    @patch("src.t212_executor._t212_session.request")
    def test_buy_fill_returns_broker_position(self, mock_request):
        position = _pos(qty=9.5, avg=101.0)
        router = _Router(positions=[_resp(200, []), _resp(200, [position])])
        mock_request.side_effect = router

        result = self.confirm_fill("X_EQ", {}, side="BUY")

        self.assertEqual(result, position)

    @patch("src.t212_executor._t212_session.request")
    def test_buy_fill_times_out(self, mock_request):
        router = _Router()  # always empty positions
        mock_request.side_effect = router

        result = self.confirm_fill("X_EQ", {}, side="BUY")

        self.assertIsNone(result)
        self.assertEqual(router.count("GET", "/equity/positions"), self.max_attempts)

    @patch("src.t212_executor._t212_session.request")
    def test_sell_fill_from_history_prefers_matching_quantity(self, mock_request):
        good = {"order": {"status": "FILLED", "side": "SELL"}, "fill": {"quantity": -9.5, "price": 99.0}}
        older = {"order": {"status": "FILLED", "side": "SELL"}, "fill": {"quantity": -4.0, "price": 95.0}}
        router = _Router(history=[_resp(200, {"items": [older, good]})])
        mock_request.side_effect = router

        result = self.confirm_fill("X_EQ", {}, side="SELL", expected_qty=9.5)

        self.assertEqual(result, good["fill"])


@patch("time.sleep", MagicMock())
class TestExecuteBuyOrder(unittest.TestCase):
    """GO-gates 2+3 on the buy path: TP attached, fill-confirmed state, broker stop."""

    def _run_buy(self, mock_request, router):
        from src.t212_executor import _execute_buy_order

        state = {"current_capital": 1000.0, "active_position": None}
        with patch("src.t212_executor.get_real_price_eur", return_value=100.0), \
             patch("src.t212_executor.save_portfolio_state") as mock_save, \
             patch("src.t212_executor.insert_transaction") as mock_tx:
            _execute_buy_order(
                state, None, "CRUDP.PA", "X_EQ",
                {"cash": 2000.0, "positions": []}, "https://x", {}, "2026-08-19 10:00:00", "TEST", 1.0,
            )
        return state, mock_save, mock_tx

    @patch("src.t212_executor._t212_session.request")
    def test_buy_payload_has_take_profit_and_state_uses_fill_price(self, mock_request):
        filled = _pos(qty=9.5, avg=101.0, value=959.5)
        router = _Router(
            positions=[
                _resp(200, []),           # post_order_market existed_before
                _resp(200, [filled]),     # _confirm_fill
            ],
            market=[_resp(201, {"id": 10})],
            stop=[_resp(201, {"id": 77})],
        )
        mock_request.side_effect = router

        state, mock_save, mock_tx = self._run_buy(mock_request, router)

        # GO-gate 2: takeProfit attached to the market order (absolute price, 2 decimals)
        post_kwargs = [kw for m, u, kw in router.calls if m == "POST" and u.endswith("/equity/orders/market")][0]
        self.assertEqual(post_kwargs["json"]["takeProfit"], 108.0)

        # GO-gate 2: dedicated GTC stop placed at -10% of the REAL fill (101 * 0.9)
        stop_kwargs = [kw for m, u, kw in router.calls if m == "POST" and u.endswith("/equity/orders/stop")][0]
        self.assertEqual(stop_kwargs["json"]["stopPrice"], 90.9)
        self.assertEqual(stop_kwargs["json"]["timeValidity"], "GOOD_TILL_CANCEL")
        self.assertLess(stop_kwargs["json"]["quantity"], 0)

        # GO-gate 3: state built on the broker fill (101.0), not the signal price (100.0)
        pos = state["active_position"]
        self.assertEqual(pos["entry_price_etf"], 101.0)
        self.assertEqual(pos["stop_order_id"], 77)
        self.assertEqual(pos["stop_price"], 90.9)
        self.assertTrue(mock_save.called)

        # GO-gate 3: DB row records the real fill price
        tx_kwargs = mock_tx.call_args.kwargs
        self.assertEqual(tx_kwargs["price"], 101.0)
        self.assertEqual(tx_kwargs["cost"], 101.0 * 9.5)
        self.assertIn("Fill Confirmed", tx_kwargs["reason"])

    @patch("src.t212_executor._t212_session.request")
    def test_buy_unconfirmed_fill_writes_nothing(self, mock_request):
        router = _Router(
            positions=[
                _resp(200, []),  # existed_before
                # _confirm_fill polls stay empty -> never confirmed
            ],
            market=[_resp(201, {"id": 10})],
        )
        mock_request.side_effect = router

        state, mock_save, mock_tx = self._run_buy(mock_request, router)

        self.assertIsNone(state.get("active_position"))
        mock_save.assert_not_called()
        mock_tx.assert_not_called()

    @patch("src.t212_executor._t212_session.request")
    def test_buy_attachment_rejected_falls_back_to_bare_order(self, mock_request):
        filled = _pos(qty=9.5, avg=101.0)
        router = _Router(
            positions=[
                _resp(200, []),  # first post_order_market existed_before
                _resp(200, []),  # fallback post_order_market existed_before
                _resp(200, [filled]),
            ],
            market=[
                _resp(400, {"detail": "takeProfit not supported"}),  # attachment refused
                _resp(201, {"id": 11}),                              # bare retry accepted
            ],
            stop=[_resp(201, {"id": 78})],
        )
        mock_request.side_effect = router

        state, _, _ = self._run_buy(mock_request, router)

        market_posts = [kw for m, u, kw in router.calls if m == "POST" and u.endswith("/equity/orders/market")]
        self.assertEqual(len(market_posts), 2)
        self.assertNotIn("takeProfit", market_posts[1]["json"])
        self.assertEqual(state["active_position"]["stop_order_id"], 78)


@patch("time.sleep", MagicMock())
class TestExecuteSellOrder(unittest.TestCase):
    """GO-gate 3 on the sell path: real fill proceeds, stop order cleaned up."""

    @patch("src.t212_executor._t212_session.request")
    def test_sell_records_confirmed_fill_and_cancels_stop(self, mock_request):
        from src.t212_executor import _execute_sell_order

        current_pos = _pos(qty=5.0, avg=100.0, value=510.0)
        state = {
            "current_capital": 510.0,
            "total_realized_pl": 0.0,
            "active_position": {
                "ticker": "X_EQ",
                "quantity": 5.0,
                "buy_budget": 500.0,
                "entry_price_etf": 100.0,
                "entry_time": "2026-08-19T09:00:00",
                "stop_order_id": 42,
                "stop_price": 90.0,
            },
        }
        router = _Router(
            positions=[
                _resp(200, [current_pos]),  # existed_before (position still there)
            ],
            market=[_resp(201, {"id": 12})],
            history=[
                _resp(200, {"items": [
                    {"order": {"status": "FILLED", "side": "SELL"}, "fill": {"quantity": -5.0, "price": 102.0}},
                ]}),
            ],
            delete=[_resp(204, None)],
        )
        mock_request.side_effect = router

        with patch("src.t212_executor.save_portfolio_state"), \
             patch("src.t212_executor.insert_transaction") as mock_tx, \
             patch("src.t212_executor._update_feedback_loop") as mock_fb:
            _execute_sell_order(
                state, current_pos, "CRUDP.PA", "X_EQ",
                "https://x", {}, "2026-08-19 10:00:00", "TEST", force_stop_loss=True,
            )

        # Real fill used: 5 @ 102 -> proceeds 510, P&L +10 (not the 510 snapshot at 100/share)
        self.assertEqual(state["total_realized_pl"], 10.0)
        self.assertIsNone(state["active_position"])
        tx_kwargs = mock_tx.call_args.kwargs
        self.assertEqual(tx_kwargs["price"], 102.0)
        self.assertEqual(tx_kwargs["cost"], 510.0)
        self.assertEqual(tx_kwargs["quantity"], 5.0)

        # Standing stop order cancelled
        self.assertEqual(router.count("DELETE"), 1)
        mock_fb.assert_called_once()

    @patch("src.t212_executor._t212_session.request")
    def test_sell_unconfirmed_fill_writes_nothing(self, mock_request):
        from src.t212_executor import _execute_sell_order

        current_pos = _pos(qty=5.0, avg=100.0, value=510.0)
        state = {
            "current_capital": 510.0,
            "total_realized_pl": 0.0,
            "active_position": {"buy_budget": 500.0, "entry_time": "2026-08-19T09:00:00"},
        }
        router = _Router(
            positions=[_resp(200, [current_pos])],
            market=[_resp(201, {"id": 13})],
            history=[],  # polls stay empty -> never confirmed
        )
        mock_request.side_effect = router

        with patch("src.t212_executor.save_portfolio_state") as mock_save, \
             patch("src.t212_executor.insert_transaction") as mock_tx:
            _execute_sell_order(
                state, current_pos, "CRUDP.PA", "X_EQ",
                "https://x", {}, "2026-08-19 10:00:00", "TEST", force_stop_loss=True,
            )

        self.assertIsNotNone(state["active_position"])  # untouched
        self.assertEqual(state["total_realized_pl"], 0.0)
        mock_save.assert_not_called()
        mock_tx.assert_not_called()


@patch("time.sleep", MagicMock())
class TestRatchetStopOrder(unittest.TestCase):
    """GO-gate 2 (user decision: moving stop) — cancel-and-replace, upward only."""

    def _state(self, stop_price=90.0, highest=600.0, qty=5.0, stop_id=42):
        return {
            "current_capital": 600.0,
            "total_realized_pl": 0.0,
            "active_position": {
                "ticker": "X_EQ",
                "quantity": qty,
                "buy_budget": 500.0,
                "entry_price_etf": 100.0,
                "entry_time": "2026-08-19T09:00:00",
                "highest_value": highest,
                "stop_order_id": stop_id,
                "stop_price": stop_price,
            },
        }

    @patch("src.t212_executor._t212_session.request")
    def test_ratchet_moves_stop_up(self, mock_request):
        from src.t212_executor import _ratchet_stop_order

        router = _Router(delete=[_resp(204, None)], stop=[_resp(201, {"id": 43})])
        mock_request.side_effect = router
        state = self._state()
        with patch("src.t212_executor.save_portfolio_state") as mock_save:
            _ratchet_stop_order(state, _pos(qty=5.0), "X_EQ", {})

        # peak = 600/5 = 120 -> desired = 120 * 0.90 = 108.0
        self.assertEqual(state["active_position"]["stop_order_id"], 43)
        self.assertEqual(state["active_position"]["stop_price"], 108.0)
        stop_kwargs = [kw for m, u, kw in router.calls if m == "POST" and u.endswith("/equity/orders/stop")][0]
        self.assertEqual(stop_kwargs["json"]["stopPrice"], 108.0)
        mock_save.assert_called_once()

    @patch("src.t212_executor._t212_session.request")
    def test_ratchet_never_lowers_and_skips_small_moves(self, mock_request):
        from src.t212_executor import _ratchet_stop_order

        router = _Router()
        mock_request.side_effect = router
        # desired (108.0) <= current (108.0) + 0.01 -> no-op
        state = self._state(stop_price=108.0)
        with patch("src.t212_executor.save_portfolio_state") as mock_save:
            _ratchet_stop_order(state, _pos(qty=5.0), "X_EQ", {})

        self.assertEqual(router.count("DELETE"), 0)
        self.assertEqual(router.count("POST", "/equity/orders/stop"), 0)
        mock_save.assert_not_called()

    @patch("src.t212_executor._t212_session.request")
    def test_ratchet_failed_delete_keeps_old_stop(self, mock_request):
        from src.t212_executor import _ratchet_stop_order

        router = _Router(delete=[_resp(400, {"detail": "cannot delete"})])
        mock_request.side_effect = router
        state = self._state()
        with patch("src.t212_executor.save_portfolio_state"):
            _ratchet_stop_order(state, _pos(qty=5.0), "X_EQ", {})

        # Delete refused -> stop unchanged (still the old order id/price)
        self.assertEqual(state["active_position"]["stop_order_id"], 42)
        self.assertEqual(state["active_position"]["stop_price"], 90.0)
        self.assertEqual(router.count("POST", "/equity/orders/stop"), 0)

    @patch("src.t212_executor._t212_session.request")
    def test_ratchet_emergency_replacement_after_failed_replace(self, mock_request):
        from src.t212_executor import _ratchet_stop_order

        # Delete OK, first replace (108.0) refused -> emergency re-place at old level (90.0)
        router = _Router(
            delete=[_resp(204, None)],
            stop=[_resp(400, {"detail": "no"}), _resp(201, {"id": 44})],
        )
        mock_request.side_effect = router
        state = self._state()
        with patch("src.t212_executor.save_portfolio_state") as mock_save:
            _ratchet_stop_order(state, _pos(qty=5.0), "X_EQ", {})

        self.assertEqual(state["active_position"]["stop_order_id"], 44)
        self.assertEqual(state["active_position"]["stop_price"], 90.0)
        prices = [kw["json"]["stopPrice"] for m, u, kw in router.calls if m == "POST" and u.endswith("/equity/orders/stop")]
        self.assertEqual(prices, [108.0, 90.0])
        mock_save.assert_called_once()

    @patch("src.t212_executor._t212_session.request")
    def test_ratchet_total_failure_clears_stop_fields_for_self_heal(self, mock_request):
        from src.t212_executor import _ratchet_stop_order

        router = _Router(
            delete=[_resp(204, None)],
            stop=[_resp(400, {}), _resp(400, {})],  # replace + emergency both fail
        )
        mock_request.side_effect = router
        state = self._state()
        with patch("src.t212_executor.save_portfolio_state") as mock_save:
            _ratchet_stop_order(state, _pos(qty=5.0), "X_EQ", {})

        self.assertNotIn("stop_order_id", state["active_position"])
        self.assertNotIn("stop_price", state["active_position"])
        mock_save.assert_called_once()

    @patch("src.t212_executor._t212_session.request")
    def test_ratchet_self_heals_position_without_stop(self, mock_request):
        from src.t212_executor import _ratchet_stop_order

        router = _Router(stop=[_resp(201, {"id": 50})])
        mock_request.side_effect = router
        state = self._state()
        state["active_position"].pop("stop_order_id")
        state["active_position"].pop("stop_price")
        with patch("src.t212_executor.save_portfolio_state") as mock_save:
            _ratchet_stop_order(state, _pos(qty=5.0), "X_EQ", {})

        # max(desired=108.0, entry*0.90=90.0) -> 108.0 placed immediately
        self.assertEqual(state["active_position"]["stop_order_id"], 50)
        self.assertEqual(state["active_position"]["stop_price"], 108.0)
        self.assertEqual(router.count("DELETE"), 0)
        mock_save.assert_called_once()


if __name__ == "__main__":
    unittest.main()
