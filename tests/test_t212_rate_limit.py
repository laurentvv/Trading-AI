"""Régulation des appels T212 : limites officielles par endpoint (relevées le 2026-09-29).

GET /equity/orders : 1 req / 5 s ; /equity/positions : 1 / 1 s ; /equity/history/orders : 6 / min ;
POST /equity/orders/stop : 1 / 2 s. Avant : 429 sur la lecture des stops à chaque cycle.
"""

import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from src.t212_rate_limit import (
    GATE,
    T212_INTERVALS,
    GatedSession,
    RateGate,
    bucket_for,
    retry_after_seconds,
)


class FakeClock:
    def __init__(self):
        self.now = 1000.0
        self.slept = []

    def time(self):
        return self.now

    def sleep(self, seconds):
        self.slept.append(seconds)
        self.now += seconds


def _gate(intervals=None):
    clock = FakeClock()
    return RateGate(intervals or T212_INTERVALS, clock=clock.time, sleeper=clock.sleep, enabled=True), clock


class TestBucketFor:
    def test_paths_are_normalised(self):
        base = "https://demo.trading212.com/api/v0"
        assert bucket_for("GET", f"{base}/equity/orders") == "GET /equity/orders"
        assert bucket_for("get", f"{base}/equity/positions?x=1") == "GET /equity/positions"
        assert bucket_for("GET", f"{base}/equity/history/orders?limit=50&ticker=X") == "GET /equity/history/orders"
        assert bucket_for("POST", f"{base}/equity/orders/market") == "POST /equity/orders/market"
        assert bucket_for("POST", f"{base}/equity/orders/stop") == "POST /equity/orders/stop"

    def test_order_ids_collapse_to_one_bucket(self):
        base = "https://live.trading212.com/api/v0"
        assert bucket_for("DELETE", f"{base}/equity/orders/55701244265") == "DELETE /equity/orders/{id}"
        assert bucket_for("GET", f"{base}/equity/orders/123") == "GET /equity/orders/{id}"

    def test_every_documented_bucket_has_an_interval(self):
        for bucket in (
            "GET /equity/orders",
            "GET /equity/positions",
            "GET /equity/history/orders",
            "POST /equity/orders/stop",
            "POST /equity/orders/limit",
            "POST /equity/orders/stop_limit",
        ):
            assert T212_INTERVALS[bucket] > 0

    def test_every_order_placement_endpoint_is_gated(self):
        """limit / stop_limit sont disponibles en réel : ils ne doivent pas échapper au régulateur."""
        base = "https://live.trading212.com/api/v0/equity/orders"
        for kind in ("market", "limit", "stop", "stop_limit"):
            assert T212_INTERVALS[bucket_for("POST", f"{base}/{kind}")] > 0, kind


class TestRateGate:
    def test_first_call_is_free_then_calls_are_spaced(self):
        gate, clock = _gate()
        assert gate.wait("GET /equity/orders") == 0.0
        waited = gate.wait("GET /equity/orders")
        assert abs(waited - T212_INTERVALS["GET /equity/orders"]) < 1e-9
        assert len(clock.slept) == 1

    def test_no_wait_once_the_interval_has_elapsed(self):
        gate, clock = _gate()
        gate.wait("GET /equity/orders")
        clock.now += 6.0
        assert gate.wait("GET /equity/orders") == 0.0

    def test_buckets_are_independent(self):
        gate, _ = _gate()
        gate.wait("GET /equity/orders")
        assert gate.wait("GET /equity/positions") == 0.0

    def test_unknown_bucket_is_never_delayed(self):
        gate, _ = _gate()
        gate.wait("GET /equity/unknown")
        assert gate.wait("GET /equity/unknown") == 0.0

    def test_penalize_delays_the_next_call(self):
        gate, _ = _gate()
        gate.penalize("GET /equity/orders", 7.0)
        assert abs(gate.wait("GET /equity/orders") - 7.0) < 1e-9

    def test_disabled_gate_never_sleeps(self):
        clock = FakeClock()
        gate = RateGate(T212_INTERVALS, clock=clock.time, sleeper=clock.sleep, enabled=False)
        gate.wait("GET /equity/orders")
        gate.wait("GET /equity/orders")
        assert clock.slept == []

    def test_concurrent_callers_are_queued_not_served_together(self):
        """Créneaux réservés sous verrou : 4 threads simultanés obtiennent 4 créneaux distincts."""
        waits = []
        lock = threading.Lock()
        clock = FakeClock()

        def sleeper(seconds):
            with lock:
                waits.append(seconds)

        gate = RateGate({"B": 1.0}, clock=clock.time, sleeper=sleeper, enabled=True)
        threads = [threading.Thread(target=gate.wait, args=("B",)) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        # Le 1er passe sans attendre ; les 3 suivants attendent 1 s, 2 s et 3 s (dans un ordre quelconque).
        assert sorted(waits) == [1.0, 2.0, 3.0]


class TestGatedSession:
    def test_every_verb_goes_through_the_gate(self):
        session = GatedSession()
        with patch.object(GATE, "wait") as mock_wait, patch(
            "requests.Session.request", return_value=MagicMock()
        ) as mock_req:
            session.get("https://demo.trading212.com/api/v0/equity/orders")
            session.delete("https://demo.trading212.com/api/v0/equity/orders/42")
            session.post("https://demo.trading212.com/api/v0/equity/orders/stop", json={})
        assert [c.args[0] for c in mock_wait.call_args_list] == [
            "GET /equity/orders",
            "DELETE /equity/orders/{id}",
            "POST /equity/orders/stop",
        ]
        assert mock_req.call_count == 3


class TestRetryAfter:
    def test_header_is_honoured_and_capped(self):
        assert retry_after_seconds(SimpleNamespace(headers={"Retry-After": "4"}), 9.0) == 4.0
        assert retry_after_seconds(SimpleNamespace(headers={"Retry-After": "999"}), 9.0) == 15.0

    def test_falls_back_to_default(self):
        assert retry_after_seconds(SimpleNamespace(headers={}), 5.5) == 5.5
        assert retry_after_seconds(None, 2.0) == 2.0
        assert retry_after_seconds(SimpleNamespace(headers={"Retry-After": "soon"}), 3.0) == 3.0


class TestStopFetchRetriesOnce429:
    def test_429_then_200_returns_found(self):
        import src.t212_executor as t212

        order = {
            "instrument": {"ticker": "SXRVd_EQ"},
            "type": "STOP",
            "side": "SELL",
            "status": "WORKING",
            "id": 1,
            "stopPrice": 1300.0,
        }
        r429 = SimpleNamespace(status_code=429, headers={}, text="")
        r200 = SimpleNamespace(status_code=200, headers={}, text="", json=lambda: [order])
        with patch.object(t212, "_t212_session") as session, patch.object(t212.GATE, "penalize"):
            session.get.side_effect = [r429, r200]
            status, found = t212._get_active_stop_order("SXRVd_EQ", headers={})
        assert (status, found["id"]) == ("FOUND", 1)
        assert session.get.call_count == 2

    def test_400_too_many_requests_is_retried_like_a_429(self):
        """Le broker signale aussi la limite par un 400 « TooManyRequests » (cf. safe_request)."""
        import src.t212_executor as t212

        order = {"instrument": {"ticker": "SXRVd_EQ"}, "type": "STOP", "side": "SELL", "status": "WORKING", "id": 7}
        r400 = SimpleNamespace(status_code=400, headers={"Retry-After": "3"}, text='{"type":"TooManyRequests"}')
        r200 = SimpleNamespace(status_code=200, headers={}, text="", json=lambda: [order])
        with patch.object(t212, "_t212_session") as session, patch.object(t212.GATE, "penalize") as penalize:
            session.get.side_effect = [r400, r200]
            status, found = t212._get_active_stop_order("SXRVd_EQ", headers={})
        assert (status, found["id"]) == ("FOUND", 7)
        # le délai est appliqué explicitement via la porte (pas seulement par la session)
        penalize.assert_called_once_with("GET /equity/orders", 3.0)

    def test_other_400_is_not_retried(self):
        import src.t212_executor as t212

        r400 = SimpleNamespace(status_code=400, headers={}, text='{"type":"BadRequest"}')
        with patch.object(t212, "_t212_session") as session, patch.object(t212.GATE, "penalize") as penalize:
            session.get.side_effect = [r400]
            assert t212._get_active_stop_order("SXRVd_EQ", headers={}) == ("ERROR", None)
        assert session.get.call_count == 1
        penalize.assert_not_called()

    def test_two_429_stay_an_error_never_an_empty_list(self):
        """Invariant « Failed fetch ≠ empty » : deux 429 = ERROR, jamais NOT_FOUND."""
        import src.t212_executor as t212

        r429 = SimpleNamespace(status_code=429, headers={}, text="")
        with patch.object(t212, "_t212_session") as session, patch.object(t212.GATE, "penalize"):
            session.get.side_effect = [r429, r429]
            status, found = t212._get_active_stop_order("SXRVd_EQ", headers={})
        assert (status, found) == ("ERROR", None)
        assert session.get.call_count == 2


class TestSellConfirmationStaysShort:
    """Revue Kilo PR #96 : l'espacement de 10,5 s de l'historique ne doit pas rallonger la fenêtre de confirmation."""

    def test_sell_polls_history_only_a_few_times_and_never_sleeps_on_top_of_the_gate(self):
        import src.t212_executor as t212

        empty = SimpleNamespace(status_code=200, headers={}, text="", json=lambda: {"items": []})
        with patch.object(t212, "_t212_session") as session, patch.object(t212.time, "sleep") as sleep:
            session.get.return_value = empty
            assert t212._confirm_fill("SXRVd_EQ", {}, side="SELL", expected_qty=1.0) is None
        assert t212.SELL_CONFIRM_ATTEMPTS <= 3
        assert session.get.call_count == t212.SELL_CONFIRM_ATTEMPTS
        sleep.assert_not_called()

    def test_buy_keeps_its_fast_positions_polling(self):
        import src.t212_executor as t212

        empty = SimpleNamespace(status_code=200, headers={}, text="", json=lambda: [])
        with patch.object(t212, "_t212_session") as session, patch.object(t212.time, "sleep") as sleep:
            session.get.return_value = empty
            assert t212._confirm_fill("SXRVd_EQ", {}, side="BUY") is None
        assert session.get.call_count == t212.FILL_CONFIRM_ATTEMPTS
        assert sleep.call_count == t212.FILL_CONFIRM_ATTEMPTS - 1
