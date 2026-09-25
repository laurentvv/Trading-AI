"""Unit tests for the 7 P0 remediation fixes from PLAN.md §2.1.

Tests cover:
1. Exit management (TP, SL, trailing, time-stop, ratchet) runs on HOLD cycles.
2. Upstream risk manager receives market_data["price_series"] when holding.
3. main.py respects risk-adjusted signal and avoids duplicate broken calls.
4. sync_state_from_t212 merges local state (highest_value never drops, entry_time/entry_price_index preserved).
5. _evaluate_time_stop applies TIME_STOP_SOFT_LOSS (5%).
6. _get_active_stop_order distinguishes 3 states (FOUND / NOT_FOUND / ERROR) and blocks blind self-heal on network error.
7. get_t212_order_history follows nextPagePath for paginated order history (>120 fills).
"""

import json
import unittest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch, call
from pathlib import Path
import sys

# Ensure src/ is on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import src.t212_executor as t212
from src.enhanced_trading_example import EnhancedTradingSystem
from src.enhanced_decision_engine import EnhancedDecisionEngine, HybridDecision, ModelDecision, SignalStrength
from src.advanced_risk_manager import AdvancedRiskManager, RiskMetrics, RiskLevel


def _mock_response(status_code, json_data):
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data
    resp.text = json.dumps(json_data) if json_data is not None else ""
    return resp


class TestP0_1_ExitOnHoldCycle(unittest.TestCase):
    """P0-1: When a position is open, exit strategies must be evaluated even on HOLD cycles."""

    def test_take_profit_triggered_on_hold_cycle(self):
        """A position at +9% with incoming HOLD signal must execute take-profit sale."""
        state = {
            "active_position": {
                "ticker": "SXRVd_EQ",
                "quantity": 1.0,
                "buy_budget": 100.0,
                "entry_price_etf": 100.0,
                "entry_price_index": 100.0,
                "highest_value": 109.0,
                "entry_time": (datetime.now() - timedelta(days=2)).isoformat(),
            }
        }
        current_pos = {
            "instrument": {"ticker": "SXRVd_EQ"},
            "quantity": 1.0,
            "quantityAvailableForTrading": 1.0,
            "averagePricePaid": 100.0,
            "currentPrice": 109.0,
            "walletImpact": {"currentValue": 109.0},  # +9% profit
        }

        portfolio_info = {
            "cash": 500.0,
            "positions": [current_pos],
            "positions_ok": True,
        }

        with patch.object(t212, "load_portfolio_state", return_value=state), \
             patch.object(t212, "_get_portfolio_info", return_value=portfolio_info), \
             patch.object(t212, "_validate_and_recalibrate_entry_price", return_value=state), \
             patch.object(t212, "_execute_sell_order") as mock_sell, \
             patch.object(t212, "get_auth_header", return_value={"Authorization": "Bearer test"}):

            # Call execute_t212_trade with HOLD signal
            t212.execute_t212_trade(
                signal="HOLD",
                confidence=0.50,
                ticker="SXRV.DE",
            )

            # mock_sell MUST be called despite signal being HOLD!
            mock_sell.assert_called_once()
            args, kwargs = mock_sell.call_args
            self.assertEqual(kwargs.get("exit_reason"), "take-profit")

    def test_ratchet_evaluated_on_hold_cycle_when_no_exit(self):
        """When no exit triggers on HOLD, the ratchet stop must still be evaluated."""
        state = {
            "active_position": {
                "ticker": "SXRVd_EQ",
                "quantity": 1.0,
                "buy_budget": 100.0,
                "entry_price_etf": 100.0,
                "entry_price_index": 100.0,
                "highest_value": 104.0,
                "stop_order_id": 12345,
                "stop_price": 90.0,
                "entry_time": (datetime.now() - timedelta(days=1)).isoformat(),
            }
        }
        current_pos = {
            "instrument": {"ticker": "SXRVd_EQ"},
            "quantity": 1.0,
            "quantityAvailableForTrading": 1.0,
            "averagePricePaid": 100.0,
            "currentPrice": 104.0,
            "walletImpact": {"currentValue": 104.0},  # +4% profit (below TP 8%, above peak)
        }

        portfolio_info = {
            "cash": 500.0,
            "positions": [current_pos],
            "positions_ok": True,
        }

        with patch.object(t212, "load_portfolio_state", return_value=state), \
             patch.object(t212, "_get_portfolio_info", return_value=portfolio_info), \
             patch.object(t212, "_validate_and_recalibrate_entry_price", return_value=state), \
             patch.object(t212, "_ratchet_stop_order") as mock_ratchet, \
             patch.object(t212, "_execute_sell_order") as mock_sell, \
             patch.object(t212, "get_auth_header", return_value={"Authorization": "Bearer test"}):

            t212.execute_t212_trade(
                signal="HOLD",
                confidence=0.50,
                ticker="SXRV.DE",
            )

            mock_sell.assert_not_called()
            mock_ratchet.assert_called_once()


class TestP0_2_PriceSeriesAndRiskUpstream(unittest.TestCase):
    """P0-2: market_data['price_series'] must be exposed and price_data cannot be None when is_holding=True."""

    def test_risk_manager_fails_if_price_data_none_when_holding(self):
        """AdvancedRiskManager must raise ValueError if price_data is None while is_holding=True."""
        rm = AdvancedRiskManager()
        risk_metrics = RiskMetrics(
            volatility_risk=0.20,
            drawdown_risk=0.15,
            correlation_risk=0.25,
            liquidity_risk=0.10,
            overall_risk_score=0.20,
            risk_level=RiskLevel.LOW,
        )

        with self.assertRaises(ValueError):
            rm.get_risk_adjusted_signal(
                original_signal="HOLD",
                confidence=0.5,
                risk_metrics=risk_metrics,
                price_data=None,  # Missing price_data while is_holding=True!
                ticker="SXRV.DE",
                is_holding=True,
                entry_price_index=1500.0,
            )

    def test_perform_enhanced_analysis_exposes_price_series(self):
        """perform_enhanced_analysis must expose price_series in market_data."""
        import pandas as pd
        system = EnhancedTradingSystem(ticker="SXRV.DE", write_db=False)
        dates = pd.date_range("2026-01-01", periods=100)
        prices = [100.0 + i for i in range(100)]
        df = pd.DataFrame({
            "Close": prices,
            "Volume": [1000] * 100,
            "RSI": [55.0] * 100,
            "MACD": [0.1] * 100,
            "BB_Position": [0.5] * 100,
        }, index=dates)

        model_preds = {
            "classic": {"prediction": 1, "confidence": 0.8},
            "text_llm": MagicMock(signal="BUY", confidence=0.8, reasoning="ok"),
            "visual_llm": MagicMock(signal="BUY", confidence=0.8, reasoning="ok"),
            "sentiment": MagicMock(signal="BUY", confidence=0.8, reasoning="ok"),
            "timesfm": MagicMock(signal="BUY", confidence=0.8, reasoning="ok"),
            "tensortrade": MagicMock(signal="BUY", confidence=0.8, reasoning="ok"),
            "grebenkov": MagicMock(signal="HOLD", confidence=0.0, reasoning="ok"),
            "hmm_model": MagicMock(signal="HOLD", confidence=0.0, reasoning="ok"),
        }

        results = system.perform_enhanced_analysis(
            data_with_features=df,
            model_predictions=model_preds,
            current_etf_price=150.0,
        )

        self.assertIn("price_series", results["market_data"])
        self.assertIsInstance(results["market_data"]["price_series"], pd.Series)
        self.assertEqual(len(results["market_data"]["price_series"]), 100)


class TestP0_3_RiskAdjustedSignalPipeline(unittest.TestCase):
    """P0-3: main.py uses risk_adjusted_signal and routes open positions on HOLD cycles."""

    def test_main_respects_engine_risk_adjusted_signal_and_manages_hold(self):
        """When engine downgrades low-confidence BUY to HOLD, main.py passes HOLD to executor and manages open position."""
        import pandas as pd
        import main

        mock_system = MagicMock()
        mock_system.risk_manager = AdvancedRiskManager()

        # Engine decision had BUY as final_signal, but risk_adjusted_signal is HOLD (confidence < 0.20)
        decision = MagicMock()
        decision.final_signal = "BUY"
        decision.risk_adjusted_signal = "HOLD"
        decision.final_confidence = 0.15

        risk_metrics = RiskMetrics(
            volatility_risk=0.20,
            drawdown_risk=0.15,
            correlation_risk=0.25,
            liquidity_risk=0.10,
            overall_risk_score=0.20,
            risk_level=RiskLevel.LOW,
        )

        prices = pd.Series([100.0, 101.0, 102.0])
        results = {
            "risk_adjusted_signal": "HOLD",
            "market_data": {"price_series": prices},
        }

        t212_state = {
            "active_position": {
                "ticker": "SXRVd_EQ",
                "quantity": 1.0,
                "buy_budget": 100.0,
                "entry_price_etf": 100.0,
                "entry_price_index": 100.0,
            }
        }

        with patch("main.load_t212_state", return_value=t212_state), \
             patch("main.execute_t212_trade") as mock_exec:

            final_signal = main._execute_t212_orders(
                ticker="SXRV.DE",
                system=mock_system,
                decision=decision,
                risk=risk_metrics,
                results=results,
                cancel_event=None,
                console=MagicMock(),
            )

            # Signal returned must be HOLD
            self.assertEqual(final_signal, "HOLD")
            # execute_t212_trade must be called with "HOLD" to evaluate open position exits
            mock_exec.assert_called_once()
            args, kwargs = mock_exec.call_args
            self.assertEqual(args[0], "HOLD")



class TestP0_4_StateMergeSync(unittest.TestCase):
    """P0-4: sync_state_from_t212 must merge local state (highest_value, entry_time, entry_price_index)."""

    def test_sync_preserves_highest_value_and_entry_metadata(self):
        """When a position exists locally with a higher highest_value, sync must not reset it."""
        t212_ticker = "SXRVd_EQ"
        local_state = {
            "tickers": {
                t212_ticker: {
                    "initial_budget": 1000.0,
                    "current_capital": 1050.0,
                    "total_realized_pl": 10.0,
                    "unrealized_pl": 50.0,
                    "equity": 1060.0,
                    "active_position": {
                        "ticker": t212_ticker,
                        "quantity": 1.0,
                        "buy_budget": 100.0,
                        "entry_price_etf": 100.0,
                        "entry_price_index": 15000.0,  # NASDAQ index price
                        "entry_time": "2026-09-01T10:00:00",
                        "highest_value": 115.0,  # Peak was 115
                        "stop_order_id": 99999,
                        "stop_price": 103.5,
                    },
                    "t212_synced": True,
                }
            }
        }

        # Broker returns current value of 108.0 (pulled back from 115.0)
        current_pos = {
            "instrument": {"ticker": t212_ticker},
            "quantity": 1.0,
            "averagePricePaid": 100.0,
            "currentPrice": 108.0,
            "walletImpact": {"currentValue": 108.0},
        }

        portfolio_info = {
            "cash": 900.0,
            "positions": [current_pos],
            "positions_ok": True,
        }

        with patch.object(t212, "_read_with_retry", return_value=local_state), \
             patch.object(t212, "get_t212_positions", return_value=[current_pos]), \
             patch.object(t212, "get_t212_order_history", return_value={"items": []}), \
             patch.object(t212, "_get_active_stop_order", return_value=("FOUND", {"id": 99999, "stopPrice": 103.5})), \
             patch.object(t212, "get_auth_header", return_value={"Authorization": "Bearer test"}):

            synced = t212.sync_state_from_t212(t212_ticker)

            pos = synced["active_position"]
            self.assertIsNotNone(pos)
            # highest_value must be 115.0 (local peak), NOT reset to 108.0!
            self.assertEqual(pos["highest_value"], 115.0)
            # entry_time must remain 2026-09-01T10:00:00, NOT overwritten with now()!
            self.assertEqual(pos["entry_time"], "2026-09-01T10:00:00")
            # entry_price_index must remain 15000.0, NOT overwritten with 100.0!
            self.assertEqual(pos["entry_price_index"], 15000.0)


class TestP0_5_TimeStopSoftLoss(unittest.TestCase):
    """P0-5: _evaluate_time_stop must apply TIME_STOP_SOFT_LOSS (5%)."""

    def test_time_stop_sells_if_loss_under_5_percent(self):
        """Aged position with small loss (-2%) triggers time-stop sell."""
        state = {
            "active_position": {
                "buy_budget": 100.0,
                "entry_time": (datetime.now() - timedelta(days=16)).isoformat(),
            }
        }
        current_pos = {
            "quantity": 1.0,
            "quantityAvailableForTrading": 1.0,
            "averagePricePaid": 100.0,
            "walletImpact": {"currentValue": 98.0},  # -2% loss <= 5%
        }
        signal, force = t212._evaluate_time_stop(state, "SXRVd_EQ", current_pos=current_pos)
        self.assertEqual(signal, "SELL")
        self.assertTrue(force)

    def test_time_stop_does_not_sell_if_loss_exceeds_5_percent(self):
        """Aged position with deeper loss (-7%) does NOT trigger time-stop (leaves to hard stop)."""
        state = {
            "active_position": {
                "buy_budget": 100.0,
                "entry_time": (datetime.now() - timedelta(days=16)).isoformat(),
            }
        }
        current_pos = {
            "quantity": 1.0,
            "quantityAvailableForTrading": 1.0,
            "averagePricePaid": 100.0,
            "walletImpact": {"currentValue": 93.0},  # -7% loss > 5%
        }
        signal, force = t212._evaluate_time_stop(state, "SXRVd_EQ", current_pos=current_pos)
        self.assertIsNone(signal)
        self.assertFalse(force)


class TestP0_6_GetActiveStopOrderThreeStates(unittest.TestCase):
    """P0-6: _get_active_stop_order must distinguish FOUND, NOT_FOUND, and ERROR."""

    def test_network_error_returns_error_status(self):
        """On request failure, _get_active_stop_order must return ('ERROR', None)."""
        with patch.object(t212, "_t212_session") as mock_session:
            mock_session.get.side_effect = Exception("Connection refused")
            status, order = t212._get_active_stop_order("SXRVd_EQ", headers={})
            self.assertEqual(status, "ERROR")
            self.assertIsNone(order)

    def test_not_found_returns_not_found_status(self):
        """When broker returns empty order list with 200, return ('NOT_FOUND', None)."""
        with patch.object(t212, "_t212_session") as mock_session:
            mock_session.get.return_value = _mock_response(200, [])
            status, order = t212._get_active_stop_order("SXRVd_EQ", headers={})
            self.assertEqual(status, "NOT_FOUND")
            self.assertIsNone(order)

    def test_found_returns_found_status(self):
        """When broker returns matching STOP order, return ('FOUND', order)."""
        matching_order = {
            "instrument": {"ticker": "SXRVd_EQ"},
            "type": "STOP",
            "side": "SELL",
            "status": "WORKING",
            "id": 123456,
            "stopPrice": 1300.0,
        }
        with patch.object(t212, "_t212_session") as mock_session:
            mock_session.get.return_value = _mock_response(200, [matching_order])
            status, order = t212._get_active_stop_order("SXRVd_EQ", headers={})
            self.assertEqual(status, "FOUND")
            self.assertEqual(order["id"], 123456)

    def test_ratchet_does_not_place_blind_stop_on_error(self):
        """If _get_active_stop_order returns ERROR, ratchet must NOT place a duplicate stop."""
        state = {
            "active_position": {
                "quantity": 1.0,
                "highest_value": 100.0,
                "stop_order_id": None,  # Missing locally
            }
        }
        current_pos = {
            "quantity": 1.0,
            "walletImpact": {"currentValue": 100.0},
        }

        with patch.object(t212, "_get_active_stop_order", return_value=("ERROR", None)), \
             patch.object(t212, "_place_stop_order") as mock_place:
            t212._ratchet_stop_order(state, current_pos, "SXRVd_EQ", headers={})
            # MUST NOT place a stop on network error!
            mock_place.assert_not_called()


class TestP0_7_OrderHistoryPagination(unittest.TestCase):
    """P0-7: get_t212_order_history must traverse nextPagePath for >50 orders."""

    def test_pagination_fetches_all_pages(self):
        """Mock 3 pages of 50 orders (150 total) and verify all are returned."""
        page1 = {
            "items": [{"order": {"id": i, "status": "FILLED"}} for i in range(1, 51)],
            "nextPagePath": "/equity/history/orders?cursor=page2",
        }
        page2 = {
            "items": [{"order": {"id": i, "status": "FILLED"}} for i in range(51, 101)],
            "nextPagePath": "/equity/history/orders?cursor=page3",
        }
        page3 = {
            "items": [{"order": {"id": i, "status": "FILLED"}} for i in range(101, 151)],
            "nextPagePath": None,
        }

        with patch.object(t212, "get_auth_header", return_value={"Authorization": "Bearer test"}), \
             patch.object(t212, "safe_request") as mock_safe_request:
            mock_safe_request.side_effect = [
                _mock_response(200, page1),
                _mock_response(200, page2),
                _mock_response(200, page3),
            ]

            result = t212.get_t212_order_history(ticker="SXRVd_EQ", limit=50)

            self.assertIsNotNone(result)
            items = result.get("items", [])
            self.assertEqual(len(items), 150)
            self.assertEqual(items[0]["order"]["id"], 1)
            self.assertEqual(items[-1]["order"]["id"], 150)
            self.assertEqual(mock_safe_request.call_count, 3)


if __name__ == "__main__":
    unittest.main()
