"""GO-gate 7 tests — per-ticker equity, FIFO P&L, journal column, monitoring.

Audit 2026-08-19 (docs/AUDIT_PROD_INDEPENDANT_2026-08-19.md):
  - T212_Capital in the journal mixed position value (open) and cash (flat)
    -> fake -71.6% drawdown; no real equity curve existed.
  - performance_monitor.db had portfolio_value=1000.00 constant on all 649
    rows (decorative monitoring).
"""

import io
import sys
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.t212_executor import _fifo_pnl


def _filled(side, qty, price):
    return {
        "order": {"status": "FILLED", "side": side},
        "fill": {"quantity": qty if side == "BUY" else -qty, "price": price},
    }


class TestFifoPnl(unittest.TestCase):
    """FIFO matching over broker fills: realized P&L + open cost basis."""

    def test_flat_history(self):
        items = [_filled("BUY", 10, 100.0), _filled("SELL", 10, 105.0)]
        realized, open_cost = _fifo_pnl(items)
        self.assertAlmostEqual(realized, 50.0)
        self.assertAlmostEqual(open_cost, 0.0)

    def test_open_position_leaves_cost_basis(self):
        items = [_filled("BUY", 10, 100.0), _filled("SELL", 4, 110.0)]
        realized, open_cost = _fifo_pnl(items)
        self.assertAlmostEqual(realized, 40.0)
        self.assertAlmostEqual(open_cost, 6 * 100.0)  # 6 remaining @ 100

    def test_multiple_lots_matched_in_order(self):
        items = [
            _filled("BUY", 5, 100.0),
            _filled("BUY", 5, 120.0),
            _filled("SELL", 7, 130.0),
        ]
        realized, open_cost = _fifo_pnl(items)
        # 5 @ (130-100) + 2 @ (130-120) = 150 + 20
        self.assertAlmostEqual(realized, 170.0)
        self.assertAlmostEqual(open_cost, 3 * 120.0)

    def test_equity_formula_end_to_end(self):
        """equity = budget + realized + (position_value - open_cost)."""
        budget = 1000.0
        items = [_filled("BUY", 10, 100.0), _filled("SELL", 4, 110.0)]
        realized, open_cost = _fifo_pnl(items)
        current_value = 6 * 115.0  # remaining 6 shares now worth 115
        equity = budget + realized + (current_value - open_cost)
        # cash: 1000 - 1000 spent + 440 received = 440 ; position 690 -> 1130
        self.assertAlmostEqual(equity, 1130.0)

    def test_ignores_unfilled_and_junk(self):
        items = [
            {"order": {"status": "NEW", "side": "BUY"}, "fill": {"quantity": 5, "price": 100.0}},
            {"order": {"status": "FILLED", "side": "BUY"}, "fill": {}},
            "junk",
            None,
            _filled("BUY", 2, 50.0),
        ]
        realized, open_cost = _fifo_pnl(items)
        self.assertAlmostEqual(realized, 0.0)
        self.assertAlmostEqual(open_cost, 100.0)

    def test_empty_history(self):
        realized, open_cost = _fifo_pnl([])
        self.assertEqual((realized, open_cost), (0.0, 0.0))
        self.assertEqual(_fifo_pnl(None), (0.0, 0.0))


class TestSyncEquity(unittest.TestCase):
    """sync_state_from_t212 persists equity/unrealized_pl (open + flat)."""

    def _run_sync(self, positions, history_items):
        from src import t212_executor as ex

        with patch.object(ex, "get_t212_positions", return_value=positions), \
             patch.object(ex, "get_t212_order_history", return_value={"items": history_items}), \
             patch.object(ex, "get_auth_header", side_effect=ValueError("no creds")):
            return ex.sync_state_from_t212("X_EQ")

    def test_flat_equity_is_budget_plus_realized(self):
        history = [_filled("BUY", 10, 100.0), _filled("SELL", 10, 105.0)]
        state = self._run_sync([], history)
        self.assertAlmostEqual(state["equity"], 1050.0)
        self.assertAlmostEqual(state["total_realized_pl"], 50.0)
        self.assertAlmostEqual(state["unrealized_pl"], 0.0)

    def test_open_equity_uses_broker_position(self):
        history = [_filled("BUY", 10, 100.0)]
        positions = [{
            "instrument": {"ticker": "X_EQ"},
            "quantity": 10,
            "quantityAvailableForTrading": 10,
            "averagePricePaid": 100.0,
            "currentPrice": 103.0,
            "walletImpact": {"currentValue": 1030.0},
            "createdAt": "2026-08-19T09:00:00",
        }]
        state = self._run_sync(positions, history)
        self.assertAlmostEqual(state["equity"], 1030.0)
        self.assertAlmostEqual(state["unrealized_pl"], 30.0)
        self.assertAlmostEqual(state["current_capital"], 1030.0)  # sizing semantics unchanged
        self.assertAlmostEqual(state["active_position"]["buy_budget"], 1000.0)


class TestJournalEquityColumn(unittest.TestCase):
    """The CSV journal writes T212_Equity fed by state['equity']."""

    def test_header_and_value(self):
        import csv
        import os
        import tempfile
        import main as main_mod

        decision = MagicMock()
        decision.final_signal = "HOLD"
        decision.final_confidence = 0.25
        decision.individual_decisions = []

        fake_state = {"equity": 1042.17, "current_capital": 999.0}
        cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as tmp:
            os.chdir(tmp)
            try:
                with patch.object(main_mod, "load_t212_state", return_value=fake_state), \
                     patch("t212_executor.get_t212_ticker", side_effect=lambda t: t):
                    main_mod._write_trading_journal(
                        ticker="SXRV.DE", decision=decision, confidence=0.25,
                        risk_level="MODERATE", signal="HOLD", is_t212=True,
                    )
                with open("trading_journal.csv", newline="", encoding="utf-8") as f:
                    rows = list(csv.reader(f))
            finally:
                os.chdir(cwd)

        header, row = rows[0], rows[1]
        self.assertIn("T212_Equity", header)
        self.assertNotIn("T212_Capital", header)
        idx = header.index("T212_Equity")
        self.assertEqual(row[idx], "1042.17 €")


if __name__ == "__main__":
    unittest.main()
