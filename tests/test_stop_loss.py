"""Tests for the unified exit strategy (June 2026 audit).

Covers the four exit mechanisms added to protect real capital:
  - hard stop-loss (-10% drawdown -> forced SELL, unconditional)
  - soft stop alert (-5% drawdown -> WARNING, no sale)
  - take-profit (+8% gain -> SELL)
  - trailing stop (-3% from peak, existing, regression-checked)
  - time-stop (15 days stale -> forced exit)
  - entry_price corruption guard (recalibrates from trading_history.db)

These are pure-function tests — no Ollama, no T212 API. They run in the
deterministic mocked suite (see AGENTS.md §3).
"""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

import pandas as pd

# Make the repo root importable (src.* layout)
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _pos(current_value: float, qty: float, avg_price: float) -> dict:
    """Build a T212-shaped current_pos dict."""
    return {
        "quantityAvailableForTrading": qty,
        "quantity": qty,
        "averagePrice": avg_price,
        "walletImpact": {"currentValue": current_value},
    }


def _state(buy_budget: float, entry_price: float, entry_time: str, highest_value: float = None) -> dict:
    """Build a local portfolio-state dict with an active position."""
    return {
        "active_position": {
            "ticker": "CRUDl_EQ",
            "quantity": 70.8,
            "buy_budget": buy_budget,
            "entry_price_etf": entry_price,
            "entry_price_index": entry_price,
            "entry_time": entry_time,
            "highest_value": highest_value if highest_value is not None else buy_budget,
        }
    }


class TestTakeProfit(unittest.TestCase):
    """Phase 1B: direct take-profit at +8%."""

    def test_take_profit_triggers_above_target(self):
        from src.t212_executor import _evaluate_take_profit, TAKE_PROFIT_TARGET

        # Buy at 100€/share, qty 10 -> cost 1000€. Now worth 1090€ (+9%).
        st = _state(buy_budget=1000.0, entry_price=100.0, entry_time="2026-06-01T10:00:00")
        pos = _pos(current_value=1090.0, qty=10, avg_price=100.0)
        signal, force = _evaluate_take_profit(st, pos, "CRUDl_EQ")
        self.assertEqual(signal, "SELL")
        self.assertFalse(force, "Take-profit is always in profit — guard must NOT be bypassed")

    def test_take_profit_does_not_trigger_below_target(self):
        from src.t212_executor import _evaluate_take_profit

        st = _state(buy_budget=1000.0, entry_price=100.0, entry_time="2026-06-01T10:00:00")
        pos = _pos(current_value=1050.0, qty=10, avg_price=100.0)  # +5% < 8%
        signal, _ = _evaluate_take_profit(st, pos, "CRUDl_EQ")
        self.assertIsNone(signal)

    def test_take_profit_no_position(self):
        from src.t212_executor import _evaluate_take_profit

        signal, _ = _evaluate_take_profit({"active_position": None}, _pos(0, 0, 0), "CRUDl_EQ")
        self.assertIsNone(signal)


class TestTimeStop(unittest.TestCase):
    """Phase 1C: stale positions (>15 days) are force-exited."""

    def test_time_stop_triggers_when_aged(self):
        from src.t212_executor import _evaluate_time_stop, MAX_HOLDING_DAYS

        # Entry 20 days ago -> beyond the 15-day threshold.
        import datetime as dt
        entry = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=20)).isoformat()
        st = _state(buy_budget=1000.0, entry_price=100.0, entry_time=entry)
        signal, force = _evaluate_time_stop(st, "CRUDl_EQ")
        self.assertEqual(signal, "SELL")
        self.assertTrue(force, "Time-stop must bypass the sell-loss guard for stale dead positions")

    def test_time_stop_does_not_trigger_when_fresh(self):
        from src.t212_executor import _evaluate_time_stop

        import datetime as dt
        entry = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=3)).isoformat()
        st = _state(buy_budget=1000.0, entry_price=100.0, entry_time=entry)
        signal, _ = _evaluate_time_stop(st, "CRUDl_EQ")
        self.assertIsNone(signal)


class TestTrailingStop(unittest.TestCase):
    """Existing trailing stop (-3% from peak with >0.5% profit) — regression."""

    def test_trailing_stop_secures_gain(self):
        from src.t212_executor import _evaluate_trailing_stop

        # Peak 1100€, now 1060€ (-3.6% from peak), cost basis 1000€ (+6% profit).
        st = _state(buy_budget=1000.0, entry_price=100.0, entry_time="2026-06-01T10:00:00",
                    highest_value=1100.0)
        pos = _pos(current_value=1060.0, qty=10, avg_price=100.0)
        signal = _evaluate_trailing_stop(st, pos, "CRUDl_EQ")
        self.assertEqual(signal, "SELL")

    def test_trailing_stop_no_trigger_without_profit(self):
        from src.t212_executor import _evaluate_trailing_stop

        # Peak 950€, now 910€ (-4.2% from peak), cost 1000€ -> in loss, no profit to secure.
        st = _state(buy_budget=1000.0, entry_price=100.0, entry_time="2026-06-01T10:00:00",
                    highest_value=950.0)
        pos = _pos(current_value=910.0, qty=10, avg_price=100.0)
        signal = _evaluate_trailing_stop(st, pos, "CRUDl_EQ")
        self.assertIsNone(signal)


class TestHardStopLoss(unittest.TestCase):
    """Phase 1A: unconditional emergency stop-loss in the risk manager."""

    def _engine(self):
        from src.advanced_risk_manager import AdvancedRiskManager
        return AdvancedRiskManager()

    def test_emergency_sell_on_deep_drawdown_even_if_signal_is_buy(self):
        # The whole point: a BUY signal must become SELL when down >10%.
        eng = self._engine()
        # Entry 100, now 89 -> -11% drawdown (beyond the -10% hard stop).
        prices = pd.Series([100.0] * 60 + [89.0])
        signal, reason = eng.get_risk_adjusted_signal(
            original_signal="BUY",
            confidence=0.9,
            risk_metrics=MagicMock(overall_risk_score=0.3, risk_level=MagicMock(name="LOW"),
                                   volatility_risk=0.2),
            price_data=prices,
            ticker="CRUDP.PA",
            is_holding=True,
            entry_price_index=100.0,
        )
        self.assertEqual(signal, "SELL")
        self.assertIn("EMERGENCY STOP-LOSS", reason)

    def test_soft_stop_alert_does_not_force_sale(self):
        eng = self._engine()
        # Entry 100, now 96 -> -4% (above -5% soft alert, below threshold) — no alert, no sale.
        prices = pd.Series([100.0] * 60 + [96.0])
        signal, _ = eng.get_risk_adjusted_signal(
            original_signal="HOLD",
            confidence=0.5,
            risk_metrics=MagicMock(overall_risk_score=0.3, risk_level=MagicMock(name="LOW"),
                                   volatility_risk=0.2),
            price_data=prices,
            ticker="CRUDP.PA",
            is_holding=True,
            entry_price_index=100.0,
        )
        # -4% is between soft alert (5%) and hard stop (10%) — should NOT force a sell.
        self.assertNotEqual(signal, "SELL")

    def test_soft_stop_alert_logs_at_minus_five(self):
        eng = self._engine()
        # Entry 100, now 94.5 -> -5.5% -> soft alert fires (WARNING only).
        prices = pd.Series([100.0] * 60 + [94.5])
        with patch("src.advanced_risk_manager.logger.warning") as mock_warn:
            signal, _ = eng.get_risk_adjusted_signal(
                original_signal="HOLD",
                confidence=0.5,
                risk_metrics=MagicMock(overall_risk_score=0.3, risk_level=MagicMock(name="LOW"),
                                       volatility_risk=0.2),
                price_data=prices,
                ticker="CRUDP.PA",
                is_holding=True,
                entry_price_index=100.0,
            )
            self.assertNotEqual(signal, "SELL")
            self.assertTrue(any("SOFT STOP ALERT" in str(c) for c in mock_warn.call_args_list),
                            "Soft stop at -5% must emit a WARNING log")


class TestEntryPriceGuard(unittest.TestCase):
    """Phase 4: corruption defence — recalibrate from trading_history.db."""

    def test_recalibrates_when_db_disagrees(self):
        from src.t212_executor import _validate_and_recalibrate_entry_price

        # Stored corrupted price (15.27) vs DB truth (13.42): >5% discrepancy.
        st = _state(buy_budget=1080.97, entry_price=15.27, entry_time="2026-06-09T10:00:00")
        fake_db = ("2026-06-09", "BUY", 70.8, 13.42, 949.99)
        with patch("src.database.get_latest_transaction", return_value=fake_db):
            out = _validate_and_recalibrate_entry_price(st, "CRUDP.PA")
        self.assertAlmostEqual(out["active_position"]["entry_price_etf"], 13.42)
        self.assertAlmostEqual(out["active_position"]["buy_budget"], 949.99)

    def test_no_recalibration_when_close(self):
        from src.t212_executor import _validate_and_recalibrate_entry_price

        # Stored price within 5% of DB — must not be touched.
        st = _state(buy_budget=1000.0, entry_price=100.0, entry_time="2026-06-09T10:00:00")
        fake_db = ("2026-06-09", "BUY", 10.0, 101.5, 1015.0)  # 1.5% diff
        with patch("src.database.get_latest_transaction", return_value=fake_db):
            out = _validate_and_recalibrate_entry_price(st, "CRUDP.PA")
        self.assertAlmostEqual(out["active_position"]["entry_price_etf"], 100.0)

    def test_no_crash_without_position(self):
        from src.t212_executor import _validate_and_recalibrate_entry_price

        out = _validate_and_recalibrate_entry_price({"active_position": None}, "CRUDP.PA")
        self.assertIsNone(out["active_position"])

    def test_handles_missing_db(self):
        from src.t212_executor import _validate_and_recalibrate_entry_price

        st = _state(buy_budget=1000.0, entry_price=100.0, entry_time="2026-06-09T10:00:00")
        with patch("src.database.get_latest_transaction", return_value=None):
            out = _validate_and_recalibrate_entry_price(st, "CRUDP.PA")
        # No DB record -> state unchanged, no crash.
        self.assertAlmostEqual(out["active_position"]["entry_price_etf"], 100.0)


if __name__ == "__main__":
    unittest.main()
