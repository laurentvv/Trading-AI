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


class TestHardStopExecutor(unittest.TestCase):
    """Executor-side hard stop-loss — the last line of defence.

    Covers the gap flagged in PR #73 review: the risk-manager stop requires
    is_holding/entry_price_index/price_data, which a caller can omit. The
    executor reads the live broker position directly, so a deep drawdown is
    ALWAYS caught here regardless of how the upstream signal was produced.
    """

    def test_executor_hard_stop_forces_sell_on_deep_drawdown(self):
        from src.t212_executor import _evaluate_hard_stop

        # Cost 1000€ (10 shares @ 100€), now worth 880€ -> -12% drawdown.
        st = _state(buy_budget=1000.0, entry_price=100.0, entry_time="2026-06-01T10:00:00")
        pos = _pos(current_value=880.0, qty=10, avg_price=100.0)
        signal, force = _evaluate_hard_stop(st, pos, "CRUDl_EQ")
        self.assertEqual(signal, "SELL")
        self.assertTrue(force, "Hard stop must bypass _check_sell_loss_guard")

    def test_executor_hard_stop_no_trigger_above_threshold(self):
        from src.t212_executor import _evaluate_hard_stop

        # Cost 1000€, now 950€ -> -5% (at soft alert, below hard stop -10%).
        st = _state(buy_budget=1000.0, entry_price=100.0, entry_time="2026-06-01T10:00:00")
        pos = _pos(current_value=950.0, qty=10, avg_price=100.0)
        signal, _ = _evaluate_hard_stop(st, pos, "CRUDl_EQ")
        self.assertIsNone(signal)

    def test_executor_hard_stop_forces_sell_even_when_incoming_signal_is_buy(self):
        """The whole point of the executor-side stop: it catches the case the
        upstream risk layer misses (e.g. no entry_price_index passed). The
        function itself doesn't see the incoming signal, but it must return
        SELL independently — the caller in execute_t212_trade overrides the
        BUY with this SELL because hard-stop is evaluated at priority 0."""
        from src.t212_executor import _evaluate_hard_stop

        # Deep loss -15% — must return SELL regardless of what the consensus said.
        st = _state(buy_budget=1000.0, entry_price=100.0, entry_time="2026-06-01T10:00:00")
        pos = _pos(current_value=850.0, qty=10, avg_price=100.0)
        signal, force = _evaluate_hard_stop(st, pos, "CRUDl_EQ")
        self.assertEqual(signal, "SELL")
        self.assertTrue(force)


class TestEntryPriceGuard(unittest.TestCase):
    """Phase 4: corruption defence — recalibrate from the AUTHORITATIVE source.

    Source-of-truth priority (changed July 2026 after the DB-trust bug):
      1. current_pos.averagePricePaid (broker's real fill) — primary
      2. trading_history.db (signal-time price) — fallback only
    The local DB can record a WRONG price (July incident: DB=10.876 while the
    real T212 fill was 12.4469), so trusting it blindly would corrupt a correct
    state. The broker price always wins when available.
    """

    def test_recalibrates_when_broker_disagrees(self):
        from src.t212_executor import _validate_and_recalibrate_entry_price

        # June incident: stored ghost 15.27 vs broker real fill 13.42.
        st = _state(buy_budget=1080.97, entry_price=15.27, entry_time="2026-06-09T10:00:00")
        pos = {"averagePricePaid": 13.42, "quantity": 70.8,
               "quantityAvailableForTrading": 70.8}
        with patch("src.database.get_latest_transaction", return_value=None):
            out = _validate_and_recalibrate_entry_price(st, "CRUDP.PA", current_pos=pos)
        self.assertAlmostEqual(out["active_position"]["entry_price_etf"], 13.42)

    def test_does_NOT_corrupt_when_db_lies_but_broker_right(self):
        """The July regression: DB recorded 10.876 but the real T212 fill was
        12.4469. The state (12.4469, correct) must NOT be overwritten by the
        lying DB. Broker price is authoritative."""
        from src.t212_executor import _validate_and_recalibrate_entry_price

        # State correct (= broker), DB lies.
        st = _state(buy_budget=326.11, entry_price=12.4469, entry_time="2026-07-01T12:48:00")
        pos = {"averagePricePaid": 12.4469, "quantity": 26.2,
               "quantityAvailableForTrading": 26.2}
        lying_db = ("2026-07-01", "BUY", 26.2, 10.876, 284.95)
        with patch("src.database.get_latest_transaction", return_value=lying_db):
            out = _validate_and_recalibrate_entry_price(st, "CRUDP.PA", current_pos=pos)
        # Broker matches state -> no recalibration, DB ignored even though it lies.
        self.assertAlmostEqual(out["active_position"]["entry_price_etf"], 12.4469)

    def test_falls_back_to_db_when_no_broker_position(self):
        from src.t212_executor import _validate_and_recalibrate_entry_price

        # No current_pos -> DB is the fallback source.
        st = _state(buy_budget=1080.97, entry_price=15.27, entry_time="2026-06-09T10:00:00")
        fake_db = ("2026-06-09", "BUY", 70.8, 13.42, 949.99)
        with patch("src.database.get_latest_transaction", return_value=fake_db):
            out = _validate_and_recalibrate_entry_price(st, "CRUDP.PA", current_pos=None)
        self.assertAlmostEqual(out["active_position"]["entry_price_etf"], 13.42)

    def test_no_recalibration_when_close(self):
        from src.t212_executor import _validate_and_recalibrate_entry_price

        st = _state(buy_budget=1000.0, entry_price=100.0, entry_time="2026-06-09T10:00:00")
        pos = {"averagePricePaid": 101.5, "quantity": 10, "quantityAvailableForTrading": 10}
        out = _validate_and_recalibrate_entry_price(st, "CRUDP.PA", current_pos=pos)
        self.assertAlmostEqual(out["active_position"]["entry_price_etf"], 100.0)

    def test_no_crash_without_position(self):
        from src.t212_executor import _validate_and_recalibrate_entry_price

        out = _validate_and_recalibrate_entry_price({"active_position": None}, "CRUDP.PA")
        self.assertIsNone(out["active_position"])

    def test_handles_missing_sources(self):
        from src.t212_executor import _validate_and_recalibrate_entry_price

        st = _state(buy_budget=1000.0, entry_price=100.0, entry_time="2026-06-09T10:00:00")
        with patch("src.database.get_latest_transaction", return_value=None):
            out = _validate_and_recalibrate_entry_price(st, "CRUDP.PA", current_pos=None)
        self.assertAlmostEqual(out["active_position"]["entry_price_etf"], 100.0)


if __name__ == "__main__":
    unittest.main()
