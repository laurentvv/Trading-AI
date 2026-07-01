"""Tests for AdvancedRiskManager — focused on the EXIT INERTIA hard-stop
bypass added in ADR-002.

Previously, an open position could drift indefinitely (CRUDP.PA hit -18%)
because every SELL was squelched to HOLD by exit inertia. A hard stop now
bypasses inertia once the drawdown exceeds ``hard_stop_drawdown``.
"""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from advanced_risk_manager import AdvancedRiskManager, RiskMetrics, RiskLevel


def _risk_metrics(level=RiskLevel.MODERATE):
    return RiskMetrics(
        volatility_risk=0.3,
        drawdown_risk=0.3,
        correlation_risk=0.3,
        liquidity_risk=0.3,
        overall_risk_score=0.3,
        risk_level=level,
    )


def _flat_prices(n=60, start=100.0):
    return pd.Series(np.linspace(start, start + 0.01, n))


class TestExitInertiaHardStop(unittest.TestCase):
    def setUp(self):
        self.rm = AdvancedRiskManager()

    def test_inertia_blocks_weak_sell_in_profit(self):
        # Holding, in profit, weak SELL conviction -> inertia keeps the position.
        prices = pd.Series(np.linspace(100.0, 110.0, 60))  # index rose 10%
        signal, reason = self.rm.get_risk_adjusted_signal(
            original_signal="SELL",
            confidence=0.30,
            risk_metrics=_risk_metrics(),
            price_data=prices,
            ticker="SXRV.DE",
            is_holding=True,
            entry_price_index=100.0,
        )
        self.assertEqual(signal, "HOLD")
        self.assertIn("INERTIA", reason.upper())

    def test_hard_stop_releases_sell_when_deep_underwater(self):
        # Holding, deep loss beyond hard_stop_drawdown -> SELL forced despite
        # low conviction. This is the CRUDP.PA -18% regression guard.
        prices = pd.Series(np.linspace(100.0, 82.0, 60))  # index fell to -18%
        signal, reason = self.rm.get_risk_adjusted_signal(
            original_signal="SELL",
            confidence=0.10,
            risk_metrics=_risk_metrics(),
            price_data=prices,
            ticker="CRUDP.PA",
            is_holding=True,
            entry_price_index=100.0,
        )
        self.assertEqual(signal, "SELL")
        # June 2026: wording changed from "HARD STOP" to "EMERGENCY STOP-LOSS"
        # (clearer) and the stop is now UNCONDITIONAL (triggers on drawdown
        # alone, even when the incoming signal is not SELL).
        self.assertIn("EMERGENCY STOP-LOSS", reason.upper())

    def test_hard_stop_is_unconditional_even_on_buy_signal(self):
        # The core June 2026 fix: a BUY signal must become SELL when the
        # position is deeply underwater. Previously the hard stop only fired
        # when the consensus already said SELL — which the biased models
        # almost never did, so the stop never triggered.
        prices = pd.Series(np.linspace(100.0, 85.0, 60))  # ~ -15% drawdown
        signal, reason = self.rm.get_risk_adjusted_signal(
            original_signal="BUY",
            confidence=0.9,
            risk_metrics=_risk_metrics(),
            price_data=prices,
            ticker="CRUDP.PA",
            is_holding=True,
            entry_price_index=100.0,
        )
        self.assertEqual(signal, "SELL")
        self.assertIn("INCOMING SIGNAL WAS BUY", reason.upper())

    def test_hard_stop_threshold_respected(self):
        # Between the soft alert (-5%) and the hard stop (-10%): no forced
        # SELL. At -9% with a weak SELL conviction, exit inertia still applies.
        prices = pd.Series(np.linspace(100.0, 91.0, 60))  # ~ -9%
        signal, _ = self.rm.get_risk_adjusted_signal(
            original_signal="SELL",
            confidence=0.10,
            risk_metrics=_risk_metrics(),
            price_data=prices,
            ticker="CRUDP.PA",
            is_holding=True,
            entry_price_index=100.0,
        )
        # -9% is above the -10% hard stop -> not forced; inertia can hold.
        self.assertEqual(signal, "HOLD")

    def test_hard_stop_customizable(self):
        rm = AdvancedRiskManager(config={"risk_parameters": {"hard_stop_drawdown": 0.05}})
        self.assertEqual(rm.hard_stop_drawdown, 0.05)


if __name__ == "__main__":
    unittest.main()
