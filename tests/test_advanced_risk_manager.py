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
        # Holding, deep loss beyond hard_stop_drawdown -> SELL passes despite
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
        self.assertIn("HARD STOP", reason.upper())

    def test_hard_stop_threshold_respected(self):
        # Just above the threshold (-12%) but not beyond -> still inertia if
        # conviction is low.
        prices = pd.Series(np.linspace(100.0, 89.0, 60))  # ~ -11%
        signal, _ = self.rm.get_risk_adjusted_signal(
            original_signal="SELL",
            confidence=0.10,
            risk_metrics=_risk_metrics(),
            price_data=prices,
            ticker="CRUDP.PA",
            is_holding=True,
            entry_price_index=100.0,
        )
        # At -11% (within -12% stop) and low conviction -> held by inertia.
        self.assertEqual(signal, "HOLD")

    def test_hard_stop_customizable(self):
        rm = AdvancedRiskManager(config={"risk_parameters": {"hard_stop_drawdown": 0.05}})
        self.assertEqual(rm.hard_stop_drawdown, 0.05)


if __name__ == "__main__":
    unittest.main()
