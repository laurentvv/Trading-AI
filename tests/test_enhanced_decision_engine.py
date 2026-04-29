import unittest
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from enhanced_decision_engine import (
    EnhancedDecisionEngine,
    ModelDecision,
    SignalStrength,
)


class TestEnhancedDecisionEngine(unittest.TestCase):
    def setUp(self):
        self.engine = EnhancedDecisionEngine()

    def create_decision(self, signal, confidence, strength_val):
        # Maps integer strength_val to SignalStrength enum roughly
        strength_map = {
            2: SignalStrength.STRONG_BUY,
            1: SignalStrength.BUY,
            0: SignalStrength.HOLD,
            -1: SignalStrength.SELL,
            -2: SignalStrength.STRONG_SELL,
        }
        return ModelDecision(
            signal=signal,
            confidence=confidence,
            strength=strength_map[strength_val],
            timestamp=datetime.now(),
            model_name="test_model",
        )

    def test_calculate_consensus_score_empty(self):
        """Test with empty list of decisions"""
        self.assertEqual(self.engine._calculate_consensus_score([]), 1.0)

    def test_calculate_consensus_score_single(self):
        """Test with a single decision"""
        decisions = [self.create_decision("BUY", 0.8, 1)]
        self.assertEqual(self.engine._calculate_consensus_score(decisions), 1.0)

    def test_calculate_consensus_score_zero_confidence(self):
        """Test with multiple decisions that all have 0.0 confidence (ZeroDivisionError fix)"""
        decisions = [
            self.create_decision("BUY", 0.0, 1),
            self.create_decision("HOLD", 0.0, 0),
            self.create_decision("SELL", 0.0, -1),
        ]
        score = self.engine._calculate_consensus_score(decisions)
        self.assertTrue(0 <= score <= 1)

    def test_calculate_consensus_score_perfect_consensus(self):
        """Test with multiple decisions that agree perfectly"""
        decisions = [
            self.create_decision("STRONG_BUY", 0.9, 2),
            self.create_decision("STRONG_BUY", 0.9, 2),
            self.create_decision("STRONG_BUY", 0.9, 2),
        ]
        score = self.engine._calculate_consensus_score(decisions)
        self.assertEqual(score, 1.0)

    def test_calculate_consensus_score_high_disagreement(self):
        """Test with decisions that are polar opposites"""
        decisions = [
            self.create_decision("STRONG_BUY", 1.0, 2),
            self.create_decision("STRONG_SELL", 1.0, -2),
        ]
        score = self.engine._calculate_consensus_score(decisions)
        # Variance of [2, -2] is 4, max_variance is 4, signal_agreement = 0
        # weighted_signal is 0. confidence_alignment = 1 - average(abs(2-0), abs(-2-0)) = 1 - 2 = -1
        # score = (0 + -1)/2 = -0.5, clamped to 0
        self.assertEqual(score, 0.0)

    def test_calculate_consensus_score_partial_agreement(self):
        """Test with a mix of decisions"""
        decisions = [
            self.create_decision("BUY", 0.8, 1),
            self.create_decision("STRONG_BUY", 0.6, 2),
            self.create_decision("HOLD", 0.4, 0),
        ]
        score = self.engine._calculate_consensus_score(decisions)
        self.assertTrue(0 < score < 1.0)


class TestVincentGanneModel(unittest.TestCase):
    def setUp(self):
        from enhanced_decision_engine import VincentGanneModel

        self.model = VincentGanneModel()

    def test_evaluate_with_brent_spread(self):
        indicators = {
            "WTI_price": 75.0,
            "Brent_price": 78.0,
            "Brent_spread": 2.0,  # Ideal
            "Nasdaq_above_ma200": True,
        }
        result = self.model.evaluate(indicators)
        self.assertIn("Brent Spread IDEAL", result["analysis"])
        self.assertGreater(result["confidence"], 0.5)

    def test_evaluate_with_extreme_brent_spread(self):
        indicators = {
            "WTI_price": 85.0,
            "Brent_price": 90.0,
            "Brent_spread": 20.0,  # Extreme
        }
        result = self.model.evaluate(indicators)
        self.assertIn("Brent Spread EXTREME", result["analysis"])


if __name__ == "__main__":
    unittest.main()
