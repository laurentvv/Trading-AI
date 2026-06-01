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


def _make_decision(signal, confidence, strength_val, model_name="test_model"):
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
        model_name=model_name,
    )


class TestEnhancedDecisionEngine(unittest.TestCase):
    def setUp(self):
        self.engine = EnhancedDecisionEngine()

    def test_calculate_consensus_score_empty(self):
        self.assertEqual(self.engine._calculate_consensus_score([]), 1.0)

    def test_calculate_consensus_score_single(self):
        decisions = [_make_decision("BUY", 0.8, 1)]
        self.assertEqual(self.engine._calculate_consensus_score(decisions), 1.0)

    def test_calculate_consensus_score_zero_confidence(self):
        decisions = [
            _make_decision("BUY", 0.0, 1),
            _make_decision("HOLD", 0.0, 0),
            _make_decision("SELL", 0.0, -1),
        ]
        score = self.engine._calculate_consensus_score(decisions)
        self.assertTrue(0 <= score <= 1)

    def test_calculate_consensus_score_perfect_consensus(self):
        decisions = [
            _make_decision("STRONG_BUY", 0.9, 2),
            _make_decision("STRONG_BUY", 0.9, 2),
            _make_decision("STRONG_BUY", 0.9, 2),
        ]
        score = self.engine._calculate_consensus_score(decisions)
        self.assertEqual(score, 1.0)

    def test_calculate_consensus_score_high_disagreement(self):
        decisions = [
            _make_decision("STRONG_BUY", 1.0, 2),
            _make_decision("STRONG_SELL", 1.0, -2),
        ]
        score = self.engine._calculate_consensus_score(decisions)
        self.assertEqual(score, 0.0)

    def test_calculate_consensus_score_partial_agreement(self):
        decisions = [
            _make_decision("BUY", 0.8, 1),
            _make_decision("STRONG_BUY", 0.6, 2),
            _make_decision("HOLD", 0.4, 0),
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
            "Brent_spread": 2.0,
            "Nasdaq_above_ma200": True,
        }
        result = self.model.evaluate(indicators)
        self.assertIn("Brent Spread IDEAL", result["analysis"])
        self.assertGreater(result["confidence"], 0.5)

    def test_evaluate_with_extreme_brent_spread(self):
        indicators = {
            "WTI_price": 85.0,
            "Brent_price": 90.0,
            "Brent_spread": 20.0,
        }
        result = self.model.evaluate(indicators)
        self.assertIn("Brent Spread EXTREME", result["analysis"])


class TestDetectMarketRegime(unittest.TestCase):
    def setUp(self):
        self.engine = EnhancedDecisionEngine()

    def test_detect_market_regime_trending(self):
        self.assertEqual(self.engine._detect_market_regime({"adx": 30}), "trending")

    def test_detect_market_regime_ranging(self):
        self.assertEqual(self.engine._detect_market_regime({"adx": 15}), "ranging")

    def test_detect_market_regime_unknown(self):
        self.assertEqual(self.engine._detect_market_regime({}), "unknown")
        self.assertEqual(self.engine._detect_market_regime({"adx": 22}), "neutral")


class TestAdjustWeightsByRegime(unittest.TestCase):
    def setUp(self):
        self.engine = EnhancedDecisionEngine()

    def test_adjust_weights_trending_boosts_grebenkov(self):
        weights = {"grebenkov": 0.05, "timesfm": 0.20, "classic": 0.13}
        adjusted = self.engine._adjust_weights_by_regime(weights, "trending")
        raw_ratio = adjusted["grebenkov"] / sum(adjusted.values())
        base_ratio = weights["grebenkov"] / sum(weights.values())
        self.assertGreater(raw_ratio, base_ratio)

    def test_adjust_weights_ranging_reduces_grebenkov(self):
        weights = {"classic": 0.13, "tensortrade": 0.05, "grebenkov": 0.05}
        adjusted = self.engine._adjust_weights_by_regime(weights, "ranging")
        raw_ratio = adjusted["grebenkov"] / sum(adjusted.values())
        base_ratio = weights["grebenkov"] / sum(weights.values())
        self.assertLess(raw_ratio, base_ratio)

    def test_adjust_weights_neutral_no_change(self):
        weights = {"grebenkov": 0.05, "classic": 0.13}
        adjusted = self.engine._adjust_weights_by_regime(weights, "neutral")
        self.assertEqual(adjusted, weights)

    def test_adjust_weights_custom_multipliers(self):
        custom = {
            "trending": {"grebenkov": 3.0, "classic": 0.1},
            "ranging": {"classic": 2.0, "grebenkov": 0.2},
        }
        engine = EnhancedDecisionEngine(regime_adjustments=custom)
        weights = {"grebenkov": 0.05, "classic": 0.13, "timesfm": 0.20}
        adjusted = engine._adjust_weights_by_regime(weights, "trending")
        raw_ratio = adjusted["grebenkov"] / sum(adjusted.values())
        base_ratio = weights["grebenkov"] / sum(weights.values())
        self.assertGreater(raw_ratio, base_ratio * 2)


if __name__ == "__main__":
    unittest.main()
