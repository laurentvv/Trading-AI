import unittest
from src.sentiment_analysis import get_sentiment_decision_from_score


class TestSentimentAnalysis(unittest.TestCase):
    def test_buy_signal(self):
        """Test that scores > 0.15 result in BUY signal with correct confidence."""
        result = get_sentiment_decision_from_score(0.20)
        self.assertEqual(result["signal"], "BUY")
        self.assertAlmostEqual(result["confidence"], 0.70)
        self.assertEqual(result["analysis"], "Sentiment score from news API is 0.20.")

        # Test extreme high score (confidence capping)
        result_extreme = get_sentiment_decision_from_score(1.5)
        self.assertEqual(result_extreme["signal"], "BUY")
        self.assertEqual(result_extreme["confidence"], 1.0)

    def test_sell_signal(self):
        """Test that scores < -0.15 result in SELL signal with correct confidence."""
        result = get_sentiment_decision_from_score(-0.20)
        self.assertEqual(result["signal"], "SELL")
        self.assertAlmostEqual(result["confidence"], 0.70)
        self.assertEqual(result["analysis"], "Sentiment score from news API is -0.20.")

        # Test extreme low score (confidence capping)
        result_extreme = get_sentiment_decision_from_score(-1.5)
        self.assertEqual(result_extreme["signal"], "SELL")
        self.assertEqual(result_extreme["confidence"], 1.0)

    def test_hold_signal(self):
        """Test that scores between -0.15 and 0.15 result in HOLD signal."""
        result = get_sentiment_decision_from_score(0.0)
        self.assertEqual(result["signal"], "HOLD")
        self.assertAlmostEqual(result["confidence"], 0.50)

        result_pos = get_sentiment_decision_from_score(0.10)
        self.assertEqual(result_pos["signal"], "HOLD")
        self.assertAlmostEqual(result_pos["confidence"], 0.60)

        result_neg = get_sentiment_decision_from_score(-0.10)
        self.assertEqual(result_neg["signal"], "HOLD")
        self.assertAlmostEqual(result_neg["confidence"], 0.60)

    def test_boundary_values(self):
        """Test exact boundary values (0.15 and -0.15)."""
        result_upper = get_sentiment_decision_from_score(0.15)
        self.assertEqual(result_upper["signal"], "HOLD")
        self.assertAlmostEqual(result_upper["confidence"], 0.65)

        result_lower = get_sentiment_decision_from_score(-0.15)
        self.assertEqual(result_lower["signal"], "HOLD")
        self.assertAlmostEqual(result_lower["confidence"], 0.65)


if __name__ == "__main__":
    unittest.main()
