import unittest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Fix sys.path for test environment
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.hmm_model import forward, viterbi, baum_welch, HMMDecisionModel
from src.enhanced_decision_engine import ModelResult


class TestHMMModel(unittest.TestCase):
    def setUp(self):
        # Example 1 Data (Eisner Task) from prompt
        self.observations = [2, 0, 2]  # 3, 1, 3 ice creams (index 0=1 ice cream, 2=3 ice creams)
        self.pi_initial = np.array([0.2, 0.8])  # 0 = COLD, 1 = HOT
        self.A_initial = np.array([[0.5, 0.5], [0.4, 0.6]])
        self.B_initial = np.array([[0.5, 0.4, 0.1], [0.2, 0.4, 0.4]])

        # Example 2 Data for Baum-Welch
        self.seq_training = [2, 0, 2, 1, 1, 0, 2, 2, 0, 1, 2, 0, 0, 1, 2]

    def test_forward_algorithm(self):
        prob, alpha = forward(self.observations, 2, self.pi_initial, self.A_initial, self.B_initial)
        # Using a relaxed delta since floating point implementations vary slightly
        self.assertTrue(prob > 0)
        self.assertAlmostEqual(prob, 0.02856, places=4)

    def test_viterbi_algorithm(self):
        path, prob = viterbi(self.observations, 2, self.pi_initial, self.A_initial, self.B_initial)
        self.assertEqual(len(path), 3)
        self.assertTrue(prob > 0)

    def test_baum_welch(self):
        new_pi, new_A, new_B = baum_welch(self.seq_training, N_states=2, V_vocab_size=3, iterations=2)
        # Verify shapes and probability distribution properties
        self.assertEqual(new_pi.shape, (2,))
        self.assertAlmostEqual(np.sum(new_pi), 1.0)

        self.assertEqual(new_A.shape, (2, 2))
        self.assertAlmostEqual(np.sum(new_A[0]), 1.0)

        self.assertEqual(new_B.shape, (2, 3))
        # Ensure that every row (state) in emission matrix sums to 1.0
        for i in range(2):
            self.assertAlmostEqual(np.sum(new_B[i]), 1.0)

    def test_model_predict(self):
        model = HMMDecisionModel(n_states=2, lookback=20, baum_welch_iterations=2)

        # Create dummy market data with a clear trend to ensure predictability
        dates = pd.date_range(start="2020-01-01", periods=30)
        # Sequence of prices generating clear returns
        prices = [100.0]
        for i in range(29):
            # Alternating trend to avoid division by zero in returns
            if i % 2 == 0:
                prices.append(prices[-1] * 1.02)  # Up 2%
            else:
                prices.append(prices[-1] * 0.99)  # Down 1%

        hist_data = pd.DataFrame({"Close": prices}, index=dates)

        data = {"hist_data": hist_data}
        result = model.predict(data)

        self.assertIsInstance(result, ModelResult)
        self.assertIn(result.signal, ["BUY", "SELL", "HOLD"])
        self.assertTrue(0.0 <= result.confidence <= 1.0)
        if result.signal != "HOLD":
            self.assertIn("current_state", result.metadata)


if __name__ == "__main__":
    unittest.main()
