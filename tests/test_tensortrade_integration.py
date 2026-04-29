import unittest
from datetime import datetime
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd

from tensortrade_model import get_tensortrade_prediction
from enhanced_decision_engine import (
    EnhancedDecisionEngine,
    ModelDecision,
    SignalStrength,
)


def _make_df(n=100):
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    data = {"Close": np.linspace(100, 110, n)}
    return pd.DataFrame(data, index=dates)


def _make_other_decision(signal="HOLD", confidence=0.5):
    return ModelDecision(
        signal=signal,
        confidence=confidence,
        strength=SignalStrength.HOLD,
        timestamp=datetime.now(),
        model_name="test_model",
    )


class TestTensorTradeIntegration(unittest.TestCase):
    @patch("tensortrade_model.PPO")
    def test_full_chain_creates_model_decision(self, mock_ppo_cls):
        mock_model = MagicMock()
        mock_model.predict.return_value = (np.array(1), None)
        mock_policy = MagicMock()
        mock_policy.obs_to_tensor.return_value = (MagicMock(),)
        mock_dist = MagicMock()
        mock_dist.distribution.probs.detach().numpy.return_value = np.array(
            [[0.1, 0.8, 0.1]]
        )
        mock_policy.get_distribution.return_value = mock_dist
        mock_model.policy = mock_policy
        mock_ppo_cls.return_value = mock_model

        df = _make_df(100)
        tensortrade_result = get_tensortrade_prediction(df)

        self.assertIn("signal", tensortrade_result)
        self.assertIn("confidence", tensortrade_result)

        engine = EnhancedDecisionEngine()

        classic_pred = 1
        classic_conf = 0.6
        text_llm_decision = {"signal": "HOLD", "confidence": 0.5}
        visual_llm_decision = {"signal": "HOLD", "confidence": 0.5}
        sentiment_decision = {"signal": "HOLD", "confidence": 0.5}

        result = engine.make_enhanced_decision(
            classic_pred=classic_pred,
            classic_conf=classic_conf,
            text_llm_decision=text_llm_decision,
            visual_llm_decision=visual_llm_decision,
            sentiment_decision=sentiment_decision,
            tensortrade_decision=tensortrade_result,
        )

        self.assertIsNotNone(result)
        tensortrade_models = [
            d for d in result.individual_decisions if d.model_name == "tensortrade"
        ]
        self.assertEqual(len(tensortrade_models), 1)
        td = tensortrade_models[0]
        self.assertEqual(td.signal, "BUY")
        self.assertAlmostEqual(td.confidence, 0.8)

    def test_tensortrade_weight_in_base_weights(self):
        engine = EnhancedDecisionEngine()
        self.assertIn("tensortrade", engine.base_weights)
        self.assertEqual(engine.base_weights["tensortrade"], 0.10)

    @patch("tensortrade_model.PPO")
    def test_consensus_score_includes_tensortrade(self, mock_ppo_cls):
        mock_model = MagicMock()
        mock_model.predict.return_value = (np.array(1), None)
        mock_policy = MagicMock()
        mock_policy.obs_to_tensor.return_value = (MagicMock(),)
        mock_dist = MagicMock()
        mock_dist.distribution.probs.detach().numpy.return_value = np.array(
            [[0.1, 0.9, 0.0]]
        )
        mock_policy.get_distribution.return_value = mock_dist
        mock_model.policy = mock_policy
        mock_ppo_cls.return_value = mock_model

        df = _make_df(100)
        tensortrade_result = get_tensortrade_prediction(df)

        engine = EnhancedDecisionEngine()
        td_signal = tensortrade_result.get("signal", "HOLD")
        td_strength = engine._normalize_signal(td_signal)

        decision = ModelDecision(
            signal=td_signal,
            confidence=tensortrade_result["confidence"],
            strength=td_strength,
            timestamp=datetime.now(),
            model_name="tensortrade",
        )

        score = engine._calculate_consensus_score([decision])
        self.assertTrue(0 <= score <= 1.0)

    @patch("tensortrade_model.PPO")
    def test_analysis_key_fallback(self, mock_ppo_cls):
        mock_model = MagicMock()
        mock_model.predict.return_value = (np.array(0), None)
        mock_policy = MagicMock()
        mock_policy.obs_to_tensor.return_value = (MagicMock(),)
        mock_dist = MagicMock()
        mock_dist.distribution.probs.detach().numpy.return_value = np.array(
            [[0.7, 0.2, 0.1]]
        )
        mock_policy.get_distribution.return_value = mock_dist
        mock_model.policy = mock_policy
        mock_ppo_cls.return_value = mock_model

        df = _make_df(100)
        result = get_tensortrade_prediction(df)
        self.assertNotIn("analysis", result)

        analysis = result.get("analysis", "TensorTrade RL policy output")
        self.assertEqual(analysis, "TensorTrade RL policy output")


if __name__ == "__main__":
    unittest.main()
