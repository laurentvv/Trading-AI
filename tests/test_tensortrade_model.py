import unittest
from unittest.mock import patch, MagicMock

import gymnasium as gym
import numpy as np
import pandas as pd

from src.tensortrade_model import get_tensortrade_prediction


def _make_df(n=100, close_col="Close"):
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    data = {close_col: np.linspace(100, 110, n)}
    return pd.DataFrame(data, index=dates)


class SimpleTradingEnv(gym.Env):
    def __init__(self, prices):
        super().__init__()
        self.prices = prices
        self.current_step = 0
        self.action_space = gym.spaces.Discrete(3)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32
        )

    def step(self, action):
        self.current_step += 1
        done = self.current_step >= len(self.prices) - 1
        reward = 0
        if not done:
            price_change = (
                self.prices[self.current_step]
                - self.prices[self.current_step - 1]
            )
            if action == 1:
                reward = price_change
            elif action == 2:
                reward = -price_change
        obs = self._get_obs()
        return obs, reward, done, False, {}

    def reset(self, seed=None, options=None):
        self.current_step = 5
        return self._get_obs(), {}

    def _get_obs(self):
        idx = self.current_step
        diffs = np.diff(self.prices[idx - 5 : idx + 1])
        return np.array(diffs, dtype=np.float32)


class TestInsufficientData(unittest.TestCase):
    def test_returns_hold_with_low_confidence(self):
        df = _make_df(n=30)
        result = get_tensortrade_prediction(df)
        self.assertEqual(result["signal"], "HOLD")
        self.assertEqual(result["confidence"], 0.5)


class TestMissingCloseColumn(unittest.TestCase):
    def test_returns_hold_with_zero_confidence(self):
        df = pd.DataFrame({"Open": np.random.rand(100)})
        result = get_tensortrade_prediction(df)
        self.assertEqual(result["signal"], "HOLD")
        self.assertEqual(result["confidence"], 0.0)


class TestLowercaseCloseColumn(unittest.TestCase):
    @patch("stable_baselines3.PPO", create=True)
    def test_works_with_lowercase_close(self, mock_ppo_cls):
        mock_model = MagicMock()
        mock_model.predict.return_value = (np.array(1), None)
        mock_policy = MagicMock()
        mock_policy.obs_to_tensor.return_value = (MagicMock(),)
        mock_dist = MagicMock()
        mock_dist.distribution.probs.detach().numpy.return_value = np.array(
            [[0.2, 0.6, 0.2]]
        )
        mock_policy.get_distribution.return_value = mock_dist
        mock_model.policy = mock_policy
        mock_ppo_cls.return_value = mock_model

        df = _make_df(n=100, close_col="close")
        result = get_tensortrade_prediction(df)
        self.assertIn(result["signal"], ["BUY", "SELL", "HOLD"])
        self.assertTrue(0.0 <= result["confidence"] <= 1.0)


class TestSuccessfulPrediction(unittest.TestCase):
    @patch("stable_baselines3.PPO", create=True)
    def test_returns_valid_signal_and_confidence(self, mock_ppo_cls):
        mock_model = MagicMock()
        mock_model.predict.return_value = (np.array(2), None)
        mock_policy = MagicMock()
        mock_policy.obs_to_tensor.return_value = (MagicMock(),)
        mock_dist = MagicMock()
        mock_dist.distribution.probs.detach().numpy.return_value = np.array(
            [[0.1, 0.3, 0.6]]
        )
        mock_policy.get_distribution.return_value = mock_dist
        mock_model.policy = mock_policy
        mock_ppo_cls.return_value = mock_model

        df = _make_df(n=100)
        result = get_tensortrade_prediction(df)
        self.assertEqual(result["signal"], "SELL")
        self.assertAlmostEqual(result["confidence"], 0.6)


class TestEnvStepLogic(unittest.TestCase):
    def test_buy_reward_positive_when_price_up(self):
        prices = np.array([100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0])
        env = SimpleTradingEnv(prices)
        env.reset()
        _, reward_buy, _, _, _ = env.step(1)
        self.assertGreater(reward_buy, 0)

    def test_sell_reward_positive_when_price_down(self):
        prices = np.array(
            [107.0, 106.0, 105.0, 104.0, 103.0, 102.0, 101.0, 100.0]
        )
        env = SimpleTradingEnv(prices)
        env.reset()
        _, reward_sell, _, _, _ = env.step(2)
        self.assertGreater(reward_sell, 0)


class TestEnvObservationShape(unittest.TestCase):
    def test_observation_shape_is_5(self):
        prices = np.array([100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0])
        env = SimpleTradingEnv(prices)
        obs, info = env.reset()
        self.assertEqual(obs.shape, (5,))


class TestEnvReset(unittest.TestCase):
    def test_reset_returns_obs_and_info_dict(self):
        prices = np.array([100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0])
        env = SimpleTradingEnv(prices)
        result = env.reset()
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        obs, info = result
        self.assertIsInstance(info, dict)


class TestExceptionHandling(unittest.TestCase):
    @patch("stable_baselines3.PPO", side_effect=RuntimeError("PPO crash"))
    def test_returns_hold_on_exception(self, mock_ppo_cls):
        df = _make_df(n=100)
        result = get_tensortrade_prediction(df)
        self.assertEqual(result["signal"], "HOLD")
        self.assertEqual(result["confidence"], 0.0)


if __name__ == "__main__":
    unittest.main()
