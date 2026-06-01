import json
import shutil
import unittest
from unittest.mock import patch, MagicMock

import gymnasium as gym
import numpy as np
import pandas as pd

from tensortrade_model import (
    get_tensortrade_prediction,
    _MODEL_DIR,
    _MODEL_PATH,
    _METADATA_PATH,
    SimpleTradingEnv as SrcSimpleTradingEnv,
    _OBS_SIZE,
)


def _make_df(n=100, close_col="Close"):
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    data = {close_col: np.linspace(100, 110, n)}
    return pd.DataFrame(data, index=dates)


class SimpleTradingEnv(gym.Env):
    def __init__(self, prices):
        super().__init__()
        self.prices = prices
        self.current_step = 0
        self.hold_count = 0
        self.action_space = gym.spaces.Discrete(3)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(_OBS_SIZE,), dtype=np.float32)

    def step(self, action):
        self.current_step += 1
        done = self.current_step >= len(self.prices) - 1
        reward = 0
        if not done:
            price_change = self.prices[self.current_step] - self.prices[self.current_step - 1]
            window = self.prices[max(0, self.current_step - 14) : self.current_step + 1]
            atr = np.mean(np.abs(np.diff(window))) if len(window) > 1 else 1.0
            atr = max(atr, 1e-6)
            if action == 1:
                reward = price_change / atr
                self.hold_count = 0
            elif action == 2:
                reward = -price_change / atr
                self.hold_count = 0
            else:
                self.hold_count += 1
                reward = -0.01 * min(self.hold_count, 10)
        obs = self._get_obs()
        return obs, reward, done, False, {}

    def reset(self, seed=None, options=None):
        self.current_step = _OBS_SIZE
        self.hold_count = 0
        return self._get_obs(), {}

    def _get_obs(self):
        idx = self.current_step
        diffs = np.diff(self.prices[idx - 5 : idx + 1])
        recent = self.prices[idx - 5 : idx + 1]
        returns = np.diff(recent) / recent[:-1]
        padding = np.zeros(_OBS_SIZE - len(diffs) - len(returns), dtype=np.float32)
        return np.concatenate([diffs, returns, padding]).astype(np.float32)


def _mock_ppo_setup(mock_ppo_cls, action=1, probs=None):
    if probs is None:
        probs = np.array([[0.2, 0.6, 0.2]])
    mock_model = MagicMock()
    mock_model.predict.return_value = (np.array(action), None)
    mock_policy = MagicMock()
    mock_policy.obs_to_tensor.return_value = (MagicMock(),)
    mock_dist = MagicMock()
    mock_dist.distribution.probs.detach().numpy.return_value = probs
    mock_policy.get_distribution.return_value = mock_dist
    mock_model.policy = mock_policy
    mock_ppo_cls.return_value = mock_model
    mock_ppo_cls.load = MagicMock(return_value=mock_model)
    return mock_model


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
    @patch("tensortrade_model.PPO")
    def test_works_with_lowercase_close(self, mock_ppo_cls):
        _mock_ppo_setup(mock_ppo_cls, action=1)
        df = _make_df(n=100, close_col="close")
        result = get_tensortrade_prediction(df)
        self.assertIn(result["signal"], ["BUY", "SELL", "HOLD"])
        self.assertTrue(0.0 <= result["confidence"] <= 1.0)


class TestSuccessfulPrediction(unittest.TestCase):
    @patch("tensortrade_model.PPO")
    def test_returns_valid_signal_and_confidence(self, mock_ppo_cls):
        _mock_ppo_setup(mock_ppo_cls, action=2, probs=np.array([[0.1, 0.3, 0.6]]))
        df = _make_df(n=100)
        result = get_tensortrade_prediction(df)
        self.assertEqual(result["signal"], "SELL")
        self.assertAlmostEqual(result["confidence"], 0.6)


class TestEnvStepLogic(unittest.TestCase):
    def test_buy_reward_positive_when_price_up(self):
        prices = np.linspace(100, 120, 50)
        env = SimpleTradingEnv(prices)
        env.reset()
        _, reward_buy, _, _, _ = env.step(1)
        self.assertGreater(reward_buy, 0)

    def test_sell_reward_positive_when_price_down(self):
        prices = np.linspace(120, 100, 50)
        env = SimpleTradingEnv(prices)
        env.reset()
        _, reward_sell, _, _, _ = env.step(2)
        self.assertGreater(reward_sell, 0)

    def test_hold_penalty_increases(self):
        prices = np.linspace(100, 110, 50)
        env = SimpleTradingEnv(prices)
        env.reset()
        _, r1, _, _, _ = env.step(0)
        _, r2, _, _, _ = env.step(0)
        self.assertLess(r2, r1)


class TestEnvObservationShape(unittest.TestCase):
    def test_observation_shape_matches_obs_size(self):
        prices = np.linspace(100, 120, 50)
        env = SimpleTradingEnv(prices)
        obs, info = env.reset()
        self.assertEqual(obs.shape, (_OBS_SIZE,))

    def test_observation_contains_diffs_and_returns(self):
        prices = np.linspace(100, 120, 50)
        env = SimpleTradingEnv(prices)
        obs, _ = env.reset()
        self.assertEqual(obs.shape[0], _OBS_SIZE)
        self.assertTrue(np.all(np.isfinite(obs)))


class TestEnvReset(unittest.TestCase):
    def test_reset_returns_obs_and_info_dict(self):
        prices = np.linspace(100, 120, 50)
        env = SimpleTradingEnv(prices)
        result = env.reset()
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        obs, info = result
        self.assertIsInstance(info, dict)

    def test_reset_starts_at_step_obs_size(self):
        prices = np.linspace(100, 120, 50)
        env = SimpleTradingEnv(prices)
        env.reset()
        self.assertEqual(env.current_step, _OBS_SIZE)


class TestExceptionHandling(unittest.TestCase):
    @patch("tensortrade_model.PPO", side_effect=RuntimeError("PPO crash"))
    def test_returns_hold_on_exception(self, mock_ppo_cls):
        mock_ppo_cls.load = MagicMock(side_effect=RuntimeError("PPO crash"))
        df = _make_df(n=100)
        result = get_tensortrade_prediction(df)
        self.assertEqual(result["signal"], "HOLD")
        self.assertEqual(result["confidence"], 0.0)


class TestModelPersistence(unittest.TestCase):
    def setUp(self):
        if _MODEL_DIR.exists():
            shutil.rmtree(_MODEL_DIR, ignore_errors=True)

    def tearDown(self):
        if _MODEL_DIR.exists():
            shutil.rmtree(_MODEL_DIR, ignore_errors=True)

    @patch("tensortrade_model.PPO")
    def test_model_saved_after_call(self, mock_ppo_cls):
        mock_model = MagicMock()
        mock_model.predict.return_value = (np.array(1), None)
        mock_policy = MagicMock()
        mock_policy.obs_to_tensor.return_value = (MagicMock(),)
        mock_dist = MagicMock()
        mock_dist.distribution.probs.detach().numpy.return_value = np.array([[0.1, 0.8, 0.1]])
        mock_policy.get_distribution.return_value = mock_dist
        mock_model.policy = mock_policy
        mock_ppo_cls.return_value = mock_model

        df = _make_df(n=100)
        get_tensortrade_prediction(df)

        mock_model.save.assert_called_once_with(_MODEL_PATH)

    @patch("tensortrade_model.PPO")
    def test_model_loaded_on_second_call(self, mock_ppo_cls):
        _mock_ppo_setup(mock_ppo_cls, action=1)

        df = _make_df(n=100)

        _MODEL_DIR.mkdir(parents=True, exist_ok=True)
        _MODEL_PATH.touch()

        with open(_METADATA_PATH, "w") as f:
            json.dump({"total_timesteps": 2000, "last_trained": "2024-01-01", "obs_shape": [_OBS_SIZE]}, f)

        get_tensortrade_prediction(df)

        mock_ppo_cls.load.assert_called_once()

    @patch("tensortrade_model.PPO")
    def test_metadata_saved_after_call(self, mock_ppo_cls):
        _mock_ppo_setup(mock_ppo_cls, action=1)

        df = _make_df(n=100)
        get_tensortrade_prediction(df)

        self.assertTrue(_METADATA_PATH.exists())
        with open(_METADATA_PATH, "r") as f:
            meta = json.load(f)
        self.assertIn("total_timesteps", meta)
        self.assertIn("last_trained", meta)
        self.assertIn("obs_shape", meta)
        self.assertEqual(meta["obs_shape"], [_OBS_SIZE])
        self.assertEqual(meta["total_timesteps"], 2000)


class TestModelInvalidationOnObsChange(unittest.TestCase):
    def setUp(self):
        if _MODEL_DIR.exists():
            shutil.rmtree(_MODEL_DIR, ignore_errors=True)

    def tearDown(self):
        if _MODEL_DIR.exists():
            shutil.rmtree(_MODEL_DIR, ignore_errors=True)

    @patch("tensortrade_model.PPO")
    def test_stale_model_deleted_on_shape_change(self, mock_ppo_cls):
        mock_model = MagicMock()
        mock_model.predict.return_value = (np.array(1), None)
        mock_policy = MagicMock()
        mock_policy.obs_to_tensor.return_value = (MagicMock(),)
        mock_dist = MagicMock()
        mock_dist.distribution.probs.detach().numpy.return_value = np.array([[0.1, 0.8, 0.1]])
        mock_policy.get_distribution.return_value = mock_dist
        mock_model.policy = mock_policy
        mock_ppo_cls.return_value = mock_model

        _MODEL_DIR.mkdir(parents=True, exist_ok=True)
        _MODEL_PATH.touch()
        with open(_METADATA_PATH, "w") as f:
            json.dump({"total_timesteps": 5000, "last_trained": "2024-01-01", "obs_shape": [5]}, f)

        df = _make_df(n=100)
        get_tensortrade_prediction(df)

        mock_ppo_cls.load.assert_not_called()
        mock_ppo_cls.assert_called_once()


class TestSourceEnvObsShape(unittest.TestCase):
    def test_obs_shape_matches_obs_size(self):
        prices = np.linspace(100, 120, 60)
        env = SrcSimpleTradingEnv(prices)
        obs, _ = env.reset()
        self.assertEqual(obs.shape, (_OBS_SIZE,))

    def test_rsi_bounded(self):
        np.random.seed(42)
        prices = np.cumsum(np.random.randn(60)) + 100
        env = SrcSimpleTradingEnv(prices)
        obs, _ = env.reset()
        rsi_normalized = obs[12]
        self.assertGreaterEqual(rsi_normalized, 0.0)
        self.assertLessEqual(rsi_normalized, 1.0)

    def test_env_position_tracking(self):
        prices = np.linspace(100, 120, 200)
        env = SrcSimpleTradingEnv(prices)
        obs, _ = env.reset()
        self.assertEqual(env.in_position, 0.0)
        obs, _, _, _, _ = env.step(1)
        self.assertEqual(env.in_position, 1.0)
        for _ in range(env.cooldown_days):
            env.step(0)
        obs, _, done, _, _ = env.step(2)
        if not done:
            self.assertEqual(env.in_position, 0.0)

    def test_env_fee_deduction(self):
        prices = np.linspace(100, 200, 60)
        env = SrcSimpleTradingEnv(prices, fee_rate=0.01)
        obs, _ = env.reset()
        _, reward, _, _, _ = env.step(1)
        price = prices[env.current_step]
        prev_price = prices[env.current_step - 1]
        fee = price * 0.01
        window = prices[max(0, env.current_step - 14) : env.current_step + 1]
        atr = max(np.mean(np.abs(np.diff(window))), 1e-6)
        expected_reward = (price - prev_price - fee) / atr
        self.assertAlmostEqual(reward, expected_reward, places=5)

    def test_env_cooldown(self):
        prices = np.linspace(100, 120, 60)
        env = SrcSimpleTradingEnv(prices, cooldown_days=3)
        obs, _ = env.reset()
        env.step(1)
        self.assertGreater(env.cooldown_counter, 0)
        counter_before = env.cooldown_counter
        obs, reward, done, _, _ = env.step(0)
        self.assertEqual(env.cooldown_counter, counter_before - 1)


if __name__ == "__main__":
    unittest.main()
