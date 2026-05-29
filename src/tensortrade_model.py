import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import gymnasium as gym
from stable_baselines3 import PPO

logger = logging.getLogger(__name__)

_MODEL_DIR = Path("data_cache") / "tensortrade"
_MODEL_PATH = _MODEL_DIR / "ppo_model.zip"
_METADATA_PATH = _MODEL_DIR / "metadata.json"
_INITIAL_TIMESTEPS = 2000
_FINE_TUNE_TIMESTEPS = 500


def _load_metadata():
    if _METADATA_PATH.exists():
        with open(_METADATA_PATH, "r") as f:
            return json.load(f)
    return {"total_timesteps": 0, "last_trained": None, "obs_shape": None}


def _save_metadata(total_timesteps, obs_shape):
    _MODEL_DIR.mkdir(parents=True, exist_ok=True)
    metadata = {
        "total_timesteps": total_timesteps,
        "last_trained": datetime.now().isoformat(),
        "obs_shape": list(obs_shape),
    }
    with open(_METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)


class SimpleTradingEnv(gym.Env):
    def __init__(self, prices):
        super().__init__()
        self.prices = prices
        self.current_step = 0
        self.hold_count = 0
        self.action_space = gym.spaces.Discrete(3)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32
        )

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
        self.current_step = 10
        self.hold_count = 0
        return self._get_obs(), {}

    def _get_obs(self):
        idx = self.current_step
        diffs = np.diff(self.prices[idx - 5 : idx + 1])
        recent = self.prices[idx - 5 : idx + 1]
        returns = np.diff(recent) / recent[:-1]
        return np.concatenate([diffs, returns]).astype(np.float32)


def get_tensortrade_prediction(df: pd.DataFrame) -> dict:
    """
    Génère une prédiction RL basée sur l'action actuelle via PPO (stable-baselines3).
    Le modèle est persisté entre les appels pour accumuler l'apprentissage.
    """
    logger.info("Démarrage de l'analyse TensorTrade...")

    try:
        if len(df) < 50:
            logger.warning("Pas assez de données pour TensorTrade, retour au neutre.")
            return {"signal": "HOLD", "confidence": 0.5}

        close_col = "Close" if "Close" in df.columns else "close"
        if close_col not in df.columns:
            logger.warning("Colonne Close manquante, TensorTrade ignoré.")
            return {"signal": "HOLD", "confidence": 0.0}

        data = df.copy()

        prices = data[close_col].values
        env = SimpleTradingEnv(prices)

        _MODEL_DIR.mkdir(parents=True, exist_ok=True)

        if _MODEL_PATH.exists():
            metadata = _load_metadata()
            if metadata.get("obs_shape") and metadata["obs_shape"] != list(
                env.observation_space.shape
            ):
                logger.info("Observation space changed, retraining from scratch.")
                _MODEL_PATH.unlink(missing_ok=True)
            else:
                try:
                    model = PPO.load(_MODEL_PATH, env=env)
                    logger.info(
                        f"Modèle PPO chargé depuis le cache ({metadata.get('total_timesteps', '?')} timesteps cumulés)"
                    )
                    model.learn(total_timesteps=_FINE_TUNE_TIMESTEPS)
                    total_ts = metadata.get("total_timesteps", 0) + _FINE_TUNE_TIMESTEPS
                    _save_metadata(total_ts, env.observation_space.shape)
                    model.save(_MODEL_PATH)
                    logger.info(
                        f"Modèle PPO sauvegardé ({_FINE_TUNE_TIMESTEPS} timesteps ajoutés, total: {total_ts})"
                    )
                except Exception as e:
                    logger.warning(f"Erreur chargement modèle PPO, retraining from scratch: {e}")
                    model = PPO("MlpPolicy", env, n_steps=64, batch_size=32, verbose=0)
                    model.learn(total_timesteps=_INITIAL_TIMESTEPS)
                    _save_metadata(_INITIAL_TIMESTEPS, env.observation_space.shape)
                    model.save(_MODEL_PATH)

        if not _MODEL_PATH.exists():
            logger.info("Aucun modèle PPO en cache, entraînement initial...")
            model = PPO("MlpPolicy", env, n_steps=128, batch_size=64, verbose=0)
            model.learn(total_timesteps=_INITIAL_TIMESTEPS)
            _save_metadata(_INITIAL_TIMESTEPS, env.observation_space.shape)
            model.save(_MODEL_PATH)
            logger.info(f"Modèle PPO initial sauvegardé ({_INITIAL_TIMESTEPS} timesteps)")

        obs, _ = env.reset()
        for _ in range(len(prices) - 11):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, _, _ = env.step(action)
            if done:
                break

        action, state = model.predict(obs, deterministic=True)

        signal = "HOLD"
        if action == 1:
            signal = "BUY"
        elif action == 2:
            signal = "SELL"

        try:
            obs_tensor = model.policy.obs_to_tensor(obs)[0]
            dist = model.policy.get_distribution(obs_tensor)
            probs = dist.distribution.probs.detach().numpy()[0]
            confidence = float(probs[action])
        except Exception:
            confidence = 0.5

        logger.info(f"TensorTrade prédiction finale : {signal} (Confiance: {confidence:.2f})")

        return {"signal": signal, "confidence": confidence}

    except Exception as e:
        logger.error(f"Erreur lors de l'exécution de TensorTrade: {e}")
        import traceback

        logger.debug(traceback.format_exc())
        return {"signal": "HOLD", "confidence": 0.0}
