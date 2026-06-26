import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import gymnasium as gym
from stable_baselines3 import PPO
from src.enhanced_decision_engine import ModelResult

logger = logging.getLogger(__name__)

_MODEL_DIR = Path("data_cache") / "tensortrade"
_MODEL_PATH = _MODEL_DIR / "ppo_model.zip"
_METADATA_PATH = _MODEL_DIR / "metadata.json"
_INITIAL_TIMESTEPS = 2000
_FINE_TUNE_TIMESTEPS = 500
_FEE_RATE = 0.001
_COOLDOWN_DAYS = 5
# PPO confidence cap. The policy distribution collapses toward one
# near-deterministic action after repeated in-call fine-tuning
# (_FINE_TUNE_TIMESTEPS each cycle, no entropy regularization), so the raw
# probs[action] is an over-confident, uncalibrated value (~0.88 systematically
# in prod). Capping it prevents this single model from inflating the consensus
# weighted_score and final_confidence. See ADR-002.
_CONFIDENCE_CAP = 0.75
# 20 features: diffs(5) + returns(5) + pos(1) + days_since(1) + vol(1) +
# rsi_14(1) + price_vs_sma(1) + fee_impact(1) + mom_10(1) + mom_20(1) + cum_ret_5(1) + norm_price(1)
_OBS_SIZE = 20


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
    def __init__(self, prices, fee_rate=_FEE_RATE, cooldown_days=_COOLDOWN_DAYS):
        super().__init__()
        self.prices = np.asarray(prices, dtype=np.float64)
        self.fee_rate = fee_rate
        self.cooldown_days = cooldown_days
        self.current_step = 0
        self.hold_count = 0
        self.in_position = 0.0
        self.entry_price = 0.0
        self.cooldown_counter = 0
        self.action_space = gym.spaces.Discrete(3)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(_OBS_SIZE,), dtype=np.float32)

    def step(self, action):
        self.current_step += 1
        done = self.current_step >= len(self.prices) - 1

        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            action = 0

        reward = 0.0
        if not done:
            price = self.prices[self.current_step]
            prev_price = self.prices[self.current_step - 1]
            price_change = price - prev_price
            window = self.prices[max(0, self.current_step - 14) : self.current_step + 1]
            atr = np.mean(np.abs(np.diff(window))) if len(window) > 1 else 1.0
            atr = max(atr, 1e-6)

            if action == 1 and self.in_position == 0:
                fee = price * self.fee_rate
                reward = (price_change - fee) / atr
                self.in_position = 1.0
                self.entry_price = price
                self.hold_count = 0
                self.cooldown_counter = self.cooldown_days
            elif action == 2 and self.in_position > 0:
                fee = price * self.fee_rate
                pnl = (price - self.entry_price) - fee
                reward = pnl / atr
                self.in_position = 0.0
                self.entry_price = 0.0
                self.hold_count = 0
                self.cooldown_counter = self.cooldown_days
            elif action == 0 and self.in_position > 0:
                reward = price_change / atr
                self.hold_count += 1
            else:
                self.hold_count += 1
                reward = -0.01 * min(self.hold_count, 10)

        obs = self._get_obs()
        return obs, reward, done, False, {}

    def reset(self, seed=None, **kwargs):
        self.current_step = 20
        self.hold_count = 0
        self.in_position = 0.0
        self.entry_price = 0.0
        self.cooldown_counter = 0
        return self._get_obs(), {}

    def _get_obs(self):
        idx = self.current_step
        diffs = np.diff(self.prices[idx - 5 : idx + 1])
        recent = self.prices[idx - 5 : idx + 1]
        returns = np.diff(recent) / np.maximum(recent[:-1], 1e-8)

        pos = np.array([self.in_position], dtype=np.float32)
        days_since = np.array([self.cooldown_counter / max(self.cooldown_days, 1)], dtype=np.float32)

        w20 = self.prices[max(0, idx - 19) : idx + 1]
        vol_20 = np.std(np.diff(w20) / np.maximum(w20[:-1], 1e-8)) if len(w20) > 2 else 0.0
        vol = np.array([vol_20], dtype=np.float32)

        changes = self.prices[idx - 13 : idx + 1] - self.prices[idx - 14 : idx]
        gains = np.maximum(changes, 0.0)
        losses = np.maximum(-changes, 0.0)
        avg_gain = np.mean(gains) if len(gains) > 0 else 0.0
        avg_loss = np.mean(losses) if len(losses) > 0 else 1e-8
        rs = avg_gain / max(avg_loss, 1e-8)
        rsi_14 = np.array([100.0 - (100.0 / (1.0 + rs))], dtype=np.float32) / 100.0

        sma20 = np.mean(self.prices[max(0, idx - 19) : idx + 1]) if idx >= 5 else self.prices[idx]
        price_vs_sma = np.array([(self.prices[idx] - sma20) / max(sma20, 1e-8)], dtype=np.float32)

        fee_impact = np.array([self.fee_rate], dtype=np.float32)

        mom_10 = np.array(
            [(self.prices[idx] - self.prices[max(0, idx - 9)]) / max(self.prices[max(0, idx - 9)], 1e-8)],
            dtype=np.float32,
        )
        mom_20 = np.array(
            [(self.prices[idx] - self.prices[max(0, idx - 19)]) / max(self.prices[max(0, idx - 19)], 1e-8)],
            dtype=np.float32,
        )
        cum_ret_5 = np.array(
            [(self.prices[idx] / max(self.prices[max(0, idx - 4)], 1e-8)) - 1.0],
            dtype=np.float32,
        )
        norm_price = np.array(
            [self.prices[idx] / max(np.max(self.prices[max(0, idx - 49) : idx + 1]), 1e-8)],
            dtype=np.float32,
        )

        return np.concatenate(
            [
                diffs,
                returns,
                pos,
                days_since,
                vol,
                rsi_14,
                price_vs_sma,
                fee_impact,
                mom_10,
                mom_20,
                cum_ret_5,
                norm_price,
            ]
        ).astype(np.float32)


def get_tensortrade_prediction(df: pd.DataFrame) -> ModelResult:
    """
    Génère une prédiction RL basée sur l'action actuelle via PPO (stable-baselines3).
    Le modèle est persisté entre les appels pour accumuler l'apprentissage.
    """
    logger.info("Démarrage de l'analyse TensorTrade...")

    try:
        if len(df) < 50:
            logger.warning("Pas assez de données pour TensorTrade, retour au neutre.")
            return ModelResult("HOLD", 0.5, "Not enough data for TensorTrade")

        close_col = "Close" if "Close" in df.columns else "close"
        if close_col not in df.columns:
            logger.warning("Colonne Close manquante, TensorTrade ignoré.")
            return ModelResult("HOLD", 0.0, "Missing Close column")

        data = df.copy()

        prices = data[close_col].values
        env = SimpleTradingEnv(prices)

        _MODEL_DIR.mkdir(parents=True, exist_ok=True)

        if _MODEL_PATH.exists():
            metadata = _load_metadata()
            if metadata.get("obs_shape") and metadata["obs_shape"] != list(env.observation_space.shape):
                logger.info("Observation space changed, retraining from scratch.")
                _MODEL_PATH.unlink(missing_ok=True)
            else:
                try:
                    model = PPO.load(_MODEL_PATH, env=env)
                    logger.info(
                        f"Modèle PPO chargé depuis le cache ({metadata.get('total_timesteps', '?')} timesteps cumulés)"
                    )
                    # Désactivé (Juin 2026) : l'entraînement continu (_FINE_TUNE_TIMESTEPS=500)
                    # à chaque appel sur les mêmes données provoquait un sur-apprentissage
                    # extrême (policy collapse) et figeait le modèle sur une seule action (BUY).
                    # model.learn(total_timesteps=_FINE_TUNE_TIMESTEPS)
                    # total_ts = metadata.get("total_timesteps", 0) + _FINE_TUNE_TIMESTEPS
                    # _save_metadata(total_ts, env.observation_space.shape)
                    # model.save(_MODEL_PATH)
                    # logger.info(f"Modèle PPO sauvegardé ({_FINE_TUNE_TIMESTEPS} timesteps ajoutés, total: {total_ts})")
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
            # Cap uncalibrated PPO overconfidence (see _CONFIDENCE_CAP).
            confidence = min(confidence, _CONFIDENCE_CAP)
        except Exception:
            confidence = 0.5

        logger.info(f"TensorTrade prédiction finale : {signal} (Confiance: {confidence:.2f})")

        return ModelResult(signal=signal, confidence=confidence, reasoning=f"PPO predicted action {action} with prob {confidence:.2f}")

    except Exception as e:
        logger.error(f"Erreur lors de l'exécution de TensorTrade: {e}")
        import traceback

        logger.debug(traceback.format_exc())
        return ModelResult("HOLD", 0.0, f"TensorTrade Error: {e}")
