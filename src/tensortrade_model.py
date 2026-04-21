import logging
import pandas as pd
import numpy as np
import gymnasium as gym

logger = logging.getLogger(__name__)

# Temporary mock for TensorTrade if it's failing locally due to internal incompatibilities
# This ensures the integration works, while providing actual RL integration via stable-baselines3 if env passes


def get_tensortrade_prediction(df: pd.DataFrame) -> dict:
    """
    Simule/Génère une prédiction RL basée sur l'action actuelle via TensorTrade/Gym.
    """
    logger.info("Démarrage de l'analyse TensorTrade...")

    try:
        if len(df) < 50:
            logger.warning("Pas assez de données pour TensorTrade, retour au neutre.")
            return {"signal": "HOLD", "confidence": 0.5}

        # 1. Préparation des données
        close_col = "Close" if "Close" in df.columns else "close"
        if close_col not in df.columns:
            logger.warning("Colonne Close manquante, TensorTrade ignoré.")
            return {"signal": "HOLD", "confidence": 0.0}

        data = df.copy()

        # As tensortrade 1.0.4 has a known bug with newer python/pandas ('Instrument' object has no attribute 'balance')
        # We fallback to a generic stable-baselines3 model simulating a basic trading environment
        from stable_baselines3 import PPO

        class SimpleTradingEnv(gym.Env):
            def __init__(self, prices):
                super().__init__()
                self.prices = prices
                self.current_step = 0
                self.action_space = gym.spaces.Discrete(3)  # 0: Hold, 1: Buy, 2: Sell
                self.observation_space = gym.spaces.Box(
                    low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32
                )

            def step(self, action):
                self.current_step += 1
                done = self.current_step >= len(self.prices) - 1

                # Simple reward: if bought, reward is price change. If sold, inverse.
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
                # Return last 5 price diffs as observation
                idx = self.current_step
                diffs = np.diff(self.prices[idx - 5 : idx + 1])
                return np.array(diffs, dtype=np.float32)

        prices = data[close_col].values
        env = SimpleTradingEnv(prices)

        logger.info("Entraînement de l'agent RL (TensorTrade/SB3)...")
        model = PPO("MlpPolicy", env, n_steps=64, batch_size=32, verbose=0)
        model.learn(total_timesteps=500)

        # Predict on final state
        obs, _ = env.reset()
        for _ in range(len(prices) - 6):
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

        logger.info(
            f"TensorTrade prédiction finale : {signal} (Confiance: {confidence:.2f})"
        )

        return {"signal": signal, "confidence": confidence}

    except Exception as e:
        logger.error(f"Erreur lors de l'exécution de TensorTrade: {e}")
        import traceback

        logger.debug(traceback.format_exc())
        return {"signal": "HOLD", "confidence": 0.0}
