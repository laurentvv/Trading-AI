import logging
import os
import numpy as np
import pandas as pd
from typing import Dict
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# Tentative d'importation de l'API 2.5
# (Elle doit être installée via setup_timesfm.py qui patche __init__.py)
try:
    # import timesfm
    from timesfm.timesfm_2p5.timesfm_2p5_torch import TimesFM_2p5_200M_torch
    from timesfm.configs import ForecastConfig

    TIMESFM_2P5_AVAILABLE = True
    logger.info("API TimesFM 2.5 (Torch) chargée avec succès.")
except ImportError:
    TIMESFM_2P5_AVAILABLE = False
    logger.error("API TimesFM 2.5 non trouvée. Veuillez lancer 'python setup_timesfm.py' pour l'installer.")


class TimesFMModel:
    """Wrapper pour le modèle TimesFM 2.5 de Google Research"""

    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self, vol_multiplier: float = 0.5):
        self.model = None
        self.initialized = False
        self.vol_multiplier = vol_multiplier
        self._positions: Dict[str, str] = {}

        if not TIMESFM_2P5_AVAILABLE:
            return

        try:
            hf_token = os.getenv("HF_TOKEN")
            if hf_token:
                try:
                    from huggingface_hub import login

                    login(token=hf_token, add_to_git_credential=False)
                    logger.info("HF_TOKEN detecte — authentification HuggingFace effectuee.")
                except Exception as hf_err:
                    logger.warning(f"Authentification HF echouee (non bloquant): {hf_err}")
            else:
                logger.info("HF_TOKEN non defini — telechargements sans authentification.")

            logger.info("Initialisation de TimesFM 2.5 (google/timesfm-2.5-200m-pytorch)...")

            # Initialisation selon l'API 2.5 (sans torch_compile pour Windows)
            self.model = TimesFM_2p5_200M_torch(torch_compile=False).from_pretrained("google/timesfm-2.5-200m-pytorch")

            self.model.compile(
                ForecastConfig(
                    max_context=1024,
                    max_horizon=256,
                    normalize_inputs=True,
                    use_continuous_quantile_head=True,
                    force_flip_invariance=True,
                    infer_is_positive=True,
                    fix_quantile_crossing=True,
                )
            )
            logger.info("TimesFM 2.5 initialisé et compilé avec succès.")
            self.initialized = True

        except Exception as e:
            logger.warning(f"Erreur lors de l'initialisation de TimesFM 2.5: {e}")
            self.initialized = False

    def update_position(self, position: str, ticker: str = "default"):
        """Manually set the current position for *ticker* to LONG or FLAT."""
        if position.upper() in ("LONG", "FLAT"):
            self._positions[ticker] = position.upper()

    def reset(self, ticker: str = None):
        """Clear position state. If *ticker* is given, reset only that ticker; otherwise clear all."""
        if ticker:
            self._positions.pop(ticker, None)
        else:
            self._positions.clear()

    def _get_position(self, ticker: str) -> str:
        return self._positions.get(ticker, "FLAT")

    def _adaptive_threshold(self, prices: np.ndarray) -> float:
        if len(prices) < 20:
            return 0.005
        returns = np.diff(prices[-20:]) / np.maximum(prices[-20:-1], 1e-8)
        realised_vol = float(np.std(returns))
        return max(0.005, realised_vol * self.vol_multiplier)

    def predict(self, df: pd.DataFrame, horizon: int = 5, ticker: str = "default") -> Dict:
        """Generate a trading signal from TimesFM 2.5 price forecast.

        Uses an ATR-adaptive threshold and position-aware filtering to avoid
        redundant BUY (when already LONG) or SELL (when FLAT) signals.
        """
        if not self.initialized or self.model is None:
            return {
                "signal": "HOLD",
                "confidence": 0.0,
                "analysis": "Model not initialized.",
            }

        try:
            prices = df["Close"].values
            if len(prices) > 1024:
                prices = prices[-1024:]

            point_forecast, _ = self.model.forecast(horizon=horizon, inputs=[prices])
            predictions = point_forecast[0]

            current_price = prices[-1]
            last_pred = predictions[-1]
            expected_return = (last_pred - current_price) / current_price if current_price != 0 else 0.0

            signal = "HOLD"
            confidence = min(1.0, abs(expected_return) * 50)

            threshold = self._adaptive_threshold(prices)
            if expected_return > threshold:
                signal = "BUY"
            elif expected_return < -threshold:
                signal = "SELL"

            if signal == "BUY" and self._get_position(ticker) == "LONG":
                signal = "HOLD"
                confidence *= 0.5
            elif signal == "SELL" and self._get_position(ticker) == "FLAT":
                signal = "HOLD"
                confidence *= 0.5

            if signal == "BUY":
                self._positions[ticker] = "LONG"
            elif signal == "SELL":
                self._positions[ticker] = "FLAT"

            analysis = (
                f"TimesFM 2.5 forecasts price move: {current_price:.2f} -> {last_pred:.2f} "
                f"({expected_return * 100:+.2f}%) over {horizon} days. "
                f"Adaptive threshold={threshold * 100:.2f}%, position={self._get_position(ticker)}"
            )

            logger.info(f"TimesFM 2.5 prediction: {signal} ({confidence:.2f})")

            return {
                "signal": signal,
                "confidence": round(float(confidence), 2),
                "analysis": analysis,
                "predictions": predictions.tolist(),
            }

        except Exception as e:
            logger.error(f"Erreur prédiction TimesFM 2.5: {e}")
            return {"signal": "HOLD", "confidence": 0.0, "analysis": f"Error: {e}"}


def get_timesfm_prediction(df: pd.DataFrame, ticker: str = "default") -> Dict:
    """Convenience wrapper: get a TimesFM prediction for *ticker* using the singleton model."""
    try:
        model = TimesFMModel.get_instance()
        return model.predict(df, ticker=ticker)
    except Exception as e:
        logger.error(f"TimesFM prediction failed: {e}")
        return {"signal": "HOLD", "confidence": 0.0, "analysis": f"Model error: {e}"}
