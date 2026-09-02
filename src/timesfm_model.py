import logging
import os
import numpy as np
import pandas as pd
from typing import Dict, Any
from dotenv import load_dotenv
from src.enhanced_decision_engine import BaseModel, ModelResult

load_dotenv()

logger = logging.getLogger(__name__)

# Contexte passé à TimesFM 3.0 (jours de cotations). L'API 3.0 accepte jusqu'à
# 15360 et gère elle-même l'alignement par patch (padding) ; 2048 ~ 8 ans de daily.
TIMESFM_CONTEXT = 2048

# Tentative d'importation de l'API TimesFM 3.0
# (package PyPI `timesfm>=3.0.1` — plus de clone vendor ni de patch __init__.py)
try:
    from timesfm3 import TimesFM3Forecaster
    import torch
    torch.set_float32_matmul_precision('high')

    TIMESFM3_AVAILABLE = True
    logger.info("API TimesFM 3.0 (Torch) chargée avec succès.")
except ImportError:
    TIMESFM3_AVAILABLE = False
    logger.error("API TimesFM 3.0 non trouvée. Veuillez lancer 'uv sync' pour installer timesfm>=3.0.1.")


class TimesFMModel(BaseModel):
    """Wrapper pour le modèle TimesFM 3.0 de Google Research (google/timesfm-3.0-pytorch)"""

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

        self._try_init()

    def _try_init(self):
        """Charge le modèle TimesFM 3.0 (idempotent, sans lever d'exception).

        Ré-appelable : predict() retente l'init si elle avait échoué (le 1er
        téléchargement du checkpoint fait ~1,3 Go et peut échouer en réseau —
        l'ancien comportement « HOLD 0.0 pour toujours » laissait le modèle
        mort jusqu'au redémarrage du process).
        """
        if not TIMESFM3_AVAILABLE:
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

            logger.info("Initialisation de TimesFM 3.0 (google/timesfm-3.0-pytorch)...")

            # from_pretrained télécharge config.json + model.safetensors (~1,3 Go)
            # dans le cache HF au premier appel. Device auto : cuda si dispo, sinon cpu.
            self.model = TimesFM3Forecaster.from_pretrained("google/timesfm-3.0-pytorch")

            self.initialized = True
            logger.info("TimesFM 3.0 initialisé avec succès (device=%s).", getattr(self.model, "device", "auto"))

        except Exception as e:
            logger.warning(f"Erreur lors de l'initialisation de TimesFM 3.0: {e}")
            self.initialized = False
            self.model = None

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

    def predict(self, data: Dict[str, Any]) -> ModelResult:
        """Generate a trading signal from TimesFM 3.0 price forecast (médiane).

        Uses an ATR-adaptive threshold and position-aware filtering to avoid
        redundant BUY (when already LONG) or SELL (when FLAT) signals.
        """
        if (not self.initialized or self.model is None) and TIMESFM3_AVAILABLE:
            # Retry opportuniste (init initiale échouée, ex. download interrompu).
            self._try_init()
        if not self.initialized or self.model is None:
            return ModelResult("HOLD", 0.0, "Model not initialized.")

        try:
            df = data.get("df")
            horizon = data.get("horizon", 5)
            ticker = data.get("ticker", "default")

            if df is None or df.empty:
                return ModelResult("HOLD", 0.0, "No data provided.")
            prices = df["Close"].values
            if len(prices) > TIMESFM_CONTEXT:
                prices = prices[-TIMESFM_CONTEXT:]

            # API 3.0 : predict(context, horizon, ...) -> ForecastOutput(forecast, quantiles).
            # .forecast est la trajectoire médiane (quantile 0.5), .quantiles (horizon, 9).
            out = self.model.predict(
                prices,
                horizon=horizon,
                return_quantiles=True,
                make_positive=True,
            )
            predictions = np.asarray(out.forecast)
            quantiles = np.asarray(out.quantiles) if out.quantiles is not None else None

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

            # Position-aware de-churn: avoid re-buying when already long.
            # NOTE: the previous branch also forced SELL -> HOLD when flat, which
            # was meant to avoid shorting but in practice suppressed EVERY bearish
            # vote (the default position is FLAT). Over 610 prod predictions this
            # produced 0 SELL, removing 20% of the consensus weight from the
            # bearish side and contributing to the structural bullish bias.
            # A SELL signal here is now a directional vote only; whether to act
            # on it (close a long, or short) is decided downstream by the risk
            # manager. See ADR-002.
            if signal == "BUY" and self._get_position(ticker) == "LONG":
                signal = "HOLD"
                confidence *= 0.5

            if signal == "BUY":
                self._positions[ticker] = "LONG"
            elif signal == "SELL":
                self._positions[ticker] = "FLAT"

            analysis = (
                f"TimesFM 3.0 forecasts price move: {current_price:.2f} -> {last_pred:.2f} "
                f"({expected_return * 100:+.2f}%) over {horizon} days. "
                f"Adaptive threshold={threshold * 100:.2f}%, position={self._get_position(ticker)}"
            )

            logger.info(f"TimesFM 3.0 prediction: {signal} ({confidence:.2f})")

            metadata = {"predictions": predictions.tolist()}
            if quantiles is not None:
                metadata["quantiles"] = quantiles.tolist()

            return ModelResult(
                signal=signal,
                confidence=round(float(confidence), 2),
                reasoning=analysis,
                metadata=metadata,
            )

        except Exception as e:
            logger.error(f"Erreur prédiction TimesFM 3.0: {e}")
            return ModelResult("HOLD", 0.0, f"Error: {e}")


def get_timesfm_prediction(df: pd.DataFrame, ticker: str = "default") -> ModelResult:
    """Convenience wrapper: get a TimesFM prediction for *ticker* using the singleton model."""
    try:
        model = TimesFMModel.get_instance()
        return model.predict({"df": df, "ticker": ticker})
    except Exception as e:
        logger.error(f"TimesFM prediction failed: {e}")
        return ModelResult("HOLD", 0.0, f"Model error: {e}")
