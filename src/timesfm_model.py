import logging
import numpy as np
import pandas as pd
from typing import Dict, Tuple

logger = logging.getLogger(__name__)

# Essayons d'importer timesfm. Si non installé, on gérera avec un fallback ou on le loggera
try:
    import timesfm
    TIMESFM_AVAILABLE = True
except ImportError:
    TIMESFM_AVAILABLE = False
    logger.warning("Le module 'timesfm' n'est pas installé. Les prédictions TimesFM seront désactivées.")

class TimesFMModel:
    """Wrapper pour le modèle TimesFM 2.5 de Google Research"""

    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        self.model = None
        self.initialized = False
        if TIMESFM_AVAILABLE:
            try:
                logger.info("Initialisation de TimesFM 2.5...")
                # Initialize TimesFM 2.5
                self.model = timesfm.TimesFm(
                    hparams=timesfm.TimesFmHparams(
                        backend="torch",
                        per_core_batch_size=32,
                        horizon_len=5,
                        context_len=512,
                        num_layers=20,
                        model_dims=1280,
                        input_patch_len=32,
                    ),
                    checkpoint=timesfm.TimesFmCheckpoint(
                        huggingface_repo_id="google/timesfm-2.5-200m-pytorch"
                    ),
                )

                # Initialize state
                self.initialized = True
                logger.info("TimesFM initialisé avec succès.")
            except Exception as e:
                # Often fails due to huggingface cache paths missing the exact .ckpt file
                if isinstance(e, FileNotFoundError) or "[Errno 2]" in str(e):
                    logger.warning(f"TimesFM checkpoint introuvable, le modèle sera ignoré: {e}")
                else:
                    logger.warning(f"Erreur lors de l'initialisation de TimesFM: {e}")
                self.initialized = False

    def predict(self, df: pd.DataFrame, horizon: int = 5) -> Dict:
        """
        Prédit les valeurs futures et génère une décision.

        Args:
            df: DataFrame contenant la colonne 'Close'
            horizon: Nombre de jours à prédire

        Returns:
            Dict avec 'signal', 'confidence', et 'analysis'
        """
        if not self.initialized or self.model is None:
            return {
                "signal": "HOLD",
                "confidence": 0.0,
                "analysis": "TimesFM model not initialized or unavailable."
            }

        if 'Close' not in df.columns:
            logger.error("La colonne 'Close' est requise pour TimesFM.")
            return {
                "signal": "HOLD",
                "confidence": 0.0,
                "analysis": "Missing 'Close' column for prediction."
            }

        try:
            # Prepare inputs (needs numpy array)
            # Use last max_context data points if available
            prices = df['Close'].values
            if len(prices) > 512:
                prices = prices[-512:]

            # Need a list of 1D arrays for input
            inputs = [prices]

            # Predict
            point_forecast, _ = self.model.forecast(
                inputs=inputs,
                freq=[0], # 0 for daily/high freq usually
            )

            # point_forecast shape is (1, horizon)
            predictions = point_forecast[0]
            current_price = prices[-1]
            last_pred = predictions[-1]

            # Calculate expected return in %
            if current_price != 0:
                expected_return = (last_pred - current_price) / current_price
            else:
                logger.warning("Current price is 0, cannot calculate expected return.")
                expected_return = 0.0

            # Determine signal
            signal = "HOLD"
            confidence = min(1.0, abs(expected_return) * 10)  # Scale confidence based on expected return (e.g., 5% return -> 0.5 conf)

            # Add basic thresholds for signal
            threshold = 0.01  # 1% move expected
            if expected_return > threshold:
                signal = "BUY"
            elif expected_return < -threshold:
                signal = "SELL"

            analysis = f"TimesFM forecasts price to move from {current_price:.2f} to {last_pred:.2f} ({expected_return*100:.2f}% expected return) over next {horizon} days."

            logger.info(f"TimesFM prediction: {signal} with {confidence:.2f} confidence")

            return {
                "signal": signal,
                "confidence": round(float(confidence), 2),
                "analysis": analysis,
                "predictions": predictions.tolist() # Keep predictions if needed by UI
            }

        except Exception as e:
            logger.error(f"Erreur lors de la prédiction TimesFM: {e}")
            return {
                "signal": "HOLD",
                "confidence": 0.0,
                "analysis": f"Error during prediction: {e}"
            }

def get_timesfm_prediction(df: pd.DataFrame) -> Dict:
    """Helper function to get prediction from singleton model"""
    model = TimesFMModel.get_instance()
    return model.predict(df)
