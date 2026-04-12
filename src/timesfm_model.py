import logging
import pandas as pd
from typing import Dict

logger = logging.getLogger(__name__)

# Tentative d'importation de l'API 2.5
# (Elle doit être installée via setup_timesfm.py qui patche __init__.py)
try:
    import timesfm
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

    def __init__(self):
        self.model = None
        self.initialized = False
        
        if not TIMESFM_2P5_AVAILABLE:
            return

        try:
            logger.info("Initialisation de TimesFM 2.5 (google/timesfm-2.5-200m-pytorch)...")
            
            # Initialisation selon l'API 2.5 (sans torch_compile pour Windows)
            self.model = TimesFM_2p5_200M_torch(torch_compile=False).from_pretrained(
                "google/timesfm-2.5-200m-pytorch"
            )
            
            # Configuration et compilation obligatoire pour l'API 2.5
            from timesfm.configs import ForecastConfig
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

    def predict(self, df: pd.DataFrame, horizon: int = 5) -> Dict:
        """Prédit les valeurs futures et génère une décision."""
        if not self.initialized or self.model is None:
            return {"signal": "HOLD", "confidence": 0.0, "analysis": "Model not initialized."}

        try:
            prices = df['Close'].values
            if len(prices) > 1024: prices = prices[-1024:]
            
            # API 2.5: point_forecast est un numpy array de shape (batch_size, horizon)
            point_forecast, _ = self.model.forecast(horizon=horizon, inputs=[prices])
            predictions = point_forecast[0]

            current_price = prices[-1]
            last_pred = predictions[-1]
            expected_return = (last_pred - current_price) / current_price if current_price != 0 else 0.0

            signal = "HOLD"
            # On scale la confiance : 1% de mouvement -> 0.5 de confiance, 2% -> 1.0
            confidence = min(1.0, abs(expected_return) * 50)

            threshold = 0.005  # Seuil de 0.5% (plus réaliste pour un ETF sur 5 jours)
            if expected_return > threshold:
                signal = "BUY"
            elif expected_return < -threshold:
                signal = "SELL"

            analysis = (f"TimesFM 2.5 forecasts price move: {current_price:.2f} -> {last_pred:.2f} "
                       f"({expected_return*100:+.2f}%) over {horizon} days.")

            logger.info(f"TimesFM 2.5 prediction: {signal} ({confidence:.2f})")

            return {
                "signal": signal,
                "confidence": round(float(confidence), 2),
                "analysis": analysis,
                "predictions": predictions.tolist()
            }

        except Exception as e:
            logger.error(f"Erreur prédiction TimesFM 2.5: {e}")
            return {"signal": "HOLD", "confidence": 0.0, "analysis": f"Error: {e}"}

def get_timesfm_prediction(df: pd.DataFrame) -> Dict:
    """
    Wrapper function to get predictions from the TimesFM model.
    Handles singleton initialization automatically.
    """
    try:
        model = TimesFMModel.get_instance()
        return model.predict(df)
    except Exception as e:
        logger.error(f"TimesFM prediction failed: {e}")
        return {"signal": "HOLD", "confidence": 0.0, "analysis": f"Model error: {e}"}
