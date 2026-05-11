import logging
import pandas as pd
import torch
from pathlib import Path
import sys

logger = logging.getLogger(__name__)

# Add vendor/kronos to the path to import model.kronos
vendor_dir = Path(__file__).parent.parent / "vendor" / "kronos"
if str(vendor_dir) not in sys.path:
    sys.path.append(str(vendor_dir))

try:
    from model import KronosTokenizer, Kronos, KronosPredictor

    KRONOS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Could not import Kronos: {e}")
    KRONOS_AVAILABLE = False


class KronosModel:
    """Wrapper for Kronos trading model."""

    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        if not KRONOS_AVAILABLE:
            logger.error("Kronos not available, check imports.")
            return

        self.tokenizer = None
        self.model = None
        self.predictor = None
        self.max_context = 512
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

    def _load_model(self):
        if self.predictor is not None:
            return True

        logger.info("Loading Kronos-base model and tokenizer...")
        try:
            self.tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
            self.model = Kronos.from_pretrained("NeoQuasar/Kronos-base")

            # Initialize predictor
            self.predictor = KronosPredictor(
                self.model, self.tokenizer, device=self.device, max_context=self.max_context
            )
            logger.info("Kronos model loaded successfully.")
            return True
        except Exception as e:
            logger.error(f"Failed to load Kronos model: {e}")
            return False

    def predict(self, df: pd.DataFrame, pred_len: int = 24) -> dict:
        """
        Generate predictions using the Kronos model.
        Returns a dictionary with signal, confidence, analysis, and the forecasted dataframe.
        """
        if not KRONOS_AVAILABLE:
            return {"signal": "HOLD", "confidence": 0.0, "analysis": "Kronos not available", "forecast_df": None}

        if not self._load_model():
            return {"signal": "HOLD", "confidence": 0.0, "analysis": "Failed to load Kronos", "forecast_df": None}

        try:
            # Ensure index is datetime
            if not isinstance(df.index, pd.DatetimeIndex):
                # Try to use Date column if index is not datetime
                if "Date" in df.columns:
                    df = df.set_index("Date")
                    df.index = pd.to_datetime(df.index)
                else:
                    logger.warning(
                        "DataFrame index is not DatetimeIndex and 'Date' column not found. Kronos requires timestamps."
                    )
                    return {
                        "signal": "HOLD",
                        "confidence": 0.0,
                        "analysis": "Invalid DataFrame format for Kronos",
                        "forecast_df": None,
                    }

            # Kronos expects lowercase column names for OHLCV
            input_df = df.copy()
            input_df.columns = [col.lower() for col in input_df.columns]

            # Ensure we have required columns
            required_cols = ["open", "high", "low", "close"]
            if not all(col in input_df.columns for col in required_cols):
                logger.warning(f"Missing required columns for Kronos: {required_cols}")
                return {
                    "signal": "HOLD",
                    "confidence": 0.0,
                    "analysis": "Missing required OHLC columns",
                    "forecast_df": None,
                }

            # Truncate to max_context if necessary
            if len(input_df) > self.max_context:
                input_df = input_df.iloc[-self.max_context :]

            x_timestamp = pd.Series(input_df.index)

            # Generate future timestamps
            # For simplicity, assuming daily frequency, can be adjusted based on input data frequency
            freq = pd.infer_freq(input_df.index)
            if not freq:
                # Fallback to estimating frequency from last two points
                delta = input_df.index[-1] - input_df.index[-2]
                last_date = input_df.index[-1]
                y_timestamp = pd.Series([last_date + delta * i for i in range(1, pred_len + 1)])
            else:
                y_timestamp = pd.Series(pd.date_range(start=input_df.index[-1], periods=pred_len + 1, freq=freq)[1:])

            logger.info(f"Generating {pred_len} period forecast with Kronos...")

            # We must reset the index to match Kronos expectations
            input_df = input_df.reset_index(drop=True)

            pred_df = self.predictor.predict(
                df=input_df,
                x_timestamp=x_timestamp,
                y_timestamp=y_timestamp,
                pred_len=pred_len,
                T=1.0,
                top_p=0.9,
                sample_count=1,
            )

            # Put datetime index back for convenience
            pred_df.index = y_timestamp

            # Convert prediction to signal
            last_close = input_df["close"].iloc[-1]
            
            # Use the first 5 periods (1 week) for trading decision to avoid long-term noise
            eval_len = min(5, pred_len)
            short_term_forecast = pred_df["close"].iloc[:eval_len]
            
            # Calculate average forecasted price over the short term
            avg_future_close = short_term_forecast.mean()
            price_change_pct = ((avg_future_close - last_close) / last_close) * 100
            
            # Maximum drawdown and runup in the full forecast to contextualize
            max_future = pred_df["close"].max()
            min_future = pred_df["close"].min()
            max_up_pct = ((max_future - last_close) / last_close) * 100
            max_down_pct = ((min_future - last_close) / last_close) * 100

            # Logic: We want the short-term average to be solidly positive or negative
            threshold = 0.5

            if price_change_pct > threshold:
                signal = "BUY"
            elif price_change_pct < -threshold:
                signal = "SELL"
            else:
                signal = "HOLD"

            # Confidence is based on the magnitude of the predicted move, but we CAP it strictly
            # so hallucinations don't override the system. Max confidence 0.65
            base_confidence = abs(price_change_pct) / 2.0
            confidence = min(base_confidence, 0.65)

            analysis = f"Kronos avg {eval_len}d forecast: {price_change_pct:+.2f}%. (24d max: {max_up_pct:+.2f}%, min: {max_down_pct:+.2f}%)"

            logger.info(f"Kronos prediction: {signal} (Conf: {confidence:.2f}) - {analysis}")

            return {"signal": signal, "confidence": confidence, "analysis": analysis, "forecast_df": pred_df}

        except Exception as e:
            logger.error(f"Error during Kronos prediction: {e}")
            import traceback

            traceback.print_exc()
            return {"signal": "HOLD", "confidence": 0.0, "analysis": f"Error: {e}", "forecast_df": None}


def get_kronos_prediction(df: pd.DataFrame, pred_len: int = 10) -> dict:
    model = KronosModel.get_instance()
    return model.predict(df, pred_len=pred_len)
