import pandas as pd
import torch
from model.kronos import Kronos, KronosTokenizer, KronosPredictor


class KronosForecaster:
    """
    A wrapper class for the Kronos model to provide a simple interface for forecasting.
    """

    def __init__(self, device="cuda:0" if torch.cuda.is_available() else "cpu"):
        """
        Initializes the KronosForecaster by loading the pre-trained model and tokenizer.
        """
        print("Initializing KronosForecaster...")
        self.device = device

        try:
            # Load the pre-trained tokenizer and model from Hugging Face Hub
            self.tokenizer = KronosTokenizer.from_pretrained(
                "NeoQuasar/Kronos-Tokenizer-base"
            )
            self.model = Kronos.from_pretrained("NeoQuasar/Kronos-small")

            # Initialize the predictor
            self.predictor = KronosPredictor(
                self.model, self.tokenizer, device=self.device
            )
            print("Kronos model and tokenizer loaded successfully.")

        except Exception as e:
            print(f"Error loading Kronos model: {e}")
            self.model = None
            self.tokenizer = None
            self.predictor = None

    def predict(self, df_history: pd.DataFrame, pred_len: int = 1) -> dict:
        """
        Generates a trading signal and confidence based on the Kronos forecast.

        Args:
            df_history (pd.DataFrame): A DataFrame with historical k-line data.
                                       It must contain ['open', 'high', 'low', 'close'] columns
                                       and a 'timestamps' column.
            pred_len (int): The number of future periods to predict.

        Returns:
            dict: A dictionary with 'signal' and 'confidence'.
        """
        if self.predictor is None:
            return {
                "signal": "HOLD",
                "confidence": 0.0,
                "analysis": "Kronos model not loaded.",
            }

        try:
            # Prepare the input data for the predictor
            df_history["timestamps"] = pd.to_datetime(df_history["timestamps"])

            # Ensure volume is present, if not, add a column of zeros
            if "volume" not in df_history.columns:
                df_history["volume"] = 0

            x_df = df_history[["open", "high", "low", "close", "volume"]]
            x_timestamp = df_history["timestamps"]

            # Create future timestamps for prediction
            last_timestamp = x_timestamp.iloc[-1]
            freq = pd.infer_freq(x_timestamp)
            if freq is None:
                freq = "D"  # Default to daily frequency if it cannot be inferred

            y_timestamp = pd.to_datetime(
                pd.date_range(
                    start=last_timestamp + pd.Timedelta(freq),
                    periods=pred_len,
                    freq=freq,
                )
            )

            # Generate the forecast
            pred_df = self.predictor.predict(
                df=x_df,
                x_timestamp=x_timestamp,
                y_timestamp=y_timestamp,
                pred_len=pred_len,
                verbose=False,  # Keep the log clean
            )

            # Determine the trading signal based on the forecast
            current_price = x_df["close"].iloc[-1]
            predicted_price = pred_df["close"].iloc[-1]

            signal = "HOLD"
            confidence = 0.5

            price_change_ratio = (predicted_price - current_price) / current_price

            if price_change_ratio > 0.002:  # Stricter threshold for BUY
                signal = "BUY"
                confidence = min(
                    0.95, 0.5 + (price_change_ratio * 10)
                )  # Scale confidence
            elif price_change_ratio < -0.002:  # Stricter threshold for SELL
                signal = "SELL"
                confidence = min(
                    0.95, 0.5 + (abs(price_change_ratio) * 10)
                )  # Scale confidence

            analysis = f"Kronos predicts a price change of {price_change_ratio:.2%}. Current: {current_price:.2f}, Predicted: {predicted_price:.2f}."

            return {"signal": signal, "confidence": confidence, "analysis": analysis}

        except Exception as e:
            print(f"Error during Kronos prediction: {e}")
            return {
                "signal": "HOLD",
                "confidence": 0.0,
                "analysis": f"An error occurred: {e}",
            }
