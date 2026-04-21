import pandas as pd
import mplfinance as mpf
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def generate_chart_image(
    data_with_indicators: pd.DataFrame,
    output_path: Path,
    title: str = "Financial Chart",
) -> bool:
    """
    Generates and saves a financial chart image using pre-calculated indicators.

    Args:
        data_with_indicators (pd.DataFrame): DataFrame containing historical market data
                                             and pre-calculated indicators.
        output_path (Path): The path where the generated chart image will be saved.
        title (str): The title for the chart.

    Returns:
        bool: True if the chart was generated successfully, False otherwise.
    """
    try:
        # Select last 6 months of data using modern pandas slicing
        end_date = data_with_indicators.index[-1]
        start_date = end_date - pd.DateOffset(months=6)
        six_months_data = data_with_indicators.loc[start_date:]

        if len(six_months_data) < 2:
            logger.warning(
                "Not enough data to plot a chart (less than 2 data points in the last 6 months)."
            )
            return False

        # Define the additional plots using the pre-calculated indicator columns
        # Ensure the columns exist before trying to plot them
        required_indicators = [
            "MA_50",
            "MA_200",
            "RSI",
            "MACD_Histogram",
            "MACD",
            "MACD_Signal",
        ]
        if not all(col in six_months_data.columns for col in required_indicators):
            logger.error(
                f"Data must contain the following indicator columns: {required_indicators}"
            )
            return False

        addplots = [
            mpf.make_addplot(six_months_data["MA_50"], color="blue", width=0.7),
            mpf.make_addplot(six_months_data["MA_200"], color="orange", width=0.7),
            mpf.make_addplot(
                six_months_data["RSI"], panel=2, color="purple", ylabel="RSI"
            ),
            mpf.make_addplot(
                six_months_data["MACD_Histogram"],
                type="bar",
                panel=3,
                color="gray",
                ylabel="MACD",
            ),
            mpf.make_addplot(six_months_data["MACD"], panel=3, color="fuchsia"),
            mpf.make_addplot(six_months_data["MACD_Signal"], panel=3, color="cyan"),
        ]

        logger.info(f"Generating chart image and saving to {output_path}...")

        # Generate and save the plot
        mpf.plot(
            six_months_data,
            type="candle",
            style="yahoo",
            title=title,
            ylabel="Price ($)",
            volume=True,
            addplot=addplots,
            panel_ratios=(6, 3, 3, 2),  # Ratios for price, volume, rsi, macd
            figscale=1.5,
            savefig=dict(fname=str(output_path), dpi=100, pad_inches=0.25),
        )

        logger.info("Chart generated successfully.")
        return True

    except Exception as e:
        logger.error(f"Failed to generate chart image: {e}")
        return False
