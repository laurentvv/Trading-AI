import unittest
import pandas as pd
from pathlib import Path
from src.chart_generator import generate_chart_image

class TestChartGenerator(unittest.TestCase):
    def test_generate_chart_image_missing_columns(self):
        """Test that missing required indicator columns returns False."""
        # Create a DataFrame with a DatetimeIndex and at least 2 rows
        dates = pd.date_range('2023-01-01', periods=5, freq='D')
        df = pd.DataFrame({
            'Open': [100, 101, 102, 103, 104],
            'High': [105, 106, 107, 108, 109],
            'Low': [95, 96, 97, 98, 99],
            'Close': [102, 103, 104, 105, 106],
            'Volume': [1000, 1100, 1200, 1300, 1400],
            # Intentionally missing 'MA_50', 'MA_200', 'RSI', 'MACD_Histogram', 'MACD', 'MACD_Signal'
            'MA_50': [100] * 5,
            # 'MA_200' is missing
        }, index=dates)

        output_path = Path("dummy_output.png")

        # Should return False due to missing columns
        result = generate_chart_image(df, output_path)
        self.assertFalse(result)

    def test_generate_chart_image_not_enough_data(self):
        """Test that having less than 2 data points returns False."""
        dates = pd.date_range('2023-01-01', periods=1, freq='D')
        df = pd.DataFrame({
            'Open': [100],
            'High': [105],
            'Low': [95],
            'Close': [102],
            'Volume': [1000],
            'MA_50': [100],
            'MA_200': [100],
            'RSI': [50],
            'MACD_Histogram': [0],
            'MACD': [0],
            'MACD_Signal': [0],
        }, index=dates)

        output_path = Path("dummy_output.png")

        result = generate_chart_image(df, output_path)
        self.assertFalse(result)

if __name__ == '__main__':
    unittest.main()
