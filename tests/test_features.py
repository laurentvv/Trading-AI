import unittest
import pandas as pd
from src.features import select_features, create_technical_indicators
import numpy as np

class TestFeatures(unittest.TestCase):
    def test_select_features_missing_critical(self):
        """Test that missing critical features raises a ValueError."""
        # Create a DataFrame missing 'RSI' which is a critical feature
        df = pd.DataFrame({
            'Returns': [0.1, 0.2],
            'MA_20': [100, 101],
            'MACD': [1.5, 1.6],
            'BB_Position': [0.5, 0.6],
            'Volume_Ratio': [1.1, 1.2],
            'Target': [1, 0]
        })

        with self.assertRaisesRegex(ValueError, "Critical features missing"):
            select_features(df)

    def test_select_features_success(self):
        """Test that having all critical features succeeds."""
        # Create a DataFrame with all critical features
        df = pd.DataFrame({
            'Returns': [0.1, 0.2],
            'MA_20': [100, 101],
            'RSI': [50, 55],
            'MACD': [1.5, 1.6],
            'BB_Position': [0.5, 0.6],
            'Volume_Ratio': [1.1, 1.2],
            'Target': [1, 0],
            'Other_Feature': [5, 6]  # Will be ignored if not in potential features
        })

        X, y, features_list = select_features(df)

        # Verify it successfully returns X, y, features_list
        self.assertIsInstance(X, pd.DataFrame)
        self.assertIsInstance(y, pd.Series)
        self.assertIsInstance(features_list, list)

        # Verify the returned features list matches the columns in X
        self.assertEqual(list(X.columns), features_list)

        # Verify all critical features are in X
        for col in ['Returns', 'MA_20', 'RSI', 'MACD', 'BB_Position', 'Volume_Ratio']:
            self.assertIn(col, X.columns)



    def test_create_technical_indicators_success(self):
        """Test successful calculation of technical indicators."""
        # Create mock OHLCV data with enough rows for 200-day MA
        dates = pd.date_range(start='2020-01-01', periods=250, freq='D')
        np.random.seed(42)

        # Generate some realistic-looking mock data
        close_prices = 100 + np.random.randn(250).cumsum()
        high_prices = close_prices + np.random.uniform(0, 2, 250)
        low_prices = close_prices - np.random.uniform(0, 2, 250)
        open_prices = close_prices + np.random.uniform(-1, 1, 250)
        volumes = np.random.randint(1000, 10000, 250)

        df = pd.DataFrame({
            'Open': open_prices,
            'High': high_prices,
            'Low': low_prices,
            'Close': close_prices,
            'Volume': volumes
        }, index=dates)

        # Calculate indicators
        result_df = create_technical_indicators(df)

        # Verify it successfully returns a DataFrame
        self.assertIsInstance(result_df, pd.DataFrame)

        # Verify that rows with NaN values are dropped (since max window is 200)
        # 250 rows total - 200 (for MA_200) = 50 rows remaining (actually 51 depending on shift)
        self.assertGreater(len(result_df), 0)
        self.assertLess(len(result_df), len(df))

        # Verify no NaN values in the resulting dataframe
        self.assertEqual(result_df.isna().sum().sum(), 0)

        # Verify presence of expected columns
        expected_columns = [
            'Returns', 'Log_Returns',
            'MA_5', 'MA_20', 'MA_50', 'MA_100', 'MA_200',
            'EMA_12', 'EMA_26', 'EMA_50',
            'RSI', 'RSI_MA',
            'MACD', 'MACD_Signal', 'MACD_Histogram',
            'BB_Upper', 'BB_Lower', 'BB_Middle', 'BB_Width', 'BB_Position',
            'Stoch_K', 'Stoch_D',
            'Volume_MA', 'Volume_Ratio',
            'Volatility', 'ATR', 'Support', 'Resistance'
        ]

        for col in expected_columns:
            self.assertIn(col, result_df.columns)

    def test_create_technical_indicators_missing_columns(self):
        """Test that missing required columns raises an error."""
        df = pd.DataFrame({
            'Close': [100, 101, 102],
            # Missing Open, High, Low, Volume
        })

        with self.assertRaises(KeyError):
            create_technical_indicators(df)

    def test_create_technical_indicators_insufficient_data(self):
        """Test with insufficient data for rolling windows."""
        # Only 50 rows, not enough for MA_200
        dates = pd.date_range(start='2020-01-01', periods=50, freq='D')
        df = pd.DataFrame({
            'Open': [100] * 50,
            'High': [105] * 50,
            'Low': [95] * 50,
            'Close': [100] * 50,
            'Volume': [1000] * 50
        }, index=dates)

        result_df = create_technical_indicators(df)

        # Since dropna() is called and MA_200 requires 200 rows,
        # the result should be an empty DataFrame
        self.assertTrue(result_df.empty)

if __name__ == '__main__':
    unittest.main()
