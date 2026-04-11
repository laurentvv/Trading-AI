import unittest
import pandas as pd
from src.features import select_features

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

if __name__ == '__main__':
    unittest.main()
