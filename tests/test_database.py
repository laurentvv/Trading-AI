import unittest
import sqlite3
import tempfile
import os
from pathlib import Path
from unittest.mock import patch
import pandas as pd

import src.database as db

class TestDatabase(unittest.TestCase):
    def setUp(self):
        # Create a temporary file for the database
        self.temp_db_fd, self.temp_db_path = tempfile.mkstemp(suffix=".db")

        # Patch the DB_PATH in the database module
        self.db_patcher = patch('src.database.DB_PATH', Path(self.temp_db_path))
        self.mock_db_path = self.db_patcher.start()

        # Initialize the database (create tables)
        db.init_db()

    def tearDown(self):
        # Stop patching
        self.db_patcher.stop()

        # Close the file descriptor and remove the temporary file
        os.close(self.temp_db_fd)
        if os.path.exists(self.temp_db_path):
            os.remove(self.temp_db_path)

    def test_init_db(self):
        # Verify that tables are created
        conn = sqlite3.connect(self.temp_db_path, timeout=5.0)
        cursor = conn.cursor()

        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = {row[0] for row in cursor.fetchall()}
        conn.close()

        self.assertIn('transactions', tables)
        self.assertIn('portfolio_history', tables)
        self.assertIn('model_signals', tables)

    def test_insert_and_get_transaction(self):
        db.insert_transaction(
            date='2024-01-01',
            ticker='AAPL',
            type='BUY',
            quantity=10.0,
            price=150.0,
            cost=1500.0,
            signal_source='test',
            reason='test reason'
        )

        result = db.get_latest_transaction(ticker='AAPL')
        self.assertIsNotNone(result)

        # Expected: (date, type, quantity, price, cost)
        self.assertEqual(result[0], '2024-01-01')
        self.assertEqual(result[1], 'BUY')
        self.assertEqual(result[2], 10.0)
        self.assertEqual(result[3], 150.0)
        self.assertEqual(result[4], 1500.0)

    def test_transaction_constraints(self):
        # Should fail with an invalid transaction type (CHECK constraint)
        with self.assertRaises(sqlite3.IntegrityError):
            db.insert_transaction(
                date='2024-01-01',
                ticker='AAPL',
                type='INVALID_TYPE',
                quantity=10.0,
                price=150.0,
                cost=1500.0
            )

    def test_insert_and_get_portfolio_state(self):
        db.insert_portfolio_state(
            date='2024-01-01',
            ticker='AAPL',
            position=10.0,
            cash=1000.0,
            total_value=2500.0,
            benchmark_value=2000.0
        )

        result = db.get_latest_portfolio_state(ticker='AAPL')
        self.assertIsNotNone(result)

        # Expected: (position, cash, total_value, benchmark_value)
        self.assertEqual(result[0], 10.0)
        self.assertEqual(result[1], 1000.0)
        self.assertEqual(result[2], 2500.0)
        self.assertEqual(result[3], 2000.0)

        # Test INSERT OR REPLACE behavior
        db.insert_portfolio_state(
            date='2024-01-01',
            ticker='AAPL',
            position=20.0,
            cash=500.0,
            total_value=3500.0,
            benchmark_value=2100.0
        )

        result_updated = db.get_latest_portfolio_state(ticker='AAPL')
        self.assertEqual(result_updated[0], 20.0)

        # Ensure only one record exists for the given date and ticker (due to UNIQUE constraint on date)
        conn = sqlite3.connect(self.temp_db_path, timeout=5.0)
        count = conn.execute("SELECT COUNT(*) FROM portfolio_history WHERE date='2024-01-01' AND ticker='AAPL'").fetchone()[0]
        conn.close()
        self.assertEqual(count, 1)

    def test_insert_model_signal_and_constraints(self):
        db.insert_model_signal(
            date='2024-01-01',
            ticker='AAPL',
            model_type='classic',
            signal='BUY',
            confidence=0.85,
            details='{"reason": "good"}'
        )

        conn = sqlite3.connect(self.temp_db_path, timeout=5.0)
        result = conn.execute("SELECT model_type, signal, confidence FROM model_signals WHERE ticker='AAPL'").fetchone()
        conn.close()

        self.assertIsNotNone(result)
        self.assertEqual(result[0], 'classic')
        self.assertEqual(result[1], 'BUY')
        self.assertEqual(result[2], 0.85)

        # Test CHECK constraint for model_type
        with self.assertRaises(sqlite3.IntegrityError):
            db.insert_model_signal(
                date='2024-01-01',
                ticker='AAPL',
                model_type='invalid_model',
                signal='BUY',
                confidence=0.85
            )

        # Test CHECK constraint for signal
        with self.assertRaises(sqlite3.IntegrityError):
            db.insert_model_signal(
                date='2024-01-01',
                ticker='AAPL',
                model_type='classic',
                signal='INVALID_SIGNAL',
                confidence=0.85
            )

    def test_get_history_dataframes(self):
        db.insert_transaction(date='2024-01-01', ticker='AAPL', type='BUY', quantity=10, price=150, cost=1500)
        db.insert_transaction(date='2024-01-02', ticker='AAPL', type='SELL', quantity=5, price=160, cost=800)

        db.insert_portfolio_state(date='2024-01-01', ticker='AAPL', position=10, cash=1000, total_value=2500, benchmark_value=2500)
        db.insert_portfolio_state(date='2024-01-02', ticker='AAPL', position=5, cash=1800, total_value=2600, benchmark_value=2550)

        df_transactions = db.get_transactions_history(ticker='AAPL')
        self.assertIsInstance(df_transactions, pd.DataFrame)
        self.assertEqual(len(df_transactions), 2)
        self.assertEqual(df_transactions.iloc[0]['type'], 'BUY')
        self.assertEqual(df_transactions.iloc[1]['type'], 'SELL')

        df_portfolio = db.get_portfolio_history(ticker='AAPL')
        self.assertIsInstance(df_portfolio, pd.DataFrame)
        self.assertEqual(len(df_portfolio), 2)
        self.assertEqual(df_portfolio.iloc[0]['position'], 10.0)
        self.assertEqual(df_portfolio.iloc[1]['position'], 5.0)

if __name__ == '__main__':
    unittest.main()
