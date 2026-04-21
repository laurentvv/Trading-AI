"""Test database fix for UNIQUE constraint on (date, ticker)."""

import sqlite3
import os
import tempfile


def test_unique_date_ticker_constraint():
    """Test that the UNIQUE constraint works on (date, ticker) pair."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    try:
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        c.execute("""CREATE TABLE portfolio_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            ticker TEXT NOT NULL,
            position REAL NOT NULL,
            cash REAL NOT NULL,
            total_value REAL NOT NULL,
            benchmark_value REAL NOT NULL,
            UNIQUE(date, ticker)
        )""")

        # Should succeed: different tickers, same date
        c.execute(
            "INSERT INTO portfolio_history (date, ticker, position, cash, total_value, benchmark_value) VALUES (?, ?, ?, ?, ?, ?)",
            ("2025-01-01", "AAPL", 10, 1000, 2000, 2000),
        )
        c.execute(
            "INSERT INTO portfolio_history (date, ticker, position, cash, total_value, benchmark_value) VALUES (?, ?, ?, ?, ?, ?)",
            ("2025-01-01", "GOOGL", 5, 500, 1000, 1000),
        )
        print("PASS: Can insert different tickers with same date")

        # Should fail: same date and ticker
        try:
            c.execute(
                "INSERT INTO portfolio_history (date, ticker, position, cash, total_value, benchmark_value) VALUES (?, ?, ?, ?, ?, ?)",
                ("2025-01-01", "AAPL", 20, 2000, 4000, 4000),
            )
            print("FAIL: Should have raised UNIQUE constraint error")
        except sqlite3.IntegrityError:
            print("PASS: UNIQUE constraint prevents duplicate (date, ticker)")

        conn.close()
    finally:
        if os.path.exists(db_path):
            os.remove(db_path)


if __name__ == "__main__":
    test_unique_date_ticker_constraint()
