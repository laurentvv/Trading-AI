import sqlite3
import logging
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

DB_PATH = Path("trading_history.db")



def init_db():
    """Initializes the SQLite database and creates tables if they don't exist."""
    conn = sqlite3.connect(DB_PATH, timeout=5.0)
    cursor = conn.cursor()

    cursor.execute("PRAGMA foreign_keys = ON")

    # Create transactions table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS transactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            ticker TEXT NOT NULL,
            type TEXT NOT NULL CHECK(type IN ('BUY', 'SELL')),
            quantity REAL NOT NULL,
            price REAL NOT NULL,
            cost REAL NOT NULL,
            signal_source TEXT,
            reason TEXT
        )
    """)

    # Create portfolio_history table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS portfolio_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            ticker TEXT NOT NULL,
            position REAL NOT NULL,
            cash REAL NOT NULL,
            total_value REAL NOT NULL,
            benchmark_value REAL NOT NULL,
            UNIQUE(date, ticker)
        )
    """)

    # Create model_signals table with updated constraints
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS model_signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            ticker TEXT NOT NULL,
            model_type TEXT NOT NULL CHECK(model_type IN ('classic', 'llm_text', 'llm_visual', 'sentiment', 'hybrid', 'oil_bench', 'vincent_ganne', 'timesfm', 'tensortrade', 'grebenkov')),
            signal TEXT NOT NULL CHECK(signal IN ('BUY', 'SELL', 'HOLD', 'STRONG_BUY', 'STRONG_SELL')),
            confidence REAL,
            details TEXT
        )
    """)

    conn.commit()
    conn.close()
    _migrate_model_signals_table()
    logger.info(f"Database initialized at {DB_PATH}")


def get_latest_portfolio_state(
    ticker: str = "QQQ",
) -> Optional[Tuple[float, float, float, float]]:
    """
    Retrieves the latest portfolio state for a given ticker.
    Returns a tuple: (position, cash, total_value, benchmark_value) or None if no record.
    """
    conn = sqlite3.connect(DB_PATH, timeout=5.0)
    query = """
        SELECT position, cash, total_value, benchmark_value
        FROM portfolio_history
        WHERE ticker = ?
        ORDER BY date DESC
        LIMIT 1
    """
    result = conn.execute(query, (ticker,)).fetchone()
    conn.close()
    return result  # This will be a tuple or None


def get_latest_transaction(
    ticker: str = "QQQ",
) -> Optional[Tuple[str, str, float, float, float]]:
    """
    Retrieves the latest transaction for a given ticker.
    Returns a tuple: (date, type, quantity, price, cost) or None if no record.
    """
    conn = sqlite3.connect(DB_PATH, timeout=5.0)
    query = """
        SELECT date, type, quantity, price, cost
        FROM transactions
        WHERE ticker = ?
        ORDER BY date DESC
        LIMIT 1
    """
    result = conn.execute(query, (ticker,)).fetchone()
    conn.close()
    return result  # This will be a tuple or None


def insert_transaction(
    date: str,
    ticker: str,
    type: str,
    quantity: float,
    price: float,
    cost: float,
    signal_source: str = "",
    reason: str = "",
):
    """Inserts a new transaction record."""
    conn = sqlite3.connect(DB_PATH, timeout=5.0)
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO transactions (date, ticker, type, quantity, price, cost, signal_source, reason)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (date, ticker, type, quantity, price, cost, signal_source, reason),
        )
        conn.commit()
        logger.info(f"Inserted transaction: {type} {quantity} of {ticker} on {date}")
    finally:
        conn.close()


def insert_portfolio_state(
    date: str,
    ticker: str,
    position: float,
    cash: float,
    total_value: float,
    benchmark_value: float,
):
    """Inserts or updates the portfolio state for a given date."""
    conn = sqlite3.connect(DB_PATH, timeout=5.0)
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT OR REPLACE INTO portfolio_history (date, ticker, position, cash, total_value, benchmark_value)
            VALUES (?, ?, ?, ?, ?, ?)
        """,
            (date, ticker, position, cash, total_value, benchmark_value),
        )
        conn.commit()
        logger.info(f"Updated portfolio state for {ticker} on {date}")
    finally:
        conn.close()


def insert_model_signal(
    date: str,
    ticker: str,
    model_type: str,
    signal: str,
    confidence: float,
    details: str = "",
):
    """Inserts a model signal."""
    conn = sqlite3.connect(DB_PATH, timeout=5.0)
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO model_signals (date, ticker, model_type, signal, confidence, details)
            VALUES (?, ?, ?, ?, ?, ?)
        """,
            (date, ticker, model_type, signal, confidence, details),
        )
        conn.commit()
        logger.info(f"Inserted {model_type} signal '{signal}' for {ticker} on {date}")
    finally:
        conn.close()



def insert_transactions_batch(transactions: list[tuple]):
    """
    Inserts multiple transactions in a single batch using executemany.
    Format of each tuple: (date, ticker, type, quantity, price, cost, signal_source, reason)
    """
    conn = sqlite3.connect(DB_PATH, timeout=5.0)
    try:
        cursor = conn.cursor()
        cursor.executemany(
            '''
            INSERT INTO transactions
            (date, ticker, type, quantity, price, cost, signal_source, reason)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''',
            transactions
        )
        conn.commit()
        logger.info(f"Inserted batch of {len(transactions)} transactions.")
    except Exception as e:
        logger.error(f"Error inserting transactions batch: {e}")
        conn.rollback()
    finally:
        conn.close()


def get_portfolio_history(ticker: str = "QQQ") -> pd.DataFrame:
    """Retrieves the full portfolio history for a given ticker as a DataFrame."""
    conn = sqlite3.connect(DB_PATH, timeout=5.0)
    query = "SELECT * FROM portfolio_history WHERE ticker = ? ORDER BY date"
    df = pd.read_sql_query(query, conn, params=(ticker,), parse_dates=["date"])
    conn.close()
    return df


def get_transactions_history(ticker: str = "QQQ") -> pd.DataFrame:
    """Retrieves the full transaction history for a given ticker as a DataFrame."""
    conn = sqlite3.connect(DB_PATH, timeout=5.0)
    query = "SELECT * FROM transactions WHERE ticker = ? ORDER BY date"
    df = pd.read_sql_query(query, conn, params=(ticker,), parse_dates=["date"])
    conn.close()
    return df


def _migrate_model_signals_table():
    conn = sqlite3.connect(DB_PATH, timeout=5.0)
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='model_signals'")
        row = cursor.fetchone()
        if row and "grebenkov" not in row[0]:
            logger.info("Migrating model_signals table to add new model types...")
            cursor.execute("DROP TABLE IF EXISTS model_signals_old")
            cursor.execute("ALTER TABLE model_signals RENAME TO model_signals_old")

            cursor.execute("""CREATE TABLE model_signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                ticker TEXT NOT NULL,
                model_type TEXT NOT NULL CHECK(model_type IN ('classic', 'llm_text', 'llm_visual', 'sentiment', 'hybrid', 'oil_bench', 'vincent_ganne', 'timesfm', 'tensortrade', 'grebenkov')),
                signal TEXT NOT NULL CHECK(signal IN ('BUY', 'SELL', 'HOLD', 'STRONG_BUY', 'STRONG_SELL')),
                confidence REAL,
                details TEXT
            )""")

            cursor.execute("""
                INSERT INTO model_signals (date, ticker, model_type, signal, confidence, details)
                SELECT date, ticker, model_type, signal, confidence, details FROM model_signals_old
            """)
            conn.commit()
            cursor.execute("DROP TABLE model_signals_old")
            conn.commit()
            logger.info("Migration complete.")
    except Exception as e:
        logger.warning(f"Migration check for model_signals: {e}")
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='model_signals_old'")
            if cursor.fetchone():
                logger.info("Attempting to recover model_signals from backup...")
                cursor.execute("DROP TABLE IF EXISTS model_signals")
                cursor.execute("ALTER TABLE model_signals_old RENAME TO model_signals")
                conn.commit()
                logger.info("Recovery successful.")
        except Exception as recovery_err:
            logger.error(f"Recovery also failed: {recovery_err}")
    finally:
        conn.close()
