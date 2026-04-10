# Technical Context

## 1. Core Technologies
- **Language**: Python 3.12 (Pinned for TimesFM compatibility)
- **Dependency Management**: **uv** (Astral) - Replaces pip and requirements.txt.
- **AI / Machine Learning**:
    - **Scikit-learn**: Quantitative models (RandomForest, etc.) with `TimeSeriesSplit`.
    - **TimesFM**: Foundation model for time-series forecasting (Google Research).
    - **Ollama**: Local serving of **Gemma 4 (e4b)** for text and visual chart analysis.
- **Data Architecture**:
    - **yfinance**: Market data download.
    - **pandas-datareader**: Macroeconomic data (FRED).
    - **pyarrow (Parquet)**: Local multi-layer caching.
    - **SQLite**: Persistent storage for simulation transactions and portfolio states.

## 2. Development Environment
- The project is initialized and run via `uv`.
- Virtual environment is located in `.venv`.
- Entry point: `main.py`.

## 3. Technical Constraints & Assumptions
- **Ollama**: Must be running locally with `gemma4:e4b`.
- **Operating System**: win32 (optimized for Windows terminal with CP1252/UTF-8 handling).
- **Network**: Required for initial API calls (Yahoo, FRED, Alpha Vantage).

## 4. Key Components
- `main.py`: CLI controller with `--simul` and `--ticker` flags.
- `src/enhanced_trading_example.py`: Main engine orchestrating all models.
- `src/database.py`: DAO layer for SQLite persistence.
- `src/read_simul.py`: Reporting tool for simulation performance.

## 5. Monitoring & Diagnostic
- **CSV Journal (`trading_journal.csv`)**: (Test Phase) Detailed audit trail of AI reasoning and decisions.
- **SQLite DB**: Performance and simulation state.
- **Dashboard**: `enhanced_performance_dashboard.png`.
