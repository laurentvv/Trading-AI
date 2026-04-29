# Technical Context

## 1. Core Technologies
- **Language**: Python 3.12 (Pinned for TimesFM compatibility)
- **Dependency Management**: **uv** (Astral) - Replaces pip and requirements.txt.
- **AI / Machine Learning**:
    - **Scikit-learn**: Quantitative models (RandomForest, etc.) with `TimeSeriesSplit`.
    - **TimesFM**: Foundation model for time-series forecasting (Google Research).
    - **stable-baselines3 (PPO)**: Reinforcement Learning agent via custom Gymnasium environment (TensorTrade integration).
    - **Ollama**: Local serving of **Gemma 4 (e4b)** for text and visual chart analysis.
- **Data & Research Architecture**:
    - **yfinance**: Market data download.
    - **pandas-datareader**: Macroeconomic data (FRED).
    - **hyperliquid-python-sdk**: Decentralized sentiment data (Funding, Open Interest).
    - **Crawl4AI**: High-fidelity asynchronous web crawler for macro research.
    - **pyarrow (Parquet)**: Local multi-layer caching.
    - **SQLite**: Persistent storage for simulation transactions and portfolio states.

## 2. Development Environment
- The project is initialized and run via `uv`.
- Virtual environment is located in `.venv`.
- Entry point: `main.py`.

## 3. Technical Constraints & Assumptions
- **Ollama**: Must be running locally with `gemma4:e4b`.
- **Trading 212 API**: Specific quantity precision required per instrument (e.g., 2 decimals for CRUDl_EQ, 4 for SXRVd_EQ).
- **Operating System**: win32 (optimized for Windows terminal with **UTF-8 logging** to support emojis).
- **Network**: Required for initial API calls (Yahoo, FRED, Alpha Vantage, Hyperliquid).

## 4. Key Components
- `main.py`: CLI controller with `--simul`, `--t212` and `--ticker` flags.
- `src/enhanced_trading_example.py`: Main engine orchestrating all models.
- `src/t212_executor.py`: Real-world execution layer via Trading 212 API. Includes `get_t212_price()` for live ETF price retrieval.
- `src/data.py`: Market data layer with yfinance circuit breaker (separate trackers for `info` vs `download`), 10s timeouts, and cache auto-invalidation (stale > 2 days).
- `src/tensortrade_model.py`: Reinforcement Learning signal using PPO (stable-baselines3) in a Gymnasium trading environment.
- `src/database.py`: DAO layer for SQLite persistence.
- `src/read_simul.py`: Reporting tool for simulation performance.
- `refresh_cache.py`: CLI utility to force-refresh Parquet cache for all tickers (`^NDX`, `CL=F`, `SXRV.DE`, `CRUDP.PA`).

## 5. Monitoring & Diagnostic
- **CSV Journal (`trading_journal.csv`)**: (Test Phase) Detailed audit trail of AI reasoning and decisions per individual model.
- **SQLite DB**: Performance and simulation state history (generated locally, **not tracked in git** — `performance_monitor.db`, `trading_history.db`).
- **Dashboard**: `enhanced_performance_dashboard.png`.

## 6. Resilience Architecture
- **yfinance Circuit Breakers** (`src/data.py`): Two independent trackers — `_yf_info_tracker` for metadata (non-critical) and `_yf_download_tracker` for data downloads. After 3 consecutive failures, calls are skipped for 120s.
- **T212 Live Price Fallback** (`src/t212_executor.py`): `get_t212_price()` retrieves real-time EUR prices from Trading 212 positions when available (0.2s vs 10s+ yfinance timeout).
- **Price Hierarchy**: T212 live → MarketDataManager (yfinance) → yfinance history → cache parquet last close.
- **_yf_ticker_info() skipped when cache exists**: The `info` metadata call is bypassed when loading from parquet cache, saving ~30-50s per cycle.
- **Stale Cache Auto-Invalidation** (`src/data.py` lines 148-154): When loading from Parquet cache, if `last_date` is > 2 days old, the cache is bypassed and fresh data is downloaded. Prevents stale data decisions in PROD.
- **MA50 Fallback** (`src/data.py`): When MA200 is NaN (insufficient price history, e.g. Urea/UME=F), the system falls back to MA50 for the cross-asset moving average check. Ensures the Vincent Ganne model always has a valid reference.
