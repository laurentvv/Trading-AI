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
- **Trading 212 API**: Specific quantity precision required per instrument (e.g., 2 decimals for CRUDl_EQ, 4 for SXRVd_EQ). API response structure may omit `averagePrice` — code uses fallback calculation (`currentValue / quantity`).
- **Operating System**: win32 (optimized for Windows terminal with **UTF-8 logging** to support emojis).
- **Network**: Required for initial API calls (Yahoo, FRED, Alpha Vantage, Hyperliquid).

## 4. Key Components
- `main.py`: CLI controller with `--simul`, `--t212` and `--ticker` flags.
- `src/enhanced_trading_example.py`: Main engine orchestrating all models.
- `src/t212_executor.py`: Real-world execution layer via Trading 212 API. Includes `get_t212_price()` for live ETF price retrieval.
- `src/data.py`: Market data layer with yfinance circuit breaker (separate trackers for `info` vs `download`), 10s timeouts, and cache auto-invalidation (stale > **1 day**).
- `src/adaptive_weight_manager.py`: Manages dynamic model weights based on `model_performance.db`. Includes `update_outcomes_for_date()` for batch-updating prediction outcomes after trade closes, and on-the-fly weight normalization.
- `src/tensortrade_model.py`: Reinforcement Learning signal using PPO (stable-baselines3) in a Gymnasium trading environment.
- `src/lean_bridge.py`: *(removed 2026-05-05)* Replaced by `backtest_prod.py`.
- `src/lean_validator.py`: *(removed 2026-05-05)* Replaced by `backtest_prod.py`.
- `backtest_prod.py`: Standalone production backtest engine. Reads `logs_prod/trading_journal.csv`, loads real prices from `data_cache/` parquet files, simulates trades with T212 fees (0.1%), and compares signal strategy vs buy-and-hold baseline. Reports Sharpe, MaxDD, Win Rate, Alpha per ticker.
- `run_lean_backtest.py`: *(removed 2026-05-05)* Replaced by `backtest_prod.py`.
- `TradingAI-Lean/`: *(removed 2026-05-05)* Replaced by `backtest_prod.py`.
- `src/database.py`: DAO layer for SQLite persistence.
- `src/read_simul.py`: Reporting tool for simulation performance.
- `refresh_cache.py`: CLI utility to force-refresh Parquet cache for all tickers (`^NDX`, `CL=F`, `SXRV.DE`, `CRUDP.PA`).
- `backtest_prod.py`: Standalone production backtest engine. Replays actual prod signals against real parquet prices with T212 fees. Compares vs buy-and-hold baseline.

## 5. Monitoring & Diagnostic
- **CSV Journal (`trading_journal.csv`)**: (Test Phase) Detailed audit trail of AI reasoning and decisions per individual model.
- **SQLite DB**: Performance and simulation state history (generated locally, **not tracked in git** — `performance_monitor.db`, `trading_history.db`).
- **Dashboard**: `enhanced_performance_dashboard.png`.
- **Diagnostic Scripts (`tests/`)**:
  - `check_cache.py`: Lists Parquet cache files with modification dates and sizes.
  - `check_db.py`: Inspects SQLite tables, columns, and latest rows for `trading_history.db` and `performance_monitor.db`.
  - `check_live.py`: Fetches live prices via yfinance for all traded tickers.

## 6. Resilience Architecture
- **yfinance Circuit Breakers** (`src/data.py`): Two independent trackers — `_yf_info_tracker` for metadata (non-critical) and `_yf_download_tracker` for data downloads. After 3 consecutive failures, calls are skipped for 120s.
- **T212 Live Price Fallback** (`src/t212_executor.py`): `get_t212_price()` retrieves real-time EUR prices from Trading 212 positions when available (0.2s vs 10s+ yfinance timeout).
- **Price Hierarchy**: T212 live → MarketDataManager (yfinance) → yfinance history → cache parquet last close.
- **_yf_ticker_info() skipped when cache exists**: The `info` metadata call is bypassed when loading from parquet cache, saving ~30-50s per cycle.
- **Stale Cache Auto-Invalidation** (`src/data.py` lines 148-154): When loading from Parquet cache, if `last_date` is > **1 day** old, the cache is bypassed and fresh data is downloaded. Cache age is logged in fractional days for precise diagnostics. Prevents stale data decisions in PROD.
- **MA50 Fallback** (`src/data.py`): When MA200 is NaN (insufficient price history, e.g. Urea/UME=F), the system falls back to MA50 for the cross-asset moving average check. Ensures the Vincent Ganne model always has a valid reference.
- **Feedback Loop** (`src/t212_executor.py` + `src/adaptive_weight_manager.py`): After a confirmed T212 SELL, `update_outcomes_for_date()` batch-updates `model_performance_history` with actual trade results using a single SQLite connection. Only updates models that had predictions on the entry date (`actual_outcome IS NULL`). This feeds the AdaptiveWeightManager for dynamic weight adjustment.
- **Grokipedia Log Filter** (`src/web_researcher.py`): A `_GrokipediaFilter` logging filter suppresses non-blocking crawl4ai "grokipedia" warnings from polluting production logs.
