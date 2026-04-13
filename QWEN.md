# Trading-AI — QWEN Context

## Project Overview

**Trading-AI** is a hybrid AI-powered trading decision support system for trading ETFs on NASDAQ. It combines six distinct AI models (tri-modal approach) to produce robust, nuanced trading signals:

1. **Classic Quantitative Model** — Ensemble of RandomForest/GradientBoosting/LogisticRegression trained on technical indicators and macroeconomic data
2. **TimesFM 2.5 (Google Research)** — Foundation model for time series forecasting
3. **Textual LLM (Gemma 4:e4b via Ollama)** — Contextual analysis of raw data and real-time news via AlphaEar skill
4. **Visual LLM (Gemma 4:e4b)** — Direct analysis of technical chart patterns
5. **Sentiment Analysis** — Hybrid analysis combining Alpha Vantage and AlphaEar social trends
6. **Decentralized Sentiment (Hyperliquid)** — Real-time blockchain data (Funding Rate, Open Interest) for Oil/WTI contrarian signals

### Key Architectural Decisions

- **Dual-Ticker Strategy**: Models analyze reference indices (`^NDX`, `CL=F`) for cleaner signals; actual orders execute on corresponding ETFs (`SXRV.DE`, `CRUDP.PA`) via Trading 212
- **"Cognitive Prudence" Philosophy**: Requires strong consensus across models before taking action. Cognitive models weight 75% vs 25% for the classic model. A 40% confidence threshold is required for BUY/SELL decisions; below that, signal degrades to HOLD
- **Risk-Adjusted Execution**: All signals pass through `AdvancedRiskManager` before execution. High volatility can override signals to protect capital

## Tech Stack

| Category | Technologies |
|----------|-------------|
| Language | Python 3.12+ |
| Package Manager | **uv** (Astral) |
| ML/Data | pandas, numpy, scikit-learn, yfinance, shap |
| Deep Learning | PyTorch, TimesFM 2.5, JAX |
| LLM | Ollama (Gemma 4:e4b) |
| Visualization | matplotlib, seaborn, mplfinance |
| API Integration | Trading 212 API, Alpha Vantage, AlphaEar (skill), **Hyperliquid SDK** |
| Utilities | python-dotenv, schedule, rich, tqdm |

## Project Structure

```
Trading-AI/
├── main.py                          # Unified entry point (--simul, --t212, --ticker)
├── schedule.py                      # Automated scheduler (Mon-Fri 8:30-18:00, every 30min)
├── setup_timesfm.py                 # TimesFM 2.5 installation and patching script
├── pyproject.toml                   # Project dependencies (uv)
├── backtest_engine.py               # Backtesting engine
├── run_short_backtest.py            # Quick backtest script
├── test_full_cycle.py               # Full cycle test
├── test_t212.py                     # Trading 212 integration test
│
├── src/
│   ├── enhanced_trading_example.py  # Main trading system logic
│   ├── classic_model.py             # Scikit-learn ensemble model
│   ├── timesfm_model.py             # TimesFM integration
│   ├── llm_client.py               # Ollama LLM client
│   ├── sentiment_analysis.py        # News/sentiment analysis
│   ├── chart_generator.py           # Technical chart generation
│   ├── features.py                  # Technical indicator creation
│   ├── data.py                      # Market data fetching (API, cache)
│   ├── database.py                  # SQLite persistence (trading_history.db)
│   ├── t212_executor.py             # Trading 212 API execution
│   ├── advanced_risk_manager.py     # Risk filtering layer
│   ├── enhanced_decision_engine.py  # Hybrid decision fusion
│   ├── adaptive_weight_manager.py   # Dynamic model weighting
│   ├── news_fetcher.py              # News data retrieval
│   └── performance_monitor.py       # Performance tracking
│
├── vendor/timesfm/                  # TimesFM 2.5 (installed via setup_timesfm.py)
├── data_cache/                      # Cached market data
├── memory-bank/                     # Documentation for AI agents
└── .agents/                         # Agent configuration
```

## Building and Running

### Prerequisites
- Python 3.12+
- **uv** package manager (https://astral.sh/uv)
- **Ollama** running locally with `gemma4:e4b` pulled
- Alpha Vantage API key (in `.env`)
- Trading 212 API credentials (in `.env.t212`, for live trading)

### Setup
```bash
# 1. Install and patch TimesFM 2.5 (CRITICAL STEP)
# This must be done BEFORE uv sync because uv needs the vendor/timesfm path to exist
python setup_timesfm.py

# 2. Install dependencies
uv sync

# 3. Configure API keys
# Create .env with ALPHA_VANTAGE_API_KEY
# Create .env.t212 with T212_API_KEY, T212_API_SECRET, T212_ENV
```

### Running

```bash
# Basic analysis (default tickers: CRUDP.PA, SXRV.DE)
uv run main.py

# Specific ticker
uv run main.py --ticker QQQ

# Simulation mode (1000€ paper trading, persists in trading_history.db)
uv run main.py --simul

# Live trading via Trading 212
uv run main.py --t212

# Multiple tickers
uv run main.py --ticker QQQ AAPL

# Start the automated scheduler (Mon-Fri, 8:30-18:00, every 30min)
uv run python schedule.py
# Or on Windows:
start_scheduler.bat
```

### Testing
```bash
# Full cycle test
uv run python test_full_cycle.py

# Trading 212 test
uv run python test_t212.py

# Short backtest
uv run python run_short_backtest.py
```

## Development Conventions

- **Module organization**: All core logic lives in `src/`. Entry points are at root level
- **Database**: SQLite (`trading_history.db`) stores portfolio state, transactions, and model signals. Schema defined in `src/database.py`
- **Caching**: Market data is cached in `data_cache/` to reduce API calls
- **Logging**: Dual output to file (`trading.log`, `scheduler.log`) and console
- **CSV Journal**: Every analysis run appends to `trading_journal.csv` for post-trade review
- **Generated artifacts**: `enhanced_trading_chart.png`, `enhanced_performance_dashboard.png`
- **Risk-first approach**: Signals are always filtered through risk management before execution. The system prioritizes capital protection over aggressive trading

## Key Files for Understanding the System

| File | Purpose |
|------|---------|
| `src/enhanced_trading_example.py` | Core trading system orchestration |
| `src/enhanced_decision_engine.py` | Multi-model signal fusion logic |
| `src/advanced_risk_manager.py` | Risk filtering and signal overrides |
| `src/t212_executor.py` | Trading 212 API integration |
| `src/classic_model.py` | Scikit-learn ensemble ML model |
| `src/database.py` | SQLite schema and helpers |
| `schedule.py` | Market-hours scheduler with dashboard |

## Important Notes

- TimesFM 2.5 is vendored as a git submodule in `vendor/timesfm/` and requires patching during setup
- The `.gitignore` excludes all generated artifacts (`.db`, `.csv`, `.json`, `.log`, `.png`, `vendor/`, `data_cache/`)
- Environment files (`.env`, `.env.t212`) are git-ignored; examples are provided (`.env.t212.example`)
- T212 executor uses instrument IDs like `SXRVd_EQ` and `CRUDl_EQ` (not ticker symbols directly)
