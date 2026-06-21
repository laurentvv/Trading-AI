<p align="center">
  <a href="README.md">English</a> |
  <a href="i18n/README_zh.md">中文</a> |
  <a href="i18n/README_hi.md">हिंदी</a> |
  <a href="i18n/README_es.md">Español</a> |
  <a href="i18n/README_fr.md">Français</a> |
  <a href="i18n/README_ar.md">العربية</a> |
  <a href="i18n/README_bn.md">বাংলা</a> |
  <a href="i18n/README_ru.md">Русский</a> |
  <a href="i18n/README_pt.md">Português</a> |
  <a href="i18n/README_id.md">Bahasa Indonesia</a>
</p>

<p align="center">
  <img src="assets/banner.png" alt="Hybrid AI Trading Banner" width="100%"/>
</p>

<div align="center">
  <br />
  <h1>📈 Hybrid AI Trading System 📈</h1>
  <p>
    An expert decision-support system for NASDAQ and Oil (WTI) ETF trading, leveraging a 12-model hybrid artificial intelligence for robust and nuanced trading signals.
  </p>
</div>

<div align="center">

[![Project Status](https://img.shields.io/badge/status-in--development-green.svg)](https://github.com/laurentvv/Trading-AI)
[![Python Version](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-lightgrey.svg)](https://opensource.org/licenses/MIT)

</div>

---

## 📚 Table of Contents

- [🌟 About the Project](#-about-the-project)
  - [✨ Key Features](#-key-features)
  - [💻 Tech Stack](#-tech-stack)
  - [⚙️ Performance & Hardware](#️-performance--hardware)
- [📂 Project Structure](#-project-structure)
- [🚀 Quick Start](#-quick-start)
  - [✅ Prerequisites](#-prerequisites)
  - [⚙️ Installation](#️-installation)
- [🛠️ Usage](#️-usage)
  - [Manual Analysis](#-manual-analysis)
  - [Automated Analysis with Intelligent Scheduler](#-automated-analysis-with-intelligent-scheduler)
- [🤝 Contributing](#-contributing)
- [📜 License](#-license)
- [📧 Contact](#-contact)

---

## 🌟 About the Project

This project is an expert decision-support system for ETF trading, using a 12-model hybrid AI approach. It is designed to provide a comprehensive and robust analysis by combining several AI perspectives.

### 🚀 Dual-Ticker Strategy (Analysis vs. Trading)
The system uses an innovative approach to maximize model accuracy:
- **High-Fidelity Analysis**: AI models analyze **global reference indices** (`^NDX` for Nasdaq, `CL=F` for WTI Crude Oil). These indices offer longer history and "purer" trends, without the noise related to trading hours or ETF fees.
- **ETF Execution**: Real orders are placed on the corresponding tickers on **Trading 212** (`SXRV.DE`, `CRUDP.PA`), using **T212 live prices** (via positions API) for position sizing. Portfolio state is synchronized directly from T212 (`sync_state_from_t212()`), and live prices are injected into the analysis pipeline (`_inject_t212_live_price()` in `src/data.py`).

### 🧠 Hybrid AI Engine
The system merges twelve distinct signals (plus a meta-model):
1.  **Classic Quantitative Model**: RandomForest/GradientBoosting/LogisticRegression ensemble trained on technical and macroeconomic indicators.
2.  **TimesFM 2.5 (Google Research)**: State-of-the-art foundation model for time-series forecasting.
3.  **TensorTrade / PPO (Reinforcement Learning)**: RL agent (stable-baselines3) training a PPO policy in a custom Gymnasium trading environment with persistence across cycles.
4.  **Oil-Bench Model (Gemma 4 12B (Unsloth))**: Energy-specialized model merging **EIA** fundamental data (Stocks, Imports, Refinery utilization) and sentiment for WTI trading.
5.  **Textual LLM (Gemma 4 12B (Unsloth))**: Contextual analysis of raw data, real-time news via the **AlphaEar** skill, and integration of dynamic **macro-economic web research**. It explicitly consumes the overnight **Morning Brief** report to gain deep fundamental awareness before making decisions.
6.  **Visual LLM (Gemma 4 12B (Unsloth))**: Direct analysis of technical charts (`enhanced_trading_chart.png`).
7.  **Sentiment Analysis**: Hybrid analysis combining Alpha Vantage and "hot" trends from **AlphaEar** (Weibo, WallstreetCN).
8.  **Decentralized Data (Hyperliquid)**: Analysis of speculative sentiment on Oil (WTI) via *Funding Rate* and *Open Interest*.
9.  **Vincent Ganne Model**: Geopolitical and cross-asset analysis (WTI, Brent, Gas, DXY, MA200) for detecting macroeconomic bottoms.
10. **Grebenkov Model**: Trend-Following mathematical model calibrated for cross-asset analysis using Agnostic Risk Parity.
11. **Hidden Markov Model (HMM)**: Probabilistic model for market regime detection (bullish/bearish) based on historical price variations.
12. **FinAcumen (Experience Memory Engine)**: An intelligent ReAct agent loop that evaluates market conditions by writing and executing raw Python queries against simulated datasets, equipped with a vector "Financial Memory".
13. **Hybrid Fusion Engine**: The meta-model orchestrating dynamic weighting and cognitive consensus across all sub-models.

The goal is to produce a final decision (`BUY`, `SELL`, `HOLD`) with an absolute priority on **Accuracy First**.

### 🧘 Decision Philosophy: "Cognitive Prudence"
Unlike classic trading algorithms that panic as soon as volatility explodes, this system applies an informed investor approach:
- **Strong Consensus Required**: A quantitative model (Classic) may cry wolf (`SELL`), but if cognitive models (Text LLM, Vision, TimesFM) remain neutral, the system will prefer `HOLD`.
- **Confidence Filter**: A movement decision (Buy or Sell) is only validated if the global confidence exceeds a safety threshold (generally 40%). Below this, the system considers the signal as "noise" and remains on standby.
- **Capital Protection**: In `VERY_HIGH` risk mode, `HOLD` serves as a shield. It prevents entering an unstable market and avoids exiting prematurely on a simple technical correction if fundamentals (News/Vision/Hyperliquid) do not confirm an imminent crash.

### ✨ Key Features

- **Dual-Ticker Approach**: Analyze the index, trade the ETF.
- **T212 Live Prices**: Real-time recovery of EUR prices via the Trading 212 API (0.2s), with yfinance fallback and parquet cache.
- **Dated Brent Spread**: Monitoring of physical market tension via the spread between Brent Spot (Dated) and Brent Futures.
- **Network Resilience**: yfinance circuit breaker with separate trackers (info vs. download), 10s timeout on all network calls.
- **Cache Auto-Invalidation**: Parquet cache auto-detects staleness (> 2 days) and forces a refresh. Use `refresh_cache.py` for manual cache clearing.
- **LLM Call Parallelization**: Independent model calls (`text_llm`, `visual_llm`, `search_query`, `timesfm`, `tensortrade`, `grebenkov`) run in a `ThreadPoolExecutor` to overlap Ollama inference with I/O. Critical path typically 4–6 min on CPU vs 10+ min sequential.
- **24h Search-Query Cache**: The LLM-generated web-search query is cached under `data_cache/search_queries/<ticker>_<date>_<price-sig>.json`. Keyed by date + a price-action signature (log2 bucketing of close + RSI bucket), so a regime change invalidates it. Fallback queries are **never** cached (one transient Ollama failure cannot poison the cache for 24h).
- **Hard Cycle Timeout**: Each ticker cycle is wrapped in a 15-min budget (`CYCLE_TIMEOUT_SECONDS` in `main.py`). On timeout, the worker thread is `shutdown(wait=False)` so the next ticker starts immediately; HOLD is applied to the timed-out ticker. Individual futures have their own per-task timeouts (search 240s, visual 300s, text 240s, CPU models 180s each, news 90s, web crawl 30s).
- **Orphan-Thread Safety**: On cycle timeout, a per-ticker `threading.Event` is set so the orphan worker bails out before any `execute_t212_trade` call — preventing real-money trades after the user has been shown the "HOLD appliqué" panel. A per-ticker `threading.Lock` further serializes T212 order placement, eliminating double-trade risk under scheduler overlap or duplicate `--ticker` invocations.
- **LLM Failure Sentinel**: When `_query_ollama` exhausts all retries, the fallback dict carries a `"failed": True` flag so downstream consensus logic can distinguish "model chose HOLD" from "model crashed" (currently propagated but not filtered — a known follow-up).
- **Advanced Cognition**: Use of **Gemma 4 12B** with **dual-layer JSON defence**:
  1. **Server-side schema enforcement** (`format: SCHEMA_*` with `additionalProperties: false`) — the load-bearing layer; passed via Ollama's `format` parameter at every call site. Schemas defined in `src/llm_client.py` (`SCHEMA_TRADING_DECISION`, `SCHEMA_SEARCH_QUERY`, `SCHEMA_OIL_ALLOCATION`).
  2. **Defensive system-prompt suffix** (`"...never add a 'thought' key."`) — redundant-but-harmless second line, kept as belt-and-braces against any future regression of the schema layer.

  The `<|think|>` reasoning token is **active** in all four production system prompts (re-enabled 2026-06-06 on `main` after validation on `think-mode` branch). The schema layer is what actually neutralises the historical `<|channel>thought` JSON-debris defect (May 2026 root cause): `tests/check_llm_json.py` confirms that schema-strict cases (`v3_schema`, `v6_schema`, `v7_schema_strict`) produce clean JSON even with `<|think|>` enabled, while the loose `format:json` variants fail. See `docs/ADR-001-think-mode-dual-layer-defence.md` for the full analysis and reversal procedure.
- **Autonomous Morning Brief Agent**: An overnight `smolagents`-based workflow (`morning_brief/morning_brief.py`) scheduled to run automatically at 01:00 AM via `schedule.py`. It independently crawls daily API logs, downloads fundamental EIA inventory data, and arbitrates a *Bull vs Bear* debate. The resulting markdown report (`morning_market_brief.md`) is automatically injected into the Textual LLM's system prompt during the daily trading cycle, granting the main AI deep contextual memory and fundamental awareness without slowing down live market execution.
- **News & Blockchain Sentiment**: Integration of **AlphaEar** and **Hyperliquid** to capture social and speculative sentiment.
- **Automated Scheduler**: `schedule.py` script for continuous execution (8:30 AM - 6:00 PM) on a server.
- **Centralized Risk Management**: The `AdvancedRiskManager` centralizes Anti-Loss (Stop-Loss) and Trailing Stop logic. Individual models no longer manage these risks, ensuring a unified and strict capital protection strategy across varying market regimes.
- **Strict Data Contracts**: All AI models are fully standardized to return a strongly-typed `ModelResult` dataclass (`signal`, `confidence`, `reasoning`), ensuring 100% uniformity across the consensus engine.
- **Code Health Audited**: Project maintains a **Grade B** code health standard via automated audits (0 dead code, high maintainability index).
- **Production Backtesting**: Standalone backtest engine (`backtest_prod.py`) replaying real prod signals against real prices with T212 fees — no external dependencies.
- **Debug Dump Control**: Set `TRADING_DEBUG_DUMP=0` to disable the capped (5 MB) `data_cache/llm_debug_fail.txt` LLM-failure dump.

### 💻 Tech Stack

- **Language**: `Python 3.12+`
- **Calculations & Data**: `pandas`, `numpy`, `yfinance`, `pyarrow`, `pandas_datareader`, `hyperliquid-python-sdk`
- **Machine Learning**: `scikit-learn`, `shap`
- **AI & LLM**: `requests`, `ollama`
- **Web Scraping & Search**: `beautifulsoup4`, `duckduckgo_search`, `crawl4ai`
- **Visualization**: `matplotlib` (Agg backend for thread safety), `seaborn`, `mplfinance`
- **Utilities**: `tqdm`, `rich`, `python-dotenv`, `schedule`

### ⚙️ Performance & Hardware
The system is designed to be **performant on consumer hardware** without requiring a dedicated GPU.
- **CPU Only**: LLM inference (Gemma 4 12B Q6_K via Ollama) and TimesFM run entirely on CPU. Throughput is ~3–4 tokens/s on a modern 8-core CPU.
- **Recommended RAM**: 16 GB minimum (32 GB suggested to run Gemma 4 12B comfortably alongside TimesFM and TensorTrade).
- **Ollama Concurrency**: Set `OLLAMA_NUM_PARALLEL=8` (already in the recommended `.env`) so multiple LLM calls can share model load. With the default 4 GB context budget, parallel slots get ~512 tokens each — Ollama will serialize if prompts exceed the per-slot ctx, but the `ThreadPoolExecutor` keeps the wall-clock overlap beneficial for I/O-bound steps (news fetch, web crawl, CPU models).
- **Execution Time**: ~6 to 9 minutes per ticker on CPU (cold), ~3 to 5 minutes per ticker with search-query cache hit. Default runs two tickers (CRUDP.PA + SXRV.DE), so plan ~15 min total.
- **Cycle Timeout**: Each ticker cycle is bounded at 15 min (`CYCLE_TIMEOUT_SECONDS`). If exceeded, HOLD is applied and the next ticker starts immediately.
- **API Speed**: Ultra-fast Trading 212 integration (<1s for live price recovery).

---


### 🧠 FinAcumen (Financial Memory)
L'architecture FinAcumen a été intégrée pour doter les modèles IA locaux d'une **mémoire d'expérience** et d'outils déterministes. Cela résout le problème de l'amnésie des LLMs.
- Lancez `uv run python src/finacumen_main.py --ticker CL=F` pour générer une analyse structurée qui interrogera la base de données vectorielle locale.
- Le résultat est mis en cache (JSON) et lu automatiquement par `main.py` au cycle suivant pour l'inclure dans le système hybride.

## 📂 Project Structure

The project is organized modularly for better maintainability.

```
Trading-AI/
├── morning_brief/                   # Overnight autonomous agent for deep fundamental analysis
│   ├── morning_brief.py             # Agent orchestrator and smolagents configuration
│   └── output/                      # Generated daily markdown reports (morning_market_brief.md)
├── src/                             # Core modules
│   ├── adaptive_weight_manager.py   # Dynamic model weighting based on performance
│   ├── advanced_risk_manager.py     # Trend-Aware risk management and sizing
│   ├── bootstrap.py                 # Core initialization logic
│   ├── chart_generator.py           # Generates technical charts for visual LLM
│   ├── classic_model.py             # Scikit-learn quantitative models ensemble
│   ├── config_weights.py            # Base weights configuration for the hybrid engine
│   ├── data.py                      # Data fetching, caching, and preprocessing
│   ├── database.py                  # SQLite database management for metrics
│   ├── eia_client.py                # Energy Information Administration API client
│   ├── enhanced_decision_engine.py  # Hybrid fusion engine orchestrating all models
│   ├── enhanced_trading_example.py  # Example scripts for model utilization
│   ├── features.py                  # Technical and macroeconomic feature engineering
│   ├── grebenkov_model.py           # Trend-Following math model (Agnostic Risk Parity)
│   ├── hmm_model.py                 # Hidden Markov Model for regime detection
│   ├── llm_client.py                # Ollama integration for local LLM inference
│   ├── news_fetcher.py              # Financial news crawling and parsing
│   ├── oil_bench_model.py           # Energy-specialized WTI trading model
│   ├── performance_monitor.py       # Tracking model accuracy and history
│   ├── read_simul.py                # Tools for reading simulation outputs
│   ├── sentiment_analysis.py        # Alpha Vantage & AlphaEar sentiment integration
│   ├── t212_executor.py             # Trading 212 API real execution and portfolio
│   ├── tensortrade_model.py         # Reinforcement Learning (PPO) signal
│   ├── timesfm_model.py             # TimesFM 2.5 time-series forecasting integration
│   └── web_researcher.py            # Macro-economic web scraping with Crawl4AI
├── data_cache/                       # All caches (gitignored)
│   ├── *.parquet                     # OHLCV data per ticker (yfinance)
│   ├── macro/                        # Macro time series (FRED, multi-source)
│   ├── search_queries/               # 24h LLM search-query cache (per ticker+date+price-sig)
│   └── llm_debug_fail.txt            # Capped (5 MB) LLM failure dump — disable with TRADING_DEBUG_DUMP=0
├── tests/                            # Test and validation scripts
│   ├── test_full_cycle.py            # End-to-end T212 buy/wait/sell test
│   ├── test_enhanced_decision_engine.py # Tests for the hybrid fusion engine
│   ├── check_llm_json.py             # LLM JSON-schema diagnostic (tests all 4 Ollama call sites)
│   ├── check_live.py                 # Live market prices verification script
│   └── ...                           # Other unit and integration tests
├── i18n/                            # Internationalization (Translated READMEs)
├── assets/                          # Static assets (images, banners)
├── memory-bank/                     # AI assistant memory and context
├── backtest_prod.py                 # Standalone production backtest engine
├── main.py                          # Single entry point (Analysis & Trading)
├── pyproject.toml                   # Project dependencies and configuration (uv)
├── refresh_cache.py                 # CLI utility to force-refresh Parquet cache
├── schedule.py                      # Live scheduler for automated execution
├── setup_timesfm.py                 # Installation script for TimesFM 2.5 vendor
├── .env.example                     # Example environment variables
└── README.md                        # This documentation
```

---

## 🚀 Quick Start

Follow these steps to set up your local development environment.

### ✅ Prerequisites

- Python 3.12+ (via `uv`)
- [Ollama](https://ollama.com/) installed and running locally.
- Downloaded LLM model: `ollama pull hf.co/unsloth/gemma-4-12b-it-GGUF:Q6_K`

### ⚙️ Installation

1.  **Clone the repository:**
    ```sh
    git clone https://github.com/laurentvv/Trading-AI.git
    cd Trading-AI
    ```
2.  **Install `uv` (if not already done):**
    See [astral.sh/uv](https://astral.sh/uv) for installation instructions.

3.  **Create and activate the virtual environment (CRUCIAL Step):**
    You must create and activate the `.venv` before installing the foundation models.
    ```bash
    uv venv
    source .venv/bin/activate  # On Windows, use `.\.venv\Scripts\activate.ps1`
    ```

4.  **Install Foundation Models:**
    Run the installation scripts to clone the models into `vendor/` and apply patches:
    ```bash
    python setup_timesfm.py
    ```

5.  **Initialize and synchronize the environment:**
    ```bash
    uv sync
    ```

6.  **Install browsers for Web research (Crawl4AI):**
    ```bash
    uv run python -m playwright install chromium
    ```

7.  **Configure your API keys:**
    Create a `.env` file in the project root:
    ```
    ALPHA_VANTAGE_API_KEY="YOUR_KEY"
    EIA_API_KEY="YOUR_KEY"
    ```

---

## 🛠️ Usage

The system trains its models on the most recent data at each execution before giving a decision.

### Simulation Mode (Paper Trading)

To test the system without risk with a fictitious capital of €1000, use the `--simul` flag. The system will manage a strict history of buys and sells.

```sh
# Run a simulated analysis (Default: SXRV.DE - Nasdaq 100 EUR)
uv run main.py --simul

# Run on Oil (WTI)
uv run main.py --ticker CRUDP.PA --simul
```

### Real Execution (Trading 212)

The system is now **fully integrated** with Trading 212:
- **Portfolio Verification**: Before any action, the robot consults your real cash and positions.
- **API Management**: Includes automatic retry mechanisms against request limits (Rate Limiting).

```sh
# Run analysis with real execution (Demo or Real according to .env)
uv run main.py --t212
```

---

## 🧪 Production Backtesting

The system includes a **standalone production backtest engine** (`backtest_prod.py`) that replays actual prod signals from `logs_prod/trading_journal.csv` against real prices from `data_cache/` Parquet files.

### Features
- **Real signals**: Replays the exact decisions of the 12-model hybrid engine.
- **Real prices**: Uses actual ETF OHLCV data (SXRV.DE, CRUDP.PA) — no US proxies.
- **T212 fees**: Simulates Trading 212's 0.1% per-trade fee model.
- **Baseline comparison**: Automatically computes buy-and-hold performance as benchmark.
- **Metrics**: Sharpe Ratio, Maximum Drawdown, Win Rate, Alpha, Total Return per ticker.

### Usage

```bash
uv run python backtest_prod.py
```

Results saved to `logs_prod/backtest_report.json` with equity curves CSV.

---

## 🤝 Contributing

Contributions are welcome! Feel free to fork the project and open a Pull Request.

---

## 📜 License

Distributed under the MIT License.

---

## 📧 Contact

Project Link: [https://github.com/laurentvv/Trading-AI](https://github.com/laurentvv/Trading-AI)
