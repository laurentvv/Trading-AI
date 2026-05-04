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
    An expert decision-support system for NASDAQ and Oil (WTI) ETF trading, leveraging a tri-modal hybrid artificial intelligence for robust and nuanced trading signals.
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

This project is an expert decision-support system for ETF trading, using a tri-modal hybrid AI approach. It is designed to provide a comprehensive and robust analysis by combining several AI perspectives.

### 🚀 Dual-Ticker Strategy (Analysis vs. Trading)
The system uses an innovative approach to maximize model accuracy:
- **High-Fidelity Analysis**: AI models analyze **global reference indices** (`^NDX` for Nasdaq, `CL=F` for WTI Crude Oil). These indices offer longer history and "purer" trends, without the noise related to trading hours or ETF fees.
- **ETF Execution**: Real orders are placed on the corresponding tickers on **Trading 212** (`SXRV.DE`, `CRUDP.PA`), using **T212 live prices** (via positions API) for position sizing.

### 🧠 Hybrid AI Engine
The system merges nine distinct signals:
1.  **Classic Quantitative Model**: RandomForest/GradientBoosting/LogisticRegression ensemble trained on technical and macroeconomic indicators.
2.  **TimesFM 2.5 (Google Research)**: State-of-the-art foundation model for time-series forecasting.
3.  **TensorTrade / PPO (Reinforcement Learning)**: RL agent (stable-baselines3) training a PPO policy in a custom Gymnasium trading environment at each run.
4.  **Oil-Bench Model (Gemma 4:e4b)**: Energy-specialized model merging **EIA** fundamental data (Stocks, Imports, Refinery utilization) and sentiment for WTI trading.
5.  **Textual LLM (Gemma 4:e4b)**: Contextual analysis of raw data, real-time news via the **AlphaEar** skill, and integration of dynamic **macro-economic web research**.
6.  **Visual LLM (Gemma 4:e4b)**: Direct analysis of technical charts (`enhanced_trading_chart.png`).
7.  **Sentiment Analysis**: Hybrid analysis combining Alpha Vantage and "hot" trends from **AlphaEar** (Weibo, WallstreetCN).
8.  **Decentralized Data (Hyperliquid)**: Analysis of speculative sentiment on Oil (WTI) via *Funding Rate* and *Open Interest*.
9.  **Vincent Ganne Model**: Geopolitical and cross-asset analysis (WTI, Brent, Gas, DXY, MA200) for detecting macroeconomic bottoms.

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
- **Advanced Cognition**: Use of **Gemma 4** for better technical/fundamental synthesis.
- **News & Blockchain Sentiment**: Integration of **AlphaEar** and **Hyperliquid** to capture social and speculative sentiment.
- **Automated Scheduler**: `schedule.py` script for continuous execution (8:30 AM - 6:00 PM) on a server.
- **Advanced Risk Management**: Automatic signal adjustment based on volatility and market regime.
- **Institutional Backtesting (QuantConnect Lean)**: Backtesting engine with realistic fill simulation, slippage, T212 fee model, and Sharpe/Sortino/MaxDD reporting — without modifying the production pipeline.

### 💻 Tech Stack

- **Language**: `Python 3.12+`
- **Calculations & Data**: `pandas`, `numpy`, `yfinance`, `pyarrow`, `pandas_datareader`, `hyperliquid-python-sdk`
- **Machine Learning**: `scikit-learn`, `shap`
- **AI & LLM**: `requests`, `ollama`
- **Web Scraping & Search**: `beautifulsoup4`, `duckduckgo_search`, `crawl4ai`
- **Visualization**: `matplotlib`, `seaborn`, `mplfinance`
- **Utilities**: `tqdm`, `rich`, `python-dotenv`, `schedule`

### ⚙️ Performance & Hardware
The system is designed to be **performant on consumer hardware** without requiring a dedicated GPU.
- **CPU Only**: LLM inference (Gemma 4 via Ollama) and TimesFM are optimized for fast CPU execution if enough RAM is available.
- **Recommended RAM**: 16 GB minimum (32 GB suggested to run Gemma 4 comfortably).
- **Execution Time**: ~2 to 5 minutes for a full cycle (including web crawling, ML training, TimesFM predictions, and 3 LLM analyses).
- **API Speed**: Ultra-fast Trading 212 integration (<1s for live price recovery).

---

## 📂 Project Structure

The project is organized modularly for better maintainability.

```
Trading-AI/
├── src/                     # Core modules
│   ├── eia_client.py               # Energy fundamental data client
│   ├── oil_bench_model.py          # Energy specialized model
│   ├── tensortrade_model.py        # Reinforcement Learning signal (PPO/SB3)
│   ├── enhanced_decision_engine.py # Fusion engine and Vincent Ganne model
│   ├── advanced_risk_manager.py    # Trend-Aware risk management
│   ├── adaptive_weight_manager.py  # Dynamic model weighting
│   ├── t212_executor.py            # Real execution on Trading 212
│   ├── timesfm_model.py            # TimesFM 2.5 integration
│   └── ...                         # Data, Features, LLM Client
├── TradingAI-Lean/           # QuantConnect Lean backtesting (Docker-based)
│   ├── main.py                       # Baseline buy-and-hold algorithm (benchmark)
│   ├── TradingAIFrameworkAlgorithm.py # Full Alpha/Portfolio/Risk/Execution framework
│   ├── AlphaModels/                  # Trading-AI models encapsulated as Lean Alpha Models
│   ├── CustomData/                   # Custom data feeds (EIA macro)
│   └── lean_config.json              # Lean CLI configuration
├── tests/                   # Test and validation scripts
│   ├── check_cache.py               # Inspect Parquet cache files (dates, sizes)
│   ├── check_db.py                  # Inspect SQLite databases (tables, rows)
│   ├── check_live.py                # Live market prices via yfinance
│   ├── test_full_cycle.py           # End-to-end T212 buy/wait/sell test
│   └── ...                          # Unit & integration tests
├── data_cache/              # Market and macro data (Parquet)
├── src/
│   ├── lean_bridge.py                # Converts trading_journal.csv → Lean signal format
│   ├── lean_validator.py             # Automated validation via Lean backtest
│   └── ...                           # Core modules (decision engine, risk, T212, etc.)
├── main.py                  # Single entry point (Analysis & Trading)
├── run_lean_backtest.py     # Lean backtest launcher (--validate, --compare)
├── schedule.py              # Live scheduler (8:30 AM - 6:00 PM)
├── refresh_cache.py         # CLI utility to force-refresh Parquet cache
├── .env                     # API Keys (Alpha Vantage, T212, EIA)
└── README.md                # This documentation
```

---

## 🚀 Quick Start

Follow these steps to set up your local development environment.

### ✅ Prerequisites

- Python 3.12+ (via `uv`)
- [Ollama](https://ollama.com/) installed and running locally.
- Downloaded LLM model: `ollama pull gemma4:e4b`

### ⚙️ Installation

1.  **Clone the repository:**
    ```sh
    git clone https://github.com/laurentvv/Trading-AI.git
    cd Trading-AI
    ```
2.  **Install `uv` (if not already done):**
    See [astral.sh/uv](https://astral.sh/uv) for installation instructions.

3.  **Install and Patch TimesFM 2.5 (CRUCIAL Step):**
    Run the installation script to clone the model into `vendor/` and apply patches:
    ```bash
    python setup_timesfm.py
    ```

4.  **Initialize and synchronize the environment:**
    ```bash
    uv sync
    ```

5.  **Install browsers for Web research (Crawl4AI):**
    ```bash
    uv run python -m playwright install chromium
    ```

6.  **Configure your API keys:**
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

## 🧪 Institutional Backtesting with QuantConnect Lean

The system includes an **additive backtesting layer** based on [QuantConnect Lean](https://github.com/QuantConnect/Lean) — an open-source institutional-grade algorithmic trading engine. This module is **completely separate** from the production pipeline and does not modify any existing code.

### Why?

The built-in backtest (`+221.95%` over 10 years) does not account for slippage, bid-ask spread, or realistic fill simulation. Lean provides:
- **Realistic execution**: Fill simulation with volume-share slippage and T212 fee model (0.1% per trade).
- **Per-model validation**: Each of the 9 AI models can be tested independently as a Lean Alpha Model.
- **Institutional metrics**: Sharpe Ratio, Sortino, Maximum Drawdown, Alpha/Beta, Tracking Error.
- **Parameter optimization**: Grid search or Bayesian optimization on model weights and confidence thresholds.

### Architecture

```
Trading-AI (Production)                    Lean Engine (Backtesting)
┌─────────────────────┐                    ┌──────────────────────────┐
│ 9 IA Models →       │                    │ Lean CLI (Docker)        │
│ Decision Engine →   │                    │ Alpha Models (proxies)   │
│ T212 Executor       │                    │ Fill/Slippage simulation │
│ trading_journal.csv │── LeanSignalBridge─→│ Sharpe/MaxDD/Sortino     │
└─────────────────────┘                    └──────────────────────────┘
```

### Usage

```sh
# 1. Export Trading-AI signals to Lean format
python run_lean_backtest.py --export-signals

# 2. Run baseline backtest (buy-and-hold QQQ/USO)
cd TradingAI-Lean && lean backtest

# 3. Run full framework with Alpha Models
cd TradingAI-Lean && lean backtest --algorithm TradingAIFrameworkAlgorithm

# 4. Validate changes against thresholds (CI/CD style)
python run_lean_backtest.py --validate

# 5. Compare baseline vs framework algorithms
python run_lean_backtest.py --compare
```

### Ticker Proxies

Lean uses US market data (free via QuantConnect). European ETFs are mapped to US equivalents:

| Trading-AI Ticker | Lean Proxy | Asset |
|---|---|---|
| SXRV.DE | QQQ | Nasdaq 100 ETF |
| CRUDP.PA | USO | Oil ETF |
| ^NDX | QQQ | Nasdaq Index |
| CL=F | USO | WTI Crude Futures |

### Key Files

| File | Role |
|---|---|
| `src/lean_bridge.py` | Converts `trading_journal.csv` → Lean-compatible signals (CSV + JSON) |
| `src/lean_validator.py` | Runs Lean backtest, validates Sharpe/MaxDD/Return against thresholds |
| `run_lean_backtest.py` | CLI launcher (`--export-signals`, `--validate`, `--compare`) |
| `TradingAI-Lean/main.py` | Baseline buy-and-hold algorithm with T212 fee model |
| `TradingAI-Lean/TradingAIFrameworkAlgorithm.py` | Full Alpha/Portfolio/Risk/Execution framework |
| `TradingAI-Lean/AlphaModels/` | Individual and composite Alpha Models (Classic, TimesFM, VG, etc.) |

---

## 🤝 Contributing

Contributions are welcome! Feel free to fork the project and open a Pull Request.

---

## 📜 License

Distributed under the MIT License.

---

## 📧 Contact

Project Link: [https://github.com/laurentvv/Trading-AI](https://github.com/laurentvv/Trading-AI)
