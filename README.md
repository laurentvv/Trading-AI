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
    An expert decision-support system for NASDAQ and Oil (WTI) ETF trading, leveraging a 14-model hybrid artificial intelligence powered by unified cloud LLMs for robust and nuanced trading signals.
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
  - [🚀 Dual-Ticker Strategy (Analysis vs. Trading)](#-dual-ticker-strategy-analysis-vs-trading)
  - [🧠 Hybrid AI Engine](#-hybrid-ai-engine)
  - [🧘 Decision Philosophy: "Cognitive Prudence"](#-decision-philosophy-cognitive-prudence)
  - [✨ Key Features](#-key-features)
  - [💻 Tech Stack](#-tech-stack)
  - [🧠 AI & LLM Architecture (NexusAI-Client Cloud Multi-Provider)](#-ai--llm-architecture-nexusai-client-cloud-multi-provider)
  - [🧠 FinAcumen (Financial Memory)](#-finacumen-financial-memory)
- [📂 Project Structure](#-project-structure)
- [🚀 Quick Start](#-quick-start)
  - [✅ Prerequisites](#-prerequisites)
  - [⚙️ Installation](#️-installation)
- [🛠️ Usage](#️-usage)
  - [Simulation Mode (Paper Trading)](#simulation-mode-paper-trading)
  - [Real Execution (Trading 212)](#real-execution-trading-212)
- [🧪 Production Backtesting](#-production-backtesting)
- [🤝 Contributing](#-contributing)
- [📜 License](#-license)

---

## 🌟 About the Project

This project is an expert decision-support system for ETF trading, using a 14-model hybrid AI approach. It is designed to provide a comprehensive and robust analysis by combining several AI perspectives.

### 🚀 Dual-Ticker Strategy (Analysis vs. Trading)
The system uses an innovative approach to maximize model accuracy:
- **High-Fidelity Analysis**: AI models analyze **global reference indices** (`^NDX` for Nasdaq, `CL=F` for WTI Crude Oil). These indices offer longer history and "purer" trends, without the noise related to trading hours or ETF fees.
- **ETF Execution**: Real orders are placed on the corresponding tickers on **Trading 212** (`SXRV.DE`, `CRUDP.PA`), using **T212 live prices** (via positions API) for position sizing. Portfolio state is synchronized directly from T212 (`sync_state_from_t212()`), and live prices are injected into the analysis pipeline (`_inject_t212_live_price()` in `src/data.py`).

### 🧠 Hybrid AI Engine
The system merges thirteen distinct signals (plus a meta-model):
1.  **Classic Quantitative Model**: RandomForest/GradientBoosting/LogisticRegression ensemble trained on technical and macroeconomic indicators.
2.  **TimesFM 3.0 (Google Research)**: State-of-the-art foundation model for time-series forecasting (median forecast + 9 quantiles, CPU inference).
3.  **TensorTrade / PPO (Reinforcement Learning)**: RL agent (stable-baselines3) training a PPO policy in a custom Gymnasium trading environment with persistence across cycles.
4.  **Oil-Bench Model**: Energy-specialized model merging **EIA** fundamental data (Stocks, Imports, Refinery utilization) and sentiment for WTI trading.
5.  **Textual LLM (NexusAI Cloud Multi-Provider)**: Contextual analysis of raw data, real-time news via the **AlphaEar** skill, and integration of dynamic **macro-economic web research**. Powered by `NexusAI-Client` with automatic failover across multiple free and paid frontier cloud providers.
6.  **Visual LLM (NexusAI Multimodal Vision)**: Direct chart pattern analysis (`enhanced_trading_chart.png`).
7.  **Sentiment Analysis**: Hybrid analysis combining Alpha Vantage and "hot" trends from **AlphaEar** (Weibo, WallstreetCN).
8.  **Decentralized Data (Hyperliquid)**: Analysis of speculative sentiment on Oil (WTI) via *Funding Rate* and *Open Interest*.
9.  **Vincent Ganne Model**: Geopolitical and cross-asset analysis (WTI, Brent, Gas, DXY, MA200) for detecting macroeconomic bottoms.
10. **Grebenkov Model**: Trend-Following mathematical model calibrated for cross-asset analysis using Agnostic Risk Parity.
11. **Hidden Markov Model (HMM)**: Probabilistic model for market regime detection (bullish/bearish) based on historical price variations.
12. **FinAcumen (Experience Memory Engine)**: An intelligent ReAct agent loop that evaluates market conditions by writing and executing Python queries, equipped with a vector "Financial Memory".
13. **🏛️ Weekend Council (Strategic Retrospective)**: A weekly, async, multi-persona LLM deliberation running every **Saturday and Sunday at 09:00**. Six personas (Stratège / Risk Manager / Quant / Sceptique / Tacticien / Comportementaliste) each execute on **distinct cloud LLM providers** (Groq, Cerebras, Mistral, Cohere, Nvidia NIM / OpenRouter / OrcaRouter, Gemini Free & Pro) for genuine reasoning diversity. The Judge emits a per-ticker stance that becomes the **11th weighted vote** in the real-time consensus (9.5% weight, decaying linearly over 7 days).
14. **Hybrid Fusion Engine**: The meta-model orchestrating dynamic weighting and cognitive consensus across all sub-models.

The goal is to produce a final decision (`BUY`, `SELL`, `HOLD`) with an absolute priority on **Accuracy First**.

### 🧘 Decision Philosophy: "Cognitive Prudence"
Unlike classic trading algorithms that panic as soon as volatility explodes, this system applies an informed investor approach:
- **Strong Consensus Required**: A quantitative model (Classic) may cry wolf (`SELL`), but if cognitive models (Text LLM, Vision, TimesFM) remain neutral, the system will prefer `HOLD`.
- **Confidence Filter**: A movement decision (Buy or Sell) is only validated if the global confidence exceeds a safety threshold (generally 40%). Below this, the system considers the signal as "noise" and remains on standby.
- **Capital Protection**: In `VERY_HIGH` risk mode, `HOLD` serves as a shield. It prevents entering an unstable market and avoids exiting prematurely on a simple technical correction.

### ✨ Key Features

- **Unified Cloud LLM Architecture via NexusAI-Client**: Complete removal of heavy local Ollama and GGUF dependencies. Direct high-speed API calls with zero-downtime automatic failover across 9+ cloud providers (Gemini Free/Pro, Groq, Cerebras, Mistral, Cohere, Nvidia NIM, OpenRouter, OrcaRouter, DeepSeek).
- **Sub-Minute Cycle Execution**: High-speed parallelized cloud inference brings the full multi-model analysis cycle down from 15 minutes to **~30-45 seconds**.
- **Dual-Ticker Approach**: Analyze the index, trade the ETF.
- **T212 Live Prices**: Real-time recovery of EUR prices via the Trading 212 API (<1s), with yfinance fallback and parquet cache.
- **Dated Brent Spread**: Monitoring of physical market tension via the spread between Brent Spot (Dated) and Brent Futures.
- **Network Resilience**: Circuit breakers and timeouts across all external network calls.
- **Cache Auto-Invalidation**: Parquet cache auto-detects staleness (> 1 day) and forces a refresh.
- **Autonomous Morning Brief Agent**: An overnight analytical workflow (`morning_brief/morning_brief.py`) running daily via `schedule.py`. Synthesizes market reports and injects fundamental awareness into daily trading cycles.
- **News & Blockchain Sentiment**: Integration of **AlphaEar** and **Hyperliquid** to capture social and speculative sentiment.
- **Centralized Risk Management**: The `AdvancedRiskManager` centralizes Anti-Loss (Stop-Loss) and Trailing Stop logic.

### 💻 Tech Stack

- **Language**: `Python 3.12+`
- **Calculations & Data**: `pandas`, `numpy`, `yfinance`, `pyarrow`, `pandas_datareader`, `hyperliquid-python-sdk`
- **Machine Learning**: `scikit-learn`, `shap`, `stable-baselines3`, `gymnasium`
- **AI & LLM**: [`nexusai-client`](https://github.com/laurentvv/NexusAI-Client), `google-genai`
- **Web Scraping & Search**: `beautifulsoup4`, `duckduckgo_search`, `crawl4ai`
- **Visualization**: `matplotlib`, `seaborn`, `mplfinance`
- **Utilities**: `tqdm`, `rich`, `python-dotenv`, `schedule`

---

### 🧠 AI & LLM Architecture (NexusAI-Client Cloud Multi-Provider)

The system leverages **[`NexusAI-Client`](https://github.com/laurentvv/NexusAI-Client)** to unify all cloud AI model calls into a resilient, zero-maintenance gateway:
- **Zero Local Footprint**: No local LLMs to download or run.
- **Automatic Fallback Chain**: Free tiers (Gemini Free, Groq, Cerebras, Nvidia NIM, OrcaRouter, Cohere) are queried first, seamlessly cascading to paid models or alternative providers on 429/503 errors.
- **Multimodal Vision**: Seamless technical chart analysis via multimodal frontier models.
- **Strict JSON Parsing**: Automatic extraction and validation of trading decisions, search queries, and oil allocations.

---

## 📂 Project Structure

```
Trading-AI/
├── morning_brief/                   # Overnight autonomous agent for fundamental analysis
│   ├── morning_brief.py             # Brief orchestrator via NexusAI-Client
│   └── output/                      # Generated daily markdown reports
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
│   ├── enhanced_trading_example.py  # Pipeline execution and orchestration
│   ├── features.py                  # Technical and macroeconomic feature engineering
│   ├── grebenkov_model.py           # Trend-Following math model
│   ├── hmm_model.py                 # Hidden Markov Model for regime detection
│   ├── llm_client.py                # Unified LLM inference via NexusAI-Client
│   ├── news_fetcher.py              # Financial news crawling and parsing
│   ├── oil_bench_model.py           # Energy-specialized WTI trading model
│   ├── performance_monitor.py       # P&L and risk metrics monitoring
│   ├── sentiment_analysis.py        # Sentiment analysis engine
│   ├── t212_executor.py             # Trading 212 order execution & state sync
│   ├── tensortrade_model.py         # Reinforcement learning model
│   ├── timesfm_model.py             # Google TimesFM foundation model wrapper
│   ├── vincent_ganne_model.py       # Geopolitical bottom-detection model
│   ├── web_researcher.py            # Dynamic web research query generator
│   ├── council/                     # Weekend AI Council deliberation suite
│   │   ├── weekend_council.py       # 3-round multi-provider debate orchestrator
│   │   └── council_prompts.py       # Personas and debate templates
│   └── agents/                      # FinAcumen cognitive ReAct agent
├── tests/                           # Comprehensive unit and integration test suite
├── main.py                          # Pipeline entry point
├── schedule.py                      # Production scheduler
└── scheduler_config.json            # Centralized parameter configuration
```

---

## 🚀 Quick Start

### ✅ Prerequisites

- Python 3.12+ (via `uv`)
- API keys in `.env` (Gemini, Groq, Mistral, Nvidia, Alpha Vantage, EIA, etc. — see `.env.example`)

### ⚙️ Installation

1. **Install `uv`**: [astral.sh/uv](https://astral.sh/uv)
2. **Install Dependencies (incl. TimesFM 3.0 from PyPI)**:
   ```bash
   uv sync
   uv run python -m playwright install chromium
   ```
3. **Configure Environment**:
   Copy `.env.example` to `.env` and fill in your API keys.
4. **Pre-download the TimesFM 3.0 checkpoint (~1.3 GB, once per machine)**:
   ```bash
   uv run python tests/smoke_timesfm3.py
   ```

---

## 🛠️ Usage

```bash
# Paper trading simulation (NASDAQ)
uv run main.py --simul

# Paper trading simulation (Oil)
uv run main.py --simul --ticker CRUDP.PA

# Live/Demo execution on Trading 212
uv run main.py --t212

# Run full automated scheduler (8:30 AM - 6:00 PM)
uv run schedule.py

# Run Weekend AI Council on demand
uv run python -m src.council.weekend_council --days 7

# Run Morning Market Brief on demand
uv run python morning_brief/morning_brief.py
```

---

## 🤝 Contributing

Contributions are welcome! Please open an issue or submit a PR.

## 📜 License

This project is licensed under the MIT License - see the `LICENSE` file for details.
