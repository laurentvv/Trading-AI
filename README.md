<p align="center">
  <a href="README.md">English</a> |
  <a href="i18n/README_fr.md">Français</a>
</p>

<p align="center">
  <img src="assets/banner.png" alt="Hybrid AI Trading Banner" width="100%"/>
</p>

<div align="center">
  <br />
  <h1>📈 Hybrid AI Trading System 📈</h1>
  <p>
    <b>High-conviction, multi-model algorithmic trading decision pipeline for NASDAQ and Energy Sector ETFs.</b><br />
    Guided by the quantitative principle: <i>"In trading, complexity doesn't pay — start with the simplest technique and demand proof for every layer of complexity."</i>
  </p>
</div>

<div align="center">

[![Project Status](https://img.shields.io/badge/status-active%20production-success.svg)](https://github.com/laurentvv/Trading-AI)
[![Python Version](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-lightgrey.svg)](https://opensource.org/licenses/MIT)
[![Engine](https://img.shields.io/badge/LLM%20Gateway-NexusAI--Client-orange.svg)](https://github.com/laurentvv/NexusAI-Client)
[![Foundation Model](https://img.shields.io/badge/Forecasting-TimesFM%203.0-purple.svg)](https://github.com/google-research/timesfm)

</div>

---

## 📚 Table of Contents

- [🌟 Architectural Vision](#-architectural-vision)
  - [Dual-Ticker Strategy (Index Analysis vs. ETF Execution)](#dual-ticker-strategy-index-analysis-vs-etf-execution)
  - [The Streamlined Model Stack (Complexity Reduction)](#the-streamlined-model-stack-complexity-reduction)
  - [Energy Pivot: From Synthetic ETC to Physical Sector ETF](#energy-pivot-from-synthetic-etc-to-physical-sector-etf)
- [🛡️ Production Invariants & Safety GO-Gates](#️-production-invariants--safety-go-gates)
- [🧠 The High-Conviction Model Ensemble](#-the-high-conviction-model-ensemble)
- [🧪 Vectorized Backtesting Engine](#-vectorized-backtesting-engine)
- [📂 Project Structure](#-project-structure)
- [🚀 Quick Start](#-quick-start)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Pre-warming Foundation Checkpoints](#pre-warming-foundation-checkpoints)
- [🛠️ Operational Execution](#️-operational-execution)
  - [Paper Trading Simulation](#paper-trading-simulation)
  - [Live / Demo Broker Execution (Trading 212)](#live--demo-broker-execution-trading-212)
  - [Continuous Automated Scheduler](#continuous-automated-scheduler)
  - [Weekend Strategic Council Deliberation](#weekend-strategic-council-deliberation)
  - [Autonomous Morning Brief](#autonomous-morning-brief)
- [📜 License](#-license)

---

## 🌟 Architectural Vision

### Dual-Ticker Strategy (Index Analysis vs. ETF Execution)
Financial retail instruments (ETFs) frequently exhibit discontinuous quotes, wide market-maker spreads outside core hours, or feed anomalies. This system separates analytical perception from financial execution:
1. **Perception on Reference Indices**: AI models analyze global liquid underlying benchmarks (**`^NDX`** for tech equities, **`CL=F`** for crude energy macro drivers). These benchmarks offer decades of deep historical data, clean volatility structures, and high continuous liquidity.
2. **Execution on European UCITS ETFs**: Validated decisions route to specific liquid EUR instruments on **Trading 212**:
   - Tech Index: **`SXRV.DE`** (iShares Nasdaq 100 UCITS ETF EUR, T212 ticker `SXRVd_EQ`).
   - Energy Sector: **`QDVF.DE`** (iShares S&P 500 Energy Sector UCITS ETF EUR Acc, T212 ticker `QDVFd_EQ`).
3. **Live Reconciled Pricing**: Executable prices are queried live via the Trading 212 position/market API (<0.5s), guarding against stale or unrepresentative closes.

---

### The Streamlined Model Stack (Complexity Reduction)
Following an extensive quant audit based on the rule *"In trading, complexity doesn't pay"*, the system eliminated underperforming "zombie" models (PPO reinforcement learning trained on noisy samples, discretized HMMs, dead macro formulas) to focus capital allocations on **proven, high-conviction decision engines**:

| Model Engine | Technology | Base Weight | Role & Edge |
|---|---|:---:|---|
| **TimesFM 3.0** | Google Foundation Model (`timesfm3`) | **25%** | Deep zero-shot time series autoregressive forecasting (median + 9 quantiles). |
| **Classic Quant Ensemble** | Scikit-Learn (RF, GB, Logistic) | **20%** | Multi-feature technical and cross-asset momentum estimation. |
| **Grebenkov Model** | Mathematical Trend Parity | **20%** | Agnostic risk parity and trend persistence indicator. |
| **Unified Text LLM** | `NexusAI-Client` Cloud Gateway | **15%** | Real-time news analysis, financial filings synthesis, and dynamic macro web search. |
| **Multimodal Vision LLM** | Frontier Cloud Multimodal Models | **10%** | Technical candlestick chart pattern and breakout analysis (`enhanced_trading_chart.png`). |
| **Weekend Council** | Multi-Provider 6-Persona Deliberation | **10%** | Weekly strategic retrospective with 7-day linear decay (11th weighted vote). |
| **Oil-Bench Model** | EIA Fundamental Integration | *Dynamic (10%)* | Energy-specialized physical supply/demand model (Crude stocks, imports, refinery utilization). |

> **Quarantined Zombie Models**: `tensortrade` (PPO RL), `hmm_model`, `vincent_ganne`, and legacy `sentiment` are quarantined at **0.0 base weight** with thread execution bypassed to preserve CPU and eliminate decision noise.

---

### Energy Pivot: From Synthetic ETC to Physical Sector ETF
Historically, oil exposure was attempted via commodity futures ETCs (`CRUDP.PA` / `OD7Fd_EQ`). Empirical walk-forward analysis demonstrated fatal structural defects in futures-based ETCs:
- **The Contango Roll Decay Trap**: Continual negative roll yield causes commodity ETCs to decay structurally over time (**-10.8% CAGR** with MA200 hysteresis and **-68.7% Max Drawdown**).
- **Feed Freeze**: European ETC feeds suffer severe data gaps (81.9% feed freezes on `CRUDP.PA`).

**The Solution — `QDVF.DE` (iShares S&P 500 Energy Sector UCITS ETF EUR)**:
- **Physical Equity Basket**: Backed by top US energy producers (ExxonMobil, Chevron, ConocoPhillips, EOG).
- **Zero Roll Decay**: Generates **+15.4% Buy & Hold CAGR** over 11 years (2,732 bars, 97.6% clean data).
- **Positive Real Dividends**: Physical cash generation (~3.5% distribution yield reinvested).
- **Direct Correlation**: Strong beta to crude oil price shocks while capturing equity value.

---

## 🛡️ Production Invariants & Safety GO-Gates

To ensure institutional safety and eliminate catastrophic trading anomalies, the system enforces **7 strict, non-negotiable GO-Gates**:

```
+-------------------------------------------------------------------------------+
|                             PRODUCTION GO-GATES                               |
+---+----------------------------+----------------------------------------------+
| 1 | Idempotent Order Posting   | Market BUYs POST bare payloads with a 15s    |
|   |                            | timeout. Reconciliation occurs before retry. |
+---+----------------------------+----------------------------------------------+
| 2 | Broker Stop-Loss Ratchet   | Every open position has a dedicated GTC stop |
|   |                            | placed at broker (peak x 0.90), ratchet UP.  |
+---+----------------------------+----------------------------------------------+
| 3 | Observed Fill Confirmation | State and database writes only happen after  |
|   |                            | confirmed fill (/equity/portfolio/positions).|
+---+----------------------------+----------------------------------------------+
| 4 | Daily Volatility Standard  | Volatility is strictly daily std (never      |
|   |                            | annualized) to match decision thresholds.    |
+---+----------------------------+----------------------------------------------+
| 5 | Synthetic Macro Banned     | No synthetic data generation. Stale price    |
|   |                            | caches (>3 days) are rejected at source.     |
+---+----------------------------+----------------------------------------------+
| 6 | Single-Instance Scheduler  | Enforced by atomic file lock (scheduler.lock)|
|   |                            | with PID monitoring and lock-keeper thread.  |
+---+----------------------------+----------------------------------------------+
| 7 | True FIFO Equity           | Equity = Initial Budget + Realized (FIFO) +  |
|   |                            | Unrealized, tracked in trading_journal.csv.  |
+---+----------------------------+----------------------------------------------+
```

- **Order Anti-Churn**: 4-hour minimum holding time blocks intraday flip-flop noise (`MIN_HOLDING_HOURS = 4`), while emergency stops bypass this immediately.
- **Selling Guard vs Broker Reservations**: A standing stop reserves shares; the executor cancels the stop first before selling, and re-places it if the sale fails.

---

## 🧠 The High-Conviction Model Ensemble

### 1. Google TimesFM 3.0 (Time-Series Foundation Model)
- Directly integrated via PyPI package `timesfm>=3.0.1` (package name `timesfm3`).
- Loads official PyTorch checkpoint `google/timesfm-3.0-pytorch` (~1.3 GB, cached locally in HF cache).
- Performs 2048-token context window autoregressive inference on CPU (~0.35s per forecast).
- Median forecast drives the primary trend signal with 9 quantile boundaries exported to telemetry.

### 2. Unified Cloud LLM Gateway via NexusAI-Client
- Powered by **[`NexusAI-Client`](https://github.com/laurentvv/NexusAI-Client)**: zero local LLM footprint (no Ollama, no heavy GGUF downloads).
- Resilient zero-cost automatic failover across 9+ cloud providers:
  - **Gemini Free / Gemini Pro** (Google)
  - **Groq & Cerebras** (Ultra-fast LPU inference)
  - **Mistral AI & Cohere**
  - **Nvidia NIM & OpenRouter / OrcaRouter**
- **Dual-Layer JSON Defence**: Guarantees parseable, schema-compliant `{signal, confidence, analysis}` dictionaries even under high model creativity.

### 3. Weekend AI Strategic Council (11th Weighted Vote)
- Runs autonomously every weekend (`src/council/weekend_council.py`).
- Convenes 6 distinct personas (Macro Strategist, Risk Manager, Quantitative Scientist, Bearish Skeptic, Market Tactician, Behavioral Analyst).
- Each persona queries a **different cloud provider** to guarantee cognitive and structural diversity.
- Three deliberation rounds culminate in a per-ticker stance that feeds the real-time consensus engine as a weighted vote decaying linearly over 7 days.

---

## 🧪 Vectorized Backtesting Engine

The system includes a dedicated pure NumPy / pandas vectorized backtesting suite located in [`src/backtest/`](file:///C:/GIT/Trading-AI/src/backtest):
- **High-Fidelity Replay**: Simulates realistic Trading 212 execution with 0.1% transaction friction.
- **Walk-Forward Baselines**: Evaluates standard trend-following benchmarks (Buy & Hold, MA50, MA200, MA200 with 1.5% hysteresis, EMA20/50, RSI28/60, MACD, Bollinger Breakouts).
- **Candidate Evaluation**: Automated reports comparing candidate instruments across Sharpe, CAGR, Max Drawdown, Win Rate, Profit Factor, and Calmar ratio.

To generate a comparative benchmark:
```bash
uv run python -m src.backtest.run_benchmark
```

---

## 📂 Project Structure

```
Trading-AI/
├── src/                             # Core production codebase
│   ├── adaptive_weight_manager.py   # Bayesian/win-rate dynamic weight adjustment
│   ├── advanced_risk_manager.py     # Trend-aware sizing and stop-loss logic
│   ├── backtest/                    # Vectorized backtest & walk-forward engine
│   │   ├── engine.py                # Pure vector backtester with costs & hysteresis
│   │   ├── data.py                  # Clean data loader and frozen-bar maskers
│   │   ├── metrics.py               # CAGR, Sharpe, Sortino, MaxDD calculations
│   │   └── report.py                # Automated Markdown report generator
│   ├── chart_generator.py           # Candlestick chart renderer for Vision LLM
│   ├── classic_model.py             # RandomForest / GradientBoosting ensemble
│   ├── config_weights.py            # Centralized model weights & quarantine status
│   ├── data.py                      # Data ingestion, caching, and freshness gates
│   ├── database.py                  # SQLite persistence for transactions & telemetry
│   ├── eia_client.py                # EIA API v2 client for energy fundamentals
│   ├── enhanced_decision_engine.py  # Multi-model consensus and quorum validator
│   ├── enhanced_trading_example.py  # Pipeline orchestrator and parallel worker pool
│   ├── features.py                  # Technical indicators and feature engineering
│   ├── grebenkov_model.py           # Agnostic risk parity trend model
│   ├── llm_client.py                # NexusAI-Client gateway wrapper (Text & Vision)
│   ├── news_fetcher.py              # Real-time financial news crawler
│   ├── oil_bench_model.py           # Energy-specific fundamental EIA model
│   ├── performance_monitor.py       # P&L tracking and win-rate calculation
│   ├── t212_executor.py             # Trading 212 execution, stop ratchet & FIFO P&L
│   ├── timesfm_model.py             # TimesFM 3.0 foundation model wrapper
│   ├── web_researcher.py            # Automated web search query engine
│   └── council/                     # Weekend Strategic Council multi-agent suite
│       ├── weekend_council.py       # 3-round multi-provider debate orchestrator
│       └── council_prompts.py       # Personas and prompt templates
├── morning_brief/                   # Autonomous overnight fundamental synthesis
│   └── morning_brief.py             # Overnight brief generator
├── memory-bank/                     # Deterministic state management
│   ├── feature_list.json            # Complete feature lifecycle registry
│   ├── contract.md                  # Testable technical validation contract
│   ├── progress.md                  # Current sprint dashboard
│   └── log.md                       # Append-only chronological event journal
├── tests/                           # 430+ unit, integration, and safety tests
├── main.py                          # Single-cycle pipeline CLI entry point
├── schedule.py                      # Production continuous scheduler
└── scheduler_config.json            # Centralized runtime configuration
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.12+ installed
- Fast virtualenv and package management via [`uv`](https://astral.sh/uv)
- Trading 212 API credentials (demo or live)
- Cloud LLM API keys configured in `.env` (Gemini, Groq, Mistral, Nvidia NIM, etc.)

### Installation
```powershell
# 1. Clone the repository
git clone https://github.com/laurentvv/Trading-AI.git
Set-Location Trading-AI

# 2. Synchronize virtual environment with uv
uv sync

# 3. Install browser dependencies for web research
uv run python -m playwright install chromium
```

### Pre-warming Foundation Checkpoints
Before launching the production scheduler, download and cache the Google TimesFM 3.0 model (~1.3 GB, cached in HuggingFace cache):
```powershell
uv run python tests/smoke_timesfm3.py
```

---

## 🛠️ Operational Execution

### Paper Trading Simulation
Run a single analytical cycle in simulation mode (virtual capital of 1,000€ per ticker, no live broker orders, DB writes enabled):
```powershell
# Default tickers (QDVF.DE and SXRV.DE)
uv run main.py --simul

# Single specific ticker
uv run main.py --simul --ticker QDVF.DE
```

### Live / Demo Broker Execution (Trading 212)
Execute live orders through the Trading 212 API (environment configured via `T212_ENV=demo` or `live` in `.env.t212`):
```powershell
uv run main.py --t212
```

### Continuous Automated Scheduler
Launch the supervised production scheduler (runs every 30 minutes from 08:30 to 18:00 CET, manages atomic lock, morning brief catch-up, and weekend council triggers):
```powershell
# Direct execution
uv run schedule.py

# Supervised loop with automatic restart on crash
.\start_scheduler.bat
```

### Weekend Strategic Council Deliberation
Trigger the multi-provider 6-persona debate on demand:
```powershell
uv run python -m src.council.weekend_council --days 7
```

### Autonomous Morning Brief
Generate the overnight synthesis of news, fundamental EIA releases, and macro developments:
```powershell
uv run python morning_brief/morning_brief.py
```

---

## 🧪 Validation & Test Suite

Run the full test suite (over 430 assertions covering order safety, broker stop ratchets, FIFO accounting, model consensus, and data freshness):

```powershell
# Full mocked test suite
.venv\Scripts\python.exe -m pytest tests/ -q --basetemp=data_cache/test_tmp

# Specific broker order safety and equity tracking tests
.venv\Scripts\python.exe -m pytest tests/test_t212_orders.py tests/test_equity_tracking.py tests/test_data_safety.py -v
```

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.
Google TimesFM 3.0 model weights are subject to the `timesfm-non-commercial-license-v1.0`.
