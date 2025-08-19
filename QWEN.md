# QWEN.md - AI Trading System for NASDAQ ETFs

## Project Overview

This project is a sophisticated trading decision support system for NASDAQ ETFs (e.g., QQQ). It employs a **tri-modal hybrid AI** approach to generate robust trading signals by combining insights from three distinct models:

1.  **Classic Quantitative Model:** A `scikit-learn` ensemble classifier trained on technical indicators and **macroeconomic data** (e.g., interest rates, inflation, unemployment).
2.  **Text-Based LLM:** An LLM (accessed via Ollama, e.g., Gemma 3) that analyzes numerical market data and provides a textual decision and analysis.
3.  **Visual LLM:** An LLM that performs technical analysis by interpreting a generated financial chart image.

The system integrates these models, runs a robust walk-forward backtest to evaluate historical performance, and provides a final, hybrid decision for the most recent data point. It is designed for decision support, not automated trading.

## Key Technologies

*   **Language:** Python 3.10+
*   **Data & Numerics:** `pandas`, `numpy`, `yfinance` (for data fetching), `pyarrow` (for Parquet caching)
*   **ML Framework:** `scikit-learn`
*   **Visualization:** `matplotlib`, `seaborn`, `mplfinance` (for chart generation)
*   **AI/LLM Interface:** `requests` (to interact with a local Ollama instance)
*   **Utilities:** `python-dotenv` (for environment variables), `tqdm` (progress bars), `rich` (formatted console output)

## Project Structure

```
.
├── memory-bank/             # Comprehensive project documentation (context, progress, decisions)
├── src/                     # Source code
│   ├── main.py              # Main orchestrator script
│   ├── data.py              # Data fetching and caching logic
│   ├── features.py          # Feature engineering (technical indicators)
│   ├── classic_model.py     # Scikit-learn model training and prediction
│   ├── llm_client.py        # Client for interacting with text and visual LLMs via Ollama
│   ├── chart_generator.py   # Generates financial chart images for visual AI analysis
│   ├── backtest.py          # Walk-forward backtesting engine with transaction costs
│   ├── sentiment_analysis.py # Analyzes sentiment from news headlines
│   └── news_fetcher.py      # Fetches recent news headlines for sentiment analysis
├── data_cache/              # Directory for cached market data (Parquet files)
├── requirements.txt         # Python dependencies
├── .env                     # (User-created) File to store sensitive API keys (e.g., ALPHA_VANTAGE_API_KEY)
├── README.md                # Project overview and usage instructions
└── QWEN.md                  # This file (Instructional context for AI)
```

## Building and Running

### Prerequisites

1.  **Python:** Ensure Python 3.10 or higher is installed.
2.  **Ollama:** Install and run [Ollama](https://ollama.com/) locally. Pull a suitable LLM (e.g., `gemma3:27b`):
    ```bash
    ollama pull gemma3:27b
    ```
3.  **API Key:** Obtain an API key from Alpha Vantage for sentiment analysis.

### Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```
2.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    # On Windows
    venv\Scripts\activate
    # On macOS/Linux
    # source venv/bin/activate
    ```
3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Configure API Key:**
    Create a file named `.env` in the project root and add your Alpha Vantage API key:
    ```env
    ALPHA_VANTAGE_API_KEY="YOUR_ACTUAL_API_KEY_HERE"
    ```

### Usage

To execute the full trading system pipeline, run the main script from the project root directory:

```bash
python src/main.py
```

**What `src/main.py` does:**

1.  **Fetches Data:** Loads or downloads historical market data for QQQ (and VIX) using `yfinance`, caching it locally in `data_cache/`.
2.  **Engineers Features:** Calculates technical indicators (RSI, MACD, Bollinger Bands, Moving Averages, etc.).
3.  **Backtests:** Runs a walk-forward validation backtest using the classic model's signals, simulating the LLM's behavior for performance evaluation. Prints metrics like Sharpe Ratio and Drawdown.
4.  **Generates Chart:** Creates a `trading_chart.png` image of the last 6 months of data (candlesticks, MAs, RSI, MACD) for the visual LLM.
5.  **Makes Final Decision:**
    *   Trains the final classic model on all available data.
    *   Queries the **text LLM** with the latest numerical indicators.
    *   Queries the **visual LLM** with the `trading_chart.png`.
    *   Fetches news sentiment via `news_fetcher.py` and analyzes it.
    *   Combines the outputs of all four models into a final hybrid decision using weighted scoring.
6.  **Displays Results:** Shows a formatted output in the console detailing each model's decision and the final hybrid signal ("STRONG BUY", "BUY", "HOLD", "SELL", "STRONG SELL") along with a reliability score.
7.  **Saves Analysis:** Generates a `backtest_analysis.png` plot showing backtested performance vs. buy-and-hold.

## Development Conventions

*   **Modularity:** Code is organized into distinct modules (`src/`) for data, features, models, LLM interaction, charting, and backtesting, promoting maintainability and clarity.
*   **Documentation:** The project uses a "Memory Bank" system (`memory-bank/`) to store evolving context, architecture decisions, and progress. This is the primary source of truth for understanding the project's design.
*   **Configuration:** API keys and other secrets are managed via a `.env` file, not hardcoded.
*   **Data Caching:** Market data is cached as Parquet files to improve performance and reduce redundant API calls.
*   **Robust Backtesting:** A walk-forward validation approach is used to simulate realistic trading conditions and avoid lookahead bias.
*   **Logging:** Uses Python's `logging` module for informative console output.
*   **Formatted Output:** Uses `rich` to provide clear, structured, and colorized output for the final decision.