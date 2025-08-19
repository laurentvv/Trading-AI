# Tri-Modal Hybrid AI Trading System for NASDAQ ETFs

This project is a sophisticated trading decision support system that uses a tri-modal hybrid AI approach to generate trading signals for NASDAQ ETFs. It combines a traditional quantitative model, a text-based Large Language Model (LLM), and a visual (multi-modal) LLM for a uniquely robust and nuanced analysis.

## Key Features

- **Tri-Modal Hybrid AI Engine**: Combines three different AI models for a consensus-based decision:
    1.  A `scikit-learn` classifier trained on quantitative technical indicators and **macroeconomic data** (e.g., interest rates, inflation).
    2.  An LLM that performs analysis on the raw numerical data.
    3.  A multi-modal LLM that performs visual analysis on a generated chart image.
- **Robust Backtesting**: Utilizes a **walk-forward validation** methodology to prevent lookahead bias and provide a realistic assessment of the strategy's historical performance.
- **Transaction Cost Simulation**: The backtester accounts for transaction costs to provide more realistic return calculations.
- **Local Data Caching**: Fetched market data is cached locally in Parquet files to speed up subsequent runs. Macroeconomic data is also cached.
- **Modular Codebase**: The code is organized into a clean, modular structure for easy maintenance and extension.
- **Comprehensive Documentation**: The project's evolution, architecture, and context are meticulously documented in the `memory-bank/` directory, following a structured documentation-driven development process.

## Tech Stack

- **Python 3.10+**
- **Pandas & NumPy**: For data manipulation.
- **Scikit-learn**: For the classic ML model.
- **Ollama**: To serve the local LLM (tested with `gemma3:27b`).
- **yfinance**: For fetching market data.
- **Matplotlib & Seaborn**: For plotting.
- **PyArrow**: For Parquet file handling.
- **Tqdm**: For progress bars.
- **mplfinance**: For generating financial charts.

## Prerequisites

Before you begin, ensure you have the following installed:
- Python 3.10 or higher.
- [Ollama](https://ollama.com/) running locally.
- A downloaded LLM (e.g., Gemma 3): `ollama pull gemma3:27b`

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up your API Key:**
    Create a file named `.env` in the root directory of the project and add your Alpha Vantage API key to it like this:
    ```
    ALPHA_VANTAGE_API_KEY="YOUR_API_KEY_HERE"
    ```

## Usage

To run the trading system, execute the main script from the root directory:

```bash
python src/main.py
```

The script will perform the following actions:
1.  Fetch or load market data from the cache.
2.  Run the walk-forward backtest and print the performance summary to the console.
3.  Generate a chart of the recent market data.
4.  Generate a final trading decision for the most recent data point by combining the outputs from the classic model, a text-based LLM, and a visual LLM's analysis of the chart.
5.  Save a plot of the backtest analysis as `backtest_analysis.png`.

## Project Structure

```
.
├── memory-bank/        # Project documentation (context, progress, decisions)
├── src/                # Source code
│   ├── main.py         # Main orchestrator script
│   ├── data.py         # Data fetching and caching
│   ├── features.py     # Feature engineering
│   ├── classic_model.py # Scikit-learn model training
│   ├── llm_client.py   # Client for text and visual LLMs
│   ├── chart_generator.py # Creates chart images for visual AI
│   └── backtest.py     # Backtesting logic
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

## The Memory Bank

This project follows a "Memory Bank" philosophy. The `memory-bank/` directory is the single source of truth for the project's context, architecture, and progress. It is designed to be a living documentation that allows any developer (or AI assistant) to quickly get up to speed with the project's state.
