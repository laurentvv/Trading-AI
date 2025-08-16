# Hybrid AI Trading System for NASDAQ ETFs

This project is a sophisticated trading decision support system that uses a hybrid AI approach to generate trading signals for NASDAQ ETFs. It combines a traditional quantitative model with a Large Language Model (LLM) for a more robust and nuanced analysis.

## Key Features

- **Hybrid AI Engine**: Combines a `scikit-learn` classifier (trained on technical indicators) with an LLM (powered by Ollama) to generate trading decisions.
- **Robust Backtesting**: Utilizes a **walk-forward validation** methodology to prevent lookahead bias and provide a realistic assessment of the strategy's historical performance.
- **Transaction Cost Simulation**: The backtester accounts for transaction costs to provide more realistic return calculations.
- **Local Data Caching**: Fetched market data is cached locally in Parquet files to speed up subsequent runs.
- **Modular Codebase**: The code is organized into a clean, modular structure for easy maintenance and extension.
- **Comprehensive Documentation**: The project's evolution, architecture, and context are meticulously documented in the `memory-bank/` directory, following a structured documentation-driven development process.

## Tech Stack

- **Python 3.10+**
- **Pandas & NumPy**: For data manipulation.
- **Scikit-learn**: For the classic ML model.
- **Ollama**: To serve the local LLM (tested with `qwen:latest`).
- **yfinance**: For fetching market data.
- **Matplotlib & Seaborn**: For plotting.
- **PyArrow**: For Parquet file handling.
- **Tqdm**: For progress bars.

## Prerequisites

Before you begin, ensure you have the following installed:
- Python 3.10 or higher.
- [Ollama](https://ollama.com/) running locally.
- A downloaded Ollama model, such as Qwen. You can get it by running:
  ```bash
  ollama pull qwen
  ```

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

## Usage

To run the trading system, execute the main script from the root directory:

```bash
python src/main.py
```

The script will perform the following actions:
1.  Fetch or load market data from the cache.
2.  Run the walk-forward backtest and print the performance summary to the console.
3.  Generate a final trading decision for the most recent data point, combining the classic model's prediction with a live call to the Ollama LLM.
4.  Save a plot of the backtest analysis as `backtest_analysis.png`.

## Project Structure

```
.
├── memory-bank/        # Project documentation (context, progress, decisions)
├── src/                # Source code
│   ├── main.py         # Main orchestrator script
│   ├── data.py         # Data fetching and caching
│   ├── features.py     # Feature engineering
│   ├── classic_model.py # Scikit-learn model training
│   ├── llm_client.py   # Ollama LLM client
│   └── backtest.py     # Backtesting logic
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

## The Memory Bank

This project follows a "Memory Bank" philosophy. The `memory-bank/` directory is the single source of truth for the project's context, architecture, and progress. It is designed to be a living documentation that allows any developer (or AI assistant) to quickly get up to speed with the project's state.
