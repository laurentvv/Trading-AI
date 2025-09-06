# GEMINI.md

## Project Overview

This project is a sophisticated trading decision support system for NASDAQ ETFs. It utilizes a tri-modal hybrid AI approach to generate trading signals, combining a traditional quantitative model, a text-based Large Language Model (LLM), and a visual (multi-modal) LLM for a robust and nuanced analysis.

The system is written in Python and leverages a variety of libraries for data analysis, machine learning, and visualization. It is designed to be run from the command line and provides a comprehensive analysis of a given ETF, including a final trading decision.

## Building and Running

### Prerequisites

- Python 3.10+
- Ollama running locally
- A downloaded LLM (e.g., `ollama pull gemma3:27b`)

### Installation

1.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`
    ```

2.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Set up your API Key:**
    Create a file named `.env` in the root directory of the project and add your Alpha Vantage API key to it like this:
    ```
    ALPHA_VANTAGE_API_KEY="YOUR_API_KEY_HERE"
    ```

### Running the System

There are two ways to run the system:

**1. Manual Analysis**

To run a single, on-demand analysis, execute the main script from the root directory:

```bash
python src/main.py
```

The script will:
1.  Fetch or load market data.
2.  Run a walk-forward backtest.
3.  Generate a chart for visual analysis.
4.  Generate a final trading decision based on the three models.
5.  Save a plot of the backtest analysis as `backtest_analysis.png`.

**2. Automated Analysis with the Scheduler**

The project includes a scheduler to run the analysis automatically every day.

To run the scheduler, execute the following command from the root directory:

```bash
python src/scheduler.py
```

The scheduler will run in the background and perform the daily analysis at the configured time (default 18:00). All scheduler activities are logged in `scheduler.log`.

## Development Conventions

*   **Modularity:** The codebase is organized into a clean, modular structure with a clear separation of concerns.
*   **Logging:** The project uses the `logging` module for informative output.
*   **Data Caching:** Market data is cached locally in Parquet files to speed up subsequent runs.
*   **Documentation-Driven Development:** The `memory-bank/` directory contains documentation about the project's evolution, architecture, and context.
*   **Configuration:** Key parameters and constants are defined at the beginning of the scripts.
*   **Error Handling:** The code includes error handling for API requests and file operations.
