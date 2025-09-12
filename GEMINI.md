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

**2. Automated Analysis with the Intelligent Scheduler**

The project includes an intelligent scheduler to manage the entire Trading AI deployment timeline, including automated daily analysis, reporting, and phase management.

For Windows, the easiest way to run the scheduler is to use the provided batch script:
```bash
start_scheduler.bat
```

Alternatively, you can run the Python script directly:
```bash
python src/intelligent_scheduler.py
```

The scheduler will run in the background, perform daily analysis, generate reports, and manage the project's deployment phases. All scheduler activities are logged in `scheduler.log`.

**3. Forcing Immediate Analysis**

In cases where the scheduler does not start or when an immediate analysis is required, you can use the `run_now.py` script. This script directly triggers the daily analysis task without waiting for the scheduled time.

```bash
python run_now.py
```
This is useful for debugging or manual intervention.

## Configuration

The behavior of the Intelligent Scheduler can be customized via a `scheduler_config.json` file placed in the root directory of the project. If this file is not present, the scheduler will use a default configuration.

Creating this file allows you to control parameters such as phase durations, performance targets, and the project's start date.

### Example `scheduler_config.json`

```json
{
    "project_start_date": "2025-08-25T18:05:27.149745",
    "trading_ticker": "QQQ",
    "daily_execution_time": "18:00",
    "weekly_report_day": "friday",
    "monthly_report_day": 28,
    "phase_transitions": {
        "phase_1_duration_days": 7,
        "phase_2_duration_days": 21,
        "phase_3_duration_days": 30,
        "phase_4_duration_days": 120
    },
    "performance_targets": {
        "phase_2": {
            "sharpe_ratio": 0.5,
            "max_drawdown": 0.05,
            "win_rate": 0.45
        },
        "phase_3": {
            "sharpe_ratio": 1.0,
            "max_drawdown": 0.03,
            "win_rate": 0.55
        },
        "phase_4": {
            "sharpe_ratio": 1.5,
            "max_drawdown": 0.02,
            "win_rate": 0.60
        }
    },
    "alerts": {
        "email_notifications": false,
        "performance_alerts": true,
        "phase_completion_alerts": true
    }
}
```

### Key Parameters

*   `project_start_date`: The official start date of the project. This is crucial for calculating phase transitions.
*   `phase_transitions`: Allows you to define the duration (in days) for each of the four project phases. This is the primary way to control the automatic transition between phases.

## Development Conventions

*   **Modularity:** The codebase is organized into a clean, modular structure with a clear separation of concerns.
*   **Logging:** The project uses the `logging` module for informative output.
*   **Data Caching:** Market data is cached locally in Parquet files to speed up subsequent runs.
*   **Documentation-Driven Development:** The `memory-bank/` directory contains documentation about the project's evolution, architecture, and context.
*   **Configuration:** Key parameters and constants are defined at the beginning of the scripts.
*   **Error Handling:** The code includes error handling for API requests and file operations.
