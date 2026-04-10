# GEMINI.md

## Project Overview

This project is a sophisticated trading decision support system for NASDAQ ETFs. It utilizes a tri-modal hybrid AI approach to generate trading signals, combining a traditional quantitative model, a text-based Large Language Model (LLM), and a visual (multi-modal) LLM for a robust and nuanced analysis.

The system is written in Python and leverages a variety of libraries for data analysis, machine learning, and visualization. It is designed to be run from the command line and provides a comprehensive analysis of a given ETF, including a final trading decision. The system manages a hypothetical portfolio to simulate the performance of the AI's decisions and provide realistic performance metrics.

## Building and Running

### Prerequisites

- Python 3.10+
- Ollama running locally
- A downloaded LLM (e.g., `ollama pull gemma3:4b`)

### Installation

1.  **Installer `uv` (si ce n'est pas déjà fait) :**
    Consultez [astral.sh/uv](https://astral.sh/uv) pour les instructions d'installation.

2.  **Initialiser l'environnement et installer TimesFM 2.5 :**
    Exécutez la commande suivante qui va synchroniser les dépendances et configurer le modèle de prévision de Google :
    ```bash
    uv run setup
    ```
    Cela va cloner le dépôt TimesFM, appliquer les patchs nécessaires pour l'API 2.5 et l'installer en mode éditable dans votre environnement virtuel.

3.  **Set up your API Key:**
    Create a file named `.env` in the root directory of the project and add your Alpha Vantage API key to it like this:
    ```
    ALPHA_VANTAGE_API_KEY="YOUR_API_KEY_HERE"
    ```

### Running the System

To run a full analysis and get a clear trading signal, use the `main.py` script:

```bash
uv run main.py --ticker SXRV.FRK
```

To run with **automatic execution** on your Trading 212 account (starts with a 1000€ budget):

```bash
uv run main.py --t212
```

The script will:
1.  **Fetch/Load Data**: Retrieves the latest market data for SXRV.FRK (iShares Nasdaq 100 EUR).
2.  **Train Models**: Trains the AI models on the available history.
3.  **Generate Decision**: Executes the hybrid analysis.
4.  **Execute Trade**: If `--t212` is set, it **verifies your real portfolio (cash/positions)**, calculates the exact fractional quantity for a 1000€ budget (or available cash), and places a market order on Trading 212.

Analysis charts are saved as `enhanced_trading_chart.png`.

The scheduler will run in the background, perform daily analysis, generate reports, and manage the project's deployment phases. All scheduler activities are logged in `scheduler.log`.

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