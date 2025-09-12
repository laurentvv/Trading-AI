# Technical Context

## 1. Core Technologies
- **Language**: Python 3.10+
- **AI / Machine Learning**:
    - **Scikit-learn**: For building the traditional quantitative trading model (e.g., RandomForest, GradientBoosting), now enhanced with macroeconomic features.
    - **Pandas & NumPy**: For data manipulation and numerical operations.
    - **Ollama**: For serving the local Large Language Model (LLM). The system will interact with Ollama via its REST API.
- **Data Fetching**:
    - **yfinance**: To download historical market data from Yahoo Finance.
    - **pandas-datareader**: To fetch macroeconomic data from FRED.
- **Data Storage**:
    - **pyarrow (Parquet)**: For efficiently storing cached kline data and cached macroeconomic data.
- **Visualization**:
    - **Matplotlib & Seaborn**: For generating plots of backtesting results.
    - **mplfinance**: For generating professional financial chart images for the visual AI.

## 2. Development Environment
- The project is managed using standard Python tooling.
- Dependencies are listed in `requirements.txt` and should be installed into a virtual environment.

## 3. Technical Constraints & Assumptions
- **Ollama Service**: The system assumes that an Ollama service is running locally and is accessible at `http://localhost:11434`. It must provide an LLM capable of both text and visual analysis (e.g., `gemma3:27b`).
- **Internet Connection**: An internet connection is required for the initial data download (or when the cache is empty/stale).
- **No GUI**: This is a command-line application. All output is text-based or saved as image files.

## 4. Scheduler Configuration
The `intelligent_scheduler.py` can be configured by placing a `scheduler_config.json` file in the project root. This file allows for overriding the default scheduler settings.

- **Purpose**: To provide user-configurable parameters for the scheduler's operation, especially for managing phase transitions.
- **Format**: JSON
- **Key Parameters**:
    - `project_start_date`: An ISO 8601 formatted string representing the project's start date, which is used to calculate phase durations.
    - `phase_transitions`: An object containing the duration in days for each phase (`phase_1_duration_days`, `phase_2_duration_days`, etc.).

If this file is not present, the scheduler will use a default configuration.
