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
