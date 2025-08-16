# System Patterns

## 1. System Architecture
The application will follow a modular, pipeline-based architecture orchestrated by a main script (`src/main.py`). The pipeline consists of the following stages:

1.  **Data Caching & Loading (`src/data.py`)**: Responsible for fetching data from yfinance and managing the local Parquet-based cache.
2.  **Feature Engineering (`src/features.py`)**: Calculates technical indicators and prepares features for the models.
3.  **Classic Model Prediction (`src/classic_model.py`)**: Handles the training and prediction of the scikit-learn based classification model.
4.  **LLM Prediction (`src/llm_client.py`)**: Manages communication with the Ollama service, including prompt construction and response parsing.
5.  **Hybrid Decision Engine (`src/main.py`)**: A central component that takes inputs from both the classic model and the LLM to produce a final, consolidated trading decision.
6.  **Backtesting (`src/backtest.py`)**: Simulates the execution of the trading strategy on historical data to evaluate its performance.

## 2. Key Design Patterns
- **Modular Pipeline**: Each logical part of the system is encapsulated in its own module. This promotes separation of concerns and makes the system easier to maintain and test.
- **Hybrid AI Model**: The core of the system is a hybrid model pattern. It combines a fast, quantitative model (scikit-learn) for baseline signal generation with a powerful, qualitative model (LLM) for deeper analysis and confirmation. This pattern aims to leverage the strengths of both approaches.
- **Local Caching (Memory Bank)**: A caching pattern will be used for market data to improve performance and reduce reliance on the network.
- **Dependency Injection (Implicit)**: The main script will create instances of the various components and pass them to where they are needed, decoupling the components from each other.
