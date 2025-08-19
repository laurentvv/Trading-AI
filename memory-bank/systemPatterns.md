# System Patterns

## 1. System Architecture
The application follows a modular, pipeline-based architecture orchestrated by `src/main.py`. The pipeline consists of:

1.  **Data Caching & Loading (`src/data.py`)**: Fetches and caches market data, and now also fetches and caches macroeconomic data from FRED.
2.  **Feature Engineering (`src/features.py`)**: Calculates technical indicators and integrates macroeconomic data as new features.
3.  **Chart Generation (`src/chart_generator.py`)**: Creates chart images for visual analysis.
4.  **Classic Model Prediction (`src/classic_model.py`)**: Trains and runs the quantitative `scikit-learn` model, now leveraging both technical and macroeconomic features.
5.  **LLM Prediction (`src/llm_client.py`)**: Manages communication with both text and visual Ollama LLMs.
6.  **Hybrid Decision Engine (`src/main.py`)**: A central component that takes inputs from all three models (classic, text LLM, visual LLM) to produce a final, consolidated trading decision.
7.  **Backtesting (`src/backtest.py`)**: Simulates the strategy using a walk-forward validation method to evaluate performance without lookahead bias.

## 2. Key Design Patterns
- **Modular Pipeline**: Each logical part of the system is encapsulated in its own module.
- **Tri-Modal Hybrid AI**: The core of the system is a hybrid model pattern that leverages three distinct modes of analysis:
    1.  **Quantitative Model**: A `scikit-learn` classifier for rapid, data-driven signal generation, enhanced with macroeconomic context.
    2.  **Textual LLM**: An LLM that analyzes numerical indicator data and provides a qualitative summary and signal.
    3.  **Visual LLM**: A multi-modal LLM that analyzes a generated chart image for technical patterns.
    This pattern aims to create a robust decision through the consensus of diverse analytical methods.
- **Local Caching (Memory Bank)**: A caching pattern is used for both market data and macroeconomic data to improve performance and reduce reliance on external APIs.
- **Dependency Injection (Implicit)**: The main script creates instances of the various components and passes them to where they are needed, decoupling the components from each other.
