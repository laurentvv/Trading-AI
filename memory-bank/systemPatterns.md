# System Patterns

## 1. System Architecture
The application follows a modular, ensemble-based architecture orchestrated by the root `main.py`. The pipeline consists of:

1.  **Ensemble Engine (`src/enhanced_trading_example.py`)**: The central coordinator that manages the lifecycle of an analysis run.
2.  **Data & Macro Layer (`src/data.py`)**: Fetches market and FRED data with a multi-source fallback strategy and local Parquet caching.
3.  **Advanced Decision Logic (`src/enhanced_decision_engine.py`)**: Uses a consensus-based approach with adaptive weighting to combine signals from:
    - `ClassicModel` (Scikit-Learn)
    - `TimesFM` (Time-series Foundation Model)
    - `LLM Client` (Textual & Visual analysis via Gemma 3)
    - `Sentiment Analysis` (Financial news)
4.  **Risk & Sizing Layer (`src/advanced_risk_manager.py`)**: Adjusts the final signal based on market volatility and uses the Kelly Criterion for position sizing.
5.  **Persistence Layer (`src/database.py`)**: Manages the SQLite database for simulation state, transaction history, and model performance tracking.
6.  **Monitoring Layer (`src/performance_monitor.py`)**: Generates real-time performance reports and visual dashboards.

## 2. Key Design Patterns
- **Ensemble Hybrid AI**: Leverages the "wisdom of the crowd" by combining traditional ML, specialized foundation models, and generative AI.
- **Adaptive Weighting**: Dynamically adjusts the influence of each model based on its recent accuracy and the current market regime (volatility).
- **Strict Simulation (Paper Trading)**: A state-machine pattern that enforces valid trade sequences (cannot SELL if position is 0) and tracks persistent state across runs.
- **Graceful Degradation**: The system is designed to continue functioning even if a complex component (like TimesFM or a specific API) fails, by falling back to the remaining reliable models.
- **Time-Series Integrity**: All training and validation steps use forward-looking-safe methods (`TimeSeriesSplit`, `ffill`) to prevent data leakage.
