# System Patterns

## 1. System Architecture
The application follows a modular, ensemble-based architecture orchestrated by the root `main.py`. The pipeline consists of:

1.  **Ensemble Engine (`src/enhanced_trading_example.py`)**: The central coordinator that manages the lifecycle of an analysis run.
2.  **Data & Macro Layer (`src/data.py`)**: Fetches market and FRED data with a multi-source fallback strategy and local Parquet caching.
3.  **Advanced Decision Logic (`src/enhanced_decision_engine.py`)**: Uses a consensus-based approach with adaptive weighting to combine signals from:
    - `ClassicModel` (Scikit-Learn)
    - `TimesFM` (Time-series Foundation Model)
    - `LLM Client` (Textual & Visual analysis via Gemma 3)
    - `Sentiment Analysis` (Hybrid Alpha Vantage + AlphaEar)
    - `VincentGanneModel` (Geopolitical & Cross-asset filtering)
4.  **Risk & Sizing Layer (`src/advanced_risk_manager.py`)**: Implements trend-aware confidence thresholds and progressive sizing logic.
5.  **Persistence Layer (`src/database.py`)**: Manages the SQLite database for simulation state, transaction history, and model performance tracking.
6.  **Monitoring Layer (`src/performance_monitor.py`)**: Generates real-time performance reports and visual dashboards.

## 2. Key Design Patterns
- **Accuracy First (Confidence Filter)**: A strict security pattern where BUY/SELL signals are only executed if global confidence exceeds a 40% threshold. Between 20-40%, signals are automatically downgraded to HOLD to avoid market noise.
- **Cognitive Majority Weighting**: Cognitive models (LLM, Vision, Sentiment, TimesFM) hold 75% of the decision weight, ensuring that qualitative context tempers the 25% weight of the aggressive quantitative Classic model.
- **Dual-Ticker Analysis**: Decouples the asset being analyzed from the asset being traded. AI models analyze high-fidelity global indices (`^NDX`, `CL=F`) while trades are executed on specific exchange-listed ETFs (`SXRV.DE`, `CRUDP.PA`).
- **Ensemble Hybrid AI**: Leverages the "wisdom of the crowd" by combining traditional ML, specialized foundation models (TimesFM 2.5), and generative AI.
- **Adaptive Weighting**: Dynamically adjusts the influence of each model based on its recent accuracy and the current market regime (volatility).
- **Super-Consensus Boost**: Increases global confidence when objective quantitative models (Classic + TimesFM) agree on a strong directional signal.
- **Strict Simulation (Paper Trading)**: A state-machine pattern that enforces valid trade sequences (cannot SELL if position is 0) and tracks persistent state across runs.
- **Graceful Degradation**: The system is designed to continue functioning even if a complex component (like TimesFM or a specific API) fails, by falling back to the remaining reliable models.
- **Time-Series Integrity**: All training and validation steps use forward-looking-safe methods (`TimeSeriesSplit`, `ffill`) to prevent data leakage.
