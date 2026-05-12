# System Patterns

## 1. System Architecture
The application follows a modular, ensemble-based architecture orchestrated by the root `main.py`. The pipeline consists of:

1.  **Ensemble Engine (`src/enhanced_trading_example.py`)**: The central coordinator that manages the lifecycle of an analysis run.
2.  **Data & Macro Layer (`src/data.py`)**: Fetches market and FRED data with a multi-source fallback strategy and local Parquet caching.
3.  **Advanced Decision Logic (`src/enhanced_decision_engine.py`)**: Uses a consensus-based approach with adaptive weighting to combine signals from:
    - `ClassicModel` (Scikit-Learn)
    - `TimesFM` (Time-series Foundation Model)
    - `TensorTrade / PPO` (Reinforcement Learning via stable-baselines3)
    - `LLM Client` (Textual & Visual analysis via **Gemma 4**, enriched by **Crawl4AI** high-fidelity research)
    - `Sentiment Analysis` (Hybrid Alpha Vantage + AlphaEar)
    - **`Hyperliquid`** (Sentiment Décentralisé : Funding Rate, Open Interest)
    - `VincentGanneModel` (**Nasdaq-exclusive** Geopolitical & Cross-asset validatory BUY signal)
4.  **Risk & Sizing Layer (`src/advanced_risk_manager.py`)**: Implements trend-aware confidence thresholds and progressive sizing logic.
5.  **Persistence Layer (`src/database.py`)**: Manages the SQLite database for simulation state, transaction history, and model performance tracking.
6.  **Monitoring Layer (`src/performance_monitor.py`)**: Generates real-time performance reports and visual dashboards.

## 2. Key Design Patterns
- **Accuracy First (Confidence Filter)**: A strict security pattern where BUY/SELL signals are only executed if global confidence exceeds a 40% threshold. Between 20-40%, signals are automatically downgraded to HOLD to avoid market noise.
- **Unidirectional Macro Validation (VG Model)**: A specialized pattern where the Vincent Ganne model only generates `BUY` or `HOLD` signals. It acts as a safety gate for Nasdaq entries based on energy price stability but never forces a `SELL`.
- **Temporal & Ticker-Aware Prompting**: The LLM Client uses a dynamic prompting strategy that injects the current month/year, the specific ticker name, and 5-day price trends into both search queries and decision prompts to minimize hallucinations and maximize context relevance.
- **High-Fidelity Web Crawling (Crawl4AI)**: Replaces simple search snippets with a full-page asynchronous crawling pattern, allowing the LLM to analyze dense macro reports and deep financial insights.
- **Oil Risk-On Resilience**: A specialized risk management pattern that recognizes Oil as a hedge or "risk-on" asset during volatility. It automatically lowers buy confidence thresholds for energy assets during HIGH/VERY_HIGH risk regimes to capture geopolitical spikes.
- **Cognitive Majority Weighting**: Cognitive models hold 75% of the decision weight, ensuring that qualitative context tempers the 25% weight of the aggressive quantitative Classic model.
- **Physical Supply/Demand Analysis**: Uses the `EIAClient` and `OilBenchModel` to inject real-world physical constraints (US crude inventories, refinery utilization, STEO forecasts) into the energy asset decision process.
- **Sentiment Décentralisé (Blockchain)**: Pattern utilisant les données on-chain (perps Hyperliquid) pour détecter les excès spéculatifs (Funding Rates) comme indicateurs contrariens.
- **Dual-Ticker Analysis**: Decouples the asset being analyzed from the asset being traded. AI models analyze high-fidelity global indices (`^NDX`, `CL=F`) while trades are executed on specific exchange-listed ETFs (`SXRV.DE`, `CRUDP.PA`).
- **Ensemble Hybrid AI**: Leverages the "wisdom of the crowd" by combining traditional ML, specialized foundation models (TimesFM 2.5), and generative AI.
- **Adaptive Weighting**: Dynamically adjusts the influence of each model based on its recent accuracy and the current market regime (volatility).
- **Weight Normalization**: Base weights may intentionally sum ≠ 1.0 (e.g., 1.05 with progressive test allocations). They are normalized at the point of use (`weights = {k: v / total for k, v in weights.items()}`) in both `enhanced_decision_engine.py` and `adaptive_weight_manager.py` to ensure consistent score scales.
- **Feedback Loop**: On T212 SELL confirmation, `update_outcomes_for_date()` updates `model_performance_history` with actual outcomes (return_1d, win/loss) for all models that predicted on the entry date. Uses a single SQLite connection (batch UPDATE). This closes the learning loop — the AdaptiveWeightManager adjusts weights based on real trade results.
- **Super-Consensus Boost**: Increases global confidence when objective quantitative models (Classic + TimesFM) agree on a strong directional signal.
- **Strict Simulation (Paper Trading)**: A state-machine pattern that enforces valid trade sequences (cannot SELL if position is 0) and tracks persistent state across runs.
- **Graceful Degradation**: The system is designed to continue functioning even if a complex component (like TimesFM or a specific API) fails, by falling back to the remaining reliable models.
- **yfinance Circuit Breaker**: Separate failure trackers for `info` (metadata, non-critical) and `download` (data). After 3 consecutive failures, calls are blocked for 120s. Prevents cascading slowdowns when Yahoo Finance is down.
- **T212 Live Price Priority**: ETF prices are first fetched from Trading 212 positions API (real-time EUR, <0.5s). Falls back to yfinance if no position exists. Index prices (`^NDX`, `CL=F`) always use yfinance/cache.
- **Time-Series Integrity**: All training and validation steps use forward-looking-safe methods (`TimeSeriesSplit`, `ffill`) to prevent data leakage.
- **Reinforcement Learning Signal (TensorTrade)**: A PPO agent trains on price history within a custom Gymnasium environment (SimpleTradingEnv) at each run. The learned policy outputs BUY/SELL/HOLD with a confidence derived from the action probability distribution. Adds a non-correlated, behavior-based signal to the ensemble.
- **Stale Cache Auto-Invalidation**: Parquet cache files are checked for staleness at load time. If the last data point is > **1 day** old (`pd.Timestamp.now() - last_date > 1 day`), the cache is bypassed and fresh data is downloaded. Cache age is logged in fractional days. Prevents decisions based on outdated market data in PROD.
- **MA50 Fallback for Insufficient History**: When computing the MA200 cross-asset indicator (used by Vincent Ganne), if the 200-day moving average is NaN due to insufficient history (common for commodities like Urea/UME=F), the system falls back to MA50. Ensures all cross-asset checks have a valid reference point.
- **T212 API Response Resilience**: The Trading 212 positions API may return objects with varying field names (e.g., `averagePrice` may be absent). The `t212_executor.py` uses defensive `.get()` with a fallback calculation (`currentValue / quantity`) to compute the entry price for portfolio state synchronization. This prevents `KeyError` crashes when the API response structure changes.
- **Production Backtest Engine**: `backtest_prod.py` replays actual prod signals from `logs_prod/trading_journal.csv` against real parquet prices from `data_cache/`. Applies T212 fees (0.1%), aggregates intraday signals to daily, and simulates BUY/SELL execution. Compares signal strategy vs buy-and-hold baseline with Sharpe, MaxDD, Win Rate, and Alpha metrics. No external dependencies — runs with existing data and `uv run`.
- **Kronos Sanity Guard (Double Protection)**: A two-layer defense against unrealistic Kronos predictions: (1) at the model level, confidence is clamped to 0.05 if the 5-day average prediction implies > 15% drop; (2) at the decision engine level, Kronos weight is capped to 0.01 if the implied market impact (signal × confidence) exceeds 15%. This prevents hallucinated crashes from distorting the ensemble consensus.
- **Progressive Test Weights**: Experimental models (vincent_ganne, oil_bench, tensortrade, kronos) are given a minimal weight of 0.05 instead of 0.0, allowing them to contribute to decisions while limiting their influence. The system can increase their weight via the adaptive feedback loop as they prove accuracy.
- **T212 Per-Ticker Budget**: Portfolio initialization uses `INITIAL_BUDGETS` dict mapping T212 tickers to budget amounts (default 1000€) instead of a hardcoded 5000€. Enables differentiated capital allocation per asset.
