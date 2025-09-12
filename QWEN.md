# QWEN.md - Trading-AI Project Context

## Project Overview

This is a sophisticated AI-powered trading decision system for NASDAQ ETFs (specifically QQQ), featuring a hybrid tri-modal approach that combines:
1. A quantitative classic model (scikit-learn) trained on technical indicators and macroeconomic data
2. A Large Language Model (LLM) for textual market analysis
3. A Visual LLM (V-LLM) for chart pattern analysis

The system generates consensus-based trading decisions (BUY/SELL/HOLD) with confidence scores, incorporating advanced risk management, adaptive weighting, and real-time performance monitoring.

## System Architecture

### Core Components

1. **Data Layer** (`src/data.py`)
   - Retrieves ETF and VIX data using yfinance with local caching
   - Fetches macroeconomic data from multiple sources (FRED, Alpha Vantage) with fallbacks
   - Provides robust data caching system in `data_cache/` directory

2. **Feature Engineering** (`src/features.py`)
   - Creates technical indicators (RSI, MACD, Bollinger Bands, Moving Averages, etc.)
   - Integrates macroeconomic features (treasury yields, CPI, unemployment, GDP)
   - Generates target variables for model training

3. **Classic Model** (`src/classic_model.py`)
   - Trains ensemble models (RandomForest, GradientBoosting, LogisticRegression)
   - Uses cross-validation for model selection
   - Provides prediction with confidence scores

4. **LLM Integration** (`src/llm_client.py`)
   - Interfaces with Ollama for both text and visual LLM analysis
   - Queries Gemma3:27b model for market data analysis
   - Processes chart images for visual pattern recognition

5. **Chart Generation** (`src/chart_generator.py`)
   - Creates financial charts with technical indicators using mplfinance
   - Generates 6-month candlestick charts with MA, RSI, MACD

6. **Hybrid Decision Engine** (`src/enhanced_decision_engine.py`)
   - Combines predictions from all models with adaptive weighting
   - Calculates consensus scores and disagreement factors
   - Applies market regime adjustments

7. **Advanced Risk Management** (`src/advanced_risk_manager.py`)
   - Calculates comprehensive risk metrics (volatility, drawdown, correlation, liquidity)
   - Provides position sizing recommendations using Kelly Criterion
   - Adjusts signals based on risk levels

8. **Adaptive Weight Management** (`src/adaptive_weight_manager.py`)
   - Dynamically adjusts model weights based on recent performance
   - Detects market regimes and applies appropriate weight adjustments
   - Tracks model performance in SQLite database

9. **Backtesting** (`src/backtest.py`)
   - Walk-forward backtesting with transaction costs
   - Performance metrics calculation (Sharpe ratio, max drawdown, win rate)
   - Logs transactions and portfolio states to database

10. **XAI Explainer** (`src/xai_explainer.py`)
    - Uses SHAP to explain classic model predictions
    - Generates waterfall plots for feature importance visualization

11. **Database Layer** (`src/database.py`)
    - SQLite database for storing transactions, portfolio history, and model signals
    - Provides functions for inserting and retrieving trading data

12. **Performance Monitoring** (`src/performance_monitor.py`)
    - Real-time performance tracking with alerts
    - Generates performance dashboards and reports
    - Monitors model accuracy and risk metrics

### Automation & Scheduling

1. **Intelligent Scheduler** (`src/intelligent_scheduler.py`)
   - Automated daily analysis execution
   - Weekly and monthly performance reporting
   - Phase-based deployment management (4-phase implementation plan)
   - Performance alerts and system maintenance

2. **Manual Execution** (`run_now.py`)
   - Immediate execution of analysis without waiting for scheduled time

## Key Features

### Multi-Model Approach
- **Quantitative Model**: Technical and macroeconomic analysis
- **Text LLM**: Contextual market analysis from technical indicators
- **Visual LLM**: Chart pattern recognition and technical analysis
- **Sentiment Analysis**: News sentiment from Alpha Vantage API

### Advanced Risk Management
- Comprehensive risk scoring (volatility, drawdown, correlation, liquidity)
- Dynamic position sizing with Kelly Criterion
- Risk-adjusted signal modification
- Market regime detection and adaptation

### Performance Tracking
- Real-time performance monitoring with alerts
- Database-backed performance history
- Automated reporting and dashboard generation
- Model performance tracking for adaptive weighting

### Data Integration
- Multiple data sources (yfinance, FRED, Alpha Vantage)
- Robust caching system to avoid redundant downloads
- Macro-economic data integration (interest rates, inflation, GDP)

### Explainability
- SHAP-based explanations for classic model decisions
- Waterfall plots for feature importance visualization

## System Workflow

1. **Data Retrieval**: Fetch market data and macroeconomic indicators
2. **Feature Engineering**: Create technical indicators and combine with macro data
3. **Model Training**: Train classic model on historical data
4. **Signal Generation**: 
   - Classic model prediction
   - Text LLM analysis of market data
   - Visual LLM analysis of generated charts
   - Sentiment analysis from news API
5. **Decision Fusion**: Combine all signals with adaptive weighting
6. **Risk Assessment**: Evaluate market risk and adjust position sizing
7. **Performance Tracking**: Log decisions and update performance metrics
8. **Reporting**: Generate visual reports and performance dashboards

## Development Environment

### Technologies Used
- Python 3.10+
- Pandas, NumPy for data processing
- Scikit-learn for machine learning
- YFinance for market data
- Ollama for LLM integration
- SQLite for data storage
- Matplotlib/Seaborn for visualization
- SHAP for explainability
- Rich for enhanced console output

### Dependencies
See `requirements.txt` for complete list:
- pandas, numpy, yfinance, scikit-learn
- matplotlib, seaborn, mplfinance
- pyarrow, requests, tqdm
- beautifulsoup4, python-dotenv, rich
- pandas_datareader, setuptools, shap
- schedule

## Configuration

### Environment Variables
Create a `.env` file with:
```
ALPHA_VANTAGE_API_KEY="your_api_key_here"
```

### Ollama Setup
- Install Ollama locally
- Pull Gemma3:27b model: `ollama pull gemma3:27b`

## Usage

### Manual Analysis
```bash
python src/main.py
```

### Automated Scheduling
```bash
# Windows
start_scheduler.bat

# Or directly
python src/intelligent_scheduler.py
```

### Immediate Execution
```bash
python run_now.py
```

## Project Status

The system is in Phase 3 (Finalization and Documentation) with all core components implemented:
- Hybrid tri-modal decision engine
- Macro-economic data integration
- Advanced risk management
- Performance monitoring and alerts
- XAI explanations with SHAP
- Robust backtesting framework
- Automated scheduler with phase management

## File Structure
```
Trading-AI/
├── src/                     # Source code
│   ├── main.py              # Main entry point
│   ├── intelligent_scheduler.py # Automated scheduler
│   ├── data.py              # Data retrieval and caching
│   ├── features.py          # Feature engineering
│   ├── classic_model.py     # Quantitative model
│   ├── llm_client.py        # LLM integration
│   ├── backtest.py          # Backtesting engine
│   ├── chart_generator.py   # Chart creation
│   ├── enhanced_decision_engine.py # Hybrid decision making
│   ├── advanced_risk_manager.py # Risk management
│   ├── adaptive_weight_manager.py # Model weight adaptation
│   ├── performance_monitor.py # Performance tracking
│   ├── xai_explainer.py     # SHAP explanations
│   ├── database.py          # Database interface
│   ├── sentiment_analysis.py # Sentiment processing
│   └── news_fetcher.py      # News API integration
├── data_cache/              # Cached market data
├── memory-bank/             # Project documentation
├── *.db                     # SQLite databases
├── requirements.txt         # Python dependencies
├── run_now.py               # Manual execution script
├── start_scheduler.bat      # Scheduler launcher (Windows)
└── README.md                # Project documentation
```