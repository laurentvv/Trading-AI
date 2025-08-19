# Project Brief: AI-Powered Trading System for NASDAQ ETFs

## 1. Core Goal
The primary objective of this project is to develop a sophisticated, AI-driven trading decision system focused on NASDAQ-listed Exchange Traded Funds (ETFs), specifically targeting tickers available on Euronext Paris (e.g., `FR0011871110.PA`).

## 2. Key Requirements
The system must perform the following key functions:
- **Data Ingestion & Caching**: Fetch historical market data (klines) and cache it locally to avoid redundant downloads.
- **Hybrid AI Decision Engine**: Generate trading signals (`BUY`/`SELL`/`HOLD`) using a hybrid approach that combines:
    1. A traditional quantitative model (e.g., `scikit-learn` classifier) trained on technical indicators.
    2. A Large Language Model (LLM) like Gemma 3 (via Ollama) that provides both a direct signal and a qualitative market analysis.
- **Backtesting**: Provide a robust backtesting framework to evaluate the performance of the trading strategy, including metrics like Sharpe ratio, max drawdown, and win rate, and accounting for transaction costs.
- **Modularity and Maintainability**: The codebase must be well-structured, modular, and easy to extend.

## 3. Scope
- **In Scope**:
    - Development of the trading logic and AI integration.
    - Creation of a local data cache ("kline memory bank").
    - Implementation of a backtesting engine.
    - Comprehensive project documentation via the "Memory Bank" system.
    - The system is for decision support, not automated execution.
- **Out of Scope**:
    - Live trading execution and broker integration.
    - A user interface (the system will run as a script).
    - Hosting and deployment of the Ollama model (it is assumed to be running locally).
