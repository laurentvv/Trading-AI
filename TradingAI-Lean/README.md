# TradingAI-Lean: QuantConnect Lean Backtesting Integration

This directory contains Lean CLI-compatible Python algorithms for backtesting
Trading-AI signals with institutional-grade fill simulation, slippage, and reporting.

## Setup

```bash
# 1. Install Lean CLI
pip install lean

# 2. Login to QuantConnect (optional, for cloud data)
lean login

# 3. Run baseline backtest
lean backtest

# 4. Run framework algorithm
lean backtest --algorithm TradingAIFrameworkAlgorithm
```

## Structure

- `main.py` — Baseline buy-and-hold algorithm with T212 fee model
- `TradingAIFrameworkAlgorithm.py` — Full Alpha/Portfolio/Risk/Execution framework
- `AlphaModels/` — Individual and composite Alpha Models from Trading-AI
- `CustomData/` — Custom data feeds (EIA macro, etc.)
- `Research/` — Jupyter notebooks for analysis

## Trading-AI Signal Bridge

Before running backtests with pre-computed signals, export them:

```bash
# From the Trading-AI root directory:
python -c "from src.lean_bridge import LeanSignalBridge; LeanSignalBridge().export_to_lean_format('../TradingAI-Lean/lean_data')"
```

## Ticker Proxies (US market data)

| Trading-AI Ticker | Lean Proxy | Asset |
|---|---|---|
| SXRV.DE | QQQ | Nasdaq 100 ETF |
| CRUDP.PA | USO | Oil ETF |
| ^NDX | QQQ | Nasdaq Index |
| CL=F | USO | WTI Crude Futures |
