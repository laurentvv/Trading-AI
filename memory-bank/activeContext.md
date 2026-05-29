# Active Context

## Current Status
The project is now in a **high-fidelity production/demo phase** with a **standalone production backtest engine** and an **active adaptive feedback loop**. The decision engine has been refined for extreme accuracy, and the web research capabilities have been significantly upgraded. The system is operating under an "Accuracy First" (Justesse) mandate. `backtest_prod.py` replays actual prod signals against real prices with T212 fees for performance validation.

### Key Recent Changes
- **T212 Live Price Injection (2026-05-15)**: `_inject_t212_live_price()` in `src/data.py` patches the last OHLCV bar of tradeable ETFs (SXRV.DE, SXRV.FRK, CRUDP.PA) with the live T212 price from `/equity/positions`. This ensures the analysis engine works with real-time prices instead of stale EOD Yahoo data. Only affects mapped ETF tickers — indices (^NDX, ^VIX, CL=F) remain untouched.
- **T212 Real Portfolio Sync (2026-05-15)**: `sync_state_from_t212()` in `src/t212_executor.py` rebuilds portfolio state from real T212 data using `/equity/positions` (open positions) and `/equity/history/orders` (realized P&L via FIFO matching). `load_portfolio_state()` now uses T212 as primary source, with local JSON as offline fallback. New helper functions: `get_t212_positions()`, `get_t212_account_summary()`, `get_t212_order_history()`.
- **T212 API Exploration (2026-05-15)**: Exhaustive testing of 12 potential OHLCV/candle endpoints on T212 API v0 — all returned 404. T212 has no historical price data API. Yahoo remains necessary for OHLCV historical data.
- **Adaptive Feedback Loop (2026-05-12)**: After each confirmed T212 SELL, `update_outcomes_for_date()` records actual trade returns in `model_performance.db` for all models that had predictions on the entry date. Uses a single SQLite connection. This closes the loop — the `AdaptiveWeightManager` can now dynamically adjust weights based on real observed performance.
- **Progressive Model Weights (2026-05-12)**: All experimental models (vincent_ganne, oil_bench, tensortrade) now have a test weight of 0.05 instead of 0.0. Base weights are normalized to sum=1.0 at the point of use in `enhanced_decision_engine.py` and `adaptive_weight_manager.py`.
- **Cache Staleness 2→1 day (2026-05-12)**: Parquet cache auto-invalidates after 1 day instead of 2. Cache age logged in fractional days for precise diagnostics.
- **Grokipedia Log Suppression (2026-05-12)**: Added `_GrokipediaFilter` to suppress non-blocking crawl4ai grokipedia errors in logs.
- **T212 Budgets per Ticker (2026-05-12)**: Replaced hardcoded 5000€ default with `INITIAL_BUDGETS` dict (1000€ per ticker: SXRVd_EQ, SXRV_EQ, CRUDl_EQ).
- **T212 Budget Bug Fix (2026-05-13)**: Fixed two bugs in `load_portfolio_state()`: (1) early return when state file missing skipped ticker initialization — now falls through to the `if ticker:` block; (2) buy fallback changed from hardcoded `5000.0` to `DEFAULT_INITIAL_BUDGET` (1000€). Ensures correct 1000€ per-ticker budget even on fresh state file.
- **Production Backtest Engine (2026-05-05)**: Replaced QuantConnect Lean integration with a standalone `backtest_prod.py` that replays actual prod signals from `trading_journal.csv` against real parquet prices with T212 fees (0.1%). No external dependencies (no Docker, no Lean CLI). Compares signal strategy vs buy-and-hold baseline with Sharpe, MaxDD, and alpha metrics. Removed `TradingAI-Lean/`, `src/lean_bridge.py`, `src/lean_validator.py`, `run_lean_backtest.py`.
- **T212 API Resilience Fix**: Fixed `KeyError: 'averagePrice'` crash in `t212_executor.py` when the Trading 212 positions API omits the `averagePrice` field. Now uses defensive `.get()` with fallback calculation.
- **Diagnostic Scripts**: Moved `check_cache.py`, `check_db.py`, `check_live.py` from `logs_prod/` to `tests/` with relative paths for reusability.
- **TensorTrade / PPO Integration**: Added a 9th signal — a Reinforcement Learning agent (PPO via stable-baselines3, Gymnasium environment) that learns buy/sell/hold policies from price history. Weight: 5% (progressive test).
- **Cache Auto-Invalidation**: Parquet cache files now auto-detect staleness — if `last_date` is > **1 day** old, a force refresh is triggered automatically (`src/data.py` lines 148-154). Age logged in fractional days.
- **MA50 Fallback**: When MA200 is NaN (insufficient history, e.g. Urea/UME=F), the system falls back to MA50 for the cross-asset indicators used by the Vincent Ganne model.
- **Cache Utility Script**: `refresh_cache.py` forces refresh of all 4 tickers (`^NDX`, `CL=F`, `SXRV.DE`, `CRUDP.PA`).
- **DB Files Removed from Git**: `performance_monitor.db` and `trading_history.db` are no longer tracked — generated locally only.
- **Vincent Ganne Model Refinement**: Now explicitly exclusive to **Nasdaq** assets for market bottom validation. It only generates `BUY` signals and acts as a geopolitical safety lock (blocking Nasdaq buys if energy prices are > $94). It is disabled for Oil trading to avoid self-referential bias.
- **Crawl4AI Integration**: Replaced simple DuckDuckGo snippets with full-page asynchronous crawling for macro research, providing the LLM with dense, high-quality context.
- **Dynamic Prompt Engineering**: LLM prompts are now ticker-aware, include qualified indicators (e.g., RSI qualificators like 'Overbought'), and incorporate current temporal context (Month/Year) and 5-day price trends for search query generation.
- **Trading 212 Dashboard**: Fixed display bugs and improved the detailed logging in `trading_journal.csv` to track each individual model's contribution (Classic, LLM, TimesFM, VG, Sentiment, TensorTrade).
- **Project Cleanup**: Root directory has been cleaned; all test scripts moved to `tests/`.

## Immediate Objectives
- [x] Integrate Vincent Ganne geopolitical safety rules.
- [x] Implement ticker-specific LLM prompts.
- [x] Enable high-fidelity web crawling with Crawl4AI.
- [x] Validate full cycle on Nasdaq and Oil tickers.
- [x] Implement detailed per-model logging.
- [x] Integrate TensorTrade / PPO RL agent as 9th signal.
- [x] Implement cache auto-invalidation (stale > 1 day).
- [x] Implement MA50 fallback for insufficient MA200 data.
- [x] Implement adaptive feedback loop for model weight adjustment.
- [x] Give progressive test weights to experimental models (0.05 each).
- [x] Remove Kronos model (fully deprecated and cleaned).
- [ ] Monitor real-time performance in Demo Mode.
- [ ] Synchronize i18n translations (9 languages) with README.md updates.
- [ ] Optimize model weights via backtest_prod.py grid search.
- [ ] Set HF_TOKEN on PROD server for TimesFM model download.
- [ ] Explore alternative live price source for ETFs without open T212 positions (SXRV.DE).

## Decision Log
- **Nasdaq Exclusivity for VG**: Decided to restrict the Vincent Ganne model to Nasdaq because its energy-price-to-stock-bottom logic is fundamentally a cross-asset indicator for equities, not a directional signal for energy itself.
- **BUY-Only Signal for VG**: Restricted VG model to BUY signals to prevent it from forcing exits on Nasdaq when energy prices spike, which might be noise rather than a structural top.
- **Dynamic Search Queries**: Improved query generation by adding the current date to search queries, ensuring the LLM finds recent news instead of outdated macro reports.
