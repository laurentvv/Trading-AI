# Active Context

## Current Status
The project is now in a **high-fidelity production/demo phase** with an **additive institutional backtesting layer**. The decision engine has been refined for extreme accuracy, and the web research capabilities have been significantly upgraded. The system is operating under an "Accuracy First" (Justesse) mandate. A QuantConnect Lean integration provides independent backtesting validation without touching the production pipeline.

### Key Recent Changes
- **QuantConnect Lean Integration (2026-05-04)**: Added an additive backtesting layer based on QuantConnect Lean (Docker-based). The `LeanSignalBridge` (`src/lean_bridge.py`) converts `trading_journal.csv` into Lean-compatible signals. The `LeanValidator` (`src/lean_validator.py`) provides CI/CD-style validation. 5 Alpha Models (Classic, TimesFM, Sentiment, RiskMomentum, VincentGanne) with the same weights as the `EnhancedDecisionEngine`. T212 fee model (0.1%) and volume-share slippage included. Zero impact on production code.
- **T212 API Resilience Fix**: Fixed `KeyError: 'averagePrice'` crash in `t212_executor.py` when the Trading 212 positions API omits the `averagePrice` field. Now uses defensive `.get()` with fallback calculation.
- **Diagnostic Scripts**: Moved `check_cache.py`, `check_db.py`, `check_live.py` from `logs_prod/` to `tests/` with relative paths for reusability.
- **TensorTrade / PPO Integration**: Added a 9th signal — a Reinforcement Learning agent (PPO via stable-baselines3, Gymnasium environment) that learns buy/sell/hold policies from price history. Weight: 10% in the decision engine.
- **Cache Auto-Invalidation**: Parquet cache files now auto-detect staleness — if `last_date` is > 2 days old, a force refresh is triggered automatically (`src/data.py` lines 148-154).
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
- [x] Implement cache auto-invalidation (stale > 2 days).
- [x] Implement MA50 fallback for insufficient MA200 data.
- [ ] Monitor real-time performance in Demo Mode.
- [ ] Add automated Stop-Loss rules in AdvancedRiskManager.
- [ ] Synchronize i18n translations (9 languages) with README.md updates.
- [ ] Inject real journal signals into Lean backtests (via insights.json).
- [ ] Optimize model weights via Lean Optimizer (grid search).

## Decision Log
- **Nasdaq Exclusivity for VG**: Decided to restrict the Vincent Ganne model to Nasdaq because its energy-price-to-stock-bottom logic is fundamentally a cross-asset indicator for equities, not a directional signal for energy itself.
- **BUY-Only Signal for VG**: Restricted VG model to BUY signals to prevent it from forcing exits on Nasdaq when energy prices spike, which might be noise rather than a structural top.
- **Dynamic Search Queries**: Improved query generation by adding the current date to search queries, ensuring the LLM finds recent news instead of outdated macro reports.
