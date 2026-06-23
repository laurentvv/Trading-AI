# Active Context

## Current Status
**PROD restart from scratch (2026-05-29)**. Kronos fully removed, TensorTrade PPO persistence deployed (enriched 10-feature env, DB integration). PROD DBs wiped clean — fresh start with a **1-month validation period until end of June 2026** to confirm all models perform correctly with real T212 trades.

### Validation Period: 2026-05-29 → 2026-06-30
- **Premier lancement validé (2026-05-29 13:48)**: Cycle complet OK pour SXRV.DE + CRUDP.PA en 488 sec. TensorTrade PPO persistance confirmée (2000→2500 timesteps). Tous modèles actifs sans erreur. Zéro référence Kronos.
- Monitoring TensorTrade PPO persistence: first cycle trains (2000 steps), subsequent cycles load + fine-tune (500 steps).
- Monitoring all model signals via `trading_journal.csv` and `model_performance.db`.
- Adaptive feedback loop active — weights will adjust based on real outcomes.
- End-of-June review: evaluate Sharpe, win rate, per-model accuracy, and decide on weight adjustments.

### Key Recent Changes
- **FinAcumen Repaired + Prod Logs Auditor (2026-06-23)**: FinAcumen (`src/finacumen_main.py`) était en `status: timeout` à chaque run prod. Six bugs corrigés dans `src/core/tools.py` (`lookup_ohlc` accepte `str|list`→dict ; indicateurs dérivés rsi/sma/macd ajoutés ; symboles prod mappés ; `pd`/`np` pré-injectés dans le sandbox) et `src/agents/solver.py` (prompt documente la vraie API ; l'observation renvoie `data` quand le LLM oublie `print` ; la branche execute-vs-final-answer se base sur le *contenu* non sur la présence de clé). **Convergence prouvée en live** (Ollama + gemma-4-12b) : CRUDP.PA → HOLD 0.75, SXRV.DE → BUY 0.85, chacun citant des prix réels.
  - **Chaîne de dépendance documentée** : FinAcumen n'est pas dans le consensus temps réel (`enhanced_decision_engine.py`), MAIS il **contribue au morning brief** qui consomme les sorties de `main.py` via fichiers partagés : `main.py` écrit `trading_journal.csv`/`trading.log`/`performance_monitor.db` → `morning_brief/tools/` les lit → `schedule.py` append la section FinAcumen dans `morning_market_brief.md`. Couplage par données, pas par import.
  - Nouveau script `audit_prod_logs.py` : valide **tous** les fichiers de `logs_prod/` (catalogue, intégrité SQLite, fraîcheur parquet/JSON/pkl, état FinAcumen) + backtest corrigé (lit `logs_prod/data_cache/`, pas le cache repo-root périmé). Émet `logs_prod/audit_report.md`.
  - Suite de tests mockés : 20/20 verts (8 tests FinAcumen ajoutés en régression des 6 bugs).
- **Think Mode Re-enabled (2026-06-06)**: Le token `<|think|>` a été ré-introduit dans les 4 prompts système de production (`src/llm_client.py:188` texte, `:236` visuel, `src/oil_bench_model.py:158` oil, `src/web_researcher.py:205` recherche). Validation complète sur branche `think-mode` puis mergée sur `main`. La sécurité est désormais portée par la **défense bi-couche JSON** (schema strict serveur + suffixe défensif dans le prompt), pas par la suppression du token. Preuves : `tests/check_llm_json.py` → tous les cas schema-strict OK avec `<|think|>` actif ; `uv run main.py --t212` → exit 0, 0 erreur "Could not find valid JSON", 2 nouvelles lignes dans `trading_journal.csv` (CRUDP.PA HOLD 19.16%, SXRV.DE BUY 35.18%), cycle 4.66 min. Voir `docs/ADR-001-think-mode-dual-layer-defence.md`.
- **PROD Fresh Start (2026-05-29)**: All PROD DBs wiped (`trading_history.db`, `model_performance.db`). `data_cache/tensortrade/` created fresh on first cycle. Kronos references fully removed from code and docs.
- **TensorTrade PPO Persistence (2026-05-29)**: PPO model now persists to `data_cache/tensortrade/ppo_model.zip` with metadata. Enriched Gymnasium env with 10 features (returns, volatility, RSI, MACD, Bollinger bands). First cycle: 2000 timesteps initial training. Subsequent cycles: load + 500 timestep fine-tune.
- **T212 Live Price Injection (2026-05-15)**: `_inject_t212_live_price()` in `src/data.py` patches the last OHLCV bar of tradeable ETFs (SXRV.DE, SXRV.FRK, CRUDP.PA) with the live T212 price from `/equity/positions`. This ensures the analysis engine works with real-time prices instead of stale EOD Yahoo data. Only affects mapped ETF tickers — indices (^NDX, ^VIX, CL=F) remain untouched.
- **T212 Real Portfolio Sync (2026-05-15)**: `sync_state_from_t212()` in `src/t212_executor.py` rebuilds portfolio state from real T212 data using `/equity/positions` (open positions) and `/equity/history/orders` (realized P&L via FIFO matching). `load_portfolio_state()` now uses T212 as primary source, with local JSON as offline fallback. New helper functions: `get_t212_positions()`, `get_t212_account_summary()`, `get_t212_order_history()`.
- **Adaptive Feedback Loop (2026-05-12)**: After each confirmed T212 SELL, `update_outcomes_for_date()` records actual trade returns in `model_performance.db` for all models that had predictions on the entry date. Uses a single SQLite connection. This closes the loop — the `AdaptiveWeightManager` can now dynamically adjust weights based on real observed performance.
- **Progressive Model Weights (2026-05-12)**: All experimental models (vincent_ganne, oil_bench, tensortrade) now have a test weight of 0.05 instead of 0.0. Base weights are normalized to sum=1.0 at the point of use in `enhanced_decision_engine.py` and `adaptive_weight_manager.py`.
- **Cache Staleness 2→1 day (2026-05-12)**: Parquet cache auto-invalidates after 1 day instead of 2. Cache age logged in fractional days for precise diagnostics.
- **T212 Budgets per Ticker (2026-05-12)**: Replaced hardcoded 5000€ default with `INITIAL_BUDGETS` dict (1000€ per ticker: SXRVd_EQ, SXRV_EQ, CRUDl_EQ).
- **T212 Budget Bug Fix (2026-05-13)**: Fixed two bugs in `load_portfolio_state()`: (1) early return when state file missing skipped ticker initialization — now falls through to the `if ticker:` block; (2) buy fallback changed from hardcoded `5000.0` to `DEFAULT_INITIAL_BUDGET` (1000€). Ensures correct 1000€ per-ticker budget even on fresh state file.
- **Production Backtest Engine (2026-05-05)**: Replaced QuantConnect Lean integration with a standalone `backtest_prod.py` that replays actual prod signals from `trading_journal.csv` against real parquet prices with T212 fees (0.1%). No external dependencies (no Docker, no Lean CLI). Compares signal strategy vs buy-and-hold baseline with Sharpe, MaxDD, and alpha metrics.
- **Vincent Ganne Model Refinement**: Now explicitly exclusive to **Nasdaq** assets for market bottom validation. It only generates `BUY` signals and acts as a geopolitical safety lock (blocking Nasdaq buys if energy prices are > $94). It is disabled for Oil trading to avoid self-referential bias.
- **Crawl4AI Integration**: Replaced simple DuckDuckGo snippets with full-page asynchronous crawling for macro research, providing the LLM with dense, high-quality context.
- **Dynamic Prompt Engineering**: LLM prompts are now ticker-aware, include qualified indicators (e.g., RSI qualificators like 'Overbought'), and incorporate current temporal context (Month/Year) and 5-day price trends for search query generation.

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
- [x] Deploy TensorTrade PPO persistence + DB migration to PROD.
- [x] PROD fresh start — all DBs wiped, clean slate.
- [ ] **[Validation Period]** Monitor PROD performance until end of June 2026.
- [ ] **[End June]** Review model accuracy, Sharpe, win rate — adjust weights if needed.
- [ ] Synchronize i18n translations (9 languages) with README.md updates.
- [ ] Optimize model weights via backtest_prod.py grid search.
- [ ] Set HF_TOKEN on PROD server for TimesFM model download.
- [ ] Explore alternative live price source for ETFs without open T212 positions (SXRV.DE).
- [x] **[DONE 2026-06-23]** Repair FinAcumen convergence (was `status: timeout` every prod run).
- [ ] **[Follow-up]** Wire FinAcumen as an 11th per-cycle consensus vote in `enhanced_decision_engine.py` / `model_performance.db` (currently only feeds the morning brief via `schedule.py`).
- [ ] Fix `backtest_prod.py` stale-source bug (reads repo-root `data_cache/` instead of `logs_prod/data_cache/`); or migrate users to `audit_prod_logs.py`.

## Decision Log
- **Nasdaq Exclusivity for VG**: Decided to restrict the Vincent Ganne model to Nasdaq because its energy-price-to-stock-bottom logic is fundamentally a cross-asset indicator for equities, not a directional signal for energy itself.
- **BUY-Only Signal for VG**: Restricted VG model to BUY signals to prevent it from forcing exits on Nasdaq when energy prices spike, which might be noise rather than a structural top.
- **Dynamic Search Queries**: Improved query generation by adding the current date to search queries, ensuring the LLM finds recent news instead of outdated macro reports.
