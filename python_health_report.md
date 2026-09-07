---
audit:
  date: 2026-09-07 22:42
  grade: D
metrics:
  ruff: 0
  vulture: 91
  hotspots_cd: 72
  hotspots_ef: 0
  avg_mi: 55.63
  duplication: 11
  sloc: 12344
  ruff_density: 0.0
  vulture_density: 7.4
history:
  - 2026-08-18: C
  - 2026-09-07: F
---

# Python Health Report — Trading-AI

Generated on 2026-09-07 22:42 by python-health-audit (v2).

## 1. Executive Summary
- Global grade: D
- Reason: Grade D assigned: 0 F and 0 E hotspots (all critical hotspots eliminated), 0 local Ruff errors (density 0.0/kLOC), average MI 55.63, Vulture density 7.4/kLOC, and 11 duplication blocks.
- Since last audit (2026-09-07): F → D; all Rank F and Rank E hotspots eliminated (-1 F, -2 E, hotspots_ef: 3 → 0), local Ruff unused import removed (1 → 0), and full test suite passing (285/285 tests).

## 2. Dead Code
### 2.1 Local — Ruff
None (0 findings).

### 2.2 Global — Vulture
- `.agents/skills/alphaear-news/scripts/database_manager.py:19`: unused attribute `row_factory` (60% confidence)
- `.agents/skills/alphaear-news/scripts/database_manager.py:87`: unused method `get_daily_news` (60% confidence)
- `.agents/skills/alphaear-news/scripts/database_manager.py:106`: unused method `delete_news` (60% confidence)
- `.agents/skills/alphaear-news/scripts/database_manager.py:112`: unused method `update_news_content` (60% confidence)
- `.agents/skills/alphaear-news/scripts/news_tools.py:117`: unused method `fetch_news_content` (60% confidence)
- `.agents/skills/alphaear-news/scripts/news_tools.py:129`: unused method `get_unified_trends` (60% confidence)
- `.agents/skills/alphaear-news/scripts/news_tools.py:163`: unused class `PolymarketTools` (60% confidence)
- `.agents/skills/alphaear-news/scripts/news_tools.py:232`: unused method `get_market_summary` (60% confidence)
- `audit_prod_logs.py:564`: unused variable `traj_fail` (60% confidence)
- `morning_brief/tools/analyze_market_sentiment.py:15`: unused variable `description` (60% confidence)
- `morning_brief/tools/analyze_nasdaq.py:26`: unused variable `description` (60% confidence)
- `morning_brief/tools/analyze_trading_logs.py:6`: unused variable `description` (60% confidence)
- `morning_brief/tools/analyze_wti_market.py:102`: unused variable `description` (60% confidence)
- `morning_brief/tools/audit_portfolio_performance.py:98`: unused variable `description` (60% confidence)
- `morning_brief/tools/base.py:9`: unused variable `description` (60% confidence)
- `src/adaptive_weight_manager.py:302`: unused method `update_prediction_outcome` (60% confidence)
- `src/adaptive_weight_manager.py:368`: unused method `calculate_model_performance` (60% confidence)
- `src/adaptive_weight_manager.py:992`: unused method `get_current_weights` (60% confidence)
- `src/advanced_risk_manager.py:113`: unused attribute `max_drawdown_warning` (60% confidence)
- `src/advanced_risk_manager.py:114`: unused attribute `max_drawdown_critical` (60% confidence)
- `src/advanced_risk_manager.py:259`: unused method `assess_market_regime` (60% confidence)
- `src/classic_model.py:292`: unused function `retrain_if_stale` (60% confidence)
- `src/data.py:17`: unused attribute `retries` (60% confidence)
- `src/data.py:73`: unused function `_yf_ticker_info` (60% confidence)
- `src/database.py:160`: unused function `insert_model_signal` (60% confidence)
- `src/enhanced_decision_engine.py:290`: unused variable `BULLISH_BIAS` (60% confidence)
- `src/enhanced_decision_engine.py:296`: unused variable `CLASSIC_BUY_BONUS_THRESHOLD` (60% confidence)
- `src/enhanced_decision_engine.py:297`: unused variable `TIMESFM_BUY_BONUS_THRESHOLD` (60% confidence)
- `src/enhanced_decision_engine.py:298`: unused variable `QUANT_MODEL_BUY_BONUS` (60% confidence)
- `src/enhanced_decision_engine.py:367`: unused attribute `model_performance_history` (60% confidence)
- `src/enhanced_decision_engine.py:808`: unused method `update_adaptive_thresholds` (60% confidence)
- `src/enhanced_decision_engine.py:829`: unused method `get_model_weights_recommendation` (60% confidence)
- `src/grebenkov_model.py:30`: unused attribute `target_volatility` (60% confidence)
- `src/grebenkov_model.py:32`: unused attribute `_position_type` (60% confidence)
- `src/llm_client.py:17`: unused variable `SCHEMA_TRADING_DECISION` (60% confidence)
- `src/llm_client.py:28`: unused variable `SCHEMA_SEARCH_QUERY` (60% confidence)
- `src/llm_client.py:35`: unused variable `SCHEMA_FINACUMEN_SOLVER` (60% confidence)
- `src/llm_client.py:47`: unused variable `SCHEMA_FINACUMEN_ANNOTATOR` (60% confidence)
- `src/llm_client.py:54`: unused variable `SCHEMA_OIL_ALLOCATION` (60% confidence)
- `src/llm_client.py:102`: unused function `_dump_llm_failure` (60% confidence)
- `src/llm_client.py:141`: unused function `check_ollama_health` (60% confidence)
- `src/llm_client.py:567`: unused function `_query_ollama` (60% confidence)
- `src/news_fetcher.py:166`: unused variable `gn_sentiment` (60% confidence)
- `src/performance_monitor.py:120`: unused attribute `active_alerts` (60% confidence)
- `src/performance_monitor.py:126`: unused attribute `benchmark_history` (60% confidence)
- `src/t212_executor.py:16`: unused variable `STATE_LOCK_TIMEOUT` (60% confidence)
- `src/t212_executor.py:124`: unused variable `TIME_STOP_SOFT_LOSS` (60% confidence)
- `src/tensortrade_model.py:62`: unused attribute `action_space` (60% confidence)
- `src/timesfm_model.py:88`: unused method `update_position` (60% confidence)
- `web_dashboard/app.py:69`: unused function `read_root` (60% confidence)
- `web_dashboard/app.py:83`: unused function `read_performance` (60% confidence)

> ⚠️ Vulture produces false positives by construction (global static
> detection). Verify each entry before removal.

## 3. Complexity Hotspots (Radon)
### Rank F Hotspots (Score > 40)
*None (0 hotspots)*

### Rank E Hotspots (Score 31-40)
*None (0 hotspots)*

### Rank D Hotspots (Score 21-30)
- `audit_prod_logs.py:397`: `run_backtest` (Rank D, 25)
- `backtest_prod.py:93`: `run_backtest` (Rank D, 26)
- `morning_brief/tools/analyze_trading_logs.py:4`: `AnalyzeTradingLogsTool` (Rank D, 25)
- `morning_brief/tools/analyze_trading_logs.py:31`: `AnalyzeTradingLogsTool.forward` (Rank D, 24)
- `scripts/backtest_ensemble_10y.py:18`: `run_ensemble_backtest` (Rank D, 21)
- `src/council/weekend_council.py:338`: `run_council` (Rank D, 25)
- `src/data.py:757`: `get_vincent_ganne_indicators` (Rank D, 23)
- `src/data.py:182`: `get_etf_data` (Rank D, 21)
- `src/enhanced_trading_example.py:296`: `EnhancedTradingSystem.get_model_predictions` (Rank D, 27)
- `src/t212_executor.py:1180`: `_execute_buy_order` (Rank D, 30)
- `src/t212_executor.py:890`: `_ratchet_stop_order` (Rank D, 22)
- `src/t212_executor.py:716`: `post_order_market` (Rank D, 21)
- `src/t212_executor.py:1624`: `execute_t212_trade` (Rank D, 21)

### Rank C Hotspots (Score 11-20)
- `schedule.py:301`: `scheduler_tick` (Rank C, 15)
- `reset_for_fresh_test.py:323`: `_wipe_data_cache` (Rank C, 15)
- `reset_for_fresh_test.py:396`: `_print_reset_preview` (Rank C, 13)
- `reset_for_fresh_test.py:481`: `_execute_full_reset` (Rank C, 11)
- `src/adaptive_weight_manager.py:847`: `AdaptiveWeightManager._extract_returns_and_outcomes` (Rank C, 12)
- `src/adaptive_weight_manager.py:976`: `AdaptiveWeightManager.resolve_previous_predictions` (Rank C, 12)
- `src/adaptive_weight_manager.py:656`: `AdaptiveWeightManager.calculate_adaptive_weights` (Rank C, 19)
- `src/adaptive_weight_manager.py:472`: `AdaptiveWeightManager.calculate_all_models_performance` (Rank C, 16)
- `src/adaptive_weight_manager.py:368`: `AdaptiveWeightManager.calculate_model_performance` (Rank C, 12)
- `src/advanced_risk_manager.py:511`: `AdvancedRiskManager.get_risk_adjusted_signal` (Rank C, 16)
- `src/advanced_risk_manager.py:462`: `AdvancedRiskManager.should_override_signal` (Rank C, 15)
- `src/classic_model.py:135`: `train_ensemble_model` (Rank C, 16)
- `src/data.py:427`: `get_alpha_vantage_data` (Rank C, 18)
- `src/data.py:520`: `get_macro_data_multi_source` (Rank C, 15)
- `src/data.py:717`: `get_hyperliquid_oil_data` (Rank C, 12)
- `src/data.py:864`: `fetch_macro_data_for_date` (Rank C, 12)
- `src/data.py:396`: `_av_parse_items` (Rank C, 11)
- `src/eia_client.py:326`: `EIAClient.format_for_llm` (Rank C, 20)
- `src/eia_client.py:451`: `EIAClient._make_request` (Rank C, 13)
- `src/eia_client.py:68`: `EIAClient.get_fundamental_context` (Rank C, 12)
- `src/eia_client.py:148`: `EIAClient.get_crude_imports` (Rank C, 11)
- `src/enhanced_decision_engine.py:668`: `EnhancedDecisionEngine.make_enhanced_decision` (Rank C, 17)
- `src/enhanced_decision_engine.py:523`: `EnhancedDecisionEngine._calculate_weighted_score` (Rank C, 12)
- `src/enhanced_decision_engine.py:128`: `VincentGanneModel._evaluate_oil` (Rank C, 11)
- `src/enhanced_decision_engine.py:194`: `VincentGanneModel._evaluate_macro` (Rank C, 11)
- `src/enhanced_trading_example.py:635`: `EnhancedTradingSystem.perform_enhanced_analysis` (Rank C, 12)
- `src/enhanced_trading_example.py:878`: `EnhancedTradingSystem._execute_hypothetical_trade` (Rank C, 12)
- `src/enhanced_trading_example.py:1054`: `EnhancedTradingSystem.display_enhanced_results` (Rank C, 12)
- `src/features.py:221`: `select_features` (Rank C, 17)
- `src/features.py:119`: `create_features` (Rank C, 12)
- `src/grebenkov_model.py:64`: `GrebenkovTrendModel.predict` (Rank C, 15)
- `src/hmm_model.py:76`: `baum_welch` (Rank C, 14)
- `src/llm_client.py:366`: `_find_dict_with_keys` (Rank C, 19)
- `src/llm_client.py:263`: `construct_llm_prompt` (Rank C, 17)
- `src/llm_client.py:431`: `_async_query_nexus` (Rank C, 11)
- `src/news_fetcher.py:42`: `fetch_alpha_vantage_news` (Rank C, 12)
- `src/oil_bench_model.py:100`: `OilBenchModel._construct_prompt` (Rank C, 13)
- `src/performance_monitor.py:709`: `PerformanceMonitor.update_monitoring` (Rank C, 18)
- `src/performance_monitor.py:551`: `PerformanceMonitor._assess_current_risk` (Rank C, 11)
- `src/t212_executor.py:193`: `_validate_and_recalibrate_entry_price` (Rank C, 20)
- `src/t212_executor.py:517`: `load_portfolio_state` (Rank C, 19)
- `src/t212_executor.py:776`: `_confirm_fill` (Rank C, 19)
- `src/t212_executor.py:404`: `sync_state_from_t212` (Rank C, 18)
- `src/t212_executor.py:365`: `_fifo_pnl` (Rank C, 17)
- `src/t212_executor.py:1419`: `_reconcile_sell_fill_price` (Rank C, 12)
- `src/t212_executor.py:1453`: `_release_standing_stop_if_reserved` (Rank C, 13)
- `src/t212_executor.py:1568`: `_execute_sell_order` (Rank C, 11)
- `src/tensortrade_model.py:176`: `get_tensortrade_prediction` (Rank C, 15)
- `src/timesfm_model.py:110`: `TimesFMModel.predict` (Rank C, 19)
- `src/web_researcher.py:147`: `generate_search_query` (Rank C, 15)
- `src/web_researcher.py:255`: `fetch_and_clean` (Rank C, 11)
- `src/agents/solver.py:21`: `SolverAgent.run_react_loop` (Rank C, 12)
- `src/core/tools.py:69`: `lookup_ohlc` (Rank C, 18)
- `src/core/tools.py:205`: `AnswerConsolidationGate` (Rank C, 13)
- `src/core/tools.py:211`: `AnswerConsolidationGate.verify` (Rank C, 12)
- `src/council/weekend_council.py:152`: `fetch_portfolio_monitoring` (Rank C, 13)

## 4. Code Duplication (Pylint)
- `audit_prod_logs.py` & `backtest_prod.py`: duplicate backtesting logic and journal loader functions (`run_backtest`, `load_journal`, `aggregate_daily_signals`)
- `clean_phantom_trades.py` & `reset_for_fresh_test.py`: duplicate backup and CLI argument parsing routines (`_move_to_backup`, `--yes`, `--dry-run`)
- `schedule.py` & `schedule_test.py`: duplicate council post-execution verification and rich dashboard generator (`get_dashboard`)
- Triplicate skill scripts across `.agents/skills/alphaear-news/scripts/`, `.kilocode/skills/alphaear-news/scripts/`, and `.qwen/skills/alphaear-news/scripts/` (`database_manager.py`, `news_tools.py`, `content_extractor.py`)

## 5. Recommended Action Plan
1. **Consolidate Code Duplication (Pylint)**: Extract duplicated backtesting utilities shared by `audit_prod_logs.py` and `backtest_prod.py` into a shared module to bring duplicate code blocks under 5.
2. **Harmonize Skills Copies**: Clean up redundant identical script duplicates across `.agents/skills/alphaear-news/`, `.kilocode/skills/alphaear-news/`, and `.qwen/skills/alphaear-news/`.
3. **Continue Modularization of Rank D Functions**: Progressively simplify functions like `_execute_buy_order` (score 30) and `EnhancedTradingSystem.get_model_predictions` (score 27) into modular helpers to elevate the codebase towards Grade C/B.
