# Python Health Report — Trading-AI

Generated on 2026-08-18 15:43 by python-health-audit.

## 1. Executive Summary
- Global grade: C
- Reason: Grade C assigned: 0 Ruff findings, zero E hotspots in application codebase (refactored to Rank C), and average Maintainability Index of 60.12.

## 2. Dead Code
### 2.1 Local — Ruff
*No finding* — 0 errors detected across the codebase (all F821, F401, F541, and F841 resolved).

### 2.2 Global — Vulture
- `vendor/kronos/examples/get_akshare_date_2024-2025_x.py:353`: unused variable `retry_count` (100% confidence)
- `vendor/kronos/examples/get_date_new.py:354`: unused variable `retry_count` (100% confidence)
- `vendor/kronos/examples/prediction_new.py:89`: unused variable `retry_count` (100% confidence)
- `vendor/kronos/examples/prediction_new_GUI.py:17`: unused import `FigureCanvasTkAgg` (90% confidence)
- `vendor/kronos/examples/prediction_new_GUI.py:391`: unused variable `retry_count` (100% confidence)
- `vendor/kronos/finetune/qlib_test.py:16`: unused import `CommonInfrastructure` (90% confidence)
- `vendor/kronos/finetune/qlib_test.py:19`: unused import `flatten_dict` (90% confidence)
- `vendor/kronos/webui/run.py:15`: unused import `flask` (90% confidence)
- `vendor/kronos/webui/run.py:16`: unused import `flask_cors` (90% confidence)
- `vendor/timesfm/src/timesfm/timesfm_2p5/timesfm_2p5_flax.py:190`: unused variable `unused_iter` (100% confidence)
- `vendor/timesfm/v1/src/adapter/dora_layers.py:79`: unused variable `objtype` (100% confidence)
- `vendor/timesfm/v1/src/adapter/lora_layers.py:70`: unused variable `objtype` (100% confidence)
- `vendor/timesfm/v1/src/finetuning/finetuning_example.py:12`: unused import `asdict` (90% confidence)

> ⚠️ Vulture produces false positives by construction (global static
> detection). Verify each entry before removal.

## 3. Complexity Hotspots (Radon)
### Rank D Hotspots (Score 31-40)
- `audit_prod_logs.py:397`: `run_backtest` (Rank D)
- `backtest_prod.py:93`: `run_backtest` (Rank D)
- `reset_for_fresh_test.py:327`: `main` (Rank D)
- `src/data.py:724`: `get_vincent_ganne_indicators` (Rank D)
- `src/enhanced_trading_example.py:260`: `EnhancedTradingSystem.get_model_predictions` (Rank D)
- `src/council/weekend_council.py:302`: `run_council` (Rank D)
- `vendor/kronos/finetune_csv/finetune_base_model.py:239`: `train_model` (Rank D)
- `vendor/kronos/finetune_csv/finetune_tokenizer.py:151`: `train_tokenizer` (Rank D)
- `vendor/timesfm/src/timesfm/utils/xreg_lib.py:210`: `BatchedInContextXRegBase._assert_covariates` (Rank D)
- `vendor/timesfm/timesfm-forecasting/examples/anomaly-detection/detect_anomalies.py:391`: `main` (Rank D)
- `vendor/timesfm/v1/src/timesfm/timesfm_base.py:429`: `TimesFmBase.forecast_with_covariates` (Rank D)

### Rank C Hotspots (Score 21-30)
- `main.py:93`: `_execute_t212_orders` (Rank C, improved from Rank E)
- `src/adaptive_weight_manager.py:578`: `AdaptiveWeightManager.calculate_adaptive_weights` (Rank C, improved from Rank E)
- `audit_prod_logs.py:221`: `audit_parquet` (Rank C)
- `audit_prod_logs.py:585`: `audit_finacumen_tools_proof` (Rank C)
- `reset_for_fresh_test.py:254`: `_wipe_data_cache` (Rank C)
- `schedule.py:215`: `main` (Rank C)
- `morning_brief/tools/analyze_trading_logs.py:4`: `AnalyzeTradingLogsTool` (Rank C)
- `scripts/backtest_ensemble_10y.py:18`: `run_ensemble_backtest` (Rank C)
- `scripts/backtest_hmm_10y.py:13`: `run_benchmark` (Rank C)
- `src/adaptive_weight_manager.py:405`: `AdaptiveWeightManager.calculate_all_models_performance` (Rank C)
- `src/adaptive_weight_manager.py:755`: `AdaptiveWeightManager.resolve_previous_predictions` (Rank C)
- `src/advanced_risk_manager.py:511`: `AdvancedRiskManager.get_risk_adjusted_signal` (Rank C)
- `src/advanced_risk_manager.py:462`: `AdvancedRiskManager.should_override_signal` (Rank C)
- `src/classic_model.py:135`: `train_ensemble_model` (Rank C)
- `src/data.py:156`: `get_etf_data` (Rank C)
- `src/data.py:378`: `get_alpha_vantage_data` (Rank C)
- `src/data.py:471`: `get_macro_data_multi_source` (Rank C)
- `src/data.py:684`: `get_hyperliquid_oil_data` (Rank C)
- `src/data.py:835`: `fetch_macro_data_for_date` (Rank C)
- `src/data.py:347`: `_av_parse_items` (Rank C)
- `src/eia_client.py:299`: `EIAClient.format_for_llm` (Rank C)
- `src/eia_client.py:424`: `EIAClient._make_request` (Rank C)
- `src/eia_client.py:63`: `EIAClient.get_fundamental_context` (Rank C)
- `src/enhanced_decision_engine.py:663`: `EnhancedDecisionEngine.make_enhanced_decision` (Rank C)
- `src/enhanced_decision_engine.py:518`: `EnhancedDecisionEngine._calculate_weighted_score` (Rank C)
- `src/enhanced_decision_engine.py:128`: `VincentGanneModel._evaluate_oil` (Rank C)
- `src/enhanced_decision_engine.py:194`: `VincentGanneModel._evaluate_macro` (Rank C)
- `src/enhanced_trading_example.py:596`: `EnhancedTradingSystem.perform_enhanced_analysis` (Rank C)
- `src/enhanced_trading_example.py:831`: `EnhancedTradingSystem._execute_hypothetical_trade` (Rank C)
- `src/enhanced_trading_example.py:981`: `EnhancedTradingSystem.display_enhanced_results` (Rank C)
- `src/features.py:202`: `select_features` (Rank C)
- `src/grebenkov_model.py:64`: `GrebenkovTrendModel.predict` (Rank C)
- `src/hmm_model.py:76`: `baum_welch` (Rank C)
- `src/llm_client.py:366`: `_find_dict_with_keys` (Rank C)
- `src/llm_client.py:263`: `construct_llm_prompt` (Rank C)
- `src/llm_client.py:431`: `_async_query_nexus` (Rank C)
- `src/news_fetcher.py:42`: `fetch_alpha_vantage_news` (Rank C)
- `src/oil_bench_model.py:100`: `OilBenchModel._construct_prompt` (Rank C)
- `src/performance_monitor.py:709`: `PerformanceMonitor.update_monitoring` (Rank C)
- `src/performance_monitor.py:551`: `PerformanceMonitor._assess_current_risk` (Rank C)
- `src/t212_executor.py:172`: `_validate_and_recalibrate_entry_price` (Rank C)
- `src/t212_executor.py:373`: `load_portfolio_state` (Rank C)
- `src/t212_executor.py:756`: `_execute_buy_order` (Rank C)
- `src/t212_executor.py:987`: `execute_t212_trade` (Rank C)
- `src/t212_executor.py:304`: `sync_state_from_t212` (Rank C)
- `src/t212_executor.py:937`: `_execute_sell_order` (Rank C)
- `src/tensortrade_model.py:176`: `get_tensortrade_prediction` (Rank C)
- `src/timesfm_model.py:107`: `TimesFMModel.predict` (Rank C)
- `src/web_researcher.py:147`: `generate_search_query` (Rank C)
- `src/web_researcher.py:255`: `fetch_and_clean` (Rank C)
- `src/agents/solver.py:21`: `SolverAgent.run_react_loop` (Rank C)
- `src/core/tools.py:69`: `lookup_ohlc` (Rank C)
- `src/core/tools.py:205`: `AnswerConsolidationGate` (Rank C)
- `src/core/tools.py:211`: `AnswerConsolidationGate.verify` (Rank C)

## 4. Code Duplication (Pylint)
- `vendor/kronos/kronos.py` & `vendor/kronos/model/kronos.py` (duplicate model definitions in legacy vendor package)
- `vendor/kronos/examples/prediction_new.py` & `vendor/kronos/examples/prediction_new_GUI.py` (duplicate factor analyzer methods in vendor package)

## 5. Recommended Action Plan
1. **Archive or isolate `vendor/kronos/`**: remove unreferenced Kronos scripts from active search paths to clear remaining third-party duplication and Vulture notices.
2. **Decompose Rank D indicators**: simplify `get_vincent_ganne_indicators` in `src/data.py` into dedicated sub-extractors for each asset class (Oil, Gas, Urea, DXY).
3. **Modularize Council execution**: break down `run_council` in `src/council/weekend_council.py` by isolating synthesis generation and report writing into standalone functions.
