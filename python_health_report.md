# Python Health Report — trading-ai

Generated on 2026-06-04 18:43 by python-health-audit.

## 1. Executive Summary
- Global grade: C
- Reason: Grade C assigned: Ruff is clean (0 finding) and average MI is 52.4 (≥ 50), satisfying the C-tier criteria despite 4 E/F hotspots (1 F + 3 E) and 6 Pylint duplications.

## 2. Dead Code
### 2.1 Local — Ruff

No finding — `ruff check . --exclude "venv,.venv,.venv_uv,tests"` returned "All checks passed!".

### 2.2 Global — Vulture

Vulture reported 7 entries (2 in project source, 5 in the vendored `vendor/timesfm/` tree). Confidence ≥ 80 %.

Project source:

| File | Line | Symbol | Confidence |
|------|------|--------|------------|
| `setup_timesfm.py` | 9 | unused variable `exc_info` | 100 % |
| `src/tensortrade_model.py` | 99 | unused variable `options` | 100 % |

Vendored `vendor/timesfm/` (third-party, not actionable in this audit):

| File | Line | Symbol | Confidence |
|------|------|--------|------------|
| `vendor/timesfm/build/lib/timesfm/timesfm_2p5/timesfm_2p5_flax.py` | 190 | unused variable `unused_iter` | 100 % |
| `vendor/timesfm/src/timesfm/timesfm_2p5/timesfm_2p5_flax.py` | 190 | unused variable `unused_iter` | 100 % |
| `vendor/timesfm/v1/src/adapter/dora_layers.py` | 79 | unused variable `objtype` | 100 % |
| `vendor/timesfm/v1/src/adapter/lora_layers.py` | 70 | unused variable `objtype` | 100 % |
| `vendor/timesfm/v1/src/finetuning/finetuning_example.py` | 12 | unused import `asdict` | 90 % |

> ⚠️ Vulture produces false positives by construction (global static
> detection). Verify each entry before removal.

## 3. Complexity Hotspots (Radon)

Only functions/classes graded **C, D, E or F** are listed (ranks A and B hidden).
Project-source only — `tests/`, `vendor/` and `venv*` excluded.

### F (critical)

| File | Line | Symbol | Type |
|------|------|--------|------|
| `src/t212_executor.py` | 411 | `execute_t212_trade` | Function |

### E (high)

| File | Line | Symbol | Type |
|------|------|--------|------|
| `src/data.py` | 326 | `get_alpha_vantage_data` | Function |
| `src/enhanced_decision_engine.py` | 470 | `EnhancedDecisionEngine.make_enhanced_decision` | Method |
| `src/enhanced_decision_engine.py` | 128 | `VincentGanneModel.evaluate` | Method |

### D (medium)

| File | Line | Symbol | Type |
|------|------|--------|------|
| `backtest_prod.py` | 93 | `run_backtest` | Function |
| `main.py` | 67 | `run_trading_analysis` | Function |
| `src/adaptive_weight_manager.py` | 537 | `AdaptiveWeightManager.calculate_adaptive_weights` | Method |
| `src/data.py` | 749 | `get_vincent_ganne_indicators` | Function |
| `src/llm_client.py` | 157 | `_query_ollama` | Function |

### C (moderate)

| File | Line | Symbol | Type |
|------|------|--------|------|
| `debug_t212.py` | 150 | `cmd_buy` | Function |
| `src/adaptive_weight_manager.py` | 370 | `AdaptiveWeightManager.calculate_all_models_performance` | Method |
| `src/adaptive_weight_manager.py` | 663 | `AdaptiveWeightManager.resolve_previous_predictions` | Method |
| `src/advanced_risk_manager.py` | 435 | `AdvancedRiskManager.should_override_signal` | Method |
| `src/advanced_risk_manager.py` | 484 | `AdvancedRiskManager.get_risk_adjusted_signal` | Method |
| `src/classic_model.py` | 85 | `train_ensemble_model` | Function |
| `src/data.py` | 156 | `get_etf_data` | Function |
| `src/data.py` | 496 | `get_macro_data_multi_source` | Function |
| `src/data.py` | 709 | `get_hyperliquid_oil_data` | Function |
| `src/data.py` | 860 | `fetch_macro_data_for_date` | Function |
| `src/eia_client.py` | 54 | `EIAClient.get_fundamental_context` | Method |
| `src/eia_client.py` | 243 | `EIAClient.format_for_llm` | Method |
| `src/eia_client.py` | 368 | `EIAClient._make_request` | Method |
| `src/enhanced_decision_engine.py` | 101 | `VincentGanneModel` | Class |
| `src/enhanced_trading_example.py` | 367 | `EnhancedTradingSystem.perform_enhanced_analysis` | Method |
| `src/enhanced_trading_example.py` | 581 | `EnhancedTradingSystem._execute_hypothetical_trade` | Method |
| `src/enhanced_trading_example.py` | 710 | `EnhancedTradingSystem.display_enhanced_results` | Method |
| `src/features.py` | 202 | `select_features` | Function |
| `src/grebenkov_model.py` | 70 | `GrebenkovTrendModel.predict` | Method |
| `src/llm_client.py` | 26 | `construct_llm_prompt` | Function |
| `src/news_fetcher.py` | 42 | `fetch_alpha_vantage_news` | Function |
| `src/oil_bench_model.py` | 87 | `OilBenchModel._construct_prompt` | Method |
| `src/performance_monitor.py` | 549 | `PerformanceMonitor._assess_current_risk` | Method |
| `src/performance_monitor.py` | 705 | `PerformanceMonitor.update_monitoring` | Method |
| `src/t212_executor.py` | 161 | `sync_state_from_t212` | Function |
| `src/t212_executor.py` | 230 | `load_portfolio_state` | Function |
| `src/tensortrade_model.py` | 168 | `get_tensortrade_prediction` | Function |
| `src/timesfm_model.py` | 106 | `TimesFMModel.predict` | Method |
| `src/web_researcher.py` | 35 | `generate_search_query` | Function |
| `src/web_researcher.py` | 130 | `fetch_and_clean` | Function |

**Average complexity:** C (17.7) over 62 blocks (project + vendored).
**Average MI (project source only, 32 files):** 52.4 — bounded by two B-ranked files (`src/data.py` 18.13, `src/eia_client.py` 16.88) and a long tail of A-ranked files.

## 4. Code Duplication (Pylint)

`pylint --disable=all --enable=duplicate-code` flagged 6 pairs. The three largest come from the `alphaear-news` skill scripts (located under `.agents/`, `.kilocode/` and `.qwen/` skills folders) being inlined into `src/web_researcher.py`.

| # | File A | File B | Lines | Approx. size |
|---|--------|--------|-------|---------------|
| 1 | `src/web_researcher.py` | `.agents/skills/alphaear-news/scripts/news_tools.py` (also mirrored in `.kilocode/` and `.qwen/`) | 11–259 | ~249 lines (`NewsNowTools` + `PolymarketTools` classes) |
| 2 | `src/web_researcher.py` | `.agents/skills/alphaear-news/scripts/content_extractor.py` | 10–122 | ~113 lines (`ContentExtractor` class) |
| 3 | `src/web_researcher.py` | `.agents/skills/alphaear-news/scripts/database_manager.py` | 8–135 | ~128 lines (`DatabaseManager` class) |
| 4 | `src/adaptive_weight_manager.py` | `src/enhanced_decision_engine.py` | 93–103 / 289–299 | ~11 lines (base weights dict literal) |
| 5 | `main.py` | `schedule.py` | 31–41 / 19–29 | ~11 lines (UTF-8 stdout reconfigure + `logging.basicConfig`) |
| 6 | `src/adaptive_weight_manager.py` | `src/performance_monitor.py` | 149–159 / 125–135 | ~11 lines (`_init_database` preamble) |

Total: ~523 duplicated lines, of which ~490 concentrate in the `web_researcher` ↔ skill-scripts cluster.

## 5. Recommended Action Plan

1. **Refactor the lone F hotspot `execute_t212_trade` in `src/t212_executor.py:411`** — split into validation / sizing / order-submission / post-trade state-update helpers; this single function drives the worst complexity score and concentrates trading risk. Pair with the two E hotspots in `src/enhanced_decision_engine.py:470` (`make_enhanced_decision`) and `:128` (`VincentGanneModel.evaluate`), plus `src/data.py:326` (`get_alpha_vantage_data`), to bring the project below the E/F threshold and unlock grade B.
2. **Extract the `NewsNowTools` / `ContentExtractor` / `DatabaseManager` cluster (~490 duplicated lines) into a shared module** imported by both `src/web_researcher.py` and the `alphaear-news` skill scripts (or remove the inlined copy from `web_researcher.py` if the skill is the canonical source). Also extract the `base_weights` literal duplicated between `src/adaptive_weight_manager.py:93` and `src/enhanced_decision_engine.py:289` into a single constant to eliminate the configuration drift risk.
3. **Quick wins (< 30 min):** remove the two project Vulture hits (`exc_info` at `setup_timesfm.py:9`, `options` at `src/tensortrade_model.py:99`) and the UTF-8 setup block duplicated between `main.py:31` and `schedule.py:19` (move it to a tiny `src/bootstrap.py` helper). Ruff already being clean means these small fixes translate directly into a higher grade tier with no regression risk.
