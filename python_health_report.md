# Python Health Report — Trading-AI

Generated on 2026-07-28 11:15 by python-health-audit.

## 1. Executive Summary
- Global grade: D
- Reason: Grade D assigned: 2 E hotspots detected in `main.py:run_trading_analysis` and `src/adaptive_weight_manager.py:AdaptiveWeightManager.calculate_adaptive_weights` (plus 4 D-rank functions); average MI is excellent (≈A) but the E hotspots and 7 Ruff findings in project code drive the rank.

> Note on scope: findings inside `vendor/` (Kronos, TimesFM vendored copies) and `.agents/skills/` (third-party skill scripts) are excluded from the grade — they are not first-party code and not maintained here. The metrics below report only first-party project files.

## 2. Dead Code
### 2.1 Local — Ruff

15 findings total; **7 in first-party project code** (8 are in `.agents/skills/ui-ux-pro-max/scripts/`, out of scope for remediation):

| Rule | File:line | Issue |
|---|---|---|
| F401 | `reset_for_fresh_test.py:73` | `import sys` unused |
| F541 | `setup_council_models.py:65` | f-string without placeholders |
| F401 | `src/gemini_gateway.py:64` | `LIMITS_FREE` imported but unused |
| F401 | `src/gemini_gateway.py:65` | `LIMITS_PAID` imported but unused |
| E402 | `web_dashboard/app.py:74` | Module-level import not at top (`import sqlite3`) |
| E402 | `web_dashboard/app.py:75` | Module-level import not at top (`import pandas`) |
| E402 | `web_dashboard/app.py:149` | Module-level import not at top (`from src.t212_executor import ...`) |

All 7 are auto-fixable (`ruff check --fix`), no behaviour change.

### 2.2 Global — Vulture

2 entries (both 100% confidence) — well below the F-grade threshold of >20:

| File:line | Symbol | Confidence |
|---|---|---|
| `src/eia_client.py:134` | unused variable `months` | 100% |
| `src/gemini_quota.py:183` | unused variable `output_chars` | 100% |

> ⚠️ Vulture produces false positives by construction (global static detection). Verify each entry before removal. Both here are local variables (not exported symbols), low false-positive risk.

## 3. Complexity Hotspots (Radon)

Only functions/classes graded C, D, E or F (ranks A and B hidden). First-party project files only (`vendor/` excluded).

### E rank (high priority)
| File | Block | Type |
|---|---|---|
| `main.py` | `run_trading_analysis` (line 92) | Function |
| `src/adaptive_weight_manager.py` | `AdaptiveWeightManager.calculate_adaptive_weights` (line 578) | Method |

### D rank
| File | Block | Type |
|---|---|---|
| `audit_prod_logs.py` | `run_backtest` (line 397) | Function |
| `backtest_prod.py` | `run_backtest` (line 93) | Function |
| `reset_for_fresh_test.py` | `main` (line 336) | Function |
| `src/data.py` | `get_vincent_ganne_indicators` (line 724) | Function |
| `src/enhanced_trading_example.py` | `EnhancedTradingSystem.get_model_predictions` (line 260) | Method |
| `src/t212_executor.py` | `_validate_and_recalibrate_entry_price` (line 140) | Function |
| `src/council/weekend_council.py` | `run_council` (line 548) | Function |

### C rank (numerous — summarized)
~30 C-rank blocks across `src/` (the bulk of the trading pipeline: `enhanced_decision_engine`, `t212_executor`, `data`, `llm_client`, `eia_client`, `performance_monitor`, `oil_bench_model`, etc.). Average complexity across all 109 analyzed blocks: **C (16.4)**.

## 4. Code Duplication (Pylint)

2 pairs of duplicated code detected (Pylint score 9.98/10):

1. **`setup_council_models.py`** (lines 38-49) ⟷ **`src/council/weekend_council.py`** (lines 440-485) — the `_is_model_installed` / Ollama-model-listing helper is duplicated between the setup script and the council runtime.
2. **`clean_phantom_trades.py`** (lines 162-168) ⟷ **`reset_for_fresh_test.py`** (lines 420-426) — the dry-run/confirm CLI block is shared verbatim between the two reset scripts.

Both are small (≤12 lines) and low-risk; extraction into a shared helper would reduce maintenance drift.

## 5. Recommended Action Plan

1. **Refactor the 2 E hotspots** (`main.py:run_trading_analysis`, `src/adaptive_weight_manager.py:calculate_adaptive_weights`) — these are the pipeline's two largest orchestrators and the single biggest maintainability debt. Extract sub-steps (data fetch / model-prediction fan-out / consensus / T212 execution) into named helpers to bring both below rank C. This is what blocks a move from D → B.
2. **Apply `ruff check --fix` for the 7 first-party findings** — all are auto-fixable (unused imports, f-string prefixes, import ordering) with zero behaviour risk. Quick win that clears the lint baseline.
3. **Extract the 2 duplicated blocks** (council model-listing helper into `src/council/`, and the reset dry-run/confirm block into a shared `reset_lib.py`) — small, removes the only duplication the audit found and prevents the two reset scripts from drifting further apart.
