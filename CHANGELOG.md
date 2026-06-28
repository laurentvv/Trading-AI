# Changelog

All notable changes to this project are documented in this file.
The format is loosely based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added — 2026-06-28 — Weekend Council (11th consensus voice)
A weekly, async, multi-persona LLM retrospective that runs every **Saturday at 01:00** and feeds its verdict back into the real-time consensus as the **11th weighted vote** (9.5%). Adapted from [`0xNyk/council-of-high-intelligence`](https://github.com/0xNyk/council-of-high-intelligence). See `docs/ADR-003-weekend-council-11th-voice.md`.

- **6 personas on 5 distinct model families** for genuine reasoning diversity (not costume changes on one model): Le Stratège (Gemma 4 12B), Le Gestionnaire de Risque (GLM-4.6V-Flash), Le Quant (Qwen 3.5 9B), Le Sceptique (LFM 2.5 Mamba), Le Tacticien (Mistral Nemo 12B), Le Comportementaliste (LFM 2.5). The Judge runs on **Qwen3.5-9B-MTP** (IFEval 91.5, 262K context).
- **4-round protocol** with anti-groupthink: (0) Problem Restate Gate, (1) targeted analysis with explicit `STANCE: BUY|SELL|HOLD (X%)`, dissent quota that forces a steelman if ≥2/3 majority converges, (2) 1-vs-1 directed debate, (3) Judge verdict (Unresolved-Questions-first).
- **Level 3 integration** — the Judge emits a parseable `VERDICT_TICKER:` block per ticker; `get_council_ticker_stance()` parses it and adds a `ModelDecision(model_name="council")` at weight 0.10. Confidence **decays linearly** (full at day 0 → 0 at day 7). Council is exempt from the adaptive weight loop (`fixed_weight_models`) since its correctness can't be measured against per-cycle market direction.
- **Real PROD context** — analyses model accuracy (`model_performance.db`), portfolio metrics + critical alerts (`performance_monitor.db`), and the trading journal (bias detector), not a generic market feed.
- **Generous token budgets** tuned for thinking models on CPU: `num_predict` up to 12000 (Judge), `num_ctx` up to 65536, 15-min per-call Ollama timeout, 48-hour scheduler window (`COUNCIL_TIMEOUT = 172800`).
- **Setup**: `uv run python setup_council_models.py` installs the 6 required models (~40 GB, idempotent). `schedule_test.py` forces a one-off trigger for validation.

### Fixed — 2026-06-28 — Council code review (6 bugs, incl. critical inert vote)
A thorough audit of the initial Level 3 commit found the council vote was **inert in production**. The fixes (commit `2ccd26c`):
- **Ticker mismatch (CRITICAL)** — the call site passed `self.analysis_ticker` (`^NDX`) but the verdict uses trading tickers (`SXRV.DE`); never matched → always skipped. Now passes `self.ticker`.
- **Freshness duplication** — unified into `_load_fresh_council_report()` (text-injection and vote previously had separate, drift-prone copies).
- **Adaptive weight drift** — `fixed_weight_models` exemption prevents the manager from silently rescaling the 0.10 weight via a neutral performance score.
- **Parser robustness** — French decimal comma (`0,65`), percent rescaling (`85` → `0.85`), `rfind` for block isolation.
- **DB pollution** — council excluded from outcome tracking (no resolvable outcome).

### Tests — 2026-06-28
- `tests/test_weekend_council.py` (22 tests): routing, dissent quota, vote tally, models footer, graceful degradation.
- `tests/test_llm_client.py`: `TestCouncilTickerStance` (8 tests) + `TestStripThinkingDebris` (4) + `TestCouncilVerdictExtraction` (5) + `TestCouncilVerdictContext` (4).
- `tests/test_enhanced_decision_engine.py`: `TestCouncilVoteIntegration` (4 tests) — confirms a council SELL vote measurably lowers the weighted score.
- Full suite: **84 green**, Ruff clean.

### Added — 2026-06-23
- **`audit_prod_logs.py`** — new standalone auditor that validates **all** files in `logs_prod/` (catalogue, SQLite integrity/row-counts, parquet freshness + June-2026 coverage, JSON search-query caches, pickle models + TensorTrade metadata), runs a **corrected backtest** against the prod cache (`logs_prod/data_cache/`, current — not the stale repo-root `data_cache/`), and a dedicated FinAcumen section (state-file analysis + deterministic tool-chain proof). Emits `logs_prod/audit_report.md` with an OK/WARN/FAIL verdict.

### Fixed — 2026-06-23 — FinAcumen convergence (was `status: timeout` on every prod run)
FinAcumen (`src/finacumen_main.py`) converged to `status: success` after fixing six bugs in `src/core/tools.py` and `src/agents/solver.py`. Verified live with Ollama + gemma-4-12b: **CRUDP.PA → HOLD 0.75, SXRV.DE → BUY 0.85**, each citing real fetched prices (close, SMA50/200, RSI, MACD).
- **`lookup_ohlc` API mismatch** — now accepts `indicator: str | list[str]`; a list returns a `{indicator: value}` dict (the form the LLM always generated). A single string still returns a float (backward compatible).
- **Missing indicators** — added `rsi` (Wilder 14), `sma_50`, `sma_200`, `ema_12`, `ema_26`, `macd`, computed from the yfinance history.
- **Symbol mapping** — aliases (`WTI`, `NASDAQ`, `NDX`, `BRENT`, `SP500`) plus direct pass-through of any yfinance ticker (`CRUDP.PA`, `SXRV.DE`).
- **Sandbox `__import__` ban** — `NumericalReasoningEngine` now pre-injects `pd` (pandas) and `np` (numpy) so the LLM never needs `import` (the `__import__` restriction is preserved for security).
- **Invisible fetched data** — the solver observation now echoes the `data` variable when the LLM assigns `data = lookup_ohlc(...)` without `print`-ing it (the root cause of the prod fetch-loop timeouts).
- **Branch logic** — the solver now distinguishes *execute* (`python_code` non-empty) from *final answer* (`action in BUY|SELL|HOLD`). Previously it always took the execute branch because `python_code` is a mandatory schema key, so final answers were never accepted.
- **Solver prompt** — documents the real `lookup_ohlc` signature, tells the LLM to call it **once** then `print`, then decide (no fetch loop). `max_iterations` raised 5 → 6.

### Tests — 2026-06-23
- `tests/test_finacumen.py` extended from 2 → 8 tests: regression coverage for all six bugs (list→dict, derived indicators, single-string backward compat, pd/np without import, `__import__` still blocked, full LLM-style code in the sandbox). Mocked suite **20/20 green**.

### Known follow-up (not in scope)
- FinAcumen is repaired and verified converging. It is coupled to the main project **via shared data**: `main.py` writes `trading_journal.csv` / `trading.log` / `performance_monitor.db`, which `morning_brief` reads; `schedule.py` then appends the FinAcumen section into `morning_market_brief.md`. So FinAcumen influences the daily decision brief. It is **not** a per-cycle vote in the real-time consensus (`enhanced_decision_engine.py` / `model_performance.db`); wiring it as an 11th vote is a deliberate follow-up.
- `backtest_prod.py` still reads the stale repo-root `data_cache/`; use `audit_prod_logs.py` for a backtest against the prod cache.

## [Older]

### Added — 2026-06-06
- **Think mode re-enabled on Gemma 4 12B**. The `<|think|>` token is now present in all four production system prompts (`src/llm_client.py:188`, `src/llm_client.py:236`, `src/oil_bench_model.py:158`, `src/web_researcher.py:205`). Restores the model's internal reasoning channel.
- **Dual-layer JSON defence** documented as the load-bearing architecture for JSON-extraction safety:
  - Layer 1 (load-bearing): `format: SCHEMA_*` with `additionalProperties: false`, enforced server-side by Ollama.
  - Layer 2 (belt-and-braces): defensive system-prompt suffix `"...never add a 'thought' key."`.
- **Architecture Decision Record** `docs/ADR-001-think-mode-dual-layer-defence.md` capturing the rationale, validation evidence, and reversal procedure.
- **Top-of-file docstring** in `tests/check_llm_json.py` updated to clarify that `*_v1_buggy` cases are now the production path, with two acceptable outcomes documented (all-OK or `*_v1_buggy` fail under loose format → negative-result validation).

### Changed — 2026-06-06
- `README.md` "Advanced Cognition" bullet: replaced the "disables the model's thinking mode" wording with the dual-layer defence explanation and a pointer to the ADR.
- `memory-bank/techContext.md`: thinking mode now described as **enabled**, with the two-layer defence explained.
- `memory-bank/activeContext.md`: new "Think Mode Re-enabled (2026-06-06)" entry under "Key Recent Changes".
- `memory-bank/systemPatterns.md`: new "Dual-Layer JSON Defence (Think Mode)" pattern documented.
- `memory-bank/progress.md`: new dated correction entry (2026-06-06) under "Corrections Récentes".

### Validation Evidence — 2026-06-06
- Mocked pytest gate: **12/12 pass** (`test_llm_client.py`, `test_llm_prompts.py`, `test_oil_bench_model.py`).
- Live `tests/check_llm_json.py` harness: **6/10 OK** — all schema-strict cases pass with `<|think|>` active; failures are exclusively on loose `format:json` (not used in production).
- End-to-end `uv run main.py --t212`: **exit 0**, **4.66 min** total, **2 new `trading_journal.csv` rows**, **0 `"Could not find valid JSON"` log lines**.
- All four `<|think|>`-enabled LLM call sites produced validated JSON in real production conditions.

### Reversibility
- Soft: `git revert <merge-commit>` or `git switch main` (pre-merge).
- Hard: `git switch main && git branch -D think-mode` (pre-merge, erases branch).
- Targeted: remove `"<|think|> "` prefix from the four `"system"` strings — schema layer keeps carrying safety.
