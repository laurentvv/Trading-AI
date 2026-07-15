# AGENTS.md — Operating Notes for AI Assistants

This file gives future AI agents (Kilo, Codex, Copilot, etc.) the minimum context needed to work safely in this repo. Read it once before editing production code.

**Project in one paragraph** — Python 3.12 trading-decision pipeline. Multi-model ensemble (Scikit-Learn, TimesFM 2.5, TensorTrade PPO, Gemma 4 12B LLM for text + vision + oil fundamentals + web research, sentiment, Vincent Ganne model) orchestrated by `main.py`. Outputs to `trading_journal.csv`, with optional Trading 212 demo/live execution via `--t212`. Virtualenv managed by `uv`. See `README.md` for the full feature list.

---

## 1. Deterministic Memory: The Four-File Discipline

Never rely solely on your context window to track project progress — it degrades, compresses, and is wiped. The **single source of truth** for system state lives in **four files on disk**, in `memory-bank/`. At every initialization, crash, or restart, read them to rebuild your state deterministically.

### 1.1 The four files (strict formats)

| File | Role | Lifecycle |
|---|---|---|
| **`memory-bank/feature_list.json`** | Complete structured map of all features (`id`, `name`, `description`, `status: pending\|in_progress\|completed`, `dependencies`). | Generated at initial planning; updated whenever a feature changes status. |
| **`memory-bank/contract.md`** | The technical validation contract — a list of strict, testable assertions (15–30 criteria). | **Frozen** just before the first line of code is written; the generator may not modify it afterwards. |
| **`memory-bank/progress.md`** | Macroscopic dashboard of the **current sprint** (objective + iteration milestones). Lets the agent instantly know what it is doing on restart. | Updated at the end of each loop iteration. |
| **`memory-bank/log.md`** | Chronological execution journal, **append-only**. Nothing is ever erased; each major event is stacked. | One entry added at the start and end of every action. |

> `log.md` entry format (mandatory): `## [YYYY-MM-DD] <phase> | <description>` where `<phase>` ∈ `init`, `gen`, `eval`, `fix`, …
> *(Historical product corrections live in `memory-bank/changelog.md` — kept separate from the sprint dashboard.)*

### 1.2 Operational directives for the execution loop

1. **Bootstrap phase** — Before any action, check these four files exist. If missing, create them per the formats above (see `memory-bank/contract.md` and `feature_list.json` for the live templates). If present, read them to rebuild immediate memory.
2. **Action phase** — Before executing a task, write the corresponding line to `log.md`.
3. **Sync phase** — After every file write or test run, update the matching status file (`progress.md` or `feature_list.json`).
4. **Error handling** — If an exception occurs or the process is interrupted, the valid state is the one extracted from the **last line of `log.md`** combined with the assertions of `progress.md`.

### 1.3 Long-form context (supplementary, not state-of-truth)

The rest of `memory-bank/` holds long-form, human-maintained context — useful background, **not** the deterministic state: `techContext.md`, `activeContext.md`, `systemPatterns.md`, `projectbrief.md`, `productContext.md`, `improvement_proposals.md`, `changelog.md` (historical product corrections).

---

## 2. Non-obvious safety invariants — DO NOT BREAK

### 2.1 Dual-Layer JSON Defence (Gemma think mode)

Gemma 4 12B's `<|think|>` reasoning channel is **active** in production. JSON-extraction safety is a **two-layer** defence — neither layer alone is sufficient:

- **Layer 1 (load-bearing)** — schema-strict `format: SCHEMA_*` parameter at every Ollama call site. Defined in `src/llm_client.py` (`SCHEMA_TRADING_DECISION`, `SCHEMA_SEARCH_QUERY`, `SCHEMA_OIL_ALLOCATION`) with `additionalProperties: false`. Enforced server-side by Ollama.
- **Layer 2 (belt-and-braces)** — every system prompt ends with `"...never add a 'thought' key."`.

**When editing the 4 Ollama call sites** — `get_llm_decision` and `get_visual_llm_decision` (`src/llm_client.py`), `OilBenchModel._query_llm` (`src/oil_bench_model.py`), `generate_search_query` (`src/web_researcher.py`) — **preserve both layers**: keep the `"<|think|> "` prefix and the `"...never add a 'thought' key."` suffix. (Function names are cited instead of line numbers — lines drift, names don't.) New LLM call sites **must** use a schema-strict `format: SCHEMA_*` (never the loose `format:json` — `tests/check_llm_json.py` proves it leaks `<|channel>thought` debris). On schema change, run `tests/check_llm_json.py` (acceptance: all `*_schema*` and `oil_*` cases OK). Full evidence/reversal: `docs/ADR-001-think-mode-dual-layer-defence.md`.

> **Gemini surface (different mechanism, same intent):** the Gemini path enforces Layer-2 via `_NO_THOUGHT_SUFFIX` (`src/gemini_gateway.py`) and Layer-1 via `response_schema`. It does **not** use `<|think|>` (that channel is Gemma-only). The asymmetry is intentional — do not "fix" it by adding `<|think|>` to Gemini prompts.

### 2.2 Other invariants (non-exhaustive)

- **T212 demo vs live** is governed by `T212_ENV` in `.env.t212` (demo is rate-limit-tolerant; live is not). Never commit credentials.
- **Per-ticker budget**: `INITIAL_BUDGETS` dict (default 1000€ per ticker), **not** the historical 5000€ hardcoded fallback.
- **DB write isolation (`write_db`)**: `EnhancedTradingSystem(write_db=...)` controls whether the simulation step (`_execute_hypothetical_trade`) writes to `trading_history.db`. In T212 mode, `main.py` passes `write_db=not is_t212` → **only `t212_executor` writes to the DB, after a broker-confirmed fill**. The simulation still runs (for internal reporting) but does not pollute the DB with phantom trades. **Never set `write_db=True` in `--t212` mode** — it re-creates the July 2026 phantom-trades desync (DB showed trades the broker never executed).
- **T212 quantity precision**: governed by `TICKER_QUANTITY_PRECISION` dict in `src/t212_executor.py` (not the old `"CRUD" in ticker` heuristic, which broke when CRUDP.PA was remapped to `OD7Fd_EQ`). Add new instruments here; fallback is `DEFAULT_QUANTITY_PRECISION = 2` (safe — T212 rejects excess precision, never insufficient).
- **win_rate sentinel**: `_calculate_win_rate` (`src/performance_monitor.py`) returns `-1.0` (not `0.0`) when no closed trades exist or on error. The alert condition excludes it via `>= 0`. **Never return `0.0` for "not computable"** — it triggers false "Win rate critically low: 0.00%" HIGH alerts.
- **Cache staleness**: 1 day (`src/data.py`) — Parquet files older than that are auto-refreshed.
- **Cycle timeout**: 40 min (`CYCLE_TIMEOUT_SECONDS` in `main.py:38`). On timeout, `cancel_event` is set so the orphan thread cannot place a T212 order even if it finishes later; a per-ticker `threading.Lock` serializes order placement.

---

## 3. Testing & validation commands

PowerShell note: `uv run pytest ...` may fail with "Failed to canonicalize script path" — prefer the direct `.venv\Scripts\python.exe -m pytest` form.

| Goal | Command |
|---|---|
| Mocked unit tests (deterministic, no Ollama) | `.venv\Scripts\python.exe -m pytest tests/test_llm_client.py tests/test_llm_prompts.py tests/test_oil_bench_model.py tests/test_weekend_council.py -v` |
| Live weekend council (Gemini members + Ollama locals, ~5 min) | `uv run python src/council/weekend_council.py --days 7` |
| Live LLM JSON harness (requires `ollama serve`) | `.venv\Scripts\python.exe tests/check_llm_json.py` |
| Full pipeline, T212 demo | `uv run main.py --t212` |
| Prod logs audit + corrected backtest | `uv run python audit_prod_logs.py` |
| Refresh parquet cache | `uv run refresh_cache.py` |

---

## 4. Where things live

- `main.py` — entry point (`--t212` / `--simul` / `--ticker`).
- `src/llm_client.py` — schemas, `_query_ollama`, `get_llm_decision` / `get_visual_llm_decision`, context injectors into `construct_llm_prompt` (`get_morning_brief_context`, `get_council_verdict_context`).
- `src/oil_bench_model.py` — `OilBenchModel._query_llm` (call site #3). `src/web_researcher.py` — `generate_search_query` (call site #4).
- `src/enhanced_decision_engine.py` — consensus aggregator. ⚠️ Does **not** currently consume the `failed: True` flag from `_fallback_decision` (silent HOLD bias risk if JSON extraction fails intermittently — see §6).
- `src/council/` — **Weekend Council**: async, multi-persona LLM retrospective (`run_council`), runs Sat & Sun at 09:00 via `schedule.py`, writes `docs/council_reports/council_report_YYYY-MM-DD.md`. Its Judge verdict is the **11th weighted vote** in the consensus (9.5%, age-decayed over 7 days) via `get_council_ticker_stance`, **and** soft context via `get_council_verdict_context`. 6 personas on 5 distinct model families (Gemma 4 12B / GLM-4.6V-Flash / Qwen 3.5 9B / Mistral Nemo 12B + Gemini 2.5 Flash), Judge on Gemini 2.5 Pro. 4-round protocol + anti-groupthink. Models prefixed `gemini:` route through the shared `GeminiGateway.deliberate()`; others on local Ollama. See `docs/ADR-003-weekend-council-11th-voice.md` and §8.
- `src/gemini_gateway.py` + `src/gemini_quota.py` — the single shared two-tier Gemini cloud gateway. Cascades (Pro → 3.5-flash → … → safety nets) + `QuotaTracker` (RPM/RPD + rolling 30-day **cost budget** `GEMINI_PAY_MONTHLY_BUDGET_EUR` computed from actual token usage × per-model price, with a `GEMINI_PAY_DAILY_CAP` call-count backstop; SQLite ledger). Returns `None` on exhaustion so callers fall back to free-llm / Ollama. See §8.
- `tests/check_llm_json.py` — live JSON diagnostic harness (10 cases × 3 schema families × 2 modes).
- `docs/ADR-*.md` — architecture decision records.
- **`memory-bank/`** — the deterministic 4-file state (§1) **plus** long-form context: `techContext.md`, `activeContext.md`, `systemPatterns.md`, `projectbrief.md`, `productContext.md`, `improvement_proposals.md`, `changelog.md` (historical corrections).

## 5. Branching, Commits & Environment Rules

- **OS Constraint (CRITICAL)**: DEV and PROD both run on **Windows**. Never suggest Linux-only commands (`rm -rf`, `ls`, `cat`) — use PowerShell (`Remove-Item -Recurse -Force`, `Get-ChildItem`) or CMD (`rmdir /s /q`).
- **Cache Invalidation**: `data_cache/` and `logs_prod/` are gitignored. When pushing fixes for corrupted data (TensorTrade policy collapse, broken EIA timestamps, …), **explicitly remind the user** to delete the affected caches on PROD after `git pull` (e.g. `Remove-Item -Recurse -Force data_cache\tensortrade`, or `uv run python refresh_cache.py`).
- **Council Models (CRITICAL for the weekend council)**: each persona runs on a **different** model lineage — this is the core of the design (genuine reasoning diversity, not costume changes on one model). When pushing the council feature, **remind the user to** (1) run `uv run python setup_council_models.py` on PROD after `git pull` (else local members silently fall back to Gemma), AND (2) set `GEMINI_API_KEY` (free) + `GEMINI_API_KEY_PAY` (paid) in `.env` (else Sceptique/Comportementaliste/Judge fall back to local Ollama — slower but functional).
- Long-lived branch: `main` (production state, schema-defended). Validation branches: short-lived (`think-mode`, …) for risky re-enablements.
- **Never commit secrets** (`.env*`, `*.db`, `data_cache/`, `logs_prod/` are gitignored — keep them that way).
- **Do not push** without explicit user request.

---

## 6. Known follow-ups (flagged, not in scope)

- `_THINKING_TOKENS` is defined once in `src/llm_client.py` (`_strip_thinking_prefix`); `tests/check_llm_json.py` exercises the same `<|think|>` / `<|channel>thought` debris inline (no shared constant) — if the token set changes, keep both in sync.
- `_fallback_decision`'s `failed: True` flag is **not** consumed by the consensus aggregator (`enhanced_decision_engine.py`) — a failed LLM call is weighted as a plain HOLD vote (silent HOLD bias on intermittent extraction failure).
- `tests/check_llm_json.py` returns exit 1 if any case fails, including the documented-to-fail loose-format cases (could be refactored to return 0 when only expected failures fail).
- `backtest_prod.py` reads `data_cache/` (repo root, ends ~2026-05-27) instead of `logs_prod/data_cache/` (prod, current) → empty tables when the journal spans an uncovered period. Use `audit_prod_logs.py` instead.
- **FinAcumen (June 2026) — repaired; converges; feeds the morning brief** (which consumes `main.py` outputs via shared files, no direct import: `main.py` writes `trading_journal.csv`/`trading.log`/`performance_monitor.db` → `morning_brief/tools/` read them → `schedule.py` appends FinAcumen into `morning_market_brief.md`). By design it is **not** an 11th per-cycle vote (that slot is the Weekend Council — §4). Key repair details: `src/core/tools.py` (`lookup_ohlc` accepts a list → dict; rsi/sma/macd added; `pd`/`np` pre-injected so the sandbox's `__import__` ban is harmless) and `src/agents/solver.py` (execute-vs-final-answer branches on *non-empty* `python_code` / `action in BUY|SELL|HOLD`, not key presence).

### 6.0 Resolved follow-ups (15 July 2026 — risk-manager calibration, SELL bias, EIA)

The 15 July 2026 PROD audit (`audit_prod_logs.py` → WARN) found three behavioural bugs. All reproduced against `logs_prod/data_cache/`; root causes + fixes:

- **Risk manager 100% VERY_HIGH** (`src/advanced_risk_manager.py:94-100, 236-242`) — the composite 0-1 `overall_risk_score` was bucketed by `volatility_thresholds = {0.01, 0.015, 0.025, 0.04, inf}`, magnitudes sized for *annualised-volatility fractions*. Any realistic composite > 0.04 collapsed to `VERY_HIGH`, so 100% of 294 PROD cycles (both SXRV.DE and CRUDP.PA) were `VERY_HIGH` — which in turn activated the BUY→HOLD gate in `should_override_signal` and neutralised every SXRV.DE BUY (147/147 `Risk_Adjusted=HOLD`). Aggravated by `calculate_liquidity_risk`'s `pattern_risk = 1-|corr(volume,returns)|` ≈ 0.98 for thin ETFs (volume decorrelated from price by construction). **Fix**: thresholds rescaled to composite range (`VERY_LOW 0.20 → HIGH 0.65 → VERY_HIGH inf`); `pattern_risk` capped at 0.5 and down-weighted (40%) vs `volume_risk` (60%).
- **SELL never emitted** (`src/enhanced_decision_engine.py:511-540, 344`; `src/enhanced_trading_example.py:651-662`) — 0 SELL across 294 cycles despite ~400 individual SELL votes. Three compounding causes: (a) `_calculate_weighted_score` summed raw weighted votes, so every HOLD model (strength 0) diluted the score toward 0 without renormalising — the most-bearish PROD cycle reached only -0.139 (vs the -0.15 cutoff); (b) the SELL threshold sat at -0.15, just past that -0.139 floor; (c) `VincentGanneModel` voted BUY/STRONG_BUY 147/147 times on SXRV.DE (a macro "market-bottom" model applied to an equity ETF — pure structural bullish noise). **Fix**: score renormalised by the *voting* weight (non-HOLD only), with a **quorum guard** (fallback to raw score when <25% of weight voted, so a lone model can't force a STRONG signal — 47/294 cycles had <2 voters); SELL threshold loosened -0.15 → -0.10; VG disabled on non-oil tickers (`effective_vg_indicators = None`). Regression test `tests/test_prod_regression.py` proves SELL now fires on the most-bearish cycle.
- **EIA crude_imports degenerate cache** (`src/eia_client.py:134-170`) — `get_crude_imports` called `/crude-oil-imports/data` with no `facets` filter, so after `groupby("period")` it collapsed to a 1-row payload pinned at 2026-04-01; `_save_to_cache` bumped the mtime unconditionally, so the mtime-based TTL hid the degeneracy (file looked "fresh" while 3.5 months stale). **Fix**: ascending sort + `length` raised to `months*50` so all origins×months fit before aggregation; payloads with <3 rows are refused caching (logged WARNING). (NB: `/crude-oil-imports/data` does NOT accept a `facets[process]` param — its valid facets are `originId`/`originType`/`destinationId`/`destinationType`/`gradeId`; an earlier revision tried `process` and got HTTP 400 on every cycle.)
- **`force_stop_loss` UnboundLocalError** (`src/t212_executor.py:~957`) — `force_stop_loss` was initialised *inside* the `if current_pos:` block. A SELL signal arriving with **no open position** skipped that block, leaving `force_stop_loss` unbound at the `_execute_sell_order(...)` call → crash. This path was unreachable while SELL never fired (pre-fix); the SELL-unblocking exposed it. **Fix**: initialise `force_stop_loss = False` / `exit_reason = None` before the branch; `_execute_sell_order` already no-ops cleanly on `current_pos is None`.

> **PROD cache invalidation reminder:** after `git pull`, delete `logs_prod/data_cache/eia/eia_crude_imports.parquet` to force a live re-fetch.

### 6.1 Resolved follow-ups (July 2026 — phantom-trades & T212 precision suite)

The July 2026 PROD audit found and fixed four bugs (PR #78/#79):

- **Phantom trades (CRITICAL)** — `_execute_hypothetical_trade` wrote SIMULATED trades to `trading_history.db` BEFORE the real T212 order was attempted. When the order failed (precision mismatch) or was skipped (HOLD), the phantom BUY stayed → persistent DB/broker desync (0 positions at broker, phantom BUYs in DB). **Fix**: `write_db=not is_t212` — the simulation no longer writes in T212 mode; only `t212_executor` writes after a broker-confirmed fill.
- **T212 quantity precision** — the heuristic `if "CRUD" in ticker` broke when CRUDP.PA was remapped to `OD7Fd_EQ` (no longer contains "CRUD") → wrong precision → order rejected. **Fix**: explicit `TICKER_QUANTITY_PRECISION` table.
- **win_rate false alerts** — `_calculate_win_rate` returned `0.0` (instead of the `-1.0` sentinel) on empty/error → 72 false "Win rate critically low: 0.00%" HIGH alerts. **Fix**: `-1.0` sentinel consistently; alert condition excludes it.
- **EIA stale false positive** — `audit_prod_logs.py` used `df.index` instead of the `period` column → RangeIndex interpreted as Unix timestamps → `@1970-01-01`. **Fix**: column cascade `period` → `date` → `index`.

### 6.2 Resolved follow-ups (Late June 2026 — ADR-002 Suite)

The end-of-June decision-model audit (`docs/ADR-002-decision-model-quality-audit.md`) remediated the structural bullish bias and the market-derived win_rate metric:
- **win_rate / Accuracy** — `BUY` is rewarded only if return exceeds `HOLD_NEUTRAL_RETURN_THRESHOLD` (0.5% buffer); `HOLD` is rewarded when the market is flat (stops optimistic models farming drift).
- **Bullish bias / Overfitting** — *TensorTrade*: disabled continuous `model.learn(total_timesteps=500)` during cached inference (was causing catastrophic policy collapse, stuck on BUY 1.00). *LLM Visual*: temperature 0.4 + hardened prompt forcing `HOLD` on ambiguous charts.
- **Hybrid LLM Architecture** — `get_llm_decision` prioritizes a cloud "Frontier Model" via `free-llm-api-keys` with instant silent fallback to local Ollama on 503. Vision (VL) stays on Ollama (free proxies reject images).

**Still open after ADR-002:** isotonic TensorTrade recalibration (current cap is interim), upstream sentiment-data skew, unifying the two `regime_adjustments` dicts, `return_5d` population.

---

## 7. Tooling scripts

- `audit_prod_logs.py` — validates **all** `logs_prod/` files (catalogue, SQLite integrity, parquet freshness, JSON/pkl, FinAcumen state) and runs a corrected backtest against `logs_prod/data_cache/`. Emits `logs_prod/audit_report.md`. (`uv run python audit_prod_logs.py`)
- `setup_council_models.py` — installs the **four local** Ollama model families the Weekend Council requires (Gemma 4 12B / GLM-4.6V-Flash / Qwen 3.5 9B / Mistral Nemo 12B); reports (without pulling) the cloud Gemini models. Idempotent, skips installed. **Must run on every PROD server** after `git pull` of the council feature (else members fall back to Gemma and diversity is lost). (`uv run python setup_council_models.py`)
- `reset_for_fresh_test.py` — **MAX reset** for a truly virgin restart. Wipes by pattern (any gitignored file at repo root: `*.db *.log *.json *.csv *.png` + `data_cache/` + runtime dirs). Preserves `.env*`, `.venv`, `.git`, `logs_prod`, `memory-bank`, source code. Gemini quota ledger wiped by default (DEMO); pass `--keep-quota-ledger` on PAID PROD to avoid overspend. Handles Windows locked files (copy+truncate fallback). (`uv run python reset_for_fresh_test.py --dry-run`)
- `clean_phantom_trades.py` — **targeted cleanup** of the 3 artefacts polluted by the phantom-trades bug (`trading_history.db`, `performance_monitor.db`, `t212_portfolio_state.json`). Preserves all caches, models, prices, EIA. Use after pulling the July 2026 fix, before relaunching. (`uv run python clean_phantom_trades.py --yes`)

---

## 8. Gemini cloud gateway (two-tier, shared)

There is a **single** Gemini cloud layer — `src/gemini_gateway.py` (`GeminiGateway`) — shared by every cloud-routed caller: real-time decision/vision (`get_llm_decision`, `get_visual_llm_decision`), web summary, AND the Weekend Council. **There is no second Gemini path; do not build one.** (Two keys, both in `.env`, gitignored — https://aistudio.google.com/apikey):

| Env var | Tier | Notable models | Used by |
|---|---|---|---|
| `GEMINI_API_KEY` | **Free** | `gemini-2.5-flash` (Pro is 0/0/0 here) | Council members (Sceptique, Comportementaliste); web summary; fallback when paid exhausted |
| `GEMINI_API_KEY_PAY` | **Paid** | `gemini-2.5-pro`, `gemini-3.5-flash` | Council Judge (`gemini:pro`); real-time decision + vision |

**Routing convention**: a model string prefixed `gemini:` is routed to the gateway. In the council, the tier is inferred from the id — `"pro"` → paid cascade, otherwise free. The gateway internally runs a **cascade** (head model → safety nets) per tier and the `QuotaTracker` (`src/gemini_quota.py`) pre-flights each call against RPM/RPD caps plus **two billing guards** on the paid tier: (1) a rolling 30-day **cost budget** `GEMINI_PAY_MONTHLY_BUDGET_EUR` (default 8.6 € — the load-bearing guard; each call's cost is computed from the real `usage_metadata` token counts × the per-model `PRICE_TABLE_EUR_PER_MTOKEN`), and (2) a daily call-count backstop `GEMINI_PAY_DAILY_CAP` (default 200). When the gateway can't serve a request it returns `None` (never raises) and the caller falls back to local Ollama.

**Public methods** (add new use-cases here, don't duplicate the SDK elsewhere): `decide(prompt)` → JSON dict (paid-led) · `analyze_chart(image_path)` → dict (vision, paid-led) · `summarize_web_context(text)` → str (free-only) · `deliberate(system, user, use_paid)` → str (prose, no JSON schema, used by the Council). Performance: a Gemini call is **0.4–30 s** vs **5–9 min** on CPU Ollama.
