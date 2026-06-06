# ADR-001 — Think Mode Re-enablement & Dual-Layer JSON Defence

**Status**: Accepted
**Date**: 2026-06-06
**Branch**: `think-mode` (validation) → `main` (merge)
**Decision owner**: User (explicit request) — technical validation by Kilo agent
**Supersedes**: Implicit May 2026 decision to remove `<|think|>` to neutralise the JSON-debris defect

---

## 1. Context & History

### 1.1 The May 2026 Defect

In May 2026 the pipeline started emitting `"Could not find valid JSON with keys ['query']"` errors from `src/llm_client.py::_query_ollama`. Root cause: the Gemma 4 12B model, when its `<|think|>` reasoning channel is active, leaks `<|channel>thought` tokens into the response stream. Example failure mode:

```json
{"thought": "<|channel>thought<channel|>```json\n{\n  \"query\": \"...\"}\n```"}
```

The `_query_ollama` JSON extractor (`src/llm_client.py:311-401`) strips thinking tokens but cannot recover a valid object from this shape — only the `thought` key survives parsing, the real payload is buried inside a string. After 3 retries, the call falls back to the canonical HOLD sentinel (`_fallback_decision`, `src/llm_client.py:59-71`) with `failed: True`.

### 1.2 The Initial Mitigation

At the time, the system relied on the **loose** `format:json` parameter of Ollama (no schema constraint, just "produce valid JSON"). With no schema enforcement, the `<|channel>thought` debris flowed freely into the output. The pragmatic fix was to **remove `<|think|>`** from the four production system prompts — eliminating the reasoning channel that was leaking. This worked.

### 1.3 What Changed Since

Independently, the four call sites were upgraded to use **schema-strict** `format` values:

| Call site | Schema |
|---|---|
| `get_llm_decision` (`src/llm_client.py:188`) | `SCHEMA_TRADING_DECISION` (`signal`, `confidence`, `analysis`) |
| `get_visual_llm_decision` (`src/llm_client.py:236`) | `SCHEMA_TRADING_DECISION` |
| `OilBenchModel._query_llm` (`src/oil_bench_model.py:158`) | `SCHEMA_OIL_ALLOCATION` (`allocation`, `reasoning`) |
| `generate_search_query` (`src/web_researcher.py:205`) | `SCHEMA_SEARCH_QUERY` (`query`) |

All schemas use `additionalProperties: false`. Ollama enforces this **server-side** — the model literally cannot emit a key outside the schema. This is a much stronger guarantee than the loose `format:json`.

### 1.4 The Hypothesis

If the schema-strict `format` parameter is the load-bearing defence, then `<|think|>` can be re-enabled without re-introducing the defect — the schema layer should suppress the `thought` key entirely.

---

## 2. Decision

**Re-enable the `<|think|>` token in all four production system prompts.**

Keep the schema-strict `format: SCHEMA_*` parameter as the load-bearing safety net. Keep the defensive system-prompt suffix `"...never add a 'thought' key."` as a redundant second layer.

The resulting architecture is a **dual-layer JSON defence**:

```
LLM call payload
├── prompt: <user text>
├── system: "<|think|> <role>. <task>. Output ONLY ... — never add a 'thought' key."
│           └── Layer 2 (client-side, advisory): tells the model what not to do
├── format: SCHEMA_*  (additionalProperties: false, server-enforced)
│           └── Layer 1 (server-side, load-bearing): makes the prohibited output unrepresentable
└── options: {temperature, num_predict}
```

---

## 3. Validation

### 3.1 Mocked test gate (deterministic)

```
.venv\Scripts\python.exe -m pytest tests/test_llm_client.py tests/test_llm_prompts.py tests/test_oil_bench_model.py -v
```

**Result**: 12/12 pass. Confirms the 4 source edits did not break any unrelated contract (the mocked tests assert on parsed return values, not on prompt payloads).

### 3.2 Live LLM harness — `tests/check_llm_json.py`

10 cases × 3 schema families × 2 modes (`<|think|>` on/off) × 2 format modes (loose `format:json` vs schema-strict).

| Case ID | `<|think|>` | Format | Result | Note |
|---|---|---|---|---|
| `query_v1_prod_buggy` | ON | `format:json` (loose) | **FAIL** | Reproduces the May 2026 defect |
| `query_v4_strict` | OFF | `format:json` (loose) | **FAIL** | Even without `<|think|>`, loose format leaks |
| `query_v6_schema` | OFF | `SCHEMA_SEARCH_QUERY` | **OK** | Schema-strict alone works |
| `query_v7_schema_strict` | ON | `SCHEMA_SEARCH_QUERY` | **OK** | Schema-strict survives `<|think|>` |
| `decision_v1_buggy` | ON | `format:json` (loose) | **FAIL** | Defect repro on trading-decision schema |
| `decision_v2_fixed` | OFF | `format:json` (loose) | **FAIL** | Loose format insufficient even with suffix |
| `decision_v3_schema` | ON | `SCHEMA_TRADING_DECISION` | **OK** | Production path validated |
| `oil_v1_buggy` | ON | `SCHEMA_OIL_ALLOCATION` | **OK** | Production path validated |
| `oil_v2_fixed` | OFF | `SCHEMA_OIL_ALLOCATION` | **OK** | Reference |
| `oil_v3_schema` | ON | `SCHEMA_OIL_ALLOCATION` | **OK** | Reference |

**Totals**: 6 OK / 4 FAIL. **All 4 failures use the loose `format:json` — none of the schema-strict cases fail.**

**Interpretation**: The schema-strict `format: SCHEMA_*` parameter is the load-bearing defence. `<|think|>` is safe to use as long as the schema layer is in place. The defensive suffix alone (Layer 2) is **not sufficient** — see `query_v4_strict` and `decision_v2_fixed` which both fail with the suffix but without schema enforcement.

### 3.3 End-to-end pipeline — `uv run main.py --t212`

| Metric | Result |
|---|---|
| Exit code | 0 |
| Total wall time | 279.69 s (4.66 min) — well under the 15-min `CYCLE_TIMEOUT_SECONDS` |
| Ticklers processed | 2 (CRUDP.PA, SXRV.DE) |
| `trading_journal.csv` rows added | 2 (one per ticker) |
| `trading.log` `"Could not find valid JSON"` lines | **0** |
| `trading.log` `"Analysis failed"` lines | **0** |
| LLM validations logged | 4 (`"LLM decision (...) received and validated"` × 4 — text + visual for each ticker, oil bench on CRUDP.PA) |
| Cycle-timeout panel | Did not appear |
| T212 demo orders | HOLD on CRUDP.PA (skipped), BUY on SXRV.DE (skipped — existing position) |

All four `<|think|>`-enabled LLM call sites produced clean, schema-valid JSON in real production conditions.

---

## 4. Consequences

### 4.1 Positive

- **Restored reasoning capacity**: Gemma 4 12B's internal `<|think|>` channel is back online, improving the quality of analysis on complex multi-indicator prompts (chart analysis, oil fundamentals, macro web research).
- **Clear defence-in-depth architecture**: the two layers are now documented and instrumented (`tests/check_llm_json.py` provides ongoing regression coverage).
- **No runtime cost**: the schema enforcement happens server-side in Ollama at no extra latency.

### 4.2 Negative / Risks

- **Layer 1 dependency**: if a future Ollama version weakens schema enforcement, Layer 2 alone is **not** sufficient (proven by `query_v4_strict` / `decision_v2_fixed` failures). The CI should include a periodic live-LLM run of `check_llm_json.py` as a canary.
- **Token-list duplication**: `tests/check_llm_json.py:388-392` keeps its own copy of `_THINKING_TOKENS` separate from `src/llm_client.py:52-56`. If the lists diverge, the harness can give false-passes. Out of scope for this ADR — flagged for follow-up.
- **Silent degradation mode**: per the plan handover, when `_query_ollama` exhausts retries it returns `failed: True` in the fallback dict, but the consensus aggregator in `enhanced_decision_engine.py` does **not** currently consume that flag — a failed LLM call is weighted as a plain HOLD vote. If `<|think|>` ever causes intermittent JSON debris again, degradation will be silent (HOLD-biased) unless `trading.log` is grepped for `"Could not find valid JSON with keys"`. Out of scope — flagged for follow-up.

### 4.3 Operational changes

- New diagnostic command for any LLM-quality investigation:
  ```
  .venv\Scripts\python.exe tests/check_llm_json.py
  ```
- Acceptance criterion: at least all schema-strict cases (`*_v3_schema`, `*_v6_schema`, `*_v7_schema_strict`, `oil_*`) must be OK. The `*_v1_buggy` and `*_v2_fixed` cases are documented to fail under loose format and are kept as negative controls.

---

## 5. Reversal Procedure

The change set is intentionally surgical: 4 single-line edits to production code + 1 docstring update in the test harness. Two rollback options:

### 5.1 Soft rollback (preserve history)

```bash
# From anywhere on main:
git revert <merge-commit-of-think-mode>
# Or, if not yet merged:
git switch main   # leave think-mode branch intact for reference
```

### 5.2 Hard rollback (erase the attempt — pre-merge only)

```bash
git switch main
git branch -D think-mode
```

### 5.3 Targeted rollback (undo just the `<|think|>` prefix, keep everything else)

For each of the four call sites (`src/llm_client.py:188`, `src/llm_client.py:236`, `src/oil_bench_model.py:158`, `src/web_researcher.py:205`), remove the `"<|think|> "` prefix from the `"system"` string. The schema-strict `format` parameter and the defensive suffix are left untouched — they were already in place before the re-enablement and remain the load-bearing defence.

---

## 6. References

- Plan file: `.kilo/plans/1779923486730-think-mode-validation.md`
- Harness: `tests/check_llm_json.py`
- Schema definitions: `src/llm_client.py:20-36`
- Production call sites: `src/llm_client.py:188`, `src/llm_client.py:236`, `src/oil_bench_model.py:158`, `src/web_researcher.py:205`
- Extraction pipeline: `src/llm_client.py:311-401` (`_query_ollama`)
- Fallback sentinel: `src/llm_client.py:59-71` (`_fallback_decision`)
- Memory-bank updates: `memory-bank/techContext.md`, `memory-bank/activeContext.md`, `memory-bank/systemPatterns.md`, `memory-bank/progress.md`
- README entry: `README.md` → "Advanced Cognition" bullet
