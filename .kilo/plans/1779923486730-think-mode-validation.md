# Plan: Technical validation of Gemma "think mode" re-enablement

## Context — what "think mode" actually means in this repo

The repo has **no runtime feature** called "think mode". The term refers to the Gemma 4 12B LLM's `<|think|>` system-prompt token, which activates the model's internal reasoning channel. Per `README.md:98`, `memory-bank/techContext.md:11`, and the full bug history in `tests/check_llm_json.py:539`, this token was **deliberately removed** in May 2026 because it caused `<|channel>thought` debris to leak into JSON responses, breaking every `_query_ollama` caller with `"Could not find valid JSON with keys ['query']"` errors.

The user has explicitly confirmed the intent: **re-enable `<|think|>` in the four production system prompts**, run the LLM test harness (adapted to the new reality), then run `uv run main.py --t212` against the demo T212 account (`T212_ENV="demo"` in `.env.t212`).

Pre-flight checks already performed (read-only):
- Working branch: **`think-mode`** (créée par l'utilisateur, propre à l'exception de ce fichier de plan). Tous les edits et commits de la validation seront isolés sur cette branche — aucun impact sur `main` / production.
- `uv 0.9.9`, Ollama `0.30.6`, venv present at `.venv\Scripts\python.exe`
- Ollama reachable on `localhost:11434`, 16 models loaded including `hf.co/unsloth/gemma-4-12b-it-GGUF:Q4_K_M` (the production model)
- `tests/test_llm_client.py`, `tests/test_llm_prompts.py`, `tests/test_oil_bench_model.py` all collect cleanly via `.venv\Scripts\python.exe -m pytest`
- T212 demo credentials are present in `.env.t212`

## Scope of changes

### A. Re-enable `<|think|>` in the four production system prompts

Prepend the literal string `"<|think|> "` to the `"system"` value of these four payload sites:

| # | File | Line | Caller |
|---|------|------|--------|
| 1 | `src/llm_client.py` | 188 | `get_llm_decision` (text LLM trading decision) |
| 2 | `src/llm_client.py` | 236 | `get_visual_llm_decision` (vision chart decision) |
| 3 | `src/oil_bench_model.py` | 158 | `OilBenchModel._query_llm` (oil allocation) |
| 4 | `src/web_researcher.py` | 205 | `generate_search_query` (DuckDuckGo query) |

Concretely, e.g. site #1 becomes:
```python
"system": "<|think|> You are an expert financial analyst. Your task is to analyze market data and news to provide a trading decision in a valid JSON format. Output ONLY the JSON object requested — never add a 'thought' key.",
```
The defensive suffix (`"... never add a 'thought' key."`) is intentionally kept — it is part of the **schema-layer** defence and is orthogonal to the `<|think|>` token. The user asked for the token to be re-enabled, not for the schema defence to be torn down. If the user later wants the full revert, the schema strings can also be reverted in a follow-up.

### B. Adapt `tests/check_llm_json.py` to the new baseline

`tests/check_llm_json.py` is the only file that actually exercises the live Gemma model with `<|think|>`. As written it treats the `*_v1_buggy` cases as ordinary passing tests and exits non-zero if any fails. After the re-enablement, those cases are the **production baseline**, not regressions — they should be expected to succeed. The adaptation is a **single comment-only change** to clarify intent:

- Add a top-of-file note stating that `<|think|>` has been re-enabled and the `v1_buggy` cases are now the **production** path; the `v2_fixed` / `v3_schema` cases remain as the documented fallback for comparison.
- No assertions change — the script already reports OK/FAIL per case and exits 0 only if everything passes. The acceptance criterion is "no extraction errors across all cases", which is exactly what the existing logic tests.

If a stricter positive check is wanted, add a new `unittest`-style assertion in a new `tests/test_think_mode_enabled.py`:
```python
def test_system_prompts_contain_think_token():
    from src.llm_client import get_llm_decision, get_visual_llm_decision
    # inspect the source via inspect.getsource and assert "<|think|>" appears in the
    # system string for all four call sites (text, visual, oil, web_research).
```
This is optional but cheap; it gives the test suite a deterministic, mock-free assertion that the re-enablement is in place.

### C. Test-suite execution order

1. **Mocked LLM unit tests** (deterministic, no Ollama needed):
   ```
   .venv\Scripts\python.exe -m pytest tests/test_llm_client.py tests/test_llm_prompts.py tests/test_oil_bench_model.py -v
   ```
   These use `unittest.mock.patch` on `requests.post`, so they are unaffected by the `<|think|>` change. **Expected:** 9 / 9 pass. If any fails, abort — the re-enablement has broken an unrelated contract.

2. **Live Gemma "think mode" harness** (requires `ollama serve` running, ~3-6 min):
   ```
   .venv\Scripts\python.exe tests/check_llm_json.py
   ```
   Exercises all 11 cases (3 `*_v1_buggy`, 7 fixed/schema variants, no skip). **Expected outcome for "validation successful":** the harness reports all cases OK — meaning the Gemma model now produces clean JSON even with `<|think|>` active (because the strict `SCHEMA_*` `format:` parameter and the defensive system-prompt suffix still constrain the output).
   
   **Alternative expected outcome (still "validation complete", just informative):** the `*_v1_buggy` cases fail with the original May-2026 JSON-debris symptom — this proves the defect is reproducible and that the schema defence is what's actually keeping production alive. In this case the harness exits non-zero; the plan treats this as a **successful negative-result validation** rather than a failure, and the summary will be reported to the user.

3. **Pre-flight health checks** before the final `--t212` run:
   ```
   .venv\Scripts\python.exe -c "from src.llm_client import check_ollama_health; print(check_ollama_health())"
   .venv\Scripts\python.exe -c "from t212_executor import load_portfolio_state; print(load_portfolio_state('CRUDP'))"
   ```
   Confirms both upstreams (Ollama + T212 demo API) are reachable from the current shell.

### D. Final execution: `uv run main.py --t212`

```
uv run main.py --t212
```

This runs the full pipeline for the two default tickers (`CRUDP.PA`, `SXRV.DE`) on the T212 **demo** account (`T212_ENV="demo"`). The pipeline calls both LLM call sites (#1 text and #2 visual at minimum) and possibly #4 (`web_researcher`) and #3 (only for `CRUDP.PA`/oil — note `CRUDP.PA` resolves to WTI via `get_t212_ticker`). All four `<|think|>`-enabled paths are therefore exercised end-to-end.

**Acceptance criteria for the final step:**
- Process exits 0 (no unhandled exception in `main.py`).
- `trading.log` contains no `"Analysis failed for ..."` lines and no `"Could not find valid JSON with keys"` errors.
- `trading_journal.csv` gets two new rows (one per ticker).
- The final rich `Panel` summary prints cleanly for both tickers (BUY/SELL/HOLD + confidence + risk level + T212 demo capital/position).
- Cycle-timeout panel must **not** appear (would indicate the 15-min `CYCLE_TIMEOUT_SECONDS` was hit, which on a 12B Gemma model with thinking enabled is plausible but not blocking).

## Reversibility

The change set is intentionally surgical: 4 single-line edits to production code + an optional clarifying comment in the test harness. Comme tout est isolé sur la branche `think-mode`, deux options de retour-arrière sont disponibles :
- **Soft** (garder l'historique pour archivage) : `git switch main` — la branche `think-mode` reste disponible pour référence ultérieure.
- **Hard** (effacer la tentative) : `git switch main && git branch -D think-mode` — supprime la branche et tous les changements.

Aucune modification hors-branche n'est effectuée pendant la validation ; `main` reste sur l'état défendu actuel (sans `<|think|>`).

## Risks and mitigations

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| Gemma still leaks `<|channel>thought` debris into JSON with `<|think|>` re-enabled | High (this is the historical defect) | The strict `SCHEMA_*` `format:` parameter is still enforced by Ollama server-side, which is the actual safety net. If debris leaks anyway, `_query_ollama` falls back to HOLD after 3 retries — no real money moves on demo regardless. |
| `main.py --t212` cycle times out (>15 min) due to slower reasoning | Medium | Already handled in `main.py:380-404` — `cancel_event` is set and the orphan thread cannot place a T212 order even if it completes after the timeout. |
| T212 demo API quota or transient error | Low | Demo account is rate-limit-tolerant; failures surface in `trading.log` and are non-fatal. |
| Tests that previously mocked the system-prompt string break | None | Inspected `test_llm_client.py` (12-79): all assertions are on the parsed **return value**, not on the prompt payload. `test_llm_prompts.py` only inspects `construct_llm_prompt` output, not the `system` field. |

## Definition of done

- [ ] Quatre modifications de ré-activation de `<|think|>` appliquées et visibles via `git diff` sur la branche `think-mode`.
- [ ] `pytest tests/test_llm_client.py tests/test_llm_prompts.py tests/test_oil_bench_model.py` → tous réussissent.
- [ ] `python tests/check_llm_json.py` s'exécute jusqu'au bout et le résumé est imprimé (résultat rapporté honnêtement, qu'il soit tout-OK ou que les `v1_buggy` échouent).
- [ ] `uv run main.py --t212` termine avec le code 0 et produit deux nouvelles lignes dans `trading_journal.csv` sans erreurs `"Could not find valid JSON"` dans `trading.log`.
- [ ] Résumé des constatations fourni à l'utilisateur (un court paragraphe : la configuration avec pensée activée a-t-elle survécu à la défense par schéma strict, oui/non, avec des preuves issues du résumé de `check_llm_json.py`).
- [ ] **Commit isolé sur la branche `think-mode`** (si l'utilisateur le demande explicitement — sinon les changements restent en working tree pour revue).
