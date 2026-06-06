# Changelog

All notable changes to this project are documented in this file.
The format is loosely based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

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
