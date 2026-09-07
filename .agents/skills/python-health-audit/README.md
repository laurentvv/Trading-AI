<p align="center">
  <img src="assets/banner.jpg" alt="python-health-audit banner" width="100%" />
</p>

# python-health-audit

[![Skill](https://img.shields.io/badge/skill-v2.0.0-5b2ddb?style=flat-square)](CHANGELOG.md)
[![CI](https://github.com/laurentvv/python-health-audit/actions/workflows/ci.yml/badge.svg)](https://github.com/laurentvv/python-health-audit/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg?style=flat-square)](LICENSE)
[![GitHub](https://img.shields.io/badge/GitHub-laurentvv%2Fpython--health--audit-181717?style=flat-square&logo=github)](https://github.com/laurentvv/python-health-audit)

> A read-only static audit skill for Python projects: one command, one Markdown report, one A–F grade.

`python-health-audit` is a **Kilo / Claude Code skill** that any developer (lead, solo, acquirer) can invoke to get an instant, opinionated health check of a Python codebase. It runs four industry-standard linters through [`uvx`](https://docs.astral.sh/uv/) — no virtualenv, no install — and writes a single `python_health_report.md` at the root of the audited project.

---

## Why

| Without the skill | With the skill |
|---|---|
| Remember 4 CLI invocations + flags + exclusions | One natural-language request (EN or FR) |
| Stitch outputs from Ruff, Vulture, Radon, Pylint by hand | One Markdown report with deterministic A–F grade |
| Risk of `ruff --fix` / `pylint --fix` mutating source | Strict read-only guarantee (even `.ruff_cache` is suppressed) |
| Ad-hoc structure every time | Stable template + machine-readable YAML header + 3-item action plan |
| No memory between audits | "Since last audit" delta + grade history carried forward |

## Install

The skill works in any agent runtime that supports the [`skills`](https://skills.sh) format — Kilo, Claude Code, Roo Code, Cline, Cursor, Continue, etc.

**Recommended (one-liner):**

```powershell
npx skills add laurentvv/python-health-audit
```

The CLI accepts the GitHub shorthand `owner/repo` and auto-discovers `SKILL.md` at the repo root.

**Per-runtime:**

```powershell
# Kilo
kilo skills add laurentvv/python-health-audit

# Claude Code
claude skills add laurentvv/python-health-audit

# Global install (shared across all projects on the machine, symlinked)
npx skills add -g laurentvv/python-health-audit
```

## Usage

Open your agent in the Python project you want to audit and ask, in plain English (or French):

```
I'm taking over a 3-year-old Flask project. Can you give me a full health check
before I start refactoring? The code lives in ./backend. Tell me where the
technical debt is.
```

The agent will:

1. `cd` into the target project (or use the `workdir` parameter)
2. Run sequentially, capturing all stdout + stderr in memory:
   - `uvx ruff check . --isolated --no-cache --select E4,E7,E9,F` — local dead code (unused imports, unused vars). The rule set is pinned so counts stay comparable across Ruff versions; `--no-cache` keeps the audit strictly read-only.
   - `uvx vulture . --min-confidence 60` — global dead code (unused functions, classes, variables — confidence 60 is where they live; 90 is imports only)
   - `uvx radon cc . -a -s -nc` — cyclomatic complexity hotspots (ranks C+)
   - `uvx radon mi . -s` — Maintainability Index per file (numeric scores)
   - `uvx radon raw . -s` — SLOC, to normalize thresholds by project size
   - `uvx pylint --recursive=y --disable=all --enable=duplicate-code ...` — copy-paste detection (portable, no `find` substitution)
3. Compute an **A–F grade** from per-kLOC densities (deterministic heuristic, see below)
4. Write `python_health_report.md` at the root of the audited project, with a machine-readable YAML header and, if a previous report exists, a *"Since last audit"* delta

That's it. No source files are touched — not even a cache directory.

**Extended audit** (opt-in — just ask for it): additionally checks dependency
vulnerabilities (`pip-audit`), docstring coverage (`interrogate`), a
TODO/FIXME census and git churn hot spots. Extended findings live in report
section 6 and never affect the grade.

## Example output

Excerpt from a real v2 audit of a 101-file / 10.6k-SLOC project
(full report at the root of the audited project):

```markdown
---
audit:
  date: 2026-09-03 15:12
  grade: F
metrics:
  ruff: 189
  vulture: 133
  hotspots_cd: 35
  hotspots_ef: 3
  avg_mi: 64.7
  duplication: 156
  sloc: 10602
  ruff_density: 17.8
  vulture_density: 12.5
---

# Python Health Report — generator-assets

Generated on 2026-09-03 15:12 by python-health-audit (v2).

## 1. Executive Summary
- Global grade: F
- Reason: Grade F assigned — 1 F-rank hotspot (`lancer_mode_interactif`
  in `main.py`, cyclomatic complexity 69) and 156 duplicated blocks (R0801)
  concentrated in `scripts/`, despite an average MI of 64.7 and Ruff
  density at 17.8/kLOC (D-level).

## 2. Dead Code
### 2.1 Local — Ruff
189 findings (145 unused imports F401 + 13 f-string F541 + 13 E402 +
9 E741 + 5 unused variables F841 + …). All fixable mechanically; no
auto-fix applied per audit policy.

### 2.2 Global — Vulture
133 entries at confidence ≥ 60 %, e.g. unused function
`verifier_mpfb_disponible` (core/mpfb_ops.py:74).

> ⚠️ Vulture produces false positives by construction (global static
> detection). Verify each entry before removal.

## 3. Complexity Hotspots (Radon)
| Rank | File:Line | Block | cc |
|------|-----------|-------|----|
| **F** | main.py:62 | `lancer_mode_interactif` | **69** |
| E | core/clothes_catalog.py:79 | `construire_catalogue_vetements` | 35 |
| … 36 more C/D blocks … |

## 4. Code Duplication (Pylint)
156 duplicated blocks (R0801), concentrated in `scripts/`:
test_bake_and_render ↔ test_clean_render / test_soft_face / …
(Blender setup/bake boilerplate, up to ~50 identical lines per pair).

## 5. Recommended Action Plan
1. **Split `lancer_mode_interactif` (main.py:62, cc 69, rank F)**.
2. **Factor the Blender script boilerplate in `scripts/`** into shared
   helpers — kills the bulk of the 156 R0801 duplications.
3. **Purge the 145 unused imports (F401)** — mechanical, −77 % of findings.
```

## Grading heuristic

The grade is computed deterministically from per-kLOC densities. First
matching rule wins (A → F); **F is the catch-all**, and MI intervals are
half-open, so every project gets exactly one grade — no undefined edge cases.

Collected metrics: `ruff` (findings), `vulture` (dead-code entries at
confidence ≥ 60), `cd`/`ef` (Radon blocks ranked C/D and E/F), `mi` (mean
numeric Maintainability Index), `dup` (Pylint R0801 count), `sloc`.

Size normalization: `dk = max(sloc / 1000, 1)` (effective kLOC, floored),
`ruff_d = ruff / dk`, `vulture_d = vulture / dk`.

| Grade | Criteria (all required on the same row) |
|-------|------------------------------------------|
| **A** | ruff = 0 **AND** cd + ef = 0 **AND** mi ≥ 80 **AND** dup = 0 **AND** vulture_d ≤ 0.5 |
| **B** | ruff_d ≤ 1 **AND** ef = 0 **AND** cd ≤ 3 **AND** mi ≥ 65 **AND** dup ≤ 1 **AND** vulture_d ≤ 2 |
| **C** | ruff_d ≤ 5 **AND** ef = 0 **AND** cd ≤ 10 **AND** mi ≥ 50 **AND** dup ≤ 5 **AND** vulture_d ≤ 5 |
| **D** | no F hotspot **AND** mi ≥ 30 **AND** ruff_d ≤ 20 **AND** vulture_d ≤ 15 **AND** dup ≤ 15 |
| **F** | otherwise (≥ 1 F hotspot, or mi < 30, or ruff_d > 20, or vulture_d > 15, or dup > 15) |

Design notes:

- Findings and dead symbols scale with project size, so their thresholds are
  **densities**; hotspots and duplication are structural and stay absolute.
- A single F-rank block (cyclomatic complexity ≥ 41) is an automatic F — one
  god-function is a due-diligence red flag regardless of the rest.
- The v1 heuristic had a rule-ordering bug (a project with an E hotspot
  graded C whenever Ruff ≤ 20) and MI boundary gaps (49.5 matched no rule);
  both are fixed in v2 — see [CHANGELOG](CHANGELOG.md).

## Constraints (what the skill will NOT do)

1. **Strict read-only** — no `Edit` / `Write` / `Set-Content` on `.py` files,
   no cache written into the project (Ruff runs with `--no-cache`). The only
   artifact written is the report itself.
2. **No auto-fix** — never invokes `ruff --fix`, `autoflake`, `autopep8`,
   `black`, or `pylint --fix`.
3. **Silent execution** — no progress messages in chat. Only the path of the
   generated report is returned.
4. **Grade stability** — extended-mode findings never influence the grade.

## Benchmark & testing

- 5 eval cases in [`evals/evals.json`](evals/evals.json) (EN + FR prompts,
  duplication focus, extended mode, re-audit with previous report).
- [`evals/fixtures/debt-ridden-project/`](evals/fixtures/debt-ridden-project/) —
  a committed synthetic codebase with planted debt (unused imports, dead
  symbols, rank-D/E hotspots, duplicated block, TODOs, outdated dependency)
  and its expected metrics. Expected grade: **D**.
- Historical benchmark (v1.0, 2 prompts × 2 configurations): with skill
  **100 %** (13/13) vs baseline 29 % (4/13) — see the
  [v1.0 release artifacts](https://github.com/laurentvv/python-health-audit/releases/tag/v1.0).
- **CI** validates the JSON/frontmatter and runs every tool invocation
  against the fixture on every push and weekly (Ubuntu + Windows), guarding
  the intentionally unpinned `uvx` tools against upstream breakage.

## Repository structure

Single-skill repo, flat layout (auto-discovered by `npx skills add`):

```
python-health-audit/
├── README.md
├── CHANGELOG.md
├── LICENSE
├── .gitignore
├── SKILL.md          # frontmatter + role / objective / steps / grading / format / constraints
├── assets/
│   └── banner.jpg    # GitHub header banner
├── .github/workflows/
│   └── ci.yml        # validate + weekly smoke test against the fixture
└── evals/
    ├── evals.json    # 5 benchmark test cases
    └── fixtures/
        └── debt-ridden-project/   # synthetic codebase with planted debt
```

## Compatibility

| Runtime | Status |
|---|---|
| Kilo | ✅ supported |
| Claude Code | ✅ supported (extra hardening: `allowed-tools` frontmatter restricts to Bash/Read/Write/Glob/Grep) |
| Roo Code, Cline, Cursor, Continue, … | ✅ any agent that reads the `SKILL.md` frontmatter format |

Required on the host machine:

- [`uv`](https://docs.astral.sh/uv/) (provides the `uvx` command) — **or** `pipx` as fallback
- PowerShell 7+ or Git Bash on Windows, bash on Linux/macOS

Tool versions are intentionally **unpinned** (fresher rules); comparability
comes from the pinned Ruff rule set (`--select E4,E7,E9,F`) and drift is
caught by the weekly CI smoke test.

## Contributing

This is a single-skill repo (flat layout). To iterate:

1. Edit `SKILL.md` at the repo root.
2. Add / modify test prompts in `evals/evals.json`; extend the fixture in
   `evals/fixtures/debt-ridden-project/` if you need new planted debt (and
   update its README with the expected metrics).
3. `ci.yml` runs locally-verifiable assertions for every tool command — keep
   them green.
4. Re-run benchmarks with the [`skill-creator`](https://github.com/Kilo-Org/skills/tree/main/skill-creator) workflow.
5. Open a PR — describe the change in the skill's behavior, not just the file diff.

## License

MIT — see [LICENSE](LICENSE).
