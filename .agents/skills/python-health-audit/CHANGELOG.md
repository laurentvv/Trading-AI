# Changelog

All notable changes to this project are documented here.
Format: [Keep a Changelog](https://keepachangelog.com/); versions mirror git tags.

## [2.0.0] — 2026-09-03

Calibrated against a synthetic fixture (`evals/fixtures/debt-ridden-project`)
and a real ~10.6k-SLOC / 101-file codebase. **Breaking**: grade semantics and
report format change — grades computed by v1 and v2 are not comparable.

### Fixed

- **Grading**: rule C's `OR` no longer swallows E-rank hotspots. A project with
  an E hotspot now grades **D** (as intended) instead of C whenever Ruff ≤ 20,
  which made the D criterion "≥ 1 E hotspot" effectively unreachable.
- **Grading**: MI intervals are now half-open (`[50, 65)`, `[30, 50)`, …). A MI
  of 49.5 previously matched no rule at all (undefined grade).
- **Grading**: Vulture and duplication now influence every grade band instead
  of only F and A respectively.
- **Read-only guarantee**: Ruff runs with `--no-cache` — it previously wrote
  `.ruff_cache/` into the audited project, contradicting the "no intermediate
  file" promise.
- `radon mi` now runs with `-s` so numeric scores are actually available to
  compute the average MI (letters only before).
- Pylint step: replaced the fragile `$(find … | grep …)` substitution (breaks
  on paths with spaces, hits the Windows argument-length limit on large repos,
  required a separate PowerShell variant) with `--recursive=y` — one portable
  command. Exclusions use `--ignore` base names: the `--ignore-paths` regex
  variant silently failed to exclude `.venv` on Windows (caught on a real
  project whose virtualenv holds 2014 files — the audit was analyzing 21×
  more files than intended).
- `evals.json`: eval 1 expected "5 sections" but listed 6 names (2.1/2.2 are
  subsections).

### Added

- **Machine-readable YAML header** in the report (grade, metrics, densities,
  history) for CI diffing and tooling.
- **"Since last audit"** comparison: a pre-existing report is read (read-only)
  and its grade, metric deltas and history are carried forward.
- **Size normalization**: Ruff and Vulture thresholds are now densities per
  kLOC (`dk = max(SLOC/1000, 1)`), so large codebases are no longer
  over-penalized by absolute counts and tiny toxic projects no longer slide.
- **Opt-in extended pass** (explicit request only): dependency vulnerabilities
  (`pip-audit`), docstring coverage (`interrogate`), TODO/FIXME census, git
  churn top-10 — reported in section 6 and never affecting the grade.
- **Committed eval fixture** `evals/fixtures/debt-ridden-project/` with planted
  debt (unused imports, dead symbols, rank-D/E hotspots, duplicated block,
  TODOs, outdated dependency) — reproducible benchmarks and CI smoke tests.
- **GitHub Actions CI**: JSON/frontmatter validation on every push, plus a
  weekly smoke test that runs every tool invocation against the fixture to
  catch upstream breakage early (tool versions stay unpinned).
- **5 eval cases** (was 2): French prompt, extended mode, re-audit with
  previous report.
- `allowed-tools` frontmatter hardening (Bash/Read/Write/Glob/Grep — no Edit).
- This CHANGELOG.md.

### Changed

- Ruff runs with `--select E4,E7,E9,F --isolated`: the rule set is pinned and
  the audited project's own Ruff config is ignored, so counts are comparable
  across projects **and across Ruff versions** (2026 defaults had silently
  expanded to I/SIM, inflating counts ~4× on real code).
- Vulture `--min-confidence` lowered from 80 to 60: at 80 only unused imports
  (90 % confidence) survive; dead functions/classes/variables sit at 60 % and
  were invisible. The false-positive warning stays prominent.
- Adaptive Pylint timeout ladder (120 s flat / 300 s above 50 Python files /
  600 s above 200) kept as headroom: with `.venv` properly excluded, a real
  101-file project audits in ~2 s — the timeouts only matter if exclusions
  ever regress again.
- `radon raw` (SLOC) added as core step 3c to feed the size normalization.
- Report template versioned (v2) with optional section 6.

## [1.1.0] — 2026-06-27

- Flattened repo layout (single-skill auto-discovery by `npx skills add`).
- Shorter one-liner install instructions.
- Renamed report file to `python_health_report.md` (English).
- Fixed Vulture/Ruff/Radon exclude patterns.

## [1.0.1] — 2026-06-14

- Added README, LICENSE (MIT), `.gitignore` — clean repo for skills.sh.
- Pointed install URLs and release link to `laurentvv/python-health-audit`.

## [1.0.0] — 2026-06-10

- Initial skill: read-only audit (Ruff, Vulture, Radon cc/mi, Pylint
  duplicate-code), deterministic A-F grade, 5-section Markdown report,
  3-item action plan.
