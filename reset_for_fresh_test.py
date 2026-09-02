#!/usr/bin/env python3
"""
MAX reset — wipe everything learned/recorded for a truly virgin restart.

WHY THIS EXISTS
---------------
After the June/July 2026 audit (capital-protection fixes on main), the safest
way to validate the corrected system is to start from a clean slate: no stale
model trained with the old biased code, no corrupted portfolio state, no
history polluted by the DEMO run. A partial reset left room for a stale pickle
to be reloaded by hash, silently undoing the fixes. This script does a COMPLETE
wipe so the next cycle re-downloads data and retrains every model from scratch.

Run it once on PROD/DEMO after pulling, before launching the fresh test:

    uv run python reset_for_fresh_test.py            # interactive confirm
    uv run python reset_for_fresh_test.py --dry-run  # preview only
    uv run python reset_for_fresh_test.py --yes      # no prompt (automation)

    # PROD reset complet (logs_prod/ = CWD du scheduler, doit etre inclus) :
    uv run python reset_for_fresh_test.py --yes --include-logs-prod

WIPE STRATEGY — pattern-based (robust, maintenance-free)
--------------------------------------------------------
Instead of an explicit file list (which drifts out of sync with the code —
the July 2026 version missed 7 runtime artifacts: scheduler.log,
analyse_morning.log, weekend_council.log, performance_dashboard.png,
morning_brief/output/, docs/council_reports/, ...), this script wipes by
PATTERN. Anything gitignored at the repo root is a runtime artifact by
definition and is eliminated automatically.

WHAT IT WIPES (everything learned/recorded)
-------------------------------------------
  - data_cache/   ENTIRE tree: prices, models, EIA, tensortrade, finacumen,
                  macro, search_queries, gemini_quota.db... ALL of it.
  - *.csv *.json *.pkl *.pickle *.db *.db-shm *.db-wal *.log *.png
                  at the repo ROOT — catches every runtime artifact present
                  and future (state, DBs, journals, logs, dashboards).
  - morning_brief/output/   generated briefs + logs (recreated next run)
  - docs/council_reports/   weekly council reports (regenerated next Saturday)
  - backtest_results/ trading_data/ logs/ models/   gitignored, if present

WHAT IT PRESERVES (whitelist — never touched)
--------------------------------------------
  - .env / .env.t212            credentials — NEVER purged
  - .venv / .git                environment + version control
  - logs_prod/                  prod log archive (read-only reference)
  - memory-bank/                deterministic state + docs
  - src/ tests/ docs/ scripts/  source code + docs
  - *.py *.md *.toml *.yaml *.bat *.lock   source/config files at root

LOGS_PROD/ RUNTIME ARTIFACTS (scheduler PROD CWD — IMPORTANT)
-------------------------------------------------------------
On PROD the scheduler is launched FROM logs_prod/, so the live runtime
artifacts (DBs, t212_portfolio_state.json, trading_journal.csv, data_cache/,
scheduler.lock, logs) live INSIDE logs_prod/ — which the whitelist above
would otherwise preserve. The 2026-08-19 reset missed exactly that:
logs_prod/model_performance.db survived with pre-reset data and polluted the
adaptive weights for 5 days (incident fixed manually on 2026-08-24).

This script now DETECTS runtime artifacts inside logs_prod/ and requires an
explicit choice:
  --include-logs-prod   wipe them too (backup + remove) — PROD migration path
  --keep-logs-prod      preserve them (DEV: logs_prod/ is an audit snapshot)
Interactive mode asks; `--yes` without either flag REFUSES to run (exit 2).
`.md` files (audit reports) inside logs_prod/ are never touched.

GEMINI QUOTA LEDGER (note for PAID PROD)
----------------------------------------
By DEFAULT, data_cache/gemini_quota.db is WIPED (demo mode — start fresh).
This tracks the rolling 30-day EUR cost budget of the Gemini gateway; wiping
it makes the gateway think it has a fresh full budget.

  - DEMO / fresh test  : default is fine (no real spend at stake).
  - PAID PROD          : pass --keep-quota-ledger to preserve the budget and
                         avoid a potential OVERSPEND.

SAFETY
------
- Nothing is deleted without confirmation (unless --yes).
- Everything is MOVED to a timestamped backup (reset_backup/) preserving the
  relative path tree, so the operation is fully reversible.
- Idempotent: re-running on an already-clean tree is a no-op.
- Windows-safe: pathlib + shutil only, no shell rm/del.
"""

from __future__ import annotations

import argparse
import datetime as dt
import shutil
from pathlib import Path

from reset_lib import confirm, dry_run_or_confirm

REPO_ROOT = Path(__file__).resolve().parent


def _backup_root() -> Path:
    """Backup root, resolved at CALL time (REPO_ROOT is monkeypatchable in tests)."""
    return REPO_ROOT / "reset_backup"

# ---------------------------------------------------------------------------
# KEEP WHITELIST — these directories are NEVER touched. The wipe only ever
# operates on: (a) the repo ROOT for pattern-matched files, and (b) the
# explicit WIPE_DIRS below. KEEP_PATHS is here for clarity + a final safety
# guard in _safe_to_wipe().
# ---------------------------------------------------------------------------
KEEP_PATHS = {
    ".git",                 # version control
    ".venv",                # python environment
    ".env", ".env.t212",    # credentials — NEVER purge
    ".env.example", ".env.t212.example",
    "logs_prod",            # prod log archive (read-only reference)
    "reset_backup",         # our own backup output
    "memory-bank",          # deterministic state / docs
    "src", "tests", "docs", "scripts", "morning_brief", "i18n",
    ".agents", "vendor", "assets",
    ".pytest_cache", ".ruff_cache", "__pycache__",
    ".kilo", ".kilocode", ".qwen",
}

# ---------------------------------------------------------------------------
# ROOT-LEVEL FILE PATTERNS — gitignored extensions (from .gitignore lines
# 36-44). Any file at the repo root matching one of these is a runtime
# artifact and gets backed up + removed. This is robust to FUTURE artifacts:
# when the code starts writing a new "macro_cache.json" or "regime.db", it is
# caught automatically without editing this script.
# ---------------------------------------------------------------------------
WIPE_ROOT_EXTENSIONS = {
    ".csv",                 # trading_journal.csv
    ".json",                # t212_portfolio_state.json, scheduler_config.json
    ".pkl", ".pickle",      # serialized models at root (rare but possible)
    ".db",                  # trading_history.db, model_performance.db, performance_monitor.db
    ".db-shm", ".db-wal",   # SQLite sidecar files
    ".log",                 # trading.log, scheduler.log, analyse_morning.log, weekend_council.log
    ".png",                 # enhanced_*.png, performance_dashboard.png
}

# ---------------------------------------------------------------------------
# WIPE DIRS — directories whose ENTIRE content is moved to backup.
#   - data_cache/    handled by _wipe_data_cache (with --keep-quota-ledger
#                    support to preserve gemini_quota.db if requested).
#   - the rest       handled by _wipe_generic_dir.
# Each of these is regenerated automatically by the code on the next run.
# ---------------------------------------------------------------------------
WIPE_DATACACHE_DIR = "data_cache"
WIPE_DIRS = [
    "morning_brief/output",  # regenerated by morning_brief.py (OUTPUT_DIR.mkdir)
    "docs/council_reports",  # regenerated by weekend_council.py (mkdir parents=True)
    "backtest_results",      # gitignored, empty/absent on DEV but possible on PROD
    "trading_data",          # gitignored
    "logs",                  # gitignored
    "models",                # gitignored
]

# File preserved inside data_cache/ ONLY when --keep-quota-ledger is passed.
KEEP_QUOTA_LEDGER_NAME = "gemini_quota.db"

# ---------------------------------------------------------------------------
# logs_prod/ — scheduler CWD on PROD. The live runtime artifacts live THERE
# (DBs, state, journal, data_cache/, lock), not at the repo root. The whitelist
# above would preserve them, so they are detected explicitly (see incident
# 2026-08-24: model_performance.db survived the 2026-08-19 reset).
# ---------------------------------------------------------------------------
LOGS_PROD_DIR = "logs_prod"
LOGS_PROD_LOCK_NAME = "scheduler.lock"


def _is_runtime_file_name(name: str) -> bool:
    """Heuristique partagée : ce fichier est-il un artefact runtime ?

    Utilisée pour la racine du repo ET pour logs_prod/ : extensions gitignorées
    (*.csv/*.json/*.db/... + composés .db-shm/.db-wal), sauvegardes de DB
    (.db.bak-<date>), et le verrou du scheduler. Les .md/.lock génériques
    (uv.lock, rapports d'audit) ne correspondent PAS.
    """
    lowered = name.lower()
    suffix = Path(name).suffix.lower()
    compound = ""
    if "." in name:
        stem_and_ext = name.rsplit(".", 2)
        if len(stem_and_ext) >= 3:
            compound = ("." + stem_and_ext[-2] + "." + stem_and_ext[-1]).lower()
    return (
        suffix in WIPE_ROOT_EXTENSIONS
        or compound in WIPE_ROOT_EXTENSIONS
        or ".db.bak-" in lowered
        or name == LOGS_PROD_LOCK_NAME
    )


def _collect_logs_prod_runtime_artifacts() -> list[Path]:
    """Artefacts runtime présents dans logs_prod/ (CWD du scheduler PROD).

    Retourne les chemins absolus : fichiers runtime à la racine de logs_prod/
    + les sous-répertoires régénérables (data_cache/, morning_brief/output/,
    docs/council_reports/...). Les .md (rapports d'audit) et tout le reste
    sont ignorés. Vide si logs_prod/ n'existe pas ou est propre.
    """
    base = REPO_ROOT / LOGS_PROD_DIR
    if not base.is_dir():
        return []
    targets: list[Path] = []
    for item in sorted(base.iterdir()):
        if item.is_file() and _is_runtime_file_name(item.name):
            targets.append(item)
    # data_cache/ est géré à part pour la racine (WIPE_DATACACHE_DIR) — ici on
    # le regroupe avec les autres dirs régénérables présents dans logs_prod/.
    for d in [WIPE_DATACACHE_DIR] + WIPE_DIRS:
        sub = base / d
        if sub.exists():
            targets.append(sub)
    return targets


def _backup_timestamp() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def _move_to_backup(target: Path, backup_dir: Path) -> bool:
    """Move target into backup_dir, PRESERVING its path relative to the repo
    root. This keeps the backup restorable: a file at data_cache/eia/x.parquet
    is backed up to <backup>/data_cache/eia/x.parquet (not flattened).

    On Windows, a file held open by another process (e.g. scheduler.log kept
    open by the long-running schedule.py via its RotatingFileHandler) cannot
    be renamed or unlinked — shutil.move raises PermissionError [WinError 32].
    In that case we fall back to copy+truncate: the bytes are preserved in the
    backup, and the live file is truncated in place so the holder keeps its
    open handle on a now-empty file (no crash, equivalent to a reset).
    """
    if not target.exists():
        return False
    try:
        rel = target.resolve().relative_to(REPO_ROOT)
    except ValueError:
        # target outside repo: fall back to flat name to be safe
        rel = Path(target.name)
    dest = backup_dir / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    # If a same-named item already exists in the backup, append a suffix.
    if dest.exists():
        dest = backup_dir / f"{rel}_{_backup_timestamp()}"
        dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        shutil.move(str(target), str(dest))
    except PermissionError:
        # File locked by another process (typical: scheduler.log on Windows
        # while schedule.py is running). Copy contents to backup, then
        # truncate the live file in place so it is effectively reset without
        # needing to close the foreign handle.
        shutil.copy2(str(target), str(dest))
        try:
            target.write_bytes(b"")
        except PermissionError:
            # Even truncation is blocked — still report success since the
            # content was backed up; the foreign process owns the file.
            pass
    return True


def _safe_to_wipe(target: Path) -> bool:
    """Final safety guard: refuse to wipe anything inside a KEEP_PATHS dir,
    UNLESS the path itself (or a parent) is explicitly listed in WIPE_DIRS.

    Why the exception: KEEP_PATHS protects top-level dirs like `docs/` and
    `morning_brief/`, but WIPE_DIRS legitimately targets subdirs of those
    (docs/council_reports, morning_brief/output) which are runtime-generated.
    Without this exception, _wipe_generic_dir silently skips them — the July
    2026 PROD reset reported success but left them on disk.
    """
    try:
        rel = target.resolve().relative_to(REPO_ROOT)
    except ValueError:
        return False
    parts = rel.parts
    if not parts:
        return False
    # Explicit WIPE_DIRS entry (or a parent of target) always wins.
    for wipe_dir in WIPE_DIRS:
        wd = Path(wipe_dir)
        if rel == wd or wd in rel.parents:
            return True
    # Otherwise, the top-level dir must not be in the keep set.
    if parts[0] in KEEP_PATHS:
        return False
    return True


# ---------------------------------------------------------------------------
# Root-level pattern wipe
# ---------------------------------------------------------------------------
def _collect_root_runtime_files() -> list[Path]:
    """Return every file at the repo ROOT whose name matches the runtime
    heuristics (see _is_runtime_file_name). Does NOT recurse into
    subdirectories (those are handled by WIPE_DIRS / KEEP_PATHS)."""
    found = []
    for item in REPO_ROOT.iterdir():
        if not item.is_file():
            continue
        if _is_runtime_file_name(item.name):
            found.append(item)
    return sorted(found)


def _wipe_root_files(files: list[Path], backup_dir: Path, dry: bool) -> int:
    moved = 0
    for f in files:
        if not _safe_to_wipe(f):
            continue
        if dry:
            moved += 1
            continue
        if _move_to_backup(f, backup_dir):
            moved += 1
    return moved


# ---------------------------------------------------------------------------
# data_cache/ wipe (preserving gemini_quota.db optionally)
# ---------------------------------------------------------------------------
def _wipe_data_cache(backup_dir: Path, dry: bool, keep_quota: bool) -> tuple[int, list[str], int]:
    """Full wipe of data_cache/. Returns (dir_moved, kept_names, file_count).

    If keep_quota is True, gemini_quota.db is staged aside, the rest is moved
    wholesale to backup, then the ledger is restored into a fresh empty
    data_cache/. Otherwise everything goes.
    """
    cache = REPO_ROOT / WIPE_DATACACHE_DIR
    if not cache.exists():
        return 0, [], 0

    # Count files for the preview (excluding the quota ledger if kept).
    file_count = sum(1 for _ in cache.rglob("*") if _.is_file())

    kept: list[tuple[Path, Path]] = []  # (original_path, temp_backup)
    if keep_quota:
        for f in cache.glob(KEEP_QUOTA_LEDGER_NAME):
            if f.is_file():
                tmp = REPO_ROOT / f".keep_tmp_{f.name}"
                if not dry:
                    shutil.move(str(f), str(tmp))
                kept.append((f, tmp))

    moved = 0
    if not dry:
        if _move_to_backup(cache, backup_dir):
            moved = 1

    kept_names = []
    for original, tmp in kept:
        kept_names.append(original.name)
        if not dry:
            original.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(tmp), str(original))
        else:
            # dry-run: clean up the temp move we made for the preview.
            if tmp.exists():
                shutil.move(str(tmp), str(original))

    if keep_quota and kept_names:
        file_count = max(0, file_count - len(kept_names))

    return moved, kept_names, file_count


# ---------------------------------------------------------------------------
# Generic directory wipe (for WIPE_DIRS other than data_cache)
# ---------------------------------------------------------------------------
def _wipe_generic_dir(rel: str, backup_dir: Path, dry: bool) -> bool:
    """Backup+remove a whole directory (e.g. morning_brief/output). Returns
    True if it existed."""
    target = REPO_ROOT / rel
    if not target.exists() or not target.is_dir():
        return False
    if not _safe_to_wipe(target):
        return False
    if dry:
        return True
    return _move_to_backup(target, backup_dir)


def _count_files_in_dir(rel: str) -> int:
    target = REPO_ROOT / rel
    if not target.exists():
        return 0
    return sum(1 for _ in target.rglob("*") if _.is_file())


def _resolve_existing(rel: str) -> Path | None:
    p = REPO_ROOT / rel
    return p if p.exists() else None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="MAX reset: wipe all learned state, caches and history for a virgin restart."
    )
    parser.add_argument(
        "--yes", action="store_true",
        help="Skip the interactive confirmation (for automation).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what WOULD be done without moving or writing anything.",
    )
    parser.add_argument(
        "--keep-quota-ledger", action="store_true",
        help="PRESERVE data_cache/gemini_quota.db (the 30-day cost-budget ledger). "
             "Use this on PAID PROD to avoid the gateway thinking it has a fresh "
             "full budget and potentially overspending. By default (DEMO mode) "
             "the ledger is wiped for a true ground-zero restart.",
    )
    parser.add_argument(
        "--include-logs-prod", action="store_true",
        help="WIPE the runtime artifacts inside logs_prod/ (scheduler CWD on PROD: "
             "DBs, t212_portfolio_state.json, journal, data_cache/, lock...). "
             "Required for a true PROD reset — without it the 2026-08-19 reset "
             "left logs_prod/model_performance.db alive (incident 2026-08-24). "
             "Everything is backed up to reset_backup/ first.",
    )
    parser.add_argument(
        "--keep-logs-prod", action="store_true",
        help="PRESERVE logs_prod/ runtime artifacts (DEV case: logs_prod/ is an "
             "audit snapshot, not a live scheduler CWD).",
    )
    args = parser.parse_args()

    if args.include_logs_prod and args.keep_logs_prod:
        parser.error("--include-logs-prod et --keep-logs-prod sont mutuellement exclusifs.")

    logs_prod_targets = _collect_logs_prod_runtime_artifacts()
    include_logs_prod = args.include_logs_prod

    print("=" * 72)
    print("  MAX RESET — vidage complet pour redemarrage vierge")
    print("=" * 72)
    print()
    print("ATTENTION: ce script efface TOUT l'etat appris et l'historique :")
    print("  caches (modeles, prix, EIA, tensortrade, finacumen, quota...),")
    print("  DBs, journaux, etat de portefeuille, dashboards, briefs, rapports.")
    print("  Le systeme repart de zero au prochain cycle (re-telechargement")
    print("  des donnees + retraining complet = lent).")
    print()
    print("PRESERVE: .env* (cles), .venv, .git, logs_prod (archive), code source.")
    quota_label = "CONSERVE (--keep-quota-ledger)" if args.keep_quota_ledger else "EFFACE (mode demo)"
    print(f"Ledger quota Gemini: {quota_label}.")
    print()

    # ---- Preview (always shown) -----------------------------------------
    if args.dry_run:
        print("[DRY-RUN] Apercu (rien ne sera modifie):\n")

    # data_cache preview
    cache = REPO_ROOT / WIPE_DATACACHE_DIR
    print("-- A EFFACER -> backup puis suppression --")
    n_targets = 0
    if cache.exists():
        _, kept_preview, cache_files = _wipe_data_cache(
            _backup_root() / "_preview", dry=True, keep_quota=args.keep_quota_ledger,
        )
        keep_note = f"; conserve: {', '.join(kept_preview)}" if kept_preview else ""
        print(f"  [dir]  data_cache/  ({cache_files} fichiers{keep_note})")
        n_targets += 1

    # other wipe dirs
    for d in WIPE_DIRS:
        p = _resolve_existing(d)
        if p:
            n = _count_files_in_dir(d)
            print(f"  [dir]  {d}/  ({n} fichiers)")
            n_targets += 1

    # root-level pattern files
    root_files = _collect_root_runtime_files()
    if root_files:
        print(f"  [root] {len(root_files)} fichier(s) runtime a la racine :")
        # Group by extension for compactness.
        by_ext: dict[str, list[str]] = {}
        for f in root_files:
            by_ext.setdefault(f.suffix.lower() or "(no-ext)", []).append(f.name)
        for ext in sorted(by_ext):
            names = by_ext[ext]
            preview = ", ".join(sorted(names)[:4])
            extra = f" +{len(names)-4} autres" if len(names) > 4 else ""
            print(f"           *{ext}: {preview}{extra}")
        n_targets += 1

    # logs_prod/ runtime artifacts (scheduler CWD on PROD)
    if logs_prod_targets:
        mode = (
            "VIDE (--include-logs-prod)" if include_logs_prod
            else "PRESERVE (--keep-logs-prod)" if args.keep_logs_prod
            else "NON DECIDE (interactif / --yes: choix explicite requis)"
        )
        print(f"\n-- logs_prod/ — {len(logs_prod_targets)} artefact(s) runtime detecte(s) "
              f"[{mode}] :")
        for t in logs_prod_targets[:8]:
            print(f"           {t.relative_to(REPO_ROOT)}")
        if len(logs_prod_targets) > 8:
            print(f"           +{len(logs_prod_targets) - 8} autres")
        n_targets += 1

    if n_targets == 0:
        print("  (rien a effacer — deja vierge)")

    print("\n-- CONSERVE (jamais touche) --")
    for k in sorted(KEEP_PATHS):
        # Directories get a trailing slash, files (like .env) don't.
        is_dir = (REPO_ROOT / k).is_dir()
        print(f"  [keep] {k}{'/' if is_dir else ''}")
    print("  [keep] *.py *.md *.toml *.yaml *.bat *.lock (fichiers source/config)")

    # ---- logs_prod/ explicit choice (before the global confirm gate) -------
    if logs_prod_targets and not include_logs_prod and not args.keep_logs_prod:
        if args.yes:
            # Automation safety: NEVER silently wipe NOR silently keep the live
            # PROD state (the 2026-08-19 reset silently kept it -> polluted
            # adaptive weights until 2026-08-24).
            print()
            print("[ERREUR] Artefacts runtime detectes dans logs_prod/ et aucun choix")
            print("         explicite fourni. Avec --yes, passez soit :")
            print("           --include-logs-prod  (reset PROD complet)")
            print("           --keep-logs-prod      (logs_prod/ = snapshot d'audit DEV)")
            return 2
        if not args.dry_run:
            print()
            print("logs_prod/ contient des artefacts runtime (CWD du scheduler PROD).")
            if confirm("Les inclure dans le vidage (backup puis suppression) ?", False):
                include_logs_prod = True
                print("-> logs_prod/ sera inclus dans le vidage.")
            else:
                print("-> logs_prod/ sera PRESERVE.")

    gate = dry_run_or_confirm(
        args.dry_run,
        args.yes,
        "\nConfirmer le VIDAGE COMPLET ? (tout est backup puis efface)",
    )
    if gate is not None:
        return gate

    # ---- Real execution --------------------------------------------------
    stamp = _backup_timestamp()
    backup_dir = _backup_root() / stamp
    backup_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nBackup -> {backup_dir.relative_to(REPO_ROOT)}")

    actions = 0

    # 1. Full wipe of data_cache/ (preserving quota ledger if requested).
    cache_moved, kept_names, _ = _wipe_data_cache(backup_dir, dry=False, keep_quota=args.keep_quota_ledger)
    if cache_moved:
        keep_note = f" (conserve: {', '.join(kept_names)})" if kept_names else ""
        print(f"  wipe   data_cache/{keep_note}")
        actions += 1

    # 2. Wipe each generic runtime directory.
    for d in WIPE_DIRS:
        if _wipe_generic_dir(d, backup_dir, dry=False):
            print(f"  wipe   {d}/")
            actions += 1

    # 3. Wipe root-level runtime files by pattern (state, DBs, logs, dashboards).
    root_files = _collect_root_runtime_files()
    n_root = _wipe_root_files(root_files, backup_dir, dry=False)
    if n_root:
        print(f"  wipe   {n_root} fichier(s) runtime racine (*.csv/*.json/*.db/*.log/*.png...)")
        actions += 1

    # 4. Wipe logs_prod/ runtime artifacts (PROD scheduler CWD) when included.
    if logs_prod_targets and include_logs_prod:
        n_lp = 0
        for t in logs_prod_targets:
            # Paranoia guard: never touch anything outside logs_prod/ here.
            if t.resolve().relative_to(REPO_ROOT).parts[0] != LOGS_PROD_DIR:
                continue
            if _move_to_backup(t, backup_dir):
                n_lp += 1
        if n_lp:
            print(f"  wipe   logs_prod/ ({n_lp} artefact(s) runtime)")
            actions += 1

    print()
    print("=" * 72)
    print(f"  VIDAGE COMPLET TERMINE — {actions} element(s) efface(s).")
    print("=" * 72)
    print()
    print("PROCHAINES ETAPES:")
    print("  1. Le 1er cycle va RE-TELECHARGER les donnees de marche (~5 ans),")
    print("     reentrainer classic (calibration isotonic), le PPO depuis zero,")
    print("     et re-fetcher les donnees EIA -> il sera LONG (plusieurs min).")
    print("  2. T212 (DEMO) : reset du compte demo dans l'app T212 (annule positions,")
    print("     stops GTC et historique -> l'equity FIFO repart a 1000 EUR/ticker),")
    print("     ou clôturez manuellement toute position residuelle avant de")
    print("     relancer, sinon le state se re-synchronise dessus.")
    print("  3. Lancez en DEMO pour valider les mecanismes de sortie")
    print("     (stop-loss -5/-10%, take-profit +8%, trailing -3%, time-stop 15j).")
    print(f"  4. Backup disponible dans {backup_dir.relative_to(REPO_ROOT)}/")
    print("     (a supprimer manuellement une fois le test valide).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
