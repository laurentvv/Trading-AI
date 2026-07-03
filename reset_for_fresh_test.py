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
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
BACKUP_ROOT = REPO_ROOT / "reset_backup"

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


def _confirm(prompt: str, assume_yes: bool) -> bool:
    if assume_yes:
        return True
    try:
        answer = input(f"{prompt} [y/N] ").strip().lower()
    except EOFError:
        return False
    return answer in ("y", "yes", "o", "oui")


def _backup_timestamp() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def _move_to_backup(target: Path, backup_dir: Path) -> bool:
    """Move target into backup_dir, PRESERVING its path relative to the repo
    root. This keeps the backup restorable: a file at data_cache/eia/x.parquet
    is backed up to <backup>/data_cache/eia/x.parquet (not flattened)."""
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
    shutil.move(str(target), str(dest))
    return True


def _safe_to_wipe(target: Path) -> bool:
    """Final safety guard: refuse to wipe anything inside a KEEP_PATHS dir."""
    try:
        rel = target.resolve().relative_to(REPO_ROOT)
    except ValueError:
        return False
    parts = rel.parts
    if not parts:
        return False
    # Top-level dir name must not be in the keep set.
    if parts[0] in KEEP_PATHS:
        return False
    return True


# ---------------------------------------------------------------------------
# Root-level pattern wipe
# ---------------------------------------------------------------------------
def _collect_root_runtime_files() -> list[Path]:
    """Return every file at the repo ROOT whose extension is in
    WIPE_ROOT_EXTENSIONS. Does NOT recurse into subdirectories (those are
    handled by WIPE_DIRS / KEEP_PATHS)."""
    found = []
    for item in REPO_ROOT.iterdir():
        if not item.is_file():
            continue
        # Suffix matching: ".db-shm" / ".db-wal" have a dot inside the suffix,
        # so compare on the last dotted suffix AND the two-part suffix.
        name = item.name
        suffix = item.suffix.lower()
        # Also handle compound suffixes like "foo.db-shm", "foo.db-wal".
        compound = ""
        if "." in name:
            stem_and_ext = name.rsplit(".", 2)
            if len(stem_and_ext) >= 3:
                compound = ("." + stem_and_ext[-2] + "." + stem_and_ext[-1]).lower()
        if suffix in WIPE_ROOT_EXTENSIONS or compound in WIPE_ROOT_EXTENSIONS:
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
    args = parser.parse_args()

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
            BACKUP_ROOT / "_preview", dry=True, keep_quota=args.keep_quota_ledger,
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

    if n_targets == 0:
        print("  (rien a effacer — deja vierge)")

    print("\n-- CONSERVE (jamais touche) --")
    for k in sorted(KEEP_PATHS):
        # Directories get a trailing slash, files (like .env) don't.
        is_dir = (REPO_ROOT / k).is_dir()
        print(f"  [keep] {k}{'/' if is_dir else ''}")
    print("  [keep] *.py *.md *.toml *.yaml *.bat *.lock (fichiers source/config)")

    print()
    if args.dry_run:
        print("[DRY-RUN] Termine. Relancez sans --dry-run pour executer.")
        return 0

    if not _confirm(
        "\nConfirmer le VIDAGE COMPLET ? (tout est backup puis efface)",
        args.yes,
    ):
        print("Annule. Aucune modification effectuee.")
        return 1

    # ---- Real execution --------------------------------------------------
    stamp = _backup_timestamp()
    backup_dir = BACKUP_ROOT / stamp
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

    print()
    print("=" * 72)
    print(f"  VIDAGE COMPLET TERMINE — {actions} element(s) efface(s).")
    print("=" * 72)
    print()
    print("PROCHAINES ETAPES:")
    print("  1. Le 1er cycle va RE-TELECHARGER les donnees de marche (~5 ans),")
    print("     reentrainer classic (calibration isotonic), le PPO depuis zero,")
    print("     et re-fetcher les donnees EIA -> il sera LONG (plusieurs min).")
    print("  2. T212: clôturez manuellement toute position DEMO residuelle avant")
    print("     de relancer, sinon le state se re-synchronise dessus.")
    print("  3. Lancez en DEMO pour valider les mecanismes de sortie")
    print("     (stop-loss -5/-10%, take-profit +8%, trailing -3%, time-stop 15j).")
    print(f"  4. Backup disponible dans {backup_dir.relative_to(REPO_ROOT)}/")
    print("     (a supprimer manuellement une fois le test valide).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
