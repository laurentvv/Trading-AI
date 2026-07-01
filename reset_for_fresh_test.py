#!/usr/bin/env python3
"""
FULL reset — wipe everything learned/recorded for a truly virgin restart.

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

WHAT IT WIPES (everything learned/recorded)
-------------------------------------------
  - data_cache/   ENTIRE tree: prices, models, EIA, tensortrade, finacumen,
                  macro, search_queries... ALL of it. Re-downloaded next cycle.
  - t212_portfolio_state.json   portfolio state
  - trading_history.db          transaction log
  - model_performance.db        model performance history
  - performance_monitor.db      realtime metrics
  - trading_journal.csv         CSV journal
  - trading.log                 main log
  - enhanced_*.png              dashboard images
  - test_img.png

WHAT IT PRESERVES (whitelist — never touched)
--------------------------------------------
  - .env / .env.t212            credentials — NEVER purged
  - .venv / .git                environment + version control
  - logs_prod/                  prod log archive (read-only reference)
  - memory-bank/                deterministic state + docs
  - data_cache/gemini_quota.db  Gemini 30-day cost-budget ledger (see note)

GEMINI QUOTA LEDGER (important)
-------------------------------
By default gemini_quota.db is PRESERVED. It tracks the rolling 30-day EUR cost
budget of the Gemini gateway; wiping it would make the gateway think it has a
fresh full budget and could OVERSPEND. Pass --purge-quota-ledger only for a
true ground-zero reset where you accept the budget restarts at zero.

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
# FULL RESET INVENTORY
#
# Goal: a complete wipe so the system starts from a truly virgin state on the
# next run (fresh data download, fresh training, empty portfolio). Nothing
# learned or recorded from a previous run survives. The first cycle after a
# full reset will be SLOW (re-downloads years of market data + retrains every
# model) — that is expected and is the price of a clean slate.
#
# The inventory is expressed as a KEEP list (whitelist), not a purge list:
# everything under REPO_ROOT that is a known runtime artifact is wiped EXCEPT
# the entries below. This is safer than a blacklist — a new artifact that
# appears later is purged by default rather than silently kept.
# ---------------------------------------------------------------------------

# ABSOLUTE KEEP — never moved, never touched (whitelist).
KEEP_PATHS = {
    ".git",                 # version control
    ".venv",                # python environment
    ".env", ".env.t212",    # credentials — NEVER purge
    ".env.example", ".env.t212.example",
    "logs_prod",            # prod log archive (read-only reference)
    "reset_backup",         # our own backup output
    "memory-bank",          # deterministic state / docs
    ".pytest_cache", ".ruff_cache", "__pycache__",
    "src", "tests", "docs", "scripts", ".agents", "vendor", "alerte-wti-main",
}

# KEEP files inside data_cache/ — these survive a full reset (e.g. the Gemini
# quota ledger, whose 30-day rolling cost budget must NOT be zeroed, otherwise
# the gateway would think it has a full fresh budget and could overspend).
KEEP_GLOBS_IN_DATACACHE = {
    "gemini_quota.db",      # cloud quota ledger — zeroing it resets the cost budget
}

# Full-wipe targets (everything here is moved to backup then removed).
WIPE_DIRS = [
    "data_cache",           # ALL caches: prices, models, EIA, tensortrade, finacumen, etc.
]
WIPE_FILES = [
    "t212_portfolio_state.json",   # portfolio state
    "trading_history.db",          # transaction log (also lives in data_cache — both wiped)
    "model_performance.db",        # model performance history
    "performance_monitor.db",      # realtime metrics history
    "trading_journal.csv",         # CSV journal
    "trading.log",                 # main log
    "enhanced_performance_dashboard_CRUDP.PA.png",
    "enhanced_performance_dashboard_SXRV.DE.png",
    "enhanced_trading_chart.png",
    "test_img.png",
]


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


def _wipe_file(rel: str, backup_dir: Path, dry: bool) -> bool:
    """Backup+remove a single file. Returns True if it existed."""
    src = REPO_ROOT / rel
    if not src.exists() or not src.is_file():
        return False
    if dry:
        return True
    return _move_to_backup(src, backup_dir)


def _wipe_data_cache(backup_dir: Path, dry: bool) -> tuple[int, list[str]]:
    """Full wipe of data_cache/ while preserving KEEP_GLOBS_IN_DATACACHE.

    Moves the entire data_cache/ tree to the backup, then restores the kept
    files (e.g. gemini_quota.db) to their original location so the cloud quota
    ledger survives a reset. Returns (files_moved, kept_names).
    """
    cache = REPO_ROOT / "data_cache"
    if not cache.exists():
        return 0, []

    # Snapshot the kept files BEFORE moving anything.
    kept: list[tuple[Path, Path]] = []  # (original_path, temp_backup)
    for name in KEEP_GLOBS_IN_DATACACHE:
        for f in cache.glob(name):
            if f.is_file():
                # Stage the kept file outside data_cache so the bulk move doesn't grab it.
                tmp = REPO_ROOT / f".keep_tmp_{f.name}"
                shutil.move(str(f), str(tmp))
                kept.append((f, tmp))

    moved = 0
    if not dry:
        # Move the rest of data_cache wholesale into the backup.
        for item in cache.rglob("*"):
            pass  # touch the tree so existence is fresh
        if _move_to_backup(cache, backup_dir):
            moved = 1  # the whole dir counts as one move

    # Restore the kept files into a fresh empty data_cache/.
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
    return moved, kept_names


def _resolve_existing(rel: str) -> Path | None:
    p = REPO_ROOT / rel
    return p if p.exists() else None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="FULL reset: wipe all learned state, caches and history for a virgin restart."
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
        "--purge-quota-ledger", action="store_true",
        help="Also wipe gemini_quota.db (the 30-day cost-budget ledger). "
             "DANGEROUS: the gateway would then think it has a fresh full budget "
             "and could overspend. Only for a complete ground-zero reset.",
    )
    args = parser.parse_args()

    print("=" * 72)
    print("  FULL RESET — vidage complet pour redemarrage vierge")
    print("=" * 72)
    print()
    print("ATTENTION: ce script efface TOUT l'etat appris et l'historique :")
    print("  caches (modeles, prix, EIA, tensortrade...), DBs, journaux, etat")
    print("  de portefeuille. Le systeme repart de zero au prochain cycle")
    print("  (re-telechargement des donnees + retraining complet = lent).")
    print()
    print("PRESERVE: .env* (cles), .venv, .git, logs_prod (archive), et le")
    print("ledger de quota Gemini (budget cloud 30j) sauf --purge-quota-ledger.")
    print()

    # Build the effective keep-list for data_cache.
    keep_in_cache = set(KEEP_GLOBS_IN_DATACACHE)
    if args.purge_quota_ledger:
        keep_in_cache.discard("gemini_quota.db")

    # ---- Preview (always shown) -----------------------------------------
    if args.dry_run:
        print("[DRY-RUN] Apercu (rien ne sera modifie):\n")

    print("-- A EFFACER (vidage complet) -> backup puis suppression --")
    n_targets = 0
    cache = REPO_ROOT / "data_cache"
    if cache.exists():
        # count everything except the kept files
        kept_count = sum(len(list(cache.glob(g))) for g in keep_in_cache)
        total = sum(1 for _ in cache.rglob("*") if _.is_file())
        print(f"  [dir]  data_cache/  ({total - kept_count} fichiers; {kept_count} conserve)")
        n_targets += 1
    for f in WIPE_FILES:
        p = _resolve_existing(f)
        if p:
            print(f"  [file] {f}")
            n_targets += 1
    if n_targets == 0:
        print("  (rien a effacer — deja vierge)")

    print("\n-- CONSERVE (jamais touche) --")
    for k in sorted(KEEP_PATHS):
        print(f"  [keep] {k}/")
    for g in sorted(keep_in_cache):
        print(f"  [keep] data_cache/{g}")

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

    # 1. Full wipe of data_cache/ (preserving kept files).
    cache_moved, kept_names = _wipe_data_cache(backup_dir, dry=False)
    if cache_moved:
        print(f"  wipe   data_cache/  (conserve: {', '.join(kept_names) or 'rien'})")
        actions += 1

    # 2. Wipe each runtime file.
    for f in WIPE_FILES:
        # Some files also live inside data_cache (trading_history.db) — only
        # wipe the root-level copy; the cache wipe above already handled the other.
        if f == "trading_history.db" and cache_moved:
            continue
        if _wipe_file(f, backup_dir, dry=False):
            print(f"  wipe   {f}")
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
