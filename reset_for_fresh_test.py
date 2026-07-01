#!/usr/bin/env python3
"""
Reset the trading system to a clean virgin state for a fresh test run.

WHY THIS EXISTS
---------------
The June 2026 audit found that learned caches had absorbed the old code's
structural bullish bias, and the portfolio state was corrupted (a ghost
entry_price that blocked every SELL). Code fixes (calibration, unconditional
exit strategy, corruption guard) live on branch
``fix/prod-exit-strategy-bias-sizing`` — but those fixes only take effect on
the NEXT training cycle. If the old caches survive, the stale biased model is
reloaded by hash and the fixes silently never apply. This script performs the
surgical reset that makes the fixes effective, while PRESERVING expensive raw
market data.

Run it once on PROD (or DEMO) after pulling the fix branch, before launching
the fresh test month:

    uv run python reset_for_fresh_test.py            # interactive confirm
    uv run python reset_for_fresh_test.py --yes      # no prompt (CI / automation)

WHAT IT DOES
------------
  PURGE  (re-learned next cycle, currently biased/corrupt):
    - data_cache/models/classic_model_*.pkl   trained WITHOUT isotonic calibration
                                              (cached by data-hash, so the fix
                                              would never apply unless purged)
    - data_cache/tensortrade/ppo_model.zip    PPO policy collapsed onto BUY
                                              (171x BUY(1.00) in the prod journal)
    - data_cache/eia/*                        fundamental caches with 1970-01-01
                                              timestamps (forced interpolation)

  RESET  (trading state tied to the old DEMO history):
    - t212_portfolio_state.json   -> {"tickers": {}}   (corrupted entry_price)
    - trading_history.db          -> archived + recreated empty by init_db()
    - data_cache/finacumen/       states/trajectories from old behaviour

  KEEP   (expensive, neutral, not re-learnable cheaply):
    - data_cache/*_max_with_vix.parquet   ~5y raw OHLCV from Yahoo
    - data_cache/macro/                   FRED macro (CPI, rates, GDP, ...)
    - data_cache/search_queries/          cached web research

SAFETY
------
- Nothing is deleted without confirmation (unless --yes).
- Every purged item is moved to a timestamped backup folder before removal,
  so the operation is fully reversible until you delete the backup yourself.
- The script is idempotent: re-running it is a no-op on an already-clean tree.
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
# Inventory: explicit, reviewed lists. Editing here is the only way to change
# what the script touches — keeps the blast radius auditable.
# ---------------------------------------------------------------------------

# Learned caches that absorbed the old biased logic. Purge = force retrain with
# the fixed code on the next cycle.
PURGE_DIRS = [
    "data_cache/models",
    "data_cache/tensortrade",
]
PURGE_GLOBS_IN_CACHE = [
    "data_cache/eia/*.parquet",   # corrupt 1970 timestamps — re-fetched fresh next cycle
]

# Trading state tied to the DEMO history we are discarding.
RESET_FILES = [
    "t212_portfolio_state.json",
    "trading_history.db",
]
RESET_DIRS = [
    "data_cache/finacumen",
]

# Raw market data preserved on purpose (expensive to regenerate, neutral).
KEEP_DIRS = [
    "data_cache",
    "data_cache/macro",
    "data_cache/search_queries",
]
KEEP_GLOBS = [
    "data_cache/*_max_with_vix.parquet",
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


def _purge_dir(rel: str, backup_dir: Path, dry: bool) -> list[Path]:
    """Purge (backup+remove) an entire directory under repo root."""
    src = REPO_ROOT / rel
    if not src.exists():
        return []
    moved = []
    if dry:
        moved.append(src)
        return moved
    if _move_to_backup(src, backup_dir):
        moved.append(src)
    return moved


def _reset_file(rel: str, backup_dir: Path, dry: bool) -> list[Path]:
    """Backup a file then recreate it in its virgin form."""
    src = REPO_ROOT / rel
    moved = []
    if src.exists():
        if dry:
            moved.append(src)
        elif _move_to_backup(src, backup_dir):
            moved.append(src)
    if not dry:
        # Recreate the virgin form immediately.
        if rel == "t212_portfolio_state.json":
            src.write_text('{\n    "tickers": {}\n}\n', encoding="utf-8")
        # trading_history.db is recreated by init_db() on the next run; we do
        # not create an empty binary here so the schema bootstrap is owned by
        # src/database.py, not duplicated in this script.
    return moved


def _reset_dir(rel: str, backup_dir: Path, dry: bool) -> list[Path]:
    """Backup+remove a directory, leaving it absent (recreated on demand)."""
    return _purge_dir(rel, backup_dir, dry)


def _collect_glob(rel_glob: str) -> list[Path]:
    base = REPO_ROOT
    # split into dir + pattern (single-level glob in cache root is enough here)
    parts = rel_glob.split("/")
    return sorted(base.glob(rel_glob)) if len(parts) <= 2 else sorted(base.glob(rel_glob))


def _verify_keep_list_present() -> list[str]:
    """Warn about kept items that are missing (operator may have lost data)."""
    warnings = []
    for g in KEEP_GLOBS:
        if not _collect_glob(g):
            warnings.append(f"  - aucun fichier pour '{g}' (donnee brute manquante ?)")
    for d in KEEP_DIRS:
        if not (REPO_ROOT / d).exists():
            warnings.append(f"  - dossier conserve absent: {d}")
    return warnings


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Reset the trading system to a virgin state for a fresh test."
    )
    parser.add_argument(
        "--yes", action="store_true",
        help="Skip the interactive confirmation (for automation).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what WOULD be done without moving or writing anything.",
    )
    args = parser.parse_args()

    print("=" * 72)
    print("  RESET FOR FRESH TEST — June 2026 exit-strategy fix")
    print("=" * 72)
    print()
    print("Ce script purge les caches APPRIS avec l'ancien code biaise, reset")
    print("l'etat de trading DEMO, et CONSERVE les donnees de marche brutes.")
    print("Les correctifs code (calibration, stop-loss, etc.) ne s'appliqueront")
    print("au prochain cycle QUE si ces caches sont purges.")
    print()

    # ---- Preview of what will be touched (always shown) ------------------
    if args.dry_run:
        print("[DRY-RUN] Apercu des actions (rien ne sera modifie):\n")

    def _resolve(rel_or_glob: str) -> list[Path]:
        p = REPO_ROOT / rel_or_glob
        if "*" in rel_or_glob:
            return sorted(REPO_ROOT.glob(rel_or_glob))
        return [p] if p.exists() else []

    print("-- A PURGER (appris / biaise / corrompu) -> backup puis suppression --")
    purge_targets: list[Path] = []
    for d in PURGE_DIRS:
        purge_targets += _resolve(d)
    for g in PURGE_GLOBS_IN_CACHE:
        purge_targets += _resolve(g)
    purge_targets = [t for t in purge_targets if t.exists()]
    if purge_targets:
        for t in purge_targets:
            kind = "dir" if t.is_dir() else "file"
            print(f"  [{kind}] {t.relative_to(REPO_ROOT)}")
    else:
        print("  (rien a purger — deja propre)")

    print("\n-- A RESETER (etat de trading DEMO) -> backup puis forme vierge --")
    reset_targets: list[Path] = []
    for f in RESET_FILES:
        reset_targets += _resolve(f)
    for d in RESET_DIRS:
        reset_targets += _resolve(d)
    reset_targets = [t for t in reset_targets if t.exists()]
    if reset_targets:
        for t in reset_targets:
            kind = "dir" if t.is_dir() else "file"
            print(f"  [{kind}] {t.relative_to(REPO_ROOT)}")
    else:
        print("  (rien a reseter — deja vierge)")

    print("\n-- CONSERVE (donnees de marche brutes, neutres, couteuses) --")
    kept = 0
    for g in KEEP_GLOBS:
        for f in _collect_glob(g):
            print(f"  [keep] {f.relative_to(REPO_ROOT)}")
            kept += 1
    for d in KEEP_DIRS:
        if (REPO_ROOT / d).exists():
            print(f"  [keep] {d}/")
            kept += 1
    if kept == 0:
        print("  (aucune donnee brute trouvee — voir warnings)")

    warns = _verify_keep_list_present()
    if warns:
        print("\n-- ATTENTION: donnees brutes conservees manquantes --")
        for w in warns:
            print(w)

    print()
    if args.dry_run:
        print("[DRY-RUN] Termine. Relancez sans --dry-run pour executer.")
        return 0

    if not _confirm(
        "\nConfirmer le reset ? (backup -> reset; DONNEES BRUTES CONSERVEES)",
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
    for d in PURGE_DIRS:
        actions += len(_purge_dir(d, backup_dir, dry=False))
    for g in PURGE_GLOBS_IN_CACHE:
        for f in _collect_glob(g):
            if _move_to_backup(f, backup_dir):
                print(f"  purge  {f.relative_to(REPO_ROOT)}")
                actions += 1
    for f in RESET_FILES:
        moved = _reset_file(f, backup_dir, dry=False)
        actions += len(moved)
        if moved:
            print(f"  reset  {f}  (-> vierge)")
    for d in RESET_DIRS:
        moved = _reset_dir(d, backup_dir, dry=False)
        actions += len(moved)
        if moved:
            print(f"  reset  {d}/  (supprime, recree a la demande)")

    print()
    print("=" * 72)
    print(f"  RESET TERMINE — {actions} element(s) traite(s).")
    print("=" * 72)
    print()
    print("PROCHAINES ETAPES:")
    print("  1. Le 1er cycle va retrainer classic (calibration isotonic),")
    print("     reentrainer le PPO depuis zero, et re-fetcher les donnees EIA.")
    print("     -> il sera plus lent qu'en regime normal (retraining).")
    print("  2. Lancez d'abord en DEMO pour valider le comportement des 4")
    print("     mecanismes de sortie (stop-loss -5/-10%, take-profit +8%,")
    print("     trailing -3%, time-stop 15j) avant tout passage en reel.")
    print(f"  3. Backup disponible dans {backup_dir.relative_to(REPO_ROOT)}/")
    print("     (a supprimer manuellement une fois le test valide).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
