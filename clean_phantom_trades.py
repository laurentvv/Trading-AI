#!/usr/bin/env python3
"""
Clean phantom trades — targeted wipe of DBs polluted by the simulation-before-T212 bug.

WHY THIS EXISTS
---------------
The July 2026 audit found that `enhanced_trading_example._execute_hypothetical_trade`
wrote SIMULATED trades to `trading_history.db` BEFORE the real T212 order was
attempted. When the T212 order failed (precision mismatch, rate limit) or was
skipped (signal became HOLD), the phantom transaction stayed in the DB —
desyncing it from broker truth (0 positions at the broker, 2 phantom BUYs in
the DB). This also generated 72 false "win_rate critically low: 0.00%" alerts.

The fix (write_db=False in T212 mode) prevents NEW phantom trades, but the
PROD DBs already contain the polluted data from before the fix. This script
purges ONLY those three artefacts so the next cycle starts from a clean,
broker-truth-aligned state — WITHOUT wiping the freshly-trained models,
market data caches, or EIA data (unlike reset_for_fresh_test.py which is a
full ground-zero reset).

Run it on PROD after pulling the fix, before relaunching the pipeline:

    uv run python clean_phantom_trades.py --dry-run   # preview
    uv run python clean_phantom_trades.py --yes        # execute (no prompt)

WHAT IT WIPES (only phantom-polluted artefacts)
-----------------------------------------------
  - trading_history.db          transactions + portfolio_history (phantom BUYs)
  - performance_monitor.db      realtime_metrics + false win_rate alerts
  - t212_portfolio_state.json   portfolio state (will resync from broker next cycle)

WHAT IT PRESERVES (everything else — especially the expensive caches)
--------------------------------------------------------------------
  - data_cache/                 models, prices, EIA, tensortrade, finacumen, etc.
  - model_performance.db        model performance history (not polluted)
  - trading_journal.csv         CSV journal (decision log, not transaction log)
  - trading.log / scheduler.log main logs
  - *.png dashboards
  - .env* credentials, .venv, .git, logs_prod, memory-bank, source code

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

REPO_ROOT = Path(__file__).resolve().parent
BACKUP_ROOT = REPO_ROOT / "reset_backup"

# Only these three artefacts are polluted by the phantom-trades bug.
WIPE_FILES = [
    "trading_history.db",          # phantom BUY transactions + portfolio_history
    "performance_monitor.db",      # false win_rate alerts + realtime_metrics
    "t212_portfolio_state.json",   # portfolio state (resyncs from broker)
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
    """Move target into backup_dir, preserving its path relative to the repo
    root. Falls back to copy+truncate if the file is locked (Windows)."""
    if not target.exists():
        return False
    try:
        rel = target.resolve().relative_to(REPO_ROOT)
    except ValueError:
        rel = Path(target.name)
    dest = backup_dir / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        dest = backup_dir / f"{rel}_{_backup_timestamp()}"
        dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        shutil.move(str(target), str(dest))
    except PermissionError:
        # File locked by another process — copy then truncate in place.
        shutil.copy2(str(target), str(dest))
        try:
            target.write_bytes(b"")
        except PermissionError:
            pass
    return True


def _resolve_existing(rel: str) -> Path | None:
    p = REPO_ROOT / rel
    return p if p.exists() else None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Clean phantom trades: wipe DBs polluted by the "
                    "simulation-before-T212 bug (targeted, not a full reset)."
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
    print("  CLEAN PHANTOM TRADES — nettoyage ciblé des DBs polluées")
    print("=" * 72)
    print()
    print("Supprime UNIQUEMENT les artefacts pollués par les trades fantômes :")
    print("  - trading_history.db        (transactions + portfolio_history)")
    print("  - performance_monitor.db    (fausses alertes win_rate + metriques)")
    print("  - t212_portfolio_state.json (state, re-sync depuis le broker)")
    print()
    print("PRESERVE tout le reste (data_cache, modeles, prix, EIA, journaux,")
    print("logs, dashboards, credentials, code source).")
    print()

    # ---- Preview ----
    if args.dry_run:
        print("[DRY-RUN] Apercu (rien ne sera modifie):\n")

    print("-- A EFFACER -> backup puis suppression --")
    n_targets = 0
    for f in WIPE_FILES:
        p = _resolve_existing(f)
        if p:
            size_kb = p.stat().st_size / 1024
            print(f"  [file] {f}  ({size_kb:.1f} Ko)")
            n_targets += 1
    if n_targets == 0:
        print("  (rien a effacer — deja propre)")

    print("\n-- CONSERVE (jamais touche) --")
    print("  [keep] data_cache/  (modeles, prix, EIA, tensortrade, finacumen)")
    print("  [keep] model_performance.db")
    print("  [keep] trading_journal.csv, trading.log, scheduler.log")
    print("  [keep] *.png dashboards, .env*, .venv, .git, logs_prod, memory-bank")
    print("  [keep] src/ tests/ docs/ scripts/  (code source)")

    print()
    if args.dry_run:
        print("[DRY-RUN] Termine. Relancez sans --dry-run pour executer.")
        return 0

    if not _confirm(
        "\nConfirmer le nettoyage ? (3 fichiers backup puis effaces)",
        args.yes,
    ):
        print("Annule. Aucune modification effectuee.")
        return 1

    # ---- Execution ----
    stamp = _backup_timestamp()
    backup_dir = BACKUP_ROOT / stamp
    backup_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nBackup -> {backup_dir.relative_to(REPO_ROOT)}")

    actions = 0
    for f in WIPE_FILES:
        src = REPO_ROOT / f
        if not src.exists() or not src.is_file():
            continue
        if _move_to_backup(src, backup_dir):
            print(f"  wipe   {f}")
            actions += 1

    print()
    print("=" * 72)
    print(f"  NETTOYAGE TERMINE — {actions} fichier(s) efface(s).")
    print("=" * 72)
    print()
    print("PROCHAINES ETAPES:")
    print("  1. Le prochain cycle va re-sync le state T212 depuis le broker")
    print("     (0 position = state propre, plus de trades fantômes en DB).")
    print("  2. Les alertes win_rate fausses ne se reproduiront plus")
    print("     (bug #1 corrige : write_db=False en mode T212).")
    print(f"  3. Backup disponible dans {backup_dir.relative_to(REPO_ROOT)}/")
    print("     (a supprimer manuellement une fois le test valide).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
