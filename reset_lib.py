"""Shared helpers for the reset/cleanup scripts.

`reset_for_fresh_test.py` (full ground-zero reset) and `clean_phantom_trades.py`
(targeted cleanup) both implement the same `_confirm` prompt and the same
dry-run/confirm control-flow block. Pylint flagged the duplication
(python-health-audit 2026-07-28). This module centralizes both.

Importing scripts keep their own `argparse` setup — they only delegate the
confirmation UX here.
"""

from __future__ import annotations


def confirm(prompt: str, assume_yes: bool) -> bool:
    """Yes/no prompt. Returns True if confirmed.

    ``assume_yes`` short-circuits to True (for automation / ``--yes``).
    Accepts y/yes/o/oui (EN + FR). On EOF (non-interactive stdin) returns False
    so an unexpected piped invocation never silently wipes anything.
    """
    if assume_yes:
        return True
    try:
        answer = input(f"{prompt} [y/N] ").strip().lower()
    except EOFError:
        return False
    return answer in ("y", "yes", "o", "oui")


def dry_run_or_confirm(
    dry_run: bool,
    assume_yes: bool,
    confirm_prompt: str,
) -> int | None:
    """Handle the dry-run/confirm gate shared by the reset scripts.

    Returns:
        - ``0`` if ``dry_run`` is True (script should exit 0 after previewing).
        - ``1`` if the user declines confirmation (script should exit 1, no-op).
        - ``None`` if execution should proceed (caller continues past the gate).
    """
    print()
    if dry_run:
        print("[DRY-RUN] Termine. Relancez sans --dry-run pour executer.")
        return 0

    if not confirm(confirm_prompt, assume_yes):
        print("Annule. Aucune modification effectuee.")
        return 1

    return None
