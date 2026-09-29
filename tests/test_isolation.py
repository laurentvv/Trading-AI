"""Garde-fous de l'isolation des tests (voir tests/conftest.py)."""

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_default_cwd_is_not_the_repo():
    assert Path.cwd().resolve() != REPO_ROOT


def test_runtime_paths_resolve_outside_the_repo():
    """Les chemins runtime relatifs ne doivent pas pointer vers le dépôt (donc vers le démo)."""
    from src import database

    resolved = database.DB_PATH.resolve()
    # (basetemp peut vivre sous le dépôt : on compare au dossier racine exact, pas à l'arborescence)
    assert resolved.parent != REPO_ROOT
    assert Path("trading_journal.csv").resolve().parent != REPO_ROOT


def test_insert_transaction_lands_in_the_temp_cwd(tmp_path):
    """Une écriture DB « naïve » atterrit dans le CWD temporaire, jamais dans trading_history.db du dépôt."""
    from src import database

    database.init_db()
    database.insert_transaction("2026-01-01 00:00:00", "TEST", "BUY", 1.0, 1.0, 1.0, "TEST", "isolation")
    assert database.DB_PATH.resolve() == (tmp_path / "trading_history.db").resolve()
    assert (tmp_path / "trading_history.db").exists()


@pytest.mark.repo_cwd
def test_repo_cwd_marker_opts_out():
    assert Path.cwd().resolve() == REPO_ROOT
