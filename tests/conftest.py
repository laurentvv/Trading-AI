"""Isolation des tests : aucune écriture relative ne doit atteindre les artefacts PROD/démo.

Contexte (analyse 2026-09-29) : le démo T212 tourne depuis le checkout de dev et tous les
chemins runtime (`trading_history.db`, `trading_journal.csv`, `t212_portfolio_state.json`,
`model_performance.db`, `performance_monitor.db`, `data_cache/`...) sont RELATIFS au CWD.
`TestMaxAvailableSizing` appelait `_execute_buy_order` sans neutraliser l'insertion DB : chaque
exécution de la suite ajoutait une fausse ligne « BUY 10 @ 100.00 » dans `trading_history.db`
du run démo (8 lignes constatées), faussant équity et historique.

Règle : chaque test s'exécute dans un CWD temporaire vide. Les tests qui ont réellement
besoin des fichiers du dépôt doivent porter `@pytest.mark.repo_cwd` (lecture seule
recommandée) ou résoudre leurs chemins via `REPO_ROOT`.
"""

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(autouse=True)
def _isolated_cwd(request, tmp_path, monkeypatch):
    """Exécute chaque test dans un CWD temporaire (sauf marqueur `repo_cwd`)."""
    if request.node.get_closest_marker("repo_cwd") is not None:
        yield
        return
    monkeypatch.chdir(tmp_path)
    yield


@pytest.fixture
def repo_root() -> Path:
    """Racine du dépôt, pour les tests qui doivent lire un fichier versionné."""
    return REPO_ROOT
