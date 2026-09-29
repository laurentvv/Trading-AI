"""Modèle « livres » du mode cœur + poche active (plan de passage en réel, phase 2, PR 2-a).

Trois livres se partagent le portefeuille :
  - ``core``   : cœur Nasdaq-100, acheté en une fois et conservé (jamais vendu par le logiciel) ;
  - ``sleeve`` : poche active pilotée par l'ensemble de modèles ;
  - ``oil``    : satellite pétrole tactique, **compté dans les 10 %** de la poche (décision du 2026-09-30).

Le poids cible de ``sleeve`` + ``oil`` ne dépasse jamais ``ACTIVE_CAP`` (10 % de l'equity totale).

Un livre et le cœur peuvent détenir le MÊME instrument broker (le Nasdaq-100 en cœur et en poche) : le broker ne
connaît qu'une position, la répartition entre livres est donc tenue ici, en quantités, et vérifiée contre le
broker par ``reconcile`` (un écart est signalé, jamais corrigé par un ordre automatique).

Ce module est pur et n'ouvre aucune connexion : il n'active rien. Le mode ``legacy`` (un stop GTC par position,
``INITIAL_BUDGETS``) reste le défaut tant que ``STRATEGY_MODE=core_sleeve`` n'est pas posé.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

LEGACY = "legacy"
CORE_SLEEVE = "core_sleeve"
_MODES = (LEGACY, CORE_SLEEVE)

ACTIVE_CAP = 0.10  # poids max de sleeve + oil dans l'equity totale
WEIGHT_TOLERANCE = 1e-9
QTY_TOLERANCE = 1e-6  # écart de quantité toléré (arrondis broker)

BOOKS_STATE_FILE = "portfolio_books_state.json"


def get_strategy_mode() -> str:
    """Mode de stratégie (`STRATEGY_MODE`, défaut ``legacy``). Une valeur inconnue lève : on ne devine pas avec de l'argent."""
    mode = os.getenv("STRATEGY_MODE", LEGACY).strip().lower() or LEGACY
    if mode not in _MODES:
        raise ValueError(f"STRATEGY_MODE={mode!r} inconnu (attendu : {', '.join(_MODES)})")
    return mode


@dataclass(frozen=True)
class Book:
    name: str  # clé stable : "core", "sleeve", "oil"
    role: str  # "core" | "sleeve" | "oil"
    target_weight: float  # part cible de l'equity totale
    instruments: tuple[str, ...]  # tickers T212 autorisés pour ce livre
    protection: str = "none"  # "none" (pas de stop broker) | "gtc_stop"

    @property
    def is_active(self) -> bool:
        return self.role in ("sleeve", "oil")


def default_books() -> dict[str, Book]:
    """Livres par défaut. Cœur 90 % ; poche active 10 % dont la répartition sleeve/oil est un paramètre du banc
    (départ : tout en ``sleeve`` tant que le pétrole n'a pas de source de prix fiable)."""
    return {
        "core": Book("core", "core", 0.90, ("SXRVd_EQ",)),
        "sleeve": Book("sleeve", "sleeve", 0.10, ("SXRVd_EQ",)),
        "oil": Book("oil", "oil", 0.0, ("OD7Fd_EQ",)),
    }


def validate_books(books: dict[str, Book]) -> None:
    """Invariants de configuration ; lève ValueError."""
    if not books:
        raise ValueError("aucun livre défini")
    total = sum(b.target_weight for b in books.values())
    if total > 1.0 + WEIGHT_TOLERANCE:
        raise ValueError(f"poids cibles > 100 % ({total:.4f})")
    active = sum(b.target_weight for b in books.values() if b.is_active)
    if active > ACTIVE_CAP + WEIGHT_TOLERANCE:
        raise ValueError(f"poche active + pétrole = {active:.4f} > plafond {ACTIVE_CAP:.2f}")
    for b in books.values():
        if b.target_weight < 0:
            raise ValueError(f"livre {b.name}: poids négatif")
        if b.role not in ("core", "sleeve", "oil"):
            raise ValueError(f"livre {b.name}: rôle inconnu {b.role!r}")
        if b.protection not in ("none", "gtc_stop"):
            raise ValueError(f"livre {b.name}: protection inconnue {b.protection!r}")
        if b.role == "core" and b.protection != "none":
            raise ValueError("le cœur n'a pas de stop broker (décision du 2026-09-29)")
        if not b.instruments:
            raise ValueError(f"livre {b.name}: aucun instrument")


def target_amounts(books: dict[str, Book], total_equity: float) -> dict[str, float]:
    """Montant cible en euros par livre pour une equity totale donnée (le reste est la réserve de cash)."""
    validate_books(books)
    if total_equity < 0:
        raise ValueError("equity négative")
    return {name: round(total_equity * b.target_weight, 2) for name, b in books.items()}


@dataclass
class BooksState:
    """Quantités détenues par livre et par instrument, plus le pic d'equity du portefeuille."""

    quantities: dict[str, dict[str, float]] = field(default_factory=dict)  # livre -> ticker -> quantité
    equity_peak: float = 0.0

    def set_quantity(self, book: str, ticker: str, qty: float) -> None:
        if qty < 0:
            raise ValueError("quantité négative")
        self.quantities.setdefault(book, {})[ticker] = qty

    def quantity(self, book: str, ticker: str) -> float:
        return self.quantities.get(book, {}).get(ticker, 0.0)

    def total_quantity(self, ticker: str) -> float:
        return sum(q.get(ticker, 0.0) for q in self.quantities.values())

    def update_peak(self, equity: float) -> float:
        """Le pic ne fait que monter ; renvoie le drawdown courant (≤ 0) depuis ce pic."""
        self.equity_peak = max(self.equity_peak, equity)
        return (equity / self.equity_peak - 1.0) if self.equity_peak > 0 else 0.0


def reconcile(state: BooksState, broker_quantities: dict[str, float]) -> dict[str, float]:
    """Écart par instrument : broker moins somme des livres. Ne renvoie que les écarts au-delà de la tolérance.

    Un écart signale une opération manuelle ou un état perdu ; l'appelant alerte, il ne corrige pas par un ordre.
    """
    tickers = set(broker_quantities) | {t for q in state.quantities.values() for t in q}
    drift = {}
    for t in sorted(tickers):
        d = broker_quantities.get(t, 0.0) - state.total_quantity(t)
        if abs(d) > QTY_TOLERANCE:
            drift[t] = d
    return drift


def load_state(path: str | Path = BOOKS_STATE_FILE) -> BooksState:
    p = Path(path)
    if not p.exists():
        return BooksState()
    data = json.loads(p.read_text(encoding="utf-8"))
    return BooksState(
        quantities={b: {t: float(q) for t, q in m.items()} for b, m in data.get("quantities", {}).items()},
        equity_peak=float(data.get("equity_peak", 0.0)),
    )


def save_state(state: BooksState, path: str | Path = BOOKS_STATE_FILE) -> None:
    """Écriture atomique (fichier temporaire puis remplacement)."""
    p = Path(path)
    payload = json.dumps({"quantities": state.quantities, "equity_peak": state.equity_peak}, indent=2, sort_keys=True)
    fd, tmp = tempfile.mkstemp(dir=p.parent if str(p.parent) else ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(payload)
        os.replace(tmp, p)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
