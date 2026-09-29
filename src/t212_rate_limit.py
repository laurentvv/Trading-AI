"""Régulation des appels à l'API Trading 212 (limites par endpoint, par COMPTE).

Contexte (analyse 2026-09-29) : ``GET /equity/orders`` est limité à **1 requête / 5 s**
(``GET /equity/positions`` : 1 / 1 s, ``/equity/account/summary`` : 1 / 5 s,
``/equity/history/orders`` : 6 / min, ``POST /equity/orders/stop`` : 1 / 2 s).
Un cycle enchaîne plusieurs de ces appels en quelques secondes (sync, vérification des stops,
cliquet, vente) : résultat, un 429 à chaque cycle sur la lecture des stops
(« Stop fetch error — local stop state preserved » ×16), donc un cliquet qui travaille à l'aveugle.

Ce module espace les appels d'un même endpoint AVANT de les émettre, au lieu de réessayer après
un 429. Il ne modifie jamais la sémantique d'un appel (pas de retry, pas de cache) : il ne fait
que retarder. Les POST d'ordres restent donc soumis à la règle « jamais de retry aveugle ».

Limites officielles (docs.trading212.com, relevées le 2026-09-29). Marge de ~10 % ajoutée.
"""

from __future__ import annotations

import logging
import math
import os
import re
import threading
import time
from typing import Callable
from urllib.parse import urlparse

import requests

logger = logging.getLogger(__name__)

# Variable d'environnement pour désactiver l'espacement (suite de tests : aucune attente réelle).
DISABLE_ENV = "T212_RATE_GATE_DISABLED"

# bucket -> intervalle minimal entre deux appels (secondes)
T212_INTERVALS: dict[str, float] = {
    "GET /equity/orders": 5.5,  # officiel : 1 req / 5 s
    "GET /equity/orders/{id}": 1.1,  # 1 req / 1 s
    "GET /equity/positions": 1.1,  # 1 req / 1 s
    "GET /equity/account/summary": 5.5,  # 1 req / 5 s
    "GET /equity/account/cash": 2.2,  # 1 req / 2 s
    "GET /equity/history/orders": 10.5,  # 6 req / 1 min
    "POST /equity/orders/stop": 2.2,  # 1 req / 2 s
    "POST /equity/orders/limit": 2.2,  # 1 req / 2 s
    "POST /equity/orders/stop_limit": 2.2,  # 1 req / 2 s
    "POST /equity/orders/market": 1.3,  # 50 req / 1 min
    "DELETE /equity/orders/{id}": 1.3,  # 50 req / 1 min
}

_ID_SEGMENT = re.compile(r"/orders/[^/]+$")


def bucket_for(method: str, url: str) -> str:
    """Nom du bucket de limitation pour (méthode, URL) : chemin normalisé sans préfixe ni identifiant."""
    path = urlparse(url).path
    idx = path.find("/equity/")
    if idx >= 0:
        path = path[idx:]
    path = path.rstrip("/")
    if path.startswith("/equity/orders/") and path.count("/") == 3 and not path.endswith(
        ("/market", "/limit", "/stop", "/stop_limit")
    ):
        path = _ID_SEGMENT.sub("/orders/{id}", path)
    return f"{method.upper()} {path}"


class RateGate:
    """Espacement minimal entre appels d'un même bucket, sûr en multi-thread.

    Le créneau est RÉSERVÉ sous verrou avant l'attente : deux threads qui arrivent ensemble
    sont mis en file (jamais servis au même instant).
    """

    def __init__(
        self,
        intervals: dict[str, float] | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] | None = None,
        enabled: bool | None = None,
    ):
        self._intervals = dict(T212_INTERVALS if intervals is None else intervals)
        self._clock = clock
        self._sleeper = sleeper
        self._enabled = enabled
        self._next_slot: dict[str, float] = {}
        self._lock = threading.Lock()
        self.waited_total = 0.0

    def _active(self) -> bool:
        # enabled=None : suit la variable d'environnement (désactivée par la suite de tests) ;
        # True/False : forcé (tests unitaires du régulateur lui-même).
        if self._enabled is not None:
            return self._enabled
        return os.environ.get(DISABLE_ENV) != "1"

    def wait(self, bucket: str) -> float:
        """Bloque jusqu'à ce que le bucket soit libre ; renvoie la durée attendue (s)."""
        interval = self._intervals.get(bucket, 0.0)
        if interval <= 0 or not self._active():
            return 0.0
        with self._lock:
            now = self._clock()
            start = max(now, self._next_slot.get(bucket, -math.inf))
            self._next_slot[bucket] = start + interval
            delay = start - now
        if delay > 0:
            logger.debug(f"T212 rate gate: {bucket} attend {delay:.1f}s")
            (self._sleeper or time.sleep)(delay)
            self.waited_total += delay
        return delay

    def penalize(self, bucket: str, seconds: float) -> None:
        """Après un 429 : impose au moins `seconds` avant le prochain appel de ce bucket."""
        if seconds <= 0 or not self._active():
            return
        with self._lock:
            floor = self._clock() + seconds
            self._next_slot[bucket] = max(self._next_slot.get(bucket, -math.inf), floor)


# Instance partagée par tout le processus (les tickers d'un cycle passent l'un après l'autre).
GATE = RateGate()


class GatedSession(requests.Session):
    """``requests.Session`` dont chaque requête passe d'abord par ``GATE`` (get/delete/post inclus)."""

    def request(self, method, url, *args, **kwargs):  # type: ignore[override]
        GATE.wait(bucket_for(method, str(url)))
        return super().request(method, url, *args, **kwargs)


def retry_after_seconds(resp: requests.Response | None, default: float, cap: float = 15.0) -> float:
    """Délai à respecter après un 429 : `Retry-After` (secondes) si fourni, sinon `default`, borné à `cap`."""
    if resp is not None:
        raw = resp.headers.get("Retry-After") if getattr(resp, "headers", None) else None
        try:
            if isinstance(raw, (str, int, float)):
                return max(0.0, min(float(raw), cap))
        except ValueError:
            pass
    return min(default, cap)
