"""Notifications d'exploitation (alertes du watchdog), sans dépendance ni secret dans le code.

Canaux (tous optionnels, configurés par variables d'environnement, jamais versionnés) :
  - ntfy.sh (ou serveur ntfy perso) : ``NTFY_TOPIC`` (+ ``NTFY_SERVER``, défaut https://ntfy.sh) ;
  - Telegram : ``TELEGRAM_BOT_TOKEN`` + ``TELEGRAM_CHAT_ID``.

Sans aucun canal configuré, l'alerte est seulement journalisée : le watchdog reste utilisable et l'absence
de canal est signalée dans son résumé. Une erreur d'envoi ne lève jamais (une alerte ratée ne doit pas
faire tomber le watchdog).
"""

from __future__ import annotations

import base64
import logging
import os

import requests

logger = logging.getLogger(__name__)

SEND_TIMEOUT = 10  # secondes

_NTFY_PRIORITY = {"INFO": "default", "WARNING": "high", "CRITICAL": "urgent"}


def configured_channels() -> list[str]:
    """Canaux réellement configurés (pour que le watchdog puisse signaler « aucune alerte ne partira »)."""
    channels = []
    if os.getenv("NTFY_TOPIC"):
        channels.append("ntfy")
    if os.getenv("TELEGRAM_BOT_TOKEN") and os.getenv("TELEGRAM_CHAT_ID"):
        channels.append("telegram")
    return channels


def _ntfy_title(title: str) -> str:
    """En-tête HTTP `Title` : ASCII tel quel, sinon RFC 2047 (`=?UTF-8?B?...?=`, accepté par ntfy)."""
    if title.isascii():
        return title
    return "=?UTF-8?B?" + base64.b64encode(title.encode("utf-8")).decode("ascii") + "?="


def notify(title: str, message: str, level: str = "WARNING") -> list[str]:
    """Envoie l'alerte sur tous les canaux configurés ; renvoie la liste des canaux ayant accepté le message."""
    level = level.upper()
    logger.log(logging.CRITICAL if level == "CRITICAL" else logging.WARNING, f"[ALERTE {level}] {title} — {message}")
    delivered: list[str] = []

    topic = os.getenv("NTFY_TOPIC")
    if topic:
        server = os.getenv("NTFY_SERVER", "https://ntfy.sh").rstrip("/")
        try:
            resp = requests.post(
                f"{server}/{topic}",
                data=message.encode("utf-8"),
                headers={"Title": _ntfy_title(title), "Priority": _NTFY_PRIORITY.get(level, "default")},
                timeout=SEND_TIMEOUT,
            )
            if resp.ok:
                delivered.append("ntfy")
            else:
                logger.warning(f"ntfy: HTTP {resp.status_code}")
        except Exception as e:  # noqa: BLE001 - une alerte ratée ne doit jamais interrompre le watchdog
            logger.warning(f"ntfy: envoi impossible ({type(e).__name__})")

    token, chat_id = os.getenv("TELEGRAM_BOT_TOKEN"), os.getenv("TELEGRAM_CHAT_ID")
    if token and chat_id:
        try:
            resp = requests.post(
                f"https://api.telegram.org/bot{token}/sendMessage",
                json={"chat_id": chat_id, "text": f"[{level}] {title}\n{message}"},
                timeout=SEND_TIMEOUT,
            )
            if resp.ok:
                delivered.append("telegram")
            else:
                logger.warning(f"telegram: HTTP {resp.status_code}")
        except Exception as e:  # noqa: BLE001 - une alerte ratée ne doit jamais interrompre le watchdog
            # Ne jamais logger l'exception brute : l'URL contient le jeton du bot.
            logger.warning(f"telegram: envoi impossible ({type(e).__name__})")

    return delivered
