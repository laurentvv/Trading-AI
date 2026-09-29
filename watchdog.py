"""Watchdog d'exploitation : détecte un scheduler mort/bloqué ou une position sans stop, alerte, relance.

Contexte (analyse 2026-09-29) : le scheduler est resté arrêté 6,4 jours (15/09 15:34 → 22/09 00:07, quatre
séances perdues) sans relance ni alerte ; disponibilité mesurée 66 % pour un critère de 95 %. Le superviseur
`start_scheduler.bat` ne relance qu'après un crash Python, pas si la fenêtre est fermée ou la machine
redémarrée, et le verrou (rafraîchi par un thread dédié) masque un scheduler bloqué.

Ce script est lancé toutes les 15 minutes par le Planificateur de tâches Windows
(voir `scripts/install_watchdog_task.ps1`). Il vérifie :
  1. scheduler vivant (verrou présent, PID vivant, verrou frais) ;
  2. cycle récent pendant la séance (lun-ven 09:00-18:15) ;
  3. aucune position ouverte sans stop broker (`t212_portfolio_state.json`) ;
  4. pas de rafale d'erreurs dans `trading.log`.

Alertes dédupliquées (rappel toutes les 2 h par cause, message « résolu » à la disparition). ``--restart``
relance le scheduler s'il est mort (verrou protégeant contre les doublons, 3 relances/heure max).
``--pause`` / ``--resume`` : arrêt volontaire (aucune alerte, aucune relance) — à utiliser avant toute maintenance.

Usage : python watchdog.py [--restart] [--dry-run] | --pause | --resume | --status
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

logger = logging.getLogger("watchdog")

# --- Paramètres ---------------------------------------------------------------------------------
TRADING_WINDOW_START = dt.time(9, 0)  # 1er cycle à 08:30 : on ne juge qu'à partir de 09:00
TRADING_WINDOW_END = dt.time(18, 15)  # dernier cycle 18:00 (+ durée)
MAX_CYCLE_AGE_MIN = 60  # cadence 30 min + durée d'un cycle (jusqu'à ~5 min) + marge
LOCK_MAX_AGE_SEC = 180  # le thread « lock-keeper » rafraîchit toutes les 30 s
ERROR_BURST_WINDOW_MIN = 60
ERROR_BURST_THRESHOLD = 25
ALERT_COOLDOWN_MIN = 120
MAX_RESTARTS_PER_HOUR = 3
LOG_TAIL_BYTES = 3_000_000

_TS = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")


@dataclass(frozen=True)
class Alert:
    key: str
    level: str  # INFO | WARNING | CRITICAL
    message: str


# --- Lecture des artefacts ----------------------------------------------------------------------
def _pid_alive(pid: int) -> bool:
    try:
        import psutil

        return psutil.pid_exists(pid)
    except ImportError:  # pragma: no cover - psutil est une dépendance du projet
        if os.name == "nt":
            out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}"], capture_output=True, text=True).stdout
            return str(pid) in out
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False


def _tail_text(path: Path, max_bytes: int = LOG_TAIL_BYTES) -> str:
    try:
        size = path.stat().st_size
        with open(path, "rb") as f:
            if size > max_bytes:
                f.seek(size - max_bytes)
            return f.read().decode("utf-8", errors="replace")
    except OSError:
        return ""


def is_trading_window(now: dt.datetime) -> bool:
    return now.weekday() < 5 and TRADING_WINDOW_START <= now.time() <= TRADING_WINDOW_END


def last_cycle_start(scheduler_log: Path) -> dt.datetime | None:
    """Horodatage du dernier « Lancement du cycle de trading » dans scheduler.log."""
    last = None
    for line in _tail_text(scheduler_log).splitlines():
        if "Lancement du cycle de trading" in line:
            m = _TS.match(line)
            if m:
                last = dt.datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
    return last


def count_recent_errors(trading_log: Path, now: dt.datetime, window_min: int = ERROR_BURST_WINDOW_MIN) -> int:
    since = now - dt.timedelta(minutes=window_min)
    count = 0
    for line in _tail_text(trading_log).splitlines():
        if " - ERROR - " in line or " - CRITICAL - " in line:
            m = _TS.match(line)
            if m and dt.datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S") >= since:
                count += 1
    return count


# --- Contrôles ----------------------------------------------------------------------------------
def check_scheduler_alive(base: Path, now: dt.datetime, pid_alive: Callable[[int], bool] = _pid_alive) -> Alert | None:
    # Volontairement PAS limité à la séance : le scheduler tourne 24 h/24 (brief de 01:00, council du samedi).
    # Un arrêt voulu se déclare avec `--pause`, sinon l'alerte revient toutes les 2 h.
    lock = base / "scheduler.lock"
    if not lock.exists():
        return Alert("scheduler-dead", "CRITICAL", "scheduler.lock absent : le scheduler n'est pas lancé.")
    try:
        pid = int(lock.read_text().strip())
    except (OSError, ValueError):
        return Alert("scheduler-dead", "CRITICAL", "scheduler.lock illisible : état du scheduler inconnu.")
    if not pid_alive(pid):
        return Alert("scheduler-dead", "CRITICAL", f"Le PID {pid} du verrou n'existe plus (arrêt brutal ?).")
    age = now.timestamp() - lock.stat().st_mtime
    if age > LOCK_MAX_AGE_SEC:
        return Alert("scheduler-dead", "CRITICAL", f"Verrou non rafraîchi depuis {age / 60:.0f} min (processus gelé ?).")
    return None


def check_cycle_freshness(base: Path, now: dt.datetime) -> Alert | None:
    if not is_trading_window(now):
        return None
    last = last_cycle_start(base / "scheduler.log")
    if last is None:
        return Alert("no-cycle", "CRITICAL", "Aucun cycle de trading trouvé dans scheduler.log.")
    age_min = (now - last).total_seconds() / 60
    if age_min > MAX_CYCLE_AGE_MIN:
        return Alert(
            "no-cycle",
            "CRITICAL",
            f"Dernier cycle il y a {age_min:.0f} min (seuil {MAX_CYCLE_AGE_MIN}) : scheduler bloqué ou arrêté.",
        )
    return None


def check_positions_have_stops(base: Path) -> list[Alert]:
    path = base / "t212_portfolio_state.json"
    if not path.exists():
        return []
    try:
        tickers = json.loads(path.read_text(encoding="utf-8")).get("tickers", {})
    except (OSError, ValueError):
        return [Alert("state-unreadable", "WARNING", "t212_portfolio_state.json illisible.")]
    alerts = []
    for name, st in tickers.items():
        pos = (st or {}).get("active_position")
        if pos and not pos.get("stop_order_id"):
            alerts.append(
                Alert(
                    f"no-stop:{name}",
                    "CRITICAL",
                    f"Position ouverte sur {name} SANS stop broker (aucune protection si la machine tombe).",
                )
            )
    return alerts


def check_error_burst(base: Path, now: dt.datetime) -> Alert | None:
    n = count_recent_errors(base / "trading.log", now)
    if n >= ERROR_BURST_THRESHOLD:
        return Alert("error-burst", "WARNING", f"{n} erreurs/critiques dans trading.log sur {ERROR_BURST_WINDOW_MIN} min.")
    return None


def collect_alerts(base: Path, now: dt.datetime, pid_alive: Callable[[int], bool] = _pid_alive) -> list[Alert]:
    alerts: list[Alert] = []
    for a in (
        check_scheduler_alive(base, now, pid_alive),
        check_cycle_freshness(base, now),
        check_error_burst(base, now),
    ):
        if a:
            alerts.append(a)
    alerts.extend(check_positions_have_stops(base))
    return alerts


# --- État (déduplication, relances, pause) ------------------------------------------------------
def _load_state(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _save_state(path: Path, state: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def select_alerts_to_send(alerts: list[Alert], state: dict, now: dt.datetime) -> tuple[list[Alert], list[str]]:
    """Alertes à (r)envoyer selon le délai de rappel, et clés résolues depuis le dernier passage."""
    sent = state.setdefault("sent", {})
    cooldown = dt.timedelta(minutes=ALERT_COOLDOWN_MIN)
    to_send = []
    for a in alerts:
        prev = sent.get(a.key)
        if prev is None or now - dt.datetime.fromisoformat(prev) >= cooldown:
            to_send.append(a)
            sent[a.key] = now.isoformat(timespec="seconds")
    active = {a.key for a in alerts}
    resolved = [k for k in list(sent) if k not in active]
    for k in resolved:
        del sent[k]
    return to_send, resolved


def _default_launcher(base: Path) -> None:  # pragma: no cover - lance une vraie fenêtre Windows
    subprocess.Popen(
        'start "Trading AI - Scheduler" cmd /k start_scheduler.bat',
        shell=True,
        cwd=str(base),
    )


def maybe_restart(
    base: Path,
    state: dict,
    now: dt.datetime,
    launcher: Callable[[Path], None] = _default_launcher,
    pid_alive: Callable[[int], bool] = _pid_alive,
) -> str:
    """Relance le scheduler mort. Retourne une description de l'action (vide si rien n'a été fait)."""
    recent = [t for t in state.get("restarts", []) if now - dt.datetime.fromisoformat(t) < dt.timedelta(hours=1)]
    state["restarts"] = recent
    if len(recent) >= MAX_RESTARTS_PER_HOUR:
        return f"relance ignorée : déjà {len(recent)} relances sur la dernière heure (boucle de crash ?)"
    lock = base / "scheduler.lock"
    if lock.exists():
        try:
            pid = int(lock.read_text().strip())
        except (OSError, ValueError):
            pid = None
        if pid is not None and pid_alive(pid) and (now.timestamp() - lock.stat().st_mtime) <= LOCK_MAX_AGE_SEC:
            return ""  # vivant : ne jamais relancer par-dessus
        try:
            lock.unlink()  # verrou orphelin d'un arrêt brutal
        except OSError:
            pass
    launcher(base)
    state["restarts"] = recent + [now.isoformat(timespec="seconds")]
    return "scheduler relancé (start_scheduler.bat)"


# --- Orchestration ------------------------------------------------------------------------------
def run_once(
    base: Path,
    now: dt.datetime | None = None,
    restart: bool = False,
    dry_run: bool = False,
    notifier: Callable[[str, str, str], object] | None = None,
    launcher: Callable[[Path], None] = _default_launcher,
    pid_alive: Callable[[int], bool] = _pid_alive,
) -> dict:
    now = now or dt.datetime.now()
    pause_file = base / "watchdog.pause"
    if pause_file.exists():
        return {"paused": True, "alerts": [], "sent": [], "resolved": [], "action": ""}

    if notifier is None:
        from src.notifier import notify as notifier  # import tardif : le watchdog reste autonome sans le projet

    state_path = base / "watchdog_state.json"
    state = _load_state(state_path)
    alerts = collect_alerts(base, now, pid_alive)

    action = ""
    if restart and not dry_run and any(a.key == "scheduler-dead" for a in alerts):
        action = maybe_restart(base, state, now, launcher, pid_alive)

    to_send, resolved = select_alerts_to_send(alerts, state, now)
    if not dry_run:
        for a in to_send:
            msg = a.message + (f" Action : {action}." if action and a.key == "scheduler-dead" else "")
            notifier(f"Trading-AI : {a.key}", msg, a.level)
        for key in resolved:
            notifier(f"Trading-AI : {key} résolu", "Le problème n'est plus détecté.", "INFO")
        _save_state(state_path, state)
    return {"paused": False, "alerts": alerts, "sent": to_send, "resolved": resolved, "action": action}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Watchdog d'exploitation Trading-AI")
    parser.add_argument("--restart", action="store_true", help="relance le scheduler s'il est mort")
    parser.add_argument("--dry-run", action="store_true", help="affiche les constats sans alerter ni relancer")
    parser.add_argument("--pause", action="store_true", help="arrêt volontaire : plus d'alerte ni de relance")
    parser.add_argument("--resume", action="store_true", help="reprend la surveillance")
    parser.add_argument("--status", action="store_true", help="affiche l'état de la surveillance")
    parser.add_argument("--base", default=".", help="dossier du projet (défaut : dossier courant)")
    args = parser.parse_args(argv)

    base = Path(args.base).resolve()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    pause_file = base / "watchdog.pause"

    if args.pause:
        pause_file.write_text(dt.datetime.now().isoformat(timespec="seconds"), encoding="utf-8")
        print(f"Surveillance en PAUSE ({pause_file}). Reprendre : python watchdog.py --resume")
        return 0
    if args.resume:
        pause_file.unlink(missing_ok=True)
        print("Surveillance reprise.")
        return 0

    try:  # les canaux d'alerte (NTFY_TOPIC, TELEGRAM_*) se configurent dans .env, comme le reste du projet
        from dotenv import load_dotenv

        load_dotenv(base / ".env")
    except ImportError:  # pragma: no cover
        pass
    from src.notifier import configured_channels

    channels = configured_channels()
    if args.status:
        print(f"Pause : {'OUI' if pause_file.exists() else 'non'} | canaux d'alerte : {channels or 'AUCUN'}")

    # --status est en lecture seule : ni alerte, ni relance, ni écriture de watchdog_state.json (sinon il
    # réinitialiserait le délai de rappel de 2 h et retarderait une vraie alerte).
    result = run_once(base, restart=args.restart and not args.status, dry_run=args.dry_run or args.status)
    if result["paused"]:
        print("Surveillance en pause : rien à faire.")
        return 0
    if not channels:
        print("ATTENTION : aucun canal d'alerte configuré (NTFY_TOPIC ou TELEGRAM_BOT_TOKEN+TELEGRAM_CHAT_ID) : "
              "les alertes ne sont que journalisées.")
    if not result["alerts"]:
        print("OK : scheduler vivant, cycles récents, positions protégées.")
        return 0
    for a in result["alerts"]:
        print(f"[{a.level}] {a.key}: {a.message}")
    if result["action"]:
        print(f"Action : {result['action']}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
