"""
schedule_test.py — copie de schedule.py pour valider le déclencheur week-end
du council SANS attendre le vrai samedi/dimanche 09:00.

Différences avec l'original :
  - COUNCIL_HOUR est forcé à (maintenant + 2 minutes) pour déclencher dans ~2 min.
  - Le check weekday() >= 5 est forcé à True (on simule un week-end).
  - Le cycle de trading ET le morning brief sont désactivés (on ne teste qu'eux
    secondairement ; ici on valide le pipe scheduler → subprocess → council).
  - Sleep réduit à 15s pour la réactivité du test.

Une fois le council déclenché une fois, le script s'arrête tout seul (exit 0)
pour ne pas tourner indéfiniment.

Usage : python schedule_test.py   (puis attendre ~2 min)
"""

import sys
import time
import logging
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# --- Configuration de test -------------------------------------------------
# Cible : l'heure actuelle + 2 minutes. On arrondit la minute pour la lisibilité.
_NOW = datetime.now()
_TRIGGER_AT = _NOW.replace(second=0, microsecond=0) + timedelta(minutes=2)
# Si les 2 minutes tombent sur l'heure suivante (ex: 01:59 → 02:01), c'est
# exactement ce qu'on veut : un créneau réaliste dans ~2 min.

COUNCIL_HOUR = _TRIGGER_AT.hour
COUNCIL_MINUTE = _TRIGGER_AT.minute
COUNCIL_DAYS_ANALYZED = 7
COUNCIL_TIMEOUT = 3600
# Forcé : on simule un jour de week-end quel que soit le vrai jour.
FORCE_WEEKEND = True

# --- Logging ---------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("ScheduleTest")
console = Console()


def run_weekend_council():
    """Identique à schedule.run_weekend_council (subprocess isolé + timeout)."""
    logger.info("🏛️ Lancement du Conseil d'IA (test week-end)")
    try:
        cmd = [
            "uv", "run", "python", "-m", "src.council.weekend_council",
            "--days", str(COUNCIL_DAYS_ANALYZED),
        ]
        with open("weekend_council_test.log", "a", encoding="utf-8") as f:
            f.write(f"\n--- Lancement test {datetime.now().isoformat()} ---\n")
            result = subprocess.run(
                cmd, stdout=f, stderr=subprocess.STDOUT, text=True, timeout=COUNCIL_TIMEOUT
            )

        if result.returncode == 0:
            logger.info("✅ Conseil d'IA terminé avec succès")
        else:
            logger.error(f"❌ Erreur lors du Conseil d'IA : Code {result.returncode}")

        # Confirme que le rapport a bien été produit.
        date_str = datetime.now().strftime("%Y-%m-%d")
        report_path = Path("docs/council_reports") / f"council_report_{date_str}.md"
        if report_path.exists():
            logger.info(f"📄 Rapport du conseil disponible : {report_path}")
        else:
            logger.warning(f"⚠️ Aucun rapport du conseil trouvé à {report_path}")

    except subprocess.TimeoutExpired:
        logger.error(f"⏱ Timeout ({COUNCIL_TIMEOUT}s) dépassé pour le Conseil d'IA.")
    except Exception as e:
        logger.error(f"💥 Erreur critique lors du Conseil d'IA : {e}")


def get_dashboard(status_msg, council_status):
    table = Table(box=None, expand=True)
    table.add_column("Propriété", style="cyan")
    table.add_column("Valeur", style="white")
    table.add_row("Statut test", f"[bold]{status_msg}[/bold]")
    table.add_row("Council déclenché à", f"[bold green]{COUNCIL_HOUR:02d}:{COUNCIL_MINUTE:02d}[/bold green]")
    table.add_row("Conseil (simulé week-end)", f"{council_status}")
    table.add_row("Mode", "TEST (cycle trading + morning brief désactivés)")
    return Panel(
        table,
        title="[bold magenta]Schedule TEST — déclenchement council[/bold magenta]",
        subtitle="Ctrl+C pour arrêter",
        border_style="magenta",
    )


def main():
    last_council_date = None
    triggered = False

    console.print(
        Panel(
            f"[bold magenta]Test du déclencheur week-end du council[/bold magenta]\n"
            f"Heure actuelle : {_NOW.strftime('%H:%M:%S')}\n"
            f"Déclenchement prévu à : {COUNCIL_HOUR:02d}:{COUNCIL_MINUTE:02d} "
            f"(dans ~{( _TRIGGER_AT - _NOW).total_seconds()//60:.0f} min)\n"
            f"Sortie automatique après le 1er déclenchement.",
            border_style="magenta",
        )
    )

    try:
        while not triggered:
            now = datetime.now()

            # Conditions de déclenchement : on simule week-end + créneau atteint.
            is_weekend = FORCE_WEEKEND or now.weekday() >= 5
            in_council_window = (
                now.hour == COUNCIL_HOUR and now.minute >= COUNCIL_MINUTE
            )

            if is_weekend and in_council_window:
                if last_council_date != now.date():
                    console.print(
                        f"\n[bold green]>>> Déclenchement du council à "
                        f"{now.strftime('%H:%M:%S')} <<<[/bold green]\n"
                    )
                    run_weekend_council()
                    last_council_date = now.date()
                    triggered = True
                    status_display = "[bold green]DÉCLENCHÉ[/bold green]"
                else:
                    status_display = "[dim]Déjà fait aujourd'hui[/dim]"
            else:
                # Affiche combien de temps avant le déclenchement.
                if now < _TRIGGER_AT:
                    remaining = _TRIGGER_AT - now
                    status_display = (
                        f"[bold yellow]EN ATTENTE[/bold yellow] — "
                        f"déclenchement dans {int(remaining.total_seconds())}s"
                    )
                else:
                    status_display = "[bold yellow]VEILLE[/bold yellow] (hors fenêtre)"

            council_status = (
                "[bold green]Terminé[/bold green]" if triggered
                else f"[bold yellow]En attente ({COUNCIL_HOUR:02d}:{COUNCIL_MINUTE:02d})[/bold yellow]"
            )

            console.clear()
            console.print(get_dashboard(status_display, council_status))
            time.sleep(15)

    except KeyboardInterrupt:
        console.print("\n[bold red]Test arrêté manuellement.[/bold red]")
        sys.exit(130)

    console.print(
        Panel(
            "[bold green]✅ Test terminé : le scheduler a bien déclenché le council "
            "via subprocess.[/bold green]\n"
            "Vérifie weekend_council_test.log et docs/council_reports/.",
            border_style="green",
        )
    )
    sys.exit(0)


if __name__ == "__main__":
    main()
